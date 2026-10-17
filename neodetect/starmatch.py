"""Registration by star matching, for frames without a WCS (PNG/TIFF/JPEG).

Each frame's stars are detected (local maxima above ``DETECT_NSIG`` sigma of
a smooth-sky-subtracted image, centroided), and the brightest are matched to
the reference frame's: a phase correlation of the images gives the shift,
nearest neighbours within ``MATCH_PX`` give candidate pairs, and RANSAC fits
an affine transform (shift, small rotation, scale, shear) to them, refined
once on all pairs it then explains. Each frame gets a linear TAN WCS: the
reference frame's (centred on the given pointing, or (0, 0), with the given
pixel scale, or ``DEFAULT_SCALE_ARCSEC``) composed with its affine map, so
everything downstream (registration onto a common grid, rates, positions)
works as with FITS input. Without a real pointing and scale, positions and
rates come out in those nominal units.

Phase correlation tolerates small rotations only (a few degrees over the
field); fields rotating more between frames (alt-az mounts without a
derotator) need a plate solution instead.
"""

import math
from typing import List, Optional

import numpy as np
from astropy.wcs import WCS
from scipy import ndimage

from .cosmics import estimate_sky
from .io import Frame, robust_sigma

DETECT_NSIG = 8.0
N_STARS = 80
MATCH_PX = 3.0
MIN_MATCHES = 6
DEFAULT_SCALE_ARCSEC = 1.0


def detect_stars(data: np.ndarray, bad: np.ndarray, n: int = N_STARS):
    """(x, y, flux) of the ``n`` brightest unsaturated, isolated peaks."""
    sky = estimate_sky(data, bad)
    r = np.where(bad, 0.0, data - sky)
    sig = robust_sigma(r[~bad][::7])
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros((0, 3))
    sm = ndimage.gaussian_filter(r, 1.0)
    peak = (sm == ndimage.maximum_filter(sm, 7)) & (sm > DETECT_NSIG * sig / 2.0)
    peak &= ~ndimage.binary_dilation(bad, np.ones((5, 5), bool))
    peak[:4, :] = peak[-4:, :] = False
    peak[:, :4] = peak[:, -4:] = False
    ys, xs = np.nonzero(peak)
    if ys.size == 0:
        return np.zeros((0, 3))
    out = []
    oy, ox = np.mgrid[-2:3, -2:3]
    for y, x in zip(ys, xs):
        st = np.clip(r[y - 2:y + 3, x - 2:x + 3], 0, None)
        f = st.sum()
        if f <= 0:
            continue
        out.append((x + (st * ox).sum() / f, y + (st * oy).sum() / f, f))
    out = np.array(out)
    return out[np.argsort(-out[:, 2])][:n]


def measure_fwhm_px(data: np.ndarray, bad: np.ndarray, stars) -> Optional[float]:
    """Median FWHM (px) of the stars from their second moments."""
    sky = estimate_sky(data, bad)
    r = data - sky
    vals = []
    oy, ox = np.mgrid[-4:5, -4:5]
    for x, y, _ in stars[:40]:
        xi, yi = int(round(x)), int(round(y))
        if not (4 <= xi < r.shape[1] - 4 and 4 <= yi < r.shape[0] - 4):
            continue
        st = np.clip(r[yi - 4:yi + 5, xi - 4:xi + 5], 0, None)
        f = st.sum()
        if f <= 0:
            continue
        dx, dy = ox - (x - xi), oy - (y - yi)
        var = ((st * dx * dx).sum() + (st * dy * dy).sum()) / (2 * f)
        if var > 0:
            vals.append(2.3548 * math.sqrt(var))
    return float(np.median(vals)) if len(vals) >= 5 else None


def _shift(ref_img, img):
    from skimage.registration import phase_cross_correlation
    (dy, dx), _, _ = phase_cross_correlation(ref_img, img, upsample_factor=4, normalization=None)
    return dx, dy


def match_affine(ref_stars, stars, shift):
    """3x3 matrix mapping this frame's pixels to the reference frame's, or None."""
    from skimage.measure import ransac
    from skimage.transform import AffineTransform
    p = stars[:, :2] + np.array(shift)
    d = np.hypot(p[:, None, 0] - ref_stars[None, :, 0], p[:, None, 1] - ref_stars[None, :, 1])
    j = np.argmin(d, axis=1)
    ok = d[np.arange(len(p)), j] < MATCH_PX
    if ok.sum() < MIN_MATCHES:
        return None
    src, dst = stars[ok, :2], ref_stars[j[ok], :2]
    kw = dict(min_samples=3, residual_threshold=1.0, max_trials=500)
    try:
        model, inl = ransac((src, dst), AffineTransform, rng=0, **kw)
    except TypeError:  # scikit-image < 0.21
        model, inl = ransac((src, dst), AffineTransform, random_state=0, **kw)
    if model is None or inl.sum() < MIN_MATCHES:
        return None
    # Refine on every star the fitted transform explains.
    q = model(stars[:, :2])
    d = np.hypot(q[:, None, 0] - ref_stars[None, :, 0], q[:, None, 1] - ref_stars[None, :, 1])
    j = np.argmin(d, axis=1)
    ok = d[np.arange(len(q)), j] < 1.0
    if ok.sum() >= MIN_MATCHES:
        model = AffineTransform()
        model.estimate(stars[ok, :2], ref_stars[j[ok], :2])
    return model.params


def assign_wcs(frames: List[Frame], ref: Optional[int] = None, scale_arcsec: Optional[float] = None,
               ra_deg: float = 0.0, dec_deg: float = 0.0, log=None) -> List[np.ndarray]:
    """Give every frame a linear TAN WCS from star matching to ``ref``
    (default the middle frame); returns the affine maps (frame px -> ref px).
    Raises if a frame can't be matched."""
    log = log or (lambda *a: None)
    ref = len(frames) // 2 if ref is None else ref
    scale = (scale_arcsec or DEFAULT_SCALE_ARCSEC) / 3600.0
    rf = frames[ref]
    h, w = rf.shape
    cd = np.array([[-scale, 0.0], [0.0, scale]])
    crpix0 = np.array([(w - 1) / 2.0, (h - 1) / 2.0])  # 0-based
    stars = [detect_stars(f.data, f.bad) for f in frames]
    clip = lambda f: np.clip(np.where(f.bad, 0.0, f.data - estimate_sky(f.data, f.bad)), 0, None)
    ref_img = clip(rf)
    maps = []
    for k, f in enumerate(frames):
        if k == ref:
            m = np.eye(3)
        else:
            if f.shape != rf.shape:
                raise ValueError(f"{f.path}: shape {f.shape} differs from the reference {rf.shape}")
            m = match_affine(stars[ref], stars[k], _shift(ref_img, clip(f)))
            if m is None:
                raise ValueError(f"{f.path}: too few stars matched to the reference frame")
        a, t = m[:2, :2], m[:2, 2]
        p0 = np.linalg.solve(a, crpix0 - t)  # this frame's pixel at the reference centre
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra_deg, dec_deg]
        wcs.wcs.crpix = p0 + 1.0
        wcs.wcs.cd = cd @ a
        f.wcs = wcs
        maps.append(m)
        log(f"  frame {k}: {len(stars[k])} stars, shift {t[0]:+.2f} {t[1]:+.2f} px, "
            f"rotation {math.degrees(math.atan2(a[1, 0], a[0, 0])):+.3f} deg")
    return maps
