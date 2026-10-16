"""Resample every frame onto one common pixel grid.

The grid is a TAN projection with the reference frame's pixel scale and
orientation (so most frames need little rotation), centered on and sized to
the union of all frame footprints, optionally limited to a region. Frames
cover only part of it (WISE frames drift ~4' per orbit), so every pixel
carries a per-frame validity flag.

Per frame, in native pixels: subtract a smooth sky, mask sharp outliers
(cosmic rays, hot pixels), fill masked pixels by normalized Gaussian
convolution of their good neighbors so the spline prefilter doesn't ring;
then resample the image with a cubic spline and the variance bilinearly.
On blank sky a grid sample is invalid only where masked pixels carry more
than ``MAX_BAD_WEIGHT`` of its bilinear interpolation weight, so a lone bad
pixel costs only the samples nearest to it. Where the fill is uncertain --
a masked pixel next to significant signal, e.g. in a star's core -- every
sample whose cubic stencil touches it is invalid, as the fill error would
not subtract out. (Invalidating the whole cubic
stencil instead cost ~45% of every real WISE frame, whose masks flag ~3% of
pixels, scattered.) The matched filter treats invalid samples as missing
data, so a source with a bad pixel in its core still counts with reduced
weight. Images are scaled to the reference frame's photometric zero point
so a source has the same flux everywhere.

Interpolation smooths the noise (lower per-pixel variance, correlated
neighbors); ``preprocess`` calibrates matched-filter significance
empirically, which absorbs this.
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from astropy.wcs import WCS
from scipy import ndimage
from scipy.ndimage import map_coordinates

from .cosmics import estimate_sky, measured_sharpness, sharp_outliers, star_sharpness_limits
from .io import Frame, Sequence, default_psf_fwhm_arcsec, robust_sigma
from .psf import FWHM_TO_SIGMA

MIN_STATIC_FRAMES = 3  # valid frames needed for the second pass's static-sky estimate
STATIC_RESID_FRAC = 0.1  # second pass: noise allowance for static-sky subtraction errors
MAX_BAD_WEIGHT = 0.25  # bilinear weight of masked native pixels that invalidates a sample
SIGNAL_NSIG = 3.0  # a masked pixel whose fill exceeds this many sigma is "on a source"


@dataclass
class Registered:
    wcs: WCS  # common grid
    data: np.ndarray  # (n, h, w) float32, sky-subtracted, reference-ZP DN; 0 where invalid
    var: np.ndarray  # (n, h, w) float32 variance (resampled, not yet noise-calibrated)
    valid: np.ndarray  # (n, h, w) bool
    times: np.ndarray  # mid-exposure MJD per frame
    psf_sigma_px: np.ndarray  # per frame, in grid pixels
    flux_scale: np.ndarray  # multiplier applied to each frame (zero-point matching)
    magzp: Optional[float]  # zero point of the grid's flux units
    n_outlier_px: List[int]
    frame_wcs: List[WCS]
    frame_shapes: List[tuple]

    @property
    def shape(self):
        return self.data.shape[1:]

    @property
    def n(self):
        return self.data.shape[0]

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.wcs.pixel_scale_matrix))) * 3600.0)


def _footprint_world(f: Frame, n_side: int = 8):
    h, w = f.shape
    xs = np.linspace(-0.5, w - 0.5, n_side)
    ys = np.linspace(-0.5, h - 0.5, n_side)
    edge_x = np.concatenate([xs, np.full(n_side, w - 0.5), xs[::-1], np.full(n_side, -0.5)])
    edge_y = np.concatenate([np.full(n_side, -0.5), ys, np.full(n_side, h - 0.5), ys[::-1]])
    return f.wcs.all_pix2world(edge_x, edge_y, 0)


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def build_grid(frames: List[Frame], ref: int, region=None, margin_px: int = 4,
               max_pixels: int = 25_000_000) -> WCS:
    """Common grid WCS. ``region`` = (ra_deg, dec_deg, size_deg) limits it to a square."""
    rf = frames[ref]
    if region is not None:
        ra_c, dec_c = region[0], region[1]
    else:
        centers = np.array([f.wcs.all_pix2world([(f.shape[1] - 1) / 2], [(f.shape[0] - 1) / 2], 0)
                            for f in frames])[:, :, 0]
        v = _unit(centers[:, 0], centers[:, 1]).mean(axis=0)
        ra_c = math.degrees(math.atan2(v[1], v[0])) % 360.0
        dec_c = math.degrees(math.asin(v[2] / np.linalg.norm(v)))

    g = WCS(naxis=2)
    g.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    g.wcs.crval = [ra_c, dec_c]
    g.wcs.cd = rf.wcs.pixel_scale_matrix
    g.wcs.crpix = [0.0, 0.0]
    g.wcs.radesys = "ICRS"

    if region is not None:
        half = 0.5 * region[2] * 3600.0 / rf.pixel_scale_arcsec
        lo_x = lo_y = -half
        hi_x = hi_y = half
    else:
        xs, ys = [], []
        for f in frames:
            ra, dec = _footprint_world(f)
            x, y = g.all_world2pix(ra, dec, 0)
            xs.append(x)
            ys.append(y)
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        lo_x, hi_x, lo_y, hi_y = xs.min(), xs.max(), ys.min(), ys.max()
    w = int(math.ceil(hi_x - lo_x)) + 2 * margin_px
    h = int(math.ceil(hi_y - lo_y)) + 2 * margin_px
    if w * h > max_pixels:
        raise ValueError(f"common grid would be {w}x{h} px; restrict it with a region")
    # Shift so grid pixel (0, 0) sits at the lower-left corner of the bounds.
    g.wcs.crpix = [1.0 - (lo_x - margin_px), 1.0 - (lo_y - margin_px)]
    g.pixel_shape = (w, h)
    return g


def _native_resid(f: Frame):
    """Sky-subtracted image and 1-sigma map."""
    sky = estimate_sky(f.data, f.bad)
    resid = f.data - sky
    if f.unc is not None:
        sig = f.unc.astype(np.float32)
    else:
        sig = np.full(f.shape, robust_sigma(resid[~f.bad][::7]), dtype=np.float32)
    return resid, sig


def _sharpness(f: Frame, resid, sig):
    return measured_sharpness(np.where(f.bad, 0.0, resid).astype(np.float32),
                              np.where(f.bad | ~np.isfinite(sig), np.inf, sig).astype(np.float32),
                              f.bad)


def _prepare_native(f: Frame, psf_sigma_native: float, outlier_nsig: float, resid, sig,
                    sharp_fallback=None):
    """Bad mask including sharp outliers. The outlier tests' ratio limits come
    from the frame's stars, else ``sharp_fallback`` (the visit's)."""
    bad = f.bad.copy()
    r = np.where(bad, 0.0, resid).astype(np.float32)
    s = np.where(bad | ~np.isfinite(sig), np.inf, sig).astype(np.float32)
    limits = star_sharpness_limits(r, s, bad, psf_sigma_native, fallback=sharp_fallback)
    hits = sharp_outliers(resid, np.where(bad, np.inf, sig), bad, psf_sigma_native,
                          nsig=outlier_nsig, limits=limits)
    n_hits = int(hits.sum())
    bad |= hits
    return resid, sig, bad, n_hits, limits


def _fill(img, bad, sigma):
    """Replace ``bad`` pixels by a Gaussian-weighted mean of the good ones."""
    good = (~bad).astype(np.float64)
    num = ndimage.gaussian_filter(np.where(bad, 0.0, img).astype(np.float64), sigma, mode="nearest")
    den = ndimage.gaussian_filter(good, sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = np.where(den > 1e-3, num / den, 0.0)
    return np.where(bad, fill, img).astype(np.float32)


def _static_sky(data, valid):
    """Per-pixel median over valid frames; NaN with fewer than MIN_STATIC_FRAMES."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(valid, data, np.nan), axis=0)
    return np.where(valid.sum(axis=0) >= MIN_STATIC_FRAMES, med, np.nan).astype(np.float32)


def _second_pass(f: Frame, grid: WCS, static, scale, psf_sigma_native, outlier_nsig, resid, sig,
                 bad, limits):
    """Sharp outliers in ``resid`` minus the static sky (grid, mapped back to
    native pixels), not already in ``bad``: hits on stars, where the star
    broadens a cosmic ray enough to pass the first pass's tests. Returns
    (hit mask, native x, y of the hits on the grid)."""
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ra, dec = f.wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
    gx, gy = grid.all_world2pix(ra, dec, 0)
    del ra, dec
    gh, gw = static.shape
    ok = np.isfinite(static)
    inside = (gx >= 0) & (gx <= gw - 1) & (gy >= 0) & (gy <= gh - 1)
    coords = np.array([gy[inside], gx[inside]])
    tn = np.full(h * w, np.nan)
    tn[inside] = map_coordinates(np.where(ok, static, 0.0).astype(np.float64), coords, order=3,
                                 mode="nearest") / scale
    # Every sample within the cubic stencil of an unknown static sky is unknown.
    near_nan = map_coordinates(ndimage.binary_dilation(~ok, np.ones((5, 5), bool)).astype(np.float32),
                               coords, order=0, mode="nearest") > 0
    tn[np.flatnonzero(inside)[near_nan]] = np.nan
    tn = tn.reshape(h, w)
    known = np.isfinite(tn) & ~bad & np.isfinite(sig)
    diff = np.where(known, resid - np.where(known, tn, 0.0), 0.0).astype(np.float32)
    # Subtraction errors scale with the static sky (PSF and registration mismatch).
    noise = np.where(known, np.sqrt(sig.astype(np.float64) ** 2
                                    + (STATIC_RESID_FRAC * np.where(known, tn, 0.0)) ** 2),
                     np.inf).astype(np.float32)
    hits = sharp_outliers(diff, noise, ~known, psf_sigma_native, nsig=outlier_nsig, limits=limits)
    hits &= ~bad
    return hits, gx.reshape(h, w)[hits], gy.reshape(h, w)[hits]


def register(seq: Sequence, ref: Optional[int] = None, region=None,
             outlier_nsig: float = 4.0, second_pass: bool = True, log=None) -> Registered:
    log = log or (lambda *a: None)
    frames = seq.frames
    n = len(frames)
    ref = n // 2 if ref is None else ref
    grid = build_grid(frames, ref, region)
    gw, gh = grid.pixel_shape
    yy, xx = np.mgrid[0:gh, 0:gw]
    g_ra, g_dec = grid.all_pix2world(xx.ravel(), yy.ravel(), 0)
    del xx, yy
    grid_scale = float(np.sqrt(abs(np.linalg.det(grid.pixel_scale_matrix))) * 3600.0)

    zp_ref = frames[ref].magzp
    data = np.zeros((n, gh, gw), np.float32)
    var = np.zeros((n, gh, gw), np.float32)
    valid = np.zeros((n, gh, gw), bool)
    psf_sigma = np.zeros(n)
    scale = np.ones(n)
    n_out = []
    # Native residuals, and the outlier tests' sharpness limits pooled over
    # the frames whose stars measure them (for frames with too few stars).
    natives = [_native_resid(f) for f in frames]
    meas = [m for m in (_sharpness(f, *nr) for f, nr in zip(frames, natives)) if m is not None]
    pooled = tuple(float(v) for v in np.median(meas, axis=0)) if meas else None
    for k, f in enumerate(frames):
        fwhm = default_psf_fwhm_arcsec(f)
        sig_native = fwhm * FWHM_TO_SIGMA / f.pixel_scale_arcsec
        psf_sigma[k] = fwhm * FWHM_TO_SIGMA / grid_scale
        if zp_ref is not None and f.magzp is not None:
            scale[k] = 10 ** (0.4 * (zp_ref - f.magzp))

        resid, sig = natives[k]
        natives[k] = None
        resid, sig, bad, n_hits, limits = _prepare_native(f, sig_native, outlier_nsig, resid, sig,
                                                          pooled)
        n_out.append(n_hits)
        if second_pass:
            natives[k] = (resid, sig, bad, limits, sig_native)
        filled = _fill(resid, bad, max(sig_native, 1.0))
        sig = _fill(sig, bad | ~np.isfinite(sig), max(sig_native, 1.0))

        fx, fy = f.wcs.all_world2pix(g_ra, g_dec, 0)
        fh, fw = f.shape
        inside = (fx > 1.0) & (fx < fw - 2.0) & (fy > 1.0) & (fy < fh - 2.0)
        coords = np.array([fy[inside], fx[inside]])
        img = map_coordinates(filled.astype(np.float64), coords, order=3, mode="nearest")
        v = map_coordinates((sig.astype(np.float64)) ** 2, coords, order=1, mode="nearest")
        on_source = bad & (np.abs(filled) > SIGNAL_NSIG * sig)
        strict = ndimage.binary_dilation(on_source, structure=np.ones((3, 3), bool))
        b = ((map_coordinates(bad.astype(np.float32), coords, order=1, mode="nearest") > MAX_BAD_WEIGHT)
             | (map_coordinates(strict.astype(np.float32), coords, order=1, mode="nearest") > 0))

        ok = np.zeros(gh * gw, bool)
        ok[np.flatnonzero(inside)[~b]] = True
        dk = np.zeros(gh * gw, np.float32)
        vk = np.zeros(gh * gw, np.float32)
        dk[inside] = img * scale[k]
        vk[inside] = v * scale[k] ** 2
        dk[~ok] = 0.0
        vk[~ok] = 0.0
        data[k] = dk.reshape(gh, gw)
        var[k] = vk.reshape(gh, gw)
        valid[k] = ok.reshape(gh, gw)
        log(f"  frame {k}: {ok.mean():.0%} of grid covered, {n_hits} outlier px masked")

    if second_pass and n >= MIN_STATIC_FRAMES + 1:
        # Second pass on the static-sky-subtracted frames: a cosmic ray on a
        # star looks as broad as the star in the first pass. Its grid samples
        # (every one whose cubic stencil touches a hit) become invalid.
        static = _static_sky(data, valid)
        n2 = []
        for k, f in enumerate(frames):
            resid, sig, bad, limits, sig_native = natives[k]
            natives[k] = None
            hits, hx, hy = _second_pass(f, grid, static, scale[k], sig_native, outlier_nsig,
                                        resid, sig, bad, limits)
            n2.append(int(hits.sum()))
            if not n2[-1]:
                continue
            inv = np.zeros((gh, gw), bool)
            x0, y0 = np.floor(hx).astype(int), np.floor(hy).astype(int)
            for dy in range(-1, 3):
                for dx in range(-1, 3):
                    xi, yi = x0 + dx, y0 + dy
                    m = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
                    inv[yi[m], xi[m]] = True
            valid[k] &= ~inv
            data[k][inv] = 0.0
            var[k][inv] = 0.0
            n_out[k] += n2[-1]
        log(f"  second pass (static sky subtracted): {n2} more outlier px masked")

    return Registered(wcs=grid, data=data, var=var, valid=valid, times=seq.times,
                      psf_sigma_px=psf_sigma, flux_scale=scale, magzp=zp_ref,
                      n_outlier_px=n_out, frame_wcs=[f.wcs for f in frames],
                      frame_shapes=[f.shape for f in frames])
