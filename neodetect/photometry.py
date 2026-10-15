"""Aperture correction for the matched-filter photometry.

Track fluxes are fits of a Gaussian PSF (``psf.stamp``). A real PSF has
wings the Gaussian lacks, so the fit sees only part of the flux: on WISE W1
the Gaussian-fit flux is ~0.87 of the total (0.14 mag). The correction is
measured per sequence from bright, isolated stars on the static-sky
template, which went through the same resampling as the asteroid's frames:

    apcorr = median(aperture flux / Gaussian-fit flux) / ee_aperture

with the aperture flux in a radius of ``aperture_radius`` px (background
from the median in an annulus outside it) and ``ee_aperture`` the fraction
of the total flux inside that radius, a property of the instrument (WISE
W1/W2: 0.959 within 8 px, from the official single-exposure PSF; 1 when
unknown). Detection fluxes are multiplied by ``apcorr``. Crowded fields can
lack isolated stars; then an instrument default is used
(``io.default_aperture_correction``).
"""

import math
from typing import Optional

import numpy as np
from scipy import ndimage

from .io import robust_sigma
from .psf import clip_stamp, stamp

BG_WIDTH = 6  # px: annulus from radius + 1 to radius + 1 + BG_WIDTH
MIN_STARS = 8


def aperture_radius(fwhm_px: float) -> int:
    """Aperture radius in px: ~3.6 FWHM (8 px for WISE), at least 5."""
    return max(5, int(round(3.6 * fwhm_px)))


def _stars(img, ok, r_ap, r_out, snr_min=20.0, snr_max=3000.0, isolation=0.05):
    """Peaks of bright isolated point sources on ``img``: (x, y) integers."""
    sig = robust_sigma(img[ok][::7])
    if not sig > 0:
        return []
    z = np.where(ok, img / sig, 0.0)
    sm = ndimage.gaussian_filter(np.where(ok, img, 0.0), 1.0)
    sm_sig = robust_sigma(sm[ok][::7])
    peaks = (sm == ndimage.maximum_filter(sm, size=9)) & (z > snr_min) & (z < snr_max)
    # The aperture must be fully valid (the background annulus uses what is).
    full = ndimage.minimum_filter(ok.astype(np.uint8), size=2 * r_ap + 3).astype(bool)
    h, w = img.shape
    out = []
    for y, x in zip(*np.nonzero(peaks & full)):
        if not (r_out <= x < w - r_out and r_out <= y < h - r_out):
            continue
        win = sm[y - r_out:y + r_out + 1, x - r_out:x + r_out + 1].copy()
        win[r_out - 4:r_out + 5, r_out - 4:r_out + 5] = 0.0
        # A neighbour would bias both fluxes; below 5 sigma it is just noise.
        if win.max() > max(isolation * sm[y, x], 5.0 * sm_sig):
            continue
        out.append((int(x), int(y)))
    return out


def _gauss_fit(img, x0, y0, sigma):
    """Least-squares flux and position of a Gaussian PSF (uniform weights, as
    for a faint source on sky noise), by a few Gauss-Newton steps on the
    position with the flux solved linearly."""
    x, y = float(x0), float(y0)
    f = 0.0
    for _ in range(6):
        st, r0, c0 = stamp(x, y, sigma)
        c = clip_stamp(st, r0, c0, img.shape)
        if c is None:
            return None
        t, (a, b, cc, d) = c
        data = img[a:b, cc:d]
        f = float((data * t).sum() / (t * t).sum())
        # Numerical position gradient of the model.
        e = 0.05
        tx = clip_stamp(*stamp(x + e, y, sigma), img.shape)[0]
        ty = clip_stamp(*stamp(x, y + e, sigma), img.shape)[0]
        if tx.shape != t.shape or ty.shape != t.shape:
            return None
        jx, jy = f * (tx - t) / e, f * (ty - t) / e
        res = data - f * t
        jtj = np.array([[(jx * jx).sum(), (jx * jy).sum()], [(jx * jy).sum(), (jy * jy).sum()]])
        try:
            dx, dy = np.linalg.solve(jtj, [(jx * res).sum(), (jy * res).sum()])
        except np.linalg.LinAlgError:
            return None
        x, y = x + float(np.clip(dx, -0.5, 0.5)), y + float(np.clip(dy, -0.5, 0.5))
        if abs(x - x0) > 1.5 or abs(y - y0) > 1.5:
            return None
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            break
    return f, x, y


def _aperture(img, ok, x, y, r):
    r_in, r_out = r + 1, r + 1 + BG_WIDTH
    ix, iy = int(round(x)), int(round(y))
    rows, cols = np.mgrid[iy - r_out:iy + r_out + 1, ix - r_out:ix + r_out + 1]
    d = np.hypot(cols - x, rows - y)
    cut = img[rows, cols]
    ann = (d >= r_in) & (d <= r_out) & ok[rows, cols]
    if ann.sum() < 30:
        return None
    bg = float(np.median(cut[ann]))
    # Fractional pixel weights on the aperture edge (4x4 subsampling).
    sub = (np.arange(4) + 0.5) / 4 - 0.5
    frac = np.zeros_like(d)
    for sy in sub:
        for sx in sub:
            frac += np.hypot(cols + sx - x, rows + sy - y) <= r
    frac /= 16.0
    return float(((cut - bg) * frac).sum())


def aperture_correction(template: np.ndarray, valid: np.ndarray, psf_sigma_px: float,
                        ee_aperture: float = 1.0, fallback: float = 1.0,
                        radius: Optional[int] = None, max_stars: int = 300, log=None) -> dict:
    """Measure the correction on ``template`` (static sky, sky ~ 0) where
    ``valid``. Returns a dict with ``apcorr`` (``fallback`` if there are too
    few stars, as in crowded fields) and diagnostics."""
    log = log or (lambda *a: None)
    fwhm = psf_sigma_px * 2.0 * math.sqrt(2.0 * math.log(2.0))
    r = radius or aperture_radius(fwhm)
    r_out = r + 1 + BG_WIDTH
    img = np.where(valid, template, 0.0).astype(np.float64)
    stars = _stars(img, valid, r, r_out)
    if len(stars) > max_stars:
        stars = stars[:: len(stars) // max_stars + 1]
    ratios = []
    for x0, y0 in stars:
        g = _gauss_fit(img, x0, y0, psf_sigma_px)
        if g is None or g[0] <= 0:
            continue
        ap = _aperture(img, valid, g[1], g[2], r)
        if ap is not None:
            ratios.append(ap / g[0])
    ratios = np.asarray(ratios)
    out = {"radius_px": r, "ee_aperture": ee_aperture, "n_stars": int(ratios.size)}
    if ratios.size < MIN_STARS:
        log(f"aperture correction: only {ratios.size} stars, using the default {fallback}")
        out.update(apcorr=fallback, ratio_median=None, ratio_scatter=None, fallback=True)
        return out
    med = float(np.median(ratios))
    out.update(apcorr=med / ee_aperture, ratio_median=med,
               ratio_scatter=float(robust_sigma(ratios)), fallback=False)
    log(f"aperture correction: {out['apcorr']:.3f} ({-2.5 * math.log10(out['apcorr']):+.3f} mag) "
        f"from {ratios.size} stars, aperture/Gaussian {med:.3f} +- {out['ratio_scatter']:.3f}, "
        f"EE({r} px) {ee_aperture}")
    return out
