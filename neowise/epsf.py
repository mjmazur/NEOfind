"""Empirical effective PSF (ePSF) from the stars in a set of frames.

An effective PSF is the instrumental PSF already integrated over a pixel:
its value at offset (dx, dy) is the fraction of a star's flux landing in a
pixel whose center is (dx, dy) from the star. It is built on a grid
oversampled ``OVERSAMPLE`` times (Anderson & King 2000): each star's pixels,
divided by its flux and placed at their offsets from its center, sample it
densely. Starting from a Gaussian, each iteration fits every star (flux,
position, background) with the current model and adds the smoothed mean
residual to it. WISE's PSF is only ~2.2 px
FWHM, so naive stacking of shifted stamps would blur it.

Used to inject realistic synthetic asteroids into real frames (and later as
the matched filter).
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage, optimize

from neodetect.cosmics import estimate_sky
from neodetect.io import robust_sigma

OVERSAMPLE = 4
RADIUS = 8  # native pixels
BG_ANNULUS = (RADIUS + 1, RADIUS + 6)  # native pixels: local background, outside the ePSF


@dataclass
class EPSF:
    grid: np.ndarray  # (2R*os+1)^2, value per native pixel at offset (i - c)/os
    oversample: int
    radius: int
    n_stars: int

    def evaluate(self, dx, dy) -> np.ndarray:
        """ePSF at native-pixel offsets (pixel center minus star position)."""
        c = self.radius * self.oversample
        coords = np.array([np.asarray(dy) * self.oversample + c, np.asarray(dx) * self.oversample + c])
        return ndimage.map_coordinates(self.grid, coords, order=3, mode="constant", cval=0.0)

    def stamp(self, x: float, y: float):
        """(values, row0, col0): the pixels around (x, y), 0-based centers,
        normalized to sum to 1."""
        r = self.radius
        ix, iy = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
        rows, cols = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
        v = self.evaluate(cols - x, rows - y)
        return v / v.sum(), iy - r, ix - r

    def fwhm_px(self) -> float:
        """FWHM from the area above half maximum of the (pixel-convolved) profile."""
        g = self.grid
        area = (g >= 0.5 * g.max()).sum() / self.oversample ** 2
        return 2.0 * math.sqrt(area / math.pi)

    def to_dict(self):
        return {"oversample": self.oversample, "radius": self.radius, "n_stars": self.n_stars,
                "fwhm_px": self.fwhm_px(), "grid": self.grid.tolist()}


def _gaussian_grid(sigma: float, os: int, r: int) -> np.ndarray:
    u = (np.arange(2 * r * os + 1) - r * os) / os
    # Pixel-integrated Gaussian (as an effective PSF).
    from scipy.special import erf
    s2 = math.sqrt(2.0) * sigma
    f = 0.5 * (erf((u + 0.5) / s2) - erf((u - 0.5) / s2))
    return np.outer(f, f)


def find_stars(resid, bad, snr_min=40.0, snr_max=3000.0, isolation=BG_ANNULUS[1], edge=None):
    """Bright, unsaturated, isolated point sources: (x, y) integer peaks.
    SNR is the peak over the sky noise."""
    edge = BG_ANNULUS[1] + 2 if edge is None else edge
    sky_sig = robust_sigma(resid[~bad][::5])
    z = np.where(bad, 0.0, resid / sky_sig)
    sm = ndimage.gaussian_filter(np.where(bad, 0.0, resid), 1.0)
    peaks = (sm == ndimage.maximum_filter(sm, size=2 * isolation + 1)) & (z > snr_min) & (z < snr_max)
    # Masked pixels are left out of the fits; just require most of the core.
    bad_near = ndimage.uniform_filter(bad.astype(np.float32), size=5) * 25 > 3.5
    # A second source above 5% of the peak within the isolation radius (beyond
    # the star's own core) disqualifies.
    h, w = resid.shape
    out = []
    for y, x in zip(*np.nonzero(peaks & ~bad_near)):
        if not (edge <= x < w - edge and edge <= y < h - edge):
            continue
        win = sm[y - isolation:y + isolation + 1, x - isolation:x + isolation + 1].copy()
        win[isolation - 5:isolation + 6, isolation - 5:isolation + 6] = 0
        if win.max() > 0.05 * sm[y, x]:
            continue
        out.append((x, y))
    return out


def _annulus_background(img, bad, x, y):
    r0, r1 = BG_ANNULUS
    ix, iy = int(round(x)), int(round(y))
    rows, cols = np.mgrid[iy - r1:iy + r1 + 1, ix - r1:ix + r1 + 1]
    d = np.hypot(cols - x, rows - y)
    sel = (d >= r0) & (d <= r1) & ~bad[rows, cols]
    return float(np.median(img[rows, cols][sel])) if sel.sum() > 20 else None


def _fit_star(img, sig, bad, x0, y0, psf: EPSF, fit_r=3):
    """Least-squares flux and position of one star with ``psf``; the
    background is the median of an annulus outside the ePSF (fitting it
    locally would absorb the PSF wings). Returns (flux, x, y, background)."""
    b = _annulus_background(img, bad, x0, y0)
    if b is None:
        return None
    ix, iy = int(round(x0)), int(round(y0))
    rows, cols = np.mgrid[iy - fit_r:iy + fit_r + 1, ix - fit_r:ix + fit_r + 1]
    d = img[rows, cols] - b
    s = sig[rows, cols]
    ok = ~bad[rows, cols] & (s > 0)
    if ok.sum() < 20:
        return None
    s = np.where(ok, s, 1.0)

    def res(p):
        f, x, y = p
        return ((d - f * psf.evaluate(cols - x, rows - y)) / s)[ok]

    f0 = max(float(d[ok].sum()), 1.0)
    try:
        r = optimize.least_squares(res, [f0, x0, y0], method="lm")
    except Exception:
        return None
    f, x, y = r.x
    if f <= 0 or abs(x - x0) > 1.5 or abs(y - y0) > 1.5:
        return None
    return f, x, y, b


def build_epsf(frames, sigma0_px: float, iterations: int = 10, max_stars: int = 600,
               log=None) -> EPSF:
    """ePSF from the stars on ``frames`` (``neodetect.io.Frame`` list).
    ``sigma0_px``: Gaussian sigma for the initial guess."""
    log = log or (lambda *a: None)
    os, r = OVERSAMPLE, RADIUS
    psf = EPSF(_gaussian_grid(sigma0_px, os, r), os, r, 0)
    stars = []  # (img, sig, bad, x, y)
    for f in frames:
        resid = f.data - estimate_sky(f.data, f.bad)
        sig = f.unc if f.unc is not None else np.full(f.shape, np.std(resid[~f.bad]), np.float32)
        for x, y in find_stars(resid, f.bad):
            stars.append((resid, sig, f.bad, float(x), float(y)))
    stars = stars[:: max(1, len(stars) // max_stars)]
    if len(stars) < 10:
        raise ValueError(f"only {len(stars)} usable stars for an ePSF")

    n_side = 2 * r * os + 1
    for it in range(iterations):
        # Residuals of every star against the current model, binned on the
        # oversampled grid; the model moves by their smoothed mean, so the
        # smoothing only blurs the correction, not the PSF itself.
        bins, vals = [], []
        fits_ok = 0
        for img, sig, bad, x, y in stars:
            p = _fit_star(img, sig, bad, x, y, psf)
            if p is None:
                continue
            fits_ok += 1
            f, xs, ys, b = p
            ix, iy = int(round(xs)), int(round(ys))
            rows, cols = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
            u = np.rint((cols - xs) * os).astype(int) + r * os
            v = np.rint((rows - ys) * os).astype(int) + r * os
            s = sig[rows, cols]
            res = (img[rows, cols] - b) / f - psf.evaluate(cols - xs, rows - ys)
            ok = (~bad[rows, cols] & (s > 0) & (u >= 0) & (u < n_side) & (v >= 0) & (v < n_side)
                  & (np.abs(res) * f < 5.0 * np.where(s > 0, s, np.inf)))
            bins.append(v[ok] * n_side + u[ok])
            vals.append(res[ok])
        bins, vals = np.concatenate(bins), np.concatenate(vals)
        order = np.argsort(bins, kind="stable")
        bins, vals = bins[order], vals[order]
        uniq, first, counts = np.unique(bins, return_index=True, return_counts=True)
        med = np.zeros(n_side * n_side)
        wt = np.zeros(n_side * n_side)
        for bi, st, c in zip(uniq, first, counts):
            if c >= 3:  # a median of fewer is no protection against outliers
                med[bi] = np.median(vals[st:st + c])
                wt[bi] = c
        # Smooth (and fill sparse bins) by count-weighted normalized convolution.
        k = (0.5 if it < iterations // 2 else 0.3) * os
        num = ndimage.gaussian_filter((med * wt).reshape(n_side, n_side), k, mode="constant")
        den = ndimage.gaussian_filter(wt.reshape(n_side, n_side), k, mode="constant")
        corr = np.where(den > 0.05, num / np.maximum(den, 1e-12), 0.0)
        g = psf.grid + corr
        # Normalize: summed over a native-pixel lattice it is 1, on average
        # over sub-pixel phases.
        g /= g.sum() / os ** 2
        psf = EPSF(g, os, r, fits_ok)
        log(f"  ePSF iteration {it + 1}: {fits_ok} stars, FWHM {psf.fwhm_px():.2f} px")
    return psf
