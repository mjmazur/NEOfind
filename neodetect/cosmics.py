"""Sky estimation and sharp-outlier (cosmic ray, hot pixel) masking in a
frame's native pixels.

Cosmic-ray hits and hot pixels are much sharper than the PSF. Two tests,
each bounded by what a Gaussian PSF of the frame's width can produce
(worst case over sub-pixel position), with a safety margin and a
significance requirement:

* **directional**: a pixel against the mean of an opposite pair of
  neighbors, along both axes and both diagonals. Catches single pixels and
  tracks of any length (the pair perpendicular to a track is dark).
* **2x2 box**: the flux in a 2x2 box against the 12-pixel ring around it.
  Catches bright hits split over 2-4 pixels, which on their own look
  PSF-like at the single-pixel level.
* **model comparison**: every significant local peak is fit (5x5 pixels,
  best of a grid of sub-pixel offsets) with the PSF and with a sharp
  charge-diffusion profile; a peak is a hit if the sharp model is better by
  ``dchi2`` or more. For SNR S the models differ by roughly 0.5 S^2 in
  chi^2, so this catches faint split hits the ratio tests' noise margins
  let through, while real point sources essentially never prefer the sharp
  model.

The ratio limits are those of a Gaussian PSF of the frame's nominal width,
or, where the frame's own bright stars are sharper (``star_sharpness_limits``:
the real WISE core is, at some sub-pixel phases), the 95th percentile of
the stars' ratios. With the Gaussian limits alone the tests flagged 9-31
pixels of the bright asteroid (120) Lachesis (peak SNR ~40) in 4 of 12
frames, wherever it fell near a pixel center.

Flagged pixels are grown by one pixel. This must run before resampling,
which would blur the distinction.
"""

import math
import warnings
from functools import lru_cache

import numpy as np
from scipy import ndimage

from .psf import profile_1d


def estimate_sky(data: np.ndarray, bad: np.ndarray, block: int = 32) -> np.ndarray:
    """Smooth sky: block-wise median (ignoring bad pixels and most stars),
    median-filtered, interpolated back to full resolution."""
    h, w = data.shape
    bh, bw = max(1, h // block), max(1, w // block)
    d = np.where(bad, np.nan, data)[:bh * block, :bw * block]
    blocks = d.reshape(bh, block, bw, block).transpose(0, 2, 1, 3).reshape(bh, bw, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coarse = np.nanmedian(blocks, axis=2)
    if np.isnan(coarse).all():
        return np.zeros_like(data)
    coarse = np.where(np.isnan(coarse), np.nanmedian(coarse), coarse)
    coarse = ndimage.median_filter(coarse, size=3, mode="nearest")
    return ndimage.zoom(coarse, (h / bh, w / bw), order=1, mode="nearest",
                        grid_mode=True)[:h, :w].astype(np.float32)


@lru_cache(maxsize=64)
def psf_sharpness_limits(sigma: float):
    """(axis ratio, diagonal ratio, box ratio): the largest values a Gaussian
    PSF reaches for peak / mean(opposite pair) and box / ring."""
    f = profile_1d(0.0, sigma, 1)
    axis = float(f[1] / f[0])
    diag = axis ** 2
    g = profile_1d(0.5, sigma, 2)  # source on a pixel corner: pixels -2..2
    st = np.outer(g, g)
    box = st[2:4, 2:4].sum()
    ring = st[1:5, 1:5].sum() - box
    return axis, diag, float(box / ring)


STAR_SNR_MIN = 30.0  # peak SNR of the stars that calibrate the ratio limits
STAR_MIN = 10
STAR_QUANTILE = 0.95
STAR_MIN_WIDTH = 0.7  # px: second-moment width that tells a star from a hit (~0.4)


def _ratios(r, ys, xs):
    """Per peak: max over axes of peak / mean(opposite pair), the same over
    diagonals, and the largest 2x2 box containing the peak against its ring."""
    def at(dy, dx):
        return r[ys + dy, xs + dx]
    c = at(0, 0)
    axis = np.maximum(c / np.maximum(0.5 * (at(0, 1) + at(0, -1)), 1e-30),
                      c / np.maximum(0.5 * (at(1, 0) + at(-1, 0)), 1e-30))
    diag = np.maximum(c / np.maximum(0.5 * (at(1, 1) + at(-1, -1)), 1e-30),
                      c / np.maximum(0.5 * (at(1, -1) + at(-1, 1)), 1e-30))
    box_r = np.zeros(c.shape)
    for oy in (-1, 0):
        for ox in (-1, 0):
            box = sum(at(oy + dy, ox + dx) for dy in (0, 1) for dx in (0, 1))
            big = sum(at(oy + dy, ox + dx) for dy in range(-1, 3) for dx in range(-1, 3))
            box_r = np.maximum(box_r, box / np.maximum(big - box, 1e-30))
    return axis, diag, box_r


def star_sharpness_limits(r: np.ndarray, s: np.ndarray, bad: np.ndarray, psf_sigma: float,
                          fallback=None):
    """Ratio limits (axis, diagonal, box): the Gaussian ones, raised to the
    stars' (``measured_sharpness``), or to ``fallback`` (e.g. pooled over a
    visit) when the frame has too few bright stars."""
    gauss = psf_sharpness_limits(round(float(psf_sigma), 3))
    meas = measured_sharpness(r, s, bad)
    if meas is None:
        meas = fallback
    return gauss if meas is None else tuple(max(g, m) for g, m in zip(gauss, meas))


def measured_sharpness(r: np.ndarray, s: np.ndarray, bad: np.ndarray):
    """The ``STAR_QUANTILE`` of bright unmasked stars' (axis, diagonal, box)
    ratios, or None with fewer than ``STAR_MIN`` stars. Bright cosmic-ray
    hits are peaks too; they are told apart by their width (5x5 second
    moment)."""
    h, w = r.shape
    clean = ndimage.minimum_filter((~bad & np.isfinite(s)).astype(np.uint8), size=5).astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(clean, r / s, 0.0)
    peak = (r == ndimage.maximum_filter(r, 5)) & (snr > STAR_SNR_MIN)
    peak[:3, :] = peak[-3:, :] = False
    peak[:, :3] = peak[:, -3:] = False
    ys, xs = np.nonzero(peak)
    oy, ox = np.mgrid[-2:3, -2:3]
    cut = np.maximum(r[ys[:, None] + oy.ravel(), xs[:, None] + ox.ravel()], 0.0)
    tot = cut.sum(axis=1)
    m2 = (cut * (oy.ravel() ** 2 + ox.ravel() ** 2)).sum(axis=1) / np.maximum(tot, 1e-30)
    star = np.sqrt(0.5 * m2) >= STAR_MIN_WIDTH
    ys, xs = ys[star], xs[star]
    if ys.size < STAR_MIN:
        return None
    return tuple(float(np.quantile(v, STAR_QUANTILE)) for v in _ratios(r, ys, xs))


def _shift(a, dy, dx):
    """a shifted so out[y, x] = a[y + dy, x + dx] (zero outside)."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def sharp_outliers(resid: np.ndarray, sigma_map: np.ndarray, bad: np.ndarray,
                   psf_sigma: float, nsig: float = 4.0, margin: float = 1.2,
                   grow: int = 1, limits=None) -> np.ndarray:
    """Mask of pixels too sharp to be a PSF (plus ``grow`` px around them).
    ``limits``: (axis, diagonal, box) ratio limits; default from the
    frame's stars (``star_sharpness_limits``)."""
    r = np.where(bad, 0.0, resid).astype(np.float32)
    s = np.where(bad | ~np.isfinite(sigma_map), np.inf, sigma_map).astype(np.float32)
    if limits is None:
        limits = star_sharpness_limits(r, s, bad, psf_sigma)
    axis, diag, box_lim = limits
    hit = np.zeros(r.shape, bool)
    # Masked neighbors are left out of the comparisons (counted as 0 they
    # would make a source next to a masked pixel look twice as sharp).
    ok = (~bad & np.isfinite(s)).astype(np.float32)

    for (dy, dx), lim in (((0, 1), axis), ((1, 0), axis), ((1, 1), diag), ((1, -1), diag)):
        lim *= margin
        n_ok = _shift(ok, dy, dx) + _shift(ok, -dy, -dx)
        pair = (_shift(r, dy, dx) + _shift(r, -dy, -dx)) / np.maximum(n_ok, 1.0)
        noise = s * np.sqrt(1.0 + lim * lim / np.maximum(n_ok, 1.0))
        hit |= (r - lim * np.maximum(pair, 0.0) > nsig * noise) & (r > nsig * s) & (n_ok > 0)

    # 2x2 boxes anchored at (y, x) .. (y+1, x+1) against the surrounding ring.
    lim = margin * box_lim
    s2 = np.where(np.isfinite(s), s, 0.0)
    box = r + _shift(r, 0, 1) + _shift(r, 1, 0) + _shift(r, 1, 1)
    big = sum(_shift(r, dy, dx) for dy in range(-1, 3) for dx in range(-1, 3))
    n_big = sum(_shift(ok, dy, dx) for dy in range(-1, 3) for dx in range(-1, 3))
    n_box = ok + _shift(ok, 0, 1) + _shift(ok, 1, 0) + _shift(ok, 1, 1)
    n_ring = n_big - n_box
    ring = (big - box) * 12.0 / np.maximum(n_ring, 1.0)
    sig_box = np.sqrt(sum(_shift(s2, dy, dx) ** 2 for dy in (0, 1) for dx in (0, 1)))
    noise = sig_box * math.sqrt(1.0 + 3.0 * lim * lim)  # ring has 3x the box's pixels
    box_hit = ((box - lim * np.maximum(ring, 0.0) > nsig * noise) & (box > nsig * sig_box)
               & (n_ring >= 6))
    box_hit &= np.isfinite(s)
    if box_hit.any():
        # Mark all four pixels of each flagged box.
        hit |= box_hit | _shift(box_hit, 0, -1) | _shift(box_hit, -1, 0) | _shift(box_hit, -1, -1)

    hit |= sharp_model_hits(r, s, bad, psf_sigma)
    hit &= ~bad
    if grow > 0 and hit.any():
        hit = ndimage.binary_dilation(hit, structure=np.ones((3, 3), bool), iterations=grow)
    return hit


SHARP_SIGMA_PX = 0.4  # charge-diffusion width of a cosmic-ray hit
_OFFSETS = np.linspace(-0.5, 0.5, 5)


@lru_cache(maxsize=64)
def _templates(sigma: float):
    """(25 offsets, 25 pixels) unit-flux 5x5 templates on a sub-pixel grid."""
    out = []
    for oy in _OFFSETS:
        fy = profile_1d(oy, sigma, 2)
        for ox in _OFFSETS:
            out.append(np.outer(fy, profile_1d(ox, sigma, 2)).ravel())
    return np.array(out)


def sharp_model_hits(r: np.ndarray, s: np.ndarray, bad: np.ndarray, psf_sigma: float,
                     peak_nsig: float = 3.0, sum_nsig: float = 5.0,
                     dchi2: float = 12.0) -> np.ndarray:
    """Pixels (3x3 around each peak) where a sharp profile beats the PSF by ``dchi2``."""
    h, w = r.shape
    finite = np.isfinite(s)
    s0 = np.where(finite, s, 1.0)
    box = ndimage.uniform_filter(r, 3) * 9.0
    box_sig = np.sqrt(ndimage.uniform_filter(np.where(finite, s0 * s0, 0.0), 3) * 9.0)
    peak = ((r == ndimage.maximum_filter(r, 3)) & (r > peak_nsig * s0) & finite
            & (box > sum_nsig * box_sig))
    peak[:2, :] = peak[-2:, :] = False
    peak[:, :2] = peak[:, -2:] = False
    ys, xs = np.nonzero(peak)
    out = np.zeros((h, w), bool)
    if ys.size == 0:
        return out
    oy, ox = np.mgrid[-2:3, -2:3]
    yy, xx = ys[:, None] + oy.ravel()[None], xs[:, None] + ox.ravel()[None]
    d = r[yy, xx].astype(np.float64)
    wt = np.where(finite[yy, xx] & ~bad[yy, xx], 1.0 / s0[yy, xx] ** 2, 0.0)
    best = []
    for sig in (psf_sigma, SHARP_SIGMA_PX):
        t = _templates(round(float(sig), 3))
        num = (wt * d) @ t.T
        den = wt @ (t * t).T
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where((den > 0) & (num > 0), num * num / den, 0.0)
        best.append(gain.max(axis=1))  # chi^2 reduction of the best fit
    is_hit = best[1] - best[0] > dchi2
    out[ys[is_hit], xs[is_hit]] = True
    return ndimage.binary_dilation(out, structure=np.ones((3, 3), bool))
