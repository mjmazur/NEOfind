"""Pixel-integrated Gaussian PSF helpers (same model as neosim.render).

Coordinates are 0-based with pixel centers on integers.
"""

import math

import numpy as np
from scipy.ndimage import correlate1d
from scipy.special import erf

FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
RADIUS_FACTOR = 4.0
_SQRT2 = math.sqrt(2.0)


def radius_px(sigma: float) -> int:
    return max(2, int(math.ceil(RADIUS_FACTOR * sigma)))


def profile_1d(offset: float, sigma: float, radius: int) -> np.ndarray:
    """Flux fraction in pixels -radius..radius for a source at ``offset``."""
    edges = np.arange(-radius, radius + 2) - 0.5 - offset
    return 0.5 * np.diff(erf(edges / (_SQRT2 * sigma)))


def stamp(x: float, y: float, sigma: float):
    """PSF stamp for a source at (x, y): returns (stamp, row0, col0)."""
    r = radius_px(sigma)
    ix, iy = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
    fx = profile_1d(x - ix, sigma, r)
    fy = profile_1d(y - iy, sigma, r)
    return np.outer(fy, fx), iy - r, ix - r


def clip_stamp(st: np.ndarray, r0: int, c0: int, shape):
    """Intersect a stamp with an image; returns (sub-stamp, (r_a, r_b, c_a, c_b)) or None."""
    h, w = shape
    sh, sw = st.shape
    ra, rb = max(r0, 0), min(r0 + sh, h)
    ca, cb = max(c0, 0), min(c0 + sw, w)
    if ra >= rb or ca >= cb:
        return None
    return st[ra - r0:rb - r0, ca - c0:cb - c0], (ra, rb, ca, cb)


def separable_correlate(image: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    out = correlate1d(image, kernel_1d, axis=0, mode="constant", cval=0.0)
    return correlate1d(out, kernel_1d, axis=1, mode="constant", cval=0.0)


def centered_kernel(sigma: float) -> np.ndarray:
    return profile_1d(0.0, sigma, radius_px(sigma))
