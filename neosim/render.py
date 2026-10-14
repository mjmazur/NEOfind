"""Rasterization (adapted from meteorsim.render).

Pixel coordinates follow the FITS/astropy 0-based convention used throughout:
pixel (i, j) is centered on x = i, y = j and spans [i - 0.5, i + 0.5].
"""

import math

import numpy as np
from scipy.special import erf

RADIUS_FACTOR = 4.5  # stamp half-width, in units of sigma
MIN_RADIUS_PX = 2

_SQRT2 = math.sqrt(2.0)

# Sub-step spacing when integrating a source's motion within one exposure, px.
TRAIL_STEP_PX = 0.25
MAX_TRAIL_SUBSTEPS = 400


def add_gaussian_stamp(image: np.ndarray, cx: float, cy: float,
                       total_flux: float, sigma: float) -> None:
    """Add a pixel-integrated 2D Gaussian to ``image`` in place (x=col, y=row).

    Copied from meteorsim: pixel i spans [i, i+1] in *these* coordinates;
    use ``add_point`` for 0-based pixel-center coordinates.
    """
    if total_flux <= 0 or sigma <= 0:
        return
    height, width = image.shape
    r = max(MIN_RADIUS_PX, int(math.ceil(RADIUS_FACTOR * sigma)))

    x0, x1 = int(math.floor(cx)) - r, int(math.floor(cx)) + r + 1
    y0, y1 = int(math.floor(cy)) - r, int(math.floor(cy)) + r + 1
    xs0, xs1 = max(0, x0), min(width, x1)
    ys0, ys1 = max(0, y0), min(height, y1)
    if xs0 >= xs1 or ys0 >= ys1:
        return

    xe = np.arange(xs0, xs1 + 1) - cx
    ye = np.arange(ys0, ys1 + 1) - cy
    fx = 0.5 * np.diff(erf(xe / (_SQRT2 * sigma)))
    fy = 0.5 * np.diff(erf(ye / (_SQRT2 * sigma)))
    image[ys0:ys1, xs0:xs1] += total_flux * np.outer(fy, fx)


def add_point(image: np.ndarray, x: float, y: float, flux: float, sigma: float) -> None:
    """Point source at 0-based pixel-center coordinates (x, y)."""
    add_gaussian_stamp(image, x + 0.5, y + 0.5, flux, sigma)


def render_points(shape, xs, ys, fluxes, sigma: float) -> np.ndarray:
    height, width = shape
    margin = RADIUS_FACTOR * sigma + 1
    layer = np.zeros(shape)
    keep = (xs > -margin) & (xs < width + margin) & (ys > -margin) & (ys < height + margin)
    for x, y, f in zip(xs[keep], ys[keep], fluxes[keep]):
        add_point(layer, x, y, f, sigma)
    return layer


def trail_substeps(trail_px: float) -> int:
    return int(min(MAX_TRAIL_SUBSTEPS, max(1, math.ceil(trail_px / TRAIL_STEP_PX))))


def apply_noise(model_e: np.ndarray, read_noise_e: float,
                np_rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise on the expected charge plus Gaussian read noise."""
    return np_rng.poisson(np.clip(model_e, 0.0, None)).astype(np.float64) + \
        np_rng.normal(0.0, read_noise_e, size=model_e.shape)
