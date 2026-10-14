"""Synthetic background star field (adapted from meteorsim.starfield).

Cumulative whole-sky-average star counts per square degree follow a
quadratic in magnitude, so the count slope can flatten toward faint limits
(meteorsim's single 0.4 dex/mag slope over-predicts by ~4x at V = 20):

    log10 N(<= m) = c0 + c1 * (m - m1) + c2 * (m - m1) * (m - m2)

The V model passes through the commonly quoted whole-sky counts (~9,100 stars
to V = 6, ~3.4e5 to V = 10, ~1e9 to V = 20). The W1/W2 model is a rough
approximation to near-infrared (2MASS/AllWISE-like) counts; stars have
W1 - W2 ~= 0, so both WISE bands share it. Neither is a real catalog. Density
is scaled by the same illustrative galactic-latitude factor as meteorsim,
evaluated at the field center.

Stars are sampled uniformly in a tangent-plane box around the target that
covers every frame's footprint, then fixed in RA/Dec so each frame's WCS
places them consistently.
"""

import math
from dataclasses import dataclass

import numpy as np

from .astrometry import galactic_latitude_deg, tangent_to_equatorial

GALACTIC_BASE = 0.6
GALACTIC_PEAK = 2.4
GALACTIC_SCALE_DEG = 10.0

# (c0, c1, c2, m1, m2)
COUNT_MODELS = {
    "V": (-0.66, 0.395, -0.0035, 6.0, 10.0),
    "W1": (0.60, 0.433, -0.0148, 8.0, 14.0),
}
COUNT_MODELS["W2"] = COUNT_MODELS["W1"]

BRIGHT_LIMIT_MAG = 2.0  # brightest star generated
FAINT_MARGIN_MAG = 0.5  # stars are generated to limiting_mag + this


@dataclass
class StarField:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray


def log_cumulative_density(mag, band: str):
    """log10 of the whole-sky-average stars/deg^2 brighter than ``mag``."""
    c0, c1, c2, m1, m2 = COUNT_MODELS[band]
    mag = np.asarray(mag, dtype=float)
    return c0 + c1 * (mag - m1) + c2 * (mag - m1) * (mag - m2)


def galactic_density_factor(ra0_deg: float, dec0_deg: float) -> float:
    b = galactic_latitude_deg(ra0_deg, dec0_deg)
    return GALACTIC_BASE + GALACTIC_PEAK * math.exp(-abs(b) / GALACTIC_SCALE_DEG)


def generate_star_field(band: str, limiting_mag: float, ra0_deg: float, dec0_deg: float,
                        half_size_deg: float, density_scale: float,
                        np_rng: np.random.Generator) -> StarField:
    if band not in COUNT_MODELS:
        raise ValueError(f"no star-count model for band {band!r} ({', '.join(COUNT_MODELS)})")
    faint = limiting_mag + FAINT_MARGIN_MAG
    n_faint = 10 ** log_cumulative_density(faint, band)
    n_bright = 10 ** log_cumulative_density(BRIGHT_LIMIT_MAG, band)
    area_deg2 = (2.0 * half_size_deg) ** 2
    expected = (n_faint - n_bright) * area_deg2 * galactic_density_factor(ra0_deg, dec0_deg) \
        * density_scale
    n_stars = int(np_rng.poisson(max(expected, 0.0)))

    half = math.radians(half_size_deg)
    xi = np_rng.uniform(-half, half, size=n_stars)
    eta = np_rng.uniform(-half, half, size=n_stars)
    ra, dec = tangent_to_equatorial(xi, eta, ra0_deg, dec0_deg)

    # Inverse-CDF magnitude sampling, numerically on a fine grid.
    grid = np.linspace(BRIGHT_LIMIT_MAG, faint, 2000)
    cdf = 10 ** log_cumulative_density(grid, band)
    u = np_rng.uniform(n_bright, n_faint, size=n_stars)
    mags = np.interp(u, cdf, grid)
    return StarField(ra_deg=np.asarray(ra), dec_deg=np.asarray(dec), mag=mags)
