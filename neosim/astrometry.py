"""Minimal spherical astrometry: gnomonic (tangent-plane) projection and
galactic latitude, used to place a synthetic star field at a given sky
pointing without depending on astropy.
"""

import math
from typing import Tuple

# North Galactic Pole, J2000 (deg).
NGP_RA_DEG = 192.85948
NGP_DEC_DEG = 27.12825


def galactic_latitude_deg(ra_deg: float, dec_deg: float) -> float:
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    ngp_ra, ngp_dec = math.radians(NGP_RA_DEG), math.radians(NGP_DEC_DEG)
    sin_b = math.sin(dec) * math.sin(ngp_dec) + \
        math.cos(dec) * math.cos(ngp_dec) * math.cos(ra - ngp_ra)
    sin_b = max(-1.0, min(1.0, sin_b))
    return math.degrees(math.asin(sin_b))


def equatorial_to_tangent(ra_deg, dec_deg, ra0_deg: float, dec0_deg: float):
    """Gnomonic projection: sky (ra, dec) -> tangent-plane (xi, eta), radians.

    Inputs may be scalars or numpy arrays.
    """
    import numpy as np

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ra0, dec0 = math.radians(ra0_deg), math.radians(dec0_deg)

    d_ra = ra - ra0
    cos_c = math.sin(dec0) * np.sin(dec) + math.cos(dec0) * np.cos(dec) * np.cos(d_ra)
    xi = np.cos(dec) * np.sin(d_ra) / cos_c
    eta = (math.cos(dec0) * np.sin(dec) - math.sin(dec0) * np.cos(dec) * np.cos(d_ra)) / cos_c
    return xi, eta


def tangent_to_equatorial(xi, eta, ra0_deg: float, dec0_deg: float):
    """Inverse gnomonic projection: tangent-plane (xi, eta), radians -> sky (ra, dec), deg."""
    import numpy as np

    ra0, dec0 = math.radians(ra0_deg), math.radians(dec0_deg)
    rho = np.sqrt(xi ** 2 + eta ** 2)
    c = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)

    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(cos_c * math.sin(dec0) + eta * sin_c * math.cos(dec0) / rho)
        ra = ra0 + np.arctan2(xi * sin_c, rho * math.cos(dec0) * cos_c - eta * math.sin(dec0) * sin_c)

    # rho == 0 (the pointing center itself) is a removable singularity.
    dec = np.where(rho == 0, dec0, dec)
    ra = np.where(rho == 0, ra0, ra)

    return np.degrees(ra) % 360.0, np.degrees(dec)


def tangent_to_pixel(xi, eta, plate_scale_rad_px: float, width: int, height: int,
                      rotation_deg: float = 0.0) -> Tuple[float, float]:
    """Tangent-plane (xi, eta), radians -> pixel (x, y).

    xi grows with increasing RA (east), eta grows with increasing Dec
    (north/"up"); ``rotation_deg`` rotates the image's up direction relative
    to north. Row index increases downward, so eta is negated.
    """
    import numpy as np

    xi_px = np.asarray(xi) / plate_scale_rad_px
    eta_px = np.asarray(eta) / plate_scale_rad_px

    theta = math.radians(rotation_deg)
    dx = xi_px * math.cos(theta) - eta_px * math.sin(theta)
    dy = xi_px * math.sin(theta) + eta_px * math.cos(theta)

    x = width / 2.0 + dx
    y = height / 2.0 - dy
    return x, y
