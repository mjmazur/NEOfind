"""Per-frame pointing and WCS.

``dither``: ground-style; each frame is centered on the target plus a small
random offset, with a fixed roll plus jitter.

``wise``: WISE survey geometry. The spacecraft scans great circles through the
ecliptic poles, so the image +y axis lies along ecliptic north, and the scan
circle precesses ~1 deg/day, so successive orbits' frames covering the target
step ``cross_scan_step_arcmin`` along ecliptic longitude. Frames are taken
every ~11 s along the scan, so where the target lands along-scan in each
orbit's frame is effectively random (``along_scan_offset_frac`` of the FOV).
"""

import math
from dataclasses import dataclass
from typing import List

import astropy.units as u
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.wcs import WCS

from .astrometry import tangent_to_equatorial
from .cadence import MIN_PER_DAY
from .config import InstrumentConfig, PointingConfig


@dataclass
class FrameGeometry:
    ra_deg: float
    dec_deg: float
    rotation_deg: float  # position angle of image +y, east of north
    wcs: WCS


def make_wcs(ra_deg: float, dec_deg: float, rotation_deg: float, pixel_scale_arcsec: float,
             width: int, height: int) -> WCS:
    """TAN WCS centered on the array, east to the left at zero rotation."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.crpix = [(width + 1) / 2.0, (height + 1) / 2.0]
    s = pixel_scale_arcsec / 3600.0
    th = math.radians(rotation_deg)
    w.wcs.cd = s * np.array([[-math.cos(th), math.sin(th)],
                             [math.sin(th), math.cos(th)]])
    w.wcs.radesys = "ICRS"
    w.pixel_shape = (width, height)
    return w


def _ecliptic_north_pa(lon_deg: float, lat_deg: float) -> float:
    c = SkyCoord(lon_deg * u.deg, lat_deg * u.deg, frame=BarycentricMeanEcliptic)
    n = SkyCoord(lon_deg * u.deg, (lat_deg + 0.01) * u.deg, frame=BarycentricMeanEcliptic)
    return float(c.icrs.position_angle(n.icrs).to(u.deg).value)


def build_pointings(pnt: PointingConfig, inst: InstrumentConfig, times_mjd: np.ndarray,
                    interval_min: float, np_rng: np.random.Generator) -> List[FrameGeometry]:
    n = len(times_mjd)
    frames = []
    if pnt.mode == "dither":
        sigma = math.radians(pnt.dither_arcsec / 3600.0)
        for _ in range(n):
            xi, eta = np_rng.normal(0.0, sigma, size=2)
            ra, dec = tangent_to_equatorial(xi, eta, pnt.ra_deg, pnt.dec_deg)
            rot = pnt.rotation_deg + np_rng.normal(0.0, pnt.rotation_jitter_deg)
            frames.append((float(ra), float(dec), rot))
    elif pnt.mode == "wise":
        target = SkyCoord(pnt.ra_deg * u.deg, pnt.dec_deg * u.deg).transform_to(
            BarycentricMeanEcliptic)
        lon0, lat0 = target.lon.deg, target.lat.deg
        fov_h_deg = inst.height * inst.pixel_scale_arcsec / 3600.0
        t_mid = 0.5 * (times_mjd[0] + times_mjd[-1])
        for t in times_mjd:
            n_orbits = (t - t_mid) * MIN_PER_DAY / interval_min
            d_cross = n_orbits * pnt.cross_scan_step_arcmin / 60.0
            d_along = np_rng.uniform(-1.0, 1.0) * pnt.along_scan_offset_frac * fov_h_deg
            lat = lat0 + d_along
            lon = lon0 + d_cross / max(math.cos(math.radians(lat)), 1e-6)
            c = SkyCoord(lon * u.deg, lat * u.deg, frame=BarycentricMeanEcliptic).icrs
            rot = _ecliptic_north_pa(lon, lat) + np_rng.normal(0.0, pnt.rotation_jitter_deg)
            frames.append((float(c.ra.deg), float(c.dec.deg), rot))
    else:
        raise ValueError(f"unknown pointing mode {pnt.mode!r}")

    return [FrameGeometry(ra, dec, rot,
                          make_wcs(ra, dec, rot, inst.pixel_scale_arcsec, inst.width, inst.height))
            for ra, dec, rot in frames]
