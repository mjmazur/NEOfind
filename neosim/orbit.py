"""Physical sky paths: heliocentric two-body motion seen from Earth or from a
spacecraft near it.

A WISE visit spans ~1.4 days, and observes at ~90 deg solar elongation,
where the observer's acceleration toward the Sun is nearly perpendicular to
the line of sight. Seen from there, a main-belt asteroid's path curves by
~15"/day^2 -- up to ~3" (1 WISE pixel) away from a straight line over a
visit -- and an NEO's by far more (the curvature scales as 1/distance).
Straight-line or ad hoc quadratic paths miss that, so injected asteroids
are integrated in heliocentric space and projected from the observer at
each frame time, light-time corrected.

An asteroid is set up from what is observed -- sky position, rate and
direction at ``t_ref`` -- plus a distance and a radial velocity (neither
of which a single visit measures): its position is the observer's plus
``delta`` along the line of sight, and its velocity the observer's plus
the relative velocity that produces that motion. Positions and velocities
are ICRS (equatorial J2000) heliocentric, AU and AU/day.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

K2 = 0.01720209895 ** 2  # GM_sun, AU^3/day^2
C_AU_DAY = 173.1446326846693
KM_S_TO_AU_DAY = 86400.0 / 149597870.7


def earth_heliocentric(mjd) -> np.ndarray:
    """Earth's heliocentric ICRS position, AU, shape (n, 3) (astropy's
    built-in ephemeris, ~km accuracy)."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(np.atleast_1d(np.asarray(mjd, float)), format="mjd", scale="utc")
    e = get_body_barycentric("earth", t).get_xyz().to_value("AU")
    s = get_body_barycentric("sun", t).get_xyz().to_value("AU")
    return (e - s).T


def earth_heliocentric_velocity(mjd: float, dt: float = 0.01) -> np.ndarray:
    p = earth_heliocentric([mjd - dt, mjd + dt])
    return (p[1] - p[0]) / (2 * dt)


def unit_vector(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def sky_basis(ra_deg: float, dec_deg: float):
    """(line of sight, east, north) unit vectors."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    los = unit_vector(ra_deg, dec_deg)
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
    return los, east, north


def _accel(r):
    return -K2 * r / np.linalg.norm(r) ** 3


def propagate(r0, v0, t0: float, times, step: float = 0.02) -> np.ndarray:
    """Two-body positions at ``times`` (MJD) from the state at ``t0``, by
    fixed-step RK4 (error << 1 km over days for r > 0.3 AU)."""
    times = np.atleast_1d(np.asarray(times, float))
    out = np.zeros((times.size, 3))
    for i, t in enumerate(times):
        r, v = np.array(r0, float), np.array(v0, float)
        span = t - t0
        n = max(1, int(math.ceil(abs(span) / step)))
        h = span / n
        for _ in range(n):
            k1v, k1r = _accel(r), v
            k2v, k2r = _accel(r + 0.5 * h * k1r), v + 0.5 * h * k1v
            k3v, k3r = _accel(r + 0.5 * h * k2r), v + 0.5 * h * k2v
            k4v, k4r = _accel(r + h * k3r), v + h * k3v
            r = r + h / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
            v = v + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[i] = r
    return out


@dataclass
class OrbitMover:
    """An asteroid on a heliocentric two-body orbit. ``rate_deg_day`` and
    ``pa_deg`` describe its apparent motion at ``t_ref`` (as for
    ``motion.Asteroid``); ``delta_au`` and ``vr_km_s`` its distance and
    radial velocity then."""
    id: int
    mag: float
    rate_deg_day: float
    pa_deg: float
    ra0_deg: float
    dec0_deg: float
    delta_au: float
    vr_km_s: float
    t_ref: float
    r0: np.ndarray = field(repr=False)
    v0: np.ndarray = field(repr=False)

    @classmethod
    def from_sky(cls, id: int, mag: float, ra_deg: float, dec_deg: float, rate_deg_day: float,
                 pa_deg: float, delta_au: float, vr_km_s: float, t_ref: float,
                 observer_offset_au=None) -> "OrbitMover":
        obs = earth_heliocentric(t_ref)[0] + (np.zeros(3) if observer_offset_au is None
                                              else np.asarray(observer_offset_au, float))
        v_obs = earth_heliocentric_velocity(t_ref)
        los, east, north = sky_basis(ra_deg, dec_deg)
        w = math.radians(rate_deg_day)  # rad/day
        pa = math.radians(pa_deg)
        v_rel = delta_au * w * (math.sin(pa) * east + math.cos(pa) * north) \
            + vr_km_s * KM_S_TO_AU_DAY * los
        # The state is set at the emission time, so that the light-time
        # corrected position at t_ref is (ra, dec).
        return cls(id=id, mag=mag, rate_deg_day=rate_deg_day, pa_deg=pa_deg, ra0_deg=ra_deg,
                   dec0_deg=dec_deg, delta_au=delta_au, vr_km_s=vr_km_s, t_ref=t_ref,
                   r0=obs + delta_au * los, v0=v_obs + v_rel)

    @property
    def t_state(self) -> float:
        """Epoch of (r0, v0): t_ref minus the light time."""
        return self.t_ref - self.delta_au / C_AU_DAY

    def radec(self, mjd, observer_offset_au=None):
        """Apparent (light-time corrected) RA, Dec in degrees at ``mjd`` (UTC
        MJD, scalar or array), seen from Earth's center plus ``observer_offset_au``
        (ICRS, AU; e.g. WISE's ``ERTH2SC`` vector)."""
        mjd = np.atleast_1d(np.asarray(mjd, float))
        obs = earth_heliocentric(mjd)
        if observer_offset_au is not None:
            obs = obs + np.asarray(observer_offset_au, float)
        t_emit = mjd.copy()
        for _ in range(2):  # light time: the asteroid where it was when the light left
            d = propagate(self.r0, self.v0, self.t_state, t_emit) - obs
            t_emit = mjd - np.linalg.norm(d, axis=1) / C_AU_DAY
        d = propagate(self.r0, self.v0, self.t_state, t_emit) - obs
        ra = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0
        dec = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
        return ra, dec

    def to_dict(self) -> dict:
        return {"id": self.id, "mag": self.mag, "rate_deg_day": self.rate_deg_day,
                "pa_deg": self.pa_deg, "ra0_deg": self.ra0_deg, "dec0_deg": self.dec0_deg,
                "delta_au": self.delta_au, "vr_km_s": self.vr_km_s, "t_ref": self.t_ref,
                "r0_au": self.r0.tolist(), "v0_au_day": self.v0.tolist(),
                "model": "heliocentric two-body, light-time corrected"}


def draw_distance(rate_deg_day: float, rng, v_perp_km_s=(8.0, 20.0),
                  delta_range_au=(0.01, 4.0)) -> float:
    """A distance consistent with a sky rate: delta = v_perp / rate, with a
    relative transverse speed typical of asteroids near quadrature."""
    v = rng.uniform(*v_perp_km_s) * KM_S_TO_AU_DAY
    return float(np.clip(v / math.radians(rate_deg_day), *delta_range_au))
