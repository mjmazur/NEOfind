"""Asteroid population and sky-plane ephemerides.

Each asteroid moves on a quadratic path in the tangent plane centered on the
field target:  p(t) = p0 + v (t - t_ref) + a (t - t_ref)^2 / 2,  with t_ref the
reference frame's mid-exposure time; the acceleration term is an ad hoc
curvature (``--accel``). With ``motion = "orbit"`` asteroids instead move on
heliocentric two-body orbits seen from Earth (``orbit.OrbitMover``), set up
from the same drawn position, rate and direction at t_ref: their paths curve
as real ones do (~15"/day^2 for a main-belt asteroid at quadrature, up to
~3" from a straight line over a 1.4-day WISE visit; NEOs far more), which a
detector fitting curvature has to be tested on.

Brightness is the mean magnitude plus a double-peaked rotational light curve,
which matters here because frames are hours apart:
    m(t) = m0 + (A / 2) * cos(4 pi (t - t_ref) / P + phi).
"""

import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np

from .astrometry import equatorial_to_tangent, tangent_to_equatorial
from .config import InstrumentConfig, PopulationConfig

ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


@dataclass
class Asteroid:
    id: int
    mag: float  # mean magnitude
    rate_deg_day: float
    pa_deg: float  # direction of motion, east of north
    xi0: float  # tangent-plane position at t_ref, rad (east)
    eta0: float  # rad (north)
    vxi: float  # rad/day
    veta: float
    axi: float  # rad/day^2
    aeta: float
    lc_amp: float  # peak-to-peak, mag
    lc_period_h: float
    lc_phase: float  # rad

    def tangent(self, t_mjd, t_ref: float):
        dt = np.asarray(t_mjd, dtype=float) - t_ref
        xi = self.xi0 + self.vxi * dt + 0.5 * self.axi * dt * dt
        eta = self.eta0 + self.veta * dt + 0.5 * self.aeta * dt * dt
        return xi, eta

    def radec(self, t_mjd, t_ref: float, ra0_deg: float, dec0_deg: float):
        xi, eta = self.tangent(t_mjd, t_ref)
        return tangent_to_equatorial(xi, eta, ra0_deg, dec0_deg)

    def mag_at(self, t_mjd, t_ref: float):
        dt_h = (np.asarray(t_mjd, dtype=float) - t_ref) * 24.0
        return self.mag + 0.5 * self.lc_amp * np.cos(
            4.0 * math.pi * dt_h / self.lc_period_h + self.lc_phase)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accel_arcsec_hr2"] = [self.axi * ARCSEC_PER_RAD / 576.0,
                                 self.aeta * ARCSEC_PER_RAD / 576.0]
        return d


@dataclass
class OrbitAsteroid(Asteroid):
    """An ``Asteroid`` whose sky path comes from a two-body orbit (``mover``);
    the tangent-plane terms describe its motion at t_ref only."""
    mover: Optional[object] = field(default=None, repr=False)

    def radec(self, t_mjd, t_ref: float, ra0_deg: float, dec0_deg: float):
        ra, dec = self.mover.radec(t_mjd)
        if np.ndim(t_mjd) == 0:
            return float(ra[0]), float(dec[0])
        return ra, dec

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in Asteroid.__dataclass_fields__}
        d["accel_arcsec_hr2"] = [0.0, 0.0]  # not quadratic: see "orbit"
        d["orbit"] = self.mover.to_dict()
        return d


def _orbit_movers(asteroids: List[Asteroid], ra0_deg, dec0_deg, t_ref, np_rng) -> List[Asteroid]:
    """Two-body versions of ``asteroids`` (distance from the rate; radial
    velocity the observer's own along the line of sight plus up to 10 km/s,
    as for ``neowise.inject``)."""
    from .orbit import (KM_S_TO_AU_DAY, OrbitMover, draw_distance,
                        earth_heliocentric_velocity, sky_basis)
    v_obs = earth_heliocentric_velocity(t_ref)
    out = []
    for a in asteroids:
        ra, dec = (float(v) for v in tangent_to_equatorial(a.xi0, a.eta0, ra0_deg, dec0_deg))
        los = sky_basis(ra, dec)[0]
        vr = -float(v_obs @ los) / KM_S_TO_AU_DAY + np_rng.uniform(-10.0, 10.0)
        mover = OrbitMover.from_sky(a.id, a.mag, ra, dec, a.rate_deg_day, a.pa_deg,
                                    draw_distance(a.rate_deg_day, np_rng), vr, t_ref)
        out.append(OrbitAsteroid(**{k: getattr(a, k) for k in Asteroid.__dataclass_fields__},
                                 mover=mover))
    return out


def build_population(pop: PopulationConfig, inst: InstrumentConfig, ra0_deg: float,
                     dec0_deg: float, ref_wcs, np_rng: np.random.Generator,
                     t_ref: Optional[float] = None) -> List[Asteroid]:
    n = pop.n_asteroids
    if n <= 0:
        return []

    mag_lo = pop.mag_min if pop.mag_min is not None else inst.limiting_mag - 4.0
    mag_hi = pop.mag_max if pop.mag_max is not None else inst.limiting_mag + 0.5
    mags = np.full(n, pop.mag) if pop.mag is not None else np_rng.uniform(mag_lo, mag_hi, n)

    if pop.rate_deg_day is not None:
        rates = np.full(n, pop.rate_deg_day)
    else:
        lo, hi = math.log(pop.rate_min_deg_day), math.log(pop.rate_max_deg_day)
        rates = np.exp(np_rng.uniform(lo, hi, n))
    pas = np.full(n, pop.pa_deg) if pop.pa_deg is not None else np_rng.uniform(0.0, 360.0, n)

    if pop.placement_box_arcmin is not None:
        half = math.radians(pop.placement_box_arcmin / 60.0) / 2.0
        xi0 = np_rng.uniform(-half, half, n)
        eta0 = np_rng.uniform(-half, half, n)
    else:
        # Uniform over the central 90% of the reference frame.
        x = np_rng.uniform(0.05, 0.95, n) * inst.width - 0.5
        y = np_rng.uniform(0.05, 0.95, n) * inst.height - 0.5
        ra, dec = ref_wcs.all_pix2world(x, y, 0)
        xi0, eta0 = equatorial_to_tangent(ra, dec, ra0_deg, dec0_deg)

    rate_rad = np.radians(rates)
    pa_rad = np.radians(pas)
    acc = pop.accel_rms_arcsec_hr2 * 576.0 / ARCSEC_PER_RAD  # arcsec/hr^2 -> rad/day^2
    accs = np_rng.normal(0.0, acc, size=(n, 2)) if acc > 0 else np.zeros((n, 2))
    amps = np_rng.uniform(0.0, pop.lc_amp_max, n)
    periods = np.exp(np_rng.uniform(math.log(pop.lc_period_min_h),
                                    math.log(pop.lc_period_max_h), n))
    phases = np_rng.uniform(0.0, 2.0 * math.pi, n)

    out = [Asteroid(id=i, mag=float(mags[i]), rate_deg_day=float(rates[i]),
                     pa_deg=float(pas[i]), xi0=float(xi0[i]), eta0=float(eta0[i]),
                     vxi=float(rate_rad[i] * math.sin(pa_rad[i])),
                     veta=float(rate_rad[i] * math.cos(pa_rad[i])),
                     axi=float(accs[i, 0]), aeta=float(accs[i, 1]),
                     lc_amp=float(amps[i]), lc_period_h=float(periods[i]),
                     lc_phase=float(phases[i]))
           for i in range(n)]
    if pop.motion == "orbit":
        if t_ref is None:
            raise ValueError("orbit motion needs t_ref")
        out = _orbit_movers(out, ra0_deg, dec0_deg, t_ref, np_rng)
    elif pop.motion != "linear":
        raise ValueError(f"unknown motion {pop.motion!r} (linear, orbit)")
    return out
