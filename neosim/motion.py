"""Asteroid population and sky-plane ephemerides.

Each asteroid moves on a quadratic path in the tangent plane centered on the
field target:  p(t) = p0 + v (t - t_ref) + a (t - t_ref)^2 / 2,  with t_ref the
reference frame's mid-exposure time. Over the ~1-day span of a WISE visit (or
a few hours from the ground) main-belt motion is linear to well under a PSF
width; the acceleration term stands in for the curvature of nearby NEOs.

Brightness is the mean magnitude plus a double-peaked rotational light curve,
which matters here because frames are hours apart:
    m(t) = m0 + (A / 2) * cos(4 pi (t - t_ref) / P + phi).
"""

import math
from dataclasses import dataclass, asdict
from typing import List

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


def build_population(pop: PopulationConfig, inst: InstrumentConfig, ra0_deg: float,
                     dec0_deg: float, ref_wcs, np_rng: np.random.Generator) -> List[Asteroid]:
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

    return [Asteroid(id=i, mag=float(mags[i]), rate_deg_day=float(rates[i]),
                     pa_deg=float(pas[i]), xi0=float(xi0[i]), eta0=float(eta0[i]),
                     vxi=float(rate_rad[i] * math.sin(pa_rad[i])),
                     veta=float(rate_rad[i] * math.cos(pa_rad[i])),
                     axi=float(accs[i, 0]), aeta=float(accs[i, 1]),
                     lc_amp=float(amps[i]), lc_period_h=float(periods[i]),
                     lc_phase=float(phases[i]))
            for i in range(n)]
