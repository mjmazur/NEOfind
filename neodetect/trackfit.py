"""Coherent multi-frame matched filter for a moving point source.

A track is a position ``(x0, y0)`` on the common grid at reference time
``t_ref`` (MJD) plus a velocity ``(vx, vy)`` in grid px/day and an
acceleration ``(ax, ay)`` in px/day^2. In frame k (mid-exposure time t_k)
the source sits at ``p_k = p0 + v dt + a dt^2 / 2``, ``dt = t_k - t_ref``.
Seeds are straight; ``refine`` can fit the acceleration: seen from Earth
(or WISE), a main-belt asteroid's path curves by ~15"/day^2, up to ~1 WISE
pixel off a straight line over a ~1.4-day visit, and an NEO's by more.
Per frame, with the frame's PSF stamp P placed exactly at p_k,

    num_k = <w_k r_k, P>      den_k = <w_k, P^2>

so ``num_k/den_k`` is the PSF-fit flux and ``num_k/sqrt(den_k)`` its SNR z_k.
For a constant-brightness source the optimal combination is

    S = sum_k num_k / sqrt(sum_k den_k),

N(0, 1) under noise for any fixed hypothesis (the weights are calibrated in
``preprocess``). Asteroids vary (rotation, phase) and single frames carry
artifacts (unmasked cosmic rays), so the **robust score** first drops frames
whose flux f_k exceeds the median flux F by more than
``clip * sigma_k + frac_tol * F`` -- measurement noise plus a fixed allowance
for real variability (rotation light curves) -- and then evaluates S over
the remaining frames. Design choices, each from a false alarm seen in
simulations:

* only *high* outliers are dropped: artifacts add flux, and a frame showing
  significantly negative flux is evidence against the hypothesis, which the
  consistency test (``consistency_p``) must get to see;
* the median is unweighted, so one high-weight artifact frame cannot define
  the reference everyone else is compared with;
* the allowance is additive: scaling it by ``clip`` would let a bright
  artifact frame "agree" with frames several times fainter.

``bilinear_sums`` approximates num/den from the psi/phi maps for fast
triage of many hypotheses; final scores always use exact stamps.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.stats import chi2

from .preprocess import Preprocessed
from .psf import clip_stamp, stamp

MIN_PSF_COVERAGE = 0.6
SOLID_COVERAGE = 0.9  # frames that constrain a final track's geometry (``refit_geometry``)


@dataclass
class Track:
    x0: float
    y0: float
    vx: float  # grid px / day
    vy: float
    t_ref: float  # MJD
    ax: float = 0.0  # grid px / day^2
    ay: float = 0.0

    def position(self, t):
        dt = np.asarray(t, dtype=float) - self.t_ref
        return (self.x0 + self.vx * dt + 0.5 * self.ax * dt * dt,
                self.y0 + self.vy * dt + 0.5 * self.ay * dt * dt)

    def velocity(self, t):
        dt = np.asarray(t, dtype=float) - self.t_ref
        return self.vx + self.ax * dt, self.vy + self.ay * dt

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def params(self):
        return np.array([self.x0, self.y0, self.vx, self.vy, self.ax, self.ay])

    def with_params(self, p) -> "Track":
        """Track from 4 (straight) or 6 parameters."""
        a = (float(p[4]), float(p[5])) if len(p) > 4 else (0.0, 0.0)
        return Track(float(p[0]), float(p[1]), float(p[2]), float(p[3]), self.t_ref, *a)


@dataclass
class TrackScore:
    track: Track
    score: float  # robust S
    score_all: float  # S over every covered frame (no clipping)
    used: np.ndarray  # bool per frame: included in the robust score
    covered: np.ndarray  # bool per frame: den > 0
    num: np.ndarray = field(repr=False)
    den: np.ndarray = field(repr=False)

    @property
    def n_used(self) -> int:
        return int(self.used.sum())

    @property
    def flux(self) -> float:
        d = float(self.den[self.used].sum())
        return float(self.num[self.used].sum()) / d if d > 0 else 0.0

    def consistency_p(self, frac_tol: float) -> float:
        """p-value of the used frames' fluxes being one constant flux plus real
        variability of rms ``frac_tol / 2`` (``frac_tol`` is the maximum
        deviation; a vetting statistic: chance alignments of unrelated
        single-frame peaks are wildly inconsistent)."""
        u = self.used
        if u.sum() < 2:
            return 1.0
        f = self.num[u] / self.den[u]
        F = self.flux
        var = 1.0 / self.den[u] + (0.5 * frac_tol * max(F, 0.0)) ** 2
        return float(chi2.sf(float(np.sum((f - F) ** 2 / var)), int(u.sum()) - 1))

    @property
    def z(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.den > 0, self.num / np.sqrt(self.den), np.nan)

    @property
    def frame_flux(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.den > 0, self.num / self.den, np.nan)

    @property
    def frame_flux_err(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(self.den > 0, 1.0 / np.sqrt(self.den), np.nan)


def robust_combine(num, den, clip: float = 4.0, frac_tol: float = 0.35,
                   min_frames: int = 3):
    """(robust S, used mask) from per-frame num/den; S = -inf if too few frames."""
    covered = den > 0
    used = covered.copy()
    if covered.sum() < min_frames:
        return -math.inf, used
    f = np.where(covered, num / np.where(covered, den, 1.0), 0.0)
    for _ in range(3):
        f_ref = float(np.median(f[used]))
        tol = clip / np.sqrt(np.where(covered, den, 1.0)) + frac_tol * max(f_ref, 0.0)
        new = covered & (f - f_ref <= tol)
        if new.sum() < min_frames or np.array_equal(new, used):
            used = new if new.sum() >= min_frames else used
            break
        used = new
    d = den[used].sum()
    s = float(num[used].sum() / math.sqrt(d)) if d > 0 else -math.inf
    return s, used


class TrackEvaluator:
    def __init__(self, pre: Preprocessed, clip: float = 4.0, frac_tol: float = 0.35,
                 min_frames: int = 3):
        self.pre = pre
        self.n = pre.n
        self.shape = pre.resid.shape[1:]
        self.times = pre.times
        self.clip, self.frac_tol, self.min_frames = clip, frac_tol, min_frames

    def frame_sums(self, track: Track):
        num = np.zeros(self.n)
        den = np.zeros(self.n)
        xs, ys = track.position(self.times)
        for k in range(self.n):
            st, r0, c0 = stamp(xs[k], ys[k], self.pre.psf_sigma_px[k])
            clipped = clip_stamp(st, r0, c0, self.shape)
            if clipped is None:
                continue
            t, (a, b, c, d) = clipped
            w = self.pre.weight[k, a:b, c:d]
            t2 = t * t
            if float((t2 * (w > 0)).sum()) < MIN_PSF_COVERAGE * float((st * st).sum()):
                continue
            num[k] = float((w * self.pre.resid[k, a:b, c:d] * t).sum())
            den[k] = float((w * t2).sum())
        return num, den

    def frame_coverage(self, track: Track) -> np.ndarray:
        """Per frame, the fraction of the PSF's squared weight on valid pixels
        at the track position (0 off the grid)."""
        cov = np.zeros(self.n)
        xs, ys = track.position(self.times)
        for k in range(self.n):
            st, r0, c0 = stamp(xs[k], ys[k], self.pre.psf_sigma_px[k])
            clipped = clip_stamp(st, r0, c0, self.shape)
            if clipped is None:
                continue
            t, (a, b, c, d) = clipped
            cov[k] = float((t * t * (self.pre.weight[k, a:b, c:d] > 0)).sum()) / float((st * st).sum())
        return cov

    def refit_geometry(self, best: "TrackScore", accel_max: float = 0.0) -> "TrackScore":
        """Re-fit a refined track's path on its fully covered frames only.

        A frame whose PSF footprint is partly masked counts only while the
        track keeps >= ``MIN_PSF_COVERAGE`` of it valid, so maximizing S can
        bend the path toward valid pixels to keep such a frame: on (120)
        Lachesis, whose core is masked in several frames, that bent the
        fitted curvature by 9"/day^2 and the positions by up to 0.8". Here
        the path is fitted to the used frames with >= ``SOLID_COVERAGE``
        (enough of them for the parameters), then the score is recomputed
        on all frames for that path."""
        cov = self.frame_coverage(best.track)
        solid = best.used & (cov >= SOLID_COVERAGE)
        fit_a = accel_max > 0
        if solid.sum() == best.used.sum() or solid.sum() < (4 if fit_a else self.min_frames):
            return best
        tau = max(float(np.max(np.abs(self.times - best.track.t_ref))), 1e-3)
        accel_max = max(accel_max, abs(best.track.ax), abs(best.track.ay))
        npar = 6 if fit_a else 4
        scale = np.array([0.5, 0.5, 0.5 / tau, 0.5 / tau, 1.0 / tau ** 2, 1.0 / tau ** 2])[:npar]
        p0 = best.track.params()[:npar]

        def params(u):
            p = p0 + u * scale
            if fit_a:
                p[4:] = np.clip(p[4:], -accel_max, accel_max)
            return p

        def neg(u):
            sc = self.score(best.track.with_params(params(u)), used=solid)
            return -sc.score if sc is not None else 1e3

        res = minimize(neg, np.zeros(npar), method="Nelder-Mead",
                       options={"xatol": 0.02, "fatol": 1e-3, "maxfev": 150 * npar,
                                "initial_simplex": np.vstack([np.zeros(npar), np.eye(npar)])})
        out = self.score(best.track.with_params(params(res.x)))
        return out if out is not None else best

    def score(self, track: Track, used=None) -> Optional[TrackScore]:
        num, den = self.frame_sums(track)
        covered = den > 0
        if used is None:
            s, used = robust_combine(num, den, self.clip, self.frac_tol, self.min_frames)
        else:
            used = used & covered
            d = den[used].sum()
            s = float(num[used].sum() / math.sqrt(d)) if (d > 0 and used.sum() >= self.min_frames) \
                else -math.inf
        if not math.isfinite(s):
            return None
        d_all = den[covered].sum()
        return TrackScore(track=track, score=s, score_all=float(num[covered].sum() / math.sqrt(d_all)),
                          used=used, covered=covered, num=num, den=den)

    def refine(self, track: Track, rounds: int = 3, accel_max: float = 0.0) -> Optional[TrackScore]:
        """Maximize S over (x0, y0, vx, vy) -- and (ax, ay), bounded by
        ``accel_max`` px/day^2 per axis (or the seed's own, if larger), when
        that is > 0 -- re-selecting frames between rounds."""
        best = self.score(track)
        if best is None:
            return None
        tau = max(float(np.max(np.abs(self.times - track.t_ref))), 1e-3)
        fit_a = accel_max > 0
        accel_max = max(accel_max, abs(track.ax), abs(track.ay))
        npar = 6 if fit_a else 4
        # Steps that move the end frames by ~0.5 px.
        scale = np.array([0.5, 0.5, 0.5 / tau, 0.5 / tau, 1.0 / tau ** 2, 1.0 / tau ** 2])[:npar]
        for _ in range(rounds):
            used = best.used
            p0 = best.track.params()[:npar]

            def params(u):
                p = p0 + u * scale
                if fit_a:
                    p[4:] = np.clip(p[4:], -accel_max, accel_max)
                return p

            def neg(u):
                sc = self.score(track.with_params(params(u)), used=used)
                return -sc.score if sc is not None else 1e3

            res = minimize(neg, np.zeros(npar), method="Nelder-Mead",
                           options={"xatol": 0.02, "fatol": 1e-3, "maxfev": 100 * npar,
                                    "initial_simplex": np.vstack([np.zeros(npar), np.eye(npar)])})
            cand = self.score(track.with_params(params(res.x)))
            if cand is None or cand.score <= best.score + 1e-3:
                break
            best = cand
        return best

    def bilinear_sums(self, x0, y0, vx, vy, t_ref, ax=0.0, ay=0.0):
        """Approximate (num, den), shape (m, n), for m hypotheses from psi/phi."""
        x0, y0, vx, vy = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (x0, y0, vx, vy))
        ax, ay = (np.broadcast_to(np.asarray(a, dtype=float), x0.shape) for a in (ax, ay))
        m = x0.size
        num = np.zeros((m, self.n))
        den = np.zeros((m, self.n))
        h, w = self.shape
        for k in range(self.n):
            dt = self.times[k] - t_ref
            x, y = x0 + vx * dt + 0.5 * ax * dt * dt, y0 + vy * dt + 0.5 * ay * dt * dt
            inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
            if not inside.any():
                continue
            coords = np.array([y[inside], x[inside]])
            ph = map_coordinates(self.pre.phi[k], coords, order=1)
            ps = map_coordinates(self.pre.psi[k], coords, order=1)
            # Interpolating across a coverage edge mixes in zeros; require all four corners.
            ok = map_coordinates((self.pre.phi[k] > 0).astype(np.float32), coords, order=1) > 0.999
            idx = np.flatnonzero(inside)
            num[idx[ok], k] = ps[ok]
            den[idx[ok], k] = ph[ok]
        return num, den
