"""Coherent multi-frame matched filter for a moving point source.

A track is a position ``(x0, y0)`` on the common grid at reference time
``t_ref`` (MJD) plus a velocity ``(vx, vy)`` in grid px/day. In frame k
(mid-exposure time t_k) the source sits at ``p_k = p0 + v (t_k - t_ref)``.
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


@dataclass
class Track:
    x0: float
    y0: float
    vx: float  # grid px / day
    vy: float
    t_ref: float  # MJD

    def position(self, t):
        dt = np.asarray(t, dtype=float) - self.t_ref
        return self.x0 + self.vx * dt, self.y0 + self.vy * dt

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def params(self):
        return np.array([self.x0, self.y0, self.vx, self.vy])

    def with_params(self, p) -> "Track":
        return Track(float(p[0]), float(p[1]), float(p[2]), float(p[3]), self.t_ref)


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

    def refine(self, track: Track, rounds: int = 3) -> Optional[TrackScore]:
        """Maximize S over (x0, y0, vx, vy), re-selecting frames between rounds."""
        best = self.score(track)
        if best is None:
            return None
        tau = max(float(np.max(np.abs(self.times - track.t_ref))), 1e-3)
        scale = np.array([0.5, 0.5, 0.5 / tau, 0.5 / tau])
        for _ in range(rounds):
            used = best.used
            p0 = best.track.params()

            def neg(u):
                sc = self.score(track.with_params(p0 + u * scale), used=used)
                return -sc.score if sc is not None else 1e3

            res = minimize(neg, np.zeros(4), method="Nelder-Mead",
                           options={"xatol": 0.02, "fatol": 1e-3, "maxfev": 400,
                                    "initial_simplex": np.vstack([np.zeros(4), np.eye(4)])})
            cand = self.score(track.with_params(p0 + res.x * scale))
            if cand is None or cand.score <= best.score + 1e-3:
                break
            best = cand
        return best

    def bilinear_sums(self, x0, y0, vx, vy, t_ref):
        """Approximate (num, den), shape (m, n), for m hypotheses from psi/phi."""
        x0, y0, vx, vy = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (x0, y0, vx, vy))
        m = x0.size
        num = np.zeros((m, self.n))
        den = np.zeros((m, self.n))
        h, w = self.shape
        for k in range(self.n):
            dt = self.times[k] - t_ref
            x, y = x0 + vx * dt, y0 + vy * dt
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
