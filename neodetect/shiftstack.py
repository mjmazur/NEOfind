"""Shift-and-stack (synthetic tracking) over a grid of velocities.

For each velocity hypothesis every frame's likelihood maps are shifted to
where the source would be at t_ref and summed (see ``backend``). Integer-pixel
shifts keep this a pure streaming sum; the velocity grid spacing is chosen so
the worst-case positional mismatch at the most distant frame is
``step_px`` (default 1 px, ~1 PSF sigma for WISE), and the exact track fit
recovers the residual loss afterwards.

Per-frame contributions are clipped at ``zclip`` sigma before stacking, so
no single frame (an unmasked cosmic ray, a subtraction residual) can carry a
hypothesis: one frame adds at most ``zclip * sqrt(phi_k)`` to the numerator.
Genuinely bright asteroids lose stacked SNR from the clipping but are far
above threshold anyway (and the linker finds them). Clipping also bounds what
m frames can reach, ``zclip * sqrt(m)``, so a threshold above
``zclip * sqrt(min_frames - 1)`` guarantees every candidate uses at least
``min_frames`` frames.
"""

import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage

from .backend import get_stack_max
from .preprocess import Preprocessed
from .trackfit import Track


@dataclass
class StackCandidate:
    track: Track
    score: float  # stacked (clipped, integer-shift) SNR


def velocity_grid(times, t_ref, v_min, v_max, step_px=1.0):
    """Velocities (px/day) on a square grid inside the annulus, deduplicated by
    their per-frame integer offsets. Returns (vx, vy, dx, dy)."""
    dt = np.asarray(times) - t_ref
    tau = max(float(np.max(np.abs(dt))), 1e-6)
    dv = step_px / tau
    m = int(math.ceil(v_max / dv))
    g = np.arange(-m, m + 1) * dv
    vx, vy = np.meshgrid(g, g)
    vx, vy = vx.ravel(), vy.ravel()
    sp = np.hypot(vx, vy)
    keep = (sp >= v_min) & (sp <= v_max)
    vx, vy = vx[keep], vy[keep]
    dx = np.rint(vx[:, None] * dt[None, :]).astype(np.int32)
    dy = np.rint(vy[:, None] * dt[None, :]).astype(np.int32)
    _, first = np.unique(np.hstack([dx, dy]), axis=0, return_index=True)
    first = np.sort(first)
    return vx[first], vy[first], dx[first], dy[first]


def shift_and_stack(pre: Preprocessed, t_ref: float, v_min: float, v_max: float,
                    threshold: float, zclip: float = 4.0, step_px: float = 1.0,
                    min_frames: int = 3, max_candidates: int = 2000,
                    backend: str = "auto", threads: int = 0, log=None) -> List[StackCandidate]:
    log = log or (lambda *a: None)
    floor = zclip * math.sqrt(max(min_frames - 1, 1))
    if threshold <= floor:
        raise ValueError(f"stack threshold {threshold} must exceed zclip*sqrt(min_frames-1) "
                         f"= {floor:.2f} to enforce min_frames")
    vx, vy, dx, dy = velocity_grid(pre.times, t_ref, v_min, v_max, step_px)
    with np.errstate(invalid="ignore"):
        lim = zclip * np.sqrt(pre.phi)
    psi_c = np.clip(pre.psi, -lim, lim).astype(np.float32)
    del lim
    valid = (pre.phi > 0).astype(np.float32)
    fn = get_stack_max(backend, threads)
    t = time.time()
    best_s, best_v = fn(psi_c, pre.phi.astype(np.float32), valid, dx, dy, min_frames)
    del psi_c, valid
    elapsed = time.time() - t

    # Only positive stacked SNRs are tracked; everything else is -inf.
    peaks = (best_s >= threshold) & (best_s == ndimage.maximum_filter(best_s, size=5))
    ys, xs = np.nonzero(peaks)
    s = best_s[ys, xs]
    order = np.argsort(-s)[:max_candidates]
    cands = [StackCandidate(Track(float(xs[i]), float(ys[i]), float(vx[best_v[ys[i], xs[i]]]),
                                  float(vy[best_v[ys[i], xs[i]]]), t_ref), float(s[i]))
             for i in order]
    log(f"shift-and-stack: {len(vx)} velocities ({v_min:.0f}-{v_max:.0f} px/day), "
        f"{elapsed:.1f}s, {int(peaks.sum())} peaks >= {threshold}, max {np.max(best_s):.1f}")
    return cands
