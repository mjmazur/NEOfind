"""Link single-frame detections into tracklets.

Every pair of detections in two different frames, separated by a rate in
``[v_min, v_max]`` px/day, defines a trial track. It is propagated to every
other frame, where the nearest detection within a tolerance counts as
support. The tolerance follows the propagated position error: with
per-detection error ``sigma_pos`` and a pair spanning ``dt`` centered at
``t_mid``, the prediction error at ``t`` is
``sigma_pos * sqrt(1/2 + 2 ((t - t_mid) / dt)^2)``, so interpolation is held
tightly and extrapolation more loosely (capped at ``r_max``).

Trial tracks with at least ``min_detections`` supporting detections are
refit by least squares (with a bounded acceleration once there are five
points: real paths curve) and re-matched; duplicates (the same object found from
many pairs) are merged by keeping the best-supported candidate among those
sharing two or more detections.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial import cKDTree

from .sources import FrameDetections
from .trackfit import Track


@dataclass
class Tracklet:
    track: Track
    members: List[tuple]  # (frame, detection index)
    sum_z: float


def _lsq_track(frames, xs, ys, times, t_ref, accel_max: float = 0.0,
               curve_resid: float = 1.0) -> Track:
    """Least-squares track: straight, or with a constant acceleration (clipped
    to ``accel_max`` px/day^2 per axis) when there are >= 5 points, or 3-4
    whose straight fit misses one by > ``curve_resid`` px (a curved path: a
    fast NEO's 3 detections can sit 2-7 px off a line)."""
    t = times[frames] - t_ref
    a = np.vstack([np.ones_like(t), t]).T
    (x0, vx), *_ = np.linalg.lstsq(a, xs, rcond=None)
    (y0, vy), *_ = np.linalg.lstsq(a, ys, rcond=None)
    straight = Track(float(x0), float(y0), float(vx), float(vy), t_ref)
    if accel_max <= 0 or len(t) < 3:
        return straight
    if len(t) < 5:
        px, py = straight.position(times[frames])
        if np.max(np.hypot(px - xs, py - ys)) <= curve_resid:
            return straight
    q = np.vstack([np.ones_like(t), t, 0.5 * t * t]).T
    cx, *_ = np.linalg.lstsq(q, xs, rcond=None)
    cy, *_ = np.linalg.lstsq(q, ys, rcond=None)
    if len(t) < 5:
        ax, ay = (float(np.clip(c, -accel_max, accel_max)) for c in (cx[2], cy[2]))
        # Refit position and velocity with the clipped acceleration.
        (x0, vx), *_ = np.linalg.lstsq(a, xs - 0.5 * ax * t * t, rcond=None)
        (y0, vy), *_ = np.linalg.lstsq(a, ys - 0.5 * ay * t * t, rcond=None)
        return Track(float(x0), float(y0), float(vx), float(vy), t_ref, ax, ay)
    if abs(cx[2]) > accel_max or abs(cy[2]) > accel_max:
        return straight
    return Track(float(cx[0]), float(cy[0]), float(cx[1]), float(cy[1]), t_ref,
                 float(cx[2]), float(cy[2]))


def query_radii(tree, px, py, r):
    """Nearest neighbour of each point within its own radius ``r``."""
    r = np.broadcast_to(np.asarray(r, float), px.shape)
    dist, idx = tree.query(np.column_stack([px, py]), distance_upper_bound=float(r.max()))
    out = np.isfinite(dist) & (dist <= r)
    return np.where(out, dist, np.inf), idx


def link(dets: List[FrameDetections], times: np.ndarray, t_ref: float, v_min: float,
         v_max: float, min_detections: int = 3, sigma_pos: float = 0.5,
         n_sigma: float = 3.0, r_max: float = 6.0, min_dt: float = 0.0,
         max_candidates: int = 200000, accel_bound=None, curve_bound=None,
         log=None) -> List[Tracklet]:
    """``accel_bound(speed_px_day)``: the largest path curvature (px/day^2)
    to fit for a tracklet of >= 5 detections moving that fast; None =
    straight tracklets. ``curve_bound(speed_px_day)``: the curvature allowed
    for when matching detections to a pair's chord and fitting 3-4
    detections -- only close, fast objects curve enough to need it, and in
    crowded fields a wider match radius for every rate links chance
    alignments; None = none."""
    log = log or (lambda *a: None)
    n = len(dets)
    trees = [cKDTree(np.column_stack([d.x, d.y])) if len(d) else None for d in dets]
    raw = []
    n_pairs = 0
    frame_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in frame_pairs:
        dt = times[j] - times[i]
        if dt <= min_dt or not len(dets[i]) or not len(dets[j]):
            continue
        di, dj = dets[i], dets[j]
        ddx = dj.x[None, :] - di.x[:, None]
        ddy = dj.y[None, :] - di.y[:, None]
        rate = np.hypot(ddx, ddy) / dt
        ii, jj = np.nonzero((rate >= v_min) & (rate <= v_max))
        if ii.size == 0:
            continue
        n_pairs += ii.size
        vx, vy = ddx[ii, jj] / dt, ddy[ii, jj] / dt
        t_mid = 0.5 * (times[i] + times[j])
        xm = 0.5 * (di.x[ii] + dj.x[jj])
        ym = 0.5 * (di.y[ii] + dj.y[jj])
        support = np.full(ii.size, 2)
        matches = {i: ii, j: jj}
        # A curved path leaves the chord through the pair by up to
        # a_max/2 |t_k - t_i| |t_k - t_j| at frame k.
        a_pair = (np.array([curve_bound(v) for v in rate[ii, jj]]) if curve_bound
                  else np.zeros(ii.size))
        for k in range(n):
            if k in (i, j) or trees[k] is None:
                continue
            tk = times[k] - t_mid
            r = min(r_max, n_sigma * sigma_pos * np.sqrt(0.5 + 2.0 * (tk / dt) ** 2))
            r = r + 0.5 * np.sqrt(2.0) * a_pair * abs((times[k] - times[i]) * (times[k] - times[j]))
            px, py = xm + vx * tk, ym + vy * tk
            dist, idx = query_radii(trees[k], px, py, r)
            hit = np.isfinite(dist)
            support += hit
            matches[k] = np.where(hit, idx, -1)
        good = np.flatnonzero(support >= min_detections)
        for g in good:
            members = [(k, int(m[g])) for k, m in sorted(matches.items()) if m[g] >= 0]
            raw.append(members)
        if len(raw) > max_candidates:
            log(f"linking: stopped at {max_candidates} candidates")
            break

    # Refit, re-match with the fitted track, dedupe.
    cands = []
    seen = set()
    for members in raw:
        key = tuple(members)
        if key in seen:
            continue
        seen.add(key)
        for _ in range(2):
            fr = np.array([k for k, _ in members])
            xs = np.array([dets[k].x[m] for k, m in members])
            ys = np.array([dets[k].y[m] for k, m in members])
            v_fit = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) / max(times[fr[-1]] - times[fr[0]], 1e-6)
            if len(members) >= 5:
                amax = accel_bound(v_fit) if accel_bound else 0.0
            else:
                amax = curve_bound(v_fit) if curve_bound else 0.0
            tr = _lsq_track(fr, xs, ys, times, t_ref, amax, curve_resid=2.0 * sigma_pos)
            new = []
            for k in range(n):
                if trees[k] is None:
                    continue
                px, py = tr.position(times[k])
                dist, idx = trees[k].query([px, py], distance_upper_bound=n_sigma * sigma_pos * 1.5)
                if np.isfinite(dist):
                    new.append((k, int(idx)))
            if len(new) < min_detections or new == members:
                break
            members = new
        if len(members) < min_detections or tr.speed < v_min or tr.speed > v_max:
            continue
        sum_z = float(sum(dets[k].z[m] for k, m in members))
        cands.append(Tracklet(tr, members, sum_z))

    cands.sort(key=lambda c: (-len(c.members), -c.sum_z))
    kept: List[Tracklet] = []
    used_by = {}
    for c in cands:
        overlap = {}
        for mem in c.members:
            for owner in used_by.get(mem, ()):
                overlap[owner] = overlap.get(owner, 0) + 1
        if any(v >= 2 for v in overlap.values()):
            continue
        idx = len(kept)
        kept.append(c)
        for mem in c.members:
            used_by.setdefault(mem, []).append(idx)
    log(f"linking: {n_pairs} pairs, {len(raw)} supported trial tracks, {len(kept)} tracklets")
    return kept
