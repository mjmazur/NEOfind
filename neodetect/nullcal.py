"""Per-visit detection thresholds from time-scrambled copies of the visit.

On real data the false-alarm tail varies far more between visits than one
global threshold can follow: on the M3 real-data sweep the highest score on
a time-scrambled (asteroid-free) copy was ~7.5-10 on most visits but 17-27
on one with a bright, artifact-rich field. A global threshold set for a
fixed false-alarm rate is then dictated by the worst visits and costs
completeness everywhere else.

Instead each visit's thresholds scale with its own *null level*: the
``NULL_RANK``-th highest candidate score on a time-scrambled copy (the
median over copies), a mid-tail statistic that one copy already measures
well (thresholds from one copy and from two agree to ~0.1). Per frame-count
class ``c``

    threshold_c = null level x NULL_RATIO[c]

with the ratios calibrated on a sweep (``calibrate_ratios``) so that the
false-alarm rate on held-out scrambled copies -- each scored with the level
measured on the *other* copies of its visit, so the level's own noise is
included -- meets the target. On real NEOWISE (``sims/real_v1``) with the
early M4 detector this raised completeness at W1 15.5-16 from 0.38 to 0.50
at the same false-alarm rate, with thresholds for >= 5 frames from 9.5
(clean visits) to 20. The final M4 detector's artifact handling (streak
mask, veto, star-residual and single-frame vetting) evens the visits out
(levels 6.7-8.2, >= 5-frame thresholds 9.8-11.9), and per-visit and global
thresholds then perform alike (0.62 vs 0.68 at W1 15.5-16); held out one
visit at a time, per-visit thresholds gave twice the false rate of global
ones (0.089 vs 0.044 per copy).

A second sample of 15 visits (sims/real_v3: 11 in the Galactic plane, the
bulge, off-ecliptic fields) reversed that: plane fields have null levels of
9-12 and the bulge 11-16, against 6.5-8 elsewhere, and global thresholds
calibrated on the ecliptic visits gave ~2 false detections per scrambled
copy there. Held out one visit at a time over all 30 visits, per-visit
thresholds from one copy gave 0.067 false per copy with completeness 0.65
at W1 15.5-16 off the plane (0.15 in it); global thresholds (dominated by
the plane) gave 0.35 off the plane, and the guard below -- global
thresholds raised by level / GUARD_LEVEL -- the same. So per-visit mode is
the real-WISE default again (one copy; a second didn't help). Guard mode
remains available.

In production ``self_calibrate`` scrambles the frame times of the sequence
being searched, runs the detector on the copies, and returns the thresholds
(``neodetect detect --null-scrambles N``, default 1 for real WISE frames);
each copy costs one detector run.

**Artifact veto.** Some false tracks recur at the same place in every
scrambled copy: a fixed artifact (scattered light along a cutout edge in a
crowded field) lines up for any frame order, so it is not a moving object.
A candidate within ``VETO_RADIUS_ARCSEC`` (at the reference time) of a
scrambled-copy candidate scoring >= ``VETO_MIN_RATIO`` x the null level is
vetoed. Ratio calibration applies the same veto to each held-out copy using
the other copies, so one recurring artifact no longer sets the ratios, and
``evaluate`` measures what the veto costs in completeness.
"""

import dataclasses
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import numpy as np

FRAME_CLASSES = ("3", "4", "5+")
NULL_RANK = 5
# Calibrated on real NEOWISE W1 single exposures: 30 visits x 3 scrambled
# copies (sims/real_v1 along the ecliptic, sims/real_v3 incl. 11 Galactic-
# plane fields), final M4 detector, held-out copies, <= 0.05 false per visit.
# Held out one visit at a time (calibrated on the other 29): 0.067 false per
# copy with the level from one copy. Recalibrate with ``neodetect evaluate``
# (``per_visit.ratios``) after detector changes.
NULL_RATIO = {"3": 1.81, "4": 1.62, "5+": 1.66}
# Guard mode: a visit's thresholds are raised by level / GUARD_LEVEL when
# its null level (one scrambled copy) exceeds the highest single-copy level
# of the calibration sample (sims/real_v1, final M4 detector: 8.76 in the
# bulge visit v011 with curved linking, 8.39 without; median 7.1). Held out
# per visit this matched global thresholds' false rate (0.044 per copy)
# where per-visit thresholds gave 0.089.
GUARD_LEVEL = 8.8
VETO_RADIUS_ARCSEC = 60.0
VETO_MIN_RATIO = 1.2


def frame_class(n_used: int) -> str:
    return "3" if n_used <= 3 else ("4" if n_used == 4 else "5+")


def null_level(scramble_scores: Seq[Seq[float]], rank: int = NULL_RANK,
               floor: float = 5.0) -> float:
    """Median over scrambled copies of the ``rank``-th highest candidate
    score (``floor``, the report threshold, when a copy has fewer)."""
    vals = []
    for scores in scramble_scores:
        s = sorted(scores, reverse=True)
        vals.append(s[rank - 1] if len(s) >= rank else floor)
    return float(np.median(vals)) if vals else float("nan")


def thresholds(level: float, ratios: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    ratios = ratios or NULL_RATIO
    return {c: float(level * ratios[c]) for c in FRAME_CLASSES}


def passes(score: float, n_used: int, thr: Dict[str, float]) -> bool:
    return score >= thr[frame_class(n_used)]


def veto_sites(copies, level: float, min_ratio: float = VETO_MIN_RATIO) -> np.ndarray:
    """(ra, dec) of the scrambled-copy candidates that can veto: ``copies``
    is a list of [(score, n_used, ra_deg, dec_deg), ...] per copy."""
    sites = [(c[2], c[3]) for cands in copies for c in cands
             if len(c) >= 4 and c[0] >= min_ratio * level]
    return np.asarray(sites, float).reshape(-1, 2)


def is_vetoed(ra: float, dec: float, sites: np.ndarray,
              radius_arcsec: float = VETO_RADIUS_ARCSEC) -> bool:
    if not len(sites):
        return False
    dra = (sites[:, 0] - ra + 180.0) % 360.0 - 180.0
    d = np.hypot(dra * np.cos(np.radians(dec)), sites[:, 1] - dec) * 3600.0
    return bool(np.any(d < radius_arcsec))


def calibrate_ratios(visit_nulls: Dict[str, List[List[Tuple]]], target_far: float,
                     rank: int = NULL_RANK, veto: bool = True) -> Optional[dict]:
    """Ratios per frame class from scrambled copies, ``visit -> [[(score,
    n_used[, ra_deg, dec_deg]), ...] per copy]`` (positions enable the veto). Each copy's candidates are scaled by the level
    of the other copies of its visit (leave one out), and each class gets a
    third of the ``target_far`` budget (false detections per copy). None
    if no visit has two copies."""
    scaled = {c: [] for c in FRAME_CLASSES}
    n_copies = 0
    for copies in visit_nulls.values():
        if len(copies) < 2:
            continue
        for i, cands in enumerate(copies):
            others = [o for j, o in enumerate(copies) if j != i]
            level = null_level([[c[0] for c in o] for o in others], rank)
            sites = veto_sites(others, level) if veto else np.zeros((0, 2))
            n_copies += 1
            for c in cands:
                if len(c) >= 4 and is_vetoed(c[2], c[3], sites):
                    continue
                scaled[frame_class(c[1])].append(c[0] / level)
    if not n_copies:
        return None
    grid = np.round(np.arange(1.0, 6.0001, 0.01), 2)
    ratios, far = {}, {}
    for c in FRAME_CLASSES:
        x = np.asarray(scaled[c])
        curve = np.array([np.sum(x >= g) for g in grid]) / n_copies
        ok = np.nonzero(curve <= target_far / len(FRAME_CLASSES))[0]
        ratios[c] = float(grid[ok[0]]) if ok.size else float(grid[-1])
        far[c] = curve.tolist()
    # The held-out false-alarm rate at these ratios.
    n_false = sum(int(np.sum(np.asarray(scaled[c]) >= ratios[c])) for c in FRAME_CLASSES)
    return {"rank": rank, "ratios": ratios, "n_copies": n_copies,
            "false_per_copy": n_false / n_copies,
            "far_curve": {"ratio": grid.tolist(), "by_class": far}}


def scrambled(seq, rng):
    """Copy of ``seq`` with the frame times permuted (frames re-sorted by
    their new times): nothing real moves linearly any more, while every
    artifact and the static sky stay."""
    times = seq.times
    perm = rng.permutation(len(times))
    while len(times) > 2 and np.all(perm == np.arange(len(times))):
        perm = rng.permutation(len(times))
    frames = [dataclasses.replace(f, mjd=float(t)) for f, t in zip(seq.frames, times[perm])]
    frames.sort(key=lambda f: f.mjd)
    return dataclasses.replace(seq, frames=frames, truth=None, known=None)


def guard_thresholds(level: float, base: Dict[str, float],
                     guard_level: float = GUARD_LEVEL) -> Dict[str, float]:
    """``base`` thresholds, raised in proportion when ``level`` exceeds ``guard_level``."""
    f = max(1.0, level / guard_level)
    return {c: float(t * f) for c, t in base.items()}


def self_calibrate(seq, cfg=None, n: int = 1, seed: int = 0, ratios=None,
                   rank: int = NULL_RANK, log=None, mode: str = "per-visit",
                   base: Optional[Dict[str, float]] = None) -> dict:
    """Run the detector on ``n`` time-scrambled copies of ``seq``: the null
    level, artifact veto sites and thresholds -- per-visit (level x ratios)
    or, in ``guard`` mode, ``base`` raised for an unusually noisy visit."""
    from .detector import detect
    log = log or (lambda *a: None)
    rng = np.random.default_rng(seed)
    copies = []
    for i in range(n):
        log(f"null calibration: scrambled copy {i + 1}/{n}")
        res = detect(scrambled(seq, rng), cfg)
        copies.append([(d["score"], d["n_frames_used"], d["ra_deg"], d["dec_deg"])
                       for d in res.detections])
    level = null_level([[c[0] for c in cp] for cp in copies], rank)
    if mode == "guard":
        if base is None:
            raise ValueError("guard mode needs base thresholds")
        thr = guard_thresholds(level, base)
    elif mode == "per-visit":
        thr = thresholds(level, ratios)
    else:
        raise ValueError(f"unknown null mode {mode!r}")
    sites = veto_sites(copies, level)
    log(f"null level {level:.2f} (rank {rank}, {n} copies, {mode}) -> thresholds "
        + ", ".join(f"{c} frames {t:.2f}" for c, t in thr.items())
        + f"; {len(sites)} artifact veto sites")
    return {"n_copies": n, "seed": seed, "rank": rank, "level": level, "mode": mode,
            "guard_level": GUARD_LEVEL if mode == "guard" else None,
            "ratios": dict(ratios or NULL_RATIO) if mode == "per-visit" else None,
            "thresholds": thr,
            "veto_sites": sites.tolist(),
            "copy_top_scores": [sorted((c[0] for c in cp), reverse=True)[:rank] for cp in copies]}


def apply(detections: List[dict], thr: Dict[str, float], sites=None) -> int:
    """Re-flag ``detected`` with per-visit thresholds and the artifact veto
    (``sites``: (ra, dec) from ``veto_sites``); the number detected."""
    sites = np.asarray(sites if sites is not None else [], float).reshape(-1, 2)
    for d in detections:
        d["artifact_veto"] = bool(len(sites)) and is_vetoed(d["ra_deg"], d["dec_deg"], sites)
        d["detected"] = passes(d["score"], d["n_frames_used"], thr) and not d["artifact_veto"]
    return sum(d["detected"] for d in detections)
