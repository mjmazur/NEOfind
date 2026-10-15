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
included -- meets the target. On real NEOWISE (``sims/real_v1``) this
raised completeness at W1 15.5-16 from 0.38 to 0.50 at the same false-alarm
rate, with thresholds for >= 5 frames from 9.5 (clean visits) to 20.

In production ``self_calibrate`` scrambles the frame times of the sequence
being searched, runs the detector on the copies, and returns the thresholds
(``neodetect detect --null-scrambles N``, default 2 for real WISE frames:
on the Oceana visit single copies gave levels of 7.7-8.4, so the median of
two steadies the thresholds); each copy costs one detector run.
"""

import dataclasses
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import numpy as np

FRAME_CLASSES = ("3", "4", "5+")
NULL_RANK = 5
# Calibrated on real NEOWISE W1 single exposures (sims/real_v1 with the M4
# detector, 15 visits x 3 scrambled copies, <= 0.05 false detections per
# visit). Recalibrate with ``neodetect evaluate`` (``per_visit.ratios``)
# after detector changes.
NULL_RATIO = {"3": 1.81, "4": 1.53, "5+": 1.39}


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


def calibrate_ratios(visit_nulls: Dict[str, List[List[Tuple[float, int]]]], target_far: float,
                     rank: int = NULL_RANK) -> Optional[dict]:
    """Ratios per frame class from scrambled copies, ``visit -> [[(score,
    n_used), ...] per copy]``. Each copy's candidates are scaled by the level
    of the other copies of its visit (leave one out), and each class gets a
    third of the ``target_far`` budget (false detections per copy). None
    if no visit has two copies."""
    scaled = {c: [] for c in FRAME_CLASSES}
    n_copies = 0
    for copies in visit_nulls.values():
        if len(copies) < 2:
            continue
        for i, cands in enumerate(copies):
            level = null_level([[s for s, _ in o] for j, o in enumerate(copies) if j != i], rank)
            n_copies += 1
            for s, n in cands:
                scaled[frame_class(n)].append(s / level)
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


def self_calibrate(seq, cfg=None, n: int = 1, seed: int = 0, ratios=None,
                   rank: int = NULL_RANK, log=None) -> dict:
    """Run the detector on ``n`` time-scrambled copies of ``seq``; the null
    level and per-visit thresholds."""
    from .detector import detect
    log = log or (lambda *a: None)
    rng = np.random.default_rng(seed)
    copies = []
    for i in range(n):
        log(f"null calibration: scrambled copy {i + 1}/{n}")
        res = detect(scrambled(seq, rng), cfg)
        copies.append([d["score"] for d in res.detections])
    level = null_level(copies, rank)
    thr = thresholds(level, ratios)
    log(f"null level {level:.2f} (rank {rank}, {n} copies) -> thresholds "
        + ", ".join(f"{c} frames {t:.2f}" for c, t in thr.items()))
    return {"n_copies": n, "seed": seed, "rank": rank, "level": level,
            "ratios": dict(ratios or NULL_RATIO), "thresholds": thr,
            "copy_top_scores": [sorted(c, reverse=True)[:rank] for c in copies]}


def apply(detections: List[dict], thr: Dict[str, float]) -> int:
    """Re-flag ``detected`` with per-visit thresholds; the number detected."""
    for d in detections:
        d["detected"] = passes(d["score"], d["n_frames_used"], thr)
    return sum(d["detected"] for d in detections)
