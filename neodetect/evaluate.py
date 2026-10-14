"""Scoring the detector against simulator ground truth.

``run_one`` runs the detector on one simulation directory and writes
``detections.json``, including a ``truth_eval`` block per true asteroid:

* ``ideal_snr``: matched-filter SNR if the track and per-frame flux were
  known exactly (sum over frames the detector could use, from truth fluxes
  and the detector's own noise model);
* ``measured_at_truth``: the detector's robust statistic evaluated at the
  true track (the ideal plus this noise realization, minus model losses);
* the best matching detection's score and errors, if any.

Comparing a detection's score with ``measured_at_truth`` separates search
losses (the right track was never seeded or refined) from SNR limits.

``evaluate_sweep`` does this for every run of a ``sweep.py`` directory in
parallel and summarizes completeness vs magnitude and rate, false alarms on
the asteroid-free runs (and unmatched detections in asteroid runs), and
astrometric/photometric accuracy.
"""

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from .detector import DetectorConfig, detect
from .io import load_sequence
from .trackfit import Track

RESULT_NAME = "detections.json"
MATCH_TOL_PX = 2.0


def truth_track(asteroid: dict, reg, t_ref: float) -> Optional[Track]:
    """Least-squares linear track on the detector grid through the truth positions."""
    pts = asteroid["frames"]
    t = np.array([p["mjd"] for p in pts]) - t_ref
    x, y = reg.wcs.all_world2pix([p["ra_deg"] for p in pts], [p["dec_deg"] for p in pts], 0)
    a = np.vstack([np.ones_like(t), t]).T
    (x0, vx), *_ = np.linalg.lstsq(a, x, rcond=None)
    (y0, vy), *_ = np.linalg.lstsq(a, y, rcond=None)
    return Track(float(x0), float(y0), float(vx), float(vy), t_ref)


def match_detection(det: dict, asteroid: dict, scale_arcsec: float,
                    tol_px: float = MATCH_TOL_PX) -> Optional[dict]:
    """Median sky separation over frames where the truth is on a frame and the
    detection was measured; a match if below ``tol_px`` pixels."""
    seps = []
    for p, q in zip(asteroid["frames"], det["frames"]):
        if not (p["in_frame"] and q["covered"]):
            continue
        dra = (q["ra_deg"] - p["ra_deg"] + 180.0) % 360.0 - 180.0
        dra *= math.cos(math.radians(p["dec_deg"]))
        seps.append(math.hypot(dra, q["dec_deg"] - p["dec_deg"]) * 3600.0)
    if len(seps) < 2:
        return None
    med = float(np.median(seps))
    if med > tol_px * scale_arcsec:
        return None
    dpa = (det["pa_deg"] - asteroid["pa_deg"] + 180.0) % 360.0 - 180.0
    return {"sep_arcsec_median": med,
            "rate_err_frac": det["rate_deg_day"] / asteroid["rate_deg_day"] - 1.0,
            "pa_err_deg": dpa,
            "mag_err": (det["mag"] - asteroid["mag"]) if det.get("mag") is not None else None}


def truth_eval(seq, result) -> list:
    if not seq.truth:
        return []
    reg, pre, ev = result.reg, result.pre, result.evaluator
    scale = reg.pixel_scale_arcsec
    out = []
    for a in seq.truth["asteroids"]:
        tr = truth_track(a, reg, result.t_ref)
        sc = ev.score(tr)
        num, den = ev.frame_sums(tr)
        # Ideal SNR from truth fluxes (scaled into grid units) and detector weights.
        ideal2 = 0.0
        for k, p in enumerate(a["frames"]):
            if den[k] > 0 and p["in_frame"]:
                f = p["flux_dn"] * reg.flux_scale[k]
                ideal2 += f * f * den[k]
        best, best_err, matches = None, None, []
        for d in result.detections:
            err = match_detection(d, a, scale)
            if err is None:
                continue
            matches.append({"score": d["score"], "n_frames_used": d["n_frames_used"], "errors": err})
            if best is None or d["score"] > best:
                best, best_err = d["score"], err
        out.append({
            "id": a["id"], "mag": a["mag"], "rate_deg_day": a["rate_deg_day"],
            "n_frames_in_frame": a["n_frames_in_frame"],
            "n_frames_covered": int((den > 0).sum()),
            "ideal_snr": math.sqrt(ideal2),
            "measured_at_truth": sc.score if sc is not None else None,
            "detection_score": best, "errors": best_err, "matches": matches,
        })
    return out


def unmatched(result_doc: dict, truth: Optional[dict], scale_arcsec: float) -> list:
    """Detections matching no true asteroid."""
    ast = truth["asteroids"] if truth else []
    return [d for d in result_doc["detections"]
            if not any(match_detection(d, a, scale_arcsec) for a in ast)]


# --- running ------------------------------------------------------------------

def run_one(run_dir, cfg: DetectorConfig = None, force: bool = False,
            name: str = RESULT_NAME) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / name
    if out.exists() and not force:
        return json.loads(out.read_text())
    seq = load_sequence(run_dir)
    result = detect(seq, cfg)
    doc = result.to_dict()
    doc["truth_eval"] = truth_eval(seq, result)
    out.write_text(json.dumps(doc, indent=1))
    return doc


def _run_one_job(args):
    run_dir, cfg_dict, force, name = args
    try:
        run_one(run_dir, DetectorConfig(**cfg_dict), force, name)
        return run_dir, None
    except Exception as e:  # keep the sweep going; report at the end
        return run_dir, repr(e)


def evaluate_sweep(sweep_dir, cfg: DetectorConfig = None, jobs: Optional[int] = None,
                   force: bool = False, target_far: float = 0.05, name: str = RESULT_NAME,
                   log=print) -> dict:
    """Run (or reuse cached) detections for every run and summarize.
    ``target_far``: false alarms per sequence for the calibrated threshold."""
    sweep_dir = Path(sweep_dir)
    index = json.loads((sweep_dir / "index.json").read_text())
    runs = index["runs"]
    jobs = jobs or max(1, (os.cpu_count() or 2) // 8)
    threads = max(1, (os.cpu_count() or 2) // jobs)
    cfg = replace(cfg or DetectorConfig(), threads=threads)
    todo = [(r["dir"], cfg.to_dict(), force, name) for r in runs]
    errors = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, (d, err) in enumerate(pool.map(_run_one_job, todo), 1):
            if err:
                errors.append((d, err))
            log(f"[{i}/{len(todo)}] {d}" + (f"  ERROR {err}" if err else ""))
    summary = summarize(runs, target_far=target_far, name=name)
    summary["errors"] = errors
    stem = Path(name).stem.replace("detections", "evaluation")
    (sweep_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    try:
        plot_summary(summary, sweep_dir / f"{stem}.png")
    except ImportError:
        log("matplotlib not available; skipping plots")
    return summary


FRAME_CLASSES = ("3", "4", "5+")


def frame_class(n_used: int) -> str:
    return "3" if n_used <= 3 else ("4" if n_used == 4 else "5+")


def summarize(runs, target_far: float = 0.05, name: str = RESULT_NAME) -> dict:
    """Calibrate one threshold per frame-count class, each with a third of the
    false-alarm budget, then measure completeness and accuracy with them."""
    null_scores = {c: [] for c in FRAME_CLASSES}
    false_dets, null_max, rows = [], [], []
    n_ast_runs = 0
    for r in runs:
        path = Path(r["dir"]) / name
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        truth = json.loads((Path(r["dir"]) / "truth.json").read_text())
        scale = doc["stats"]["pixel_scale_arcsec"]
        if not truth["asteroids"]:
            for d in doc["detections"]:
                null_scores[frame_class(d["n_frames_used"])].append(d["score"])
            null_max.append(max((d["score"] for d in doc["detections"]), default=0.0))
            continue
        n_ast_runs += 1
        false_dets += [(d["score"], d["n_frames_used"]) for d in unmatched(doc, truth, scale)]
        for te in doc["truth_eval"]:
            rows.append({"dir": r["dir"], **te})

    n_null = len(null_max)
    thresholds = np.round(np.arange(5.0, 20.01, 0.25), 2)
    far_by_class, calibrated = {}, {}
    for c in FRAME_CLASSES:
        sc = np.asarray(null_scores[c])
        far = [float(np.sum(sc >= t)) / max(n_null, 1) for t in thresholds]
        far_by_class[c] = far
        ok = [t for t, f in zip(thresholds, far) if f <= target_far / len(FRAME_CLASSES)]
        calibrated[c] = float(ok[0]) if ok else float(thresholds[-1])

    def passes(score, n_used):
        return score >= calibrated[frame_class(n_used)]

    false_null = sum(int(np.sum(np.asarray(null_scores[c]) >= calibrated[c]))
                     for c in FRAME_CLASSES) / max(n_null, 1)
    false_ast = sum(passes(s, n) for s, n in false_dets) / max(n_ast_runs, 1)

    usable = [row for row in rows if row["n_frames_covered"] >= 3]
    for row in usable:
        hits = [m for m in row.get("matches", []) if passes(m["score"], m["n_frames_used"])]
        row["found"] = bool(hits)
        row["found_errors"] = max(hits, key=lambda m: m["score"])["errors"] if hits else None

    def table(key, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [row for row in usable if lo <= row[key] < hi]
            if not sel:
                continue
            out.append({"lo": lo, "hi": hi, "n": len(sel),
                        "completeness": float(np.mean([row["found"] for row in sel])),
                        "median_ideal_snr": float(np.median([row["ideal_snr"] for row in sel])),
                        "median_measured_at_truth": _median(row["measured_at_truth"] for row in sel)})
        return out

    mags = [row["mag"] for row in usable] or [0.0]
    mag_edges = np.arange(math.floor(min(mags) * 2) / 2, max(mags) + 0.5, 0.5).tolist()
    rate_edges = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 100.0]
    snr_edges = [0, 5, 7.5, 10, 12.5, 15, 20, 30, 50, 1e9]
    matched = [row["found_errors"] for row in usable if row["found_errors"]]
    accuracy = {}
    for key in ("sep_arcsec_median", "rate_err_frac", "pa_err_deg", "mag_err"):
        vals = np.array([e[key] for e in matched if e[key] is not None])
        if vals.size:
            accuracy[key] = {"median": float(np.median(vals)),
                             "rms": float(np.sqrt(np.mean(vals ** 2))),
                             "p90_abs": float(np.quantile(np.abs(vals), 0.9))}
    return {
        "n_null_runs": n_null, "n_asteroid_runs": n_ast_runs, "n_asteroids": len(usable),
        "target_far_per_sequence": target_far, "calibrated_thresholds": calibrated,
        "false_per_null_run": false_null, "unmatched_per_asteroid_run": false_ast,
        "far_curve": {"threshold": thresholds.tolist(), "by_class": far_by_class},
        "null_max_scores": sorted(null_max),
        "completeness_by_mag": table("mag", mag_edges),
        "completeness_by_rate": table("rate_deg_day", rate_edges),
        "completeness_by_ideal_snr": table("ideal_snr", snr_edges),
        "accuracy": accuracy, "asteroids": usable,
    }


def _median(values):
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def plot_summary(summary: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    thr = summary["calibrated_thresholds"]["5+"]
    ax = axes[0]
    comp = summary["completeness_by_mag"]
    ax.plot([0.5 * (c["lo"] + c["hi"]) for c in comp], [c["completeness"] for c in comp], "o-",
            color="#2a6fdb")
    ax.set_xlabel("asteroid mean magnitude")
    ax.set_ylabel("completeness (calibrated thresholds)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.set_title("Completeness")

    ax = axes[1]
    rows = summary["asteroids"]
    mags = np.array([r["mag"] for r in rows])
    ax.scatter(mags, [r["detection_score"] or 0.1 for r in rows], s=10, color="#2a6fdb",
               label="detector score (missed at 0.1)")
    ax.scatter(mags, [max(r["measured_at_truth"] or 0.1, 0.1) for r in rows], s=10, marker="x",
               color="#e07b00", label="statistic at true track")
    ax.scatter(mags, [r["ideal_snr"] for r in rows], s=6, marker=".", color="#777777",
               label="ideal SNR")
    ax.axhline(thr, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("asteroid mean magnitude")
    ax.set_ylabel("SNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Scores vs truth")

    ax = axes[2]
    far = summary["far_curve"]
    for c, color in zip(("3", "4", "5+"), ("#c0392b", "#e67e22", "#2a6fdb")):
        ax.semilogy(far["threshold"], np.maximum(far["by_class"][c], 1e-3), "o-", ms=3,
                    color=color, label=f"{c} frames")
        ax.axvline(summary["calibrated_thresholds"][c], color=color, lw=0.8, ls="--")
    ax.set_xlabel("score threshold")
    ax.set_ylabel("false detections per null sequence (0 at 1e-3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"False alarms ({summary['n_null_runs']} null runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
