"""Command-line interface: ``python -m neodetect {detect,evaluate,validate}``.

    python -m neodetect detect data/neowise/<visit> --ades obs.psv --gifs cutouts/
"""

import argparse
import json
import sys
from pathlib import Path

from .detector import DetectorConfig, detect

RESULT_NAME = "detections.json"


def _add_detector_args(p: argparse.ArgumentParser) -> None:
    d = DetectorConfig()
    g = p.add_argument_group("detector")
    g.add_argument("--region", type=float, nargs=3, metavar=("RA", "DEC", "SIZE"), default=None,
                   help="search only this square (degrees); default: union of all frames")
    g.add_argument("--ref-frame", type=int, default=None, dest="ref_frame",
                   help="frame defining the grid orientation and zero point (default: middle)")
    g.add_argument("--no-link", action="store_false", dest="link",
                   help="disable the detection-linking search")
    g.add_argument("--no-stack", action="store_false", dest="stack",
                   help="disable the shift-and-stack search")
    g.add_argument("--peak-threshold", type=float, default=d.peak_threshold,
                   help=f"single-frame SNR for linking (default: {d.peak_threshold})")
    g.add_argument("--min-detections", type=int, default=d.min_detections,
                   help=f"detections needed to form a tracklet (default: {d.min_detections})")
    g.add_argument("--link-rate-max", type=float, default=d.link_rate_max,
                   help=f"fastest linked motion, deg/day (default: {d.link_rate_max})")
    g.add_argument("--stack-rate-min", type=float, default=d.stack_rate_min,
                   help=f"slowest stacked motion, deg/day (default: {d.stack_rate_min})")
    g.add_argument("--stack-rate-max", type=float, default=d.stack_rate_max,
                   help=f"fastest stacked motion, deg/day (default: {d.stack_rate_max})")
    g.add_argument("--stack-threshold", type=float, default=d.stack_threshold,
                   help=f"stacked SNR for candidates (default: {d.stack_threshold})")
    g.add_argument("--backend", default=d.backend, help="shift-and-stack backend: auto, numba, numpy")
    g.add_argument("--min-frames", type=int, default=d.min_frames,
                   help=f"frames a track must use (default: {d.min_frames})")
    g.add_argument("--threshold", type=float, default=d.detect_threshold, dest="detect_threshold",
                   help="detection threshold for tracks using >= 5 frames "
                        f"(default: {d.detect_threshold})")
    g.add_argument("--threshold-4", type=float, default=d.detect_threshold_4,
                   dest="detect_threshold_4",
                   help=f"... using 4 frames (default: {d.detect_threshold_4})")
    g.add_argument("--threshold-3", type=float, default=d.detect_threshold_3,
                   dest="detect_threshold_3",
                   help=f"... using 3 frames (default: {d.detect_threshold_3})")
    g.add_argument("--report-threshold", type=float, default=d.report_threshold,
                   help=f"keep candidates above this for diagnostics (default: {d.report_threshold})")
    g.add_argument("--threads", type=int, default=0, help="threads (default: all cores)")


def _config(args) -> DetectorConfig:
    fields = DetectorConfig.__dataclass_fields__
    kw = {k: v for k, v in vars(args).items() if k in fields}
    if kw.get("region") is not None:
        kw["region"] = tuple(kw["region"])
    return DetectorConfig(**kw)


def _print_detections(doc):
    print(f"{'#':>3} {'score':>6} {'det':>3} {'used':>4} {'RA (deg)':>10} {'Dec (deg)':>10} "
          f"{'deg/day':>8} {'PA':>6} {'mag':>6}  known")
    for i, d in enumerate(doc["detections"]):
        mag = f"{d['mag']:6.2f}" if d["mag"] is not None else "     -"
        print(f"{i:3d} {d['score']:6.1f} {'yes' if d['detected'] else ' no':>3} "
              f"{d['n_frames_used']:2d}/{d['n_frames_covered']:<2d}{d['ra_deg']:10.5f} "
              f"{d['dec_deg']:+10.5f} {d['rate_deg_day']:8.3f} {d['pa_deg']:6.1f} {mag}  "
              f"{d.get('known') or ''}")


def _label_known(doc, known, scale_arcsec):
    """Name the known object each detection matches, if any."""
    from .evaluate import match_detection
    for d in doc["detections"]:
        for a in known["asteroids"]:
            if match_detection(d, a, scale_arcsec):
                d["known"] = a.get("name", a["id"])
                d["known_id"] = a["id"]
                break


def _print_known(doc, known):
    """Known objects the search could have seen, brightest first."""
    rows = sorted((k for k in doc.get("known_eval", []) if k["n_frames_covered"] >= 3),
                  key=lambda k: k["mag"] if k["mag"] == k["mag"] else 99.0)
    names = {a["id"]: a.get("name", a["id"]) for a in known["asteroids"]}
    print(f"known objects on >= 3 frames ({known.get('magnitude', '')}):")
    print(f"  {'name':>18} {'mag':>5} {'deg/day':>7} {'frames':>6} {'@truth':>7} {'detection':>9}")
    for k in rows[:25]:
        at = f"{k['measured_at_truth']:7.1f}" if k["measured_at_truth"] is not None else "      -"
        det = f"{k['detection_score']:9.1f}" if k["detection_score"] is not None else "        -"
        print(f"  {names.get(k['id'], k['id']):>18} {k['mag']:5.1f} {k['rate_deg_day']:7.3f} "
              f"{k['n_frames_covered']:6d} {at} {det}")


def cmd_detect(args) -> int:
    from .evaluate import truth_eval
    from .io import load_sequence

    inputs = args.inputs if len(args.inputs) > 1 else args.inputs[0]
    seq = load_sequence(inputs)
    cfg = _config(args)
    log = lambda m: print(m, file=sys.stderr)
    result = detect(seq, cfg, log=log)
    doc = result.to_dict()
    n_null = args.null_scrambles
    if n_null is None:
        from .io import default_null_scrambles
        n_null = default_null_scrambles(seq.frames[0])
    if n_null:
        from . import nullcal
        ratios = dict(zip(nullcal.FRAME_CLASSES, args.null_ratio)) if args.null_ratio else None
        cal = nullcal.self_calibrate(seq, cfg, n=n_null, ratios=ratios, log=log)
        nullcal.apply(doc["detections"], cal["thresholds"])
        doc["null_calibration"] = cal
    doc["frames"] = [{"path": f.path, "mjd": f.mjd} for f in seq.frames]
    if seq.truth:
        doc["truth_eval"] = truth_eval(seq, result)
    elif seq.known:
        doc["known_eval"] = truth_eval(seq, result, seq.known)
        _label_known(doc, seq.known, result.reg.pixel_scale_arcsec)
    if args.output:
        out = Path(args.output)
    elif seq.source:
        out = Path(seq.source) / RESULT_NAME
    else:
        out = Path(RESULT_NAME)
    out.write_text(json.dumps(doc, indent=1))
    _print_detections(doc)
    if seq.truth:
        found = [t for t in doc["truth_eval"] if t["detection_score"] is not None]
        print(f"truth: {len(found)}/{len(doc['truth_eval'])} asteroids matched by a candidate")
    if "known_eval" in doc:
        _print_known(doc, seq.known)
    print(f"wrote {out}")
    if args.ades:
        from .report import ades_rows, write_ades_psv
        rows = ades_rows(doc, [f.header for f in seq.frames], station=args.station)
        stn = rows[0]["stn"] if rows else (args.station or "XXX")
        path = write_ades_psv(rows, args.ades, observatory=stn)
        print(f"wrote {path} ({len(rows)} observations)")
    if args.gifs:
        from .report import write_cutout_gifs
        paths = write_cutout_gifs(result, doc, args.gifs)
        print(f"wrote {len(paths)} cutout GIFs to {args.gifs}")
    if args.plot:
        from .overlay import plot_detections
        png = out.with_suffix(".png")
        plot_detections(result, png, truth=seq.truth)
        print(f"wrote {png}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate_sweep

    s = evaluate_sweep(args.sweep_dir, _config(args), jobs=args.jobs, force=args.force,
                       target_far=args.target_far, name=args.name)
    print(f"\nnull runs: {s['n_null_runs']}, asteroid runs: {s['n_asteroid_runs']}, "
          f"asteroids (>= 3 frames covered): {s['n_asteroids']}")
    th = s["calibrated_thresholds"]
    print(f"calibrated thresholds (<= {s['target_far_per_sequence']} false/sequence on null runs, "
          f"split across classes): 3 frames {th['3']:.2f}, 4 frames {th['4']:.2f}, "
          f">=5 frames {th['5+']:.2f}")
    print(f"  -> --threshold {th['5+']:.2f} --threshold-4 {th['4']:.2f} --threshold-3 {th['3']:.2f}")
    print(f"null-run max scores (highest 8): "
          f"{', '.join(f'{v:.1f}' for v in s['null_max_scores'][-8:])}")
    print(f"false detections at these thresholds: {s['false_per_null_run']:.3f} per null run, "
          f"{s['unmatched_per_asteroid_run']:.3f} unmatched per asteroid run")
    fmt = lambda v: f"{v:.1f}" if v is not None else "-"
    for name, key in (("magnitude", "completeness_by_mag"), ("rate deg/day", "completeness_by_rate")):
        print(f"\n{name:>14} {'n':>4} {'complete':>8} {'med ideal':>9} {'med @truth':>10}")
        for c in s[key]:
            print(f"{c['lo']:6.2f}-{c['hi']:<7.2f} {c['n']:4d} {c['completeness']:8.2f} "
                  f"{c['median_ideal_snr']:9.1f} {fmt(c['median_measured_at_truth']):>10}")
    pv = s.get("per_visit")
    if pv:
        r = pv["ratios"]
        print(f"\nper-visit thresholds (null level = rank-{pv['rank']} score on scrambled copies): "
              f"level x {r['3']:.2f} / {r['4']:.2f} / {r['5+']:.2f} for 3 / 4 / >=5 frames, "
              f"{pv['false_per_null_run']:.3f} false per held-out null run")
        print(f"  -> neodetect detect --null-ratio {r['3']:.2f} {r['4']:.2f} "
              f"{r['5+']:.2f}")
        print(f"  {'visit':>40} {'level':>6} {'thr 5+':>6} {'n':>4} {'complete':>8}")
        for v, x in pv["visits"].items():
            c = f"{x['completeness']:8.2f}" if x["completeness"] is not None else "       -"
            print(f"  {v[-40:]:>40} {x['level']:6.2f} {x['thresholds']['5+']:6.2f} "
                  f"{x['n_asteroids']:4d} {c}")
        for name, key in (("magnitude", "completeness_by_mag"), ("rate deg/day", "completeness_by_rate")):
            print(f"\n{name:>14} {'n':>4} {'complete':>8}   (per-visit thresholds)")
            for c in pv[key]:
                print(f"{c['lo']:6.2f}-{c['hi']:<7.2f} {c['n']:4d} {c['completeness']:8.2f}")
    if s["accuracy"]:
        print("\naccuracy of detected asteroids (global thresholds):")
        for k, v in s["accuracy"].items():
            print(f"  {k:18s} median {v['median']:+.3f}  rms {v['rms']:.3f}  |90%| {v['p90_abs']:.3f}")
    if s["errors"]:
        print(f"\n{len(s['errors'])} runs failed, first: {s['errors'][0]}")
    stem = Path(args.name).stem.replace("detections", "evaluation")
    print(f"\nwrote {Path(args.sweep_dir) / stem}.json and .png")
    return 0


def cmd_validate(args) -> int:
    from .validate import validate_noise

    inputs = args.inputs if len(args.inputs) > 1 else args.inputs[0]
    validate_noise(inputs, region=tuple(args.region) if args.region else None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neodetect",
                                description="Moving-object (asteroid) detector for image sequences.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="run the detector on a set of frames")
    d.add_argument("inputs", nargs="+", help="a directory, FITS files, or glob patterns")
    d.add_argument("-o", "--output", help="output JSON (default: DIR/detections.json)")
    d.add_argument("--plot", action="store_true", help="also write an overlay PNG")
    d.add_argument("--ades", metavar="FILE", help="write detections as ADES PSV (MPC format)")
    d.add_argument("--station", help="MPC observatory code for --ades (default: C51 for WISE, "
                                     "else the header's MPCCODE)")
    d.add_argument("--gifs", metavar="DIR", help="write an animated cutout GIF per detection")
    d.add_argument("--null-scrambles", type=int, default=None, metavar="N",
                   help="set this visit's thresholds from N time-scrambled copies of it "
                        "(one detector run each; see nullcal), overriding --threshold*; "
                        "0 = off (default: 2 for real WISE frames, else 0)")
    d.add_argument("--null-ratio", type=float, nargs=3, metavar=("R3", "R4", "R5"),
                   help="threshold / null level for 3, 4, >=5 frames (default: the real-NEOWISE "
                        "calibration in nullcal; `evaluate` prints new ones)")
    _add_detector_args(d)
    d.set_defaults(func=cmd_detect)

    e = sub.add_parser("evaluate", help="run + score the detector on a sweep.py directory")
    e.add_argument("sweep_dir")
    e.add_argument("-j", "--jobs", type=int, default=None, help="parallel sequences")
    e.add_argument("--force", action="store_true", help="re-run even if results exist")
    e.add_argument("--name", default=RESULT_NAME,
                   help="per-run result file name, to keep results of different configurations "
                        f"apart (default: {RESULT_NAME})")
    e.add_argument("--target-far", type=float, default=0.05,
                   help="false alarms per sequence for threshold calibration (default: 0.05)")
    _add_detector_args(e)
    e.set_defaults(func=cmd_evaluate)

    v = sub.add_parser("validate", help="check noise whitening on an asteroid-free sequence")
    v.add_argument("inputs", nargs="+")
    v.add_argument("--region", type=float, nargs=3, metavar=("RA", "DEC", "SIZE"), default=None)
    v.set_defaults(func=cmd_validate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
