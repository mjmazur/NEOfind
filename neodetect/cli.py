"""Command-line interface: ``python -m neodetect {detect,evaluate,review,validate}``.

    python -m neodetect detect data/neowise/<visit> --ades obs.psv --gifs cutouts/
    python -m neodetect review data/neowise/<visit>     # confirm suspected unknowns
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
    g.add_argument("--backend", default=d.backend,
                   help="shift-and-stack backend: auto (cupy on an NVIDIA GPU once its self-test "
                        "passes, else numba, else numpy), cupy, numba, numpy")
    g.add_argument("--min-frames", type=int, default=d.min_frames,
                   help=f"frames a track must use (default: {d.min_frames})")
    g.add_argument("--threshold", type=float, default=None, dest="detect_threshold",
                   help="detection threshold for tracks using >= 5 frames "
                        f"(default: the instrument's calibration, else {d.detect_threshold})")
    g.add_argument("--threshold-4", type=float, default=None,
                   dest="detect_threshold_4",
                   help=f"... using 4 frames (default: instrument's, else {d.detect_threshold_4})")
    g.add_argument("--threshold-3", type=float, default=None,
                   dest="detect_threshold_3",
                   help=f"... using 3 frames (default: instrument's, else {d.detect_threshold_3})")
    g.add_argument("--report-threshold", type=float, default=d.report_threshold,
                   help=f"keep candidates above this for diagnostics (default: {d.report_threshold})")
    g.add_argument("--skip-crowded", action=argparse.BooleanOptionalAction, default=None,
                   help=f"skip the search in a crowded field (>= {d.crowded_frac * 100:.0f}%% of the sky "
                        "on static sources, e.g. the Galactic bulge), flagged either way "
                        "(default: on for real WISE frames, else off)")
    g.add_argument("--threads", type=int, default=0, help="threads (default: all cores)")


def _config(args) -> DetectorConfig:
    fields = DetectorConfig.__dataclass_fields__
    kw = {k: v for k, v in vars(args).items() if k in fields}
    thr = ("detect_threshold", "detect_threshold_3", "detect_threshold_4")
    if any(kw.get(k) is not None for k in thr):
        kw["instrument_thresholds"] = False  # explicit thresholds win
    for k in thr:
        if kw.get(k) is None:
            kw.pop(k, None)
    if kw.get("region") is not None:
        kw["region"] = tuple(kw["region"])
    return DetectorConfig(**kw)


def _print_detections(doc, rows=None):
    print(f"{'#':>3} {'score':>6} {'det':>3} {'used':>4} {'RA (deg)':>10} {'Dec (deg)':>10} "
          f"{'deg/day':>8} {'PA':>6} {'mag':>6}  status")
    for i in (range(len(doc["detections"])) if rows is None else rows):
        d = doc["detections"][i]
        mag = f"{d['mag']:6.2f}" if d["mag"] is not None else "     -"
        print(f"{i:3d} {d['score']:6.1f} {'yes' if d['detected'] else ' no':>3} "
              f"{d['n_frames_used']:2d}/{d['n_frames_covered']:<2d}{d['ra_deg']:10.5f} "
              f"{d['dec_deg']:+10.5f} {d['rate_deg_day']:8.3f} {d['pa_deg']:6.1f} {mag}  "
              f"{_status_text(d)}")


def _status_text(d) -> str:
    known = d.get("known")
    if not d["detected"]:
        return "" if known is None else str(known)
    s = d.get("status", "")
    if known is not None:
        s += f" {known}"
    elif d.get("near_known"):
        s += f" ({d['near_known']['name']} {d['near_known']['sep_arcsec']:.0f}\")"
    if d.get("review"):
        s += f" [{d['review']['verdict']}]"
    return s


def _print_crowded(doc) -> None:
    st = doc["stats"]
    if st.get("crowded"):
        what = ("search skipped" if st.get("skipped") else
                "expect false candidates and low completeness")
        print(f"crowded field: {st['confusion_frac']:.0%} of the sky lies on static sources -- {what}")


def _print_suspects(doc, where) -> None:
    from .review import REVIEWABLE
    rows = [i for i, d in enumerate(doc["detections"])
            if d["detected"] and d.get("status") in REVIEWABLE and not d.get("review")]
    if rows:
        kind = "suspected unknown" if any(doc["detections"][i]["status"] != "unchecked"
                                          for i in rows) else "unchecked"
        print(f"{len(rows)} {kind} detection(s) to review: "
              + ", ".join(f"#{i}" for i in rows)
              + f" -- run `python -m neodetect review {where}`")


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
    if n_null and not doc["stats"].get("skipped"):
        from . import nullcal
        ratios = dict(zip(nullcal.FRAME_CLASSES, args.null_ratio)) if args.null_ratio else None
        mode = args.null_mode or "per-visit"
        c = doc["config"]  # thresholds as resolved for this instrument
        base = {"3": c["detect_threshold_3"], "4": c["detect_threshold_4"], "5+": c["detect_threshold"]}
        cal = nullcal.self_calibrate(seq, cfg, n=n_null, ratios=ratios, log=log, mode=mode,
                                     base=base)
        nullcal.apply(doc["detections"], cal["thresholds"], cal["veto_sites"])
        doc["null_calibration"] = cal
    doc["frames"] = [{"path": f.path, "mjd": f.mjd} for f in seq.frames]
    if seq.truth:
        doc["truth_eval"] = truth_eval(seq, result)
    elif seq.known:
        doc["known_eval"] = truth_eval(seq, result, seq.known)
    if args.output:
        out = Path(args.output)
    elif seq.source:
        out = Path(seq.source) / RESULT_NAME
    else:
        out = Path(RESULT_NAME)
    from . import review
    review.classify(doc, seq.known or seq.truth)
    review.attach_verdicts(doc, review.load_verdicts(out.parent))
    out.write_text(json.dumps(doc, indent=1))
    _print_detections(doc)
    if seq.truth:
        found = [t for t in doc["truth_eval"] if t["detection_score"] is not None]
        print(f"truth: {len(found)}/{len(doc['truth_eval'])} asteroids matched by a candidate")
    if "known_eval" in doc:
        _print_known(doc, seq.known)
    print(f"wrote {out}")
    _print_crowded(doc)
    _print_suspects(doc, out if out.name != RESULT_NAME else out.parent)
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
        if pv.get("veto"):
            print(f"  artifact veto (nullcal): {pv['n_asteroids_vetoed']} asteroids lost to it")
        print(f"  -> neodetect detect --null-ratio {r['3']:.2f} {r['4']:.2f} "
              f"{r['5+']:.2f}")
        print(f"  {'visit':>40} {'level':>6} {'thr 5+':>6} {'vetos':>5} {'n':>4} {'complete':>8}")
        for v, x in pv["visits"].items():
            c = f"{x['completeness']:8.2f}" if x["completeness"] is not None else "       -"
            print(f"  {v[-40:]:>40} {x['level']:6.2f} {x['thresholds']['5+']:6.2f} "
                  f"{x.get('n_veto_sites', 0):5d} {x['n_asteroids']:4d} {c}")
        for name, key in (("magnitude", "completeness_by_mag"), ("rate deg/day", "completeness_by_rate")):
            print(f"\n{name:>14} {'n':>4} {'complete':>8}   (per-visit thresholds)")
            for c in pv[key]:
                print(f"{c['lo']:6.2f}-{c['hi']:<7.2f} {c['n']:4d} {c['completeness']:8.2f}")
    gd = s.get("guard")
    if gd:
        raised = ", ".join(f"{v[-30:]} x{f:.2f}" for v, f in gd["raised"].items()) or "none"
        print(f"\nguard (real-WISE default: global thresholds, raised when a visit's one-copy "
              f"null level > {gd['guard_level']}; veto from that copy): "
              f"{gd['false_per_null_run']:.3f} false per null run; raised: {raised}")
        print(f"{'magnitude':>14} {'n':>4} {'complete':>8}   (guard)")
        for c in gd["completeness_by_mag"]:
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


def _review_frames(doc, visit_dir):
    """The frames a detections file was made from, in its frame order."""
    from .io import load_sequence
    paths = [f["path"] for f in doc.get("frames", [])]
    if paths and all(Path(q).exists() for q in paths):
        seq = load_sequence(paths)
    else:  # moved since, or an `evaluate` result: the visit directory's images
        seq = load_sequence(str(visit_dir))
    mjd = [f["mjd"] for f in doc["frames"]] if "frames" in doc else \
        [f["mjd"] for f in doc["detections"][0]["frames"]] if doc["detections"] else []
    if len(seq.frames) != len(mjd) or any(abs(a.mjd - b) > 1e-6 for a, b in zip(seq.frames, mjd)):
        raise SystemExit(f"the images in {visit_dir} don't match the detections file's frames")
    return seq


def _skybot_catalog(seq, visit_dir, station, log):
    """known.json for the visit from SkyBoT, at its station (C51 for WISE,
    via the spacecraft offsets)."""
    from neowise.known import write_known
    fr = seq.frames[0]
    wise = str(fr.header.get("TELESCOP", "")).strip().upper() == "WISE"
    seq_json = Path(visit_dir) / "sequence.json"
    if fr.path.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")):
        meta = json.loads(seq_json.read_text()) if seq_json.exists() else {}
        if "ra_deg" not in meta or "pixel_scale_arcsec" not in meta:
            raise SystemExit("--skybot needs the field's pointing and pixel scale: ra_deg, dec_deg "
                             "and pixel_scale_arcsec in sequence.json")
    loc = "500" if wise else (station or str(fr.header.get("MPCCODE", "") or ""))
    if not loc:
        raise SystemExit("--skybot needs the observatory: --station CODE (or MPCCODE in the headers)")
    meta_path = Path(visit_dir) / "visit.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("center") and meta.get("size_deg"):  # a neowise visit: its own region
        c = tuple(meta["center"])
        radius = meta["size_deg"] / 2 ** 0.5 + 0.05
    else:  # every frame's corners (one frame can be a sliver at the region's edge)
        corners = []
        for f in seq.frames:
            ny, nx = f.shape
            ra, dec = f.wcs.all_pix2world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1], 0)
            corners += list(zip(ra, dec))
        ra, dec = np.array(corners).T
        v = np.array([np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                      np.cos(np.radians(dec)) * np.sin(np.radians(ra)), np.sin(np.radians(dec))]).mean(axis=1)
        c = (float(np.degrees(np.arctan2(v[1], v[0])) % 360), float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))))
        cosd = np.cos(np.radians(c[1]))
        radius = float(np.max(np.hypot(((ra - c[0] + 180) % 360 - 180) * cosd, dec - c[1]))) + 0.05
    log(f"querying SkyBoT ({'C51 via 500 + spacecraft offset' if wise else loc}) per frame, "
        f"{radius:.2f} deg around {c[0]:.4f} {c[1]:+.4f}")
    band = int(str(fr.band or "1").upper().lstrip("W") or 1) if wise else 1
    kw = {} if wise else {"loc": loc, "v_minus": 0.0, "source": f"SkyBoT (IMCCE), station {loc}"}
    return json.loads(write_known(seq, visit_dir, c, radius, band=band, log=log, **kw).read_text())


def cmd_review(args) -> int:
    from . import review

    target = Path(args.target)
    det_path = target / args.name if target.is_dir() else target
    visit_dir = det_path.parent
    doc = json.loads(det_path.read_text())
    log = lambda m: print(m, file=sys.stderr)
    catalog = None
    for name in ("known.json", "truth.json"):
        if (visit_dir / name).exists():
            catalog = json.loads((visit_dir / name).read_text())
            break
    seq = None
    if args.skybot:
        seq = _review_frames(doc, visit_dir)
        catalog = _skybot_catalog(seq, visit_dir, args.station, log)
    review.classify(doc, catalog)
    verdicts = review.load_verdicts(visit_dir)
    review.attach_verdicts(doc, verdicts)

    dets = doc["detections"]
    wanted = review.REVIEWABLE + (("known",) if args.all else ())
    rows = [i for i, d in enumerate(dets) if d["detected"] and d.get("status") in wanted]
    changed = False
    for verdict, idx in (("confirmed", args.confirm), ("rejected", args.reject)):
        for i in idx or []:
            if not (0 <= i < len(dets)) or not dets[i]["detected"]:
                raise SystemExit(f"#{i} is not a detection in {det_path}")
            review.set_verdict(verdicts, doc, i, verdict, args.note or "")
            changed = True

    sheet_dir = visit_dir / review.SHEET_DIR
    if rows and not args.no_sheets:
        seq = seq or _review_frames(doc, visit_dir)
        for i in rows:
            review.write_sheet(seq.frames, doc, i, sheet_dir / f"det{i:03d}.png")

    interactive = args.interactive if args.interactive is not None else (
        sys.stdin.isatty() and not (args.confirm or args.reject or args.list))
    if interactive:
        todo = [i for i in rows if not dets[i].get("review")]
        print(f"{len(todo)} detection(s) to review ({len(rows) - len(todo)} already have a verdict); "
              "c = confirm, r = reject, s = skip, q = quit; add a note after the letter")
        for i in todo:
            d = dets[i]
            print()
            _print_detections(doc, [i])
            if not args.no_sheets:
                print(f"    sheet: {sheet_dir / f'det{i:03d}.png'}")
            try:
                ans = input(f"#{i} [c/r/s/q]? ").strip()
            except EOFError:
                break
            key, note = (ans[:1].lower(), ans[1:].strip()) if ans else ("s", "")
            if key == "q":
                break
            if key in "cr" and key:
                review.set_verdict(verdicts, doc, i, "confirmed" if key == "c" else "rejected",
                                   note or args.note or "")
                changed = True

    if changed:
        print(f"wrote {review.save_verdicts(visit_dir, verdicts)}")
    det_path.write_text(json.dumps(doc, indent=1))
    if rows and not args.no_sheets:
        page = review.write_index(doc, rows, sheet_dir, f"{visit_dir}: detections to review")
        print(f"cutout sheets: {sheet_dir}/det*.png (all on {page})")
    print()
    _print_detections(doc, rows)
    if not rows:
        print("nothing to review: every detection matches a known object" if catalog
              else "no detections")
    elif catalog is None:
        print("no known.json: statuses are 'unchecked' (--skybot checks SkyBoT)")
    _print_crowded(doc)
    confirmed = [i for i, d in enumerate(dets) if d["detected"]
                 and (d.get("review") or {}).get("verdict") == "confirmed"]
    if args.ades:
        from .report import ades_rows, write_ades_psv
        seq = seq or _review_frames(doc, visit_dir)
        sub = dict(doc, detections=[dict(d) if i in confirmed else dict(d, detected=False)
                                    for i, d in enumerate(dets)])
        rows_ = ades_rows(sub, [f.header for f in seq.frames], station=args.station)
        stn = rows_[0]["stn"] if rows_ else (args.station or "XXX")
        path = write_ades_psv(rows_, args.ades, observatory=stn)
        print(f"wrote {path}: {len(rows_)} observations of {len(confirmed)} confirmed detection(s)")
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
                   help="run the detector on N time-scrambled copies of the visit (see "
                        "nullcal): guard mode raises the thresholds of an unusually noisy "
                        "visit, per-visit mode sets them from the copies; both veto "
                        "candidates that recur in the copies. 0 = off (default: 1 for real "
                        "WISE frames, else 0)")
    d.add_argument("--null-mode", choices=("guard", "per-visit"), default=None,
                   help="what the scrambled copies set: per-visit (default; the visit's null "
                        "level x NULL_RATIO) or guard (the instrument's global thresholds, "
                        "raised only for an unusually noisy visit)")
    d.add_argument("--null-ratio", type=float, nargs=3, metavar=("R3", "R4", "R5"),
                   help="per-visit mode: threshold / null level for 3, 4, >=5 frames (default: "
                        "the real-NEOWISE calibration in nullcal; `evaluate` prints new ones)")
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

    r = sub.add_parser("review", help="inspect and confirm suspected unknown asteroids")
    r.add_argument("target", help="a visit directory (its detections file) or a detections JSON")
    r.add_argument("--name", default=RESULT_NAME, help=f"detections file in the directory "
                                                       f"(default: {RESULT_NAME})")
    r.add_argument("--skybot", action="store_true",
                   help="(re)build known.json from SkyBoT before classifying (needs a real WCS "
                        "and the observatory: C51 for WISE, else --station or MPCCODE)")
    r.add_argument("--station", help="MPC observatory code (ground data; also for --ades)")
    r.add_argument("--all", action="store_true", help="also list detections of known objects")
    r.add_argument("--confirm", type=int, nargs="+", metavar="N", help="confirm detections #N")
    r.add_argument("--reject", type=int, nargs="+", metavar="N", help="reject detections #N")
    r.add_argument("--note", help="note stored with the verdicts")
    r.add_argument("--list", action="store_true", help="only list, don't prompt")
    r.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=None,
                   help="prompt for a verdict on each unreviewed detection (default: when run "
                        "in a terminal without --confirm/--reject/--list)")
    r.add_argument("--no-sheets", action="store_true", help="don't write cutout sheets")
    r.add_argument("--ades", metavar="FILE",
                   help="write the confirmed detections as ADES PSV (trkSub; no designation "
                        "unless matched to a known object)")
    r.set_defaults(func=cmd_review)

    v = sub.add_parser("validate", help="check noise whitening on an asteroid-free sequence")
    v.add_argument("inputs", nargs="+")
    v.add_argument("--region", type=float, nargs=3, metavar=("RA", "DEC", "SIZE"), default=None)
    v.set_defaults(func=cmd_validate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
