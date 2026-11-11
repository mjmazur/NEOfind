"""Command-line interface: ``python -m neowise {visits,fetch,known,inject,scramble,sweep,survey,campaign,gate,reviewer}``.

    python -m neowise visits 150.0 2.0                  # list visits covering a position
    python -m neowise fetch 150.0 2.0 --visit 3         # download that visit (0.5 deg cutouts)
    python -m neowise known data/neowise/<visit>        # (re)build known.json from SkyBoT
    python -m neowise inject data/neowise/<visit> -o OUT --n 20   # synthetic asteroids
    python -m neowise scramble data/neowise/<visit> -o OUT          # null test copy
    python -m neowise sweep data/neowise/*/* -o sims/real_v1        # both, for `neodetect evaluate`
    python -m neowise campaign run OUT --max-visits 200   # W3 + W4 campaign (test batch), 10-17 h
    python -m neowise campaign run OUT --automate         # ... around the detection runs (SLAVEMSG, UDP 17633)
    python -m neowise campaign pause OUT                  # free the GPU; `resume` to continue
    python -m neowise reviewer OUT                        # blind review page: confirm / likely / discard
"""

import argparse
import sys
from pathlib import Path

from astropy.time import Time

from . import query
from .download import download_visit

DATA_DIR = Path("data/neowise")


def _time_zone(name):
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        name and ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise argparse.ArgumentTypeError(f"unknown time zone {name!r} (e.g. America/Toronto)")
    return name


def _log(msg):
    print(msg, flush=True)


def _tables(a):
    """Mission phases to search: the given ones, else the band's default."""
    if a.tables:
        return a.tables.split(",")
    return list(query.DEFAULT_TABLES[a.band])


def _visits(a):
    tables = _tables(a)
    frames = query.search(a.ra, a.dec, band=a.band, tables=tables, size_deg=a.size)
    visits = [query.one_per_orbit(v, a.ra, a.dec) for v in
              query.group_visits(frames, max_span_days=a.max_span)]
    return [v for v in visits if len(v) >= a.min_frames]


def _date(mjd):
    return Time(mjd, format="mjd").strftime("%Y-%m-%d")


def cmd_visits(a):
    visits = _visits(a)
    print(f"{len(visits)} visits with >= {a.min_frames} frames covering "
          f"({a.ra}, {a.dec}) +- {a.size / 2} deg in W{a.band}:")
    for i, v in enumerate(visits):
        print(f"  {i:3d}  {_date(v[0].mjd)}  MJD {v[0].mjd:.3f}  {len(v):3d} frames  "
              f"span {v[-1].mjd - v[0].mjd:5.2f} d  [{v[0].table}]")
    return 0


def default_visit_dir(a, visit) -> Path:
    tag = f"r{a.ra:07.3f}{a.dec:+07.3f}_s{a.size:g}_w{a.band}"
    return DATA_DIR / tag / Time(visit[0].mjd, format="mjd").strftime("%Y%m%d")


def cmd_fetch(a):
    visits = _visits(a)
    if not visits:
        print("no visits found", file=sys.stderr)
        return 1
    if a.visit >= len(visits):
        print(f"--visit {a.visit} out of range (0..{len(visits) - 1}); see `visits`", file=sys.stderr)
        return 1
    v = visits[a.visit]
    out = Path(a.output) if a.output else default_visit_dir(a, v)
    _log(f"visit {a.visit}: {_date(v[0].mjd)}, {len(v)} frames -> {out}")
    size = None if a.full_frames else a.size
    download_visit(v, out, center=(a.ra, a.dec), size_deg=size, log=_log)
    if not a.no_known:
        _known(out, (a.ra, a.dec), a.size, a.band, a.max_v)
    print(out)
    return 0


def _known(visit_dir, center, size, band, max_v):
    if max_v is None:  # W3/W4 see asteroids by their thermal glow, far fainter than V 21
        max_v = 21.0 if band <= 2 else 99.0
    from neodetect.io import load_sequence
    from .known import write_known
    seq = load_sequence(visit_dir)
    _log("querying SkyBoT per frame for known objects")
    path = write_known(seq, visit_dir, center, radius_deg=size / 2 ** 0.5 + 0.05, band=band,
                       max_v=max_v, log=_log)
    _log(f"wrote {path}")


def cmd_known(a):
    import json
    meta = json.loads((Path(a.visit_dir) / "visit.json").read_text())
    center = meta["center"] or [meta["frames"][0]["ra"], meta["frames"][0]["dec"]]
    size = meta["size_deg"] or 0.8
    _known(Path(a.visit_dir), center, size, meta["frames"][0]["band"], a.max_v)
    return 0


def cmd_inject(a):
    from .inject import inject_visit
    out = inject_visit(a.visit_dir, a.output, n=a.n, mag_range=a.mags, rate_range=a.rates,
                       seed=a.seed, log=_log)
    print(out)
    return 0


def cmd_scramble(a):
    from .inject import scramble_visit
    print(scramble_visit(a.visit_dir, a.output, seed=a.seed, log=_log))
    return 0


def cmd_sweep(a):
    """Injected and time-scrambled copies of each visit, with an index.json
    that ``neodetect evaluate`` reads like a simulator sweep."""
    import json
    from .inject import inject_visit, scramble_visit, visit_epsf
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rng_seed = a.seed if a.seed is not None else 0
    runs = []
    for vi, vd in enumerate(a.visit_dirs):
        vd = Path(vd)
        visit_epsf(vd, log=_log)
        for j in range(a.injections):
            seed = rng_seed * 100000 + vi * 100 + j
            d = out / f"v{vi:03d}_inj{j:02d}"
            if not (d / "truth.json").exists():
                inject_visit(vd, d, n=a.n, mag_range=a.mags, rate_range=a.rates, seed=seed, log=_log)
            runs.append({"dir": str(d), "visit": str(vd), "asteroids": a.n, "seed": seed})
        for j in range(a.scrambles):
            seed = rng_seed * 100000 + vi * 100 + 50 + j
            d = out / f"v{vi:03d}_scr{j:02d}"
            if not (d / "truth.json").exists():
                scramble_visit(vd, d, seed=seed, log=_log)
            runs.append({"dir": str(d), "visit": str(vd), "asteroids": 0, "seed": seed})
    (out / "index.json").write_text(json.dumps({"kind": "neowise real-data sweep",
                                                "argv": sys.argv, "runs": runs}, indent=1))
    print(f"{len(runs)} runs -> {out / 'index.json'}")
    return 0


def cmd_survey(a):
    from .survey import Survey, ecliptic_fields
    if a.fields:
        fields = [tuple(float(x) for x in line.split()[:2]) for line in Path(a.fields).read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
    else:
        fields = ecliptic_fields(a.lon_step, [float(x) for x in a.lats.split(",")])
    if a.limit:
        fields = fields[:a.limit]
    Survey(a.output, band=a.band, size=a.size, jobs=a.jobs, threads=a.threads, log=_log).run(fields)
    return 0


def cmd_campaign(a):
    from . import campaign as C
    if a.action == "status":
        C.print_status(a.output)
        return 0
    if a.action == "retry-failed":
        print(f"{C.retry_failed(a.output)} failed visit(s) requeued")
        return 0
    if a.action == "backfill-sheets":
        print(f"{C.backfill_sheets(a.output)} searched visit(s) queued to be fetched again for blind sheets "
              "(the running campaign picks them up before new visits)")
        return 0
    if a.action in ("pause", "resume", "stop"):
        kw = {"pause": {"paused": True, "reason": a.reason or "paused by the user"},
              "resume": {"paused": False, "reason": ""}, "stop": {"stop": True}}[a.action]
        C.set_control(a.output, **kw)
        print(f"{a.output}: {a.action} requested (a running search is killed within seconds)"
              if a.action != "resume" else f"{a.output}: resumed")
        return 0
    s = C.Settings(hours=a.hours, tz=a.tz, automate="wait" if a.automate else "now" if a.automate_now else "",
                   udp_port=a.udp_port, controller_ip=a.controller_ip, resume_delay=a.resume_delay, jobs=a.jobs, threads=a.threads, gpu_limit=a.gpu_limit,
                   yield_gpu=a.yield_gpu, yield_min_mib=a.yield_min_mib, yield_ignore=a.yield_ignore,
                   stack_rate_max=a.stack_rate_max, size=a.size, tables=a.tables,
                   w4_min=a.w4_min, keep_fits=a.keep_fits, lat_max=a.lat_max, wide_share=a.wide_share)
    camp = C.Campaign(a.output, s, log=_log)
    if a.fields:
        fields = []
        for line in Path(a.fields).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                ra, dec = (float(x) for x in line.split()[:2])
                fields.append((ra, dec, float("nan"), float("nan")))
        n = C.add_fields(camp.db, fields)
    else:
        n = C.add_fields(camp.db, C.ecliptic_band_fields(a.lat_max, a.size))
    if n:
        camp.log(f"added {n} fields" + ("" if a.fields else f" (|ecliptic lat| <= {a.lat_max} deg)"))
    if a.wide_share > 0 and a.wide_lat_max > a.lat_max:  # after the band's: their own random order
        n = C.add_fields(camp.db, C.ecliptic_band_fields(a.wide_lat_max, a.size))
        if n:
            camp.log(f"added {n} fields beyond the band ({a.lat_max:g} < |ecliptic lat| <= {a.wide_lat_max:g} deg)")
    if a.action == "add-fields":
        return 0
    camp.run(max_visits=a.max_visits)
    return 0


def cmd_sheets(a):
    from .campaign import blind_sheets
    blind_sheets(a.visit_dir, log=_log)


def cmd_slavemsg(a):
    from . import automate
    t = {"detbeg": automate.SLAVEMSG_DETBEG, "detend": automate.SLAVEMSG_DETEND}[a.type]
    automate.send(t, a.host, a.port, a.text or "")
    print(f"sent {automate.type_name(t)} to {a.host}:{a.port}")


def cmd_gate(a):
    from .campaign import w4_gate
    w4_gate(a.visit_dir, a.min_score, log=_log)
    return 0


def cmd_skymap(a):
    from .skymap import write_skymap
    path = write_skymap(a.output, a.out, uncatalogued_only=a.uncatalogued, include_likely=not a.confirmed_only)
    print(f"wrote {path}")
    return 0


def cmd_plots(a):
    from .plots import PLOTS, write_plots
    for path in write_plots(a.output, a.out, include_likely=not a.confirmed_only, names=a.only or PLOTS):
        print(f"wrote {path}")
    return 0


def cmd_report(a):
    from .report import write_report
    print(f"wrote {write_report(a.output, a.out, include_likely=not a.confirmed_only, lookup=not a.no_lookup)}")
    return 0


def cmd_reviewer(a):
    from .reviewer import serve
    serve(a.output, host=a.host, port=a.port)
    return 0


def _range(s):
    lo, hi = (float(x) for x in s.split(":"))
    return lo, hi


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neowise", description="WISE/NEOWISE data access")
    sub = p.add_subparsers(dest="cmd", required=True)

    def position_args(q):
        q.add_argument("ra", type=float, help="deg")
        q.add_argument("dec", type=float, help="deg")
        q.add_argument("--size", type=float, default=0.5, help="square region side, deg (default 0.5)")
        q.add_argument("--band", type=int, default=1, choices=(1, 2, 3, 4),
                       help="WISE band; W3 (12 um) and W4 (22 um) exist only in the "
                            "cryogenic mission, 2010-01-07 .. 2010-08-06 (default 1)")
        q.add_argument("--tables", default=None,
                       help=f"comma-separated mission phases: {', '.join(query.TABLES)} "
                            "(default: neowiser for W1/W2, allsky for W3/W4)")
        q.add_argument("--min-frames", type=int, default=5, help="skip shorter visits (default 5)")
        q.add_argument("--max-span", type=float, default=None,
                       help="split visits longer than this many days")

    q = sub.add_parser("visits", help="list visits covering a position")
    position_args(q)
    q.set_defaults(func=cmd_visits)

    q = sub.add_parser("fetch", help="download a visit (cutouts) and its known objects")
    position_args(q)
    q.add_argument("--visit", type=int, default=0, help="index from `visits` (default 0)")
    q.add_argument("-o", "--output", default=None,
                   help=f"directory (default: {DATA_DIR}/<position>/<date>)")
    q.add_argument("--full-frames", action="store_true", help="whole frames, not cutouts")
    q.add_argument("--no-known", action="store_true", help="skip the SkyBoT query")
    q.add_argument("--max-v", type=float, default=None,
                   help="faintest known object (V) to keep (default: 21 for W1/W2, none for W3/W4)")
    q.set_defaults(func=cmd_fetch)

    q = sub.add_parser("known", help="(re)build known.json for a downloaded visit")
    q.add_argument("visit_dir")
    q.add_argument("--max-v", type=float, default=None,
                   help="faintest known object (V) to keep (default: 21 for W1/W2, none for W3/W4)")
    q.set_defaults(func=cmd_known)

    q = sub.add_parser("inject", help="copy a visit with synthetic asteroids added")
    q.add_argument("visit_dir")
    q.add_argument("-o", "--output", required=True)
    q.add_argument("--n", type=int, default=20, help="asteroids to inject (default 20)")
    q.add_argument("--mags", type=_range, default=(13.5, 16.5), help="W mag range (default 13.5:16.5)")
    q.add_argument("--rates", type=_range, default=(0.05, 3.0),
                   help="deg/day, log-uniform (default 0.05:3)")
    q.add_argument("--seed", type=int, default=None)
    q.set_defaults(func=cmd_inject)

    q = sub.add_parser("scramble", help="copy a visit with frame times permuted (null test)")
    q.add_argument("visit_dir")
    q.add_argument("-o", "--output", required=True)
    q.add_argument("--seed", type=int, default=None)
    q.set_defaults(func=cmd_scramble)

    q = sub.add_parser("sweep", help="injected + scrambled copies of visits, for `neodetect evaluate`")
    q.add_argument("visit_dirs", nargs="+")
    q.add_argument("-o", "--output", required=True)
    q.add_argument("--injections", type=int, default=2, help="injected copies per visit (default 2)")
    q.add_argument("--scrambles", type=int, default=2, help="scrambled copies per visit (default 2)")
    q.add_argument("--n", type=int, default=20, help="asteroids per injected copy (default 20)")
    q.add_argument("--mags", type=_range, default=(13.5, 17.0), help="default 13.5:17")
    q.add_argument("--rates", type=_range, default=(0.05, 3.0), help="deg/day (default 0.05:3)")
    q.add_argument("--seed", type=int, default=None)
    q.set_defaults(func=cmd_sweep)

    q = sub.add_parser("survey", help="fetch, search and review many visits (resumable)")
    q.add_argument("output", help="survey directory (state in survey.json, suspects.txt)")
    q.add_argument("--band", type=int, default=3, choices=(1, 2, 3, 4),
                   help="band (default 3: W3, 2010; W4 twins are fetched to confirm short tracks)")
    q.add_argument("--fields", help="file of 'ra dec' lines (default: an ecliptic grid)")
    q.add_argument("--lon-step", type=float, default=3.0, help="ecliptic grid step, deg (default 3)")
    q.add_argument("--lats", default="0", help="ecliptic latitudes, comma-separated (default 0)")
    q.add_argument("--size", type=float, default=0.6, help="cutout side, deg (default 0.6)")
    q.add_argument("--jobs", type=int, default=4, help="searches at a time (default 4)")
    q.add_argument("--threads", type=int, default=8, help="threads per search (default 8)")
    q.add_argument("--limit", type=int, default=None, help="only the first N fields")
    q.set_defaults(func=cmd_survey)

    from .campaign import Settings, W4_MIN_SCORE
    d = Settings()
    q = sub.add_parser("campaign", help="unattended W3 + W4 search of the 2010 survey (resumable)")
    q.add_argument("action", choices=("run", "add-fields", "status", "pause", "resume", "stop", "retry-failed",
                                          "backfill-sheets"),
                   help="run (or resume) the campaign; add-fields only adds fields; status; pause / "
                        "resume (frees the GPU at once); stop; retry-failed requeues failed visits; "
                        "backfill-sheets makes blind sheets for visits searched before the review was blind")
    q.add_argument("output", help="campaign directory (campaign.sqlite, data/, campaign.log)")
    q.add_argument("--lat-max", type=float, default=10.0,
                   help="fields with |ecliptic latitude| <= this, deg (default 10; 90 = whole sky); "
                        "a larger value later adds the extra fields")
    q.add_argument("--wide-share", type=float, default=d.wide_share,
                   help="share of the search time for fields beyond --lat-max (e.g. 0.25; default 0: none); "
                        "the next visit comes from the band behind on its share")
    q.add_argument("--wide-lat-max", type=float, default=90.0,
                   help="with --wide-share: fields up to this |ecliptic latitude|, deg (default 90: the poles)")
    q.add_argument("--fields", help="file of 'ra dec' lines instead of the ecliptic band")
    q.add_argument("--max-visits", type=int, default=None,
                   help="stop after this many visits in all (a test batch; fields go in random order)")
    q.add_argument("--hours", default=d.hours, help=f"active hours in --tz (default {d.hours})")
    q.add_argument("--tz", default=d.tz, type=_time_zone,
                   help=f"time zone of --hours, IANA name (default {d.tz}; '' = the machine's)")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--automate", action="store_true",
                   help="run around the night's detection runs instead of --hours: stop on SLAVEMSG_DETBEG, "
                        "start --resume-delay after SLAVEMSG_DETEND; wait for a DETEND before starting")
    g.add_argument("--automate-now", action="store_true",
                   help="as --automate, but start at once (the GPU is known to be free)")
    q.add_argument("--udp-port", type=int, default=d.udp_port,
                   help=f"UDP port of the controller's SLAVEMSGs (default {d.udp_port})")
    q.add_argument("--controller-ip", default=d.controller_ip,
                   help=f"the controller's address; others are ignored (default {d.controller_ip})")
    q.add_argument("--resume-delay", type=float, default=d.resume_delay,
                   help=f"minutes after SLAVEMSG_DETEND before searching again (default {d.resume_delay:g})")
    q.add_argument("--jobs", type=int, default=d.jobs, help=f"searches at a time (default {d.jobs})")
    q.add_argument("--threads", type=int, default=d.threads, help="CPU threads per search (default: CPUs / jobs)")
    q.add_argument("--gpu-limit", type=float, default=d.gpu_limit,
                   help=f"share of the GPU searches may use (default {d.gpu_limit})")
    q.add_argument("--no-yield-gpu", dest="yield_gpu", action="store_false",
                   help="don't pause while other programs compute on the GPU")
    q.add_argument("--yield-min-mib", type=float, default=d.yield_min_mib,
                   help=f"pause for another program only if it holds this much GPU memory, MiB "
                        f"(default {d.yield_min_mib:g})")
    q.add_argument("--yield-ignore", default=d.yield_ignore,
                   help=f"comma-separated process names never paused for (default {d.yield_ignore})")
    q.add_argument("--stack-rate-max", type=float, default=d.stack_rate_max,
                   help=f"shift-and-stack rate limit, deg/day (default {d.stack_rate_max})")
    q.add_argument("--size", type=float, default=d.size, help=f"field side and spacing, deg (default {d.size})")
    q.add_argument("--tables", default=d.tables,
                   help="mission phases (default allsky: the 4-band survey; add cryo_3band for W3-only "
                        "visits of Aug-Sep 2010)")
    q.add_argument("--w4-min", type=float, default=d.w4_min,
                   help=f"W4 track score a suspect needs (default {d.w4_min})")
    q.add_argument("--keep-fits", action="store_true", help="keep every visit's FITS files")
    q.add_argument("--reason", help="pause: a note shown in the status")
    q.set_defaults(func=cmd_campaign)

    q = sub.add_parser("slavemsg", help="send a SLAVEMSG datagram (try --automate by hand)")
    q.add_argument("type", choices=("detbeg", "detend"))
    q.add_argument("--host", default="127.0.0.1")
    q.add_argument("--port", type=int, default=17633)
    q.add_argument("--text", help="the message's text")
    q.set_defaults(func=cmd_slavemsg)

    q = sub.add_parser("sheets", help="blind cutout sheets for every detection of a searched visit")
    q.add_argument("visit_dir")
    q.set_defaults(func=cmd_sheets)

    q = sub.add_parser("gate", help="score a W3 visit's detections in its W4 twin (the campaign's W4 gate)")
    q.add_argument("visit_dir")
    q.add_argument("--min-score", type=float, default=W4_MIN_SCORE)
    q.set_defaults(func=cmd_gate)

    q = sub.add_parser("reviewer", help="web page for the blind review of a campaign's detections")
    q.add_argument("output", help="campaign directory")
    q.add_argument("--host", default="127.0.0.1")
    q.add_argument("--port", type=int, default=8765)
    q.set_defaults(func=cmd_reviewer)

    q = sub.add_parser("skymap", help="sky map (PNG) of a campaign's confirmed and likely asteroids")
    q.add_argument("output", help="campaign directory")
    q.add_argument("-o", "--out", help="image file (.png, .svg or .pdf; default OUT/skymap.png)")
    q.add_argument("--uncatalogued", action="store_true", help="only the ones no catalogued asteroid matches")
    q.add_argument("--confirmed-only", action="store_true", help="leave out the ones marked likely")
    q.set_defaults(func=cmd_skymap)

    q = sub.add_parser("plots", help="result plots (PNGs) of a campaign: completeness, motion, the detections "
                                     "not catalogued, astrometry, and checks on the search")
    q.add_argument("output", help="campaign directory")
    q.add_argument("-o", "--out", help="directory for the PNGs (default OUT/plots)")
    q.add_argument("--only", nargs="+", choices=("completeness", "motion", "gallery", "astrometry", "discovery", "scores",
                                                 "brightness", "galactic", "ecliptic", "solar_system", "progress"),
                   help="just these plots")
    q.add_argument("--confirmed-only", action="store_true", help="leave out the detections marked likely")
    q.set_defaults(func=cmd_plots)

    q = sub.add_parser("report", help="a PDF report of a campaign's results and significant findings")
    q.add_argument("output", help="campaign directory")
    q.add_argument("-o", "--out", help="the PDF (default OUT/report.pdf)")
    q.add_argument("--confirmed-only", action="store_true", help="leave out the detections marked likely")
    q.add_argument("--no-lookup", action="store_true",
                   help="don't look up missing discovery dates in JPL's Small-Body Database first")
    q.set_defaults(func=cmd_report)

    a = p.parse_args(argv)
    return a.func(a)
