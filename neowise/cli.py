"""Command-line interface: ``python -m neowise {visits,fetch,known,inject}``.

    python -m neowise visits 150.0 2.0                  # list visits covering a position
    python -m neowise fetch 150.0 2.0 --visit 3         # download that visit (0.5 deg cutouts)
    python -m neowise known data/neowise/<visit>        # (re)build known.json from SkyBoT
    python -m neowise inject data/neowise/<visit> -o OUT --n 20   # synthetic asteroids
    python -m neowise scramble data/neowise/<visit> -o OUT          # null test copy
    python -m neowise sweep data/neowise/*/* -o sims/real_v1        # both, for `neodetect evaluate`
"""

import argparse
import sys
from pathlib import Path

from astropy.time import Time

from . import query
from .download import download_visit

DATA_DIR = Path("data/neowise")


def _log(msg):
    print(msg, flush=True)


def _visits(a):
    tables = a.tables.split(",")
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
        q.add_argument("--band", type=int, default=1, choices=(1, 2))
        q.add_argument("--tables", default="neowiser",
                       help=f"comma-separated mission phases: {', '.join(query.TABLES)} "
                            "(default: neowiser)")
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
    q.add_argument("--max-v", type=float, default=21.0,
                   help="faintest known object (V) to keep (default 21)")
    q.set_defaults(func=cmd_fetch)

    q = sub.add_parser("known", help="(re)build known.json for a downloaded visit")
    q.add_argument("visit_dir")
    q.add_argument("--max-v", type=float, default=21.0)
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

    a = p.parse_args(argv)
    return a.func(a)
