#!/usr/bin/env python3
"""Generate a batch of simulations for detector evaluation.

Produces ``--runs`` sequences, each with ``--n-asteroids`` asteroids at mean
magnitudes drawn uniformly from ``--mags`` and rates log-uniform in
``--rates``, plus ``--nulls`` asteroid-free sequences, all under one
directory with an ``index.json`` listing every run:

    python sweep.py -o sims/sweeps/wise_v1 --runs 24 --nulls 24
    python -m neodetect evaluate sims/sweeps/wise_v1
"""

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from neosim.cli import build_configs, build_parser
from neosim.simulate import run_simulation


def _run(job):
    args = build_parser().parse_args(job["argv"])
    cfg = build_configs(args)
    run_simulation(cfg["instrument"], cfg["cadence"], cfg["pointing"], cfg["population"],
                   cfg["artifacts"], cfg["sim"])
    return job["dir"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--preset", default="wise-w1")
    p.add_argument("--runs", type=int, default=24, help="sequences with asteroids (default: 24)")
    p.add_argument("--nulls", type=int, default=24, help="asteroid-free sequences (default: 24)")
    p.add_argument("--n-asteroids", type=int, default=12, help="asteroids per run (default: 12)")
    p.add_argument("--mags", default="13.5:16.5",
                   help="mean magnitude range lo:hi, uniform (default: 13.5:16.5)")
    p.add_argument("--rates", default="0.05:3",
                   help="rate range lo:hi deg/day, log-uniform (default: 0.05:3)")
    p.add_argument("--seed", type=int, default=0, help="master seed (default: 0)")
    p.add_argument("-j", "--jobs", type=int, default=None, help="parallel workers")
    p.add_argument("sim_args", nargs=argparse.REMAINDER,
                   help="extra simulate.py options after '--', applied to every run")
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    m_lo, m_hi = args.mags.split(":")
    r_lo, r_hi = args.rates.split(":")
    extra = [a for a in args.sim_args if a != "--"]

    jobs = []
    for i in range(args.runs + args.nulls):
        has = i < args.runs
        d = str(out / f"run_{i:04d}")
        # Spread pointings around the ecliptic so star fields differ.
        ra = rng.uniform(0.0, 360.0)
        argv = ["--preset", args.preset, "-o", d, "--seed", str(rng.randrange(2 ** 31)),
                "--no-previews", "--ra", f"{ra:.4f}", "--dec", f"{rng.uniform(-10, 10):.4f}",
                "--n-asteroids", str(args.n_asteroids if has else 0),
                "--mag-min", m_lo, "--mag-max", m_hi, "--rate-min", r_lo, "--rate-max", r_hi,
                *extra]
        jobs.append({"dir": d, "asteroids": has, "argv": argv})

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for n, d in enumerate(pool.map(_run, jobs), 1):
            print(f"[{n}/{len(jobs)}] {d}", flush=True)

    with open(out / "index.json", "w") as f:
        json.dump({"args": vars(args), "runs": jobs}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
