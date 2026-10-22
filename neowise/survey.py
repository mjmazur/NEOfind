"""Resumable survey: fetch, search and review many visits.

``neowise survey OUT`` takes fields (``--fields FILE``: lines ``ra dec``)
or a grid along the ecliptic (``--lon-step``, ``--lats``), lists each
field's visits in the band (``--band``, W3 by default: the 2010 cryogenic
survey), fetches each visit (and, for W3, its W4 twin, which confirms short
W3 tracks: ``neodetect.confirm``), runs ``neodetect detect`` with the real
WISE defaults and ``neodetect review --list`` (cutout sheets of the
suspects). Fetching runs in a few threads and waits out IRSA outages;
searches run as subprocesses, ``--jobs`` at a time with ``--threads`` each.

State is kept in ``OUT/survey.json`` (per visit: directory, status,
detections, suspects), so an interrupted survey resumes where it stopped;
``OUT/suspects.txt`` lists each suspect with its sheet as soon as it's found.
"""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord
import astropy.units as u

from . import query
from .cli import DATA_DIR, _known
from .download import download_visit


def ecliptic_fields(lon_step: float, lats: List[float]) -> List[Tuple[float, float]]:
    """(ra, dec) of a grid in ecliptic longitude x latitude."""
    out = []
    lon = 0.0
    while lon < 360.0 - 1e-9:
        for lat in lats:
            c = SkyCoord(lon * u.deg, lat * u.deg, frame=GeocentricTrueEcliptic(equinox="J2000")).icrs
            out.append((round(float(c.ra.deg), 3), round(float(c.dec.deg), 3)))
        lon += lon_step
    return out


def visit_dir(ra, dec, size, band, mjd) -> Path:
    from astropy.time import Time
    tag = f"r{ra:07.3f}{dec:+07.3f}_s{size:g}_w{band}"
    return DATA_DIR / tag / Time(mjd, format="mjd").strftime("%Y%m%d")


class Survey:
    def __init__(self, out, band=3, size=0.6, jobs=4, threads=8, companion=True, log=print):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.band, self.size, self.jobs, self.threads = band, size, jobs, threads
        self.companion = companion and band == 3
        self.log = log
        self.lock = threading.Lock()
        p = self.out / "survey.json"
        self.state = json.loads(p.read_text()) if p.exists() else {"band": band, "size": size, "visits": {}}

    def save(self):
        with self.lock:
            tmp = self.out / "survey.json.part"
            tmp.write_text(json.dumps(self.state, indent=1))
            tmp.replace(self.out / "survey.json")

    def fetch(self, ra, dec) -> List[str]:
        """Fetch every visit of the field (and companions); the band's visit dirs."""
        tables = list(query.DEFAULT_TABLES[self.band])
        for attempt in range(36):
            try:
                frames = query.search(ra, dec, band=self.band, tables=tables, size_deg=self.size)
                break
            except RuntimeError as e:
                self.log(f"  search ({ra}, {dec}) failed: {e}; retrying in 10 min")
                time.sleep(600)
        else:
            return []
        visits = [query.one_per_orbit(v, ra, dec) for v in query.group_visits(frames)]
        dirs = []
        for v in (v for v in visits if len(v) >= 5):
            bands = [self.band] + ([4] if self.companion else [])
            for b in bands:
                fr = v if b == self.band else [type(f)(**{**f.__dict__, "band": b}) for f in v]
                d = visit_dir(ra, dec, self.size, b, v[0].mjd)
                for attempt in range(36):
                    try:
                        if not (d / "known.json").exists():
                            download_visit(fr, d, center=(ra, dec), size_deg=self.size)
                            _known(d, (ra, dec), self.size, b, None)
                        break
                    except Exception as e:
                        missing = [f for f in fr if "404" in str(e) and f.frame_id in str(e)]
                        if missing:  # not in the archive (some W4 frames): drop it, go on
                            self.log(f"  {d}: {missing[0].frame_id} not in the archive (404), skipped")
                            fr = [f for f in fr if f not in missing]
                            if len(fr) < 3:
                                break
                            continue
                        self.log(f"  fetch {d} failed: {str(e)[:120]}; retrying in 10 min")
                        time.sleep(600)
                if b == self.band:
                    dirs.append(str(d))
        return dirs

    def search(self, d: str) -> dict:
        """detect + review --list in subprocesses; the visit's summary."""
        env_py = [sys.executable, "-m", "neodetect"]
        with open(Path(d) / "detect.log", "w") as fh:
            subprocess.run(env_py + ["detect", d, "--threads", str(self.threads)], stdout=fh,
                           stderr=subprocess.STDOUT, check=False)
        with open(Path(d) / "review.log", "w") as fh:
            subprocess.run(env_py + ["review", d, "--list", "--no-interactive"], stdout=fh,
                           stderr=subprocess.STDOUT, check=False)
        det = Path(d) / "detections.json"
        if not det.exists():
            return {"status": "failed"}
        doc = json.loads(det.read_text())
        dets = [x for x in doc["detections"] if x["detected"]]
        sus = [i for i, x in enumerate(doc["detections"]) if x["detected"] and x.get("status") == "suspect"]
        return {"status": "skipped" if doc["stats"].get("skipped") else "searched",
                "detected": len(dets), "known": sum(x.get("status") == "known" for x in dets),
                "suspects": sus,
                "sheets": [str(Path(d) / "review" / f"det{i:03d}.png") for i in sus]}

    def run(self, fields: List[Tuple[float, float]], fetch_threads: int = 3):
        todo = [f for f in fields if f"{f[0]} {f[1]}" not in self.state.get("fields_done", [])]
        self.log(f"survey: {len(fields)} fields, {len(todo)} to do; band W{self.band}")
        with ThreadPoolExecutor(self.jobs) as searchers, ThreadPoolExecutor(fetch_threads) as fetchers:
            pending = []

            def done_search(d, fut):
                res = fut.result()
                with self.lock:
                    self.state["visits"][d] = res
                self.save()
                if res.get("suspects"):
                    with open(self.out / "suspects.txt", "a") as fh:
                        for i, sheet in zip(res["suspects"], res["sheets"]):
                            fh.write(f"{d} #{i} {sheet}\n")
                self.log(f"  {d}: {res.get('status')}, {res.get('detected', 0)} detected "
                         f"({res.get('known', 0)} known), suspects {res.get('suspects', [])}")

            def field(f):
                dirs = self.fetch(*f)
                for d in dirs:
                    if self.state["visits"].get(d, {}).get("status") in ("searched", "skipped"):
                        continue
                    fut = searchers.submit(self.search, d)
                    fut.add_done_callback(lambda fu, d=d: done_search(d, fu))
                    pending.append(fut)
                with self.lock:
                    self.state.setdefault("fields_done", []).append(f"{f[0]} {f[1]}")
                self.save()

            list(fetchers.map(field, todo))
            for fut in pending:
                fut.result()
        v = self.state["visits"].values()
        self.log(f"survey done: {len(v)} visits, {sum(x.get('status') == 'searched' for x in v)} searched, "
                 f"{sum(x.get('detected', 0) for x in v)} detected, "
                 f"{sum(len(x.get('suspects', [])) for x in v)} suspects")
