"""Unattended W3 + W4 campaign over the 2010 cryogenic survey.

``neowise campaign run OUT`` tiles a band of ecliptic latitude (``--lat-max``,
default 10 deg; 90 = the whole sky) with square fields, lists each field's
W3 visits, fetches each with its W4 twin (the same exposures) and searches
it: ``neodetect detect`` (the joint W3 + W4 matched filter; W3 alone when a
visit has no W4), ``neodetect review --list`` (catalogue match), the W4
gate (``w4_gate``) and blind cutout sheets of every detection
(``blind_sheets``). Fields are worked in a fixed random order, so the first
N visits (``--max-visits``: a test batch) are a fair sample of the rest and
the projection in ``campaign status`` holds; rerunning with a larger
``--lat-max`` adds the extra fields after the ones already there.

**Higher latitudes.** With ``--wide-share`` (e.g. 0.25) the fields beyond
``--lat-max``, up to ``--wide-lat-max`` (default 90: the poles), are added
too, in their own random order, and get that share of the search time:
the next visit fetched is from the band behind on its share of the
seconds searched since the mix began (``next_band``). Time, not visits:
fields far from the ecliptic have many more visits (at the ecliptic poles
WISE looked every orbit). Searches take the fetched visits first in, first
out, so the mix holds.

**W4 gate.** W4 frames are noisier than W3 (2x2-binned), so the search
itself keeps the optimal weighting (``io.REAL_WISE_CAL``: W3 + 0.56 W4), and
W4 is scored afterwards: each track's score in the W4 frames
(``confirm.forced_scores``) passes at >= ``W4_MIN_SCORE`` (> 3 for 94% of
205 catalogued asteroids, for none of 101 null candidates). Visits without
W4 (frames missing from the archive) are searched in W3 alone (``no-w4``).
The review is blind: every detection goes on it, catalogued or not.

**Schedule and control.** Work happens only inside ``--hours`` (default
10-17) in the time zone ``--tz`` (default America/Toronto; the machine's clock
is UTC) and while not paused (``campaign pause OUT``, or the review
page). A search running when the window closes or a pause arrives is killed
and redone later, which frees the GPU at once. With ``--yield-gpu`` (on by
default) the campaign also pauses while another program computes on the GPU.
Searches run with ``NEOFIND_GPU_LIMIT`` (``--gpu-limit``, default 0.8),
shared among them through ``NEOFIND_GPU_LOCK``.

**Resuming.** State is in ``OUT/campaign.sqlite`` (fields, visits,
candidates, verdicts), written as each step finishes; a visit interrupted by
a crash is simply searched again. ``campaign run OUT`` picks up where it
stopped; ``scripts/neofind-campaign.service`` restarts it after a crash or
reboot. A searched visit's W4 FITS files are deleted at once, its W3 ones
when nothing in it is uncatalogued or once the review has settled it
(``settle_fits``; ``--keep-fits`` keeps all). ``backfill_sheets`` queues
visits searched before the review was blind to be fetched again for
their sheets.

``neowise reviewer OUT`` serves the blind review page (``reviewer.py``).
"""

import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

W4_MIN_SCORE = 3.0
REVIEWABLE = ("suspect", "near-known", "unchecked")
DB_NAME = "campaign.sqlite"
CONTROL_NAME = "control.json"
STATUS_NAME = "status.json"
FITS_GLOBS = ("*-int-1b.fits*", "*-unc-1b.fits*", "*-msk-1b.fits*")
BLIND_DIR = "review/blind"  # sheets without the catalogue match, for the blind review
FETCH_TRIES = 3  # downloads of a visit tried this many times in a row (2 min apart) ...
FETCH_DEFER_S = 3600.0  # ... before it goes back in the queue for an hour, so one stuck visit can't hold up the rest
FETCH_MAX_DEFERS = 12  # put back this many times: failed (``campaign retry-failed`` requeues it)
OUTAGE_DEFERS = 3  # this many visits in a row put back: the archive is down, wait 10 min

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS fields (
    id INTEGER PRIMARY KEY, ra REAL, dec REAL, elon REAL, elat REAL, ord INTEGER,
    status TEXT DEFAULT 'pending', n_visits INTEGER, error TEXT, UNIQUE (ra, dec));
CREATE TABLE IF NOT EXISTS visits (
    dir TEXT PRIMARY KEY, field_id INTEGER, mjd REAL, n_frames INTEGER, w4 INTEGER,
    status TEXT, attempts INTEGER DEFAULT 0, detected INTEGER, known INTEGER,
    n_candidates INTEGER, seconds REAL, error TEXT, updated TEXT, redo INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY, visit_dir TEXT, det INTEGER, status TEXT, gate TEXT,
    w4_score REAL, score REAL, n_frames INTEGER, ra REAL, dec REAL, t_ref REAL,
    rate REAL, pa REAL, mag REAL, near TEXT, sheet TEXT,
    verdict TEXT, note TEXT, reviewed TEXT, UNIQUE (visit_dir, det));
CREATE INDEX IF NOT EXISTS visits_status ON visits (status);
CREATE INDEX IF NOT EXISTS fields_ord ON fields (ord);
"""


def connect(out) -> sqlite3.Connection:
    """The campaign database (WAL: the review page reads while the campaign writes)."""
    db = sqlite3.connect(str(Path(out) / DB_NAME), timeout=60, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(SCHEMA)
    have = {r[1] for r in db.execute("PRAGMA table_info(visits)")}  # added since the first campaigns
    for col, decl in (("redo", "INTEGER DEFAULT 0"), ("fetch_after", "REAL"), ("fetch_tries", "INTEGER DEFAULT 0"),
                      ("searched_at", "REAL")):
        if col not in have:
            db.execute(f"ALTER TABLE visits ADD COLUMN {col} {decl}")
    have = {r[1] for r in db.execute("PRAGMA table_info(candidates)")}
    for col in ("known_name TEXT", "arc_arcsec REAL", "span_hours REAL"):
        if col.split()[0] not in have:
            db.execute(f"ALTER TABLE candidates ADD COLUMN {col}")
    return db


def arc(det: dict):
    """(arc length in arcsec, time span in hours) of a detection: from the
    first to the last frame its track was measured on; (None, None) if it
    has fewer than two."""
    fr = sorted((f for f in det.get("frames") or [] if f.get("used") and f.get("mjd") is not None), key=lambda f: f["mjd"])
    if len(fr) < 2:
        return None, None
    a, b = fr[0], fr[-1]
    r1, d1, r2, d2 = map(math.radians, (a["ra_deg"], a["dec_deg"], b["ra_deg"], b["dec_deg"]))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return round(math.degrees(2 * math.asin(min(1.0, math.sqrt(h)))) * 3600, 2), round((b["mjd"] - a["mjd"]) * 24, 3)


def fill_arcs(db) -> int:
    """Arc length and time span of the detections recorded before they were
    kept, from their visits' detections.json; how many."""
    rows = db.execute("SELECT id, visit_dir, det FROM candidates WHERE arc_arcsec IS NULL AND span_hours IS NULL "
                      "ORDER BY visit_dir").fetchall()
    dets, todo = {}, []
    for r in rows:
        if r["visit_dir"] not in dets:
            dets.clear()  # one visit at a time
            try:
                dets[r["visit_dir"]] = json.loads((Path(r["visit_dir"]) / "detections.json").read_text())["detections"]
            except (OSError, ValueError, KeyError):
                dets[r["visit_dir"]] = []
        d = dets[r["visit_dir"]]
        a, h = arc(d[r["det"]]) if r["det"] < len(d) else (None, None)
        todo.append((a if a is not None else -1, h if h is not None else -1, r["id"]))  # -1: none to give
    with db:
        db.executemany("UPDATE candidates SET arc_arcsec=?, span_hours=? WHERE id=?", todo)
    return len(todo)


def known_name(det: dict):
    """The catalogued asteroid a detection is, "2001 VS44 (64477)", or None."""
    name, num = det.get("known"), det.get("known_id")
    if not name:
        return None
    return f"{name} ({num})" if num and str(num) != str(name) else str(name)


def fill_known_names(db) -> int:
    """Name the catalogued detections recorded before ``known_name`` was
    kept, from their visits' detections.json; how many."""
    rows = db.execute("SELECT id, visit_dir, det FROM candidates WHERE status='known' AND known_name IS NULL").fetchall()
    dets, todo = {}, []
    for r in rows:
        if r["visit_dir"] not in dets:
            try:
                dets[r["visit_dir"]] = json.loads((Path(r["visit_dir"]) / "detections.json").read_text())["detections"]
            except (OSError, ValueError, KeyError):
                dets[r["visit_dir"]] = []
        d = dets[r["visit_dir"]]
        name = known_name(d[r["det"]]) if r["det"] < len(d) else None
        todo.append((name or "", r["id"]))  # "": looked for, none given
    with db:
        db.executemany("UPDATE candidates SET known_name=? WHERE id=?", todo)
    return sum(bool(n) for n, _ in todo)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, obj) -> None:
    """Atomic: a crash leaves the old file or the new one, never half of one."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


# ------------------------------------------------------------------- fields

def ecliptic_band_fields(lat_max: float, step: float) -> List[Tuple[float, float, float, float]]:
    """(ra, dec, elon, elat) of square fields ``step`` deg apart tiling
    |ecliptic latitude| <= ``lat_max``. Rows sit at multiples of ``step``,
    so a narrower band's fields are a subset of a wider one's."""
    from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord
    import astropy.units as u
    k = int(np.floor(lat_max / step + 1e-9))
    lons, lats = [], []
    for lat in np.arange(-k, k + 1) * step:
        if abs(lat) > 90.0 - step / 2:  # the poles: one field each
            lons.append(0.0)
            lats.append(float(np.sign(lat) * 90.0))
            continue
        n = max(1, int(np.ceil(360.0 * np.cos(np.radians(abs(lat) + step / 2)) / step)))
        lons += list(np.arange(n) * 360.0 / n)
        lats += [float(lat)] * n
    c = SkyCoord(np.array(lons) * u.deg, np.array(lats) * u.deg,
                 frame=GeocentricTrueEcliptic(equinox="J2000")).icrs
    return [(round(float(r), 3), round(float(d), 3), round(lo, 3), round(la, 3))
            for r, d, lo, la in zip(c.ra.deg, c.dec.deg, lons, lats)]


def add_fields(db, fields: Iterable[Tuple[float, float, float, float]], seed: int = 2010) -> int:
    """Insert new fields in a fixed random order after the existing ones."""
    fields = list(fields)
    rng = np.random.default_rng(seed + len(fields))
    start = (db.execute("SELECT MAX(ord) FROM fields").fetchone()[0] or 0) + 1
    n = 0
    with db:
        for o, i in enumerate(rng.permutation(len(fields))):
            ra, dec, elon, elat = fields[i]
            cur = db.execute("INSERT OR IGNORE INTO fields (ra, dec, elon, elat, ord) VALUES (?,?,?,?,?)",
                             (ra, dec, elon, elat, start + o))
            n += cur.rowcount
    return n


# ----------------------------------------------------------------- W4 gate

def w4_gate(visit_dir, min_score: float = W4_MIN_SCORE, log=print) -> dict:
    """Score each detection's track in the W4 twin and record
    ``w4_score`` and ``w4_gate`` ("pass", "fail" or "no-w4") on it, in the
    visit's ``detections.json``; counts per gate value."""
    from neodetect.confirm import companion_dir, forced_scores
    path = Path(visit_dir) / "detections.json"
    doc = json.loads(path.read_text())
    rows = [d for d in doc["detections"] if d["detected"]]
    comp = companion_dir(str(visit_dir), "4")
    todo = [d for d in rows if d.get("confirm_score") is None]
    scores = forced_scores(comp, todo) if comp is not None and todo else [None] * len(todo)
    for d, s in zip(todo, scores):
        d["w4_score"] = s
    for d in rows:
        if d.get("confirm_score") is not None:  # already scored in W4 (short-track confirmation)
            d["w4_score"] = d["confirm_score"]
        s = d.get("w4_score")
        d["w4_gate"] = "no-w4" if s is None else ("pass" if s >= min_score else "fail")
    _write_json(path, doc)
    counts = {g: sum(d["w4_gate"] == g for d in rows) for g in ("pass", "fail", "no-w4")}
    log(f"W4 gate (>= {min_score}) on {visit_dir}: {counts}"
        + ("" if comp is not None else " (no W4 twin: W3 alone)"))
    return counts


def blind_sheets(visit_dir, log=print) -> int:
    """A cutout sheet for every detection, without its catalogue match
    (``BLIND_DIR``/detNNN.png): the blind review sees catalogued and
    unknown objects alike. Needs the visit's FITS files."""
    from neodetect.cli import _review_frames
    from neodetect.review import write_sheet
    vdir = Path(visit_dir)
    doc = json.loads((vdir / "detections.json").read_text())
    rows = [i for i, d in enumerate(doc["detections"]) if d["detected"]]
    if rows:
        seq = _review_frames(doc, vdir)
        for i in rows:
            write_sheet(seq.frames, doc, i, vdir / BLIND_DIR / f"det{i:03d}.png", blind=True)
    log(f"{len(rows)} blind sheet(s) in {vdir / BLIND_DIR}")
    return len(rows)


def delete_fits(visit_dir, bands=("3", "4")) -> None:
    """Delete a W3 visit's FITS files (band "3") and its W4 twin's ("4")."""
    from neodetect.confirm import companion_dir
    for b in bands:
        v = Path(visit_dir) if b == "3" else companion_dir(str(visit_dir), b)
        for g in FITS_GLOBS if v is not None else ():
            for f in v.glob(g):
                f.unlink()


def settle_fits(db, visit_dir) -> bool:
    """Once every detection in the visit not catalogued has a verdict and
    none is confirmed or likely, its W3 FITS (kept for an ADES report) go
    unless ``--keep-fits``; whether they were deleted."""
    row = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
    if row and json.loads(row[0]).get("keep_fits"):
        return False
    open_ = db.execute("SELECT COUNT(*) FROM candidates WHERE visit_dir=? AND status!='known' AND "
                       "(verdict IS NULL OR verdict IN ('confirmed','likely'))", (str(visit_dir),)).fetchone()[0]
    if open_:
        return False
    delete_fits(visit_dir, ("3",))
    return True


# ----------------------------------------------------------------- control

def read_control(out) -> dict:
    p = Path(out) / CONTROL_NAME
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def set_control(out, **kw) -> dict:
    c = {**read_control(out), **kw, "updated": _now()}
    _write_json(Path(out) / CONTROL_NAME, c)
    return c


def parse_hours(s: str) -> Tuple[float, float]:
    """"10-17" or "9:30-17:15" -> (start, end) in hours of the day."""
    def h(x):
        hh, _, mm = x.strip().partition(":")
        return int(hh) + int(mm or 0) / 60.0
    a, b = s.split("-")
    return h(a), h(b)


def local_now(tz: str = "") -> datetime:
    """Wall-clock time in ``tz`` (an IANA name; "" = the machine's own zone)."""
    return datetime.now(ZoneInfo(tz)) if tz else datetime.now()


def in_window(hours: Tuple[float, float], now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    t = now.hour + now.minute / 60.0 + now.second / 3600.0
    a, b = hours
    return a <= t < b if a <= b else (t >= a or t < b)  # a window past midnight wraps


def other_gpu_users(own_pids, min_mib: float = 300.0, ignore=()) -> List[str]:
    """Compute processes on the GPU other than ours ("pid name MiB") holding
    at least ``min_mib``, except those whose name contains an ``ignore``
    entry. Remote-desktop video encoders (RustDesk: ~140 MiB) hold a small
    CUDA context all the time; real GPU work takes far more."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit() or int(parts[0]) in own_pids:
            continue
        pid, name, mib = parts[0], parts[1], float(parts[2]) if parts[2].replace(".", "").isdigit() else 0.0
        if mib < min_mib or any(x and x.lower() in name.lower() for x in ignore):
            continue
        out.append(f"{pid} {Path(name).name} {mib:.0f} MiB")
    return out


# ---------------------------------------------------------------- campaign

@dataclass
class Settings:
    hours: str = "10-17"
    tz: str = "America/Toronto"  # the hours' time zone; "" = the machine's
    jobs: int = 2
    threads: int = 0  # per search; 0 = the CPUs / jobs
    gpu_limit: float = 0.8
    yield_gpu: bool = True
    yield_min_mib: float = 300.0
    yield_ignore: str = "rustdesk"
    stack_rate_max: float = 0.25
    size: float = 0.6
    tables: str = "allsky"
    max_span: float = 1.5
    min_frames: int = 5
    w4_min: float = W4_MIN_SCORE
    keep_fits: bool = False
    prefetch: int = 4
    max_attempts: int = 3
    automate: str = ""  # "wait" / "now": the controller's SLAVEMSGs decide, not --hours (automate.py)
    udp_port: int = 17633
    controller_ip: str = "10.24.8.10"
    resume_delay: float = 30.0  # minutes after SLAVEMSG_DETEND
    lat_max: float = 10.0  # the band near the ecliptic, |ecliptic lat| <= this
    wide_share: float = 0.0  # share of the search time for the fields beyond it (``next_band``)


class Interrupted(Exception):
    pass


class Campaign:
    def __init__(self, out, settings: Settings = None, log=None):
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.data = self.out / "data"
        self.s = settings or Settings()
        self.db = connect(self.out)
        self.lock = threading.RLock()
        self._logfh = open(self.out / "campaign.log", "a", buffering=1)
        self._log_ext = log
        self.procs = {}  # worker -> Popen
        self.current = {}  # worker -> (visit dir, redo)
        self.stop = threading.Event()
        self.active = threading.Event()
        self.state = "starting"
        self.reason = ""
        self.gpu_users: List[str] = []
        self._last_gpu_check = 0.0
        self.max_visits: Optional[int] = None
        self._active_since: Optional[float] = None
        self.gate = None  # automate.SlaveGate, with --automate / --automate-now

    # ---- bookkeeping
    def log(self, msg: str) -> None:
        line = f"{_now()} {msg}"
        self._logfh.write(line + "\n")
        if self._log_ext:
            self._log_ext(line)

    def sql(self, q, args=()):
        with self.lock, self.db:
            return self.db.execute(q, args).fetchall()

    def meta(self, key, default=None):
        r = self.sql("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(r[0][0]) if r else default

    def set_meta(self, key, value):
        self.sql("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))

    def _count(self, q, args=()) -> int:
        return self.sql(q, args)[0][0]

    def visits_processed(self) -> int:  # a redo (``backfill_sheets``) was processed before
        return self._count("SELECT COUNT(*) FROM visits WHERE status IN ('done','failed') OR redo=1")

    # ---- schedule
    def _update_active(self) -> None:
        ctl = read_control(self.out)
        if ctl.get("stop"):
            self.stop.set()
        state, reason = "running", ""
        if ctl.get("paused"):
            state, reason = "paused", ctl.get("reason") or "paused by the user"
        elif self.gate is not None:
            if not self.gate.is_open():
                state, reason = "waiting", self.gate.describe()
        elif not in_window(parse_hours(self.s.hours), local_now(self.s.tz)):
            state, reason = "waiting", f"outside the active hours {self.s.hours} {self.s.tz}".rstrip()
        if state == "running" and self.s.yield_gpu:
            if time.time() - self._last_gpu_check > 15:
                own = {p.pid for p in list(self.procs.values())}
                self.gpu_users = other_gpu_users(own, self.s.yield_min_mib, self.s.yield_ignore.split(","))
                self._last_gpu_check = time.time()
            if self.gpu_users:
                state, reason = "yielding", "GPU in use by " + ", ".join(self.gpu_users)
        if (self.max_visits is not None and self.visits_processed() >= self.max_visits
                and not self._count("SELECT COUNT(*) FROM visits WHERE redo=1 AND status!='failed'")):
            state, reason = "done", f"test batch of {self.max_visits} visits finished"
            self.stop.set()
        if state != self.state:
            self.log(f"state: {state}" + (f" ({reason})" if reason else ""))
        now = time.time()
        if state == "running" and self._active_since is None:
            self._active_since = now
        elif state != "running" and self._active_since is not None:
            self.set_meta("active_seconds", self.meta("active_seconds", 0.0) + now - self._active_since)
            self._active_since = None
        self.state, self.reason = state, reason
        (self.active.set if state == "running" else self.active.clear)()

    def active_seconds(self) -> float:
        extra = time.time() - self._active_since if self._active_since else 0.0
        return self.meta("active_seconds", 0.0) + extra

    def _wait_active(self) -> None:
        while not self.stop.is_set():
            if self.active.wait(5):
                return
        raise Interrupted("stopping")

    def _sleep(self, seconds: float) -> None:
        """Sleep, returning early (Interrupted) when stopping."""
        if self.stop.wait(seconds):
            raise Interrupted("stopping")

    # ---- fetching
    def _visit_dir(self, ra, dec, band, mjd) -> Path:
        from astropy.time import Time
        tag = f"r{ra:07.3f}{dec:+07.3f}_s{self.s.size:g}_w{band}"
        return self.data / tag / Time(mjd, format="mjd").strftime("%Y%m%d")

    def _retrying(self, what, fn, tries=36, wait=600.0):
        """``fn()``, retried every ``wait`` s through IRSA/SkyBoT outages."""
        for attempt in range(tries):
            self._wait_active()
            try:
                return fn()
            except (Interrupted, _Missing):
                raise
            except Exception as e:  # network
                if attempt + 1 == tries:
                    raise _GaveUp(f"{what}: gave up after {tries} tries ({str(e)[:160]})")
                self.log(f"{what} failed ({str(e)[:160]}); retrying in {wait / 60:g} min")
                self._sleep(wait)

    def list_field(self, f) -> int:
        """List the field's W3 visits into the database; how many."""
        from . import query
        ra, dec = f["ra"], f["dec"]
        frames = self._retrying(f"search ({ra}, {dec})", lambda: query.search(
            ra, dec, band=3, tables=self.s.tables.split(","), size_deg=self.s.size))
        visits = [query.one_per_orbit(v, ra, dec)
                  for v in query.group_visits(frames, max_span_days=self.s.max_span)]
        visits = [v for v in visits if len(v) >= self.s.min_frames]
        with self.lock, self.db:
            for v in visits:
                d = self._visit_dir(ra, dec, 3, v[0].mjd)
                (d.parent / ".frames").mkdir(parents=True, exist_ok=True)
                (d.parent / ".frames" / f"{d.name}.json").write_text(json.dumps([x.to_dict() for x in v]))
                self.db.execute("INSERT OR IGNORE INTO visits (dir, field_id, mjd, n_frames, status, updated) "
                                "VALUES (?,?,?,?,?,?)", (str(d), f["id"], v[0].mjd, len(v), "listed", _now()))
            self.db.execute("UPDATE fields SET status=?, n_visits=? WHERE id=?",
                            ("listed" if visits else "empty", len(visits), f["id"]))
        return len(visits)

    def fetch_visit(self, row) -> None:
        """Download the W3 visit, its W4 twin and both known.json."""
        from .cli import _known
        from .download import download_visit
        from .query import FrameInfo
        d3 = Path(row["dir"])
        frames = [FrameInfo(**x) for x in json.loads((d3.parent / ".frames" / f"{d3.name}.json").read_text())]
        ra, dec = self.sql("SELECT ra, dec FROM fields WHERE id=?", (row["field_id"],))[0]
        have_w4 = 0
        for band in (3, 4):  # W4: the twins of the W3 frames kept
            fr = frames if band == 3 else [FrameInfo(**{**f.__dict__, "band": 4}) for f in frames]
            d = d3 if band == 3 else self._visit_dir(ra, dec, 4, frames[0].mjd)
            while not ((d / "known.json").exists() and _have_frames(d, fr)):
                try:
                    self._retrying(f"fetch {d}", lambda: self._download(fr, d, ra, dec, band, download_visit,
                                                                          _known), tries=FETCH_TRIES, wait=120)
                except _Missing as m:  # not in the archive (some W4 frames): drop it, go on
                    if band == 3 and row["redo"]:  # it was there when the visit was searched: IRSA's hiccup
                        raise _GaveUp(f"fetch {d}: {m.frame_id} not found (404), but it was searched with it")
                    self.log(f"{d}: {m.frame_id} not in the archive (404), skipped")
                    fr = [f for f in fr if f.frame_id != m.frame_id]
                    if len(fr) < 3:
                        break
            if band == 3:
                frames = fr
            if band == 4 and (d / "known.json").exists():
                have_w4 = 1
        if not (d3 / "known.json").exists():
            raise RuntimeError("W3 frames not available")
        self.sql("UPDATE visits SET status='fetched', w4=?, updated=? WHERE dir=?", (have_w4, _now(), str(d3)))

    def _download(self, fr, d, ra, dec, band, download_visit, _known):
        try:
            download_visit(fr, d, center=(ra, dec), size_deg=self.s.size)  # skips files present
        except Exception as e:
            for f in fr:
                if "404" in str(e) and f.frame_id in str(e):
                    raise _Missing(f.frame_id)
            raise
        if not (d / "known.json").exists():
            _known(d, (ra, dec), self.s.size, band, None)

    def _wide(self, col="elat") -> str:
        """SQL: a field beyond the near band (fields of --fields, no
        ecliptic latitude, count as near)."""
        return f"COALESCE(ABS({col}), 0) > {float(self.s.lat_max)}"

    def band_seconds(self) -> dict:
        """Search seconds per band since the mix began, the visits in
        flight (fetched or searching) counted at their band's mean."""
        since = self.meta("wide_since", 0.0)
        secs, n, fly = {0: 0.0, 1: 0.0}, {0: 0, 1: 0}, {0: 0, 1: 0}
        for w, sec, k in self.sql(f"SELECT {self._wide('f.elat')}, COALESCE(SUM(v.seconds), 0), COUNT(*) "
                                  "FROM visits v JOIN fields f ON f.id=v.field_id WHERE v.status='done' "
                                  "AND v.searched_at >= ? AND v.seconds IS NOT NULL GROUP BY 1", (since,)):
            secs[w], n[w] = sec, k
        for w, k in self.sql(f"SELECT {self._wide('f.elat')}, COUNT(*) FROM visits v JOIN fields f "
                             "ON f.id=v.field_id WHERE v.status IN ('fetched','searching') AND v.redo=0 GROUP BY 1"):
            fly[w] = k
        mean = (secs[0] + secs[1]) / (n[0] + n[1]) if n[0] + n[1] else 600.0
        return {("wide" if w else "near"): secs[w] + fly[w] * (secs[w] / n[w] if n[w] else mean) for w in (0, 1)}

    def next_band(self) -> Optional[str]:
        """The band to fetch from next: the one behind on its share of the
        time (``wide_share``); None without the mix (field order only)."""
        share = self.s.wide_share
        if share <= 0:
            return None
        if share >= 1:
            return "wide"
        t = self.band_seconds()
        return "wide" if t["wide"] < share * (t["wide"] + t["near"]) else "near"

    def pick(self, full: bool = False):
        """What to fetch next: ([a listed visit], None), or ([], a field to
        list), from the band ``next_band`` says (the other if it has none
        left); visits to redo first. ``full``: only those."""
        band = self.next_band()
        rows, field = [], None
        for b in [band, {"near": "wide", "wide": "near"}[band]] if band else [None]:
            cond = "1" if b is None else self._wide() if b == "wide" else f"NOT {self._wide()}"
            rows = self.sql("SELECT * FROM visits WHERE status='listed' AND COALESCE(fetch_after, 0) <= ?"
                            + (" AND redo=1" if full else "")
                            + f" AND (redo=1 OR field_id IN (SELECT id FROM fields WHERE {cond}))"
                            + " ORDER BY redo DESC, (SELECT ord FROM fields WHERE id=field_id), mjd LIMIT 1",
                            (time.time(),))
            if rows or full:
                break
            f = self.sql(f"SELECT * FROM fields WHERE status='pending' AND {cond} ORDER BY ord LIMIT 1")
            if f:
                field = f[0]
                break
        return rows, field

    def fetcher(self) -> None:
        """Keep ``prefetch`` visits fetched ahead of the searches. A visit
        whose downloads keep failing is put back in the queue for an hour
        and the next one fetched instead."""
        put_back = 0  # visits in a row
        while not self.stop.is_set():
            try:
                self._wait_active()
                if self._count("SELECT COUNT(*) FROM visits WHERE status='fetched'") >= self.s.prefetch:
                    self._sleep(5)
                    continue
                full = self.max_visits is not None and (self.visits_processed() + self._count(
                    "SELECT COUNT(*) FROM visits WHERE status IN ('fetched','searching') AND redo=0")) >= self.max_visits
                rows, field = self.pick(full)
                if full and not rows:
                    self._sleep(10)
                    continue
                if rows:
                    try:
                        self.fetch_visit(rows[0])
                        put_back = 0
                    except Interrupted:
                        raise
                    except _GaveUp as e:
                        put_back += 1
                        self.put_back(rows[0], e)
                        if put_back >= OUTAGE_DEFERS:
                            self.log(f"{put_back} visits in a row failed to download: the archive seems down, "
                                     "waiting 10 min")
                            self._sleep(600)
                    except Exception as e:
                        self.log(f"fetch {rows[0]['dir']} failed: {e}")
                        self.sql("UPDATE visits SET status='failed', error=?, updated=? WHERE dir=?",
                                 (f"fetch: {e}"[:500], _now(), rows[0]["dir"]))
                    continue
                if field is None:
                    self._sleep(30)  # everything listed and fetched
                    continue
                n = self.list_field(field)
                lat = f" (ecliptic lat {field['elat']})" if field["elat"] is not None else ""
                self.log(f"field {field['ra']} {field['dec']}{lat}: {n} visit(s)")
            except Interrupted:
                return
            except Exception as e:  # keep the campaign alive
                self.log(f"fetcher error: {e!r}")
                try:
                    self._sleep(60)
                except Interrupted:
                    return

    def put_back(self, row, e) -> None:
        """Back in the queue for FETCH_DEFER_S; failed after FETCH_MAX_DEFERS."""
        n = (row["fetch_tries"] or 0) + 1
        if n >= FETCH_MAX_DEFERS:
            self.log(f"fetch {row['dir']} failed {n} times, giving up: {e}")
            self.sql("UPDATE visits SET status='failed', fetch_tries=?, error=?, updated=? WHERE dir=?",
                     (n, f"fetch: {e}"[:500], _now(), row["dir"]))
            return
        at = time.time() + FETCH_DEFER_S
        self.log(f"{e}; put back in the queue until {time.strftime('%H:%M', time.localtime(at))} "
                 f"({n}/{FETCH_MAX_DEFERS}), fetching the next visit")
        self.sql("UPDATE visits SET fetch_after=?, fetch_tries=?, updated=? WHERE dir=?",
                 (at, n, _now(), row["dir"]))

    # ---- searching
    def _run(self, worker, args, log_path, env) -> int:
        """A step as a subprocess (its own process group); killed, and
        Interrupted raised, when the campaign leaves the active state."""
        with open(log_path, "w") as fh:
            p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 cwd=str(Path(__file__).resolve().parent.parent), start_new_session=True)
            self.procs[worker] = p
            pidfile = self.out / "running" / f"worker{worker}.pid"
            pidfile.parent.mkdir(exist_ok=True)
            pidfile.write_text(str(p.pid))
            try:
                while p.poll() is None:
                    if not self.active.is_set() or self.stop.is_set():
                        os.killpg(p.pid, signal.SIGTERM)
                        try:
                            p.wait(20)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL)
                            p.wait()
                        raise Interrupted(self.reason or "stopping")
                    time.sleep(1)
                return p.returncode
            finally:
                self.procs.pop(worker, None)
                pidfile.unlink(missing_ok=True)

    def search_visit(self, worker, d: str, redo: bool = False) -> None:
        """Search visit ``d`` and record it; ``redo``: it was searched
        before, only gate and make the blind sheets again."""
        t0 = time.time()
        env = {**os.environ, "NEOFIND_GPU_LIMIT": str(self.s.gpu_limit), "PYTHONUNBUFFERED": "1",
               "NEOFIND_GPU_LOCK": str(self.out / "gpu.lock")}  # the limit is for all searches together
        threads = self.s.threads or max(1, (os.cpu_count() or 4) // self.s.jobs)
        py = [sys.executable, "-m"]
        steps = [
            ("detect", py + ["neodetect", "detect", d, "--threads", str(threads),
                             "--stack-rate-max", str(self.s.stack_rate_max)]),
            ("review", py + ["neodetect", "review", d, "--list", "--no-interactive"]),
            ("gate", py + ["neowise", "gate", d, "--min-score", str(self.s.w4_min)]),
            ("sheets", py + ["neowise", "sheets", d]),
        ]
        if redo:
            steps = steps[2:]
        for name, args in steps:
            rc = self._run(worker, args, Path(d) / f"{name}.log", env)
            if rc != 0:
                tail = (Path(d) / f"{name}.log").read_text().strip().splitlines()[-1:] or [""]
                raise RuntimeError(f"{name} exited {rc}: {tail[0][:300]}")
        self.record(d, time.time() - t0, redo)

    def record(self, d: str, seconds: float, redo: bool = False) -> None:
        doc = json.loads((Path(d) / "detections.json").read_text())
        dets = doc["detections"]
        found = [(i, x) for i, x in enumerate(dets) if x["detected"]]
        cands = [(i, x) for i, x in found if x.get("status") in REVIEWABLE]
        with self.lock, self.db:
            for i, x in found:  # all of them: the review is blind
                nk = x.get("near_known") or {}
                near = f"{nk['name']} {nk['sep_arcsec']}\"" if nk else None
                self.db.execute(
                    "INSERT INTO candidates (visit_dir, det, status, gate, w4_score, score, n_frames, ra, dec, "
                    "t_ref, rate, pa, mag, near, sheet, known_name, arc_arcsec, span_hours) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT (visit_dir, det) DO UPDATE SET status=excluded.status, gate=excluded.gate, "
                    "w4_score=excluded.w4_score, score=excluded.score, near=excluded.near, sheet=excluded.sheet, "
                    "known_name=excluded.known_name, arc_arcsec=excluded.arc_arcsec, span_hours=excluded.span_hours",
                    (d, i, x.get("status"), x.get("w4_gate"), x.get("w4_score"), x["score"], x["n_frames_used"],
                     x["ra_deg"], x["dec_deg"], x["t_ref_mjd"], x["rate_deg_day"], x["pa_deg"], x.get("mag"),
                     near, str(Path(d) / BLIND_DIR / f"det{i:03d}.png"), known_name(x) or "", *arc(x)))
            self.db.execute("UPDATE visits SET status='done', detected=?, known=?, n_candidates=?, "
                            "seconds=COALESCE(?, seconds), searched_at=COALESCE(searched_at, ?), error=NULL, "
                            "updated=?, redo=0 WHERE dir=?",
                            (len(found), sum(x.get("status") == "known" for _, x in found), len(cands),
                             None if redo else seconds, None if redo else time.time(), _now(), d))
        if not self.s.keep_fits:  # W3 FITS stay for the ADES report until the review settles it
            self.delete_fits(d, ("3", "4") if not cands else ("4",))
        sus = [i for i, x in cands if x.get("status") == "suspect" and x.get("w4_gate") in ("pass", "no-w4")]
        self.log(f"{d}: done in {seconds / 60:.1f} min, {len(found)} detected "
                 f"({sum(x.get('status') == 'known' for _, x in found)} known), "
                 f"{len(cands)} to classify, suspects for review {sus}")

    def delete_fits(self, d: str, bands=("3", "4")) -> None:
        delete_fits(d, bands)

    def worker(self, k: int) -> None:
        while not self.stop.is_set():
            try:
                self._wait_active()
            except Interrupted:
                return
            with self.lock, self.db:
                full = self.max_visits is not None and (
                    self.visits_processed() + sum(1 for x in self.current.values() if not x[1])) >= self.max_visits
                r = self.db.execute("SELECT dir, attempts, redo FROM visits WHERE status='fetched'"
                                    + (" AND redo=1" if full else "") + " ORDER BY redo DESC, updated, "  # first in, first out
                                    "(SELECT ord FROM fields WHERE id=field_id), mjd LIMIT 1").fetchall()
                row = r[0] if r else None
                if row:
                    self.db.execute("UPDATE visits SET status='searching', attempts=attempts+1, updated=? "
                                    "WHERE dir=?", (_now(), row["dir"]))
            if row is None:
                if self.stop.wait(5):
                    return
                continue
            d, redo = row["dir"], bool(row["redo"])
            self.current[k] = (d, redo)
            self.log(f"worker {k}: {'blind sheets for' if redo else 'searching'} {d} (attempt {row['attempts'] + 1})")
            try:
                self.search_visit(k, d, redo)
            except Interrupted as e:
                self.log(f"worker {k}: {d} interrupted ({e}); it will be searched again")
                self.sql("UPDATE visits SET status='fetched', attempts=attempts-1, updated=? WHERE dir=?",
                         (_now(), d))
            except Exception as e:
                failed = row["attempts"] + 1 >= self.s.max_attempts
                self.log(f"worker {k}: {d} failed: {e}" + (" (giving up)" if failed else " (will retry)"))
                self.sql("UPDATE visits SET status=?, error=?, updated=? WHERE dir=?",
                         ("failed" if failed else "fetched", str(e)[:500], _now(), d))
            finally:
                self.current.pop(k, None)

    # ---- main loop
    def write_status(self) -> None:
        st = status_summary(self.out, db=self.db, lock=self.lock)
        st.update(state=self.state, reason=self.reason, hours=self.s.hours, pid=os.getpid(),
                  current=[x[0] for x in self.current.values()], updated=_now(),
                  active_seconds=self.active_seconds(), max_visits=self.max_visits,
                  settings=asdict(self.s))
        _write_json(self.out / STATUS_NAME, st)

    def kill_orphans(self) -> None:
        """Searches left running by a campaign process that died (their own
        process groups outlive it): kill them before searching again."""
        for f in (self.out / "running").glob("worker*.pid"):
            try:
                pid = int(f.read_text())
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except (OSError, ValueError):
                cmd = ""
            if "neodetect" in cmd or "neowise gate" in cmd or "neowise sheets" in cmd:
                self.log(f"killing search {pid} left by the previous run")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass
            f.unlink(missing_ok=True)

    def run(self, max_visits: Optional[int] = None) -> None:
        self.max_visits = max_visits
        if (self.out / STATUS_NAME).exists():
            prev = json.loads((self.out / STATUS_NAME).read_text())
            if prev.get("pid") and prev["pid"] != os.getpid() and _is_campaign(prev["pid"]):
                raise SystemExit(f"campaign already running in {self.out} (pid {prev['pid']})")
        self.kill_orphans()
        # A crash leaves visits marked 'searching': search them again.
        n = self._count("SELECT COUNT(*) FROM visits WHERE status='searching'")
        if n:
            self.sql("UPDATE visits SET status='fetched', attempts=MAX(attempts-1, 0) WHERE status='searching'")
            self.log(f"resuming: {n} interrupted search(es) requeued")
        if read_control(self.out).get("stop"):
            set_control(self.out, stop=False)
        self.set_meta("settings", asdict(self.s))
        if self.s.wide_share > 0 and self.meta("wide_since") is None:
            self.set_meta("wide_since", time.time())
        when = (f"--automate{'-now' if self.s.automate == 'now' else ''} (UDP {self.s.udp_port})"
                if self.s.automate else f"hours {self.s.hours}")
        self.log(f"campaign {self.out}: {self._count('SELECT COUNT(*) FROM fields')} fields, "
                 f"{self.visits_processed()} visits processed; {when}, {self.s.jobs} jobs, "
                 f"GPU limit {self.s.gpu_limit:.0%}" + (f", stop after {max_visits} visits" if max_visits else "")
                 + (f"; {self.s.wide_share:.0%} of the time beyond ecliptic latitude {self.s.lat_max:g}"
                    if self.s.wide_share > 0 else ""))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *a: self.stop.set())
        if self.s.automate:
            from .automate import SlaveGate
            self.gate = SlaveGate(self.out, self.s.automate, self.s.udp_port, self.s.controller_ip,
                                  self.s.resume_delay * 60, log=self.log)
            self.gate.start()
        threads = [threading.Thread(target=self.fetcher, name="fetcher", daemon=True)]
        threads += [threading.Thread(target=self.worker, args=(k,), name=f"worker{k}", daemon=True)
                    for k in range(self.s.jobs)]
        for t in threads:
            t.start()
        try:
            while not self.stop.is_set():
                self._update_active()
                if self._finished():
                    self.state, self.reason = "done", "every field searched"
                    self.log("campaign finished: every field searched")
                    break
                self.write_status()
                self.stop.wait(2)
        finally:
            self.stop.set()
            self.active.clear()
            for t in threads:
                t.join(60)
            if self.gate is not None:
                self.gate.close()
            if self._active_since is not None:
                self.set_meta("active_seconds", self.meta("active_seconds", 0.0) + time.time() - self._active_since)
                self._active_since = None
            if self.state not in ("done",):
                self.state, self.reason = "stopped", self.reason or "stop requested (campaign stop, Ctrl+C or SIGTERM)"
            self.write_status()
            self.log(f"campaign stopped: {self.reason}")

    def _finished(self) -> bool:
        return (self._count("SELECT COUNT(*) FROM fields WHERE status='pending'") == 0
                and self._count("SELECT COUNT(*) FROM visits WHERE status IN ('listed','fetched','searching')") == 0)


def _have_frames(d: Path, frames) -> bool:
    """Every frame's int, unc and msk files are in ``d`` (a download cut
    short, by a stop or by giving up, leaves only some)."""
    return all(any(d.glob(f"{f.frame_id}-w{f.band}-{kind}-1b.fits*"))
               for f in frames for kind in ("int", "unc", "msk"))


class _GaveUp(RuntimeError):
    """Retried as many times as allowed."""


class _Missing(Exception):
    def __init__(self, frame_id):
        super().__init__(frame_id)
        self.frame_id = frame_id


# ------------------------------------------------------------------ status

def status_summary(out, db=None, lock=None) -> dict:
    """Counts, throughput and the projected time to finish."""
    db = db or connect(out)
    lock = lock or threading.RLock()
    with lock:
        q = lambda s, a=(): db.execute(s, a).fetchone()[0]
        fields = {r[0]: r[1] for r in db.execute("SELECT status, COUNT(*) FROM fields GROUP BY status")}
        visits = {r[0]: r[1] for r in db.execute("SELECT status, COUNT(*) FROM visits WHERE redo=0 GROUP BY status")}
        redo = {r[0]: r[1] for r in db.execute("SELECT status, COUNT(*) FROM visits WHERE redo=1 GROUP BY status")}
        gates = {f"{r[0]}/{r[1]}": r[2] for r in db.execute(
            "SELECT status, gate, COUNT(*) FROM candidates GROUP BY status, gate")}
        review = {str(r[0] or "to review"): r[1] for r in db.execute(  # blind: every detection
            f"SELECT verdict, COUNT(*) FROM candidates WHERE sheet LIKE '%/{BLIND_DIR}/%' GROUP BY verdict")}
        waiting = q(f"SELECT COUNT(*) FROM candidates WHERE sheet IS NULL OR sheet NOT LIKE '%/{BLIND_DIR}/%'")
        put_back = q("SELECT COUNT(*) FROM visits WHERE status='listed' AND fetch_after > ?", (time.time(),))
        per_visit = [r[0] for r in db.execute(
            "SELECT seconds FROM visits WHERE (status='done' OR redo=1) AND seconds IS NOT NULL "
            "ORDER BY updated DESC LIMIT 100")]
        listed = q("SELECT COUNT(*) FROM fields WHERE status IN ('listed','empty')")
        n_listed_visits = q("SELECT COALESCE(SUM(n_visits),0) FROM fields WHERE status IN ('listed','empty')")
        active = json.loads((db.execute("SELECT value FROM meta WHERE key='active_seconds'").fetchone()
                             or ["0"])[0])
        settings = json.loads((db.execute("SELECT value FROM meta WHERE key='settings'").fetchone() or ["{}"])[0])
        share = settings.get("wide_share", 0.0)
        bands = None
        if share > 0:  # the mix: the projection is for the near band, at its share of the time
            wide = f"COALESCE(ABS(elat), 0) > {float(settings.get('lat_max', 10.0))}"
            since = json.loads((db.execute("SELECT value FROM meta WHERE key='wide_since'").fetchone() or ["0"])[0])
            bands, secs = {}, {}
            for w, name in ((0, "near"), (1, "wide")):
                inb = f"field_id IN (SELECT id FROM fields WHERE ({wide}) = {w})"
                secs[name] = q(f"SELECT COALESCE(SUM(seconds), 0) FROM visits WHERE status='done' "
                               f"AND searched_at >= ? AND {inb}", (since,))
                bands[name] = {
                    "fields": q(f"SELECT COUNT(*) FROM fields WHERE ({wide}) = {w}"),
                    "fields_listed": q(f"SELECT COUNT(*) FROM fields WHERE ({wide}) = {w} AND status IN ('listed','empty')"),
                    "visits_done": q(f"SELECT COUNT(*) FROM visits WHERE status='done' AND {inb}"),
                    "hours_since_mix": round(secs[name] / 3600, 2)}
            tot = secs["near"] + secs["wide"]
            bands["wide"]["share_since_mix"] = round(secs["wide"] / tot, 3) if tot else None
            bands["target_share"] = share
            near = f"field_id IN (SELECT id FROM fields WHERE NOT {wide})"
            near_fields = q(f"SELECT COUNT(*) FROM fields WHERE NOT {wide}")
            listed = q(f"SELECT COUNT(*) FROM fields WHERE NOT {wide} AND status IN ('listed','empty')")
            n_listed_visits = q(f"SELECT COALESCE(SUM(n_visits),0) FROM fields WHERE NOT {wide} "
                                "AND status IN ('listed','empty')")
            near_visits = {r[0]: r[1] for r in db.execute(
                f"SELECT status, COUNT(*) FROM visits WHERE redo=0 AND {near} GROUP BY status")}
    processed = visits.get("done", 0) + visits.get("failed", 0) + sum(redo.values())
    st = {"fields": fields, "visits": visits, "candidates": gates, "review": review, "put_back": put_back,
          "blind_sheets_pending": {"visits": sum(n for k, n in redo.items() if k != "failed"),
                                   "failed": redo.get("failed", 0), "detections": waiting},
          "median_minutes_per_visit": round(float(np.median(per_visit)) / 60, 2) if per_visit else None}
    total_fields = near_fields if bands else sum(fields.values())
    if bands:
        st["bands"] = bands
    if listed and processed and active > 600:
        vpf = n_listed_visits / listed
        queued = near_visits if bands else visits
        remaining = (total_fields - listed) * vpf + sum(queued.get(s, 0) for s in ("listed", "fetched", "searching"))
        rate = processed / (active / 3600.0)  # visits per active hour
        if bands:  # the near band gets 1 - share of the time
            rate *= 1.0 - share
        st.update(visits_per_field=round(vpf, 2), visits_per_active_hour=round(rate, 1),
                  remaining_visits=int(round(remaining)), projected_active_hours=round(remaining / rate))
        if not settings.get("automate"):  # with --automate the hours aren't known in advance
            a, b = parse_hours(settings.get("hours", "10-17"))
            st["projected_days"] = round(remaining / rate / ((b - a) % 24 or 24), 1)
    return st


def backfill_sheets(out) -> int:
    """Queue the searched visits that have no blind sheets (searched before
    the review was blind) to be fetched again, gated and given sheets for
    every detection; they go before new visits. How many."""
    db = connect(out)
    rows = db.execute("SELECT dir FROM visits WHERE status='done' AND detected > 0").fetchall()
    todo = [r["dir"] for r in rows if not (Path(r["dir"]) / BLIND_DIR).is_dir()]
    with db:
        db.executemany("UPDATE visits SET status='listed', redo=1, attempts=0 WHERE dir=?", [(d,) for d in todo])
    return len(todo)


def retry_failed(out) -> int:
    """Requeue failed visits (e.g. after a long archive outage); those
    failed at a download, and blind-sheet redos (their files are checked
    and fetched again), go back to be fetched."""
    db = connect(out)
    with db:
        return db.execute("UPDATE visits SET status = CASE WHEN error LIKE 'fetch:%' OR redo=1 THEN 'listed' "
                          "ELSE 'fetched' END, attempts=0, fetch_tries=0, fetch_after=NULL, error=NULL "
                          "WHERE status='failed'").rowcount


def print_status(out) -> None:
    st = status_summary(out)
    live = {}
    p = Path(out) / STATUS_NAME
    if p.exists():
        live = json.loads(p.read_text())
    alive = bool(live.get("pid")) and _is_campaign(live["pid"]) and live.get("state") not in ("stopped", "done")
    print(f"campaign {out}: " + (f"{live.get('state')} ({live.get('reason') or 'working'})" if alive
                                 else f"not running (last state: {live.get('state', 'never run')})"))
    ctl = read_control(out)
    if ctl.get("paused"):
        print("  paused (campaign resume to continue)")
    for d in live.get("current", []) if alive else []:
        print(f"  searching {d}")
    print(f"  fields: {st['fields']}")
    print(f"  visits: {st['visits']}")
    print(f"  candidates (status/gate): {st['candidates']}")
    print(f"  blind review (all detections with sheets): {st['review']}")
    bp = st["blind_sheets_pending"]
    if bp["visits"] or bp["failed"] or bp["detections"]:
        print(f"  waiting for blind sheets: {bp['detections']} detections; {bp['visits']} visits queued"
              + (f", {bp['failed']} failed (retry-failed)" if bp["failed"] else ""))
    if st["put_back"]:
        print(f"  {st['put_back']} visit(s) put back in the queue for an hour after download failures")
    if st.get("median_minutes_per_visit"):
        print(f"  {st['median_minutes_per_visit']} min per visit (median)")
    b = st.get("bands")
    if b:
        w = b["wide"]
        print(f"  near the ecliptic: {b['near']['visits_done']} visits searched, {b['near']['fields_listed']}/"
              f"{b['near']['fields']} fields listed; beyond it: {w['visits_done']} visits, {w['fields_listed']}/"
              f"{w['fields']} fields")
        if w["share_since_mix"] is not None:
            print(f"  time beyond the band since the mix began: {w['share_since_mix']:.0%} "
                  f"(target {b['target_share']:.0%}; {w['hours_since_mix']} h of "
                  f"{w['hours_since_mix'] + b['near']['hours_since_mix']:.2f} h searching)")
    if st.get("projected_active_hours") is not None:
        left = (f"~{st['projected_days']} days at these hours" if st.get("projected_days") is not None
                else f"~{st['projected_active_hours']} active hours")
        print(f"  {st['visits_per_active_hour']} visits per active hour, {st['visits_per_field']} visits per field; "
              f"~{st['remaining_visits']} visits left{' near the ecliptic' if b else ''}: {left}")


def _is_campaign(pid: int) -> bool:
    """``pid`` is a running ``neowise campaign`` (not a reused pid after a reboot)."""
    try:
        return b"campaign" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
