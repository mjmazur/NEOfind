"""Result plots of a campaign: ``neowise plots OUT`` (PNGs) and the review
page's Plots tab (``data``, drawn in the browser).

* Completeness: of the catalogued asteroids that crossed the searched
  fields (on at least ``MIN_FRAMES`` frames), the fraction found, against
  the signal the detector measures at the asteroid's predicted track, its
  rate, and its orbit class. This is the search's detection limit.
* Motion: each confirmed or likely detection's motion in ecliptic
  coordinates (degrees per day in longitude and latitude). Catalogued main-
  belt asteroids form a tight clump; a detection not catalogued far from
  it moves unlike a main-belt asteroid, the first sign of an NEO.
* Gallery: the confirmed and likely detections not catalogued, with their
  cutout sheets, most unusual motion first.
* Astrometry: the confirmed and likely catalogued detections' positions
  minus the catalogue's (SkyBoT, shifted to the spacecraft) on the frames
  they were measured on; what the MPC submission's positions are worth.
* Before discovery: for each catalogued asteroid confirmed (or likely),
  its discovery date minus the date of the earliest WISE image it was
  found on: those to the right were seen by WISE before anyone discovered
  them. Numbered asteroids use their credited discovery (JPL's Small-Body
  Database, kept in ``OUT/discovery_cache.json``); unnumbered ones, which
  have none, the date their provisional designation encodes.

and, to check the search itself:

* Scores: the track scores of the detections by verdict, and the fraction
  of those reviewed judged real (confirmed or likely) by score. The review
  goes in a random order, so this also estimates how many of the ones not
  yet reviewed are real.
* Brightness: the W3 magnitudes of the catalogued confirmed detections and
  of those not catalogued.
* Galactic latitude: detections per visit, catalogued or not, by the
  field's galactic latitude: crowded star fields make false tracks.
* Along the ecliptic: each visit's detections by ecliptic longitude, with
  the galactic plane's crossings marked.
* Solar system: where the catalogued asteroids found were, seen from
  above the ecliptic, by orbit class; grey, every catalogued asteroid on a
  searched field.
* Progress: visits searched and detections over time (from the campaign's
  log), with the times the campaign wasn't running shaded.

What each visit contributes is read once from its detections.json and
known.json (about 1.3 MB a visit) and kept in ``OUT/plots_cache.json``.
"""

import functools
import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import campaign as C

CACHE_NAME = "plots_cache.json"
CACHE_VERSION = 3
MIN_FRAMES = 3  # the detector's fewest frames for a track
OBLIQUITY = 23.4392911
WISE_PIXEL = 2.75  # arcsec
SNR_EDGES = (2, 4, 6, 8, 10, 12, 15, 20, 30, 50, 100)
RATE_EDGES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0)
CLASS_ORDER = ("Hungaria", "MB inner", "MB middle", "MB outer", "Cybele", "Hilda", "Trojan",
               "Mars-crosser", "NEA", "other")
# Categorical slots 1 and 2 of the reference palette, light surface (as the sky map)
COLORS = {"catalogued": "#2a78d6", "not catalogued": "#eb6834"}


def orbit_class(c: str) -> str:
    """SkyBoT's class, grouped: "MB>Inner" -> "MB inner", "NEA>Apollo" -> "NEA"."""
    c = c or ""
    if c.startswith("NEA"):
        return "NEA"
    if c.startswith("MB>"):
        sub = c[3:]
        return {"Inner": "MB inner", "Middle": "MB middle", "Outer": "MB outer"}.get(sub, sub)
    if c == "Mars-Crosser":
        return "Mars-crosser"
    return c if c in ("Hungaria", "Trojan") else "other"


# ---- reading the visits

def _visit_summary(d: Path) -> dict:
    """What the plots need from one visit: every catalogued asteroid that
    crossed it (found or not), and the catalogued detections' residuals."""
    doc = json.loads((d / "detections.json").read_text())
    try:
        known = json.loads((d / "known.json").read_text())["asteroids"]
    except (OSError, ValueError, KeyError):
        known = []
    by_id = {str(a["id"]): a for a in known}
    found = {str(x["known_id"]): i for i, x in enumerate(doc["detections"])
             if x["detected"] and x.get("known_id") is not None}
    crossed = []
    for k in doc.get("known_eval", []):
        a = by_id.get(str(k["id"]), {})
        on = [p for p in a.get("frames", []) if p["in_frame"]] or a.get("frames", [])
        mid = on[len(on) // 2] if on else None
        # [id, class, rate, frames covered, signal at the track, found, its detection, distance (AU),
        #  ra, dec, mjd, then heliocentric ecliptic x, y (AU)]
        crossed.append([str(k["id"]), orbit_class(a.get("class")), k["rate_deg_day"], k["n_frames_covered"],
                        k["measured_at_truth"], str(k["id"]) in found, found.get(str(k["id"]), -1),
                        a.get("dist_au"), mid and round(mid["ra_deg"], 5), mid and round(mid["dec_deg"], 5),
                        mid and round(mid["mjd"], 5)])
    where = [c for c in crossed if c[7] is not None and c[10] is not None]
    if where:
        x, y = heliocentric(*zip(*[(c[8], c[9], c[7], c[10]) for c in where]))
        for c, p, q in zip(where, x, y):
            c += [round(float(p), 3), round(float(q), 3)]
    astrom = {}
    for aid, i in found.items():
        a, x = by_id.get(aid), doc["detections"][i]
        if a is None:
            continue
        res, errs = [], []
        for p, q in zip(a["frames"], x["frames"]):
            if not (p["in_frame"] and q["used"]):
                continue
            dra = ((q["ra_deg"] - p["ra_deg"] + 180.0) % 360.0 - 180.0) * math.cos(math.radians(p["dec_deg"]))
            res.append([round(dra * 3600.0, 3), round((q["dec_deg"] - p["dec_deg"]) * 3600.0, 3)])
        errs = [e for e, p in zip(a.get("err") or [], a["frames"]) if p["in_frame"]]
        if res:
            astrom[str(i)] = {"class": orbit_class(a.get("class")), "res": res,
                              "eph_err": max(errs) if errs else a.get("err_arcsec")}
    return {"crossed": crossed, "astrom": astrom}


def load_visits(out, db=None, cache_path=None) -> dict:
    """{visit dir: summary} for every searched visit, from the cache where
    the visit's detections.json hasn't changed since."""
    out = Path(out)
    cache_path = Path(cache_path) if cache_path else out / CACHE_NAME
    try:
        cache = json.loads(cache_path.read_text())
        if cache.get("version") != CACHE_VERSION:
            cache = {}
    except (OSError, ValueError):
        cache = {}
    old = cache.get("visits", {})
    db = db or C.connect(out)
    dirs = [r[0] for r in db.execute("SELECT dir FROM visits WHERE status='done' ORDER BY dir")]
    visits, changed = {}, False
    for d in dirs:
        p = Path(d) / "detections.json"
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        v = old.get(d)
        if v is None or v["mtime"] != mtime:
            try:
                v = dict(_visit_summary(Path(d)), mtime=mtime)
            except (OSError, ValueError, KeyError):
                continue
            changed = True
        visits[d] = v
    if changed or len(visits) != len(old):
        tmp = cache_path.with_name(cache_path.name + ".part")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "visits": visits}))
        tmp.replace(cache_path)
    return visits


# ---- the four plots' data

def _wilson(k: int, n: int, z: float = 1.0):
    """The fraction k/n and its 68% (z=1) Wilson interval."""
    if n == 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, min(p, max(0.0, mid - half)), max(p, min(1.0, mid + half))


def _binned(rows, value, edges, labels=None):
    out = []
    for i, (lo, hi) in enumerate(zip((-math.inf,) + tuple(edges), tuple(edges) + (math.inf,))):
        sel = [r for r in rows if r[value] is not None and lo <= r[value] < hi]
        if not sel:
            continue
        k = sum(r["found"] for r in sel)
        p, a, b = _wilson(k, len(sel))
        xs = sorted(r[value] for r in sel)
        out.append({"lo": None if lo == -math.inf else lo, "hi": None if hi == math.inf else hi,
                    "x": xs[len(xs) // 2], "n": len(sel), "found": k, "frac": p, "lo68": a, "hi68": b})
    return out


def completeness(visits: dict, snr_floor: float = 10.0) -> dict:
    """Fraction found of the catalogued asteroids on >= MIN_FRAMES frames,
    against the signal at their track ("snr", the detector's statistic on
    the predicted track: what a perfect search would score), their rate,
    and their class; rate and class only for those with ``snr`` above
    ``snr_floor``, so it's the search, not the signal, that is compared."""
    rows = [{"id": c[0], "class": c[1], "rate": c[2], "snr": c[4], "found": bool(c[5])}
            for v in visits.values() for c in v["crossed"] if c[3] >= MIN_FRAMES]
    for r in rows:
        if r["snr"] is not None and r["snr"] < SNR_EDGES[0]:
            r["snr"] = SNR_EDGES[0] * 0.75  # nothing there: plotted in the first bin
    bright = [r for r in rows if r["snr"] is not None and r["snr"] >= snr_floor]
    by_class = []
    for c in CLASS_ORDER:
        sel = [r for r in bright if r["class"] == c]
        if sel:
            k = sum(r["found"] for r in sel)
            p, a, b = _wilson(k, len(sel))
            by_class.append({"class": c, "n": len(sel), "found": k, "frac": p, "lo68": a, "hi68": b})
    k = sum(r["found"] for r in rows)
    return {"n": len(rows), "found": k, "snr_floor": snr_floor, "min_frames": MIN_FRAMES,
            "n_bright": len(bright), "found_bright": sum(r["found"] for r in bright),
            "by_snr": _binned(rows, "snr", SNR_EDGES), "by_rate": _binned(bright, "rate", RATE_EDGES),
            "by_class": by_class}


def ecliptic_motion(ra, dec, rate, pa):
    """(lambda, beta, d lambda cos beta / dt, d beta / dt) in degrees and
    degrees per day, for a motion of ``rate`` deg/day at position angle
    ``pa`` (east of north)."""
    h = 1e-3  # day
    e = math.radians(OBLIQUITY)

    def ecl(r, d):
        r, d = math.radians(r), math.radians(d)
        x, y, z = math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d)
        y, z = y * math.cos(e) + z * math.sin(e), -y * math.sin(e) + z * math.cos(e)
        return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, z))))

    s = rate * h
    ra2 = ra + s * math.sin(math.radians(pa)) / max(math.cos(math.radians(dec)), 1e-9)
    dec2 = dec + s * math.cos(math.radians(pa))
    l1, b1 = ecl(ra, dec)
    l2, b2 = ecl(ra2, dec2)
    dl = (l2 - l1 + 180.0) % 360.0 - 180.0
    return l1, b1, dl * math.cos(math.radians(b1)) / h, (b2 - b1) / h


def motion(db, visits: dict, include_likely: bool = True) -> dict:
    """Every confirmed (and likely) detection's ecliptic motion, its
    catalogue and orbit class; for those not catalogued, how far the motion
    is from the catalogued main-belt asteroids' ("sigma": the distance in
    units of their robust spread, the larger axis of it)."""
    verdicts = ("confirmed", "likely") if include_likely else ("confirmed",)
    q = (f"SELECT id, visit_dir, det, ra, dec, rate, pa, mag, score, t_ref, verdict, status, known_name, near, "
         f"n_frames, NULLIF(arc_arcsec, -1) arc_arcsec, NULLIF(span_hours, -1) span_hours "
         f"FROM candidates WHERE verdict IN ({','.join('?' * len(verdicts))}) ORDER BY id")
    objs = []
    for r in db.execute(q, verdicts).fetchall():
        o = dict(r)
        o["lam"], o["beta"], o["vl"], o["vb"] = (round(v, 5) for v in
                                                 ecliptic_motion(o["ra"], o["dec"], o["rate"], o["pa"]))
        o["catalogue"] = "catalogued" if o["status"] == "known" else "not catalogued"
        a = visits.get(o["visit_dir"], {}).get("astrom", {}).get(str(o["det"]))
        o["class"] = a["class"] if a else None
        objs.append(o)
    mb = np.array([[o["vl"], o["vb"]] for o in objs if (o["class"] or "").startswith("MB")])
    ref = None
    if len(mb) >= 10:
        med = np.median(mb, axis=0)
        mad = 1.4826 * np.median(np.abs(mb - med), axis=0)
        ref = {"vl": float(med[0]), "vb": float(med[1]), "sl": float(max(mad[0], 1e-3)),
               "sb": float(max(mad[1], 1e-3)), "n": len(mb)}
        for o in objs:
            o["sigma"] = round(math.hypot((o["vl"] - med[0]) / ref["sl"], (o["vb"] - med[1]) / ref["sb"]), 1)
    return {"objects": objs, "main_belt": ref}


def gallery(mot: dict) -> list:
    """The detections not catalogued, the most unusual motion first."""
    sel = [o for o in mot["objects"] if o["catalogue"] == "not catalogued"]
    return sorted(sel, key=lambda o: (-(o.get("sigma") or 0.0), -o["rate"]))


def astrometry(db, visits: dict, include_likely: bool = True, max_eph_err: float = 2.0) -> dict:
    """The confirmed (and likely) catalogued detections' offsets from the
    catalogue, in arcsec: each one's median over its frames, and every
    frame's; only where the ephemeris is good to ``max_eph_err``."""
    verdicts = ("confirmed", "likely") if include_likely else ("confirmed",)
    q = (f"SELECT id, visit_dir, det, mag, score, n_frames, rate, t_ref, verdict, known_name FROM candidates "
         f"WHERE status='known' AND verdict IN ({','.join('?' * len(verdicts))}) ORDER BY id")
    objs, frames, loose = [], [], 0
    for r in db.execute(q, verdicts).fetchall():
        a = visits.get(r["visit_dir"], {}).get("astrom", {}).get(str(r["det"]))
        if not a:
            continue
        if a["eph_err"] is not None and a["eph_err"] > max_eph_err:
            loose += 1
            continue
        res = np.array(a["res"])
        o = dict(r)
        o["dra"], o["ddec"] = (round(float(v), 3) for v in np.median(res, axis=0))
        o["sep"] = round(math.hypot(o["dra"], o["ddec"]), 3)
        o["eph_err"], o["n_obs"], o["class"] = a["eph_err"], len(res), a["class"]
        objs.append(o)
        frames += a["res"]
    f = np.array(frames) if frames else np.zeros((0, 2))
    stats = None
    if len(f):
        med = np.median(f, axis=0)
        stats = {"n_obs": len(f), "median_dra": round(float(med[0]), 3), "median_ddec": round(float(med[1]), 3),
                 "rms_dra": round(float(np.sqrt(np.mean(f[:, 0] ** 2))), 3),
                 "rms_ddec": round(float(np.sqrt(np.mean(f[:, 1] ** 2))), 3),
                 "p68": round(float(np.percentile(np.hypot(f[:, 0], f[:, 1]), 68)), 3),
                 "p95": round(float(np.percentile(np.hypot(f[:, 0], f[:, 1]), 95)), 3)}
    return {"objects": objs, "frames": [[round(a, 2), round(b, 2)] for a, b in frames], "stats": stats,
            "max_eph_err": max_eph_err, "loose_ephemeris": loose, "pixel": WISE_PIXEL}


MIN_REVIEWED = 5  # fewer reviewed in a score bin: its fraction isn't drawn
SCORE_EDGES = (8, 9, 10, 11, 12, 14, 17, 20, 25, 35, 50, 100)
VERDICT_ROWS = (("confirmed", "verdict='confirmed'"), ("likely", "verdict='likely'"),
                ("discarded", "verdict='rejected'"), ("not reviewed", "verdict IS NULL"))


def scores(db) -> dict:
    """Detections per score bin by verdict, and per bin the fraction of the
    reviewed judged real; with it, the real ones expected among those not
    reviewed (the review order is random, so the reviewed are a fair sample)."""
    edges = (-math.inf,) + SCORE_EDGES + (math.inf,)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = {"lo": None if lo == -math.inf else lo, "hi": None if hi == math.inf else hi}
        for name, where in VERDICT_ROWS:
            b[name] = db.execute(f"SELECT COUNT(*) FROM candidates WHERE {where} AND score >= ? AND score < ?",
                                 (lo, hi)).fetchone()[0]
        real, reviewed = b["confirmed"] + b["likely"], b["confirmed"] + b["likely"] + b["discarded"]
        b["reviewed"] = reviewed
        b["frac"], b["lo68"], b["hi68"] = _wilson(real, reviewed)
        xs = [r[0] for r in db.execute("SELECT score FROM candidates WHERE score >= ? AND score < ? ORDER BY score",
                                       (lo, hi))]
        if xs:
            b["x"] = xs[len(xs) // 2]
            bins.append(b)
    expected = sum(b["not reviewed"] * b["frac"] for b in bins if b["frac"] is not None)
    unjudged = sum(b["not reviewed"] for b in bins if b["frac"] is None)
    return {"bins": bins, "rows": [n for n, _ in VERDICT_ROWS], "expected_real": round(expected),
            "not_reviewed": sum(b["not reviewed"] for b in bins), "unjudged": unjudged}


def brightness(db, include_likely: bool = True) -> dict:
    """W3 magnitudes: a histogram of the catalogued, and each one not catalogued."""
    verdicts = ("confirmed", "likely") if include_likely else ("confirmed",)
    rows = db.execute(f"SELECT id, mag, verdict, status, rate, score, near FROM candidates WHERE mag IS NOT NULL "
                      f"AND verdict IN ({','.join('?' * len(verdicts))}) ORDER BY id", verdicts).fetchall()
    cat = [r["mag"] for r in rows if r["status"] == "known"]
    new = [dict(r, catalogue="not catalogued") for r in rows if r["status"] != "known"]
    step = 0.25
    lo = math.floor(min([r["mag"] for r in rows] + [8.0]) / step) * step
    hi = math.ceil(max([r["mag"] for r in rows] + [11.0]) / step) * step
    edges = np.arange(lo, hi + step / 2, step)
    n, _ = np.histogram(cat, edges)
    return {"edges": [round(float(e), 2) for e in edges], "catalogued": [int(k) for k in n], "objects": new,
            "median_catalogued": float(np.median(cat)) if cat else None,
            "median_new": float(np.median([o["mag"] for o in new])) if new else None}


def _visit_rows(db) -> list:
    """Every searched visit: its field's position, and its detections (any
    verdict or none: counts only) catalogued, not catalogued and discarded."""
    return [dict(r) for r in db.execute(
        "SELECT v.dir, v.mjd, f.ra, f.dec, f.elon, f.elat, "
        "(SELECT COUNT(*) FROM candidates c WHERE c.visit_dir=v.dir AND c.status='known') AS cat, "
        "(SELECT COUNT(*) FROM candidates c WHERE c.visit_dir=v.dir AND COALESCE(c.status,'')!='known') AS new, "
        "(SELECT COUNT(*) FROM candidates c WHERE c.visit_dir=v.dir AND c.verdict='rejected') AS discarded "
        "FROM visits v JOIN fields f ON f.id=v.field_id WHERE v.status='done' ORDER BY v.dir")]


def _galactic_b(ra, dec):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    return SkyCoord(np.asarray(ra, float) * u.deg, np.asarray(dec, float) * u.deg).galactic.b.deg


GLAT_EDGES = (0, 5, 10, 15, 20, 30, 45, 60, 90)


def galactic(rows: list) -> dict:
    """Mean detections per visit (catalogued, not catalogued, discarded) by
    the field's |galactic latitude|."""
    if not rows:
        return {"bins": []}
    b = np.abs(_galactic_b([r["ra"] for r in rows], [r["dec"] for r in rows]))
    for r, x in zip(rows, b):
        r["glat"] = round(float(x), 2)
    bins = []
    for lo, hi in zip(GLAT_EDGES[:-1], GLAT_EDGES[1:]):
        sel = [r for r in rows if lo <= r["glat"] < hi]
        if sel:
            bins.append({"lo": lo, "hi": hi, "x": (lo + hi) / 2, "visits": len(sel),
                         **{k: round(sum(r[k] for r in sel) / len(sel), 2) for k in ("cat", "new", "discarded")},
                         **{f"n_{k}": sum(r[k] for r in sel) for k in ("cat", "new", "discarded")}})
    return {"bins": bins}


@functools.lru_cache(maxsize=4)
def ecliptic_plane_crossings(half_width: float = 10.0) -> list:
    """[[lambda from, lambda to], ...] where the ecliptic is within
    ``half_width`` deg of the galactic plane."""
    from .skymap import ecliptic_line
    ra, dec = ecliptic_line(0.0, 3601)
    near = np.abs(_galactic_b(ra, dec)) < half_width
    lam = np.linspace(0, 360, 3601)
    out, start = [], None
    for x, n in zip(lam, near):
        if n and start is None:
            start = x
        if not n and start is not None:
            out.append([round(float(start), 1), round(float(x), 1)])
            start = None
    if start is not None:
        out.append([round(float(start), 1), 360.0])
    return out


def ecliptic(rows: list, visits: dict) -> dict:
    """Each visit by its field's ecliptic longitude: its detections, and
    the catalogued asteroids on it (on >= MIN_FRAMES frames)."""
    out = []
    for r in rows:
        v = visits.get(r["dir"], {})
        out.append({"elon": round(r["elon"], 3), "elat": round(r["elat"], 3), "mjd": r["mjd"], "cat": r["cat"],
                    "new": r["new"], "crossing": sum(c[3] >= MIN_FRAMES for c in v.get("crossed", [])),
                    "glat": r.get("glat"), "field": f"{r['ra']:.3f} {r['dec']:+.3f}"})
    return {"visits": out, "galactic_plane": ecliptic_plane_crossings()}


SOLAR_GROUPS = {"main belt": ("Hungaria", "MB inner", "MB middle", "MB outer"),
                "Cybele, Hilda, Trojan": ("Cybele", "Hilda", "Trojan"),
                "NEA, Mars-crosser": ("NEA", "Mars-crosser")}
PLANETS = {"Earth": 1.0, "Mars": 1.524, "Jupiter": 5.203}


def heliocentric(ra, dec, dist, mjd):
    """Heliocentric ecliptic (x, y) in AU of objects at (ra, dec) and
    ``dist`` AU from the Earth (WISE: within 1e-4 AU of it) at ``mjd``."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(np.asarray(mjd, float), format="mjd", scale="utc")
    e = get_body_barycentric("earth", t).xyz.to_value("AU") - get_body_barycentric("sun", t).xyz.to_value("AU")
    r, d, dist = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float)), np.asarray(dist, float)
    p = e + dist * np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    ob = math.radians(OBLIQUITY)
    return p[0], p[1] * math.cos(ob) + p[2] * math.sin(ob)


def solar_system(db, visits: dict, include_likely: bool = True) -> dict:
    """The catalogued confirmed (and likely) detections' asteroids where they
    were, from above the ecliptic, by group; every catalogued asteroid on a
    searched field (found or not) as context; the Earth's path."""
    verdicts = ("confirmed", "likely") if include_likely else ("confirmed",)
    judged = {(r["visit_dir"], r["det"]): dict(r) for r in db.execute(
        f"SELECT id, visit_dir, det, verdict, known_name, rate FROM candidates WHERE status='known' "
        f"AND verdict IN ({','.join('?' * len(verdicts))})", verdicts)}
    ctx, objs, times = [], [], []
    for d, v in visits.items():
        for c in v.get("crossed", []):
            if len(c) < 13 or c[3] < MIN_FRAMES:
                continue
            ctx.append([c[11], c[12]])
            times.append(c[10])
            o = judged.get((d, c[6]))
            if o:
                group = next((g for g, cls in SOLAR_GROUPS.items() if c[1] in cls), "other")
                objs.append(dict(o, x=c[11], y=c[12], r=round(math.hypot(c[11], c[12]), 3), dist=c[7], mjd=c[10],
                                 orbit=c[1], group=group, catalogue="catalogued"))
    out = {"context": ctx, "objects": objs, "earth": [], "planets": PLANETS,
           "groups": list(SOLAR_GROUPS) + ["other"]}
    if not ctx:
        return out
    t = np.arange(min(times), max(times) + 2, 2.0)
    x, y = heliocentric(np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), t)
    out["earth"] = [[round(float(p), 4), round(float(q), 4)] for p, q in zip(x, y)]
    return out


_DONE = re.compile(r"^(\S+) (/\S+): done in [\d.]+ min, (\d+) detected \((\d+) known\)")
_STATE = re.compile(r"^(\S+) (?:state: (.+)|(campaign stopped).*|campaign (/\S+): .*)$")


IDLE_KINDS = ("outside the hours", "detection run", "paused", "waiting", "not running")


def _idle_kind(why: str) -> str:
    w = why.lower()
    if "active hours" in w:
        return "outside the hours"
    if "slavemsg" in w or "detection run" in w:
        return "detection run"
    if w.startswith("paused"):
        return "paused"
    if w.startswith(("stopped", "done")):
        return "not running"
    return "waiting"


def progress(out) -> dict:
    """From the campaign's log: when each visit was first searched (its
    detections, catalogued or not), and the intervals the campaign wasn't
    running (outside the hours, paused, waiting for a detection run, not
    started), as [start, end, why]; times are the log's, local."""
    try:
        lines = (Path(out) / "campaign.log").read_text().splitlines()
    except OSError:
        return {"visits": [], "idle": []}
    seen, visits, idle, cur = set(), [], [], None  # cur: (since, why) while not running
    t = None
    for ln in lines:
        m = _DONE.match(ln)
        if m:
            t = m[1]
            if m[2] not in seen:
                seen.add(m[2])
                visits.append([t, int(m[3]), int(m[4])])
            continue
        m = _STATE.match(ln)
        if not m:
            continue
        t = m[1]
        why = m[2] if m[2] is not None else ("stopped" if m[3] else "running")
        running = why.startswith("running")
        why = _idle_kind(why)
        if running and cur:
            idle.append([cur[0], t, cur[1]])
            cur = None
        elif not running and cur is None:
            cur = (t, why)
        elif not running and cur and why != cur[1]:
            idle.append([cur[0], t, cur[1]])
            cur = (t, why)
    if cur:
        idle.append([cur[0], None, cur[1]])  # still
    def iso(t):  # the log's local time, with its offset
        return datetime.fromisoformat(t).astimezone().isoformat(timespec="seconds") if t else None

    visits = [[iso(r[0]), r[1], r[2]] for r in visits]
    idle = [[iso(a), iso(b), k] for a, b, k in idle]
    merged = []
    for iv in idle:
        if merged and merged[-1][2] == iv[2] and merged[-1][1] == iv[0]:
            merged[-1][1] = iv[1]
        elif iv[1] is None or iv[1] > iv[0]:
            merged.append(list(iv))
    return {"visits": visits, "idle": merged, "kinds": IDLE_KINDS}


# ---- discovery dates

DISCOVERY_CACHE = "discovery_cache.json"
LAG_MIN = -30  # years; earlier discoveries go in one bin
# Half-month letters of a provisional designation (no I): A = Jan 1-15, B = Jan 16-31, ...
HALF_MONTHS = "ABCDEFGHJKLMNOPQRSTUVWXY"
# The survey designations' observing dates (Palomar-Leiden and the Trojan surveys)
SURVEYS = {"P-L": "1960-09-24", "T-1": "1971-03-25", "T-2": "1973-09-19", "T-3": "1977-10-16"}
_PROV = re.compile(r"\b(1[89]\d\d|20\d\d) ([A-HJ-Y])[A-HJ-Z]\d*\b")
_SURVEY = re.compile(r"\b\d{4} (P-L|T-[123])\b")


def _known(name: str):
    """(number or None, designation or None) of a catalogued name such as
    '2002 EH16 (244276)', '(244276) 2002 EH16' or '2015 DM182'."""
    num = re.search(r"\((\d+)\)", name or "")
    if num:
        rest = (name[:num.start()] + name[num.end():]).strip()
    else:
        m = re.fullmatch(r"\s*(\d+)\s+(.*)", name or "")
        num, rest = (m, m.group(2)) if m and not _PROV.search(name) else (None, (name or "").strip())
    return (num.group(1) if num else None), (rest or None)


def full_name(info) -> str:
    return (info or {}).get("fullname") or ""


def designation_date(desig: str):
    """The date a provisional designation encodes (the middle of its half
    month), or a survey designation's observing date, as a datetime; None
    for anything else."""
    m = _SURVEY.search(desig or "")
    if m:
        return datetime.fromisoformat(SURVEYS[m.group(1)])
    m = _PROV.search(desig or "")
    if not m:
        return None
    k = HALF_MONTHS.index(m.group(2))
    return datetime(int(m.group(1)), k // 2 + 1, 8 if k % 2 == 0 else 23)


def _parse_date(s):
    for f in ("%Y-%b-%d", "%Y-%m-%d", "%Y-%b", "%Y"):
        try:
            return datetime.strptime(str(s).strip(), f)
        except (TypeError, ValueError):
            pass
    return None


def _confirmed_known(db, include_likely: bool) -> list:
    """The catalogued objects confirmed (or likely), each with the earliest
    image (t_ref) it was found on."""
    verdicts = ("confirmed", "likely") if include_likely else ("confirmed",)
    rows = db.execute(f"SELECT id, known_name, t_ref, verdict, mag, score FROM candidates WHERE status='known' "
                      f"AND known_name IS NOT NULL AND t_ref IS NOT NULL AND verdict IN ({','.join('?' * len(verdicts))}) "
                      f"ORDER BY t_ref", verdicts).fetchall()
    first = {}
    for r in rows:
        first.setdefault(r["known_name"], r)
    return list(first.values())


def class_label(c) -> str:
    """SkyBoT's class in words: "MB>Inner" -> "main belt, inner",
    "NEA>Apollo" -> "NEA, Apollo", "KBO>Resonant>3:2" -> "KBO, resonant 3:2"."""
    if not c:
        return ""
    head, *rest = str(c).split(">")
    head = {"MB": "main belt", "Mars-Crosser": "Mars-crosser"}.get(head, head)
    rest = [r if r[:1].isdigit() or r in ("Apollo", "Aten", "Amor", "Atira", "Cybele", "Hilda", "SDO") else r.lower()
            for r in rest]
    return ", ".join([head] + ([" ".join(rest)] if rest else []))


def skybot_classes(db, ids) -> dict:
    """candidate id -> SkyBoT's class of the catalogued asteroid it is, or
    for one not catalogued, of the catalogued asteroid near it (from its
    visit's detections.json and known.json); "" if neither."""
    ids = list(ids)
    if not ids:
        return {}
    rows = db.execute(f"SELECT id, visit_dir, det FROM candidates WHERE id IN ({','.join('?' * len(ids))}) "
                      f"ORDER BY visit_dir", ids).fetchall()
    out, visit, dets, known = {}, None, [], {}
    for r in rows:
        if r["visit_dir"] != visit:
            visit, dets, known = r["visit_dir"], [], {}
            try:
                dets = json.loads((Path(visit) / "detections.json").read_text())["detections"]
                known = {str(a["id"]): a.get("class") for a in
                         json.loads((Path(visit) / "known.json").read_text()).get("asteroids", [])}
            except (OSError, ValueError, KeyError):
                pass
        det = dets[r["det"]] if r["det"] < len(dets) else {}
        kid = det.get("known_id") or (det.get("near_known") or {}).get("id")
        out[r["id"]] = known.get(str(kid), "") or ""
    return out


def discovery_cache(out) -> dict:
    """number -> SBDB discovery info, from ``OUT/discovery_cache.json`` and
    the review sheets' caches (``review/discovery.json`` in each visit)."""
    out = Path(out)
    try:
        cache = json.loads((out / DISCOVERY_CACHE).read_text())
    except (OSError, ValueError):
        cache = {}
    for f in (out / "data").glob("*/*/review/discovery.json"):
        try:
            for k, v in json.loads(f.read_text()).items():
                cache.setdefault(k, v)
        except (OSError, ValueError):
            pass
    return cache


def fetch_discoveries(out, db=None, include_likely: bool = True, pause: float = 0.5, log=None, stop=None) -> int:
    """Look up in JPL's Small-Body Database the discovery of every numbered
    confirmed (or likely) catalogued object not in the cache yet (only its
    number is sent), one at a time (JPL refuses parallel queries for a
    while), saving every 50; returns how many were added. Stops early after
    5 failures in a row (no network, or JPL refusing) or when ``stop`` is set."""
    from neodetect.review import discovery
    db = db or C.connect(out)
    cache = discovery_cache(out)
    todo = sorted({n for n, _ in (_known(r["known_name"]) for r in _confirmed_known(db, include_likely))
                   if n and n not in cache}, key=int)
    if not todo:
        return 0
    if log:
        log(f"looking up {len(todo)} discovery dates in JPL's Small-Body Database")
    found, fails = {}, 0

    def save():
        cache = {**discovery_cache(out), **found}
        tmp = Path(out) / (DISCOVERY_CACHE + ".part")
        tmp.write_text(json.dumps(cache))
        tmp.replace(Path(out) / DISCOVERY_CACHE)

    for n in todo:
        if stop is not None and stop.is_set():
            break
        info = discovery(n)
        if info is None:  # no network or refused: asked again next time
            fails += 1
            if fails >= 5:
                if log:
                    log("JPL's Small-Body Database isn't answering; the rest are looked up next time")
                break
        else:
            found[n], fails = info, 0
            if len(found) % 50 == 0:
                save()
        time.sleep(pause)
    if found:
        save()
    return len(found)


def discovery_lag(db, cache: dict, include_likely: bool = True) -> dict:
    """For each catalogued object confirmed (or likely): its discovery date
    minus the date of the earliest WISE image it was found on, in years
    (positive: WISE saw it before it was discovered). Numbered objects use
    their credited discovery (SBDB); those without one (the unnumbered, and
    a few numbered ones SBDB has no date for) the date their provisional
    designation encodes (to within a half month)."""
    objs, pending, undated = [], 0, 0
    for r in _confirmed_known(db, include_likely):
        num, desig = _known(r["known_name"])
        info = cache.get(num) if num else None
        if num and num not in cache:
            pending += 1
            continue
        when, source = _parse_date((info or {}).get("date")), "discovery"
        if when is None:
            full = full_name(info)
            when, source = designation_date(desig) or designation_date(full), "designation"
        if when is None:
            undated += 1
            continue
        image = datetime(1858, 11, 17) + timedelta(days=float(r["t_ref"]))
        lag = (when - image).total_seconds() / 86400 / 365.25
        # a designation's date is only good to its half month (+-8 days)
        slack = 8 / 365.25 if source == "designation" and _PROV.search(desig or full_name(info)) else 0.0
        objs.append({"id": r["id"], "known_name": r["known_name"], "verdict": r["verdict"], "source": source,
                     "image": image.strftime("%Y-%m-%d"), "discovered": when.strftime("%Y-%m-%d"),
                     "who": (info or {}).get("who") if source == "discovery" else None,
                     "site": (info or {}).get("location") if source == "discovery" else None,
                     "lag": round(lag, 3), "unsure": bool(abs(lag) <= slack)})
    lags = [o["lag"] for o in objs]
    hi = max(1, math.ceil(max(lags + [0]) + 1e-9))
    edges = list(range(LAG_MIN, hi + 1))
    counts = {s: np.histogram(np.clip([o["lag"] for o in objs if o["source"] == s], LAG_MIN, hi - 1e-6), edges)[0]
              for s in ("discovery", "designation")}
    before = [o for o in objs if o["lag"] > 0 and not o["unsure"]]
    unsure = [o for o in objs if o["unsure"]]
    wise = [o for o in objs if "WISE" in (o["who"] or "").upper()]
    return {"edges": edges, "discovery": [int(k) for k in counts["discovery"]],
            "designation": [int(k) for k in counts["designation"]],
            "n": len(objs), "before": len(before), "before_designation": sum(o["source"] == "designation" for o in before),
            "by_wise": len(wise), "unsure": len(unsure), "pending": pending, "undated": undated, "lag_min": LAG_MIN,
            "earliest": min(lags) if lags else None, "median": float(np.median(lags)) if lags else None,
            "objects": sorted(before, key=lambda o: -o["lag"])}


def data(out, db=None, include_likely: bool = True, cache_path=None) -> dict:
    db = db or C.connect(out)
    visits = load_visits(out, db, cache_path)
    mot = motion(db, visits, include_likely)
    rows = _visit_rows(db)
    return {"visits": len(visits), "completeness": completeness(visits), "motion": mot,
            "gallery": gallery(mot), "astrometry": astrometry(db, visits, include_likely),
            "scores": scores(db), "brightness": brightness(db, include_likely), "galactic": galactic(rows),
            "ecliptic": ecliptic(rows, visits), "solar_system": solar_system(db, visits, include_likely),
            "progress": progress(out), "discovery": discovery_lag(db, discovery_cache(out), include_likely)}


# ---- PNGs

INK, MUTE, GRID, CAT_LIGHT = "#0b0b0b", "#52514e", "#e1e0d9", "#9fc0ea"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_completeness(c: dict, path) -> Path:
    plt = _plt()
    from matplotlib.ticker import FuncFormatter, NullFormatter
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    color = COLORS["catalogued"]
    for ax, key, xlabel in ((axs[0], "by_snr", "signal at the predicted track (detector score)"),
                            (axs[1], "by_rate", "rate (deg/day)")):
        b = c[key]
        x = [r["x"] for r in b]
        y = [100 * r["frac"] for r in b]
        err = [[100 * (r["frac"] - r["lo68"]) for r in b], [100 * (r["hi68"] - r["frac"]) for r in b]]
        ax.errorbar(x, y, yerr=err, color=color, lw=2, marker="o", ms=5, capsize=0, elinewidth=1)
        if key == "by_snr":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(0, None)
        ax.set_ylim(-3, 103)
        ax.set_xlabel(xlabel, fontsize=9)
        _style(ax)
        for r, xi, yi in zip(b, x, y):
            ax.annotate(f"{r['n']}", (xi, yi), textcoords="offset points", xytext=(0, -13 if yi > 50 else 7),
                        ha="center", fontsize=6.5, color=MUTE)
    axs[0].set_ylabel("found (%)", fontsize=9)
    axs[0].set_title(f"All {c['n']} catalogued asteroids on ≥{c['min_frames']} frames: {c['found']} found",
                     fontsize=10, color=INK, loc="left")
    axs[1].set_title(f"Signal ≥ {c['snr_floor']:g} ({c['n_bright']}): by rate", fontsize=10, color=INK, loc="left")
    ax = axs[2]
    b = c["by_class"]
    ys = np.arange(len(b))[::-1]
    ax.errorbar([100 * r["frac"] for r in b], ys,
                xerr=[[100 * (r["frac"] - r["lo68"]) for r in b], [100 * (r["hi68"] - r["frac"]) for r in b]],
                fmt="o", color=color, ms=5, elinewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['class']} ({r['n']})" for r in b], fontsize=8)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("found (%)", fontsize=9)
    ax.set_title(f"Signal ≥ {c['snr_floor']:g}: by orbit class", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.text(0.01, 0.01, "Numbers: asteroids in the bin. Bars: 68% interval.", fontsize=7.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def _motion_ax(ax, m, zoom):
    objs = m["objects"]
    for cat in ("catalogued", "not catalogued"):
        for verdict, filled in (("confirmed", True), ("likely", False)):
            sel = [o for o in objs if o["catalogue"] == cat and o["verdict"] == verdict]
            if not sel:
                continue
            new = cat == "not catalogued"
            ax.scatter([o["vl"] for o in sel], [o["vb"] for o in sel], s=(34 if new else 12) * (1.2 if not filled else 1),
                       facecolors=COLORS[cat] if filled else "none", edgecolors="white" if filled else COLORS[cat],
                       linewidths=0.6 if filled else 1.3, alpha=1 if new else 0.75, zorder=4 if new else 3,
                       label=f"{verdict}, {cat} ({len(sel)})")
    ref = m["main_belt"]
    if ref:
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(ref["vl"] + 5 * ref["sl"] * np.cos(t), ref["vb"] + 5 * ref["sb"] * np.sin(t), "--", color=MUTE,
                lw=0.9, zorder=2, label="5σ of the main belt")
    for o in objs:
        if o["catalogue"] == "not catalogued" and (o.get("sigma") or 0) > (1.4 if zoom else 5):
            ax.annotate(f"#{o['id']}", (o["vl"], o["vb"]), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=INK)
    ax.axhline(0, color=MUTE, lw=0.6, zorder=1)
    ax.axvline(0, color=MUTE, lw=0.6, zorder=1)
    if zoom and ref:
        ax.set_xlim(ref["vl"] - 8 * ref["sl"], ref["vl"] + 8 * ref["sl"])
        ax.set_ylim(ref["vb"] - 8 * ref["sb"], ref["vb"] + 8 * ref["sb"])
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("motion in ecliptic longitude, dλ cos β / dt (deg/day)", fontsize=9)
    ax.set_ylabel("motion in ecliptic latitude, dβ / dt (deg/day)", fontsize=9)
    _style(ax)


def plot_motion(m: dict, path) -> Path:
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150)
    _motion_ax(axs[0], m, zoom=False)
    _motion_ax(axs[1], m, zoom=True)
    axs[0].set_title("Motion of the confirmed and likely detections", fontsize=10, color=INK, loc="left")
    axs[1].set_title("Close up of the main belt", fontsize=10, color=INK, loc="left")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_astrometry(a: dict, path) -> Path:
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    ax = axs[0]
    f = np.array(a["frames"]) if a["frames"] else np.zeros((0, 2))
    ax.scatter(f[:, 0], f[:, 1], s=4, color=COLORS["catalogued"], alpha=0.35, lw=0)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(a["pixel"] * np.cos(t), a["pixel"] * np.sin(t), "--", color=MUTE, lw=0.9, label=f"1 pixel ({a['pixel']}\")")
    s = a["stats"]
    if s:
        ax.plot([s["median_dra"]], [s["median_ddec"]], "+", color=INK, ms=12, mew=1.5, label="median")
    lim = 2.2 * a["pixel"]
    ax.set_xlim(lim, -lim)  # east to the left, as on the sky
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("ΔRA cos Dec (arcsec, measured − catalogue)", fontsize=9)
    ax.set_ylabel("ΔDec (arcsec)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    out_of = int(np.sum((np.abs(f[:, 0]) > lim) | (np.abs(f[:, 1]) > lim))) if len(f) else 0
    ax.set_title(f"Every observation ({len(f)}" + (f", {out_of} off the plot" if out_of else "") + ")",
                 fontsize=10, color=INK, loc="left")
    _style(ax)
    ax = axs[1]
    o = a["objects"]
    ax.scatter([x["score"] for x in o], [x["sep"] for x in o], s=12, color=COLORS["catalogued"], alpha=0.7, lw=0)
    ax.set_xscale("log")
    ax.axhline(a["pixel"], ls="--", color=MUTE, lw=0.9)
    ax.set_xlabel("track score", fontsize=9)
    ax.set_ylabel("offset of the detection's median position (arcsec)", fontsize=9)
    ax.set_ylim(0, max([x["sep"] for x in o] + [a["pixel"] * 1.3]))
    ax.set_title(f"Each detection ({len(o)})", fontsize=10, color=INK, loc="left")
    _style(ax)
    if s:
        fig.text(0.01, 0.01, f"Median offset ΔRA {s['median_dra']:+.2f}\", ΔDec {s['median_ddec']:+.2f}\"; "
                             f"RMS {s['rms_dra']:.2f}\" and {s['rms_ddec']:.2f}\"; 68% within {s['p68']:.2f}\", "
                             f"95% within {s['p95']:.2f}\". Catalogue: SkyBoT, shifted to the spacecraft; "
                             f"{a['loose_ephemeris']} detections left out (ephemeris worse than {a['max_eph_err']:g}\").",
                 fontsize=7.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def _caption(o) -> str:
    when = (np.datetime64("1858-11-17") + np.timedelta64(int(round(o["t_ref"] * 86400)), "s")).astype(str)[:16]
    s = (f"#{o['id']} {o['verdict']}   RA {o['ra']:.4f}  Dec {o['dec']:+.4f}   {when.replace('T', ' ')} UTC   "
         f"{o['rate']:.3f} deg/day at PA {o['pa']:.0f}   score {o['score']:.1f}")
    if o.get("sigma") is not None:
        s += f"   motion {o['sigma']:.0f}σ from the main belt"
    if o.get("near"):
        s += f"\nnear {o['near']}"
    return s


def plot_gallery(g: list, sheets: dict, path) -> Path:
    """``sheets``: {id: PNG path}, the blind cutout sheets."""
    plt = _plt()
    import matplotlib.image as mpimg
    imgs = []
    for o in g:
        p = sheets.get(o["id"])
        try:
            imgs.append(mpimg.imread(p) if p else None)
        except (OSError, ValueError, SyntaxError):
            imgs.append(None)
    width = max([i.shape[1] for i in imgs if i is not None] + [900])
    heights = [(i.shape[0] if i is not None else 60) + 40 + 14 * bool(o.get("near")) for o, i in zip(g, imgs)] or [60]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, sum(heights) / dpi + 0.5), dpi=dpi)
    fig.text(0.01, 1 - 0.25 / (sum(heights) / dpi + 0.5), f"Detections not catalogued ({len(g)}), "
             "most unusual motion first", fontsize=12, color=INK, va="center")
    top = sum(heights) + 50
    y = top - 50
    for o, im, h in zip(g, imgs, heights):
        cap = 40 + 14 * bool(o.get("near"))
        ax = fig.add_axes((0, (y - h) / top, (im.shape[1] if im is not None else width) / width,
                           (h - cap) / top))
        ax.axis("off")
        if im is not None:
            ax.imshow(im, interpolation="nearest", aspect="auto")
        fig.text(0.005, (y - cap / 2) / top, _caption(o), fontsize=9, color=INK, va="center", linespacing=1.3)
        y -= h
    if not g:
        fig.text(0.5, 0.5, "None yet", ha="center", color=MUTE)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return Path(path)


def sheets(db, ids) -> dict:
    ids = list(ids)
    if not ids:
        return {}
    return {r[0]: r[1] for r in db.execute(f"SELECT id, sheet FROM candidates WHERE id IN ({','.join('?' * len(ids))})",
                                           ids)}


AQUA = "#1baf7a"  # categorical slot 3
IDLE_SHADE = {"outside the hours": "#efeee9", "detection run": "#e1e0d9", "paused": "#e1e0d9",
              "waiting": "#efeee9", "not running": "#f4f3ef"}


def _bin_label(b, unit=""):
    if b.get("lo") is None:
        return f"<{b['hi']:g}{unit}"
    if b.get("hi") is None:
        return f"≥{b['lo']:g}{unit}"
    return f"{b['lo']:g}–{b['hi']:g}{unit}"


def plot_scores(sc: dict, path) -> Path:
    plt = _plt()
    from matplotlib.ticker import FuncFormatter, NullFormatter
    rows = sc["rows"]
    fig = plt.figure(figsize=(14, 6.4), dpi=150)
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.18)
    bins = sc["bins"]
    labels = [_bin_label(b) for b in bins]
    xs = np.arange(len(bins))
    for k, r in enumerate(rows):
        top = max(b[r] for b in bins) or 1
        ax = fig.add_subplot(gs[k, 0])
        ax.bar(xs, [b[r] for b in bins], width=0.8, color=COLORS["catalogued"] if r != "not reviewed" else "#9a9890")
        for x, b in zip(xs, bins):
            if b[r]:
                ax.annotate(str(b[r]), (x, b[r]), textcoords="offset points", xytext=(0, 2), ha="center",
                            fontsize=6.5, color=MUTE)
        ax.set_ylim(0, top * 1.25)
        ax.set_ylabel(f"{r}\n({sum(b[r] for b in bins)})", fontsize=8, rotation=0, ha="right", va="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels if k == len(rows) - 1 else [], fontsize=7.5)
        _style(ax)
        if k == 0:
            ax.set_title("Detections by track score and verdict", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("track score", fontsize=9)
    ax = fig.add_subplot(gs[:, 1])
    j = [b for b in bins if b["frac"] is not None and b["reviewed"] >= MIN_REVIEWED]
    ax.errorbar([b["x"] for b in j], [100 * b["frac"] for b in j],
                yerr=[[100 * (b["frac"] - b["lo68"]) for b in j], [100 * (b["hi68"] - b["frac"]) for b in j]],
                color=COLORS["catalogued"], lw=2, marker="o", ms=5, elinewidth=1)
    for b in j:
        ax.annotate(str(b["reviewed"]), (b["x"], 100 * b["frac"]), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=6.5, color=MUTE)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(min([60] + [100 * b["lo68"] - 5 for b in j]), 103)
    ax.set_xlabel("track score", fontsize=9)
    ax.set_ylabel("judged real, of those reviewed (%)", fontsize=9)
    ax.set_title(f"Expected real among the {sc['not_reviewed']} not reviewed: about {sc['expected_real']}",
                 fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.text(0.01, 0.01, "Real: confirmed or likely. Numbers on the right: detections reviewed in the bin "
             f"(bins with fewer than {MIN_REVIEWED} not drawn); bars: 68% interval. "
             "The review order is random, so the reviewed are a fair sample.",
             fontsize=7.5, color=MUTE)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_brightness(br: dict, path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    e = np.array(br["edges"])
    n = np.array(br["catalogued"])
    ax.bar(e[:-1], n, width=np.diff(e) * 0.9, align="edge", color=COLORS["catalogued"],
           label=f"catalogued ({int(n.sum())})")
    objs = br["objects"]
    if objs:
        y0 = n.max() * 1.08 if n.size and n.max() else 1
        ax.scatter([o["mag"] for o in objs], [y0] * len(objs), s=40, color=COLORS["not catalogued"],
                   edgecolors="white", lw=0.8, zorder=3, label=f"not catalogued ({len(objs)})")
        for k, o in enumerate(sorted(objs, key=lambda o: o["mag"])):
            ax.annotate(f"#{o['id']}", (o["mag"], y0), textcoords="offset points",
                        xytext=(0, 7 + 9 * (k % 3)), ha="center", fontsize=6.5, color=INK)
        ax.set_ylim(0, y0 * 1.18)
    for m, c, lab in ((br["median_catalogued"], COLORS["catalogued"], "median, catalogued"),
                      (br["median_new"], COLORS["not catalogued"], "median, not catalogued")):
        if m is not None:
            ax.axvline(m, color=c, ls="--", lw=1, label=f"{lab} {m:.2f}")
    ax.set_xlabel("W3 magnitude (fainter →)", fontsize=9)
    ax.set_ylabel("detections", fontsize=9)
    ax.set_title("Brightness of the confirmed and likely detections", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_galactic(g: dict, path) -> Path:
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    b = g["bins"]
    x = [f"{x['lo']}–{x['hi']}°\n{x['visits']} visits" for x in b]
    for ax, key, color, title in ((axs[0], "cat", COLORS["catalogued"], "Catalogued asteroids found per visit"),
                                  (axs[1], "new", COLORS["not catalogued"],
                                   "Detections not catalogued per visit (any verdict)")):
        ax.bar(range(len(b)), [r[key] for r in b], color=color, width=0.75)
        for i, r in enumerate(b):
            ax.annotate(f"{r[key]:.2f}", (i, r[key]), textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=7, color=MUTE)
        ax.set_xticks(range(len(b)))
        ax.set_xticklabels(x, fontsize=7.5)
        ax.set_xlabel("|galactic latitude| of the field", fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_ecliptic(e: dict, path) -> Path:
    plt = _plt()
    fig, axs = plt.subplots(3, 1, figsize=(13, 7.5), dpi=150, sharex=True)
    v = e["visits"]
    lam = [r["elon"] for r in v]
    for ax, key, color, label in ((axs[0], "crossing", "#9a9890", "catalogued asteroids on the field (≥3 frames)"),
                                  (axs[1], "cat", COLORS["catalogued"], "catalogued asteroids found"),
                                  (axs[2], "new", COLORS["not catalogued"], "detections not catalogued")):
        for lo, hi in e["galactic_plane"]:
            ax.axvspan(lo, hi, color="#efeee9", lw=0, zorder=0)
        ax.scatter(lam, [r[key] for r in v], s=10, color=color, lw=0, zorder=3)
        ax.set_ylabel("per visit", fontsize=8)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        _style(ax)
    axs[0].annotate("galactic plane (|b| < 10°)", (e["galactic_plane"][0][0], 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -10), textcoords="offset points", fontsize=7, color=MUTE) if e["galactic_plane"] else None
    axs[-1].set_xlim(0, 360)
    axs[-1].set_xticks(range(0, 361, 30))
    axs[-1].set_xlabel("ecliptic longitude of the field (deg)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_solar_system(ss: dict, path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=150)
    t = np.linspace(0, 2 * np.pi, 360)
    for name, r in ss["planets"].items():
        ax.plot(r * np.cos(t), r * np.sin(t), color=GRID if name != "Earth" else MUTE, lw=0.8, zorder=1)
        ax.annotate(name, (r * np.cos(np.radians(100)), r * np.sin(np.radians(100))), fontsize=7, color=MUTE)
    c = np.array(ss["context"]) if ss["context"] else np.zeros((0, 2))
    ax.scatter(c[:, 0], c[:, 1], s=2, color="#c3c2b7", lw=0, zorder=2,
               label=f"catalogued asteroids on searched fields ({len(c)})")
    if ss["earth"]:
        e = np.array(ss["earth"])
        ax.plot(e[:, 0], e[:, 1], color=INK, lw=2.2, zorder=4, label="the Earth (WISE) during the visits")
    colors = dict(zip(ss["groups"], (COLORS["catalogued"], AQUA, COLORS["not catalogued"], MUTE)))
    for g in ss["groups"]:
        sel = [o for o in ss["objects"] if o["group"] == g]
        if sel:
            big = g != "main belt"
            ax.scatter([o["x"] for o in sel], [o["y"] for o in sel], s=30 if big else 9, color=colors[g],
                       edgecolors="white", lw=0.6 if big else 0.3, zorder=5 if big else 3,
                       label=f"found: {g} ({len(sel)})")
    for o in ss["objects"]:
        if o["group"] == "NEA, Mars-crosser":
            ax.annotate(o["known_name"] or "", (o["x"], o["y"]), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=INK)
    ax.plot([0], [0], "o", color="#eda100", ms=9, zorder=6)
    lim = 5.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU, toward the vernal equinox)", fontsize=9)
    ax.set_ylabel("y (AU)", fontsize=9)
    ax.set_title("Where the asteroids were, seen from ecliptic north", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_progress(pr: dict, path) -> Path:
    plt = _plt()
    import matplotlib.dates as mdates

    def local(t):
        return datetime.fromisoformat(t).astimezone().replace(tzinfo=None)

    fig, axs = plt.subplots(3, 1, figsize=(13, 7.5), dpi=150, sharex=True)
    v = pr["visits"]
    if not v:
        fig.text(0.5, 0.5, "No visits searched yet", ha="center", color=MUTE)
    else:
        t = [local(r[0]) for r in v]
        det = np.cumsum([r[1] for r in v])
        known = np.cumsum([r[2] for r in v])
        series = ((np.arange(1, len(v) + 1), "#52514e", "visits searched"),
                  (known, COLORS["catalogued"], "catalogued asteroids found"),
                  (det - known, COLORS["not catalogued"], "detections not catalogued"))
        end = max(t[-1], datetime.now())
        for ax, (y, color, label) in zip(axs, series):
            seen = set()
            for a, b, kind in pr["idle"]:
                ax.axvspan(local(a), local(b) if b else end,
                           color=IDLE_SHADE.get(kind, "#efeee9"), lw=0, zorder=0,
                           label=None if kind in seen or ax is not axs[0] else kind)
                seen.add(kind)
            ax.step(t, y, where="post", color=color, lw=2, zorder=3)
            ax.set_title(f"{label}: {int(y[-1])}", fontsize=9, color=INK, loc="left")
            _style(ax)
        if pr["idle"]:
            axs[0].legend(fontsize=7.5, frameon=False, loc="upper left", title="not searching:", title_fontsize=7.5)
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axs[-1].set_xlim(t[0], end)
    axs[-1].set_xlabel(f"date ({time.strftime('%Z')})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_discovery(dl: dict, path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    e = np.array(dl["edges"], float)
    a, b = np.array(dl["discovery"]), np.array(dl["designation"])
    top = max(1, int((a + b).max()) if a.size else 1) * 1.25
    ax.axvspan(0, e[-1], color=AQUA, alpha=0.07, lw=0, zorder=0)
    ax.bar(e[:-1], a, width=0.86, align="edge", color=COLORS["catalogued"],
           label=f"credited discovery date ({int(a.sum())})")
    ax.bar(e[:-1], b, width=0.86, align="edge", bottom=a, color=AQUA, edgecolor="white", lw=0.8,
           label=f"no credited discovery (unnumbered): provisional designation date ({int(b.sum())})")
    ax.axvline(0, color=INK, lw=1)
    n, k = dl["n"], dl["before"]
    if n:
        ax.text(0.4, top * 0.93, f"seen by WISE before discovery: {k} of {n} ({100 * k / n:.0f}%) →"
                + (f"\n({dl['unsure']} more too close to tell)" if dl["unsure"] else ""),
                fontsize=9, color=INK, va="top")
        ax.text(-0.4, top * 0.93, "← discovered before the WISE image", fontsize=9, color=MUTE, va="top", ha="right")
    ax.set_ylim(0, top)
    ax.set_xlim(e[0] - 0.3, e[-1] + 0.3)
    ticks = [t for t in range(int(e[0]), int(e[-1]) + 1) if t % 5 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"≤{t}" if t == dl["lag_min"] else (f"+{t}" if t > 0 else str(t)) for t in ticks])
    ax.set_xlabel("discovery date − date of the WISE image it was found on (years)", fontsize=9)
    ax.set_ylabel("catalogued asteroids", fontsize=9)
    ax.set_title("When the catalogued asteroids found were discovered, relative to the WISE images",
                 fontsize=10, color=INK, loc="left")
    notes = []
    if dl["pending"]:
        notes.append(f"{dl['pending']} discovery dates not looked up yet")
    if dl["earliest"] is not None and dl["earliest"] < dl["lag_min"]:
        notes.append(f"leftmost bar: discovered {-dl['lag_min']} or more years before (earliest {-dl['earliest']:.0f})")
    if notes:
        ax.set_title("\n".join(notes), fontsize=7.5, color=MUTE, loc="right")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


PLOTS = ("completeness", "motion", "gallery", "astrometry", "discovery",
         "scores", "brightness", "galactic", "ecliptic", "solar_system", "progress")


def render(name: str, d: dict, db, path) -> Path:
    if name == "completeness":
        return plot_completeness(d["completeness"], path)
    if name == "motion":
        return plot_motion(d["motion"], path)
    if name == "astrometry":
        return plot_astrometry(d["astrometry"], path)
    if name == "gallery":
        return plot_gallery(d["gallery"], sheets(db, [o["id"] for o in d["gallery"]]), path)
    fn = {"scores": plot_scores, "brightness": plot_brightness, "galactic": plot_galactic,
          "ecliptic": plot_ecliptic, "solar_system": plot_solar_system, "progress": plot_progress,
          "discovery": plot_discovery}.get(name)
    if fn is None:
        raise ValueError(name)
    return fn(d[name], path)


def write_plots(out, directory=None, include_likely: bool = True, names=PLOTS) -> list:
    """The plots as PNGs in ``directory`` (default ``OUT/plots``)."""
    db = C.connect(out)
    if "discovery" in names:
        fetch_discoveries(out, db, include_likely, log=print)
    d = data(out, db, include_likely)
    directory = Path(directory) if directory else Path(out) / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return [render(n, d, db, directory / f"{n}.png") for n in names]
