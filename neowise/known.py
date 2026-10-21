"""Known solar-system objects in a visit, from IMCCE SkyBoT.

SkyBoT is queried once per frame epoch (cone search around the visit
center). It returns geocentric positions -- the WISE observatory code C51 is
not applied -- but WISE orbits ~6900 km from the geocenter, a parallax of
~3.5" (1.3 px) for a main-belt asteroid and far more for NEOs. Each frame
header carries the Earth-to-spacecraft vector (``ERTH2SC[XYZ]``, AU, J2000
equatorial), so positions are shifted to the spacecraft using SkyBoT's
geocentric distance.

The result is written as ``known.json`` in the simulator's truth format
(``asteroids`` with per-frame RA/Dec, pixel position, ``in_frame``,
magnitude, flux), so ``neodetect.evaluate`` matches detections to it
unchanged. Magnitudes are *predicted W1/W2* from SkyBoT's V with a fixed
reflected-light colour (``V_MINUS_W``); real colours vary by a magnitude or
more (taxonomy; thermal emission for NEOs near the Sun), so treat them as
rough.
"""

import json
import math
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from neodetect.io import world2pix

from .query import http_get

SKYBOT = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
V_MINUS_W = {1: 1.8, 2: 1.9}  # sun V-W1 ~1.6, asteroids a little redder
# W3/W4 (12/22 um) see an asteroid's own thermal emission, which reflected
# light (V) doesn't predict: their magnitudes are left NaN (positions only).


def skybot_cone(mjd_utc: float, ra: float, dec: float, radius_deg: float,
                loc: str = "500") -> List[dict]:
    """Objects within ``radius_deg`` at the epoch, seen from the MPC
    observatory ``loc`` (default 500, the geocentre)."""
    def query(obj_filter):
        q = urllib.parse.urlencode({"-ep": f"{mjd_utc + 2400000.5:.7f}", "-ra": f"{ra:.6f}",
                                    "-dec": f"{dec:.6f}", "-rd": f"{radius_deg:.4f}",
                                    "-mime": "text", "-output": "all", "-loc": loc,
                                    "-objFilter": obj_filter})
        return parse_skybot(http_get(f"{SKYBOT}?{q}", timeout=180.0).decode("utf-8", "replace"))
    try:
        return query("111")  # asteroids, planets, comets
    except RuntimeError as e:
        # SkyBoT's comet ephemerides don't cover every epoch (early 2010:
        # "comet conesearch -> Bad request: Epoch ... not available"), and
        # then the whole query fails: retry without comets.
        if "comet" not in str(e).lower():
            raise
        return query("110")


def parse_skybot(text: str) -> List[dict]:
    rows = []
    header = None
    for line in text.splitlines():
        if line.startswith("# Flag:") and line.split(":")[1].strip() == "-1":
            raise RuntimeError(f"SkyBoT error: {text[:300]}")
        if line.startswith("# Num"):
            header = [h.strip() for h in line[1:].split("|")]
            continue
        if line.startswith("#") or not line.strip() or header is None:
            continue
        v = dict(zip(header, [c.strip() for c in line.split("|")]))
        if not v.get("Mv"):  # no predicted magnitude (some comets): can't be scored
            continue
        h, m, s = (float(x) for x in v["RA(h)"].split())
        sign = -1.0 if v["DE(deg)"].lstrip().startswith("-") else 1.0
        d, dm, ds = (abs(float(x)) for x in v["DE(deg)"].split())
        rows.append({
            "num": v["Num"], "name": v["Name"], "class": v["Class"],
            "ra_deg": 15.0 * (h + m / 60.0 + s / 3600.0),
            "dec_deg": sign * (d + dm / 60.0 + ds / 3600.0),
            "v_mag": float(v["Mv"]), "err_arcsec": _num(v.get("Err(arcsec)")),
            "dra_arcsec_h": float(v["dRA(arcsec/h)"]), "ddec_arcsec_h": float(v["dDEC(arcsec/h)"]),
            "dist_au": float(v["Dg(ua)"]),
        })
    return rows


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def topocentric(ra_deg: float, dec_deg: float, dist_au: float, earth_to_sc_au) -> tuple:
    """Shift a geocentric position at distance ``dist_au`` to an observer
    displaced by ``earth_to_sc_au`` (equatorial J2000, AU)."""
    r = dist_au * _unit(ra_deg, dec_deg) - np.asarray(earth_to_sc_au, float)
    ra = math.degrees(math.atan2(r[1], r[0])) % 360.0
    dec = math.degrees(math.asin(r[2] / np.linalg.norm(r)))
    return ra, dec


def _fit_motion(mjd, ra, dec):
    """Rate (deg/day) and position angle (east of north) by least squares."""
    t = np.asarray(mjd) - np.mean(mjd)
    dra = ((np.asarray(ra) - ra[0] + 180.0) % 360.0 - 180.0) * math.cos(math.radians(dec[0]))
    if len(t) < 2 or np.ptp(t) == 0:
        return 0.0, 0.0
    ve = np.polyfit(t, dra, 1)[0]
    vn = np.polyfit(t, np.asarray(dec), 1)[0]
    return float(math.hypot(ve, vn)), float(math.degrees(math.atan2(ve, vn)) % 360.0)


def known_objects(seq, center, radius_deg: float, band: int = 1, max_v: float = 99.0,
                  log=None, loc: str = "500", v_minus: Optional[float] = None,
                  source: str = "SkyBoT (IMCCE), shifted to the WISE spacecraft") -> dict:
    """Known objects on the frames of ``seq`` (a ``neodetect.io.Sequence``),
    in truth.json format. Positions are shifted by each frame's
    ``ERTH2SC[XYZ]`` if present (WISE); a ground station passes its MPC
    code as ``loc`` and its band's V-minus colour as ``v_minus``."""
    log = log or (lambda *a: None)
    vw = V_MINUS_W.get(band, math.nan) if v_minus is None else v_minus
    per_obj: Dict[str, dict] = {}
    with ThreadPoolExecutor(4) as ex:  # SkyBoT takes seconds per query
        results = list(ex.map(lambda fr: skybot_cone(fr.mjd, center[0], center[1], radius_deg, loc),
                              seq.frames))
    for k, (fr, rows) in enumerate(zip(seq.frames, results)):
        h = fr.header
        e2s = [h.get("ERTH2SCX", 0.0), h.get("ERTH2SCY", 0.0), h.get("ERTH2SCZ", 0.0)]
        log(f"  frame {k}: {len(rows)} known objects")
        for r in rows:
            if r["class"].lower().startswith("comet") and r["v_mag"] == 0.0:
                r["v_mag"] = math.nan  # SkyBoT's placeholder for "unknown"
            if r["v_mag"] > max_v:
                continue
            key = r["num"] if r["num"] not in ("", "-") else r["name"]
            ra, dec = topocentric(r["ra_deg"], r["dec_deg"], r["dist_au"], e2s)
            x, y = world2pix(fr.wcs, ra, dec)
            ny, nx = fr.shape
            iy, ix = int(round(float(y))), int(round(float(x)))
            in_frame = bool(0 <= ix < nx and 0 <= iy < ny and not fr.bad[iy, ix])
            mag = r["v_mag"] - vw
            flux = 10.0 ** (-0.4 * (mag - fr.magzp)) if fr.magzp and math.isfinite(mag) else None
            o = per_obj.setdefault(key, {"id": key, "name": r["name"], "class": r["class"],
                                         "frames_by_k": {}, "v": [], "err": [], "dist": []})
            o["frames_by_k"][k] = {"frame": k, "mjd": fr.mjd, "ra_deg": ra, "dec_deg": dec,
                                   "ra_geo_deg": r["ra_deg"], "dec_geo_deg": r["dec_deg"],
                                   "x": float(x), "y": float(y), "in_frame": in_frame,
                                   "mag": mag, "flux_dn": flux}
            o["v"].append(r["v_mag"])
            o["err"].append(r["err_arcsec"])
            o["dist"].append(r["dist_au"])

    asteroids = []
    for key, o in per_obj.items():
        fr_by_k = o.pop("frames_by_k")
        if len(fr_by_k) < 2:
            continue
        ks = sorted(fr_by_k)
        rate, pa = _fit_motion([fr_by_k[k]["mjd"] for k in ks], [fr_by_k[k]["ra_deg"] for k in ks],
                               [fr_by_k[k]["dec_deg"] for k in ks])
        # Frames where SkyBoT did not return the object (outside the cone).
        frames = [fr_by_k.get(k, {"frame": k, "mjd": seq.frames[k].mjd, "ra_deg": math.nan,
                                  "dec_deg": math.nan, "in_frame": False, "mag": None,
                                  "flux_dn": 0.0}) for k in range(seq.n_frames)]
        v = float(np.median(o.pop("v")))  # nan if unknown
        asteroids.append({
            **o, "v_mag": v, "mag": v - vw, "err_arcsec": float(np.max(o.pop("err"))),
            "dist_au": float(np.median(o.pop("dist"))), "rate_deg_day": rate, "pa_deg": pa,
            "n_frames_in_frame": sum(p["in_frame"] for p in frames), "frames": frames,
        })
    asteroids.sort(key=lambda a: a["mag"] if math.isfinite(a["mag"]) else 99.0)
    if v_minus is None and not math.isfinite(vw):
        mag = f"W{band}: thermal, not predicted"
    else:
        mag = f"predicted W{band} = V - {vw}" if v_minus is None else f"predicted V - {vw}"
    return {"source": source, "magnitude": mag, "asteroids": asteroids}


def write_known(seq, out_dir, center, radius_deg: float, band: int = 1, max_v: float = 99.0,
                log=None) -> Path:
    doc = known_objects(seq, center, radius_deg, band, max_v, log)
    path = Path(out_dir) / "known.json"
    path.write_text(json.dumps(doc, indent=1))
    return path
