"""Suspected unknown asteroids: classification, cutout sheets, and the
user's confirmation.

Every detection that passes its threshold gets a ``status``:

- ``known``: matches a catalogued object (``known.json``, from SkyBoT) to
  within ``evaluate.MATCH_TOL_PX`` pixels.
- ``near-known``: no match, but a catalogued object lies within
  ``NEAR_KNOWN_ARCSEC`` (or 3x its ephemeris uncertainty, if larger) at a
  rate within ``NEAR_KNOWN_RATE_FRAC``: most likely that object with a
  poor orbit, or a fragment of its detection; worth a look.
- ``suspect``: nothing catalogued nearby, a suspected unknown asteroid.
- ``unchecked``: there's no catalogue to check against (``neodetect review
  --skybot`` fetches one).

``neodetect review`` writes a cutout sheet per suspect (each frame around
the track, and the same with the other frames' median subtracted, which
removes the stars and leaves the mover) and records the user's verdict
(confirmed or rejected, with a note) in ``review.json`` beside
``detections.json``. Verdicts are matched to detections by position, not
index, so they carry over when the visit is searched again. Confirmed
suspects can be written as ADES for the MPC, as tracklets without a
designation (trkSub only).
"""

import getpass
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .evaluate import MATCH_TOL_PX

SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api"
NEAR_KNOWN_ARCSEC = 60.0
NEAR_KNOWN_RATE_FRAC = 0.3
REVIEW_NAME = "review.json"
SHEET_DIR = "review"
REVIEWABLE = ("suspect", "near-known", "unchecked")


def _seps_arcsec(det: dict, asteroid: dict) -> List[float]:
    """Per-frame sky separation where the object is on a frame and the
    detection was measured."""
    seps = []
    for p, q in zip(asteroid["frames"], det["frames"]):
        if not (p.get("in_frame") and q["covered"]):
            continue
        dra = (q["ra_deg"] - p["ra_deg"] + 180.0) % 360.0 - 180.0
        dra *= math.cos(math.radians(p["dec_deg"]))
        seps.append(math.hypot(dra, q["dec_deg"] - p["dec_deg"]) * 3600.0)
    return seps


def classify(doc: dict, catalog: Optional[dict]) -> dict:
    """Set ``status`` (and ``known`` / ``near_known``) on every detected
    detection; returns the count per status."""
    scale = doc["stats"]["pixel_scale_arcsec"]
    counts = {}
    for d in doc["detections"]:
        for k in ("status", "near_known", "known", "known_id"):
            d.pop(k, None)
        if not d["detected"]:
            continue
        if catalog is None:
            d["status"] = "unchecked"
        else:
            best = None
            for a in catalog["asteroids"]:
                seps = _seps_arcsec(d, a)
                if len(seps) < 2:
                    continue
                sep = float(np.median(seps))
                if best is None or sep < best[0]:
                    best = (sep, a)
            if best and best[0] <= MATCH_TOL_PX * scale:
                d["status"] = "known"
                d.setdefault("known", best[1].get("name", best[1]["id"]))
                d.setdefault("known_id", best[1]["id"])
            elif best and _near(d, best[1], best[0]):
                d["status"] = "near-known"
                d["near_known"] = {"name": best[1].get("name", best[1]["id"]), "id": best[1]["id"],
                                   "sep_arcsec": round(best[0], 1),
                                   "err_arcsec": best[1].get("err_arcsec")}
            else:
                d["status"] = "suspect"
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return counts


def _near(d: dict, a: dict, sep: float) -> bool:
    err = a.get("err_arcsec")
    radius = max(NEAR_KNOWN_ARCSEC, 3.0 * err if err and math.isfinite(err) else 0.0)
    rate = a.get("rate_deg_day") or 0.0
    return sep <= radius and rate > 0 and abs(d["rate_deg_day"] / rate - 1.0) <= NEAR_KNOWN_RATE_FRAC


# ----------------------------------------------------------------- verdicts

def _positions(d: dict) -> list:
    return [[f["mjd"], f["ra_deg"], f["dec_deg"]] for f in d["frames"] if f["used"]]


def load_verdicts(visit_dir) -> List[dict]:
    p = Path(visit_dir) / REVIEW_NAME
    return json.loads(p.read_text())["verdicts"] if p.exists() else []


def save_verdicts(visit_dir, verdicts: List[dict]) -> Path:
    p = Path(visit_dir) / REVIEW_NAME
    p.write_text(json.dumps({"verdicts": verdicts}, indent=1))
    return p


def _same_object(v: dict, d: dict, tol_arcsec: float) -> bool:
    by_t = {round(f["mjd"], 5): f for f in d["frames"] if f["covered"]}
    seps = []
    for t, ra, dec in v["positions"]:
        f = by_t.get(round(t, 5))
        if f is None:
            continue
        dra = ((f["ra_deg"] - ra + 180.0) % 360.0 - 180.0) * math.cos(math.radians(dec))
        seps.append(math.hypot(dra, f["dec_deg"] - dec) * 3600.0)
    return len(seps) >= 2 and float(np.median(seps)) <= tol_arcsec


def attach_verdicts(doc: dict, verdicts: List[dict]) -> None:
    """Copy each verdict onto the detection it was made for (``review``)."""
    tol = MATCH_TOL_PX * doc["stats"]["pixel_scale_arcsec"]
    for d in doc["detections"]:
        d.pop("review", None)
        for v in verdicts:
            if _same_object(v, d, tol):
                d["review"] = {k: v[k] for k in ("verdict", "note", "reviewer", "time")}
                break


def set_verdict(verdicts: List[dict], doc: dict, i: int, verdict: str, note: str = "") -> dict:
    """Record ``verdict`` ("confirmed" or "rejected") for detection ``i``,
    replacing any earlier verdict on the same object."""
    if verdict not in ("confirmed", "rejected"):
        raise ValueError(verdict)
    d = doc["detections"][i]
    tol = MATCH_TOL_PX * doc["stats"]["pixel_scale_arcsec"]
    verdicts[:] = [v for v in verdicts if not _same_object(v, d, tol)]
    v = {"verdict": verdict, "note": note, "reviewer": getpass.getuser(),
         "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "status": d.get("status"), "score": round(d["score"], 2), "mag": d.get("mag"),
         "rate_deg_day": d["rate_deg_day"], "pa_deg": d["pa_deg"], "positions": _positions(d)}
    verdicts.append(v)
    d["review"] = {k: v[k] for k in ("verdict", "note", "reviewer", "time")}
    return v


# --------------------------------------------------------- object details

def discovery(key, cache: Optional[dict] = None) -> Optional[dict]:
    """Full name and discovery circumstances of a catalogued object (number
    or designation) from JPL's Small-Body Database, or None (unknown to it,
    or no network). ``cache`` (key -> result) avoids repeated queries."""
    import urllib.parse
    import urllib.request
    key = str(key).strip()
    if cache is not None and key in cache:
        return cache[key]
    info = None
    try:
        url = f"{SBDB}?{urllib.parse.urlencode({'sstr': key, 'discovery': 'true'})}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "NEOfind"}),
                                    timeout=30) as r:
            doc = json.loads(r.read().decode())
        disc = doc.get("discovery") or {}
        info = {"fullname": doc["object"]["fullname"], "date": disc.get("date"),
                "who": disc.get("who"), "location": disc.get("location")}
    except Exception:
        return None  # not cached: a later run may have the network
    if cache is not None:
        cache[key] = info
    return info


def sheet_info(d: dict, catalog: Optional[dict], cache: Optional[dict] = None) -> dict:
    """What the stack panel says about detection ``d``: the catalogued
    object's name, discovery date and V magnitude, or "not catalogued"."""
    if d.get("known_id") is not None:
        oid, pre = d["known_id"], ""
    elif d.get("near_known"):
        oid, pre = d["near_known"]["id"], "near "
    else:
        return {"name": "not catalogued"}
    obj = next((a for a in (catalog or {}).get("asteroids", []) if str(a["id"]) == str(oid)), {})
    disc = discovery(oid, cache)
    v = obj.get("v_mag")
    return {"name": pre + (disc["fullname"] if disc else str(obj.get("name", oid))),
            "discovered": (disc or {}).get("date"), "site": (disc or {}).get("location"),
            "v_mag": v if v is not None and v == v else None}


# ------------------------------------------------------------ cutout sheets

def _local_sky(fr, x: float, y: float, r: int = 40) -> float:
    ny, nx = fr.shape
    ya, yb = max(int(y) - r, 0), min(int(y) + r + 1, ny)
    xa, xb = max(int(x) - r, 0), min(int(x) + r + 1, nx)
    box = fr.data[ya:yb, xa:xb][~fr.bad[ya:yb, xa:xb]]
    return float(np.median(box)) if box.size else 0.0


def _sample(fr, xs, ys) -> np.ndarray:
    """Bilinear samples of ``fr``, weighted by pixel validity (nan where
    mostly bad or off the frame)."""
    from scipy import ndimage
    good = (~fr.bad).astype(np.float64)
    num = ndimage.map_coordinates(np.where(fr.bad, 0.0, fr.data).astype(np.float64), [ys, xs],
                                  order=1, mode="constant", cval=0.0)
    den = ndimage.map_coordinates(good, [ys, xs], order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.3, num / den, np.nan)


def _tangent_grid(ra0: float, dec0: float, half: int, scale_deg: float):
    """RA/Dec of a north-up, east-left stamp grid centred on (ra0, dec0)."""
    oy, ox = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    xi, eta = np.radians(-ox * scale_deg), np.radians(oy * scale_deg)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    den = np.cos(d0) - eta * np.sin(d0)
    ra = a0 + np.arctan2(xi, den)
    dec = np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, den))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def stamps(frames, d: dict, half: int = 12, scale_arcsec: float = None):
    """Per frame with the track on it: (k, image, difference), sky-subtracted
    stamps centred on the track, resampled north up, east left (so stars
    shift between frames and the mover stays put). The difference
    subtracts the median of the other frames on the same grid (frames where
    the mover itself is within 3 FWHM of this position are left out)."""
    from .io import default_psf_fwhm_arcsec, world2pix
    fwhm = default_psf_fwhm_arcsec(frames[0])
    scale = (scale_arcsec or frames[0].pixel_scale_arcsec) / 3600.0
    shape = (2 * half + 1, 2 * half + 1)

    def resample(j, ra, dec):
        px, py = world2pix(frames[j].wcs, ra.ravel(), dec.ravel())
        px, py = np.asarray(px).reshape(shape), np.asarray(py).reshape(shape)
        s = _local_sky(frames[j], float(px[half, half]), float(py[half, half]))
        return _sample(frames[j], px, py) - s

    out = []
    for k, f in enumerate(d["frames"]):
        if not f["in_frame"]:
            continue
        fr = frames[k]
        ra, dec = _tangent_grid(f["ra_deg"], f["dec_deg"], half, scale)
        img = resample(k, ra, dec)
        refs = []
        for j, g in enumerate(d["frames"]):
            if j == k:
                continue
            dra = ((g["ra_deg"] - f["ra_deg"] + 180.0) % 360.0 - 180.0) * math.cos(math.radians(f["dec_deg"]))
            if math.hypot(dra, g["dec_deg"] - f["dec_deg"]) * 3600.0 < 3.0 * fwhm:
                continue
            ref = resample(j, ra, dec)
            if frames[j].magzp and fr.magzp:
                ref *= 10.0 ** (0.4 * (fr.magzp - frames[j].magzp))
            refs.append(ref)
        diff = img - np.nanmedian(refs, axis=0) if len(refs) >= 2 else np.full_like(img, np.nan)
        out.append((k, img, diff))
    return out


def write_sheet(frames, doc: dict, i: int, path, half: int = 12, zoom: int = 4,
                max_cols: int = 9, info: Optional[dict] = None) -> Path:
    """PNG contact sheet for detection ``i``: per frame, the image (upper)
    and the difference (lower) around the track, north up, marked at its
    position; frames the score didn't use have an orange border, masked
    or off-frame pixels are dark blue. A last panel (green border) stacks
    the used frames along the track: the stamps are centred on the track,
    so their mean is a shift-and-stack, and a mover too faint to see in
    any one frame (z ~ 2-4 each) shows there. ``info`` (``sheet_info``) is
    written under it: name, discovery date and site, V magnitude."""
    import warnings
    from PIL import Image, ImageDraw

    d = doc["detections"][i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan medians off the frame edge
        st = stamps(frames, d, half, doc["stats"]["pixel_scale_arcsec"])
        used = [s for s in st if d["frames"][s[0]]["used"]]
        stack = (np.nanmean([s[1] for s in used], axis=0), np.nanmean([s[2] for s in used], axis=0)) \
            if used else None
    vals = np.concatenate([s[2].ravel() for s in st] or [np.zeros(1)])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        sig = 1.4826 * float(np.median(np.abs(vals - np.median(vals)))) or 1.0
    else:
        sig = 1.0
    size = (2 * half + 1) * zoom
    head = 30
    n_panels = len(st) + (stack is not None)
    ncol = min(max(n_panels, 1), max_cols)
    nrow = max(1, math.ceil(n_panels / max_cols))
    lines = []
    if info:
        lines = [info.get("name") or ""]
        if info.get("discovered"):
            lines.append(f"disc. {info['discovered']}")
        if info.get("site"):
            lines.append(info["site"])
        if info.get("v_mag") is not None:
            lines.append(f"V {info['v_mag']:.1f}")
    foot = 12 * len(lines) + 4 if lines else 0
    W, H = ncol * (size + 4), head + nrow * (2 * size + 22) + foot
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    mag = f"mag {d['mag']:.2f}" if d.get("mag") is not None else ""
    nk = d.get("near_known")
    tag = d.get("status", "")
    if nk:
        tag += f" ({nk['name']} {nk['sep_arcsec']:.0f}\")"
    elif d.get("known"):
        tag += f" ({d['known']})"
    dr.text((4, 2), f"#{i}  score {d['score']:.1f}  {d['n_frames_used']}/{d['n_frames_covered']} frames  "
                    f"{d['rate_deg_day']:.3f} deg/day  PA {d['pa_deg']:.0f}  {mag}", fill=(255, 255, 255))
    conf = (f"  {d['confirm_band']} {d['confirm_score']:.1f}" if d.get("confirm_score") is not None else "")
    dr.text((4, 15), f"RA {d['ra_deg']:.5f}  Dec {d['dec_deg']:+.5f}  {tag}{conf}", fill=(200, 200, 200))

    def panel(n, label, img, diff, s, border=None):
        r, c = divmod(n, max_cols)
        x0, y0 = c * (size + 4), head + r * (2 * size + 22)
        for row, (a, lo, hi) in enumerate(((img, -3 * s, 15 * s), (diff, -3 * s, 8 * s))):
            u8 = (np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            rgb = np.repeat(u8[::-1, :, None], 3, axis=2)
            rgb[np.isnan(a)[::-1]] = (20, 30, 70)  # masked / off the frame
            canvas.paste(Image.fromarray(rgb).resize((size, size), Image.NEAREST), (x0, y0 + 12 + row * (size + 2)))
            yy = y0 + 12 + row * (size + 2)
            cx, cy = x0 + size / 2, yy + size / 2
            for a0, b0 in ((-3 * zoom - 6, -3 * zoom), (3 * zoom, 3 * zoom + 6)):
                dr.line([cx + a0, cy, cx + b0, cy], fill=(255, 60, 60))
                dr.line([cx, cy + a0, cx, cy + b0], fill=(255, 60, 60))
            if border:
                dr.rectangle([x0, yy, x0 + size - 1, yy + size - 1], outline=border)
        dr.text((x0 + 2, y0), label, fill=(255, 255, 255))

    for n, (k, img, diff) in enumerate(st):
        z = d["frames"][k]["z"]
        panel(n, f"{k}" + (f" z={z:.1f}" if z is not None else ""), img, diff, sig,
              None if d["frames"][k]["used"] else (255, 180, 0))
    if stack is not None:
        panel(len(st), f"stack of {len(used)}", stack[0], stack[1], sig / math.sqrt(len(used)), (60, 220, 90))
        r, c = divmod(len(st), max_cols)
        x0, y0 = c * (size + 4), head + r * (2 * size + 22) + 2 * size + 18
        if x0 + 6 * max(map(len, lines or [""])) > W:  # keep the text on the sheet
            x0 = max(0, W - 6 * max(map(len, lines)) - 2)
        for j, line in enumerate(lines):
            dr.text((x0 + 2, y0 + 12 * j), line, fill=(120, 230, 140))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def write_index(doc: dict, rows: List[int], sheet_dir, title: str) -> Path:
    """A local HTML page with every sheet and its verdict, for browsing."""
    from html import escape
    items = []
    for i in rows:
        d = doc["detections"][i]
        rv = d.get("review")
        verdict = f"{rv['verdict']}" + (f": {escape(rv['note'])}" if rv and rv["note"] else "") if rv else "not reviewed"
        items.append(f"<section><h2>#{i} &middot; {escape(d.get('status', ''))} &middot; {verdict}</h2>"
                     f"<img src=\"det{i:03d}.png\" alt=\"detection {i}\"></section>")
    html = ("<!doctype html><meta charset=utf-8><title>" + escape(title) + "</title>"
            "<style>body{background:#111;color:#ddd;font:14px sans-serif;margin:16px}"
            "img{max-width:100%;image-rendering:pixelated}h2{font-size:15px}</style>"
            f"<h1>{escape(title)}</h1>" + "".join(items))
    p = Path(sheet_dir) / "index.html"
    p.write_text(html)
    return p
