"""Reporting: ADES PSV astrometry for the Minor Planet Center, and animated
cutouts of each detection for visual vetting.

**ADES** (Astrometry Data Exchange Standard, the MPC's submission format;
PSV serialization, version 2017). One row per frame a detection uses, at
the track's fitted position at that frame's mid-exposure time. For WISE the
station is C51 and each row carries the spacecraft's geocentric position
from the frame header (``ERTH2SC[XYZ]``, AU -> km, ``sys=ICRF_KM``,
``ctr=399``), as ADES requires for space-based observations; elsewhere the
station comes from ``--station`` or the header's ``MPCCODE``. Positional
uncertainties combine the frame's centroid error (``sqrt(2) sigma_PSF / z``,
sky-limited Gaussian) with ``ASTROM_FLOOR_ARCSEC``, the median track error
measured on injected asteroids in real NEOWISE frames (README), and
magnitude uncertainties with ``PHOT_FLOOR_MAG``. Submitter
and observer names are placeholders unless given (``neowise reviewer``'s
MPC export asks for them). ``mode`` CCD and ``astCat`` 2MASS are from the
MPC's lists of ADES values; W1-W4 are not among its photometric bands, so
the MPC export leaves the magnitudes out.

Positions are those of the fitted track, which includes a constant
acceleration (``trackfit``): enough for a main-belt orbit over a WISE visit
(the quadratic fits ephemerides to ~0.02"), not necessarily for a close NEO.
"""

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.time import Time

AU_KM = 149597870.7
ASTROM_FLOOR_ARCSEC = 0.25
PHOT_FLOOR_MAG = 0.03  # calibration floor (injected asteroids in real frames: 0.04 rms)
ADES_FIELDS = ["permID", "provID", "trkSub", "mode", "stn", "sys", "ctr", "pos1", "pos2", "pos3",
               "obsTime", "ra", "dec", "rmsRA", "rmsDec", "astCat", "mag", "rmsMag", "band",
               "photCat", "notes", "remarks"]


def _is_wise(h) -> bool:
    return str(h.get("TELESCOP", "")).strip().upper() == "WISE"


def _trksub(i: int, t_ref: float) -> str:
    """Up to 8 characters, unique within a submission: NF + day stamp + index."""
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    n, s = int(t_ref) % 36 ** 3, ""
    for _ in range(3):
        n, r = divmod(n, 36)
        s = digits[r] + s
    return f"NF{s}{i:03d}"


def _designation(d: dict):
    """(permID, provID) for a detection matched to a known object."""
    kid = d.get("known_id")
    if kid is None:
        return "", ""
    kid = str(kid).strip()
    return (kid, "") if kid.isdigit() else ("", kid)


def ades_rows(doc: dict, headers: List, station: Optional[str] = None,
              fwhm_arcsec: Optional[float] = None, detected_only: bool = True) -> List[dict]:
    """One dict per (detection, used frame); ``headers`` are the frames'
    FITS headers in the detection's frame order."""
    t_ref = doc["stats"]["t_ref_mjd"]
    scale = doc["stats"]["pixel_scale_arcsec"]
    rows = []
    for i, d in enumerate(doc["detections"]):
        if detected_only and not d["detected"]:
            continue
        perm, prov = _designation(d)
        for f in d["frames"]:
            if not f["used"]:
                continue
            h = headers[f["frame"]]
            wise = _is_wise(h)
            stn = station or ("C51" if wise else str(h.get("MPCCODE", "") or "XXX"))
            fw = fwhm_arcsec or (6.1 if wise else 2.0 * scale)
            sig_psf = fw / 2.3548
            z = max(float(f["z"] or 0.0), 3.0)
            rms = math.hypot(math.sqrt(2.0) * sig_psf / z, ASTROM_FLOOR_ARCSEC)
            row = {
                "permID": perm, "provID": prov, "trkSub": _trksub(i, t_ref), "mode": "CCD",
                "stn": stn, "sys": "", "ctr": "", "pos1": "", "pos2": "", "pos3": "",
                "obsTime": Time(f["mjd"], format="mjd", scale="utc").isot[:23] + "Z",
                "ra": f"{f['ra_deg']:.6f}", "dec": f"{f['dec_deg']:+.6f}",
                "rmsRA": f"{rms:.3f}", "rmsDec": f"{rms:.3f}",
                "astCat": "2MASS" if wise else "Gaia2", "mag": "", "rmsMag": "", "band": "",
                "photCat": "", "notes": "", "remarks": "",
            }
            if wise and "ERTH2SCX" in h:
                row.update(sys="ICRF_KM", ctr="399",
                           **{f"pos{j + 1}": f"{float(h[f'ERTH2SC{c}']) * AU_KM:.3f}"
                              for j, c in enumerate("XYZ")})
            if f.get("flux") and f["flux"] > 0 and d.get("mag") is not None and d.get("flux"):
                # Per-frame magnitude on the detection's (aperture-corrected) scale.
                mag = d["mag"] - 2.5 * math.log10(f["flux"] / d["flux"])
                row["mag"] = f"{mag:.2f}"
                row["rmsMag"] = f"{math.hypot(1.0857 / max(z, 1.0), PHOT_FLOOR_MAG):.2f}"
                band = str(h.get("BAND", "")).strip()
                row["band"] = f"W{band}" if wise and band else band
            rows.append(row)
    return rows


WISE_TELESCOPE = {"name": "WISE", "aperture": "0.4", "design": "Reflector", "detector": "Si:As"}


def write_ades_psv(rows: List[dict], path, observatory: str = "C51",
                   submitter: str = "SUBMITTER NAME", observers=("OBSERVER NAMES",),
                   measurers=("NEOfind (automated)",), institution: str = "", telescope: Optional[dict] = None,
                   software: Optional[dict] = None, comments=()) -> Path:
    """ADES PSV file: the obsContext header blocks, then a fixed-width
    ``|``-separated table (only columns with any value). ``observers`` and
    ``measurers`` are lists of names, one ``! name`` line each, as ADES
    wants; ``telescope`` needs aperture, design and detector (default: WISE)."""
    names = lambda v: [v] if isinstance(v, str) else [n for n in v if n]
    cols = [c for c in ADES_FIELDS if any(r.get(c) for r in rows)] or ADES_FIELDS[:3]
    width = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in rows]) for c in cols}
    lines = ["# version=2017", "# observatory", f"! mpcCode {observatory}",
             "# submitter", f"! name {submitter}"] + ([f"! institution {institution}"] if institution else [])
    if names(observers):
        lines += ["# observers"] + [f"! name {n}" for n in names(observers)]
    lines += ["# measurers"] + [f"! name {n}" for n in names(measurers)]
    lines += ["# telescope"] + [f"! {k} {v}" for k, v in (telescope or WISE_TELESCOPE).items() if v]
    if software:
        lines += ["# software"] + [f"! {k} {v}" for k, v in software.items() if v]
    if comments:
        lines += ["# comment"] + [f"! line {c}" for c in comments]
    lines.append("|".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        lines.append("|".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def check_ades(rows: List[dict], submitter: str, measurers) -> List[str]:
    """Problems the MPC's validator would reject (the required fields and
    their formats), as messages; empty when none."""
    import re
    probs = []
    if not submitter or submitter == "SUBMITTER NAME":
        probs.append("no submitter name")
    if not [n for n in ([measurers] if isinstance(measurers, str) else measurers) if n]:
        probs.append("no measurer names")
    for k, r in enumerate(rows):
        if not (r.get("permID") or r.get("provID") or r.get("trkSub")):
            probs.append(f"row {k + 1}: no permID, provID or trkSub")
        if r.get("trkSub") and not re.fullmatch(r"[-A-Za-z0-9_]{1,8}", r["trkSub"]):
            probs.append(f"row {k + 1}: trkSub {r['trkSub']!r} is not 1-8 letters, digits, - or _")
        for f in ("mode", "stn", "obsTime", "ra", "dec", "astCat"):
            if not r.get(f):
                probs.append(f"row {k + 1}: no {f}")
        if any(r.get(f) for f in ("sys", "ctr", "pos1", "pos2", "pos3")) and not all(
                r.get(f) for f in ("sys", "ctr", "pos1", "pos2", "pos3")):
            probs.append(f"row {k + 1}: incomplete spacecraft position")
    return probs


def write_cutout_gifs(result, doc: dict, out_dir, half: int = 15, zoom: int = 6,
                      frame_ms: int = 400, detected_only: bool = True) -> List[Path]:
    """Per detection, an animated GIF over the frames: the registered image
    (left) and the static-sky-subtracted residual (right) around the track
    position, the prediction marked; frames the score did not use are
    marked by a dashed frame border."""
    from PIL import Image, ImageDraw

    reg, pre = result.reg, result.pre
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    gh, gw = reg.shape
    for i, d in enumerate(doc["detections"]):
        if detected_only and not d["detected"]:
            continue
        tr = d["grid_track"]
        dt = pre.times - d["t_ref_mjd"]
        xs = tr["x0"] + tr["vx_px_day"] * dt + 0.5 * tr.get("ax_px_day2", 0.0) * dt * dt
        ys = tr["y0"] + tr["vy_px_day"] * dt + 0.5 * tr.get("ay_px_day2", 0.0) * dt * dt
        # Display limits from the residual noise, shared by every frame.
        sig = []
        for k in range(pre.n):
            w = pre.weight[k]
            if np.any(w > 0):
                sig.append(float(np.median(1.0 / np.sqrt(w[w > 0][::97]))))
        s = float(np.median(sig)) if sig else 1.0
        images = []
        for k in range(pre.n):
            cx, cy = int(round(xs[k])), int(round(ys[k]))
            panels = []
            for src, lo, hi in ((reg.data[k], -3 * s, 12 * s), (pre.resid[k], -3 * s, 8 * s)):
                cut = np.full((2 * half + 1, 2 * half + 1), lo, np.float32)
                ya, yb = max(cy - half, 0), min(cy + half + 1, gh)
                xa, xb = max(cx - half, 0), min(cx + half + 1, gw)
                if ya < yb and xa < xb:
                    cut[ya - (cy - half):yb - (cy - half), xa - (cx - half):xb - (cx - half)] = \
                        np.where(reg.valid[k, ya:yb, xa:xb], src[ya:yb, xa:xb], lo)
                u8 = (np.clip((cut - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)[::-1]
                panels.append(Image.fromarray(u8).resize(((2 * half + 1) * zoom,) * 2, Image.NEAREST))
            size = (2 * half + 1) * zoom
            canvas = Image.new("RGB", (2 * size + 4, size + 16), (0, 0, 0))
            for j, p in enumerate(panels):
                canvas.paste(p.convert("RGB"), (j * (size + 4), 16))
            dr = ImageDraw.Draw(canvas)
            px = (xs[k] - cx + half + 0.5) * zoom
            py = size - (ys[k] - cy + half + 0.5) * zoom + 16
            for j in range(2):
                ox = j * (size + 4)
                for a, b in ((-12, -5), (5, 12)):
                    dr.line([ox + px + a, py, ox + px + b, py], fill=(255, 60, 60))
                    dr.line([ox + px, py + a, ox + px, py + b], fill=(255, 60, 60))
            fr = d["frames"][k]
            used = "used" if fr["used"] else ("not used" if fr["covered"] else "no data")
            z = f"z={fr['z']:.1f}" if fr["z"] is not None else ""
            dr.text((2, 2), f"#{i} frame {k} {used} {z}", fill=(255, 255, 255))
            if not fr["used"]:
                dr.rectangle([0, 16, 2 * size + 3, size + 15], outline=(255, 180, 0))
            images.append(canvas)
        path = out_dir / f"det{i:03d}.gif"
        images[0].save(path, save_all=True, append_images=images[1:], duration=frame_ms, loop=0)
        paths.append(path)
    return paths
