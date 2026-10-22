"""Cross-band confirmation of short W3 tracks by W4.

WISE took W3 and W4 in the same exposures. In 2010 W3 data a few
exposures in dense fields carry single-frame, star-shaped transients that
chain into 3-frame tracks scoring up to ~21, so 3-frame W3 tracks need a
score of 20 on their own (``io.REAL_WISE_CAL``). A real asteroid shows in
W4 as well: along the W3 track, in the companion W4 visit's frames, the
detector's own track score (``forced_scores``) was > 3 for 94% of 205
catalogued asteroids detected in W3 (4 of 5 on 3 frames), and for none of
101 null-copy 3-frame W3 candidates scoring 9-20 (median 0.12). So a
3-frame W3 track below the minimum but above the per-visit 3-frame
threshold counts when W4 confirms it (``confirm_short_tracks``).

The companion visit is the same directory with ``_w3`` replaced by
``_w4`` (``neowise fetch --band 4`` of the same position and size).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np


def companion_dir(source, band: str) -> Optional[Path]:
    """The ``band`` (e.g. "4") twin of a ``neowise`` visit directory, if fetched."""
    if not source:
        return None
    parts = list(Path(source).parts)
    for i, p in enumerate(parts):
        if "_w3" in p:
            parts[i] = p.replace("_w3", f"_w{band}")
            d = Path(*parts)
            return d if d.is_dir() and any(d.glob("*-int-1b.fits*")) else None
    return None


def forced_scores(companion, detections: List[dict], cfg=None) -> List[Optional[float]]:
    """Track score of each detection's path in the companion visit (same
    exposures: frame times must match), on the frames the detection used."""
    from .detector import DetectorConfig
    from .evaluate import truth_track
    from .io import load_sequence
    from .preprocess import preprocess
    from .register import register
    from .trackfit import TrackEvaluator
    cfg = cfg or DetectorConfig()
    seq = load_sequence(str(companion))
    reg = register(seq, second_pass=cfg.sharp_second_pass, streaks=cfg.mask_streaks)
    pre = preprocess(reg, local_noise_px=cfg.local_noise_px)
    ev = TrackEvaluator(pre, clip=cfg.clip_sigma, frac_tol=cfg.frac_tol, min_frames=2)
    t_ref = 0.5 * (pre.times[0] + pre.times[-1])
    out = []
    for d in detections:
        fr = d["frames"]
        if len(fr) != seq.n_frames or any(abs(f["mjd"] - t) > 1e-5 for f, t in zip(fr, seq.times)):
            out.append(None)
            continue
        a = {"frames": [{"ra_deg": f["ra_deg"], "dec_deg": f["dec_deg"], "in_frame": f["used"],
                         "mjd": f["mjd"]} for f in fr]}
        tr = truth_track(a, reg, t_ref)
        sc = ev.score(tr, used=np.array([f["used"] for f in fr])) if tr is not None else None
        out.append(None if sc is None else float(sc.score))
    return out


def confirm_short_tracks(doc: dict, source, rule: dict, thr3: float, log=None) -> int:
    """Flag as detected the 3-frame tracks scoring >= ``thr3`` (the per-visit
    3-frame threshold before the minimum) but below it, that the companion
    band confirms (forced score >= ``rule["min_score"]``). Records
    ``confirm_score`` on each track checked; the number confirmed."""
    log = log or (lambda *a: None)
    cands = [d for d in doc["detections"] if d["n_frames_used"] == 3 and not d["detected"]
             and not d.get("artifact_veto") and d["score"] >= thr3]
    if not cands:
        return 0
    comp = companion_dir(source, rule["band"])
    if comp is None:
        log(f"W{rule['band']} confirmation: no companion visit for {source}; "
            f"{len(cands)} 3-frame track(s) need >= the minimum")
        return 0
    n = 0
    for d, s in zip(cands, forced_scores(comp, cands)):
        d["confirm_band"] = f"W{rule['band']}"
        d["confirm_score"] = s
        if s is not None and s >= rule["min_score"]:
            d["detected"] = True
            n += 1
    log(f"W{rule['band']} confirmation: {n} of {len(cands)} 3-frame track(s) confirmed ({comp})")
    return n
