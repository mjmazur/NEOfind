"""End-to-end pipeline: register -> static-sky subtraction -> two seeding
paths (detection linking; shift-and-stack) -> triage -> exact robust track
fit -> vetting -> detections."""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .io import Sequence, world2pix
from .linking import link
from .preprocess import Preprocessed, preprocess
from .register import Registered, register
from .shiftstack import shift_and_stack
from .sources import find_sources
from .trackfit import Track, TrackEvaluator, TrackScore, robust_combine


@dataclass
class DetectorConfig:
    # Sky region (ra_deg, dec_deg, size_deg) to search; None = union of frames.
    region: Optional[tuple] = None
    ref_frame: Optional[int] = None  # grid orientation/zero point; None = middle frame

    # Linking path: single-frame peaks >= peak_threshold, rates in deg/day.
    link: bool = True
    peak_threshold: float = 4.0
    max_peaks_per_frame: int = 3000
    link_rate_max: float = 10.0
    min_detections: int = 3
    link_sigma_pos_px: float = 0.5

    # Shift-and-stack path (the expensive one; rates in deg/day).
    stack: bool = True
    stack_rate_min: float = 0.0
    stack_rate_max: float = 0.25
    stack_threshold: float = 6.0
    stack_zclip: float = 4.0
    stack_step_px: float = 1.0
    backend: str = "auto"

    # Track fit and vetting.
    n_refine: int = 400
    min_frames: int = 3
    clip_sigma: float = 4.0
    frac_tol: float = 0.35  # max fractional flux variation between frames (~0.3 mag)
    stationary_fwhm: float = 2.0  # reject motion < this many PSF FWHM over the span
    min_consistency_p: float = 1e-3  # reject tracks whose per-frame fluxes disagree

    # Coherent-score thresholds: the operating point, by how many frames a
    # track uses. Short tracks need more: three or four single-frame artifacts
    # (cosmic rays that look PSF-like) align by chance far more often than
    # five or more. Calibrate on asteroid-free runs with ``neodetect
    # evaluate``. Candidates above ``report_threshold`` are kept for diagnostics.
    detect_threshold: float = 8.25  # tracks using >= 5 frames
    detect_threshold_4: float = 9.5
    detect_threshold_3: float = 10.25
    report_threshold: float = 5.0

    threads: int = 0  # 0 = all cores

    def threshold_for(self, n_frames: int) -> float:
        if n_frames <= 3:
            return self.detect_threshold_3
        if n_frames == 4:
            return self.detect_threshold_4
        return self.detect_threshold

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionResult:
    detections: List[dict]
    config: dict
    stats: dict
    reg: Optional[Registered] = field(default=None, repr=False)
    pre: Optional[Preprocessed] = field(default=None, repr=False)
    evaluator: Optional[TrackEvaluator] = field(default=None, repr=False)
    t_ref: float = 0.0

    def to_dict(self):
        return {"config": self.config, "stats": self.stats, "detections": self.detections}


def deg_day_to_px(rate_deg_day: float, scale_arcsec: float) -> float:
    return rate_deg_day * 3600.0 / scale_arcsec


def same_track(a: Track, b: Track, times, tol_px: float = 2.0) -> bool:
    xa, ya = a.position(times)
    xb, yb = b.position(times)
    return bool(np.median(np.hypot(xa - xb, ya - yb)) < tol_px)


def detect(seq: Sequence, cfg: DetectorConfig = None, log=None) -> DetectionResult:
    cfg = cfg or DetectorConfig()
    log = log or (lambda *a: None)
    timing = {}
    t = time.time()
    reg = register(seq, ref=cfg.ref_frame, region=cfg.region)
    timing["register_s"] = time.time() - t
    log(f"registered {reg.n} frames onto a {reg.shape[1]}x{reg.shape[0]} grid "
        f"({timing['register_s']:.1f}s), outlier px masked per frame: {reg.n_outlier_px}")

    t = time.time()
    pre = preprocess(reg, log=log)
    timing["preprocess_s"] = time.time() - t

    times = pre.times
    t_ref = 0.5 * (times[0] + times[-1])
    span = times[-1] - times[0]
    scale = reg.pixel_scale_arcsec
    fwhm_px = float(np.median(pre.psf_sigma_px)) * 2.3548
    v_stationary = cfg.stationary_fwhm * fwhm_px / max(span, 1e-6)
    stats = {"grid_shape": list(reg.shape), "pixel_scale_arcsec": scale,
             "time_span_h": span * 24.0, "t_ref_mjd": t_ref,
             "noise_alpha": pre.alpha.tolist(), "noise_beta": pre.beta,
             "snr_calibration": pre.calib.tolist(), "n_outlier_px": reg.n_outlier_px,
             "v_stationary_px_day": v_stationary}

    seeds: List[Track] = []
    if cfg.link:
        t = time.time()
        dets = find_sources(pre, cfg.peak_threshold, cfg.max_peaks_per_frame)
        tracklets = link(dets, times, t_ref, v_stationary,
                         deg_day_to_px(cfg.link_rate_max, scale),
                         min_detections=cfg.min_detections, sigma_pos=cfg.link_sigma_pos_px,
                         log=log)
        seeds += [tl.track for tl in tracklets]
        timing["link_s"] = time.time() - t
        stats.update(n_sources=[len(d) for d in dets], n_tracklets=len(tracklets))

    if cfg.stack:
        t = time.time()
        sc = shift_and_stack(pre, t_ref, max(deg_day_to_px(cfg.stack_rate_min, scale), v_stationary),
                             deg_day_to_px(cfg.stack_rate_max, scale), cfg.stack_threshold,
                             zclip=cfg.stack_zclip, step_px=cfg.stack_step_px,
                             min_frames=cfg.min_frames, backend=cfg.backend,
                             threads=cfg.threads, log=log)
        seeds += [c.track for c in sc]
        timing["stack_s"] = time.time() - t
        stats.update(n_stack_candidates=len(sc))

    # Triage every seed with the fast bilinear approximation.
    t = time.time()
    evaluator = TrackEvaluator(pre, clip=cfg.clip_sigma, frac_tol=cfg.frac_tol,
                               min_frames=cfg.min_frames)
    chosen: List[Track] = []
    if seeds:
        p = np.array([s.params() for s in seeds])
        num, den = evaluator.bilinear_sums(p[:, 0], p[:, 1], p[:, 2], p[:, 3], t_ref)
        quick = np.array([robust_combine(num[i], den[i], cfg.clip_sigma, cfg.frac_tol,
                                         cfg.min_frames)[0] for i in range(len(seeds))])
        for i in np.argsort(-quick):
            if len(chosen) >= cfg.n_refine or not math.isfinite(quick[i]):
                break
            if quick[i] < cfg.report_threshold - 1.5:
                break
            if not any(same_track(c, seeds[i], times) for c in chosen):
                chosen.append(seeds[i])
    timing["triage_s"] = time.time() - t

    t = time.time()
    refined: List[TrackScore] = []
    n_inconsistent = 0
    for tr in chosen:
        r = evaluator.refine(tr)
        if r is None or r.score < cfg.report_threshold:
            continue
        if r.track.speed < v_stationary or r.n_used < cfg.min_frames:
            continue
        if r.consistency_p(cfg.frac_tol) < cfg.min_consistency_p:
            n_inconsistent += 1
            continue
        refined.append(r)
    refined.sort(key=lambda r: -r.score)
    final: List[TrackScore] = []
    for r in refined:
        if any(same_track(f.track, r.track, times[r.used]) for f in final):
            continue
        r = _release_shared_frames(r, final, evaluator, times)
        if r is not None and r.score >= cfg.report_threshold:
            final.append(r)
    final.sort(key=lambda r: -r.score)
    timing["refine_s"] = time.time() - t
    log(f"triage + refine: {len(seeds)} seeds -> {len(chosen)} refined -> "
        f"{len(final)} candidates >= {cfg.report_threshold} "
        f"({timing['triage_s'] + timing['refine_s']:.1f}s)")

    stats.update(n_seeds=len(seeds), n_refined=len(chosen), n_inconsistent=n_inconsistent,
                 timing=timing,
                 max_score=max((r.score for r in refined), default=None))
    detections = [to_detection(r, reg, pre, cfg) for r in final]
    return DetectionResult(detections=detections, config=cfg.to_dict(), stats=stats,
                           reg=reg, pre=pre, evaluator=evaluator, t_ref=t_ref)


def _release_shared_frames(r: TrackScore, accepted: List[TrackScore], evaluator: TrackEvaluator,
                           times, tol_px: float = 2.0) -> Optional[TrackScore]:
    """Drop frames where ``r`` sits on a stronger accepted track (the same
    source can't belong to two objects) and rescore without them."""
    if not accepted:
        return r
    xs, ys = r.track.position(times)
    shared = np.zeros(len(times), bool)
    for f in accepted:
        fx, fy = f.track.position(times)
        shared |= f.covered & (np.hypot(xs - fx, ys - fy) < tol_px)
    if not (shared & r.used).any():
        return r
    return evaluator.score(r.track, used=r.used & ~shared)


def to_detection(r: TrackScore, reg: Registered, pre: Preprocessed,
                 cfg: DetectorConfig) -> dict:
    tr = r.track
    xs, ys = tr.position(pre.times)
    ra, dec = reg.wcs.all_pix2world(xs, ys, 0)
    ra0, dec0 = reg.wcs.all_pix2world([tr.x0], [tr.y0], 0)
    dt = 0.01
    (xa, xb), (ya, yb) = tr.position([tr.t_ref - dt, tr.t_ref + dt])
    (raa, rab), (deca, decb) = reg.wcs.all_pix2world([xa, xb], [ya, yb], 0)
    c1 = SkyCoord(raa * u.deg, deca * u.deg)
    c2 = SkyCoord(rab * u.deg, decb * u.deg)
    rate = float(c1.separation(c2).deg) / (2 * dt)
    pa = float(c1.position_angle(c2).deg)
    flux = r.flux
    mag = reg.magzp - 2.5 * math.log10(flux) if (reg.magzp is not None and flux > 0) else None
    frames = []
    z, ff, fe = r.z, r.frame_flux, r.frame_flux_err
    for k in range(pre.n):
        fx, fy = world2pix(reg.frame_wcs[k], [ra[k]], [dec[k]])
        fh, fw = reg.frame_shapes[k]
        frames.append({
            "frame": k, "mjd": float(pre.times[k]), "ra_deg": float(ra[k]),
            "dec_deg": float(dec[k]), "x_frame": float(fx[0]), "y_frame": float(fy[0]),
            "in_frame": bool(-0.5 <= fx[0] < fw - 0.5 and -0.5 <= fy[0] < fh - 0.5),
            "covered": bool(r.covered[k]), "used": bool(r.used[k]),
            "z": None if not r.covered[k] else float(z[k]),
            "flux": None if not r.covered[k] else float(ff[k]),
            "flux_err": None if not r.covered[k] else float(fe[k]),
        })
    return {
        "score": float(r.score), "score_all_frames": float(r.score_all),
        "detected": bool(r.score >= cfg.threshold_for(r.n_used)),
        "n_frames_used": r.n_used, "n_frames_covered": int(r.covered.sum()),
        "consistency_p": r.consistency_p(cfg.frac_tol),
        "t_ref_mjd": tr.t_ref, "ra_deg": float(ra0[0]), "dec_deg": float(dec0[0]),
        "rate_deg_day": rate, "rate_arcsec_hr": rate * 150.0, "pa_deg": pa,
        "flux": flux, "mag": mag,
        "grid_track": {"x0": tr.x0, "y0": tr.y0, "vx_px_day": tr.vx, "vy_px_day": tr.vy},
        "frames": frames,
    }
