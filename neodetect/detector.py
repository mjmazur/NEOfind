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

from .io import (Sequence, default_aperture_correction, default_aperture_ee,
                 default_centroid_offset, world2pix)
from .linking import link
from .preprocess import Preprocessed, preprocess
from .register import Registered, register
from .shiftstack import shift_and_stack
from .sources import find_sources
from .photometry import aperture_correction
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
    n_refine: int = 2000  # seeds refined, best first (crowded real fields exceed 400)
    min_frames: int = 3
    clip_sigma: float = 4.0
    frac_tol: float = 0.35  # max fractional flux variation between frames (~0.3 mag)
    stationary_fwhm: float = 2.0  # search no slower than this many PSF FWHM over the span
    # Reject candidates whose path moves less than this many FWHM over the
    # visit (largest separation of its positions at the frame times, so a
    # curved path can't pass on its speed at t_ref). On real NEOWISE the top
    # scrambled-copy false tracks after the artifact veto were slow tracks on
    # bright-star residuals moving 2.9-3.7 FWHM over a 33 h visit (0.003-0.005
    # deg/day); the slowest of 87 known asteroids in those visits moved 0.084.
    min_path_fwhm: float = 4.0
    min_consistency_p: float = 1e-3  # reject tracks whose per-frame fluxes disagree
    # Path curvature fitted in refinement, bound per axis by
    # max(accel_max, accel_per_rate * rate, accel_per_rate2 * rate^2) in
    # arcsec/day^2 (rate in deg/day;
    # accel_max 0 = straight tracks). Seen from Earth, curvature comes mostly
    # from the observer's acceleration toward the Sun divided by the distance,
    # as the rate does from relative velocity: main-belt asteroids show
    # ~15"/day^2 at ~0.2 deg/day, |a|/rate 40-120 (``trackfit``).
    accel_max_arcsec_day2: float = 60.0
    accel_per_rate: float = 150.0
    # Close objects: the rate itself changes as the distance does, curving
    # the path by ~2 (v_r / v_perp) rate^2 -- up to ~1000"/day^2 at 1.7
    # deg/day for injected NEOs at 0.2 AU closing at 30 km/s. With v_r /
    # v_perp up to ~4 that is <= accel_per_rate2 * rate^2 (rate in deg/day).
    accel_per_rate2: float = 500.0
    # Tracks using fewer frames are fitted straight: with 3 frames the six
    # path parameters pass through any 3 points, and curved 3-frame chance
    # alignments of artifacts set the M4 false-alarm tail on simulations
    # (9 of 11 null candidates >= 9, all with 85-1270"/day^2). Over 3
    # consecutive frames real curvature is ~1 px (a 1.7 deg/day NEO: 38.2
    # straight vs 40.3 curved).
    accel_min_frames: int = 4

    # Repeat the cosmic-ray tests with the static sky subtracted (``register``):
    # a hit on a star is as broad as the star in the first pass.
    sharp_second_pass: bool = True
    # In the same pass, mask long thin features (scattered-light streaks from
    # bright stars beyond a cutout edge, satellite trails; ``cosmics.streak_mask``).
    mask_streaks: bool = True

    # Deflate weights where a frame's local SNR scatter exceeds the noise
    # model (scattered light, crowded-field edges; ``preprocess``); 0 = off.
    local_noise_px: int = 31

    # Photometry: scale Gaussian-fit fluxes to total flux using stars (see
    # ``photometry``); aperture_ee None = instrument default (io).
    aperture_correction: bool = True
    aperture_ee: Optional[float] = None
    # Astrometry: (dx, dy) native px subtracted from Gaussian-fit positions
    # in every frame; None = instrument default (``io.default_centroid_offset``).
    centroid_offset: Optional[tuple] = None

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


def _accel_bound_px(cfg: DetectorConfig, speed_px_day: float, scale_arcsec: float) -> float:
    """Largest path curvature (px/day^2 per axis) to fit for a track this fast."""
    if cfg.accel_max_arcsec_day2 <= 0:
        return 0.0
    rate = speed_px_day * scale_arcsec / 3600.0
    return max(cfg.accel_max_arcsec_day2, cfg.accel_per_rate * rate,
               cfg.accel_per_rate2 * rate * rate) / scale_arcsec


def _path_extent(track: Track, times) -> float:
    """Largest separation (grid px) between the path's positions at ``times``."""
    x, y = track.position(np.asarray(times))
    return float(np.max(np.hypot(x[:, None] - x[None], y[:, None] - y[None])))


def _straight(track: Track, times) -> Track:
    """The straight track through ``track``'s positions at ``times`` (least
    squares): dropping the acceleration alone would move a track far from
    its frames when they lie far from ``t_ref``."""
    xs, ys = track.position(times)
    dt = np.asarray(times) - track.t_ref
    A = np.column_stack([np.ones_like(dt), dt])
    (x0, vx), (y0, vy) = (np.linalg.lstsq(A, v, rcond=None)[0] for v in (xs, ys))
    return Track(float(x0), float(y0), float(vx), float(vy), track.t_ref)


def same_track(a: Track, b: Track, times, tol_px: float = 2.0) -> bool:
    xa, ya = a.position(times)
    xb, yb = b.position(times)
    return bool(np.median(np.hypot(xa - xb, ya - yb)) < tol_px)


def detect(seq: Sequence, cfg: DetectorConfig = None, log=None) -> DetectionResult:
    cfg = cfg or DetectorConfig()
    log = log or (lambda *a: None)
    timing = {}
    t = time.time()
    reg = register(seq, ref=cfg.ref_frame, region=cfg.region,
                   second_pass=cfg.sharp_second_pass, streaks=cfg.mask_streaks)
    timing["register_s"] = time.time() - t
    log(f"registered {reg.n} frames onto a {reg.shape[1]}x{reg.shape[0]} grid "
        f"({timing['register_s']:.1f}s), outlier px masked per frame: {reg.n_outlier_px}")

    t = time.time()
    pre = preprocess(reg, log=log, local_noise_px=cfg.local_noise_px)
    timing["preprocess_s"] = time.time() - t

    apc = {"apcorr": 1.0}
    if cfg.aperture_correction:
        ee = cfg.aperture_ee if cfg.aperture_ee is not None else default_aperture_ee(seq.frames[0])
        apc = aperture_correction(pre.template, pre.coverage >= min(3, reg.n),
                                  float(np.median(pre.psf_sigma_px)), ee,
                                  fallback=default_aperture_correction(seq.frames[0]), log=log)

    times = pre.times
    t_ref = 0.5 * (times[0] + times[-1])
    span = times[-1] - times[0]
    scale = reg.pixel_scale_arcsec
    fwhm_px = float(np.median(pre.psf_sigma_px)) * 2.3548
    v_stationary = cfg.stationary_fwhm * fwhm_px / max(span, 1e-6)
    min_path = cfg.min_path_fwhm * fwhm_px
    stats = {"grid_shape": list(reg.shape), "pixel_scale_arcsec": scale,
             "time_span_h": span * 24.0, "t_ref_mjd": t_ref,
             "noise_alpha": pre.alpha.tolist(), "noise_beta": pre.beta,
             "snr_calibration": pre.calib.tolist(), "n_outlier_px": reg.n_outlier_px,
             "v_stationary_px_day": v_stationary, "aperture_correction": apc}

    seeds: List[Track] = []
    if cfg.link:
        t = time.time()
        dets = find_sources(pre, cfg.peak_threshold, cfg.max_peaks_per_frame)
        tracklets = link(dets, times, t_ref, v_stationary,
                         deg_day_to_px(cfg.link_rate_max, scale),
                         min_detections=cfg.min_detections, sigma_pos=cfg.link_sigma_pos_px,
                         accel_bound=lambda v: _accel_bound_px(cfg, v, scale), log=log)
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
        num, den = evaluator.bilinear_sums(p[:, 0], p[:, 1], p[:, 2], p[:, 3], t_ref,
                                           p[:, 4], p[:, 5])
        quick = np.array([robust_combine(num[i], den[i], cfg.clip_sigma, cfg.frac_tol,
                                         cfg.min_frames)[0] for i in range(len(seeds))])
        # The bilinear coverage test is stricter than the exact one near
        # masked pixels: a seed it sees on too few frames gets the exact
        # score instead (a 1.7 deg/day mover on 3 frames, exact score 38,
        # was dropped here).
        for i in np.flatnonzero(~np.isfinite(quick)):
            ex = evaluator.score(seeds[i])
            quick[i] = ex.score if ex is not None else -math.inf
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
        r = evaluator.refine(tr, accel_max=_accel_bound_px(cfg, tr.speed, scale))
        if r is not None and r.n_used < cfg.accel_min_frames and (r.track.ax or r.track.ay):
            r = evaluator.refine(_straight(r.track, times[r.used]), accel_max=0.0)
        if r is None or r.score < cfg.report_threshold:
            continue
        if _path_extent(r.track, times) < min_path or r.n_used < cfg.min_frames:
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
    # Final paths from fully covered frames only (``refit_geometry``).
    final = [evaluator.refit_geometry(r, accel_max=_accel_bound_px(cfg, r.track.speed, scale)
                                      if r.n_used >= cfg.accel_min_frames else 0.0)
             for r in final]
    final = [r for r in final if r.score >= cfg.report_threshold]
    final.sort(key=lambda r: -r.score)
    timing["refine_s"] = time.time() - t
    log(f"triage + refine: {len(seeds)} seeds -> {len(chosen)} refined -> "
        f"{len(final)} candidates >= {cfg.report_threshold} "
        f"({timing['triage_s'] + timing['refine_s']:.1f}s)")

    stats.update(n_seeds=len(seeds), n_refined=len(chosen), n_inconsistent=n_inconsistent,
                 timing=timing,
                 max_score=max((r.score for r in refined), default=None))
    offsets = [tuple(cfg.centroid_offset) if cfg.centroid_offset is not None
               else default_centroid_offset(f) for f in seq.frames]
    stats["centroid_offset_px"] = [list(o) for o in offsets]
    detections = [to_detection(r, reg, pre, cfg, apc["apcorr"], offsets) for r in final]
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
                 cfg: DetectorConfig, apcorr: float = 1.0, offsets=None) -> dict:
    """Detection record; fluxes are scaled to total flux by ``apcorr``, and
    each frame's position moved by ``-offsets[k]`` native px (the centroid
    convention; the reference position by their mean sky shift)."""
    tr = r.track
    xs, ys = tr.position(pre.times)
    ra, dec = reg.wcs.all_pix2world(xs, ys, 0)
    ra0, dec0 = reg.wcs.all_pix2world([tr.x0], [tr.y0], 0)
    if offsets is not None and np.any(offsets):
        shifts = []
        for k in range(pre.n):
            fx, fy = world2pix(reg.frame_wcs[k], [ra[k]], [dec[k]])
            r2, d2 = reg.frame_wcs[k].all_pix2world(fx - offsets[k][0], fy - offsets[k][1], 0)
            shifts.append((float(r2[0]) - ra[k], float(d2[0]) - dec[k]))
            ra[k], dec[k] = float(r2[0]), float(d2[0])
        dra, ddec = np.median(np.array(shifts), axis=0)
        ra0, dec0 = ra0 + dra, dec0 + ddec
    dt = 0.01
    (xa, xb), (ya, yb) = tr.position([tr.t_ref - dt, tr.t_ref + dt])
    (raa, rab), (deca, decb) = reg.wcs.all_pix2world([xa, xb], [ya, yb], 0)
    c1 = SkyCoord(raa * u.deg, deca * u.deg)
    c2 = SkyCoord(rab * u.deg, decb * u.deg)
    rate = float(c1.separation(c2).deg) / (2 * dt)
    pa = float(c1.position_angle(c2).deg)
    flux = r.flux * apcorr
    mag = reg.magzp - 2.5 * math.log10(flux) if (reg.magzp is not None and flux > 0) else None
    frames = []
    z, ff, fe = r.z, r.frame_flux * apcorr, r.frame_flux_err * apcorr
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
        "grid_track": {"x0": tr.x0, "y0": tr.y0, "vx_px_day": tr.vx, "vy_px_day": tr.vy,
                       "ax_px_day2": tr.ax, "ay_px_day2": tr.ay},
        "accel_arcsec_day2": [tr.ax * reg.pixel_scale_arcsec, tr.ay * reg.pixel_scale_arcsec],
        "frames": frames,
    }
