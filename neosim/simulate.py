"""Top-level orchestration: build a scene and render every frame."""

import json
import math
import random
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .artifacts import MASK_COSMIC, MASK_SATURATED, build_defects, render_cosmic_rays
from .astrometry import galactic_latitude_deg
from .cadence import build_times
from .config import (ArtifactConfig, CadenceConfig, InstrumentConfig, PointingConfig,
                     PopulationConfig, SimConfig)
from .fitsio import frame_header, write_frame
from .motion import build_population
from .optics import fwhm_arcsec_to_sigma_px
from .photometry import build_photometric_model, point_source_snr
from .pointing import build_pointings
from .preview import write_previews
from .render import apply_noise, render_points, trail_substeps
from .starfield import galactic_density_factor, generate_star_field

COSMIC_FLAG_NSIGMA = 0.5  # flagged-hit pixels above this many sky sigma get MASK_COSMIC


def default_output_dir() -> Path:
    return Path("sims") / datetime.now().strftime("%Y%m%d_%H%M%S")


def _star_region_half_size_deg(frames, pnt: PointingConfig, inst: InstrumentConfig,
                               psf_sigma_px: float) -> float:
    target = SkyCoord(pnt.ra_deg * u.deg, pnt.dec_deg * u.deg)
    centers = SkyCoord([f.ra_deg for f in frames] * u.deg, [f.dec_deg for f in frames] * u.deg)
    max_sep = float(np.max(target.separation(centers).deg))
    half_diag = 0.5 * math.hypot(inst.width, inst.height) * inst.pixel_scale_arcsec / 3600.0
    margin = 6 * psf_sigma_px * inst.pixel_scale_arcsec / 3600.0
    return max_sep + half_diag + margin


def run_simulation(inst: InstrumentConfig, cad: CadenceConfig, pnt: PointingConfig,
                   pop: PopulationConfig, art: ArtifactConfig, sim: SimConfig) -> Path:
    seed = sim.seed if sim.seed is not None else random.SystemRandom().randrange(2 ** 32)
    root = np.random.SeedSequence(seed)
    # Independent streams so e.g. changing the asteroid count leaves stars and
    # noise unchanged for the same seed.
    (rng_time, rng_point, rng_frame, rng_stars, rng_ast, rng_defects,
     rng_noise, rng_cosmic) = [np.random.default_rng(s) for s in root.spawn(8)]

    out_dir = Path(sim.output_dir) if sim.output_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    times = build_times(cad, rng_time)
    n = len(times)
    frames = build_pointings(pnt, inst, times, cad.interval_min, rng_point)
    ref = sim.ref_frame if sim.ref_frame is not None else n // 2
    if not 0 <= ref < n:
        raise ValueError(f"ref_frame {ref} out of range for {n} frames")
    t_ref = float(times[ref])

    sigma0 = fwhm_arcsec_to_sigma_px(inst.psf_fwhm_arcsec, inst.pixel_scale_arcsec)
    phot = build_photometric_model(inst, sigma0)

    # Per-frame conditions.
    fwhm = inst.psf_fwhm_arcsec * np.clip(1 + rng_frame.normal(0, inst.psf_fwhm_jitter, n), 0.5, 2)
    sky = inst.background_e * np.clip(1 + rng_frame.normal(0, inst.background_jitter, n), 0.1, 10)
    dzp = rng_frame.normal(0, inst.zp_jitter_mag, n)
    transparency = 10 ** (-0.4 * dzp)

    if sim.stars:
        half = _star_region_half_size_deg(frames, pnt, inst, sigma0 * 1.5)
        stars = generate_star_field(inst.band, inst.limiting_mag, pnt.ra_deg, pnt.dec_deg,
                                    half, sim.star_density_scale, rng_stars)
        star_flux = phot.total_flux_e(stars.mag)
    else:
        stars = None

    asteroids = build_population(pop, inst, pnt.ra_deg, pnt.dec_deg, frames[ref].wcs, rng_ast)
    defects = build_defects(art, inst, rng_defects)
    shape = (inst.height, inst.width)
    dark = defects.dark_image(shape)
    static_mask = defects.static_mask(shape)
    scale_rad_px = math.radians(inst.pixel_scale_arcsec / 3600.0)

    images_dn, file_records, frame_meta, cosmic_truth = [], [], [], []
    ast_truth = {a.id: [] for a in asteroids}
    for k, (t_mid, geom) in enumerate(zip(times, frames)):
        sigma = fwhm_arcsec_to_sigma_px(fwhm[k], inst.pixel_scale_arcsec)
        model = np.full(shape, sky[k]) + dark

        if stars is not None:
            sx, sy = geom.wcs.all_world2pix(stars.ra_deg, stars.dec_deg, 0)
            model += render_points(shape, sx, sy, star_flux * transparency[k], sigma)

        for a in asteroids:
            trail_px = math.radians(a.rate_deg_day) * inst.exposure_s / 86400.0 / scale_rad_px
            n_sub = trail_substeps(trail_px)
            frac = (np.arange(n_sub) + 0.5) / n_sub - 0.5
            t_sub = t_mid + frac * inst.exposure_s / 86400.0
            ra, dec = a.radec(t_sub, t_ref, pnt.ra_deg, pnt.dec_deg)
            x, y = geom.wcs.all_world2pix(ra, dec, 0)
            mag = float(a.mag_at(t_mid, t_ref))
            flux = phot.total_flux_e(mag) * transparency[k]
            model += render_points(shape, x, y, np.full(n_sub, flux / n_sub), sigma)

            ra_m, dec_m = a.radec(t_mid, t_ref, pnt.ra_deg, pnt.dec_deg)
            xm, ym = geom.wcs.all_world2pix(ra_m, dec_m, 0)
            in_frame = bool(-0.5 <= xm < inst.width - 0.5 and -0.5 <= ym < inst.height - 0.5)
            ast_truth[a.id].append({
                "frame": k, "mjd": float(t_mid), "ra_deg": float(ra_m), "dec_deg": float(dec_m),
                "x": float(xm), "y": float(ym), "in_frame": in_frame, "mag": mag,
                "flux_e": float(flux), "flux_dn": float(flux / inst.gain_e_per_dn),
                "snr": float(point_source_snr(flux, sigma, sky[k], inst.read_noise_e)),
                "trail_px": float(trail_px),
            })

        image = apply_noise(model, inst.read_noise_e, rng_noise)
        cr, cr_flagged, hits = render_cosmic_rays(art, inst, rng_cosmic)
        image += cr
        if len(defects.bad_columns):
            image[:, defects.bad_columns] = rng_noise.normal(
                0.0, inst.read_noise_e, size=(inst.height, len(defects.bad_columns)))
        saturated = image >= phot.full_well_e
        image[saturated] = phot.full_well_e

        mask = static_mask.copy()
        mask[saturated] |= MASK_SATURATED
        noise_sigma = math.sqrt(sky[k] + inst.read_noise_e ** 2)
        mask[cr_flagged > COSMIC_FLAG_NSIGMA * noise_sigma] |= MASK_COSMIC

        g = inst.gain_e_per_dn
        image_dn = image / g
        unc_dn = np.sqrt(np.clip(image, 0.0, None) + inst.read_noise_e ** 2) / g
        magzp_dn = phot.zero_point_dn(g) - dzp[k]
        header = frame_header(geom.wcs, instrument=inst.name, band=inst.band,
                              mjd_mid=float(t_mid), exposure_s=inst.exposure_s,
                              magzp_dn=magzp_dn, gain=g, read_noise_dn=inst.read_noise_e / g,
                              saturate_dn=phot.full_well_e / g, psf_fwhm_arcsec=float(fwhm[k]),
                              sky_dn=float(sky[k] / g), index=k)
        file_records.append(write_frame(out_dir, k, image_dn, unc_dn, mask, header,
                                        sim.compress, seed))
        images_dn.append(image_dn)
        cosmic_truth.append(hits)
        frame_meta.append({
            "index": k, "mjd_mid": float(t_mid), "center_ra_deg": geom.ra_deg,
            "center_dec_deg": geom.dec_deg, "rotation_deg": geom.rotation_deg,
            "psf_fwhm_arcsec": float(fwhm[k]), "psf_sigma_px": sigma,
            "sky_e": float(sky[k]), "zp_offset_mag": float(dzp[k]), "magzp_dn": magzp_dn,
            "n_cosmic_rays": len(hits), "n_saturated_px": int(saturated.sum()),
            "files": file_records[-1],
        })

    if sim.previews:
        tracks = {a.id: [(p["ra_deg"], p["dec_deg"]) for p in ast_truth[a.id]] for a in asteroids}
        write_previews(out_dir, images_dn, [f.wcs for f in frames], ref, tracks)

    metadata = {
        "seed": seed,
        "instrument": inst.to_dict(), "cadence": cad.to_dict(), "pointing": pnt.to_dict(),
        "population": pop.to_dict(), "artifacts": art.to_dict(), "sim": sim.to_dict(),
        "derived": {
            "n_frames": n, "ref_frame": ref, "t_ref_mjd": t_ref,
            "time_span_h": float((times[-1] - times[0]) * 24.0),
            "psf_sigma_px_nominal": sigma0,
            "photometric_zero_point_e": phot.zero_point,
            "photometric_zero_point_dn": phot.zero_point_dn(inst.gain_e_per_dn),
            "noise_equivalent_area_px": phot.nea_px,
            "full_well_e": phot.full_well_e,
            "n_stars": 0 if stars is None else int(len(stars.mag)),
            "galactic_latitude_deg": galactic_latitude_deg(pnt.ra_deg, pnt.dec_deg),
            "galactic_density_factor": galactic_density_factor(pnt.ra_deg, pnt.dec_deg),
            "n_hot_pixels": int(len(defects.hot_x)),
            "bad_columns": [int(c) for c in defects.bad_columns],
        },
        "frames": frame_meta,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    truth = {
        "n_frames": n, "ref_frame": ref, "t_ref_mjd": t_ref,
        "target_ra_deg": pnt.ra_deg, "target_dec_deg": pnt.dec_deg,
        "pixel_convention": "0-based, pixel centers at integers (astropy origin=0)",
        "asteroids": [],
        "cosmic_rays": cosmic_truth,
        "hot_pixels": [[int(x), int(y), float(e), bool(fl)] for x, y, e, fl in
                       zip(defects.hot_x, defects.hot_y, defects.hot_dark_e, defects.hot_flagged)],
    }
    for a in asteroids:
        per_frame = ast_truth[a.id]
        visible = [p for p in per_frame if p["in_frame"]]
        entry = a.to_dict()
        entry.update({
            "rate_arcsec_hr": a.rate_deg_day * 3600.0 / 24.0,
            "rate_px_hr": a.rate_deg_day * 3600.0 / 24.0 / inst.pixel_scale_arcsec,
            "n_frames_in_frame": len(visible),
            "combined_snr": float(math.sqrt(sum(p["snr"] ** 2 for p in visible))),
            "frames": per_frame,
        })
        truth["asteroids"].append(entry)
    with open(out_dir / "truth.json", "w") as f:
        json.dump(truth, f, indent=1)

    return out_dir
