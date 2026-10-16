"""Simulator self-consistency: calibration, geometry, noise and bookkeeping."""

import json
import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from neosim.artifacts import MASK_HOT, MASK_SATURATED
from neosim.cadence import build_times
from neosim.cli import build_configs, build_parser
from neosim.config import (ArtifactConfig, CadenceConfig, InstrumentConfig, PointingConfig,
                           PopulationConfig, SimConfig)
from neosim.optics import fwhm_arcsec_to_sigma_px
from neosim.photometry import build_photometric_model, point_source_snr
from neosim.pointing import build_pointings
from neosim.simulate import run_simulation

NO_ARTIFACTS = ArtifactConfig(cosmic_rate_cm2_s=0.0, hot_pixel_frac=0.0, n_bad_columns=0)


def _run(tmp_path, name, inst=None, cad=None, pnt=None, pop=None, art=None, **sim):
    sim.setdefault("seed", 3)
    sim.setdefault("previews", False)
    out = run_simulation(inst or InstrumentConfig(width=256, height=256),
                         cad or CadenceConfig(n_frames=6),
                         pnt or PointingConfig(along_scan_offset_frac=0.05,
                                               cross_scan_step_arcmin=0.5),
                         pop or PopulationConfig(n_asteroids=3, placement_box_arcmin=4.0),
                         art or NO_ARTIFACTS,
                         SimConfig(output_dir=str(tmp_path / name), **sim))
    truth = json.load(open(out / "truth.json"))
    meta = json.load(open(out / "metadata.json"))
    return out, truth, meta


def _load(out, meta, k, kind="int"):
    with fits.open(out / meta["frames"][k]["files"][kind]) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        return hdu.data.astype(np.float64), hdu.header


def test_limiting_magnitude_is_5_sigma():
    inst = InstrumentConfig()
    sigma = fwhm_arcsec_to_sigma_px(inst.psf_fwhm_arcsec, inst.pixel_scale_arcsec)
    phot = build_photometric_model(inst, sigma)
    snr = point_source_snr(phot.total_flux_e(inst.limiting_mag), sigma, inst.background_e,
                           inst.read_noise_e)
    assert snr == pytest.approx(5.0, rel=1e-9)


def test_cadence_modes():
    rng = np.random.default_rng(0)
    t = build_times(CadenceConfig(n_frames=12, drop_prob=0.0, time_jitter_s=0.0), rng)
    assert np.allclose(np.diff(t) * 1440.0, 95.0)
    t = build_times(CadenceConfig(n_frames=12, drop_prob=0.3, time_jitter_s=0.0), rng)
    steps = np.diff(t) * 1440.0 / 95.0
    assert len(t) == 12 and np.allclose(steps, np.round(steps)) and steps.min() > 0.99
    t = build_times(CadenceConfig(mode="visits", n_visits=4, frames_per_visit=3,
                                  time_jitter_s=0.0), rng)
    assert len(t) == 12 and (t[-1] - t[0]) * 1440.0 == pytest.approx(3 * 30 + 2 * 2)


def test_wise_frames_are_aligned_with_ecliptic_north():
    inst = InstrumentConfig()
    times = build_times(CadenceConfig(time_jitter_s=0.0, drop_prob=0.0), np.random.default_rng(1))
    frames = build_pointings(PointingConfig(ra_deg=150.0, dec_deg=20.0, rotation_jitter_deg=0.0),
                             inst, times, 95.0, np.random.default_rng(2))
    for f in frames:
        ra, dec = f.wcs.all_pix2world([507.5, 507.5], [507.5, 607.5], 0)
        ecl = SkyCoord(ra * u.deg, dec * u.deg).transform_to(BarycentricMeanEcliptic)
        d_lon = (ecl.lon[1] - ecl.lon[0]).wrap_at(180 * u.deg).arcsec
        assert ecl.lat[1].arcsec - ecl.lat[0].arcsec == pytest.approx(275.7, abs=1.0)
        assert abs(d_lon) < 1.0
    # The field steps along ecliptic longitude, ~3.9' per orbit.
    c = SkyCoord([f.ra_deg for f in frames] * u.deg, [f.dec_deg for f in frames] * u.deg)
    lon = c.transform_to(BarycentricMeanEcliptic).lon.wrap_at(180 * u.deg).arcmin
    assert np.allclose(np.diff(lon) * np.cos(np.radians(c[0].transform_to(
        BarycentricMeanEcliptic).lat.rad)), 3.9, atol=0.05)


def test_asteroid_positions_and_fluxes_match_truth(tmp_path):
    # Bright, isolated asteroids on an otherwise empty, noiseless-ish sky.
    # ~4e5 e- per asteroid: centroid noise ~0.002 px, flux noise ~0.2%.
    inst = InstrumentConfig(width=256, height=256, read_noise_e=1.0, background_e=1.0,
                            limiting_mag=20.0, saturation_mag=5.0, psf_fwhm_jitter=0.0,
                            zp_jitter_mag=0.0, background_jitter=0.0)
    pop = PopulationConfig(n_asteroids=3, mag=10.0, rate_min_deg_day=0.2, rate_max_deg_day=1.0,
                           placement_box_arcmin=3.0, lc_amp_max=0.0)
    out, truth, meta = _run(tmp_path, "pos", inst=inst, pop=pop, stars=False)
    sky = 1.0 / inst.gain_e_per_dn
    n_checked = 0
    for a in truth["asteroids"]:
        for p in a["frames"]:
            others = [b["frames"][p["frame"]] for b in truth["asteroids"] if b is not a]
            blended = any(math.hypot(o["x"] - p["x"], o["y"] - p["y"]) < 15 for o in others)
            if blended or not (10 < p["x"] < 245 and 10 < p["y"] < 245):
                continue
            n_checked += 1
            img, _ = _load(out, meta, p["frame"])
            x0, y0 = int(round(p["x"])), int(round(p["y"]))
            cut = img[y0 - 6:y0 + 7, x0 - 6:x0 + 7] - sky
            yy, xx = np.mgrid[y0 - 6:y0 + 7, x0 - 6:x0 + 7]
            flux = cut.sum()
            assert flux == pytest.approx(p["flux_dn"], rel=0.01)
            assert (cut * xx).sum() / flux == pytest.approx(p["x"], abs=0.03)
            assert (cut * yy).sum() / flux == pytest.approx(p["y"], abs=0.03)
    assert n_checked >= 10


def test_matched_filter_snr_and_uncertainty_match_prediction(tmp_path):
    inst = InstrumentConfig(width=400, height=400, psf_fwhm_jitter=0.0, background_jitter=0.0,
                            zp_jitter_mag=0.0)
    pop = PopulationConfig(n_asteroids=40, mag=inst.limiting_mag - 1.0, rate_deg_day=0.3,
                           placement_box_arcmin=14.0, lc_amp_max=0.0)
    out, truth, meta = _run(tmp_path, "snr", inst=inst, pop=pop, stars=False,
                            pnt=PointingConfig(along_scan_offset_frac=0.0,
                                               cross_scan_step_arcmin=0.0))
    sigma = meta["frames"][0]["psf_sigma_px"]
    r = 5
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    measured, predicted = [], []
    for k in range(meta["derived"]["n_frames"]):
        img, hdr = _load(out, meta, k)
        unc, _ = _load(out, meta, k, "unc")
        sky_dn = hdr["SKYLEVEL"]
        # The unc image should describe the actual scatter of sky pixels.
        resid = (img - sky_dn) / unc
        assert np.std(resid[:40, :40]) == pytest.approx(1.0, abs=0.05)
        for a in truth["asteroids"]:
            p = a["frames"][k]
            if not (r + 1 < p["x"] < 400 - r - 2 and r + 1 < p["y"] < 400 - r - 2):
                continue
            x0, y0 = int(round(p["x"])), int(round(p["y"]))
            dx, dy = p["x"] - x0, p["y"] - y0
            psf = np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * sigma ** 2))
            psf /= psf.sum()
            cut = img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1] - sky_dn
            w = 1.0 / unc[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1] ** 2
            measured.append((w * psf * cut).sum() / math.sqrt((w * psf ** 2).sum()))
            predicted.append(p["snr"])
    measured, predicted = np.array(measured), np.array(predicted)
    assert len(measured) > 150
    # Optimal-filter SNR averages to the prediction; scatter is ~N(0, 1).
    assert measured.mean() == pytest.approx(predicted.mean(), rel=0.05)
    assert np.std(measured - predicted) == pytest.approx(1.0, abs=0.2)


def test_masks_and_artifacts(tmp_path):
    inst = InstrumentConfig(width=256, height=256)
    art = ArtifactConfig(hot_pixel_frac=1e-3, hot_flag_frac=1.0, cosmic_flag_frac=0.0)
    pop = PopulationConfig(n_asteroids=1, mag=6.0, rate_deg_day=0.01,
                           placement_box_arcmin=2.0)  # saturates
    out, truth, meta = _run(tmp_path, "art", inst=inst, art=art, pop=pop)
    ref = truth["ref_frame"]
    msk, _ = _load(out, meta, ref, "msk")
    msk = msk.astype(int)
    for x, y, _, flagged in truth["hot_pixels"]:
        assert msk[y, x] & MASK_HOT
    p = truth["asteroids"][0]["frames"][ref]
    assert msk[int(round(p["y"])), int(round(p["x"]))] & MASK_SATURATED
    area = 256 * 256 * (18e-4) ** 2
    expected = art.cosmic_rate_cm2_s * area * inst.exposure_s
    counts = [len(c) for c in truth["cosmic_rays"]]
    assert abs(np.mean(counts) - expected) < 4 * math.sqrt(expected / len(counts))


def test_seed_reproducible_and_null_run_shares_scene(tmp_path):
    art = ArtifactConfig()
    out1, t1, m1 = _run(tmp_path, "a", art=art)
    out2, t2, m2 = _run(tmp_path, "b", art=art)
    out3, t3, m3 = _run(tmp_path, "null", art=art,
                        pop=PopulationConfig(n_asteroids=0, placement_box_arcmin=4.0))
    assert np.array_equal(_load(out1, m1, 2)[0], _load(out2, m2, 2)[0])
    assert t1["cosmic_rays"] == t3["cosmic_rays"]
    assert t1["hot_pixels"] == t3["hot_pixels"]
    assert m1["derived"]["n_stars"] == m3["derived"]["n_stars"]
    assert [f["mjd_mid"] for f in m1["frames"]] == [f["mjd_mid"] for f in m3["frames"]]
    assert t3["asteroids"] == []


def test_wcs_header_round_trip(tmp_path):
    out, truth, meta = _run(tmp_path, "wcs")
    for k, f in enumerate(meta["frames"]):
        _, hdr = _load(out, meta, k)
        w = WCS(hdr)
        ra, dec = w.all_pix2world([127.5], [127.5], 0)
        assert ra[0] == pytest.approx(f["center_ra_deg"], abs=1e-9)
        assert dec[0] == pytest.approx(f["center_dec_deg"], abs=1e-9)
        assert hdr["MJD-MID"] == pytest.approx(f["mjd_mid"])


def test_cli_preset_and_overrides():
    args = build_parser().parse_args(["--preset", "ground", "--limiting-mag", "19",
                                      "--no-asteroids", "--no-previews", "--ra", "45"])
    cfg = build_configs(args)
    assert cfg["instrument"].name == "ground" and cfg["instrument"].limiting_mag == 19.0
    assert cfg["instrument"].exposure_s == 60.0
    assert cfg["cadence"].mode == "visits" and cfg["pointing"].mode == "dither"
    assert cfg["pointing"].ra_deg == 45.0
    assert cfg["population"].n_asteroids == 0
    assert cfg["sim"].previews is False and cfg["sim"].preset == "ground"


def test_ground_trailing(tmp_path):
    args = build_parser().parse_args(["--preset", "ground", "--width", "200", "--height", "200",
                                      "--rate", "8", "--n-asteroids", "1", "--no-stars"])
    cfg = build_configs(args)
    cfg["population"].placement_box_arcmin = 1.0
    cfg["sim"].output_dir = str(tmp_path / "trail")
    cfg["sim"].previews = False
    cfg["sim"].seed = 1
    out = run_simulation(*(cfg[s] for s in
                           ("instrument", "cadence", "pointing", "population", "artifacts", "sim")))
    truth = json.load(open(out / "truth.json"))
    # 8 deg/day for 60 s is 20" = 16.7 px at 1.2"/px.
    assert truth["asteroids"][0]["frames"][0]["trail_px"] == pytest.approx(16.67, rel=0.01)


def test_orbit_motion_starts_where_drawn_and_curves():
    from neosim.astrometry import tangent_to_equatorial
    from neosim.motion import build_population

    pop = PopulationConfig(n_asteroids=6, rate_min_deg_day=0.1, rate_max_deg_day=1.0,
                           motion="orbit", placement_box_arcmin=20.0)
    t_ref = 57000.4
    ra0, dec0 = 120.0, 5.0
    ast = build_population(pop, InstrumentConfig(), ra0, dec0, None, np.random.default_rng(2),
                           t_ref=t_ref)
    for a in ast:
        ra, dec = a.radec(t_ref, t_ref, ra0, dec0)
        ra_l, dec_l = tangent_to_equatorial(a.xi0, a.eta0, ra0, dec0)
        assert abs((ra - ra_l) * math.cos(math.radians(dec))) * 3600 < 0.01
        assert abs(dec - dec_l) * 3600 < 0.01
        # Deviation from the chord over +-0.7 day: the path is not straight.
        t = np.array([t_ref - 0.7, t_ref, t_ref + 0.7])
        r, d = a.radec(t, t_ref, ra0, dec0)
        x = (r - r[1]) * math.cos(math.radians(dec0)) * 3600
        y = (d - d[1]) * 3600
        assert math.hypot(0.5 * (x[0] + x[2]), 0.5 * (y[0] + y[2])) > 0.1
        assert json.dumps(a.to_dict())  # truth serializes
