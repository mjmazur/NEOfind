"""Per-visit thresholds from time-scrambled copies (``nullcal``)."""

import numpy as np

from neodetect import nullcal


def test_null_level_is_median_rank_score():
    copies = [[9.0, 8.0, 7.0], [10.0, 6.5, 6.0], [7.0]]
    # 2nd highest per copy: 8.0, 6.5, floor 5.0 -> median 6.5
    assert nullcal.null_level(copies, rank=2) == 6.5


def test_calibrate_ratios_meets_target_on_held_out_copies():
    rng = np.random.default_rng(0)
    visit_nulls = {}
    for v, scale in enumerate([1.0, 1.0, 1.2, 2.0]):  # one noisy visit
        visit_nulls[v] = [[(5.0 + scale * rng.exponential(1.0) * 1.5, int(rng.integers(3, 9)))
                           for _ in range(80)] for _ in range(3)]
    cal = nullcal.calibrate_ratios(visit_nulls, target_far=0.5)
    assert cal["false_per_copy"] <= 0.5
    assert set(cal["ratios"]) == set(nullcal.FRAME_CLASSES)
    # The noisy visit gets the highest thresholds.
    levels = {v: nullcal.null_level([[s for s, _ in c] for c in cs]) for v, cs in visit_nulls.items()}
    assert max(levels, key=levels.get) == 3
    assert nullcal.calibrate_ratios({0: visit_nulls[0][:1]}, 0.05) is None


def test_apply_and_passes():
    thr = nullcal.thresholds(7.0, {"3": 2.0, "4": 1.5, "5+": 1.4})
    dets = [{"score": 12.0, "n_frames_used": 3}, {"score": 14.5, "n_frames_used": 3},
            {"score": 10.0, "n_frames_used": 5}, {"score": 10.0, "n_frames_used": 4}]
    assert nullcal.apply(dets, thr) == 2
    assert [d["detected"] for d in dets] == [False, True, True, False]


def test_real_wise_defaults_guard_copy_and_thresholds():
    from types import SimpleNamespace
    from astropy.io import fits
    from neodetect.io import REAL_WISE_THRESHOLDS, default_null_scrambles, default_thresholds
    h = fits.Header({"TELESCOP": "WISE"})
    assert default_null_scrambles(SimpleNamespace(header=h)) == 1
    assert default_thresholds(SimpleNamespace(header=h)) == REAL_WISE_THRESHOLDS
    h["SIMULATD"] = True
    assert default_null_scrambles(SimpleNamespace(header=h)) == 0
    assert default_thresholds(SimpleNamespace(header=h)) is None
    g = fits.Header({"INSTRUME": "ground", "SIMULATD": True})
    from neodetect.io import GROUND_SIM_THRESHOLDS
    assert default_thresholds(SimpleNamespace(header=g)) == GROUND_SIM_THRESHOLDS


def test_guard_raises_only_noisy_visits():
    base = {"3": 10.0, "4": 9.25, "5+": 10.25}
    assert nullcal.guard_thresholds(7.1, base) == base
    raised = nullcal.guard_thresholds(2 * nullcal.GUARD_LEVEL, base)
    assert raised == {c: 2 * t for c, t in base.items()}


def test_artifact_veto_drops_recurring_site_from_calibration():
    rng = np.random.default_rng(1)

    def copy(extra=()):
        c = [(float(s), 5, 10.0 + rng.uniform(0, 0.5), 20.0 + rng.uniform(0, 0.5))
             for s in rng.uniform(5.0, 8.0, 40)]
        return c + list(extra)

    # One visit has the same strong artifact (score 14) in every copy.
    art = (14.0, 6, 10.3, 20.3)
    visits = {f"v{i}": [copy() for _ in range(3)] for i in range(8)}
    visits["bad"] = [copy([(art[0], art[1], art[2] + 0.002, art[3])]) for _ in range(3)]
    plain = nullcal.calibrate_ratios(visits, target_far=0.05, veto=False)
    vetoed = nullcal.calibrate_ratios(visits, target_far=0.05, veto=True)
    assert plain["ratios"]["5+"] > 1.8  # set by the artifact
    assert vetoed["ratios"]["5+"] < 1.4

    level = 7.0
    sites = nullcal.veto_sites([[art]], level)
    assert nullcal.is_vetoed(art[2] + 0.01, art[3], sites)  # 34" away
    assert not nullcal.is_vetoed(art[2] + 0.03, art[3], sites)  # 100" away
    dets = [{"score": 20.0, "n_frames_used": 6, "ra_deg": art[2], "dec_deg": art[3]},
            {"score": 20.0, "n_frames_used": 6, "ra_deg": 11.0, "dec_deg": 21.0}]
    assert nullcal.apply(dets, nullcal.thresholds(level), sites) == 1
    assert dets[0]["artifact_veto"] and not dets[0]["detected"] and dets[1]["detected"]


def test_null_level_has_a_minimum():
    # A quiet copy with fewer than NULL_RANK candidates (extendedness
    # vetting removed the rest) must not drag thresholds down to noise.
    assert nullcal.null_level([[8.0, 6.0, 5.5]]) == nullcal.MIN_LEVEL
    assert nullcal.null_level([[9.0, 8.5, 8.0, 7.5, 7.2]]) == 7.2


def test_real_wise_skips_crowded_fields_by_default():
    from types import SimpleNamespace
    from astropy.io import fits
    from neodetect.io import default_skip_crowded
    h = fits.Header({"TELESCOP": "WISE"})
    assert default_skip_crowded(SimpleNamespace(header=h))
    h["SIMULATD"] = True
    assert not default_skip_crowded(SimpleNamespace(header=h))


def test_known_masks_cover_catalogued_asteroids():
    from types import SimpleNamespace
    frames = [SimpleNamespace(shape=(60, 80), pixel_scale_arcsec=2.75, psf_fwhm_arcsec=6.5, band="3")
              for _ in range(2)]
    known = {"asteroids": [
        {"err_arcsec": 0.5, "frames": [{"in_frame": True, "x": 20.0, "y": 30.0},
                                       {"in_frame": True, "x": 25.0, "y": 30.0}]},
        {"err_arcsec": 60.0, "frames": [{"in_frame": False, "x": None, "y": None},
                                        {"in_frame": True, "x": 60.0, "y": 10.0}]},
    ]}
    m = nullcal.known_masks(frames, known)
    r = nullcal.MASK_FWHM * 6.5 / 2.75  # 5.9 px
    assert m[0][30, 20] and m[0][30, int(20 + r - 0.5)] and not m[0][30, int(20 + r + 1.5)]
    assert m[1][30, 25] and not m[1][30, 20 - 2]
    # A poorly known orbit gets a wider mask, capped at MASK_MAX_PX.
    assert m[1][10, 60 - 14] and not m[1][10, 60 - 17]
    assert not m[0][10, 60]  # off the frame in frame 0
    assert not any(x.any() for x in nullcal.known_masks(frames, None))


def test_band_calibration_and_w3_floor():
    from types import SimpleNamespace
    from astropy.io import fits
    from neodetect.io import REAL_WISE_CAL, band_calibration, default_thresholds
    h = fits.Header({"TELESCOP": "WISE"})
    f3, f4, f1 = (SimpleNamespace(header=h, band=b) for b in ("3", "4", None))
    assert band_calibration(f3) is REAL_WISE_CAL["3"] and band_calibration(f1) is REAL_WISE_CAL["1"]
    assert default_thresholds(f3) == REAL_WISE_CAL["3"]["fixed"] and default_thresholds(f4)[0] == 9.0
    assert REAL_WISE_CAL["1"]["ratios"] == nullcal.NULL_RATIO
    thr = nullcal.thresholds(7.0, REAL_WISE_CAL["3"]["ratios"], REAL_WISE_CAL["3"]["floor"])
    assert thr["3"] == 20.0 and abs(thr["4"] - 7.0 * 1.64) < 1e-9
    assert band_calibration(SimpleNamespace(header=fits.Header({"SIMULATD": True}), band="3")) is None
