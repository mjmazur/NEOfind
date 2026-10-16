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


def test_null_scrambles_default_on_for_real_wise_only():
    from types import SimpleNamespace
    from astropy.io import fits
    from neodetect.io import default_null_scrambles
    h = fits.Header({"TELESCOP": "WISE"})
    assert default_null_scrambles(SimpleNamespace(header=h)) == 2
    h["SIMULATD"] = True
    assert default_null_scrambles(SimpleNamespace(header=h)) == 0


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
