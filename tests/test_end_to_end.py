"""Simulate a small WISE-like sequence with a few asteroids, check the detector
finds them with accurate motion and photometry, and finds nothing in the
matching asteroid-free run."""

import pytest

from neodetect.detector import DetectorConfig, detect
from neodetect.evaluate import truth_eval
from neodetect.io import load_sequence
from neosim.config import (ArtifactConfig, CadenceConfig, InstrumentConfig, PointingConfig,
                           PopulationConfig, SimConfig)
from neosim.simulate import run_simulation

DETECT = DetectorConfig(stack_rate_max=0.15)


def _simulate(tmp_path, name, n_asteroids):
    out = tmp_path / name
    run_simulation(InstrumentConfig(width=320, height=320),
                   CadenceConfig(n_frames=8, drop_prob=0.0),
                   PointingConfig(ra_deg=150.0, dec_deg=12.0, cross_scan_step_arcmin=0.5,
                                  along_scan_offset_frac=0.05),
                   PopulationConfig(n_asteroids=n_asteroids, mag_min=13.0, mag_max=14.0,
                                    rate_min_deg_day=0.04, rate_max_deg_day=0.12,
                                    placement_box_arcmin=6.0),
                   ArtifactConfig(),
                   SimConfig(output_dir=str(out), seed=11, previews=False))
    return load_sequence(out)


@pytest.mark.slow
def test_detects_asteroids_and_matches_truth(tmp_path):
    seq = _simulate(tmp_path, "ast", 3)
    res = detect(seq, DETECT)
    evals = truth_eval(seq, res)
    assert len(evals) == 3
    for te in evals:
        assert te["detection_score"] is not None, f"asteroid {te['id']} not detected"
        assert te["detection_score"] >= DETECT.threshold_for(5)
        err = te["errors"]
        assert err["sep_arcsec_median"] < 1.5
        assert abs(err["rate_err_frac"]) < 0.02
        assert abs(err["mag_err"]) < 0.2


@pytest.mark.slow
def test_no_detection_without_asteroids(tmp_path):
    seq = _simulate(tmp_path, "null", 0)
    res = detect(seq, DETECT)
    assert not any(d["detected"] for d in res.detections)


@pytest.mark.slow
def test_scrambled_copy_sets_visit_thresholds(tmp_path):
    import numpy as np
    from neodetect import nullcal
    seq = _simulate(tmp_path, "ast", 3)
    copy = nullcal.scrambled(seq, np.random.default_rng(0))
    assert sorted(copy.times.tolist()) == sorted(seq.times.tolist())
    assert list(copy.times) == sorted(copy.times)
    assert any(a.path != b.path for a, b in zip(copy.frames, seq.frames))
    cal = nullcal.self_calibrate(seq, DETECT, n=1)
    assert 5.0 <= cal["level"] < 9.0
    res = detect(seq, DETECT)
    assert nullcal.apply(res.detections, cal["thresholds"]) >= 3
