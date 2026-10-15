"""WISE data-access helpers that don't need the network: mask bits, visit
grouping, footprint overlap, parallax, SkyBoT parsing, the empirical PSF
and injection into frames."""

import json
import math

import numpy as np
import pytest

from neodetect.io import load_sequence, read_image
from neosim.config import (ArtifactConfig, CadenceConfig, InstrumentConfig, PointingConfig,
                           PopulationConfig, SimConfig)
from neosim.simulate import run_simulation
from neowise import maskbits, query
from neowise.epsf import EPSF, _gaussian_grid, build_epsf
from neowise.inject import EE_WITHIN_STAMP, inject_visit, scramble_visit
from neowise.known import parse_skybot, topocentric


def test_mask_bits_leave_temporal_outliers_usable():
    assert not maskbits.BAD_BITS & (1 << 27)  # moving objects are temporal outliers
    for b in (0, 1, 2, 9, 10, 18, 21, 28):
        assert maskbits.BAD_BITS & (1 << b)
    assert "spike" in " ".join(maskbits.describe(1 << 28))


def _frame(mjd, ra, dec, half=0.39):
    corners = tuple((ra + dx / math.cos(math.radians(dec)), dec + dy)
                    for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)))
    return query.FrameInfo("neowiser", "1a", "1a", 1, 1, mjd, ra, dec, 20.75, 10, 90.0, 90.0, corners)


def test_overlap_fraction_and_visit_grouping():
    assert query.overlap_fraction(_frame(0, 150.0, 10.0), 150.0, 10.0, 0.5) == pytest.approx(1.0)
    assert query.overlap_fraction(_frame(0, 150.5, 10.0), 150.0, 10.0, 0.5) == pytest.approx(0.30, abs=0.03)
    assert query.overlap_fraction(_frame(0, 152.0, 10.0), 150.0, 10.0, 0.5) == 0.0
    orbit = query.ORBIT_DAYS
    frames = []
    for i in range(12):  # one visit, two overlapping frames on some orbits
        frames.append(_frame(100 + i * orbit, 150.0 + 0.05 * i, 10.0))
        if i % 3 == 0:
            frames.append(_frame(100 + i * orbit + 11 / 86400, 150.0 + 0.05 * i, 10.6))
    frames += [_frame(280 + i * orbit, 150.0, 10.0) for i in range(5)]  # six months later
    visits = query.group_visits(sorted(frames, key=lambda f: f.mjd))
    assert [len(v) for v in visits] == [16, 5]
    one = query.one_per_orbit(visits[0], 150.0, 10.0)
    assert len(one) == 12 and all(f.dec == 10.0 for f in one)
    assert len(query.group_visits(visits[0], max_span_days=0.5)) == 2


def test_topocentric_parallax():
    # Observer 6900 km (4.6e-5 AU) perpendicular to the line of sight to an
    # object 2.5 AU away: shift = 4.6e-5 / 2.5 rad = 3.8".
    e2s = np.array([0.0, 0.0, 6900.0 / 1.495978707e8])
    ra, dec = topocentric(10.0, 0.0, 2.5, e2s)
    assert ra == pytest.approx(10.0, abs=1e-9)
    assert (0.0 - dec) * 3600 == pytest.approx(6900 / 1.495978707e8 / 2.5 * 206265, rel=1e-3)


SKYBOT_TEXT = """# Flag: 1
# Ticket: 1
# Num | Name | RA(h) | DE(deg) | Class | Mv | Err(arcsec) | d(arcsec) | dRA(arcsec/h) | dDEC(arcsec/h) | Dg(ua) | Dh(ua)
 224 | Oceana | 10 07 53.3 | +11 38 19.2 | MB>Middle | 13.6 | 0.031 | 900.1 | 25.1 | -4.2 | 2.1 | 2.7
 - | 2015 AB1 | 10 07 00.0 | -00 30 00.0 | NEA | 19.0 | 12.0 | 100.0 | 100.0 | 5.0 | 0.2 | 1.1
"""


def test_parse_skybot():
    rows = parse_skybot(SKYBOT_TEXT)
    assert [r["name"] for r in rows] == ["Oceana", "2015 AB1"]
    assert rows[0]["ra_deg"] == pytest.approx(15 * (10 + 7 / 60 + 53.3 / 3600))
    assert rows[0]["dec_deg"] == pytest.approx(11 + 38 / 60 + 19.2 / 3600)
    assert rows[1]["dec_deg"] == pytest.approx(-0.5)  # sign of "-00"
    assert rows[1]["dist_au"] == 0.2


@pytest.fixture(scope="module")
def sim_visit(tmp_path_factory):
    out = tmp_path_factory.mktemp("visit") / "null"
    run_simulation(InstrumentConfig(width=500, height=500),
                   CadenceConfig(n_frames=6, drop_prob=0.0),
                   PointingConfig(ra_deg=150.0, dec_deg=12.0, cross_scan_step_arcmin=0.5,
                                  along_scan_offset_frac=0.05),
                   PopulationConfig(n_asteroids=0), ArtifactConfig(),
                   SimConfig(output_dir=str(out), seed=5, previews=False))
    return out


def test_epsf_recovers_simulated_gaussian(sim_visit):
    seq = load_sequence(sim_visit)
    sigma = 6.1 / 2.3548 / 2.757
    psf = build_epsf(seq.frames, sigma * 1.15)  # start from a wrong width
    ref = EPSF(_gaussian_grid(sigma, psf.oversample, psf.radius), psf.oversample, psf.radius, 0)
    assert psf.n_stars >= 10
    assert psf.fwhm_px() == pytest.approx(ref.fwhm_px(), rel=0.05)
    yy, xx = np.mgrid[-3:4, -3:4]
    for dx, dy in ((0.0, 0.0), (0.3, -0.4)):  # peak is ~0.16
        assert np.allclose(psf.evaluate(xx - dx, yy - dy), ref.evaluate(xx - dx, yy - dy), atol=0.012)


def test_inject_adds_flux_at_truth_positions(sim_visit, tmp_path):
    out = inject_visit(sim_visit, tmp_path / "inj", n=4, mag_range=(12.0, 12.5),
                       rate_range=(0.05, 0.1), seed=3, half_size_deg=0.12)
    truth = json.loads((out / "truth.json").read_text())
    assert len(truth["asteroids"]) == 4
    orig = load_sequence(sim_visit)
    new = load_sequence(out)
    assert new.n_frames == orig.n_frames
    for k in range(orig.n_frames):
        diff = read_image(new.frames[k].path)[0].astype(float) - read_image(orig.frames[k].path)[0]
        for a in truth["asteroids"]:
            p = a["frames"][k]
            if not p["in_frame"]:
                continue
            ix, iy = int(round(p["x"])), int(round(p["y"]))
            box = diff[max(iy - 8, 0):iy + 9, max(ix - 8, 0):ix + 9]
            if box.shape == (17, 17):  # not cut by the frame edge
                assert box.sum() == pytest.approx(EE_WITHIN_STAMP * p["flux_dn"], rel=0.01)
            peak = np.unravel_index(np.argmax(box), box.shape)
            assert abs(peak[0] - min(iy, 8)) <= 1 and abs(peak[1] - min(ix, 8)) <= 1


def test_scramble_permutes_times(sim_visit, tmp_path):
    out = scramble_visit(sim_visit, tmp_path / "scr", seed=1)
    a, b = load_sequence(sim_visit), load_sequence(out)
    assert sorted(a.times) == pytest.approx(sorted(b.times))
    by_name = {f.path.split("/")[-1].split("-int")[0]: f.mjd for f in a.frames}
    moved = [abs(by_name[f.path.split("/")[-1].split("-int")[0]] - f.mjd) > 1e-6 for f in b.frames]
    assert any(moved)
    assert json.loads((out / "truth.json").read_text())["asteroids"] == []
