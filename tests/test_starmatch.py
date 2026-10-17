"""PNG input: sidecar times and registration by star matching."""

import json
import math

import numpy as np
import pytest

imageio = pytest.importorskip("imageio")
pytest.importorskip("skimage")

from neodetect.io import load_sequence  # noqa: E402


def _field(shape, stars, dx, dy, rot_deg, rng):
    """Stars (x, y, flux) seen through a frame shifted by (dx, dy) and rotated."""
    img = rng.normal(1000.0, 10.0, shape)
    c = np.array([(shape[1] - 1) / 2, (shape[0] - 1) / 2])
    th = math.radians(rot_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for x, y, f in stars:
        p = rot @ (np.array([x, y]) - c) + c + np.array([dx, dy])
        img += f / (2 * math.pi * 1.2 ** 2) * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * 1.2 ** 2))
    return img


def test_png_sequence_registered_by_star_matching(tmp_path):
    rng = np.random.default_rng(4)
    shape = (200, 220)
    stars = [(rng.uniform(15, 205), rng.uniform(15, 185), rng.uniform(2e3, 3e4)) for _ in range(60)]
    offsets = [(0.0, 0.0, 0.0), (3.4, -2.1, 0.3), (-5.2, 4.7, -0.5), (7.9, 1.3, 0.8)]
    rows = ["file,time"]
    for k, (dx, dy, rot) in enumerate(offsets):
        img = np.clip(_field(shape, stars, dx, dy, rot, rng), 0, 65535).astype(np.uint16)
        imageio.v3.imwrite(tmp_path / f"f{k}.png", img)
        rows.append(f"f{k}.png,{57000.0 + 0.01 * k}")
    (tmp_path / "times.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "sequence.json").write_text(json.dumps({"pixel_scale_arcsec": 2.0}))

    seq = load_sequence(tmp_path)
    assert [round(f.mjd, 3) for f in seq.frames] == [57000.0, 57000.01, 57000.02, 57000.03]
    assert seq.frames[0].pixel_scale_arcsec == pytest.approx(2.0, rel=1e-3)
    assert seq.frames[0].psf_fwhm_arcsec == pytest.approx(2.3548 * 1.2 * 2.0, rel=0.25)
    # Every star lands on the same sky position in every frame.
    ref = seq.frames[len(seq.frames) // 2]
    c = np.array([(shape[1] - 1) / 2, (shape[0] - 1) / 2])
    for (x, y, _) in stars[:10]:
        sky = []
        for f, (dx, dy, rot) in zip(seq.frames, offsets):
            th = math.radians(rot)
            rmat = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            p = rmat @ (np.array([x, y]) - c) + c + np.array([dx, dy])
            sky.append(f.wcs.all_pix2world(p[0], p[1], 0))
        sky = np.array(sky, float)
        spread = np.hypot(*(sky - sky.mean(axis=0)).T * 3600.0) / 2.0  # px
        assert spread.max() < 0.05
