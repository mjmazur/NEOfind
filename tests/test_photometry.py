"""Aperture correction of the Gaussian matched-filter photometry."""

import math

import numpy as np

from neodetect.photometry import aperture_correction
from neodetect.psf import stamp


def _star_field(core=0.85, sig1=0.95, sig2=3.0, n=60, size=600, noise=1.0, seed=3):
    """Stars with a Gaussian core plus wide wings, on a regular grid with
    random sub-pixel offsets."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise, (size, size))
    side = int(math.ceil(math.sqrt(n)))
    step = size // (side + 1)
    for i in range(side):
        for j in range(side):
            x, y = step * (j + 1) + rng.uniform(-0.5, 0.5), step * (i + 1) + rng.uniform(-0.5, 0.5)
            flux = rng.uniform(2000, 8000)
            for frac, s in ((core, sig1), (1.0 - core, sig2)):
                st, r0, c0 = stamp(x, y, s)
                img[r0:r0 + st.shape[0], c0:c0 + st.shape[1]] += frac * flux * st
    return img


def test_aperture_correction_recovers_wing_loss():
    core, s1, s2, r = 0.85, 0.95, 3.0, 8
    img = _star_field(core, s1, s2)
    # Gaussian-fit flux of a core+wing source, as a fraction of the total
    # (continuous approximation): <G2, G1> / <G1, G1> = 2 s1^2 / (s1^2 + s2^2).
    fit = core + (1 - core) * 2 * s1 ** 2 / (s1 ** 2 + s2 ** 2)
    ee = core + (1 - core) * (1 - math.exp(-r ** 2 / (2 * s2 ** 2)))
    out = aperture_correction(img, np.ones(img.shape, bool), s1, ee_aperture=ee, radius=r)
    assert not out["fallback"] and out["n_stars"] >= 30
    assert abs(out["apcorr"] - 1 / fit) < 0.02 * (1 / fit)


def test_aperture_correction_is_one_for_gaussian_stars_and_falls_back():
    img = _star_field(core=1.0)
    out = aperture_correction(img, np.ones(img.shape, bool), 0.95)
    assert abs(out["apcorr"] - 1.0) < 0.01
    empty = np.random.default_rng(0).normal(0, 1, (300, 300))
    out = aperture_correction(empty, np.ones(empty.shape, bool), 0.95, fallback=1.15)
    assert out["fallback"] and out["apcorr"] == 1.15


def test_centroid_offset_default_only_for_real_wise_w1():
    from types import SimpleNamespace
    from astropy.io import fits
    from neodetect.io import WISE_W1_CENTROID_OFFSET_PX, default_centroid_offset
    h = fits.Header({"TELESCOP": "WISE"})
    assert default_centroid_offset(SimpleNamespace(header=h, band="1")) == WISE_W1_CENTROID_OFFSET_PX
    assert default_centroid_offset(SimpleNamespace(header=h, band="2")) == (0.0, 0.0)
    h["SIMULATD"] = True
    assert default_centroid_offset(SimpleNamespace(header=h, band="1")) == (0.0, 0.0)
