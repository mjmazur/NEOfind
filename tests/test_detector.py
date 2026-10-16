"""Detector components and a small end-to-end check against the simulator."""

import math

import numpy as np
import pytest

from neodetect.backend import HAVE_NUMBA, get_stack_max
from neodetect.cosmics import sharp_outliers
from neodetect.linking import link
from neodetect.psf import stamp
from neodetect.shiftstack import velocity_grid
from neodetect.sources import FrameDetections
from neodetect.trackfit import robust_combine

PSF_SIGMA = 0.94  # WISE W1 in native pixels


def _add_psf(img, x, y, flux, sigma=PSF_SIGMA):
    st, r0, c0 = stamp(x, y, sigma)
    img[r0:r0 + st.shape[0], c0:c0 + st.shape[1]] += flux * st


def test_sharp_outliers_flags_cosmic_rays_but_not_point_sources():
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 1.0, (200, 200))
    sources = []
    # Point sources from SNR ~8 to very bright, at random sub-pixel phases.
    for i, flux in enumerate(np.geomspace(40, 20000, 30)):
        x, y = 15 + (i % 6) * 30 + rng.uniform(-0.5, 0.5), 15 + (i // 6) * 30 + rng.uniform(-0.5, 0.5)
        _add_psf(img, x, y, flux)
        sources.append((x, y))
    sig = np.sqrt(1.0 + np.clip(img, 0, None) * 0.01)
    # Cosmic rays: single pixel, split across two pixels, split across a corner.
    crs = [(30, 180, [(0, 0, 60.0)]), (90, 180, [(0, 0, 30.0), (0, 1, 30.0)]),
           (150, 180, [(0, 0, 15.0), (0, 1, 15.0), (1, 0, 15.0), (1, 1, 15.0)])]
    for x, y, parts in crs:
        for dy, dx, v in parts:
            img[y + dy, x + dx] += v
    hits = sharp_outliers(img, sig, np.zeros(img.shape, bool), PSF_SIGMA)
    for x, y, _ in crs:
        assert hits[y, x], f"cosmic ray at {(x, y)} not flagged"
    for x, y in sources:
        ix, iy = int(round(x)), int(round(y))
        assert not hits[iy - 1:iy + 2, ix - 1:ix + 2].any(), f"point source at {(x, y)} flagged"


def test_velocity_grid_offsets_unique_and_in_range():
    times = np.array([0.0, 0.066, 0.132, 0.198, 0.264, 0.33, 0.396])
    t_ref = 0.198
    vx, vy, dx, dy = velocity_grid(times, t_ref, 10.0, 60.0, step_px=1.0)
    sp = np.hypot(vx, vy)
    assert sp.min() >= 10.0 and sp.max() <= 60.0
    keys = {tuple(r) for r in np.hstack([dx, dy])}
    assert len(keys) == len(vx)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_numba_and_numpy_backends_agree():
    rng = np.random.default_rng(1)
    n, h, w = 5, 40, 50
    phi = rng.uniform(0.5, 2.0, (n, h, w)).astype(np.float32)
    phi[:, :5, :] = 0.0
    psi = rng.normal(0.0, 1.0, (n, h, w)).astype(np.float32) * np.sqrt(phi)
    dx = rng.integers(-6, 7, (30, n)).astype(np.int32)
    dy = rng.integers(-6, 7, (30, n)).astype(np.int32)
    valid = (phi > 0).astype(np.float32)
    psi[phi == 0] = 0.0
    # min_frames=1: the numba kernel leaves the frame count to the threshold.
    s1, v1 = get_stack_max("numba")(psi, phi, valid, dx, dy, 1)
    s2, v2 = get_stack_max("numpy")(psi, phi, valid, dx, dy, 1)
    s2 = np.where(s2 > 0, s2, -np.inf)  # numba tracks positive S only
    v2 = np.where(s2 > 0, v2, -1)
    finite = np.isfinite(s2)
    assert np.array_equal(finite, np.isfinite(s1))
    assert np.allclose(s1[finite], s2[finite], atol=1e-4)
    assert np.mean(v1[finite] == v2[finite]) > 0.999


def test_robust_combine_rejects_single_frame_outlier():
    den = np.full(8, 4.0)
    num = np.full(8, 2.0 * 4.0)  # flux 2, z = 4 per frame
    s_clean, used = robust_combine(num, den)
    assert used.all() and s_clean == pytest.approx(num.sum() / math.sqrt(den.sum()))
    num_cr = num.copy()
    num_cr[3] = 40.0 * 4.0  # one frame 20x brighter (a cosmic ray on the track)
    s, used = robust_combine(num_cr, den)
    assert not used[3] and used.sum() == 7
    assert s == pytest.approx(num[used].sum() / math.sqrt(den[used].sum()))
    num_neg = num.copy()
    num_neg[5] = -40.0 * 4.0  # strongly negative frames are kept: evidence against
    s, used = robust_combine(num_neg, den)
    assert used[5]


def test_link_recovers_linear_track_among_noise():
    rng = np.random.default_rng(2)
    times = 57000.0 + np.arange(8) * 95.0 / 1440.0
    t_ref = times.mean()
    x0, y0, vx, vy = 300.0, 250.0, 180.0, -95.0  # px/day
    dets = []
    for k, t in enumerate(times):
        xs = list(rng.uniform(0, 600, 60))
        ys = list(rng.uniform(0, 600, 60))
        if k not in (2, 5):  # missed in two frames
            xs.append(x0 + vx * (t - t_ref) + rng.normal(0, 0.3))
            ys.append(y0 + vy * (t - t_ref) + rng.normal(0, 0.3))
        dets.append(FrameDetections(np.array(xs), np.array(ys), np.full(len(xs), 5.0)))
    tracklets = link(dets, times, t_ref, v_min=5.0, v_max=2000.0, min_detections=4)
    best = max(tracklets, key=lambda t: len(t.members))
    assert len(best.members) == 6
    tr = best.track
    assert tr.x0 == pytest.approx(x0, abs=0.5) and tr.y0 == pytest.approx(y0, abs=0.5)
    assert tr.vx == pytest.approx(vx, rel=0.02) and tr.vy == pytest.approx(vy, rel=0.02)


def _star_field_sequence(cr_flux, seed=1):
    """8 slightly dithered frames of 9 stars (300-30000 DN). Frame 2: a
    diffused 1.3 px cosmic-ray track on the 949 DN star; frame 3: an asteroid
    crossing the 533 DN star."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from neodetect.io import Frame, Sequence

    scale, sigma = 2.75, 6.1 / 2.3548 / 2.75
    rng = np.random.default_rng(seed)
    stars = [(25 + 30 * (i % 3) + rng.uniform(-.5, .5), 25 + 30 * (i // 3) + rng.uniform(-.5, .5), f)
             for i, f in enumerate(np.geomspace(300, 30000, 9))]
    frames = []
    for k in range(8):
        dx, dy = 0.37 * k, -0.23 * k
        img = rng.normal(100.0, 5.0, (120, 120))
        for x, y, f in stars:
            _add_psf(img, x + dx, y + dy, f, sigma)
        if k == 2:
            x, y, _ = stars[2]
            for t in (0.25, 0.75):
                _add_psf(img, x + dx + 0.2 + 1.04 * t, y + dy - 0.1 + 0.78 * t, 0.5 * cr_flux, 0.35)
        if k == 3:
            _add_psf(img, stars[1][0] + dx + 0.3, stars[1][1] + dy - 0.2, 300.0, sigma)
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [150.0, 10.0]
        w.wcs.crpix = [60.5 + dx, 60.5 + dy]
        w.wcs.cd = [[-scale / 3600.0, 0.0], [0.0, scale / 3600.0]]
        frames.append(Frame(path=f"f{k}", data=img.astype(np.float32),
                            unc=np.full(img.shape, 5.0, np.float32), bad=np.zeros(img.shape, bool),
                            wcs=w, mjd=57000.0 + 0.066 * k, exptime=7.7, magzp=20.5,
                            psf_fwhm_arcsec=6.1, saturate=None, band="W1", header=fits.Header()))
    return Sequence(frames=frames), stars


def test_second_pass_masks_cosmic_ray_on_star_but_not_asteroid():
    from neodetect.register import register

    seq, stars = _star_field_sequence(200.0)
    x, y, _ = stars[2]
    ra, dec = seq.frames[2].wcs.all_pix2world(x + 0.74 + 0.72, y - 0.46 + 0.29, 0)
    valid = {}
    for second in (False, True):
        reg = register(seq, ref=0, second_pass=second)
        gx, gy = reg.wcs.all_world2pix(ra, dec, 0)
        valid[second] = bool(reg.valid[2, int(round(float(gy))), int(round(float(gx)))])
        assert reg.n_outlier_px[3] == 0, "asteroid on a star masked"
    # The star broadens the hit past the first pass; with the star subtracted it is sharp.
    assert valid[False] and not valid[True]


def test_streak_mask_flags_lines_not_points_or_noise():
    from neodetect.cosmics import streak_mask

    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 1.0, (200, 200))
    ok = np.ones(z.shape, bool)
    clean = streak_mask(z, ok)
    # A faint horizontal streak (z ~1.5 per pixel, 3 px wide, 120 px long) ...
    z[99:102, 40:160] += 1.5
    # ... and bright point sources, in a row and alone.
    for x in (20, 30, 40, 50, 60):
        _add_psf(z, x, 170.0, 400.0)
    _add_psf(z, 150.0, 40.0, 5000.0)
    m = streak_mask(z, ok)
    assert not clean.any()
    assert m[100, 60:140].all()
    assert not m[30:50, 130:170].any()
    assert m[160:180, :].sum() < 50  # a row of stars is not a streak


def test_path_extent_counts_curvature():
    from neodetect.detector import _path_extent
    from neodetect.trackfit import Track

    times = np.linspace(0.0, 1.0, 11)
    straight = Track(0.0, 0.0, 3.0, 0.0, 0.5)
    curved = Track(0.0, 0.0, 0.0, 0.0, 0.5, ax=24.0, ay=0.0)
    assert abs(_path_extent(straight, times) - 3.0) < 1e-9
    assert abs(_path_extent(curved, times) - 3.0) < 1e-9  # 0.5 * 24 * 0.5^2 each way: 0 to 3
