"""Physical sky paths of injected asteroids (neosim.orbit)."""

import math

import numpy as np

from neosim.orbit import OrbitMover, earth_heliocentric


def test_earth_is_one_au_from_the_sun():
    r = np.linalg.norm(earth_heliocentric([57000.0, 57180.0]), axis=1)
    assert np.all(np.abs(r - 1.0) < 0.02)


def test_path_matches_requested_motion_and_curves_like_main_belt():
    t_ref = 57154.8
    m = OrbitMover.from_sky(0, 15.0, 152.0, 11.6, rate_deg_day=0.17, pa_deg=120.0,
                            delta_au=2.33, vr_km_s=22.0, t_ref=t_ref)
    t = t_ref + np.linspace(-0.7, 0.7, 15)
    ra, dec = m.radec(t)
    c = math.cos(math.radians(11.6))
    x, y = (ra - 152.0) * c * 3600, (dec - 11.6) * 3600
    # Light-time corrected position at t_ref is the requested one.
    assert abs(x[7]) < 0.01 and abs(y[7]) < 0.01
    # Rate and direction at t_ref.
    vx, vy = np.polyfit(t - t_ref, x, 2)[1], np.polyfit(t - t_ref, y, 2)[1]
    assert abs(math.hypot(vx, vy) / 3600 - 0.17) < 0.002
    assert abs(math.degrees(math.atan2(vx, vy)) % 360 - 120.0) < 1.0
    # Curvature: arcseconds off a straight line, but a quadratic fits closely.
    for deg, lo, hi in ((1, 0.5, 5.0), (2, 0.0, 0.05)):
        r = np.hypot(x - np.polyval(np.polyfit(t, x, deg), t), y - np.polyval(np.polyfit(t, y, deg), t))
        assert lo <= r.max() <= hi, (deg, r.max())
