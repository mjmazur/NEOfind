"""Suspected unknown asteroids: classification, verdicts, cutout stamps (``review``)."""

import math
from types import SimpleNamespace

import numpy as np
from astropy.wcs import WCS

from neodetect import review

SCALE = 2.75  # arcsec / px
T = [60000.0 + 0.1 * k for k in range(6)]


def _det(ra0, dec0, rate_deg_day=0.2, score=12.0):
    frames = [{"frame": k, "mjd": t, "ra_deg": ra0 + rate_deg_day * (t - T[0]) / math.cos(math.radians(dec0)),
               "dec_deg": dec0, "in_frame": True, "covered": True, "used": True, "z": 5.0,
               "x_frame": 0.0, "y_frame": 0.0} for k, t in enumerate(T)]
    return {"score": score, "detected": True, "n_frames_used": 6, "n_frames_covered": 6,
            "ra_deg": ra0, "dec_deg": dec0, "rate_deg_day": rate_deg_day, "pa_deg": 90.0,
            "mag": 15.0, "frames": frames}


def _obj(det, name, offset_arcsec=0.0, err=None):
    frames = [dict(f, dec_deg=f["dec_deg"] + offset_arcsec / 3600.0) for f in det["frames"]]
    return {"id": name, "name": name, "rate_deg_day": det["rate_deg_day"], "err_arcsec": err,
            "frames": frames}


def _doc(*dets):
    return {"stats": {"pixel_scale_arcsec": SCALE}, "detections": list(dets)}


def test_classify_known_near_known_suspect_unchecked():
    a, b, c = _det(10.0, 5.0), _det(10.2, 5.0), _det(10.4, 5.0)
    undetected = dict(_det(10.6, 5.0), detected=False)
    cat = {"asteroids": [_obj(a, "Alpha", 2.0), _obj(b, "Beta", 40.0)]}
    doc = _doc(a, b, c, undetected)
    assert review.classify(doc, cat) == {"known": 1, "near-known": 1, "suspect": 1}
    assert a["status"] == "known" and a["known"] == "Alpha"
    assert b["status"] == "near-known" and b["near_known"]["name"] == "Beta"
    assert c["status"] == "suspect" and "status" not in undetected
    # A poorly known orbit widens the near-known radius.
    far = _det(10.8, 5.0)
    review.classify(_doc(far), {"asteroids": [_obj(far, "Gamma", 150.0, err=60.0)]})
    assert far["status"] == "near-known"
    review.classify(_doc(c), None)
    assert c["status"] == "unchecked"


def test_verdicts_follow_the_object_not_the_index():
    a, b = _det(10.0, 5.0), _det(10.2, 5.0)
    doc = _doc(a, b)
    verdicts = []
    review.set_verdict(verdicts, doc, 1, "confirmed", "clean mover")
    review.set_verdict(verdicts, doc, 1, "rejected", "on a ghost")  # replaces the first
    assert len(verdicts) == 1 and verdicts[0]["verdict"] == "rejected"
    # A new run lists the detections in another order, slightly shifted.
    b2 = _det(10.2 + 1.0 / 3600.0, 5.0)
    new = _doc(_det(10.4, 5.0), b2, _det(10.0, 5.0))
    review.attach_verdicts(new, verdicts)
    assert b2["review"]["note"] == "on a ghost"
    assert [("review" in d) for d in new["detections"]] == [False, True, False]


def _frame(rot_deg, star, mover, rng):
    """A 100x100 frame at RA 10, Dec 5 rotated by ``rot_deg``, with a static
    star and a mover at the given sky positions."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 5.0]
    w.wcs.crpix = [50.5, 50.5]
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    w.wcs.cd = SCALE / 3600.0 * np.array([[-c, s], [s, c]])
    yy, xx = np.mgrid[0:100, 0:100]
    data = 100.0 + rng.normal(0, 1.0, (100, 100))
    for (ra, dec), flux in ((star, 400.0), (mover, 200.0)):
        x, y = w.all_world2pix(ra, dec, 0)
        data += flux / (2 * np.pi) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.0)
    return SimpleNamespace(data=data.astype(np.float32), bad=np.zeros((100, 100), bool), wcs=w,
                           magzp=20.0, psf_fwhm_arcsec=2.355 * SCALE, band=None,
                           pixel_scale_arcsec=SCALE, shape=(100, 100))


def test_difference_stamps_remove_stars_and_keep_the_mover():
    rng = np.random.default_rng(3)
    star = (10.0, 5.0 + 6 * SCALE / 3600.0)
    det = _det(10.0 - 12 * SCALE / 3600.0, 5.0 - 4 * SCALE / 3600.0, rate_deg_day=0.03)
    frames = []
    for k, f in enumerate(det["frames"]):
        frames.append(_frame(25.0 * k, star, (f["ra_deg"], f["dec_deg"]), rng))
    st = review.stamps(frames, det, half=20, scale_arcsec=SCALE)
    assert len(st) == 6
    for k, img, diff in st:
        assert diff[20, 20] > 15.0  # mover at the centre (peak ~32)
        # The star, wherever it falls on the stamp, is gone from the difference.
        yx = np.unravel_index(np.nanargmax(np.where(np.hypot(*np.mgrid[-20:21, -20:21]) > 4, img, -1e9)),
                              img.shape)
        assert img[yx] > 30.0 and abs(diff[yx]) < 10.0


def test_sheet_info_names_the_catalogued_object(monkeypatch):
    calls = []

    def fake_discovery(key, cache=None):
        calls.append(key)
        return {"fullname": "380673 (2005 GN25)", "date": "2005-Apr-02", "location": "Mount Lemmon"}
    monkeypatch.setattr(review, "discovery", fake_discovery)
    cat = {"asteroids": [{"id": "380673", "name": "2005 GN25", "v_mag": 22.06}]}
    info = review.sheet_info({"known_id": "380673"}, cat)
    assert info == {"name": "380673 (2005 GN25)", "discovered": "2005-Apr-02", "site": "Mount Lemmon",
                    "v_mag": 22.06}
    assert review.sheet_info({"near_known": {"id": "380673"}}, cat)["name"].startswith("near ")
    assert review.sheet_info({}, cat) == {"name": "not catalogued"}
