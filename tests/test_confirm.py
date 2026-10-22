"""Cross-band confirmation of short W3 tracks (``confirm``)."""

from neodetect import confirm


def test_companion_dir(tmp_path):
    w3 = tmp_path / "r152.090+11.472_s0.6_w3" / "20100514"
    w4 = tmp_path / "r152.090+11.472_s0.6_w4" / "20100514"
    w3.mkdir(parents=True)
    assert confirm.companion_dir(w3, "4") is None  # not fetched
    w4.mkdir(parents=True)
    (w4 / "04566a150-w4-int-1b.fits").write_bytes(b"")
    assert confirm.companion_dir(w3, "4") == w4
    assert confirm.companion_dir(tmp_path / "r1_s0.6_w1" / "x", "4") is None


def test_confirm_short_tracks(monkeypatch, tmp_path):
    rule = {"band": "4", "min_score": 3.0, "fixed_3": 11.75}
    det = lambda score, n, detected=False: {"score": score, "n_frames_used": n, "detected": detected}
    doc = {"detections": [det(15.0, 3), det(16.0, 3), det(9.0, 3), det(25.0, 3, True), det(12.0, 4)]}
    monkeypatch.setattr(confirm, "companion_dir", lambda src, band: tmp_path)
    monkeypatch.setattr(confirm, "forced_scores", lambda comp, ds: [4.2, 0.3][:len(ds)])
    assert confirm.confirm_short_tracks(doc, "x", rule, thr3=11.7) == 1
    d = doc["detections"]
    assert d[0]["detected"] and d[0]["confirm_score"] == 4.2  # W4 confirms
    assert not d[1]["detected"] and d[1]["confirm_band"] == "W4"  # W4 doesn't
    assert not d[2]["detected"] and "confirm_score" not in d[2]  # below the 3-frame threshold
    assert "confirm_score" not in d[4]  # only 3-frame tracks
