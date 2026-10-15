"""ADES PSV reporting."""

from astropy.io import fits

from neodetect.report import AU_KM, ades_rows, write_ades_psv


def _doc():
    frames = [{"frame": k, "mjd": 57154.5 + 0.066 * k, "ra_deg": 152.0 + 0.01 * k, "dec_deg": 11.6,
               "used": k != 1, "covered": True, "z": 20.0, "flux": 1000.0, "flux_err": 50.0}
              for k in range(3)]
    det = {"detected": True, "frames": frames, "mag": 15.0, "flux": 1000.0, "known_id": "224"}
    return {"stats": {"t_ref_mjd": 57154.566, "pixel_scale_arcsec": 2.75}, "detections": [det]}


def _wise_header():
    h = fits.Header()
    h["TELESCOP"] = "WISE"
    h["BAND"] = 1
    for c, v in zip("XYZ", (4e-5, 2e-5, 1e-5)):
        h[f"ERTH2SC{c}"] = v
    return h


def test_wise_rows_carry_spacecraft_position(tmp_path):
    rows = ades_rows(_doc(), [_wise_header()] * 3)
    assert len(rows) == 2  # frame 1 not used
    r = rows[0]
    assert r["stn"] == "C51" and r["sys"] == "ICRF_KM" and r["ctr"] == "399"
    assert abs(float(r["pos1"]) - 4e-5 * AU_KM) < 1e-3
    assert r["permID"] == "224" and r["band"] == "W1" and r["mag"] == "15.00"
    assert r["obsTime"].startswith("2015-05-12T12:00:00.000") and r["obsTime"].endswith("Z")
    path = write_ades_psv(rows, tmp_path / "obs.psv")
    lines = path.read_text().splitlines()
    assert lines[0] == "# version=2017" and "! mpcCode C51" in lines
    header = [c.strip() for c in lines[-3].split("|")]
    assert header[:3] == ["permID", "trkSub", "mode"] and "provID" not in header
    assert len(lines[-1].split("|")) == len(header)


def test_ground_rows_use_station_and_no_position():
    h = fits.Header()
    h["MPCCODE"] = "W84"
    rows = ades_rows(_doc(), [h] * 3)
    assert rows[0]["stn"] == "W84" and rows[0]["sys"] == ""
