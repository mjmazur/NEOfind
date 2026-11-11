"""The W3 + W4 campaign (``neowise.campaign``) and its review page (``neowise.reviewer``)."""

import csv
import json
import math
import threading
import time
import urllib.request
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from neowise import campaign as C
from neowise import reviewer as R

T = [55396.0 + 0.066 * k for k in range(6)]


def _det(ra0, dec0, status="suspect", score=12.0, **kw):
    frames = [{"frame": k, "mjd": t, "ra_deg": ra0 + 0.2 * (t - T[0]) / math.cos(math.radians(dec0)),
               "dec_deg": dec0, "in_frame": True, "covered": True, "used": True} for k, t in enumerate(T)]
    return {"score": score, "detected": True, "n_frames_used": 6, "ra_deg": ra0, "dec_deg": dec0,
            "t_ref_mjd": T[2], "rate_deg_day": 0.2, "pa_deg": 90.0, "mag": 10.9, "status": status,
            "frames": frames, **kw}


def _visit(tmp_path, dets, w4=True):
    d3 = tmp_path / "camp" / "data" / "r025.057+10.405_s0.6_w3" / "20100719"
    d4 = tmp_path / "camp" / "data" / "r025.057+10.405_s0.6_w4" / "20100719"
    for d in (d3,) + ((d4,) if w4 else ()):
        d.mkdir(parents=True)
        for k in ("int", "unc", "msk"):
            (d / f"01234a001-w{d.parent.name[-1]}-{k}-1b.fits").write_bytes(b"x")
    (d3 / "detections.json").write_text(json.dumps({"stats": {"pixel_scale_arcsec": 2.75}, "detections": dets}))
    return d3, d4


def test_ecliptic_fields_nest_and_cover_the_band():
    narrow = C.ecliptic_band_fields(2.0, 0.6)
    wide = C.ecliptic_band_fields(4.0, 0.6)
    assert {f[:2] for f in narrow} < {f[:2] for f in wide}
    assert max(abs(f[3]) for f in narrow) <= 2.0
    # 7 latitude rows of ~600 fields at 0.6 deg
    assert 6 * 590 < len(narrow) < 7 * 610


def test_add_fields_appends_new_ones_in_random_order(tmp_path):
    db = C.connect(tmp_path)
    assert C.add_fields(db, C.ecliptic_band_fields(1.0, 0.6)) > 0
    n1 = db.execute("SELECT COUNT(*) FROM fields").fetchone()[0]
    assert C.add_fields(db, C.ecliptic_band_fields(1.0, 0.6)) == 0  # idempotent
    added = C.add_fields(db, C.ecliptic_band_fields(2.0, 0.6))
    assert added > 0
    first = db.execute("SELECT elat FROM fields ORDER BY ord LIMIT ?", (n1,)).fetchall()
    assert max(abs(r[0]) for r in first) <= 1.0  # the old band still comes first
    lats = [r[0] for r in db.execute("SELECT elat FROM fields ORDER BY ord LIMIT 50")]
    assert len(set(lats)) > 1  # shuffled, not row by row


def test_active_hours_window():
    h = C.parse_hours("10-17")
    assert h == (10, 17)
    assert C.in_window(h, datetime(2026, 10, 28, 10, 0))
    assert C.in_window(h, datetime(2026, 10, 28, 16, 59))
    assert not C.in_window(h, datetime(2026, 10, 28, 17, 0))
    assert not C.in_window(h, datetime(2026, 10, 28, 9, 59))
    assert C.parse_hours("9:30-17:15") == (9.5, 17.25)
    assert C.in_window((22, 6), datetime(2026, 10, 28, 2, 0))  # wraps past midnight


def test_hours_follow_the_time_zone():
    assert C.local_now("UTC").utcoffset().total_seconds() == 0
    toronto = C.local_now("America/Toronto")
    assert toronto.utcoffset().total_seconds() in (-4 * 3600, -5 * 3600)  # EDT / EST
    camp_hour = toronto.hour  # what the campaign compares with --hours
    assert C.in_window((camp_hour, camp_hour + 1), toronto)


def test_gpu_yield_ignores_small_users_and_named_ones(monkeypatch):
    out = ("300402, /usr/share/rustdesk/rustdesk, 142\n"
           "4000, /usr/bin/blender, 2900\n"
           "4001, /usr/bin/obs, 900\n"
           "5000, /opt/neofind/python, 3000\n")
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out))
    assert C.other_gpu_users({5000}, 300, ["obs"]) == ["4000 blender 2900 MiB"]
    # by name, even if big
    out2 = "1, /usr/share/rustdesk/rustdesk, 900\n"
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out2))
    assert C.other_gpu_users(set(), 300, ["rustdesk"]) == []


def test_w4_gate_pass_fail_and_no_w4(tmp_path, monkeypatch):
    from neodetect import confirm
    dets = [_det(25.0, 10.3), _det(25.1, 10.3), _det(25.2, 10.3, status="known"),
            _det(25.3, 10.3, confirm_score=5.0, n_frames_used=3)]
    d3, d4 = _visit(tmp_path, dets)
    monkeypatch.setattr(confirm, "forced_scores", lambda comp, ds: [4.2, 1.0, 3.5][:len(ds)])
    counts = C.w4_gate(d3, 3.0, log=lambda *a: None)
    assert counts == {"pass": 3, "fail": 1, "no-w4": 0}  # catalogued ones too: the review is blind
    out = json.loads((d3 / "detections.json").read_text())["detections"]
    assert [d.get("w4_gate") for d in out] == ["pass", "fail", "pass", "pass"]
    assert out[3]["w4_score"] == 5.0  # the short-track confirmation's score is reused

    d3b, _ = _visit(tmp_path / "b", [_det(25.0, 10.3)], w4=False)
    assert C.w4_gate(d3b, 3.0, log=lambda *a: None) == {"pass": 0, "fail": 0, "no-w4": 1}


def _campaign(tmp_path, **kw):
    camp = C.Campaign(tmp_path / "camp", C.Settings(hours="0-24", yield_gpu=False, **kw))
    C.add_fields(camp.db, [(25.057, 10.405, 0.0, 0.0)])
    return camp


def test_record_lists_every_detection_and_deletes_fits_of_catalogued_only(tmp_path):
    camp = _campaign(tmp_path)
    d3, d4 = _visit(tmp_path, [_det(25.0, 10.3, status="known", w4_gate="pass"),
                               _det(25.1, 10.3, status="known", w4_gate="fail")])
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES (?, 1, 'searching')", (str(d3),))
    camp.record(str(d3), 300.0)
    v = camp.sql("SELECT * FROM visits")[0]
    assert (v["status"], v["detected"], v["known"], v["n_candidates"]) == ("done", 2, 2, 0)
    rows = camp.sql("SELECT status, sheet FROM candidates ORDER BY det")
    assert [r["status"] for r in rows] == ["known", "known"]  # the blind review sees them too
    assert rows[0]["sheet"].endswith("review/blind/det000.png")
    assert not list(d3.glob("*.fits")) and not list(d4.glob("*.fits"))  # nothing new to report
    assert (d3 / "detections.json").exists()


def test_record_keeps_fits_of_a_visit_with_anything_not_catalogued(tmp_path):
    camp = _campaign(tmp_path)
    d3, d4 = _visit(tmp_path, [_det(25.1, 10.3, w4_gate="fail", w4_score=1.0)])
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES (?, 1, 'searching')", (str(d3),))
    camp.record(str(d3), 300.0)
    assert len(list(d3.glob("*.fits"))) == 3 and not list(d4.glob("*.fits"))  # W3 kept for an ADES report
    rv = R.Reviewer(camp.out)
    cid = rv.candidates("todo")[0]["id"]
    rv.verdict(cid, "likely")
    assert len(list(d3.glob("*.fits"))) == 3
    rv.verdict(cid, "rejected")
    assert not list(d3.glob("*.fits"))  # settled: nothing to report


def test_backfill_sheets_redoes_searched_visits_first(tmp_path):
    camp = _campaign(tmp_path)
    d3, _ = _visit(tmp_path, [_det(25.1, 10.3)])
    camp.sql("INSERT INTO visits (dir, field_id, mjd, status, detected, seconds) VALUES "
             "(?, 1, 55396, 'done', 1, 400), ('new', 1, 55397, 'fetched', NULL, NULL), "
             "('none', 1, 55398, 'done', 0, 300)", (str(d3),))
    assert C.backfill_sheets(camp.out) == 1  # not the visit with no detections
    assert camp.sql("SELECT status, redo FROM visits WHERE dir=?", (str(d3),))[0][:] == ("listed", 1)
    assert camp.visits_processed() == 2  # a redo doesn't count against --max-visits again
    camp.max_visits = 2
    camp._update_active()
    assert camp.state != "done"  # the test batch isn't finished until the redos are
    camp.sql("UPDATE visits SET status='fetched' WHERE dir=?", (str(d3),))
    steps = []
    camp._run = lambda worker, args, log, env: steps.append(args[3]) or 0
    camp.record = lambda d, sec, redo=False: steps.append(("record", redo))
    camp.max_visits = 2
    camp.active.set()
    t = threading.Thread(target=camp.worker, args=(0,), daemon=True)
    t.start()
    for _ in range(100):
        if ("record", True) in steps:
            break
        time.sleep(0.05)
    camp.stop.set()
    t.join(10)
    assert steps[:3] == ["gate", "sheets", ("record", True)]  # no new search
    assert "detect" not in steps  # and --max-visits 2 is reached: 'new' waits


def test_pause_and_window_deactivate(tmp_path):
    camp = _campaign(tmp_path)
    camp._update_active()
    assert camp.active.is_set() and camp.state == "running"
    C.set_control(camp.out, paused=True, reason="GPU needed")
    camp._update_active()
    assert not camp.active.is_set() and camp.state == "paused" and "GPU needed" in camp.reason
    C.set_control(camp.out, paused=False)
    camp.s.hours = "0-0:01" if datetime.now().hour or datetime.now().minute > 0 else "1-2"
    camp._update_active()
    assert camp.state == "waiting"


def test_run_requeues_interrupted_searches(tmp_path):
    camp = _campaign(tmp_path)
    camp.sql("INSERT INTO visits (dir, field_id, status, attempts) VALUES ('v', 1, 'searching', 1)")
    camp.run(max_visits=0)  # a batch of 0 visits: finished at once
    assert tuple(camp.sql("SELECT status, attempts FROM visits")[0]) == ("fetched", 0)
    assert json.loads((camp.out / C.STATUS_NAME).read_text())["state"] == "done"


def test_reviewer_verdicts_reach_review_json_and_the_final_list(tmp_path):
    camp = _campaign(tmp_path)
    d3, _ = _visit(tmp_path, [_det(25.1, 10.3, w4_gate="pass", w4_score=6.0),
                              _det(25.2, 10.3, w4_gate="no-w4"), _det(25.3, 10.3, w4_gate="fail"),
                              _det(25.4, 10.3, status="known", w4_gate="pass", known="2001 VS44", known_id="64477")])
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES (?, 1, 'searching')", (str(d3),))
    camp.record(str(d3), 300.0)
    camp.sql("UPDATE candidates SET known_name=NULL WHERE status='known'")  # recorded before the names were kept
    rv = R.Reviewer(camp.out)  # names them from detections.json
    todo = rv.candidates("todo")
    assert len(todo) == 4  # every detection, whatever its W4 gate or catalogue match
    assert not any(k in c for c in todo for k in ("status", "near", "gate", "known_name"))  # blind
    by_det = {c["det"]: c["id"] for c in todo}

    rv.verdict(by_det[0], "confirmed", "clear in the stack")
    rv.verdict(by_det[1], "likely")
    rv.verdict(by_det[2], "rejected")
    rv.verdict(by_det[3], "confirmed")
    verdicts = json.loads((d3 / "review.json").read_text())["verdicts"]
    assert sorted(v["verdict"] for v in verdicts) == ["confirmed", "confirmed", "likely", "rejected"]
    final = list(csv.DictReader(open(camp.out / "confirmed.csv")))
    assert [(r["verdict"], r["catalogue"]) for r in final] == [
        ("confirmed", "suspect"), ("confirmed", "known"), ("likely", "suspect")]
    assert final[0]["note"] == "clear in the stack"
    assert rv.candidates("todo") == []
    assert rv.candidates("results")[0]["status"]  # with a verdict, the match is shown
    assert {c["det"]: c["known_name"] for c in rv.candidates("confirmed")} == {0: "", 3: "2001 VS44 (64477)"}
    assert rv.candidates("rejected")[0]["status"] == "suspect"
    assert rv.results() == {"confirmed": {"suspect": 1, "known": 1}, "likely": {"suspect": 1},
                            "rejected": {"suspect": 1}}

    rv.verdict(by_det[0], None)  # undo
    assert sorted(v["verdict"] for v in json.loads((d3 / "review.json").read_text())["verdicts"]) == [
        "confirmed", "likely", "rejected"]
    assert len((camp.out / "confirmed.csv").read_text().splitlines()) == 3


def test_reviewer_waits_for_blind_sheets(tmp_path):
    camp = _campaign(tmp_path)
    camp.sql("INSERT INTO candidates (visit_dir, det, status, score, sheet) VALUES "
             "('v', 0, 'suspect', 9, 'v/review/det000.png')")  # searched before the review was blind
    rv = R.Reviewer(camp.out)
    assert rv.candidates("todo") == [] and rv.state()["counts"]["waiting"] == 1


def test_reviewer_http_api(tmp_path):
    camp = _campaign(tmp_path)
    d3, _ = _visit(tmp_path, [_det(25.1, 10.3, w4_gate="pass", w4_score=6.0)])
    (d3 / C.BLIND_DIR).mkdir(parents=True)
    (d3 / C.BLIND_DIR / "det000.png").write_bytes(b"\x89PNG fake")
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES (?, 1, 'searching')", (str(d3),))
    camp.record(str(d3), 300.0)
    rv = R.Reviewer(camp.out)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), R.make_handler(rv))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
            return json.loads(data) if r.headers["Content-Type"].startswith("application/json") else data
    try:
        assert b"NEOfind blind review" in call("/")
        cands = call("/api/candidates?view=todo")
        assert len(cands) == 1
        assert call(f"/sheet/{cands[0]['id']}.png").startswith(b"\x89PNG")
        call("/api/control", {"action": "pause"})
        assert C.read_control(camp.out)["paused"] is True
        st = call("/api/state")
        assert st["counts"]["todo"] == 1 and st["control"]["paused"]
        call("/api/verdict", {"id": cands[0]["id"], "verdict": "confirmed"})
        assert call("/api/candidates?view=confirmed")[0]["verdict"] == "confirmed"
        assert call("/api/results") == {"confirmed": {"suspect": 1}}
        assert call("/api/candidates?view=notcat")[0]["status"] == "suspect"
        r = call("/api/export", {"submitter": "J. Smith"})  # the fake FITS can't be read: skipped, not a crash
        assert r["observations"] == 0 and len(r["skipped"]) == 1
        assert call("/api/mpc")["submitter"] == "J. Smith"  # kept for the next export
    finally:
        srv.shutdown()


def test_retry_failed_requeues_by_stage(tmp_path):
    camp = _campaign(tmp_path)
    camp.sql("INSERT INTO visits (dir, field_id, status, attempts, error) VALUES "
             "('a', 1, 'failed', 3, 'fetch: SkyBoT 502'), ('b', 1, 'failed', 3, 'detect exited 1'), "
             "('c', 1, 'done', 1, NULL)")
    assert C.retry_failed(camp.out) == 2
    rows = {r["dir"]: (r["status"], r["attempts"]) for r in camp.sql("SELECT * FROM visits")}
    assert rows == {"a": ("listed", 0), "b": ("fetched", 0), "c": ("done", 1)}


def test_stuck_download_is_put_back_and_the_next_visit_fetched(tmp_path, monkeypatch):
    camp = C.Campaign(tmp_path / "camp", C.Settings())
    camp.active.set()
    camp.db.execute("INSERT INTO fields (id, ra, dec, ord, status) VALUES (1, 10, 0, 0, 'listed')")
    camp.db.executemany("INSERT INTO visits (dir, field_id, mjd, status) VALUES (?, 1, ?, 'listed')",
                        [(str(tmp_path / "a"), 1.0), (str(tmp_path / "b"), 2.0)])
    camp.db.commit()
    monkeypatch.setattr(camp, "_sleep", lambda s: None)
    tried = []

    def fetch(row):
        tried.append(Path(row["dir"]).name)
        if row["dir"].endswith("a"):
            camp._retrying("fetch a", lambda: 1 / 0, tries=C.FETCH_TRIES, wait=120)
        camp.sql("UPDATE visits SET status='fetched' WHERE dir=?", (row["dir"],))
        camp.stop.set()

    monkeypatch.setattr(camp, "fetch_visit", fetch)
    camp.fetcher()
    assert tried == ["a", "b"]  # a put back, b fetched without waiting for it
    a = camp.sql("SELECT * FROM visits WHERE dir=?", (str(tmp_path / "a"),))[0]
    assert a["status"] == "listed" and a["fetch_tries"] == 1 and a["fetch_after"] > time.time() + 3000
    assert C.status_summary(camp.out)["put_back"] == 1

    camp.sql("UPDATE visits SET fetch_tries=?, fetch_after=NULL WHERE dir=?",
             (C.FETCH_MAX_DEFERS - 1, str(tmp_path / "a")))
    camp.put_back(camp.sql("SELECT * FROM visits WHERE dir=?", (str(tmp_path / "a"),))[0], "gave up")
    assert camp.sql("SELECT status FROM visits WHERE dir=?", (str(tmp_path / "a"),))[0][0] == "failed"
    assert C.retry_failed(camp.out) == 1
    a = camp.sql("SELECT * FROM visits WHERE dir=?", (str(tmp_path / "a"),))[0]
    assert a["status"] == "listed" and a["fetch_tries"] == 0 and a["fetch_after"] is None


def test_mpc_export_of_the_confirmed_not_catalogued(tmp_path, monkeypatch):
    from astropy.io import fits
    from neodetect import cli as ncli
    camp = _campaign(tmp_path)
    d3, _ = _visit(tmp_path, [_det(25.1, 10.3, w4_gate="pass", flux=1000.0),
                              _det(25.2, 10.3, w4_gate="pass"),
                              _det(25.4, 10.3, status="known", known="2001 VS44", known_id="64477")])
    doc = json.loads((d3 / "detections.json").read_text())
    doc["stats"] = {**doc.get("stats", {}), "t_ref_mjd": T[2], "pixel_scale_arcsec": 2.75}
    for d in doc["detections"]:
        for f in d["frames"]:
            f.update(z=10.0, flux=1000.0)
    (d3 / "detections.json").write_text(json.dumps(doc))
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES (?, 1, 'searching')", (str(d3),))
    camp.record(str(d3), 300.0)
    rv = R.Reviewer(camp.out)
    by_det = {c["det"]: c["id"] for c in rv.candidates("todo")}
    rv.verdict(by_det[0], "confirmed")
    rv.verdict(by_det[1], "likely")
    rv.verdict(by_det[2], "confirmed")
    assert {c["det"] for c in rv.candidates("notcat")} == {0, 1}  # the known one is catalogued

    h = fits.Header({"TELESCOP": "WISE", "BAND": 3, "ERTH2SCX": 4e-5, "ERTH2SCY": 2e-5, "ERTH2SCZ": 1e-5})
    monkeypatch.setattr(ncli, "_review_frames",
                        lambda doc, d: SimpleNamespace(frames=[SimpleNamespace(header=h)] * len(T)))
    assert rv.export_mpc()["problems"] == ["no submitter name", "no measurer names"]
    rv.save_mpc_settings({"submitter": "J. Smith", "observers": ["A. B", " "], "include_likely": False})
    r = rv.export_mpc()
    assert r["objects"] == 1 and r["observations"] == len(T) and not r["problems"]
    text = Path(r["path"]).read_text()
    for line in ("# version=2017", "! mpcCode C51", "! name J. Smith", "! name A. B", "! aperture 0.4",
                 "! design Reflector", "! detector Si:As"):
        assert line in text.splitlines()
    table = [ln for ln in text.splitlines() if not ln.startswith(("#", "!"))]
    cols = [c.strip() for c in table[0].split("|")]
    assert "mag" not in cols and "band" not in cols and "permID" not in cols  # no W3 photometry, no designation
    assert {ln.split("|")[cols.index("trkSub")].strip() for ln in table[1:]} == {R.trksub(by_det[0])}
    listed = list(csv.DictReader(open(Path(r["path"]).with_suffix(".csv"))))
    assert [x["trkSub"] for x in listed] == [R.trksub(by_det[0])]
    assert rv.export_mpc(include_likely=True)["objects"] == 2
    assert R.trksub(1) == "NF000001" and R.trksub(36) == "NF000010"


def test_a_visit_counts_as_fetched_only_with_every_frame(tmp_path):
    fr = [SimpleNamespace(frame_id=f"0404{k}b165", band=3) for k in range(3)]
    for f in fr:
        for kind in ("int", "unc", "msk"):
            (tmp_path / f"{f.frame_id}-w3-{kind}-1b.fits.gz").write_bytes(b"x")
    assert C._have_frames(tmp_path, fr)
    (tmp_path / "04041b165-w3-int-1b.fits.gz").unlink()  # a download cut short
    assert not C._have_frames(tmp_path, fr)


def test_skymap_points_and_png(tmp_path):
    from neowise import skymap
    camp = _campaign(tmp_path)
    d3, _ = _visit(tmp_path, [_det(25.1, 10.3), _det(25.2, 10.3), _det(25.4, 10.3, status="known", known="Koyo",
                                                                            known_id="5591")])
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES (?, 1, 'searching')", (str(d3),))
    camp.record(str(d3), 300.0)
    rv = R.Reviewer(camp.out)
    by_det = {c["det"]: c["id"] for c in rv.candidates("todo")}
    rv.verdict(by_det[0], "confirmed")
    rv.verdict(by_det[1], "likely")
    rv.verdict(by_det[2], "confirmed")
    d = rv.skymap()
    assert sorted((o["verdict"], o["catalogue"]) for o in d["objects"]) == [
        ("confirmed", "catalogued"), ("confirmed", "not catalogued"), ("likely", "not catalogued")]
    assert [o["known_name"] for o in d["objects"] if o["catalogue"] == "catalogued"] == ["Koyo (5591)"]
    assert d["searched"] == [[25.057, 10.405]] and d["lat_max"] == 0.0  # record() marked the visit done
    assert [o["id"] for o in rv.skymap(True, False)["objects"]] == [by_det[0]]
    ra, dec = skymap.ecliptic_line(0.0)
    assert abs(dec).max() < 23.44 + 1e-6 and abs(dec[90] - 23.439) < 0.01  # lambda 90: the solstice
    png = rv.skymap_png(uncatalogued_only=True)
    assert png.startswith(b"\x89PNG")


def test_result_plots(tmp_path, monkeypatch):
    import neodetect.review
    from neowise import plots as P
    asked = []

    def fake_discovery(key, cache=None):  # instead of JPL's Small-Body Database
        asked.append(key)
        return {"fullname": "5591 Koyo (1990 VO)", "date": "2014-Nov-14", "who": "Someone", "location": "Somewhere"}
    monkeypatch.setattr(neodetect.review, "discovery", fake_discovery)
    camp = _campaign(tmp_path)
    fast = _det(25.3, 10.3, rate_deg_day=2.0, pa_deg=0.0)
    d3, _ = _visit(tmp_path, [_det(25.1, 10.3, status="known", known="Koyo", known_id="5591"), fast, _det(25.2, 10.3)])
    doc = json.loads((d3 / "detections.json").read_text())
    doc["known_eval"] = [{"id": "5591", "rate_deg_day": 0.2, "n_frames_covered": 6, "measured_at_truth": 30.0},
                         {"id": "2010 AB", "rate_deg_day": 0.3, "n_frames_covered": 6, "measured_at_truth": 3.0},
                         {"id": "2010 CD", "rate_deg_day": 0.3, "n_frames_covered": 2, "measured_at_truth": 50.0}]
    (d3 / "detections.json").write_text(json.dumps(doc))
    pred = [dict(f, dec_deg=f["dec_deg"] + 1.0 / 3600) for f in doc["detections"][0]["frames"]]  # 1" north
    (d3 / "known.json").write_text(json.dumps({"asteroids": [
        {"id": "5591", "name": "Koyo", "class": "MB>Inner", "err": [0.3] * 6, "dist_au": 1.5, "frames": pred},
        {"id": "2010 AB", "class": "NEA>Apollo", "frames": []}]}))
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES (?, 1, 'searching')", (str(d3),))
    camp.record(str(d3), 300.0)
    rv = R.Reviewer(camp.out)
    by_det = {c["det"]: c["id"] for c in rv.candidates("todo")}
    rv.verdict(by_det[0], "confirmed")
    rv.verdict(by_det[1], "confirmed")
    rv.verdict(by_det[2], "likely")
    d = rv.plots()
    c = d["completeness"]
    assert (c["n"], c["found"]) == (2, 1)  # the one on 2 frames doesn't count
    assert [(b["n"], b["found"]) for b in c["by_snr"]] == [(1, 0), (1, 1)]
    assert [(b["class"], b["n"], b["found"]) for b in c["by_class"]] == [("MB inner", 1, 1)]  # signal >= 10 only
    m = {o["id"]: o for o in d["motion"]["objects"]}
    assert m[by_det[0]]["class"] == "MB inner" and m[by_det[0]]["catalogue"] == "catalogued"
    assert abs(math.hypot(m[by_det[1]]["vl"], m[by_det[1]]["vb"]) - 2.0) < 0.01
    assert [o["id"] for o in d["gallery"]] == [by_det[1], by_det[2]]  # fastest first (no main belt to compare)
    assert [o["id"] for o in rv.plots(include_likely=False)["gallery"]] == [by_det[1]]
    (a,) = d["astrometry"]["objects"]
    assert abs(a["dra"]) < 0.01 and abs(a["ddec"] + 1.0) < 0.01 and a["n_obs"] == 6 and a["eph_err"] == 0.3
    assert (camp.out / P.CACHE_NAME).exists()
    sc = d["scores"]  # all three score 12: one bin, every one reviewed and judged real
    (b,) = sc["bins"]
    assert (b["confirmed"], b["likely"], b["not reviewed"], b["frac"]) == (2, 1, 0, 1.0)
    br = d["brightness"]
    assert sum(br["catalogued"]) == 1 and [o["id"] for o in br["objects"]] == [by_det[1], by_det[2]]
    (g,) = d["galactic"]["bins"]
    assert (g["visits"], g["n_cat"], g["n_new"]) == (1, 1, 2) and 40 <= g["lo"] < g["hi"] <= 60  # b = -51.6
    (v,) = d["ecliptic"]["visits"]
    assert (v["cat"], v["new"], v["crossing"]) == (1, 2, 2)
    assert [[round(a), round(b)] for a, b in d["ecliptic"]["galactic_plane"]] in ([[79, 102], [259, 282]],
                                                                                 [[78, 102], [258, 282]])
    ss = d["solar_system"]
    (o,) = ss["objects"]
    assert o["id"] == by_det[0] and o["group"] == "main belt" and 1.0 < o["r"] < 2.6 and len(ss["context"]) == 1
    pr = d["progress"]
    assert len(pr["visits"]) == 1 and pr["visits"][0][1:] == [3, 1]
    dl = d["discovery"]  # looked up in the background, then from the cache
    if dl["pending"]:
        rv._disc_thread.join(10)
        dl = rv.discovery()
    assert asked == ["5591"] and not rv.discovery()["pending"]
    assert json.loads((camp.out / P.DISCOVERY_CACHE).read_text())["5591"]["date"] == "2014-Nov-14"
    (o,) = dl["objects"]
    assert (dl["n"], dl["before"], o["id"], o["source"], o["discovered"]) == (1, 1, by_det[0], "discovery", "2014-11-14")
    assert sum(dl["discovery"]) == 1 and sum(dl["designation"]) == 0 and dl["edges"][0] == P.LAG_MIN
    for name in P.PLOTS:
        assert rv.plot_png(name).startswith(b"\x89PNG")
    assert rv.plot_png("nope") is None
    pdf = rv.report_pdf()
    assert pdf.startswith(b"%PDF") and len(pdf) > 50_000
    from neowise import report as REP
    heads = [h for h, _ in REP.findings(rv.plots(), C.status_summary(camp.out, db=camp.db), True)]
    assert heads[:2] == ["Detections not catalogued", "Seen before discovery"]
    c = {x["id"]: x for x in rv.candidates("results")}
    a = c[by_det[0]]
    assert a["arc_arcsec"] > 0 and a["span_hours"] > 0  # kept by record()
    assert P.skybot_classes(camp.db, [by_det[0], by_det[1]]) == {by_det[0]: "MB>Inner", by_det[1]: ""}


def test_arc_is_first_to_last_frame_used():
    f = lambda mjd, ra, dec, used=True: {"mjd": mjd, "ra_deg": ra, "dec_deg": dec, "used": used}
    det = {"frames": [f(55000.5, 10.0, 0.0, False), f(55000.0, 10.0, 0.0), f(55000.25, 10.05, 0.0),
                      f(55001.0, 10.1, 0.0)]}
    a, h = C.arc(det)
    assert abs(a - 360.0) < 0.01 and h == 24.0  # 0.1 deg over a day; the unused frame doesn't count
    assert C.arc({"frames": [f(55000.0, 1.0, 1.0)]}) == (None, None)


def test_discovery_dates_from_names_and_designations():
    from datetime import datetime
    from neowise import plots as P
    assert P._known("2002 EH16 (244276)") == ("244276", "2002 EH16")
    assert P._known("(1) Ceres") == ("1", "Ceres")
    assert P._known("2015 DM182") == (None, "2015 DM182")
    assert P.designation_date("2015 DM182") == datetime(2015, 2, 23)  # D: Feb 16-29
    assert P.designation_date("2002 EH16") == datetime(2002, 3, 8)  # E: Mar 1-15
    assert P.designation_date("2010 YA") == datetime(2010, 12, 23)
    assert P.designation_date("6344 P-L") == datetime(1960, 9, 24)
    assert P.designation_date("Ceres") is None
    assert [P.class_label(c) for c in ("MB>Inner", "NEA>Apollo", "KBO>Resonant>3:2", "Mars-Crosser", None)] == \
        ["main belt, inner", "NEA, Apollo", "KBO, resonant 3:2", "Mars-crosser", ""]
    import sqlite3
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE candidates (id INTEGER, known_name TEXT, t_ref REAL, verdict TEXT, mag REAL, score REAL, status TEXT)")
    mjd = lambda y, m, d: (datetime(y, m, d) - datetime(1858, 11, 17)).days
    db.executemany("INSERT INTO candidates VALUES (?,?,?,?,NULL,NULL,'known')", [
        (1, "2015 DM182", mjd(2010, 3, 1), "confirmed"),  # designated 5 years later
        (2, "2010 KS6", mjd(2010, 5, 17), "confirmed"),  # May 16-31: can't tell
        (3, "2002 EH16 (244276)", mjd(2010, 3, 1), "confirmed"),
        (4, "2016 AB1 (999999)", mjd(2010, 3, 1), "likely"),  # not looked up yet
        (5, "2015 DM182", mjd(2010, 7, 1), "confirmed")])  # same object later: the earliest counts
    dl = P.discovery_lag(db, {"244276": {"fullname": "244276 (2002 EH16)", "date": "2002-Mar-06"}})
    assert (dl["n"], dl["before"], dl["unsure"], dl["pending"]) == (3, 1, 1, 1)
    (o,) = dl["objects"]
    assert o["id"] == 1 and o["source"] == "designation" and o["image"] == "2010-03-01"
    assert sum(dl["discovery"]) == 1 and sum(dl["designation"]) == 2
    assert P.discovery_lag(db, {}, include_likely=False)["pending"] == 1


def test_wide_share_splits_search_time_between_bands(tmp_path):
    camp = C.Campaign(tmp_path / "camp", C.Settings(hours="0-24", yield_gpu=False, lat_max=10.0, wide_share=0.25))
    near = C.ecliptic_band_fields(10.0, 5.0)
    C.add_fields(camp.db, near)
    n_wide = C.add_fields(camp.db, C.ecliptic_band_fields(30.0, 5.0))  # only the new ones: beyond 10 deg
    assert n_wide > 0 and camp._count(f"SELECT COUNT(*) FROM fields WHERE {camp._wide()}") == n_wide
    assert camp._count("SELECT MIN(ord) FROM fields WHERE " + camp._wide()) > len(near)  # after the band's
    camp.set_meta("wide_since", 1000.0)
    ids = {w: camp.sql(f"SELECT id FROM fields WHERE ({camp._wide()}) = {w} ORDER BY ord LIMIT 2") for w in (0, 1)}

    def done(fid, seconds, at=2000.0, name=None):
        camp.sql("INSERT INTO visits (dir, field_id, status, seconds, searched_at) VALUES (?,?,?,?,?)",
                 (name or f"v{fid}-{seconds}-{at}", fid, "done", seconds, at))

    assert camp.next_band() == "near"  # nothing yet: 0 < 0.25 * 0 is false
    rows, f = camp.pick()
    assert not rows and f["id"] == ids[0][0]["id"]  # lists the first near field
    done(ids[0][0]["id"], 3000.0)
    done(ids[0][0]["id"], 99999.0, at=10.0)  # before the mix: not counted
    assert camp.next_band() == "wide"
    rows, f = camp.pick()
    assert f["id"] == ids[1][0]["id"] and abs(f["elat"]) > 10  # the first wide field, in its own order
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES ('wl', ?, 'listed')", (ids[1][0]["id"],))
    rows, f = camp.pick()
    assert rows[0]["dir"] == "wl"
    done(ids[1][0]["id"], 900.0)  # 900 / 3900 = 23%: still behind
    assert camp.next_band() == "wide"
    done(ids[1][0]["id"], 200.0)  # 1100 / 4100 = 27%
    assert camp.next_band() == "near"
    camp.sql("INSERT INTO visits (dir, field_id, status) VALUES ('nf', ?, 'fetched')", (ids[0][1]["id"],))
    assert camp.band_seconds()["near"] == 3000.0 + 3000.0  # in flight, at the band's mean
    # a band with nothing left: the other one
    camp.sql(f"UPDATE fields SET status='listed' WHERE NOT ({camp._wide()})")
    rows, f = camp.pick()
    assert rows[0]["dir"] == "wl"
    st = C.status_summary(camp.out, db=camp.db)
    assert "bands" not in st  # the settings aren't saved until run()
    camp.set_meta("settings", {"lat_max": 10.0, "wide_share": 0.25})
    b = C.status_summary(camp.out, db=camp.db)["bands"]
    assert b["wide"]["fields"] == n_wide and b["wide"]["visits_done"] == 2 and b["wide"]["share_since_mix"] == 0.268


def test_worker_takes_fetched_visits_first_in_first_out(tmp_path):
    camp = _campaign(tmp_path)
    C.add_fields(camp.db, [(200.0, 10.0, 0.0, 30.0)])  # a later field (higher ord)
    late, early = camp.sql("SELECT id FROM fields ORDER BY ord DESC")
    camp.sql("INSERT INTO visits (dir, field_id, status, updated) VALUES ('a', ?, 'fetched', '2026-01-01T00:00:01')",
             (late["id"],))
    camp.sql("INSERT INTO visits (dir, field_id, status, updated) VALUES ('b', ?, 'fetched', '2026-01-01T00:00:02')",
             (early["id"],))
    seen = []
    camp.search_visit = lambda k, d, redo: (seen.append(d), camp.stop.set())
    camp.active.set()
    camp.worker(0)
    assert seen == ["a"]  # fetched first, though its field comes later
