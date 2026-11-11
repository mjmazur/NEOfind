"""Blind review page for a campaign's detections: ``neowise reviewer OUT``.

A small local web server (http://127.0.0.1:8765 by default) over
``OUT/campaign.sqlite``. Every detection, catalogued asteroid or not, comes
up in a random order with its blind cutout sheet (``campaign.BLIND_DIR``:
nothing from the catalogue on it), to confirm, call likely or discard.
The catalogue match is withheld from the To review tab, in the page and
the API alike; once a detection has a verdict, the other tabs show it
(which asteroid it is, or that it is not catalogued). Verdicts go
to the database and to the visit's ``review.json`` (``neodetect.review``,
so ``neodetect review --ades`` sees the confirmed ones too).
``OUT/confirmed.csv``, the final list (confirmed and likely, with the
catalogue match), is rewritten after every verdict. "Export for MPC"
writes the confirmed detections not catalogued (and, if asked, the likely
ones) as an ADES PSV file for submission, ``OUT/mpc/neofind_*.psv``, with
a CSV beside it naming each tracklet's detection (``export_mpc``). A
visit's W3 FITS files, whose headers this needs, are kept until all its
detections not catalogued have a verdict, and after that if one is
confirmed or likely. The Sky map and Plots tabs (``skymap``, ``plots``)
show the results. The page also shows the campaign's state and can
pause or resume it.
"""

import csv
import json
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import campaign as C

BLIND = f"sheet LIKE '%/{C.BLIND_DIR}/%'"  # visits searched before the review was blind wait for theirs
VIEWS = {
    "todo": f"verdict IS NULL AND {BLIND}",
    "confirmed": "verdict='confirmed'",
    "likely": "verdict='likely'",
    "notcat": "verdict IN ('confirmed','likely') AND COALESCE(status, '') != 'known'",  # possible discoveries
    "rejected": "verdict='rejected'",
    "results": "verdict IS NOT NULL",
}
VERDICTS = ("confirmed", "likely", "rejected")
SHUFFLE = "(id * 2654435761) % 4294967296"  # a fixed random order, not grouped by visit or score
ORDER = {"todo": SHUFFLE, "results": "reviewed DESC"}
COLUMNS = ("id", "visit_dir", "det", "w4_score", "score", "n_frames", "ra", "dec",
           "t_ref", "rate", "pa", "mag", "verdict", "note", "reviewed",
           "NULLIF(arc_arcsec, -1) arc_arcsec", "NULLIF(span_hours, -1) span_hours")
UNBLINDED = ("status", "near", "gate", "known_name")  # once there is a verdict: not in "todo"
MPC_DIR = "mpc"
MPC_SETTINGS = "mpc.json"
MPC_DEFAULTS = {"submitter": "", "institution": "", "observers": [], "measurers": [], "include_likely": False,
                "comments": ["Archival WISE single-exposure (L1b) W3 and W4 images, 2010",
                             "Found by shift-and-stack, tracklets reviewed by eye"]}


class Reviewer:
    def __init__(self, out):
        self.out = Path(out).resolve()
        if not (self.out / C.DB_NAME).exists():
            raise SystemExit(f"{self.out} has no {C.DB_NAME}: not a campaign directory")
        self.db = C.connect(self.out)
        self.lock = threading.RLock()
        self._disc_thread = None
        self.mpl_lock = threading.Lock()
        C.fill_known_names(self.db)
        C.fill_arcs(self.db)

    def candidates(self, view: str):
        """The detections in ``view``; the catalogue match in all but "todo"."""
        view = view if view in VIEWS else "todo"
        cols = COLUMNS + (UNBLINDED if view != "todo" else ())
        with self.lock:
            rows = self.db.execute(f"SELECT {', '.join(cols)} FROM candidates WHERE {VIEWS[view]} "
                                   f"ORDER BY {ORDER.get(view, 'reviewed DESC')}").fetchall()
        return [dict(r) for r in rows]

    def results(self) -> dict:
        """Verdict against catalogue match, counts (the Results tab)."""
        with self.lock:
            rows = self.db.execute("SELECT COALESCE(verdict, 'unreviewed') v, status, COUNT(*) n FROM candidates "
                                   f"WHERE {BLIND} OR verdict IS NOT NULL GROUP BY v, status").fetchall()
        out = {}
        for r in rows:
            out.setdefault(r["v"], {})[r["status"] or "unchecked"] = r["n"]
        return out

    def state(self) -> dict:
        live = {}
        p = self.out / C.STATUS_NAME
        if p.exists():
            try:
                live = json.loads(p.read_text())
            except ValueError:
                pass
        running = bool(live.get("pid")) and C._is_campaign(live["pid"]) and live.get("state") not in ("stopped", "done")
        with self.lock:
            summary = C.status_summary(self.out, db=self.db, lock=self.lock)
            counts = {v: self.db.execute(f"SELECT COUNT(*) FROM candidates WHERE {w}").fetchone()[0]
                      for v, w in VIEWS.items()}
        counts["waiting"] = self.db.execute(f"SELECT COUNT(*) FROM candidates WHERE NOT ({BLIND})").fetchone()[0]
        return {"running": running, "state": live.get("state") if running else "not running",
                "reason": live.get("reason", "") if running else "", "current": live.get("current", []) if running else [],
                "control": C.read_control(self.out), "summary": summary, "counts": counts, "out": str(self.out)}

    def sheet(self, cid: int):
        with self.lock:
            r = self.db.execute("SELECT sheet FROM candidates WHERE id=?", (cid,)).fetchone()
        p = Path(r[0]) if r and r[0] else None
        return p.read_bytes() if p and p.exists() else None

    def verdict(self, cid: int, verdict, note: str = "") -> dict:
        from neodetect import review
        from neodetect.evaluate import MATCH_TOL_PX
        if verdict not in VERDICTS + (None,):
            raise ValueError(verdict)
        with self.lock:
            r = self.db.execute("SELECT visit_dir, det FROM candidates WHERE id=?", (cid,)).fetchone()
            if r is None:
                raise KeyError(cid)
            vdir = Path(r["visit_dir"])
            doc = json.loads((vdir / "detections.json").read_text())
            verdicts = review.load_verdicts(vdir)
            if verdict is None:  # undo
                d = doc["detections"][r["det"]]
                tol = MATCH_TOL_PX * doc["stats"]["pixel_scale_arcsec"]
                verdicts = [v for v in verdicts if not review._same_object(v, d, tol)]
            else:
                review.set_verdict(verdicts, doc, r["det"], verdict, note)
            review.save_verdicts(vdir, verdicts)
            with self.db:
                self.db.execute("UPDATE candidates SET verdict=?, note=?, reviewed=? WHERE id=?",
                                (verdict, note or None, datetime.now().isoformat(timespec="seconds")
                                 if verdict else None, cid))
            C.settle_fits(self.db, vdir)
            self.write_final_list()
        return {"ok": True}

    def write_final_list(self) -> Path:
        """``confirmed.csv``: the confirmed and likely detections, with their
        catalogue match ("suspect": nothing catalogued there)."""
        rows = [c for c in self.candidates("results") if c["verdict"] in ("confirmed", "likely")]
        rows.sort(key=lambda c: (c["verdict"] != "confirmed", c["status"] == "known", c["ra"]))
        path = self.out / "confirmed.csv"
        tmp = path.with_suffix(".csv.part")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["verdict", "catalogue", "near", "ra_deg", "dec_deg", "t_ref_mjd", "rate_deg_day", "pa_deg",
                        "arc_arcsec", "span_hours", "w3_mag", "score", "w4_score", "gate", "n_frames", "note", "reviewed", "visit_dir",
                        "detection"])
            for c in rows:
                w.writerow([c["verdict"], c["status"], c["near"] or "",
                            f"{c['ra']:.5f}", f"{c['dec']:+.5f}", f"{c['t_ref']:.5f}", f"{c['rate']:.4f}",
                            f"{c['pa']:.1f}", "" if c["arc_arcsec"] is None else f"{c['arc_arcsec']:.1f}",
                            "" if c["span_hours"] is None else f"{c['span_hours']:.3f}", "" if c["mag"] is None else f"{c['mag']:.2f}", f"{c['score']:.1f}",
                            "" if c["w4_score"] is None else f"{c['w4_score']:.1f}", c["gate"], c["n_frames"],
                            c["note"] or "", c["reviewed"], c["visit_dir"], c["det"]])
        tmp.replace(path)
        return path

    # ---- sky map
    def skymap(self, uncatalogued_only=False, include_likely=True) -> dict:
        from .skymap import points
        with self.lock:
            return points(self.db, uncatalogued_only, include_likely)

    def skymap_png(self, uncatalogued_only=False, include_likely=True) -> bytes:
        import tempfile
        from .skymap import write_skymap
        with tempfile.TemporaryDirectory() as tmp:
            return write_skymap(self.out, Path(tmp) / "skymap.png", uncatalogued_only, include_likely).read_bytes()

    # ---- result plots
    def plots(self, include_likely=True) -> dict:
        from . import plots as P
        with self.lock:
            d = P.data(self.out, self.db, include_likely)
        if d["discovery"]["pending"]:
            self.fetch_discoveries()
        d["discovery"]["fetching"] = self._disc_thread is not None and self._disc_thread.is_alive()
        return d

    def fetch_discoveries(self):
        """Look up the discovery dates not cached yet, in the background
        (JPL's Small-Body Database, one at a time: a few minutes for a
        thousand); the Plots tab polls ``discovery()`` meanwhile."""
        from . import plots as P
        if self._disc_thread is not None and self._disc_thread.is_alive():
            return
        self._disc_thread = threading.Thread(target=P.fetch_discoveries, args=(self.out,), daemon=True)
        self._disc_thread.start()

    def discovery(self, include_likely=True) -> dict:
        from . import plots as P
        with self.lock:
            d = P.discovery_lag(self.db, P.discovery_cache(self.out), include_likely)
        d["fetching"] = self._disc_thread is not None and self._disc_thread.is_alive()
        return d

    def plot_png(self, name: str, include_likely=True):
        import tempfile
        from . import plots as P
        if name not in P.PLOTS:
            return None
        with self.lock:
            d = P.data(self.out, self.db, include_likely)
            with self.mpl_lock, tempfile.TemporaryDirectory() as tmp:  # pyplot isn't thread-safe
                return P.render(name, d, self.db, Path(tmp) / f"{name}.png").read_bytes()

    def report_pdf(self, include_likely=True) -> bytes:
        """The PDF report (``report.py``), made with its own database
        connection so the review page carries on meanwhile (~10 s)."""
        import tempfile
        from . import report as R
        with self.mpl_lock, tempfile.TemporaryDirectory() as tmp:
            path = R.write_report(self.out, Path(tmp) / "report.pdf", include_likely, lookup=False)
            data = path.read_bytes()
        if R.P.discovery_lag(self.db, R.P.discovery_cache(self.out), include_likely)["pending"]:
            self.fetch_discoveries()  # for the next one
        return data

    # ---- MPC submission
    def mpc_settings(self) -> dict:
        """Who to name in the ADES header, kept in ``OUT/mpc.json``."""
        try:
            saved = json.loads((self.out / MPC_SETTINGS).read_text())
        except (OSError, ValueError):
            saved = {}
        return {**MPC_DEFAULTS, **saved}

    def save_mpc_settings(self, body: dict) -> dict:
        st = self.mpc_settings()
        for k in MPC_DEFAULTS:
            if k in body:
                v = body[k]
                st[k] = [n.strip() for n in v if n.strip()] if isinstance(MPC_DEFAULTS[k], list) else v
        (self.out / MPC_SETTINGS).write_text(json.dumps(st, indent=1))
        return st

    def export_mpc(self, include_likely: bool = None) -> dict:
        """The confirmed detections not catalogued (and the likely ones,
        with ``include_likely``) as an ADES PSV file to submit to the MPC:
        one tracklet each, its trkSub ``NF`` and the detection's id in base
        36 (the same in every export), the spacecraft's position from the
        frame headers, no photometry (W3/W4 aren't ADES bands)."""
        from neodetect.cli import _review_frames
        from neodetect.report import ades_rows, check_ades, write_ades_psv
        st = self.mpc_settings()
        if include_likely is None:
            include_likely = st["include_likely"]
        wanted = ("confirmed", "likely") if include_likely else ("confirmed",)
        cands = [c for c in self.candidates("notcat") if c["verdict"] in wanted]
        by_visit = {}
        for c in cands:
            by_visit.setdefault(c["visit_dir"], {})[c["det"]] = c
        rows, listed, skipped = [], [], []
        for vdir, dets in sorted(by_visit.items()):
            doc = json.loads((Path(vdir) / "detections.json").read_text())
            try:
                seq = _review_frames(doc, Path(vdir))
            except (Exception, SystemExit) as e:  # FITS files deleted or moved
                skipped.append(f"{vdir}: {e}")
                continue
            heads = [f.header for f in seq.frames]
            for det, c in sorted(dets.items()):
                one = dict(doc, detections=[doc["detections"][det]])
                trk = trksub(c["id"])
                new = [dict(r, trkSub=trk, permID="", provID="", mag="", rmsMag="", band="")
                       for r in ades_rows(one, heads)]
                rows += new
                listed.append({"trkSub": trk, "id": c["id"], "verdict": c["verdict"], "catalogue": c["status"],
                               "near": c["near"] or "", "observations": len(new), "ra_deg": c["ra"],
                               "dec_deg": c["dec"], "t_ref_mjd": c["t_ref"], "rate_deg_day": c["rate"],
                               "note": c["note"] or "", "visit_dir": vdir, "det": det})
        submitter = st["submitter"].strip()
        measurers = st["measurers"] or ([submitter] if submitter else [])
        problems = check_ades(rows, submitter, measurers) if rows else []
        out = {"path": None, "observations": len(rows), "objects": len(listed), "skipped": skipped,
               "problems": problems}
        if not rows or problems:
            return out
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        d = self.out / MPC_DIR
        d.mkdir(exist_ok=True)
        path = write_ades_psv(rows, d / f"neofind_C51_{stamp}.psv", observatory="C51", submitter=submitter,
                              institution=st["institution"].strip(), observers=st["observers"],
                              measurers=measurers, software={"objectDetection": "NEOfind shift-and-stack"},
                              comments=[c for c in st["comments"] if c])
        with open(path.with_suffix(".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(listed[0]))
            w.writeheader()
            w.writerows(listed)
        out["path"] = str(path)
        out["file"] = path.name
        return out

    def mpc_file(self, name: str):
        if not re.fullmatch(r"neofind_C51_\d{8}T\d{6}\.(psv|csv)", name):
            return None
        p = self.out / MPC_DIR / name
        return p.read_bytes() if p.exists() else None


def trksub(cid: int) -> str:
    """``NF`` and the detection's id in base 36: 8 characters, unique in the campaign."""
    digits, s, n = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", "", int(cid)
    while True:
        n, r = divmod(n, 36)
        s = digits[r] + s
        if not n:
            break
    return "NF" + s.rjust(6, "0")


def make_handler(rv: Reviewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._send(200, rv.state())
            if u.path == "/api/candidates":
                return self._send(200, rv.candidates(q.get("view", ["todo"])[0]))
            if u.path == "/api/results":
                return self._send(200, rv.results())
            if u.path in ("/api/skymap", "/skymap.png"):
                only = q.get("uncatalogued", ["0"])[0] == "1"
                likely = q.get("likely", ["1"])[0] == "1"
                if u.path == "/api/skymap":
                    return self._send(200, rv.skymap(only, likely))
                return self._send(200, rv.skymap_png(only, likely), "image/png")
            if u.path == "/api/discovery":
                return self._send(200, rv.discovery(q.get("likely", ["1"])[0] == "1"))
            if u.path == "/api/plots":
                return self._send(200, rv.plots(q.get("likely", ["1"])[0] == "1"))
            if u.path.startswith("/plots/") and u.path.endswith(".png"):
                png = rv.plot_png(u.path[len("/plots/"):-4], q.get("likely", ["1"])[0] == "1")
                return self._send(200, png, "image/png") if png else self._send(404, {"error": "no such plot"})
            if u.path == "/api/mpc":
                return self._send(200, rv.mpc_settings())
            if u.path.startswith("/mpc/"):
                name = u.path.split("/")[2]
                data = rv.mpc_file(name)
                if data is None:
                    return self._send(404, {"error": "no such file"})
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if u.path == "/report.pdf":
                data = rv.report_pdf(q.get("likely", ["1"])[0] == "1")
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="neofind_report_{datetime.now():%Y%m%d_%H%M}.pdf"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if u.path.startswith("/sheet/"):
                png = rv.sheet(int(u.path.split("/")[2].split(".")[0]))
                return self._send(200, png, "image/png") if png else self._send(404, {"error": "no sheet"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/verdict":
                    return self._send(200, rv.verdict(int(body["id"]), body.get("verdict"), body.get("note", "")))
                if u.path == "/api/control":
                    a = body.get("action")
                    if a == "pause":
                        C.set_control(rv.out, paused=True, reason="paused from the review page")
                    elif a == "resume":
                        C.set_control(rv.out, paused=False, reason="")
                    else:
                        return self._send(400, {"error": f"unknown action {a}"})
                    return self._send(200, {"ok": True})
                if u.path == "/api/export":
                    rv.save_mpc_settings(body)
                    return self._send(200, rv.export_mpc())
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(404, {"error": "not found"})
    return Handler


def serve(out, host="127.0.0.1", port=8765):
    rv = Reviewer(out)
    rv.write_final_list()
    srv = ThreadingHTTPServer((host, port), make_handler(rv))
    print(f"reviewing {rv.out}: open http://{host}:{port}/  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEOfind blind review</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--mute:#5d6678;--line:#dde1e8;--accent:#2f6fde;
 --ok:#1f8a4c;--bad:#c23b3b;--warn:#b7791f;--chip:#eef1f6;--cat:#2a78d6;--new:#eb6834;--aqua:#1baf7a;--field:#d6d4cc;--idle:rgba(137,135,129,.13);--idle2:rgba(137,135,129,.26);color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--panel:#1a1f27;--ink:#e6e9ef;--mute:#9aa3b5;
 --line:#2c3340;--accent:#6c9cff;--ok:#4cc38a;--bad:#ff6b6b;--warn:#e0a84a;--chip:#232a35;--cat:#3987e5;--new:#d95926;--aqua:#199e70;--field:#3a4250;--idle:rgba(137,135,129,.12);--idle2:rgba(137,135,129,.24);color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--panel);border-bottom:1px solid var(--line);
 padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
h1{font-size:16px;margin:0;font-weight:650}
.state{display:flex;gap:8px;align-items:center;color:var(--mute)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mute)}
.dot.running{background:var(--ok)}.dot.paused,.dot.yielding{background:var(--warn)}
.tabs{display:flex;gap:4px;flex-wrap:wrap}
.tabs button{border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:999px;
 padding:4px 12px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button.act{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;
 padding:6px 12px;cursor:pointer;font:inherit}
button.act:hover{border-color:var(--accent)}
.spacer{flex:1}
#sky{padding:16px}#sky[hidden],main[hidden]{display:none}
#sky .bar{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
#sky .bar label{display:flex;gap:6px;align-items:center;cursor:pointer}
#sky .leg{display:flex;gap:16px;flex-wrap:wrap;color:var(--mute);font-size:12px;align-items:center}
#sky .leg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
#skyWrap{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px}
#skySvg{width:100%;height:auto;display:block}
#skySvg .grid{fill:none;stroke:var(--line);stroke-width:1}
#skySvg .ecl{fill:none;stroke:var(--mute);stroke-width:1.5}#skySvg .band{fill:none;stroke:var(--mute);stroke-width:1;stroke-dasharray:3 4}
#skySvg .lab{fill:var(--mute);font-size:11px}
#skySvg .pt{cursor:pointer}#skySvg .pt:hover .dot{stroke-width:3}
#skyTip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:6px 9px;
 border-radius:7px;white-space:nowrap;display:none;z-index:3}
#skyTable{margin-top:12px}#skyTable table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
#skyTable th,#skyTable td{border-bottom:1px solid var(--line);padding:3px 10px;text-align:left}
#plots{padding:16px;position:relative}#plots[hidden]{display:none}
#plots .bar{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
#plots .bar label{display:flex;gap:6px;align-items:center;cursor:pointer}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:16px}
.card h2{font-size:15px;margin:0 0 2px}.card .hint{margin-top:2px}
.card .charts{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}
.card .charts>div{flex:1 1 380px;min-width:0}.card .charts h3{font-size:12px;font-weight:600;margin:0 0 4px;color:var(--mute)}
.card a.act{text-decoration:none;font-size:12px;padding:4px 10px;border:1px solid var(--line);border-radius:8px;color:var(--ink);background:var(--panel)}
.card a.act:hover{border-color:var(--accent)}
.leg{display:flex;gap:16px;flex-wrap:wrap;color:var(--mute);font-size:12px;align-items:center;margin-top:6px}
.leg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
svg.chart .grid{stroke:var(--line);stroke-width:1}svg.chart .axis{stroke:var(--mute);stroke-width:1}
svg.chart .lab{fill:var(--mute);font-size:11px}svg.chart .ttl{fill:var(--ink);font-size:11px}
svg.chart .ref{fill:none;stroke:var(--mute);stroke-width:1.2;stroke-dasharray:4 4}
svg.chart [data-k]{cursor:default}svg.chart .click{cursor:pointer}
#plotTip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:6px 9px;
 border-radius:7px;white-space:nowrap;display:none;z-index:3}
.gal{display:grid;gap:12px;margin-top:10px}
.gcard{border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.gcard:hover{border-color:var(--accent)}
.gcard .gh{padding:6px 10px;display:flex;gap:6px 14px;flex-wrap:wrap;font-size:13px;font-variant-numeric:tabular-nums;align-items:baseline}
.gcard .gh b{font-size:14px}.gcard .gh .sig{color:var(--ink);font-weight:600}
.gcard .sheet{background:none;border-radius:0}.gcard .sheet img{max-width:100%;height:auto}
#plots nav{display:flex;gap:6px 14px;flex-wrap:wrap;font-size:13px}#plots nav a{color:var(--accent);text-decoration:none}
#plots h2.group{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:22px 0 8px}
.mult h3{margin:6px 0 0!important}
#plots table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
#plots td,#plots th{border-bottom:1px solid var(--line);padding:3px 10px;text-align:right}
main{display:grid;grid-template-columns:minmax(220px,300px) 1fr;gap:16px;padding:16px;align-items:start}
@media (max-width:800px){main{grid-template-columns:1fr}}
#list{background:var(--panel);border:1px solid var(--line);border-radius:10px;max-height:calc(100vh - 110px);
 overflow:auto}
.item{padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:grid;
 grid-template-columns:1fr auto;gap:2px 8px}
.item:hover{background:var(--chip)}.item.sel{background:var(--chip);box-shadow:inset 3px 0 var(--accent)}
.item .t{font-weight:600;font-variant-numeric:tabular-nums}.item .s{color:var(--mute);font-size:12px}
.item .cat{grid-column:1/-1;color:var(--ink)}
.pill{font-size:11px;padding:1px 7px;border-radius:999px;background:var(--chip);color:var(--mute);
 white-space:nowrap;align-self:start}
.pill.pass{color:var(--ok)}.pill.no-w4{color:var(--warn)}.pill.fail{color:var(--bad)}
.pill.confirmed{background:var(--ok);color:#fff}.pill.rejected{background:var(--bad);color:#fff}
.pill.likely{background:var(--warn);color:#fff}
table.res{border-collapse:collapse;margin:4px 0 16px;font-variant-numeric:tabular-nums}
table.res th,table.res td{border:1px solid var(--line);padding:4px 10px;text-align:right}
table.res th{color:var(--mute);font-weight:600}table.res td:first-child,table.res th:first-child{text-align:left}
#detail{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}
.sheet{overflow-x:auto;background:#000;border-radius:6px}
.sheet img{display:block;image-rendering:pixelated;max-width:none}
dl{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px 16px;margin:14px 0}
dt{color:var(--mute);font-size:12px}dd{margin:0;font-variant-numeric:tabular-nums}
.verdict{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:8px}
.big{font-size:15px;padding:9px 18px;border-radius:9px;border:0;cursor:pointer;color:#fff;font-weight:600}
.big.ok{background:var(--ok)}.big.mid{background:var(--warn)}.big.bad{background:var(--bad)}.big.skip{background:var(--mute)}
input[type=text]{flex:1;min-width:180px;padding:8px 10px;border-radius:8px;border:1px solid var(--line);
 background:var(--bg);color:var(--ink);font:inherit}
.hint{color:var(--mute);font-size:12px;margin-top:10px}
.empty{color:var(--mute);padding:30px;text-align:center}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--ink);color:var(--bg);
 padding:8px 14px;border-radius:8px;opacity:0;transition:opacity .2s;pointer-events:none;max-width:90vw}
#toast.show{opacity:1}
dialog{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:12px;padding:18px 20px;
 width:min(560px,calc(100vw - 32px))}
dialog::backdrop{background:rgba(0,0,0,.45)}
dialog h2{margin:0 0 4px;font-size:17px}
dialog label{display:block;margin:10px 0 3px;font-size:13px;color:var(--mute)}
dialog input[type=text],dialog textarea{width:100%;box-sizing:border-box;padding:7px 9px;border-radius:8px;
 border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
dialog .row{display:flex;gap:10px;justify-content:flex-end;margin-top:14px;flex-wrap:wrap}
#mpcOut{margin-top:10px;font-size:13px}#mpcOut .bad{color:var(--bad)}#mpcOut a{color:var(--accent)}
.zoom{display:flex;gap:6px;align-items:center;color:var(--mute);font-size:12px;margin:8px 0}
</style></head><body>
<header>
 <h1>NEOfind blind review</h1>
 <div class="state"><span class="dot" id="dot"></span><span id="stateText">…</span></div>
 <button class="act" id="pauseBtn" hidden></button>
 <div class="spacer"></div>
 <div class="tabs" id="tabs">
  <button data-v="todo" class="on">To review <span></span></button>
  <button data-v="confirmed">Confirmed <span></span></button>
  <button data-v="likely">Likely <span></span></button>
  <button data-v="notcat" title="Confirmed or likely, and no catalogued asteroid there: possible discoveries">Not catalogued <span></span></button>
  <button data-v="rejected">Discarded <span></span></button>
  <button data-v="results" title="Every detection with a verdict, and the verdicts against the catalogue">Results <span></span></button>
  <button data-v="skymap" title="Where the confirmed (and likely) asteroids are on the sky">Sky map <span></span></button>
  <button data-v="plots" title="Completeness, motion, the detections not catalogued, and astrometry">Plots <span></span></button>
 </div>
 <a class="act" id="reportBtn" href="/report.pdf?likely=1" download title="A PDF of the results and the significant findings: summary, the detections not catalogued, every plot, the asteroids seen before discovery">PDF report</a>
 <button class="act" id="exportBtn" title="ADES PSV file of the confirmed detections not catalogued, to submit to the MPC">Export for MPC</button>
</header>
<main>
 <div id="list"></div>
 <section id="detail"><div class="empty">Loading…</div></section>
</main>
<section id="sky" hidden>
 <div class="bar">
  <label><input type="checkbox" id="skyNew"> Only not catalogued</label>
  <label><input type="checkbox" id="skyLikely" checked> Include likely</label>
  <span class="leg"><span><i style="background:var(--new)"></i>not catalogued</span><span><i style="background:var(--cat)"></i>catalogued</span>
   <span><i style="border:2px solid var(--mute);box-sizing:border-box"></i>ring: likely</span><span><i style="background:var(--field);border-radius:2px"></i>field searched</span>
   <span>— ecliptic, ┄ edge of the band</span></span>
  <span id="skyCount" class="hint" style="margin:0"></span>
  <a class="act" id="skyPng" download>Download PNG</a>
 </div>
 <div id="skyWrap"><svg id="skySvg" viewBox="0 0 1000 520" role="img" aria-label="Sky map of the asteroid detections"></svg><div id="skyTip"></div></div>
 <details id="skyTable"><summary>Table</summary><div></div></details>
</section>
<section id="plots" hidden>
 <div class="bar">
  <label><input type="checkbox" id="plLikely" checked> Include likely</label>
  <span id="plCount" class="hint" style="margin:0"></span>
 </div>
 <nav>Results: <a href="#pl-completeness">Completeness</a><a href="#pl-motion">Motion</a><a href="#pl-gallery">Not catalogued</a>
  <a href="#pl-astrometry">Astrometry</a><a href="#pl-discovery">Before discovery</a> · Checks: <a href="#pl-scores">Scores</a><a href="#pl-brightness">Brightness</a>
  <a href="#pl-galactic">Galactic latitude</a><a href="#pl-ecliptic">Along the ecliptic</a><a href="#pl-solar">Solar system</a><a href="#pl-progress">Progress</a></nav>
 <h2 class="group">Results</h2>
 <div class="card" id="pl-completeness"><h2>Completeness</h2>
  <div class="hint">Of the catalogued asteroids that crossed a searched field on at least <span class="minfr">3</span> frames, the fraction
   found: against the signal the detector measures on the asteroid's predicted track (what a perfect search would score), then, for those
   with a signal of at least <span class="floor">10</span>, by rate and by orbit class. Bars: 68% interval.</div>
  <div class="charts"><div><h3 id="cTitle"></h3><svg class="chart" id="cSnr" viewBox="0 0 440 290"></svg></div>
   <div><h3>By rate</h3><svg class="chart" id="cRate" viewBox="0 0 440 290"></svg></div>
   <div><h3>By orbit class</h3><svg class="chart" id="cClass" viewBox="0 0 440 290"></svg></div></div>
  <details><summary>Table</summary><div id="cTable"></div></details>
  <div class="leg"><a class="act" data-png="completeness" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-motion"><h2>Motion</h2>
  <div class="hint">Each confirmed or likely detection's motion along and across the ecliptic. Catalogued main-belt asteroids form the
   clump (dashed: 5σ of it); a detection not catalogued far from it doesn't move like a main-belt asteroid. Hover for details, click to open.</div>
  <div class="leg"><span><i style="background:var(--new)"></i>not catalogued</span><span><i style="background:var(--cat)"></i>catalogued</span>
   <span><i style="border:2px solid var(--mute);box-sizing:border-box"></i>ring: likely</span><span>┄ 5σ of the main belt</span></div>
  <div class="charts"><div><h3>All</h3><svg class="chart" id="mAll" viewBox="0 0 520 420"></svg></div>
   <div><h3>Close up of the main belt</h3><svg class="chart" id="mZoom" viewBox="0 0 520 420"></svg></div></div>
  <div class="leg"><a class="act" data-png="motion" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-gallery"><h2>Not catalogued</h2>
  <div class="hint">The confirmed and likely detections with no catalogued asteroid there, the most unusual motion first
   (σ: distance from the main-belt clump above). Click one to open it.</div>
  <div class="gal" id="gal"></div>
  <div class="leg"><a class="act" data-png="gallery" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-astrometry"><h2>Astrometry</h2>
  <div class="hint" id="aStats"></div>
  <div class="leg"><span><i style="background:var(--field)"></i>each observation</span><span><i style="background:var(--cat)"></i>each detection (median of its frames)</span>
   <span>┄ 1 pixel</span></div>
  <div class="charts"><div><h3>Measured − catalogue position (arcsec; east to the left)</h3><svg class="chart" id="aSky" viewBox="0 0 460 440"></svg></div>
   <div><h3>Each detection's offset against its track score</h3><svg class="chart" id="aScore" viewBox="0 0 460 440"></svg></div></div>
  <div class="leg"><a class="act" data-png="astrometry" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-discovery"><h2>Before discovery</h2>
  <div class="hint">For each catalogued asteroid confirmed<span class="lk"> or likely</span>: its discovery date minus the date of the earliest WISE image
   it was found on. Right of the line, WISE saw it before anyone discovered it. Numbered asteroids use their credited discovery (JPL's Small-Body
   Database); unnumbered ones have none, so the date their provisional designation encodes (to a half month) stands in. <b id="dSum"></b></div>
  <div class="leg"><span><i style="background:var(--cat);border-radius:2px"></i>credited discovery date</span>
   <span><i style="background:var(--aqua);border-radius:2px"></i>no credited discovery (unnumbered): provisional designation date</span><span id="dNote"></span></div>
  <svg class="chart" id="dHist" viewBox="0 0 1000 340"></svg>
  <details><summary>Seen before discovery</summary><div id="dTable"></div></details>
  <div class="leg"><a class="act" data-png="discovery" download>Download PNG</a></div>
 </div>
 <h2 class="group">Checks on the search</h2>
 <div class="card" id="pl-scores"><h2>Scores</h2>
  <div class="hint">Detections by track score and verdict, and per score the fraction of those reviewed judged real (confirmed or likely).
   The review goes in a random order, so the reviewed are a fair sample: <b id="sExp"></b></div>
  <div class="charts"><div class="mult"><h3>Confirmed</h3><svg class="chart" id="sR0" viewBox="0 0 640 120"></svg>
    <h3>Likely</h3><svg class="chart" id="sR1" viewBox="0 0 640 120"></svg><h3>Discarded</h3><svg class="chart" id="sR2" viewBox="0 0 640 120"></svg>
    <h3>Not reviewed</h3><svg class="chart" id="sR3" viewBox="0 0 640 150"></svg></div>
   <div><h3>Judged real, of those reviewed</h3><svg class="chart" id="sFrac" viewBox="0 0 460 420"></svg></div></div>
  <div class="leg"><a class="act" data-png="scores" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-brightness"><h2>Brightness</h2>
  <div class="hint">W3 magnitudes of the confirmed and likely detections: the catalogued as a histogram, each one not catalogued as a dot
   (click to open). Dashed: the medians.</div>
  <div class="leg"><span><i style="background:var(--cat);border-radius:2px"></i>catalogued</span><span><i style="background:var(--new)"></i>not catalogued</span><span id="bMed"></span></div>
  <svg class="chart" id="bHist" viewBox="0 0 1000 340"></svg>
  <div class="leg"><a class="act" data-png="brightness" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-galactic"><h2>Galactic latitude</h2>
  <div class="hint">Detections per visit by the field's galactic latitude, every detection whatever its verdict. Crowded star fields near the
   plane make false tracks: more detections not catalogued there, fewer catalogued ones found.</div>
  <div class="charts"><div><h3>Catalogued asteroids found per visit</h3><svg class="chart" id="gCat" viewBox="0 0 460 280"></svg></div>
   <div><h3>Detections not catalogued per visit</h3><svg class="chart" id="gNew" viewBox="0 0 460 280"></svg></div></div>
  <div class="leg"><a class="act" data-png="galactic" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-ecliptic"><h2>Along the ecliptic</h2>
  <div class="hint">Each visit by its field's ecliptic longitude. Shaded: where the ecliptic is within 10° of the galactic plane.</div>
  <div class="mult"><h3>Catalogued asteroids on the field (on ≥3 frames)</h3><svg class="chart" id="eCross" viewBox="0 0 1000 170"></svg>
   <h3>Catalogued asteroids found</h3><svg class="chart" id="eCat" viewBox="0 0 1000 170"></svg>
   <h3>Detections not catalogued</h3><svg class="chart" id="eNew" viewBox="0 0 1000 190"></svg></div>
  <div class="leg"><a class="act" data-png="ecliptic" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-solar"><h2>Solar system</h2>
  <div class="hint">Where the catalogued confirmed and likely detections' asteroids were, seen from ecliptic north (x toward the vernal equinox),
   from their catalogued distances; grey, every catalogued asteroid on a searched field. Black: the Earth, and WISE with it, during the visits.</div>
  <div class="leg"><span><i style="background:var(--cat)"></i>main belt</span><span><i style="background:var(--aqua)"></i>Cybele, Hilda, Trojan</span>
   <span><i style="background:var(--new)"></i>NEA, Mars-crosser</span><span><i style="background:var(--field)"></i>on a searched field</span></div>
  <svg class="chart" id="solar" viewBox="0 0 640 640" style="max-width:720px"></svg>
  <div class="leg"><a class="act" data-png="solar_system" download>Download PNG</a></div>
 </div>
 <div class="card" id="pl-progress"><h2>Progress</h2>
  <div class="hint">From the campaign's log, in your local time: visits searched and their detections. Shaded: the campaign wasn't searching
   (hover for why). A stretch running but flat is the campaign retrying a download or redoing visits' sheets.</div>
  <div class="mult"><h3 id="pV"></h3><svg class="chart" id="prV" viewBox="0 0 1000 170"></svg><h3 id="pC"></h3><svg class="chart" id="prC" viewBox="0 0 1000 170"></svg>
   <h3 id="pN"></h3><svg class="chart" id="prN" viewBox="0 0 1000 190"></svg></div>
  <div class="leg"><a class="act" data-png="progress" download>Download PNG</a></div>
 </div>
 <div id="plotTip"></div>
</section>
<div id="toast"></div>
<dialog id="mpcDlg"><form method="dialog" id="mpcForm">
 <h2>Export for the MPC</h2>
 <div class="hint">An ADES PSV file (station C51, WISE) of the confirmed detections that are not catalogued, one tracklet each,
  to submit at minorplanetcenter.net (it can be checked first with their PSV test page). The names go in the file's header.</div>
 <label for="mSub">Submitter name (required, e.g. J. Smith)</label><input type="text" id="mSub">
 <label for="mInst">Institution</label><input type="text" id="mInst">
 <label for="mMeas">Measurers, one per line (default: the submitter)</label><textarea id="mMeas" rows="2"></textarea>
 <label for="mObs">Observers, one per line (optional)</label><textarea id="mObs" rows="2"></textarea>
 <label for="mCom">Comment lines</label><textarea id="mCom" rows="2"></textarea>
 <label><input type="checkbox" id="mLikely"> Include the ones marked likely</label>
 <div id="mpcOut"></div>
 <div class="row"><button class="act" value="cancel">Close</button><button class="big ok" id="mpcGo" type="button">Write file</button></div>
</form></dialog>
<script>
let view="todo", items=[], sel=0, zoom=1;
const VNAME={confirmed:"confirmed",likely:"likely",rejected:"discarded"};
const $=s=>document.querySelector(s);
const fmt=(x,d)=>x==null?"–":Number(x).toFixed(d);
// arc (first to last frame used) and the time it spans: 1871″ -> 31.2′
const fmtArc=a=>a==null?"–":a>=60?`${(a/60).toFixed(1)}′`:`${a.toFixed(1)}″`;
const fmtSpan=h=>h==null?"–":h>=48?`${(h/24).toFixed(1)} days`:`${h.toFixed(1)} h`;
const arcSpan=c=>c.arc_arcsec==null?"–":`${fmtArc(c.arc_arcsec)} over ${fmtSpan(c.span_hours)}`;
function toast(m){const t=$("#toast");t.textContent=m;t.classList.add("show");clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove("show"),2600)}
async function api(path,body){const r=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});
 const j=await r.json(); if(!r.ok) throw new Error(j.error||r.status); return j}
function catLabel(c){return c.status==="known"?(c.known_name||"catalogued asteroid"):c.status==="near-known"?"near "+(c.near||"a catalogued asteroid"):"not catalogued"}
function w4Label(c){return c.w4_score==null?"no W4":`W4 ${fmt(c.w4_score,1)}`}
async function refreshState(){
 try{const s=await api("/api/state");
  const st=s.state, dot=$("#dot"); dot.className="dot "+st.replace(/ /g,"-");
  let txt=st==="not running"?"Campaign not running":st[0].toUpperCase()+st.slice(1)+(s.reason?` — ${s.reason}`:"");
  const v=s.summary.visits||{}; txt+=` · ${(v.done||0)} visits searched`;
  if(s.summary.projected_days!=null) txt+=` · ~${s.summary.projected_days} days left`;
  else if(s.summary.projected_active_hours!=null) txt+=` · ~${s.summary.projected_active_hours} active hours left`;
  if(s.counts.waiting) txt+=` · ${s.counts.waiting} detections waiting for blind sheets`;
  $("#stateText").textContent=txt;
  const b=$("#pauseBtn"); const paused=!!(s.control&&s.control.paused);
  b.hidden=false; b.textContent=paused?"Resume campaign":"Pause campaign"; b.dataset.a=paused?"resume":"pause";
  document.querySelectorAll("#tabs button").forEach(x=>x.querySelector("span").textContent=s.counts[x.dataset.v]==null?"":`(${s.counts[x.dataset.v]})`);
 }catch(e){$("#stateText").textContent="Review server unreachable"}
}
async function load(keep){
 const cur=items[sel]&&items[sel].id;
 items=await api("/api/candidates?view="+view);
 if(keep&&cur!=null){const i=items.findIndex(x=>x.id===cur); sel=i>=0?i:Math.min(sel,items.length-1)} else sel=Math.min(sel,Math.max(0,items.length-1));
 renderList(); renderDetail(); refreshState();
}
function renderList(){
 const L=$("#list");
 if(!items.length){L.innerHTML=`<div class="empty">${view==="todo"?"Nothing left to review.":"None yet."}</div>`;return}
 L.innerHTML=items.map((c,i)=>`<div class="item${i===sel?" sel":""}" data-i="${i}">
  <span class="t">${fmt(c.ra,4)} ${c.dec>=0?"+":""}${fmt(c.dec,4)}</span>
  <span class="pill ${c.verdict||""}">${c.verdict?VNAME[c.verdict]:w4Label(c)}</span>
  <span class="s">score ${fmt(c.score,1)} · ${c.n_frames} fr · ${fmt(c.rate,3)}°/d</span>${view!=="todo"?`<span class="s cat">${catLabel(c)}</span>`:""}</div>`).join("");
 L.querySelectorAll(".item").forEach(e=>e.onclick=()=>{sel=+e.dataset.i;renderList();renderDetail()});
 const s=L.querySelector(".sel"); s&&s.scrollIntoView({block:"nearest"});
}
let resTable="";
async function loadResults(){
 const r=await api("/api/results"), cols=["known","near-known","suspect","unchecked"], rowsV=["confirmed","likely","rejected","unreviewed"];
 resTable=`<table class="res"><tr><th>Verdict \\ catalogue</th>${cols.map(c=>`<th>${c==="suspect"?"not catalogued":c}</th>`).join("")}</tr>`+
  rowsV.map(v=>`<tr><td>${VNAME[v]||v}</td>${cols.map(c=>`<td>${(r[v]||{})[c]||0}</td>`).join("")}</tr>`).join("")+"</table>";
}
function renderDetail(){
 const D=$("#detail"), c=items[sel];
 const head=view==="results"?resTable:"";
 if(!c){D.innerHTML=head+`<div class="empty">${view==="todo"?"Nothing left to review. New detections appear as the campaign searches, and as blind sheets are made for visits searched earlier.":"Nothing here."}</div>`;return}
 const date=new Date((c.t_ref-40587)*86400000).toISOString().replace("T"," ").slice(0,16);
 D.innerHTML=head+`
 <div class="zoom">Zoom <button class="act" data-z="-1">−</button><button class="act" data-z="1">+</button>
  <span>${sel+1} of ${items.length}</span></div>
 <div class="sheet"><img id="img" src="/sheet/${c.id}.png" alt="cutout sheet"></div>
 <dl>
  <div><dt>Position (J2000)</dt><dd>${fmt(c.ra,5)} ${c.dec>=0?"+":""}${fmt(c.dec,5)}</dd></div>
  <div><dt>Epoch (UTC)</dt><dd>${date}</dd></div>
  <div><dt>Rate · PA</dt><dd>${fmt(c.rate,3)} °/day · ${fmt(c.pa,1)}°</dd></div>
  <div><dt>Arc · time span</dt><dd title="first to last of the ${c.n_frames} frames the track was measured on${c.arc_arcsec!=null?": "+fmt(c.arc_arcsec,1)+"″ in "+fmt(c.span_hours,2)+" h":""}">${arcSpan(c)}</dd></div>
  <div><dt>W3 magnitude</dt><dd>${fmt(c.mag,2)}</dd></div>
  <div><dt>Track score (W3+W4)</dt><dd>${fmt(c.score,1)} on ${c.n_frames} frames</dd></div>
  <div><dt>W4 track score</dt><dd>${c.w4_score==null?"no W4 frames":fmt(c.w4_score,1)}</dd></div>
  ${view!=="todo"?`<div><dt>Catalogue</dt><dd>${catLabel(c)}</dd></div>`:""}
  <div><dt>Verdict</dt><dd>${c.verdict?VNAME[c.verdict]+(c.note?" — "+c.note:""):"none"}</dd></div>
 </dl>
 <div class="hint">${c.visit_dir} #${c.det}</div>
 <div class="verdict">
  <button class="big ok" id="bOk">Confirm (C)</button>
  <button class="big mid" id="bMid">Likely (L)</button>
  <button class="big bad" id="bBad">Discard (D)</button>
  ${c.verdict?'<button class="big skip" id="bUndo">Undo (U)</button>':'<button class="big skip" id="bSkip">Skip (S)</button>'}
  <input type="text" id="note" placeholder="Note (optional)" value="${(c.note||"").replace(/"/g,"&quot;")}">
 </div>
 <div class="hint">${view==="todo"?"Blind: whether this is a catalogued asteroid is shown once it has a verdict.":"The catalogue match is shown because this detection has a verdict."} Upper row: each frame, stars and all. Lower row: the other frames' median subtracted. Green panel: the frames stacked along the track.
  Keys: C confirm · L likely · D discard · S/→ next · ← previous · U undo · +/− zoom.</div>`;
 const img=$("#img"); img.onload=()=>{img.style.width=img.naturalWidth*zoom+"px"};
 $("#bOk").onclick=()=>decide("confirmed"); $("#bMid").onclick=()=>decide("likely"); $("#bBad").onclick=()=>decide("rejected");
 $("#bSkip")&&($("#bSkip").onclick=()=>move(1)); $("#bUndo")&&($("#bUndo").onclick=()=>decide(null));
 D.querySelectorAll("[data-z]").forEach(b=>b.onclick=()=>setZoom(+b.dataset.z));
}
function setZoom(d){zoom=Math.min(3,Math.max(0.5,zoom+d*0.25));const i=$("#img");i&&(i.style.width=i.naturalWidth*zoom+"px")}
function move(d){if(!items.length)return; sel=(sel+d+items.length)%items.length; renderList(); renderDetail()}
async function decide(v){
 const c=items[sel]; if(!c)return;
 const note=($("#note")||{}).value||"";
 try{await api("/api/verdict",{id:c.id,verdict:v,note});
  toast(v==="confirmed"?"Confirmed — added to confirmed.csv":v==="likely"?"Likely — added to confirmed.csv":v==="rejected"?"Discarded":"Verdict removed");
  if(view==="results") await loadResults();
  if(view==="todo"&&v){items.splice(sel,1); if(sel>=items.length)sel=Math.max(0,items.length-1); renderList(); renderDetail(); refreshState()}
  else await load(true);
 }catch(e){toast("Error: "+e.message)}
}
document.addEventListener("keydown",e=>{
 if(e.target.id==="note"){if(e.key==="Enter")decide("confirmed");if(e.key==="Escape")e.target.blur();return}
 if(view==="skymap"||view==="plots"||$("#mpcDlg").open||/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;
 const k=e.key.toLowerCase();
 if(k==="c")decide("confirmed");else if(k==="l")decide("likely");else if(k==="d")decide("rejected");else if(k==="u")decide(null);
 else if(k==="s"||k==="arrowright"||k==="j")move(1);else if(k==="arrowleft"||k==="k")move(-1);
 else if(k==="+"||k==="=")setZoom(1);else if(k==="-")setZoom(-1);
});
function showTab(v){
 document.querySelectorAll("#tabs button").forEach(x=>x.classList.toggle("on",x.dataset.v===v));
 view=v; sel=0; $("main").hidden=v==="skymap"||v==="plots"; $("#sky").hidden=v!=="skymap"; $("#plots").hidden=v!=="plots"}
document.querySelectorAll("#tabs button").forEach(b=>b.onclick=async()=>{
 showTab(b.dataset.v); if(view==="skymap") return drawSky(); if(view==="plots") return drawPlots(); if(view==="results") await loadResults(); load(false)});
// ---- sky map: Hammer-Aitoff, RA 180 in the middle, increasing to the left
const R2=Math.SQRT2, SW=1000, SH=520, SX=SW/2/(2*R2)*0.97, SY=SX;
function hammer(ra,dec){const l=-(((ra%360)+360)%360-180)*Math.PI/180, p=dec*Math.PI/180, d=Math.sqrt(1+Math.cos(p)*Math.cos(l/2));
 return [SW/2+SX*2*R2*Math.cos(p)*Math.sin(l/2)/d, SH/2-SY*R2*Math.sin(p)/d]}
function eclPts(beta){const e=23.4392911*Math.PI/180, b=beta*Math.PI/180, out=[];
 for(let k=0;k<=360;k++){const L=k*Math.PI/180;
  const dec=Math.asin(Math.sin(b)*Math.cos(e)+Math.cos(b)*Math.sin(e)*Math.sin(L));
  const ra=Math.atan2(Math.sin(L)*Math.cos(e)-Math.tan(b)*Math.sin(e),Math.cos(L))*180/Math.PI;
  out.push([(ra+360)%360,dec*180/Math.PI])}return out}
function line(pts){let d="",prev=null;for(const [ra,dec] of pts){const [x,y]=hammer(ra,dec);
 d+=(prev==null||Math.abs(x-prev)>SW/3?"M":"L")+x.toFixed(1)+" "+y.toFixed(1);prev=x}return d}
let skyData=null;
async function drawSky(){
 const only=$("#skyNew").checked, likely=$("#skyLikely").checked, q=`uncatalogued=${only?1:0}&likely=${likely?1:0}`;
 $("#skyPng").href="/skymap.png?"+q;
 skyData=await api("/api/skymap?"+q); const D=skyData, svg=$("#skySvg");
 let g="";
 for(let ra=0;ra<360;ra+=30){const pts=[];for(let dec=-90;dec<=90;dec+=3)pts.push([ra===180?179.999:ra,dec]);g+=`<path class="grid" d="${line(pts)}"/>`}
 for(let dec=-60;dec<=60;dec+=30){const pts=[];for(let ra=0.001;ra<=360;ra+=3)pts.push([ra,dec]);g+=`<path class="grid" d="${line(pts)}"/>`}
 g+=`<ellipse class="grid" cx="${SW/2}" cy="${SH/2}" rx="${SX*2*R2}" ry="${SY*R2}" style="stroke:var(--mute)"/>`;
 for(let ra=0;ra<360;ra+=60){const [x,y]=hammer(ra===180?179.999:ra,0);g+=`<text class="lab" x="${x+3}" y="${y-4}">${ra}°</text>`}
 for(const dec of [-60,-30,30,60]){const [x,y]=hammer(180,dec);g+=`<text class="lab" x="${x+4}" y="${y-4}">${dec>0?"+":""}${dec}°</text>`}
 g+=`<path fill="var(--field)" d="${D.searched.map(([ra,dec])=>{const [x,y]=hammer(ra,dec);return `M${(x-2.5).toFixed(1)} ${(y-2.5).toFixed(1)}h5v5h-5z`}).join("")}"/>`;
 g+=`<path class="ecl" d="${line(eclPts(0))}"/>`;
 if(D.lat_max) g+=`<path class="band" d="${line(eclPts(D.lat_max))}"/><path class="band" d="${line(eclPts(-D.lat_max))}"/>`;
 const objs=[...D.objects].sort((a,b)=>(a.catalogue==="not catalogued")-(b.catalogue==="not catalogued"));
 g+=objs.map(o=>{const [x,y]=hammer(o.ra,o.dec), c=o.catalogue==="not catalogued"?"var(--new)":"var(--cat)";
  const dot=o.verdict==="likely"?`<circle class="dot" cx="${x}" cy="${y}" r="5" fill="none" stroke="${c}" stroke-width="2"/>`
   :`<circle class="dot" cx="${x}" cy="${y}" r="5" fill="${c}" stroke="var(--panel)" stroke-width="1.5"/>`;
  return `<g class="pt" data-id="${o.id}"><circle cx="${x}" cy="${y}" r="10" fill="transparent"/>${dot}</g>`}).join("");
 svg.innerHTML=g;
 const n=D.objects.length, nn=D.objects.filter(o=>o.catalogue==="not catalogued").length;
 $("#skyCount").textContent=`${n} shown (${nn} not catalogued) · ${D.searched.length} fields searched`;
 const tip=$("#skyTip"), byId=Object.fromEntries(D.objects.map(o=>[o.id,o]));
 const label=o=>o.catalogue==="not catalogued"?(o.near?"not catalogued, near "+o.near:"not catalogued"):(o.known_name||"catalogued asteroid");
 const when=o=>new Date((o.t_ref-40587)*86400000).toISOString().slice(0,16).replace("T"," ");
 svg.querySelectorAll(".pt").forEach(el=>{const o=byId[el.dataset.id];
  el.onmousemove=ev=>{const r=$("#skyWrap").getBoundingClientRect();
   tip.innerHTML=`<b>${esc(label(o))}</b><br>${VNAME[o.verdict]} · RA ${fmt(o.ra,3)} Dec ${o.dec>=0?"+":""}${fmt(o.dec,3)}<br>${when(o)} UTC · ${fmt(o.rate,3)}°/day<br><i>click to open</i>`;
   tip.style.display="block"; let lx=ev.clientX-r.left+14; if(lx>r.width-220) lx=ev.clientX-r.left-230;
   tip.style.left=lx+"px"; tip.style.top=(ev.clientY-r.top+12)+"px"};
  el.onmouseleave=()=>tip.style.display="none";
  el.onclick=()=>openCandidate(o)});
 $("#skyTable div").innerHTML=`<table><tr><th>Object</th><th>Verdict</th><th>RA</th><th>Dec</th><th>Epoch (UTC)</th><th>Rate °/day</th></tr>`+
  D.objects.map(o=>`<tr><td>${esc(label(o))}</td><td>${VNAME[o.verdict]}</td><td>${fmt(o.ra,4)}</td><td>${fmt(o.dec,4)}</td><td>${when(o)}</td><td>${fmt(o.rate,3)}</td></tr>`).join("")+"</table>";
}
async function openCandidate(o){
 $("#skyTip").style.display="none"; $("#plotTip").style.display="none";
 showTab(o.catalogue==="not catalogued"?"notcat":o.verdict); await load(false);
 const i=items.findIndex(c=>c.id===o.id); if(i>=0){sel=i; renderList(); renderDetail()}}
$("#skyNew").onchange=drawSky; $("#skyLikely").onchange=drawSky;
// ---- plots: small SVG charts with a hover tip; points that are detections open them
function ticks(lo,hi,log){
 if(log){const t=[];for(let e=Math.floor(Math.log10(lo));e<=Math.ceil(Math.log10(hi));e++)for(const m of [1,2,5]){const v=m*10**e;if(v>=lo*0.999&&v<=hi*1.001)t.push(v)}return t}
 const p=10**Math.floor(Math.log10((hi-lo)/5||1)), step=[1,2,2.5,5,10].map(m=>m*p).find(s=>(hi-lo)/s<=6), t=[];
 for(let v=Math.ceil(lo/step-1e-9)*step;v<=hi+1e-9;v+=step)t.push(+v.toFixed(6));return t}
function frame(svg,o){ // o: x/y [lo,hi], xlog, ylog, flipX, xlab, ylab, ycats
 const [w,h]=svg.getAttribute("viewBox").split(" ").slice(2).map(Number), m={l:o.ycats?110:50,r:12,t:8,b:o.b??40}, W=w-m.l-m.r, H=h-m.t-m.b;
 const f=(v,[a,b],log)=>log?(Math.log10(v)-Math.log10(a))/(Math.log10(b)-Math.log10(a)):(v-a)/(b-a);
 const X=v=>m.l+W*(o.flipX?1-f(v,o.x,o.xlog):f(v,o.x,o.xlog)), Y=v=>m.t+H*(1-f(v,o.y,o.ylog));
 let g=`<defs><clipPath id="clip_${svg.id}"><rect x="${m.l}" y="${m.t}" width="${W}" height="${H}"/></clipPath></defs>`;
 const xt=o.xcats?o.xcats.map((c,i)=>[i,c]):(o.xticks||ticks(...o.x,o.xlog)).map(t=>[t,o.xfmt?o.xfmt(t):+t.toPrecision(3)]);
 for(const [t,l] of xt)g+=`<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${m.t+H}"/><text class="lab" x="${X(t)}" y="${m.t+H+14}" text-anchor="middle">${esc(l)}</text>`;
 const yt=o.ycats?o.ycats.map((c,i)=>[i,c]):ticks(...o.y,o.ylog).map(t=>[t,+t.toPrecision(3)]);
 for(const [t,l] of yt)g+=`<line class="grid" x1="${m.l}" x2="${m.l+W}" y1="${Y(t)}" y2="${Y(t)}"/><text class="lab" x="${m.l-6}" y="${Y(t)+4}" text-anchor="end">${esc(l)}</text>`;
 g+=`<line class="axis" x1="${m.l}" x2="${m.l+W}" y1="${m.t+H}" y2="${m.t+H}"/><line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t+H}"/>`;
 if(o.xlab)g+=`<text class="ttl" x="${m.l+W/2}" y="${h-4}" text-anchor="middle">${o.xlab}</text>`;
 if(o.ylab)g+=`<text class="ttl" transform="translate(12 ${m.t+H/2}) rotate(-90)" text-anchor="middle">${o.ylab}</text>`;
 return {X,Y,g,clip:`clip-path="url(#clip_${svg.id})"`}}
function tips(svg,items,label,click){
 svg.querySelectorAll("[data-k]").forEach(el=>{const it=items[+el.dataset.k], tip=$("#plotTip");
  el.onmousemove=ev=>{const r=$("#plots").getBoundingClientRect(); tip.innerHTML=label(it); tip.style.display="block";
   let lx=ev.clientX-r.left+14; if(lx>r.width-260) lx=ev.clientX-r.left-270;
   tip.style.left=lx+"px"; tip.style.top=(ev.clientY-r.top+12)+"px"};
  el.onmouseleave=()=>tip.style.display="none";
  if(click){el.classList.add("click"); el.onclick=()=>click(it)}})}
const pct=x=>x==null?"–":(100*x).toFixed(0)+"%";
function binLabel(b,unit){return b.lo==null?`below ${b.hi}${unit}`:b.hi==null?`${b.lo}${unit} and above`:`${b.lo}–${b.hi}${unit}`}
function fracChart(svg,bins,o){
 const F=frame(svg,o); let g=F.g, path="";
 bins.forEach((b,i)=>{const x=F.X(b.x),y=F.Y(100*b.frac); path+=(i?"L":"M")+x.toFixed(1)+" "+y.toFixed(1);
  g+=`<line x1="${x}" x2="${x}" y1="${F.Y(100*b.lo68)}" y2="${F.Y(100*b.hi68)}" stroke="var(--cat)" stroke-width="1.5"/>`});
 g+=`<path d="${path}" fill="none" stroke="var(--cat)" stroke-width="2"/>`;
 g+=bins.map((b,i)=>`<g data-k="${i}"><circle cx="${F.X(b.x)}" cy="${F.Y(100*b.frac)}" r="11" fill="transparent"/><circle cx="${F.X(b.x)}" cy="${F.Y(100*b.frac)}" r="4.5" fill="var(--cat)" stroke="var(--panel)" stroke-width="1.5"/></g>`).join("");
 svg.innerHTML=g; tips(svg,bins,b=>`<b>${o.what} ${binLabel(b,o.unit)}</b><br>${b.found} of ${b.n} ${o.verb||"found"} (${pct(b.frac)})<br>68% interval ${pct(b.lo68)}–${pct(b.hi68)}`)}
function catLabelM(o){return o.catalogue==="not catalogued"?(o.near?"not catalogued, near "+o.near:"not catalogued"):(o.known_name||"catalogued asteroid")}
function motionChart(svg,M,zoom){
 const ref=M.main_belt, os=M.objects; let x,y;
 if(zoom&&ref){x=[ref.vl-8*ref.sl,ref.vl+8*ref.sl]; y=[ref.vb-8*ref.sb,ref.vb+8*ref.sb]}
 else{const vx=os.map(o=>o.vl).concat([0]), vy=os.map(o=>o.vb).concat([0]), px=(Math.max(...vx)-Math.min(...vx))*0.06+0.05, py=(Math.max(...vy)-Math.min(...vy))*0.06+0.05;
  x=[Math.min(...vx)-px,Math.max(...vx)+px]; y=[Math.min(...vy)-py,Math.max(...vy)+py]}
 const F=frame(svg,{x,y,xlab:"along the ecliptic, dλ cos β / dt (°/day)",ylab:"across it, dβ / dt (°/day)"});
 let g=F.g+`<g ${F.clip}>`;
 if(x[0]<0&&x[1]>0)g+=`<line class="axis" x1="${F.X(0)}" x2="${F.X(0)}" y1="${F.Y(y[0])}" y2="${F.Y(y[1])}"/>`;
 if(y[0]<0&&y[1]>0)g+=`<line class="axis" x1="${F.X(x[0])}" x2="${F.X(x[1])}" y1="${F.Y(0)}" y2="${F.Y(0)}"/>`;
 if(ref){let d="";for(let k=0;k<=72;k++){const t=k/72*2*Math.PI;d+=(k?"L":"M")+F.X(ref.vl+5*ref.sl*Math.cos(t)).toFixed(1)+" "+F.Y(ref.vb+5*ref.sb*Math.sin(t)).toFixed(1)}g+=`<path class="ref" d="${d}"/>`}
 const order=os.map((o,i)=>i).sort((a,b)=>(os[a].catalogue==="not catalogued")-(os[b].catalogue==="not catalogued"));
 for(const i of order){const o=os[i], X=F.X(o.vl), Y=F.Y(o.vb), nw=o.catalogue==="not catalogued", c=nw?"var(--new)":"var(--cat)", r=nw?5.5:3.2;
  const dot=o.verdict==="likely"?`<circle cx="${X}" cy="${Y}" r="${r}" fill="none" stroke="${c}" stroke-width="1.8"/>`:`<circle cx="${X}" cy="${Y}" r="${r}" fill="${c}" stroke="var(--panel)" stroke-width="1"/>`;
  g+=`<g data-k="${i}"><circle cx="${X}" cy="${Y}" r="${nw?11:7}" fill="transparent"/>${dot}</g>`;
  if(nw&&(o.sigma||0)>(zoom?1.4:5))g+=`<text class="ttl" x="${X+7}" y="${Y-6}">#${o.id}</text>`}
 svg.innerHTML=g+"</g>";
 tips(svg,os,o=>`<b>${esc(catLabelM(o))}</b> #${o.id}${o.class?" · "+o.class:""}<br>${VNAME[o.verdict]} · ${fmt(o.rate,3)}°/day at PA ${fmt(o.pa,0)}°<br>arc ${arcSpan(o)} (${o.n_frames} frames)<br>`+
  `${fmt(o.vl,3)}°/day along, ${fmt(o.vb,3)}°/day across the ecliptic`+(o.sigma!=null?`<br>${fmt(o.sigma,1)}σ from the main belt`:"")+"<br><i>click to open</i>",openCandidate)}
function astromCharts(A){
 const pix=A.pixel, lim=2.2*pix, os=A.objects.map(o=>({...o,catalogue:"catalogued"}));
 let F=frame($("#aSky"),{x:[-lim,lim],y:[-lim,lim],flipX:true,xlab:"ΔRA cos Dec (″)",ylab:"ΔDec (″)"}), g=F.g+`<g ${F.clip}>`;
 g+=`<path fill="var(--field)" d="${A.frames.map(([a,b])=>`M${F.X(a).toFixed(1)} ${F.Y(b).toFixed(1)}h1.6v1.6h-1.6z`).join("")}"/>`;
 let d="";for(let k=0;k<=72;k++){const t=k/72*2*Math.PI;d+=(k?"L":"M")+F.X(pix*Math.cos(t)).toFixed(1)+" "+F.Y(pix*Math.sin(t)).toFixed(1)}
 g+=`<path class="ref" d="${d}"/>`;
 g+=os.map((o,i)=>`<g data-k="${i}"><circle cx="${F.X(o.dra)}" cy="${F.Y(o.ddec)}" r="7" fill="transparent"/><circle cx="${F.X(o.dra)}" cy="${F.Y(o.ddec)}" r="2.8" fill="var(--cat)" stroke="var(--panel)" stroke-width="0.8"/></g>`).join("");
 $("#aSky").innerHTML=g+"</g>";
 const lab=o=>`<b>${esc(o.known_name||"catalogued asteroid")}</b> #${o.id}${o.class?" · "+o.class:""}<br>ΔRA ${fmt(o.dra,2)}″ ΔDec ${fmt(o.ddec,2)}″ (median of ${o.n_obs} frames)<br>track score ${fmt(o.score,1)} · ephemeris ±${fmt(o.eph_err,2)}″<br><i>click to open</i>`;
 tips($("#aSky"),os,lab,openCandidate);
 const sc=os.map(o=>o.score), ymax=Math.max(pix*1.3,...os.map(o=>o.sep)), xr=[Math.min(...sc,10)*0.9,Math.max(...sc,20)*1.1];
 F=frame($("#aScore"),{x:xr,xlog:true,y:[0,ymax],xlab:"track score",ylab:"offset (″)"}); g=F.g;
 g+=`<line class="ref" x1="${F.X(xr[0])}" x2="${F.X(xr[1])}" y1="${F.Y(pix)}" y2="${F.Y(pix)}"/>`;
 g+=os.map((o,i)=>`<g data-k="${i}"><circle cx="${F.X(o.score)}" cy="${F.Y(o.sep)}" r="7" fill="transparent"/><circle cx="${F.X(o.score)}" cy="${F.Y(o.sep)}" r="2.8" fill="var(--cat)" stroke="var(--panel)" stroke-width="0.8"/></g>`).join("");
 $("#aScore").innerHTML=g; tips($("#aScore"),os,lab,openCandidate);
 const s=A.stats;
 $("#aStats").textContent=s?`The confirmed${$("#plLikely").checked?" and likely":""} catalogued detections against the catalogue (SkyBoT, shifted to the spacecraft), ${s.n_obs} observations of ${os.length} detections: `+
  `median offset ΔRA ${s.median_dra>=0?"+":""}${s.median_dra}″, ΔDec ${s.median_ddec>=0?"+":""}${s.median_ddec}″; RMS ${s.rms_dra}″ and ${s.rms_ddec}″; 68% within ${s.p68}″, 95% within ${s.p95}″. `+
  `${A.loose_ephemeris} left out, their ephemeris worse than ${A.max_eph_err}″. These are the positions an MPC submission gives.`:"No catalogued detections with a verdict yet."}
function galleryCards(G){
 $("#gal").innerHTML=G.length?G.map((o,i)=>`<div class="gcard" data-g="${i}"><div class="gh"><b>#${o.id}</b><span class="pill ${o.verdict}">${VNAME[o.verdict]}</span>`+
  (o.sigma!=null?`<span class="sig">${fmt(o.sigma,0)}σ from the main belt</span>`:"")+
  `<span>${fmt(o.rate,3)}°/day at PA ${fmt(o.pa,0)}°</span><span>arc ${arcSpan(o)}</span><span>RA ${fmt(o.ra,4)} Dec ${o.dec>=0?"+":""}${fmt(o.dec,4)}</span>`+
  `<span>${new Date((o.t_ref-40587)*86400000).toISOString().slice(0,16).replace("T"," ")} UTC</span><span>score ${fmt(o.score,1)}</span>`+
  (o.near?`<span>near ${esc(o.near)}</span>`:"")+`</div><div class="sheet"><img loading="lazy" src="/sheet/${o.id}.png" alt="cutout sheet of #${o.id}"></div></div>`).join("")
  :`<div class="empty">None yet.</div>`;
 $("#gal").querySelectorAll(".gcard").forEach(el=>el.onclick=()=>openCandidate(G[+el.dataset.g]))}
async function drawPlots(){
 const likely=$("#plLikely").checked;
 $("#plCount").textContent="Reading the visits…";
 document.querySelectorAll("#plots [data-png]").forEach(a=>a.href=`/plots/${a.dataset.png}.png?likely=${likely?1:0}`);
 const D=await api("/api/plots?likely="+(likely?1:0)), C=D.completeness;
 $("#plCount").textContent=`${D.visits} visits searched · ${D.motion.objects.length} confirmed${likely?" or likely":""} detections · ${D.gallery.length} not catalogued`;
 document.querySelectorAll("#plots .minfr").forEach(e=>e.textContent=C.min_frames); document.querySelectorAll("#plots .floor").forEach(e=>e.textContent=C.snr_floor);
 $("#cTitle").textContent=`All ${C.n} on ≥${C.min_frames} frames (${C.found} found), by signal`;
 if(C.by_snr.length){
  fracChart($("#cSnr"),C.by_snr,{x:[1,Math.max(200,...C.by_snr.map(b=>b.x*1.3))],xlog:true,y:[0,100],xlab:"signal on the predicted track",ylab:"found (%)",what:"signal",unit:""});
  fracChart($("#cRate"),C.by_rate,{x:[0,Math.max(0.5,...C.by_rate.map(b=>b.x*1.15))],y:[0,100],xlab:"rate (°/day)",ylab:"found (%)",what:"rate",unit:"°/day"});
  const B=C.by_class, F=frame($("#cClass"),{x:[0,100],y:[B.length-0.5,-0.5],ycats:B.map(b=>`${b.class} (${b.n})`),xlab:"found (%)"}); let g=F.g;
  g+=B.map((b,i)=>`<line x1="${F.X(100*b.lo68)}" x2="${F.X(100*b.hi68)}" y1="${F.Y(i)}" y2="${F.Y(i)}" stroke="var(--cat)" stroke-width="1.5"/>`+
   `<g data-k="${i}"><circle cx="${F.X(100*b.frac)}" cy="${F.Y(i)}" r="10" fill="transparent"/><circle cx="${F.X(100*b.frac)}" cy="${F.Y(i)}" r="4.5" fill="var(--cat)" stroke="var(--panel)" stroke-width="1.5"/></g>`).join("");
  $("#cClass").innerHTML=g; tips($("#cClass"),B,b=>`<b>${b.class}</b><br>${b.found} of ${b.n} found (${pct(b.frac)})<br>68% interval ${pct(b.lo68)}–${pct(b.hi68)}`);
  const row=(b,l)=>`<tr><td style="text-align:left">${l}</td><td>${b.n}</td><td>${b.found}</td><td>${pct(b.frac)}</td><td>${pct(b.lo68)}–${pct(b.hi68)}</td></tr>`;
  $("#cTable").innerHTML=`<table><tr><th style="text-align:left">Bin</th><th>Asteroids</th><th>Found</th><th>Fraction</th><th>68%</th></tr>`+
   C.by_snr.map(b=>row(b,"signal "+binLabel(b,""))).join("")+C.by_rate.map(b=>row(b,"rate "+binLabel(b,"°/day"))).join("")+B.map(b=>row(b,b.class)).join("")+"</table>"}
 motionChart($("#mAll"),D.motion,false); motionChart($("#mZoom"),D.motion,true);
 galleryCards(D.gallery); astromCharts(D.astrometry); discChart(D.discovery);
 document.querySelectorAll("#plots .lk").forEach(e=>e.style.display=likely?"":"none");
 scoreCharts(D.scores); brightChart(D.brightness); galChart(D.galactic); eclCharts(D.ecliptic); solarChart(D.solar_system); progressCharts(D.progress);
}
function discChart(Dd){
 const e=Dd.edges, a=Dd.discovery, b=Dd.designation, n=a.length, tot=a.map((v,i)=>v+b[i]), top=Math.max(1,...tot)*1.25, lo=Dd.lag_min;
 const F=frame($("#dHist"),{x:[e[0]-0.3,e[n]+0.3],y:[0,top],xticks:e.filter(t=>t%5===0),xfmt:t=>t===lo?"≤"+t:t>0?"+"+t:""+t,
  xlab:"discovery date − date of the WISE image it was found on (years)",ylab:"catalogued asteroids"});
 let g=F.g; const w=(F.X(1)-F.X(0))*0.86;
 g+=`<rect x="${F.X(0)}" y="${F.Y(top)}" width="${F.X(e[n]+0.3)-F.X(0)}" height="${F.Y(0)-F.Y(top)}" fill="var(--aqua)" opacity="0.07"/>`;
 tot.forEach((t,i)=>{const x=F.X(e[i])+(F.X(1)-F.X(0)-w)/2, ya=F.Y(a[i]), yt=F.Y(t);
  g+=`<g data-k="${i}"><rect x="${F.X(e[i])}" y="${F.Y(top)}" width="${F.X(1)-F.X(0)}" height="${F.Y(0)-F.Y(top)}" fill="transparent"/>`+
   (a[i]?`<rect x="${x}" y="${ya}" width="${w}" height="${F.Y(0)-ya}" fill="var(--cat)" rx="2"/>`:"")+
   (b[i]?`<rect x="${x}" y="${yt}" width="${w}" height="${Math.max(0,ya-yt-(a[i]?2:0))}" fill="var(--aqua)" rx="2"/>`:"")+"</g>"});
 g+=`<line x1="${F.X(0)}" x2="${F.X(0)}" y1="${F.Y(0)}" y2="${F.Y(top)}" stroke="var(--ink)" stroke-width="1.5"/>`;
 g+=`<text class="ttl" x="${F.X(0)+8}" y="${F.Y(top)+16}">seen by WISE before discovery →</text><text class="lab" x="${F.X(0)-8}" y="${F.Y(top)+16}" text-anchor="end">← discovered before the image</text>`;
 $("#dHist").innerHTML=g;
 const os=Dd.objects, span=i=>i===0?`${-e[1]} or more years before`:e[i]>=0?`${e[i]} to ${e[i+1]} years after`:`${-e[i+1]} to ${-e[i]} years before`;
 tips($("#dHist"),tot.map((t,i)=>i),i=>{const inBin=os.filter(o=>o.lag>=e[i]&&o.lag<e[i+1]);
  return `<b>discovered ${span(i)} the image</b><br>${tot[i]} asteroids: ${a[i]} by credited discovery, ${b[i]} by designation date`+
   (inBin.length?"<br>"+inBin.slice(0,6).map(o=>esc(o.known_name)).join("<br>")+(inBin.length>6?`<br>and ${inBin.length-6} more`:""):"")});
 $("#dSum").textContent=Dd.n?`${Dd.before} of ${Dd.n} (${pct(Dd.before/Dd.n)}) were seen by WISE before discovery`+
  (Dd.before?`, ${Dd.before_designation} of them dated by their designation; the earliest image came ${fmt(os[0].lag,1)} years before (${esc(os[0].known_name)}).`:".")+
  (Dd.unsure?` ${Dd.unsure} more were designated in the half month of their image, too close to tell.`:""):"";
 $("#dNote").textContent=Dd.pending?`${Dd.pending} discovery dates still to look up${Dd.fetching?" (looking them up now…)":""}`:
  (Dd.earliest!=null&&Dd.earliest<lo?`leftmost bar: ${-lo} or more years before (earliest ${fmt(-Dd.earliest,0)})`:"");
 $("#dTable").innerHTML=os.length?`<table><tr><th style="text-align:left">Asteroid</th><th>WISE image</th><th>Discovered</th><th>Years before</th><th style="text-align:left">Date from</th><th style="text-align:left">Discoverer</th></tr>`+
  os.map(o=>`<tr class="click" data-c="${o.id}" data-v="${o.verdict}"><td style="text-align:left">${esc(o.known_name)}</td><td>${o.image}</td><td>${o.discovered}</td><td>${fmt(o.lag,1)}</td><td style="text-align:left">${o.source==="discovery"?"credited discovery":"designation"}</td><td style="text-align:left">${esc(o.who||"")}</td></tr>`).join("")+"</table>":`<div class="empty">None yet.</div>`;
 $("#dTable").querySelectorAll("tr[data-c]").forEach(tr=>tr.onclick=()=>openCandidate({id:+tr.dataset.c,verdict:tr.dataset.v,catalogue:"catalogued"}));
 clearTimeout(discChart.t);
 if(Dd.pending&&Dd.fetching) discChart.t=setTimeout(async()=>discChart(await api("/api/discovery?likely="+($("#plLikely").checked?1:0))),8000)}
function bars(svg,vals,o){ // o: cats, color, tip(i), fmt, b
 const n=vals.length, top=Math.max(1e-9,...vals)*1.18, F=frame(svg,{x:[-0.5,n-0.5],y:[0,top],xcats:o.cats,xlab:o.xlab,ylab:o.ylab,b:o.b});
 const w=(F.X(1)-F.X(0))*0.78; let g=F.g;
 vals.forEach((v,i)=>{g+=`<g data-k="${i}"><rect x="${F.X(i)-w/2}" y="${F.Y(top)}" width="${w}" height="${F.Y(0)-F.Y(top)}" fill="transparent"/>`+
  `<rect x="${F.X(i)-w/2}" y="${F.Y(v)}" width="${w}" height="${Math.max(0,F.Y(0)-F.Y(v))}" fill="${o.color}" rx="2"/>`+
  (v?`<text class="lab" x="${F.X(i)}" y="${F.Y(v)-3}" text-anchor="middle">${o.fmt?o.fmt(v):v}</text>`:"")+"</g>"});
 svg.innerHTML=g; tips(svg,vals.map((v,i)=>i),o.tip)}
function scoreCharts(S){
 const B=S.bins, cats=B.map(b=>b.lo==null?`<${b.hi}`:b.hi==null?`≥${b.lo}`:`${b.lo}–${b.hi}`), names=["confirmed","likely","discarded","not reviewed"];
 names.forEach((r,k)=>bars($("#sR"+k),B.map(b=>b[r]),{cats:k===3?cats:cats.map(()=>""),b:k===3?40:16,xlab:k===3?"track score":"",
  color:r==="not reviewed"?"var(--mute)":"var(--cat)",tip:i=>`<b>score ${cats[i]}: ${B[i][r]} ${r}</b><br>${names.map(x=>`${x} ${B[i][x]}`).join(" · ")}`}));
 const J=B.filter(b=>b.frac!=null&&b.reviewed>=5);
 fracChart($("#sFrac"),J.map(b=>({...b,n:b.reviewed,found:Math.round(b.frac*b.reviewed)})),{x:[Math.min(...J.map(b=>b.x))*0.9,Math.max(...J.map(b=>b.x))*1.15],xlog:true,
  y:[Math.max(0,Math.floor(Math.min(...J.map(b=>100*b.lo68))/10)*10-10),100],xlab:"track score",ylab:"judged real (%)",what:"score",unit:"",verb:"reviewed judged real"});
 $("#sExp").textContent=`about ${S.expected_real} of the ${S.not_reviewed} not yet reviewed are expected to be real.`}
function brightChart(Bd){
 const e=Bd.edges, n=Bd.catalogued, top=Math.max(1,...n)*1.35, F=frame($("#bHist"),{x:[e[0],e[e.length-1]],y:[0,top],xlab:"W3 magnitude (fainter →)",ylab:"detections"});
 let g=F.g; const w=F.X(e[1])-F.X(e[0]);
 n.forEach((v,i)=>{g+=`<g data-k="${i}"><rect x="${F.X(e[i])}" y="${F.Y(top)}" width="${w}" height="${F.Y(0)-F.Y(top)}" fill="transparent"/><rect x="${F.X(e[i])+1}" y="${F.Y(v)}" width="${w-2}" height="${F.Y(0)-F.Y(v)}" fill="var(--cat)" rx="2"/></g>`});
 for(const [m,c] of [[Bd.median_catalogued,"var(--cat)"],[Bd.median_new,"var(--new)"]]) if(m!=null) g+=`<line x1="${F.X(m)}" x2="${F.X(m)}" y1="${F.Y(0)}" y2="${F.Y(top)}" stroke="${c}" stroke-width="1.5" stroke-dasharray="4 4"/>`;
 const y0=Math.max(1,...n)*1.15, os=Bd.objects;
 g+=os.map((o,i)=>`<g data-o="${i}"><circle cx="${F.X(o.mag)}" cy="${F.Y(y0)}" r="10" fill="transparent"/><circle cx="${F.X(o.mag)}" cy="${F.Y(y0)}" r="5.5" fill="var(--new)" stroke="var(--panel)" stroke-width="1.2"/></g>`).join("");
 $("#bHist").innerHTML=g;
 tips($("#bHist"),n.map((v,i)=>i),i=>`<b>W3 ${e[i]}–${e[i+1]}</b><br>${n[i]} catalogued`);
 $("#bHist").querySelectorAll("[data-o]").forEach(el=>{el.dataset.k=el.dataset.o; delete el.dataset.o});
 const objTip=o=>`<b>#${o.id} ${esc(catLabelM(o))}</b><br>${VNAME[o.verdict]} · W3 ${fmt(o.mag,2)} · ${fmt(o.rate,3)}°/day · score ${fmt(o.score,1)}<br><i>click to open</i>`;
 [...$("#bHist").querySelectorAll("g[data-k]")].slice(n.length).forEach(el=>{const o=os[+el.dataset.k], tip=$("#plotTip");
  el.onmousemove=ev=>{const r=$("#plots").getBoundingClientRect(); tip.innerHTML=objTip(o); tip.style.display="block"; tip.style.left=(ev.clientX-r.left+14)+"px"; tip.style.top=(ev.clientY-r.top+12)+"px"};
  el.onmouseleave=()=>tip.style.display="none"; el.classList.add("click"); el.onclick=()=>openCandidate(o)});
 $("#bMed").textContent=Bd.median_catalogued!=null?`median ${fmt(Bd.median_catalogued,2)} catalogued`+(Bd.median_new!=null?`, ${fmt(Bd.median_new,2)} not catalogued`:""):""}
function galChart(G){
 const B=G.bins, cats=B.map(b=>`${b.lo}–${b.hi}°`);
 const tip=i=>`<b>|b| ${cats[i]}: ${B[i].visits} visits</b><br>catalogued found ${B[i].n_cat} (${B[i].cat} per visit)<br>not catalogued ${B[i].n_new} (${B[i].new} per visit)<br>discarded ${B[i].n_discarded}`;
 bars($("#gCat"),B.map(b=>b.cat),{cats,color:"var(--cat)",fmt:v=>v.toFixed(1),xlab:"|galactic latitude|",tip});
 bars($("#gNew"),B.map(b=>b.new),{cats,color:"var(--new)",fmt:v=>v.toFixed(2),xlab:"|galactic latitude|",tip})}
function eclCharts(E){
 const V=E.visits, date=m=>new Date((m-40587)*86400000).toISOString().slice(0,10);
 [["eCross","crossing","var(--mute)"],["eCat","cat","var(--cat)"],["eNew","new","var(--new)"]].forEach(([id,key,c],k)=>{
  const svg=$("#"+id), top=Math.max(1,...V.map(v=>v[key]))*1.1, F=frame(svg,{x:[0,360],xticks:[0,30,60,90,120,150,180,210,240,270,300,330,360],
   y:[0,top],b:k===2?40:16,xfmt:t=>k===2?t+"°":"",xlab:k===2?"ecliptic longitude of the field":""});
  let g=E.galactic_plane.map(([a,b])=>`<rect x="${F.X(a)}" y="${F.Y(top)}" width="${F.X(b)-F.X(a)}" height="${F.Y(0)-F.Y(top)}" fill="var(--idle2)"/>`).join("")+F.g;
  g+=V.map((v,i)=>`<g data-k="${i}"><circle cx="${F.X(v.elon)}" cy="${F.Y(v[key])}" r="6" fill="transparent"/><circle cx="${F.X(v.elon)}" cy="${F.Y(v[key])}" r="2.6" fill="${c}"/></g>`).join("");
  svg.innerHTML=g;
  tips(svg,V,v=>`<b>field ${v.field}</b> · ${date(v.mjd)}<br>λ ${fmt(v.elon,1)}° β ${fmt(v.elat,1)}° · galactic |b| ${fmt(v.glat,1)}°<br>${v.crossing} catalogued on the field · ${v.cat} found · ${v.new} not catalogued`)})}
function solarChart(P){
 const lim=5.8, F=frame($("#solar"),{x:[-lim,lim],y:[-lim,lim],xlab:"x (AU)",ylab:"y (AU)"});
 let g=F.g+`<g ${F.clip}>`; const sc=F.X(1)-F.X(0);
 for(const [nm,r] of Object.entries(P.planets)) g+=`<circle cx="${F.X(0)}" cy="${F.Y(0)}" r="${r*sc}" fill="none" stroke="${nm==="Earth"?"var(--mute)":"var(--line)"}"/><text class="lab" x="${F.X(0)+4}" y="${F.Y(r)-3}">${nm}</text>`;
 g+=`<path fill="var(--field)" d="${P.context.map(([x,y])=>`M${F.X(x).toFixed(1)} ${F.Y(y).toFixed(1)}h1.5v1.5h-1.5z`).join("")}"/>`;
 if(P.earth.length) g+=`<path d="${P.earth.map(([x,y],i)=>(i?"L":"M")+F.X(x).toFixed(1)+" "+F.Y(y).toFixed(1)).join("")}" fill="none" stroke="var(--ink)" stroke-width="3"/>`;
 g+=`<circle cx="${F.X(0)}" cy="${F.Y(0)}" r="6" fill="#eda100"/>`;
 const col={"main belt":"var(--cat)","Cybele, Hilda, Trojan":"var(--aqua)","NEA, Mars-crosser":"var(--new)"}, os=P.objects;
 const order=os.map((o,i)=>i).sort((a,b)=>(os[a].group!=="main belt")-(os[b].group!=="main belt"));
 for(const i of order){const o=os[i], big=o.group!=="main belt";
  g+=`<g data-k="${i}"><circle cx="${F.X(o.x)}" cy="${F.Y(o.y)}" r="${big?10:6}" fill="transparent"/><circle cx="${F.X(o.x)}" cy="${F.Y(o.y)}" r="${big?5:2.6}" fill="${col[o.group]||"var(--mute)"}" stroke="var(--panel)" stroke-width="${big?1.2:0.6}"/></g>`;
  if(o.group==="NEA, Mars-crosser") g+=`<text class="ttl" x="${F.X(o.x)+7}" y="${F.Y(o.y)-6}">${esc(o.known_name||"")}</text>`}
 $("#solar").innerHTML=g+"</g>";
 tips($("#solar"),os,o=>`<b>${esc(o.known_name||"catalogued asteroid")}</b> #${o.id} · ${o.orbit}<br>${fmt(o.r,2)} AU from the Sun, ${fmt(o.dist,2)} AU from WISE<br>${new Date((o.mjd-40587)*86400000).toISOString().slice(0,10)} · ${VNAME[o.verdict]}<br><i>click to open</i>`,openCandidate)}
function progressCharts(P){
 const V=P.visits; if(!V.length){$("#pV").textContent="No visits searched yet";return}
 const t=V.map(v=>Date.parse(v[0])), t0=t[0], t1=Math.max(Date.now(),t[t.length-1]);
 let cv=0,cc=0,cn=0; const Y={prV:[],prC:[],prN:[]};
 V.forEach(v=>{cv+=1;cc+=v[2];cn+=v[1]-v[2];Y.prV.push(cv);Y.prC.push(cc);Y.prN.push(cn)});
 $("#pV").textContent=`Visits searched: ${cv}`; $("#pC").textContent=`Catalogued asteroids found: ${cc}`; $("#pN").textContent=`Detections not catalogued: ${cn}`;
 const day=86400000, nd=Math.ceil((t1-t0)/day), step=Math.max(1,Math.ceil(nd/10)), xt=[];
 for(let d=new Date(t0);d.setHours(0,0,0,0),d<=t1;d=new Date(d.getTime()+step*day)) if(d>=t0) xt.push(d.getTime());
 const dfmt=x=>new Date(x).toLocaleDateString(undefined,{month:"short",day:"numeric"}), when=x=>x?new Date(x).toLocaleString(undefined,{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}):"now";
 const idle=P.idle.map(([a,b,k])=>({a:Date.parse(a),b:b?Date.parse(b):t1,k,open:!b}));
 [["prV","var(--ink)"],["prC","var(--cat)"],["prN","var(--new)"]].forEach(([id,c],k)=>{
  const svg=$("#"+id), y=Y[id], top=Math.max(1,y[y.length-1])*1.08, F=frame(svg,{x:[t0,t1],y:[0,top],xticks:xt,xfmt:k===2?dfmt:()=>"",b:k===2?40:16,xlab:k===2?"date (your local time)":""});
  let g=idle.map((iv,i)=>`<rect data-k="${i}" x="${F.X(Math.max(iv.a,t0))}" y="${F.Y(top)}" width="${Math.max(1,F.X(Math.min(iv.b,t1))-F.X(Math.max(iv.a,t0)))}" height="${F.Y(0)-F.Y(top)}" fill="${iv.k==="outside the hours"||iv.k==="not running"?"var(--idle)":"var(--idle2)"}"/>`).join("")+F.g;
  let d=`M${F.X(t0)} ${F.Y(0)}`; t.forEach((x,i)=>{d+=`H${F.X(x).toFixed(1)}V${F.Y(y[i]).toFixed(1)}`}); d+=`H${F.X(t1)}`;
  g+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="2"/>`;
  svg.innerHTML=g; tips(svg,idle,iv=>`<b>${iv.k}</b><br>${when(iv.a)} – ${iv.open?"now":when(iv.b)}`)})}
$("#plLikely").onchange=drawPlots;
$("#pauseBtn").onclick=async()=>{const a=$("#pauseBtn").dataset.a; await api("/api/control",{action:a});
 toast(a==="pause"?"Pausing: the GPU is freed within seconds":"Resumed"); refreshState()};
const lines=v=>v.split("\n").map(x=>x.trim()).filter(Boolean);
const esc=v=>String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;");
$("#reportBtn").onclick=()=>toast("Making the PDF report: it downloads in about 10 seconds");
$("#exportBtn").onclick=async()=>{const m=await api("/api/mpc");
 $("#mSub").value=m.submitter; $("#mInst").value=m.institution; $("#mMeas").value=m.measurers.join("\n");
 $("#mObs").value=m.observers.join("\n"); $("#mCom").value=m.comments.join("\n"); $("#mLikely").checked=!!m.include_likely;
 $("#mpcOut").innerHTML=""; $("#mpcDlg").showModal()};
$("#mpcGo").onclick=async()=>{const o=$("#mpcOut");
 if(!$("#mSub").value.trim()){o.innerHTML='<span class="bad">The MPC needs a submitter name.</span>';return}
 o.textContent="Writing…";
 try{const r=await api("/api/export",{submitter:$("#mSub").value.trim(),institution:$("#mInst").value.trim(),
   measurers:lines($("#mMeas").value),observers:lines($("#mObs").value),comments:lines($("#mCom").value),
   include_likely:$("#mLikely").checked});
  let h=r.path?`Wrote ${r.observations} observations of ${r.objects} object(s): <a href="/mpc/${r.file}" download>${esc(r.file)}</a>
   (and <a href="/mpc/${r.file.replace(/psv$/,"csv")}" download>the list of what is in it</a>), in ${esc(r.path.replace(/[^/]*$/,""))}`
   :r.problems.length?"Not written:":"Nothing to export: no confirmed"+($("#mLikely").checked?" or likely":"")+" detections that are not catalogued.";
  if(r.problems.length) h+=`<div class="bad">${r.problems.slice(0,8).map(esc).join("<br>")}</div>`;
  if(r.skipped.length) h+=`<div class="bad">Skipped (FITS files missing): ${r.skipped.map(esc).join("<br>")}</div>`;
  o.innerHTML=h}catch(e){o.innerHTML=`<span class="bad">Export failed: ${esc(e.message)}</span>`}};
load(false); setInterval(refreshState,5000); setInterval(()=>{if(view==="todo"&&!items.length)load(false)},30000);
</script></body></html>
"""
