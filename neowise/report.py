"""A PDF report of a campaign's results: ``neowise report OUT`` or the
review page's "PDF report" button.

Page 1 sums up the campaign and lists the significant findings, worked out
from the same numbers as the Plots tab (``plots.data``): the detections not
catalogued (those moving unlike a main-belt asteroid first), the catalogued
asteroids WISE saw before they were discovered, the search's detection
limit, the astrometry, what the review has left, and warnings (false tracks
near the galactic plane, suspiciously bright detections). Then a table of
the detections not catalogued, the plots with what each shows, and the
asteroids seen before discovery. Drawn with matplotlib (no other PDF
library needed).
"""

import json
import math
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import campaign as C
from . import plots as P

A4 = (8.27, 11.69)  # inches
MARGIN = 0.7
INK, MUTE, RULE = "#0b0b0b", "#52514e", "#c9c8c1"
WRAP = 96  # characters per line of body text (9 pt)

CAPTIONS = {
    "completeness": "Of the catalogued asteroids that crossed a searched field on at least {min_frames} frames, the fraction "
                    "found: against the signal the detector measures on the asteroid's predicted track, then (signal at "
                    "least {snr_floor}) by rate and by orbit class. Bars: 68% interval. This is the search's detection limit.",
    "motion": "Each confirmed or likely detection's motion along and across the ecliptic. Catalogued main-belt asteroids "
              "form the clump (dashed: 5σ of it); a detection not catalogued far from it doesn't move like a main-belt "
              "asteroid, the first sign of a near-Earth object.",
    "gallery": "The confirmed and likely detections with no catalogued asteroid there, the most unusual motion first, "
               "with their cutout sheets.",
    "astrometry": "The catalogued detections' measured positions minus the catalogue's (SkyBoT, shifted to the "
                  "spacecraft), per frame and per detection: what the positions in an MPC submission are worth.",
    "discovery": "For each catalogued asteroid confirmed or likely: its discovery date minus the date of the earliest WISE "
                 "image it was found on; right of zero, WISE saw it before it was discovered. Numbered asteroids: their "
                 "credited discovery (JPL's Small-Body Database). Unnumbered ones have none, so the date their provisional "
                 "designation encodes (to a half month) stands in.",
    "scores": "Detections by track score and verdict, and per score the fraction of those reviewed judged real. The "
              "review goes in a random order, so this estimates how many of those not yet reviewed are real.",
    "brightness": "W3 magnitudes of the confirmed and likely detections: the catalogued as a histogram, each one not "
                  "catalogued as a dot.",
    "galactic": "Detections per visit by the field's galactic latitude. Crowded star fields near the plane make false "
                "tracks: more detections not catalogued there, fewer catalogued ones found.",
    "ecliptic": "Each visit by its field's ecliptic longitude; shaded where the ecliptic is within 10° of the galactic plane.",
    "solar_system": "Where the catalogued asteroids found were, seen from ecliptic north, by orbit class; grey, every "
                    "catalogued asteroid on a searched field.",
    "progress": "Visits searched and their detections over time, from the campaign's log; shaded when the campaign "
                "wasn't searching.",
}
TITLES = {"completeness": "Completeness", "motion": "Motion", "gallery": "Not catalogued: cutout sheets",
          "astrometry": "Astrometry", "discovery": "Seen before discovery", "scores": "Scores",
          "brightness": "Brightness", "galactic": "Galactic latitude", "ecliptic": "Along the ecliptic",
          "solar_system": "Solar system", "progress": "Progress"}
ORDER = ("completeness", "motion", "gallery", "astrometry", "discovery",
         "scores", "brightness", "galactic", "ecliptic", "solar_system", "progress")


def _date(mjd) -> str:
    return datetime.fromtimestamp((float(mjd) - 40587) * 86400, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _arc(a, h) -> str:
    if a is None:
        return "–"
    arc = f"{a / 60:.1f}′" if a >= 60 else f"{a:.1f}″"
    return f"{arc} / {h:.1f} h"


def _crossing(bins, level):
    """The signal at which the found fraction first reaches ``level``
    (linear in log signal between bins), or None."""
    prev = None
    for b in bins:
        if b["n"] >= 5 and b["frac"] >= level:
            if prev is None:
                return b["x"]
            f = (level - prev["frac"]) / max(b["frac"] - prev["frac"], 1e-9)
            return 10 ** (math.log10(prev["x"]) + f * (math.log10(b["x"]) - math.log10(prev["x"])))
        if b["n"] >= 5:
            prev = b
    return None


def findings(d: dict, st: dict, include_likely: bool) -> list:
    """The significant findings, most important first, as (heading, text)."""
    out = []
    which = "confirmed or likely" if include_likely else "confirmed"
    mot = d["motion"]
    new = [o for o in mot["objects"] if o["catalogue"] == "not catalogued"]
    if new:
        odd = sorted((o for o in new if (o.get("sigma") or 0) > 5), key=lambda o: -o["sigma"])
        txt = (f"{len(new)} {which} detections have no catalogued asteroid there "
               f"({sum(o['verdict'] == 'confirmed' for o in new)} confirmed, {sum(o['verdict'] == 'likely' for o in new)} likely).")
        if odd:
            txt += (f" {len(odd)} move unlike a main-belt asteroid (more than 5σ from the catalogued main belt's motion), "
                    "the first sign of a near-Earth or other unusual object: " +
                    "; ".join(f"#{o['id']} at {o['rate']:.2f}°/day, {o['sigma']:.0f}σ"
                              + (f", arc {_arc(o.get('arc_arcsec'), o.get('span_hours'))}" if o.get("arc_arcsec") else "")
                              for o in odd[:8]) + (f"; and {len(odd) - 8} more" if len(odd) > 8 else "") + ".")
            weak = [o for o in odd if (o.get("score") or 99) < 12]
            if weak and len(weak) == len(odd):
                txt += (" All of them have track scores below 12, where false tracks are commonest: each is worth a "
                        "second look.")
            elif weak:
                txt += (" Of these, " + ", ".join(f"#{o['id']}" for o in weak) +
                        " have track scores below 12, where false tracks are commonest: worth a second look.")
        else:
            txt += " All move like main-belt asteroids."
        txt += (" These are 2010 images, so a real one can't be followed up directly; it could be linked to later "
                "observations, or reported to the MPC as unidentified tracklets.")
        out.append(("Detections not catalogued", txt))
    dl = d.get("discovery") or {}
    if dl.get("n"):
        objs = dl["objects"]
        txt = (f"{dl['before']} of the {dl['n']} catalogued asteroids found ({100 * dl['before'] / dl['n']:.0f}%) were seen by "
               f"WISE before their discovery")
        if objs:
            o = objs[0]
            txt += (f"; the earliest by {o['lag']:.1f} years ({o['known_name']}: WISE image {o['image']}, "
                    f"{'discovered' if o['source'] == 'discovery' else 'designated'} {o['discovered']})")
        txt += (f". {dl['before_designation']} of them are dated by their provisional designation (no credited discovery); "
                f"{dl.get('unsure', 0)} more were designated within the half month of their image, too close to tell.")
        if dl.get("by_wise"):
            txt += f" {dl['by_wise']} of the asteroids found were discovered by WISE itself."
        if dl.get("pending"):
            txt += f" ({dl['pending']} discovery dates weren't looked up yet.)"
        txt += (" WISE's own detections were reported to the MPC during the mission, so many of these observations may "
                "already be in its records: check before submitting them as precoveries.")
        out.append(("Seen before discovery", txt))
    c = d["completeness"]
    if c["n"]:
        s50, s90 = _crossing(c["by_snr"], 0.5), _crossing(c["by_snr"], 0.9)
        txt = (f"Of the {c['n']:,} catalogued asteroids that crossed a searched field on at least {c['min_frames']} frames, "
               f"{c['found']:,} were found ({100 * c['found'] / c['n']:.0f}%; most of the rest were too faint for any search).")
        if s50:
            txt += f" Half are found at a signal of {s50:.1f} on the predicted track"
            txt += f", 90% at {s90:.1f}." if s90 else "."
        top = [b for b in c["by_snr"] if b["lo"] is not None and b["lo"] >= 12 and b["n"] >= 20]
        if top:
            k, n = sum(b["found"] for b in top), sum(b["n"] for b in top)
            txt += f" Above a signal of 12: {100 * k / n:.0f}% of {n:,}."
        out.append(("Detection limit", txt))
    a = d["astrometry"].get("stats")
    if a:
        txt = (f"{a['n_obs']:,} positions of catalogued detections against the catalogue: 68% within {a['p68']}″, 95% within "
               f"{a['p95']}″ (a WISE pixel is {P.WISE_PIXEL}″). Median offset ΔRA {a['median_dra']:+.2f}″, ΔDec {a['median_ddec']:+.2f}″")
        big = max(abs(a["median_dra"]), abs(a["median_ddec"]))
        txt += (", a systematic offset worth correcting before an MPC submission." if big >= 0.15 else ", no significant systematic offset.")
        out.append(("Astrometry", txt))
    sc = d["scores"]
    if sc.get("not_reviewed"):
        out.append(("Review", f"{sc['not_reviewed']:,} detections aren't reviewed yet; from the fraction of the reviewed judged "
                              f"real at each score, about {sc['expected_real']:,} of them are real."))
    gb = d["galactic"]["bins"]
    if gb:
        low = [b for b in gb if b["hi"] <= 5]
        rest = [b for b in gb if b["lo"] >= 5]
        if low and rest and sum(b["visits"] for b in low) >= 5:
            r_low = sum(b["n_new"] for b in low) / sum(b["visits"] for b in low)
            r_rest = sum(b["n_new"] for b in rest) / max(1, sum(b["visits"] for b in rest))
            if r_rest and r_low > 1.5 * r_rest:
                out.append(("Galactic plane", f"Within 5° of the galactic plane there are {r_low:.2f} detections not catalogued "
                                              f"per visit against {r_rest:.2f} elsewhere: crowded star fields make false "
                                              "tracks, so the ones there need the most scrutiny."))
    br = d["brightness"]
    if br.get("objects") and br.get("edges"):
        e, n = np.array(br["edges"]), np.array(br["catalogued"])
        if n.sum() >= 20:
            cum = np.cumsum(n) / n.sum()
            bright = e[1:][np.argmax(cum >= 0.01)]
            odd = [o for o in br["objects"] if o["mag"] is not None and o["mag"] < bright]
            if odd:
                out.append(("Bright detections not catalogued",
                            ", ".join(f"#{o['id']} (W3 {o['mag']:.1f})" for o in odd) +
                            f" {'is' if len(odd) == 1 else 'are'} brighter than 99% of the catalogued asteroids found "
                            f"(W3 {bright:.1f}): an asteroid that bright would almost surely be catalogued, so check "
                            "for a saturated star or an artefact."))
    return out


class _Doc:
    """Pages of text, tables and images, top to bottom."""

    def __init__(self, pdf, plt, title):
        self.pdf, self.plt, self.title, self.fig, self.page = pdf, plt, title, None, 0

    def new(self):
        self.close()
        self.page += 1
        self.fig = self.plt.figure(figsize=A4)
        self.y = A4[1] - MARGIN
        self.fig.text(MARGIN / A4[0], 0.4 / A4[1], self.title, fontsize=7, color=MUTE)
        self.fig.text(1 - MARGIN / A4[0], 0.4 / A4[1], str(self.page), fontsize=7, color=MUTE, ha="right")

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            self.plt.close(self.fig)
            self.fig = None

    def need(self, inches):
        if self.fig is None or self.y - inches < MARGIN:
            self.new()

    def text(self, s, size=9, color=INK, weight="normal", gap=0.06, indent=0.0, wrap=WRAP, family=None):
        lines = textwrap.wrap(s, int(wrap * 9 / size) - int(indent * 14)) or [""]
        h = size / 72 * 1.38
        for k, line in enumerate(lines):
            self.need(h)
            self.fig.text((MARGIN + indent) / A4[0], self.y / A4[1], line, fontsize=size, color=color,
                          weight=weight, va="top", family=family)
            self.y -= h
        self.y -= gap

    def rule(self):
        self.need(0.1)
        self.fig.add_artist(self.plt.Line2D([MARGIN / A4[0], 1 - MARGIN / A4[0]], [self.y / A4[1]] * 2,
                                            color=RULE, lw=0.6))
        self.y -= 0.1

    def table(self, head, rows, widths, size=7.5):
        h = size / 72 * 1.5
        xs = np.cumsum([0] + list(widths))
        scale = (A4[0] - 2 * MARGIN) / xs[-1]

        def row(vals, weight="normal", color=INK):
            self.need(h)
            for x, v in zip(xs, vals):
                self.fig.text((MARGIN + x * scale) / A4[0], self.y / A4[1], str(v), fontsize=size, va="top",
                              weight=weight, color=color)
            self.y -= h
        row(head, "bold")
        for r in rows:
            if self.y - h < MARGIN:
                self.new()
                row(head, "bold")
            row(r)
        self.y -= 0.1

    def image(self, path, max_h=None):
        img = self.plt.imread(str(path))
        w = A4[0] - 2 * MARGIN
        h = w * img.shape[0] / img.shape[1]
        if max_h and h > max_h:
            w, h = w * max_h / h, max_h
        self.need(h)
        ax = self.fig.add_axes([MARGIN / A4[0], (self.y - h) / A4[1], w / A4[0], h / A4[1]])
        ax.imshow(img, interpolation="none")  # embedded at its own resolution
        ax.axis("off")
        self.y -= h + 0.12


def _gallery(doc, objs, sheets, kinds=None):
    """Each detection not catalogued: a line about it, then its cutout sheet."""
    kinds = kinds or {}
    doc.new()
    doc.text(TITLES["gallery"], size=12, weight="bold")
    doc.text(CAPTIONS["gallery"] + " Upper row: each frame; lower row: the other frames' median subtracted; green: the "
             "frames stacked along the track.", size=8.5, color=MUTE, gap=0.14)
    if not objs:
        doc.text("None yet.")
    for o in objs:
        line = (f"#{o['id']} · {o['verdict']} · {_date(o['t_ref'])} UTC · RA {o['ra']:.4f} Dec {o['dec']:+.4f} · "
                f"{o['rate']:.3f}°/day at PA {o['pa']:.0f}° · arc {_arc(o.get('arc_arcsec'), o.get('span_hours'))} · "
                f"W3 {o['mag']:.2f} · score {o['score']:.1f}" + (f" · {o['sigma']:.0f}σ" if o.get("sigma") is not None else "")
                + (f" · near {o['near']}".replace('"', "″")
                   + (f" ({P.class_label(kinds[o['id']])})" if kinds.get(o["id"]) else "") if o.get("near") else ""))
        path = sheets.get(o["id"])
        if not path or not Path(path).exists():
            doc.text(line + " (no cutout sheet)", size=8, gap=0.12)
            continue
        img = doc.plt.imread(str(path))
        h = min((A4[0] - 2 * MARGIN) * img.shape[0] / img.shape[1], 3.4)
        doc.need(h + 0.3)
        doc.text(line, size=8, weight="bold", gap=0.04)
        doc.image(path, max_h=h)


def write_report(out, path=None, include_likely: bool = True, db=None, data=None, lookup: bool = True) -> Path:
    """The report as a PDF at ``path`` (default ``OUT/report.pdf``)."""
    from matplotlib.backends.backend_pdf import PdfPages
    out = Path(out)
    db = db or C.connect(out)
    if lookup and data is None:
        P.fetch_discoveries(out, db, include_likely)
    d = data or P.data(out, db, include_likely)
    st = C.status_summary(out, db=db)
    path = Path(path) if path else out / "report.pdf"
    plt = P._plt()
    now = datetime.now().astimezone()
    which = "confirmed and likely" if include_likely else "confirmed"
    with tempfile.TemporaryDirectory() as tmp, PdfPages(path) as pdf:
        info = pdf.infodict()
        info["Title"], info["Subject"] = "NEOfind W3+W4 campaign report", f"Campaign {out}"
        doc = _Doc(pdf, plt, f"NEOfind W3+W4 campaign report · {now:%Y-%m-%d %H:%M %Z}")
        doc.new()
        doc.text("NEOfind W3+W4 campaign report", size=17, weight="bold", gap=0.04)
        doc.text(f"{out} · made {now:%Y-%m-%d %H:%M %Z} · {which} detections", size=8.5, color=MUTE, gap=0.16)
        doc.text("Summary", size=12, weight="bold")
        rv = st.get("review", {})
        visits = st["visits"].get("done", 0)
        hours = (db.execute("SELECT SUM(seconds) FROM visits WHERE status='done'").fetchone()[0] or 0) / 3600
        b = st.get("bands") or {}
        row = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
        lat_max = json.loads(row[0]).get("lat_max", 10.0) if row else 10.0
        cover = (f" ({b['near']['visits_done']} within {lat_max:g}° of the ecliptic, "
                 f"{b['wide']['visits_done']} beyond it)") if b else ""
        n_cand = sum(st["candidates"].values())
        doc.text(f"{visits:,} WISE visits searched{cover}, on {sum(st['fields'].values()) - st['fields'].get('pending', 0):,} "
                 f"fields, in {hours:,.0f} hours of searching. {n_cand:,} detections passed the W3+W4 search; "
                 f"{rv.get('confirmed', 0):,} are confirmed, {rv.get('likely', 0):,} likely and {rv.get('rejected', 0):,} "
                 f"discarded in the blind review, and {rv.get('to review', 0):,} are still to review.")
        if st.get("projected_active_hours"):
            doc.text(f"About {st['remaining_visits']:,} visits are left near the ecliptic, roughly "
                     f"{st['projected_active_hours']:,} more hours of searching at the present rate.", gap=0.16)
        doc.text("Significant findings", size=12, weight="bold")
        for k, (head, txt) in enumerate(findings(d, st, include_likely), 1):
            doc.text(f"{k}. {head}", size=9.5, weight="bold", gap=0.02)
            doc.text(txt, indent=0.18, gap=0.1)

        # the detections not catalogued
        mot = d["motion"]
        new = sorted((o for o in mot["objects"] if o["catalogue"] == "not catalogued"),
                     key=lambda o: -(o.get("sigma") or 0))
        doc.new()
        doc.text(f"Detections not catalogued ({len(new)})", size=12, weight="bold")
        doc.text("Confirmed or likely in the blind review, with no catalogued asteroid there; the most unusual motion first "
                 "(σ: distance from the catalogued main belt's motion). Arc: first to last frame the track was measured on; "
                 "fr.: frames. Near: a catalogued asteroid close by but too far to be the detection, with its SkyBoT "
                 "orbit class.",
                 size=8.5, color=MUTE, gap=0.1)
        near_kind = P.skybot_classes(db, [o["id"] for o in new if o.get("near")])

        def near(o):
            if not o.get("near"):
                return ""
            k = P.class_label(near_kind.get(o["id"]))
            return o["near"].replace('"', "″") + (f" ({k})" if k else "")
        if new:
            doc.table(("#", "verdict", "image (UTC)", "RA", "Dec", "W3", "°/day", "PA", "arc / span", "fr.",
                       "score", "σ", "near (catalogued; type)"),
                      [(o["id"], o["verdict"], _date(o["t_ref"]), f"{o['ra']:.4f}", f"{o['dec']:+.4f}",
                        "–" if o["mag"] is None else f"{o['mag']:.2f}", f"{o['rate']:.3f}", f"{o['pa']:.0f}",
                        _arc(o.get("arc_arcsec"), o.get("span_hours")), o.get("n_frames") or "–", f"{o['score']:.1f}",
                        "–" if o.get("sigma") is None else f"{o['sigma']:.1f}", near(o)[:36])
                       for o in new],
                      (3.4, 7.0, 11.2, 6.8, 6.8, 3.9, 4.5, 3.1, 8.8, 3.0, 4.4, 3.4, 21.6), size=6.8)
        else:
            doc.text("None yet.")

        # the sky map: every detection, then those not catalogued
        from .skymap import write_skymap
        doc.new()
        doc.text("Sky map", size=12, weight="bold")
        doc.text(f"Where the {which} detections are on the sky (equatorial coordinates, Hammer projection, RA increasing "
                 "to the left), over the fields searched (grey squares); solid line: the ecliptic, dotted: the edge of "
                 "the band near it. Filled: confirmed, rings: likely.", size=8.5, color=MUTE, gap=0.12)
        half = (A4[1] - 2 * MARGIN - 1.2) / 2
        for only in (False, True):
            png = write_skymap(out, Path(tmp) / f"skymap{int(only)}.png", only, include_likely)
            doc.image(png, max_h=half)

        # the plots
        for name in ORDER:
            if name == "gallery":
                _gallery(doc, d["gallery"], P.sheets(db, [o["id"] for o in d["gallery"]]),
                         P.skybot_classes(db, [o["id"] for o in d["gallery"] if o.get("near")]))
                continue
            png = Path(tmp) / f"{name}.png"
            try:
                P.render(name, d, db, png)
            except Exception as e:  # one plot failing doesn't lose the report
                doc.new()
                doc.text(TITLES[name], size=12, weight="bold")
                doc.text(f"(could not be drawn: {e})", color=MUTE)
                continue
            cap = CAPTIONS[name].format(min_frames=d["completeness"]["min_frames"],
                                        snr_floor=d["completeness"]["snr_floor"])
            img = plt.imread(str(png))
            full = A4[1] - 2 * MARGIN - 0.9  # the page less a title and caption
            h = min((A4[0] - 2 * MARGIN) * img.shape[0] / img.shape[1], full)
            text_h = 0.45 + 0.17 * (len(textwrap.wrap(cap, int(WRAP * 9 / 8.5))) + 1)
            if name == ORDER[0] or doc.y - (text_h + h) < MARGIN:
                doc.new()  # starts a page unless it fits under the one before
            else:
                doc.y -= 0.2
            doc.text(TITLES[name], size=12, weight="bold")
            doc.text(cap, size=8.5, color=MUTE, gap=0.12)
            doc.image(png, max_h=min(full, doc.y - MARGIN))

        # seen before discovery
        dl = d.get("discovery") or {}
        if dl.get("objects"):
            doc.new()
            doc.text(f"Catalogued asteroids seen by WISE before discovery ({dl['before']})", size=12, weight="bold")
            doc.text("Years before: discovery date minus the date of the earliest WISE image the asteroid was found on. "
                     "Type: SkyBoT's orbit class. Date from: the credited discovery (JPL's Small-Body Database) or, with none, "
                     "the provisional designation.",
                     size=8.5, color=MUTE, gap=0.1)
            kinds = P.skybot_classes(db, [o["id"] for o in dl["objects"]])
            doc.table(("#", "asteroid", "type", "WISE image", "discovered", "yrs before", "date from", "discoverer"),
                      [(o["id"], o["known_name"][:26], P.class_label(kinds.get(o["id"]))[:22] or "–", o["image"],
                        o["discovered"], f"{o['lag']:.1f}", "discovery" if o["source"] == "discovery" else "designation",
                        (o.get("who") or "")[:26])
                       for o in dl["objects"]],
                      (4, 15, 12, 8.4, 8.4, 6.6, 8.6, 15), size=7)

        doc.new()
        doc.text("How the results were made", size=12, weight="bold")
        for t in ("The campaign searches WISE's 2010 W3 (12 µm) single-exposure frames of each visit for moving objects with a "
                  "matched filter along trial tracks, W3 and W4 (22 µm) together, with W4 as a check on each track "
                  "where W4 frames exist.",
                  "Every detection gets a blind review (confirm, likely, or discard) from its cutout sheet, before its "
                  "catalogue match is shown. Catalogued asteroids come from SkyBoT, their positions shifted from the Earth "
                  "to the spacecraft.",
                  "Discovery dates come from JPL's Small-Body Database (only the asteroid's number is sent). An unnumbered "
                  "asteroid has no credited discovery; the date its provisional designation encodes (year and half month) "
                  "stands in, good to about a week.",
                  "Making this report sends nothing to the Minor Planet Center; submissions are made separately "
                  "(the review page's Export for MPC)."):
            doc.text(t, gap=0.1)
        doc.close()
    return path
