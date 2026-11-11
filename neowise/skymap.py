"""Sky map of a campaign's confirmed asteroids: ``neowise skymap OUT`` (a
PNG) and the review page's Sky map tab (``points``, drawn in the browser).

Hammer-Aitoff, equatorial, RA 180 deg in the middle and increasing to the
left, as on the sky. Each detection with a verdict of confirmed (and, if
asked, likely) is a dot at its position at the visit's reference time:
orange when no catalogued asteroid is there, blue when one is; likely ones
are rings. Grey: the fields searched so far. The ecliptic and the edges
of the campaign's band are drawn for orientation.
"""

import json
import math
from pathlib import Path

import numpy as np

OBLIQUITY = 23.4392911  # deg, J2000
# Categorical slots 1 and 2 of the reference palette, light surface
COLORS = {"catalogued": "#2a78d6", "not catalogued": "#eb6834"}
SEARCHED = "#d6d4cc"


def points(db, uncatalogued_only=False, include_likely=True) -> dict:
    """The map's data: the objects (confirmed, and likely), the fields
    searched and the band's half-width in ecliptic latitude."""
    verdicts = ("confirmed", "likely") if include_likely else ("confirmed",)
    q = (f"SELECT id, ra, dec, verdict, status, known_name, near, rate, t_ref, mag FROM candidates "
         f"WHERE verdict IN ({','.join('?' * len(verdicts))})")
    if uncatalogued_only:
        q += " AND COALESCE(status, '') != 'known'"
    objects = []
    for r in db.execute(q + " ORDER BY id", verdicts).fetchall():
        o = dict(r)
        o["catalogue"] = "catalogued" if o["status"] == "known" else "not catalogued"
        objects.append(o)
    searched = [list(r) for r in db.execute(
        "SELECT ra, dec FROM fields WHERE id IN (SELECT field_id FROM visits WHERE status='done')")]
    r = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
    lat = json.loads(r[0]).get("lat_max") if r else None  # the near band's edge, not the widest field's
    if lat is None:
        lat = db.execute("SELECT MAX(ABS(elat)) FROM fields").fetchone()[0]
    return {"objects": objects, "searched": searched, "lat_max": round(lat, 1) if lat is not None else None}


def ecliptic_line(beta_deg: float, n: int = 361):
    """(ra, dec) in degrees along ecliptic latitude ``beta_deg``."""
    lam = np.radians(np.linspace(0, 360, n))
    b, e = math.radians(beta_deg), math.radians(OBLIQUITY)
    dec = np.arcsin(np.sin(b) * math.cos(e) + np.cos(b) * math.sin(e) * np.sin(lam))
    ra = np.arctan2(np.sin(lam) * math.cos(e) - math.tan(b) * math.sin(e), np.cos(lam))
    return np.degrees(ra) % 360, np.degrees(dec)


def _lon(ra):
    """Matplotlib's Hammer longitude (radians): RA 180 in the middle, increasing to the left."""
    return -np.radians((np.asarray(ra, float) - 180.0 + 180.0) % 360.0 - 180.0)


def plot(data: dict, path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6.6), dpi=150)
    ax = fig.add_subplot(111, projection="hammer")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.set_xticks(np.radians(np.arange(-150, 181, 30)))
    ax.set_xticklabels([f"{(180 - x) % 360:g}°" for x in range(-150, 181, 30)], fontsize=8, color="#52514e")
    ax.tick_params(axis="y", labelsize=8, labelcolor="#52514e")
    if data["searched"]:
        s = np.array(data["searched"])
        ax.scatter(_lon(s[:, 0]), np.radians(s[:, 1]), s=14, c=SEARCHED, marker="s", lw=0, alpha=0.6,
                   label="fields searched", zorder=1)
    for beta, style in [(0.0, "-")] + ([(data["lat_max"], ":"), (-data["lat_max"], ":")] if data["lat_max"] else []):
        ra, dec = ecliptic_line(beta)
        x, y = _lon(ra), np.radians(dec)
        cut = np.where(np.abs(np.diff(x)) > np.pi)[0] + 1  # don't draw across the map's edge
        for k, (xs, ys) in enumerate(zip(np.split(x, cut), np.split(y, cut))):
            ax.plot(xs, ys, style, color="#52514e", lw=1.0 if beta == 0 else 0.8, zorder=2,
                    label=("ecliptic" if beta == 0 else f"ecliptic latitude ±{data['lat_max']:g}°") if k == 0 and beta >= 0 else None)
    for cat, color in COLORS.items():
        for verdict, filled in (("confirmed", True), ("likely", False)):
            sel = [o for o in data["objects"] if o["catalogue"] == cat and o["verdict"] == verdict]
            if not sel:
                continue
            ax.scatter(_lon([o["ra"] for o in sel]), np.radians([o["dec"] for o in sel]), s=26 if filled else 30,
                       facecolors=color if filled else "none", edgecolors="white" if filled else color,
                       linewidths=0.8 if filled else 1.4, zorder=4 if cat == "not catalogued" else 3,
                       label=f"{verdict}, {cat} ({len(sel)})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=11, pad=18, color="#0b0b0b")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.9, bottom=0.18)
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_skymap(out, path=None, uncatalogued_only=False, include_likely=True) -> Path:
    from . import campaign as C
    db = C.connect(out)
    C.fill_known_names(db)
    data = points(db, uncatalogued_only, include_likely)
    what = ("confirmed and likely" if include_likely else "confirmed") + (", not catalogued" if uncatalogued_only else "")
    path = path or Path(out) / ("skymap_uncatalogued.png" if uncatalogued_only else "skymap.png")
    n = len(data["objects"])
    return plot(data, path, title=f"NEOfind: {n} asteroid detection{'s' * (n != 1)}, {what}; "
                                  f"{len(data['searched'])} fields searched")
