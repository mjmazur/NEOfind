"""Find WISE/NEOWISE single-exposure (L1b) frames covering a sky position via
IRSA's Image Server (IBE), and group them into visits.

IBE metadata tables (one per mission phase):

    allsky      wise/allsky/4band_p1bm_frm       2010-01 .. 2010-08  W1-W4
    cryo_3band  wise/cryo_3band/3band_p1bm_frm   2010-08 .. 2010-09  W1-W3
    postcryo    wise/postcryo/2band_p1bm_frm     2010-09 .. 2011-02  W1-W2
    neowiser    wise/neowiser/p1bm_frm           2013-12 .. 2024-08  W1-W2

A point near the ecliptic is seen on ~12 consecutive orbits (95 min apart)
every ~6 months; such a run of orbits is a *visit*, the unit the detector
searches. Successive frames along one scan overlap slightly, so one orbit
can contribute two frames to a position; ``one_per_orbit`` keeps the frame
whose center is nearest the target.
"""

import csv
import io
import math
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

IBE = "https://irsa.ipac.caltech.edu/ibe"
TABLES = {
    "allsky": "wise/allsky/4band_p1bm_frm",
    "cryo_3band": "wise/cryo_3band/3band_p1bm_frm",
    "postcryo": "wise/postcryo/2band_p1bm_frm",
    "neowiser": "wise/neowiser/p1bm_frm",
}
# Default mission phases per band: NEOWISE-R for W1/W2; the 4-band
# cryogenic survey (2010-01-07 .. 2010-08-06) is the only one with W4 and,
# with the 3-band phase after it, the only one with W3.
DEFAULT_TABLES = {1: ("neowiser",), 2: ("neowiser",), 3: ("allsky",), 4: ("allsky",)}
ORBIT_DAYS = 94.8 / 1440.0
USER_AGENT = "NEOfind/0.3 (moving-object search)"


@dataclass
class FrameInfo:
    table: str  # key of TABLES
    scan_id: str
    scangrp: str
    frame_num: int
    band: int
    mjd: float  # mid-exposure
    ra: float  # frame center
    dec: float
    magzp: float
    qual_frame: int
    moon_sep: float
    saa_sep: float
    corners: tuple = ()  # ((ra, dec) x 4) footprint

    @property
    def frame_id(self) -> str:
        return f"{self.scan_id}{self.frame_num:03d}"

    def url(self, kind: str) -> str:
        """Data URL of the ``int``, ``unc`` or ``msk`` image."""
        ext = ".fits" if kind == "int" else ".fits.gz"
        return (f"{IBE}/data/{TABLES[self.table]}/{self.scangrp}/{self.scan_id}/"
                f"{self.frame_num:03d}/{self.frame_id}-w{self.band}-{kind}-1b{ext}")

    def filename(self, kind: str) -> str:
        return self.url(kind).rsplit("/", 1)[1]

    def to_dict(self) -> dict:
        return asdict(self)


def http_get(url: str, timeout: float = 120.0, retries: int = 6) -> bytes:
    """GET with retries and exponential backoff. IRSA answers 503 when too
    many requests arrive at once, and occasionally times out."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"GET {url} failed: {e}") from e
            if e.code == 500 and b"does not overlap" in (e.read() or b""):
                raise RuntimeError(f"GET {url} failed: cutout does not overlap image") from e
        except Exception as e:  # URLError, socket timeout, ...
            if attempt == retries - 1:
                raise RuntimeError(f"GET {url} failed: {e}") from e
        time.sleep(3.0 * 2 ** attempt)


def search(ra: float, dec: float, band: int = 1, tables: Sequence[str] = ("neowiser",),
           mjd_min: Optional[float] = None, mjd_max: Optional[float] = None,
           min_qual: int = 1, size_deg: float = 0.0,
           min_overlap: float = 0.1) -> List[FrameInfo]:
    """Frames of ``band`` whose footprint contains (ra, dec) or, with
    ``size_deg`` > 0, covers at least ``min_overlap`` of that square around
    it; sorted by time."""
    where = [f"band={int(band)}", f"qual_frame>={int(min_qual)}"]
    if mjd_min is not None:
        where.append(f"mjd_obs>={mjd_min}")
    if mjd_max is not None:
        where.append(f"mjd_obs<={mjd_max}")
    out = []
    for t in tables:
        params = {"POS": f"{ra},{dec}", "ct": "csv", "where": " and ".join(where)}
        if size_deg > 0:
            params.update(SIZE=f"{size_deg}", INTERSECT="OVERLAPS")
        q = urllib.parse.urlencode(params)
        text = http_get(f"{IBE}/search/{TABLES[t]}?{q}").decode()
        for row in csv.DictReader(io.StringIO(text)):
            out.append(FrameInfo(
                table=t, scan_id=row["scan_id"], scangrp=row["scangrp"],
                frame_num=int(row["frame_num"]), band=int(row["band"]),
                mjd=float(row["mjd_obs"]), ra=float(row["crval1"]), dec=float(row["crval2"]),
                magzp=float(row["magzp"]), qual_frame=int(row["qual_frame"]),
                moon_sep=float(row["moon_sep"] or "nan"), saa_sep=float(row["saa_sep"] or "nan"),
                corners=tuple((float(row[f"ra{i}"]), float(row[f"dec{i}"])) for i in range(1, 5))))
    if size_deg > 0:  # IBE's OVERLAPS is generous; keep frames that really overlap
        out = [f for f in out if overlap_fraction(f, ra, dec, size_deg) >= min_overlap]
    return sorted(out, key=lambda f: f.mjd)


def _sep_deg(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _tangent(ra, dec, ra0, dec0):
    """Gnomonic projection about (ra0, dec0), degrees."""
    r, d, r0, d0 = (np.radians(np.asarray(v, float)) for v in (ra, dec, ra0, dec0))
    c = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0)
    xi = np.cos(d) * np.sin(r - r0) / c
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(r - r0)) / c
    return np.degrees(xi), np.degrees(eta)


def overlap_fraction(f: FrameInfo, ra: float, dec: float, size_deg: float, n: int = 24) -> float:
    """Fraction of the square region (side ``size_deg`` about ra, dec) that
    the frame's footprint covers, by sampling."""
    cx, cy = _tangent([c[0] for c in f.corners], [c[1] for c in f.corners], ra, dec)
    g = (np.arange(n) + 0.5) / n * size_deg - size_deg / 2
    px, py = (a.ravel() for a in np.meshgrid(g, g))
    # Inside a convex polygon: same sign of the cross product on every edge.
    sides = []
    for i in range(4):
        x0, y0, x1, y1 = cx[i], cy[i], cx[(i + 1) % 4], cy[(i + 1) % 4]
        sides.append((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0))
    sides = np.array(sides)
    inside = (sides >= 0).all(axis=0) | (sides <= 0).all(axis=0)
    return float(inside.mean())


def one_per_orbit(frames: List[FrameInfo], ra: float, dec: float) -> List[FrameInfo]:
    """Of frames taken within the same orbit pass (< 0.3 orbit apart), keep
    the one centered nearest (ra, dec)."""
    out: List[FrameInfo] = []
    for f in frames:
        if out and f.mjd - out[-1].mjd < 0.3 * ORBIT_DAYS:
            if _sep_deg(f.ra, f.dec, ra, dec) < _sep_deg(out[-1].ra, out[-1].dec, ra, dec):
                out[-1] = f
            continue
        out.append(f)
    return out


def group_visits(frames: List[FrameInfo], gap_days: float = 0.5,
                 max_span_days: Optional[float] = None) -> List[List[FrameInfo]]:
    """Split a time-sorted frame list at gaps > ``gap_days``; optionally cut
    long visits (high ecliptic latitude) into consecutive pieces no longer
    than ``max_span_days`` (the detector assumes linear motion)."""
    visits: List[List[FrameInfo]] = []
    for f in frames:
        if (not visits or f.mjd - visits[-1][-1].mjd > gap_days
                or (max_span_days is not None and f.mjd - visits[-1][0].mjd > max_span_days)):
            visits.append([])
        visits[-1].append(f)
    return visits
