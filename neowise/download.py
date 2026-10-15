"""Download WISE L1b frame triplets (``int``, ``unc``, ``msk``) -- whole
frames or square cutouts around a position -- into a visit directory that
``neodetect`` reads directly.

Files keep their IRSA names (``<scan><frame>-w1-int-1b.fits``,
``-unc-1b.fits.gz``, ``-msk-1b.fits.gz``) so the loader pairs them. Cutouts
keep the full header with CRPIX shifted, so the SIP WCS stays exact.
Existing non-empty files are not downloaded again.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from .query import FrameInfo, http_get

KINDS = ("int", "unc", "msk")


def fetch_file(url: str, dest: Path) -> Path:
    """Download to ``dest`` (a ``.fits`` name); the saved name gets ``.gz``
    when the server sends gzip (IRSA does for cutouts and unc/msk files)."""
    base = dest.name[:-3] if dest.name.endswith(".gz") else dest.name
    for cand in (dest.with_name(base), dest.with_name(base + ".gz")):
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    data = http_get(url)
    if data[:2] == b"\x1f\x8b":
        dest = dest.with_name(base + ".gz")
    elif data.startswith(b"SIMPLE"):
        dest = dest.with_name(base)
    else:
        raise RuntimeError(f"{url}: expected FITS or gzip data, got {data[:80]!r}")
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return dest


def frame_urls(f: FrameInfo, center=None, size_deg: Optional[float] = None):
    """(kind, url) for one frame; a cutout when ``center``/``size_deg`` given."""
    q = f"?center={center[0]:.6f},{center[1]:.6f}&size={size_deg}deg" if size_deg else ""
    return [(k, f.url(k) + q) for k in KINDS]


def download_visit(frames: List[FrameInfo], out_dir, center=None, size_deg: Optional[float] = None,
                   workers: int = 4, log=None) -> Path:
    """Fetch every frame's triplet into ``out_dir`` and write ``visit.json``."""
    log = log or (lambda *a: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(url, out / f.filename(k)) for f in frames for k, url in frame_urls(f, center, size_deg)]
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(lambda j: fetch_file(*j), jobs), 1):
            if i % 12 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} files")
    meta = {"center": list(center) if center else None, "size_deg": size_deg,
            "frames": [f.to_dict() for f in frames]}
    (out / "visit.json").write_text(json.dumps(meta, indent=1))
    return out
