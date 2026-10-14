"""Load an image sequence: FITS frames with WCS and observation times, plus
optional uncertainty and mask images.

Accepted inputs are directories, files, or glob patterns. Within a directory,
files whose name contains ``-int`` are the images (simulator and WISE L1b
layout: ``X-int...``, ``X-unc...``, ``X-msk...``); companions are found by
substituting ``-unc``/``-msk`` and trying the ``.fits``, ``.fits.gz`` and
``.fits.fz`` extensions. If no ``-int`` files exist, every FITS file is an
image with no companions.

A simulator run directory also provides ``truth.json`` and
``metadata.json``.
"""

import glob
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
import warnings

FITS_EXTS = (".fits", ".fits.gz", ".fits.fz", ".fit", ".fts")
SAT_FRACTION = 0.98  # pixels above this fraction of SATURATE are treated as saturated


@dataclass
class Frame:
    path: str
    data: np.ndarray  # float32, DN
    unc: Optional[np.ndarray]  # float32 1-sigma, DN (None: estimate from data)
    bad: np.ndarray  # bool, True = unusable
    wcs: WCS
    mjd: float  # mid-exposure
    exptime: Optional[float]
    magzp: Optional[float]
    psf_fwhm_arcsec: Optional[float]
    saturate: Optional[float]
    band: Optional[str]
    header: fits.Header = field(repr=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.wcs.pixel_scale_matrix))) * 3600.0)


@dataclass
class Sequence:
    frames: List[Frame]
    truth: Optional[dict] = None
    metadata: Optional[dict] = None
    source: Optional[str] = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def times(self) -> np.ndarray:
        return np.array([f.mjd for f in self.frames])


def _read_image(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    return np.asarray(hdu.data), hdu.header.copy()
    raise ValueError(f"{path}: no 2-D image HDU")


def _companion(path: Path, kind: str) -> Optional[Path]:
    name = path.name
    if "-int" not in name:
        return None
    stem = name.replace("-int", f"-{kind}", 1)
    base = stem
    for ext in FITS_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    for ext in FITS_EXTS:
        cand = path.with_name(base + ext)
        if cand.exists():
            return cand
    return None


def observation_mjd(h: fits.Header) -> float:
    """Mid-exposure MJD from whichever time keywords the header carries."""
    exptime = h.get("EXPTIME")
    half = 0.5 * exptime / 86400.0 if exptime else 0.0
    if "MJD-MID" in h:
        return float(h["MJD-MID"])
    if "MJD_OBS" in h:  # WISE L1b; treated as mid-exposure (check in milestone 3)
        return float(h["MJD_OBS"])
    if "MJD-OBS" in h:
        return float(h["MJD-OBS"]) + half
    for key in ("DATE-OBS", "DATE_OBS"):
        if key in h:
            return float(Time(h[key], scale="utc").mjd) + half
    raise ValueError("no observation time (MJD-MID, MJD_OBS, MJD-OBS or DATE-OBS) in header")


def load_frame(path, bad_bits: Optional[int] = None) -> Frame:
    path = Path(path)
    data, h = _read_image(path)
    data = data.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(h)
    if not wcs.has_celestial:
        raise ValueError(f"{path}: no celestial WCS")

    unc = None
    unc_path = _companion(path, "unc")
    if unc_path is not None:
        unc = _read_image(unc_path)[0].astype(np.float32)

    bad = ~np.isfinite(data)
    msk_path = _companion(path, "msk")
    if msk_path is not None:
        msk = _read_image(msk_path)[0].astype(np.int64)
        bits = bad_bits if bad_bits is not None else -1  # default: any set bit
        bad |= (msk & bits) != 0
    if unc is not None:
        bad |= ~np.isfinite(unc) | (unc <= 0)
    saturate = h.get("SATURATE")
    if saturate:
        bad |= data >= SAT_FRACTION * float(saturate)
    data[bad] = 0.0
    if unc is not None:
        unc[bad] = 0.0

    band = h.get("BAND")
    return Frame(path=str(path), data=data, unc=unc, bad=bad, wcs=wcs.celestial,
                 mjd=observation_mjd(h), exptime=h.get("EXPTIME"), magzp=h.get("MAGZP"),
                 psf_fwhm_arcsec=h.get("PSFFWHM"), saturate=saturate,
                 band=str(band) if band is not None else None, header=h)


def _is_fits(p: Path) -> bool:
    return any(p.name.lower().endswith(ext) for ext in FITS_EXTS)


def find_images(inputs) -> List[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    found = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files = sorted(q for q in p.iterdir() if _is_fits(q))
        elif p.exists():
            files = [p]
        else:
            files = sorted(Path(g) for g in glob.glob(str(item)) if _is_fits(Path(g)))
        ints = [q for q in files if "-int" in q.name]
        if ints:
            files = ints
        else:
            files = [q for q in files if not re.search(r"-(unc|msk)", q.name)]
        found += files
    return found


def load_sequence(inputs, bad_bits: Optional[int] = None) -> Sequence:
    paths = find_images(inputs)
    if not paths:
        raise ValueError(f"no FITS images found in {inputs}")
    frames = sorted((load_frame(p, bad_bits) for p in paths), key=lambda f: f.mjd)
    truth = metadata = None
    source = None
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        source = str(inputs)
        if (Path(inputs) / "truth.json").exists():
            truth = json.loads((Path(inputs) / "truth.json").read_text())
        if (Path(inputs) / "metadata.json").exists():
            metadata = json.loads((Path(inputs) / "metadata.json").read_text())
    return Sequence(frames=frames, truth=truth, metadata=metadata, source=source)


def default_psf_fwhm_arcsec(frame: Frame) -> float:
    """PSF FWHM when the header has none: WISE band values, else 2 pixels."""
    if frame.psf_fwhm_arcsec:
        return float(frame.psf_fwhm_arcsec)
    wise = {"1": 6.1, "W1": 6.1, "2": 6.4, "W2": 6.4, "3": 6.5, "W3": 6.5, "4": 12.0, "W4": 12.0}
    if frame.band is not None and frame.band.upper() in wise:
        return wise[frame.band.upper()]
    return 2.0 * frame.pixel_scale_arcsec


def robust_sigma(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    med = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - med)))
