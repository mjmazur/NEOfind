"""Load an image sequence: FITS frames with WCS and observation times, plus
optional uncertainty and mask images.

Accepted inputs are directories, files, or glob patterns. Within a directory,
files whose name contains ``-int`` are the images (simulator and WISE L1b
layout: ``X-int...``, ``X-unc...``, ``X-msk...``); companions are found by
substituting ``-unc``/``-msk`` and trying the ``.fits``, ``.fits.gz`` and
``.fits.fz`` extensions. If no ``-int`` files exist, every FITS file is an
image with no companions.

A simulator run directory also provides ``truth.json`` and
``metadata.json``; a ``neowise`` visit directory provides ``known.json``
(known solar-system objects, same format).

Raster images (PNG, TIFF, JPEG) are read when no FITS image is found. They
carry no times or WCS: times come from a sidecar -- ``times.csv`` in the
directory (``file,time`` per line, ISO UTC or MJD) or a ``<image>.json``
next to each image (``{"time": ...}``) -- and the WCS from star matching
(``starmatch``). An optional ``sequence.json`` gives ``pixel_scale_arcsec``,
``ra_deg``, ``dec_deg`` (the pointing), ``exptime_s``, ``fwhm_arcsec``,
``magzp`` and ``gain_e_per_adu``; without a FWHM it is measured from the
stars. Pixels at the format's maximum value are saturated.
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
from astropy.wcs import WCS, FITSFixedWarning, NoConvergence
import warnings

FITS_EXTS = (".fits", ".fits.gz", ".fits.fz", ".fit", ".fts")
RASTER_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
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
    known: Optional[dict] = None  # known.json from neowise: real objects, truth format

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def times(self) -> np.ndarray:
        return np.array([f.mjd for f in self.frames])


def world2pix(wcs: WCS, ra, dec):
    """``wcs.all_world2pix`` (0-based) that never raises. Far outside a frame
    the SIP distortion polynomial can make the iterative inverse diverge;
    those points get the linear (no-distortion) inverse, off by a few pixels
    at most, which is ample for positions that are off the frame anyway."""
    try:
        return wcs.all_world2pix(ra, dec, 0)
    except NoConvergence as e:
        sol = np.array(e.best_solution, dtype=float)
        bad = np.zeros(len(sol), bool)
        for idx in (e.divergent, e.slow_conv):
            if idx is not None:
                bad[np.asarray(idx, int)] = True
        bad |= ~np.all(np.isfinite(sol), axis=1)
        lx, ly = wcs.wcs_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0)
        sol[bad, 0], sol[bad, 1] = lx[bad], ly[bad]
        return sol[:, 0], sol[:, 1]


def read_image(path: Path):
    """(data, header) of the first 2-D image HDU (plain, gzip or tile-compressed FITS)."""
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
    if "MJD_OBS" in h:  # WISE L1b: "Obs. midpoint Modified Julian Day"
        return float(h["MJD_OBS"])
    if "MJD-OBS" in h:
        return float(h["MJD-OBS"]) + half
    for key in ("DATE-OBS", "DATE_OBS"):
        if key in h:
            return float(Time(h[key], scale="utc").mjd) + half
    raise ValueError("no observation time (MJD-MID, MJD_OBS, MJD-OBS or DATE-OBS) in header")


def default_bad_bits(h: fits.Header) -> int:
    """Mask bits that mark unusable pixels: WISE's definitions for WISE
    frames (see ``neowise.maskbits``), otherwise any set bit."""
    if _is_real_wise(h):
        from neowise.maskbits import BAD_BITS
        return BAD_BITS
    return -1


def _is_real_wise(h: fits.Header) -> bool:
    return str(h.get("TELESCOP", "")).strip().upper() == "WISE" and not h.get("SIMULATD")


def default_null_scrambles(frame: "Frame") -> int:
    """Time-scrambled copies ``neodetect detect`` runs (``nullcal``): 1 for
    real WISE frames, whose per-visit thresholds (the copy's null level x
    ``NULL_RATIO``) follow a false-alarm tail that varies by ~2x across the
    sky (Galactic plane, bulge), and which feeds the artifact veto; else 0.
    A second copy didn't help held out (30 visits)."""
    return 1 if _is_real_wise(frame.header) else 0


def default_skip_crowded(frame: "Frame") -> bool:
    """Skip the search in a crowded field (``DetectorConfig.crowded_frac``):
    on for real WISE frames. Over 30 real NEOWISE visits the five crowded
    ones took 1.5-3.5 h per run and found almost nothing (W1 16
    completeness 0.07 in the plane at best, none in the bulge)."""
    return _is_real_wise(frame.header)


# Fixed detection thresholds (3 / 4 / >= 5 frames) for real NEOWISE W1 when
# no scrambled copy is run (--null-scrambles 0): calibrated on the 22 quiet
# visits (median one-copy null level <= 7.5) of sims/real_v1 + real_v3 with
# static-source and extendedness vetting, so valid away from the Galactic plane only; plane
# and bulge fields need the per-visit thresholds (nullcal), the default.
REAL_WISE_THRESHOLDS = (9.75, 9.25, 9.25)


# The simulator's ``ground`` preset (sims/sweeps/ground_v1: 24 null runs,
# 0 false per run). Its frames are close in time, so fast chance alignments
# (up to the 10 deg/day linking limit) have more room than in WISE visits.
GROUND_SIM_THRESHOLDS = (11.75, 11.5, 9.75)


def default_thresholds(frame: "Frame"):
    """(3, 4, >= 5-frame) detection thresholds for this instrument, or None
    for ``DetectorConfig``'s (simulated WISE) defaults."""
    if _is_real_wise(frame.header):
        return REAL_WISE_THRESHOLDS
    h = frame.header
    if h.get("SIMULATD") and str(h.get("INSTRUME", "")).strip().lower() == "ground":
        return GROUND_SIM_THRESHOLDS
    return None


def default_aperture_ee(frame: "Frame") -> float:
    """Fraction of a point source's total flux within the photometry
    aperture (``photometry.aperture_radius``, ~3.6 FWHM). Real WISE W1/W2:
    0.959 within 8 px, from the official single-exposure PSF (Explanatory
    Supplement IV.4.c). Elsewhere (simulations, unknown instruments): 1."""
    return 0.959 if _is_real_wise(frame.header) else 1.0


# Real WISE W1: position of a Gaussian-fit centroid minus the NEOWISE
# single-exposure catalog's PSF-fit position (the convention the frame WCS
# is calibrated in), native px. Constant over the detector and over 16
# frames of 8 visits at every orientation to ~0.01 px; the real PSF is
# slightly asymmetric.
WISE_W1_CENTROID_OFFSET_PX = (0.147, 0.085)


def default_centroid_offset(frame: "Frame") -> tuple:
    """(dx, dy) native px to subtract from Gaussian-fit positions: the
    real-WISE W1 offset above, else (0, 0) (the Gaussian is the PSF in
    simulations; unknown otherwise)."""
    if _is_real_wise(frame.header) and str(frame.band or "").upper() in ("1", "W1"):
        return WISE_W1_CENTROID_OFFSET_PX
    return (0.0, 0.0)


def default_aperture_correction(frame: "Frame") -> float:
    """Gaussian-fit to total flux ratio used when a sequence has too few
    isolated stars to measure it. Real WISE W1: 1.15, the median over 15
    NEOWISE visits (1.145-1.21). Elsewhere: 1 (the Gaussian model is exact
    for simulations; unknown otherwise)."""
    return 1.15 if _is_real_wise(frame.header) else 1.0


def load_frame(path, bad_bits: Optional[int] = None) -> Frame:
    path = Path(path)
    data, h = read_image(path)
    data = data.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(h)
    if not wcs.has_celestial:
        raise ValueError(f"{path}: no celestial WCS")

    unc = None
    unc_path = _companion(path, "unc")
    if unc_path is not None:
        unc = read_image(unc_path)[0].astype(np.float32)

    bad = ~np.isfinite(data)
    msk_path = _companion(path, "msk")
    if msk_path is not None:
        msk = read_image(msk_path)[0].astype(np.int64)
        bits = bad_bits if bad_bits is not None else default_bad_bits(h)
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


def _is_raster(p: Path) -> bool:
    return p.suffix.lower() in RASTER_EXTS


def find_images(inputs) -> List[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    found = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files = sorted(q for q in p.iterdir() if _is_fits(q))
            if not files:
                files = sorted(q for q in p.iterdir() if _is_raster(q))
        elif p.exists():
            files = [p]
        else:
            files = sorted(Path(g) for g in glob.glob(str(item)) if _is_fits(Path(g)))
            if not files:
                files = sorted(Path(g) for g in glob.glob(str(item)) if _is_raster(Path(g)))
        ints = [q for q in files if "-int" in q.name]
        if ints:
            files = ints
        else:
            files = [q for q in files if not re.search(r"-(unc|msk)", q.name)]
        found += files
    return found


def _parse_time(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(Time(str(v).strip(), scale="utc").mjd)


def raster_times(paths: List[Path]) -> dict:
    """Image path -> MJD from ``times.csv`` (per directory) or ``<image>.json``."""
    out = {}
    for d in sorted({p.parent for p in paths}):
        tf = d / "times.csv"
        if tf.exists():
            for line in tf.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("file,"):
                    continue
                name, t = [x.strip() for x in line.split(",", 1)]
                out[str(d / name)] = _parse_time(t)
    for p in paths:
        if str(p) not in out:
            for side in (p.with_suffix(".json"), p.with_name(p.name + ".json")):
                if side.exists():
                    out[str(p)] = _parse_time(json.loads(side.read_text())["time"])
                    break
    missing = [p.name for p in paths if str(p) not in out]
    if missing:
        raise ValueError(f"no observation time for {missing[:3]} (times.csv or <image>.json)")
    return out


def load_raster_frame(path, mjd: float, meta: dict) -> Frame:
    """A PNG/TIFF/JPEG frame (no WCS yet; ``starmatch`` assigns one)."""
    import imageio.v3 as iio
    path = Path(path)
    raw = np.asarray(iio.imread(path))
    top = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else None
    if raw.ndim == 3:  # colour: luminance (alpha dropped); saturated if any channel is
        sat = np.any(raw[..., :3] >= top, axis=2) if top is not None else np.zeros(raw.shape[:2], bool)
        data = (raw[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
                if raw.shape[2] >= 3 else raw[..., 0].astype(np.float32))
    else:
        data = raw.astype(np.float32)
        sat = data >= top if top is not None else np.zeros(data.shape, bool)
    bad = ~np.isfinite(data) | sat
    data[bad] = 0.0
    h = fits.Header()
    for key, k in (("EXPTIME", "exptime_s"), ("MAGZP", "magzp"), ("PSFFWHM", "fwhm_arcsec"),
                   ("GAIN", "gain_e_per_adu")):
        if meta.get(k) is not None:
            h[key] = meta[k]
    h["MJD-OBS"] = mjd
    return Frame(path=str(path), data=data, unc=None, bad=bad, wcs=None, mjd=mjd,
                 exptime=meta.get("exptime_s"), magzp=meta.get("magzp"),
                 psf_fwhm_arcsec=meta.get("fwhm_arcsec"), saturate=top, band=None, header=h)


def _load_raster_sequence(paths: List[Path]) -> List[Frame]:
    from . import starmatch
    meta = {}
    seq_json = paths[0].parent / "sequence.json"
    if seq_json.exists():
        meta = json.loads(seq_json.read_text())
    times = raster_times(paths)
    frames = sorted((load_raster_frame(p, times[str(p)], meta) for p in paths), key=lambda f: f.mjd)
    starmatch.assign_wcs(frames, scale_arcsec=meta.get("pixel_scale_arcsec"),
                         ra_deg=meta.get("ra_deg", 0.0), dec_deg=meta.get("dec_deg", 0.0))
    if not meta.get("fwhm_arcsec"):
        ref = frames[len(frames) // 2]
        fwhm = starmatch.measure_fwhm_px(ref.data, ref.bad, starmatch.detect_stars(ref.data, ref.bad))
        if fwhm:
            for f in frames:
                f.psf_fwhm_arcsec = fwhm * f.pixel_scale_arcsec
                f.header["PSFFWHM"] = f.psf_fwhm_arcsec
    return frames


def load_sequence(inputs, bad_bits: Optional[int] = None) -> Sequence:
    paths = find_images(inputs)
    if not paths:
        raise ValueError(f"no FITS or raster images found in {inputs}")
    if all(_is_raster(p) for p in paths):
        frames = _load_raster_sequence(paths)
    else:
        frames = sorted((load_frame(p, bad_bits) for p in paths), key=lambda f: f.mjd)
    truth = metadata = known = None
    source = None
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        source = str(inputs)
        if (Path(inputs) / "truth.json").exists():
            truth = json.loads((Path(inputs) / "truth.json").read_text())
        if (Path(inputs) / "metadata.json").exists():
            metadata = json.loads((Path(inputs) / "metadata.json").read_text())
        if (Path(inputs) / "known.json").exists():
            known = json.loads((Path(inputs) / "known.json").read_text())
    return Sequence(frames=frames, truth=truth, metadata=metadata, source=source, known=known)


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
