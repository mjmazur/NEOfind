"""WISE L1b-style FITS output: one ``-int`` (image, DN), ``-unc`` (1-sigma
uncertainty, DN) and ``-msk`` (bit mask) file per frame, each carrying the
frame's WCS and observation metadata, so simulated and real WISE frames load
through the same code.

With ``compress`` the files are RICE-compressed ``.fits.fz`` (float images
quantized at 1/16 of the per-tile noise with subtractive dithering, the usual
fpack setting, which leaves noise statistics intact; the mask is lossless).
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time


def frame_header(wcs, *, instrument: str, band: str, mjd_mid: float, exposure_s: float,
                 magzp_dn: float, gain: float, read_noise_dn: float, saturate_dn: float,
                 psf_fwhm_arcsec: float, sky_dn: float, index: int) -> fits.Header:
    h = wcs.to_header()
    mjd_start = mjd_mid - exposure_s / 2.0 / 86400.0
    h["INSTRUME"] = (instrument, "instrument (simulated)")
    h["BAND"] = (band, "bandpass")
    h["EXPTIME"] = (exposure_s, "[s] exposure time")
    h["MJD-OBS"] = (mjd_start, "[d] MJD at exposure start")
    h["MJD-MID"] = (mjd_mid, "[d] MJD at mid-exposure")
    h["DATE-OBS"] = (Time(mjd_start, format="mjd", scale="utc").isot, "UTC exposure start")
    h["MAGZP"] = (magzp_dn, "[mag] zero point: m = MAGZP - 2.5 log10(DN)")
    h["GAIN"] = (gain, "[e-/DN]")
    h["RDNOISE"] = (read_noise_dn, "[DN] read noise")
    h["SATURATE"] = (saturate_dn, "[DN] full well")
    h["PSFFWHM"] = (psf_fwhm_arcsec, "[arcsec] PSF FWHM this frame")
    h["SKYLEVEL"] = (sky_dn, "[DN] true sky level this frame")
    h["FRAMEIDX"] = (index, "frame index in the simulated sequence")
    h["SIMULATD"] = (True, "produced by neosim")
    return h


def _write(path: Path, data: np.ndarray, header: fits.Header, compress: bool,
           lossy: bool, dither_seed: int) -> Path:
    if compress:
        path = path.with_name(path.name + ".fz")
        kw = dict(compression_type="RICE_1")
        if lossy:
            kw.update(quantize_level=16.0, quantize_method=1, dither_seed=dither_seed)
        hdus = fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data, header, **kw)])
    else:
        hdus = fits.HDUList([fits.PrimaryHDU(data, header)])
    hdus.writeto(path, overwrite=True)
    return path


def write_frame(out_dir: Path, index: int, image_dn: np.ndarray, unc_dn: np.ndarray,
                mask: np.ndarray, header: fits.Header, compress: bool,
                seed: int) -> dict:
    stem = f"frame_{index:04d}"
    dither_seed = 1 + (seed + 3 * index) % 9999
    paths = {
        "int": _write(out_dir / f"{stem}-int.fits", image_dn.astype(np.float32), header,
                      compress, True, dither_seed),
        "unc": _write(out_dir / f"{stem}-unc.fits", unc_dn.astype(np.float32), header,
                      compress, True, dither_seed + 1),
        "msk": _write(out_dir / f"{stem}-msk.fits", mask.astype(np.int16), header,
                      compress, False, 0),
    }
    return {k: p.name for k, p in paths.items()}
