"""PSF width conversions (adapted from meteorsim.optics).

The instrument is specified by its plate scale directly (``pixel_scale_arcsec``)
rather than by pixel size and focal length; ``plate_scale_arcsec`` converts
the latter when that is what's known.
"""

import math

FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # ~ 1 / 2.3548

# Below ~1 px FWHM a real sensor's point-source footprint is set by pixel-level
# effects, not the optics; keep rendered PSFs at least this wide (meteorsim
# used 2 px for its heavily undersampled wide-field camera).
MIN_PSF_FWHM_PX = 1.0


def plate_scale_arcsec(pixel_size_um: float, focal_length_mm: float) -> float:
    return math.degrees((pixel_size_um * 1e-6) / (focal_length_mm * 1e-3)) * 3600.0


def fwhm_arcsec_to_sigma_px(fwhm_arcsec: float, pixel_scale_arcsec: float) -> float:
    fwhm_px = max(fwhm_arcsec / pixel_scale_arcsec, MIN_PSF_FWHM_PX)
    return fwhm_px * FWHM_TO_SIGMA
