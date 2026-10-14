"""Magnitude <-> flux (electrons) calibration (adapted from meteorsim.photometry).

Where meteorsim tied its zero point to a limiting star's *peak pixel*, here
the limiting magnitude has the usual survey meaning: the magnitude at which an
optimal PSF-fit (matched-filter) measurement of an isolated point source has
SNR = 5 in a nominal frame. For a Gaussian PSF of width sigma integrated over
square pixels the noise-equivalent area is ``NEA ~= 4*pi*(sigma^2 + 1/12)``
pixels, and

    SNR = F / sqrt(F + NEA * (sky + read_noise^2)).

Everything is in electrons; images are written in DN using ``gain_e_per_dn``.
"""

import math
from dataclasses import dataclass

from scipy.special import erf

from .config import InstrumentConfig

LIMIT_SNR = 5.0


def noise_equivalent_area(psf_sigma_px: float) -> float:
    return 4.0 * math.pi * (psf_sigma_px ** 2 + 1.0 / 12.0)


def peak_pixel_fraction(psf_sigma_px: float) -> float:
    """Fraction of a point source's flux in the pixel it is centered on."""
    half = 0.5 / (math.sqrt(2.0) * psf_sigma_px)
    return float(erf(half)) ** 2


def point_source_snr(flux_e, psf_sigma_px: float, sky_e: float, read_noise_e: float):
    """Matched-filter SNR of a point source (scalar or array flux)."""
    var = sky_e + read_noise_e ** 2
    return flux_e / (flux_e + noise_equivalent_area(psf_sigma_px) * var) ** 0.5


@dataclass
class PhotometricModel:
    zero_point: float  # magnitude for which total flux == 1 electron
    full_well_e: float
    nea_px: float  # nominal-frame noise-equivalent area

    def total_flux_e(self, mag):
        return 10 ** (-0.4 * (mag - self.zero_point))

    def mag(self, flux_e):
        return self.zero_point - 2.5 * math.log10(flux_e)

    def zero_point_dn(self, gain_e_per_dn: float) -> float:
        """Zero point for images in DN (the FITS ``MAGZP`` convention)."""
        return self.zero_point - 2.5 * math.log10(gain_e_per_dn)


def build_photometric_model(inst: InstrumentConfig, psf_sigma_px: float) -> PhotometricModel:
    nea = noise_equivalent_area(psf_sigma_px)
    var = inst.background_e + inst.read_noise_e ** 2
    # Solve F / sqrt(F + nea*var) = k for F.
    k2 = LIMIT_SNR ** 2
    flux_limit = 0.5 * (k2 + math.sqrt(k2 * k2 + 4.0 * k2 * nea * var))
    zero_point = inst.limiting_mag + 2.5 * math.log10(flux_limit)

    flux_sat = 10 ** (-0.4 * (inst.saturation_mag - zero_point))
    full_well = inst.background_e + flux_sat * peak_pixel_fraction(psf_sigma_px)
    return PhotometricModel(zero_point=zero_point, full_well_e=full_well, nea_px=nea)
