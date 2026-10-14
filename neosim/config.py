"""Configuration dataclasses for the asteroid simulator.

Every field has a default matching the ``wise-w1`` preset; presets
(``presets.py``) and command-line options override them section by section.
"""

from dataclasses import dataclass, asdict, fields
from typing import Optional


class _Section:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def field_names(cls):
        return {f.name for f in fields(cls)}


@dataclass
class InstrumentConfig(_Section):
    name: str = "WISE"
    band: str = "W1"  # selects the star-count model and is written to headers
    width: int = 1016
    height: int = 1016
    pixel_scale_arcsec: float = 2.757
    pixel_size_um: float = 18.0  # physical pixel pitch; sets the cosmic-ray target area
    psf_fwhm_arcsec: float = 6.1
    psf_fwhm_jitter: float = 0.02  # fractional rms frame-to-frame PSF width variation
    exposure_s: float = 7.7
    limiting_mag: float = 15.3  # 5-sigma PSF-fit point source, nominal frame
    saturation_mag: float = 8.0  # point source whose peak pixel just reaches full well
    background_e: float = 250.0  # sky, electrons/pixel/exposure
    background_jitter: float = 0.05  # fractional rms frame-to-frame sky variation
    read_noise_e: float = 20.0
    gain_e_per_dn: float = 3.2
    zp_jitter_mag: float = 0.01  # rms frame-to-frame transparency/calibration variation


@dataclass
class CadenceConfig(_Section):
    mode: str = "wise"  # "wise" | "uniform" | "visits" | "file"
    n_frames: int = 12  # wise / uniform
    interval_min: float = 95.0  # wise: orbital period; uniform: frame spacing
    n_visits: int = 4  # visits mode
    frames_per_visit: int = 3
    visit_spacing_min: float = 30.0
    intra_visit_min: float = 2.0
    drop_prob: float = 0.05  # wise: chance an orbit's exposure is missing (SAA, Moon, ...)
    time_jitter_s: float = 10.0
    start_mjd: float = 57000.0
    times_file: Optional[str] = None  # file mode: one mid-exposure MJD per line


@dataclass
class PointingConfig(_Section):
    mode: str = "wise"  # "wise" (scan geometry) | "dither"
    ra_deg: float = 0.0  # target (field) center
    dec_deg: float = 0.0
    rotation_deg: float = 0.0  # dither mode: position angle of the image +y axis, E of N
    rotation_jitter_deg: float = 0.1
    dither_arcsec: float = 10.0  # dither mode: rms per-axis pointing offset
    cross_scan_step_arcmin: float = 3.9  # wise: field drift per orbit along ecliptic longitude
    along_scan_offset_frac: float = 0.45  # wise: max random along-scan offset, fraction of FOV


@dataclass
class PopulationConfig(_Section):
    n_asteroids: int = 10
    mag_min: Optional[float] = None  # None => limiting_mag - 4
    mag_max: Optional[float] = None  # None => limiting_mag + 0.5
    rate_min_deg_day: float = 0.1
    rate_max_deg_day: float = 3.0  # rates are drawn log-uniform between min and max
    mag: Optional[float] = None  # fix every asteroid's mean magnitude
    rate_deg_day: Optional[float] = None  # fix every asteroid's rate
    pa_deg: Optional[float] = None  # fix every asteroid's direction of motion (E of N)
    accel_rms_arcsec_hr2: float = 0.0  # rms per-axis sky-plane acceleration (curvature)
    lc_amp_max: float = 0.5  # rotational light curve peak-to-peak amplitude, uniform in [0, max]
    lc_period_min_h: float = 2.0
    lc_period_max_h: float = 20.0
    placement_box_arcmin: Optional[float] = 30.0  # square around target at t_ref; None => reference FOV


@dataclass
class ArtifactConfig(_Section):
    cosmic_rate_cm2_s: float = 5.0  # low-Earth-orbit-like rate
    cosmic_median_e: float = 1500.0
    cosmic_sigma_dex: float = 0.35  # log10 spread of deposited charge
    cosmic_max_len_px: float = 2.0  # thin IR array: mostly 1-3 pixel hits
    cosmic_flag_frac: float = 0.0  # fraction of hits flagged in the mask
    hot_pixel_frac: float = 2e-4
    hot_median_e: float = 1000.0  # extra dark charge per exposure
    hot_flag_frac: float = 0.9
    n_bad_columns: int = 0


@dataclass
class SimConfig(_Section):
    preset: str = "wise-w1"
    output_dir: Optional[str] = None
    seed: Optional[int] = None
    star_density_scale: float = 1.0
    stars: bool = True
    compress: bool = True  # RICE-compressed .fits.fz (quantized floats, lossless mask)
    previews: bool = True
    ref_frame: Optional[int] = None  # None => middle frame
