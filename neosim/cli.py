"""Command-line interface: ``python simulate.py [--preset NAME] [options]``.

Every option defaults to "unset", meaning the preset's value is used; the
preset in turn overrides the dataclass defaults.
"""

import argparse
import sys

from .config import (ArtifactConfig, CadenceConfig, InstrumentConfig, PointingConfig,
                     PopulationConfig, SimConfig)
from .presets import get_preset, preset_names
from .simulate import run_simulation

SECTIONS = {
    "instrument": InstrumentConfig,
    "cadence": CadenceConfig,
    "pointing": PointingConfig,
    "population": PopulationConfig,
    "artifacts": ArtifactConfig,
    "sim": SimConfig,
}

# (section, field, flag, type, help). ``bool`` entries become on/off flag pairs.
OPTIONS = [
    ("instrument", "band", "--band", str, "band label; selects the star-count model (V, W1, W2)"),
    ("instrument", "width", "--width", int, "array width, px"),
    ("instrument", "height", "--height", int, "array height, px"),
    ("instrument", "pixel_scale_arcsec", "--pixel-scale", float, "plate scale, arcsec/px"),
    ("instrument", "pixel_size_um", "--pixel-size", float, "physical pixel pitch, um"),
    ("instrument", "psf_fwhm_arcsec", "--psf-fwhm", float, "PSF FWHM, arcsec"),
    ("instrument", "psf_fwhm_jitter", "--psf-jitter", float, "fractional rms PSF width variation"),
    ("instrument", "exposure_s", "--exposure", float, "exposure time, s"),
    ("instrument", "limiting_mag", "--limiting-mag", float, "5-sigma point-source depth, mag"),
    ("instrument", "saturation_mag", "--saturation-mag", float, "point-source saturation, mag"),
    ("instrument", "background_e", "--background", float, "sky, e-/px/exposure"),
    ("instrument", "background_jitter", "--background-jitter", float, "fractional rms sky variation"),
    ("instrument", "read_noise_e", "--read-noise", float, "read noise, e-"),
    ("instrument", "gain_e_per_dn", "--gain", float, "gain, e-/DN"),
    ("instrument", "zp_jitter_mag", "--zp-jitter", float, "rms transparency variation, mag"),
    ("cadence", "mode", "--cadence", str, "wise | uniform | visits | file"),
    ("cadence", "n_frames", "--n-frames", int, "frames (wise/uniform cadence)"),
    ("cadence", "interval_min", "--interval", float, "minutes between frames / orbital period"),
    ("cadence", "n_visits", "--n-visits", int, "visits (visits cadence)"),
    ("cadence", "frames_per_visit", "--frames-per-visit", int, "frames per visit"),
    ("cadence", "visit_spacing_min", "--visit-spacing", float, "minutes between visits"),
    ("cadence", "intra_visit_min", "--intra-visit", float, "minutes between frames in a visit"),
    ("cadence", "drop_prob", "--drop-prob", float, "wise: probability an orbit is missing"),
    ("cadence", "time_jitter_s", "--time-jitter", float, "rms timing jitter, s"),
    ("cadence", "start_mjd", "--start-mjd", float, "MJD of the first frame"),
    ("cadence", "times_file", "--times-file", str, "file of mid-exposure MJDs (file cadence)"),
    ("pointing", "mode", "--pointing", str, "wise | dither"),
    ("pointing", "ra_deg", "--ra", float, "target RA, deg"),
    ("pointing", "dec_deg", "--dec", float, "target Dec, deg"),
    ("pointing", "rotation_deg", "--rotation", float, "dither: image +y position angle, deg E of N"),
    ("pointing", "rotation_jitter_deg", "--rotation-jitter", float, "rms roll jitter, deg"),
    ("pointing", "dither_arcsec", "--dither", float, "dither: rms offset per axis, arcsec"),
    ("pointing", "cross_scan_step_arcmin", "--cross-scan-step", float, "wise: drift per orbit, arcmin"),
    ("pointing", "along_scan_offset_frac", "--along-scan-offset", float,
     "wise: max along-scan offset, fraction of FOV"),
    ("population", "n_asteroids", "--n-asteroids", int, "asteroids in the field (0 = null run)"),
    ("population", "mag_min", "--mag-min", float, "brightest mean magnitude"),
    ("population", "mag_max", "--mag-max", float, "faintest mean magnitude"),
    ("population", "rate_min_deg_day", "--rate-min", float, "slowest rate, deg/day"),
    ("population", "rate_max_deg_day", "--rate-max", float, "fastest rate, deg/day"),
    ("population", "mag", "--mag", float, "fix every asteroid's mean magnitude"),
    ("population", "rate_deg_day", "--rate", float, "fix every asteroid's rate, deg/day"),
    ("population", "pa_deg", "--pa", float, "fix every asteroid's motion direction, deg E of N"),
    ("population", "accel_rms_arcsec_hr2", "--accel", float, "rms acceleration, arcsec/hr^2"),
    ("population", "motion", "--motion", str, "linear (quadratic path) or orbit (two-body)"),
    ("population", "lc_amp_max", "--lc-amp-max", float, "max light-curve amplitude, mag"),
    ("population", "placement_box_arcmin", "--placement-box", float,
     "asteroids start in this square around the target, arcmin"),
    ("artifacts", "cosmic_rate_cm2_s", "--cosmic-rate", float, "cosmic rays per cm^2 per s"),
    ("artifacts", "cosmic_flag_frac", "--cosmic-flag-frac", float, "fraction of hits flagged"),
    ("artifacts", "hot_pixel_frac", "--hot-pixel-frac", float, "fraction of hot pixels"),
    ("artifacts", "n_bad_columns", "--bad-columns", int, "dead columns"),
    ("sim", "output_dir", "--output-dir", str, "output directory (default ./sims/YYYYMMDD_HHMMSS)"),
    ("sim", "seed", "--seed", int, "random seed"),
    ("sim", "star_density_scale", "--star-density", float, "star density multiplier"),
    ("sim", "ref_frame", "--ref-frame", int, "reference frame index (default: middle)"),
    ("sim", "stars", "stars", bool, "render background stars"),
    ("sim", "compress", "compress", bool, "write RICE-compressed .fits.fz"),
    ("sim", "previews", "previews", bool, "write PNG/GIF previews"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate a sequence of sky images containing moving asteroids, "
                    "written as WISE L1b-style FITS (int/unc/msk) with ground truth.")
    p.add_argument("--preset", default="wise-w1", choices=preset_names(),
                   help="parameter preset (default: wise-w1)")
    p.add_argument("--no-asteroids", action="store_const", const=0, dest="population.n_asteroids",
                   help="null run: no asteroids (same as --n-asteroids 0)")
    p.add_argument("-o", dest="sim.output_dir", default=None, metavar="DIR",
                   help="alias for --output-dir")
    groups = {s: p.add_argument_group(f"{s} parameters") for s in SECTIONS}
    for section, field, flag, typ, help_ in OPTIONS:
        dest = f"{section}.{field}"
        g = groups[section]
        if typ is bool:
            g.add_argument(f"--{flag}", dest=dest, action="store_true", default=None,
                           help=f"{help_} (on)")
            g.add_argument(f"--no-{flag}", dest=dest, action="store_false", default=None,
                           help=f"{help_} (off)")
        else:
            g.add_argument(flag, dest=dest, type=typ, default=None, help=help_,
                           metavar=typ.__name__.upper())
    return p


def build_configs(args: argparse.Namespace) -> dict:
    preset = get_preset(args.preset)
    values = {s: dict(preset.get(s, {})) for s in SECTIONS}
    values["sim"]["preset"] = args.preset
    for dest, v in vars(args).items():
        if "." in dest and v is not None:
            section, field = dest.split(".", 1)
            values[section][field] = v
    return {s: cls(**values[s]) for s, cls in SECTIONS.items()}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = build_configs(args)
        out = run_simulation(cfg["instrument"], cfg["cadence"], cfg["pointing"],
                             cfg["population"], cfg["artifacts"], cfg["sim"])
    except ValueError as e:
        parser.error(str(e))
        return 2
    print(f"Wrote simulation output to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
