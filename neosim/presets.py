"""Named parameter presets.

A preset is a set of per-section overrides applied on top of the dataclass
defaults (which are the ``wise-w1`` values). Command-line options are then
applied on top of the preset.

WISE values: 1016x1016 arrays at 2.757"/px, 18 um pixels, ~7.7 s W1/W2
exposures, PSF FWHM ~6.1" (W1) / 6.4" (W2), and approximate single-exposure
5-sigma depths and saturation limits. Sky/read-noise/gain values are
representative rather than calibrated; the limiting magnitude fixes the
photometric scale either way.
"""

PRESETS = {
    "wise-w1": {},
    "wise-w2": {
        "instrument": dict(band="W2", psf_fwhm_arcsec=6.4, limiting_mag=14.5,
                           saturation_mag=6.7, background_e=600.0, gain_e_per_dn=3.83),
    },
    "ground": {
        "instrument": dict(name="ground", band="V", width=1024, height=1024,
                           pixel_scale_arcsec=1.2, pixel_size_um=9.0,
                           psf_fwhm_arcsec=2.0, psf_fwhm_jitter=0.15, exposure_s=60.0,
                           limiting_mag=20.0, saturation_mag=11.5, background_e=800.0,
                           background_jitter=0.2, read_noise_e=8.0, gain_e_per_dn=1.5,
                           zp_jitter_mag=0.05),
        "cadence": dict(mode="visits", drop_prob=0.0, time_jitter_s=5.0),
        "pointing": dict(mode="dither", rotation_jitter_deg=0.05),
        "population": dict(rate_min_deg_day=0.05, rate_max_deg_day=2.0,
                           placement_box_arcmin=None),
        "artifacts": dict(cosmic_rate_cm2_s=0.03, cosmic_median_e=3000.0,
                          cosmic_max_len_px=15.0, hot_pixel_frac=1e-4,
                          hot_flag_frac=1.0, n_bad_columns=1),
    },
}


def preset_names():
    return sorted(PRESETS)


def get_preset(name: str) -> dict:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(preset_names())}")
