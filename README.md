# NEOfind

Find moving objects (asteroids) in sequences of astronomical images, with
first-class support for WISE/NEOWISE single-exposure frames. See `PLAN.md` for
the full design and milestones. Status: **milestone 1 (simulator) done**; the
detector (`neodetect`) and WISE access (`neowise`) are next.

## Simulator (`neosim`)

```bash
pip install -r requirements.txt
python simulate.py                        # WISE W1-like visit, 10 random asteroids
python simulate.py --preset ground        # ground-based survey visits
python simulate.py --no-asteroids         # null run (false-alarm calibration)
python simulate.py --help                 # every option
python -m pytest                          # tests (~3 s)
```

Output goes to `./sims/YYYYMMDD_HHMMSS/` (or `-o DIR`):

| file | contents |
|---|---|
| `frame_NNNN-int.fits.fz` | image, DN (RICE-compressed; `--no-compress` for plain `.fits`) |
| `frame_NNNN-unc.fits.fz` | 1-sigma uncertainty, DN |
| `frame_NNNN-msk.fits.fz` | bit mask: 1 hot pixel, 2 bad column, 4 saturated, 8 cosmic ray (flagged ones) |
| `frame_NNNN.png`, `blink.gif` | quick looks; the GIF is resampled onto the reference frame's grid |
| `overlay.png` | median of all frames on the reference grid with the true tracks drawn |
| `metadata.json` | every parameter, the seed, derived calibration, per-frame conditions |
| `truth.json` | per asteroid: motion, light curve and, per frame, RA/Dec, pixel x/y, in-frame flag, magnitude, flux and expected matched-filter SNR; every cosmic ray and hot pixel |

Each FITS header carries a TAN WCS plus `MJD-OBS` (exposure start),
`MJD-MID`, `EXPTIME`, `MAGZP` (m = MAGZP − 2.5 log10 DN), `GAIN`, `RDNOISE`,
`SATURATE`, `PSFFWHM` and the true `SKYLEVEL`. Pixel coordinates in
`truth.json` are 0-based with pixel centers on integers (astropy `origin=0`).

### Presets

| | `wise-w1` (default) | `wise-w2` | `ground` |
|---|---|---|---|
| array, scale | 1016², 2.757"/px | same | 1024², 1.2"/px |
| PSF FWHM | 6.1" | 6.4" | 2.0" ± 15% |
| exposure | 7.7 s | 7.7 s | 60 s |
| 5σ depth / saturation | 15.3 / 8.0 | 14.5 / 6.7 | 20.0 / 11.5 |
| cadence | 12 orbits, 95 min apart (5% dropped) | same | 4 visits × 3 frames, 30 min apart |
| pointing | WISE scan geometry | same | 10" dither |
| cosmic rays | 5 /cm²/s (~130 per frame) | same | 0.03 /cm²/s |

### Model notes

- **Photometry.** `--limiting-mag` is the 5σ depth of an optimal PSF-fit
  measurement in a nominal frame:
  `SNR = F / sqrt(F + NEA·(sky + RN²))`, with `NEA = 4π(σ² + 1/12)`. This sets the zero point.
  `test_matched_filter_snr_and_uncertainty_match_prediction` checks the
  per-frame `snr` values in `truth.json` against measurements in the images.
- **WISE geometry.** Image +y lies along ecliptic north (the scan direction).
  Successive orbits' frames step 3.9' in ecliptic longitude, and land at a
  random along-scan offset, so frames overlap only partly and an asteroid
  can leave the field partway through. The span is ~17 h for 12 frames.
- **Motion.** Each asteroid moves on a quadratic path in the tangent plane about the target
  (`--accel` for curvature). Rates are log-uniform between `--rate-min` and
  `--rate-max` deg/day. Trailing within an exposure is integrated, so it
  shows up only when physically significant (fast movers from the ground).
  Brightness includes a double-peaked rotational light curve.
- **Per-frame variation.** PSF width, sky level and transparency vary
  between frames. The per-frame `MAGZP` reflects the transparency.
- **Artifacts.** Hot pixels and dead columns are fixed in detector
  coordinates, so they move on the sky between dithered or scanned frames,
  just as in real data. Cosmic rays are short log-normal-charge tracks; by
  default none are flagged (`--cosmic-flag-frac`).
- **Stars.** Synthetic stars, not a catalog. Cumulative counts are quadratic in magnitude per band
  (V; W1/W2), scaled by galactic latitude at the target. They are static
  and rendered through each frame's WCS.
- **Not yet modeled.** WISE's real PSF wings and diffraction features,
  latents, ghosts, galaxies, variable stars, and WISE's orbital parallax for
  very close NEOs.

## Origin

`neosim` reuses and adapts code from `~/source/Projects/MeteorSimulator`
(flux-conserving PSF stamping, photometric model, star field, gnomonic
helpers, config/CLI style); adapted modules say so in their docstrings.
