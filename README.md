# NEOfind

Find moving objects (asteroids) in sequences of astronomical images, with
first-class support for WISE/NEOWISE single-exposure frames. See `PLAN.md` for
the full design and milestones. Status: **milestones 1 (simulator) and 2
(detector on simulated WISE data) done**; WISE data access (`neowise`) is next.

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

## Detector (`neodetect`)

```bash
python -m neodetect detect sims/<run> --plot          # -> detections.json / .png
python -m neodetect detect frame_*-int.fits.fz         # any FITS frames with WCS + time
python -m neodetect detect DIR --region 150.1 12.3 0.5 # search one 0.5 deg square
python -m neodetect validate sims/<null-run>           # noise whitening self-check
python sweep.py -o sims/sweeps/wise_v1 --runs 24 --nulls 24
python -m neodetect evaluate sims/sweeps/wise_v1       # completeness / false alarms
python -m pytest                                       # tests (~15 s)
```

Inputs are FITS images with a celestial WCS and an observation time
(`MJD-MID`, `MJD_OBS`, `MJD-OBS` or `DATE-OBS` + `EXPTIME`). Optional
companions `X-unc*`/`X-msk*` next to `X-int*` provide uncertainties and a
bad-pixel mask; `MAGZP` and `PSFFWHM` are used when present.

Pipeline:

1. **Native-frame cleanup** (`cosmics.py`, `register.py`): subtract a smooth
   sky; mask cosmic rays and hot pixels with three tests bounded by what the
   PSF can produce: peak vs opposite-neighbor pairs, 2×2 box vs ring, and a
   PSF-vs-sharp-profile χ² comparison (catches ~91% of simulated WISE hits,
   masks no asteroids).
2. **Registration** (`register.py`): resample every frame (cubic spline)
   onto one TAN grid covering the union of footprints, scaled to a common
   zero point; per-frame validity tracks partial coverage.
3. **Static sky** (`preprocess.py`): subtract a leave-one-out, outlier-clipped
   inverse-variance template (an asteroid never subtracts itself); noise
   model `alpha_k·(var + var_template) + beta·T²`; PSF matched-filter maps
   `psi = P⋆(w r)`, `phi = P²⋆w`, with weights rescaled so the SNR map is
   N(0, 1) on blank sky (absorbs interpolation-correlated noise).
4. **Seeding**, two paths:
   - *Linking* (`sources.py`, `linking.py`): single-frame peaks ≥ 4σ;
     every pair at a plausible rate (up to 10 deg/day) is propagated to
     the other frames with error-scaled tolerances; ≥ 3 supporting
     detections make a tracklet.
   - *Shift-and-stack* (`shiftstack.py`, `backend.py`): every velocity on a
     grid up to 0.25 deg/day, per-frame contributions clipped at 4σ so no
     single frame can carry a hypothesis; numba-parallel on CPU (a CuPy GPU
     backend only needs to implement one function).
5. **Track fit** (`trackfit.py`): exact PSF-stamp matched filter per frame,
   robust combination (frames far *above* the median flux are dropped, e.g.
   a cosmic ray on the track; low frames are kept as evidence against),
   Nelder–Mead refinement of position and velocity.
6. **Vetting** (`detector.py`): stationary sources rejected; per-frame
   fluxes must be consistent with one source (χ² p ≥ 1e-3, allowing 35%
   variability); a frame already claimed by a stronger track is released.
   Detection thresholds depend on how many frames a track uses (short
   tracks need more: chance alignments of PSF-like cosmic-ray residuals).

Output per detection: score, frames used/covered, RA/Dec and rate/PA at
`t_ref`, mean flux and magnitude, and per frame the MJD, RA/Dec, native
pixel position, flux, flux error and SNR.

`evaluate` matches detections to the truth (median separation < 2 px),
calibrates the three thresholds on the asteroid-free runs (splitting the
false-alarm budget), and reports completeness by magnitude, rate and ideal
SNR, plus astrometric/photometric accuracy.

### Measured performance (simulated WISE W1)

`sims/sweeps/wise_v1`: 24 runs with 12 asteroids each (W1 13.5–16.5,
0.05–3 deg/day; 286 covered by ≥ 3 frames) and 72 null runs, default
`wise-w1` preset (5σ single-frame depth 15.3, ~130 unflagged cosmic rays
per frame). The calibrated thresholds (3 / 4 / ≥5 frames: 10.25 / 9.5 /
8.25) are the defaults. They give 0.04 false detections per null run.

| W1 mag | 13.5–15.0 | 15.0–15.5 | 15.5–16.0 | 16.0–16.5 |
|---|---|---|---|---|
| completeness | 1.00 | 0.82 | 0.40 | 0.04 |
| median ideal stacked SNR | 21–48 | 13.1 | 8.4 | 5.6 |

So the 50% point is at W1 ≈ 15.4–15.5, i.e. a stacked SNR of about 9. The
search goes about as deep as the single-frame limit while also using many
frames. By rate, completeness is 0.65–0.85 with no clear trend up to 2
deg/day (each bin mixes magnitudes). The 2–5 deg/day bin reaches 0.55;
fast movers cross the field in fewer frames. Accuracy for detected
objects is 0.37″ rms in position (0.13 px), 0.2% in rate, 0.13° in position
angle, and 0.07 mag. Per 12-frame sequence, the linker takes seconds; the
shift-and-stack up to 0.25 deg/day takes ~40 s on 32 cores; the full
evaluation (96 runs, `-j 4`) takes ~1.5 h.

## Origin

`neosim` reuses and adapts code from `~/source/Projects/MeteorSimulator`
(flux-conserving PSF stamping, photometric model, star field, gnomonic
helpers, config/CLI style); adapted modules say so in their docstrings.
