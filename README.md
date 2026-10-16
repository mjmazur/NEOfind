# NEOfind

Find moving objects (asteroids) in sequences of astronomical images, with
first-class support for WISE/NEOWISE single-exposure frames. See `PLAN.md` for
the full design and milestones. Status: **milestones 1 (simulator), 2
(detector on simulated WISE data) and 3 (real NEOWISE data) done; 4
(robustness and reporting) in progress**.

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
  (`--accel` for ad hoc curvature), or with `--motion orbit` on a heliocentric
  two-body orbit seen from Earth, set up from the same drawn position, rate and
  direction (distance from the rate, as for injections into real data): paths
  then curve as real ones do (~12"/day² for main-belt distances, ~250"/day² at
  0.4 AU). Rates are log-uniform between `--rate-min` and
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
python -m neodetect detect DIR --ades obs.psv --gifs cutouts/  # MPC report + vetting GIFs
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
   masks no asteroids). The ratio limits come from each frame's own bright
   stars when they are sharper than the nominal Gaussian, as real WISE
   cores are (frames with too few stars use the visit's median). Masked
   neighbours are left out of the comparisons. With Gaussian limits the
   tests masked parts of the bright asteroid (120) Lachesis in 4 of 12
   frames. A second pass repeats the tests with the static sky (per-pixel
   median of the registered frames, mapped back to native pixels)
   subtracted: a cosmic ray on a star is as broad as the star in the first
   pass, and these hits set the top of the simulated false-alarm tail.
   The same pass masks long thin features (`cosmics.streak_mask`):
   scattered-light streaks from bright stars beyond a cutout edge, and
   satellite trails. Significance clipped at ±1.5 is averaged along 41-px
   lines in 16 directions; a line is masked where its mean is ≥ 0.8 and
   ≥ 6× the frame's own scatter in that direction (real WISE noise is
   correlated along rows), and low across it. Compact sources (≤ 100 px
   above 5σ, grown 6 px) are left out and never masked, so asteroids
   stay; on simulations nothing is masked.
2. **Registration** (`register.py`): resample every frame (cubic spline)
   onto one TAN grid covering the union of footprints, scaled to a common
   zero point; per-frame validity tracks partial coverage. Masked pixels are
   filled from their neighbors and cost only the grid samples nearest them
   (strictly masked only on sources), which matters for real WISE masks.
3. **Static sky** (`preprocess.py`): subtract a leave-one-out, outlier-clipped
   inverse-variance template (an asteroid never subtracts itself); noise
   model `alpha_k·(var + var_template) + beta·T²`; PSF matched-filter maps
   `psi = P⋆(w r)`, `phi = P²⋆w`, with weights rescaled so the SNR map is
   N(0, 1) on blank sky (absorbs interpolation-correlated noise). A
   *local* noise map (SNR scatter in 31-px windows) then deflates the
   weights wherever a frame is noisier than the model says: scattered light
   from bright stars outside the field, crowded-field frame edges
   (`local_noise_px`, 0 = off). Compact sources (asteroids, star
   residuals, an asteroid's imprints in other frames' templates) are left
   out of the estimate, so a bright asteroid doesn't down-weight itself.
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
   Nelder–Mead refinement of position, velocity and a bounded acceleration.
   Seen from Earth or WISE, asteroid paths curve: the observer accelerates
   toward the Sun, so a main-belt asteroid's path bends ~15"/day² and
   strays up to 3" (1 WISE px) from a straight line over a visit. The bound
   is max(60, 150 × rate, 500 × rate²) "/day² (rate in deg/day). The rate²
   term is for close objects, whose rate changes with their distance.
   Tracks using fewer than 4 frames are fitted straight: with 3 frames a
   curved path fits any 3 points.
   A final track's path is re-fitted on its fully covered frames (≥ 90%
   of the PSF on valid pixels). Otherwise the fit can bend the path toward
   valid pixels to keep a partly masked frame; that put Lachesis's
   positions 0.8″ off.
6. **Vetting** (`detector.py`): paths moving less than 4 PSF FWHM over the
   visit are rejected (the largest separation of the path's positions, so
   curvature counts; 0.005 deg/day over a 33 h NEOWISE visit, while the
   slowest of 87 known asteroids in our visits moves 0.084); per-frame
   fluxes must be consistent with one source (χ² p ≥ 1e-3, allowing 35%
   variability); a frame already claimed by a stronger track is released.
   Detection thresholds depend on how many frames a track uses (short
   tracks need more: chance alignments of PSF-like cosmic-ray residuals).
   On real WISE frames they are set per visit (below).
7. **Photometry** (`photometry.py`): Gaussian-fit fluxes are scaled to
   total flux with an aperture correction measured on the visit's own
   bright isolated stars (≥ 8 needed; ~1.15 on NEOWISE W1, the fallback in
   crowded fields; ~1.00 on simulations).
8. **Astrometry**: on real WISE W1, Gaussian-fit positions sit (+0.147,
   +0.085) native px from the NEOWISE catalog's PSF-fit positions, the
   convention the frame WCS is calibrated in. That's 0.45″ and constant to
   ~0.01 px over the detector and over 8 visits at every orientation; the
   real PSF is slightly asymmetric. Reported positions are corrected per frame
   (`centroid_offset`, default from `io.default_centroid_offset`). Oceana
   then agrees with the catalog to 0.02″ along and 0.06″ across its track
   (0.46″ before). Injections draw their stamps at the same offset.

**Per-visit thresholds** (`nullcal.py`). On real data the false-alarm tail
varies far more between visits than one global threshold can follow. So
`detect` runs the detector on time-scrambled copies of the visit
(`--null-scrambles N`, default 2 on real WISE frames, 0 = off). The visit's
*null level* is the median over copies of the 5th-highest candidate score.
The thresholds are that level × 1.81 / 1.53 / 1.39 for 3 / 4 / ≥5-frame
tracks (`--null-ratio`). The factors are calibrated on `sims/real_v1` so
that held-out scrambled copies give ≤ 0.05 false detections per visit.
Each copy costs one detector run. The copies also give an **artifact
veto**: a candidate within 60″ (at the reference time) of a scrambled-copy
candidate scoring ≥ 1.2× the null level is rejected, as something that
recurs whatever the frame order isn't moving. Without it one scattered-light
artifact in one visit set the ≥ 5-frame factor.

**Reporting** (`report.py`): `--ades FILE` writes ADES PSV for the MPC,
one row per frame a detection uses: the track position at that frame's
time, uncertainties from the centroid error plus a 0.25" floor, and the
per-frame magnitude (uncertainty floor 0.03 mag). For WISE the station is C51 and each row carries the
spacecraft position from `ERTH2SC`. Known objects get their designation.
The submitter/observer names are placeholders. Check the WISE `mode` and
`astCat` values against the MPC's documentation before a real submission.
`--gifs DIR` writes an animated cutout per detection: registered image and
static-sky residual with the predicted position marked.

Output per detection: score, frames used/covered, RA/Dec and rate/PA at
`t_ref`, mean flux and magnitude, and per frame the MJD, RA/Dec, native
pixel position, flux, flux error and SNR.

`evaluate` matches detections to the truth (median separation < 2 px),
calibrates the three thresholds on the asteroid-free runs (splitting the
false-alarm budget), and reports completeness by magnitude, rate and ideal
SNR, plus astrometric/photometric accuracy. On real-data sweeps (several
scrambled copies per visit) it also calibrates the per-visit factors,
holding out each copy, and reports completeness with per-visit thresholds.

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

### Measured performance (real NEOWISE W1)

`sims/real_v1`: 15 NEOWISE-R visits along the ecliptic (0.5° cutouts,
11–18 frames each, 2014–2015). Each visit has 2 injected copies (20
asteroids each, W1 13.5–17, 0.05–3 deg/day; 489 covered by ≥ 3 frames) and
3 time-scrambled copies (45 null runs). The detector runs with its default
settings; the thresholds below are calibrated on the scrambled nulls. The
numbers are from the M4 detector with aperture correction and
`n_refine` = 2000 (`evaluation_m4.json`), before local noise and
acceleration fitting. Results with those, and with curved injected paths
(`sims/real_v2`), are pending.

| | simulated (`wise_v1`) | real (`real_v1`) |
|---|---|---|
| thresholds 3 / 4 / ≥5 frames | 10.25 / 9.5 / 8.25 | 20 / 16.75 / 13.75 |
| false detections per null run | 0.04 | 0.04 |
| completeness W1 14.5–15.0 | 1.00 | 0.94 |
| completeness W1 15.0–15.5 | 0.82 | 0.79 |
| completeness W1 15.5–16.0 | 0.40 | 0.37 |
| completeness W1 16.0–16.5 | 0.04 | 0.01 |
| median ideal SNR at W1 15.0–15.5 | 13.2 | 23.8 |

The 50% point is the same, W1 ≈ 15.4, but for different reasons. Real
frames are deeper than the `wise-w1` preset, which gives ~1.8× the SNR at
a given magnitude. Their false-alarm tail is heavier, so the thresholds are
~1.7× higher. On real data the 50% point is at a stacked SNR of ~17, not 9.

- **What sets the real thresholds:** chance alignments of real sources and
  artifacts in the crowded low-galactic-latitude visits (v002 in Taurus,
  v010–v013 toward the bulge). Every null-run score above 17 comes from
  v012. With 45 nulls, the 3-frame threshold is set by a single event;
  20 is the top of the calibration grid.
- **Crowded fields also lose completeness.** Completeness at W1 14.5–15.5
  is 0.73 in the five crowded visits, against 0.92 elsewhere. Those visits
  hit the M3 triage cap (`n_refine` = 400). Raising it to 2000 (the M4
  default) roughly doubles their refine time but didn't change completeness
  overall; the per-visit thresholds do help (0.80).
- **Known asteroids:** (224) Oceana and (120) Lachesis (W1 ≈ 11.7–11.8) are
  found in every copy at score 200–250, with positions within 0.1–0.2 px
  of the parallax-corrected SkyBoT ephemeris. The fainter known objects
  are below the thresholds above. One exception: (2882), at a stacked SNR
  of 14, is missed because its track runs along the cutout edge, where an
  edge-pixel outlier fails the flux-consistency test.
- **Accuracy:** 0.30″ rms in position, 0.2% in rate, 0.09° in position
  angle, 0.04 mag rms in magnitude (bias −0.01). Before the aperture
  correction (M3) magnitudes were 0.14 mag too faint. By rate, completeness falls from 0.7 below
  0.25 deg/day to 0.2–0.5 above, mostly because fast movers cross a 0.5°
  cutout in fewer frames.
- **Per-visit thresholds**, same detections, same false-alarm rate:

  | W1 mag | 14.5–15.0 | 15.0–15.5 | 15.5–16.0 | 16.0–16.5 |
  |---|---|---|---|---|
  | global thresholds | 0.94 | 0.79 | 0.37 | 0.01 |
  | per-visit thresholds | 0.96 | 0.90 | 0.50 | 0.19 |

  The ≥5-frame threshold ranges from 9.5 on clean visits to 20.6 on the
  bulge visit v012. The 50% point moves to W1 ≈ 15.5–15.6. A single
  scrambled copy is noisy: four copies of the Oceana visit gave levels of
  7.7–8.4, hence the default of two.
- **Cost:** ~3–6 min per visit on 32 cores. The full sweep (75 runs, `-j 4`)
  takes ~5 h.

## WISE/NEOWISE data (`neowise`)

```bash
python -m neowise visits 152.09 11.47                 # visits covering a position
python -m neowise fetch 152.09 11.47 --visit 2        # 0.5 deg cutouts + known.json
python -m neodetect detect data/neowise/<pos>/<date>  # search; labels known objects
python -m neowise inject <visit> -o OUT --n 20        # synthetic asteroids in real frames
python -m neowise scramble <visit> -o OUT             # null test: frame times permuted
python -m neowise sweep data/neowise/*/* -o sims/real_v1
python -m neodetect evaluate sims/real_v1             # completeness / false alarms on real data
```

- **Search** (`query.py`): IRSA IBE metadata tables for the cryogenic,
  post-cryo and NEOWISE-R phases (`--tables`, default `neowiser`). A
  position near the ecliptic is seen on ~12-20 consecutive orbits every ~6
  months; such a run is a *visit*. One frame per orbit (the one centered
  nearest the target) and only frames covering ≥ 10% of the region are kept.
- **Download** (`download.py`): `int`/`unc`/`msk` cutouts (or `--full-frames`)
  with their IRSA names, so `neodetect` pairs them; cached; polite retries
  (IRSA answers 503 to bursts).
- **Mask bits** (`maskbits.py`): the Explanatory Supplement's L1b
  definitions. Bit 27 (temporal outlier in co-add stacking) is *not* treated
  as bad: moving objects are temporal outliers. Static bad pixels (~3%) are
  NaN in the images anyway.
- **Known objects** (`known.py`): SkyBoT cone search per frame epoch.
  SkyBoT is geocentric; WISE orbits ~6900 km from the geocenter (a 3.5"
  parallax at 2.7 AU), so positions are shifted to the spacecraft using
  each header's `ERTH2SC` vector — they then agree with detections to
  ~0.1-0.2 px. Magnitudes are predicted as W1 = V − 1.8 (rough: real
  colours vary by a magnitude). `detect` labels matching detections and
  lists the known objects it could have seen.
- **Empirical PSF** (`epsf.py`): effective PSF from ~300 stars per visit,
  4× oversampled. W1 FWHM ≈ 5.9", with wings the Gaussian model lacks.
- **Injection** (`inject.py`): synthetic asteroids rendered with the visit's
  ePSF through each frame's real WCS and `MAGZP`, plus a `truth.json` in
  the simulator's format (known real objects included, flagged `real`).
  Each asteroid follows a heliocentric two-body orbit
  (`neosim/orbit.py`), seen from the spacecraft with light-time
  correction, so paths curve as real ones do. It is set up from the sky
  rate and direction plus a distance drawn as Δ = (8–20 km/s) / rate and a
  radial velocity. With the right radial velocity this reproduces SkyBoT
  paths of known asteroids to ~0.1".
  **Scrambled** copies keep every artifact but no real linear motion, and
  stand in for asteroid-free nulls when calibrating false alarms.

## Origin

`neosim` reuses and adapts code from `~/source/Projects/MeteorSimulator`
(flux-conserving PSF stamping, photometric model, star field, gnomonic
helpers, config/CLI style); adapted modules say so in their docstrings.
