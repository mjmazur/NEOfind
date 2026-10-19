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
python -m neodetect detect pngs/                       # PNG/TIFF/JPEG + times.csv (no WCS)
python -m neodetect review DIR                         # inspect + confirm suspected unknowns
python -m neodetect review DIR --ades new.psv          # ... confirmed ones as ADES
python -m neodetect validate sims/<null-run>           # noise whitening self-check
python sweep.py -o sims/sweeps/wise_v1 --runs 24 --nulls 24
python -m neodetect evaluate sims/sweeps/wise_v1       # completeness / false alarms
python -m pytest                                       # tests (~15 s)
```

Inputs are FITS images with a celestial WCS and an observation time
(`MJD-MID`, `MJD_OBS`, `MJD-OBS` or `DATE-OBS` + `EXPTIME`). Optional
companions `X-unc*`/`X-msk*` next to `X-int*` provide uncertainties and a
bad-pixel mask; `MAGZP` and `PSFFWHM` are used when present.

Raster images (PNG, TIFF, JPEG; 8/16-bit, colour converted to luminance)
are read when a directory holds no FITS. Times come from a `times.csv`
(`file,time`, ISO UTC or MJD) or a `<image>.json` (`{"time": ...}`) per
image. A WCS is built by star matching (`starmatch.py`): stars detected in
each frame are matched to the middle frame's (phase correlation for the
shift, then RANSAC for an affine transform), and each frame gets the
reference frame's linear TAN WCS composed with its affine map. An optional
`sequence.json` gives `pixel_scale_arcsec`, `ra_deg`/`dec_deg`,
`exptime_s`, `fwhm_arcsec`, `magzp`; without them positions and rates are
in nominal units (1"/px about (0, 0)) and the PSF FWHM is measured from
the stars. Pixels at the format's maximum are saturated; with no
uncertainty image the noise model is estimated from the data (bright
sources' own photon noise is then missing, so their scores run high).
Phase correlation tolerates small field rotations only (a few degrees).
On a simulated ground sequence converted to 16-bit PNG without WCS, all 8
asteroids were found as from the FITS, within 0.01–0.05 px and with the
same rates.

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
     detections make a tracklet. Between strong peaks (z ≥ 7) the
     tolerance allows for close-object curvature (500 × rate² "/day²) and
     3–4 detections give a curved seed: a fast NEO's third detection can sit
     2–7 px off the chord through the other two. A tracklet sharing two
     detections with a stronger one is dropped only if it is the same path.
   - *Shift-and-stack* (`shiftstack.py`, `backend.py`): every velocity on a
     grid up to 0.25 deg/day, per-frame contributions clipped at 4σ so no
     single frame can carry a hypothesis. Backends (`--backend`): numba
     (parallel CPU, cache-blocked), cupy (NVIDIA GPU: a CUDA kernel, one
     thread per grid pixel over chunks of 1024 velocities) and a numpy
     reference. `auto` uses the GPU when CuPy finds one and the kernel
     passes a self-test against the numpy reference (once per process),
     else numba. The CUDA kernel's logic is tested here through a
     line-by-line Python transliteration; it has not yet run on a GPU (no
     NVIDIA device on the development machine), hence the self-test.
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
   curved path fits any 3 points. The straight fit is tried both with
   frames re-selected and on the curved fit's own frames (a straight line
   through a curved path can reach frames the object never visits).
   The curved fit is kept instead when every straight fit fails flux
   consistency and the curved fit is strong and physical: z ≥ 9 on every
   used frame, curvature within 500 × rate² "/day², consistent fluxes.
   NEOs a few tenths of an AU away curve by several arcsec over three
   frames (WISE's orbital parallax): a W1 13.7, 1.02 deg/day mover at
   0.29 AU sat 4" off a straight line and was dropped before.
   A final track's path is re-fitted on its fully covered frames (≥ 90%
   of the PSF on valid pixels). Otherwise the fit can bend the path toward
   valid pixels to keep a partly masked frame; that put Lachesis's
   positions 0.8″ off.
6. **Vetting** (`detector.py`): paths moving less than 4 PSF FWHM over the
   visit are rejected (the largest separation of the path's positions, so
   curvature counts; 0.005 deg/day over a 33 h NEOWISE visit, while the
   slowest of 87 known asteroids in our visits moves 0.084); tracks on
   bright-star residuals in every frame they use are rejected (each used
   frame's flux error > 2× its blank-sky error, with more extra variance
   than the track's own photon noise explains; bright asteroids like
   Oceana are 2.6× from their own noise and stay); tracks whose score rests
   on one frame are rejected (without it < 0.6 of the score and < 8 left;
   real asteroids keep 0.66–0.97); tracks with ≥ 75% of their frames on a
   static source (the frames' median sky > 3× the single-frame noise
   within 1 px) are rejected -- in crowded fields slow false tracks thread
   faint, partly variable stars; extended tracks are rejected (the
   frame-combined flux fitted with a Gaussian twice the PSF width, over
   the PSF-fit flux, above 2.2; a point source gives ~1.7, 1012
   injected and known asteroid tracks ≤ 2.11) -- bright-star ghosts and halos drift slowly
   through dense fields and set the top false tracks there; per-frame
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

**Thresholds on real WISE** (`nullcal.py`, `io.default_thresholds`).
`detect` runs the detector on one time-scrambled copy of the visit
(`--null-scrambles N`, default 1 on real WISE frames, 0 = off). Its *null
level* (5th-highest candidate score) sets the visit's thresholds,
level × 1.80 / 1.38 / 1.25 for 3 / 4 / ≥ 5-frame tracks (`--null-ratio`),
and the copy gives an **artifact veto**: a candidate within 60″ (at the
reference time) of a scrambled-copy candidate scoring ≥ 1.2× the null
level is rejected, as something that recurs whatever the frame order isn't
moving.

The false-alarm tail varies by ~2× across the sky: null levels are
6.5–8 along the ecliptic and at high latitude, 9–12 in the Galactic plane
and 11–16 toward the bulge. The factors are calibrated on 30 visits
(`sims/real_v1` + `sims/real_v3`, 11 in the plane). Held out one visit at
a time (calibrated on the other 29):

| scheme | false per null copy | W1 15–15.5 / 15.5–16 / 16–16.5, off plane | in plane |
|---|---|---|---|
| per-visit, 1 copy, static-source + extendedness vetting (default) | 0.044 | 0.92 / 0.78 / 0.41 | 0.76 / 0.27 / 0.07 |
| per-visit, 1 copy, static-source vetting only | 0.067 | 0.95 / 0.77 / 0.31 | 0.67 / 0.24 / 0.02 |
| per-visit, 1 copy, neither | 0.067 | 0.92 / 0.65 / 0.11 | 0.61 / 0.15 / 0.00 |
| per-visit, 2 copies | 0.078 | 0.92 / 0.63 / 0.10 | 0.59 / 0.12 / 0.00 |
| global thresholds | 0.122 | 0.80 / 0.35 / 0.00 | 0.57 / 0.03 / 0.00 |
| guard (global, raised for noisy visits) | 0.011 | 0.80 / 0.35 / 0.00 | 0.53 / 0.03 / 0.00 |

Global thresholds calibrated across such a sky are set by the plane and
cost completeness everywhere else. (On the 15 ecliptic visits of
`real_v1` alone the opposite seemed true -- per-visit thresholds gained
nothing there -- because those visits are alike.) In the bulge (null
levels up to 15.7) the thresholds reach 20–26 and practically nothing is
detected. Without scrambled copies (`--null-scrambles 0`) real WISE uses
fixed thresholds 9.75 / 9.25 / 9.25 calibrated on the 22 quiet visits:
valid away from the plane only. `--null-mode guard` keeps those fixed
thresholds and raises them only above the null level `GUARD_LEVEL` (8.8).

**Crowded fields.** After registration the detector measures the fraction
of covered sky within 1 px of a static source (z > 3, the static-source
vetting's map). From `crowded_frac` = 0.45 on, the visit is flagged
`crowded` in the stats and the listing. Over 30 real NEOWISE visits the
fraction was ≤ 0.34 wherever injected W1 14.5–15.5 asteroids were found
reliably, and ≥ 0.54 in the five bulge and plane fields, where null levels
reached 9.4–14.8 and completeness collapsed. `--skip-crowded` stops there
(12 s on a bulge visit, instead of hours), with no candidates and no null
calibration.

**Reporting** (`report.py`): `--ades FILE` writes ADES PSV for the MPC,
one row per frame a detection uses: the track position at that frame's
time, uncertainties from the centroid error plus a 0.25" floor, and the
per-frame magnitude (uncertainty floor 0.03 mag). For WISE the station is C51 and each row carries the
spacecraft position from `ERTH2SC`. Known objects get their designation.
The submitter/observer names are placeholders. Check the WISE `mode` and
`astCat` values against the MPC's documentation before a real submission.
`--gifs DIR` writes an animated cutout per detection: registered image and
static-sky residual with the predicted position marked.

**Suspected unknown asteroids** (`review.py`). Each detection that passes
its threshold gets a status:
- `known`: it matches a catalogued object in `known.json` (written by
  `neowise fetch` from SkyBoT; truth.json for simulations) to within 2 px.
- `near-known`: no match, but a catalogued object at a similar rate (within
  30%) lies within 60", or 3× its ephemeris uncertainty if that's larger.
  It's probably that object with a poor orbit; worth a look.
- `suspect`: nothing catalogued nearby, a suspected unknown asteroid.
- `unchecked`: there's no catalogue to check against.

`detect` lists the suspects to review. `neodetect review DIR` then
classifies the detections again (`--skybot` first rebuilds `known.json`
from SkyBoT: at C51 via the spacecraft offsets for WISE, else at
`--station` or the headers' `MPCCODE`; raster input needs `ra_deg`,
`dec_deg` and `pixel_scale_arcsec` in `sequence.json`). It writes a cutout
sheet per suspect to `DIR/review/` (`index.html` shows them all):
- upper row: each frame around the track, resampled north up, east left,
  so the stars shift between frames and the mover stays at the crosshair;
- lower row: the same with the median of the other frames subtracted,
  which removes the stars;
- an orange border marks frames the score didn't use, dark blue marks
  masked or off-frame pixels.

It then asks for a verdict on each unreviewed suspect: `c` confirm, `r`
reject, `s` skip, `q` quit, optionally followed by a note. Without a
terminal, use `--confirm N ...` / `--reject N ...` with `--note`. Verdicts
go to `DIR/review.json` with the track's positions. They're matched to
detections by position (median separation < 2 px), not by index, so they
carry over when the visit is searched again; `detect` shows them in its
listing. `--ades FILE` writes the confirmed detections as ADES (trkSub
only, no designation). Before submitting to the MPC, note that a new
designation needs observations on two or more nights.

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
per frame). `sims/sweeps/wise_v2` is the same runs (same seeds) with
asteroids on two-body orbits (`--motion orbit`), so their paths curve as
real ones do; it shares the null runs. The calibrated thresholds (3 / 4 /
≥5 frames: 9.0 / 9.25 / 8.75) are the defaults. They give 0.04 false
detections per null run. Final M4 detector:

| W1 mag | 13.5–15.0 | 15.0–15.5 | 15.5–16.0 | 16.0–16.5 |
|---|---|---|---|---|
| completeness, straight paths (`wise_v1`) | 1.00 | 0.88 | 0.40 | 0.02 |
| completeness, orbits (`wise_v2`) | 0.98 | 0.90 | 0.33 | 0.02 |
| median ideal stacked SNR | 22–48 | 13.2 | 8.6 | 5.6 |

So the 50% point is at W1 ≈ 15.6, a stacked SNR of about 9: the search
goes about as deep as the single-frame limit while using many frames. By
rate, completeness is 0.65–0.9 up to 2 deg/day and 0.5–0.55 at 2–5
deg/day (fast movers cross the field in fewer frames).

Accuracy on orbit paths (`wise_v2`): 0.29″ median, 0.47″ rms in position,
0.6% in rate, 0.20° in position angle, 0.08 mag rms. Fitting curvature is
what makes that possible: on the 189 asteroids found by both, a straight
fit gives 0.99″ rms (1.67″ for paths > 1.5″ from a line) against 0.45″.
On the straight `wise_v1` paths the fit only costs (0.45″ rms against
0.37″ in M2), which is why `wise_v2` is the reference. Per 12-frame
sequence, the linker takes seconds; the shift-and-stack up to 0.25
deg/day takes ~40 s on 32 cores; the full evaluation (96 runs, `-j 4`)
takes ~1.5 h.

### Measured performance (simulated ground preset)

`sims/sweeps/ground_v1`: 24 runs with 12 asteroids (V 17–21, 0.05–2
deg/day) and 24 null runs, `ground` preset (1.2"/px, 2" seeing, 60 s
exposures, single-frame 5σ depth V = 20). Calibrated thresholds 11.75 /
11.5 / 9.75 (the instrument default for this preset, `io.default_thresholds`):
higher than for WISE, because frames close in time give fast chance
alignments (up to the 10 deg/day linking limit) more room. Completeness is
1.00 to V = 20, 0.77 at V 20–20.5 and 0.06 at 20.5–21; 0.85–0.91 at every
rate. Position 0.11" rms, rate 0.4%; magnitudes +0.05 biased, 0.15 rms.

### Measured performance (real NEOWISE W1)

`sims/real_v1`: 15 NEOWISE-R visits along the ecliptic (0.5° cutouts,
11–18 frames each, 2014–2015). Each visit has 2 injected copies (20
asteroids each, W1 13.5–17, 0.05–3 deg/day; 489 covered by ≥ 3 frames) and
3 time-scrambled copies (45 null runs). `sims/real_v2` injects on two-body
orbits seen from WISE (curved paths, 491 covered) and shares the null
runs. The detector runs with its default settings; the thresholds below
are calibrated on the scrambled nulls. Final M4 detector
(`evaluation_m4h.json`):

| | simulated (`wise_v1`) | real (`real_v1`) | real, orbits (`real_v2`) |
|---|---|---|---|
| global thresholds 3 / 4 / ≥5 frames | 9.0 / 9.25 / 8.75 | 10.25 / 9.25 / 10.25 | same |
| false detections per null run | 0.04 | 0.00 | 0.00 |
| unmatched detections per injected run | 0.04 | 0.13 | 0.07 |
| completeness W1 14.5–15.0 | 1.00 | 1.00 | 0.95 |
| completeness W1 15.0–15.5 | 0.88 | 0.95 | 0.96 |
| completeness W1 15.5–16.0 | 0.40 | 0.66 | 0.73 |
| completeness W1 16.0–16.5 | 0.02 | 0.16 | 0.30 |
| median ideal SNR at W1 15.5–16.0 | 8.6 | 14.4 | 15.1 |

The 50% point is W1 ≈ 15.9 on real data, against 15.6 simulated: real
frames are deeper than the `wise-w1` preset (~1.7× the SNR at a given
magnitude), and with the artifact handling below the real thresholds are
only ~1.1× the simulated ones. At M3 they were ~2× (20 / 16.75 / 13.75)
and the real 50% point was W1 ≈ 15.4.

- **What set the real thresholds, and what fixed it:** scattered-light
  streaks from a bright star just past a cutout edge in the bulge visit
  v012, which line up into slow tracks in every scrambled copy (streak
  mask, artifact veto); slow tracks on bright-star residuals (stationary
  test on the path's extent, star-residual vetting); and tracks resting on
  one frame (single-frame vetting). Each check removes none of the 680
  injected-asteroid detections tested.
- **Unmatched detections in injected runs** were 0.10–0.17 per visit,
  above the 0.05 target met on the held-out scrambled copies. Most
  borrowed one frame from an injected asteroid that frame release missed;
  releasing frames within 1.5 FWHM of any stronger track (`release_fwhm`)
  brings it to 0.07–0.13 at no completeness cost (confirmed in a full
  rerun). What
  remains is one event in the real frame order (v012, a 10.9 3-frame track
  on two bright-star wings and one faint source), which no scrambled copy
  reproduces by position, plus an asteroid seen in one frame only.
- **New sky** (`sims/real_v3`): 15 more visits -- 11 in the Galactic plane
  (on and off the ecliptic, including two bulge fields), plus high-latitude
  and ecliptic controls. Off the plane they behave like `real_v1`; in the
  plane the null tails are much heavier (above), mostly slow tracks
  threading faint, partly variable stars, plus a few frames with 10× the
  usual number of peaks (5% of frames, 12% of false-track frames). The
  numbers in the table above are for these ecliptic visits; see the
  threshold section for the 30-visit, held-out comparison.
- **Curved injections** (`real_v2`) are found as well as straight ones at
  W1 ≥ 15; the bright, fast ones (W1 14–15) at 0.95–1.00 (0.93–0.96
  before curved linking, which recovered 5 movers of 1.1–2.3 deg/day;
  2–5 deg/day completeness 0.33 → 0.43): close, fast movers cross a
  cutout in few frames.
- **Known asteroids:** (224) Oceana and (120) Lachesis (W1 ≈ 11.7–11.8) are
  found in every copy at score 236–264, within 0.1–0.2 px of the
  parallax-corrected SkyBoT ephemeris. (2882), at a stacked SNR of 12, is
  now found too (11.3); M3 lost it to an edge-pixel outlier.
- **Accuracy:** 0.27″ median / 0.39″ rms in position on `real_v1`, 0.32″ /
  0.60″ on the curved `real_v2`; 0.3–0.6% in rate; 0.17–0.28° in position
  angle; magnitudes −0.01 mag biased, 0.05–0.07 mag rms (aperture
  correction; M3 was 0.14 mag faint).
- **Cost:** ~3–6 min per visit on 32 cores, plus one run for the guard's
  scrambled copy. The full sweep (75 runs, `-j 4`) takes
  ~5 h.

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
