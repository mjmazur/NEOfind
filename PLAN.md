# NEOfind — plan

Detect moving objects (asteroids) in sequences of astronomical images, with
first-class support for WISE/NEOWISE single-exposure frames, a simulator that
produces matching synthetic data with ground truth, and an evaluation harness
in the style of `~/source/Projects/MeteorSimulator`.

## Decisions

- **astropy** is a required dependency (FITS, WCS, times, coordinates).
- **Reuse MeteorSimulator code** by copying and adapting modules (flux-
  conserving PSF stamping, photometric model, star-field generation, gnomonic
  helpers, CLI/config style, truth files, sweep/evaluate workflow). Copied
  modules note their origin.
- **WISE/NEOWISE is the priority.** The simulator's default preset is WISE W1;
  frames are written in WISE L1b style (`-int`, `-unc`, `-msk` FITS with WCS)
  so simulated and real data go through the same loader and detector paths.
- **CPU now, GPU later.** Compute-heavy kernels (shift-and-stack) go through a
  small array-backend layer (NumPy by default, numba optional, CuPy optional)
  so an NVIDIA GPU can be dropped in without restructuring.

## Meteors vs asteroids

| | MeteorSimulator | NEOfind |
|---|---|---|
| Per-frame signal | streak | point source (trails only for fast NEOs / long exposures) |
| Search space | template length × angle per frame | velocity (vx, vy) [+ acceleration] across 5–15 frames |
| Cadence | regular 1/fps | irregular: minutes–hours (WISE: once per ~95-min orbit, ~1 day) |
| Pointing | fixed | changes every frame (WISE: field drifts ~4'/orbit, random along-scan offset) |
| Dominant false positives | scintillation | cosmic rays, hot pixels, subtraction residuals, variable stars, chance noise alignments |
| Data format | PNG + metadata.json | FITS + WCS + MJD (+ uncertainty and mask images) |

The meteor detector's coherent multi-frame score `S = Σ num_k / sqrt(Σ den_k)`
generalizes directly to **shift-and-stack** (synthetic tracking / KBMOD-style
ψ/φ likelihood images): for each velocity hypothesis, shift each frame's
PSF-filtered difference image along the track and sum.

## Layout

```
neosim/          simulator
  config.py      InstrumentConfig, CadenceConfig, PointingConfig, PopulationConfig, ArtifactConfig, SimConfig
  presets.py     wise-w1 (default), wise-w2, ground
  optics.py      PSF FWHM <-> sigma, plate scale helpers           (from meteorsim)
  astrometry.py  gnomonic projection, galactic latitude             (from meteorsim)
  photometry.py  zero point from limiting magnitude (5σ PSF-fit), noise-equivalent area, SNR
  starfield.py   star counts per band, galactic-latitude scaling   (from meteorsim)
  cadence.py     observation times: uniform | visits | wise (orbit) | file
  pointing.py    per-frame WCS: dithered (ground) | wise scan geometry
  motion.py      asteroid population, ephemerides (linear + optional quadratic) and rotational light curves
  artifacts.py   cosmic rays, hot pixels, bad columns
  render.py      PSF stamping incl. trailing, noise, saturation      (from meteorsim)
  fitsio.py      WISE-style int/unc/msk writer
  preview.py     PNG previews, reprojected blink GIF, truth overlay
  simulate.py, cli.py

neodetect/       detector
  io.py          files / directory / glob -> frames (FITS+WCS+time, unc, mask); WISE L1b and simulator layouts
  register.py    reproject onto a common grid by WCS (fallback: star matching)
  preprocess.py  static-sky template (sigma-clipped median, partial coverage aware), PSF matching,
                 cosmic-ray masking, noise model -> per-pixel weights
  likelihood.py  per-frame ψ = PSF⋆(w·r), φ = PSF²⋆w
  backend.py     NumPy / numba / CuPy selection
  shiftstack.py  velocity-grid search, coarse-to-fine
  sources.py     single-frame detections on difference images
  linking.py     detections -> tracklets under linear motion (pair seeds, predict, extend; kd-tree)
  trackfit.py    coherent refinement of (x0, y0, vx, vy[, ax, ay]); per-frame flux, magnitude
  vet.py         stationary rejection, single-frame-dominance (sigma-G light-curve) filter, bright-star masks
  report.py      detections.json (RA/Dec per epoch, rate, PA, mag, score), MPC 80-col/ADES, cutout GIFs, overlay
  known.py       known-object association (IMCCE SkyBoT, JPL sb_ident)
  evaluate.py, validate.py, cli.py   detect | evaluate | validate

neowise/         WISE/NEOWISE data access
  query.py       IRSA IBE metadata search by position/time/band; by object via IRSA MOST
  download.py    L1b int/unc/msk frames or cutouts, local cache
  visits.py      group frames into visits
  maskbits.py    WISE mask bits -> internal flags
  inject.py      synthetic asteroids injected into real frames (injection-recovery)

sweep.py, tests/
```

## Detector pipeline

1. Load + register onto a reference WCS grid; track per-pixel coverage.
2. Static-sky subtraction, PSF matching, cosmic-ray masking, noise model;
   `validate` checks the whitened residual is N(0, 1).
3. Two seeding paths: shift-and-stack (faint/slow), detect-and-link (fast
   NEOs, where a fine velocity grid over a 1-day WISE baseline is too large).
4. Coherent track fit and robust scoring; vetting.
5. Threshold calibrated on asteroid-free runs (search inflates the noise
   maximum); completeness vs magnitude and rate from sweeps.

## WISE notes

- IRSA IBE tables `wise/allsky/4band_p1bm_frm`, `wise/neowiser/p1bm_frm`;
  frames 1016×1016 at 2.757"/px, W1/W2 PSF ≈ 6.1"/6.4", ~7.7 s exposures.
  NEOWISE-R (2013-12 to 2024-08) is W1/W2 only.
- A point near the ecliptic is seen on ~12 consecutive orbits (~95 min apart)
  as the scan circle precesses ~1°/day. Confirm details against the
  Explanatory Supplement when writing `visits.py` and `maskbits.py`.
- Real-data truth: (a) known asteroids from SkyBoT/sb_ident and MOST, (b)
  injection-recovery of synthetic asteroids in real frames.
- Learned in M3 (see `neowise/`):
  - L1b WCS is `SIN-SIP` (distortion up to ~6"); astropy handles it, and
    IRSA cutouts keep the full header with CRPIX shifted. `MJD_OBS` is the
    exposure midpoint.
  - Static bad pixels (~3%: bits 1, 3, 4, 6, 9, 26) are NaN in `-int`.
    Bits 21 and 28 keep data with 10-15x inflated `-unc`. Bit 27 (co-add
    temporal outlier) must not be masked: moving objects set it.
  - `-unc` overestimates the noise by ~1/0.55 in variance; the noise model's
    alpha absorbs it.
  - IBE `INTERSECT=OVERLAPS` is generous (cutouts of non-overlapping frames
    fail with HTTP 500), so overlap is computed from the frame corners. IRSA
    answers 503 to bursts of more than ~10 parallel requests; cutouts come
    back gzipped.
  - SkyBoT ignores observatory code C51 (WISE); positions are shifted from
    the geocenter to the spacecraft with the header's `ERTH2SC` vector
    (3.5" for the main belt, much more for NEOs).
  - Single-exposure PSF: the empirical ePSF agrees with the official
    cryogenic W1 PSF (Explanatory Supplement IV.4.c) to ~5%: effective FWHM
    ~2.2 px, NEA ~14.5 px. 4% of the flux lies beyond 8 px.
  - Each frame also has artifact tables (`-art-w1-{D,H,P}.tbl`: diffraction
    spikes, halos, latents/persistence) for M4.

## Status and lessons

- **M1 done** (simulator). **M2 done** (detector on WISE-like simulations).
  **M3 done**: `neowise` search/download/known objects/ePSF/
  injection/scrambled nulls. The detector runs on real visits unchanged
  except for the registration masking (below). Real-data results are in the
  README: same 50% point (W1 ≈ 15.4) as on simulations, with ~1.7x
  higher thresholds offset by ~1.8x deeper real frames. Known bright
  asteroids are recovered at 0.1-0.2 px.
- **M4 in progress** (robustness and reporting). Done so far, details in
  the README:
  - Aperture correction from each visit's stars (`photometry.py`): magnitude
    bias +0.14 → −0.01 mag on injected asteroids, rms 0.15 → 0.04.
    Against the NEOWISE source catalog (IRSA `neowiser_p1bs_psd`), Oceana's
    per-frame magnitudes are still ~0.08 mag faint. The likely cause is
    track position error, which acceleration fitting should reduce;
    recheck.
  - Curved paths. SkyBoT ephemerides show apparent paths bending ~15"/day²
    for main-belt asteroids (|a|/rate 40–120 "/day² per deg/day), straying
    up to 3.3" (median 2.2") from a straight line over a 1.4-day visit; a
    quadratic fits to 0.02". Cause: the observer's solar acceleration
    divided by the distance. Straight tracks lose a median 3% of the score
    (up to 45% for fast curved movers). Hence the bounded acceleration
    in track fit and linking, and two-body orbit injections
    (`neosim/orbit.py`, validated against SkyBoT to ~0.1"). The `neosim`
    simulator's own `motion.py` is still quadratic in the tangent plane.
  - ADES PSV output and cutout GIFs (`report.py`).
  - `n_refine` 400 → 2000: didn't change crowded-field completeness
    overall (0.73 at W1 14.5–15.5).
  - Per-visit thresholds from time-scrambled copies (`nullcal.py`;
    `detect --null-scrambles`, default 2 for WISE). Completeness at the
    same false-alarm rate: 0.79 → 0.90 at W1 15–15.5, 0.37 → 0.50 at
    15.5–16. The null level is a mid-tail statistic (5th-highest score per
    copy), which one copy measures with ~5–10% scatter.
  - Local noise map (`preprocess.local_sigma`). The highest-ranked false
    tracks in scrambled copies of the Oceana visit came from scattered
    light: a fan of streaks along the cutout edge in two frames, from a
    bright star just outside the field. On the bulge visit v012 the
    SNR scatter within 10 px of frame edges was 1.3–1.5 (1.0 elsewhere;
    clean visits show no edge effect, and native frames are clean at their
    edges). Deflating weights where the scatter in 31-px windows exceeds
    1.2 brings v012's edges to 1.0 and its z>5 tail down ~20×. It deflates
    1–7% of each Oceana frame.
  - Bright-object astrometry (found with ADES on the Lachesis visit:
    positions off SkyBoT by up to 0.8″ in an arch, fitted curvature −23
    vs −14″/day²). Three causes, three fixes:
    - The cosmic-ray ratio tests used Gaussian limits. Real WISE cores are
      sharper (peak/neighbour ~2.0 vs 1.67, diagonal ~4 vs 2.8), so they
      masked a bright asteroid's own pixels. Limits now come from the
      frame's stars (width-selected to exclude hits; the visit median for
      frames with < 10 stars). Masked neighbours are no longer counted as
      zero. Fewer pixels get flagged per frame (10–50% fewer), so check the
      false-alarm rate in the re-run.
    - Partly masked frames count only above 60% PSF coverage, so the fit
      bent the path to keep them. Final paths are now re-fitted on frames
      with ≥ 90% coverage (`TrackEvaluator.refit_geometry`).
    - The first local-noise version let Oceana's template imprints deflate
      its own weights (score 263 → 204). Compact |z| > 5 sources are now
      excluded from the estimate; extended streaks still count.
  - `real_v2` (curved injections; acceleration fit, before local noise and
    the fixes above): with per-visit thresholds, completeness rose at the
    faint end (0.60 at W1 15.5–16, 0.28 at 16–16.5; `real_v1`: 0.50, 0.19)
    but fell at the bright end (0.92 at 14–14.5). 12 of the 13 misses
    brighter than 15 were fast movers (0.6–2.8 deg/day) on 3–5 frames with
    no candidate at all. Two causes, on top of the cosmic-ray masking fix:
    - Close objects curve far more than main-belt ones. The rate changes
      with the distance, adding ~2 (v_r/v_⊥) rate² to the curvature:
      340–970"/day² for the misses, against a bound of 150 × rate.
      The bound now includes 500 × rate² (`accel_per_rate2`).
    - Triage's bilinear coverage test is stricter than the exact one near
      masked pixels, so a seed could score −inf and be dropped. Such seeds
      now get the exact score.
    Both checked on two of the misses (found at 52.8 and 40.3). Astrometry
    was also worse (0.54" rms vs 0.30") and needs a recheck after the
    re-run (`detections_m4b.json`). The simulated `wise_v1` run started
    before these two changes.
  - Astrometric offset: stars and asteroids alike sat ~0.45″ from the
    NEOWISE catalog and SkyBoT (the two agree to ~0.1″), along a fixed
    detector direction, so the sign flipped between visits scanned in
    opposite directions. In native px it is (+0.147, +0.085), constant to
    ~0.01 px over 16 frames of 8 visits: a Gaussian-fit vs catalog PSF-fit
    centroid convention. It's now corrected in reported positions. Injected
    stamps are drawn at the same offset (`truth.json`
    `centroid_offset_px`). Older injection sets without that key are
    evaluated with the correction off, so `real_v1`/`real_v2` numbers stay
    comparable. Residual after the fix: Oceana 0.02″/0.06″ vs the catalog;
    Lachesis ~0.2″, where masked core pixels perturb its centroids.
  - Simulated regression (`wise_v1`, `detections_m4.json`, before the
    curvature-bound, triage and astrometry changes): thresholds 12.5 /
    10.0 / 8.75 (M2/M3: 10.25 / 9.5 / 8.25), completeness 0.80 / 0.38 at
    W1 15–15.5 / 15.5–16 (0.82 / 0.40), 2–5 deg/day 0.45 (0.55). Two causes:
    - Curved 3-frame tracks: 6 parameters pass through any 3 points, so
      chance alignments of artifacts got easier (9 of 11 null candidates
      ≥ 9 used 3 frames with 85–1270"/day²). Tracks using < 4 frames are
      now fitted straight (`accel_min_frames`), restarting from the line
      through the curved track's positions at its frames.
    - The star-measured cosmic-ray limits were ~10× the Gaussian ones on
      simulations: elongated cosmic-ray tracks passed the width cut, which
      switched masking off (a 16.2 null track of unmasked hits). Stars must
      now also be round and not concentrated in one pixel, and the limits
      are capped at 1.75× the Gaussian (simulations now measure ~1.05×,
      real WISE 1.1–1.5×).
    The three worst simulated nulls now match M3 (9.9 / 8.1 / 10.8 vs
    7.9 / 7.8 / 10.8). The fast movers stay found (52.8, 38.3).
  - Pending, in order: `real_v1`, `real_v2` and the simulated `wise_v1`,
    all with every fix (`detections_m4b.json`). After that, recalibrate `nullcal.NULL_RATIO`
    and the fixed thresholds.
- Real-data limits found in M3, and what became of them:
  - Crowded fields (|b| < ~20°) dominate the false-alarm tail. They also
    saturate the triage cap (`n_refine` = 400), which costs ~20% completeness
    there. → cap raised (no gain); per-visit thresholds and local noise
    address the tail. Still open: star-subtracted residuals (fewer seeds).
  - Thresholds from 45 scrambled nulls are set by single events. → per-visit
    thresholds (done).
  - Photometry is 0.14 mag faint. → aperture correction (done).
  - Tracks along cutout edges can fail flux consistency because of
    edge-pixel outliers. → local noise in crowded fields; recheck (2882).
  - `all_world2pix` diverges far outside a SIP frame; `io.world2pix` falls
    back to the linear inverse for those points.
- Real WISE masks flag ~3% of pixels, scattered. Invalidating each bad
  pixel's whole cubic stencil (as in M2) lost ~45% of every frame; now bad
  pixels are filled by normalized convolution and only samples dominated by
  them are invalid (strict stencil masking only on sources).
- On real data the whitened SNR tails are heavier than on simulations
  (P(z>4) ~10x Gaussian): satellite/debris streaks, edge artifacts and
  noise near masked pixels. False alarms are calibrated on real data with
  time-scrambled visits (all artifacts, no linear motion).
- Search losses are negligible: refined detection scores equal the
  statistic at the true track, which tracks the ideal SNR.
- False alarms are dominated by chance alignments of 3-4 single-frame
  artifacts (cosmic rays whose footprint is genuinely PSF-like: corner hits,
  ~2 px diagonal tracks, hits on stars). Hence: three-test CR masking in
  native pixels, per-frame clipping in the stack, flux-consistency vetting,
  and detection thresholds that depend on how many frames a track uses.
- CPU cost: shift-and-stack up to 0.25 deg/day on a full WISE visit
  (12 frames, ~2000x1700 grid, 44k velocities) takes ~40 s on 32 cores
  (cache-blocked numba kernel, ~30 G adds/s, memory-bandwidth bound).
  Each doubling of the rate range is 4x the work, which is where a GPU
  backend pays off.
- False alarms on real data (scrambled `real_v1`, score ≥ 9, before local
  noise): of 238, 146 have a frame within 15 px of a cutout edge, 145 a
  bright source within 6 px, and 94 ≥ 10 bad pixels within 6 px; v012
  dominates. They skew to tracks whose leave-one-out template had few
  frames (2–5, vs 6–8 for real detections), but a cut on that costs too much
  completeness (≥ 4 frames: −32% false alarms, −11% real).
- Open items for M4+: WISE artifact tables (`-art`: spikes, halos,
  latents) as masks; a template-depth-dependent threshold; second-pass artifact masking on star-subtracted
  residuals back-projected to native pixels (cosmic rays on bright-star
  wings escape the flat-background sharpness tests; vetting currently
  catches them); real WISE PSF (wings) in both simulator and detector;
  ground-preset PSF matching between frames; trailed templates for fast
  movers in long exposures.

## Milestones

1. **Simulator.** WISE (default) and ground presets, WISE-style FITS output
   with WCS, cadence and pointing models, asteroid population, artifacts,
   truth.json, previews, tests.
2. **Detector core on WISE-like simulations.** io, registration, subtraction,
   noise model, linking + shift-and-stack seeding (NumPy backend), track fit,
   vetting, validate, sweep/evaluate; first completeness curves.
3. **Real WISE data.** query/download/cache, mask bits, visits, known-object
   association, injection-recovery, MOST-based known-NEO test set.
4. **Robustness and reporting.** WISE PSF model (wings), bright-star
   artifacts/latents, MPC/ADES output, cutout GIFs. Added during M4:
   aperture correction, curved paths (acceleration fit, orbit injection),
   per-visit thresholds, local noise map.
5. **Generic user input.** Directory/glob, PNG + sidecar times, star-matching
   registration; ground preset tuning.
6. **Performance.** numba CPU and CuPy GPU backends for shift-and-stack;
   tiling for full frames.
