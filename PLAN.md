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
  - With every fix (`detections_m4b.json`, commit aac1c79):
    - Simulated `wise_v1`: thresholds 10.25 / 10.0 / 8.75, completeness
      0.84 / 0.38 at W1 15–15.5 / 15.5–16, 2–5 deg/day 0.50; position rms
      0.45" (M2 0.37"), PA rms 0.19° (M2 0.13°) -- still open.
    - Real data: the curved 3-frame nulls of v012 (27.0, 24.8) are gone, but
      calibration is now set by slow 5-frame tracks on bright-star residuals
      in crowded fields: v012 at RA 285.99 Dec −22.70 (17.5, all three
      scrambles; 19.2 as a 3-frame track before), v003 (11.5, at the
      curvature bound; 8.4 fitted straight). Local noise lowers both (24.6
      and 14.8 without it). Fitted ratios became 1.63 / 1.39 / 1.75 and
      global thresholds 16.75 / 11.0 / 17.5, so per-visit completeness at
      15.5–16 / 16–16.5 is 0.47 / 0.00 on `real_v1` (0.47 / 0.16 with the
      old ratios at 0.09 false per null) and 0.57 / 0.06 on `real_v2`
      (0.62 / 0.26). Photometry −0.009 mag (rms 0.042); position 0.32" rms
      (`real_v1`), 0.44" (`real_v2`).
  - Second-pass cosmic-ray masking (`register._second_pass`): the run_0030
    9.9 null was a cosmic ray on a faint star, which broadened it past the
    first pass's tests (M3 masked it with Gaussian limits; the
    star-measured ones were 7% looser). Repeating the tests on frames minus
    the static sky (median of the registered frames, mapped back to native
    pixels; noise allowance 10% of the static sky for subtraction errors)
    catches these hits. The 7 worst simulated nulls go from 11.6 / 10.8 /
    9.9 / 9.9 / 9.9 / 9.9 / 9.8 to 7.8 / 7.2 / 9.9 / 7.9 / 8.5 / 7.9 / 9.8;
    Oceana, Lachesis and 33 injections score the same; +2–5 s per visit.
  - With the second pass (`detections_m4c.json`), simulated `wise_v1`:
    thresholds 9.0 / 9.25 / 8.75 (M2 10.25 / 9.5 / 8.25), completeness
    0.88 / 0.40 at W1 15–15.5 / 15.5–16, 2–5 deg/day 0.55.
  - Simulated astrometry (0.45" rms vs M2 0.37", PA 0.19° vs 0.13°) is the
    curvature fit: simulated asteroids move in straight lines (truth
    acceleration 0), and on faint tracks the two extra parameters fit noise
    (median 12"/day² fitted, 17 tracks at the bound with 1.09" rms).
    Keeping curvature only when it raises S² by ≥ 9 helps the simulations
    (0.69" → 0.56", PA 0.27° → 0.17° on 8 runs) but hurts real curved
    injections (`real_v2`, 0.58" → 0.62" on 5 visits), so the detector is
    unchanged. On the 64 simulated asteroids found by all three, M3 gives
    0.30" median / 0.51" rms / PA 0.17°, the curvature fit 0.38 / 0.69 /
    0.27, and curvature-if-significant 0.30 / 0.56 / 0.17: the fit accounts
    for the whole regression. So the simulator now curves its asteroids
    (`--motion orbit`, two-body as for `real_v2` injections), and
    `sims/sweeps/wise_v2` is `wise_v1` with orbit motion (same seeds, shared
    null runs) for measuring astrometry on realistic paths.
  - The v012 calibration artifact is scattered light from a very bright star
    just past the cutout's right edge: a fan of horizontal streaks ~200 px
    long in 3 of the 7 frames covering that spot, fixed on the detector, so
    their segments line up as a slow track in every scrambled copy. Fixes:
    - Streak mask in the second pass (`cosmics.streak_mask`). Real WISE noise
      is correlated along rows (line-mean scatter 0.22 vs 0.165 in other
      directions on v012), so the threshold is relative to each frame's own
      scatter per direction. Compact round sources (5–100 px above 5σ) are
      left out of the line sums, or a line beside Oceana picked up its wings
      and masked its core. v012 copy 2: 17.5 → 13.5; Oceana, injections
      into v012 unchanged; nothing masked on simulations.
    - Artifact veto (`nullcal`): candidates within 60" of a scrambled-copy
      candidate scoring ≥ 1.2× the null level are rejected; calibration
      applies it to each held-out copy using the other copies.
    - Stationary rejection on the path's extent (≥ 4 FWHM over the visit,
      was the speed at t_ref ≥ 2 FWHM): after the veto the ≥ 5-frame ratio
      was set by slow tracks on bright-star residuals moving 2.9–3.7 FWHM
      over 33 h, one only because of fitted curvature. The slowest of 87
      known asteroids in the real visits moves 0.084 deg/day (~70 FWHM).
    On the m4b detections, veto + path cut alone give ratios 1.48 / 1.36 /
    1.47 (was 1.63 / 1.39 / 1.75), per-visit completeness at 15.5–16 /
    16–16.5 of 0.53 / 0.10 on `real_v1` (0.47 / 0.00) and 0.65 / 0.23 on
    `real_v2` (0.57 / 0.06), 0 false per held-out copy; the veto cost 1 and
    3 injections.
  - v012 with streak mask, veto and path cut (`detections_m4d.json`): null
    level 10.2 → 7.75, injections found at per-visit thresholds 13 → 17 of
    33. Two kinds of false track remained: a 13.5 4-frame track in one copy
    at the streak spot (the other copies have only 7.0-7.1 there, below the
    veto bar), and a 14.2 3-frame track in the real frame order (both
    injection runs) whose three frames all sit on bright stars -- W1 12.5
    with z ~8 per frame, nothing in the two masked frames on its path.
    - Star-residual vetting: reject a track if every used frame's flux
      error is > 2× the frame's blank-sky error with more extra variance
      than the track's own photon noise (> 2 × flux). On the m4b detections
      it flags 24% of scrambled-copy and 28% of other false tracks ≥ 9 and
      none of 679 injected-asteroid detections (`real_v1` + `real_v2`), nor
      Oceana or Lachesis (2.6× from their own noise); never on simulations.
      Removes the 13.5, the 10.6 and the 14.2 above.
    - Veto settings re-checked (score bar 0.8-1.2 × level, radius 20-60"):
      1.2 × level within 60" is best; lower bars veto 7-19 asteroids.
    - Estimated on m4b with path cut, star-residual vetting and veto:
      ratios 1.48 / 1.36 / 1.47; per-visit completeness at W1 15–15.5 /
      15.5–16 / 16–16.5 of 0.90 / 0.56 / 0.12 (`real_v1`) and 0.87 / 0.65 /
      0.24 (`real_v2`), before the streak mask's gain.
  - `real_v1` with everything (`detections_m4e.json`): v012 no longer sets
    any ratio (level 7.68, completeness 0.52; its worst held-out false track
    1.38 × level). Global thresholds 11.5 / 18.0 / 10.25, completeness 0.88
    / 0.54 / 0.14 at W1 15–15.5 / 15.5–16 / 16–16.5 (m4b: 0.81 / 0.16 /
    0.00). The 4-frame class was then set by one 17.9 null in v002 (ratio
    2.48): one z 17 frame plus three on bright stars with little weight,
    just inside the star-residual test's photon-noise allowance.
    - Single-frame vetting: reject a track if, without its strongest frame,
      < 0.6 of the score and < 8 is left. Real asteroids keep 0.66–0.97 of
      it (n equal frames: sqrt((n-1)/n)); on `real_v1` + `real_v2` it
      removes 30% of false tracks ≥ 9 and none of 680 injected-asteroid
      detections, nor Oceana or Lachesis; nothing on simulations.
    - On the m4e detections (the rule is scale-free, so it applies exactly
      to saved results): global thresholds 10.0 / 9.25 / 10.25, ratios 1.30
      / 1.32 / 1.46, completeness 0.95 / 0.68 / 0.14 (global) and 0.93 /
      0.62 / 0.13 (per visit), 0 false per held-out copy.
  - Simulated astrometry on realistic paths (`wise_v2`, all 24 runs, 189
    asteroids found by both fits): curvature fit 0.29" median / 0.45" rms,
    PA 0.21°, vs a straight fit 0.55" / 0.99", 0.22° (paths > 1.5" from a
    line: 0.52" vs 1.67" rms, scores 28.7 vs 25.0; < 0.5": 0.47" vs 0.45",
    the noise-fitting cost).
    The `wise_v1` regression was that cost alone; `wise_v2` is now the
    simulated astrometry reference.
  - Final M4 numbers (`evaluation_m4f.json`: m4e detections plus
    single-frame vetting, applied to the saved results; `real_v2` shares
    `real_v1`'s null runs):
    - `nullcal.NULL_RATIO` 1.81 / 1.53 / 1.39 → 1.30 / 1.32 / 1.46 (held-out
      false 0, veto lost 0 asteroids). Default fixed thresholds (simulated)
      10.25 / 9.5 / 8.25 → 9.0 / 9.25 / 8.75.
    - Real (`real_v1` / `real_v2`): global thresholds 10.0 / 9.25 / 10.25,
      completeness at W1 15–15.5 / 15.5–16 / 16–16.5 of 0.95 / 0.68 / 0.14
      and 0.94 / 0.73 / 0.30 (M3: 0.79 / 0.37 / 0.01); 50% point W1 ≈ 15.9
      (M3 15.4). Per visit 0.93 / 0.62 / 0.13: with the artifacts handled
      the visits are alike (levels 6.7–8.2) and per-visit thresholds no
      longer beat global ones. Position 0.39" / 0.58" rms, magnitudes
      −0.01 ± 0.05–0.07. (2882) now found.
    - Unmatched detections in injected runs: 0.10–0.17 per visit, above the
      0.05 met on held-out scrambled copies -- each visit's real frame
      order is an extra draw of chance alignments (v012: a 10.9 3-frame
      track on two bright-star wings and one faint source).
    - Simulated: thresholds 9.0 / 9.25 / 8.75, completeness 0.88 / 0.40 /
      0.02 (`wise_v1`), 0.90 / 0.33 / 0.02 (`wise_v2`, orbits); 50% point
      W1 ≈ 15.6; 0.47" rms on orbit paths.
  - Template-depth threshold, examined: the false tracks' template depth
    (other valid frames at their positions, minimum over used frames) is
    lower than real asteroids' (median 4 vs 6; ≤ 4 for 57-67% vs 21%).
    Scaling scores by 1 + a·max(0, d0 − depth) and recalibrating beats a
    uniform threshold raise only slightly (at 0 unmatched: 0.57 / 0.12 vs
    0.57 / 0.07 at W1 15.5–16 / 16–16.5 on `real_v1`), and the "excess"
    it targets was mostly not real-order false alarms: of the unmatched
    detections in injected runs only one event (v012, 10.9) is in the real
    frame order -- counted in all four of its injected runs -- and the rest
    each borrowed one frame from an injected asteroid, 1.4–6.8" from its
    path, that frame release missed (stronger track's frame masked, or
    > 2 px). Releasing frames within 1.5 FWHM of a stronger track, covered
    or not (`release_fwhm`), removes those at no completeness cost
    (unmatched per injected run 0.167 → 0.100 `real_v1`, 0.100 → 0.067
    `real_v2`; offline estimate on m4f). So the real-order rate is ~1 event
    in 15 visits, near the 0.05 target, and a depth threshold isn't worth
    its completeness cost on this sample; it needs more visits to tune.
  - Per-visit threshold default, examined held out (calibrate on 14
    visits, score the 15th; m4f): global thresholds 0.044 false per null
    copy, completeness 0.95 / 0.68 / 0.14 (`real_v1`); per-visit with 2
    copies 0.089 and 0.93 / 0.63 / 0.13 (it also lowers the bar in the
    quietest visits: two 8.5 false tracks in r222); 1 copy 0.067. A guard
    -- global thresholds, raised by level / 8.4 when a visit's one-copy
    null level exceeds the calibration sample's highest -- 0.044 and 0.95 /
    0.65 / 0.14. Now the real-WISE default: `io.default_thresholds` (10.0 /
    9.25 / 10.25), `--null-scrambles` 1, `--null-mode guard` (per-visit
    still available); `evaluate` reports the guard scheme.
  - Bright fast movers in `real_v2` (8 of 197 at W1 < 15 missed, all
    1.1–2.9 deg/day on 3–5 frames, none in `real_v1`'s straight paths):
    all search failures, not thresholds. Causes: the linker predicts a
    pair's other frames on a straight chord, which a fast curved path
    misses by 2–7 px, and fits curvature only from 5 detections, so a
    3-4-detection seed starts straight and refines to an inconsistent
    optimum; and the straight refit of a 3-frame track reached a 4th frame
    the curved object never visits (zero flux, consistency p = 0). Two of
    the 8 are faint (ideal SNR 7, 11).
    - Straight refit also tried on the curved fit's own frames, keeping
      the better consistent fit (`straight_refit_fixed_frames`, on):
      recovers that one, no losses, scrambled copies unchanged.
    - Linker curvature (`link_curvature`, off): match radius widened by
      the close-object curvature bound (500 × rate²) and 3-4-detection
      seeds fitted curved. Recovered 4 more but lost 2 (W1 14.3, 15.1;
      not traced) and raised scrambled-copy false tracks with fast curved
      chance alignments (v012 8.6 → 11.4, v011 9.2 → 10.1, v010 8.1 → 9.1).
      Tested with a separate threshold for its extra trials: one scrambled
      copy per visit and one `real_v2` injected run per visit (15 + 15),
      against m4f. Tracks >= 0.5 deg/day need thresholds x 1.1 to get back
      to 0 false (4 in 15 copies at x 1.0, all fast and curved, mostly the
      bulge visit v011); then fast movers found 49 → 48 of 97 (bright ones
      28 → 30 of 33, fainter ones lose to the higher threshold), slower
      ones unchanged (105 / 154). A 1 deg/day split can't reach 0 false (a
      10.1 at 0.95 deg/day). Net nothing: it stays off. Since only bright
      fast movers gain, a narrower version may pay: allow for curvature
      only when every linked detection is strong (z ≥ ~7), which chance
      alignments of noise peaks rarely are.
    - That gate, tested (same 15 + 15 runs, against m4g): z ≥ 5 left one
      extra false track (10.1, 4 frames, v011); z ≥ 7 none, fast movers 49
      → 50. Every curved variant lost one asteroid (W1 15.1, 0.7 deg/day):
      a curved chance tracklet with more detections shared two of its
      detections, and the linker dropped any tracklet sharing ≥ 2
      detections with a stronger one. Dropping it only when it is the same
      path (within 2 px at all its frames) fixes that; alone it changes
      nothing on the sample. z ≥ 7 + that dedupe: fast movers 49 → 51 of 97
      (bright 28 → 29 of 33), slower ones unchanged, no extra false tracks.
      Now the default (`link_curvature`, `link_curve_min_z` 7). Full rerun
      (`detections_m4h.json`, also with the refit fix) against m4g: no
      false tracks on any null run (72 simulated, 45 real); `wise_v1`
      unchanged; `wise_v2` +2 fast movers (1.9, 2.6 deg/day); `real_v2` +5
      (1.1–2.3 deg/day; W1 14–15 0.96 / 0.93 → 1.00 / 0.95, 2–5 deg/day
      0.33 → 0.43), none lost; `real_v1` +1 / −1. The bulge visit v011
      has many z ≥ 7 star residuals, so the allowance applies often there:
      its one-copy null levels rose 8.2 / 8.4 → 8.8 / 8.7 and a 3-frame null
      9.7 → 10.05, so the real 3-frame threshold is 10.25 (was 10.0) and
      the guard level 8.8 (was 8.4); ratios 1.30 / 1.31 / 1.45. The one
      loss (v011, W1 15.6) is found as a 3-frame 9.8 instead of a 4-frame
      9.6.
  - Real WISE PSF (wings) in the detector, examined: against each visit's
    empirical ePSF (`neowise.epsf`, ~10% of the flux beyond 3 px), the
    Gaussian matched filter keeps 0.993–0.994 of the SNR at every
    sub-pixel phase, so a winged filter would gain < 1%; the PSF's other
    effects are already measured empirically (aperture correction 1.15,
    centroid offset 0.15 px). Not pursued. WISE artifact flags: the L1b
    masks carry none (spikes, halos, ghosts, latents are flagged only per
    source, as `cc_flags` in the single-exposure catalog); using them would
    need a catalog query per visit -- left for when the new visits show
    whether bright-star false alarms still matter.
  - New sky (`sims/real_v3`, 15 visits: 11 Galactic plane incl. 2 bulge
    fields, high-latitude and ecliptic controls; 4 downloads interrupted by
    IRSA 503s and refetched; the bulge needed a relaxed ePSF star
    selection). Out of sample, the `real_v1` calibration failed in the
    plane: null levels 9–12 (bulge 11–16) vs 6.5–8 elsewhere; global
    thresholds gave ~2 false per copy there, the guard 0.9. Held out one
    visit at a time over all 30 visits: per-visit thresholds from one copy
    0.067 false per copy, completeness 0.92 / 0.65 / 0.11 at W1 15 / 15.5
    / 16 off the plane (0.61 / 0.15 / 0.00 in it); global thresholds and
    the guard 0.35 at 15.5–16 off the plane (set by the plane); a floor on
    quiet-sky thresholds 0.078 and about the same completeness. So
    per-visit mode with one copy is the real-WISE default again,
    `NULL_RATIO` 1.81 / 1.62 / 1.66; fixed real-WISE thresholds (no
    copies) 10.25 / 9.25 / 11.75 from the 17 quiet visits, valid off the
    plane. The `real_v1`-only conclusion (global beats per-visit) came from
    a sample of alike visits. Plane false tracks: about half thread faint
    static (partly variable) stars -- rejecting tracks with every frame on
    a static source would remove 6–50% of them, losing no asteroids; and
    heavy-tailed frames (5% of frames, 10× the usual peaks) carry 12% of
    false-track frames -- a per-frame tail calibration would help there.
    The bulge (null levels to 15.7, 26 min per run at the refine cap) is
    effectively unsearchable at WISE resolution.
  - Static-source vetting (`static_source_frac` 0.75, `static_source_z`
    3): reject tracks with ≥ 75% of their used frames within 1 px of a
    static source (the frames' median sky > 3× the single-frame noise).
    Tested offline on the m4h detections of all 30 visits (rules: all /
    ≥ 75% of frames, z > 3 / 5), ratios recalibrated leave-one-visit-out:
    ≥ 75% at z > 3 keeps the held-out false rate at 0.067 per copy and
    raises completeness at W1 15 / 15.5 / 16 from 0.92 / 0.65 / 0.11 to
    0.95 / 0.77 / 0.31 off the plane and 0.61 / 0.15 / 0.00 to 0.67 / 0.24
    / 0.02 in it; "all frames" removed too little to lower thresholds.
    Cost: 2 of 586 injected asteroids (W1 13.6–13.7 in Puppis and Carina,
    where most positions lie on some star); Oceana, Lachesis, (2882) and
    simulated runs unchanged. NULL_RATIO 1.62 / 1.35 / 1.46; fixed
    real-WISE thresholds 10.25 / 9.75 / 10.25 (18 quiet visits).
    Confirmed in the detector (M4l): held out over 30 visits 0.067 false
    per copy with the same six events as offline and the same
    completeness, once the curved-track rule's kept tracks are gated at
    z >= 9 (two of its tracks were extra false events; see the bright
    fast movers entry);
    real_v1 W1 15.5 / 16 33 / 3 -> 40 / 10 of 68 / 69 found, real_v3
    gains as offline, simulated sweeps identical to M4h. Cost: one slow
    W1 14.9 asteroid in a field at b = -23 (7 of 9 frames on faint stars,
    static z 3.3-8.9). With extendedness vetting the calibration is
    NULL_RATIO 1.80 / 1.38 / 1.25, fixed thresholds 9.75 / 9.25 / 9.25
    (confirmed by M4m, below).
  - Crowded-field flag (`crowded_frac` 0.45, `--skip-crowded`): the
    fraction of covered sky within 1 px of a static source (z > 3) is
    <= 0.34 on the 25 visits where injected W1 14.5-15.5 asteroids were
    found reliably and >= 0.54 on the five bulge/plane fields with null
    levels 9.4-14.8 and collapsed completeness. Flagged in the stats and
    printed; `--skip-crowded` stops after registration (12 s instead of
    hours on the bulge). Default for real WISE since M4m: there the five
    crowded visits took 1.5-3.5 h per run for W1 16 completeness of 0.07 in
    the plane at best (`--no-skip-crowded` to search anyway).
  - Suspected unknown asteroids (`review.py`, `neodetect review`):
    detections get a status (known / near-known / suspect / unchecked)
    against `known.json` (or SkyBoT, `--skybot`, any station); per suspect
    a north-up cutout sheet (frames and frames minus the others' median);
    the user's verdicts (confirm / reject + note) go to `review.json`,
    matched by position so they survive reruns; confirmed ones export as
    ADES (trkSub).
  - Extendedness vetting (`extended_max` 2.2, `extended_k` 2): partial
    M4l real_v3 showed the top false tracks in a Cassiopeia field (17.0,
    0.014 deg/day, 10 weak frames) on a diffuse blob several PSF widths
    across beside a bright star -- a WISE ghost or halo, which L1b masks
    don't flag. Per track, the frame-combined flux fitted with a Gaussian
    2x the PSF sigma over the PSF-fit flux (point source 1.6 for a
    Gaussian PSF, ~1.7 measured). Offline on M4h detections of 69 real
    runs: 261 injected/known asteroid tracks <= 2.11 (7.5-10 scores <=
    1.97), 55% of scrambled-copy tracks >= 9 above 2.2 (42% at 7.5-10).
    In the detector: the 17.0 ghost gone (copy's top 13.0), v014 (11
    found incl. the 77.1 NEO), a simulated run and Oceana unchanged.
    Full measurement (150 real runs incl. real_v2's injected): 1012
    asteroid tracks <= 2.11 (99% 1.91; 89 at scores 7.5-10 <= 1.97; 17
    real known objects <= 1.84); 53% of scrambled tracks >= 9 above 2.2.
    Held out over 30 visits on top of static-source vetting (cuts 2.0 /
    2.2 / 2.4): 2.2 best -- false 0.067 -> 0.044 per copy, completeness
    W1 15 / 15.5 / 16 0.95 / 0.77 / 0.31 -> 0.92 / 0.78 / 0.41 off the
    plane and 0.67 / 0.24 / 0.02 -> 0.76 / 0.27 / 0.07 in it (2.0: 0.078
    false). With both: NULL_RATIO 1.80 / 1.38 / 1.25, fixed real-WISE
    thresholds 9.75 / 9.25 / 9.25 (22 quiet visits).
    Confirmed by a full rerun (M4m: static-source + extendedness vetting,
    curved-track gate z >= 9, minimum null level, committed calibration).
    Held out over 30 visits on its own output: 0.067 false per copy (6
    events in 90, offline estimate 4), completeness W1 15 / 15.5 / 16
    0.94 / 0.80 / 0.41 off the plane (M4l 0.94 / 0.76 / 0.31) and 0.76 /
    0.30 / 0.07 in it (M4l 0.67 / 0.24 / 0.02). Per set, each version with
    its own calibration: real_v3 (11 plane fields) no false tracks in 45
    copies (M4l and M4h 4), W1 15 / 16 52 / 13 of 65 / 60 found (M4l 49 /
    7, M4h 44 / 4), 15 more than M4l and none lost; real_v1 2 false in 45
    (0.044), W1 16 24 / 69 (M4l 10); real_v2 2 false in 45, W1 16 28 / 80
    (M4l 22), 3 fewer at W1 14.5-15 (the higher 3-frame threshold);
    wise_v1 / wise_v2 identical to M4h (212 / 209 of 286, 3 false in 72
    nulls). The crowded flag marked exactly the five dense fields.
    Refit on M4m, NULL_RATIO would be 1.66 / 1.37 / 1.25 and the quiet-sky
    thresholds 9.25 / 9.25 / 9.25; over all 135 copies 1.66 gives 7 false
    tracks vs 4 at 1.80 for 5 more of ~1400 asteroids, so the committed
    values stay.
  - Minimum null level (`nullcal.MIN_LEVEL` 6.5): in M4m a quiet 9-frame
    high-latitude visit (v008) kept fewer than 5 candidates above the
    report threshold in a scrambled copy once extendedness vetting
    removed the rest, so its null level fell to that threshold (5.0), its
    thresholds to 6.3-6.9, and five noise tracks at 6.5-8.2 passed. The
    offline estimate couldn't show it: it cut only candidates >= 7.5. 6.5
    is the 5th percentile of single-copy levels before extendedness
    vetting (M4l; 1 copy in 77 below it, 4 in M4m). With it, M4m on 10
    finished real_v3 visits: no false tracks in 30 copies (M4l 4), real_v1
    unchanged (2 in 45). Held out on all of M4m: 6.0 and 7.0 give 5 false
    events in 90 copies, 6.5 gives 6, with the same completeness within
    one asteroid per bin -- noise, so 6.5 stays.
  - W3/W4 (cryogenic WISE, 2010): asteroids glow thermally at 12/22 um,
    so these bands reach far fainter ones than W1. `neowise` fetches them
    (`--band 3/4`, default table `allsky`, 2010-01-07 .. 2010-08-06); known
    objects keep positions only (no thermal magnitude model yet) and no V
    cut. First visit (Oceana field, 2010-05-14, 18 frames, W1-calibrated
    thresholds): W3 13 detections, all catalogued, V 18.9-22.2, every one
    with V <= 20 on >= 3 frames found (4/4), 7/8 to V 21, 11/28 to V 22
    (two, 2014 OJ226 and 2016 EM130, discovered after 2010); W4 4. Noise
    model and null levels (7.07, 6.50) as in W1. A W3/W4 calibration set
    (the 30 W1 calibration positions at their 2010 epoch, mirrored sweeps)
    is running. Also fixed: `review --skybot` searched around the first
    frame (a sliver at the edge here) and kept stale labels on reclassifying.
  - W3/W4 calibration set (30 positions, 2010; sims/real_w3_*, real_w4_*):
    W3 finds half of the catalogued asteroids at V 20 and some to V 22,
    where W1 finds almost none fainter than V 17 (50% completeness off the
    plane: W1 16.1, W3 11.1, W4 7.5 in each band's magnitudes). But the
    held-out false rate was 0.107 (W3), 0.093 (W4) per copy vs 0.080 (W1)
    on the same 25 uncrowded positions, and 53 of W3's 72 strongest
    scrambled-copy candidates used images of catalogued asteroids:
    scrambling destroys their motion but not their images. → catalogued
    asteroids masked in scrambled copies (`nullcal.known_masks`, in
    `detect`'s copy and `neowise scramble`); the W3/W4 null copies are
    being redone with it. Dropping such candidates after the fact lowered
    W3's short-track ratios (2.31 / 2.16 -> 1.74 / 1.46) but left a false
    rate of 0.147: tracks of strong uncatalogued peaks in two dense fields
    (v011/v012), to be identified next.
  - Per-frame tail calibration: tested offline, not adopted. Per frame
    of all 30 visits, the 1e-4 upper quantile of the SNR map on star-free
    sky relative to the visit median: 95th percentile 1.57, 99th 3.0.
    Deflating frames' weights by that ratio (all frames, > 1.2, > 1.5;
    1e-4 or 1e-5 quantile; ratio or its square root), rescored from the
    saved per-frame fluxes (exact for the score) and recalibrated
    leave-one-visit-out on top of static-source vetting: held-out false
    0.056-0.089 per copy vs 0.067, completeness at W1 16 0.24-0.26 vs 0.31
    off the plane. The 6 remaining held-out false events in 90 copies
    aren't on heavy-tailed frames: robust combination and single-frame
    dominance already contain them.
  - Bright fast movers: of injected W1 < 15 movers >= 0.6 deg/day on 3
    frames, 47 of 58 found (on 4-6 frames 94 of 97). Five with truth SNR
    25-75 had no candidate at all: NEOs a few tenths of an AU away whose
    paths curve by several arcsec over three frames (WISE orbital
    parallax); the curved 3-frame fit found them, the straight refit of
    < 4-frame tracks failed consistency and dropped them. →
    `keep_strong_curved`: keep the curved fit when every used frame has
    z >= 7, the curvature is within the linker's bound, and fluxes are
    consistent. Partial M4l rerun: on real_v3 v001 a kept curved track
    (an injected asteroid's frame plus two unrelated peaks, 2.96
    deg/day, 3370"/day^2 -- within the bound at that rate) scored 31.6
    and took the asteroid's frame, losing its 3-frame detection (29.6).
    → kept curved tracks are accepted after all regular tracks, so they
    can't take their frames (fixed on that run; the v014 NEO still found
    at 77.1). M4l (all real sets with injections, pre-fix): 10 kept curved
    tracks matched injected asteroids missed before (weakest frame z
    8.2-43.6), 4 were false tracks in scrambled copies (z 7.1-8.3; one,
    15.4 at 4.6 deg/day, above its copy's threshold) → `keep_curved_min_z`
    9: keeps 7 recoveries, no false tracks. Kept curved tracks come last,
    so the gate can be applied to saved results exactly (M4m ran at 7).
  - W2 vs W1 on the 30 calibration visits: stopped partway (50 of 150 runs)
    for W3/W4. Partial: W2's 50% completeness at W2 14.6 vs W1 16.1 off the
    plane (8 visits).
  - Open from M4: bright fast movers (linker curvature, above);
    the global thresholds rest on 15 ecliptic visits -- recheck the guard
    level on new sky (Galactic plane).
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
  latents) as masks; a template-depth-dependent threshold; real WISE PSF (wings) in both simulator and detector;
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
   registration; ground preset tuning. Status: done. Raster frames (PNG,
   TIFF, JPEG) with `times.csv` / `<image>.json` times and optional
   `sequence.json`; `starmatch.py` gives each frame a linear TAN WCS from
   star matching (phase correlation + RANSAC affine; small field rotations
   only). A simulated ground sequence as 16-bit PNG without WCS gives the
   same 8 detections as the FITS (0.01–0.05 px). Ground preset calibrated
   (`ground_v1`: thresholds 11.75 / 11.5 / 9.75, the instrument default;
   complete to V = 20, 0.77 at 20–20.5). Open: plate solving for rotating
   fields; noise model without uncertainty images ignores a source's own
   photon noise (bright sources' scores run high).
6. **Performance.** numba CPU and CuPy GPU backends for shift-and-stack;
   tiling for full frames. Status: numba done (M2); the CuPy backend is
   written (`backend._stack_max_cupy`, a CUDA RawKernel mirroring the numba
   kernel, launched in chunks of 1024 velocities) and chosen by `auto` only
   after a self-test against the numpy reference; its logic is tested via a
   Python transliteration of the kernel, but it has not run on a GPU yet --
   first job on a GPU machine: `pytest` (the GPU test then runs) and a
   timing of shift-and-stack against numba. Tiling still open.
