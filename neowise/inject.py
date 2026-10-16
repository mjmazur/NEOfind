"""Injection-recovery and null tests on real frames.

``inject_visit`` copies a downloaded visit, adding synthetic asteroids --
on heliocentric orbits seen from the spacecraft, so their paths curve as
real ones do (``neosim.orbit``) -- to the ``-int`` images, rendered with the visit's own empirical PSF (see
``epsf``) at their predicted positions through each frame's real WCS, with
fluxes from each frame's ``MAGZP`` (total flux; the 4% beyond the ePSF's
8-px radius is left out). Pixels that are NaN in the real data
stay NaN. ``-unc``/``-msk`` are symlinked unchanged: a faint asteroid's
photon noise is negligible against the background. The copy gets a
``truth.json`` in the simulator's format, so ``neodetect evaluate`` scores
it as it scores simulations. Known real asteroids (``known.json``) are
listed too, flagged ``"real": true``: detections of them are not false
alarms, but they don't count toward completeness.

``scramble_visit`` copies a visit with its frame times permuted. Every
artifact (cosmic rays, streaks, latents, star residuals) stays, but nothing
real moves linearly any more, so detections on it calibrate the false-alarm
rate on real data -- there are no asteroid-free real visits.
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits

from neodetect.io import (default_centroid_offset, default_psf_fwhm_arcsec, load_sequence,
                          read_image, world2pix)
from neodetect.psf import FWHM_TO_SIGMA
from neosim.astrometry import tangent_to_equatorial
from neosim.orbit import (KM_S_TO_AU_DAY, OrbitMover, draw_distance, earth_heliocentric_velocity,
                          sky_basis)

from .epsf import EPSF, build_epsf

EPSF_FILE = "epsf.npz"
# Fraction of the total flux within the ePSF's 8-px radius, from the official
# WISE single-exposure PSF (W1, Explanatory Supplement IV.4.c; W2 assumed
# alike). MAGZP calibrates total flux, so an injected source puts this
# fraction of it into the stamp.
EE_WITHIN_STAMP = 0.959


def visit_epsf(visit_dir, seq=None, log=None) -> EPSF:
    """The visit's ePSF, built once and cached in the visit directory."""
    path = Path(visit_dir) / EPSF_FILE
    if path.exists():
        z = np.load(path)
        return EPSF(z["grid"], int(z["oversample"]), int(z["radius"]), int(z["n_stars"]))
    seq = seq or load_sequence(visit_dir)
    f0 = seq.frames[0]
    sigma0 = default_psf_fwhm_arcsec(f0) * FWHM_TO_SIGMA / f0.pixel_scale_arcsec
    psf = build_epsf(seq.frames, sigma0, log=log)
    np.savez(path, grid=psf.grid, oversample=psf.oversample, radius=psf.radius, n_stars=psf.n_stars)
    return psf


def _copy_companions(src_int: Path, out: Path):
    """Symlink the -unc/-msk files next to a copied -int file."""
    for p in src_int.parent.iterdir():
        if p.name.startswith(src_int.name.split("-int")[0]) and ("-unc" in p.name or "-msk" in p.name):
            dest = out / p.name
            if not dest.exists():
                os.symlink(os.path.relpath(p.resolve(), out.resolve()), dest)


def _write_int(src: Path, out: Path, data: np.ndarray, header: fits.Header) -> Path:
    name = src.name
    for ext in (".gz", ".fz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    dest = out / name
    fits.PrimaryHDU(data.astype(np.float32), header).writeto(dest, overwrite=True)
    return dest


def _sky_center(seq):
    """Mean of the frame centers (unit-vector average)."""
    v = np.zeros(3)
    for f in seq.frames:
        ra, dec = f.wcs.all_pix2world([(f.shape[1] - 1) / 2], [(f.shape[0] - 1) / 2], 0)
        r, d = math.radians(ra[0]), math.radians(dec[0])
        v += [math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d)]
    return math.degrees(math.atan2(v[1], v[0])) % 360.0, math.degrees(math.asin(v[2] / np.linalg.norm(v)))


def _observer_offset(frame):
    """Earth-to-spacecraft vector (AU, ICRS) from a WISE header; zeros otherwise."""
    h = frame.header
    return np.array([float(h.get(f"ERTH2SC{c}", 0.0)) for c in "XYZ"])


def make_asteroids(seq, n, mag_range, rate_range, rng, half_size_deg, t_ref, ra0, dec0) -> list:
    """Random movers placed uniformly in a square about the visit center at
    ``t_ref``, on heliocentric two-body orbits (``neosim.orbit``): their paths
    curve as real ones do (no light curves: real data has its own noise).
    Distance follows from the rate (``draw_distance``); the radial velocity is
    the observer's own along the line of sight plus up to 10 km/s."""
    off = np.mean([_observer_offset(f) for f in seq.frames], axis=0)
    v_obs = earth_heliocentric_velocity(t_ref)
    out = []
    for i in range(n):
        rate = math.exp(rng.uniform(math.log(rate_range[0]), math.log(rate_range[1])))
        pa = rng.uniform(0.0, 360.0)
        xi = math.radians(rng.uniform(-half_size_deg, half_size_deg))
        eta = math.radians(rng.uniform(-half_size_deg, half_size_deg))
        ra, dec = (float(v) for v in tangent_to_equatorial(xi, eta, ra0, dec0))
        los = sky_basis(ra, dec)[0]
        vr = -float(v_obs @ los) / KM_S_TO_AU_DAY + rng.uniform(-10.0, 10.0)
        out.append(OrbitMover.from_sky(i, float(rng.uniform(*mag_range)), ra, dec, rate, pa,
                                       draw_distance(rate, rng), vr, t_ref, observer_offset_au=off))
    return out


def inject_visit(visit_dir, out_dir, n: int = 20, mag_range=(13.5, 16.5), rate_range=(0.05, 3.0),
                 seed: Optional[int] = None, half_size_deg: Optional[float] = None, log=None) -> Path:
    log = log or (lambda *a: None)
    visit_dir, out = Path(visit_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seq = load_sequence(visit_dir)
    psf = visit_epsf(visit_dir, seq, log)
    rng = np.random.default_rng(seed)
    ra0, dec0 = _sky_center(seq)
    if half_size_deg is None:
        meta = json.loads((visit_dir / "visit.json").read_text()) if (visit_dir / "visit.json").exists() else {}
        half_size_deg = 0.45 * (meta.get("size_deg") or 0.7)
    t_ref = float(np.mean(seq.times))
    asteroids = make_asteroids(seq, n, mag_range, rate_range, rng, half_size_deg, t_ref, ra0, dec0)

    truth = {"target_ra_deg": ra0, "target_dec_deg": dec0, "t_ref_mjd": t_ref,
             "source_visit": str(visit_dir), "seed": seed,
             "psf": {"kind": "empirical ePSF", "fwhm_px": psf.fwhm_px(), "n_stars": psf.n_stars},
             "pixel_convention": "0-based, pixel centers at integers (astropy origin=0)",
             # Where sources sit relative to their positions: the ePSF's origin
             # is a Gaussian-fit center, real stars sit at catalog positions,
             # so stamps are drawn this far from the truth (as stars are).
             "centroid_offset_px": list(default_centroid_offset(seq.frames[0])),
             "asteroids": []}
    per_ast = {a.id: [] for a in asteroids}
    offset = default_centroid_offset(seq.frames[0])
    paths = {a.id: np.array([np.ravel(a.radec(f.mjd, _observer_offset(f))) for f in seq.frames])
             for a in asteroids}
    for k, f in enumerate(seq.frames):
        src = Path(f.path)
        raw, header = read_image(src)
        raw = raw.astype(np.float64)
        assert raw.shape == f.shape
        h, w = f.shape
        for a in asteroids:
            ra, dec = paths[a.id][k]
            x, y = (float(v) for v in world2pix(f.wcs, ra, dec))
            flux = 10.0 ** (-0.4 * (a.mag - f.magzp))
            ix, iy = int(round(x)), int(round(y))
            in_frame = bool(0 <= ix < w and 0 <= iy < h and np.isfinite(raw[iy, ix]))
            if -psf.radius <= ix < w + psf.radius and -psf.radius <= iy < h + psf.radius:
                st, r0, c0 = psf.stamp(x + offset[0], y + offset[1])
                rs, cs = slice(max(r0, 0), min(r0 + st.shape[0], h)), slice(max(c0, 0), min(c0 + st.shape[1], w))
                raw[rs, cs] += (EE_WITHIN_STAMP * flux
                                * st[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0])
            per_ast[a.id].append({"frame": k, "mjd": f.mjd, "ra_deg": float(ra), "dec_deg": float(dec),
                                  "x": x, "y": y, "in_frame": in_frame, "mag": a.mag,
                                  "flux_dn": flux})
        header["INJECTED"] = (n, "synthetic asteroids added (neowise.inject)")
        _write_int(src, out, raw, header)
        _copy_companions(src, out)

    for a in asteroids:
        frames = per_ast[a.id]
        entry = a.to_dict()
        entry.update({"n_frames_in_frame": sum(p["in_frame"] for p in frames), "frames": frames})
        truth["asteroids"].append(entry)
    known = visit_dir / "known.json"
    if known.exists():
        for o in json.loads(known.read_text())["asteroids"]:
            truth["asteroids"].append({**o, "real": True})
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    for extra in ("visit.json", "known.json", EPSF_FILE):
        if (visit_dir / extra).exists() and not (out / extra).exists():
            os.symlink(os.path.relpath((visit_dir / extra).resolve(), out.resolve()), out / extra)
    log(f"injected {n} asteroids into {seq.n_frames} frames -> {out}")
    return out


def scramble_visit(visit_dir, out_dir, seed: Optional[int] = None, log=None) -> Path:
    """Copy of the visit with frame times randomly permuted (a null test)."""
    log = log or (lambda *a: None)
    visit_dir, out = Path(visit_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seq = load_sequence(visit_dir)
    srcs = [Path(f.path) for f in seq.frames]
    times = seq.times
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(srcs))
    while len(srcs) > 2 and np.all(perm == np.arange(len(srcs))):
        perm = rng.permutation(len(srcs))
    for src, t in zip(srcs, times[perm]):
        data, header = read_image(src)
        header["MJD_OBS"] = (float(t), "SCRAMBLED observation time (null test)")
        for key in ("MJD-MID", "MJD-OBS"):
            header.remove(key, ignore_missing=True)
        header["SCRAMBLD"] = (True, "frame times permuted (neowise.inject)")
        _write_int(src, out, data, header)
        _copy_companions(src, out)
    (out / "truth.json").write_text(json.dumps({"scrambled_from": str(visit_dir), "seed": seed,
                                                "asteroids": []}, indent=1))
    log(f"scrambled {len(srcs)} frame times -> {out}")
    return out
