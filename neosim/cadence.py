"""Observation times (mid-exposure MJD) for each frame."""

import numpy as np

from .config import CadenceConfig

MIN_PER_DAY = 1440.0
S_PER_DAY = 86400.0


def build_times(cad: CadenceConfig, np_rng: np.random.Generator) -> np.ndarray:
    if cad.mode == "file":
        if not cad.times_file:
            raise ValueError("cadence mode 'file' needs times_file")
        times = np.atleast_1d(np.loadtxt(cad.times_file, dtype=float))
        return np.sort(times)

    if cad.mode == "wise":
        # One exposure per orbit at the same scan phase; some orbits are lost
        # (South Atlantic Anomaly, Moon avoidance, anneals ...). Keep drawing
        # orbits until n_frames survive.
        offsets_min = []
        k = 0
        while len(offsets_min) < cad.n_frames:
            if np_rng.uniform() >= cad.drop_prob:
                offsets_min.append(k * cad.interval_min)
            k += 1
        offsets_min = np.array(offsets_min)
    elif cad.mode == "uniform":
        offsets_min = np.arange(cad.n_frames) * cad.interval_min
    elif cad.mode == "visits":
        offsets_min = np.array([v * cad.visit_spacing_min + j * cad.intra_visit_min
                                for v in range(cad.n_visits)
                                for j in range(cad.frames_per_visit)], dtype=float)
    else:
        raise ValueError(f"unknown cadence mode {cad.mode!r}")

    jitter_s = np_rng.normal(0.0, cad.time_jitter_s, size=offsets_min.size) \
        if cad.time_jitter_s > 0 else 0.0
    times = cad.start_mjd + offsets_min / MIN_PER_DAY + jitter_s / S_PER_DAY
    return np.sort(times)
