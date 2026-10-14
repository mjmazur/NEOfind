"""Detector artifacts: hot pixels, dead columns and cosmic rays.

Hot pixels and bad columns are fixed in *detector* coordinates, so on dithered
or scanned frames they land on different sky positions every frame and look
like transients after sky registration, exactly the false positives a
moving-object search has to reject. Most are flagged in the mask, as a
pipeline's static bad-pixel map would.

Cosmic rays arrive at ``cosmic_rate_cm2_s`` over the physical array area
during each exposure. Each deposits a log-normal amount of charge along a
short straight track (thin IR arrays: 1-3 px; thick CCDs: longer), slightly
spread by charge diffusion. A fraction ``cosmic_flag_frac`` are flagged.
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import ArtifactConfig, InstrumentConfig
from .render import add_point

# Simulator mask bits (msk images). Real-data loaders map their own bits to these.
MASK_HOT = 1
MASK_BADCOL = 2
MASK_SATURATED = 4
MASK_COSMIC = 8

DIFFUSION_SIGMA_PX = 0.35


@dataclass
class DetectorDefects:
    hot_y: np.ndarray
    hot_x: np.ndarray
    hot_dark_e: np.ndarray
    hot_flagged: np.ndarray
    bad_columns: np.ndarray

    def dark_image(self, shape) -> np.ndarray:
        img = np.zeros(shape)
        img[self.hot_y, self.hot_x] = self.hot_dark_e
        return img

    def static_mask(self, shape) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.int16)
        f = self.hot_flagged
        mask[self.hot_y[f], self.hot_x[f]] |= MASK_HOT
        mask[:, self.bad_columns] |= MASK_BADCOL
        return mask


def build_defects(art: ArtifactConfig, inst: InstrumentConfig,
                  np_rng: np.random.Generator) -> DetectorDefects:
    n_pix = inst.width * inst.height
    n_hot = int(np_rng.poisson(art.hot_pixel_frac * n_pix))
    idx = np_rng.choice(n_pix, size=n_hot, replace=False)
    dark = art.hot_median_e * 10 ** np_rng.normal(0.0, 0.4, n_hot)
    return DetectorDefects(
        hot_y=idx // inst.width, hot_x=idx % inst.width, hot_dark_e=dark,
        hot_flagged=np_rng.uniform(size=n_hot) < art.hot_flag_frac,
        bad_columns=np_rng.choice(inst.width, size=art.n_bad_columns, replace=False),
    )


def render_cosmic_rays(art: ArtifactConfig, inst: InstrumentConfig,
                       np_rng: np.random.Generator):
    """Returns (charge image in electrons, charge from flagged hits only, hit records)."""
    shape = (inst.height, inst.width)
    layer = np.zeros(shape)
    flagged_layer = np.zeros(shape)
    area_cm2 = inst.width * inst.height * (inst.pixel_size_um * 1e-4) ** 2
    n = int(np_rng.poisson(art.cosmic_rate_cm2_s * area_cm2 * inst.exposure_s))
    hits = []
    for _ in range(n):
        x0 = np_rng.uniform(-0.5, inst.width - 0.5)
        y0 = np_rng.uniform(-0.5, inst.height - 0.5)
        energy = art.cosmic_median_e * 10 ** np_rng.normal(0.0, art.cosmic_sigma_dex)
        length = np_rng.uniform(0.0, art.cosmic_max_len_px)
        theta = np_rng.uniform(0.0, 2.0 * math.pi)
        flagged = bool(np_rng.uniform() < art.cosmic_flag_frac)
        n_sub = max(1, int(math.ceil(length / 0.5)))
        s = (np.arange(n_sub) + 0.5) / n_sub * length
        for xs, ys in zip(x0 + s * math.cos(theta), y0 + s * math.sin(theta)):
            add_point(layer, xs, ys, energy / n_sub, DIFFUSION_SIGMA_PX)
            if flagged:
                add_point(flagged_layer, xs, ys, energy / n_sub, DIFFUSION_SIGMA_PX)
        hits.append({"x": float(x0), "y": float(y0), "length_px": float(length),
                     "angle_deg": float(math.degrees(theta)), "charge_e": float(energy),
                     "flagged": flagged})
    return layer, flagged_layer, hits
