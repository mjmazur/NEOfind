"""Single-frame point-source detections on the matched-filter SNR maps."""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage

from .preprocess import Preprocessed


@dataclass
class FrameDetections:
    x: np.ndarray  # grid px (sub-pixel)
    y: np.ndarray
    z: np.ndarray  # SNR at the peak

    def __len__(self):
        return self.x.size


def _parabola_offset(m1, c, p1):
    d = m1 - 2.0 * c + p1
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(d < 0, 0.5 * (m1 - p1) / d, 0.0)
    return np.clip(off, -0.5, 0.5)


def find_sources(pre: Preprocessed, threshold: float = 4.0,
                 max_per_frame: int = 3000) -> List[FrameDetections]:
    out = []
    for k in range(pre.n):
        z = pre.snr(k).astype(np.float32)
        peak = (z >= threshold) & (z == ndimage.maximum_filter(z, size=3))
        ys, xs = np.nonzero(peak)
        h, w = z.shape
        keep = (xs > 0) & (xs < w - 1) & (ys > 0) & (ys < h - 1)
        ys, xs = ys[keep], xs[keep]
        zv = z[ys, xs]
        if zv.size > max_per_frame:
            top = np.argsort(-zv)[:max_per_frame]
            ys, xs, zv = ys[top], xs[top], zv[top]
        dx = _parabola_offset(z[ys, xs - 1], zv, z[ys, xs + 1])
        dy = _parabola_offset(z[ys - 1, xs], zv, z[ys + 1, xs])
        out.append(FrameDetections(x=xs + dx, y=ys + dy, z=zv.astype(float)))
    return out
