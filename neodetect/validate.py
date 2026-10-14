"""Self-checks for the assumption everything else rests on: after
preprocessing, each frame's matched-filter SNR map is N(0, 1) on blank sky,
with Gaussian tails. Run it on an asteroid-free sequence; heavy positive
tails mean artifacts (usually cosmic rays) are getting through.
"""

import numpy as np
from scipy import ndimage
from scipy.stats import norm

from .io import load_sequence
from .preprocess import preprocess
from .register import register


def validate_noise(inputs, region=None, log=print) -> dict:
    seq = load_sequence(inputs)
    reg = register(seq, region=region)
    pre = preprocess(reg)
    log(f"{reg.n} frames on a {reg.shape[1]}x{reg.shape[0]} grid; noise alpha "
        f"{np.round(pre.alpha, 3).tolist()}, beta {pre.beta:.2e}, SNR calibration "
        f"{np.round(pre.calib, 3).tolist()}")
    log(f"{'frame':>5} {'pixels':>9} {'std':>6} {'mean':>7} {'P(z>4)':>9} {'P(z<-4)':>9} "
        f"{'peaks>5':>8}   (Gaussian P(z>4) = {norm.sf(4):.1e})")
    rows = []
    for k in range(pre.n):
        z = pre.snr(k)
        sel = pre.phi[k] > 0
        v = z[sel]
        peaks = int(((z > 5) & (z == ndimage.maximum_filter(z, 5))).sum())
        row = {"frame": k, "count": int(sel.sum()), "std": float(v.std()),
               "mean": float(v.mean()), "p_hi": float((v > 4).mean()),
               "p_lo": float((v < -4).mean()), "peaks_above_5": peaks}
        rows.append(row)
        log(f"{k:5d} {row['count']:9d} {row['std']:6.3f} {row['mean']:+7.3f} "
            f"{row['p_hi']:9.1e} {row['p_lo']:9.1e} {peaks:8d}")
    return {"frames": rows}
