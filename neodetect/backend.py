"""Compute backends for the shift-and-stack kernel.

``stack_max(psi_c, phi, valid, dx, dy, min_frames)`` must return, for every
grid pixel (the source position at t_ref), the best stacked SNR over all
velocity hypotheses and the index of that velocity:

    S(x, y | v) = sum_k psi_c[k, y + dy[v,k], x + dx[v,k]] / sqrt(sum_k phi[k, ...])

requiring at least ``min_frames`` frames with ``valid`` = 1 there. ``psi_c``
and ``phi`` are zero wherever ``valid`` is zero, so the sums need no
branches (which lets the compiler vectorize the inner loop).

Backends: ``numba`` (parallel CPU, default when installed) and ``numpy``
(slow reference). A ``cupy`` backend for NVIDIA GPUs belongs here too
(milestone 6); it only has to implement this one function.
"""

import math
import os

import numpy as np

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def _stack_max_numpy(psi_c, phi, valid, dx, dy, min_frames):
    n, h, w = psi_c.shape
    best_s = np.full((h, w), -np.inf, np.float32)
    best_v = np.full((h, w), -1, np.int32)
    for v in range(dx.shape[0]):
        num = np.zeros((h, w), np.float32)
        den = np.zeros((h, w), np.float32)
        cnt = np.zeros((h, w), np.int16)
        for k in range(n):
            ox, oy = int(dx[v, k]), int(dy[v, k])
            ys, yd = (slice(oy, h), slice(0, h - oy)) if oy >= 0 else (slice(0, h + oy), slice(-oy, h))
            xs, xd = (slice(ox, w), slice(0, w - ox)) if ox >= 0 else (slice(0, w + ox), slice(-ox, w))
            if ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            num[yd, xd] += psi_c[k, ys, xs]
            den[yd, xd] += phi[k, ys, xs]
            cnt[yd, xd] += valid[k, ys, xs].astype(np.int16)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where((den > 0) & (cnt >= min_frames), num / np.sqrt(den), -np.inf)
        better = s > best_s
        best_s[better] = s[better]
        best_v[better] = v
    return best_s, best_v


ROW_BLOCK = 8  # rows per work item: keep a thread's input rows (all frames) within L2


if HAVE_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _stack_max_kernel(psi_c, phi, x_first, x_end, dx, dy):
        # Cache-blocked: each work item is a block of output rows, looping over
        # velocities innermost-but-one. Velocities arrive ordered by vy, so
        # consecutive velocities read the same input rows (only the x shift
        # differs) and those rows stay in cache instead of being re-streamed
        # from memory per velocity -- the kernel is otherwise bandwidth-bound.
        # Each frame covers only part of the grid (WISE: ~1/3), so the
        # accumulation for frame k on input row yy only spans that row's covered
        # columns [x_first, x_end). The frame count is not tracked: with
        # per-frame clipping at zclip, m frames reach at most zclip*sqrt(m), so
        # the caller's threshold enforces it (see shiftstack).
        n, h, w = psi_c.shape
        nv = dx.shape[0]
        best_s = np.full((h, w), -np.inf, np.float32)
        best_v = np.full((h, w), -1, np.int32)
        nb = (h + ROW_BLOCK - 1) // ROW_BLOCK
        for b in prange(nb):
            y0 = b * ROW_BLOCK
            rows = min(h, y0 + ROW_BLOCK) - y0
            num = np.empty((rows, w), np.float32)
            den = np.empty((rows, w), np.float32)
            best2 = np.zeros((rows, w), np.float32)
            for v in range(nv):
                num[:, :] = 0.0
                den[:, :] = 0.0
                for k in range(n):
                    ox = dx[v, k]
                    oy = dy[v, k]
                    for yi in range(rows):
                        yy = y0 + yi + oy
                        if yy < 0 or yy >= h:
                            continue
                        # Output x reads input x + ox, which must lie in the covered span.
                        x_lo = max(0, x_first[k, yy] - ox)
                        x_hi = min(w, x_end[k, yy] - ox)
                        if x_lo >= x_hi:
                            continue
                        p = psi_c[k, yy]
                        f = phi[k, yy]
                        nrow = num[yi]
                        drow = den[yi]
                        for x in range(x_lo, x_hi):
                            nrow[x] += p[x + ox]
                            drow[x] += f[x + ox]
                for yi in range(rows):
                    for x in range(w):
                        a = num[yi, x]
                        # S > best  <=>  a^2 > best^2 * den for positive S; avoids sqrt.
                        if a > 0.0 and a * a > best2[yi, x] * den[yi, x]:
                            best2[yi, x] = a * a / den[yi, x]
                            best_v[y0 + yi, x] = v
            for yi in range(rows):
                for x in range(w):
                    if best_v[y0 + yi, x] >= 0:
                        best_s[y0 + yi, x] = math.sqrt(best2[yi, x])
        return best_s, best_v

    def _stack_max_numba(psi_c, phi, valid, dx, dy, min_frames):
        n, h, w = valid.shape
        any_x = valid > 0
        has = any_x.any(axis=2)
        x_first = np.where(has, any_x.argmax(axis=2), w).astype(np.int32)
        x_end = np.where(has, w - any_x[:, :, ::-1].argmax(axis=2), 0).astype(np.int32)
        return _stack_max_kernel(psi_c, phi, x_first, x_end, dx, dy)


def available():
    names = ["numpy"]
    if HAVE_NUMBA:
        names.insert(0, "numba")
    return names


def get_stack_max(name: str = "auto", threads: int = 0):
    if name == "auto":
        name = "numba" if HAVE_NUMBA else "numpy"
    if name == "numba":
        if not HAVE_NUMBA:
            raise ValueError("numba backend requested but numba is not installed")
        if threads > 0:
            numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        return _stack_max_numba
    if name == "numpy":
        return _stack_max_numpy
    raise ValueError(f"unknown backend {name!r} (available: {', '.join(available())})")
