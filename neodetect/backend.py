"""Compute backends for the shift-and-stack kernel.

``stack_max(psi_c, phi, valid, dx, dy, min_frames)`` must return, for every
grid pixel (the source position at t_ref), the best stacked SNR over all
velocity hypotheses and the index of that velocity:

    S(x, y | v) = sum_k psi_c[k, y + dy[v,k], x + dx[v,k]] / sqrt(sum_k phi[k, ...])

requiring at least ``min_frames`` frames with ``valid`` = 1 there. ``psi_c``
and ``phi`` are zero wherever ``valid`` is zero, so the sums need no
branches (which lets the compiler vectorize the inner loop).

Backends: ``cupy`` (NVIDIA GPU: a CUDA kernel, one thread per grid pixel
looping over a chunk of velocities and all frames), ``numba`` (parallel
CPU) and ``numpy`` (slow reference). ``auto`` picks cupy when CuPy and a
GPU are present and its kernel passes a self-test against the numpy
reference on a small random problem (run once per process), else numba,
else numpy.
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


_CUDA_SRC = r"""
extern "C" __global__
void stack_max(const float* psi, const float* phi, const int* dx, const int* dy,
               const int n, const int h, const int w, const int v0, const int nv,
               float* best2, int* bestv)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const long idx = (long)y * w + x;
    float b2 = best2[idx];
    int bv = bestv[idx];
    for (int v = 0; v < nv; v++) {
        float a = 0.0f, d = 0.0f;
        for (int k = 0; k < n; k++) {
            const int xx = x + dx[v * n + k];
            const int yy = y + dy[v * n + k];
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            const long j = ((long)k * h + yy) * w + xx;
            a += psi[j];
            d += phi[j];
        }
        // S > best  <=>  a^2 > best^2 * d for positive S (as the numba kernel).
        if (a > 0.0f && a * a > b2 * d) {
            b2 = a * a / d;
            bv = v0 + v;
        }
    }
    best2[idx] = b2;
    bestv[idx] = bv;
}
"""
CUPY_CHUNK = 1024  # velocities per kernel launch (keeps each launch short)

try:  # pragma: no cover - needs an NVIDIA GPU
    import cupy as _cp
    HAVE_CUPY = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # ImportError, or CUDA without a device/driver
    _cp = None
    HAVE_CUPY = False

_cupy_kernel = None
_cupy_ok = None


def _stack_max_cupy(psi_c, phi, valid, dx, dy, min_frames):  # pragma: no cover - GPU only
    global _cupy_kernel
    cp = _cp
    if _cupy_kernel is None:
        _cupy_kernel = cp.RawKernel(_CUDA_SRC, "stack_max")
    n, h, w = psi_c.shape
    psi_g = cp.asarray(np.ascontiguousarray(psi_c, np.float32))
    phi_g = cp.asarray(np.ascontiguousarray(phi, np.float32))
    best2 = cp.zeros((h, w), cp.float32)
    bestv = cp.full((h, w), -1, cp.int32)
    block = (32, 8, 1)
    grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
    for v0 in range(0, dx.shape[0], CUPY_CHUNK):
        ddx = cp.asarray(np.ascontiguousarray(dx[v0:v0 + CUPY_CHUNK], np.int32))
        ddy = cp.asarray(np.ascontiguousarray(dy[v0:v0 + CUPY_CHUNK], np.int32))
        _cupy_kernel(grid, block, (psi_g, phi_g, ddx, ddy, np.int32(n), np.int32(h), np.int32(w),
                                   np.int32(v0), np.int32(ddx.shape[0]), best2, bestv))
    best_v = cp.asnumpy(bestv)
    best_s = np.where(best_v >= 0, np.sqrt(cp.asnumpy(best2)), -np.inf).astype(np.float32)
    return best_s, best_v


def _self_test(stack_max, seed: int = 0) -> bool:
    """Compare a backend with the numpy reference on a small random problem."""
    rng = np.random.default_rng(seed)
    n, h, w, nv = 6, 40, 48, 30
    valid = (rng.uniform(size=(n, h, w)) > 0.2).astype(np.uint8)
    psi = np.where(valid, rng.normal(0, 1, (n, h, w)), 0).astype(np.float32)
    phi = np.where(valid, rng.uniform(0.5, 1.5, (n, h, w)), 0).astype(np.float32)
    dx = rng.integers(-5, 6, (nv, n)).astype(np.int32)
    dy = rng.integers(-5, 6, (nv, n)).astype(np.int32)
    s0, v0 = _stack_max_numpy(psi, phi, valid, dx, dy, 1)
    s1, v1 = stack_max(psi, phi, valid, dx, dy, 1)
    ok = np.isfinite(s0)
    return bool(np.array_equal(ok, np.isfinite(s1)) and np.allclose(s0[ok], s1[ok], rtol=1e-4, atol=1e-5))


def cupy_ok() -> bool:
    """CuPy and a GPU are present and the kernel agrees with the reference."""
    global _cupy_ok
    if _cupy_ok is None:
        _cupy_ok = HAVE_CUPY and _self_test(_stack_max_cupy)
    return _cupy_ok


def available():
    names = ["numpy"]
    if HAVE_NUMBA:
        names.insert(0, "numba")
    if HAVE_CUPY:
        names.insert(0, "cupy")
    return names


def resolve(name: str = "auto") -> str:
    """The backend ``name`` stands for: ``auto`` is cupy on a GPU that passes
    its self-test, else numba, else numpy."""
    if name == "auto":
        return "cupy" if cupy_ok() else ("numba" if HAVE_NUMBA else "numpy")
    return name


def get_stack_max(name: str = "auto", threads: int = 0):
    name = resolve(name)
    if name == "cupy":
        if not HAVE_CUPY:
            raise ValueError("cupy backend requested but CuPy or an NVIDIA GPU is not available")
        if not cupy_ok():
            raise RuntimeError("cupy backend failed its self-test against the numpy reference")
        return _stack_max_cupy
    if name == "numba":
        if not HAVE_NUMBA:
            raise ValueError("numba backend requested but numba is not installed")
        if threads > 0:
            numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        return _stack_max_numba
    if name == "numpy":
        return _stack_max_numpy
    raise ValueError(f"unknown backend {name!r} (available: {', '.join(available())})")
