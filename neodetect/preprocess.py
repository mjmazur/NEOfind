"""Static-sky subtraction, noise model and matched-filter likelihood maps.

**Template.** Everything static (stars, galaxies, sky residuals) is removed by
subtracting, from each frame k, a *leave-one-out* inverse-variance weighted
mean of the other frames. Pixels deviating strongly from the per-pixel median
(cosmic rays that escaped masking, bright asteroids) are excluded first.
Leaving frame k out means an asteroid never subtracts itself; its leakage
into other frames' templates lands where it was at *those* times, not on its
position in frame k. The template's own noise is independent of frame k's, so
it simply adds ``1 / sum_{j != k} w_j`` to the residual variance.

**Noise model.** residual variance = ``alpha_k * (var_k + var_template_k) + beta * T^2``.
``alpha_k`` rescales the resampled uncertainty per frame (interpolation
smoothing, imperfect unc maps); ``beta`` absorbs star residuals (PSF and
registration mismatch) in proportion to the template brightness T.

**Likelihood maps.** For PSF P_k: ``psi_k = P_k (x) (w_k r_k)``,
``phi_k = P_k^2 (x) w_k`` (correlations). At any position, ``psi/phi`` is
the PSF-fit flux and ``psi/sqrt(phi)`` its SNR. Because resampled noise is
spatially correlated, the SNR map's empirical scatter on blank sky (``c_k``)
is used to rescale ``w_k`` so the SNR is N(0, 1) by construction; fluxes are
unaffected.

**Local noise.** The noise model is global per frame, but real frames have
patches where residuals are far noisier than it says: scattered light and
spikes from bright stars just outside the field (a fan of streaks along a
cutout edge in some frames of the Oceana visit, where scrambled copies
found their best false tracks), and in dense Galactic fields the frame
edges (SNR scatter 1.3-1.5 within 10 px of the edge on the bulge visit,
1.0 elsewhere). Per frame, the local SNR scatter ``s`` is estimated in
``local_noise_px`` windows from the mean of ``min(|z|, 2.5)``, leaving out
the surroundings of strong sources (a bright asteroid's negative imprints
in the templates of the other frames would otherwise deflate its own
weights: Oceana lost 23% of its score), and where ``s``
exceeds ``LOCAL_NOISE_MIN`` the weights are divided by ``s^2``, so the SNR
there is again ~N(0, 1). Clean frames are left alone (the estimate's own
scatter is ~8%).
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d

from .io import robust_sigma
from .psf import centered_kernel, separable_correlate
from .register import Registered

MIN_TEMPLATE_FRAMES = 2  # other frames required to form a pixel's template
OUTLIER_NSIG = 4.0
OUTLIER_FRAC = 0.1  # extra fractional tolerance on bright pixels when clipping
MIN_PSF_COVERAGE = 0.6  # fraction of PSF weight on valid pixels to use a position
LOCAL_NOISE_MIN = 1.2  # local SNR scatter above which weights are deflated
LOCAL_NOISE_CLIP = 2.5  # |z| clipped here in the local scatter estimate
LOCAL_NOISE_SOURCE = 5.0  # compact sources above this |z| are left out of it
LOCAL_NOISE_COMPACT_PX = 100  # largest such source (a PSF core at SNR ~1000 is ~60 px)


@dataclass
class Preprocessed:
    resid: np.ndarray  # (n, h, w) float32
    weight: np.ndarray  # (n, h, w) float32, calibrated inverse variance (0 = unusable)
    psi: np.ndarray  # (n, h, w) float32
    phi: np.ndarray  # (n, h, w) float32 (0 where PSF coverage is insufficient)
    template: np.ndarray  # (h, w) all-frame weighted mean (static sky)
    coverage: np.ndarray  # (h, w) number of frames with valid data
    alpha: np.ndarray
    beta: float
    calib: np.ndarray  # c_k
    psf_sigma_px: np.ndarray
    times: np.ndarray

    @property
    def n(self):
        return self.resid.shape[0]

    def snr(self, k: int) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.phi[k] > 0, self.psi[k] / np.sqrt(self.phi[k]), 0.0)


def _template(reg: Registered):
    n = reg.n
    with np.errstate(divide="ignore"):
        w0 = np.where(reg.valid & (reg.var > 0), 1.0 / np.maximum(reg.var, 1e-30), 0.0
                      ).astype(np.float32)
    masked = np.where(reg.valid, reg.data, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(masked, axis=0)
    del masked
    med = np.nan_to_num(med)
    use = reg.valid.copy()
    for k in range(n):
        tol = OUTLIER_NSIG * np.sqrt(reg.var[k] + (OUTLIER_FRAC * med) ** 2)
        use[k] &= np.abs(reg.data[k] - med) <= tol
    uw = np.where(use, w0, 0.0).astype(np.float32)
    W = uw.sum(axis=0)
    S = (uw * reg.data).sum(axis=0)
    cnt = use.sum(axis=0)
    return uw, W, S, cnt


def _mean_abs_clipped(s, c=LOCAL_NOISE_CLIP):
    """E[min(|z|, c)] for z ~ N(0, s^2)."""
    from scipy.special import erf
    u = c / np.asarray(s, float)
    return s * (math.sqrt(2.0 / math.pi) * (1.0 - np.exp(-0.5 * u * u))
                + u * (1.0 - erf(u / math.sqrt(2.0))))


_S_GRID = np.linspace(0.3, 6.0, 400)
_M_GRID = _mean_abs_clipped(_S_GRID)


def local_sigma(z: np.ndarray, usable: np.ndarray, size: int) -> np.ndarray:
    """Local scatter of an SNR map in ``size`` windows, from the mean of
    |z| clipped at ``LOCAL_NOISE_CLIP`` (inverted for the clipping), leaving
    out pixels within 3 px of a compact strong source (|z| >
    ``LOCAL_NOISE_SOURCE`` over at most ``LOCAL_NOISE_COMPACT_PX`` px): a
    bright asteroid or its imprints in the other frames' templates must not
    deflate its own weights, while extended structure (scattered-light
    streaks) must count. 1 where under a third of a window counts."""
    from scipy.ndimage import label, maximum_filter, uniform_filter
    lab, n = label(np.abs(z) > LOCAL_NOISE_SOURCE)
    strong = np.zeros(z.shape, bool)
    if n:
        compact = np.bincount(lab.ravel()) <= LOCAL_NOISE_COMPACT_PX
        compact[0] = False
        strong = maximum_filter(compact[lab], size=7)
    use = usable & ~strong
    a = np.where(use, np.minimum(np.abs(z), LOCAL_NOISE_CLIP), 0.0).astype(np.float32)
    num = uniform_filter(a, size, mode="constant")
    den = uniform_filter(use.astype(np.float32), size, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(den > 1.0 / 3.0, num / den, _M_GRID[np.searchsorted(_S_GRID, 1.0)])
    return np.interp(m, _M_GRID, _S_GRID).astype(np.float32)


def preprocess(reg: Registered, log=None, local_noise_px: int = 31) -> Preprocessed:
    log = log or (lambda *a: None)
    n = reg.n
    uw, W, S, cnt = _template(reg)
    with np.errstate(divide="ignore", invalid="ignore"):
        template_all = np.where(W > 0, S / W, 0.0).astype(np.float32)

    resid = np.zeros_like(reg.data)
    base_var = np.zeros_like(reg.data)
    ok = np.zeros(reg.data.shape, bool)
    tmpl_k = np.zeros_like(reg.data)
    for k in range(n):
        Wk = W - uw[k]
        cnt_k = cnt - (uw[k] > 0)
        good = reg.valid[k] & (Wk > 0) & (cnt_k >= MIN_TEMPLATE_FRAMES)
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(good, (S - uw[k] * reg.data[k]) / Wk, 0.0)
            vt = np.where(good, 1.0 / Wk, 0.0)
        resid[k] = np.where(good, reg.data[k] - T, 0.0)
        base_var[k] = np.where(good, reg.var[k] + vt, 0.0)
        tmpl_k[k] = T
        ok[k] = good
    del uw

    # --- noise model ---------------------------------------------------------
    alpha = np.ones(n)
    for k in range(n):
        sel = ok[k] & (np.abs(tmpl_k[k]) < 2.0 * np.sqrt(np.maximum(base_var[k], 1e-30)))
        if sel.sum() > 1000:
            z = resid[k][sel] / np.sqrt(base_var[k][sel])
            alpha[k] = robust_sigma(z[:: max(1, z.size // 500000)]) ** 2
    beta = _fit_star_term(resid, base_var, tmpl_k, ok, alpha)
    log(f"noise model: alpha = {np.round(alpha, 3).tolist()}, beta = {beta:.2e}")

    weight = np.zeros_like(reg.data)
    for k in range(n):
        v = alpha[k] * base_var[k] + beta * tmpl_k[k] ** 2
        weight[k] = np.where(ok[k] & (v > 0), 1.0 / np.maximum(v, 1e-30), 0.0)
        resid[k][~ok[k]] = 0.0
    del base_var, tmpl_k

    # --- likelihood maps + calibration ---------------------------------------
    psi = np.zeros_like(reg.data)
    phi = np.zeros_like(reg.data)
    calib = np.ones(n)
    n_noisy = []
    star_free = np.abs(template_all) < 3.0 * np.sqrt(np.median(1.0 / np.maximum(
        weight[weight > 0][:: 97], 1e-30)))
    for k in range(n):
        kern = centered_kernel(reg.psf_sigma_px[k])
        p = separable_correlate(weight[k] * resid[k], kern)
        f = separable_correlate(weight[k], kern ** 2)
        cov = separable_correlate(ok[k].astype(np.float32), kern ** 2) / float((kern ** 2).sum()) ** 2
        usable = (cov >= MIN_PSF_COVERAGE) & (f > 0)
        p = np.where(usable, p, 0.0)
        f = np.where(usable, f, 0.0)
        sel = usable & star_free
        if sel.sum() > 1000:
            z = p[sel] / np.sqrt(f[sel])
            calib[k] = robust_sigma(z[:: max(1, z.size // 500000)])
        c2 = calib[k] ** 2
        weight[k] /= c2
        psi[k] = p / c2
        phi[k] = f / c2
        if local_noise_px:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(usable, psi[k] / np.sqrt(np.where(usable, phi[k], 1.0)), 0.0)
            s = local_sigma(z, usable, local_noise_px)
            noisy = s > LOCAL_NOISE_MIN
            n_noisy.append(round(float(np.mean(noisy[usable])) if usable.any() else 0.0, 4))
            if noisy.any():
                weight[k] /= np.where(noisy, s * s, 1.0)
                psi[k] = np.where(usable, separable_correlate(weight[k] * resid[k], kern), 0.0)
                phi[k] = np.where(usable, separable_correlate(weight[k], kern ** 2), 0.0)
    log(f"SNR calibration c_k = {np.round(calib, 3).tolist()}")
    if local_noise_px:
        log(f"local noise: fraction of each frame deflated = {n_noisy}")

    return Preprocessed(resid=resid, weight=weight.astype(np.float32),
                        psi=psi.astype(np.float32), phi=phi.astype(np.float32),
                        template=template_all, coverage=reg.valid.sum(axis=0),
                        alpha=alpha, beta=beta, calib=calib,
                        psf_sigma_px=reg.psf_sigma_px, times=reg.times)


def _fit_star_term(resid, base_var, tmpl, ok, alpha, n_bins: int = 20) -> float:
    """Fit beta in var_z = 1 + beta * T^2 / (alpha * base_var) from robust
    per-bin variances, binning pixels by template SNR."""
    xs, ys, ws = [], [], []
    n = resid.shape[0]
    frames = np.linspace(0, n - 1, min(n, 6)).astype(int)
    z_all, x_all = [], []
    for k in frames:
        sel = ok[k]
        bv = alpha[k] * base_var[k][sel]
        z_all.append(resid[k][sel] / np.sqrt(bv))
        x_all.append(tmpl[k][sel] ** 2 / bv)
    z_all, x_all = np.concatenate(z_all), np.concatenate(x_all)
    bright = x_all > 4.0  # template > 2 sigma
    if bright.sum() < 200:
        return 0.0
    edges = np.quantile(x_all[bright], np.linspace(0, 1, n_bins + 1))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = bright & (x_all >= lo) & (x_all < hi)
        if sel.sum() < 50:
            continue
        v = robust_sigma(z_all[sel]) ** 2
        xs.append(float(np.median(x_all[sel])))
        ys.append(v - 1.0)
        ws.append(math.sqrt(sel.sum()) / max(v, 1e-6))
    if not xs:
        return 0.0
    xs, ys, ws = map(np.asarray, (xs, ys, ws))
    beta = float(np.sum(ws ** 2 * xs * ys) / np.sum(ws ** 2 * xs * xs))
    return max(beta, 0.0)
