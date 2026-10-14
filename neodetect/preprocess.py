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


def preprocess(reg: Registered, log=None) -> Preprocessed:
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
    log(f"SNR calibration c_k = {np.round(calib, 3).tolist()}")

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
