"""
Additional MR estimators and instrument-strength diagnostics.

Rounds out the MR suite with:

- :func:`mr_mode` — weighted mode estimator (Hartwig et al. 2017).
  Consistent when the plurality of SNPs are valid instruments (the
  "ZEMPA" assumption), more permissive than the median's 50%.
- :func:`mr_f_statistic` — first-stage Cragg-Donald / mean F-statistic
  across SNPs. Flags weak-instrument bias when mean F < 10.
- :func:`mr_funnel_plot` — visualization of SNP-specific Wald ratios
  against their precision, to detect directional pleiotropy.
- :func:`mr_scatter_plot` — classic beta_Y vs beta_X scatter with IVW
  and Egger lines.

References
----------
Hartwig, F.P., Davey Smith, G. & Bowden, J. (2017).
"Robust inference in summary data Mendelian randomization via the
zero modal pleiotropy assumption." *IJE*, 46(6), 1985-1998.

Staiger, D. & Stock, J.H. (1997).
"Instrumental variables regression with weak instruments."
*Econometrica*, 65(3), 557-586.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import stats


__all__ = [
    "ModeBasedResult",
    "FStatisticResult",
    "mr_mode",
    "mr_f_statistic",
    "mr_funnel_plot",
    "mr_scatter_plot",
]


# --------------------------------------------------------------------------- #
#  Mode-based estimator (Hartwig 2017)
# --------------------------------------------------------------------------- #


@dataclass
class ModeBasedResult:
    estimate: float
    se: float
    ci: tuple[float, float]
    p_value: float
    n_snps: int
    bandwidth: float
    method: str = "weighted-mode"

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"MR {self.method} (Hartwig et al. 2017)\n"
            f"  Estimate = {self.estimate:.4f}   SE = {self.se:.4f}   "
            f"95% CI [{lo:.4f}, {hi:.4f}]   p = {self.p_value:.4g}\n"
            f"  Bandwidth (Silverman) = {self.bandwidth:.4f}   "
            f"n SNPs = {self.n_snps}"
        )


def _silverman_bandwidth(x: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Silverman's rule-of-thumb bandwidth for a weighted KDE."""
    n = len(x)
    if n < 2:
        return 1.0
    if weights is None:
        sd = float(np.std(x, ddof=1))
    else:
        w = weights / weights.sum()
        mean = float(np.sum(w * x))
        sd = float(np.sqrt(np.sum(w * (x - mean) ** 2)))
    # IQR fallback
    iqr = float(np.subtract(*np.percentile(x, [75, 25]))) / 1.34
    scale = min(sd, iqr) if iqr > 0 else sd
    if scale == 0:
        scale = 1.0
    return 0.9 * scale * n ** (-1 / 5)


def _weighted_mode(
    ratios: np.ndarray, weights: np.ndarray, bandwidth: float
) -> float:
    """Gaussian-kernel weighted mode of the Wald-ratio distribution."""
    grid = np.linspace(ratios.min(), ratios.max(), 1024)
    if bandwidth <= 0:
        return float(ratios[np.argmax(weights)])
    diffs = (grid[:, None] - ratios[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2) / (bandwidth * np.sqrt(2 * np.pi))
    density = (kernel * weights[None, :]).sum(axis=1)
    return float(grid[np.argmax(density)])


def mr_mode(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    method: str = "weighted",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> ModeBasedResult:
    """Mode-based MR estimator (Hartwig, Davey Smith & Bowden 2017).

    Parameters
    ----------
    method : {"weighted", "simple"}, default "weighted"
        ``weighted`` uses IVW weights when finding the mode; ``simple``
        uses unit weights (robust but less efficient).
    n_boot : int, default 1000
        Bootstrap replicates for SE.
    """
    if method not in ("weighted", "simple"):
        raise ValueError("method must be 'weighted' or 'simple'")
    rng = np.random.default_rng(seed)
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)

    ratios = by / bx
    ratio_se = sy / np.abs(bx)
    if method == "weighted":
        w = 1.0 / ratio_se ** 2
    else:
        w = np.ones_like(ratios)
    w = w / w.sum()

    bandwidth = _silverman_bandwidth(ratios, w)
    estimate = _weighted_mode(ratios, w, bandwidth)

    # Parametric bootstrap SE
    boot = np.empty(n_boot)
    for b in range(n_boot):
        bx_b = bx + rng.normal(0, sx)
        by_b = by + rng.normal(0, sy)
        r_b = by_b / bx_b
        rs_b = sy / np.abs(bx_b)
        if method == "weighted":
            w_b = 1.0 / rs_b ** 2
        else:
            w_b = np.ones_like(r_b)
        w_b = w_b / w_b.sum()
        h_b = _silverman_bandwidth(r_b, w_b)
        boot[b] = _weighted_mode(r_b, w_b, h_b)

    se = float(np.std(boot, ddof=1))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (estimate - z_crit * se, estimate + z_crit * se)
    z = estimate / se if se > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    return ModeBasedResult(
        estimate=float(estimate),
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        p_value=p,
        n_snps=len(bx),
        bandwidth=bandwidth,
        method=f"{method}-mode",
    )


# --------------------------------------------------------------------------- #
#  F-statistic: instrument strength
# --------------------------------------------------------------------------- #


@dataclass
class FStatisticResult:
    f_mean: float
    f_min: float
    f_max: float
    weak_instrument_risk: bool
    r2_mean: float
    per_snp_F: np.ndarray

    def summary(self) -> str:
        flag = ("⚠ WEAK INSTRUMENT RISK" if self.weak_instrument_risk
                else "OK — instruments strong")
        return (
            "MR Instrument-Strength Diagnostic\n"
            f"  Mean per-SNP F       = {self.f_mean:.2f}\n"
            f"  Min  per-SNP F       = {self.f_min:.2f}\n"
            f"  Max  per-SNP F       = {self.f_max:.2f}\n"
            f"  Mean per-SNP R^2     = {self.r2_mean:.4f}\n"
            f"  Staiger-Stock rule : F >= 10 required for each SNP; {flag}"
        )


def mr_f_statistic(
    beta_exposure: np.ndarray,
    se_exposure: np.ndarray,
    *,
    n_samples: Optional[int] = None,
) -> FStatisticResult:
    """Per-SNP F-statistic as an instrument-strength summary.

    Uses the standard summary-stat approximation
    ``F_i = (beta_i / se_i)^2``, which is valid for large GWAS where
    the first-stage regression is asymptotically Normal. Flags weak-IV
    risk when any SNP's F falls below 10 (Staiger-Stock 1997).

    Parameters
    ----------
    n_samples : int, optional
        GWAS sample size. If provided, R^2 is reported via
        R^2 = F / (F + n - 2); otherwise a small-sample approximation
        ``R^2 ≈ F / (F + n_snps - 2)`` is used.
    """
    bx = np.asarray(beta_exposure, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    f_per = (bx / sx) ** 2
    n = n_samples if n_samples is not None else len(bx)
    if n > 2:
        r2_per = f_per / (f_per + n - 2)
    else:
        r2_per = f_per * 0.0
    return FStatisticResult(
        f_mean=float(np.mean(f_per)),
        f_min=float(np.min(f_per)),
        f_max=float(np.max(f_per)),
        weak_instrument_risk=bool(np.min(f_per) < 10.0),
        r2_mean=float(np.mean(r2_per)),
        per_snp_F=f_per,
    )


# --------------------------------------------------------------------------- #
#  Funnel + scatter plots
# --------------------------------------------------------------------------- #


def mr_funnel_plot(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    *,
    snp_ids: Optional[List[str]] = None,
    ax=None,
):
    """Funnel plot of SNP-specific Wald ratios vs. precision.

    An asymmetric funnel around the IVW estimate suggests directional
    horizontal pleiotropy. Returns the matplotlib axis.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for plotting") from exc

    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    ratio = by / bx
    precision = np.abs(bx) / sy

    w = 1.0 / (sy / np.abs(bx)) ** 2
    ivw_est = float(np.sum(w * ratio) / np.sum(w))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ratio, precision, s=40, alpha=0.7, edgecolors="black")
    ax.axvline(ivw_est, color="red", ls="--", lw=1.5,
               label=f"IVW = {ivw_est:.3f}")
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    if snp_ids is not None:
        for xi, yi, name in zip(ratio, precision, snp_ids):
            ax.annotate(name, (xi, yi), fontsize=8, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("SNP-specific Wald ratio (β_Y / β_X)")
    ax.set_ylabel("Precision  |β_X| / SE(β_Y)")
    ax.set_title("MR Funnel Plot (asymmetry → directional pleiotropy)")
    ax.legend()
    return ax


def mr_scatter_plot(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    ax=None,
):
    """Classic MR scatter plot with IVW and MR-Egger lines."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for plotting") from exc

    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)

    w = 1.0 / sy ** 2
    ivw = float(np.sum(w * bx * by) / np.sum(w * bx ** 2))

    X = np.column_stack([np.ones(len(bx)), bx])
    W = np.diag(w)
    try:
        coef = np.linalg.solve(X.T @ W @ X, X.T @ W @ by)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ by
    egger_intercept, egger_slope = float(coef[0]), float(coef[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(bx, by, xerr=sx, yerr=sy,
                fmt="o", capsize=2, alpha=0.7, color="tab:blue")
    xs = np.linspace(min(0, float(bx.min())), float(bx.max()), 50)
    ax.plot(xs, ivw * xs, "r--", lw=1.5,
            label=f"IVW slope = {ivw:.3f}")
    ax.plot(xs, egger_intercept + egger_slope * xs,
            "g-.", lw=1.5,
            label=f"MR-Egger: {egger_intercept:+.3f} + {egger_slope:.3f}β_X")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("SNP effect on exposure (β_X)")
    ax.set_ylabel("SNP effect on outcome (β_Y)")
    ax.set_title("MR Scatter Plot with IVW and MR-Egger Lines")
    ax.legend()
    return ax
