"""
Diagnostic plots for IV workflows.

Four plots, each targeted at a specific weak-IV / sensitivity question:

- :func:`plot_first_stage` — first-stage fitted values, instrument strength
  visualised per endogenous regressor.
- :func:`plot_ar_confidence_set` — Anderson-Rubin confidence set by grid
  inversion; stays valid even when instruments are weak.
- :func:`plot_mte_curve` — marginal treatment effect curve with confidence
  band (from :class:`sp.iv.MTEResult`).
- :func:`plot_plausibly_exogenous` — β(γ) curve with UCI envelope for
  exclusion-restriction sensitivity.

Matplotlib is imported lazily so the rest of ``sp.iv`` is useful even in
headless environments without matplotlib.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .mte import MTEResult
from .plausibly_exogenous import PlausiblyExogenousResult


def _mpl():
    import matplotlib.pyplot as plt  # lazy
    return plt


# ═══════════════════════════════════════════════════════════════════════
#  First-stage scatter / fit
# ═══════════════════════════════════════════════════════════════════════

def plot_first_stage(
    endog: Union[np.ndarray, pd.Series],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    endog_name: Optional[str] = None,
    ax=None,
    bins: int = 25,
):
    """
    First-stage fit plot: ŷ_1 = Z π̂ vs observed D, with binscatter.

    A tight diagonal pattern means the instruments have strong first-stage
    predictive power; a horizontal cloud means weak IV.
    """
    plt = _mpl()

    def grab(v, cols=False):
        if isinstance(v, str):
            return data[v].values.astype(float)
        if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
            return data[v].values.astype(float)
        return np.asarray(v, dtype=float)

    D = grab(endog).reshape(-1)
    Z = grab(instruments, cols=True)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(D)
    if exog is None:
        W = np.ones((n, 1))
    else:
        Wx = grab(exog, cols=True)
        if Wx.ndim == 1:
            Wx = Wx.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wx])

    # Partial out controls, then first-stage projection
    def _resid(M, X):
        b, *_ = np.linalg.lstsq(X, M, rcond=None)
        return M - X @ b
    Dt = _resid(D, W)
    Zt = _resid(Z, W)
    pi, *_ = np.linalg.lstsq(Zt, Dt, rcond=None)
    d_hat = Zt @ pi

    # First-stage F
    rss_full = float((Dt - d_hat) ** 2 @ np.ones_like(Dt))
    rss_red = float(Dt @ Dt)
    k = Z.shape[1]
    df_d = n - W.shape[1] - k
    f_stat = ((rss_red - rss_full) / k) / (rss_full / max(df_d, 1)) if rss_full > 0 else np.nan

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    # Binscatter
    order = np.argsort(d_hat)
    d_hat_s = d_hat[order]
    Dt_s = Dt[order]
    edges = np.linspace(d_hat_s.min(), d_hat_s.max(), bins + 1)
    idx = np.clip(np.searchsorted(edges, d_hat_s, side="right") - 1, 0, bins - 1)
    xb = np.array([d_hat_s[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    yb = np.array([Dt_s[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])

    ax.scatter(d_hat, Dt, s=6, alpha=0.15, color="0.6")
    ax.scatter(xb, yb, s=36, color="#d62728", zorder=3, label="binned mean")
    ax.axline((0, 0), slope=1, linestyle="--", color="0.3", lw=1)
    ax.set_xlabel(r"First-stage fitted $\hat{D}$ (partialled out)")
    ax.set_ylabel(fr"Observed $D$ (partialled out){' — ' + endog_name if endog_name else ''}")
    ax.set_title(f"First-stage fit   F = {f_stat:.1f}")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)

    if own_ax:
        fig.tight_layout()
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  Anderson-Rubin confidence set
# ═══════════════════════════════════════════════════════════════════════

def plot_ar_confidence_set(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    beta_grid: Optional[np.ndarray] = None,
    level: float = 0.95,
    ax=None,
):
    """
    Anderson-Rubin (1949) confidence set for β by grid inversion.

    For each candidate β₀ in ``beta_grid``, compute the AR F-statistic and
    invert the test to show the full (1 − α) confidence set. Valid even
    under arbitrarily weak instruments.
    """
    plt = _mpl()
    from scipy import stats as sstats

    def grab(v, cols=False):
        if isinstance(v, str):
            return data[v].values.astype(float)
        if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
            return data[v].values.astype(float)
        return np.asarray(v, dtype=float)

    Y = grab(y).reshape(-1)
    D = grab(endog).reshape(-1)
    Z = grab(instruments, cols=True)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(Y)

    if exog is None:
        W = np.ones((n, 1))
    else:
        Wx = grab(exog, cols=True)
        if Wx.ndim == 1:
            Wx = Wx.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wx])

    def _resid(M, X):
        b, *_ = np.linalg.lstsq(X, M, rcond=None)
        return M - X @ b

    Dt = _resid(D, W)
    Yt = _resid(Y, W)
    Zt = _resid(Z, W)
    k = Zt.shape[1]
    df_d = n - W.shape[1] - k

    # Default β-grid: ±6σ around 2SLS point estimate
    if beta_grid is None:
        b2sls = float((Zt @ (Zt.T @ Dt)) @ Yt / ((Zt @ (Zt.T @ Dt)) @ Dt))
        se_rough = np.std(Yt - b2sls * Dt) / (np.std(Dt) * np.sqrt(n))
        beta_grid = np.linspace(b2sls - 6 * se_rough, b2sls + 6 * se_rough, 201)

    ar_f = np.empty_like(beta_grid)
    for i, b0 in enumerate(beta_grid):
        u = Yt - b0 * Dt
        pi, *_ = np.linalg.lstsq(Zt, u, rcond=None)
        u_hat = Zt @ pi
        rss_full = float((u - u_hat) @ (u - u_hat))
        rss_red = float(u @ u)
        ar_f[i] = ((rss_red - rss_full) / k) / (rss_full / max(df_d, 1)) if rss_full > 0 else np.nan

    crit = sstats.f.ppf(level, k, df_d)
    in_set = ar_f <= crit

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(beta_grid, ar_f, color="#1f77b4", lw=1.8, label="AR F-stat")
    ax.axhline(crit, linestyle="--", color="#d62728",
               label=f"{int(level*100)}% critical F = {crit:.2f}")
    # Shade the confidence set
    if in_set.any():
        ax.fill_between(beta_grid, 0, ar_f, where=in_set,
                        color="#2ca02c", alpha=0.18,
                        label=f"{int(level*100)}% AR set")
        lo, hi = beta_grid[in_set].min(), beta_grid[in_set].max()
        ax.axvline(lo, color="#2ca02c", lw=0.8, linestyle=":")
        ax.axvline(hi, color="#2ca02c", lw=0.8, linestyle=":")
        ax.annotate(
            f"AR CI: [{lo:.3f}, {hi:.3f}]",
            xy=(0.02, 0.92), xycoords="axes fraction",
            fontsize=10, color="#2ca02c",
        )
    ax.set_xlabel(r"candidate $\beta_0$")
    ax.set_ylabel("AR F-statistic")
    ax.set_title("Anderson-Rubin confidence set (weak-IV-robust)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)

    if own_ax:
        fig.tight_layout()
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  MTE curve
# ═══════════════════════════════════════════════════════════════════════

def plot_mte_curve(result: MTEResult, ax=None, show_ci: bool = True, show_ate: bool = True):
    """Plot the marginal treatment effect curve with 95 % CI band."""
    plt = _mpl()
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    c = result.mte_curve
    ax.plot(c["u"], c["mte"], color="#1f77b4", lw=2, label=r"$MTE(u \mid \bar X)$")
    if show_ci and "ci_lower" in c.columns:
        ax.fill_between(c["u"], c["ci_lower"], c["ci_upper"],
                        color="#1f77b4", alpha=0.18, label="95% CI")
    if show_ate:
        ax.axhline(result.ate, linestyle="--", color="#d62728", lw=1.2,
                   label=f"ATE = {result.ate:.3f}")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel(r"unobserved resistance $u$")
    ax.set_ylabel("Marginal Treatment Effect")
    ax.set_title(
        f"MTE — poly degree {result.poly_degree}, N = {result.n_obs}, "
        f"support [{result.propensity_range[0]:.2f}, {result.propensity_range[1]:.2f}]"
    )
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)

    if own_ax:
        fig.tight_layout()
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  Plausibly exogenous sensitivity
# ═══════════════════════════════════════════════════════════════════════

def plot_plausibly_exogenous(
    result: PlausiblyExogenousResult,
    ax=None,
    show_bounds: bool = True,
):
    """
    Plot β̂(γ) across the γ grid with per-γ CI whiskers + union envelope.

    For the UCI method (``result.method`` starts with 'UCI'), each gamma
    in the grid yields a point estimate and standard error; this plot
    shows all of them along with the union-of-CIs envelope.

    For the LTZ method, only a single γ-mean is fit; the plot reduces to a
    simple point-with-whisker visualisation.
    """
    plt = _mpl()
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    grid = result.gamma_grid
    betas = result.beta_at_gamma
    ses = result.se_at_gamma

    # One-d index on the x-axis for readability
    if grid.ndim == 2 and grid.shape[1] == 1:
        xs = grid[:, 0]
        xlab = r"$\gamma$"
    elif grid.ndim == 2 and grid.shape[1] > 1:
        # L2 norm for compact display
        xs = np.linalg.norm(grid, axis=1)
        xlab = r"$\|\gamma\|_2$"
    else:
        xs = np.asarray(grid).ravel()
        xlab = r"$\gamma$"

    order = np.argsort(xs)
    xs, betas_s, ses_s = xs[order], betas[order], ses[order]

    ax.plot(xs, betas_s, color="#1f77b4", lw=1.6, label=r"$\hat\beta(\gamma)$")
    ax.fill_between(xs, betas_s - 1.96 * ses_s, betas_s + 1.96 * ses_s,
                    color="#1f77b4", alpha=0.18, label="per-γ 95% CI")

    if show_bounds:
        ax.axhline(result.ci_lower, color="#d62728", linestyle="--", lw=1.2,
                   label=f"union lower = {result.ci_lower:.3f}")
        ax.axhline(result.ci_upper, color="#d62728", linestyle="--", lw=1.2,
                   label=f"union upper = {result.ci_upper:.3f}")

    ax.axhline(result.beta_hat, color="0.3", lw=0.8, linestyle=":")
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$\hat\beta$")
    ax.set_title(f"Plausibly exogenous sensitivity — {result.method}")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)

    if own_ax:
        fig.tight_layout()
    return ax


__all__ = [
    "plot_first_stage",
    "plot_ar_confidence_set",
    "plot_mte_curve",
    "plot_plausibly_exogenous",
]
