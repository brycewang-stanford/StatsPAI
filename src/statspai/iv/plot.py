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

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .mte import MTEResult
from .plausibly_exogenous import PlausiblyExogenousResult


def _mpl() -> Any:
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
    ax: Any = None,
    bins: int = 25,
) -> Any:
    """
    First-stage fit plot: ŷ_1 = Z π̂ vs observed D, with binscatter.

    A tight diagonal pattern means the instruments have strong first-stage
    predictive power; a horizontal cloud means weak IV.
    """
    plt = _mpl()

    def grab(v: Any, cols: bool = False) -> np.ndarray:
        if isinstance(v, str):
            if data is None:
                raise ValueError("data must be provided when using column names")
            return np.asarray(data[v].values, dtype=float)
        if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
            if data is None:
                raise ValueError("data must be provided when using column names")
            return np.asarray(data[v].values, dtype=float)
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
    def _resid(M: np.ndarray, X: np.ndarray) -> np.ndarray:
        b, *_ = np.linalg.lstsq(X, M, rcond=None)
        return np.asarray(M - X @ b, dtype=float)

    Dt = _resid(D, W)
    Zt = _resid(Z, W)
    pi, *_ = np.linalg.lstsq(Zt, Dt, rcond=None)
    d_hat = Zt @ pi

    # First-stage F
    rss_full = float((Dt - d_hat) ** 2 @ np.ones_like(Dt))
    rss_red = float(Dt @ Dt)
    k = Z.shape[1]
    df_d = n - W.shape[1] - k
    f_stat = (
        ((rss_red - rss_full) / k) / (rss_full / max(df_d, 1))
        if rss_full > 0
        else np.nan
    )

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(5.5, 5))

    # Binscatter
    order = np.argsort(d_hat)
    d_hat_s = d_hat[order]
    Dt_s = Dt[order]
    edges = np.linspace(d_hat_s.min(), d_hat_s.max(), bins + 1)
    idx = np.clip(
        np.searchsorted(edges, d_hat_s, side="right") - 1,
        0,
        bins - 1,
    )
    xb = np.array(
        [d_hat_s[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
    )
    yb = np.array(
        [Dt_s[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
    )

    ax.scatter(d_hat, Dt, s=6, alpha=0.15, color="0.6")
    ax.scatter(xb, yb, s=36, color="#d62728", zorder=3, label="binned mean")
    ax.axline((0, 0), slope=1, linestyle="--", color="0.3", lw=1)
    ax.set_xlabel(r"First-stage fitted $\hat{D}$ (partialled out)")
    suffix = f" — {endog_name}" if endog_name else ""
    ax.set_ylabel(rf"Observed $D$ (partialled out){suffix}")
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
    ax: Any = None,
) -> Any:
    """
    Anderson-Rubin (1949) confidence set for β by grid inversion.

    For each candidate β₀ in ``beta_grid``, compute the AR F-statistic and
    invert the test to show the full (1 − α) confidence set. Valid even
    under arbitrarily weak instruments.
    """
    plt = _mpl()
    from scipy import stats as sstats

    def grab(v: Any, cols: bool = False) -> np.ndarray:
        if isinstance(v, str):
            if data is None:
                raise ValueError("data must be provided when using column names")
            return np.asarray(data[v].values, dtype=float)
        if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
            if data is None:
                raise ValueError("data must be provided when using column names")
            return np.asarray(data[v].values, dtype=float)
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

    def _resid(M: np.ndarray, X: np.ndarray) -> np.ndarray:
        b, *_ = np.linalg.lstsq(X, M, rcond=None)
        return np.asarray(M - X @ b, dtype=float)

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
        ar_f[i] = (
            ((rss_red - rss_full) / k) / (rss_full / max(df_d, 1))
            if rss_full > 0
            else np.nan
        )

    crit = sstats.f.ppf(level, k, df_d)
    in_set = ar_f <= crit

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(beta_grid, ar_f, color="#1f77b4", lw=1.8, label="AR F-stat")
    ax.axhline(
        crit,
        linestyle="--",
        color="#d62728",
        label=f"{int(level*100)}% critical F = {crit:.2f}",
    )
    # Shade the confidence set
    if in_set.any():
        ax.fill_between(
            beta_grid,
            0,
            ar_f,
            where=in_set,
            color="#2ca02c",
            alpha=0.18,
            label=f"{int(level*100)}% AR set",
        )
        lo, hi = beta_grid[in_set].min(), beta_grid[in_set].max()
        ax.axvline(lo, color="#2ca02c", lw=0.8, linestyle=":")
        ax.axvline(hi, color="#2ca02c", lw=0.8, linestyle=":")
        ax.annotate(
            f"AR CI: [{lo:.3f}, {hi:.3f}]",
            xy=(0.02, 0.92),
            xycoords="axes fraction",
            fontsize=10,
            color="#2ca02c",
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


def plot_mte_curve(
    result: MTEResult,
    ax: Any = None,
    show_ci: bool = True,
    show_ate: bool = True,
) -> Any:
    """Plot the marginal treatment effect curve with 95 % CI band."""
    plt = _mpl()
    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    c = result.mte_curve
    ax.plot(c["u"], c["mte"], color="#1f77b4", lw=2, label=r"$MTE(u \mid \bar X)$")
    if show_ci and "ci_lower" in c.columns:
        ax.fill_between(
            c["u"],
            c["ci_lower"],
            c["ci_upper"],
            color="#1f77b4",
            alpha=0.18,
            label="95% CI",
        )
    if show_ate:
        ax.axhline(
            result.ate,
            linestyle="--",
            color="#d62728",
            lw=1.2,
            label=f"ATE = {result.ate:.3f}",
        )
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
    ax: Any = None,
    show_bounds: bool = True,
) -> Any:
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
    ax.fill_between(
        xs,
        betas_s - 1.96 * ses_s,
        betas_s + 1.96 * ses_s,
        color="#1f77b4",
        alpha=0.18,
        label="per-γ 95% CI",
    )

    if show_bounds:
        ax.axhline(
            result.ci_lower,
            color="#d62728",
            linestyle="--",
            lw=1.2,
            label=f"union lower = {result.ci_lower:.3f}",
        )
        ax.axhline(
            result.ci_upper,
            color="#d62728",
            linestyle="--",
            lw=1.2,
            label=f"union upper = {result.ci_upper:.3f}",
        )

    ax.axhline(result.beta_hat, color="0.3", lw=0.8, linestyle=":")
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$\hat\beta$")
    ax.set_title(f"Plausibly exogenous sensitivity — {result.method}")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)

    if own_ax:
        fig.tight_layout()
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  Forest plot of IV estimates across methods (sp.iv.iv_compare output)
# ═══════════════════════════════════════════════════════════════════════


def plot_iv_forest(
    table: pd.DataFrame,
    estimate_col: str = "estimate",
    lo_col: str = "CI lower",
    hi_col: str = "CI upper",
    label_col: str = "method",
    *,
    ax: Any = None,
    sort_by: Optional[str] = None,
    reference: Optional[float] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Forest plot of point estimates and CIs across IV estimators.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`sp.iv.iv_compare` *or* a custom table with at
        least ``estimate_col`` / ``lo_col`` / ``hi_col`` / ``label_col``.
    sort_by : str, optional
        Column name to sort rows by.  By default, preserves table order.
    reference : float, optional
        Vertical guideline (e.g. OLS, prior literature, or H0).
    """
    plt = _mpl()
    df = table.copy()
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.45 * max(len(df), 4)))

    ys = np.arange(len(df))[::-1]
    estimates = df[estimate_col].astype(float).values
    lows = df[lo_col].astype(float).values
    highs = df[hi_col].astype(float).values
    labels = df[label_col].astype(str).values

    # CI whiskers
    ax.hlines(ys, lows, highs, color="#1f77b4", lw=2.0, alpha=0.85)
    # Point estimates
    ax.scatter(
        estimates, ys, s=46, color="#1f77b4", zorder=3, edgecolor="white", linewidth=0.8
    )

    if reference is not None and np.isfinite(reference):
        ax.axvline(
            reference,
            color="#d62728",
            linestyle="--",
            lw=1.0,
            label=f"reference = {reference:.3f}",
        )
        ax.legend(loc="best", frameon=False, fontsize=9)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("estimate")
    ax.set_title(title or "IV forest plot — point estimates and 95% CIs")
    ax.grid(axis="x", alpha=0.3)
    if own_ax:
        fig.tight_layout()
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  Forest plot built directly from an IVDiagResult bundle
# ═══════════════════════════════════════════════════════════════════════


def plot_iv_forest_from_diag(
    result: Any,
    ax: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Forest plot of all CIs reported by :func:`sp.iv.iv_diag`."""
    df = result.to_frame()
    return plot_iv_forest(
        df,
        estimate_col="estimate",
        lo_col="CI lower",
        hi_col="CI upper",
        label_col="estimator",
        ax=ax,
        reference=result.beta_2sls,
        title=title or "IV diagnostic bundle — point estimates and CIs",
    )


# ═══════════════════════════════════════════════════════════════════════
#  Weak-IV-robust CI overlay (AR / CLR / K)
# ═══════════════════════════════════════════════════════════════════════


def plot_weak_iv_ci_overlay(
    result: Any,
    ax: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Compact panel showing analytic, tF, AR (and optional CLR/K) sets."""
    plt = _mpl()
    rows: List[Tuple[str, float, float, float, str]] = []
    rows.append(("Wald", *result.ci_analytic_2sls, result.beta_2sls, "#1f77b4"))
    rows.append(("LMMP tF", *result.tF_adjusted_ci, result.beta_2sls, "#9467bd"))
    rows.append(("AR", *result.ar_ci, result.beta_2sls, "#2ca02c"))
    if result.clr_ci is not None:
        rows.append(("CLR", *result.clr_ci, result.beta_2sls, "#ff7f0e"))
    if result.k_ci is not None:
        rows.append(("K", *result.k_ci, result.beta_2sls, "#8c564b"))
    if result.bootstrap_ci_pairs is not None:
        rows.append(
            ("pairs boot", *result.bootstrap_ci_pairs, result.beta_2sls, "#17becf")
        )
    if result.bootstrap_ci_wild is not None:
        rows.append(
            ("wild boot", *result.bootstrap_ci_wild, result.beta_2sls, "#bcbd22")
        )
    if result.ltz_ci is not None:
        rows.append(("CHR LTZ", *result.ltz_ci, result.beta_2sls, "#7f7f7f"))

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.55 * len(rows)))
    ys = np.arange(len(rows))[::-1]
    for i, (lab, lo, hi, b, color) in enumerate(rows):
        y = ys[i]
        if not np.isfinite(lo):
            lo = b - 8 * abs(result.se_2sls)
            ax.annotate(
                "←", xy=(lo, y), fontsize=14, ha="left", va="center", color=color
            )
        if not np.isfinite(hi):
            hi = b + 8 * abs(result.se_2sls)
            ax.annotate(
                "→", xy=(hi, y), fontsize=14, ha="right", va="center", color=color
            )
        ax.hlines(y, lo, hi, color=color, lw=2.4, alpha=0.85)
        ax.scatter(
            [b], [y], s=42, color=color, zorder=3, edgecolor="white", linewidth=0.8
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(result.beta_2sls, color="0.4", linestyle=":", lw=1)
    ax.set_xlabel(r"$\beta$")
    title_text = title or (
        f"IV confidence sets — F = {result.first_stage_F:.1f}, "
        f"effective F = {result.effective_F:.1f}"
    )
    ax.set_title(title_text)
    ax.grid(axis="x", alpha=0.3)
    if own_ax:
        fig.tight_layout()
    return ax


# ═══════════════════════════════════════════════════════════════════════
#  2x2 diagnostic panel built from an IVDiagResult
# ═══════════════════════════════════════════════════════════════════════


def plot_iv_diagnostics(
    result: Any,
    fig: Any = None,
    suptitle: Optional[str] = None,
) -> Any:
    """A four-panel ``IVDiagResult`` summary:

    (top-left)  first-stage fit / instrument strength
    (top-right) AR confidence set on a β-grid
    (bot-left)  forest plot of all CI methods
    (bot-right) leverage-vs-residual diagnostic (Young 2022 spirit)
    """
    plt = _mpl()
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    else:
        axes = fig.subplots(2, 2)

    raw = result.raw

    # 1) first-stage scatter
    plot_first_stage(
        endog=raw["D"],
        instruments=raw["Z"],
        exog=raw["W_no_const"],
        endog_name=result.endog,
        ax=axes[0][0],
    )

    # 2) AR set with a quick grid (re-uses analytic AR set)
    grid_lo, grid_hi = result.ar_ci
    if not np.isfinite(grid_lo) or not np.isfinite(grid_hi):
        grid_lo = result.beta_2sls - 8 * result.se_2sls
        grid_hi = result.beta_2sls + 8 * result.se_2sls
    pad = (grid_hi - grid_lo) * 0.4 + 1e-6
    grid = np.linspace(grid_lo - pad, grid_hi + pad, 401)
    try:
        # call the legacy plot_ar_confidence_set with the bundled raw arrays
        plot_ar_confidence_set(
            y=raw["Y"],
            endog=raw["D"],
            instruments=raw["Z"],
            exog=raw["W_no_const"],
            beta_grid=grid,
            level=1.0 - result.alpha,
            ax=axes[0][1],
        )
    except Exception:  # pragma: no cover
        axes[0][1].set_visible(False)

    # 3) forest plot of all CIs
    plot_weak_iv_ci_overlay(result, ax=axes[1][0])

    # 4) leverage / residual scatter — compute the diagonal of the hat
    # matrix only; the explicit n × n form is OOM-unsafe at n ≥ 50k.
    Z, W = raw["Z"], raw["W"]
    Z_full = np.column_stack([Z, W])
    V = np.linalg.pinv(Z_full.T @ Z_full)
    leverage = np.einsum("ij,jk,ik->i", Z_full, V, Z_full)
    resid = raw.get("resid_2sls")
    ax_lev = axes[1][1]
    ax_lev.scatter(leverage, resid, s=8, alpha=0.4, color="0.5")
    high_lev = leverage > 2 * np.mean(leverage)
    if high_lev.any():
        ax_lev.scatter(
            leverage[high_lev],
            resid[high_lev],
            s=22,
            color="#d62728",
            label="high leverage",
        )
        ax_lev.legend(loc="best", frameon=False, fontsize=9)
    ax_lev.axhline(0, color="0.4", lw=0.8)
    ax_lev.set_xlabel("hat-matrix leverage on Z + W")
    ax_lev.set_ylabel("2SLS residual")
    ax_lev.set_title("Leverage diagnostic (Young 2022)")
    ax_lev.grid(alpha=0.3)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_first_stage",
    "plot_ar_confidence_set",
    "plot_mte_curve",
    "plot_plausibly_exogenous",
    "plot_iv_forest",
    "plot_iv_forest_from_diag",
    "plot_weak_iv_ci_overlay",
    "plot_iv_diagnostics",
]
