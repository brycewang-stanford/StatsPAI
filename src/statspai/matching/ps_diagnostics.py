"""
Propensity score diagnostics for matching and IPW analyses.

Comprehensive toolkit for assessing overlap, balance, and propensity score
quality — the essential diagnostics that existing Python packages handle poorly.

Functions
---------
propensity_score : Estimate propensity scores via logit, probit, or GBM.
overlap_plot     : Mirrored density plot of PS distributions by treatment.
trimming         : Crump et al. (2009) optimal PS trimming.
love_plot        : Dot plot of standardized mean differences (before/after).
ps_balance       : Comprehensive balance table with SMD, KS, variance ratios.

Classes
-------
PSBalanceResult  : Container for balance diagnostics with summary/plot methods.

References
----------
Crump, R.K., Hotz, V.J., Imbens, G.W. and Mitnik, O.A. (2009).
    "Dealing with limited overlap in estimation of average treatment effects."
    Biometrika, 96(1), 187-199.
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Austin, P.C. (2011). Multivariate Behavioral Research, 46(3), 399-424. [@crump2009dealing]
"""

from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats, optimize


# ======================================================================
# Propensity score estimation
# ======================================================================

def propensity_score(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    method: str = "logit",
    trimming: Optional[str] = None,
) -> pd.Series:
    """Estimate propensity scores P(D=1|X).

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Name of binary treatment column (0/1).
    covariates : list of str
        Covariate column names.
    method : {'logit', 'probit', 'gbm'}
        Estimation method.  ``'logit'`` uses IRLS (no sklearn needed).
        ``'probit'`` uses scipy.optimize.  ``'gbm'`` tries sklearn
        GradientBoostingClassifier, falling back to logit with interactions.
    trimming : {None, 'crump'}
        If ``'crump'``, apply Crump et al. (2009) trimming after estimation.
        Trimmed observations receive ``NaN`` scores.

    Returns
    -------
    pd.Series
        Propensity scores indexed like *data*.
    """
    D = data[treatment].values.astype(float)
    X = data[covariates].values.astype(float)

    if method == "logit":
        ps = _logit_irls(X, D)
    elif method == "probit":
        ps = _probit_mle(X, D)
    elif method == "gbm":
        ps = _gbm_ps(X, D)
    else:
        raise ValueError(f"method must be 'logit', 'probit', or 'gbm', got '{method}'")

    # Clip to avoid exact 0/1
    ps = np.clip(ps, 1e-8, 1 - 1e-8)
    ps_series = pd.Series(ps, index=data.index, name="propensity_score")

    if trimming == "crump":
        alpha = _crump_alpha(ps)
        mask = (ps < alpha) | (ps > 1 - alpha)
        ps_series[mask] = np.nan

    return ps_series


def _logit_irls(X: np.ndarray, D: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Logistic regression via iteratively reweighted least squares."""
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])
    beta = np.zeros(Xa.shape[1])

    for _ in range(max_iter):
        eta = Xa @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        W = np.maximum(W, 1e-12)
        z = eta + (D - mu) / W
        # Weighted least squares step
        XtW = Xa.T * W
        try:
            beta_new = np.linalg.solve(XtW @ Xa, XtW @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtW @ Xa, XtW @ z, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = Xa @ beta
    eta = np.clip(eta, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


def _probit_mle(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Probit regression via maximum likelihood (scipy.optimize)."""
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])

    def neg_loglik(beta):
        eta = Xa @ beta
        eta = np.clip(eta, -30, 30)
        Phi = stats.norm.cdf(eta)
        Phi = np.clip(Phi, 1e-12, 1 - 1e-12)
        return -np.sum(D * np.log(Phi) + (1 - D) * np.log(1 - Phi))

    def grad(beta):
        eta = Xa @ beta
        eta = np.clip(eta, -30, 30)
        Phi = stats.norm.cdf(eta)
        Phi = np.clip(Phi, 1e-12, 1 - 1e-12)
        phi = stats.norm.pdf(eta)
        lam = D * phi / Phi - (1 - D) * phi / (1 - Phi)
        return -Xa.T @ lam

    beta0 = np.zeros(Xa.shape[1])
    result = optimize.minimize(neg_loglik, beta0, jac=grad, method="BFGS",
                               options={"maxiter": 200})
    eta = Xa @ result.x
    eta = np.clip(eta, -30, 30)
    return stats.norm.cdf(eta)


def _gbm_ps(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """GBM propensity scores via sklearn, fallback to logit + interactions."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        gbm.fit(X, D)
        return gbm.predict_proba(X)[:, 1]
    except ImportError:
        import warnings
        warnings.warn(
            "sklearn not available for GBM; falling back to logit with "
            "pairwise interactions.",
            stacklevel=3,
        )
        # Build interaction terms
        from itertools import combinations
        interactions = []
        for i, j in combinations(range(X.shape[1]), 2):
            interactions.append(X[:, i] * X[:, j])
        if interactions:
            X_aug = np.column_stack([X] + interactions)
        else:
            X_aug = X
        return _logit_irls(X_aug, D)


# ======================================================================
# Crump et al. (2009) trimming
# ======================================================================

def _crump_alpha(ps: np.ndarray) -> float:
    """Find optimal Crump trimming threshold alpha.

    Solves: alpha = 1 / (2 * E[1/(e(1-e)) * I(alpha <= e <= 1-alpha)])
    via grid search on [0, 0.5).
    """
    ps_clean = ps[np.isfinite(ps)]
    inv_var = 1.0 / (ps_clean * (1 - ps_clean))

    best_alpha = 0.0
    alphas = np.linspace(0, 0.49, 500)
    for a in alphas:
        mask = (ps_clean >= a) & (ps_clean <= 1 - a)
        if mask.sum() < 2:
            break
        rhs = 1.0 / (2.0 * np.mean(inv_var[mask]))
        if a <= rhs:
            best_alpha = a
        else:
            break
    return best_alpha


def trimming(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    method: str = "crump",
    ps: Optional[pd.Series] = None,
    ps_method: str = "logit",
) -> pd.DataFrame:
    """Trim sample to optimal overlap region.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariates for PS estimation (if *ps* not supplied).
    method : {'crump', 'sturmer'}
        ``'crump'`` uses Crump et al. (2009) optimal rule.
        ``'sturmer'`` trims at the fixed [0.1, 0.9] interval.
    ps : Series, optional
        Pre-estimated propensity scores.  If None, estimated via *ps_method*.
    ps_method : str
        Method for PS estimation if *ps* is None.

    Returns
    -------
    DataFrame
        Trimmed data (rows with PS in the overlap region).
    """
    if ps is None:
        ps = propensity_score(data, treatment, covariates, method=ps_method)

    ps_vals = ps.values

    if method == "crump":
        alpha = _crump_alpha(ps_vals[np.isfinite(ps_vals)])
    elif method == "sturmer":
        alpha = 0.1
    else:
        raise ValueError(f"method must be 'crump' or 'sturmer', got '{method}'")

    mask = (ps_vals >= alpha) & (ps_vals <= 1 - alpha) & np.isfinite(ps_vals)
    return data.loc[mask].copy()


# ======================================================================
# Standardized mean difference & balance utilities
# ======================================================================

def _smd(x_t: np.ndarray, x_c: np.ndarray,
         w_t: Optional[np.ndarray] = None,
         w_c: Optional[np.ndarray] = None) -> float:
    """Standardized mean difference (Austin 2011 formula)."""
    if w_t is not None:
        mean_t = np.average(x_t, weights=w_t)
        var_t = np.average((x_t - mean_t) ** 2, weights=w_t)
    else:
        mean_t = np.mean(x_t)
        var_t = np.var(x_t, ddof=1) if len(x_t) > 1 else 0.0

    if w_c is not None:
        mean_c = np.average(x_c, weights=w_c)
        var_c = np.average((x_c - mean_c) ** 2, weights=w_c)
    else:
        mean_c = np.mean(x_c)
        var_c = np.var(x_c, ddof=1) if len(x_c) > 1 else 0.0

    denom = np.sqrt((var_t + var_c) / 2.0)
    if denom < 1e-12:
        return 0.0
    return (mean_t - mean_c) / denom


def _variance_ratio(x_t: np.ndarray, x_c: np.ndarray,
                    w_t: Optional[np.ndarray] = None,
                    w_c: Optional[np.ndarray] = None) -> float:
    """Variance ratio (treated / control)."""
    if w_t is not None:
        mean_t = np.average(x_t, weights=w_t)
        var_t = np.average((x_t - mean_t) ** 2, weights=w_t)
    else:
        var_t = np.var(x_t, ddof=1) if len(x_t) > 1 else 0.0

    if w_c is not None:
        mean_c = np.average(x_c, weights=w_c)
        var_c = np.average((x_c - mean_c) ** 2, weights=w_c)
    else:
        var_c = np.var(x_c, ddof=1) if len(x_c) > 1 else 0.0

    if var_c < 1e-12:
        return np.inf if var_t > 1e-12 else 1.0
    return var_t / var_c


def _ks_stat(x_t: np.ndarray, x_c: np.ndarray,
             w_t: Optional[np.ndarray] = None,
             w_c: Optional[np.ndarray] = None) -> float:
    """Kolmogorov-Smirnov statistic (unweighted or weighted)."""
    if w_t is None and w_c is None:
        return stats.ks_2samp(x_t, x_c).statistic

    # Weighted KS
    all_vals = np.sort(np.concatenate([x_t, x_c]))
    if w_t is None:
        w_t = np.ones(len(x_t))
    if w_c is None:
        w_c = np.ones(len(x_c))

    cdf_t = np.searchsorted(np.sort(x_t), all_vals, side="right")
    cdf_c = np.searchsorted(np.sort(x_c), all_vals, side="right")
    # Approximate weighted CDF via order-based approach
    idx_t = np.argsort(x_t)
    idx_c = np.argsort(x_c)
    cum_w_t = np.cumsum(w_t[idx_t])
    cum_w_c = np.cumsum(w_c[idx_c])
    ecdf_t = np.interp(all_vals, x_t[idx_t], cum_w_t / cum_w_t[-1],
                        left=0, right=1)
    ecdf_c = np.interp(all_vals, x_c[idx_c], cum_w_c / cum_w_c[-1],
                        left=0, right=1)
    return np.max(np.abs(ecdf_t - ecdf_c))


# ======================================================================
# PSBalanceResult
# ======================================================================

class PSBalanceResult:
    """Container for propensity score balance diagnostics.

    Attributes
    ----------
    table : DataFrame
        Balance statistics per covariate: mean_treat, mean_control,
        smd_raw, smd_weighted, variance_ratio, ks_stat.
    ps : Series
        Estimated propensity scores.
    """

    def __init__(self, table: pd.DataFrame, ps: pd.Series):
        self.table = table
        self.ps = ps

    def summary(self) -> str:
        """Formatted balance summary table."""
        lines = []
        lines.append("Propensity Score Balance Diagnostics")
        lines.append("=" * 70)
        lines.append("")

        fmt = table_to_string(self.table)
        lines.append(fmt)

        lines.append("")
        n_imbalanced_raw = (self.table["smd_raw"].abs() > 0.1).sum()
        n_imbalanced_wtd = (self.table["smd_weighted"].abs() > 0.1).sum()
        lines.append(f"Covariates with |SMD| > 0.1: {n_imbalanced_raw} (raw)"
                     f" -> {n_imbalanced_wtd} (weighted)")
        return "\n".join(lines)

    def love_plot(self, threshold: float = 0.1, **kwargs):
        """Convenience method: calls ``love_plot()`` from balance data."""
        return _love_plot_from_table(self.table, threshold=threshold, **kwargs)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        html = "<h4>Propensity Score Balance Diagnostics</h4>"
        html += self.table.to_html(float_format="{:.4f}".format)
        n_raw = (self.table["smd_raw"].abs() > 0.1).sum()
        n_wtd = (self.table["smd_weighted"].abs() > 0.1).sum()
        html += (f"<p>Covariates with |SMD| &gt; 0.1: {n_raw} (raw) "
                 f"&rarr; {n_wtd} (weighted)</p>")
        return html

    def __repr__(self):
        return self.summary()


def table_to_string(df: pd.DataFrame) -> str:
    """Format a DataFrame as an aligned text table."""
    return df.to_string(float_format=lambda x: f"{x:.4f}")


# ======================================================================
# ps_balance — comprehensive balance table
# ======================================================================

def ps_balance(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    method: str = "logit",
) -> PSBalanceResult:
    """Compute comprehensive propensity score balance table.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariate columns to assess balance for.
    weights : array-like, optional
        IPW or matching weights.  If None, inverse-PS weights are
        computed automatically from estimated propensity scores.
    method : str
        PS estimation method ('logit', 'probit', 'gbm').

    Returns
    -------
    PSBalanceResult
        Object with ``.table``, ``.ps``, ``.summary()``, ``.love_plot()``.
    """
    D = data[treatment].values.astype(float)
    ps = propensity_score(data, treatment, covariates, method=method)
    ps_vals = ps.values

    # Compute IPW weights if not supplied
    if weights is not None:
        w = np.asarray(weights, dtype=float)
    else:
        # ATE weights: 1/e for treated, 1/(1-e) for control
        w = np.where(D == 1, 1.0 / ps_vals, 1.0 / (1.0 - ps_vals))

    treat_mask = D == 1
    ctrl_mask = D == 0

    rows = []
    for cov in covariates:
        x = data[cov].values.astype(float)
        x_t = x[treat_mask]
        x_c = x[ctrl_mask]
        w_t = w[treat_mask]
        w_c = w[ctrl_mask]

        mean_t = np.mean(x_t)
        mean_c = np.mean(x_c)
        smd_raw = _smd(x_t, x_c)
        smd_weighted = _smd(x_t, x_c, w_t, w_c)
        vr = _variance_ratio(x_t, x_c, w_t, w_c)
        ks = _ks_stat(x_t, x_c, w_t, w_c)

        rows.append({
            "variable": cov,
            "mean_treat": mean_t,
            "mean_control": mean_c,
            "smd_raw": smd_raw,
            "smd_weighted": smd_weighted,
            "variance_ratio": vr,
            "ks_stat": ks,
        })

    table = pd.DataFrame(rows).set_index("variable")
    return PSBalanceResult(table=table, ps=ps)


# ======================================================================
# Plotting helpers
# ======================================================================

def _require_matplotlib():
    """Import matplotlib or raise a helpful error."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


# ======================================================================
# overlap_plot
# ======================================================================

def overlap_plot(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    ps: Optional[pd.Series] = None,
    method: str = "logit",
    ax=None,
    figsize: Tuple[float, float] = (8, 4),
    title: str = "Propensity Score Overlap",
) -> Tuple:
    """Mirrored density plot of propensity scores by treatment group.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariates for PS estimation (ignored if *ps* supplied).
    ps : Series, optional
        Pre-estimated propensity scores.
    method : str
        PS estimation method if *ps* is None.
    ax : matplotlib Axes, optional
        Axes to plot on.  If None, a new figure is created.
    figsize : tuple
        Figure size (width, height).
    title : str
        Plot title.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib figure and axes.
    """
    plt = _require_matplotlib()

    if ps is None:
        ps = propensity_score(data, treatment, covariates, method=method)

    D = data[treatment].values.astype(float)
    ps_vals = ps.values

    ps_treat = ps_vals[D == 1]
    ps_ctrl = ps_vals[D == 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Kernel density estimation
    grid = np.linspace(0, 1, 300)

    kde_t = stats.gaussian_kde(ps_treat, bw_method="scott")
    kde_c = stats.gaussian_kde(ps_ctrl, bw_method="scott")

    density_t = kde_t(grid)
    density_c = kde_c(grid)

    # Mirrored: treated above, control below
    ax.fill_between(grid, density_t, alpha=0.35, color="#2171b5",
                    label=f"Treated (n={len(ps_treat)})")
    ax.fill_between(grid, -density_c, alpha=0.35, color="#cb181d",
                    label=f"Control (n={len(ps_ctrl)})")
    ax.plot(grid, density_t, color="#2171b5", linewidth=1.2)
    ax.plot(grid, -density_c, color="#cb181d", linewidth=1.2)

    # Common support region
    cs_low = max(ps_treat.min(), ps_ctrl.min())
    cs_high = min(ps_treat.max(), ps_ctrl.max())
    ax.axvspan(cs_low, cs_high, alpha=0.08, color="green",
               label=f"Common support [{cs_low:.2f}, {cs_high:.2f}]")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    return fig, ax


# ======================================================================
# love_plot
# ======================================================================

def love_plot(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    threshold: float = 0.1,
    ps_method: str = "logit",
    ax=None,
    figsize: Tuple[float, float] = (7, None),
    title: str = "Covariate Balance (Love Plot)",
) -> Tuple:
    """Love plot: dot plot of standardized mean differences before/after.

    Parameters
    ----------
    data : DataFrame
        Input data.
    treatment : str
        Binary treatment column.
    covariates : list of str
        Covariate columns.
    weights : array-like, optional
        IPW or matching weights.  If None, inverse-PS weights are computed.
    threshold : float
        SMD threshold for the vertical dashed line (default 0.1).
    ps_method : str
        PS estimation method for balance computation.
    ax : matplotlib Axes, optional
    figsize : tuple
        (width, height).  Height defaults to 0.4 * n_covariates + 1.
    title : str
        Plot title.

    Returns
    -------
    (fig, ax) : tuple
    """
    result = ps_balance(data, treatment, covariates,
                        weights=weights, method=ps_method)
    return _love_plot_from_table(result.table, threshold=threshold,
                                ax=ax, figsize=figsize, title=title)


def _love_plot_from_table(
    table: pd.DataFrame,
    threshold: float = 0.1,
    ax=None,
    figsize: Tuple[float, float] = (7, None),
    title: str = "Covariate Balance (Love Plot)",
) -> Tuple:
    """Internal love plot renderer from a balance table."""
    plt = _require_matplotlib()

    n_vars = len(table)
    if figsize[1] is None:
        figsize = (figsize[0], max(3, 0.4 * n_vars + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    variables = table.index.tolist()
    smd_raw = table["smd_raw"].abs().values
    smd_wtd = table["smd_weighted"].abs().values
    y_pos = np.arange(n_vars)

    # Horizontal connecting lines
    for i in range(n_vars):
        ax.plot([smd_raw[i], smd_wtd[i]], [y_pos[i], y_pos[i]],
                color="gray", linewidth=0.8, zorder=1)

    # Dots
    ax.scatter(smd_raw, y_pos, color="#cb181d", s=50, zorder=2,
               label="Raw", marker="o", edgecolors="white", linewidth=0.5)
    ax.scatter(smd_wtd, y_pos, color="#2171b5", s=50, zorder=2,
               label="Weighted", marker="s", edgecolors="white", linewidth=0.5)

    # Threshold line
    ax.axvline(threshold, color="black", linestyle="--", linewidth=0.8,
               label=f"Threshold ({threshold})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(left=0)

    fig.tight_layout()
    return fig, ax
