"""
Multi-cutoff and multi-score RD designs.

Implements RD designs with multiple cutoffs (rdmc) and
multiple running variables / geographic RD (rdms).

Equivalent to Stata/R's ``rdmulti`` package (Cattaneo et al. 2024).

References
----------
Cattaneo, M.D., Keele, L., Titiunik, R. & Vazquez-Bare, G. (2016).
"Interpreting Regression Discontinuity Designs with Multiple Cutoffs."
*Journal of Politics*, 78(4), 1229-1248.

Cattaneo, M.D., Keele, L., Titiunik, R. & Vazquez-Bare, G. (2024).
"Extrapolating Treatment Effects in Multi-Cutoff Regression Discontinuity
Designs." *JASA*, 119(545), 118-127.

Keele, L. & Titiunik, R. (2015).
"Geographic Boundaries as Regression Discontinuities."
*Political Analysis*, 23(1), 127-155.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from ..core.results import CausalResult


class RDMultiResult:
    """Results from multi-cutoff/multi-score RD."""

    def __init__(self, cutoff_results, pooled_estimate, pooled_se,
                 pooled_ci, n_cutoffs, n_total, method):
        self.cutoff_results = cutoff_results  # list of dicts
        self.pooled_estimate = pooled_estimate
        self.pooled_se = pooled_se
        self.pooled_ci = pooled_ci
        self.n_cutoffs = n_cutoffs
        self.n_total = n_total
        self.method = method

    def summary(self) -> str:
        lines = [
            f"Multi-Cutoff Regression Discontinuity ({self.method})",
            "=" * 65,
            f"Number of cutoffs: {self.n_cutoffs}",
            f"Total observations: {self.n_total}",
            "",
            f"{'Cutoff':<10s} {'N':>6s} {'Estimate':>10s} {'SE':>10s} "
            f"{'95% CI':>22s} {'p-value':>10s}",
            "-" * 65,
        ]
        for cr in self.cutoff_results:
            ci = f"[{cr['ci_lower']:.4f}, {cr['ci_upper']:.4f}]"
            lines.append(f"{cr['cutoff']:<10.2f} {cr['n']:>6d} {cr['estimate']:>10.4f} "
                         f"{cr['se']:>10.4f} {ci:>22s} {cr['p_value']:>10.4f}")

        lines.append("-" * 65)
        ci_pooled = f"[{self.pooled_ci[0]:.4f}, {self.pooled_ci[1]:.4f}]"
        lines.append(f"{'Pooled':<10s} {self.n_total:>6d} {self.pooled_estimate:>10.4f} "
                     f"{self.pooled_se:>10.4f} {ci_pooled:>22s}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Forest plot of cutoff-specific and pooled estimates."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(self.cutoff_results) * 0.5 + 2)))

        labels = [f"c = {cr['cutoff']:.1f}" for cr in self.cutoff_results] + ['Pooled']
        estimates = [cr['estimate'] for cr in self.cutoff_results] + [self.pooled_estimate]
        ci_lowers = [cr['ci_lower'] for cr in self.cutoff_results] + [self.pooled_ci[0]]
        ci_uppers = [cr['ci_upper'] for cr in self.cutoff_results] + [self.pooled_ci[1]]

        y_pos = range(len(labels))
        errors = [[e - cl for e, cl in zip(estimates, ci_lowers)],
                  [cu - e for e, cu in zip(estimates, ci_uppers)]]

        colors = ['steelblue'] * len(self.cutoff_results) + ['red']
        ax.errorbar(estimates, y_pos, xerr=errors, fmt='o', color='steelblue',
                    capsize=3, elinewidth=1.5)
        ax.scatter([self.pooled_estimate], [len(labels)-1], color='red', s=80, zorder=5,
                   marker='D')
        ax.axvline(0, color='gray', ls='--', lw=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Treatment Effect')
        ax.set_title('Multi-Cutoff RD Estimates')
        plt.tight_layout()
        return ax


def _local_linear_rd(y, x, c, h, kernel='triangular'):
    """Local linear RD estimate at cutoff c with bandwidth h."""
    x_centered = x - c

    # Kernel weights
    u = x_centered / h
    if kernel == 'triangular':
        w = np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
    elif kernel == 'uniform':
        w = np.where(np.abs(u) <= 1, 0.5, 0.0)
    elif kernel == 'epanechnikov':
        w = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    else:
        w = np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)

    mask = w > 0
    if mask.sum() < 4:
        return np.nan, np.nan, 0

    y_m, x_m, w_m = y[mask], x_centered[mask], w[mask]
    D_m = (x_m >= 0).astype(float)

    # Local linear: y = a + b*x + tau*D + delta*D*x
    X = np.column_stack([np.ones(len(x_m)), x_m, D_m, D_m * x_m])
    W = np.diag(w_m)

    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y_m
        beta = np.linalg.solve(XtWX, XtWy)
        resid = y_m - X @ beta
        sigma2 = np.sum(w_m * resid**2) / max(mask.sum() - 4, 1)
        var_cov = sigma2 * np.linalg.inv(XtWX)
        tau = beta[2]
        se = np.sqrt(var_cov[2, 2])
    except np.linalg.LinAlgError:
        tau, se = np.nan, np.nan

    return tau, se, mask.sum()


def rdmc(
    data: pd.DataFrame,
    y: str,
    x: str,
    cutoffs: List[float],
    bandwidth: float = None,
    kernel: str = "triangular",
    pooling: str = "ivw",
    alpha: float = 0.05,
) -> RDMultiResult:
    """
    Multi-cutoff RD design.

    Estimates treatment effects at multiple cutoffs and pools them.

    Equivalent to ``rdmulti::rdmc()`` in R/Stata.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x : str
        Running variable.
    cutoffs : list of float
        Cutoff values.
    bandwidth : float, optional
        Bandwidth for local polynomial. If None, uses Silverman rule.
    kernel : str, default 'triangular'
    pooling : str, default 'ivw'
        Pooling method: 'ivw' (inverse-variance weighted) or 'equal'.
    alpha : float, default 0.05

    Returns
    -------
    RDMultiResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.rdmc(df, y='score', x='running_var', cutoffs=[50, 70, 90])
    >>> print(result.summary())
    >>> result.plot()
    """
    y_data = data[y].values.astype(float)
    x_data = data[x].values.astype(float)

    if bandwidth is None:
        bandwidth = 1.06 * np.std(x_data) * len(x_data)**(-1/5)

    cutoff_results = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for c in cutoffs:
        tau, se, n_local = _local_linear_rd(y_data, x_data, c, bandwidth, kernel)
        p_val = 2 * (1 - stats.norm.cdf(abs(tau / se))) if se > 0 else np.nan

        cutoff_results.append({
            'cutoff': c,
            'estimate': tau,
            'se': se,
            'ci_lower': tau - z_crit * se,
            'ci_upper': tau + z_crit * se,
            'p_value': p_val,
            'n': n_local,
            'bandwidth': bandwidth,
        })

    # Pool estimates
    valid = [cr for cr in cutoff_results if np.isfinite(cr['se']) and cr['se'] > 0]
    if len(valid) > 0:
        if pooling == 'ivw':
            weights = np.array([1 / cr['se']**2 for cr in valid])
            weights /= weights.sum()
            pooled = sum(w * cr['estimate'] for w, cr in zip(weights, valid))
            pooled_se = np.sqrt(1 / sum(1 / cr['se']**2 for cr in valid))
        else:
            pooled = np.mean([cr['estimate'] for cr in valid])
            pooled_se = np.sqrt(np.mean([cr['se']**2 for cr in valid]) / len(valid))
    else:
        pooled, pooled_se = np.nan, np.nan

    pooled_ci = (pooled - z_crit * pooled_se, pooled + z_crit * pooled_se)

    return RDMultiResult(
        cutoff_results=cutoff_results,
        pooled_estimate=pooled,
        pooled_se=pooled_se,
        pooled_ci=pooled_ci,
        n_cutoffs=len(cutoffs),
        n_total=len(y_data),
        method='Multi-Cutoff RD (rdmc)',
    )


def rdms(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    cutoff1: float = 0,
    cutoff2: float = 0,
    bandwidth: float = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
) -> CausalResult:
    """
    Multi-score / Geographic RD design.

    Handles two-dimensional running variables for geographic boundaries.

    Equivalent to ``rdmulti::rdms()`` and Keele & Titiunik (2015).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x1 : str
        First running variable (e.g., latitude distance to boundary).
    x2 : str
        Second running variable (e.g., longitude distance to boundary).
    cutoff1 : float, default 0
        Cutoff for x1.
    cutoff2 : float, default 0
        Cutoff for x2.
    bandwidth : float, optional
    kernel : str, default 'triangular'
    alpha : float, default 0.05

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.rdms(df, y='outcome', x1='dist_lat', x2='dist_lon')
    >>> print(result.summary())
    """
    y_data = data[y].values.astype(float)
    x1_data = data[x1].values.astype(float) - cutoff1
    x2_data = data[x2].values.astype(float) - cutoff2

    # Euclidean distance to boundary
    distance = np.sqrt(x1_data**2 + x2_data**2)

    # Treatment: positive side (both x1 and x2 > 0, or use x1 as primary)
    # Convention: treatment = 1 if x1 >= 0
    treatment = (x1_data >= 0).astype(float)

    if bandwidth is None:
        bandwidth = 1.06 * np.std(distance) * len(distance)**(-1/5)

    # Kernel weights based on distance
    u = distance / bandwidth
    if kernel == 'triangular':
        w = np.where(u <= 1, 1 - u, 0.0)
    elif kernel == 'uniform':
        w = np.where(u <= 1, 1.0, 0.0)
    else:
        w = np.where(u <= 1, 1 - u, 0.0)

    mask = w > 0
    n_local = mask.sum()

    if n_local < 10:
        warnings.warn("Very few observations within bandwidth")

    y_m = y_data[mask]
    x1_m = x1_data[mask]
    x2_m = x2_data[mask]
    D_m = treatment[mask]
    w_m = w[mask]

    # Local linear with 2D running variable
    X = np.column_stack([np.ones(n_local), x1_m, x2_m, D_m,
                         D_m * x1_m, D_m * x2_m])
    W = np.diag(w_m)

    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y_m
        beta = np.linalg.solve(XtWX, XtWy)
        resid = y_m - X @ beta
        sigma2 = np.sum(w_m * resid**2) / max(n_local - X.shape[1], 1)
        var_cov = sigma2 * np.linalg.inv(XtWX)

        tau = beta[3]  # treatment coefficient
        se = np.sqrt(var_cov[3, 3])
    except np.linalg.LinAlgError:
        tau, se = np.nan, np.nan
        beta = np.full(6, np.nan)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_val = 2 * (1 - stats.norm.cdf(abs(tau / se))) if se > 0 else np.nan

    return CausalResult(
        method='Geographic RD (rdms)',
        estimand='ATE at boundary',
        estimate=tau,
        se=se,
        pvalue=p_val,
        ci=(tau - z_crit * se, tau + z_crit * se),
        alpha=alpha,
        n_obs=len(y_data),
        model_info={
            'bandwidth': bandwidth,
            'kernel': kernel,
            'cutoff1': cutoff1,
            'cutoff2': cutoff2,
            'n_local': n_local,
        },
    )
