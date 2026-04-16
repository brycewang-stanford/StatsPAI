"""
Local randomization inference for regression discontinuity designs.

Implements the methodology of Cattaneo, Titiunik, and Vazquez-Bare (2016) for
inference in RD designs under a local randomization assumption. Within a small
window around the cutoff, units are treated as if randomly assigned to
treatment or control.

Functions
---------
rdrandinf : Main randomization inference for RD designs.
rdwinselect : Data-driven window selection for local randomization.
rdsensitivity : Sensitivity of results across different windows.
rdrbounds : Rosenbaum sensitivity bounds for hidden bias.

References
----------
Cattaneo, M.D., Titiunik, R. and Vazquez-Bare, G. (2016).
"Inference in Regression Discontinuity Designs under Local Randomization."
*The Stata Journal*, 16(2), 331-367.
"""

from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


# ======================================================================
# Citation registration
# ======================================================================

CausalResult._CITATIONS['rdlocrand'] = (
    "@article{cattaneo2016inference,\n"
    "  title={Inference in Regression Discontinuity Designs under\n"
    "  Local Randomization},\n"
    "  author={Cattaneo, Matias D and Titiunik, Roc{\\'\\i}o and\n"
    "  Vazquez-Bare, Gonzalo},\n"
    "  journal={The Stata Journal},\n"
    "  volume={16},\n"
    "  number={2},\n"
    "  pages={331--367},\n"
    "  year={2016}\n"
    "}"
)


# ======================================================================
# Internal helpers
# ======================================================================

def _select_window(data: pd.DataFrame, x: str, c: float,
                   wl: Optional[float], wr: Optional[float]):
    """Return mask for observations within [c+wl, c+wr]."""
    xv = data[x].values
    if wl is None or wr is None:
        raise ValueError(
            "Window bounds wl and wr must be specified. "
            "Use rdwinselect() to choose a data-driven window."
        )
    left = c + wl   # wl is typically negative
    right = c + wr
    return (xv >= left) & (xv <= right)


def _polynomial_residuals(y: np.ndarray, x: np.ndarray, p: int,
                          covs: Optional[np.ndarray] = None) -> np.ndarray:
    """Partial out polynomial in X (and optional covariates) from Y."""
    n = len(y)
    parts = []
    # Polynomial terms x^1, ..., x^p (if p > 0)
    if p > 0:
        parts.extend([x ** k for k in range(1, p + 1)])
    # Covariates
    if covs is not None:
        if covs.ndim == 1:
            parts.append(covs.reshape(-1, 1))
        elif covs.shape[1] > 0:
            parts.append(covs)

    if len(parts) == 0:
        return y - np.mean(y)

    X_design = np.column_stack(parts)
    X_design = np.column_stack([np.ones(n), X_design])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    return y - X_design @ beta


def _diffmeans(y: np.ndarray, d: np.ndarray) -> float:
    """Difference in means: E[Y|D=1] - E[Y|D=0]."""
    return y[d == 1].mean() - y[d == 0].mean()


def _ks_stat(y: np.ndarray, d: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic."""
    stat, _ = sp_stats.ks_2samp(y[d == 1], y[d == 0])
    return stat


def _ranksum_stat(y: np.ndarray, d: np.ndarray) -> float:
    """Wilcoxon rank-sum (Mann-Whitney U) test statistic (standardised)."""
    n1 = int((d == 1).sum())
    n0 = int((d == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.0
    stat, _ = sp_stats.mannwhitneyu(
        y[d == 1], y[d == 0], alternative='two-sided'
    )
    # Standardise to make comparable across permutations
    mu = n1 * n0 / 2
    sigma = np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
    if sigma == 0:
        return 0.0
    return abs((stat - mu) / sigma)


_STAT_FUNCS = {
    'diffmeans': _diffmeans,
    'ksmirnov': _ks_stat,
    'ranksum': _ranksum_stat,
}


def _compute_stat(y: np.ndarray, d: np.ndarray, stat_name: str) -> float:
    """Dispatch to the requested test statistic."""
    return _STAT_FUNCS[stat_name](y, d)


def _permutation_pvalue(y: np.ndarray, d: np.ndarray, stat_name: str,
                        n_perms: int, rng: np.random.Generator,
                        two_sided: bool = True) -> tuple:
    """
    Fisher randomization p-value via permutation.

    Returns (observed_stat, perm_pvalue).
    """
    obs_stat = _compute_stat(y, d, stat_name)
    abs_obs = abs(obs_stat) if two_sided else obs_stat

    count = 0
    for _ in range(n_perms):
        d_perm = rng.permutation(d)
        perm_stat = _compute_stat(y, d_perm, stat_name)
        perm_abs = abs(perm_stat) if two_sided else perm_stat
        if perm_abs >= abs_obs - 1e-14:
            count += 1

    perm_pval = count / n_perms
    return obs_stat, perm_pval


def _asymptotic_pvalue(y: np.ndarray, d: np.ndarray,
                       stat_name: str) -> tuple:
    """
    Asymptotic p-value for the chosen test statistic.

    Returns (stat, pvalue).
    """
    if stat_name == 'diffmeans':
        y1, y0 = y[d == 1], y[d == 0]
        n1, n0 = len(y1), len(y0)
        if n1 < 2 or n0 < 2:
            return _diffmeans(y, d), np.nan
        diff = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)
        if se < 1e-14:
            return diff, 0.0 if abs(diff) > 1e-14 else 1.0
        t = diff / se
        pval = 2 * (1 - sp_stats.t.cdf(abs(t), df=n1 + n0 - 2))
        return diff, pval
    elif stat_name == 'ksmirnov':
        stat, pval = sp_stats.ks_2samp(y[d == 1], y[d == 0])
        return stat, pval
    elif stat_name == 'ranksum':
        stat, pval = sp_stats.mannwhitneyu(
            y[d == 1], y[d == 0], alternative='two-sided'
        )
        return stat, pval
    else:
        raise ValueError(f"Unknown statistic: {stat_name}")


def _wald_iv(y: np.ndarray, d_actual: np.ndarray,
             z: np.ndarray) -> tuple:
    """
    Wald (IV) estimator: tau = E[Y|Z=1]-E[Y|Z=0] / E[D|Z=1]-E[D|Z=0].

    Returns (estimate, se).
    """
    y1 = y[z == 1].mean()
    y0 = y[z == 0].mean()
    d1 = d_actual[z == 1].mean()
    d0 = d_actual[z == 0].mean()
    first_stage = d1 - d0
    if abs(first_stage) < 1e-14:
        return np.nan, np.nan
    tau = (y1 - y0) / first_stage

    # Delta method SE
    n1 = int((z == 1).sum())
    n0 = int((z == 0).sum())
    var_y1 = y[z == 1].var(ddof=1) / n1 if n1 > 1 else 0
    var_y0 = y[z == 0].var(ddof=1) / n0 if n0 > 1 else 0
    var_d1 = d_actual[z == 1].var(ddof=1) / n1 if n1 > 1 else 0
    var_d0 = d_actual[z == 0].var(ddof=1) / n0 if n0 > 1 else 0

    # Gradient of g(mu_y1, mu_y0, mu_d1, mu_d0) = (mu_y1-mu_y0)/(mu_d1-mu_d0)
    num = y1 - y0
    den = first_stage
    # Var(tau) via delta method
    var_num = var_y1 + var_y0
    var_den = var_d1 + var_d0
    se = np.sqrt(var_num / den**2 + num**2 * var_den / den**4)
    return tau, se


# ======================================================================
# Public API
# ======================================================================

def rdrandinf(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    wl: Optional[float] = None,
    wr: Optional[float] = None,
    statistic: str = 'diffmeans',
    p: int = 0,
    covs: Optional[List[str]] = None,
    kernel: str = 'uniform',
    n_perms: int = 1000,
    fuzzy: Optional[str] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> CausalResult:
    """
    Randomization inference for regression discontinuity designs.

    Under the local randomization assumption, units within a small window
    around the cutoff are treated as if randomly assigned. Inference is
    based on Fisher's randomization test.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    wl : float, optional
        Window left bound offset from cutoff (typically negative).
        The left edge of the window is ``c + wl``.
    wr : float, optional
        Window right bound offset from cutoff (typically positive).
        The right edge of the window is ``c + wr``.
    statistic : str, default 'diffmeans'
        Test statistic: 'diffmeans', 'ksmirnov', 'ranksum', or 'all'.
    p : int, default 0
        Polynomial order for adjustment (0 = unadjusted).
    covs : list of str, optional
        Covariate names to partial out before testing.
    kernel : str, default 'uniform'
        Kernel weighting (only 'uniform' currently supported for local
        randomization).
    n_perms : int, default 1000
        Number of permutations for Fisher randomization test.
    fuzzy : str, optional
        Actual treatment variable for fuzzy RD. The Wald (IV) estimator
        is computed within the window.
    alpha : float, default 0.05
        Significance level.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    CausalResult
        Result with treatment effect estimate, permutation and asymptotic
        p-values, and confidence interval.

    Notes
    -----
    The confidence interval is obtained by test inversion when
    ``statistic='diffmeans'``: the set of hypothesised effect values
    tau_0 that are not rejected by the permutation test at level alpha.

    References
    ----------
    Cattaneo, M.D., Titiunik, R. and Vazquez-Bare, G. (2016).
    "Inference in Regression Discontinuity Designs under Local
    Randomization." *The Stata Journal*, 16(2), 331-367.
    """
    rng = np.random.default_rng(seed)

    # --- subset to window ---
    mask = _select_window(data, x, c, wl, wr)
    df_w = data.loc[mask].copy()
    n_obs = len(df_w)
    if n_obs < 4:
        raise ValueError(
            f"Only {n_obs} observations in window [{c+wl}, {c+wr}]. "
            "Widen the window or check your data."
        )

    yv = df_w[y].values.astype(float)
    xv = df_w[x].values.astype(float)
    # Treatment: above cutoff
    z = (xv >= c).astype(int)
    n_right = int(z.sum())
    n_left = n_obs - n_right
    if n_left < 2 or n_right < 2:
        raise ValueError(
            f"Need >= 2 observations on each side of the cutoff; "
            f"got {n_left} left and {n_right} right."
        )

    # --- polynomial / covariate adjustment ---
    if p > 0 or (covs is not None and len(covs) > 0):
        cov_mat = df_w[covs].values.astype(float) if covs else None
        yv = _polynomial_residuals(yv, xv - c, p, cov_mat)

    # --- fuzzy RD: Wald estimator ---
    if fuzzy is not None:
        d_actual = df_w[fuzzy].values.astype(float)
        tau_iv, se_iv = _wald_iv(yv, d_actual, z)

        # Permutation p-value for fuzzy: permute Z, recompute Wald
        count = 0
        for _ in range(n_perms):
            z_perm = rng.permutation(z)
            tau_perm, _ = _wald_iv(yv, d_actual, z_perm)
            if not np.isnan(tau_perm) and abs(tau_perm) >= abs(tau_iv) - 1e-14:
                count += 1
        perm_pval = count / n_perms

        # Asymptotic p-value
        if se_iv > 0 and not np.isnan(se_iv):
            t_stat = tau_iv / se_iv
            asym_pval = 2 * (1 - sp_stats.norm.cdf(abs(t_stat)))
        else:
            asym_pval = np.nan

        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (tau_iv - z_crit * se_iv, tau_iv + z_crit * se_iv)

        return CausalResult(
            method='RD Local Randomization (Fuzzy)',
            estimand='LATE',
            estimate=float(tau_iv),
            se=float(se_iv),
            pvalue=float(perm_pval),
            ci=ci,
            alpha=alpha,
            n_obs=n_obs,
            model_info={
                'cutoff': c,
                'window': (c + wl, c + wr),
                'n_left': n_left,
                'n_right': n_right,
                'statistic': 'wald_iv',
                'polynomial_order': p,
                'n_perms': n_perms,
                'pvalue_permutation': perm_pval,
                'pvalue_asymptotic': asym_pval,
                'first_stage': float(
                    d_actual[z == 1].mean() - d_actual[z == 0].mean()
                ),
                'fuzzy_treatment': fuzzy,
            },
            _citation_key='rdlocrand',
        )

    # --- sharp RD ---
    stat_names = list(_STAT_FUNCS.keys()) if statistic == 'all' else [statistic]
    if statistic != 'all' and statistic not in _STAT_FUNCS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. "
            f"Choose from: 'diffmeans', 'ksmirnov', 'ranksum', 'all'."
        )

    results = {}
    for sname in stat_names:
        obs, perm_pval = _permutation_pvalue(yv, z, sname, n_perms, rng)
        _, asym_pval = _asymptotic_pvalue(yv, z, sname)
        results[sname] = {
            'observed_stat': obs,
            'pvalue_permutation': perm_pval,
            'pvalue_asymptotic': asym_pval,
        }

    # Primary statistic for the CausalResult
    primary = stat_names[0]
    tau = _diffmeans(yv, z)
    y1, y0 = yv[z == 1], yv[z == 0]
    se = np.sqrt(y1.var(ddof=1) / n_right + y0.var(ddof=1) / n_left)

    # Confidence interval by test inversion for diffmeans
    ci = _ci_test_inversion(yv, z, n_perms, alpha, rng)

    pval_main = results[primary]['pvalue_permutation']

    # Detail DataFrame if 'all'
    detail = None
    if statistic == 'all':
        rows = []
        for sname, res in results.items():
            rows.append({
                'statistic': sname,
                'observed': res['observed_stat'],
                'pvalue_perm': res['pvalue_permutation'],
                'pvalue_asym': res['pvalue_asymptotic'],
            })
        detail = pd.DataFrame(rows)

    return CausalResult(
        method='RD Local Randomization',
        estimand='ATE (local)',
        estimate=float(tau),
        se=float(se),
        pvalue=float(pval_main),
        ci=ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info={
            'cutoff': c,
            'window': (c + wl, c + wr),
            'n_left': n_left,
            'n_right': n_right,
            'statistic': statistic,
            'polynomial_order': p,
            'n_perms': n_perms,
            'covariates': covs,
            'results_by_stat': results,
            'pvalue_permutation': pval_main,
            'pvalue_asymptotic': results[primary]['pvalue_asymptotic'],
        },
        _citation_key='rdlocrand',
    )


def _ci_test_inversion(y: np.ndarray, d: np.ndarray, n_perms: int,
                       alpha: float, rng: np.random.Generator,
                       n_grid: int = 101) -> tuple:
    """
    Confidence interval by test inversion for difference-in-means.

    Shift Y under each hypothesised tau_0 and check if the Fisher test
    rejects. The CI is the range of non-rejected tau_0 values.
    """
    tau_hat = _diffmeans(y, d)
    se_hat = np.sqrt(
        y[d == 1].var(ddof=1) / (d == 1).sum() +
        y[d == 0].var(ddof=1) / (d == 0).sum()
    )
    if se_hat < 1e-14 or np.isnan(se_hat):
        return (tau_hat, tau_hat)

    # Search range: +/- 4 SE around point estimate
    lo = tau_hat - 4 * se_hat
    hi = tau_hat + 4 * se_hat
    grid = np.linspace(lo, hi, n_grid)

    not_rejected = []
    for tau0 in grid:
        # Under H0: tau = tau0, adjust treated outcomes
        y_adj = y.copy()
        y_adj[d == 1] = y[d == 1] - tau0
        # Test stat under null: diff should be ~0
        obs = abs(_diffmeans(y_adj, d))
        count = 0
        for _ in range(n_perms):
            d_perm = rng.permutation(d)
            perm_stat = abs(_diffmeans(y_adj, d_perm))
            if perm_stat >= obs - 1e-14:
                count += 1
        pval = count / n_perms
        if pval > alpha:
            not_rejected.append(tau0)

    if len(not_rejected) == 0:
        # Fall back to normal approximation
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        return (tau_hat - z_crit * se_hat, tau_hat + z_crit * se_hat)

    return (float(min(not_rejected)), float(max(not_rejected)))


def rdwinselect(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    wmin: Optional[float] = None,
    wstep: Optional[float] = None,
    nwindows: int = 10,
    statistic: str = 'diffmeans',
    p: int = 0,
    seed: int = 42,
    alpha: float = 0.15,
) -> pd.DataFrame:
    """
    Data-driven window selection for local randomization RD.

    Tests covariate balance at successively larger windows around the
    cutoff. The recommended window is the largest for which all
    covariates remain balanced (p > alpha).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    covs : list of str, optional
        Covariate names to test balance for. If None, uses quantiles of
        the running variable as pseudo-covariates.
    wmin : float, optional
        Minimum half-window width. Defaults to the smallest gap between
        adjacent observations near the cutoff.
    wstep : float, optional
        Window increment. Defaults to ``(max_range - wmin) / nwindows``.
    nwindows : int, default 10
        Number of windows to evaluate.
    statistic : str, default 'diffmeans'
        Test statistic for balance testing.
    p : int, default 0
        Polynomial order for adjustment.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.15
        Significance level for balance (lenient by default to be
        conservative about window selection).

    Returns
    -------
    pd.DataFrame
        Columns: window_left, window_right, n_left, n_right, p_value,
        balanced. Rows sorted by window width.
    """
    rng = np.random.default_rng(seed)
    xv = data[x].values.astype(float)

    # --- determine window grid ---
    x_left = xv[xv < c]
    x_right = xv[xv >= c]
    if len(x_left) == 0 or len(x_right) == 0:
        raise ValueError("Need observations on both sides of the cutoff.")

    # Max range: distance to closest boundary
    max_left = c - x_left.min()
    max_right = x_right.max() - c
    max_range = min(max_left, max_right)

    if wmin is None:
        # Smallest gap near cutoff
        sorted_x = np.sort(xv)
        gaps = np.diff(sorted_x)
        near_cutoff = (sorted_x[:-1] >= c - max_range / 2) & (
            sorted_x[:-1] <= c + max_range / 2
        )
        if near_cutoff.any():
            wmin = float(np.median(gaps[near_cutoff]))
        else:
            wmin = float(np.median(gaps[gaps > 0]))
        wmin = max(wmin, max_range / (nwindows * 2))

    if wstep is None:
        wstep = (max_range - wmin) / max(nwindows - 1, 1)
        wstep = max(wstep, wmin)

    # --- pseudo-covariates if none given ---
    use_covs = covs
    if use_covs is None or len(use_covs) == 0:
        # Create quantile dummies of X as pseudo-covariates
        qs = [0.25, 0.5, 0.75]
        pseudo_names = []
        for q in qs:
            cname = f'_x_q{int(q*100)}'
            data = data.copy()
            data[cname] = (data[x] <= np.quantile(xv, q)).astype(float)
            pseudo_names.append(cname)
        use_covs = pseudo_names

    # --- evaluate each window ---
    rows = []
    for i in range(nwindows):
        w = wmin + i * wstep
        wl = -w
        wr = w

        mask = (xv >= c + wl) & (xv <= c + wr)
        df_w = data.loc[mask]
        n = len(df_w)
        xw = df_w[x].values.astype(float)
        z = (xw >= c).astype(int)
        n_left = int((z == 0).sum())
        n_right = int((z == 1).sum())

        if n_left < 2 or n_right < 2:
            rows.append({
                'window_left': c + wl,
                'window_right': c + wr,
                'n_left': n_left,
                'n_right': n_right,
                'p_value': np.nan,
                'balanced': False,
            })
            continue

        # Test balance for each covariate, take minimum p-value
        min_pval = 1.0
        for cv in use_covs:
            if cv not in df_w.columns:
                continue
            cv_vals = df_w[cv].values.astype(float)
            if np.std(cv_vals) < 1e-14:
                continue

            # Polynomial adjustment
            if p > 0:
                cv_vals = _polynomial_residuals(cv_vals, xw - c, p)

            _, perm_pval = _permutation_pvalue(
                cv_vals, z, statistic, 500, rng
            )
            min_pval = min(min_pval, perm_pval)

        rows.append({
            'window_left': c + wl,
            'window_right': c + wr,
            'n_left': n_left,
            'n_right': n_right,
            'p_value': min_pval,
            'balanced': min_pval > alpha,
        })

    result = pd.DataFrame(rows)

    # Clean up pseudo-covariates
    if covs is None:
        for cname in pseudo_names:
            if cname in data.columns:
                data.drop(columns=[cname], inplace=True, errors='ignore')

    return result


def rdsensitivity(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    wlist: Optional[List[float]] = None,
    nwindows: int = 20,
    statistic: str = 'diffmeans',
    p: int = 0,
    n_perms: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Sensitivity of RD estimates across different window widths.

    For each window, runs ``rdrandinf`` and records the estimate, standard
    error, and p-value. Optionally produces a plot if matplotlib is
    available.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    wlist : list of float, optional
        Symmetric half-window widths to evaluate. If None, an
        evenly-spaced grid is generated automatically.
    nwindows : int, default 20
        Number of windows when ``wlist`` is None.
    statistic : str, default 'diffmeans'
        Test statistic for inference.
    p : int, default 0
        Polynomial order for adjustment.
    n_perms : int, default 500
        Number of permutations per window.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Columns: window, estimate, se, pvalue, ci_lower, ci_upper,
        significant.
    """
    xv = data[x].values.astype(float)

    if wlist is None:
        x_left = xv[xv < c]
        x_right = xv[xv >= c]
        max_left = c - x_left.min() if len(x_left) > 0 else 1.0
        max_right = x_right.max() - c if len(x_right) > 0 else 1.0
        max_w = min(max_left, max_right)
        # Start from a small window; ensure enough obs
        sorted_gaps = np.sort(np.abs(xv - c))
        # Need at least 4 obs, so start from the 4th closest
        min_w = sorted_gaps[min(3, len(sorted_gaps) - 1)] * 1.1
        min_w = max(min_w, max_w / (nwindows * 2))
        wlist = np.linspace(min_w, max_w * 0.95, nwindows).tolist()

    rows = []
    for w in wlist:
        wl = -w
        wr = w
        mask = (xv >= c + wl) & (xv <= c + wr)
        n_in = int(mask.sum())
        z_in = (xv[mask] >= c).astype(int)
        n_left = int((z_in == 0).sum())
        n_right = int((z_in == 1).sum())

        if n_left < 2 or n_right < 2:
            rows.append({
                'window': w,
                'estimate': np.nan,
                'se': np.nan,
                'pvalue': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'significant': False,
            })
            continue

        try:
            res = rdrandinf(
                data, y, x, c=c, wl=wl, wr=wr,
                statistic=statistic, p=p, n_perms=n_perms,
                alpha=alpha, seed=seed,
            )
            rows.append({
                'window': w,
                'estimate': res.estimate,
                'se': res.se,
                'pvalue': res.pvalue,
                'ci_lower': res.ci[0],
                'ci_upper': res.ci[1],
                'significant': res.pvalue <= alpha,
            })
        except (ValueError, RuntimeError):
            rows.append({
                'window': w,
                'estimate': np.nan,
                'se': np.nan,
                'pvalue': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'significant': False,
            })

    result = pd.DataFrame(rows)

    # Auto-plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        valid = result.dropna(subset=['estimate'])
        if len(valid) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Panel 1: Estimates with CI
            ax = axes[0]
            ax.plot(valid['window'], valid['estimate'], 'o-', color='#2c3e50')
            ax.fill_between(
                valid['window'], valid['ci_lower'], valid['ci_upper'],
                alpha=0.2, color='#3498db',
            )
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Window half-width')
            ax.set_ylabel('Treatment effect estimate')
            ax.set_title('Sensitivity: Estimates across windows')

            # Panel 2: p-values
            ax = axes[1]
            ax.plot(valid['window'], valid['pvalue'], 'o-', color='#e74c3c')
            ax.axhline(alpha, color='grey', linestyle='--', linewidth=0.8,
                        label=f'alpha = {alpha}')
            ax.set_xlabel('Window half-width')
            ax.set_ylabel('Permutation p-value')
            ax.set_title('Sensitivity: P-values across windows')
            ax.legend()

            plt.tight_layout()
            plt.show()
    except ImportError:
        pass

    return result


def rdrbounds(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    wl: Optional[float] = None,
    wr: Optional[float] = None,
    gamma_list: Optional[List[float]] = None,
    statistic: str = 'ranksum',
    n_perms: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rosenbaum sensitivity bounds for RD under local randomization.

    Assesses how much hidden bias (departure from random assignment)
    would be needed to explain away the estimated treatment effect.
    For each gamma (odds ratio), computes upper and lower p-value
    bounds under worst-case confounding.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    wl : float, optional
        Window left bound offset (typically negative).
    wr : float, optional
        Window right bound offset (typically positive).
    gamma_list : list of float, optional
        Odds ratios to evaluate. Defaults to [1, 1.5, 2, 2.5, 3, 4, 5].
        gamma=1 is pure randomization; gamma>1 allows confounding.
    statistic : str, default 'ranksum'
        Test statistic (ranksum is standard for Rosenbaum bounds).
    n_perms : int, default 1000
        Number of permutations for p-value computation.
    seed : int, default 42
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: gamma, pvalue_upper, pvalue_lower.

    Notes
    -----
    Under Rosenbaum's model, if gamma=Gamma, the probability that unit i
    is treated satisfies:

        1/(1+Gamma) <= P(D_i=1) <= Gamma/(1+Gamma)

    instead of the uniform 1/2 under pure randomization. The bounds on the
    p-value are obtained by computing the worst-case assignment
    probabilities at each gamma level.
    """
    rng = np.random.default_rng(seed)

    if gamma_list is None:
        gamma_list = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # Subset to window
    mask = _select_window(data, x, c, wl, wr)
    df_w = data.loc[mask].copy()
    n_obs = len(df_w)
    if n_obs < 4:
        raise ValueError(f"Only {n_obs} observations in window.")

    yv = df_w[y].values.astype(float)
    xv = df_w[x].values.astype(float)
    z = (xv >= c).astype(int)
    n_t = int(z.sum())
    n_c = n_obs - n_t

    if n_t < 2 or n_c < 2:
        raise ValueError("Need >= 2 observations on each side of cutoff.")

    # Observed test statistic
    obs_stat = _compute_stat(yv, z, statistic)

    # Rank outcomes for Rosenbaum bounds
    ranks = sp_stats.rankdata(yv)

    rows = []
    for gamma in gamma_list:
        if gamma < 1.0:
            raise ValueError("gamma must be >= 1.")

        if abs(gamma - 1.0) < 1e-14:
            # Pure randomization: uniform permutation
            _, pval = _permutation_pvalue(yv, z, statistic, n_perms, rng)
            rows.append({
                'gamma': gamma,
                'pvalue_upper': pval,
                'pvalue_lower': pval,
            })
            continue

        # Under Rosenbaum's model with odds ratio Gamma:
        # Worst-case assignment probabilities depend on outcome ranks.
        # Upper bound: high-rank units more likely treated
        # Lower bound: low-rank units more likely treated

        # Assignment probabilities proportional to gamma^(rank indicator)
        # Upper: p_i proportional to gamma^1 if rank(y_i) is high
        # Lower: p_i proportional to gamma^1 if rank(y_i) is low

        count_upper = 0
        count_lower = 0

        for _ in range(n_perms):
            # Upper bound: bias units with higher outcomes towards treatment
            weights_upper = np.where(
                ranks > np.median(ranks),
                gamma / (1.0 + gamma),
                1.0 / (1.0 + gamma),
            )
            # Normalise to produce valid sampling weights
            weights_upper = weights_upper / weights_upper.sum()
            # Sample n_t treated units with these weights (without replacement)
            idx_upper = rng.choice(
                n_obs, size=n_t, replace=False, p=weights_upper
            )
            z_upper = np.zeros(n_obs, dtype=int)
            z_upper[idx_upper] = 1
            stat_upper = _compute_stat(yv, z_upper, statistic)
            if abs(stat_upper) >= abs(obs_stat) - 1e-14:
                count_upper += 1

            # Lower bound: bias units with lower outcomes towards treatment
            weights_lower = np.where(
                ranks <= np.median(ranks),
                gamma / (1.0 + gamma),
                1.0 / (1.0 + gamma),
            )
            weights_lower = weights_lower / weights_lower.sum()
            idx_lower = rng.choice(
                n_obs, size=n_t, replace=False, p=weights_lower
            )
            z_lower = np.zeros(n_obs, dtype=int)
            z_lower[idx_lower] = 1
            stat_lower = _compute_stat(yv, z_lower, statistic)
            if abs(stat_lower) >= abs(obs_stat) - 1e-14:
                count_lower += 1

        rows.append({
            'gamma': gamma,
            'pvalue_upper': count_upper / n_perms,
            'pvalue_lower': count_lower / n_perms,
        })

    return pd.DataFrame(rows)
