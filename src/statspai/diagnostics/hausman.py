"""
Hausman (1978) specification test.

Tests whether the fixed effects (FE) or random effects (RE) estimator
is appropriate for panel data. Under H0 (RE is consistent), both FE
and RE are consistent but RE is more efficient. Under H1, only FE
is consistent.

References
----------
Hausman, J.A. (1978).
"Specification Tests in Econometrics."
*Econometrica*, 46(6), 1251-1271.
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def hausman_test(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str,
    time: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Hausman test for FE vs RE in panel data.

    Equivalent to Stata's ``hausman fe re``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Dependent variable.
    x : list of str
        Independent variables (time-varying).
    id : str
        Unit identifier.
    time : str
        Time period identifier.
    alpha : float, default 0.05

    Returns
    -------
    dict
        ``'statistic'``: Hausman chi² statistic
        ``'df'``: degrees of freedom
        ``'pvalue'``: p-value
        ``'recommendation'``: 'FE' or 'RE'
        ``'beta_fe'``, ``'beta_re'``: coefficient vectors

    Examples
    --------
    >>> result = sp.hausman_test(df, y='wage', x=['education', 'experience'],
    ...                         id='worker', time='year')
    >>> print(f"chi2({result['df']}) = {result['statistic']:.2f}, "
    ...       f"p = {result['pvalue']:.4f}")
    >>> print(f"Recommendation: {result['recommendation']}")

    Notes
    -----
    The test statistic is:

    .. math::
        H = (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE})'
        [V(\\hat{\\beta}_{FE}) - V(\\hat{\\beta}_{RE})]^{-1}
        (\\hat{\\beta}_{FE} - \\hat{\\beta}_{RE})

    Under H0, H ~ χ²(k).

    - Reject H0 (p < 0.05) → Use Fixed Effects
    - Fail to reject → Use Random Effects (more efficient)

    See Hausman (1978, *Econometrica*).
    """
    df = data[[id, time, y] + x].dropna()
    k = len(x)

    # --- Fixed Effects (within estimator) ---
    beta_fe, vcov_fe = _within_estimator(df, y, x, id)

    # --- Random Effects (GLS estimator) ---
    beta_re, vcov_re = _re_estimator(df, y, x, id)

    # --- Hausman statistic ---
    b_diff = beta_fe - beta_re
    V_diff = vcov_fe - vcov_re

    # Ensure positive definite (regularize if needed)
    try:
        V_inv = np.linalg.inv(V_diff)
        H = float(b_diff @ V_inv @ b_diff)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V_diff)
        H = float(max(b_diff @ V_inv @ b_diff, 0))

    H = max(H, 0)  # chi² must be non-negative
    pvalue = float(1 - stats.chi2.cdf(H, k))

    recommendation = 'FE' if pvalue < alpha else 'RE'

    return {
        'statistic': H,
        'df': k,
        'pvalue': pvalue,
        'recommendation': recommendation,
        'beta_fe': pd.Series(beta_fe, index=x),
        'beta_re': pd.Series(beta_re, index=x),
        'interpretation': (
            f"chi2({k}) = {H:.4f}, p = {pvalue:.4f}. "
            f"{'Reject H0: use Fixed Effects.' if pvalue < alpha else 'Cannot reject H0: Random Effects is more efficient.'}"
        ),
    }


def _within_estimator(df, y, x, id_col):
    """Fixed Effects (within transformation)."""
    n = len(df)
    k = len(x)

    Y = df[y].values.astype(float)
    X = df[x].values.astype(float)
    ids = df[id_col].values

    # Demean within groups
    Y_dm = Y.copy()
    X_dm = X.copy()
    for uid in np.unique(ids):
        mask = ids == uid
        Y_dm[mask] -= Y_dm[mask].mean()
        X_dm[mask] -= X_dm[mask].mean(axis=0)

    # OLS on demeaned data
    XtX = X_dm.T @ X_dm
    XtY = X_dm.T @ Y_dm
    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    resid = Y_dm - X_dm @ beta
    n_groups = len(np.unique(ids))
    sigma2 = np.sum(resid ** 2) / (n - n_groups - k)
    vcov = sigma2 * np.linalg.pinv(XtX)

    return beta, vcov


def _re_estimator(df, y, x, id_col):
    """Random Effects (GLS with estimated variance components)."""
    n = len(df)
    k = len(x)
    ids = df[id_col].values
    unique_ids = np.unique(ids)
    N = len(unique_ids)

    Y = df[y].values.astype(float)
    X = np.column_stack([np.ones(n), df[x].values.astype(float)])

    # Get FE residuals for variance decomposition
    beta_fe, _ = _within_estimator(df, y, x, id_col)
    Y_dm = Y.copy()
    X_fe = df[x].values.astype(float)
    X_dm = X_fe.copy()
    for uid in unique_ids:
        mask = ids == uid
        Y_dm[mask] -= Y_dm[mask].mean()
        X_dm[mask] -= X_dm[mask].mean(axis=0)
    resid_fe = Y_dm - X_dm @ beta_fe

    # Variance components
    T_bar = n / N  # average T
    sigma2_e = np.sum(resid_fe ** 2) / (n - N - k)

    # Between estimator residuals
    group_means_y = np.array([Y[ids == uid].mean() for uid in unique_ids])
    group_means_x = np.column_stack(
        [np.ones(N)] + [np.array([df[v].values[ids == uid].mean()
                                   for uid in unique_ids])
                         for v in x])
    beta_between = np.linalg.lstsq(group_means_x, group_means_y, rcond=None)[0]
    resid_between = group_means_y - group_means_x @ beta_between
    sigma2_b = max(np.var(resid_between) - sigma2_e / T_bar, 0)

    # GLS transformation: θ = 1 - sqrt(σ²_e / (T*σ²_α + σ²_e))
    theta = 1 - np.sqrt(sigma2_e / (T_bar * sigma2_b + sigma2_e)) if sigma2_b > 0 else 0

    # Quasi-demean
    Y_gls = Y.copy()
    X_gls = X.copy()
    for uid in unique_ids:
        mask = ids == uid
        Y_gls[mask] -= theta * Y_gls[mask].mean()
        X_gls[mask] -= theta * X_gls[mask].mean(axis=0)

    # GLS OLS
    XtX = X_gls.T @ X_gls
    XtY = X_gls.T @ Y_gls
    try:
        beta_full = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        beta_full = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    resid_gls = Y_gls - X_gls @ beta_full
    sigma2_gls = np.sum(resid_gls ** 2) / (n - k - 1)
    vcov_full = sigma2_gls * np.linalg.pinv(XtX)

    # Return only slopes (exclude constant)
    beta_re = beta_full[1:]
    vcov_re = vcov_full[1:, 1:]

    return beta_re, vcov_re


# Citation
CausalResult._CITATIONS['hausman'] = (
    "@article{hausman1978specification,\n"
    "  title={Specification Tests in Econometrics},\n"
    "  author={Hausman, Jerry A.},\n"
    "  journal={Econometrica},\n"
    "  volume={46},\n"
    "  number={6},\n"
    "  pages={1251--1271},\n"
    "  year={1978},\n"
    "  publisher={Wiley}\n"
    "}"
)
