"""
Panel data diagnostic tests.

Provides helper functions used by PanelResults methods:
- Hausman (1978) test: FE vs RE
- Breusch-Pagan (1980) LM test: Pooled OLS vs RE
- F-test for joint significance of entity effects
- Pesaran (2004) CD test for cross-sectional dependence

These are called internally by PanelResults methods and can also
be used standalone via ``sp.hausman_test()``, ``sp.bp_lm_test()``, etc.

References
----------
Hausman, J.A. (1978). "Specification Tests in Econometrics."
Breusch, T.S. and Pagan, A.R. (1980). "The Lagrange Multiplier Test."
Pesaran, M.H. (2004). "General Diagnostic Tests for Cross Section
    Dependence in Panels." [@hausman1978specification]
"""

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ======================================================================
# Hausman test: FE vs RE
# ======================================================================

def _hausman_from_data(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    time_col: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Hausman (1978) specification test.

    H0: RE is consistent and efficient (use RE).
    H1: RE is inconsistent (use FE).
    """
    df = data[[id_col, time_col, y] + x].dropna()
    k = len(x)

    beta_fe, vcov_fe = _within_estimator(df, y, x, id_col)
    beta_re, vcov_re = _re_estimator(df, y, x, id_col)

    b_diff = beta_fe - beta_re
    V_diff = vcov_fe - vcov_re

    try:
        V_inv = np.linalg.inv(V_diff)
        H = float(b_diff @ V_inv @ b_diff)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V_diff)
        H = float(max(b_diff @ V_inv @ b_diff, 0))

    H = max(H, 0)
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


# ======================================================================
# Breusch-Pagan LM test: Pooled OLS vs RE
# ======================================================================

def _bp_lm_test(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    time_col: str,
) -> Dict[str, Any]:
    """
    Breusch-Pagan (1980) Lagrange Multiplier test.

    H0: Var(alpha_i) = 0 (Pooled OLS appropriate).
    H1: Var(alpha_i) > 0 (Random Effects needed).

    The test statistic is:

        LM = nT/(2(T-1)) * [sum_i (sum_t e_it)^2 / sum_i sum_t e_it^2 - 1]^2

    Under H0, LM ~ chi2(1).
    """
    df = data[[id_col, time_col, y] + x].dropna()

    # Pooled OLS residuals
    Y = df[y].values.astype(float)
    X = np.column_stack([np.ones(len(df)), df[x].values.astype(float)])

    try:
        beta = np.linalg.solve(X.T @ X, X.T @ Y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond=None)[0]

    e = Y - X @ beta

    # Group residuals by entity
    ids = df[id_col].values
    unique_ids = np.unique(ids)
    N = len(unique_ids)
    nT = len(e)

    # Average T per unit
    T_counts = np.array([np.sum(ids == uid) for uid in unique_ids])
    T_bar = np.mean(T_counts)

    # Sum of squared group-summed residuals
    sum_ei_sq = 0.0
    for uid in unique_ids:
        mask = ids == uid
        sum_ei_sq += (e[mask].sum()) ** 2

    # Total sum of squared residuals
    total_sq = np.sum(e ** 2)

    # LM statistic (Honda 1985 variant for unbalanced panels)
    ratio = sum_ei_sq / total_sq - 1
    LM = (nT / (2 * (T_bar - 1))) * ratio ** 2

    LM = max(LM, 0)
    pvalue = float(1 - stats.chi2.cdf(LM, 1))

    return {
        'statistic': LM,
        'df': 1,
        'pvalue': pvalue,
        'recommendation': 'RE' if pvalue < 0.05 else 'Pooled OLS',
        'interpretation': (
            f"LM = {LM:.4f}, p = {pvalue:.4f}. "
            f"{'Reject H0: use Random Effects.' if pvalue < 0.05 else 'Cannot reject H0: Pooled OLS is adequate.'}"
        ),
    }


# ======================================================================
# F-test for entity fixed effects
# ======================================================================

def _f_test_effects(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    time_col: str,
) -> Dict[str, Any]:
    """
    F-test for joint significance of entity fixed effects.

    H0: all alpha_i = 0.
    """
    df = data[[id_col, time_col, y] + x].dropna()
    n = len(df)
    k = len(x)

    Y = df[y].values.astype(float)
    X = np.column_stack([np.ones(n), df[x].values.astype(float)])

    # Restricted model: Pooled OLS
    try:
        beta_r = np.linalg.solve(X.T @ X, X.T @ Y)
    except np.linalg.LinAlgError:
        beta_r = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond=None)[0]
    rss_r = np.sum((Y - X @ beta_r) ** 2)

    # Unrestricted model: FE (within estimator)
    ids = df[id_col].values
    unique_ids = np.unique(ids)
    N = len(unique_ids)

    Y_dm = Y.copy()
    X_dm = df[x].values.astype(float).copy()
    for uid in unique_ids:
        mask = ids == uid
        Y_dm[mask] -= Y_dm[mask].mean()
        X_dm[mask] -= X_dm[mask].mean(axis=0)

    try:
        beta_u = np.linalg.solve(X_dm.T @ X_dm, X_dm.T @ Y_dm)
    except np.linalg.LinAlgError:
        beta_u = np.linalg.lstsq(X_dm.T @ X_dm, X_dm.T @ Y_dm, rcond=None)[0]
    rss_u = np.sum((Y_dm - X_dm @ beta_u) ** 2)

    # F-statistic
    df1 = N - 1
    df2 = n - N - k
    if df2 <= 0 or rss_u <= 0:
        return {
            'statistic': np.nan, 'df1': df1, 'df2': df2,
            'pvalue': np.nan, 'interpretation': 'Insufficient degrees of freedom.',
        }

    F = ((rss_r - rss_u) / df1) / (rss_u / df2)
    F = max(F, 0)
    pvalue = float(1 - stats.f.cdf(F, df1, df2))

    return {
        'statistic': F,
        'df1': df1,
        'df2': df2,
        'pvalue': pvalue,
        'interpretation': (
            f"F({df1}, {df2}) = {F:.4f}, p = {pvalue:.4f}. "
            f"{'Reject H0: entity effects are significant — use FE.' if pvalue < 0.05 else 'Cannot reject H0: entity effects not significant.'}"
        ),
    }


# ======================================================================
# Pesaran CD test for cross-sectional dependence
# ======================================================================

def _pesaran_cd(
    resids: pd.Series,
    entity_col: str,
    time_col: str,
    data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Pesaran (2004) CD test for cross-sectional dependence.

    H0: no cross-sectional dependence.
    """
    ids = data[entity_col].values
    unique_ids = np.unique(ids)
    N = len(unique_ids)

    # Reshape residuals to (T x N) panel
    resid_dict = {}
    for uid in unique_ids:
        mask = ids == uid
        resid_dict[uid] = pd.Series(
            resids.values[mask] if hasattr(resids, 'values') else resids[mask],
            index=data.loc[mask, time_col].values,
        )
    resid_panel = pd.DataFrame(resid_dict)

    # Pairwise correlations
    T_common = len(resid_panel.dropna())
    if T_common < 3 or N < 2:
        return {
            'statistic': np.nan,
            'pvalue': np.nan,
            'interpretation': 'Insufficient data for CD test.',
        }

    sum_rho = 0.0
    count = 0
    cols = resid_panel.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = resid_panel[[cols[i], cols[j]]].dropna()
            if len(pair) >= 3:
                T_ij = len(pair)
                rho_ij = pair.corr().iloc[0, 1]
                sum_rho += np.sqrt(T_ij) * rho_ij
                count += 1

    if count == 0:
        return {
            'statistic': np.nan,
            'pvalue': np.nan,
            'interpretation': 'No valid pairs for CD test.',
        }

    CD = np.sqrt(2.0 / (N * (N - 1))) * sum_rho
    pvalue = float(2 * (1 - stats.norm.cdf(abs(CD))))

    return {
        'statistic': float(CD),
        'pvalue': pvalue,
        'interpretation': (
            f"CD = {CD:.4f}, p = {pvalue:.4f}. "
            f"{'Reject H0: cross-sectional dependence detected.' if pvalue < 0.05 else 'Cannot reject H0: no cross-sectional dependence.'}"
        ),
    }


# ======================================================================
# Internal helpers (from diagnostics/hausman.py)
# ======================================================================

def _within_estimator(df, y, x, id_col):
    """Fixed Effects (within transformation)."""
    n = len(df)
    k = len(x)

    Y = df[y].values.astype(float)
    X = df[x].values.astype(float)
    ids = df[id_col].values

    Y_dm = Y.copy()
    X_dm = X.copy()
    for uid in np.unique(ids):
        mask = ids == uid
        Y_dm[mask] -= Y_dm[mask].mean()
        X_dm[mask] -= X_dm[mask].mean(axis=0)

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

    beta_fe, _ = _within_estimator(df, y, x, id_col)
    Y_dm = Y.copy()
    X_fe = df[x].values.astype(float)
    X_dm = X_fe.copy()
    for uid in unique_ids:
        mask = ids == uid
        Y_dm[mask] -= Y_dm[mask].mean()
        X_dm[mask] -= X_dm[mask].mean(axis=0)
    resid_fe = Y_dm - X_dm @ beta_fe

    T_bar = n / N
    sigma2_e = np.sum(resid_fe ** 2) / (n - N - k)

    group_means_y = np.array([Y[ids == uid].mean() for uid in unique_ids])
    group_means_x = np.column_stack(
        [np.ones(N)] + [np.array([df[v].values[ids == uid].mean()
                                   for uid in unique_ids])
                         for v in x])
    beta_between = np.linalg.lstsq(group_means_x, group_means_y, rcond=None)[0]
    resid_between = group_means_y - group_means_x @ beta_between
    sigma2_b = max(np.var(resid_between) - sigma2_e / T_bar, 0)

    theta = 1 - np.sqrt(sigma2_e / (T_bar * sigma2_b + sigma2_e)) if sigma2_b > 0 else 0

    Y_gls = Y.copy()
    X_gls = X.copy()
    for uid in unique_ids:
        mask = ids == uid
        Y_gls[mask] -= theta * Y_gls[mask].mean()
        X_gls[mask] -= theta * X_gls[mask].mean(axis=0)

    XtX = X_gls.T @ X_gls
    XtY = X_gls.T @ Y_gls
    try:
        beta_full = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        beta_full = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    resid_gls = Y_gls - X_gls @ beta_full
    sigma2_gls = np.sum(resid_gls ** 2) / (n - k - 1)
    vcov_full = sigma2_gls * np.linalg.pinv(XtX)

    beta_re = beta_full[1:]
    vcov_re = vcov_full[1:, 1:]

    return beta_re, vcov_re
