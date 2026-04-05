"""
Jackknife Inference for Small Clusters.

When the number of clusters is small (< 50), standard cluster-robust
standard errors are unreliable. This module provides:

1. Cluster jackknife SE (leave-one-cluster-out variance estimator)
2. CR2/CR3 bias-corrected cluster SEs (Bell & McCaffrey 2002)
3. Wild cluster bootstrap t-test with improved finite-sample inference

References
----------
Bell, R.M. and McCaffrey, D.F. (2002).
"Bias Reduction in Standard Errors for Linear Regression with
Multi-Stage Samples." *Survey Methodology*, 28(2), 169-181.

MacKinnon, J.G. and Webb, M.D. (2017).
"Wild Bootstrap Inference for Wildly Different Cluster Sizes."
*Journal of Applied Econometrics*, 32(2), 233-254.

Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2008).
"Bootstrap-Based Improvements for Inference with Clustered Errors."
*Review of Economics and Statistics*, 90(3), 414-427.
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults


def jackknife_se(
    result: EconometricResults,
    data: pd.DataFrame,
    cluster: str,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Leave-one-cluster-out jackknife standard errors.

    Computes cluster jackknife variance by re-estimating the model
    G times, each time dropping one cluster. Valid when G is small
    and conventional cluster-robust SEs are unreliable.

    Parameters
    ----------
    result : EconometricResults
        A fitted regression result from ``sp.regress()``.
    data : pd.DataFrame
        The original data used for estimation.
    cluster : str
        Name of the cluster variable in ``data``.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        New results object with jackknife standard errors,
        t-statistics based on effective DoF, and adjusted p-values.

    Examples
    --------
    >>> result = sp.regress("y ~ x1 + x2", data=df, cluster="state")
    >>> jk = sp.jackknife_se(result, data=df, cluster="state")
    >>> print(jk.summary())

    Notes
    -----
    The cluster jackknife variance is:

        V_jk = ((G-1)/G) * sum_g (theta_g - theta_bar)^2

    where theta_g is the estimate from dropping cluster g, and
    theta_bar is the mean of the leave-one-out estimates.

    The effective degrees of freedom are G - 1, which gives
    wider confidence intervals than asymptotic cluster-robust SEs.
    """
    # Extract formula components from the result
    formula = result.model_info.get('formula', '')
    model_type = result.model_info.get('model_type', 'OLS')

    # Parse the formula to get y and X variable names
    y_var, x_vars = _parse_formula(formula, result)

    # Prepare data
    cols = [y_var] + x_vars + [cluster]
    cols = [c for c in cols if c in data.columns]
    df = data[cols].dropna()

    Y = df[y_var].values.astype(float)
    X_names = [v for v in x_vars if v in df.columns]
    X = np.column_stack([np.ones(len(df)), df[X_names].values.astype(float)])
    var_names = ['_const'] + X_names

    cl = df[cluster].values
    unique_cl = np.unique(cl)
    G = len(unique_cl)
    n = len(Y)
    k = X.shape[1]

    # Full-sample OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y

    # Leave-one-cluster-out estimates
    beta_loo = np.zeros((G, k))
    for g_idx, g_val in enumerate(unique_cl):
        mask = cl != g_val
        X_g = X[mask]
        Y_g = Y[mask]
        try:
            beta_loo[g_idx] = np.linalg.inv(X_g.T @ X_g) @ X_g.T @ Y_g
        except np.linalg.LinAlgError:
            beta_loo[g_idx] = np.linalg.lstsq(X_g, Y_g, rcond=None)[0]

    # Jackknife variance: V_jk = ((G-1)/G) * sum (beta_g - beta_bar)^2
    beta_bar = beta_loo.mean(axis=0)
    deviations = beta_loo - beta_bar[np.newaxis, :]
    V_jk = ((G - 1) / G) * (deviations.T @ deviations)

    jk_se = np.sqrt(np.diag(V_jk))

    # Map back to original parameter names
    param_names = list(result.params.index)
    se_series = pd.Series(np.nan, index=param_names)
    params_series = result.params.copy()

    for i, vn in enumerate(var_names):
        # Match variable name (handle _const vs Intercept)
        matched = None
        if vn == '_const':
            for pn in param_names:
                if pn.lower() in ('intercept', '_const', 'const', '(intercept)'):
                    matched = pn
                    break
        else:
            if vn in param_names:
                matched = vn
        if matched is not None:
            se_series[matched] = jk_se[i]

    # Effective DoF = G - 1
    df_resid = G - 1

    # Build new result
    model_info = dict(result.model_info)
    model_info['se_type'] = 'cluster jackknife'
    model_info['n_clusters'] = G
    model_info['jackknife_dof'] = df_resid

    data_info = dict(result.data_info)
    data_info['df_resid'] = df_resid

    diagnostics = dict(result.diagnostics)
    diagnostics['n_clusters'] = G
    diagnostics['effective_dof'] = df_resid

    return EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def cr2_se(
    result: EconometricResults,
    data: pd.DataFrame,
    cluster: str,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    CR2 bias-corrected cluster-robust standard errors (Bell & McCaffrey 2002).

    Applies a bias correction to the cluster-robust variance estimator
    that accounts for leverage and improves finite-sample performance.
    The correction factor uses the inverse square root of (I - H_gg),
    where H_gg is the within-cluster hat matrix block.

    Parameters
    ----------
    result : EconometricResults
        A fitted regression result from ``sp.regress()``.
    data : pd.DataFrame
        The original data used for estimation.
    cluster : str
        Name of the cluster variable in ``data``.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        New results object with CR2-corrected standard errors.

    Notes
    -----
    The CR2 estimator replaces the raw residuals e_g with
    adjusted residuals:

        e*_g = (I - H_gg)^{-1/2} e_g

    where H_gg = X_g (X'X)^{-1} X_g' is the within-cluster block
    of the hat matrix. This removes the downward bias in the
    conventional cluster-robust variance estimator.

    See Bell & McCaffrey (2002) and Pustejovsky & Tipton (2018).
    """
    y_var, x_vars = _parse_formula(result.model_info.get('formula', ''), result)

    cols = [y_var] + x_vars + [cluster]
    cols = [c for c in cols if c in data.columns]
    df = data[cols].dropna()

    Y = df[y_var].values.astype(float)
    X_names = [v for v in x_vars if v in df.columns]
    X = np.column_stack([np.ones(len(df)), df[X_names].values.astype(float)])
    var_names = ['_const'] + X_names

    cl = df[cluster].values
    unique_cl = np.unique(cl)
    G = len(unique_cl)
    n = len(Y)
    k = X.shape[1]

    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y
    resid = Y - X @ beta_hat

    # CR2: apply (I - H_gg)^{-1/2} correction to cluster residuals
    meat = np.zeros((k, k))
    for g_val in unique_cl:
        idx = cl == g_val
        X_g = X[idx]
        e_g = resid[idx]
        n_g = idx.sum()

        # H_gg = X_g (X'X)^{-1} X_g'
        H_gg = X_g @ XtX_inv @ X_g.T
        I_H = np.eye(n_g) - H_gg

        # (I - H_gg)^{-1/2} via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(I_H)
        eigvals = np.maximum(eigvals, 1e-12)  # numerical stability
        I_H_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Adjusted residuals
        e_adj = I_H_inv_sqrt @ e_g
        score_g = X_g.T @ e_adj
        meat += np.outer(score_g, score_g)

    V_cr2 = XtX_inv @ meat @ XtX_inv
    cr2_ses = np.sqrt(np.diag(V_cr2))

    # Satterthwaite DoF approximation
    dof = _satterthwaite_dof(X, cl, unique_cl, XtX_inv, k)

    # Map to parameter names
    param_names = list(result.params.index)
    se_series = pd.Series(np.nan, index=param_names)
    dof_series = pd.Series(np.nan, index=param_names)

    for i, vn in enumerate(var_names):
        matched = _match_varname(vn, param_names)
        if matched is not None:
            se_series[matched] = cr2_ses[i]
            if i < len(dof):
                dof_series[matched] = dof[i]

    model_info = dict(result.model_info)
    model_info['se_type'] = 'CR2 (Bell-McCaffrey)'
    model_info['n_clusters'] = G

    data_info = dict(result.data_info)
    # Use minimum Satterthwaite DoF across coefficients
    min_dof = float(np.nanmin(dof)) if len(dof) > 0 else G - 1
    data_info['df_resid'] = min_dof

    diagnostics = dict(result.diagnostics)
    diagnostics['n_clusters'] = G
    diagnostics['satterthwaite_dof'] = {var_names[i]: float(dof[i]) for i in range(len(dof))}

    return EconometricResults(
        params=result.params.copy(),
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def wild_cluster_boot(
    result: EconometricResults,
    data: pd.DataFrame,
    cluster: str,
    variable: str,
    n_boot: int = 999,
    weight_type: str = 'rademacher',
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Wild cluster bootstrap t-test for a single coefficient.

    Implements the WCR (Wild Cluster Restricted) bootstrap from
    Cameron, Gelbach & Miller (2008), designed for inference with
    few clusters. Imposes the null hypothesis when generating
    bootstrap samples for better finite-sample performance.

    Parameters
    ----------
    result : EconometricResults
        A fitted regression result from ``sp.regress()``.
    data : pd.DataFrame
        The original data used for estimation.
    cluster : str
        Name of the cluster variable.
    variable : str
        Name of the coefficient to test (H0: beta = 0).
    n_boot : int, default 999
        Number of bootstrap replications (use odd number).
    weight_type : str, default 'rademacher'
        Bootstrap weight distribution:
        - ``'rademacher'``: +/-1 with equal probability.
        - ``'webb'``: Webb (2014) 6-point distribution for G < 12.
        - ``'mammen'``: Mammen (1993) 2-point distribution.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for confidence interval.

    Returns
    -------
    dict
        Keys:
        - ``beta_hat``: point estimate
        - ``se_cluster``: conventional cluster-robust SE
        - ``t_stat``: observed t-statistic
        - ``p_boot``: bootstrap p-value (two-sided)
        - ``ci_boot``: bootstrap percentile-t confidence interval
        - ``t_distribution``: array of bootstrap t-statistics
        - ``n_clusters``: number of clusters
        - ``n_boot``: number of replications

    Examples
    --------
    >>> result = sp.regress("y ~ x1 + x2", data=df, cluster="state")
    >>> boot = sp.wild_cluster_boot(result, data=df, cluster="state",
    ...                              variable="x1", n_boot=999)
    >>> print(f"Bootstrap p = {boot['p_boot']:.4f}")
    >>> print(f"Bootstrap CI = {boot['ci_boot']}")

    Notes
    -----
    The WCR bootstrap imposes H0: beta_variable = 0 when constructing
    bootstrap samples. Under few clusters, this yields much better
    size control than unrestricted wild bootstrap or cluster-robust SEs.

    For G < 12, use ``weight_type='webb'`` per Webb (2014).
    """
    import warnings

    rng = np.random.default_rng(seed)

    y_var, x_vars = _parse_formula(result.model_info.get('formula', ''), result)

    cols = [y_var] + x_vars + [cluster]
    cols = [c for c in cols if c in data.columns]
    df = data[cols].dropna()

    Y = df[y_var].values.astype(float)
    X_names = [v for v in x_vars if v in df.columns]
    X = np.column_stack([np.ones(len(df)), df[X_names].values.astype(float)])
    var_names = ['_const'] + X_names

    if variable not in var_names:
        raise ValueError(
            f"Variable '{variable}' not found. Available: {var_names}"
        )
    test_idx = var_names.index(variable)

    cl = df[cluster].values
    unique_cl = np.unique(cl)
    G = len(unique_cl)
    n = len(Y)
    k = X.shape[1]

    if G < 6:
        warnings.warn(
            f"Only {G} clusters. Bootstrap results may be unreliable "
            f"with fewer than 6 clusters.",
            UserWarning,
        )

    # Full-sample OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y
    resid = Y - X @ beta_hat
    beta_test = beta_hat[test_idx]

    # Cluster-robust SE (CR1)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    meat = np.zeros((k, k))
    for g_val in unique_cl:
        idx = cl == g_val
        score_g = X[idx].T @ resid[idx]
        meat += np.outer(score_g, score_g)
    vcov_cl = correction * XtX_inv @ meat @ XtX_inv
    se_cl = float(np.sqrt(vcov_cl[test_idx, test_idx]))
    t_stat = beta_test / se_cl if se_cl > 0 else 0.0

    # Restricted OLS (impose H0: beta_variable = 0)
    other_cols = [j for j in range(k) if j != test_idx]
    X_other = X[:, other_cols]
    beta_other = np.linalg.lstsq(X_other, Y, rcond=None)[0]
    beta_r = np.zeros(k)
    for i, j in enumerate(other_cols):
        beta_r[j] = beta_other[i]
    beta_r[test_idx] = 0.0
    resid_r = Y - X @ beta_r

    # Bootstrap
    t_boot = np.zeros(n_boot)
    for b in range(n_boot):
        w = _draw_weights(G, weight_type, rng)

        Y_star = np.zeros(n)
        for g_idx, g_val in enumerate(unique_cl):
            obs_idx = cl == g_val
            Y_star[obs_idx] = X[obs_idx] @ beta_r + w[g_idx] * resid_r[obs_idx]

        beta_b = XtX_inv @ X.T @ Y_star
        resid_b = Y_star - X @ beta_b

        meat_b = np.zeros((k, k))
        for g_val in unique_cl:
            idx = cl == g_val
            score_g = X[idx].T @ resid_b[idx]
            meat_b += np.outer(score_g, score_g)
        vcov_b = correction * XtX_inv @ meat_b @ XtX_inv
        se_b = np.sqrt(max(vcov_b[test_idx, test_idx], 1e-20))

        t_boot[b] = beta_b[test_idx] / se_b

    # Bootstrap p-value (two-sided)
    p_boot = float(np.mean(np.abs(t_boot) >= np.abs(t_stat)))

    # Percentile-t CI
    t_lower = np.percentile(t_boot, 100 * alpha / 2)
    t_upper = np.percentile(t_boot, 100 * (1 - alpha / 2))
    ci_boot = (
        beta_test - t_upper * se_cl,
        beta_test - t_lower * se_cl,
    )

    return {
        'beta_hat': float(beta_test),
        'se_cluster': se_cl,
        't_stat': float(t_stat),
        'p_boot': p_boot,
        'ci_boot': ci_boot,
        't_distribution': t_boot,
        'n_clusters': G,
        'n_obs': n,
        'n_boot': n_boot,
        'weight_type': weight_type,
    }


# ======================================================================
# Helper functions
# ======================================================================


def _parse_formula(
    formula: str,
    result: EconometricResults,
) -> Tuple[str, List[str]]:
    """Parse formula or extract variable names from result."""
    if formula and '~' in formula:
        lhs, rhs = formula.split('~', 1)
        y_var = lhs.strip()
        x_terms = [t.strip() for t in rhs.split('+')]
        x_vars = [t for t in x_terms if t and t != '1']
        return y_var, x_vars

    # Fallback: use result parameter names
    param_names = list(result.params.index)
    y_var = result.model_info.get('depvar', result.data_info.get('y_var', 'y'))
    x_vars = [p for p in param_names
              if p.lower() not in ('intercept', '_const', 'const', '(intercept)')]
    return y_var, x_vars


def _match_varname(vn: str, param_names: List[str]) -> Optional[str]:
    """Match a variable name to parameter names (handle _const/Intercept)."""
    if vn == '_const':
        for pn in param_names:
            if pn.lower() in ('intercept', '_const', 'const', '(intercept)'):
                return pn
        return None
    if vn in param_names:
        return vn
    return None


def _draw_weights(
    G: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw G bootstrap weights from the specified distribution."""
    if weight_type == 'rademacher':
        return rng.choice([-1.0, 1.0], size=G)
    elif weight_type == 'webb':
        vals = np.array([
            -np.sqrt(1.5), -np.sqrt(1.0), -np.sqrt(0.5),
             np.sqrt(0.5),  np.sqrt(1.0),  np.sqrt(1.5),
        ])
        return rng.choice(vals, size=G)
    elif weight_type == 'mammen':
        p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        vals = np.array([-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2])
        return rng.choice(vals, size=G, p=[p, 1 - p])
    raise ValueError(
        f"weight_type must be 'rademacher', 'webb', or 'mammen', "
        f"got '{weight_type}'"
    )


def _satterthwaite_dof(
    X: np.ndarray,
    cl: np.ndarray,
    unique_cl: np.ndarray,
    XtX_inv: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Satterthwaite degrees of freedom approximation for CR2.

    Approximates the effective DoF for each coefficient using:
        dof_j = (sum_g A_{jj,g})^2 / sum_g A_{jj,g}^2

    where A_{jj,g} is the j-th diagonal of the g-th cluster
    contribution to the variance.
    """
    G = len(unique_cl)
    A_diag = np.zeros((G, k))  # A_{jj,g} for each cluster and coeff

    for g_idx, g_val in enumerate(unique_cl):
        idx = cl == g_val
        X_g = X[idx]
        n_g = idx.sum()

        H_gg = X_g @ XtX_inv @ X_g.T
        I_H = np.eye(n_g) - H_gg

        eigvals, eigvecs = np.linalg.eigh(I_H)
        eigvals = np.maximum(eigvals, 1e-12)
        I_H_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        # Contribution: (X'X)^{-1} X_g' (I-H_gg)^{-1} X_g (X'X)^{-1}
        M_g = XtX_inv @ X_g.T @ I_H_inv @ X_g @ XtX_inv
        A_diag[g_idx] = np.diag(M_g)

    # Satterthwaite: dof = (sum A_jj)^2 / sum A_jj^2
    sum_A = A_diag.sum(axis=0)
    sum_A2 = (A_diag ** 2).sum(axis=0)
    dof = np.where(sum_A2 > 1e-20, sum_A ** 2 / sum_A2, G - 1)

    return dof


# ======================================================================
# Citation
# ======================================================================

from ..core.results import CausalResult

CausalResult._CITATIONS['jackknife_cluster'] = (
    "@article{bell2002bias,\n"
    "  title={Bias Reduction in Standard Errors for Linear Regression "
    "with Multi-Stage Samples},\n"
    "  author={Bell, Robert M. and McCaffrey, Daniel F.},\n"
    "  journal={Survey Methodology},\n"
    "  volume={28},\n"
    "  number={2},\n"
    "  pages={169--181},\n"
    "  year={2002}\n"
    "}"
)
