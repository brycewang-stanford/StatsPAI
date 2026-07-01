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
*Journal of Applied Econometrics*, 32(2), 233-254. [@mackinnon2017wild]

Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2008).
"Bootstrap-Based Improvements for Inference with Clustered Errors."
*Review of Economics and Statistics*, 90(3), 414-427. [@cameron2008bootstrap]
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility


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
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> state = rng.integers(0, 8, size=n)
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(size=n) + rng.normal(size=8)[state]
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "state": state})
    >>> result = sp.regress("y ~ x1 + x2", data=df, cluster="state")
    >>> jk = sp.jackknife_se(result, data=df, cluster="state")
    >>> jk.model_info["se_type"]
    'cluster jackknife'

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
    formula = result.model_info.get("formula", "")

    # Parse the formula to get y and X variable names
    y_var, x_vars = _parse_formula(formula, result)

    # Prepare data
    cols = [y_var] + x_vars + [cluster]
    cols = [c for c in cols if c in data.columns]
    df = data[cols].dropna()

    Y = df[y_var].values.astype(float)
    X_names = [v for v in x_vars if v in df.columns]
    X = np.column_stack([np.ones(len(df)), df[X_names].values.astype(float)])
    var_names = ["_const"] + X_names

    cl = df[cluster].values
    unique_cl = np.unique(cl)
    G = len(unique_cl)
    k = X.shape[1]

    # Leave-one-cluster-out estimates
    beta_loo = np.zeros((G, k))
    for g_idx, g_val in enumerate(unique_cl):
        mask = cl != g_val
        X_g = X[mask]
        Y_g = Y[mask]
        try:
            beta_loo[g_idx] = np.linalg.solve(X_g.T @ X_g, X_g.T @ Y_g)
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
        if vn == "_const":
            for pn in param_names:
                if pn.lower() in ("intercept", "_const", "const", "(intercept)"):
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
    model_info["se_type"] = "cluster jackknife"
    model_info["n_clusters"] = G
    model_info["jackknife_dof"] = df_resid

    data_info = dict(result.data_info)
    data_info["df_resid"] = df_resid

    diagnostics = dict(result.diagnostics)
    diagnostics["n_clusters"] = G
    diagnostics["effective_dof"] = df_resid

    _result = EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.inference.jackknife_se",
            params={"cluster": cluster, "alpha": alpha},
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> state = rng.integers(0, 10, size=n)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 + rng.normal(size=n) + rng.normal(size=10)[state]
    >>> df = pd.DataFrame({"y": y, "x1": x1, "state": state})
    >>> res = sp.regress("y ~ x1", data=df, cluster="state")
    >>> cr2 = sp.cr2_se(res, data=df, cluster="state")
    >>> cr2.model_info["se_type"]
    'CR2 (Bell-McCaffrey)'
    """
    X, Y, var_names, cl, _src = _design_from_result(result, data, cluster)

    unique_cl = np.unique(cl)
    G = len(unique_cl)
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
    model_info["se_type"] = "CR2 (Bell-McCaffrey)"
    model_info["n_clusters"] = G

    data_info = dict(result.data_info)
    # Use minimum Satterthwaite DoF across coefficients
    min_dof = float(np.nanmin(dof)) if len(dof) > 0 else G - 1
    data_info["df_resid"] = min_dof

    diagnostics = dict(result.diagnostics)
    diagnostics["n_clusters"] = G
    diagnostics["satterthwaite_dof"] = {
        var_names[i]: float(dof[i]) for i in range(len(dof))
    }

    _result = EconometricResults(
        params=result.params.copy(),
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.inference.cr2_se",
            params={"cluster": cluster, "alpha": alpha},
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def wild_cluster_boot(
    result: EconometricResults,
    data: pd.DataFrame,
    cluster: str,
    variable: str,
    n_boot: int = 999,
    weight_type: str = "rademacher",
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
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> state = rng.integers(0, 8, size=n)
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(size=n) + rng.normal(size=8)[state]
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "state": state})
    >>> result = sp.regress("y ~ x1 + x2", data=df, cluster="state")
    >>> boot = sp.wild_cluster_boot(result, data=df, cluster="state",
    ...                             variable="x1", n_boot=199, seed=0)
    >>> bool(0.0 <= boot["p_boot"] <= 1.0)
    True

    Notes
    -----
    The WCR bootstrap imposes H0: beta_variable = 0 when constructing
    bootstrap samples. Under few clusters, this yields much better
    size control than unrestricted wild bootstrap or cluster-robust SEs.

    For G < 12, use ``weight_type='webb'`` per Webb (2014).
    """
    import warnings

    rng = np.random.default_rng(seed)

    X, Y, var_names, cl, _src = _design_from_result(result, data, cluster)

    if variable not in var_names:
        raise ValueError(f"Variable '{variable}' not found. Available: {var_names}")
    test_idx = var_names.index(variable)

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
        "beta_hat": float(beta_test),
        "se_cluster": se_cl,
        "t_stat": float(t_stat),
        "p_boot": p_boot,
        "ci_boot": ci_boot,
        "t_distribution": t_boot,
        "n_clusters": G,
        "n_obs": n,
        "n_boot": n_boot,
        "weight_type": weight_type,
    }


# ======================================================================
# Helper functions
# ======================================================================


def _parse_formula(
    formula: str,
    result: EconometricResults,
) -> Tuple[str, List[str]]:
    """Parse formula or extract variable names from result."""
    if formula and "~" in formula:
        lhs, rhs = formula.split("~", 1)
        y_var = lhs.strip()
        x_terms = [t.strip() for t in rhs.split("+")]
        x_vars = [t for t in x_terms if t and t != "1"]
        return y_var, x_vars

    # Fallback: use result parameter names
    param_names = list(result.params.index)
    y_var = result.data_info.get(
        "dependent_var",
        result.model_info.get("depvar", result.data_info.get("y_var", "y")),
    )
    x_vars = [
        p
        for p in param_names
        if p.lower() not in ("intercept", "_const", "const", "(intercept)")
    ]
    return y_var, x_vars


def _match_varname(vn: str, param_names: List[str]) -> Optional[str]:
    """Match a variable name to parameter names (handle _const/Intercept)."""
    if vn == "_const":
        for pn in param_names:
            if pn.lower() in ("intercept", "_const", "const", "(intercept)"):
                return pn
        return None
    if vn in param_names:
        return vn
    return None


def _design_from_result(
    result: EconometricResults,
    data: pd.DataFrame,
    cluster: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    """Return ``(X, y, var_names, cluster_array, source)`` for residual-based SEs.

    Prefers the design matrix / outcome the estimator already stored on its
    result (``data_info['X'] / ['y'] / ['var_names']``).  For a fixed-effects
    model such as :func:`sp.feols` those stored arrays are the
    *within-transformed* (demeaned) design and outcome, so cluster-robust / CR2
    / wild-bootstrap inference operates on the correctly partialled-out model
    instead of a formula re-parse that would treat the absorbed fixed effects as
    ordinary regressors (the old path mishandled the ``|`` FE/IV separator).

    Falls back to the historical formula re-parse + plain-OLS design whenever the
    stored arrays are missing or cannot be *safely row-aligned* to ``data`` — in
    particular when rows were dropped for missing values, the fitted sample no
    longer matches ``data`` row-for-row, so the cluster key could not be aligned.
    On that fallback the behaviour is byte-for-byte the previous behaviour.

    The ``source`` flag (``"stored"`` / ``"reparsed"``) is returned so callers
    and tests can assert which path ran.
    """
    di = getattr(result, "data_info", {}) or {}
    stored_x = di.get("X")
    stored_y = di.get("y")
    stored_names = di.get("var_names")
    if (
        stored_x is not None
        and stored_y is not None
        and stored_names is not None
        and cluster in data.columns
    ):
        X = np.asarray(stored_x, dtype=float)
        y = np.asarray(stored_y, dtype=float).ravel()
        cl_col = data[cluster]
        # Trust the stored design only when it row-aligns to ``data`` and the
        # cluster key is complete; otherwise the fitted sample was filtered and
        # the rows would not correspond.
        if (
            X.ndim == 2
            and X.shape[0] == len(data)
            and y.shape[0] == X.shape[0]
            and not cl_col.isna().any()
        ):
            var_names = [str(v) for v in stored_names]
            return X, y, var_names, cl_col.to_numpy(), "stored"

    # --- fallback: formula re-parse (plain-OLS design) -----------------------
    y_var, x_vars = _parse_formula(result.model_info.get("formula", ""), result)
    cols = [y_var] + x_vars + [cluster]
    cols = [c for c in cols if c in data.columns]
    df = data[cols].dropna()
    y = df[y_var].to_numpy(dtype=float)
    x_names = [v for v in x_vars if v in df.columns]
    X = np.column_stack([np.ones(len(df)), df[x_names].to_numpy(dtype=float)])
    var_names = ["_const"] + x_names
    cl = df[cluster].to_numpy()
    return X, y, var_names, cl, "reparsed"


def _draw_weights(
    G: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw G bootstrap weights from the specified distribution."""
    if weight_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=G)
    elif weight_type == "webb":
        vals = np.array(
            [
                -np.sqrt(1.5),
                -np.sqrt(1.0),
                -np.sqrt(0.5),
                np.sqrt(0.5),
                np.sqrt(1.0),
                np.sqrt(1.5),
            ]
        )
        return rng.choice(vals, size=G)
    elif weight_type == "mammen":
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
    sum_A2 = (A_diag**2).sum(axis=0)
    dof = np.where(sum_A2 > 1e-20, sum_A**2 / sum_A2, G - 1)

    return dof


# ======================================================================
# Citation
# ======================================================================

from ..core.results import CausalResult

CausalResult._CITATIONS["jackknife_cluster"] = (
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


# ======================================================================
# Efficient OLS / 2SLS cluster SE helpers (read stored design; no re-parse)
# ======================================================================


def _ols_correction_from_cl_codes(
    cl_codes: np.ndarray,
    n: int,
    k: int,
) -> float:
    """Default small-sample correction for OLS cluster SE: (G/(G-1))·((n-1)/(n-k)).

    Matches the convention R ``sandwich::vcovCL`` uses (and the
    ``sp.regress`` standalone default). Returned as a float; callers that
    want a bias-reduced CR2/CR3 over the textbook should NOT add this on top
    (the Pustejovsky-Tipton 2018 definition already absorbs the df
    correction differently — see the dedicated helpers below).
    """
    g = int(cl_codes.max()) + 1
    return (g / (g - 1.0)) * ((n - 1) / (n - k))


def cr_vcov_ols(
    result: "EconometricResults",
    cluster_codes: np.ndarray,
    power: float = 0.5,
    small_sample: bool = True,
) -> "pd.Series":
    """Pustejovsky-Tipton CR2/CR3 cluster-robust SE on the stored OLS design.

    With ``power=0`` returns the textbook cluster-robust (CR0/CR1 — same
    formula as CR2 with A=I), ``power=0.5`` is CR2 (Bell-McCaffrey), and
    ``power=1`` is CR3 (the jackknife-type adjustment). The bias correction
    A_g = (I - H_gg)^{-p} is built on the per-cluster hat block
    ``H_gg = X_g (X'X)^{-1} X_g'`` (global bread — the convention R
    ``sandwich::vcovCL(HC2/3)`` uses). With ``small_sample=True`` the
    standard ``(G/(G-1))·((n-1)/(n-k))`` correction is applied; set it
    ``False`` to match R's bare cluster-robust vcov (the convention used by
    the bias-reduced CR variants of Pustejovsky-Tipton 2018).

    The result must carry ``data_info['X'/'y'/'var_names']`` (every
    ``sp.regress`` result with ``cluster=`` populated does); results
    without stored design fall back to the formula-reparse path
    (the legacy ``sp.cr2_se`` behavior).
    """
    iv = getattr(result, "data_info", None) or {}
    if not ({"X", "y", "var_names"} <= set(iv)):
        raise MethodIncompatibility(
            "cr_vcov_ols needs a result with stored data_info['X'/'y'/'var_names'] "
            "(i.e. from sp.regress with cluster=...). Use sp.cr2_se for the "
            "reparse fallback."
        )
    X = np.asarray(iv["X"], dtype=float)
    y = np.asarray(iv["y"], dtype=float).ravel()
    names = list(iv["var_names"])
    n, k = X.shape
    if cluster_codes.shape != (n,):
        raise MethodIncompatibility(
            "cluster_codes length must equal the number of observations in the "
            "fitted sample."
        )

    bread = np.linalg.inv(X.T @ X)  # (X'X)^-1
    beta = bread @ (X.T @ y)
    resid = y - X @ beta

    meat = np.zeros((k, k))
    g_set = int(cluster_codes.max()) + 1
    for cid in range(g_set):
        idx = cluster_codes == cid
        X_g = X[idx]
        r_g = resid[idx]
        ng = int(idx.sum())
        H_gg = X_g @ bread @ X_g.T
        i_h = np.eye(ng) - H_gg
        evals, evecs = np.linalg.eigh(i_h)
        evals = np.maximum(evals, 1e-12)
        a_g = evecs @ np.diag(evals ** (-power)) @ evecs.T
        score = X_g.T @ (a_g @ r_g)
        meat += np.outer(score, score)

    corr = _ols_correction_from_cl_codes(cluster_codes, n, k) if small_sample else 1.0
    vcov = corr * bread @ meat @ bread
    return pd.Series(np.sqrt(np.maximum(np.diag(vcov), 0)), index=names)


def two_way_correction_ols(
    result: "EconometricResults",
    c1_codes: np.ndarray,
    c2_codes: np.ndarray,
    c12_codes: np.ndarray,
    small_sample: bool = True,
) -> "pd.Series":
    """Two-way (Cameron-Gelbach-Miller 2011) cluster-robust SE on the OLS design.

    Inclusion-exclusion on the projected-score meat ``M1 + M2 - M12``, with
    ``M_g = (X_g' e_g)(X_g' e_g)'``.  Default correction is
    ``(G_min/(G_min-1))·((n-1)/(n-k))`` (Stata ``reg, cluster(a b) small``
    convention). Pass ``small_sample=False`` for the bias-reduced variant.
    """
    iv = getattr(result, "data_info", None) or {}
    X = np.asarray(iv["X"], dtype=float)
    y = np.asarray(iv["y"], dtype=float).ravel()
    names = list(iv["var_names"])
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ (X.T @ y)
    resid = y - X @ beta

    def _meat(codes: np.ndarray) -> np.ndarray:
        m = np.zeros((k, k))
        for cid in range(int(codes.max()) + 1):
            idx = codes == cid
            s = X[idx].T @ resid[idx]
            m += np.outer(s, s)
        return m

    meat = _meat(c1_codes) + _meat(c2_codes) - _meat(c12_codes)
    if small_sample:
        g_min = min(int(c1_codes.max()) + 1, int(c2_codes.max()) + 1)
        corr = (g_min / (g_min - 1.0)) * ((n - 1) / (n - k))
    else:
        corr = 1.0
    vcov = corr * bread @ meat @ bread
    return pd.Series(np.sqrt(np.maximum(np.diag(vcov), 0)), index=names)


def conley_vcov_ols(
    result: "EconometricResults",
    data: "pd.DataFrame",
    lat: str,
    lon: str,
    dist_cutoff: float,
) -> "pd.Series":
    """Conley spatial-HAC SE on the stored OLS/within design (acreg-compatible).

    Uniform spatial kernel on the cluster-robust scores ``score_a = X_a e_a``
    with ``acreg``'s planar distance (111 km/deg latitude, ``cos(lat_b)·111``
    per degree longitude, anchored at the column point ``b`` — asymmetric,
    then ``V`` is symmetrised). For a fixed-effects result the stored ``X`` is
    the within-transformed design, so this is spatial-HAC on the partialled-out
    model (matches ``acreg`` run on the FE-demeaned data). Reads
    ``data_info['X'/'y'/'var_names']``.
    """
    iv = getattr(result, "data_info", None) or {}
    if not ({"X", "y", "var_names"} <= set(iv)):
        raise MethodIncompatibility(
            "conley_vcov_ols needs a result with stored data_info['X'/'y'/"
            "'var_names'] (sp.regress with cluster=, or sp.feols)."
        )
    X = np.asarray(iv["X"], dtype=float)
    y = np.asarray(iv["y"], dtype=float).ravel()
    names = list(iv["var_names"])
    n, k = X.shape
    for c in (lat, lon):
        if c not in data.columns or data[c].isna().any() or len(data[c]) != n:
            raise MethodIncompatibility(
                f"Coordinate column {c!r} must be present, complete, and "
                "row-aligned to the fitted sample."
            )
    lat_v = data[lat].to_numpy(dtype=float)
    lon_v = data[lon].to_numpy(dtype=float)

    bread = np.linalg.inv(X.T @ X)
    beta = bread @ (X.T @ y)
    resid = y - X @ beta

    lon_scale = np.cos(np.radians(lat_v)) * 111.0
    d_lat = lat_v[:, None] - lat_v[None, :]
    d_lon = lon_v[:, None] - lon_v[None, :]
    dist = np.sqrt((111.0 * d_lat) ** 2 + (lon_scale[None, :] * d_lon) ** 2)
    weig = (dist <= dist_cutoff).astype(float)

    score = X * resid[:, None]
    core = bread @ (score.T @ weig @ score) @ bread
    vcov = 0.5 * (core + core.T)
    return pd.Series(np.sqrt(np.maximum(np.diag(vcov), 0)), index=names)


def _matrix_power_sym(mat: np.ndarray, power: float, tol: float = -12.0) -> np.ndarray:
    """Symmetric matrix power via eigendecomposition (clubSandwich ``matrix_power``).

    Eigenvalues at or below ``10**tol`` are zeroed (generalised inverse), matching
    R ``clubSandwich:::matrix_power``.
    """
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    val_p = np.where(vals > 10.0**tol, vals**power, 0.0)
    return (vecs * val_p) @ vecs.T


def glm_cr_vcov(
    X: np.ndarray,
    y: np.ndarray,
    family: Any,
    cluster_codes: np.ndarray,
    power: float = 0.5,
) -> np.ndarray:
    """clubSandwich CR2/CR3 cluster-robust SEs for a GLM (any exponential family).

    Reproduces R ``clubSandwich::vcovCR(glm_fit, type="CR2"/"CR3")`` to machine
    precision for the full (FE-as-dummies) design. ``power=0.5`` gives CR2
    (Bell-McCaffrey / Pustejovsky-Tipton 2018) and ``power=1.0`` gives CR3
    (the jackknife-type adjustment).

    The estimator generalises the OLS construction with the IRLS working weights.
    With ``d_i = dμ/dη``, ``V_i = Var(μ_i)`` and working weight ``w_i = d_i²/V_i``:

    * bread ``M = (X' diag(w) X)^{-1}``,
    * per-cluster weighted hat ``H_g = diag(d_g) X_g M X_g' diag(d_g/V_g)``,
    * CR2 adjustment ``A_g = Θ_g^{1/2} (Θ_g^{1/2}(I-H_g)Θ_g Θ_g^{1/2})^{-1/2}
      Θ_g^{1/2}`` with target ``Θ_g = diag(V_g)`` (CR3 uses ``A_g = (I-H_g)^{-1}``),
    * score ``s_g = X_g' diag(d_g/V_g) A_g (y_g-μ_g)``,
    * ``V = M (Σ_g s_g s_g') M``.

    Parameters
    ----------
    X : ndarray (n, p)
        Full model matrix (intercept + regressors + any FE dummies).
    y : ndarray (n,)
        Response.
    family : statsmodels GLM family instance
        E.g. ``sm.families.Poisson()`` or ``sm.families.Binomial()``.
    cluster_codes : ndarray (n,)
        Integer cluster labels in ``0..G-1``.
    power : float, default 0.5
        ``0.5`` → CR2, ``1.0`` → CR3 (== cluster jackknife).

    Returns
    -------
    ndarray (p,)
        Standard errors for every column of ``X`` (slice the regressor block).
    """
    import statsmodels.api as sm  # noqa: F401 — statsmodels is a core dependency

    fit = sm.GLM(y, X, family=family).fit()
    mu = np.asarray(fit.mu, dtype=float)
    n, p = X.shape
    # d = dμ/dη = 1 / (dη/dμ); V = variance(μ); working weight w = d²/V.
    d = 1.0 / family.link.deriv(mu)
    V = family.variance(mu)
    w = d**2 / V
    dV = d / V
    M = np.linalg.inv((X * w[:, None]).T @ X)
    e = y - mu

    meat = np.zeros((p, p))
    for cid in range(int(cluster_codes.max()) + 1):
        idx = cluster_codes == cid
        Xg = X[idx]
        dg = d[idx]
        Vg = V[idx]
        dVg = dV[idx]
        ng = int(idx.sum())
        Hg = (dg[:, None]) * (Xg @ M @ Xg.T) * (dVg[None, :])
        IH = np.eye(ng) - Hg
        if power == 0.5:
            thc = np.sqrt(Vg)
            G = (thc[:, None]) * IH * (Vg * thc)[None, :]
            A = (thc[:, None]) * _matrix_power_sym(G, -0.5) * thc[None, :]
        else:  # CR3 / jackknife
            A = np.linalg.inv(IH)
        s = (Xg.T * dVg) @ (A @ e[idx])
        meat += np.outer(s, s)

    vcov = M @ meat @ M
    return np.sqrt(np.maximum(np.diag(vcov), 0.0))
