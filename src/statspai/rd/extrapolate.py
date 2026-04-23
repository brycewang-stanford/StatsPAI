"""
Extrapolation and external validity for regression discontinuity designs.

Addresses the fundamental limitation that RD effects are only identified at the
cutoff by providing methods to extrapolate treatment effects away from it.

Implements:
- Angrist & Rokkanen (2015) conditional-independence extrapolation
- Multi-cutoff extrapolation (Cattaneo, Keele, Titiunik & Vazquez-Bare 2021 JASA)
- External validity diagnostics

References
----------
Angrist, J.D. and Rokkanen, M. (2015).
"Wanna Get Away? Regression Discontinuity Estimation of Exam School
Effects Away From the Cutoff."
*Journal of the American Statistical Association*, 110(512), 1331-1344.
doi:10.1080/01621459.2015.1012259
[@angrist2015wanna]

Cattaneo, M.D., Keele, L., Titiunik, R. and Vazquez-Bare, G. (2021).
"Extrapolating Treatment Effects in Multi-Cutoff Regression Discontinuity
Designs." *Journal of the American Statistical Association*, 116(536),
1941-1952. doi:10.1080/01621459.2020.1751646
[@cattaneo2021extrapolating]
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult
from .rdrobust import rdrobust


# ======================================================================
# Citations
# ======================================================================

CausalResult._CITATIONS['rd_extrapolate'] = (
    "@article{angrist2015wanna,\n"
    "  title={Wanna get away? Regression discontinuity estimation\n"
    "  of exam school effects away from the cutoff},\n"
    "  author={Angrist, Joshua D and Rokkanen, Miikka},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={110},\n"
    "  number={512},\n"
    "  pages={1331--1344},\n"
    "  year={2015},\n"
    "  doi={10.1080/01621459.2015.1012259}\n"
    "}"
)

CausalResult._CITATIONS['rd_multi_extrapolate'] = (
    "@article{cattaneo2021extrapolating,\n"
    "  title={Extrapolating treatment effects in multi-cutoff\n"
    "  regression discontinuity designs},\n"
    "  author={Cattaneo, Matias D and Keele, Luke and\n"
    "  Titiunik, Roc{\\'\\i}o and Vazquez-Bare, Gonzalo},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1941--1952},\n"
    "  year={2021},\n"
    "  doi={10.1080/01621459.2020.1751646}\n"
    "}"
)


# ======================================================================
# Internal helpers
# ======================================================================

def _ols_fit(X: np.ndarray, y: np.ndarray):
    """
    OLS regression via QR decomposition with fallback to lstsq.

    Returns (coefficients, residuals, variance-covariance matrix of beta).
    """
    n, k = X.shape
    try:
        Q, R = np.linalg.qr(X, mode='reduced')
        beta = np.linalg.solve(R, Q.T @ y)
        resid = y - X @ beta
        sigma2 = np.sum(resid ** 2) / max(n - k, 1)
        R_inv = np.linalg.inv(R)
        vcov = sigma2 * (R_inv @ R_inv.T)
    except np.linalg.LinAlgError:
        # Fallback for singular/near-singular design matrices
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = np.sum(resid ** 2) / max(n - k, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        vcov = sigma2 * XtX_inv
    return beta, resid, vcov


def _add_intercept(Z: np.ndarray) -> np.ndarray:
    """Prepend a column of ones."""
    n = Z.shape[0] if Z.ndim > 1 else len(Z)
    ones = np.ones((n, 1))
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return np.hstack([ones, Z])


def _partial_f_test(y: np.ndarray, Z: np.ndarray, x_running: np.ndarray):
    """
    Partial F-test for conditional independence: test whether the running
    variable X adds predictive power for Y beyond covariates Z.

    H0: coefficient on x_running is zero in regression Y ~ Z + X.
    """
    # Restricted model: Y ~ Z (with intercept)
    X_r = _add_intercept(Z)
    beta_r, resid_r, _ = _ols_fit(X_r, y)
    ssr_r = np.sum(resid_r ** 2)

    # Unrestricted model: Y ~ Z + x_running
    X_u = np.hstack([X_r, x_running.reshape(-1, 1)])
    beta_u, resid_u, _ = _ols_fit(X_u, y)
    ssr_u = np.sum(resid_u ** 2)

    n, k_u = X_u.shape
    q = 1  # one restriction (coefficient on x_running = 0)
    df_resid = n - k_u
    if df_resid <= 0:
        return np.nan, np.nan
    f_stat = ((ssr_r - ssr_u) / q) / (ssr_u / df_resid)
    p_value = 1.0 - sp_stats.f.cdf(f_stat, q, df_resid)
    return float(f_stat), float(p_value)


def _propensity_score(Z: np.ndarray, D: np.ndarray, max_iter: int = 50):
    """
    Logistic regression P(D=1|Z) via iteratively reweighted least squares.

    Returns predicted probabilities, clipped to [0.01, 0.99].
    """
    X = _add_intercept(Z)
    n, k = X.shape
    beta = np.zeros(k)

    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -20, 20)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1.0 - mu)
        W = np.maximum(W, 1e-10)
        z_tilde = eta + (D - mu) / W
        XtW = X.T * W[np.newaxis, :]
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ z_tilde)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta
    eta = np.clip(eta, -20, 20)
    ps = 1.0 / (1.0 + np.exp(-eta))
    return np.clip(ps, 0.01, 0.99)


def _bootstrap_cate(
    y: np.ndarray,
    Z: np.ndarray,
    x_running: np.ndarray,
    D: np.ndarray,
    eval_Z: np.ndarray,
    method: str,
    n_boot: int = 200,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Bootstrap standard errors for CATE at eval_Z points.

    Returns array of shape (n_eval, n_boot) with bootstrapped CATE values.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(y)
    n_eval = eval_Z.shape[0]
    boot_cates = np.empty((n_eval, n_boot))

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, Z_b, x_b, D_b = y[idx], Z[idx], x_running[idx], D[idx]

        treated = D_b == 1
        control = D_b == 0

        if treated.sum() < 3 or control.sum() < 3:
            boot_cates[:, b] = np.nan
            continue

        if method == 'ols':
            cate_b = _cate_ols(y_b, Z_b, D_b, eval_Z)
        elif method == 'ipw':
            cate_b = _cate_ipw(y_b, Z_b, D_b, eval_Z)
        elif method == 'doubly_robust':
            cate_b = _cate_dr(y_b, Z_b, D_b, eval_Z)
        else:
            cate_b = _cate_ols(y_b, Z_b, D_b, eval_Z)

        boot_cates[:, b] = cate_b

    return boot_cates


def _cate_ols(
    y: np.ndarray, Z: np.ndarray, D: np.ndarray, eval_Z: np.ndarray
) -> np.ndarray:
    """Estimate CATE via OLS on each side, predict at eval_Z."""
    treated = D == 1
    control = D == 0

    X_t = _add_intercept(Z[treated])
    X_c = _add_intercept(Z[control])

    beta_t, _, _ = _ols_fit(X_t, y[treated])
    beta_c, _, _ = _ols_fit(X_c, y[control])

    X_eval = _add_intercept(eval_Z)
    y1_hat = X_eval @ beta_t
    y0_hat = X_eval @ beta_c
    return y1_hat - y0_hat


def _cate_ipw(
    y: np.ndarray, Z: np.ndarray, D: np.ndarray, eval_Z: np.ndarray
) -> np.ndarray:
    """
    Estimate CATE via inverse probability weighting.

    For each eval point, compute weighted means using kernel-smoothed weights
    combined with IPW.
    """
    ps = _propensity_score(Z, D)
    n_eval = eval_Z.shape[0]
    cate = np.empty(n_eval)

    # Use a bandwidth based on Silverman's rule applied to Z
    if Z.ndim == 1:
        Z_2d = Z.reshape(-1, 1)
    else:
        Z_2d = Z
    eval_2d = eval_Z if eval_Z.ndim > 1 else eval_Z.reshape(-1, 1)

    n = len(y)
    d = Z_2d.shape[1]
    # Silverman bandwidth
    h_bw = 1.06 * np.std(Z_2d, axis=0) * n ** (-1.0 / (d + 4))
    h_bw = np.maximum(h_bw, 1e-10)

    for i in range(n_eval):
        # Gaussian kernel weights
        diff = (Z_2d - eval_2d[i]) / h_bw
        kern = np.exp(-0.5 * np.sum(diff ** 2, axis=1))

        w1 = kern * D / ps
        w0 = kern * (1 - D) / (1 - ps)

        sum_w1 = np.sum(w1)
        sum_w0 = np.sum(w0)

        if sum_w1 < 1e-10 or sum_w0 < 1e-10:
            cate[i] = np.nan
        else:
            mu1 = np.sum(w1 * y) / sum_w1
            mu0 = np.sum(w0 * y) / sum_w0
            cate[i] = mu1 - mu0

    return cate


def _cate_dr(
    y: np.ndarray, Z: np.ndarray, D: np.ndarray, eval_Z: np.ndarray
) -> np.ndarray:
    """
    Doubly robust (AIPW) CATE estimation.

    Combines outcome regression with IPW for double robustness.
    """
    ps = _propensity_score(Z, D)

    treated = D == 1
    control = D == 0

    X_t = _add_intercept(Z[treated])
    X_c = _add_intercept(Z[control])
    beta_t, _, _ = _ols_fit(X_t, y[treated])
    beta_c, _, _ = _ols_fit(X_c, y[control])

    X_all = _add_intercept(Z)
    mu1_hat = X_all @ beta_t
    mu0_hat = X_all @ beta_c

    n_eval = eval_Z.shape[0]
    eval_X = _add_intercept(eval_Z)
    mu1_eval = eval_X @ beta_t
    mu0_eval = eval_X @ beta_c

    if Z.ndim == 1:
        Z_2d = Z.reshape(-1, 1)
    else:
        Z_2d = Z
    eval_2d = eval_Z if eval_Z.ndim > 1 else eval_Z.reshape(-1, 1)

    n = len(y)
    d = Z_2d.shape[1]
    h_bw = 1.06 * np.std(Z_2d, axis=0) * n ** (-1.0 / (d + 4))
    h_bw = np.maximum(h_bw, 1e-10)

    cate = np.empty(n_eval)
    for i in range(n_eval):
        diff = (Z_2d - eval_2d[i]) / h_bw
        kern = np.exp(-0.5 * np.sum(diff ** 2, axis=1))

        # AIPW score
        aipw_1 = kern * (D * (y - mu1_hat) / ps + mu1_hat)
        aipw_0 = kern * ((1 - D) * (y - mu0_hat) / (1 - ps) + mu0_hat)

        sum_k = np.sum(kern)
        if sum_k < 1e-10:
            cate[i] = mu1_eval[i] - mu0_eval[i]
        else:
            cate[i] = np.sum(aipw_1) / sum_k - np.sum(aipw_0) / sum_k

    return cate


# ======================================================================
# Public API
# ======================================================================

def rd_extrapolate(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    treatment: Optional[str] = None,
    eval_points: Optional[np.ndarray] = None,
    n_eval: int = 20,
    method: str = 'ols',
    h_local: Optional[float] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Angrist-Rokkanen (2015) extrapolation of RD effects away from the cutoff.

    If Y(0), Y(1) are independent of X conditional on covariates Z, then
    treatment effects can be estimated at any value of the running variable,
    not just at the cutoff.

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
    covs : list of str, optional
        Covariate names for conditional independence. Required.
    treatment : str, optional
        Treatment variable for fuzzy RD. If None, sharp design assumed
        (D = 1{X >= c}).
    eval_points : np.ndarray, optional
        Running variable values at which to extrapolate the CATE.
        If None, ``n_eval`` equally spaced points spanning the data range.
    n_eval : int, default 20
        Number of evaluation points when ``eval_points`` is None.
    method : str, default 'ols'
        Estimation method: ``'ols'``, ``'ipw'``, or ``'doubly_robust'``.
    h_local : float, optional
        Bandwidth for local RD estimate at the cutoff (for comparison).
        If None, the MSE-optimal bandwidth from ``rdrobust`` is used.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``estimate`` is the average treatment effect across evaluation points.
        ``detail`` is a DataFrame with columns
        ``[x_value, cate, se, ci_lower, ci_upper]``.
        ``model_info`` contains conditional independence test results and
        the local RD estimate for comparison.

    Notes
    -----
    The key identifying assumption is *conditional independence*:
    ``Y(0), Y(1) ⊥ X | Z``.  A partial F-test is run on each side of the
    cutoff to check whether X retains predictive power for Y after
    conditioning on Z.  A rejection suggests the assumption may fail.

    References
    ----------
    Angrist, J.D. and Rokkanen, M. (2015). "Wanna Get Away? Regression
    Discontinuity Estimation of Exam School Effects Away from the Cutoff."
    *JASA*, 110(512), 1331-1344. [@angrist2015wanna]

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> Z = rng.normal(0, 1, n)
    >>> X = Z + rng.normal(0, 0.5, n)
    >>> D = (X >= 0).astype(int)
    >>> Y = 1.0 + 2.0 * Z + 3.0 * D + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X, 'z': Z})
    >>> result = rd_extrapolate(df, y='y', x='x', c=0, covs=['z'])
    >>> abs(result.estimate - 3.0) < 1.0
    True
    """
    if covs is None or len(covs) == 0:
        raise ValueError(
            "covs is required for Angrist-Rokkanen extrapolation. "
            "Conditional independence Y(d) ⊥ X | Z requires covariates Z."
        )
    if method not in ('ols', 'ipw', 'doubly_robust'):
        raise ValueError(
            f"method must be 'ols', 'ipw', or 'doubly_robust', got '{method}'"
        )

    df = data.dropna(subset=[y, x] + covs).copy()
    Y_arr = df[y].values.astype(float)
    X_arr = df[x].values.astype(float)
    Z_arr = df[covs].values.astype(float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)

    if treatment is not None:
        D_arr = df[treatment].values.astype(float)
    else:
        D_arr = (X_arr >= c).astype(float)

    n = len(Y_arr)

    # ------------------------------------------------------------------
    # Step 1: Local RD estimate at the cutoff for comparison
    # ------------------------------------------------------------------
    rd_kwargs = dict(data=df, y=y, x=x, c=c, alpha=alpha)
    if h_local is not None:
        rd_kwargs['h'] = h_local
    if treatment is not None:
        rd_kwargs['fuzzy'] = treatment

    local_rd = rdrobust(**rd_kwargs)
    local_est = local_rd.estimate
    local_se = local_rd.se
    local_ci = local_rd.ci

    # ------------------------------------------------------------------
    # Step 2: Conditional independence tests (partial F-test on each side)
    # ------------------------------------------------------------------
    treated = D_arr == 1
    control = D_arr == 0

    if control.sum() < 5 or treated.sum() < 5:
        raise ValueError(
            "Insufficient observations on one side of the cutoff "
            f"(control={control.sum()}, treated={treated.sum()})."
        )

    f_control, p_control = _partial_f_test(
        Y_arr[control], Z_arr[control], X_arr[control]
    )
    f_treated, p_treated = _partial_f_test(
        Y_arr[treated], Z_arr[treated], X_arr[treated]
    )

    ci_test = {
        'control_side': {'f_stat': f_control, 'p_value': p_control},
        'treated_side': {'f_stat': f_treated, 'p_value': p_treated},
        'ci_holds': bool(p_control > alpha and p_treated > alpha),
    }

    # ------------------------------------------------------------------
    # Step 3: Evaluation points
    # ------------------------------------------------------------------
    if eval_points is None:
        x_min, x_max = np.min(X_arr), np.max(X_arr)
        eval_points = np.linspace(x_min, x_max, n_eval)
    else:
        eval_points = np.asarray(eval_points, dtype=float)

    # Build evaluation covariate matrix: use mean of Z for each eval X
    # (since CATE(x) averages over Z distribution)
    # We evaluate at the mean covariate profile for interpretability
    Z_mean = np.mean(Z_arr, axis=0)
    n_ep = len(eval_points)
    eval_Z = np.tile(Z_mean, (n_ep, 1))

    # ------------------------------------------------------------------
    # Step 4: Estimate CATE at eval_points
    # ------------------------------------------------------------------
    if method == 'ols':
        cate_hat = _cate_ols(Y_arr, Z_arr, D_arr, eval_Z)
    elif method == 'ipw':
        cate_hat = _cate_ipw(Y_arr, Z_arr, D_arr, eval_Z)
    elif method == 'doubly_robust':
        cate_hat = _cate_dr(Y_arr, Z_arr, D_arr, eval_Z)

    # ------------------------------------------------------------------
    # Step 5: Bootstrap standard errors
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    boot_cates = _bootstrap_cate(
        Y_arr, Z_arr, X_arr, D_arr, eval_Z, method,
        n_boot=200, rng=rng,
    )
    # SE = std of bootstrap distribution (dropping NaN columns)
    valid_boot = ~np.isnan(boot_cates)
    se_hat = np.full(n_ep, np.nan)
    for i in range(n_ep):
        valid_i = boot_cates[i, valid_boot[i]]
        if len(valid_i) > 1:
            se_hat[i] = np.std(valid_i, ddof=1)

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = cate_hat - z_crit * se_hat
    ci_upper = cate_hat + z_crit * se_hat

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    detail = pd.DataFrame({
        'x_value': eval_points,
        'cate': cate_hat,
        'se': se_hat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    })

    # Average treatment effect across evaluation points
    valid = ~np.isnan(cate_hat)
    ate = float(np.nanmean(cate_hat))
    ate_se = float(np.sqrt(np.nanmean(se_hat[valid] ** 2) / max(valid.sum(), 1)))
    ate_pv = float(2 * (1 - sp_stats.norm.cdf(abs(ate) / max(ate_se, 1e-20))))
    ate_ci = (ate - z_crit * ate_se, ate + z_crit * ate_se)

    model_info = {
        'method': method,
        'cutoff': c,
        'n_control': int(control.sum()),
        'n_treated': int(treated.sum()),
        'covariates': covs,
        'conditional_independence_test': ci_test,
        'local_rd_estimate': {
            'estimate': local_est,
            'se': local_se,
            'ci': local_ci,
        },
        'n_eval_points': n_ep,
    }

    return CausalResult(
        method='RD Extrapolation (Angrist-Rokkanen 2015)',
        estimand='ATE (extrapolated)',
        estimate=ate,
        se=ate_se,
        pvalue=ate_pv,
        ci=ate_ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='rd_extrapolate',
    )


def rd_multi_extrapolate(
    data: pd.DataFrame,
    y: str,
    x: str,
    cutoffs: List[float],
    eval_points: Optional[np.ndarray] = None,
    method: str = 'linear',
    alpha: float = 0.05,
) -> CausalResult:
    """
    Multi-cutoff RD extrapolation (Cattaneo, Keele, Titiunik, Vazquez-Bare 2021).

    Estimates local RD effects at each cutoff, then interpolates/extrapolates
    a treatment effect function tau(x) through those point estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    cutoffs : list of float
        Cutoff values.  Must contain at least 2 cutoffs.
    eval_points : np.ndarray, optional
        Running variable values at which to predict tau(x).
        Defaults to 30 equally spaced points spanning the data range.
    method : str, default 'linear'
        Interpolation method:

        - ``'linear'``: tau(x) = a + b*x
        - ``'polynomial'``: polynomial of degree min(len(cutoffs)-1, 3)
        - ``'weighted'``: inverse-variance weighted local linear
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``estimate`` is the average extrapolated effect.
        ``detail`` is a DataFrame with columns
        ``[x_value, cate_extrapolated, se, ci_lower, ci_upper]``.
        ``model_info`` contains cutoff-specific estimates and heterogeneity
        test results.

    References
    ----------
    Cattaneo, M.D., Keele, L., Titiunik, R. and Vazquez-Bare, G. (2021).
    "Extrapolating Treatment Effects in Multi-Cutoff Regression Discontinuity
    Designs." *Journal of the American Statistical Association*, 116(536),
    1941-1952. doi:10.1080/01621459.2020.1751646
    [@cattaneo2021extrapolating]

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 3000
    >>> X = rng.uniform(-2, 4, n)
    >>> tau_true = 2.0 + 0.5 * X
    >>> D = ((X >= 1) | (X >= 3)).astype(int)
    >>> Y = 0.5 * X + tau_true * D + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X})
    >>> result = rd_multi_extrapolate(df, y='y', x='x', cutoffs=[1.0, 3.0])
    >>> result.estimate > 0
    True
    """
    if len(cutoffs) < 2:
        raise ValueError("At least 2 cutoffs are required for multi-cutoff extrapolation.")
    if method not in ('linear', 'polynomial', 'weighted'):
        raise ValueError(
            f"method must be 'linear', 'polynomial', or 'weighted', got '{method}'"
        )

    cutoffs = sorted(cutoffs)
    df = data.dropna(subset=[y, x]).copy()

    # ------------------------------------------------------------------
    # Step 1: Estimate local RD at each cutoff
    # ------------------------------------------------------------------
    cutoff_estimates = []
    for ci in cutoffs:
        try:
            res_i = rdrobust(df, y=y, x=x, c=ci, alpha=alpha)
            cutoff_estimates.append({
                'cutoff': ci,
                'estimate': res_i.estimate,
                'se': res_i.se,
                'ci_lower': res_i.ci[0],
                'ci_upper': res_i.ci[1],
                'n_obs': res_i.n_obs,
            })
        except (ValueError, np.linalg.LinAlgError):
            # Skip cutoffs with insufficient data
            continue

    if len(cutoff_estimates) < 2:
        raise ValueError(
            "Could not estimate local RD at enough cutoffs. "
            f"Successfully estimated at {len(cutoff_estimates)}/{len(cutoffs)} cutoffs."
        )

    c_vals = np.array([e['cutoff'] for e in cutoff_estimates])
    tau_vals = np.array([e['estimate'] for e in cutoff_estimates])
    se_vals = np.array([e['se'] for e in cutoff_estimates])

    # ------------------------------------------------------------------
    # Step 2: Fit tau(x) through the (cutoff, tau_hat) pairs
    # ------------------------------------------------------------------
    if eval_points is None:
        x_min, x_max = df[x].min(), df[x].max()
        eval_points = np.linspace(x_min, x_max, 30)
    else:
        eval_points = np.asarray(eval_points, dtype=float)

    n_ep = len(eval_points)
    k = len(cutoff_estimates)

    if method == 'linear':
        degree = 1
    elif method == 'polynomial':
        degree = min(k - 1, 3)
    else:  # weighted
        degree = 1

    # Weights for WLS
    if method == 'weighted':
        weights = 1.0 / (se_vals ** 2)
    else:
        weights = np.ones(k)

    # Construct Vandermonde matrix
    V = np.vander(c_vals, N=degree + 1, increasing=True)
    W = np.diag(weights)

    # WLS: (V'WV)^{-1} V'W tau
    VtW = V.T @ W
    try:
        A = VtW @ V
        A_inv = np.linalg.inv(A)
        beta_fit = A_inv @ (VtW @ tau_vals)
    except np.linalg.LinAlgError:
        # Fallback: simple OLS
        beta_fit = np.linalg.lstsq(V, tau_vals, rcond=None)[0]
        A_inv = np.eye(degree + 1)

    # Residual variance for SE calculation
    tau_fitted = V @ beta_fit
    residuals = tau_vals - tau_fitted
    if k > degree + 1:
        sigma2 = np.sum(weights * residuals ** 2) / (k - degree - 1)
    else:
        # Just-identified: use average SE^2 from local estimates
        sigma2 = np.mean(se_vals ** 2)

    vcov_beta = sigma2 * A_inv

    # Predict at evaluation points
    V_eval = np.vander(eval_points, N=degree + 1, increasing=True)
    tau_pred = V_eval @ beta_fit

    # SE via delta method: Var(tau(x)) = v(x)' Vcov v(x)
    se_pred = np.empty(n_ep)
    for i in range(n_ep):
        v_i = V_eval[i]
        se_pred[i] = np.sqrt(max(v_i @ vcov_beta @ v_i, 0))

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = tau_pred - z_crit * se_pred
    ci_upper = tau_pred + z_crit * se_pred

    # ------------------------------------------------------------------
    # Step 3: Heterogeneity test (Wald test: are all tau_j equal?)
    # ------------------------------------------------------------------
    if k >= 2:
        # Inverse-variance weighted mean (GLS-optimal for Wald test)
        iv_weights = 1.0 / (se_vals ** 2)
        tau_wbar = np.sum(iv_weights * tau_vals) / iv_weights.sum()
        # Wald statistic: sum of (tau_j - tau_wbar)^2 / se_j^2
        wald_stat = np.sum(iv_weights * (tau_vals - tau_wbar) ** 2)
        wald_df = k - 1
        wald_pval = 1.0 - sp_stats.chi2.cdf(wald_stat, wald_df)
        heterogeneity_test = {
            'wald_statistic': float(wald_stat),
            'df': wald_df,
            'p_value': float(wald_pval),
            'heterogeneous': bool(wald_pval < alpha),
        }
    else:
        heterogeneity_test = None

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    detail = pd.DataFrame({
        'x_value': eval_points,
        'cate_extrapolated': tau_pred,
        'se': se_pred,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    })

    ate = float(np.mean(tau_pred))
    ate_se = float(np.sqrt(np.mean(se_pred ** 2) / n_ep))
    ate_pv = float(2 * (1 - sp_stats.norm.cdf(abs(ate) / max(ate_se, 1e-20))))
    ate_ci = (ate - z_crit * ate_se, ate + z_crit * ate_se)

    cutoff_detail = pd.DataFrame(cutoff_estimates)

    model_info = {
        'method': method,
        'degree': degree,
        'coefficients': beta_fit.tolist(),
        'cutoff_estimates': cutoff_detail,
        'heterogeneity_test': heterogeneity_test,
        'n_cutoffs': k,
        'n_eval_points': n_ep,
    }

    return CausalResult(
        method='Multi-Cutoff RD Extrapolation (Cattaneo et al. 2021)',
        estimand='ATE (multi-cutoff extrapolated)',
        estimate=ate,
        se=ate_se,
        pvalue=ate_pv,
        ci=ate_ci,
        alpha=alpha,
        n_obs=len(df),
        detail=detail,
        model_info=model_info,
        _citation_key='rd_multi_extrapolate',
    )


def rd_external_validity(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    target_x_range: Optional[tuple] = None,
    alpha: float = 0.05,
) -> dict:
    """
    Diagnostic assessment of RD external validity.

    Compares covariate distributions between the cutoff neighborhood and a
    target population, runs the conditional independence test, and provides
    a recommendation on whether extrapolation is credible.

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
    covs : list of str, optional
        Covariate names for overlap and CI testing.
    target_x_range : tuple of (float, float), optional
        Running variable range ``(x_low, x_high)`` defining the target
        population. Defaults to the full data range.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys:

        - ``'local_estimate'``: local RD effect at the cutoff (CausalResult).
        - ``'ci_test'``: conditional independence test results (dict or None).
        - ``'overlap'``: covariate overlap statistics (dict or None).
        - ``'extrapolated_estimate'``: extrapolated ATE for the target
          population if CI test passes (CausalResult or None).
        - ``'recommendation'``: human-readable guidance string.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> Z = rng.normal(0, 1, n)
    >>> X = Z + rng.normal(0, 0.5, n)
    >>> D = (X >= 0).astype(int)
    >>> Y = 1.0 + 2.0 * Z + 3.0 * D + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X, 'z': Z})
    >>> diag = rd_external_validity(df, y='y', x='x', c=0, covs=['z'])
    >>> 'recommendation' in diag
    True
    """
    df = data.dropna(subset=[y, x]).copy()
    X_arr = df[x].values.astype(float)

    if target_x_range is None:
        target_x_range = (float(np.min(X_arr)), float(np.max(X_arr)))

    # ------------------------------------------------------------------
    # 1. Local RD estimate
    # ------------------------------------------------------------------
    local_rd = rdrobust(df, y=y, x=x, c=c, alpha=alpha)

    # ------------------------------------------------------------------
    # 2. Conditional independence test (if covariates provided)
    # ------------------------------------------------------------------
    ci_test = None
    if covs is not None and len(covs) > 0:
        df_clean = df.dropna(subset=covs)
        Y_arr = df_clean[y].values.astype(float)
        X_arr_c = df_clean[x].values.astype(float)
        Z_arr = df_clean[covs].values.astype(float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        D_arr = (X_arr_c >= c).astype(float)

        treated = D_arr == 1
        control = D_arr == 0

        if control.sum() >= 5 and treated.sum() >= 5:
            f_c, p_c = _partial_f_test(Y_arr[control], Z_arr[control], X_arr_c[control])
            f_t, p_t = _partial_f_test(Y_arr[treated], Z_arr[treated], X_arr_c[treated])
            ci_test = {
                'control_side': {'f_stat': f_c, 'p_value': p_c},
                'treated_side': {'f_stat': f_t, 'p_value': p_t},
                'ci_holds': bool(p_c > alpha and p_t > alpha),
            }

    # ------------------------------------------------------------------
    # 3. Covariate overlap diagnostics
    # ------------------------------------------------------------------
    overlap = None
    if covs is not None and len(covs) > 0:
        df_clean = df.dropna(subset=covs)
        X_arr_c = df_clean[x].values.astype(float)
        Z_arr = df_clean[covs].values.astype(float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        # Get bandwidth from local RD for cutoff neighborhood definition
        h_bw = local_rd.model_info.get('h_left', None)
        if h_bw is None:
            h_bw = local_rd.model_info.get('bandwidth', {})
            if isinstance(h_bw, dict):
                h_bw = h_bw.get('h_left', np.std(X_arr_c) * 0.2)
            if h_bw is None:
                h_bw = np.std(X_arr_c) * 0.2

        # Cutoff neighborhood
        near_cutoff = np.abs(X_arr_c - c) <= h_bw
        # Target population
        in_target = (X_arr_c >= target_x_range[0]) & (X_arr_c <= target_x_range[1])

        overlap_stats = {}
        if near_cutoff.sum() >= 5 and in_target.sum() >= 5:
            Z_local = Z_arr[near_cutoff]
            Z_target = Z_arr[in_target]

            for j, cov_name in enumerate(covs):
                z_loc = Z_local[:, j]
                z_tgt = Z_target[:, j]

                # Standardized mean difference
                pooled_sd = np.sqrt(
                    (np.var(z_loc, ddof=1) + np.var(z_tgt, ddof=1)) / 2
                )
                if pooled_sd < 1e-10:
                    smd = 0.0
                else:
                    smd = (np.mean(z_loc) - np.mean(z_tgt)) / pooled_sd

                # KS test for distributional overlap
                ks_stat, ks_pval = sp_stats.ks_2samp(z_loc, z_tgt)

                # Overlap coefficient: proportion of target support
                # covered by local distribution (via histogram overlap)
                bins = np.linspace(
                    min(z_loc.min(), z_tgt.min()),
                    max(z_loc.max(), z_tgt.max()),
                    50,
                )
                h_loc, _ = np.histogram(z_loc, bins=bins, density=True)
                h_tgt, _ = np.histogram(z_tgt, bins=bins, density=True)
                bin_w = bins[1] - bins[0]
                overlap_coeff = float(np.sum(np.minimum(h_loc, h_tgt)) * bin_w)

                overlap_stats[cov_name] = {
                    'std_mean_diff': float(smd),
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pval),
                    'overlap_coefficient': overlap_coeff,
                    'local_mean': float(np.mean(z_loc)),
                    'target_mean': float(np.mean(z_tgt)),
                }

            overlap = {
                'n_local': int(near_cutoff.sum()),
                'n_target': int(in_target.sum()),
                'bandwidth_used': float(h_bw),
                'covariate_diagnostics': overlap_stats,
            }

    # ------------------------------------------------------------------
    # 4. Extrapolated estimate (if CI test passes)
    # ------------------------------------------------------------------
    extrapolated = None
    if ci_test is not None and ci_test['ci_holds'] and covs is not None:
        target_eval = np.linspace(target_x_range[0], target_x_range[1], 20)
        try:
            extrapolated = rd_extrapolate(
                data=data, y=y, x=x, c=c, covs=covs,
                eval_points=target_eval, method='ols', alpha=alpha,
            )
        except (ValueError, np.linalg.LinAlgError):
            extrapolated = None

    # ------------------------------------------------------------------
    # 5. Recommendation
    # ------------------------------------------------------------------
    recommendation = _build_recommendation(ci_test, overlap, alpha)

    return {
        'local_estimate': local_rd,
        'ci_test': ci_test,
        'overlap': overlap,
        'extrapolated_estimate': extrapolated,
        'recommendation': recommendation,
    }


def _build_recommendation(ci_test, overlap, alpha):
    """Build a human-readable recommendation string."""
    lines = []

    if ci_test is None:
        lines.append(
            "No covariates provided: conditional independence cannot be tested. "
            "Extrapolation is not recommended without covariates."
        )
        return ' '.join(lines)

    ci_pass = ci_test['ci_holds']
    if ci_pass:
        lines.append(
            "Conditional independence test PASSED on both sides of the cutoff "
            f"(alpha={alpha}). The running variable does not appear to have "
            "additional predictive power for the outcome beyond the covariates."
        )
    else:
        failed_sides = []
        if ci_test['control_side']['p_value'] <= alpha:
            failed_sides.append(
                f"control (F={ci_test['control_side']['f_stat']:.2f}, "
                f"p={ci_test['control_side']['p_value']:.4f})"
            )
        if ci_test['treated_side']['p_value'] <= alpha:
            failed_sides.append(
                f"treated (F={ci_test['treated_side']['f_stat']:.2f}, "
                f"p={ci_test['treated_side']['p_value']:.4f})"
            )
        lines.append(
            "Conditional independence test FAILED on: "
            + ', '.join(failed_sides) + ". "
            "The running variable retains predictive power for the outcome "
            "after conditioning on covariates. Extrapolation may not be valid."
        )

    if overlap is not None:
        cov_diag = overlap.get('covariate_diagnostics', {})
        poor_overlap = []
        for cname, cstats in cov_diag.items():
            if abs(cstats['std_mean_diff']) > 0.25:
                poor_overlap.append(
                    f"{cname} (SMD={cstats['std_mean_diff']:.3f})"
                )
        if poor_overlap:
            lines.append(
                "Covariate balance concern: large standardized mean differences "
                "between cutoff neighborhood and target for: "
                + ', '.join(poor_overlap) + "."
            )
        else:
            lines.append(
                "Covariate distributions are reasonably similar between the "
                "cutoff neighborhood and target population."
            )

    if ci_pass and (overlap is None or not any(
        abs(s['std_mean_diff']) > 0.25
        for s in (overlap or {}).get('covariate_diagnostics', {}).values()
    )):
        lines.append(
            "RECOMMENDATION: Extrapolation appears credible. The Angrist-Rokkanen "
            "conditional independence assumption is supported by the data."
        )
    else:
        lines.append(
            "RECOMMENDATION: Exercise caution with extrapolation. Consider "
            "additional covariates or sensitivity analyses before relying on "
            "treatment effect estimates away from the cutoff."
        )

    return ' '.join(lines)


# ======================================================================
# Plot helper
# ======================================================================

def _extrapolation_plot(
    result: CausalResult,
    local_estimate: Optional[float] = None,
    cutoffs: Optional[List[float]] = None,
    ax=None,
    figsize: tuple = (10, 7),
    title: Optional[str] = None,
    xlabel: str = 'Running Variable (X)',
    ylabel: str = 'Treatment Effect',
    show_ci: bool = True,
    ci_alpha: float = 0.15,
    **kwargs,
):
    """
    Plot extrapolated CATE(x) curve with confidence bands.

    Parameters
    ----------
    result : CausalResult
        Output from ``rd_extrapolate`` or ``rd_multi_extrapolate``.
    local_estimate : float, optional
        Local RD estimate at the cutoff (shown as reference line).
        Automatically extracted from model_info if available.
    cutoffs : list of float, optional
        Cutoff values to mark with diamond markers.
        Automatically extracted from model_info if available.
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    figsize : tuple, default (10, 7)
        Figure size.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    show_ci : bool, default True
        Show confidence band.
    ci_alpha : float, default 0.15
        Transparency for CI shading.
    **kwargs
        Passed to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    if result.detail is None:
        raise ValueError("Result has no detail DataFrame to plot.")

    detail = result.detail

    # Determine column names (rd_extrapolate vs rd_multi_extrapolate)
    if 'cate' in detail.columns:
        cate_col = 'cate'
    elif 'cate_extrapolated' in detail.columns:
        cate_col = 'cate_extrapolated'
    else:
        raise ValueError("Cannot find CATE column in detail DataFrame.")

    x_vals = detail['x_value'].values
    cate_vals = detail[cate_col].values

    # Auto-extract from model_info
    mi = result.model_info or {}
    if local_estimate is None:
        local_info = mi.get('local_rd_estimate')
        if local_info is not None:
            local_estimate = local_info.get('estimate')

    if cutoffs is None:
        # From single-cutoff extrapolation
        if 'cutoff' in mi:
            cutoffs = [mi['cutoff']]
        # From multi-cutoff
        ce = mi.get('cutoff_estimates')
        if ce is not None and hasattr(ce, 'cutoff'):
            cutoffs = ce['cutoff'].tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Main CATE curve
    line_kw = dict(color='#2166ac', linewidth=2, label='Extrapolated CATE(x)')
    line_kw.update(kwargs)
    ax.plot(x_vals, cate_vals, **line_kw)

    # Confidence band
    if show_ci and 'ci_lower' in detail.columns and 'ci_upper' in detail.columns:
        ax.fill_between(
            x_vals,
            detail['ci_lower'].values,
            detail['ci_upper'].values,
            alpha=ci_alpha,
            color='#2166ac',
            label=f'{(1 - result.alpha) * 100:.0f}% CI',
        )

    # Shade interpolation vs extrapolation regions
    if cutoffs is not None and len(cutoffs) >= 2:
        c_min, c_max = min(cutoffs), max(cutoffs)
        # Interpolation region (between cutoffs)
        interp_mask = (x_vals >= c_min) & (x_vals <= c_max)
        if interp_mask.any():
            ax.axvspan(
                x_vals[interp_mask].min(), x_vals[interp_mask].max(),
                alpha=0.05, color='green', label='Interpolation region',
            )
        # Extrapolation regions
        extrap_left = x_vals < c_min
        extrap_right = x_vals > c_max
        if extrap_left.any():
            ax.axvspan(
                x_vals[extrap_left].min(), c_min,
                alpha=0.05, color='red', label='Extrapolation region',
            )
        if extrap_right.any():
            ax.axvspan(
                c_max, x_vals[extrap_right].max(),
                alpha=0.05, color='red',
            )

    # Mark cutoffs with diamonds
    if cutoffs is not None:
        for i, ci in enumerate(cutoffs):
            # Find nearest eval point or interpolate
            idx_near = np.argmin(np.abs(x_vals - ci))
            tau_at_c = cate_vals[idx_near]
            label = 'Cutoff RD estimate' if i == 0 else None
            ax.plot(
                ci, tau_at_c, marker='D', markersize=10,
                color='#b2182b', zorder=5, label=label,
            )

    # Local estimate reference line
    if local_estimate is not None:
        ax.axhline(
            local_estimate, linestyle='--', color='#b2182b',
            alpha=0.6, linewidth=1.2, label=f'Local RD = {local_estimate:.3f}',
        )

    # Zero reference
    ax.axhline(0, linestyle=':', color='grey', alpha=0.5, linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'RD Treatment Effect Extrapolation')
    ax.legend(frameon=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax
