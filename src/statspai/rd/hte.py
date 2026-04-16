"""
Heterogeneous treatment effects in regression discontinuity designs.

Implements the methodology of Calonico, Cattaneo, Farrell, Palomba, and
Titiunik (2025) for estimating conditional average treatment effects (CATE)
in RD designs using fully interacted local polynomial models.

The core idea: standard RD estimates tau = E[Y(1)-Y(0)|X=c]. When treatment
effects vary with covariates Z, we estimate CATE(z) = E[Y(1)-Y(0)|X=c, Z=z]
by fitting a fully interacted local linear model on each side of the cutoff.

References
----------
Calonico, S., Cattaneo, M.D., Farrell, M.H., Palomba, F. and Titiunik, R.
(2025). "Treatment Effect Heterogeneity in Regression Discontinuity Designs."
Working Paper.
"""

from typing import Optional, List, Union, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ._core import _kernel_fn, _kernel_mse_constant, _sandwich_variance
from .rdrobust import _select_bandwidth


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['rdhte'] = (
    "@article{calonico2025heterogeneous,\n"
    "  title={Treatment Effect Heterogeneity in Regression\n"
    "  Discontinuity Designs},\n"
    "  author={Calonico, Sebastian and Cattaneo, Matias D and\n"
    "  Farrell, Max H and Palomba, Filippo and Titiunik, Roc{\\'\\i}o},\n"
    "  year={2025},\n"
    "  journal={Working Paper}\n"
    "}"
)


# ======================================================================
# Public API
# ======================================================================

def rdhte(
    data: pd.DataFrame,
    y: str,
    x: str,
    z: Union[str, List[str]],
    c: float = 0,
    p: int = 1,
    h: Optional[float] = None,
    b: Optional[float] = None,
    kernel: str = 'triangular',
    bwselect: str = 'mserd',
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    eval_points: Optional[np.ndarray] = None,
    n_eval: int = 20,
) -> CausalResult:
    """
    Estimate conditional average treatment effects (CATE) in RD designs.

    Fits a fully interacted local polynomial model on each side of the
    cutoff and computes CATE(z) = (alpha_R - alpha_L) + z'(gamma_R - gamma_L).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    z : str or list of str
        Covariate(s) for treatment effect heterogeneity.
    c : float, default 0
        RD cutoff value.
    p : int, default 1
        Polynomial order for the running variable (1 = local linear).
    h : float, optional
        Bandwidth for estimation. If None, MSE-optimal bandwidth is selected.
    b : float, optional
        Bandwidth for bias correction. Defaults to h.
    kernel : str, default 'triangular'
        Kernel function: 'triangular', 'uniform', or 'epanechnikov'.
    bwselect : str, default 'mserd'
        Bandwidth selection method: 'mserd' or 'msetwo'.
    cluster : str, optional
        Cluster variable name for cluster-robust standard errors.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    eval_points : np.ndarray, optional
        Z values at which to evaluate CATE. Each row is a point in Z-space.
        If None, n_eval equally spaced quantiles (10th to 90th pctile) are used.
    n_eval : int, default 20
        Number of evaluation points when eval_points is not provided.

    Returns
    -------
    CausalResult
        - estimate: average CATE across evaluation points (the ATE)
        - detail: DataFrame with [z_value, cate, se, ci_lower, ci_upper, pvalue]
        - model_info: coefficients, heterogeneity test, bandwidth info

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.uniform(-1, 1, n)
    >>> Z = rng.normal(0, 1, n)
    >>> tau_z = 2.0 + 1.5 * Z  # CATE varies with Z
    >>> Y = 0.5 * X + tau_z * (X >= 0) + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X, 'z': Z})
    >>> result = rdhte(df, y='y', x='x', z='z', c=0)
    >>> abs(result.estimate - 2.0) < 1.0  # average CATE near 2
    True
    """
    # --- Validate inputs ---
    if kernel not in ('triangular', 'uniform', 'epanechnikov'):
        raise ValueError(
            f"kernel must be 'triangular', 'uniform', or 'epanechnikov', "
            f"got '{kernel}'"
        )
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    z_cols = [z] if isinstance(z, str) else list(z)
    dz = len(z_cols)

    for col in [y, x] + z_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    if cluster is not None and cluster not in data.columns:
        raise ValueError(f"Cluster column '{cluster}' not found in data")

    # --- Parse data ---
    Y_raw = data[y].values.astype(float)
    X_raw = data[x].values.astype(float)
    Z_raw = np.column_stack([data[zc].values.astype(float) for zc in z_cols])
    cl_raw = data[cluster].values if cluster else None

    # Drop NaN
    valid = np.isfinite(Y_raw) & np.isfinite(X_raw)
    for j in range(dz):
        valid &= np.isfinite(Z_raw[:, j])
    if cl_raw is not None:
        valid &= pd.notna(data[cluster].values)

    Y = Y_raw[valid]
    X_c = X_raw[valid] - c
    Z = Z_raw[valid]
    cl = cl_raw[valid] if cl_raw is not None else None

    n = len(Y)
    left = X_c < 0
    right = X_c >= 0
    n_left = int(left.sum())
    n_right = int(right.sum())

    # Minimum observations check: need at least (p+1) + dz + p*dz + 1
    # parameters per side: intercept + p running-var terms + dz covariates
    # + p*dz interactions
    n_params = (p + 1) + dz + p * dz
    if n_left < n_params + 2 or n_right < n_params + 2:
        raise ValueError(
            f"Not enough observations (left={n_left}, right={n_right}, "
            f"need >= {n_params + 2} per side for p={p}, dim(Z)={dz})."
        )

    # --- Bandwidth selection ---
    h_auto = h is None
    if h is None:
        h = rdbwhte(data, y, x, z, c=c, p=p, kernel=kernel)
    if b is None:
        b = h

    # --- Build evaluation points ---
    if eval_points is not None:
        eval_pts = np.atleast_2d(eval_points)
        if eval_pts.ndim == 1 or (eval_pts.ndim == 2 and eval_pts.shape[1] != dz):
            if dz == 1:
                eval_pts = eval_pts.reshape(-1, 1)
            else:
                raise ValueError(
                    f"eval_points must have {dz} columns, "
                    f"got shape {eval_points.shape}"
                )
    else:
        # Quantiles from 10th to 90th percentile
        if dz == 1:
            pctiles = np.linspace(10, 90, n_eval)
            z_vals = np.percentile(Z[:, 0], pctiles)
            eval_pts = z_vals.reshape(-1, 1)
        else:
            # For multivariate Z: grid over marginal quantiles
            pctiles = np.linspace(10, 90, max(int(n_eval ** (1 / dz)), 3))
            grids = [np.percentile(Z[:, j], pctiles) for j in range(dz)]
            mesh = np.meshgrid(*grids, indexing='ij')
            eval_pts = np.column_stack([m.ravel() for m in mesh])
            # Trim to at most n_eval points (take evenly spaced subset)
            if len(eval_pts) > n_eval:
                idx = np.round(np.linspace(0, len(eval_pts) - 1, n_eval)).astype(int)
                eval_pts = eval_pts[idx]

    n_eval_actual = len(eval_pts)

    # --- Fit fully interacted model on each side ---
    beta_L, vcov_L, n_eff_L = _interacted_wls(
        Y[left], X_c[left], Z[left], h, p, kernel,
        cl[left] if cl is not None else None,
    )
    beta_R, vcov_R, n_eff_R = _interacted_wls(
        Y[right], X_c[right], Z[right], h, p, kernel,
        cl[right] if cl is not None else None,
    )

    # --- Extract CATE coefficients ---
    # Model on each side: Y = alpha + beta_1*(X-c) + ... + beta_p*(X-c)^p
    #                        + Z'gamma + (X-c)*Z'delta_1 + ... + (X-c)^p*Z'delta_p
    # Parameter layout: [intercept, (X-c), ..., (X-c)^p, Z_1, ..., Z_dz,
    #                    (X-c)*Z_1, ..., (X-c)*Z_dz, ..., (X-c)^p*Z_1, ..., (X-c)^p*Z_dz]
    # CATE(z) = (alpha_R - alpha_L) + z'(gamma_R - gamma_L)
    # Indices: intercept = 0, gamma starts at p+1, gamma has dz entries

    idx_intercept = 0
    idx_gamma_start = p + 1
    idx_gamma_end = p + 1 + dz

    # Difference vector for CATE constant part and Z-coefficients
    diff_alpha = beta_R[idx_intercept] - beta_L[idx_intercept]
    diff_gamma = beta_R[idx_gamma_start:idx_gamma_end] - beta_L[idx_gamma_start:idx_gamma_end]

    # Joint variance of the differences (independent sides)
    # Indices to extract: [intercept, gamma_1, ..., gamma_dz]
    extract_idx = np.array([idx_intercept] + list(range(idx_gamma_start, idx_gamma_end)))
    vcov_diff = vcov_R[np.ix_(extract_idx, extract_idx)] + vcov_L[np.ix_(extract_idx, extract_idx)]

    # --- Evaluate CATE at each point ---
    cate_vals = np.empty(n_eval_actual)
    se_vals = np.empty(n_eval_actual)
    ci_lower_vals = np.empty(n_eval_actual)
    ci_upper_vals = np.empty(n_eval_actual)
    pv_vals = np.empty(n_eval_actual)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for i in range(n_eval_actual):
        z_i = eval_pts[i]
        # CATE(z) = diff_alpha + z'*diff_gamma
        # Weight vector w = [1, z_1, ..., z_dz] applied to [diff_alpha, diff_gamma]
        w = np.concatenate([[1.0], z_i])
        cate_i = float(w @ np.concatenate([[diff_alpha], diff_gamma]))
        var_i = float(w @ vcov_diff @ w)
        se_i = float(np.sqrt(max(var_i, 0)))

        cate_vals[i] = cate_i
        se_vals[i] = se_i
        ci_lower_vals[i] = cate_i - z_crit * se_i
        ci_upper_vals[i] = cate_i + z_crit * se_i
        z_stat = cate_i / se_i if se_i > 1e-15 else 0.0
        pv_vals[i] = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # --- Average CATE (the ATE) ---
    ate = float(np.mean(cate_vals))
    # SE of the average: mean of the linear functions
    w_avg = np.zeros(1 + dz)
    w_avg[0] = 1.0
    w_avg[1:] = np.mean(eval_pts, axis=0)
    ate_var = float(w_avg @ vcov_diff @ w_avg)
    ate_se = float(np.sqrt(max(ate_var, 0)))
    ate_z = ate / ate_se if ate_se > 1e-15 else 0.0
    ate_pv = float(2 * (1 - stats.norm.cdf(abs(ate_z))))
    ate_ci = (ate - z_crit * ate_se, ate + z_crit * ate_se)

    # --- Heterogeneity test: H0: gamma_R - gamma_L = 0 ---
    het_test = _heterogeneity_test(diff_gamma, vcov_diff, dz)

    # --- Detail DataFrame ---
    if dz == 1:
        z_display = eval_pts[:, 0]
    else:
        z_display = [tuple(row) for row in eval_pts]

    detail = pd.DataFrame({
        'z_value': z_display,
        'cate': cate_vals,
        'se': se_vals,
        'ci_lower': ci_lower_vals,
        'ci_upper': ci_upper_vals,
        'pvalue': pv_vals,
    })

    # --- Model info ---
    model_info: Dict[str, Any] = {
        'rd_type': 'Sharp',
        'polynomial_p': p,
        'kernel': kernel,
        'bandwidth_h': round(float(h), 6),
        'bandwidth_b': round(float(b), 6),
        'bwselect': bwselect if h_auto else 'manual',
        'cutoff': c,
        'z_covariates': z_cols,
        'n_z': dz,
        'n_left': n_left,
        'n_right': n_right,
        'n_effective_left': n_eff_L,
        'n_effective_right': n_eff_R,
        'n_eval_points': n_eval_actual,
        'ate': ate,
        'ate_se': ate_se,
        'ate_pvalue': ate_pv,
        'ate_ci': ate_ci,
        'diff_alpha': float(diff_alpha),
        'diff_gamma': diff_gamma.tolist(),
        'coefficients_left': beta_L.tolist(),
        'coefficients_right': beta_R.tolist(),
        'heterogeneity_test': het_test,
        'vcov_diff': vcov_diff.tolist(),
    }

    result = CausalResult(
        method='RD Heterogeneous Treatment Effects',
        estimand='CATE',
        estimate=ate,
        se=ate_se,
        pvalue=ate_pv,
        ci=ate_ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='rdhte',
    )

    # Attach plot method
    result.plot = lambda **kw: _rdhte_plot(result, **kw)

    return result


def rdbwhte(
    data: pd.DataFrame,
    y: str,
    x: str,
    z: Union[str, List[str]],
    c: float = 0,
    p: int = 1,
    kernel: str = 'triangular',
) -> float:
    """
    MSE-optimal bandwidth selection for the fully interacted RD model.

    Accounts for the additional variance introduced by the covariate
    interaction terms compared to the standard local polynomial RD bandwidth.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    z : str or list of str
        Covariate(s) for treatment effect heterogeneity.
    c : float, default 0
        RD cutoff value.
    p : int, default 1
        Polynomial order.
    kernel : str, default 'triangular'
        Kernel function.

    Returns
    -------
    float
        MSE-optimal bandwidth.
    """
    z_cols = [z] if isinstance(z, str) else list(z)
    dz = len(z_cols)

    for col in [y, x] + z_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    Y = data[y].values.astype(float)
    X_raw = data[x].values.astype(float)
    Z = np.column_stack([data[zc].values.astype(float) for zc in z_cols])

    valid = np.isfinite(Y) & np.isfinite(X_raw)
    for j in range(dz):
        valid &= np.isfinite(Z[:, j])

    Y = Y[valid]
    X_c = X_raw[valid] - c
    Z = Z[valid]

    n = len(Y)
    left = X_c < 0
    right = X_c >= 0

    # Step 1: get the standard MSE-optimal bandwidth as a baseline
    h_base = _select_bandwidth(Y, X_c, left, right, p, kernel, 'mserd')
    if isinstance(h_base, tuple):
        h_base = float(np.mean(h_base))

    # Step 2: adjust for the additional variance from interaction terms
    # The interacted model has (p+1) + dz + p*dz parameters per side vs
    # (p+1) for the standard model. The MSE-optimal bandwidth scales as
    # h ~ n^{-1/(2p+3)}. With more parameters, variance increases, so the
    # optimal bandwidth is wider. The adjustment factor comes from the
    # ratio of integrated variance constants.
    #
    # For local linear (p=1): standard has 2 params, interacted has 2 + dz + dz = 2(1+dz)
    # The variance inflation factor scales the bandwidth upward:
    # h_hte = h_base * (n_params_interacted / n_params_standard)^{1/(2p+3)}
    n_params_std = p + 1
    n_params_hte = (p + 1) + dz + p * dz
    rate_exponent = 1.0 / (2 * p + 3)
    inflation = (n_params_hte / n_params_std) ** rate_exponent

    h_hte = h_base * inflation

    # Step 3: refine using pilot residuals from interacted model
    h_pilot = min(h_hte * 1.5, 0.98 * np.ptp(X_c))

    # Pilot fit: residual variance from interacted model on each side
    sigma2_L = _interacted_residual_var(Y[left], X_c[left], Z[left], h_pilot, p, kernel)
    sigma2_R = _interacted_residual_var(Y[right], X_c[right], Z[right], h_pilot, p, kernel)

    # Curvature (second derivative of conditional mean)
    h_deriv = max(np.median(np.abs(X_c)), h_pilot) * 1.5
    m2_L = _interacted_second_deriv(Y[left], X_c[left], Z[left], h_deriv, kernel)
    m2_R = _interacted_second_deriv(Y[right], X_c[right], Z[right], h_deriv, kernel)

    # Density at cutoff
    sd_x = np.std(X_c)
    h_dens = 1.06 * sd_x * n ** (-1 / 5)
    n_near = np.sum(np.abs(X_c) <= h_dens)
    f_c = max(n_near / (2 * h_dens * n), 1e-10) if h_dens > 0 and n > 0 else 1.0

    # MSE-optimal: h = (C_K * (sigma2_L + sigma2_R) / (f_c * bias_sq * n))^{1/(2p+3)}
    C_K = _kernel_mse_constant(kernel)
    bias_sq = ((m2_R - m2_L) / 2) ** 2

    x_range = np.ptp(X_c)
    if bias_sq < 1e-12:
        h_opt = h_hte
    else:
        h_opt = (C_K * (sigma2_L + sigma2_R) /
                 (f_c * bias_sq * n)) ** rate_exponent

    h_opt = float(np.clip(h_opt, 0.02 * x_range, 0.98 * x_range))

    return h_opt


def rdhte_lincom(
    result: CausalResult,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute a weighted linear combination of CATE estimates.

    Useful for computing group-specific average treatment effects,
    e.g., the average CATE for males vs females.

    Parameters
    ----------
    result : CausalResult
        Result from rdhte().
    weights : np.ndarray
        Linear combination weights. Must have length equal to the number
        of evaluation points in result.detail.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: 'estimate', 'se', 'ci', 'pvalue'.

    Examples
    --------
    Compute the average CATE for the first and second halves of Z:

    >>> result = rdhte(df, y='y', x='x', z='z')
    >>> n_pts = len(result.detail)
    >>> w1 = np.zeros(n_pts)
    >>> w1[:n_pts//2] = 1.0 / (n_pts//2)  # first half
    >>> lincom1 = rdhte_lincom(result, w1)
    """
    weights = np.asarray(weights, dtype=float)
    detail = result.detail
    n_pts = len(detail)

    if len(weights) != n_pts:
        raise ValueError(
            f"weights has length {len(weights)}, expected {n_pts} "
            f"(number of evaluation points)"
        )

    # Recover the structural parameters to build the covariance properly
    mi = result.model_info
    vcov_diff = np.array(mi['vcov_diff'])
    diff_alpha = mi['diff_alpha']
    diff_gamma = np.array(mi['diff_gamma'])
    dz = mi['n_z']

    # Each eval point i has CATE(z_i) = w_i' * theta where
    # w_i = [1, z_i1, ..., z_idz] and theta = [diff_alpha, diff_gamma]
    # The weighted linear combination is: sum_i weights[i] * CATE(z_i)
    # = (sum_i weights[i] * w_i)' * theta
    # with variance = aggregated_w' * vcov_diff * aggregated_w

    # Reconstruct evaluation points from detail
    z_values = detail['z_value'].values
    if dz == 1:
        eval_pts = np.array(z_values, dtype=float).reshape(-1, 1)
    else:
        eval_pts = np.array([list(zv) for zv in z_values], dtype=float)

    # Aggregated weight vector
    w_agg = np.zeros(1 + dz)
    for i in range(n_pts):
        w_i = np.concatenate([[1.0], eval_pts[i]])
        w_agg += weights[i] * w_i

    theta = np.concatenate([[diff_alpha], diff_gamma])
    estimate = float(w_agg @ theta)
    variance = float(w_agg @ vcov_diff @ w_agg)
    se = float(np.sqrt(max(variance, 0)))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - z_crit * se, estimate + z_crit * se)
    z_stat = estimate / se if se > 1e-15 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return {
        'estimate': estimate,
        'se': se,
        'ci': ci,
        'pvalue': pvalue,
    }


# ======================================================================
# Internal helpers
# ======================================================================

def _build_interacted_design(
    X_c: np.ndarray,
    Z: np.ndarray,
    p: int,
) -> np.ndarray:
    """
    Build the fully interacted design matrix.

    Columns: [1, (X-c), (X-c)^2, ..., (X-c)^p,
              Z_1, ..., Z_dz,
              (X-c)*Z_1, ..., (X-c)*Z_dz,
              ...
              (X-c)^p*Z_1, ..., (X-c)^p*Z_dz]

    Parameters
    ----------
    X_c : np.ndarray, shape (n,)
        Centered running variable.
    Z : np.ndarray, shape (n, dz)
        Covariates.
    p : int
        Polynomial order.

    Returns
    -------
    np.ndarray, shape (n, (p+1) + dz + p*dz) = (n, (p+1)(1+dz))
    """
    n = len(X_c)
    dz = Z.shape[1] if Z.ndim > 1 else 1
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    cols = []

    # Running variable polynomial: 1, (X-c), ..., (X-c)^p
    for j in range(p + 1):
        cols.append(X_c ** j)

    # Z main effects
    for k in range(dz):
        cols.append(Z[:, k])

    # Interactions: (X-c)^j * Z_k for j=1..p, k=1..dz
    for j in range(1, p + 1):
        for k in range(dz):
            cols.append((X_c ** j) * Z[:, k])

    return np.column_stack(cols)


def _interacted_wls(
    y: np.ndarray,
    x_c: np.ndarray,
    Z: np.ndarray,
    h: float,
    p: int,
    kernel: str,
    cluster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Kernel-weighted WLS for the fully interacted local polynomial model.

    Returns (beta, vcov, n_effective).
    """
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    dz = Z.shape[1]

    u = x_c / h
    w = _kernel_fn(u, kernel)
    in_bw = np.abs(u) <= 1
    n_eff = int(in_bw.sum())

    n_params = (p + 1) + dz + p * dz

    if n_eff < n_params + 2:
        return np.zeros(n_params), np.eye(n_params) * 1e10, 0

    y_bw = y[in_bw]
    x_bw = x_c[in_bw]
    Z_bw = Z[in_bw]
    w_bw = w[in_bw]

    # Design matrix
    Xmat = _build_interacted_design(x_bw, Z_bw, p)
    k = Xmat.shape[1]

    # WLS via square-root weights
    sqw = np.sqrt(w_bw)
    Xw = Xmat * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        XtWX = Xw.T @ Xw
        beta = np.linalg.solve(XtWX, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        XtWX = Xw.T @ Xw

    resid = y_bw - Xmat @ beta

    cl_bw = cluster[in_bw] if cluster is not None else None
    vcov = _sandwich_variance(Xw, yw, beta, resid, n_eff, k, cl_bw)

    return beta, vcov, n_eff


def _heterogeneity_test(
    diff_gamma: np.ndarray,
    vcov_diff: np.ndarray,
    dz: int,
) -> Dict[str, float]:
    """
    Wald test for H0: gamma_R - gamma_L = 0 (no heterogeneity).

    The test statistic is:
        W = diff_gamma' * Sigma_gamma^{-1} * diff_gamma ~ chi2(dz)

    where Sigma_gamma is the submatrix of vcov_diff corresponding to
    the gamma coefficients (rows/cols 1:dz+1, excluding intercept).
    """
    # vcov_diff is (1+dz) x (1+dz): [intercept, gamma_1, ..., gamma_dz]
    # Extract the gamma block (indices 1 to dz+1)
    Sigma_gamma = vcov_diff[1:, 1:]

    try:
        Sigma_inv = np.linalg.inv(Sigma_gamma)
        wald_stat = float(diff_gamma @ Sigma_inv @ diff_gamma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_gamma)
        wald_stat = float(diff_gamma @ Sigma_inv @ diff_gamma)

    wald_stat = max(wald_stat, 0.0)
    wald_pv = float(1 - stats.chi2.cdf(wald_stat, df=dz))

    return {
        'statistic': wald_stat,
        'pvalue': wald_pv,
        'df': dz,
    }


def _interacted_residual_var(
    y: np.ndarray,
    x_c: np.ndarray,
    Z: np.ndarray,
    h: float,
    p: int,
    kernel: str,
) -> float:
    """Residual variance from the interacted model within bandwidth."""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    u = x_c / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < (p + 1) + Z.shape[1] + p * Z.shape[1] + 2:
        return float(np.var(y)) if len(y) > 0 else 1.0

    y_bw = y[in_bw]
    x_bw = x_c[in_bw]
    Z_bw = Z[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    Xmat = _build_interacted_design(x_bw, Z_bw, p)
    sqw = np.sqrt(w_bw)
    Xw = Xmat * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        resid = y_bw - Xmat @ beta
        return float(np.average(resid ** 2, weights=w_bw))
    except Exception:
        return float(np.var(y_bw))


def _interacted_second_deriv(
    y: np.ndarray,
    x_c: np.ndarray,
    Z: np.ndarray,
    h: float,
    kernel: str,
) -> float:
    """Estimate m''(0) from interacted local cubic, evaluated at mean(Z)."""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    u = x_c / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 8:
        return 0.0

    y_bw = y[in_bw]
    x_bw = x_c[in_bw]
    Z_bw = Z[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    # Fit a local cubic in x with Z controls (no interactions for pilot)
    # y = b0 + b1*x + b2*x^2 + b3*x^3 + Z'*gamma
    dz = Z_bw.shape[1]
    cols = [x_bw ** j for j in range(4)]
    for k in range(dz):
        cols.append(Z_bw[:, k])
    Xmat = np.column_stack(cols)

    sqw = np.sqrt(w_bw)
    Xw = Xmat * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        return float(2 * beta[2])  # m''(0) = 2*beta_2
    except Exception:
        return 0.0


# ======================================================================
# Plot helper
# ======================================================================

def _rdhte_plot(
    result: CausalResult,
    ax=None,
    ci_alpha: float = 0.2,
    cate_color: str = '#2171B5',
    ate_color: str = '#CB181D',
    zero_color: str = 'gray',
    xlabel: Optional[str] = None,
    ylabel: str = 'CATE',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
):
    """
    Plot CATE(z) vs z with confidence bands.

    Parameters
    ----------
    result : CausalResult
        Result from rdhte().
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.
    ci_alpha : float, default 0.2
        Transparency for confidence band shading.
    cate_color : str, default '#2171B5'
        Color for the CATE line.
    ate_color : str, default '#CB181D'
        Color for the average treatment effect line.
    zero_color : str, default 'gray'
        Color for the zero-effect line.
    xlabel : str, optional
        X-axis label. Defaults to covariate name(s).
    ylabel : str, default 'CATE'
        Y-axis label.
    title : str, optional
        Plot title.
    figsize : tuple, default (8, 5)
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. "
                          "Install it with: pip install matplotlib")

    detail = result.detail
    mi = result.model_info
    dz = mi['n_z']

    if dz > 1:
        raise NotImplementedError(
            "Plotting is only supported for scalar Z (dim=1). "
            "For multivariate Z, construct custom plots from result.detail."
        )

    z_vals = detail['z_value'].values.astype(float)
    cate = detail['cate'].values
    ci_lo = detail['ci_lower'].values
    ci_hi = detail['ci_upper'].values
    ate = mi['ate']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Sort by z for smooth line
    order = np.argsort(z_vals)
    z_sorted = z_vals[order]
    cate_sorted = cate[order]
    ci_lo_sorted = ci_lo[order]
    ci_hi_sorted = ci_hi[order]

    # CATE line + CI band
    ax.plot(z_sorted, cate_sorted, color=cate_color, linewidth=2,
            label='CATE(z)')
    ax.fill_between(z_sorted, ci_lo_sorted, ci_hi_sorted,
                    color=cate_color, alpha=ci_alpha,
                    label=f'{int((1 - result.alpha) * 100)}% CI')

    # Horizontal lines
    ax.axhline(y=0, color=zero_color, linestyle='--', linewidth=1,
               label='Zero effect')
    ax.axhline(y=ate, color=ate_color, linestyle='-.', linewidth=1.5,
               label=f'ATE = {ate:.3f}')

    # Labels
    z_names = mi.get('z_covariates', ['Z'])
    ax.set_xlabel(xlabel or z_names[0])
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'Heterogeneous Treatment Effects in RD')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return ax
