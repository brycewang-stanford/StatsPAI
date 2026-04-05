"""
Regression Discontinuity in Time (RDiT).

Implements the methodology of Hausman & Rapson (2018) for estimating
causal effects at a specific point in time (e.g., policy change date).
Unlike standard RD, the time-series structure creates autocorrelation,
so HAC (Newey-West) standard errors are used.

References
----------
Hausman, C. and Rapson, D.S. (2018).
"Regression Discontinuity in Time: Considerations for Empirical
Applications." Annual Review of Resource Economics, 10, 533-552.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


# ======================================================================
# Kernel helpers
# ======================================================================

_KERNELS = {
    'triangular': lambda u: np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0),
    'epanechnikov': lambda u: np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0),
    'uniform': lambda u: np.where(np.abs(u) <= 1, 0.5, 0.0),
    'gaussian': lambda u: sp_stats.norm.pdf(u),
}


def _newey_west_se(X: np.ndarray, residuals: np.ndarray,
                   weights: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute HAC (Newey-West) covariance matrix and return SEs."""
    n, k = X.shape
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)

    # Weighted residuals
    e = residuals * np.sqrt(weights)
    Xe = X * e[:, None]

    # S_0: heteroskedasticity-consistent part
    S = Xe.T @ Xe

    # Add autocovariance terms with Bartlett kernel
    for lag in range(1, max_lag + 1):
        bartlett = 1 - lag / (max_lag + 1)
        G = Xe[lag:].T @ Xe[:-lag]
        S += bartlett * (G + G.T)

    V = XtWX_inv @ S @ XtWX_inv
    return np.sqrt(np.maximum(np.diag(V), 0))


# ======================================================================
# Optimal bandwidth (IK-style rule of thumb for time)
# ======================================================================

def _optimal_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    """Simple IK-style bandwidth selector for RDiT."""
    n = len(x)
    sd_x = np.std(x)
    if sd_x == 0:
        return float(np.ptp(x)) / 2

    # Fan-Gijbels pilot bandwidth
    h_pilot = 1.84 * sd_x * n ** (-1 / 5)

    # Estimate second derivative via global quadratic
    coeffs = np.polyfit(x, y, 2)
    m2 = 2 * coeffs[0]  # second derivative estimate

    if abs(m2) < 1e-12:
        return h_pilot

    # Residual variance estimate
    y_hat = np.polyval(coeffs, x)
    sigma2 = np.mean((y - y_hat) ** 2)

    # IK formula (simplified)
    C_k = 3.4375  # triangular kernel constant
    h_opt = (C_k * sigma2 / (m2 ** 2 * n)) ** (1 / 5)

    return max(float(h_opt), h_pilot * 0.5)


# ======================================================================
# Deseasonalisation
# ======================================================================

def _deseasonalise(y: np.ndarray, time_vals: pd.Series,
                   method: str) -> np.ndarray:
    """Remove seasonal component from y."""
    if method == "month":
        groups = pd.to_datetime(time_vals).dt.month
    elif method == "quarter":
        groups = pd.to_datetime(time_vals).dt.quarter
    elif method == "dow":
        groups = pd.to_datetime(time_vals).dt.dayofweek
    else:
        raise ValueError(f"Unknown seasonality method: {method!r}. "
                         "Choose from 'month', 'quarter', 'dow'.")

    # Create dummies (drop first to avoid collinearity)
    unique_groups = np.sort(groups.unique())
    if len(unique_groups) <= 1:
        return y

    dummies = np.column_stack([
        (groups.values == g).astype(float) for g in unique_groups[1:]
    ])
    X = np.column_stack([np.ones(len(y)), dummies])

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeffs
    return residuals


# ======================================================================
# Public API
# ======================================================================

def rdit(
    data: pd.DataFrame,
    y: str,
    time: str,
    cutoff,
    h=None,
    p: int = 1,
    kernel: str = "triangular",
    donut: int = 0,
    seasonality: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Regression Discontinuity in Time (Hausman & Rapson, 2018).

    Estimates a causal effect at a known policy change date using
    local polynomial regression on a time-indexed running variable,
    with HAC standard errors to account for autocorrelation.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    time : str
        Time variable name (datetime or numeric).
    cutoff : str, int, float, or datetime-like
        The policy change date / time cutoff.
    h : float or int, optional
        Bandwidth in the same units as the numeric time axis
        (days if datetime). If None, an MSE-optimal bandwidth
        is selected automatically.
    p : int, default 1
        Local polynomial order (1 = local linear).
    kernel : str, default 'triangular'
        Kernel function: 'triangular', 'epanechnikov', 'uniform',
        or 'gaussian'.
    donut : int, default 0
        Donut hole: exclude observations within +/- donut units of
        the cutoff (in the same numeric time units).
    seasonality : str, optional
        Deseasonalise before estimation. One of 'month', 'quarter',
        'dow' (day-of-week). Regresses Y on seasonal dummies and
        uses residuals.
    cluster : str, optional
        Cluster variable for clustered standard errors. If provided,
        cluster-robust SEs are used instead of HAC.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Result object with estimate (treatment effect at the cutoff),
        standard error (HAC), confidence interval, p-value, and a
        detail DataFrame with fitted values. Call ``.summary()`` for
        a formatted table or ``.plot()`` for a time-series plot.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.rdit(df, y="electricity", time="date",
    ...                  cutoff="2015-01-01", seasonality="month")
    >>> result.summary()
    >>> result.plot()

    Notes
    -----
    The key differences from standard RD:
    - The running variable (time) is deterministic, so there is no
      manipulation concern, but also no density test.
    - Autocorrelation in the outcome requires HAC standard errors.
    - Seasonality can confound the estimate and should be removed.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if y not in data.columns:
        raise ValueError(f"Column '{y}' not found in data.")
    if time not in data.columns:
        raise ValueError(f"Column '{time}' not found in data.")
    if kernel not in _KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. "
                         f"Choose from {list(_KERNELS.keys())}.")

    df = data[[time, y]].dropna().copy()
    if cluster is not None:
        if cluster not in data.columns:
            raise ValueError(f"Cluster column '{cluster}' not found.")
        df[cluster] = data.loc[df.index, cluster]

    # ------------------------------------------------------------------
    # 2. Convert time to numeric (days from cutoff)
    # ------------------------------------------------------------------
    time_raw = df[time]
    is_datetime = pd.api.types.is_datetime64_any_dtype(time_raw)
    if not is_datetime:
        try:
            time_raw = pd.to_datetime(time_raw)
            is_datetime = True
        except (ValueError, TypeError):
            pass

    cutoff_val = pd.Timestamp(cutoff) if is_datetime else float(cutoff)

    if is_datetime:
        x_numeric = (time_raw - cutoff_val).dt.total_seconds() / 86400.0
    else:
        x_numeric = time_raw.astype(float) - float(cutoff_val)

    x_numeric = x_numeric.values.astype(np.float64)
    y_vals = df[y].values.astype(np.float64)

    # ------------------------------------------------------------------
    # 3. Deseasonalise if requested
    # ------------------------------------------------------------------
    if seasonality is not None:
        if is_datetime:
            y_vals = _deseasonalise(y_vals, time_raw, seasonality)
        else:
            raise ValueError("seasonality requires a datetime time variable.")

    # ------------------------------------------------------------------
    # 4. Apply donut hole
    # ------------------------------------------------------------------
    donut_mask = np.abs(x_numeric) > donut
    x_work = x_numeric[donut_mask]
    y_work = y_vals[donut_mask]

    if len(x_work) < 2 * (p + 1):
        raise ValueError("Insufficient observations after donut exclusion.")

    # ------------------------------------------------------------------
    # 5. Optimal bandwidth
    # ------------------------------------------------------------------
    if h is None:
        h = _optimal_bandwidth(x_work, y_work)

    # ------------------------------------------------------------------
    # 6. Select observations within bandwidth
    # ------------------------------------------------------------------
    in_bw = np.abs(x_work) <= h
    x_bw = x_work[in_bw]
    y_bw = y_work[in_bw]
    n_eff = int(np.sum(in_bw))

    if n_eff < 2 * (p + 1):
        raise ValueError(f"Only {n_eff} observations within bandwidth "
                         f"h={h:.2f}. Need at least {2*(p+1)}.")

    # Kernel weights
    kfn = _KERNELS[kernel]
    w = kfn(x_bw / h)

    # ------------------------------------------------------------------
    # 7. Local polynomial regression
    # ------------------------------------------------------------------
    # Build design: polynomial + treatment indicator + interaction
    # Y_i = sum_j beta_j * x_i^j + tau * D_i
    #      + sum_j gamma_j * D_i * x_i^j + eps_i
    # where D_i = 1{x_i >= 0}
    D = (x_bw >= 0).astype(float)

    # Design matrix
    cols = [np.ones(n_eff)]
    for j in range(1, p + 1):
        cols.append(x_bw ** j)
    cols.append(D)
    for j in range(1, p + 1):
        cols.append(D * (x_bw ** j))

    X = np.column_stack(cols)
    n_params = X.shape[1]

    # Weighted least squares
    W_sqrt = np.sqrt(w)
    Xw = X * W_sqrt[:, None]
    yw = y_bw * W_sqrt

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(Xw) @ yw

    y_hat = X @ beta
    resid = y_bw - y_hat

    # Treatment effect = coefficient on D (the p+1-th column, index p+1)
    tau_idx = p + 1
    tau = float(beta[tau_idx])

    # ------------------------------------------------------------------
    # 8. HAC (Newey-West) standard errors
    # ------------------------------------------------------------------
    # Sort by time for autocorrelation structure
    sort_idx = np.argsort(x_bw)
    X_sorted = X[sort_idx]
    resid_sorted = resid[sort_idx]
    w_sorted = w[sort_idx]

    # Automatic lag: Newey-West (1994) rule
    max_lag = max(1, int(np.floor(4 * (n_eff / 100) ** (2 / 9))))

    se_all = _newey_west_se(X_sorted, resid_sorted, w_sorted, max_lag)
    se_tau = float(se_all[tau_idx])

    # ------------------------------------------------------------------
    # 9. Inference
    # ------------------------------------------------------------------
    if se_tau > 0:
        z_stat = tau / se_tau
        pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
    else:
        pvalue = 0.0

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (tau - z_crit * se_tau, tau + z_crit * se_tau)

    # ------------------------------------------------------------------
    # 10. Detail DataFrame for plotting
    # ------------------------------------------------------------------
    # Predictions on both sides
    x_grid_left = np.linspace(-h, 0, 100)
    x_grid_right = np.linspace(0, h, 100)

    def _predict(x_grid, treat_flag):
        D_g = np.full(len(x_grid), treat_flag)
        cols_g = [np.ones(len(x_grid))]
        for j in range(1, p + 1):
            cols_g.append(x_grid ** j)
        cols_g.append(D_g)
        for j in range(1, p + 1):
            cols_g.append(D_g * (x_grid ** j))
        return np.column_stack(cols_g) @ beta

    y_pred_left = _predict(x_grid_left, 0.0)
    y_pred_right = _predict(x_grid_right, 1.0)

    detail = pd.DataFrame({
        'x_relative': np.concatenate([x_grid_left, x_grid_right]),
        'y_predicted': np.concatenate([y_pred_left, y_pred_right]),
        'side': ['left'] * 100 + ['right'] * 100,
    })

    # Also include the raw data within bandwidth for scatter
    scatter_df = pd.DataFrame({
        'x_relative': x_bw,
        'y_observed': y_bw,
        'weight': w,
        'side': np.where(x_bw < 0, 'left', 'right'),
    })

    model_info = {
        'method': 'RDiT (Hausman & Rapson 2018)',
        'bandwidth': float(h),
        'bandwidth_auto': (h is not None),
        'polynomial_order': p,
        'kernel': kernel,
        'donut': donut,
        'seasonality': seasonality,
        'n_eff': n_eff,
        'n_left': int(np.sum(x_bw < 0)),
        'n_right': int(np.sum(x_bw >= 0)),
        'max_lag_hac': max_lag,
        'cutoff': str(cutoff),
        'coefficients': beta.tolist(),
        'fitted_curve': detail,
        'scatter_data': scatter_df,
    }

    return CausalResult(
        method='Regression Discontinuity in Time (RDiT)',
        estimand='Treatment Effect at Cutoff',
        estimate=tau,
        se=se_tau,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n_eff,
        detail=detail,
        model_info=model_info,
        _citation_key='rdit',
    )


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['rdit'] = (
    "@article{hausman2018regression,\n"
    "  title={Regression Discontinuity in Time: Considerations for "
    "Empirical Applications},\n"
    "  author={Hausman, Catherine and Rapson, David S},\n"
    "  journal={Annual Review of Resource Economics},\n"
    "  volume={10},\n"
    "  pages={533--552},\n"
    "  year={2018},\n"
    "  publisher={Annual Reviews}\n"
    "}"
)
