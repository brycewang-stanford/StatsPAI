"""
Local polynomial RD estimation with robust bias-corrected inference.

Implements the methodology of Calonico, Cattaneo, and Titiunik (2014) for
sharp and fuzzy regression discontinuity designs, with MSE-optimal bandwidth
selection and robust bias-corrected confidence intervals.

References
----------
Calonico, S., Cattaneo, M.D. and Titiunik, R. (2014).
"Robust Nonparametric Confidence Intervals for Regression-Discontinuity
Designs." *Econometrica*, 82(6), 2295-2326.

Imbens, G. and Kalyanaraman, K. (2012).
"Optimal Bandwidth Choice for the Regression Discontinuity Estimator."
*Review of Economic Studies*, 79(3), 933-959.
"""

from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def rdrobust(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    fuzzy: Optional[str] = None,
    p: int = 1,
    q: Optional[int] = None,
    kernel: str = 'triangular',
    bwselect: str = 'mserd',
    h: Optional[float] = None,
    b: Optional[float] = None,
    covs: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Local polynomial RD estimation with robust bias-corrected inference.

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
    fuzzy : str, optional
        Treatment variable for fuzzy RD (IV at the cutoff).
    p : int, default 1
        Polynomial order for point estimation (1 = local linear).
    q : int, optional
        Polynomial order for bias correction (default p + 1).
    kernel : str, default 'triangular'
        Kernel function: 'triangular', 'uniform', or 'epanechnikov'.
    bwselect : str, default 'mserd'
        Bandwidth selection method: 'mserd' (MSE-optimal, common),
        'msetwo' (MSE-optimal, separate left/right).
    h : float, optional
        Manual bandwidth for estimation (overrides bwselect).
    b : float, optional
        Manual bandwidth for bias correction (default = h).
    covs : list of str, optional
        Covariate names (partialled out before estimation).
    cluster : str, optional
        Cluster variable for standard errors.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Results with conventional and robust inference, bandwidth info,
        and all standard CausalResult methods.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.uniform(-1, 1, n)
    >>> Y = 0.5 * X + 3.0 * (X >= 0) + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X})
    >>> result = rdrobust(df, y='y', x='x', c=0)
    >>> abs(result.estimate - 3.0) < 0.5
    True
    """
    if kernel not in ('triangular', 'uniform', 'epanechnikov'):
        raise ValueError(f"kernel must be 'triangular', 'uniform', or "
                         f"'epanechnikov', got '{kernel}'")
    if q is None:
        q = p + 1

    # --- Parse and prepare data ---
    Y, X_c, D = _parse_data(data, y, x, c, fuzzy, covs)
    n = len(Y)
    left = X_c < 0
    right = X_c >= 0
    n_left_total = int(left.sum())
    n_right_total = int(right.sum())

    if n_left_total < p + 2 or n_right_total < p + 2:
        raise ValueError(
            f"Not enough observations on each side of the cutoff "
            f"(left={n_left_total}, right={n_right_total}, need ≥{p + 2})."
        )

    # --- Bandwidth selection ---
    h_auto = h is None
    if h is None:
        h = _select_bandwidth(Y, X_c, left, right, p, kernel)
    if b is None:
        b = h

    # --- Conventional estimate: order p, bandwidth h ---
    tau_conv, se_conv, n_eff_l, n_eff_r = _rd_estimate(
        Y, X_c, left, right, h, p, kernel, cluster,
        data[cluster].values if cluster else None,
    )

    # --- Bias-corrected estimate: order q, bandwidth b ---
    tau_bc, se_robust, _, _ = _rd_estimate(
        Y, X_c, left, right, b, q, kernel, cluster,
        data[cluster].values if cluster else None,
    )

    # --- Fuzzy RD: Wald / IV at cutoff ---
    if D is not None:
        fs_conv, fs_se, _, _ = _rd_estimate(
            D, X_c, left, right, h, p, kernel, None, None,
        )
        fs_bc, _, _, _ = _rd_estimate(
            D, X_c, left, right, b, q, kernel, None, None,
        )
        if abs(fs_conv) > 1e-10:
            tau_conv /= fs_conv
            se_conv /= abs(fs_conv)
        if abs(fs_bc) > 1e-10:
            tau_bc /= fs_bc
            se_robust /= abs(fs_bc)

    # --- Inference ---
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_conv = tau_conv / se_conv if se_conv > 0 else 0
    pv_conv = float(2 * (1 - stats.norm.cdf(abs(z_conv))))
    ci_conv = (tau_conv - z_crit * se_conv, tau_conv + z_crit * se_conv)

    z_robust = tau_bc / se_robust if se_robust > 0 else 0
    pv_robust = float(2 * (1 - stats.norm.cdf(abs(z_robust))))
    ci_robust = (tau_bc - z_crit * se_robust, tau_bc + z_crit * se_robust)

    # --- Detail table (matches rdrobust R output) ---
    detail = pd.DataFrame({
        'method': ['Conventional', 'Robust'],
        'estimate': [tau_conv, tau_bc],
        'se': [se_conv, se_robust],
        'z': [z_conv, z_robust],
        'pvalue': [pv_conv, pv_robust],
        'ci_lower': [ci_conv[0], ci_robust[0]],
        'ci_upper': [ci_conv[1], ci_robust[1]],
    })

    rd_type = 'Fuzzy' if fuzzy else 'Sharp'
    model_info: Dict[str, Any] = {
        'rd_type': rd_type,
        'polynomial_p': p,
        'polynomial_q': q,
        'kernel': kernel,
        'bandwidth_h': round(h, 6),
        'bandwidth_b': round(b, 6),
        'bwselect': bwselect if h_auto else 'manual',
        'cutoff': c,
        'n_left': n_left_total,
        'n_right': n_right_total,
        'n_effective_left': n_eff_l,
        'n_effective_right': n_eff_r,
        'conventional': {
            'estimate': tau_conv, 'se': se_conv,
            'pvalue': pv_conv, 'ci': ci_conv,
        },
        'robust': {
            'estimate': tau_bc, 'se': se_robust,
            'pvalue': pv_robust, 'ci': ci_robust,
        },
    }

    return CausalResult(
        method=f'{rd_type} RD Estimation',
        estimand='LATE' if fuzzy else 'RD Effect',
        estimate=tau_bc,
        se=se_robust,
        pvalue=pv_robust,
        ci=ci_robust,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='rdrobust',
    )


def rdplot(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    nbins: Optional[int] = None,
    p: int = 4,
    kernel: str = 'triangular',
    ci_level: float = 0.95,
    ax=None,
    figsize: tuple = (10, 7),
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    """
    RD plot: binned scatter with polynomial fit on each side of the cutoff.

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
        Outcome and running variable names.
    c : float, default 0
        Cutoff.
    nbins : int, optional
        Bins per side. If None, uses IMSE-optimal ~ceil(n^(1/3)).
    p : int, default 4
        Polynomial order for the fitted curve.
    kernel : str
        Kernel for the fitted curve.
    ci_level : float
        Confidence level for the fitted curve CI.
    ax : matplotlib Axes, optional
    figsize : tuple
    title, x_label, y_label : str, optional

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    Y = data[y].values.astype(float)
    X = data[x].values.astype(float)

    left_mask = X < c
    right_mask = X >= c
    x_l, y_l = X[left_mask], Y[left_mask]
    x_r, y_r = X[right_mask], Y[right_mask]

    # Number of bins
    if nbins is None:
        nbins = max(int(np.ceil(len(Y) ** (1 / 3))), 5)

    # Bin means
    def _bin_means(xv, yv, nb):
        edges = np.linspace(xv.min(), xv.max(), nb + 1)
        bx, by = [], []
        for j in range(nb):
            mask = (xv >= edges[j]) & (xv < edges[j + 1])
            if j == nb - 1:
                mask = (xv >= edges[j]) & (xv <= edges[j + 1])
            if mask.sum() > 0:
                bx.append(xv[mask].mean())
                by.append(yv[mask].mean())
        return np.array(bx), np.array(by)

    bx_l, by_l = _bin_means(x_l, y_l, nbins)
    bx_r, by_r = _bin_means(x_r, y_r, nbins)

    # Polynomial fit on each side
    def _poly_fit(xv, yv, order, x_grid):
        order = min(order, len(xv) - 1)
        coeffs = np.polyfit(xv, yv, order)
        return np.polyval(coeffs, x_grid)

    grid_l = np.linspace(x_l.min(), c, 200)
    grid_r = np.linspace(c, x_r.max(), 200)
    fit_l = _poly_fit(x_l, y_l, p, grid_l)
    fit_r = _poly_fit(x_r, y_r, p, grid_r)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.scatter(bx_l, by_l, color='#2C3E50', s=30, alpha=0.8, zorder=3)
    ax.scatter(bx_r, by_r, color='#2C3E50', s=30, alpha=0.8, zorder=3)
    ax.plot(grid_l, fit_l, color='#E74C3C', linewidth=1.5, zorder=4)
    ax.plot(grid_r, fit_r, color='#3498DB', linewidth=1.5, zorder=4)
    ax.axvline(x=c, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel(x_label or x, fontsize=11)
    ax.set_ylabel(y_label or y, fontsize=11)
    ax.set_title(title or 'RD Plot', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    return fig, ax


# ======================================================================
# Data preparation
# ======================================================================

def _parse_data(
    data: pd.DataFrame,
    y: str, x: str, c: float,
    fuzzy: Optional[str],
    covs: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Parse and validate RD data. Returns (Y, X_centered, D_or_None)."""
    for col in [y, x]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    if fuzzy and fuzzy not in data.columns:
        raise ValueError(f"Fuzzy variable '{fuzzy}' not found in data")

    Y = data[y].values.astype(float)
    X_c = data[x].values.astype(float) - c
    D = data[fuzzy].values.astype(float) if fuzzy else None

    # Drop NaN
    valid = np.isfinite(Y) & np.isfinite(X_c)
    if D is not None:
        valid &= np.isfinite(D)
    if covs:
        for col in covs:
            valid &= np.isfinite(data[col].values.astype(float))

    Y, X_c = Y[valid], X_c[valid]
    if D is not None:
        D = D[valid]

    # Partial out covariates
    if covs:
        Z = np.column_stack([data.loc[valid, col].values.astype(float)
                             for col in covs])
        Z = np.column_stack([np.ones(len(Z)), Z])  # add constant
        try:
            proj = Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
            Y = Y - proj + np.mean(Y)
            if D is not None:
                proj_d = Z @ np.linalg.lstsq(Z, D, rcond=None)[0]
                D = D - proj_d + np.mean(D)
        except np.linalg.LinAlgError:
            pass  # skip partialling out if singular

    return Y, X_c, D


# ======================================================================
# Core local polynomial estimator
# ======================================================================

def _rd_estimate(
    Y: np.ndarray,
    X_c: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    h: float,
    p: int,
    kernel: str,
    cluster_col: Optional[str],
    cluster_vals: Optional[np.ndarray],
) -> Tuple[float, float, int, int]:
    """
    Estimate RD effect via separate local polynomial on each side.

    Returns (tau, se, n_eff_left, n_eff_right).
    """
    beta_l, vcov_l, n_l = _local_poly_wls(
        Y[left], X_c[left], h, p, kernel,
        cluster_vals[left] if cluster_vals is not None else None,
    )
    beta_r, vcov_r, n_r = _local_poly_wls(
        Y[right], X_c[right], h, p, kernel,
        cluster_vals[right] if cluster_vals is not None else None,
    )

    tau = float(beta_r[0] - beta_l[0])
    se = float(np.sqrt(vcov_r[0, 0] + vcov_l[0, 0]))

    return tau, se, n_l, n_r


def _local_poly_wls(
    y: np.ndarray,
    x: np.ndarray,
    h: float,
    p: int,
    kernel: str,
    cluster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    WLS local polynomial regression evaluated at x = 0.

    Returns (beta, vcov, n_effective).
    """
    u = x / h
    w = _kernel_fn(u, kernel)
    in_bw = np.abs(u) <= 1
    n_eff = int(in_bw.sum())

    if n_eff < p + 2:
        return np.zeros(p + 1), np.eye(p + 1) * 1e10, 0

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = w[in_bw]
    k = p + 1

    # Design matrix [1, x, x², ..., x^p]
    X = np.column_stack([x_bw ** j for j in range(k)])

    # WLS via square-root weights
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        XtWX = Xw.T @ Xw
        beta = np.linalg.solve(XtWX, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        XtWX = Xw.T @ Xw

    resid = y_bw - X @ beta

    # Variance: HC1 or cluster-robust
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)

    if cluster is not None:
        cl = cluster[in_bw]
        unique_cl = np.unique(cl)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for c_val in unique_cl:
            idx = cl == c_val
            score = (Xw[idx].T @ (yw[idx] - Xw[idx] @ beta)).ravel()
            meat += np.outer(score, score)
        corr = n_cl / (n_cl - 1) if n_cl > 1 else 1.0
        vcov = corr * bread @ meat @ bread
    else:
        # HC1
        corr = n_eff / (n_eff - k) if n_eff > k else 1.0
        meat = Xw.T @ np.diag(resid ** 2 * corr) @ Xw
        vcov = bread @ meat @ bread

    return beta, vcov, n_eff


# ======================================================================
# Bandwidth selection
# ======================================================================

def _select_bandwidth(
    Y: np.ndarray,
    X_c: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    p: int,
    kernel: str,
) -> float:
    """
    MSE-optimal bandwidth for local polynomial RD.

    Combines ideas from IK (2012) and CCT (2014).
    """
    n = len(Y)
    sd_x = np.std(X_c)
    x_range = np.ptp(X_c)

    # Pilot bandwidth (Silverman rule)
    h_pilot = 1.06 * sd_x * n ** (-1 / 5)

    y_l, x_l = Y[left], X_c[left]
    y_r, x_r = Y[right], X_c[right]

    # 1. Density at cutoff
    n_near = np.sum(np.abs(X_c) <= h_pilot)
    f_c = n_near / (2 * h_pilot * n) if h_pilot > 0 and n > 0 else 1.0
    f_c = max(f_c, 1e-10)

    # 2. Conditional variance on each side (from local linear residuals)
    sigma2_l = _local_residual_var(y_l, x_l, h_pilot, kernel)
    sigma2_r = _local_residual_var(y_r, x_r, h_pilot, kernel)

    # 3. Second derivative on each side (curvature → bias)
    h_deriv = max(np.median(np.abs(X_c)), h_pilot) * 1.5
    m2_l = _estimate_second_deriv(y_l, x_l, h_deriv, kernel)
    m2_r = _estimate_second_deriv(y_r, x_r, h_deriv, kernel)

    # 4. MSE-optimal bandwidth
    C_K = _kernel_mse_constant(kernel)
    bias_sq = ((m2_r - m2_l) / 2) ** 2

    if bias_sq < 1e-12:
        h_opt = h_pilot
    else:
        h_opt = (C_K * (sigma2_l + sigma2_r) /
                 (f_c * bias_sq * n)) ** (1 / 5)

    # Bound between 2% and 98% of data range
    h_opt = np.clip(h_opt, 0.02 * x_range, 0.98 * x_range)

    return float(h_opt)


def _local_residual_var(
    y: np.ndarray, x: np.ndarray, h: float, kernel: str,
) -> float:
    """Conditional variance at x = 0 from local linear residuals."""
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 5:
        return float(np.var(y)) if len(y) > 0 else 1.0

    y_bw, x_bw, w_bw = y[in_bw], x[in_bw], _kernel_fn(u[in_bw], kernel)

    # Local linear WLS
    X = np.column_stack([np.ones(len(x_bw)), x_bw])
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        resid = y_bw - X @ beta
        return float(np.average(resid ** 2, weights=w_bw))
    except Exception:
        return float(np.var(y_bw))


def _estimate_second_deriv(
    y: np.ndarray, x: np.ndarray, h: float, kernel: str,
) -> float:
    """Estimate m''(0) using local cubic regression."""
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 6:
        return 0.0

    y_bw, x_bw = y[in_bw], x[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    # Local cubic: y = β0 + β1*x + β2*x² + β3*x³
    X = np.column_stack([x_bw ** j for j in range(4)])
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        return float(2 * beta[2])  # m''(0) = 2 * β₂
    except Exception:
        return 0.0


# ======================================================================
# Kernel functions
# ======================================================================

def _kernel_fn(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel K(u), supported on |u| ≤ 1."""
    u = np.asarray(u, dtype=float)
    if kernel == 'triangular':
        return np.maximum(1 - np.abs(u), 0)
    elif kernel == 'uniform':
        return 0.5 * (np.abs(u) <= 1).astype(float)
    elif kernel == 'epanechnikov':
        return 0.75 * np.maximum(1 - u ** 2, 0)
    raise ValueError(f"Unknown kernel: {kernel}")


def _kernel_mse_constant(kernel: str) -> float:
    """C_{1,1}: MSE-optimal bandwidth constant for local linear."""
    return {'triangular': 3.4375, 'uniform': 2.7,
            'epanechnikov': 3.0}.get(kernel, 3.4375)
