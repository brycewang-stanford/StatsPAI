"""
Boundary Discontinuity Designs with bivariate running variables.

Implements the methodology of Cattaneo, Titiunik, and Yu (2025) for
regression discontinuity designs where treatment assignment is
determined by position relative to a known boundary curve in 2D space.

Two estimation approaches:
  - **distance-based**: project observations onto signed distance to
    boundary, then apply standard univariate local polynomial RD.
  - **location-based**: fit bivariate local polynomial on each side
    of the boundary at evaluation points along the curve.

References
----------
Cattaneo, M.D., Titiunik, R. and Yu, R. (2025).
"Boundary Discontinuity Designs." Working Paper.

Keele, L. and Titiunik, R. (2015).
"Geographic Boundaries as Regression Discontinuities."
*Political Analysis*, 23(1), 127-155.
"""

from typing import Optional, Callable, Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ..core.results import CausalResult


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['rd2d'] = (
    "@article{cattaneo2025boundary,\n"
    "  title={Boundary Discontinuity Designs},\n"
    "  author={Cattaneo, Matias D and Titiunik, Roc{\\'\\i}o and Yu, Ruoqi},\n"
    "  year={2025},\n"
    "  journal={Working Paper}\n"
    "}"
)


# ======================================================================
# Public API
# ======================================================================

def rd2d(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    treatment: str,
    boundary: Optional[Callable] = None,
    approach: str = 'distance',
    p: int = 1,
    kernel: str = 'triangular',
    h: Optional[float] = None,
    bwselect: str = 'mserd',
    eval_points: Optional[np.ndarray] = None,
    n_eval: int = 1,
    alpha: float = 0.05,
) -> CausalResult:
    """
    2D boundary regression discontinuity estimation.

    Estimates treatment effects in designs where units are assigned to
    treatment based on their position relative to a boundary curve in
    the (x1, x2) plane.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x1 : str
        First running variable (score dimension 1).
    x2 : str
        Second running variable (score dimension 2).
    treatment : str
        Binary treatment indicator (1 = treated, 0 = control).
    boundary : callable, optional
        Function ``f(x1) -> x2`` defining the boundary curve.  If None,
        the boundary is the vertical line ``x1 = 0``.
    approach : str, default 'distance'
        ``'distance'``: project onto signed distance to boundary, then
        apply univariate local polynomial RD.
        ``'location'``: fit bivariate local polynomial on each side of
        the boundary at evaluation points.
    p : int, default 1
        Polynomial order for point estimation (1 = local linear).
    kernel : str, default 'triangular'
        Kernel function: 'triangular', 'uniform', or 'epanechnikov'.
    h : float, optional
        Manual bandwidth. If None, MSE-optimal bandwidth is selected.
    bwselect : str, default 'mserd'
        Bandwidth selection method (used when ``h`` is None).
    eval_points : np.ndarray, optional
        Shape ``(k, 2)`` array of boundary evaluation points.  If None,
        points are automatically selected along the boundary.
    n_eval : int, default 1
        Number of evaluation points when ``eval_points`` is None.
        Use 1 for a single pooled effect.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Treatment effect estimate with standard errors, confidence
        intervals, and optional detail table with point-by-point
        estimates along the boundary.
    """
    if approach not in ('distance', 'location'):
        raise ValueError(
            f"approach must be 'distance' or 'location', got '{approach}'"
        )
    if kernel not in ('triangular', 'uniform', 'epanechnikov'):
        raise ValueError(
            f"kernel must be 'triangular', 'uniform', or "
            f"'epanechnikov', got '{kernel}'"
        )
    for col in [y, x1, x2, treatment]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # --- Extract and clean data ---
    Y = data[y].values.astype(float)
    X1 = data[x1].values.astype(float)
    X2 = data[x2].values.astype(float)
    T = data[treatment].values.astype(float)

    valid = np.isfinite(Y) & np.isfinite(X1) & np.isfinite(X2) & np.isfinite(T)
    Y, X1, X2, T = Y[valid], X1[valid], X2[valid], T[valid]
    n = len(Y)

    if n < 20:
        raise ValueError(f"Too few valid observations ({n}). Need at least 20.")

    treated = T == 1
    control = T == 0
    n_treated = int(treated.sum())
    n_control = int(control.sum())
    if n_treated < 5 or n_control < 5:
        raise ValueError(
            f"Too few treated ({n_treated}) or control ({n_control}) units."
        )

    if approach == 'distance':
        return _rd2d_distance(
            Y, X1, X2, T, treated, control, boundary, p, kernel,
            h, bwselect, alpha, n,
        )
    else:  # location
        return _rd2d_location(
            Y, X1, X2, T, treated, control, boundary, p, kernel,
            h, bwselect, eval_points, n_eval, alpha, n,
        )


def rd2d_bw(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    treatment: str,
    boundary: Optional[Callable] = None,
    approach: str = 'distance',
    p: int = 1,
    kernel: str = 'triangular',
) -> float:
    """
    Bandwidth selection for 2D boundary RD.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x1, x2 : str
        Running variable names.
    treatment : str
        Binary treatment indicator.
    boundary : callable, optional
        Boundary function f(x1) -> x2.  None implies x1 = 0.
    approach : str, default 'distance'
        'distance' or 'location'.
    p : int, default 1
        Polynomial order.
    kernel : str, default 'triangular'
        Kernel function.

    Returns
    -------
    float
        MSE-optimal bandwidth.
    """
    for col in [y, x1, x2, treatment]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    Y = data[y].values.astype(float)
    X1 = data[x1].values.astype(float)
    X2 = data[x2].values.astype(float)
    T = data[treatment].values.astype(float)

    valid = np.isfinite(Y) & np.isfinite(X1) & np.isfinite(X2) & np.isfinite(T)
    Y, X1, X2, T = Y[valid], X1[valid], X2[valid], T[valid]

    treated = T == 1
    control = T == 0

    if approach == 'distance':
        dist = _signed_distance(X1, X2, T, boundary)
        return _bw_mse_optimal_1d(Y, dist, p, kernel)
    else:
        return _bw_mse_optimal_2d(Y, X1, X2, T, treated, control,
                                  boundary, p, kernel)


def rd2d_plot(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    treatment: str,
    boundary: Optional[Callable] = None,
    result: Optional[CausalResult] = None,
    plot_type: str = 'scatter',
    ax=None,
    figsize: tuple = (10, 8),
) -> tuple:
    """
    2D boundary RD visualization.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x1, x2 : str
        Running variable names.
    treatment : str
        Binary treatment indicator.
    boundary : callable, optional
        Boundary function f(x1) -> x2.  None implies x1 = 0.
    result : CausalResult, optional
        Result from ``rd2d()``, used for bandwidth and effect info.
    plot_type : str, default 'scatter'
        ``'scatter'``: 2D scatter of (x1, x2) colored by treatment
        status, with boundary curve and optional bandwidth region.
        ``'heatmap'``: outcome values displayed as a heatmap with
        boundary overlay.
        ``'boundary_effects'``: treatment effect estimates along the
        boundary (requires ``result`` with multiple eval points).
    ax : matplotlib Axes, optional
        Pre-existing axes to draw on.
    figsize : tuple, default (10, 8)
        Figure size.

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    if plot_type not in ('scatter', 'heatmap', 'boundary_effects'):
        raise ValueError(
            f"plot_type must be 'scatter', 'heatmap', or "
            f"'boundary_effects', got '{plot_type}'"
        )

    for col in [y, x1, x2, treatment]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    X1 = data[x1].values.astype(float)
    X2 = data[x2].values.astype(float)
    Y = data[y].values.astype(float)
    T = data[treatment].values.astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Generate boundary curve for plotting
    x1_range = np.linspace(X1.min(), X1.max(), 300)
    if boundary is not None:
        x2_boundary = np.array([boundary(v) for v in x1_range])
    else:
        x2_boundary = None  # vertical line at x1 = 0

    if plot_type == 'scatter':
        treated_mask = T == 1
        control_mask = T == 0

        ax.scatter(X1[control_mask], X2[control_mask],
                   c='#3498DB', alpha=0.4, s=15, label='Control', zorder=2)
        ax.scatter(X1[treated_mask], X2[treated_mask],
                   c='#E74C3C', alpha=0.4, s=15, label='Treated', zorder=2)

        # Boundary
        if boundary is not None:
            ax.plot(x1_range, x2_boundary, 'k-', linewidth=2,
                    label='Boundary', zorder=3)
        else:
            ax.axvline(x=0, color='k', linewidth=2,
                       label='Boundary (x1=0)', zorder=3)

        # Bandwidth region
        if result is not None and 'bandwidth' in result.model_info:
            bw = result.model_info['bandwidth']
            if boundary is not None:
                # Draw dashed lines offset from boundary
                x2_bw_upper = x2_boundary + bw
                x2_bw_lower = x2_boundary - bw
                ax.plot(x1_range, x2_bw_upper, 'k--', linewidth=0.8,
                        alpha=0.5, label=f'h = {bw:.3f}')
                ax.plot(x1_range, x2_bw_lower, 'k--', linewidth=0.8,
                        alpha=0.5)
            else:
                ax.axvspan(-bw, bw, alpha=0.08, color='gray',
                           label=f'h = {bw:.3f}')

        ax.set_xlabel(x1, fontsize=11)
        ax.set_ylabel(x2, fontsize=11)
        ax.set_title('2D Boundary RD: Treatment Assignment', fontsize=13)
        ax.legend(fontsize=9, loc='best')

    elif plot_type == 'heatmap':
        norm = Normalize(vmin=np.nanpercentile(Y, 2),
                         vmax=np.nanpercentile(Y, 98))
        scatter = ax.scatter(X1, X2, c=Y, cmap='RdYlBu_r', norm=norm,
                             s=12, alpha=0.7, zorder=2)
        fig.colorbar(scatter, ax=ax, label=y, shrink=0.8)

        if boundary is not None:
            ax.plot(x1_range, x2_boundary, 'k-', linewidth=2.5,
                    label='Boundary', zorder=3)
        else:
            ax.axvline(x=0, color='k', linewidth=2.5,
                       label='Boundary (x1=0)', zorder=3)

        ax.set_xlabel(x1, fontsize=11)
        ax.set_ylabel(x2, fontsize=11)
        ax.set_title('2D Boundary RD: Outcome Heatmap', fontsize=13)
        ax.legend(fontsize=9, loc='best')

    elif plot_type == 'boundary_effects':
        if result is None or result.detail is None:
            raise ValueError(
                "plot_type='boundary_effects' requires a result from "
                "rd2d() with multiple eval points."
            )
        detail = result.detail
        if 'eval_x1' not in detail.columns:
            raise ValueError(
                "Result detail does not contain boundary eval points. "
                "Use approach='location' with n_eval > 1."
            )

        ax.errorbar(
            detail['eval_x1'], detail['estimate'],
            yerr=[detail['estimate'] - detail['ci_lower'],
                  detail['ci_upper'] - detail['estimate']],
            fmt='o-', color='#2C3E50', capsize=4, capthick=1.2,
            linewidth=1.5, markersize=6, zorder=3,
        )
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8,
                    alpha=0.7)

        # Mark pooled estimate
        if result.estimate is not None:
            ax.axhline(y=result.estimate, color='#E74C3C',
                       linestyle=':', linewidth=1.2, alpha=0.8,
                       label=f'Pooled = {result.estimate:.4f}')
            ax.legend(fontsize=9, loc='best')

        ax.set_xlabel(f'{x1} (along boundary)', fontsize=11)
        ax.set_ylabel('Treatment Effect', fontsize=11)
        ax.set_title('2D Boundary RD: Effects Along Boundary', fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    return fig, ax


# ======================================================================
# Distance-based approach
# ======================================================================

def _rd2d_distance(
    Y: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    T: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    boundary: Optional[Callable],
    p: int,
    kernel: str,
    h: Optional[float],
    bwselect: str,
    alpha: float,
    n: int,
) -> CausalResult:
    """Distance-based 2D RD: project to distance, run univariate RD."""
    # Compute signed distance to boundary
    dist = _signed_distance(X1, X2, T, boundary)

    # Bandwidth selection on the distance variable
    right = dist >= 0  # treated side
    left = dist < 0    # control side

    n_left = int(left.sum())
    n_right = int(right.sum())
    if n_left < p + 2 or n_right < p + 2:
        raise ValueError(
            f"Not enough observations on each side of the boundary "
            f"(left={n_left}, right={n_right}, need >= {p + 2})."
        )

    h_auto = h is None
    if h is None:
        h = _bw_mse_optimal_1d(Y, dist, p, kernel)

    # Local polynomial RD on distance
    tau, se, n_eff_l, n_eff_r = _local_poly_rd_1d(
        Y, dist, left, right, h, p, kernel
    )

    # Inference
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_stat = tau / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    ci = (tau - z_crit * se, tau + z_crit * se)

    detail = pd.DataFrame({
        'method': ['Distance-based RD'],
        'estimate': [tau],
        'se': [se],
        'z': [z_stat],
        'pvalue': [pvalue],
        'ci_lower': [ci[0]],
        'ci_upper': [ci[1]],
    })

    model_info: Dict[str, Any] = {
        'approach': 'distance',
        'polynomial_p': p,
        'kernel': kernel,
        'bandwidth': round(float(h), 6),
        'bwselect': bwselect if h_auto else 'manual',
        'n_left': n_left,
        'n_right': n_right,
        'n_effective_left': n_eff_l,
        'n_effective_right': n_eff_r,
        'boundary': 'x1=0' if boundary is None else 'custom',
    }

    return CausalResult(
        method='2D Boundary RD (distance-based)',
        estimand='Boundary RD Effect',
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='rd2d',
    )


# ======================================================================
# Location-based approach
# ======================================================================

def _rd2d_location(
    Y: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    T: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    boundary: Optional[Callable],
    p: int,
    kernel: str,
    h: Optional[float],
    bwselect: str,
    eval_points: Optional[np.ndarray],
    n_eval: int,
    alpha: float,
    n: int,
) -> CausalResult:
    """Location-based 2D RD: bivariate local polynomial at boundary."""
    # Determine evaluation points along boundary
    if eval_points is not None:
        eval_pts = np.atleast_2d(eval_points)
    else:
        eval_pts = _generate_eval_points(X1, X2, boundary, n_eval)

    # Bandwidth selection
    h_auto = h is None
    if h is None:
        h = _bw_mse_optimal_2d(Y, X1, X2, T, treated, control,
                               boundary, p, kernel)

    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Estimate effect at each evaluation point
    point_results = []
    for k in range(len(eval_pts)):
        b1, b2 = eval_pts[k, 0], eval_pts[k, 1]

        tau_k, se_k = _bivariate_local_poly_rd(
            Y, X1, X2, treated, control, b1, b2, h, p, kernel
        )

        z_k = tau_k / se_k if se_k > 0 else 0.0
        pv_k = float(2 * (1 - stats.norm.cdf(abs(z_k))))
        ci_k = (tau_k - z_crit * se_k, tau_k + z_crit * se_k)

        point_results.append({
            'eval_x1': b1,
            'eval_x2': b2,
            'estimate': tau_k,
            'se': se_k,
            'z': z_k,
            'pvalue': pv_k,
            'ci_lower': ci_k[0],
            'ci_upper': ci_k[1],
        })

    detail = pd.DataFrame(point_results)

    # Pool estimates via inverse-variance weighting
    if len(point_results) == 1:
        tau_pool = point_results[0]['estimate']
        se_pool = point_results[0]['se']
    else:
        tau_pool, se_pool = _inverse_variance_pool(
            detail['estimate'].values, detail['se'].values
        )

    z_pool = tau_pool / se_pool if se_pool > 0 else 0.0
    pv_pool = float(2 * (1 - stats.norm.cdf(abs(z_pool))))
    ci_pool = (tau_pool - z_crit * se_pool, tau_pool + z_crit * se_pool)

    model_info: Dict[str, Any] = {
        'approach': 'location',
        'polynomial_p': p,
        'kernel': kernel,
        'bandwidth': round(float(h), 6),
        'bwselect': bwselect if h_auto else 'manual',
        'n_eval_points': len(eval_pts),
        'eval_points': eval_pts.tolist(),
        'boundary': 'x1=0' if boundary is None else 'custom',
    }

    return CausalResult(
        method='2D Boundary RD (location-based)',
        estimand='Boundary RD Effect',
        estimate=tau_pool,
        se=se_pool,
        pvalue=pv_pool,
        ci=ci_pool,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='rd2d',
    )


# ======================================================================
# Distance computation helpers
# ======================================================================

def _signed_distance(
    X1: np.ndarray,
    X2: np.ndarray,
    T: np.ndarray,
    boundary: Optional[Callable],
) -> np.ndarray:
    """
    Compute signed distance from each observation to the boundary.

    Positive on the treated side, negative on the control side.
    """
    if boundary is None:
        return _signed_distance_to_vertical(X1, T, cutoff=0.0)
    else:
        return _signed_distance_to_curve(X1, X2, T, boundary)


def _signed_distance_to_vertical(
    X1: np.ndarray,
    T: np.ndarray,
    cutoff: float = 0.0,
) -> np.ndarray:
    """
    Signed distance to vertical boundary x1 = cutoff.

    For vertical boundary, distance is simply x1 - cutoff.  Sign is
    positive for treated, negative for control.
    """
    raw_dist = X1 - cutoff
    sign = np.where(T == 1, 1.0, -1.0)
    return np.abs(raw_dist) * sign


def _signed_distance_to_curve(
    X1: np.ndarray,
    X2: np.ndarray,
    T: np.ndarray,
    boundary_fn: Callable,
) -> np.ndarray:
    """
    Signed distance to an arbitrary boundary curve f(x1) -> x2.

    For each observation, numerically finds the closest point on the
    boundary and computes the Euclidean distance.  Sign is positive
    for treated, negative for control.
    """
    n = len(X1)
    dist = np.empty(n)

    # Determine search range for boundary parameter
    x1_min, x1_max = X1.min(), X1.max()
    margin = 0.1 * (x1_max - x1_min)
    search_lo = x1_min - margin
    search_hi = x1_max + margin

    for i in range(n):
        xi1, xi2 = X1[i], X2[i]

        # Minimize squared distance to boundary curve
        def sq_dist(t):
            return (xi1 - t) ** 2 + (xi2 - boundary_fn(t)) ** 2

        # Use bounded minimization with a few restarts
        best_d2 = np.inf
        # Try starting from a grid to avoid local minima
        n_starts = 5
        starts = np.linspace(search_lo, search_hi, n_starts)
        # Also try the observation's own x1 as starting point
        starts = np.append(starts, xi1)

        for s in starts:
            try:
                res = optimize.minimize_scalar(
                    sq_dist,
                    bounds=(search_lo, search_hi),
                    method='bounded',
                )
                if res.fun < best_d2:
                    best_d2 = res.fun
            except Exception:
                pass

        dist[i] = np.sqrt(max(best_d2, 0.0))

    # Apply sign: positive for treated, negative for control
    sign = np.where(T == 1, 1.0, -1.0)
    return dist * sign


# ======================================================================
# Univariate local polynomial RD (for distance approach)
# ======================================================================

def _local_poly_rd_1d(
    Y: np.ndarray,
    X: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    h: float,
    p: int,
    kernel: str,
) -> Tuple[float, float, int, int]:
    """
    Standard univariate local polynomial RD at cutoff = 0.

    Returns (tau, se, n_eff_left, n_eff_right).
    """
    beta_l, vcov_l, n_l = _wls_local_poly(Y[left], X[left], h, p, kernel)
    beta_r, vcov_r, n_r = _wls_local_poly(Y[right], X[right], h, p, kernel)

    tau = float(beta_r[0] - beta_l[0])
    se = float(np.sqrt(vcov_r[0, 0] + vcov_l[0, 0]))

    return tau, se, n_l, n_r


def _wls_local_poly(
    y: np.ndarray,
    x: np.ndarray,
    h: float,
    p: int,
    kernel: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    WLS local polynomial regression evaluated at x = 0.

    Returns (beta, vcov, n_effective).
    """
    u = x / h
    w = _kernel_fn(u, kernel)
    in_bw = np.abs(u) <= 1
    n_eff = int(in_bw.sum())

    k = p + 1
    if n_eff < k + 1:
        return np.zeros(k), np.eye(k) * 1e10, 0

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = w[in_bw]

    # Design matrix [1, x, x^2, ..., x^p]
    X_mat = np.column_stack([x_bw ** j for j in range(k)])

    sqw = np.sqrt(w_bw)
    Xw = X_mat * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        XtWX = Xw.T @ Xw
        beta = np.linalg.solve(XtWX, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        XtWX = Xw.T @ Xw

    resid = y_bw - X_mat @ beta

    # HC1 sandwich variance
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)

    corr = n_eff / (n_eff - k) if n_eff > k else 1.0
    meat = Xw.T @ np.diag(resid ** 2 * corr) @ Xw
    vcov = bread @ meat @ bread

    return beta, vcov, n_eff


# ======================================================================
# Bivariate local polynomial (for location approach)
# ======================================================================

def _bivariate_local_poly_rd(
    Y: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    b1: float,
    b2: float,
    h: float,
    p: int,
    kernel: str,
) -> Tuple[float, float]:
    """
    Estimate boundary RD effect at point (b1, b2) via bivariate
    local polynomial on each side.

    For p=1 (local linear):
        Y_i = alpha + beta1*(X1_i - b1) + beta2*(X2_i - b2) + eps_i
    with product kernel weighting.

    Returns (tau, se) where tau = alpha_R - alpha_L.
    """
    # Fit on treated (right) side
    beta_r, vcov_r, n_r = _bivariate_wls(
        Y[treated], X1[treated], X2[treated], b1, b2, h, p, kernel
    )
    # Fit on control (left) side
    beta_l, vcov_l, n_l = _bivariate_wls(
        Y[control], X1[control], X2[control], b1, b2, h, p, kernel
    )

    # Treatment effect = intercept_R - intercept_L
    tau = float(beta_r[0] - beta_l[0])
    se = float(np.sqrt(vcov_r[0, 0] + vcov_l[0, 0]))

    return tau, se


def _bivariate_wls(
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    b1: float,
    b2: float,
    h: float,
    p: int,
    kernel: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    WLS bivariate local polynomial evaluated at (b1, b2).

    Uses product kernel: K(u1) * K(u2) where u1 = (x1-b1)/h,
    u2 = (x2-b2)/h.

    For p=1: regressors are [1, (x1-b1), (x2-b2)]
    For p=2: [1, (x1-b1), (x2-b2), (x1-b1)^2, (x1-b1)*(x2-b2), (x2-b2)^2]

    Returns (beta, vcov, n_effective).
    """
    dx1 = x1 - b1
    dx2 = x2 - b2

    u1 = dx1 / h
    u2 = dx2 / h

    # Product kernel
    w1 = _kernel_fn(u1, kernel)
    w2 = _kernel_fn(u2, kernel)
    w = w1 * w2

    in_bw = (np.abs(u1) <= 1) & (np.abs(u2) <= 1)
    n_eff = int(in_bw.sum())

    # Build design matrix based on polynomial order
    cols = _bivariate_design_columns(dx1[in_bw], dx2[in_bw], p)
    k = cols.shape[1]

    if n_eff < k + 1:
        return np.zeros(k), np.eye(k) * 1e10, 0

    y_bw = y[in_bw]
    w_bw = w[in_bw]

    sqw = np.sqrt(w_bw)
    Xw = cols * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        XtWX = Xw.T @ Xw
        beta = np.linalg.solve(XtWX, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        XtWX = Xw.T @ Xw

    resid = y_bw - cols @ beta

    # HC1 sandwich variance
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)

    corr = n_eff / (n_eff - k) if n_eff > k else 1.0
    meat = Xw.T @ np.diag(resid ** 2 * corr) @ Xw
    vcov = bread @ meat @ bread

    return beta, vcov, n_eff


def _bivariate_design_columns(
    dx1: np.ndarray, dx2: np.ndarray, p: int,
) -> np.ndarray:
    """
    Build bivariate polynomial design matrix up to order p.

    p=0: [1]
    p=1: [1, dx1, dx2]
    p=2: [1, dx1, dx2, dx1^2, dx1*dx2, dx2^2]
    p=3: [1, dx1, dx2, dx1^2, dx1*dx2, dx2^2,
           dx1^3, dx1^2*dx2, dx1*dx2^2, dx2^3]
    """
    n = len(dx1)
    columns = [np.ones(n)]

    for order in range(1, p + 1):
        for j in range(order + 1):
            columns.append(dx1 ** (order - j) * dx2 ** j)

    return np.column_stack(columns)


# ======================================================================
# Bandwidth selection
# ======================================================================

def _bw_mse_optimal_1d(
    Y: np.ndarray,
    dist: np.ndarray,
    p: int,
    kernel: str,
) -> float:
    """
    MSE-optimal bandwidth for distance-based 2D RD.

    Standard univariate MSE-optimal bandwidth on the distance variable.
    """
    n = len(Y)
    left = dist < 0
    right = dist >= 0

    sd_x = np.std(dist)
    x_range = np.ptp(dist)

    # Pilot bandwidth (Silverman rule)
    h_pilot = 1.06 * sd_x * n ** (-1 / 5)
    h_pilot = max(h_pilot, 0.01 * x_range)

    # Density at zero
    n_near = np.sum(np.abs(dist) <= h_pilot)
    f_c = n_near / (2 * h_pilot * n) if (h_pilot > 0 and n > 0) else 1.0
    f_c = max(f_c, 1e-10)

    # Conditional variance from local linear residuals on each side
    sigma2_l = _residual_variance_1d(Y[left], dist[left], h_pilot, kernel)
    sigma2_r = _residual_variance_1d(Y[right], dist[right], h_pilot, kernel)

    # Second derivatives (curvature -> bias)
    h_deriv = max(np.median(np.abs(dist)), h_pilot) * 1.5
    m2_l = _second_deriv_1d(Y[left], dist[left], h_deriv, kernel)
    m2_r = _second_deriv_1d(Y[right], dist[right], h_deriv, kernel)

    C_K = _kernel_mse_constant(kernel)

    bias_sq = ((m2_r - m2_l) / 2) ** 2
    if bias_sq < 1e-12:
        h_opt = h_pilot
    else:
        h_opt = (C_K * (sigma2_l + sigma2_r) /
                 (f_c * bias_sq * n)) ** (1 / 5)

    h_opt = np.clip(h_opt, 0.02 * x_range, 0.98 * x_range)
    return float(h_opt)


def _bw_mse_optimal_2d(
    Y: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    T: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    boundary: Optional[Callable],
    p: int,
    kernel: str,
) -> float:
    """
    MSE-optimal bandwidth for location-based 2D RD.

    Uses leave-one-out cross-validation on each side to select
    a common bandwidth for the product kernel.
    """
    n = len(Y)

    # Use cross-validation on a coarse grid
    # First compute a reference scale
    scale1 = np.std(X1) if np.std(X1) > 0 else 1.0
    scale2 = np.std(X2) if np.std(X2) > 0 else 1.0
    scale = (scale1 + scale2) / 2

    h_pilot = scale * n ** (-1 / 5)

    # Grid of candidate bandwidths
    h_candidates = h_pilot * np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

    # Pick boundary centroid as evaluation point
    if boundary is not None:
        x1_med = np.median(X1)
        b1 = x1_med
        b2 = boundary(x1_med)
    else:
        b1, b2 = 0.0, np.median(X2)

    best_cv = np.inf
    best_h = h_pilot

    for h_cand in h_candidates:
        cv_score = 0.0
        for side, mask in [('treated', treated), ('control', control)]:
            y_s = Y[mask]
            x1_s = X1[mask]
            x2_s = X2[mask]
            n_s = len(y_s)
            if n_s < 10:
                continue

            # Compute product kernel weights
            dx1 = x1_s - b1
            dx2 = x2_s - b2
            u1 = dx1 / h_cand
            u2 = dx2 / h_cand
            w1 = _kernel_fn(u1, kernel)
            w2 = _kernel_fn(u2, kernel)
            w = w1 * w2
            in_bw = (np.abs(u1) <= 1) & (np.abs(u2) <= 1)

            if in_bw.sum() < 5:
                cv_score += 1e10
                continue

            # Simple LOO-CV approximation via hat matrix
            cols = _bivariate_design_columns(dx1[in_bw], dx2[in_bw], p)
            y_bw = y_s[in_bw]
            w_bw = w[in_bw]

            sqw = np.sqrt(w_bw)
            Xw = cols * sqw[:, np.newaxis]
            yw = y_bw * sqw

            try:
                XtWX = Xw.T @ Xw
                XtWX_inv = np.linalg.inv(XtWX)
                H = Xw @ XtWX_inv @ Xw.T
                resid = yw - H @ yw
                h_diag = np.diag(H)
                h_diag = np.clip(h_diag, 0, 0.999)
                loo_resid = resid / (1 - h_diag)
                cv_score += float(np.mean(loo_resid ** 2))
            except np.linalg.LinAlgError:
                cv_score += 1e10

        if cv_score < best_cv:
            best_cv = cv_score
            best_h = h_cand

    return float(best_h)


# ======================================================================
# Evaluation point generation
# ======================================================================

def _generate_eval_points(
    X1: np.ndarray,
    X2: np.ndarray,
    boundary: Optional[Callable],
    n_eval: int,
) -> np.ndarray:
    """
    Generate evaluation points along the boundary.

    Spreads points evenly along the boundary within the data range.
    """
    if n_eval < 1:
        n_eval = 1

    if boundary is None:
        # Boundary is x1 = 0
        if n_eval == 1:
            return np.array([[0.0, np.median(X2)]])
        else:
            x2_lo = np.percentile(X2, 10)
            x2_hi = np.percentile(X2, 90)
            x2_grid = np.linspace(x2_lo, x2_hi, n_eval)
            return np.column_stack([np.zeros(n_eval), x2_grid])
    else:
        # Sample along boundary within data range
        x1_lo = np.percentile(X1, 10)
        x1_hi = np.percentile(X1, 90)
        if n_eval == 1:
            x1_mid = (x1_lo + x1_hi) / 2
            return np.array([[x1_mid, boundary(x1_mid)]])
        else:
            x1_grid = np.linspace(x1_lo, x1_hi, n_eval)
            x2_grid = np.array([boundary(v) for v in x1_grid])
            return np.column_stack([x1_grid, x2_grid])


# ======================================================================
# Inverse-variance pooling
# ======================================================================

def _inverse_variance_pool(
    estimates: np.ndarray,
    se: np.ndarray,
) -> Tuple[float, float]:
    """
    Inverse-variance weighted pooling of multiple estimates.

    Returns (pooled_estimate, pooled_se).
    """
    # Guard against zero/tiny SEs
    se = np.maximum(se, 1e-10)
    weights = 1.0 / se ** 2
    w_sum = weights.sum()

    if w_sum < 1e-20:
        return float(np.mean(estimates)), float(np.mean(se))

    pooled = float(np.sum(weights * estimates) / w_sum)
    pooled_se = float(np.sqrt(1.0 / w_sum))

    return pooled, pooled_se


# ======================================================================
# Bandwidth helpers (univariate)
# ======================================================================

def _residual_variance_1d(
    y: np.ndarray, x: np.ndarray, h: float, kernel: str,
) -> float:
    """Conditional variance at x=0 from local linear residuals."""
    if len(y) == 0:
        return 1.0
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 5:
        return float(np.var(y)) if len(y) > 0 else 1.0

    y_bw, x_bw = y[in_bw], x[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    X_mat = np.column_stack([np.ones(len(x_bw)), x_bw])
    sqw = np.sqrt(w_bw)
    Xw = X_mat * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        resid = y_bw - X_mat @ beta
        return float(np.average(resid ** 2, weights=w_bw))
    except Exception:
        return float(np.var(y_bw))


def _second_deriv_1d(
    y: np.ndarray, x: np.ndarray, h: float, kernel: str,
) -> float:
    """Estimate m''(0) via local cubic regression."""
    if len(y) == 0:
        return 0.0
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 6:
        return 0.0

    y_bw, x_bw = y[in_bw], x[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    X_mat = np.column_stack([x_bw ** j for j in range(4)])
    sqw = np.sqrt(w_bw)
    Xw = X_mat * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        return float(2 * beta[2])
    except Exception:
        return 0.0


# ======================================================================
# Kernel functions
# ======================================================================

def _kernel_fn(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel K(u), supported on |u| <= 1."""
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
