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
from math import factorial

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
    deriv: int = 0,
    p: int = 1,
    q: Optional[int] = None,
    kernel: str = 'triangular',
    bwselect: str = 'mserd',
    h: Optional[float] = None,
    b: Optional[float] = None,
    covs: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    donut: float = 0,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Local polynomial RD estimation with robust bias-corrected inference.

    Supports sharp RD, fuzzy RD, regression kink design (RKD), and
    donut-hole RD through a unified interface.

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
    deriv : int, default 0
        Derivative of the regression function to estimate.
        0 = standard RD (jump in level), 1 = regression kink design
        (change in slope). See Card & Lee (2008).
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
    donut : float, default 0
        Donut-hole radius: observations with |x - c| <= donut are
        excluded. Useful when manipulation near the cutoff is suspected.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Results with conventional and robust inference, bandwidth info,
        and all standard CausalResult methods.

    Examples
    --------
    Sharp RD:

    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.uniform(-1, 1, n)
    >>> Y = 0.5 * X + 3.0 * (X >= 0) + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X})
    >>> result = rdrobust(df, y='y', x='x', c=0)
    >>> abs(result.estimate - 3.0) < 0.5
    True

    Donut-hole RD (exclude observations within 0.05 of cutoff):

    >>> result = rdrobust(df, y='y', x='x', c=0, donut=0.05)

    Regression Kink Design (estimate change in slope):

    >>> result = rdrobust(df, y='y', x='x', c=0, deriv=1)
    """
    if kernel not in ('triangular', 'uniform', 'epanechnikov'):
        raise ValueError(f"kernel must be 'triangular', 'uniform', or "
                         f"'epanechnikov', got '{kernel}'")
    if deriv < 0:
        raise ValueError(f"deriv must be non-negative, got {deriv}")
    if donut < 0:
        raise ValueError(f"donut must be non-negative, got {donut}")
    # For RKD (deriv >= 1), polynomial order must be at least deriv + 1
    if deriv > 0 and p < deriv + 1:
        p = deriv + 1
    if q is None:
        q = p + 1

    # --- Parse and prepare data ---
    Y, X_c, D = _parse_data(data, y, x, c, fuzzy, covs)

    # --- Donut hole: exclude observations within donut radius ---
    if donut > 0:
        keep = np.abs(X_c) > donut
        if keep.sum() < 10:
            raise ValueError(
                f"donut={donut} excludes too many observations "
                f"({(~keep).sum()} dropped, {keep.sum()} remain)."
            )
        Y, X_c = Y[keep], X_c[keep]
        if D is not None:
            D = D[keep]

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
        h = _select_bandwidth(Y, X_c, left, right, p, kernel, bwselect)
    if b is None:
        b = h  # bias-correction bandwidth mirrors estimation bandwidth

    # --- Cluster values (handle donut filtering) ---
    if cluster:
        cl_vals_all = data[cluster].values
        if donut > 0:
            X_raw = data[x].values.astype(float) - c
            cl_vals_all = cl_vals_all[np.abs(X_raw) > donut]
    else:
        cl_vals_all = None

    # --- Conventional estimate: order p, bandwidth h ---
    tau_conv, se_conv, n_eff_l, n_eff_r = _rd_estimate(
        Y, X_c, left, right, h, p, kernel, cluster,
        cl_vals_all, deriv=deriv,
    )

    # --- Bias-corrected estimate: order q, bandwidth b ---
    tau_bc, se_robust, _, _ = _rd_estimate(
        Y, X_c, left, right, b, q, kernel, cluster,
        cl_vals_all, deriv=deriv,
    )

    # --- Fuzzy RD: Wald / IV at cutoff ---
    if D is not None:
        fs_conv, fs_se, _, _ = _rd_estimate(
            D, X_c, left, right, h, p, kernel, None, None,
            deriv=deriv,
        )
        fs_bc, _, _, _ = _rd_estimate(
            D, X_c, left, right, b, q, kernel, None, None,
            deriv=deriv,
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

    if deriv >= 1:
        rd_type = 'Kink'
    elif fuzzy:
        rd_type = 'Fuzzy'
    else:
        rd_type = 'Sharp'

    def _round_bw(bw):
        if isinstance(bw, tuple):
            return (round(bw[0], 6), round(bw[1], 6))
        return round(bw, 6)

    model_info: Dict[str, Any] = {
        'rd_type': rd_type,
        'deriv': deriv,
        'donut': donut,
        'polynomial_p': p,
        'polynomial_q': q,
        'kernel': kernel,
        'bandwidth_h': _round_bw(h),
        'bandwidth_b': _round_bw(b),
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

    if deriv >= 1:
        estimand_str = 'RKD Effect (change in slope)'
    elif fuzzy:
        estimand_str = 'LATE'
    else:
        estimand_str = 'RD Effect'

    return CausalResult(
        method=f'{rd_type} RD Estimation',
        estimand=estimand_str,
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
    shade_ci: bool = True,
    donut: float = 0,
    show_bw: bool = False,
    h: Optional[float] = None,
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
    ci_level : float, default 0.95
        Confidence level for CI bands.
    shade_ci : bool, default True
        Show confidence interval bands around the polynomial fit.
    donut : float, default 0
        If > 0, shades the donut region |x - c| <= donut.
    show_bw : bool, default False
        If True, shades the bandwidth window. Requires `h` or
        auto-computes from rdrobust.
    h : float, optional
        Bandwidth to display. If None and show_bw=True, auto-computes.
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

    # Polynomial fit with CI on each side
    def _poly_fit_ci(xv, yv, order, x_grid, level):
        order = min(order, len(xv) - 1)
        coeffs = np.polyfit(xv, yv, order)
        fit = np.polyval(coeffs, x_grid)
        # SE via residual variance + leverage
        resid = yv - np.polyval(coeffs, xv)
        sigma2 = np.sum(resid ** 2) / max(len(xv) - order - 1, 1)
        # Design matrix at grid points
        V = np.column_stack([x_grid ** j for j in range(order, -1, -1)])
        V_data = np.column_stack([xv ** j for j in range(order, -1, -1)])
        try:
            cov_beta = sigma2 * np.linalg.inv(V_data.T @ V_data)
            se = np.sqrt(np.sum((V @ cov_beta) * V, axis=1))
        except np.linalg.LinAlgError:
            se = np.full(len(x_grid), np.sqrt(sigma2))
        z = stats.norm.ppf(1 - (1 - level) / 2)
        return fit, fit - z * se, fit + z * se

    grid_l = np.linspace(x_l.min(), c, 200)
    grid_r = np.linspace(c, x_r.max(), 200)
    fit_l, ci_lo_l, ci_hi_l = _poly_fit_ci(x_l, y_l, p, grid_l, ci_level)
    fit_r, ci_lo_r, ci_hi_r = _poly_fit_ci(x_r, y_r, p, grid_r, ci_level)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Bandwidth window shading
    if show_bw:
        if h is None:
            try:
                r = rdrobust(data, y=y, x=x, c=c, p=1)
                h = r.model_info['bandwidth_h']
            except Exception:
                h = None
        if h is not None:
            bw_h = h[0] if isinstance(h, tuple) else h
            ax.axvspan(c - bw_h, c + bw_h, alpha=0.06, color='#3498DB',
                       label=f'Bandwidth h = {bw_h:.3f}')

    # Donut hole shading
    if donut > 0:
        ax.axvspan(c - donut, c + donut, alpha=0.12, color='#E74C3C',
                   label=f'Donut ±{donut}', zorder=1)

    # CI bands
    if shade_ci:
        ax.fill_between(grid_l, ci_lo_l, ci_hi_l,
                        color='#E74C3C', alpha=0.12, zorder=2)
        ax.fill_between(grid_r, ci_lo_r, ci_hi_r,
                        color='#3498DB', alpha=0.12, zorder=2)

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
    if donut > 0 or show_bw:
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()

    return fig, ax


def rdplotdensity(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    p: int = 2,
    n_grid: int = 50,
    h: Optional[float] = None,
    ci_level: float = 0.95,
    hist: bool = True,
    nbins: int = 30,
    ax=None,
    figsize: tuple = (10, 7),
    title: Optional[str] = None,
):
    """
    Density discontinuity plot at the RD cutoff.

    Visualizes whether the running variable density is continuous at the
    cutoff. Combines a histogram with local polynomial density estimates
    and confidence intervals on each side.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Running variable.
    c : float, default 0
        Cutoff.
    p : int, default 2
        Polynomial order for density estimation.
    n_grid : int, default 50
        Grid points per side for the density curve.
    h : float, optional
        Bandwidth. If None, auto-selects.
    ci_level : float, default 0.95
        Confidence level for CI bands.
    hist : bool, default True
        Overlay histogram.
    nbins : int, default 30
        Number of histogram bins per side.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    X = data[x].values.astype(float)
    X = X[np.isfinite(X)]
    n = len(X)

    x_left = X[X < c]
    x_right = X[X >= c]

    # Auto bandwidth
    sd = np.std(X)
    if h is None:
        h_l = 1.06 * np.std(x_left) * len(x_left) ** (-1 / (2 * p + 3))
        h_r = 1.06 * np.std(x_right) * len(x_right) ** (-1 / (2 * p + 3))
    else:
        h_l = h_r = h

    # Grid points
    grid_l = np.linspace(max(c - 3 * h_l, x_left.min()), c, n_grid)
    grid_r = np.linspace(c, min(c + 3 * h_r, x_right.max()), n_grid)

    # Density estimation via KDE-like local polynomial
    def _density_at_grid(x_data, grid_pts, bw):
        densities = []
        ses = []
        for g in grid_pts:
            u = (x_data - g) / bw
            in_bw = np.abs(u) <= 1
            n_bw = in_bw.sum()
            if n_bw < 5:
                densities.append(np.nan)
                ses.append(np.nan)
                continue
            # Kernel density (triangular kernel)
            w = np.maximum(1 - np.abs(u[in_bw]), 0)
            f_hat = w.sum() / (bw * len(x_data))
            se_hat = np.sqrt(f_hat / (bw * len(x_data))) if f_hat > 0 else 0
            densities.append(f_hat)
            ses.append(se_hat)
        return np.array(densities), np.array(ses)

    f_l, se_l = _density_at_grid(x_left, grid_l, h_l)
    f_r, se_r = _density_at_grid(x_right, grid_r, h_r)

    z = stats.norm.ppf(1 - (1 - ci_level) / 2)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Histogram
    if hist:
        all_range = (X.min(), X.max())
        ax.hist(x_left, bins=nbins, density=True, alpha=0.2,
                color='#E74C3C', range=(all_range[0], c), label=None)
        ax.hist(x_right, bins=nbins, density=True, alpha=0.2,
                color='#3498DB', range=(c, all_range[1]), label=None)

    # Density curves with CI
    valid_l = np.isfinite(f_l)
    valid_r = np.isfinite(f_r)

    ax.plot(grid_l[valid_l], f_l[valid_l], color='#E74C3C',
            linewidth=2, label='Left of cutoff')
    ax.fill_between(grid_l[valid_l],
                     (f_l - z * se_l)[valid_l],
                     (f_l + z * se_l)[valid_l],
                     color='#E74C3C', alpha=0.15)

    ax.plot(grid_r[valid_r], f_r[valid_r], color='#3498DB',
            linewidth=2, label='Right of cutoff')
    ax.fill_between(grid_r[valid_r],
                     (f_r - z * se_r)[valid_r],
                     (f_r + z * se_r)[valid_r],
                     color='#3498DB', alpha=0.15)

    ax.axvline(x=c, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel(x, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title or 'Density Discontinuity at Cutoff', fontsize=13)
    ax.legend(fontsize=10)
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
    h,
    p: int,
    kernel: str,
    cluster_col: Optional[str],
    cluster_vals: Optional[np.ndarray],
    deriv: int = 0,
) -> Tuple[float, float, int, int]:
    """
    Estimate RD effect via separate local polynomial on each side.

    Parameters
    ----------
    h : float or tuple of (float, float)
        Bandwidth. If tuple, (h_left, h_right) for separate bandwidths.
    deriv : int
        Which derivative to extract. 0 = intercept (standard RD),
        1 = first derivative (regression kink design), etc.

    Returns (tau, se, n_eff_left, n_eff_right).
    """
    if isinstance(h, tuple):
        h_l, h_r = h
    else:
        h_l = h_r = h

    beta_l, vcov_l, n_l = _local_poly_wls(
        Y[left], X_c[left], h_l, p, kernel,
        cluster_vals[left] if cluster_vals is not None else None,
    )
    beta_r, vcov_r, n_r = _local_poly_wls(
        Y[right], X_c[right], h_r, p, kernel,
        cluster_vals[right] if cluster_vals is not None else None,
    )

    # For deriv-th derivative: coefficient is beta[deriv] * deriv!
    d = min(deriv, len(beta_r) - 1, len(beta_l) - 1)
    scale = float(factorial(d)) if d > 0 else 1.0
    tau = float((beta_r[d] - beta_l[d]) * scale)
    se = float(np.sqrt((vcov_r[d, d] + vcov_l[d, d])) * scale)

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
    bwselect: str = 'mserd',
) -> 'float | Tuple[float, float]':
    """
    MSE-optimal bandwidth for local polynomial RD.

    Combines ideas from IK (2012) and CCT (2014).

    Parameters
    ----------
    bwselect : str
        'mserd' → single common bandwidth.
        'msetwo' → separate (h_left, h_right) bandwidths.

    Returns
    -------
    float or tuple of (float, float)
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

    C_K = _kernel_mse_constant(kernel)

    if bwselect == 'msetwo':
        # Separate MSE-optimal bandwidth for each side
        h_opt_l = _side_optimal_bw(
            sigma2_l, m2_l, f_c, len(x_l), C_K, h_pilot, x_range)
        h_opt_r = _side_optimal_bw(
            sigma2_r, m2_r, f_c, len(x_r), C_K, h_pilot, x_range)
        return (h_opt_l, h_opt_r)
    else:
        # Common bandwidth (mserd)
        bias_sq = ((m2_r - m2_l) / 2) ** 2
        if bias_sq < 1e-12:
            h_opt = h_pilot
        else:
            h_opt = (C_K * (sigma2_l + sigma2_r) /
                     (f_c * bias_sq * n)) ** (1 / 5)
        h_opt = np.clip(h_opt, 0.02 * x_range, 0.98 * x_range)
        return float(h_opt)


def _side_optimal_bw(
    sigma2: float, m2: float, f_c: float, n_side: int,
    C_K: float, h_pilot: float, x_range: float,
) -> float:
    """MSE-optimal bandwidth for one side of the cutoff."""
    bias_sq = m2 ** 2
    if bias_sq < 1e-12 or n_side < 5:
        h_opt = h_pilot
    else:
        h_opt = (C_K * sigma2 / (f_c * bias_sq * n_side)) ** (1 / 5)
    return float(np.clip(h_opt, 0.02 * x_range, 0.98 * x_range))


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
