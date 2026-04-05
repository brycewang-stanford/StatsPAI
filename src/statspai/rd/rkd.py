"""
Regression Kink Design (RKD) estimator.

Implements the methodology of Card, Lee, Pei, and Weber (2015) for
estimating causal effects from kinks (changes in slope) in the
treatment assignment function, rather than discontinuities in the level.

References
----------
Card, D., Lee, D.S., Pei, Z. and Weber, A. (2015).
"Inference on Causal Effects in a Generalized Regression Kink Design."
*Econometrica*, 83(6), 2453-2483.

Nielsen, H.S., Sorensen, T. and Taber, C. (2010).
"Estimating the Effect of Student Aid on College Enrollment:
Evidence from a Government Grant Policy Reform."
*American Economic Journal: Economic Policy*, 2(2), 185-215.
"""

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def rkd(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    treatment: Optional[str] = None,
    h: Optional[float] = None,
    kernel: str = "triangular",
    p: int = 1,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Regression Kink Design estimator (Card et al., 2015).

    Estimates causal effects from a kink (change in slope) in the
    treatment assignment function at a known threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        Kink point (cutoff).
    treatment : str, optional
        Treatment variable for fuzzy RKD. If None, estimate the
        reduced-form kink in E[Y|X] (sharp / reduced-form RKD).
    h : float, optional
        Bandwidth. If None, an MSE-optimal bandwidth is selected
        automatically.
    kernel : str, default 'triangular'
        Kernel function: 'triangular', 'epanechnikov', or 'uniform'.
    p : int, default 1
        Local polynomial order (1 = local linear, the default and
        most common choice for RKD).
    cluster : str, optional
        Cluster variable name for clustered standard errors.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Result object with estimate, standard error, confidence interval,
        summary(), and plot() methods.

    Notes
    -----
    **Sharp RKD** (treatment=None): estimates the change in slope of
    E[Y|X] at the kink point *c*. This is the reduced-form kink.

    **Fuzzy RKD** (treatment specified): estimates the ratio of the
    change in slope of E[Y|X] to the change in slope of E[T|X] at *c*,
    analogous to fuzzy RD.

    The estimator fits separate local polynomial regressions on each
    side of the kink and computes the difference in estimated slopes
    (first derivatives) at the kink point.

    Examples
    --------
    Sharp (reduced-form) RKD:

    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.uniform(-2, 2, n)
    >>> Y = 0.5 * X + 0.8 * np.maximum(X, 0) + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X})
    >>> result = rkd(df, y='y', x='x', c=0)
    >>> abs(result.estimate - 0.8) < 0.5
    True

    Fuzzy RKD:

    >>> T = 1.0 * X + 2.0 * np.maximum(X, 0) + rng.normal(0, 0.3, n)
    >>> Y2 = 0.4 * T + rng.normal(0, 0.5, n)
    >>> df2 = pd.DataFrame({'y': Y2, 'x': X, 'treat': T})
    >>> result2 = rkd(df2, y='y', x='x', c=0, treatment='treat')
    """
    # --- Validate inputs ---
    if kernel not in ("triangular", "epanechnikov", "uniform"):
        raise ValueError(
            f"kernel must be 'triangular', 'epanechnikov', or "
            f"'uniform', got '{kernel}'"
        )
    if p < 1:
        raise ValueError(f"p must be >= 1 for RKD (need slope), got {p}")

    # --- Parse data ---
    cols = [col for col in [y, x, treatment, cluster] if col is not None]
    df = data.dropna(subset=cols)
    Y = df[y].values.astype(float)
    X = df[x].values.astype(float)
    X_c = X - c  # centre at kink point

    T = df[treatment].values.astype(float) if treatment is not None else None
    cl = df[cluster].values if cluster is not None else None

    n = len(Y)
    if n < 20:
        raise ValueError(f"Too few observations ({n}). Need at least 20.")

    # --- Bandwidth selection ---
    if h is None:
        h = _rkd_bandwidth(Y, X_c, T, p, kernel)

    # --- Kernel weights ---
    u = X_c / h
    w = _kernel_weights(u, kernel)

    left = (X_c < 0) & (w > 0)
    right = (X_c >= 0) & (w > 0)
    n_left = int(left.sum())
    n_right = int(right.sum())

    if n_left < (p + 1) or n_right < (p + 1):
        raise ValueError(
            f"Insufficient observations within bandwidth h={h:.4f}: "
            f"{n_left} left, {n_right} right. "
            f"Need at least {p + 1} on each side."
        )

    # --- Local polynomial fits ---
    b_left_y, V_left_y, resid_left_y = _local_poly_fit(
        Y[left], X_c[left], w[left], p, cl[left] if cl is not None else None
    )
    b_right_y, V_right_y, resid_right_y = _local_poly_fit(
        Y[right], X_c[right], w[right], p, cl[right] if cl is not None else None
    )

    # Slope estimates (coefficient on (X - c), i.e. index 1)
    slope_left_y = b_left_y[1]
    slope_right_y = b_right_y[1]
    kink_y = slope_right_y - slope_left_y

    se_slope_left_y = np.sqrt(V_left_y[1, 1])
    se_slope_right_y = np.sqrt(V_right_y[1, 1])
    se_kink_y = np.sqrt(V_left_y[1, 1] + V_right_y[1, 1])

    # --- Sharp vs Fuzzy ---
    if treatment is None:
        # Sharp / reduced-form: just the kink in E[Y|X]
        estimate = kink_y
        se = se_kink_y
        estimand_label = "Kink in E[Y|X]"
        design = "Sharp (Reduced-Form)"
        extra_info = {}
    else:
        # Fuzzy: ratio of kinks
        b_left_t, V_left_t, _ = _local_poly_fit(
            T[left], X_c[left], w[left], p, cl[left] if cl is not None else None
        )
        b_right_t, V_right_t, _ = _local_poly_fit(
            T[right], X_c[right], w[right], p, cl[right] if cl is not None else None
        )

        slope_left_t = b_left_t[1]
        slope_right_t = b_right_t[1]
        kink_t = slope_right_t - slope_left_t

        if np.abs(kink_t) < 1e-12:
            raise ValueError(
                "First-stage kink in treatment is effectively zero. "
                "Cannot estimate fuzzy RKD."
            )

        estimate = kink_y / kink_t

        # Delta method SE for ratio: Var(a/b) ~ (1/b^2)*Var(a) + (a^2/b^4)*Var(b)
        var_kink_y = V_left_y[1, 1] + V_right_y[1, 1]
        var_kink_t = V_left_t[1, 1] + V_right_t[1, 1]
        se = np.sqrt(
            var_kink_y / kink_t**2
            + (kink_y**2 / kink_t**4) * var_kink_t
        )

        se_slope_left_t = np.sqrt(V_left_t[1, 1])
        se_slope_right_t = np.sqrt(V_right_t[1, 1])
        se_kink_t = np.sqrt(var_kink_t)

        estimand_label = "LATE (Fuzzy RKD)"
        design = "Fuzzy"
        extra_info = {
            "slope_left_treatment": slope_left_t,
            "slope_right_treatment": slope_right_t,
            "kink_treatment": kink_t,
            "se_slope_left_treatment": se_slope_left_t,
            "se_slope_right_treatment": se_slope_right_t,
            "se_kink_treatment": se_kink_t,
        }

    # --- Inference ---
    z = estimate / se if se > 0 else np.nan
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(z)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - z_crit * se, estimate + z_crit * se)

    # --- Build model_info ---
    model_info = {
        "design": design,
        "slope_left_outcome": slope_left_y,
        "slope_right_outcome": slope_right_y,
        "kink_outcome": kink_y,
        "se_slope_left_outcome": se_slope_left_y,
        "se_slope_right_outcome": se_slope_right_y,
        "se_kink_outcome": se_kink_y,
        "bandwidth": h,
        "bw_type": "MSE-optimal" if h is not None else "manual",
        "kernel": kernel,
        "polynomial_order": p,
        "cutoff": c,
        "n_left": n_left,
        "n_right": n_right,
        "n_effective": n_left + n_right,
        **extra_info,
    }

    # --- Plot data (for result.plot()) ---
    _plot_data = {
        "Y": Y,
        "X": X,
        "X_c": X_c,
        "h": h,
        "c": c,
        "p": p,
        "kernel": kernel,
        "w": w,
        "left": left,
        "right": right,
        "b_left": b_left_y,
        "b_right": b_right_y,
    }

    result = CausalResult(
        method="Regression Kink Design (Card et al., 2015)",
        estimand=estimand_label,
        estimate=float(estimate),
        se=float(se),
        pvalue=float(pvalue),
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=_build_detail_table(model_info, treatment is not None),
        model_info=model_info,
        _citation_key="rkd",
    )

    # Attach custom summary and plot
    result._rkd_plot_data = _plot_data
    result._original_summary = result.summary
    result.summary = lambda alpha_=None: _rkd_summary(result, alpha_)
    result.plot = lambda **kw: _rkd_plot(result, **kw)

    return result


# ======================================================================
# Internal helpers
# ======================================================================

def _kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    """Compute kernel weights for scaled distances u = (X - c) / h."""
    if kernel == "triangular":
        return np.maximum(1 - np.abs(u), 0)
    elif kernel == "epanechnikov":
        return np.maximum(0.75 * (1 - u**2), 0)
    else:  # uniform
        return (np.abs(u) <= 1).astype(float)


def _local_poly_fit(
    Y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    p: int,
    cl: Optional[np.ndarray] = None,
):
    """
    Weighted local polynomial regression.

    Returns (coefficients, HC variance matrix, residuals).
    The slope estimate is coefficients[1].
    """
    n = len(Y)
    # Design matrix: [1, X, X^2, ..., X^p]
    Z = np.column_stack([X**j for j in range(p + 1)])  # (n, p+1)
    W = np.diag(w)

    ZtW = Z.T @ W  # (p+1, n)
    ZtWZ = ZtW @ Z  # (p+1, p+1)

    try:
        ZtWZ_inv = np.linalg.inv(ZtWZ)
    except np.linalg.LinAlgError:
        ZtWZ_inv = np.linalg.pinv(ZtWZ)

    beta = ZtWZ_inv @ (ZtW @ Y)
    resid = Y - Z @ beta

    # HC1 (sandwich) variance or cluster-robust variance
    if cl is not None:
        V = _cluster_variance(Z, w, resid, cl, ZtWZ_inv)
    else:
        # HC1 sandwich: (Z'WZ)^{-1} Z'W diag(e^2) WZ (Z'WZ)^{-1}
        meat = Z.T @ (W @ np.diag(resid**2) @ W) @ Z
        dfc = n / max(n - (p + 1), 1)
        V = dfc * ZtWZ_inv @ meat @ ZtWZ_inv

    return beta, V, resid


def _cluster_variance(
    Z: np.ndarray,
    w: np.ndarray,
    resid: np.ndarray,
    cl: np.ndarray,
    ZtWZ_inv: np.ndarray,
) -> np.ndarray:
    """Cluster-robust variance estimator."""
    unique_cl = np.unique(cl)
    G = len(unique_cl)
    n = len(resid)
    k = Z.shape[1]

    meat = np.zeros((k, k))
    for g in unique_cl:
        idx = cl == g
        Zg = Z[idx]
        wg = w[idx]
        eg = resid[idx]
        score_g = (Zg * (wg * eg)[:, None]).sum(axis=0)  # (k,)
        meat += np.outer(score_g, score_g)

    # Small-sample correction: G/(G-1) * (n-1)/(n-k)
    dfc = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = dfc * ZtWZ_inv @ meat @ ZtWZ_inv
    return V


def _rkd_bandwidth(
    Y: np.ndarray,
    X_c: np.ndarray,
    T: Optional[np.ndarray],
    p: int,
    kernel: str,
) -> float:
    """
    MSE-optimal bandwidth for RKD (derivative estimation).

    Uses a plug-in approach: fit a global polynomial to estimate bias
    and use local residuals for variance, then apply the IK-style
    rule adapted for derivative estimation.
    """
    n = len(Y)
    x_range = np.ptp(X_c)
    if x_range < 1e-12:
        raise ValueError("Running variable has no variation.")

    # Pilot bandwidth: use Silverman rule-of-thumb scaled up
    # (RKD needs larger bandwidth than RD for slope estimation)
    sd_x = np.std(X_c)
    h_pilot = 2.0 * 1.06 * sd_x * n ** (-1.0 / 5.0)
    h_pilot = max(h_pilot, x_range * 0.05)  # floor

    # Fit global polynomial (order p + 2) for bias estimation
    q = min(p + 2, 5)
    left = X_c < 0
    right = X_c >= 0

    # Estimate second derivative of conditional mean on each side
    # using a global polynomial within a pilot region
    pilot_left = left & (np.abs(X_c) <= h_pilot * 2)
    pilot_right = right & (np.abs(X_c) <= h_pilot * 2)

    if pilot_left.sum() < q + 1 or pilot_right.sum() < q + 1:
        # Fallback: use full data
        pilot_left = left
        pilot_right = right

    def _fit_deriv2(yy, xx):
        """Fit polynomial and return estimated 2nd derivative at 0."""
        if len(yy) < q + 1:
            return 0.0
        Z = np.column_stack([xx**j for j in range(q + 1)])
        try:
            beta = np.linalg.lstsq(Z, yy, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0
        # 2nd derivative at 0 is 2 * beta[2] (if q >= 2)
        return 2.0 * beta[2] if q >= 2 else 0.0

    d2_left = _fit_deriv2(Y[pilot_left], X_c[pilot_left])
    d2_right = _fit_deriv2(Y[pilot_right], X_c[pilot_right])

    # Estimate variance using local residuals
    u_pilot = X_c / h_pilot
    w_pilot = _kernel_weights(u_pilot, kernel)
    active = w_pilot > 0
    if np.any(active):
        var_est = np.average(Y[active]**2, weights=w_pilot[active])
    else:
        var_est = np.var(Y)

    # MSE-optimal bandwidth for derivative estimation
    # h_opt ~ C_k * (sigma^2 / (n * f * (m'' bias)^2))^{1/5}
    bias_sq = max((d2_right - d2_left) ** 2, 1e-10)
    C_k = {"triangular": 3.4375, "epanechnikov": 3.1999, "uniform": 2.7}
    c_k = C_k.get(kernel, 3.4375)

    h_opt = c_k * (var_est / (n * bias_sq)) ** (1.0 / 5.0)

    # Bound the bandwidth to reasonable range
    h_opt = np.clip(h_opt, x_range * 0.02, x_range * 0.8)

    return float(h_opt)


def _build_detail_table(model_info: Dict[str, Any], fuzzy: bool) -> pd.DataFrame:
    """Build a tidy detail DataFrame."""
    rows = [
        {
            "term": "Slope left (outcome)",
            "estimate": model_info["slope_left_outcome"],
            "se": model_info["se_slope_left_outcome"],
        },
        {
            "term": "Slope right (outcome)",
            "estimate": model_info["slope_right_outcome"],
            "se": model_info["se_slope_right_outcome"],
        },
        {
            "term": "Kink (outcome)",
            "estimate": model_info["kink_outcome"],
            "se": model_info["se_kink_outcome"],
        },
    ]
    if fuzzy:
        rows.extend([
            {
                "term": "Slope left (treatment)",
                "estimate": model_info["slope_left_treatment"],
                "se": model_info["se_slope_left_treatment"],
            },
            {
                "term": "Slope right (treatment)",
                "estimate": model_info["slope_right_treatment"],
                "se": model_info["se_slope_right_treatment"],
            },
            {
                "term": "Kink (treatment)",
                "estimate": model_info["kink_treatment"],
                "se": model_info["se_kink_treatment"],
            },
        ])
    return pd.DataFrame(rows)


# ======================================================================
# Summary
# ======================================================================

def _rkd_summary(result: CausalResult, alpha: Optional[float] = None) -> str:
    """Formatted RKD summary output."""
    a = alpha if alpha is not None else result.alpha
    z_crit = stats.norm.ppf(1 - a / 2)
    ci = (result.estimate - z_crit * result.se,
          result.estimate + z_crit * result.se)

    mi = result.model_info
    stars = CausalResult._stars(result.pvalue)
    pct = int((1 - a) * 100)

    lines = []
    bar = "\u2501" * 60
    lines.append(bar)
    lines.append("  Regression Kink Design (Card et al., 2015)")
    lines.append(bar)

    design = mi.get("design", "Sharp")
    lines.append(f"  Design:                 {design}")
    lines.append(f"  RKD estimate:           {result.estimate:.4f}{stars}")
    lines.append(f"  Robust SE:              {result.se:.4f}")
    lines.append(f"  {pct}% CI:                [{ci[0]:.4f}, {ci[1]:.4f}]")
    lines.append("")
    lines.append(f"  Slope left of kink:     {mi['slope_left_outcome']:.4f}"
                 f"  (SE: {mi['se_slope_left_outcome']:.4f})")
    lines.append(f"  Slope right of kink:    {mi['slope_right_outcome']:.4f}"
                 f"  (SE: {mi['se_slope_right_outcome']:.4f})")
    lines.append(f"  Kink (slope change):    {mi['kink_outcome']:.4f}")

    if design == "Fuzzy":
        lines.append("")
        lines.append(f"  First stage kink:       {mi['kink_treatment']:.4f}"
                     f"  (SE: {mi['se_kink_treatment']:.4f})")

    lines.append("")
    bw_label = mi.get("bw_type", "manual")
    lines.append(f"  Bandwidth:              {mi['bandwidth']:.3f} ({bw_label})")
    lines.append(f"  Kernel:                 {mi['kernel'].capitalize()}")
    lines.append(f"  Polynomial order:       {mi['polynomial_order']}")
    lines.append(f"  N left:                 {mi['n_left']}")
    lines.append(f"  N right:                {mi['n_right']}")
    lines.append(f"  N effective:            {mi['n_effective']}")
    lines.append(bar)

    summary_str = "\n".join(lines)
    print(summary_str)
    return summary_str


# ======================================================================
# Plot
# ======================================================================

def _rkd_plot(result: CausalResult, **kwargs):
    """
    RKD plot: scatter + separate polynomial fits on each side of the kink.

    Parameters
    ----------
    **kwargs
        Passed to matplotlib (e.g., figsize, title, scatter_alpha).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for RKD plots.")

    pd_ = result._rkd_plot_data
    X = pd_["X"]
    Y = pd_["Y"]
    c = pd_["c"]
    h = pd_["h"]
    b_left = pd_["b_left"]
    b_right = pd_["b_right"]
    left_mask = pd_["left"]
    right_mask = pd_["right"]

    figsize = kwargs.get("figsize", (8, 5))
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter points within bandwidth
    scatter_alpha = kwargs.get("scatter_alpha", 0.25)
    ax.scatter(
        X[left_mask], Y[left_mask],
        c="steelblue", alpha=scatter_alpha, s=8, zorder=1, label=None,
    )
    ax.scatter(
        X[right_mask], Y[right_mask],
        c="indianred", alpha=scatter_alpha, s=8, zorder=1, label=None,
    )

    # Fitted polynomials
    x_left = np.linspace(c - h, c, 200)
    x_right = np.linspace(c, c + h, 200)

    def _poly_val(b, xs, centre):
        xc = xs - centre
        yhat = np.zeros_like(xs, dtype=float)
        for j in range(len(b)):
            yhat += b[j] * xc**j
        return yhat

    y_left = _poly_val(b_left, x_left, c)
    y_right = _poly_val(b_right, x_right, c)

    ax.plot(x_left, y_left, color="navy", linewidth=2, label="Left fit")
    ax.plot(x_right, y_right, color="darkred", linewidth=2, label="Right fit")

    # Kink line
    ax.axvline(c, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Kink point")

    title = kwargs.get("title", "Regression Kink Design")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(kwargs.get("xlabel", "Running Variable"))
    ax.set_ylabel(kwargs.get("ylabel", "Outcome"))
    ax.legend(frameon=False)

    plt.tight_layout()

    if kwargs.get("show", True):
        plt.show()

    return fig
