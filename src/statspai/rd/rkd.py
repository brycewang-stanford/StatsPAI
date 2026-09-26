"""
Regression Kink Design (RKD) estimator.

Implements the methodology of Card, Lee, Pei, and Weber (2015) for
estimating causal effects from kinks (changes in slope) in the
treatment assignment function, rather than discontinuities in the level.

References
----------
Card, D., Lee, D.S., Pei, Z. and Weber, A. (2015).
"Inference on Causal Effects in a Generalized Regression Kink Design."
*Econometrica*, 83(6), 2453-2483. [@card2015inference]

Nielsen, H.S., Sorensen, T. and Taber, C. (2010).
"Estimating the Effect of Student Aid on College Enrollment:
Evidence from a Government Grant Policy Reform."
*American Economic Journal: Economic Policy*, 2(2), 185-215. [@nielsen2010estimating]
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ._core import _kernel_fn


class RKDResult(CausalResult):
    """CausalResult with RKD-specific summary and plotting methods."""

    def __init__(self, *, rkd_plot_data: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rkd_plot_data = rkd_plot_data

    # Own layout without the base class's digits= / fmt= precision options;
    # passing them raises TypeError rather than being ignored.
    def summary(self, alpha: Optional[float] = None) -> str:  # type: ignore[override]
        return _rkd_summary(self, alpha)

    def plot(self, type: str = "main", **kwargs: Any) -> Any:
        return _rkd_plot(self, **kwargs)


# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(treat="treatment", _strict=True, running="x", cutoff="c")
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
        Bandwidth. If None, the CCT MSE-optimal bandwidth for the first
        derivative (``rdrobust(deriv=1, bwselect='mserd', vce='hc1')``) is
        selected.
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

    It is ``sp.rdrobust(..., deriv=1, vce='hc1')`` -- R / Stata
    ``rdrobust(deriv=1)``, the kink estimator of Calonico, Cattaneo &
    Titiunik -- reporting the *conventional* estimate and its HC1 (or
    cluster) standard error as the headline, with the robust
    bias-corrected estimate, SE and interval in ``model_info['robust']``.
    Through 1.28.0 the default bandwidth was an ad hoc rule of thumb that
    no reference computes, and the fuzzy standard error was a delta
    method that dropped the covariance between the outcome and treatment
    kinks (3.6% off R on the reference fixture); both now come from the
    shared rdrobust path.

    Examples
    --------
    Sharp (reduced-form) RKD — slope changes by 0.8 at the kink x=0:

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 4000
    >>> X = rng.uniform(-2, 2, n)
    >>> Y = 0.5 * X + 0.8 * np.maximum(X, 0) + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X})
    >>> result = sp.rkd(df, y='y', x='x', c=0)
    >>> bool(abs(result.estimate - 0.8) < 0.5)
    True

    Fuzzy RKD — ratio of outcome kink to treatment kink:

    >>> T = 1.0 * X + 2.0 * np.maximum(X, 0) + rng.normal(0, 0.3, n)
    >>> Y2 = 0.4 * T + rng.normal(0, 0.5, n)
    >>> df2 = pd.DataFrame({'y': Y2, 'x': X, 'treat': T})
    >>> result2 = sp.rkd(df2, y='y', x='x', c=0, treatment='treat')
    >>> bool(np.isfinite(result2.estimate))
    True
    """
    # --- Validate inputs ---
    h_given = h is not None
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
    n = len(df)
    if n < 20:
        raise ValueError(f"Too few observations ({n}). Need at least 20.")

    # --- Estimation: the shared CCT path with deriv = 1 ---
    from .rdrobust import rdrobust as _rdrobust

    fit = _rdrobust(
        df,
        y=y,
        x=x,
        c=c,
        fuzzy=treatment,
        deriv=1,
        p=p,
        kernel=kernel,
        h=h,
        cluster=cluster,
        vce="hc1",
        alpha=alpha,
        manipulation_test=False,
    )
    fmi = fit.model_info
    h = float(fmi["bandwidth_h"])
    conv = fmi["conventional"]
    estimate = float(conv["estimate"])
    se = float(conv["se"])

    # --- Per-side slopes (reporting and plotting only) ---
    Y = df[y].values.astype(float)
    X = df[x].values.astype(float)
    X_c = X - c
    T = df[treatment].values.astype(float) if treatment is not None else None
    cl = df[cluster].values if cluster is not None else None
    w = _kernel_weights(X_c / h, kernel)
    left = (X_c < 0) & (w > 0)
    right = (X_c >= 0) & (w > 0)
    n_left = int(left.sum())
    n_right = int(right.sum())

    b_left_y, V_left_y, _ = _local_poly_fit(
        Y[left], X_c[left], w[left], p, cl[left] if cl is not None else None
    )
    b_right_y, V_right_y, _ = _local_poly_fit(
        Y[right], X_c[right], w[right], p, cl[right] if cl is not None else None
    )
    slope_left_y = b_left_y[1]
    slope_right_y = b_right_y[1]
    kink_y = slope_right_y - slope_left_y
    se_slope_left_y = np.sqrt(V_left_y[1, 1])
    se_slope_right_y = np.sqrt(V_right_y[1, 1])
    se_kink_y = np.sqrt(V_left_y[1, 1] + V_right_y[1, 1])

    if treatment is None:
        estimand_label = "Kink in E[Y|X]"
        design = "Sharp (Reduced-Form)"
        extra_info: Dict[str, Any] = {}
    else:
        assert T is not None
        b_left_t, V_left_t, _ = _local_poly_fit(
            T[left], X_c[left], w[left], p, cl[left] if cl is not None else None
        )
        b_right_t, V_right_t, _ = _local_poly_fit(
            T[right], X_c[right], w[right], p, cl[right] if cl is not None else None
        )
        kink_t = b_right_t[1] - b_left_t[1]
        estimand_label = "LATE (Fuzzy RKD)"
        design = "Fuzzy"
        extra_info = {
            "slope_left_treatment": b_left_t[1],
            "slope_right_treatment": b_right_t[1],
            "kink_treatment": kink_t,
            "se_slope_left_treatment": np.sqrt(V_left_t[1, 1]),
            "se_slope_right_treatment": np.sqrt(V_right_t[1, 1]),
            "se_kink_treatment": np.sqrt(V_left_t[1, 1] + V_right_t[1, 1]),
            "first_stage_F": fmi.get("first_stage_F"),
        }

    # --- Inference ---
    z = estimate / se if se > 0 else np.nan
    pvalue = 2 * stats.norm.sf(np.abs(z))
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
        "bandwidth_b": float(fmi["bandwidth_b"]),
        "bw_type": "manual" if h_given else "mserd (CCT, deriv=1)",
        "kernel": kernel,
        "polynomial_order": p,
        "cutoff": c,
        "n_left": n_left,
        "n_right": n_right,
        "n_effective": n_left + n_right,
        "vce": "cluster" if cluster is not None else "hc1",
        "conventional": dict(conv),
        "robust": dict(fmi["robust"]),
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

    result = RKDResult(
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
        rkd_plot_data=_plot_data,
    )

    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.rd.rkd",
            params={
                "y": y,
                "x": x,
                "c": c,
                "treatment": treatment,
                "h": h,
                "kernel": kernel,
                "p": p,
                "cluster": cluster,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return result


# ======================================================================
# Internal helpers
# ======================================================================


def _kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    """Compute kernel weights for scaled distances u = (X - c) / h.

    Delegates to ._core._kernel_fn. Note that the uniform kernel there
    uses the standard 0.5 * 1{|u|<=1} normalization (vs. the historical
    1{|u|<=1} used in an earlier RKD implementation); WLS fits and
    sandwich variance are invariant to this constant rescaling of the
    weights, so estimated coefficients and standard errors are unchanged.
    """
    return np.asarray(_kernel_fn(u, kernel), dtype=float)


def _local_poly_fit(
    Y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    p: int,
    cl: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted local polynomial regression.

    Returns (coefficients, HC variance matrix, residuals).
    The slope estimate is coefficients[1].
    """
    n = len(Y)
    # Design matrix: [1, X, X^2, ..., X^p]
    Z = np.column_stack([X**j for j in range(p + 1)])  # (n, p+1)
    # Kernel weights applied by row: np.diag(w) was an n x n matrix.

    ZtW = (Z * w[:, None]).T  # (p+1, n)
    ZtWZ = ZtW @ Z  # (p+1, p+1)

    try:
        ZtWZ_inv = np.linalg.inv(ZtWZ)
    except np.linalg.LinAlgError:  # pragma: no cover
        ZtWZ_inv = np.linalg.pinv(ZtWZ)

    beta = ZtWZ_inv @ (ZtW @ Y)
    resid = Y - Z @ beta

    # HC1 (sandwich) variance or cluster-robust variance
    if cl is not None:
        V = _cluster_variance(Z, w, resid, cl, ZtWZ_inv)
    else:
        # HC1 sandwich: (Z'WZ)^{-1} Z'W diag(e^2) WZ (Z'WZ)^{-1}
        meat = (Z * (w**2 * resid**2)[:, None]).T @ Z
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
    return np.asarray(V, dtype=float)


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
        rows.extend(
            [
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
            ]
        )
    return pd.DataFrame(rows)


# ======================================================================
# Summary
# ======================================================================


def _rkd_summary(result: CausalResult, alpha: Optional[float] = None) -> str:
    """Formatted RKD summary output."""
    a = alpha if alpha is not None else result.alpha
    z_crit = stats.norm.ppf(1 - a / 2)
    ci = (result.estimate - z_crit * result.se, result.estimate + z_crit * result.se)

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
    se_label = f"  SE ({mi.get('vce', 'hc1')}):"
    lines.append(f"{se_label:<26}{result.se:.4f}")
    lines.append(f"  {pct}% CI:                [{ci[0]:.4f}, {ci[1]:.4f}]")
    rb = mi.get("robust")
    if rb:
        lines.append(
            f"  Robust bias-corrected:  {rb['estimate']:.4f}"
            f"  (SE: {rb['se']:.4f}; CI [{rb['ci'][0]:.4f}, {rb['ci'][1]:.4f}])"
        )
    lines.append("")
    lines.append(
        f"  Slope left of kink:     {mi['slope_left_outcome']:.4f}"
        f"  (SE: {mi['se_slope_left_outcome']:.4f})"
    )
    lines.append(
        f"  Slope right of kink:    {mi['slope_right_outcome']:.4f}"
        f"  (SE: {mi['se_slope_right_outcome']:.4f})"
    )
    lines.append(f"  Kink (slope change):    {mi['kink_outcome']:.4f}")

    if design == "Fuzzy":
        lines.append("")
        lines.append(
            f"  First stage kink:       {mi['kink_treatment']:.4f}"
            f"  (SE: {mi['se_kink_treatment']:.4f})"
        )

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


def _rkd_plot(result: RKDResult, **kwargs: Any) -> Any:
    """
    RKD plot: scatter + separate polynomial fits on each side of the kink.

    Parameters
    ----------
    **kwargs
        Passed to matplotlib (e.g., figsize, title, scatter_alpha).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError("matplotlib is required for RKD plots.")  # pragma: no cover

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
        X[left_mask],
        Y[left_mask],
        c="steelblue",
        alpha=scatter_alpha,
        s=8,
        zorder=1,
        label=None,
    )
    ax.scatter(
        X[right_mask],
        Y[right_mask],
        c="indianred",
        alpha=scatter_alpha,
        s=8,
        zorder=1,
        label=None,
    )

    # Fitted polynomials
    x_left = np.linspace(c - h, c, 200)
    x_right = np.linspace(c, c + h, 200)

    def _poly_val(b: np.ndarray, xs: np.ndarray, centre: float) -> np.ndarray:
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
    ax.axvline(
        c, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Kink point"
    )

    title = kwargs.get("title", "Regression Kink Design")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(kwargs.get("xlabel", "Running Variable"))
    ax.set_ylabel(kwargs.get("ylabel", "Outcome"))
    ax.legend(frameon=False)

    plt.tight_layout()

    if kwargs.get("show", True):
        plt.show()

    return fig
