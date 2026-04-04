"""
Sensitivity analysis and robustness diagnostics.

Implements:
- Oster (2019) coefficient stability bounds (δ and bias-adjusted β)
- McCrary (2008) density discontinuity test for RD manipulation

References
----------
Oster, E. (2019).
"Unobservable Selection and Coefficient Stability: Theory and Evidence."
*Journal of Business & Economic Statistics*, 37(2), 187-204.

McCrary, J. (2008).
"Manipulation of the Running Variable in the Regression Discontinuity
Design: A Density Test."
*Journal of Econometrics*, 142(2), 698-714.
"""

from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Oster (2019) Coefficient Stability Bounds
# ======================================================================

def oster_bounds(
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    treat: Optional[str] = None,
    controls: Optional[List[str]] = None,
    r_max: Optional[float] = None,
    delta: float = 1.0,
    beta_short: Optional[float] = None,
    r2_short: Optional[float] = None,
    beta_long: Optional[float] = None,
    r2_long: Optional[float] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Oster (2019) coefficient stability bounds.

    Assesses robustness of a treatment effect estimate to omitted variable
    bias by computing how much unobservable selection (relative to
    observable selection) would be needed to explain away the result.

    Can be called in two ways:

    1. **From data** — runs short and long regressions internally::

        result = oster_bounds(data, y='wage', treat='training',
                              controls=['age', 'education'])

    2. **From pre-computed statistics** — supply β and R² directly::

        result = oster_bounds(beta_short=2.5, r2_short=0.15,
                              beta_long=2.0, r2_long=0.45)

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input data (required for method 1).
    y : str, optional
        Outcome variable.
    treat : str, optional
        Treatment variable.
    controls : list of str, optional
        Control variables (included in the long but not short regression).
    r_max : float, optional
        Maximum R² under full selection. Default: min(1.0, 1.3 × R²_long).
    delta : float, default 1.0
        Proportionality assumption: ratio of unobservable-to-observable
        selection. δ=1 means equal selection.
    beta_short : float, optional
        Treatment coefficient from short regression (no controls).
    r2_short : float, optional
        R² from short regression.
    beta_long : float, optional
        Treatment coefficient from long regression (with controls).
    r2_long : float, optional
        R² from long regression.
    alpha : float, default 0.05
        Significance level for the identified set.

    Returns
    -------
    dict
        Keys:
        - ``beta_short``, ``r2_short``: short regression estimates
        - ``beta_long``, ``r2_long``: long regression estimates
        - ``r_max``: maximum R² used
        - ``delta_for_zero``: δ* such that adjusted β = 0 (robustness measure)
        - ``beta_adjusted``: bias-adjusted β under (δ, R_max)
        - ``identified_set``: [min(β_adjusted, β_long), max(β_adjusted, β_long)]
        - ``robust``: True if identified set excludes zero
        - ``interpretation``: human-readable interpretation

    Examples
    --------
    >>> # From data
    >>> result = oster_bounds(df, y='wage', treat='training',
    ...                       controls=['age', 'education', 'experience'])
    >>> print(f"δ* = {result['delta_for_zero']:.2f}")
    >>> print(f"Robust: {result['robust']}")

    >>> # From statistics (e.g., read from a paper)
    >>> result = oster_bounds(beta_short=2.5, r2_short=0.15,
    ...                       beta_long=2.0, r2_long=0.45)
    """
    # --- Obtain β and R² ---
    if data is not None and y is not None and treat is not None:
        b_short, r2_s, b_long, r2_l = _run_oster_regressions(
            data, y, treat, controls or [],
        )
    elif (beta_short is not None and r2_short is not None
          and beta_long is not None and r2_long is not None):
        b_short, r2_s = beta_short, r2_short
        b_long, r2_l = beta_long, r2_long
    else:
        raise ValueError(
            "Provide either (data, y, treat, controls) or "
            "(beta_short, r2_short, beta_long, r2_long)."
        )

    # --- R_max ---
    if r_max is None:
        r_max = min(1.0, 1.3 * r2_l)
    if r_max <= r2_l:
        r_max = min(1.0, r2_l + 0.01)

    # --- Bias-adjusted coefficient (Oster eq. 4) ---
    # β*(δ, R_max) = β̃ - δ × (β̊ - β̃) × (R̃² - R̊²) / (R_max - R̃²)
    movement = b_short - b_long  # β̊ - β̃
    r2_gain_obs = r2_l - r2_s    # R̃² - R̊²
    r2_gain_unobs = r_max - r2_l  # R_max - R̃²

    if abs(r2_gain_unobs) < 1e-12:
        # R_max ≈ R²_long → no room for unobservables
        beta_adj = b_long
        delta_star = np.inf
    elif abs(r2_gain_obs) < 1e-12:
        # No R² gain from controls → controls don't matter
        beta_adj = b_long
        delta_star = np.inf
    else:
        bias_correction = delta * movement * r2_gain_obs / r2_gain_unobs
        beta_adj = b_long - bias_correction

        # δ* that drives adjusted β to zero
        # 0 = β̃ - δ* × (β̊ - β̃) × (R̃² - R̊²) / (R_max - R̃²)
        # δ* = β̃ × (R_max - R̃²) / ((β̊ - β̃) × (R̃² - R̊²))
        numerator = b_long * r2_gain_unobs
        denominator = movement * r2_gain_obs
        delta_star = numerator / denominator if abs(denominator) > 1e-12 else np.inf

    # --- Identified set: [min(β_adj, β_long), max(β_adj, β_long)] ---
    id_set = (min(beta_adj, b_long), max(beta_adj, b_long))

    # Robust if identified set excludes zero
    robust = not (id_set[0] <= 0 <= id_set[1])

    # --- Interpretation ---
    if np.isinf(delta_star):
        interp = (
            "Controls do not affect the coefficient or R². "
            "Oster bounds are uninformative."
        )
    elif abs(delta_star) > 1:
        interp = (
            f"|δ*| = {abs(delta_star):.2f} > 1. Unobservable confounding "
            f"would need to be {abs(delta_star):.1f}× stronger than observable "
            f"confounding to explain away the effect. Result is ROBUST."
        )
    else:
        interp = (
            f"|δ*| = {abs(delta_star):.2f} < 1. The effect could be "
            f"explained by unobservable confounding equal to "
            f"{abs(delta_star):.0%} of observable confounding. "
            f"Result is SENSITIVE to omitted variables."
        )

    return {
        'beta_short': b_short,
        'r2_short': r2_s,
        'beta_long': b_long,
        'r2_long': r2_l,
        'r_max': r_max,
        'delta': delta,
        'delta_for_zero': delta_star,
        'beta_adjusted': beta_adj,
        'identified_set': id_set,
        'robust': robust,
        'interpretation': interp,
    }


def _run_oster_regressions(
    data: pd.DataFrame,
    y: str,
    treat: str,
    controls: List[str],
) -> Tuple[float, float, float, float]:
    """Run short and long regressions, return (β_short, R²_short, β_long, R²_long)."""
    df = data[[y, treat] + controls].dropna()
    Y = df[y].values
    D = df[treat].values
    n = len(Y)

    # Short regression: Y ~ D
    X_short = np.column_stack([np.ones(n), D])
    beta_s = np.linalg.lstsq(X_short, Y, rcond=None)[0]
    resid_s = Y - X_short @ beta_s
    tss = np.sum((Y - Y.mean()) ** 2)
    r2_s = 1 - np.sum(resid_s ** 2) / tss

    # Long regression: Y ~ D + controls
    X_long = np.column_stack([np.ones(n), D] +
                             [df[c].values for c in controls])
    beta_l = np.linalg.lstsq(X_long, Y, rcond=None)[0]
    resid_l = Y - X_long @ beta_l
    r2_l = 1 - np.sum(resid_l ** 2) / tss

    # Treatment coefficient is index 1 in both
    return float(beta_s[1]), float(r2_s), float(beta_l[1]), float(r2_l)


# ======================================================================
# McCrary (2008) Density Discontinuity Test
# ======================================================================

def mccrary_test(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    bw: Optional[float] = None,
    n_bins: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    McCrary (2008) density discontinuity test for RD manipulation.

    Tests the null hypothesis that the density of the running variable
    is continuous at the cutoff. A rejection suggests possible manipulation
    of the running variable.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff.
    bw : float, optional
        Bandwidth for local linear density estimation.
        Default: automatic (2 × Silverman rule).
    n_bins : int, optional
        Number of histogram bins per side. Default: ceil(sqrt(n/2)).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        - ``estimate``: log density difference ln(f̂₊) - ln(f̂₋)
        - ``pvalue``: two-sided test for H₀: no discontinuity
        - ``model_info['density_left']``: estimated density from the left
        - ``model_info['density_right']``: estimated density from the right

    Examples
    --------
    >>> result = mccrary_test(df, x='score', c=0)
    >>> print(f"θ̂ = {result.estimate:.3f}, p = {result.pvalue:.4f}")
    """
    X = data[x].values.astype(float)
    X = X[np.isfinite(X)]
    n = len(X)

    if n < 20:
        raise ValueError("Need at least 20 observations for McCrary test.")

    X_c = X - c

    # Bins per side
    if n_bins is None:
        n_bins = max(int(np.ceil(np.sqrt(n / 2))), 10)

    # Bandwidth
    if bw is None:
        bw = 2 * 1.06 * np.std(X_c) * n ** (-1 / 5)

    # --- Step 1: Histogram (separate left and right) ---
    x_left = X_c[X_c < 0]
    x_right = X_c[X_c >= 0]

    if len(x_left) < 5 or len(x_right) < 5:
        raise ValueError("Not enough observations on each side of the cutoff.")

    # Bin width
    bin_w_left = abs(x_left.min()) / n_bins if len(x_left) > 0 else 1
    bin_w_right = x_right.max() / n_bins if len(x_right) > 0 else 1
    bin_w = min(bin_w_left, bin_w_right)
    bin_w = max(bin_w, 1e-10)

    # Create bins (centered at midpoints)
    # Left: bins from -n_bins*bin_w to 0
    # Right: bins from 0 to n_bins*bin_w
    edges_left = np.linspace(-n_bins * bin_w, 0, n_bins + 1)
    edges_right = np.linspace(0, n_bins * bin_w, n_bins + 1)

    counts_left, _ = np.histogram(x_left, bins=edges_left)
    counts_right, _ = np.histogram(x_right, bins=edges_right)

    # Bin midpoints
    mid_left = (edges_left[:-1] + edges_left[1:]) / 2
    mid_right = (edges_right[:-1] + edges_right[1:]) / 2

    # Normalized counts → density estimates
    density_left = counts_left / (n * bin_w)
    density_right = counts_right / (n * bin_w)

    # --- Step 2: Local linear smoothing on each side ---
    # Fit local linear to (midpoint, density) within bandwidth of cutoff
    f_left, se_left = _local_linear_density(
        mid_left, density_left, 0, bw, side='left',
    )
    f_right, se_right = _local_linear_density(
        mid_right, density_right, 0, bw, side='right',
    )

    # Ensure positive densities
    f_left = max(f_left, 1e-10)
    f_right = max(f_right, 1e-10)

    # --- Step 3: Test statistic ---
    # θ̂ = ln(f̂₊) - ln(f̂₋)
    theta = np.log(f_right) - np.log(f_left)

    # SE via delta method: se(θ) ≈ sqrt((se_r/f_r)² + (se_l/f_l)²)
    se_theta = np.sqrt((se_right / f_right) ** 2 + (se_left / f_left) ** 2)
    se_theta = max(se_theta, 1e-10)

    z = theta / se_theta
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (theta - z_crit * se_theta, theta + z_crit * se_theta)

    model_info = {
        'density_left': f_left,
        'density_right': f_right,
        'log_density_ratio': theta,
        'bandwidth': bw,
        'n_bins': n_bins,
        'bin_width': bin_w,
        'cutoff': c,
        'n_left': len(x_left),
        'n_right': len(x_right),
    }

    return CausalResult(
        method='McCrary (2008) Density Test',
        estimand='Log Density Ratio',
        estimate=theta,
        se=se_theta,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key='mccrary',
    )


def _local_linear_density(
    midpoints: np.ndarray,
    densities: np.ndarray,
    target: float,
    bw: float,
    side: str = 'left',
) -> Tuple[float, float]:
    """
    Local linear regression of bin densities on bin midpoints.

    Evaluates at ``target`` (the cutoff boundary).
    Returns (density_estimate, standard_error).
    """
    # Use only bins within bandwidth
    if side == 'left':
        in_bw = (midpoints >= target - bw) & (midpoints < target)
    else:
        in_bw = (midpoints >= target) & (midpoints <= target + bw)

    m = midpoints[in_bw]
    d = densities[in_bw]

    if len(m) < 2:
        # Fall back: simple average of nearest bins
        if len(d) > 0:
            return float(d.mean()), float(d.std() / np.sqrt(len(d))) if len(d) > 1 else 0.1
        return 0.01, 0.1

    # Triangular kernel weights
    u = (m - target) / bw
    w = np.maximum(1 - np.abs(u), 0)

    # WLS: density = α + β × (midpoint - target)
    X = np.column_stack([np.ones(len(m)), m - target])
    sqw = np.sqrt(w)
    Xw = X * sqw[:, np.newaxis]
    dw = d * sqw

    try:
        beta = np.linalg.lstsq(Xw, dw, rcond=None)[0]
        f_hat = float(beta[0])  # intercept = density at cutoff

        # SE from weighted residuals
        resid = d - X @ beta
        n_eff = len(m)
        sigma2 = np.sum(w * resid ** 2) / max(np.sum(w) - 2, 1)
        try:
            XtWX_inv = np.linalg.inv(Xw.T @ Xw)
            se_hat = float(np.sqrt(sigma2 * XtWX_inv[0, 0]))
        except np.linalg.LinAlgError:
            se_hat = float(np.std(d) / np.sqrt(n_eff))
    except Exception:
        f_hat = float(np.average(d, weights=w))
        se_hat = float(np.std(d) / np.sqrt(len(d)))

    return max(f_hat, 1e-10), max(se_hat, 1e-10)


# ======================================================================
# Citations
# ======================================================================

CausalResult._CITATIONS['oster'] = (
    "@article{oster2019unobservable,\n"
    "  title={Unobservable Selection and Coefficient Stability: "
    "Theory and Evidence},\n"
    "  author={Oster, Emily},\n"
    "  journal={Journal of Business \\& Economic Statistics},\n"
    "  volume={37},\n"
    "  number={2},\n"
    "  pages={187--204},\n"
    "  year={2019},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)

CausalResult._CITATIONS['mccrary'] = (
    "@article{mccrary2008manipulation,\n"
    "  title={Manipulation of the Running Variable in the "
    "Regression Discontinuity Design: A Density Test},\n"
    "  author={McCrary, Justin},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={142},\n"
    "  number={2},\n"
    "  pages={698--714},\n"
    "  year={2008},\n"
    "  publisher={Elsevier}\n"
    "}"
)
