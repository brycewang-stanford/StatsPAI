"""
Sensitivity analysis and robustness diagnostics.

Implements:
- Oster (2019) coefficient stability bounds (δ and bias-adjusted β)
- McCrary (2008) density discontinuity test for RD manipulation

References
----------
Oster, E. (2019).
"Unobservable Selection and Coefficient Stability: Theory and Evidence."
*Journal of Business & Economic Statistics*, 37(2), 187-204. [@oster2019unobservable]

McCrary, J. (2008).
"Manipulation of the Running Variable in the Regression Discontinuity
Design: A Density Test."
*Journal of Econometrics*, 142(2), 698-714. [@mccrary2008manipulation]
"""

from typing import Any, Dict, List, Optional, Tuple

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
        Must not exceed 1; a value at or below R²_long is replaced by
        R²_long + 0.01 with a warning.
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
        - ``beta_adjusted_alternatives``: the other real roots of Oster's
          quadratic (δ = 1) / cubic (δ ≠ 1) -- exact method only
        - ``method``: ``"exact"`` when called with data -- Oster's exact
          solution, which reproduces her Stata ``psacalc`` (``delta`` and
          ``beta``) to machine precision -- or ``"approximate"`` when only
          (β, R²) summaries are given, which uses Oster's first-order
          approximation β* ≈ β̃ − δ (β̊ − β̃)(R_max − R̃²)/(R̃² − R̊²)
        - ``identified_set``: [min(β_adjusted, β_long), max(β_adjusted, β_long)]
        - ``robust``: True if identified set excludes zero
        - ``interpretation``: human-readable interpretation

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> # From data
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> age = rng.normal(40, 10, n)
    >>> education = rng.normal(13, 3, n)
    >>> training = (rng.normal(size=n) + 0.3 * (education - 13) > 0).astype(float)
    >>> wage = 20 + 2.0 * training + 0.1 * age + 0.5 * education + rng.normal(size=n)
    >>> df = pd.DataFrame({"wage": wage, "training": training,
    ...                    "age": age, "education": education})
    >>> result = sp.oster_bounds(df, y='wage', treat='training',
    ...                          controls=['age', 'education'])
    >>> bool('delta_for_zero' in result)
    True

    >>> # From statistics (e.g., read from a paper)
    >>> result = sp.oster_bounds(beta_short=2.5, r2_short=0.15,
    ...                          beta_long=2.0, r2_long=0.45)
    >>> round(result['delta_for_zero'], 2)
    8.89
    """
    # --- Obtain β and R² ---
    from ._oster import (
        oster_approx_beta,
        oster_approx_delta,
        oster_beta_exact,
        oster_delta_exact,
        oster_inputs,
    )

    inputs: Optional[Dict[str, float]] = None
    if data is not None and y is not None and treat is not None:
        inputs = oster_inputs(data, y, treat, controls or [])
        b_short, r2_s = inputs["beta_o"], inputs["r_o"]
        b_long, r2_l = inputs["beta_t"], inputs["r_t"]
    elif (
        beta_short is not None
        and r2_short is not None
        and beta_long is not None
        and r2_long is not None
    ):
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
    elif r_max > 1.0:
        raise ValueError(f"r_max is an R-squared and cannot exceed 1 (got {r_max}).")
    if r_max <= r2_l:
        import warnings

        adjusted = min(1.0, r2_l + 0.01)
        warnings.warn(
            f"oster_bounds: r_max={r_max:.6g} does not exceed the controlled "
            f"R-squared {r2_l:.6g}; using r_max={adjusted:.6g} instead.",
            UserWarning,
            stacklevel=2,
        )
        r_max = adjusted

    movement = b_short - b_long  # β̊ - β̃
    r2_gain_obs = r2_l - r2_s  # R̃² - R̊²
    beta_alternatives: List[float] = []
    if abs(r2_gain_obs) < 1e-12 and abs(movement) < 1e-12:
        # Controls move neither the coefficient nor R²: uninformative.
        beta_adj = b_long
        delta_star = np.inf
        method = "exact" if inputs is not None else "approximate"
    elif inputs is not None:
        # Exact solution (Oster 2019; psacalc): uses the variances of y, of
        # the treatment and of the treatment residualised on the controls.
        delta_star = oster_delta_exact(inputs, r_max, beta=0.0)
        sol = oster_beta_exact(inputs, r_max, delta=delta)
        beta_adj = float(sol["beta"])  # type: ignore[arg-type]
        beta_alternatives = list(sol["alternatives"])  # type: ignore[arg-type]
        method = "exact"
    else:
        # Only coefficients and R² are known: Oster's first-order
        # approximation beta* ~= beta_t - delta (beta_o - beta_t)
        # (R_max - R_t) / (R_t - R_o).
        beta_adj = oster_approx_beta(b_short, b_long, r2_s, r2_l, r_max, delta)
        delta_star = oster_approx_delta(b_short, b_long, r2_s, r2_l, r_max)
        method = "approximate"

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
        "beta_short": b_short,
        "r2_short": r2_s,
        "beta_long": b_long,
        "r2_long": r2_l,
        "r_max": r_max,
        "delta": delta,
        "delta_for_zero": delta_star,
        "beta_adjusted": beta_adj,
        "beta_adjusted_alternatives": beta_alternatives,
        "method": method,
        "identified_set": id_set,
        "robust": robust,
        "interpretation": interp,
    }


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
    bin_width: Optional[float] = None,
) -> CausalResult:
    """
    McCrary (2008) density discontinuity test for RD manipulation.

    Tests the null hypothesis that the density of the running variable
    is continuous at the cutoff. A rejection suggests possible manipulation
    of the running variable.

    A line-for-line port of R ``rdd::DCdensity`` (the R implementation of
    McCrary's ``DCdensity``): a fine histogram with bin width
    ``2 sd(x) n^(-1/2)``, a bandwidth from fourth-order polynomial fits to
    the bin heights on each side (``3.348 (sigma^2 (c - l) /
    sum f''^2)^(1/5)``, averaged over the two sides), zero-padded bins,
    triangular-kernel local linear fits to the bin heights on each side,
    and ``theta = log f+ - log f-`` with
    ``se = sqrt(24 / (5 n h) (1/f+ + 1/f-))``.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff.
    bw : float, optional
        Bandwidth of the local linear fits. Default: McCrary's rule above.
    n_bins : int, optional
        Convenience alternative to ``bin_width``: the bin width becomes
        ``(max(x) - min(x)) / n_bins``. Ignored when ``bin_width`` is given.
    alpha : float, default 0.05
        Significance level.
    bin_width : float, optional
        Histogram bin width (``DCdensity``'s ``bin``). Default:
        ``2 sd(x) n^(-1/2)``.

    Returns
    -------
    CausalResult
        - ``estimate``: log density difference ln(f̂₊) - ln(f̂₋)
        - ``pvalue``: two-sided test for H₀: no discontinuity
        - ``model_info['density_left']``: estimated density from the left
        - ``model_info['density_right']``: estimated density from the right

    Notes
    -----
    Through 1.28.0 this was a different estimator under McCrary's name:
    ``ceil(sqrt(n/2))`` bins of a width set by the narrower side, a
    Silverman-type bandwidth, a sandwich SE, and silent fallbacks to a
    density of 0.01 with SE 0.1 when a side had fewer than two bins in the
    window. :func:`sp.rddensity` (Cattaneo, Jansson & Ma) is the modern
    replacement; this function exists for comparability with McCrary.

    References
    ----------
    mccrary2008manipulation

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> # Smooth running variable (no manipulation at the cutoff)
    >>> df = pd.DataFrame({"score": rng.normal(0, 1, size=500)})
    >>> result = sp.mccrary_test(df, x='score', c=0)
    >>> bool(result.pvalue > 0.05)   # no manipulation -> fail to reject
    True
    """
    X = data[x].to_numpy(dtype=float)
    X = X[np.isfinite(X)]
    rn = len(X)
    if rn < 20:
        raise ValueError("Need at least 20 observations for McCrary test.")
    rsd = float(np.std(X, ddof=1))
    rmin, rmax = float(X.min()), float(X.max())
    if c <= rmin or c >= rmax:
        raise ValueError("Cutpoint must lie within range of the running variable.")

    if bin_width is not None:
        b = float(bin_width)
    elif n_bins is not None:
        b = (rmax - rmin) / int(n_bins)
    else:
        b = 2 * rsd * rn ** (-0.5)
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"bin width must be positive; got {b}")

    def _mid(v):
        return np.floor((v - c) / b) * b + b / 2 + c

    l_ = _mid(rmin)
    r_ = _mid(rmax)
    lc, rc = c - b / 2, c + b / 2
    j = int(np.floor((rmax - rmin) / b)) + 2
    binnum = np.round(((_mid(X) - l_) / b) + 1).astype(int)  # 1-based
    if binnum.min() < 1 or binnum.max() > j:  # pragma: no cover - R would error
        raise ValueError("bin assignment fell outside the histogram grid")
    cellval = np.bincount(binnum - 1, minlength=j)[:j] / rn / b
    cellmp = _mid(l_ + np.arange(j) * b)

    if bw is None:
        leftofc = int(np.round(((_mid(lc) - l_) / b) + 1))
        rightofc = int(np.round(((_mid(rc) - l_) / b) + 1))
        if rightofc - leftofc != 1:  # pragma: no cover - mirrors R's guard
            raise ValueError("Error occurred in bandwidth calculation")

        def _side_h(mask, span, mp_eval):
            V = np.vander(cellmp[mask], 5, increasing=True)
            coef, *_ = np.linalg.lstsq(V, cellval[mask], rcond=None)
            resid = cellval[mask] - V @ coef
            mse4 = float(resid @ resid) / (int(mask.sum()) - 5)
            fpp = 2 * coef[2] + 6 * coef[3] * mp_eval + 12 * coef[4] * mp_eval**2
            return 3.348 * (mse4 * span / float(np.sum(fpp * fpp))) ** 0.2

        h_left = _side_h(cellmp < c, c - l_, cellmp[:leftofc])
        h_right = _side_h(cellmp >= c, r_ - c, cellmp[rightofc - 1 :])
        bw = 0.5 * (h_left + h_right)
    bw = float(bw)
    if not ((X > c - bw) & (X < c)).any() or not ((X < c + bw) & (X >= c)).any():
        raise ValueError("Insufficient data within the bandwidth.")

    pad = int(np.ceil(bw / b))
    cval, cmp_ = cellval, cellmp
    if pad >= 1:
        cval = np.concatenate([np.zeros(pad), cellval, np.zeros(pad)])
        cmp_ = np.concatenate(
            [
                (l_ - pad * b) + np.arange(pad) * b,  # R seq(l - pad*b, l - b, b)
                cellmp,
                (r_ + b) + np.arange(pad) * b,  # R seq(r + b, r + pad*b, b)
            ]
        )
    dist = cmp_ - c

    def _side_fit(mask):
        w = 1 - np.abs(dist / bw)
        w = np.where(w > 0, w * mask, 0.0)
        if np.count_nonzero(w) < 2:
            raise ValueError("Too few bins inside the bandwidth on one side.")
        sw = np.sqrt(w)
        A = np.column_stack([np.ones_like(dist), dist]) * sw[:, None]
        coef, *_ = np.linalg.lstsq(A, cval * sw, rcond=None)
        return float(coef[0])

    f_left = _side_fit(cmp_ < c)
    f_right = _side_fit(cmp_ >= c)
    if f_left <= 0 or f_right <= 0:
        raise ValueError(
            f"Non-positive density estimate at the cutoff (left {f_left:.4g}, "
            f"right {f_right:.4g}); the log-difference is undefined."
        )
    theta = float(np.log(f_right) - np.log(f_left))
    se_theta = float(np.sqrt((1 / (rn * bw)) * (24 / 5) * (1 / f_right + 1 / f_left)))
    z = theta / se_theta
    pvalue = float(2 * stats.norm.sf(abs(z)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (theta - z_crit * se_theta, theta + z_crit * se_theta)

    model_info = {
        "density_left": f_left,
        "density_right": f_right,
        "log_density_ratio": theta,
        "z": z,
        "bandwidth": bw,
        "bin_width": b,
        "n_bins": j,
        "cutoff": c,
        "n_left": int((X < c).sum()),
        "n_right": int((X >= c).sum()),
        "reference": "R rdd::DCdensity",
    }

    _result = CausalResult(
        method="McCrary (2008) Density Test",
        estimand="Log Density Ratio",
        estimate=theta,
        se=se_theta,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=rn,
        model_info=model_info,
        _citation_key="mccrary",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.diagnostics.mccrary_test",
            params={
                "x": x,
                "c": c,
                "bw": bw,
                "n_bins": n_bins,
                "bin_width": b,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
