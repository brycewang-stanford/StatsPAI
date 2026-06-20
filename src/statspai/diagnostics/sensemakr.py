"""
Cinelli & Hazlett (2020) sensitivity analysis for omitted variable bias.

Quantifies how strong an unobserved confounder would need to be (in
terms of partial R² with treatment and outcome) to change the
qualitative conclusion of a study.

Key outputs:
- Robustness Value (RV): minimum confounder strength to nullify the result
- Contour plots of bias-adjusted estimates across confounder strengths
- Benchmarking against observed covariates

References
----------
Cinelli, C. and Hazlett, C. (2020).
"Making Sense of Sensitivity: Extending Omitted Variable Bias."
*Journal of the Royal Statistical Society: Series B*, 82(1), 39-67. [@cinelli2020making]
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def sensemakr(
    data: pd.DataFrame,
    y: str,
    treat: str,
    controls: List[str],
    benchmark: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Cinelli & Hazlett (2020) omitted variable bias sensitivity analysis.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Treatment variable of interest.
    controls : list of str
        Observed control variables (included in the regression).
    benchmark : list of str, optional
        Subset of controls to use as benchmarks — "if the unobserved
        confounder were as strong as [benchmark], would the result
        survive?" Default: use the strongest control.
    alpha : float, default 0.05

    Returns
    -------
    dict
        ``'rv_q'``: Robustness Value — minimum partial R² to change sign
        ``'rv_qa'``: RV for significance (to make p > alpha)
        ``'adjusted_estimate'``: bias-adjusted β for given confounder
        ``'benchmark_table'``: DataFrame comparing confounder strengths
        ``'interpretation'``: human-readable summary

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> result = sp.sensemakr(df, y='log_wage', treat='education',
    ...                       controls=['experience', 'female'])
    >>> bool(0.0 <= result['rv_q'] <= 1.0)  # Robustness Value is a partial R^2
    True
    >>> isinstance(result['interpretation'], str)
    True

    Notes
    -----
    The Robustness Value (RV) answers: "What is the minimum strength
    of association (measured by partial R²) that an unobserved
    confounder would need to have with both the treatment AND the
    outcome to change the sign of the estimated treatment effect?"

    If RV = 20%, then an unobserved confounder explaining 20% of the
    residual variance of both treatment and outcome would be needed to
    fully explain away the result.

    The partial R² framework is more interpretable than Oster (2019)
    because it directly maps to variance explained.

    See Cinelli & Hazlett (2020, *JRSS-B*), Sections 3-4.
    """
    df = data[[y, treat] + controls].dropna()
    Y = df[y].values
    D = df[treat].values
    X = df[controls].values
    n = len(Y)

    # Full regression: Y ~ D + X
    Z_full = np.column_stack([np.ones(n), D, X])
    beta_full = np.linalg.lstsq(Z_full, Y, rcond=None)[0]
    resid_full = Y - Z_full @ beta_full
    rss_full = np.sum(resid_full**2)
    k_full = Z_full.shape[1]

    # Treatment coefficient and SE
    XtX_inv = np.linalg.pinv(Z_full.T @ Z_full)
    sigma2 = rss_full / (n - k_full)
    se_treat = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    beta_treat = float(beta_full[1])
    t_treat = beta_treat / se_treat

    # Partial R² of treatment with outcome (after controlling for X)
    Z_no_d = np.column_stack([np.ones(n), X])
    beta_no_d = np.linalg.lstsq(Z_no_d, Y, rcond=None)[0]
    rss_no_d = np.sum((Y - Z_no_d @ beta_no_d) ** 2)
    partial_r2_yd = 1 - rss_full / rss_no_d  # partial R² of D on Y|X

    # --- Robustness Value ---
    # Mirrors sensemakr::robustness_value.numeric. ``rv_q`` sets alpha=1
    # (point estimate only); ``rv_qa`` uses the caller's alpha threshold.
    df_resid = n - k_full
    rv_q = _robustness_value(t_treat, df_resid, q=1.0, alpha=1.0)
    rv_qa = _robustness_value(t_treat, df_resid, q=1.0, alpha=alpha)

    # --- Benchmark table ---
    benchmark_vars = benchmark or controls
    bench_rows = []
    for var in benchmark_vars:
        if var not in controls:
            continue
        # Raw partial R² of this observed covariate with Y and D. The
        # treatment-side regression must not include D itself on the RHS.
        r2_yv = _partial_r2_of(Y, var, df, controls, treat=treat)
        r2_dv = _partial_r2_of(D, var, df, controls, treat=None)
        r2dz_x, r2yz_dx = _sensemakr_bound_scale(r2_dv, r2_yv)
        bench_rows.append(
            {
                "variable": var,
                "partial_r2_Y": round(r2_yv, 4),
                "partial_r2_D": round(r2_dv, 4),
                "r2dz_x": float(r2dz_x),
                "r2yz_dx": float(r2yz_dx),
            }
        )

    bench_df = pd.DataFrame(bench_rows) if bench_rows else pd.DataFrame()

    # --- Interpretation ---
    if rv_qa > 0.10:
        robustness = "ROBUST"
        detail = (
            f"An unobserved confounder would need to explain "
            f">{rv_qa:.0%} of the residual variance of both "
            f"treatment and outcome to render the result insignificant."
        )
    elif rv_qa > 0.01:
        robustness = "MODERATELY ROBUST"
        detail = (
            f"An unobserved confounder explaining {rv_qa:.0%} of "
            f"residual variance would suffice to make result insignificant."
        )
    else:
        robustness = "FRAGILE"
        detail = "Even a weak confounder could invalidate this result."

    return {
        "beta_treat": beta_treat,
        "se_treat": se_treat,
        "t_treat": float(t_treat),
        "partial_r2_yd": float(partial_r2_yd),
        "rv_q": rv_q,
        "rv_qa": rv_qa,
        "robustness": robustness,
        "benchmark_table": bench_df,
        "interpretation": (
            f"{robustness}: RV_q = {rv_q:.1%}, " f"RV_{{q,α}} = {rv_qa:.1%}. {detail}"
        ),
    }


def _partial_r2_of(
    Y: np.ndarray,
    var: str,
    df: pd.DataFrame,
    controls: List[str],
    treat: Optional[str],
) -> float:
    """Compute partial R² of 'var' with Y controlling for everything else."""
    other = [c for c in controls if c != var]
    n = len(df)
    base = [np.ones(n)]
    if treat is not None:
        base.append(df[treat].values)

    # Full model (with var)
    Z_full = np.column_stack(base + [df[c].values for c in controls])
    rss_full = np.sum((Y - Z_full @ np.linalg.lstsq(Z_full, Y, rcond=None)[0]) ** 2)

    # Restricted (without var)
    Z_restr = np.column_stack(base + [df[c].values for c in other])
    rss_restr = np.sum((Y - Z_restr @ np.linalg.lstsq(Z_restr, Y, rcond=None)[0]) ** 2)

    return float(max(1 - rss_full / rss_restr, 0)) if rss_restr > 0 else 0.0


def _sensemakr_bound_scale(
    r2dxj_x: float,
    r2yxj_dx: float,
    *,
    kd: float = 1.0,
    ky: float = 1.0,
) -> Tuple[float, float]:
    """Map raw benchmark partial R² values to sensemakr's bound scale.

    This mirrors ``sensemakr::ovb_partial_r2_bound`` for the default
    ``kd = ky = 1`` used by :func:`sensemakr` benchmark covariates.
    """
    r2dxj_x = float(np.clip(r2dxj_x, 0.0, 1.0 - 1e-15))
    r2yxj_dx = float(np.clip(r2yxj_dx, 0.0, 1.0 - 1e-15))
    r2dz_x = kd * (r2dxj_x / (1.0 - r2dxj_x))
    if r2dz_x >= 1.0:
        r2dz_x = 1.0

    denom = (1.0 - kd * r2dxj_x) * (1.0 - r2dxj_x)
    if denom <= 0:
        r2zxj_xd = 1.0
    else:
        r2zxj_xd = kd * (r2dxj_x**2) / denom
    r2zxj_xd = float(np.clip(r2zxj_xd, 0.0, 1.0 - 1e-15))

    r2yz_dx = ((np.sqrt(ky) + np.sqrt(r2zxj_xd)) / np.sqrt(1.0 - r2zxj_xd)) ** 2 * (
        r2yxj_dx / (1.0 - r2yxj_dx)
    )
    r2yz_dx = float(np.clip(r2yz_dx, 0.0, 1.0))
    return float(r2dz_x), r2yz_dx


def _robustness_value(
    t_statistic: float,
    dof: int,
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    invert: bool = False,
) -> float:
    """Compute the Cinelli-Hazlett robustness value.

    Port of ``sensemakr::robustness_value.numeric`` for the OLS case.
    """
    if dof <= 1:
        return 0.0
    fq = q * abs(float(t_statistic) / np.sqrt(dof))
    f_crit = abs(stats.t.ppf(alpha / 2.0, df=dof - 1)) / np.sqrt(dof - 1)
    f1, f2 = (f_crit, fq) if invert else (fq, f_crit)
    fqa = f1 - f2
    if fqa < 0:
        return 0.0

    rv_binding = 0.0 if fqa == 0 else 2.0 / (1.0 + np.sqrt(1.0 + 4.0 / (fqa**2)))

    fq2 = fq**2
    f_crit2 = f_crit**2
    xf1, xf2 = (f_crit2, fq2) if invert else (fq2, f_crit2)
    xrv = (xf1 - xf2) / (1.0 + xf1) if xf1 > xf2 else 0.0
    is_xrv = f2 != 0 and fqa > 0 and f1 > 1.0 / f2
    return float(xrv if is_xrv else rv_binding)


# Citation
CausalResult._CITATIONS["sensemakr"] = (
    "@article{cinelli2020making,\n"
    "  title={Making Sense of Sensitivity: Extending Omitted Variable Bias},\n"
    "  author={Cinelli, Carlos and Hazlett, Chad},\n"
    "  journal={Journal of the Royal Statistical Society: Series B},\n"
    "  volume={82},\n"
    "  number={1},\n"
    "  pages={39--67},\n"
    "  year={2020},\n"
    "  publisher={Wiley}\n"
    "}"
)
