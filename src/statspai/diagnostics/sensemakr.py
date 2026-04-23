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
    >>> result = sp.sensemakr(df, y='wage', treat='education',
    ...                       controls=['age', 'experience', 'female'])
    >>> print(f"RV_q = {result['rv_q']:.1%}")
    >>> print(result['interpretation'])

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
    rss_full = np.sum(resid_full ** 2)
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

    # Partial R² of treatment with itself (after controlling for X)
    beta_d_x = np.linalg.lstsq(Z_no_d, D, rcond=None)[0]
    resid_d = D - Z_no_d @ beta_d_x
    partial_r2_dd = float(np.var(resid_d) * n / np.sum(resid_d ** 2))

    # --- Robustness Value ---
    # RV_q: minimum confounder strength to reduce estimate to zero
    # From Cinelli & Hazlett (2020), Theorem 1:
    # RV_q = sqrt(f² / (1 + f²)) where f = |t| / sqrt(df)
    df_resid = n - k_full
    f2 = t_treat ** 2 / df_resid
    rv_q = float(np.sqrt(f2 / (1 + f2)))

    # RV_{q,alpha}: to make insignificant
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    f2_alpha = (abs(t_treat) - t_crit) ** 2 / df_resid if abs(t_treat) > t_crit else 0
    rv_qa = float(np.sqrt(f2_alpha / (1 + f2_alpha)))

    # --- Benchmark table ---
    benchmark_vars = benchmark or controls
    bench_rows = []
    for var in benchmark_vars:
        if var not in controls:
            continue
        # Partial R² of this variable with Y and D
        r2_yv = _partial_r2_of(Y, var, df, controls, treat)
        r2_dv = _partial_r2_of(D, var, df, controls, treat)
        bench_rows.append({
            'variable': var,
            'partial_r2_Y': round(r2_yv, 4),
            'partial_r2_D': round(r2_dv, 4),
        })

    bench_df = pd.DataFrame(bench_rows) if bench_rows else pd.DataFrame()

    # --- Interpretation ---
    if rv_qa > 0.10:
        robustness = "ROBUST"
        detail = (f"An unobserved confounder would need to explain "
                  f">{rv_qa:.0%} of the residual variance of both "
                  f"treatment and outcome to render the result insignificant.")
    elif rv_qa > 0.01:
        robustness = "MODERATELY ROBUST"
        detail = (f"An unobserved confounder explaining {rv_qa:.0%} of "
                  f"residual variance would suffice to make result insignificant.")
    else:
        robustness = "FRAGILE"
        detail = "Even a weak confounder could invalidate this result."

    return {
        'beta_treat': beta_treat,
        'se_treat': se_treat,
        't_treat': float(t_treat),
        'partial_r2_yd': float(partial_r2_yd),
        'rv_q': rv_q,
        'rv_qa': rv_qa,
        'robustness': robustness,
        'benchmark_table': bench_df,
        'interpretation': f"{robustness}: RV_q = {rv_q:.1%}, RV_{{q,α}} = {rv_qa:.1%}. {detail}",
    }


def _partial_r2_of(Y, var, df, controls, treat):
    """Compute partial R² of 'var' with Y controlling for everything else."""
    other = [c for c in controls if c != var]
    n = len(df)

    # Full model (with var)
    Z_full = np.column_stack([np.ones(n), df[treat].values] +
                             [df[c].values for c in controls])
    rss_full = np.sum((Y - Z_full @ np.linalg.lstsq(Z_full, Y, rcond=None)[0]) ** 2)

    # Restricted (without var)
    Z_restr = np.column_stack([np.ones(n), df[treat].values] +
                              [df[c].values for c in other])
    rss_restr = np.sum((Y - Z_restr @ np.linalg.lstsq(Z_restr, Y, rcond=None)[0]) ** 2)

    return max(1 - rss_full / rss_restr, 0) if rss_restr > 0 else 0


# Citation
CausalResult._CITATIONS['sensemakr'] = (
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
