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

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


@accepts_aliases(covariates="controls")
def sensemakr(
    data: pd.DataFrame,
    y: str,
    treat: str,
    controls: List[str],
    benchmark: Optional[List[str]] = None,
    alpha: float = 0.05,
    kd: Union[float, Sequence[float]] = 1.0,
    ky: Optional[Union[float, Sequence[float]]] = None,
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
        Observed control variables (included in the regression). A
        string, categorical or boolean column enters as a set of
        indicators with the first level omitted, like a factor in R.
    benchmark : list of str, optional
        Subset of controls to use as benchmarks — "if the unobserved
        confounder were as strong as [benchmark], would the result
        survive?" Default: every control. A categorical control is
        benchmarked as a group (all its indicators together).
    alpha : float, default 0.05
    kd, ky : float or sequence of float, default 1 and ``kd``
        How many times as strong as the benchmark the confounder is
        assumed to be, in explaining the treatment (``kd``) and the
        outcome (``ky``). ``kd=[1, 2, 3]`` gives one row per multiple
        for each benchmark, as ``sensemakr(kd = 1:3)`` does in R.

    Returns
    -------
    dict
        ``'rv_q'``: Robustness Value — minimum partial R² to change sign
        ``'rv_qa'``: RV for significance (to make p > alpha)
        ``'adjusted_estimate'``: bias-adjusted β for given confounder
        ``'benchmark_table'``: DataFrame with one row per benchmark and
        multiple: the implied confounder strengths ``r2dz_x`` and
        ``r2yz_dx``, and the estimate, standard error, t statistic and
        confidence interval adjusted for a confounder of that strength
        pulling the estimate toward zero
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
    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    blocks = {c: _control_columns(df[c]) for c in controls}
    X = np.empty((len(Y), 0))
    if controls:
        X = np.column_stack([blocks[c] for c in controls])
    n = len(Y)

    # Full regression: Y ~ D + X
    Z_full = np.column_stack([np.ones(n), D, X])
    beta_full = np.linalg.lstsq(Z_full, Y, rcond=None)[0]
    resid_full = Y - Z_full @ beta_full
    rss_full = np.sum(resid_full**2)
    k_full = int(np.linalg.matrix_rank(Z_full))

    # Treatment coefficient and SE
    sigma2 = rss_full / (n - k_full)
    if k_full == Z_full.shape[1]:
        # (Z'Z)^-1[1, 1] from the QR factor: forming Z'Z squares the
        # condition number and costs eight digits on unscaled regressors.
        r_factor = np.linalg.qr(Z_full, mode="r")
        e1 = np.zeros(k_full)
        e1[1] = 1.0
        xtx_inv_dd = float(np.sum(np.linalg.solve(r_factor.T, e1) ** 2))
    else:
        xtx_inv_dd = float(np.linalg.pinv(Z_full.T @ Z_full)[1, 1])
    se_treat = float(np.sqrt(sigma2 * xtx_inv_dd))
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
    kd_list = [float(k) for k in np.atleast_1d(kd)]
    ky_list = kd_list if ky is None else [float(k) for k in np.atleast_1d(ky)]
    if len(ky_list) != len(kd_list):
        raise MethodIncompatibility(
            "sensemakr: kd and ky must have the same length.",
            recovery_hint="Pass one ky per kd, or leave ky unset to use ky = kd.",
        )
    benchmark_vars = benchmark or controls
    # sensemakr uses the regression's own degrees of freedom here
    t_crit = float(stats.t.ppf(1 - alpha / 2.0, df=df_resid))
    bench_rows = []
    for var in benchmark_vars:
        if var not in controls:
            continue
        # Raw partial R² of this observed covariate with Y and D. The
        # treatment-side regression must not include D itself on the RHS.
        r2_yv = _partial_r2_of(Y, var, blocks, treat=D)
        r2_dv = _partial_r2_of(D, var, blocks, treat=None)
        for kd_j, ky_j in zip(kd_list, ky_list):
            r2dz_x, r2yz_dx = _sensemakr_bound_scale(r2_dv, r2_yv, kd=kd_j, ky=ky_j)
            if r2yz_dx >= 1.0:
                warnings.warn(
                    f"sensemakr: a confounder {kd_j:g}x/{ky_j:g}x as strong as "
                    f"{var!r} implies a partial R2 with the outcome above 1; "
                    "it is set to 1. Try lower kd / ky.",
                    stacklevel=2,
                )
            # sensemakr::adjusted_estimate / adjusted_se with reduce = TRUE
            bias = float(
                np.sqrt(r2yz_dx * r2dz_x / (1.0 - r2dz_x))
                * se_treat
                * np.sqrt(df_resid)
            )
            adj_est = float(np.sign(beta_treat) * (abs(beta_treat) - bias))
            adj_se = float(
                np.sqrt((1.0 - r2yz_dx) / (1.0 - r2dz_x))
                * se_treat
                * np.sqrt(df_resid / (df_resid - 1.0))
            )
            bench_rows.append(
                {
                    "variable": var,
                    "kd": kd_j,
                    "ky": ky_j,
                    "partial_r2_Y": round(r2_yv, 4),
                    "partial_r2_D": round(r2_dv, 4),
                    "r2dz_x": float(r2dz_x),
                    "r2yz_dx": float(r2yz_dx),
                    "adjusted_estimate": adj_est,
                    "adjusted_se": adj_se,
                    "adjusted_t": adj_est / adj_se if adj_se > 0 else float("nan"),
                    "adjusted_lower_CI": adj_est - t_crit * adj_se,
                    "adjusted_upper_CI": adj_est + t_crit * adj_se,
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


def _control_columns(col: pd.Series) -> np.ndarray:
    """Design columns of one control: itself, or indicators for a factor."""
    if pd.api.types.is_bool_dtype(col):
        return col.to_numpy(dtype=float).reshape(-1, 1)
    if pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(dtype=float).reshape(-1, 1)
    dummies = pd.get_dummies(col.astype("category"), drop_first=True)
    return dummies.to_numpy(dtype=float)


def _partial_r2_of(
    Y: np.ndarray,
    var: str,
    blocks: Dict[str, np.ndarray],
    treat: Optional[np.ndarray],
) -> float:
    """Partial R² of control ``var`` (all its columns) with Y given the rest."""
    n = len(Y)
    base = [np.ones((n, 1))]
    if treat is not None:
        base.append(np.asarray(treat, dtype=float).reshape(-1, 1))

    # Full model (with var)
    Z_full = np.column_stack(base + list(blocks.values()))
    rss_full = np.sum((Y - Z_full @ np.linalg.lstsq(Z_full, Y, rcond=None)[0]) ** 2)

    # Restricted (without var)
    Z_restr = np.column_stack(base + [b for c, b in blocks.items() if c != var])
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
