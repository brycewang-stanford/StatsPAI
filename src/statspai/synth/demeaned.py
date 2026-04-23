"""
De-meaned / De-trended Synthetic Control Method.

Addresses transitory shocks that can deteriorate pre-treatment fit
and bias the standard SCM estimator. Two approaches:

* **demeaned** — subtract unit-specific pre-treatment means before
  optimising weights, then add them back. Removes level differences.
* **detrended** — remove unit-specific linear time trends before
  optimising, then add them back. Removes both level and slope
  differences.

References
----------
Ferman, B. and Pinto, C. (2021).
"Synthetic Control Method: Inference, Sensitivity, and Confidence Sets."
*Journal of the American Statistical Association*, 116(536), 1835-1847. [@ferman2021synthetic]

Doudchenko, N. and Imbens, G.W. (2016).
"Balancing, Regression, Difference-in-Differences and Synthetic
Control Methods: A Synthesis." NBER Working Paper 22791. [@doudchenko2016balancing]
"""

from __future__ import annotations

from typing import Any, List, Optional, Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult


def demeaned_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    variant: Literal["demeaned", "detrended"] = "demeaned",
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    De-meaned / De-trended Synthetic Control Method.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period (inclusive).
    covariates : list of str, optional
        Additional covariates to match on.
    variant : {'demeaned', 'detrended'}, default 'demeaned'
        * ``'demeaned'`` — subtract unit-level pre-treatment means.
        * ``'detrended'`` — subtract unit-level linear time trends.
    penalization : float, default 0.0
        Ridge penalty on weights.
    placebo : bool, default True
        Run in-space placebo inference.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> result = sp.demeaned_synth(df, outcome='gdp', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989)
    >>> print(result.summary())
    """
    # --- Build panel matrix ---
    pivot = data.pivot_table(index=time, columns=unit, values=outcome)
    times = pivot.index.values
    pre_mask = times < treatment_time
    post_mask = times >= treatment_time

    if pre_mask.sum() < 2:
        raise ValueError("Need at least 2 pre-treatment periods")
    if post_mask.sum() < 1:
        raise ValueError("Need at least 1 post-treatment period")

    Y_treated = pivot[treated_unit].values.astype(np.float64)
    donor_cols = [c for c in pivot.columns if c != treated_unit]
    Y_donors = pivot[donor_cols].values.astype(np.float64)  # (T, J)

    # Drop donors with NaN in pre-period
    pre_donors = Y_donors[pre_mask]
    valid = ~np.any(np.isnan(pre_donors), axis=0)
    if valid.sum() == 0:
        raise ValueError("No valid donor units")
    Y_donors = Y_donors[:, valid]
    donor_cols = [donor_cols[i] for i in range(len(donor_cols)) if valid[i]]
    J = Y_donors.shape[1]

    # --- De-mean or de-trend ---
    time_numeric = np.arange(len(times), dtype=np.float64)
    pre_idx = np.where(pre_mask)[0]

    if variant == "demeaned":
        # Subtract pre-treatment means
        mean_treated = np.mean(Y_treated[pre_mask])
        means_donors = np.mean(Y_donors[pre_mask], axis=0)  # (J,)
        Y_treated_adj = Y_treated - mean_treated
        Y_donors_adj = Y_donors - means_donors[np.newaxis, :]
    elif variant == "detrended":
        # Subtract unit-specific linear trends fit on pre-period
        def _detrend(y, t_pre, t_all):
            slope, intercept = np.polyfit(t_pre, y[pre_mask], 1)
            return y - (intercept + slope * t_all), intercept, slope

        Y_treated_adj, tr_int, tr_slope = _detrend(
            Y_treated, time_numeric[pre_mask], time_numeric
        )
        Y_donors_adj = np.empty_like(Y_donors)
        donor_params = []
        for j in range(J):
            Y_donors_adj[:, j], d_int, d_slope = _detrend(
                Y_donors[:, j], time_numeric[pre_mask], time_numeric
            )
            donor_params.append((d_int, d_slope))
    else:
        raise ValueError(f"variant must be 'demeaned' or 'detrended', got {variant!r}")

    # --- Solve weights on adjusted data ---
    Y_pre_treated = Y_treated_adj[pre_mask]
    Y_pre_donors = Y_donors_adj[pre_mask]

    weights = _solve_weights(Y_pre_treated, Y_pre_donors, penalization)

    # --- Compute synthetic with intercept correction ---
    # The gap is computed in the adjusted space, then the synthetic
    # in original space includes the treated unit's level/trend.
    gap_adj = Y_treated_adj - Y_donors_adj @ weights
    if variant == "demeaned":
        # Synthetic in original space: mean_treated + adjusted_synthetic
        Y_synth = mean_treated + Y_donors_adj @ weights
    else:
        # Detrended: add back treated unit's trend
        Y_synth = (tr_int + tr_slope * time_numeric) + Y_donors_adj @ weights
    gap = Y_treated - Y_synth
    gap_post = gap[post_mask]
    gap_pre = gap[pre_mask]
    att = float(np.mean(gap_post))
    pre_mspe = float(np.mean(gap_pre ** 2))

    # --- Placebo inference ---
    placebo_atts = []
    placebo_pre_mspes = []
    if placebo and J >= 2:
        all_Y = np.column_stack([Y_treated[:, np.newaxis], Y_donors])
        all_Y_adj = np.column_stack([Y_treated_adj[:, np.newaxis], Y_donors_adj])

        # Pre-treatment means/trends for each unit (for intercept correction)
        if variant == "demeaned":
            all_means = np.concatenate([[mean_treated], means_donors])
        else:
            all_params = [(tr_int, tr_slope)] + donor_params

        for i in range(J):
            idx_p = i + 1
            Y_p = all_Y[:, idx_p]
            Y_p_adj = all_Y_adj[:, idx_p]
            didx = [j for j in range(all_Y.shape[1]) if j != idx_p]
            Y_d = all_Y[:, didx]
            Y_d_adj = all_Y_adj[:, didx]

            try:
                w = _solve_weights(Y_p_adj[pre_mask], Y_d_adj[pre_mask], penalization)
                if variant == "demeaned":
                    synth_p = all_means[idx_p] + Y_d_adj @ w
                else:
                    p_int, p_slope = all_params[idx_p]
                    synth_p = (p_int + p_slope * time_numeric) + Y_d_adj @ w
                gap_p = Y_p - synth_p
                placebo_atts.append(float(np.mean(gap_p[post_mask])))
                placebo_pre_mspes.append(float(np.mean(gap_p[pre_mask] ** 2)))
            except Exception:
                continue

    # --- P-value ---
    if len(placebo_atts) > 0:
        post_mspe = float(np.mean(gap_post ** 2))
        ratio_treated = post_mspe / pre_mspe if pre_mspe > 1e-10 else np.inf
        placebo_ratios = [
            a ** 2 / m if m > 1e-10 else 0
            for a, m in zip(placebo_atts, placebo_pre_mspes)
        ]
        pvalue = float(np.mean(np.array(placebo_ratios) >= ratio_treated))
        pvalue = max(pvalue, 1 / (len(placebo_ratios) + 1))
        se = float(np.std(placebo_atts)) if len(placebo_atts) > 1 else 0.0
    else:
        pvalue = np.nan
        se = float(np.std(gap_post)) / max(np.sqrt(len(gap_post)), 1)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z_crit * se, att + z_crit * se)

    weight_df = pd.DataFrame({
        "unit": donor_cols, "weight": weights,
    }).sort_values("weight", ascending=False).reset_index(drop=True)
    weight_df = weight_df[weight_df["weight"] > 1e-6]

    gap_df = pd.DataFrame({
        "time": times, "treated": Y_treated, "synthetic": Y_synth,
        "gap": gap, "post_treatment": post_mask,
    })

    variant_label = "De-meaned" if variant == "demeaned" else "De-trended"

    model_info = {
        "variant": variant,
        "n_donors": J,
        "n_pre_periods": int(pre_mask.sum()),
        "n_post_periods": int(post_mask.sum()),
        "pre_treatment_mspe": round(pre_mspe, 6),
        "pre_treatment_rmse": round(np.sqrt(pre_mspe), 6),
        "penalization": penalization,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "weights": weight_df,
        "gap_table": gap_df,
        "Y_synth": Y_synth,
        "Y_treated": Y_treated,
        "times": times,
    }

    if placebo_atts:
        model_info["placebo_atts"] = placebo_atts
        model_info["n_placebos"] = len(placebo_atts)

    return CausalResult(
        method=f"{variant_label} Synthetic Control (Ferman & Pinto 2021)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(Y_treated),
        detail=weight_df,
        model_info=model_info,
        _citation_key="demeaned_synth",
    )


def _solve_weights(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    penalization: float = 0.0,
) -> np.ndarray:
    """min ||y - X w||^2 + pen ||w||^2  s.t. w >= 0, sum(w) = 1."""
    J = Y_donors_pre.shape[1]
    if J == 0:
        raise ValueError("No donor units available")

    def objective(w):
        r = Y_treated_pre - Y_donors_pre @ w
        loss = r @ r
        if penalization > 0:
            loss += penalization * (w @ w)
        return loss

    def jac(w):
        r = Y_treated_pre - Y_donors_pre @ w
        g = -2 * Y_donors_pre.T @ r
        if penalization > 0:
            g += 2 * penalization * w
        return g

    res = optimize.minimize(
        objective, np.ones(J) / J, jac=jac, method="SLSQP",
        bounds=[(0, 1)] * J,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x


# Citation
CausalResult._CITATIONS["demeaned_synth"] = (
    "@article{ferman2021synthetic,\n"
    "  title={Synthetic Control Method: Inference, Sensitivity, "
    "and Confidence Sets},\n"
    "  author={Ferman, Bruno and Pinto, Cristine},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1835--1847},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
