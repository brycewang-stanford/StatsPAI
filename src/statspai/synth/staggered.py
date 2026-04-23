"""
Staggered Adoption Synthetic Control.

Extends the synthetic control method to settings where multiple units
receive treatment at different times. Uses partially-pooled SCM weights
to borrow strength across treated cohorts while respecting the
staggered timing structure.

Model
-----
For each treated unit i with treatment time g_i:

    τ̂_i = Y_i^post - ω̂_i' Y_0^post

Then the overall ATT is the (weighted) average across treated units,
with optional pooling of SCM weights across cohorts sharing the same
adoption time.

References
----------
Ben-Michael, E., Feller, A. and Rothstein, J. (2022).
"Synthetic Controls with Staggered Adoption."
*Journal of the Royal Statistical Society: Series B*, 84(2), 351-381. [@benmichael2022synthetic]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult


def staggered_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    method: Literal["separate", "pooled"] = "separate",
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Staggered Adoption Synthetic Control.

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
    treatment : str
        Binary treatment indicator (0/1). Units transition from 0 to 1
        at their respective adoption times.
    method : {'separate', 'pooled'}, default 'separate'
        * ``'separate'`` — fit a separate SCM for each treated unit.
        * ``'pooled'`` — partially pool weights across cohorts with
          the same adoption time.
    penalization : float, default 0.0
        Ridge penalty on donor weights.
    placebo : bool, default True
        Run placebo inference.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        With ``model_info`` containing per-unit and per-cohort effects.

    Examples
    --------
    >>> result = sp.staggered_synth(df, outcome='gdp', unit='state',
    ...     time='year', treatment='treated')
    >>> print(result.summary())
    """
    # --- Identify treated units and their adoption times ---
    panel = data.pivot_table(index=unit, columns=time, values=treatment, aggfunc="first")
    all_times = sorted(panel.columns.tolist())
    all_units = panel.index.tolist()

    # Find adoption time for each unit (first period where treatment = 1)
    adoption_times: Dict[Any, Any] = {}
    never_treated: List[Any] = []

    for u in all_units:
        tvals = panel.loc[u]
        treated_periods = tvals[tvals == 1].index.tolist()
        if len(treated_periods) > 0:
            adoption_times[u] = min(treated_periods)
        else:
            never_treated.append(u)

    if len(adoption_times) == 0:
        raise ValueError("No treated units found")
    if len(never_treated) == 0:
        raise ValueError("No never-treated (pure control) units found")

    # --- Outcome panel ---
    outcome_panel = data.pivot_table(
        index=unit, columns=time, values=outcome, aggfunc="first"
    )

    # --- Group by cohort (same adoption time) ---
    cohorts: Dict[Any, List[Any]] = {}
    for u, g in adoption_times.items():
        cohorts.setdefault(g, []).append(u)

    # --- Fit SCM per unit (or per cohort) ---
    unit_results = []

    for cohort_time, cohort_units in sorted(cohorts.items()):
        pre_times_c = [t for t in all_times if t < cohort_time]
        post_times_c = [t for t in all_times if t >= cohort_time]

        if len(pre_times_c) < 2 or len(post_times_c) < 1:
            continue

        # Donor pool: never-treated + not-yet-treated at cohort_time
        donors_c = never_treated.copy()
        for u, g in adoption_times.items():
            if g > cohort_time and u not in cohort_units:
                donors_c.append(u)

        if len(donors_c) < 1:
            continue

        Y0_pre = outcome_panel.loc[donors_c, pre_times_c].values.astype(np.float64)
        Y0_post = outcome_panel.loc[donors_c, post_times_c].values.astype(np.float64)

        if method == "pooled":
            # Pool: use average of cohort units as target
            Y1_pre = outcome_panel.loc[cohort_units, pre_times_c].mean(axis=0).values
            Y1_post = outcome_panel.loc[cohort_units, post_times_c].mean(axis=0).values

            weights = _solve_weights(Y1_pre, Y0_pre.T, penalization)
            Y1_hat = Y0_post.T @ weights
            effects = Y1_post - Y1_hat
            att_c = float(np.mean(effects))

            for u in cohort_units:
                Y_u_post = outcome_panel.loc[u, post_times_c].values.astype(np.float64)
                eff_u = Y_u_post - Y1_hat
                unit_results.append({
                    "unit": u,
                    "cohort_time": cohort_time,
                    "att": float(np.mean(eff_u)),
                    "n_post": len(post_times_c),
                    "n_pre": len(pre_times_c),
                    "weights": weights,
                    "donors": donors_c,
                })
        else:  # separate
            for u in cohort_units:
                Y1_pre_u = outcome_panel.loc[u, pre_times_c].values.astype(np.float64)
                Y1_post_u = outcome_panel.loc[u, post_times_c].values.astype(np.float64)

                try:
                    weights = _solve_weights(Y1_pre_u, Y0_pre.T, penalization)
                    Y1_hat = Y0_post.T @ weights
                    effects = Y1_post_u - Y1_hat
                    unit_results.append({
                        "unit": u,
                        "cohort_time": cohort_time,
                        "att": float(np.mean(effects)),
                        "n_post": len(post_times_c),
                        "n_pre": len(pre_times_c),
                        "weights": weights,
                        "donors": donors_c,
                    })
                except Exception:
                    continue

    if len(unit_results) == 0:
        raise ValueError("Could not estimate effects for any treated unit")

    # --- Aggregate ATT ---
    # Weight by number of post-periods
    atts = np.array([r["att"] for r in unit_results])
    n_posts = np.array([r["n_post"] for r in unit_results], dtype=np.float64)
    weights_agg = n_posts / n_posts.sum()
    att = float(np.sum(atts * weights_agg))

    # --- Placebo inference ---
    placebo_atts_list = []
    if placebo and len(never_treated) >= 3:
        for p_unit in never_treated:
            p_atts = []
            remaining_donors = [u for u in never_treated if u != p_unit]

            for cohort_time in sorted(cohorts.keys()):
                pre_t = [t for t in all_times if t < cohort_time]
                post_t = [t for t in all_times if t >= cohort_time]
                if len(pre_t) < 2 or len(post_t) < 1 or len(remaining_donors) < 1:
                    continue

                Y0p = outcome_panel.loc[remaining_donors, pre_t].values.astype(np.float64)
                Y0p_post = outcome_panel.loc[remaining_donors, post_t].values.astype(np.float64)
                Yp_pre = outcome_panel.loc[p_unit, pre_t].values.astype(np.float64)
                Yp_post = outcome_panel.loc[p_unit, post_t].values.astype(np.float64)

                try:
                    w = _solve_weights(Yp_pre, Y0p.T, penalization)
                    hat = Y0p_post.T @ w
                    p_atts.append(float(np.mean(Yp_post - hat)))
                except Exception:
                    continue

            if p_atts:
                placebo_atts_list.append(float(np.mean(p_atts)))

    if len(placebo_atts_list) > 1:
        se = float(np.std(placebo_atts_list, ddof=1))
        pvalue = float(np.mean(np.abs(placebo_atts_list) >= abs(att)))
        pvalue = max(pvalue, 1 / (len(placebo_atts_list) + 1))
    else:
        se = float(np.std(atts, ddof=1)) / max(np.sqrt(len(atts)), 1)
        pvalue = np.nan

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z_crit * se, att + z_crit * se)

    # --- Build detail tables ---
    unit_df = pd.DataFrame([
        {"unit": r["unit"], "cohort_time": r["cohort_time"],
         "att": r["att"], "n_pre": r["n_pre"], "n_post": r["n_post"]}
        for r in unit_results
    ])

    cohort_df = unit_df.groupby("cohort_time").agg(
        att=("att", "mean"),
        n_units=("unit", "count"),
    ).reset_index()

    model_info = {
        "method": method,
        "n_treated_units": len(adoption_times),
        "n_control_units": len(never_treated),
        "n_cohorts": len(cohorts),
        "cohort_times": sorted(cohorts.keys()),
        "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
        "unit_effects": unit_df,
        "cohort_effects": cohort_df,
        "penalization": penalization,
        "all_times": all_times,
    }

    if placebo_atts_list:
        model_info["placebo_atts"] = placebo_atts_list
        model_info["n_placebos"] = len(placebo_atts_list)

    return CausalResult(
        method="Staggered Synthetic Control (Ben-Michael et al. 2022)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=unit_df,
        model_info=model_info,
        _citation_key="staggered_synth",
    )


def _solve_weights(
    y: np.ndarray,
    X: np.ndarray,
    penalization: float = 0.0,
) -> np.ndarray:
    """min ||y - X w||^2 + pen ||w||^2  s.t. w >= 0, sum(w) = 1."""
    J = X.shape[1]
    if J == 0:
        raise ValueError("No donor units")

    def objective(w):
        r = y - X @ w
        loss = r @ r
        if penalization > 0:
            loss += penalization * (w @ w)
        return loss

    def jac(w):
        r = y - X @ w
        g = -2 * X.T @ r
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
CausalResult._CITATIONS["staggered_synth"] = (
    "@article{benmichael2022synthetic,\n"
    "  title={Synthetic Controls with Staggered Adoption},\n"
    "  author={Ben-Michael, Eli and Feller, Avi and Rothstein, Jesse},\n"
    "  journal={Journal of the Royal Statistical Society: Series B},\n"
    "  volume={84},\n"
    "  number={2},\n"
    "  pages={351--381},\n"
    "  year={2022},\n"
    "  publisher={Wiley}\n"
    "}"
)
