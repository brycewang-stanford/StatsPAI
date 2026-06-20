"""
Goodman-Bacon (2021) decomposition for two-way fixed effects DID.

Decomposes the standard TWFE DID estimator into a weighted average of
all possible 2×2 DID comparisons, revealing which comparisons drive
the estimate and how much weight comes from "forbidden" comparisons
that use already-treated units as controls.

This is a **diagnostic** tool — it does not provide a bias-corrected
estimator. Use Callaway-Sant'Anna or Sun-Abraham for estimation.

References
----------
Goodman-Bacon, A. (2021).
"Difference-in-Differences with Variation in Treatment Timing."
*Journal of Econometrics*, 225(2), 254-277. [@goodmanbacon2021difference]

Goodman-Bacon, A., Goldring, T. and Nichols, A. (2019).
"BACONDECOMP: Stata module to perform the Bacon decomposition
of difference-in-differences estimation."
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


def bacon_decomposition(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Goodman-Bacon (2021) decomposition of the TWFE DID estimator.

    Decomposes the overall TWFE coefficient into a weighted sum of
    2×2 DID comparisons between different treatment timing groups.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment indicator (0 before treatment, 1 after).
    time : str
        Time period variable.
    id : str
        Unit identifier.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys:
        - ``beta_twfe``: overall TWFE estimate
        - ``decomposition``: pd.DataFrame with columns
          [type, treated, control, estimate, weight]
        - ``weighted_sum``: Σ(weight × estimate) — equals beta_twfe
          under the same dyad conventions as R ``bacondecomp`` and Stata
          ``bacondecomp``
        - ``n_comparisons``: number of 2×2 sub-comparisons
        - ``negative_weight_share``: fraction of signed weight mass on
          truly negative Bacon weights. This is usually zero; already-treated
          control comparisons are reported separately as
          ``already_treated_control_weight_share``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> cohorts = {1: 4, 2: 4, 3: 7, 4: 7, 5: 99, 6: 99}  # 99 = never treated
    >>> rows = []
    >>> for unit, g in cohorts.items():
    ...     for year in range(1, 11):
    ...         treated = int(year >= g)
    ...         y = unit + 0.3 * year + 2.0 * treated + rng.normal(0, 0.5)
    ...         rows.append({'unit': unit, 'year': year,
    ...                      'outcome': y, 'treated': treated})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.bacon_decomposition(df, y='outcome', treat='treated',
    ...                                 time='year', id='unit')
    >>> sorted(result['decomposition']['type'].unique())
    ['Earlier vs Later Treated', 'Later vs Earlier Treated', 'Treated vs Untreated']
    >>> bool(abs(result['weighted_sum'] - result['beta_twfe']) < 1e-8)
    True

    Notes
    -----
    The decomposition identifies three types of comparisons:

    1. **Earlier vs Later treated**: Units treated at time g₁ vs units
       treated later at g₂ (g₁ < g₂). These are "good" comparisons.
    2. **Later vs Earlier treated**: Units treated at g₂ vs already-treated
       units at g₁. These are "forbidden" — they use treated units as
       controls and can introduce negative weighting bias.
    3. **Treated vs Never treated**: Always valid comparisons.

    A large ``already_treated_control_weight_share`` signals that TWFE is
    relying heavily on comparisons with already-treated controls, so a
    heterogeneity-robust estimator (C&S, Sun-Abraham) should be used.

    See Goodman-Bacon (2021, *JEcon*), Theorem 1.
    """
    df = data.copy()

    # Validate balanced panel using the same requirement as bacondecomp:
    # each unit must contribute the same number of time observations.
    counts = df.groupby(id)[time].nunique()
    if counts.nunique() != 1:
        raise MethodIncompatibility(
            "Unbalanced Panel",
            recovery_hint=(
                "Goodman-Bacon decomposition requires the same number of "
                "time observations per unit; balance the panel or use a "
                "heterogeneity-robust DID estimator directly."
            ),
            diagnostics={"panel_balance_counts": counts.value_counts().to_dict()},
            alternative_functions=["sp.callaway_santanna", "sp.sun_abraham"],
        )

    time_periods = sorted(df[time].unique())
    time_min = min(time_periods)

    # Get treatment timing for each unit
    # g_i = first period where treat == 1 (inf for never-treated)
    unit_treat = df.groupby(id)[[treat, time]].apply(
        lambda grp: (
            grp.loc[grp[treat] == 1, time].min() if (grp[treat] == 1).any() else np.inf
        )
    )

    _validate_monotone_treatment(df, treat, time, id, unit_treat)
    df = df.merge(unit_treat.rename("_treat_time"), left_on=id, right_index=True)

    # Identify groups by treatment timing
    timing_groups = sorted(unit_treat.unique())
    never_treated = np.inf

    # Overall TWFE estimate (for reference)
    beta_twfe = _twfe_estimate(df, y, treat, time, id)

    # Enumerate all 2×2 comparisons
    comparisons = []

    for untreated_group in timing_groups:
        for treated_group in timing_groups:
            if treated_group == untreated_group:
                continue
            if treated_group == never_treated:
                continue
            if treated_group == time_min:
                continue

            data_pair = _subset_bacon_dyad(
                df, treated_group, untreated_group, time, "_treat_time"
            )
            if data_pair.empty:
                continue

            wt = _bacon_weight(
                data_pair, treat, "_treat_time", treated_group, untreated_group
            )
            if not np.isfinite(wt) or abs(wt) < 1e-15:
                continue

            est = _twfe_estimate(data_pair, y, treat, time, id)

            if untreated_group == never_treated:
                comp_type = "Treated vs Untreated"
                control = "Never"
            elif untreated_group == time_min:
                comp_type = "Later vs Always Treated"
                control = untreated_group
            elif treated_group > untreated_group:
                comp_type = "Later vs Earlier Treated"
                control = untreated_group
            else:
                comp_type = "Earlier vs Later Treated"
                control = untreated_group

            comparisons.append(
                {
                    "type": comp_type,
                    "treated": treated_group,
                    "control": control,
                    "estimate": est,
                    "weight": wt,
                }
            )

    decomp = pd.DataFrame(comparisons)

    if len(decomp) == 0:
        return {
            "beta_twfe": beta_twfe,
            "decomposition": decomp,
            "weighted_sum": 0.0,
            "n_comparisons": 0,
            "negative_weight_share": 0.0,
            "already_treated_control_weight_share": 0.0,
        }

    # Normalize weights to sum to 1
    total_w = decomp["weight"].sum()
    if total_w > 0:
        decomp["weight"] = decomp["weight"] / total_w

    weighted_sum = float((decomp["estimate"] * decomp["weight"]).sum())

    abs_weight_sum = float(decomp["weight"].abs().sum())
    if abs_weight_sum > 0:
        neg_share = float(
            decomp.loc[decomp["weight"] < 0, "weight"].abs().sum() / abs_weight_sum
        )
    else:
        neg_share = 0.0

    forbidden = decomp["type"].isin(
        {
            "Later vs Earlier Treated",
            "Later vs Always Treated",
        }
    )
    forbidden_share = float(decomp.loc[forbidden, "weight"].sum())

    return {
        "beta_twfe": beta_twfe,
        "decomposition": decomp,
        "weighted_sum": weighted_sum,
        "n_comparisons": len(decomp),
        "negative_weight_share": neg_share,
        "already_treated_control_weight_share": forbidden_share,
    }


def _twfe_estimate(
    df: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id_col: str,
) -> float:
    """Standard TWFE DID regression: Y_it = α_i + γ_t + β·D_it + ε_it."""
    # Demean by unit and time (within transformation)
    panel = df.set_index([id_col, time])
    Y = panel[y].unstack()
    D = panel[treat].unstack()

    # Double-demean
    y_dm = (
        Y.values
        - Y.values.mean(axis=1, keepdims=True)
        - Y.values.mean(axis=0, keepdims=True)
        + Y.values.mean()
    )
    d_dm = (
        D.values
        - D.values.mean(axis=1, keepdims=True)
        - D.values.mean(axis=0, keepdims=True)
        + D.values.mean()
    )

    y_flat = y_dm.ravel()
    d_flat = d_dm.ravel()
    valid = np.isfinite(y_flat) & np.isfinite(d_flat)

    if valid.sum() < 2 or np.var(d_flat[valid]) < 1e-12:
        return 0.0

    beta = float(np.sum(d_flat[valid] * y_flat[valid]) / np.sum(d_flat[valid] ** 2))
    return beta


def _validate_monotone_treatment(
    df: pd.DataFrame,
    treat: str,
    time: str,
    id_col: str,
    unit_treat: pd.Series,
) -> None:
    """Require absorbing treatment timing."""
    merged = df.merge(
        unit_treat.rename("_treat_time"),
        left_on=id_col,
        right_index=True,
    )
    expected = np.where(
        np.isinf(merged["_treat_time"].to_numpy(dtype=float)),
        0,
        (merged[time].to_numpy() >= merged["_treat_time"].to_numpy()).astype(int),
    )
    observed = merged[treat].to_numpy()
    if not np.array_equal(observed.astype(int), expected.astype(int)):
        raise MethodIncompatibility(
            "Treatment not weakly increasing with time",
            recovery_hint=(
                "Pass an absorbing 0/1 treatment-status column to "
                "bacon_decomposition, not a cohort/first-treatment column."
            ),
            diagnostics={"treatment_column": treat},
            alternative_functions=["sp.did_analysis", "sp.callaway_santanna"],
        )


def _subset_bacon_dyad(
    df: pd.DataFrame,
    treated_group: float,
    untreated_group: float,
    time: str,
    treat_time: str,
) -> pd.DataFrame:
    """Subset one bacondecomp dyad and its admissible comparison window."""
    data_pair = df[df[treat_time].isin([treated_group, untreated_group])].copy()
    if treated_group < untreated_group:
        data_pair = data_pair[data_pair[time] < untreated_group]
    elif treated_group > untreated_group:
        data_pair = data_pair[data_pair[time] >= untreated_group]
    return data_pair


def _bacon_weight(
    data_pair: pd.DataFrame,
    treat: str,
    treat_time: str,
    treated_group: float,
    untreated_group: float,
) -> float:
    """Uncontrolled Goodman-Bacon weight used by R/Stata bacondecomp."""
    if untreated_group == np.inf:
        n_u = float((data_pair[treat_time] == untreated_group).sum())
        n_k = float((data_pair[treat_time] == treated_group).sum())
        if n_k + n_u == 0:
            return 0.0
        n_ku = n_k / (n_k + n_u)
        D_k = float(data_pair.loc[data_pair[treat_time] == treated_group, treat].mean())
        V_ku = n_ku * (1 - n_ku) * D_k * (1 - D_k)
        return float((n_k + n_u) ** 2 * V_ku)

    if treated_group < untreated_group:
        n_k = float((data_pair[treat_time] == treated_group).sum())
        n_l = float((data_pair[treat_time] == untreated_group).sum())
        if n_k + n_l == 0:
            return 0.0
        n_kl = n_k / (n_k + n_l)
        D_k = float(data_pair.loc[data_pair[treat_time] == treated_group, treat].mean())
        D_l = float(
            data_pair.loc[data_pair[treat_time] == untreated_group, treat].mean()
        )
        denom = 1 - D_l
        if abs(denom) < 1e-15:
            return 0.0
        V_kl = n_kl * (1 - n_kl) * (D_k - D_l) / denom * (1 - D_k) / denom
        return float(((n_k + n_l) * denom) ** 2 * V_kl)

    n_k = float((data_pair[treat_time] == untreated_group).sum())
    n_l = float((data_pair[treat_time] == treated_group).sum())
    if n_k + n_l == 0:
        return 0.0
    n_kl = n_k / (n_k + n_l)
    D_k = float(data_pair.loc[data_pair[treat_time] == untreated_group, treat].mean())
    D_l = float(data_pair.loc[data_pair[treat_time] == treated_group, treat].mean())
    if abs(D_k) < 1e-15:
        return 0.0
    V_kl = n_kl * (1 - n_kl) * (D_l / D_k) * (D_k - D_l) / D_k
    return float(((n_k + n_l) * D_k) ** 2 * V_kl)


# Citation
CausalResult._CITATIONS["bacon_decomposition"] = (
    "@article{goodmanbacon2021difference,\n"
    "  title={Difference-in-Differences with Variation in Treatment "
    "Timing},\n"
    "  author={Goodman-Bacon, Andrew},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={225},\n"
    "  number={2},\n"
    "  pages={254--277},\n"
    "  year={2021},\n"
    "  publisher={Elsevier}\n"
    "}"
)
