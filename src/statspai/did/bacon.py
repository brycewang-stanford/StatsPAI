"""
Goodman-Bacon (2021) decomposition for two-way fixed effects DID.

Decomposes the standard TWFE DID estimator into a weighted average of
all possible 2×2 DID comparisons, revealing which comparisons drive
the estimate and whether "forbidden" comparisons (using already-treated
units as controls) contribute negative weights.

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

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from ..core.results import CausalResult


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
        - ``weighted_sum``: Σ(weight × estimate) — should equal beta_twfe
        - ``n_comparisons``: number of 2×2 sub-comparisons
        - ``negative_weight_share``: fraction of weight on comparisons
          where already-treated units serve as controls (the "forbidden"
          comparisons that can bias TWFE)

    Examples
    --------
    >>> result = bacon_decomposition(df, y='outcome', treat='treated',
    ...                              time='year', id='unit')
    >>> print(result['decomposition'])
    >>> print(f"TWFE = {result['beta_twfe']:.4f}")
    >>> print(f"Negative weight share = {result['negative_weight_share']:.1%}")

    Notes
    -----
    The decomposition identifies three types of comparisons:

    1. **Earlier vs Later treated**: Units treated at time g₁ vs units
       treated later at g₂ (g₁ < g₂). These are "good" comparisons.
    2. **Later vs Earlier treated**: Units treated at g₂ vs already-treated
       units at g₁. These are "forbidden" — they use treated units as
       controls and can introduce negative weighting bias.
    3. **Treated vs Never treated**: Always valid comparisons.

    A large ``negative_weight_share`` signals that TWFE is unreliable
    and a heterogeneity-robust estimator (C&S, Sun-Abraham) should be used.

    See Goodman-Bacon (2021, *JEcon*), Theorem 1.
    """
    df = data.copy()

    # Validate balanced panel
    panel = df.pivot_table(index=id, columns=time, values=y, aggfunc='first')
    n_units = len(panel)
    time_periods = sorted(df[time].unique())
    T = len(time_periods)

    # Get treatment timing for each unit
    # g_i = first period where treat == 1 (inf for never-treated)
    unit_treat = df.groupby(id)[[treat, time]].apply(
        lambda grp: grp.loc[grp[treat] == 1, time].min()
        if (grp[treat] == 1).any() else np.inf
    )

    # Identify groups by treatment timing
    timing_groups = sorted(unit_treat.unique())
    never_treated = np.inf

    # Overall TWFE estimate (for reference)
    beta_twfe = _twfe_estimate(df, y, treat, time, id)

    # Enumerate all 2×2 comparisons
    comparisons = []

    for i, g1 in enumerate(timing_groups):
        for g2 in timing_groups[i + 1:]:
            if g1 == never_treated:
                continue

            units_g1 = unit_treat[unit_treat == g1].index
            units_g2 = unit_treat[unit_treat == g2].index

            n1 = len(units_g1)
            n2 = len(units_g2)

            if n1 == 0 or n2 == 0:
                continue

            if g2 == never_treated:
                # Type: Treated vs Never-treated
                comp_type = 'Treated vs Never-treated'
                est, wt = _pairwise_did(
                    panel, units_g1, units_g2, g1, time_periods,
                    n1, n2, n_units, T,
                )
                comparisons.append({
                    'type': comp_type,
                    'treated': g1,
                    'control': 'Never',
                    'estimate': est,
                    'weight': wt,
                })
            else:
                # Type 1: Earlier (g1) vs Later (g2) — "good"
                est1, wt1 = _pairwise_did(
                    panel, units_g1, units_g2, g1, time_periods,
                    n1, n2, n_units, T,
                )
                comparisons.append({
                    'type': 'Earlier vs Later treated',
                    'treated': g1,
                    'control': g2,
                    'estimate': est1,
                    'weight': wt1,
                })

                # Type 2: Later (g2) vs Earlier (g1) — "forbidden"
                est2, wt2 = _pairwise_did(
                    panel, units_g2, units_g1, g2, time_periods,
                    n2, n1, n_units, T,
                )
                comparisons.append({
                    'type': 'Later vs Already-treated',
                    'treated': g2,
                    'control': g1,
                    'estimate': est2,
                    'weight': wt2,
                })

    decomp = pd.DataFrame(comparisons)

    if len(decomp) == 0:
        return {
            'beta_twfe': beta_twfe,
            'decomposition': decomp,
            'weighted_sum': 0.0,
            'n_comparisons': 0,
            'negative_weight_share': 0.0,
        }

    # Normalize weights to sum to 1
    total_w = decomp['weight'].sum()
    if total_w > 0:
        decomp['weight'] = decomp['weight'] / total_w

    weighted_sum = float((decomp['estimate'] * decomp['weight']).sum())

    # Negative weight share (from "forbidden" comparisons)
    forbidden = decomp['type'] == 'Later vs Already-treated'
    neg_share = float(decomp.loc[forbidden, 'weight'].sum())

    return {
        'beta_twfe': beta_twfe,
        'decomposition': decomp,
        'weighted_sum': weighted_sum,
        'n_comparisons': len(decomp),
        'negative_weight_share': neg_share,
    }


def _twfe_estimate(df, y, treat, time, id_col):
    """Standard TWFE DID regression: Y_it = α_i + γ_t + β·D_it + ε_it."""
    # Demean by unit and time (within transformation)
    panel = df.set_index([id_col, time])
    Y = panel[y].unstack()
    D = panel[treat].unstack()

    # Double-demean
    y_dm = Y.values - Y.values.mean(axis=1, keepdims=True) \
           - Y.values.mean(axis=0, keepdims=True) + Y.values.mean()
    d_dm = D.values - D.values.mean(axis=1, keepdims=True) \
           - D.values.mean(axis=0, keepdims=True) + D.values.mean()

    y_flat = y_dm.ravel()
    d_flat = d_dm.ravel()
    valid = np.isfinite(y_flat) & np.isfinite(d_flat)

    if valid.sum() < 2 or np.var(d_flat[valid]) < 1e-12:
        return 0.0

    beta = float(np.sum(d_flat[valid] * y_flat[valid]) /
                 np.sum(d_flat[valid] ** 2))
    return beta


def _pairwise_did(panel, units_treated, units_control, treat_time,
                  time_periods, n_t, n_c, n_total, T):
    """Compute 2×2 DID and Bacon weight for a pair of groups."""
    # Pre/post periods relative to treat_time
    pre_periods = [t for t in time_periods if t < treat_time]
    post_periods = [t for t in time_periods if t >= treat_time]

    if not pre_periods or not post_periods:
        return 0.0, 0.0

    # Outcome means
    y_t_pre = panel.loc[units_treated, pre_periods].values.mean()
    y_t_post = panel.loc[units_treated, post_periods].values.mean()
    y_c_pre = panel.loc[units_control, pre_periods].values.mean()
    y_c_post = panel.loc[units_control, post_periods].values.mean()

    did_est = (y_t_post - y_t_pre) - (y_c_post - y_c_pre)

    # Bacon weight ∝ n_k × n_l × V(D̃_kl)
    n_k = n_t
    n_l = n_c
    s = len(post_periods) / T  # share of post-treatment periods
    var_d = s * (1 - s)  # variance of demeaned treatment in this 2×2
    weight = (n_k + n_l) * var_d

    return float(did_est), float(weight)


# Citation
CausalResult._CITATIONS['bacon_decomposition'] = (
    "@article{goodman2021difference,\n"
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
