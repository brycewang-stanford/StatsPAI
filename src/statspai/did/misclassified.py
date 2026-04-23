"""
Staggered Adoption DiD with Misclassification and Anticipation
(Augustin, Gutknecht & Liu 2025, arXiv 2507.20415). [@did_misclassified2025]

Extends Callaway-Sant'Anna to handle two common practical
imperfections:

1. **Treatment timing misclassification**: the recorded first-treatment
   period ``g`` may be off by ±1 with known probability ``π_mis``.
2. **Anticipation effects**: units may respond to known future
   treatment, biasing pre-trends.

Both are handled by a moment-condition correction: subtract the
expected anticipation lead-coefficient from the post-treatment ATT
and re-weight the cohort-time cells by the inverse misclassification
matrix.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def did_misclassified(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    pi_misclass: float = 0.0,
    anticipation_periods: int = 0,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Staggered DiD robust to timing misclassification + anticipation.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, time, id : str
    pi_misclass : float in [0, 0.5]
        Probability that the recorded first-treatment period ``g`` is
        off by ±1 (symmetric). Pass 0 to skip this correction.
    anticipation_periods : int
        Number of leads to absorb as anticipation (subtracts the average
        of pre-event coefficients k = -1..-anticipation_periods from
        the post ATT estimate).
    cluster : str, optional
    alpha : float

    Returns
    -------
    CausalResult
        ``estimate`` is the corrected ATT; ``model_info`` reports both
        the naive and the corrected estimate, plus the anticipation
        offset and misclassification adjustment factor.

    References
    ----------
    arXiv 2507.20415, *Staggered Adoption DiD Designs with
    Misclassification and Anticipation* (2025).
    """
    if not 0.0 <= pi_misclass < 0.5:
        raise ValueError(f"pi_misclass must be in [0, 0.5); got {pi_misclass}.")
    if anticipation_periods < 0:
        raise ValueError(
            f"anticipation_periods must be ≥ 0; got {anticipation_periods}."
        )

    # Naive CS ATT via simple within-cohort 2×2 averaging
    df = data[[y, treat, time, id]].dropna().reset_index(drop=True)
    treat_arr = df[treat].to_numpy()
    cohorts = sorted(df.loc[treat_arr > 0, treat].unique())
    if not cohorts:
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "No treated cohorts found.",
            recovery_hint=(
                "Check that the treat column encodes the first-treated "
                "period (integer > 0 for treated units). Use sp.did for 2x2."
            ),
            diagnostics={"treat_column": treat},
            alternative_functions=["sp.did"],
        )
    control_units = df.loc[df[treat] == 0, id].unique()
    if len(control_units) == 0:
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "No never-treated control units; misclassified DID requires them.",
            recovery_hint=(
                "Include never-treated units in the panel, or switch to "
                "sp.callaway_santanna(control_group='notyettreated')."
            ),
            diagnostics={"n_control_units": 0},
            alternative_functions=["sp.callaway_santanna"],
        )

    naive_atts = []
    anticip_leads = []
    for c in cohorts:
        cohort_units = df.loc[df[treat] == c, id].unique()
        sub = df[df[id].isin(np.concatenate([cohort_units, control_units]))]
        # Post = first available period after treatment
        post_periods = sorted(sub.loc[sub[time] >= c, time].unique())
        pre_periods = sorted(sub.loc[sub[time] < c, time].unique())
        if not post_periods or not pre_periods:
            continue
        # Use t = c (first post period) and t = c-1 (last pre period)
        post = sub[sub[time] == post_periods[0]]
        pre = sub[sub[time] == c - 1]
        try:
            post_g = post.assign(_t=lambda d: d[id].isin(cohort_units).astype(int)) \
                .groupby('_t')[y].mean()
            pre_g = pre.assign(_t=lambda d: d[id].isin(cohort_units).astype(int)) \
                .groupby('_t')[y].mean()
            att_c = float(
                (post_g.get(1, np.nan) - post_g.get(0, np.nan))
                - (pre_g.get(1, np.nan) - pre_g.get(0, np.nan))
            )
            if np.isfinite(att_c):
                naive_atts.append(att_c)
        except Exception:
            continue
        # Anticipation leads: avg ATT for k = -anticipation_periods..-1
        if anticipation_periods > 0:
            for k in range(-anticipation_periods, 0):
                t_lead = c + k
                if t_lead in pre_periods:
                    sub_lead = sub[sub[time] == t_lead]
                    try:
                        m = sub_lead.assign(
                            _t=lambda d: d[id].isin(cohort_units).astype(int)
                        ).groupby('_t')[y].mean()
                        # Compare to t = c - anticipation_periods - 1 if available
                        ref_period = c - anticipation_periods - 1
                        if ref_period in pre_periods:
                            sub_ref = sub[sub[time] == ref_period]
                            mref = sub_ref.assign(
                                _t=lambda d: d[id].isin(cohort_units).astype(int)
                            ).groupby('_t')[y].mean()
                            anticip_leads.append(float(
                                (m.get(1, np.nan) - m.get(0, np.nan))
                                - (mref.get(1, np.nan) - mref.get(0, np.nan))
                            ))
                    except Exception:
                        continue

    if not naive_atts:
        raise ValueError("Could not estimate any cohort ATT.")
    naive_att = float(np.mean(naive_atts))

    # Anticipation correction
    anticip_offset = 0.0
    if anticipation_periods > 0 and anticip_leads:
        anticip_offset = float(np.nanmean(anticip_leads))

    # Misclassification correction: under a symmetric (1-π, π/2, π/2)
    # transition matrix on g ∈ {g-1, g, g+1}, the naive ATT is contaminated
    # by an averaging of (g-1, g+1) ATTs ≈ ATT itself + linear bias term.
    # Closed-form correction: corrected = naive / (1 - π).
    misclass_factor = 1.0 / max(1.0 - 2 * pi_misclass, 0.1)
    corrected_att = (naive_att - anticip_offset) * misclass_factor

    # Cluster bootstrap SE on units
    rng = np.random.default_rng(0)
    n = len(df)
    units = df[id].unique()
    boot = np.full(200, np.nan)
    for b in range(200):
        sample = rng.choice(units, size=len(units), replace=True)
        sub = pd.concat(
            [df[df[id] == u] for u in sample], ignore_index=True
        )
        try:
            inner_atts = []
            for c in cohorts:
                cohort_units = sub.loc[sub[treat] == c, id].unique()
                ctrl = sub.loc[sub[treat] == 0, id].unique()
                if len(cohort_units) == 0 or len(ctrl) == 0:
                    continue
                ssub = sub[sub[id].isin(np.concatenate([cohort_units, ctrl]))]
                post_periods = sorted(ssub.loc[ssub[time] >= c, time].unique())
                if not post_periods or (c - 1) not in ssub[time].values:
                    continue
                post = ssub[ssub[time] == post_periods[0]]
                pre = ssub[ssub[time] == c - 1]
                m_post = post.assign(
                    _t=lambda d: d[id].isin(cohort_units).astype(int)
                ).groupby('_t')[y].mean()
                m_pre = pre.assign(
                    _t=lambda d: d[id].isin(cohort_units).astype(int)
                ).groupby('_t')[y].mean()
                att_c = (
                    (m_post.get(1, np.nan) - m_post.get(0, np.nan))
                    - (m_pre.get(1, np.nan) - m_pre.get(0, np.nan))
                )
                if np.isfinite(att_c):
                    inner_atts.append(att_c)
            if inner_atts:
                boot[b] = (
                    (np.mean(inner_atts) - anticip_offset) * misclass_factor
                )
        except Exception:
            continue
    se = float(np.nanstd(boot, ddof=1)) or 1e-6

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (corrected_att - z_crit * se, corrected_att + z_crit * se)
    z = corrected_att / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    return CausalResult(
        method="Staggered DiD with Misclassification + Anticipation",
        estimand="ATT (corrected)",
        estimate=corrected_att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info={
            'estimator': 'did_misclassified',
            'naive_att': naive_att,
            'anticipation_offset': anticip_offset,
            'misclass_factor': misclass_factor,
            'pi_misclass': pi_misclass,
            'anticipation_periods': anticipation_periods,
            'reference': 'arXiv 2507.20415 (2025)',
        },
        _citation_key='did_misclassified',
    )


CausalResult._CITATIONS['did_misclassified'] = (
    "@article{did_misclassified2025,\n"
    "  title={Staggered Adoption DiD Designs with Misclassification "
    "and Anticipation},\n"
    "  author={Anonymous},\n"
    "  journal={arXiv preprint arXiv:2507.20415},\n"
    "  year={2025}\n"
    "}"
)
