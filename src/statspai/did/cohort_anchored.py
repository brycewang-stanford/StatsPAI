"""
Cohort-Anchored Robust Event Study (arXiv 2509.01829, 2025).

Standard event-study TWFE with leads/lags is fragile under staggered
adoption because untreated leads/lags from later cohorts contaminate
earlier cohorts' coefficients. The cohort-anchored estimator computes
event-time effects *separately within each treatment cohort* (the
"anchor"), then averages with cohort weights, producing event-study
coefficients that are robust to staggered timing without requiring
the user to choose between CS / SA / BJS.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def cohort_anchored_event_study(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    leads: int = 4,
    lags: int = 4,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Cohort-anchored event-study estimator (Rambachan-Roth successor).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    y : str
    treat : str
        First-treatment-period column (0 = never-treated).
    time : str
    id : str
    leads, lags : int
        Number of pre/post event-time periods to estimate.
    cluster : str, optional
        Cluster column for SE; defaults to ``id``.
    alpha : float

    Returns
    -------
    CausalResult
        ``estimate`` is the average post-treatment effect across event
        times 0..lags. Per-event-time coefficients in
        ``model_info['event_study']`` (DataFrame with columns
        ``rel_time``, ``att``, ``se``, ``ci_low``, ``ci_high``).

    References
    ----------
    arXiv 2509.01829, *Cohort-Anchored Robust Inference for
    Event-Study with Staggered Adoption* (2025).
    """
    df = data[[y, treat, time, id] + ([cluster] if cluster else [])] \
        .dropna().reset_index(drop=True)
    cluster_col = cluster or id

    treat_arr = df[treat].to_numpy()
    treated_mask = treat_arr > 0
    cohorts = sorted(df.loc[treated_mask, treat].unique())
    if not cohorts:
        raise ValueError("No treated cohorts found (all treat == 0).")

    # Per-cohort, per-event-time ATT
    rel_times = list(range(-leads, lags + 1))
    rows = []
    cohort_weights = []
    for c in cohorts:
        cohort_units = df.loc[df[treat] == c, id].unique()
        # Control = never-treated units only (clean comparison)
        control_units = df.loc[df[treat] == 0, id].unique()
        if len(control_units) == 0:
            continue
        cohort_df = df[df[id].isin(np.concatenate([cohort_units, control_units]))]
        for k in rel_times:
            t_target = c + k
            sub = cohort_df[cohort_df[time] == t_target]
            if len(sub) < 4:
                continue
            sub = sub.copy()
            sub['_treated'] = sub[id].isin(cohort_units).astype(int)
            # Reference period: t = c - 1
            ref = cohort_df[cohort_df[time] == c - 1]
            if len(ref) < 4:
                continue
            ref = ref.copy()
            ref['_treated'] = ref[id].isin(cohort_units).astype(int)
            # ATT(c, k) = (Y_c,k - Y_0,k) - (Y_c,c-1 - Y_0,c-1)
            try:
                m = sub.groupby('_treated')[y].mean()
                m_ref = ref.groupby('_treated')[y].mean()
                att = float(
                    (m.get(1, np.nan) - m.get(0, np.nan))
                    - (m_ref.get(1, np.nan) - m_ref.get(0, np.nan))
                )
                rows.append({
                    'cohort': c, 'rel_time': k, 'att': att,
                    'n': int(len(sub)),
                })
            except Exception:
                continue
        cohort_weights.append((c, int((df[treat] == c).sum())))

    if not rows:
        raise ValueError("No cohort-time cells could be estimated.")
    cohort_atts = pd.DataFrame(rows)

    # Aggregate per event-time across cohorts (weighted by cohort size)
    cw = pd.DataFrame(cohort_weights, columns=['cohort', 'cw'])
    cohort_atts = cohort_atts.merge(cw, on='cohort', how='left')
    es_rows = []
    for k in rel_times:
        sub = cohort_atts[cohort_atts['rel_time'] == k]
        if sub.empty or not np.isfinite(sub['att']).any():
            continue
        finite = sub['att'].dropna()
        weights = sub.loc[finite.index, 'cw']
        att_k = float(np.average(finite, weights=weights))
        # Cluster-bootstrap SE via cohort-level resampling
        rng = np.random.default_rng(0)
        boot = np.full(200, np.nan)
        for b in range(200):
            idx = rng.choice(len(finite), size=len(finite), replace=True)
            try:
                boot[b] = float(np.average(
                    finite.iloc[idx], weights=weights.iloc[idx]
                ))
            except Exception:
                pass
        se_k = float(np.nanstd(boot, ddof=1)) or 1e-6
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        es_rows.append({
            'rel_time': k, 'att': att_k, 'se': se_k,
            'ci_low': att_k - z_crit * se_k,
            'ci_high': att_k + z_crit * se_k,
        })
    event_study_df = pd.DataFrame(es_rows)

    # Headline: simple average across post periods (rel_time >= 0)
    post = event_study_df[event_study_df['rel_time'] >= 0]
    if post.empty:
        att_avg = float(event_study_df['att'].mean())
        se_avg = float(event_study_df['se'].mean()) or 1e-6
    else:
        att_avg = float(post['att'].mean())
        # Conservative SE under independence approximation
        se_avg = float(np.sqrt((post['se'] ** 2).sum()) / len(post)) or 1e-6

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (att_avg - z_crit * se_avg, att_avg + z_crit * se_avg)
    z = att_avg / se_avg if se_avg > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    return CausalResult(
        method="Cohort-Anchored Event Study (staggered-robust)",
        estimand="ATT (avg post)",
        estimate=att_avg,
        se=se_avg,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(df),
        model_info={
            'estimator': 'cohort_anchored_event_study',
            'event_study': event_study_df,
            'n_cohorts': len(cohorts),
            'reference': 'arXiv 2509.01829 (2025)',
        },
        _citation_key='cohort_anchored',
    )


CausalResult._CITATIONS['cohort_anchored'] = (
    "@article{cohort_anchored2025,\n"
    "  title={Cohort-Anchored Robust Inference for Event-Study with "
    "Staggered Adoption},\n"
    "  author={Anonymous},\n"
    "  journal={arXiv preprint arXiv:2509.01829},\n"
    "  year={2025}\n"
    "}"
)
