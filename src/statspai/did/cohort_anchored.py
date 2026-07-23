"""
Cohort-Anchored Robust Event Study (Liu 2025, arXiv 2509.01829). [@cohort_anchored2025]

Standard event-study TWFE with leads/lags is fragile under staggered
adoption because untreated leads/lags from later cohorts contaminate
earlier cohorts' coefficients. The cohort-anchored estimator computes
event-time effects *separately within each treatment cohort* (the
"anchor"), then averages with cohort weights, producing event-study
coefficients that are robust to staggered timing without requiring
the user to choose between CS / SA / BJS.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core._bootstrap import bootstrap_se as _bootstrap_se
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

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True,
    ...                 seed=0)
    >>> df['first_treat'] = df['first_treat'].fillna(0).astype(int)
    >>> res = sp.cohort_anchored_event_study(
    ...     df, y='y', treat='first_treat', time='time', id='unit',
    ...     leads=2, lags=2)
    >>> round(float(res.estimate), 4)
    0.3668
    >>> list(res.model_info['event_study'].columns)
    ['rel_time', 'att', 'se', 'ci_low', 'ci_high']
    """
    df = (
        data[[y, treat, time, id] + ([cluster] if cluster else [])]
        .dropna()
        .reset_index(drop=True)
    )
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
            sub["_treated"] = sub[id].isin(cohort_units).astype(int)
            # Reference period: t = c - 1
            ref = cohort_df[cohort_df[time] == c - 1]
            if len(ref) < 4:
                continue
            ref = ref.copy()
            ref["_treated"] = ref[id].isin(cohort_units).astype(int)
            # ATT(c, k) = (Y_c,k - Y_0,k) - (Y_c,c-1 - Y_0,c-1)
            try:
                m = sub.groupby("_treated")[y].mean()
                m_ref = ref.groupby("_treated")[y].mean()
                att = float(
                    (m.get(1, np.nan) - m.get(0, np.nan))
                    - (m_ref.get(1, np.nan) - m_ref.get(0, np.nan))
                )
                rows.append(
                    {
                        "cohort": c,
                        "rel_time": k,
                        "att": att,
                        "n": int(len(sub)),
                    }
                )
            except Exception:
                continue
        cohort_weights.append((c, int((df[treat] == c).sum())))

    if not rows:
        raise ValueError("No cohort-time cells could be estimated.")
    cohort_atts = pd.DataFrame(rows)

    # Aggregate per event-time across cohorts (weighted by cohort size)
    cw = pd.DataFrame(cohort_weights, columns=["cohort", "cw"])
    cohort_atts = cohort_atts.merge(cw, on="cohort", how="left")
    # Per-event-time point estimates (cohort-size weighted)
    att_by_k: dict = {}
    for k in rel_times:
        sub = cohort_atts[cohort_atts["rel_time"] == k]
        if sub.empty or not np.isfinite(sub["att"]).any():
            continue
        finite = sub["att"].dropna()
        weights = sub.loc[finite.index, "cw"]
        att_by_k[k] = float(np.average(finite, weights=weights))
    valid_ks = list(att_by_k.keys())

    # Joint cluster-bootstrap SE: resample clusters (user-specified
    # ``cluster`` column, or the unit id by default) ONCE per draw and
    # recompute the whole ATT(k) vector on that draw.  This honors a
    # user-supplied cluster level (e.g. state, region, firm) instead of
    # silently collapsing to cohort-level resampling.
    # ⚠️ correctness fix (2026-07): the historical version ran an
    # independent bootstrap loop per event time and aggregated the
    # headline SE as sqrt(sum se_k^2)/m — an independence approximation
    # across event times, which share a reference period and control
    # group.  The joint bootstrap captures the cross-period covariance
    # directly: the headline SE is the bootstrap SD of the post-period
    # average itself.
    rng = np.random.default_rng(0)
    cluster_ids = df[cluster_col].unique()
    n_boot = 200
    boot_mat = np.full((n_boot, len(valid_ks)), np.nan)
    for b in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        # Resample full observations belonging to sampled clusters
        pieces = [df[df[cluster_col] == cid] for cid in sampled]
        if not pieces:
            continue
        df_b = pd.concat(pieces, ignore_index=True)
        cohort_cache = {}
        for c_b in cohorts:
            cohort_units_b = df_b.loc[df_b[treat] == c_b, id].unique()
            control_units_b = df_b.loc[df_b[treat] == 0, id].unique()
            if len(cohort_units_b) == 0 or len(control_units_b) == 0:
                continue
            cohort_cache[c_b] = (cohort_units_b, control_units_b)
        for j, k in enumerate(valid_ks):
            try:
                att_vals_b = []
                w_vals_b = []
                for c_b, (cohort_units_b, control_units_b) in cohort_cache.items():
                    sub_b = df_b[
                        (df_b[time] == c_b + k)
                        & df_b[id].isin(
                            np.concatenate([cohort_units_b, control_units_b])
                        )
                    ]
                    ref_b = df_b[
                        (df_b[time] == c_b - 1)
                        & df_b[id].isin(
                            np.concatenate([cohort_units_b, control_units_b])
                        )
                    ]
                    if len(sub_b) < 2 or len(ref_b) < 2:
                        continue
                    sub_b = sub_b.assign(_tr=sub_b[id].isin(cohort_units_b).astype(int))
                    ref_b = ref_b.assign(_tr=ref_b[id].isin(cohort_units_b).astype(int))
                    m_b = sub_b.groupby("_tr")[y].mean()
                    mr_b = ref_b.groupby("_tr")[y].mean()
                    att_b = (m_b.get(1, np.nan) - m_b.get(0, np.nan)) - (
                        mr_b.get(1, np.nan) - mr_b.get(0, np.nan)
                    )
                    if np.isfinite(att_b):
                        att_vals_b.append(att_b)
                        w_vals_b.append(int((df_b[treat] == c_b).sum()))
                if att_vals_b:
                    boot_mat[b, j] = float(np.average(att_vals_b, weights=w_vals_b))
            except Exception:
                continue  # cell stays NaN; bootstrap_se tracks failures

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    es_rows = []
    for j, k in enumerate(valid_ks):
        att_k = att_by_k[k]
        se_k = _bootstrap_se(boot_mat[:, j], label=f"did.cohort_anchored[k={k}]")
        es_rows.append(
            {
                "rel_time": k,
                "att": att_k,
                "se": se_k,
                "ci_low": att_k - z_crit * se_k,
                "ci_high": att_k + z_crit * se_k,
            }
        )
    event_study_df = pd.DataFrame(es_rows)

    # Headline: simple average across post periods (rel_time >= 0);
    # falls back to all periods when no post period is estimable.
    post_js = [j for j, k in enumerate(valid_ks) if k >= 0]
    head_js = post_js if post_js else list(range(len(valid_ks)))
    att_avg = float(np.mean([att_by_k[valid_ks[j]] for j in head_js]))
    head_boot = boot_mat[:, head_js]
    # A draw contributes only when every headline event time succeeded,
    # so each replicate averages the same set of periods as the estimate.
    complete = np.all(np.isfinite(head_boot), axis=1)
    boot_avg = np.full(n_boot, np.nan)
    boot_avg[complete] = head_boot[complete].mean(axis=1)
    se_avg = _bootstrap_se(boot_avg, label="did.cohort_anchored.headline")

    ci = (att_avg - z_crit * se_avg, att_avg + z_crit * se_avg)
    z = att_avg / se_avg if se_avg > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    _result = CausalResult(
        method="Cohort-Anchored Event Study (staggered-robust)",
        estimand="ATT (avg post)",
        estimate=att_avg,
        se=se_avg,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(df),
        model_info={
            "estimator": "cohort_anchored_event_study",
            "event_study": event_study_df,
            "n_cohorts": len(cohorts),
            "reference": "arXiv 2509.01829 (2025)",
        },
        _citation_key="cohort_anchored",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.cohort_anchored_event_study",
            params={
                "y": y,
                "treat": treat,
                "time": time,
                "id": id,
                "leads": leads,
                "lags": lags,
                "cluster": cluster,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


CausalResult._CITATIONS["cohort_anchored"] = (
    "@article{cohort_anchored2025,\n"
    "  title={Cohort-Anchored Robust Inference for Event-Study with "
    "Staggered Adoption},\n"
    "  author={Anonymous},\n"
    "  journal={arXiv preprint arXiv:2509.01829},\n"
    "  year={2025}\n"
    "}"
)
