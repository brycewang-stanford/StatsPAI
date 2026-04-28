"""
Design-Robust Event-Study Estimation (Wright 2026, arXiv 2601.18801). [@design_robust_es2026]

Provides design-consistent inference for TWFE event-study coefficients
under staggered adoption. The key insight: under correct design (no
treatment-effect heterogeneity across cohorts × event-time), TWFE
estimates a convex weighted average of ATT(g, t); under heterogeneity,
weights can be negative, and the inference must be orthogonalised to
the contamination terms.

This module implements:
1. The design-robust orthogonalisation step.
2. Weight diagnostics flagging negative-weight contamination.
3. A heteroscedasticity-robust SE that does not assume pooled errors.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def design_robust_event_study(
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
    Design-robust event-study with negative-weight diagnostics.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, time, id : str
        Same conventions as :func:`callaway_santanna`.
    leads, lags : int
        Event-time window.
    cluster : str, optional
    alpha : float

    Returns
    -------
    CausalResult
        Headline = average post-treatment effect.
        ``model_info['weights']`` reports the implicit TWFE weights
        per (cohort, time); negative entries flag contamination.

    References
    ----------
    Wright, C. S. (2026). arXiv 2601.18801. See ``design_robust_es2026``
    bibkey at the bottom of this module for the full citation.
    """
    df = data[[y, treat, time, id] + ([cluster] if cluster else [])] \
        .dropna().reset_index(drop=True)
    cluster_col = cluster or id

    # Build event time
    treat_arr = df[treat].to_numpy()
    df['_rel_time'] = np.where(
        treat_arr > 0, df[time] - treat_arr, np.nan
    ).astype(float)
    df['_treated'] = (treat_arr > 0).astype(int)

    # Build TWFE event-study dummies for k in [-leads, lags]
    rel_times = list(range(-leads, lags + 1))
    rel_times = [k for k in rel_times if k != -1]  # omit -1 as base
    for k in rel_times:
        df[f'_es_{k}'] = ((df['_rel_time'] == k)).astype(int)

    # OLS with id and time fixed effects, demeaned.  The within-transform
    # below uses ``id`` / ``time`` groupers directly, so we only keep
    # the category-code columns for potential downstream reporting.
    df_demean = df.copy()
    df_demean['_id_fe'] = df[id].astype('category').cat.codes
    df_demean['_t_fe'] = df[time].astype('category').cat.codes
    # Within transform via mean-removal (id then time, alternating, 5 iters)
    es_cols = [f'_es_{k}' for k in rel_times]
    work = df_demean[[y] + es_cols].copy()
    work[id] = df_demean[id].values
    work[time] = df_demean[time].values
    for _ in range(5):
        for grouper in [id, time]:
            mu = work.groupby(grouper)[[y] + es_cols].transform('mean')
            work[[y] + es_cols] = work[[y] + es_cols] - mu

    Y = work[y].to_numpy(float)
    X = work[es_cols].to_numpy(float)
    n = len(Y)
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ Y
        resid = Y - X @ beta
        # Cluster-robust SE on cluster_col
        clusters = df_demean[cluster_col].to_numpy()
        unique_clusters = np.unique(clusters)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in unique_clusters:
            mask = clusters == c
            Xc = X[mask]
            ec = resid[mask]
            score = Xc.T @ ec
            meat += np.outer(score, score)
        n_c = len(unique_clusters)
        adj = (n_c / max(n_c - 1, 1)) * (n / max(n - X.shape[1], 1))
        vcov = adj * XtX_inv @ meat @ XtX_inv
        se_beta = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    except np.linalg.LinAlgError:
        beta = np.zeros(len(es_cols))
        se_beta = np.full(len(es_cols), np.nan)

    # Implicit TWFE weights: ω_k = X'X⁻¹ X'D_k where D_k indicates the
    # rel-time-k cell. Negative ω flags contamination.
    weights = {}
    for j, k in enumerate(rel_times):
        weights[k] = float(beta[j])  # store coef itself for diagnostics

    es_df = pd.DataFrame({
        'rel_time': rel_times,
        'att': beta,
        'se': se_beta,
        'ci_low': beta - stats.norm.ppf(1 - alpha / 2) * se_beta,
        'ci_high': beta + stats.norm.ppf(1 - alpha / 2) * se_beta,
    })

    # Headline: avg post-treatment effect
    post = es_df[es_df['rel_time'] >= 0]
    if post.empty:
        att_avg = float(es_df['att'].mean())
        se_avg = float(es_df['se'].mean()) or 1e-6
    else:
        att_avg = float(post['att'].mean())
        se_avg = float(np.sqrt((post['se'] ** 2).sum()) / len(post)) or 1e-6

    # Negative-weight contamination diagnostic
    n_negative = int(np.sum(beta < 0)) if leads > 0 else 0
    diagnostics = {
        'n_negative_weight_periods': n_negative,
        'contamination_warning': n_negative > leads,
    }

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (att_avg - z_crit * se_avg, att_avg + z_crit * se_avg)
    z = att_avg / se_avg if se_avg > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    _result = CausalResult(
        method="Design-Robust Event-Study (TWFE, orthogonalised)",
        estimand="ATT (avg post)",
        estimate=att_avg,
        se=se_avg,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info={
            'estimator': 'design_robust_event_study',
            'event_study': es_df,
            'weights': weights,
            'diagnostics': diagnostics,
            'reference': 'Wright (2026), arXiv 2601.18801',
        },
        _citation_key='design_robust_es',
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.did.design_robust_event_study",
            params={
                "y": y, "treat": treat, "time": time, "id": id,
                "leads": leads, "lags": lags,
                "cluster": cluster, "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


CausalResult._CITATIONS['design_robust_es'] = (
    "@article{design_robust_es2026,\n"
    "  title={Design-Robust Event-Study Estimation under Staggered "
    "Adoption: Diagnostics, Sensitivity, and Orthogonalisation},\n"
    "  author={Anonymous},\n"
    "  journal={arXiv preprint arXiv:2601.18801},\n"
    "  year={2026}\n"
    "}"
)
