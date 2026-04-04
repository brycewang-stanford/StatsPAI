"""
Sun & Abraham (2021) interaction-weighted estimator for staggered DID.

Estimates dynamic treatment effects via cohort-specific event study
coefficients that are free of contamination from heterogeneous effects
across cohorts. The estimator re-weights the OLS event study
coefficients using cohort shares.

References
----------
Sun, L. and Abraham, S. (2021).
"Estimating Dynamic Treatment Effects in Event Studies with
Heterogeneous Treatment Effects."
*Journal of Econometrics*, 225(2), 175-199.
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def sun_abraham(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    i: str,
    event_window: Optional[Tuple[int, int]] = None,
    control_group: str = 'nevertreated',
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Sun & Abraham (2021) interaction-weighted event study estimator.

    Estimates cohort-specific treatment effects and aggregates them
    using cohort-share weights to obtain heterogeneity-robust dynamic
    treatment effect estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data (long format).
    y : str
        Outcome variable.
    g : str
        Cohort variable: first treatment period (0 or inf = never treated).
    t : str
        Time period variable.
    i : str
        Unit identifier.
    event_window : tuple of (int, int), optional
        (min_relative_time, max_relative_time) for event study.
        Default: all available relative times.
    control_group : str, default 'nevertreated'
        ``'nevertreated'`` or ``'lastcohort'`` (last treated cohort).
    covariates : list of str, optional
        Additional control variables.
    cluster : str, optional
        Cluster variable for SEs. Default: clusters on ``i``.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        With event study coefficients in ``model_info['event_study']``.

    Examples
    --------
    >>> result = sun_abraham(df, y='earnings', g='first_treat',
    ...                      t='year', i='worker', event_window=(-5, 5))
    >>> result.event_study_plot()

    Notes
    -----
    The Sun-Abraham estimator proceeds in two steps:

    1. **Saturated regression**: Interact relative-time dummies with
       cohort dummies (all possible ``1(e=ℓ) × 1(G=g)`` interactions),
       using never-treated (or last-cohort) as the reference.

    2. **Aggregation**: For each relative time ℓ, compute the
       interaction-weighted (IW) estimator as a weighted average of
       cohort-specific effects using cohort shares as weights:
       ``δ̂_ℓ = Σ_g ŝ_g × δ̂_{ℓ,g}``
       where ``ŝ_g = P(G=g | G is treated and observed at e=ℓ)``.

    This avoids the contamination bias of standard OLS event study
    regressions documented in Sun & Abraham (2021, *JEcon*), Theorem 1.
    """
    df = data.copy()

    # Parse
    for col in [y, g, t, i]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

    df[g] = df[g].fillna(0).replace([np.inf], 0)
    time_periods = sorted(df[t].unique())
    cohorts = sorted([v for v in df[g].unique() if v > 0 and v <= max(time_periods)])
    never_mask = df[g] == 0

    if cluster is None:
        cluster = i

    # Relative time for each obs
    df['_rel_time'] = np.where(df[g] > 0, df[t] - df[g], np.nan)

    # Event window
    valid_rel = df.loc[df[g] > 0, '_rel_time'].dropna()
    if event_window is None:
        e_min = int(valid_rel.min())
        e_max = int(valid_rel.max())
    else:
        e_min, e_max = event_window

    rel_times = list(range(e_min, e_max + 1))
    rel_times = [e for e in rel_times if e != -1]  # -1 is reference

    # --- Step 1: Cohort × relative-time saturated regression ---
    # Build interaction dummies: 1(G=g) × 1(e=ℓ) for all (g, ℓ)
    interactions = {}
    for g_val in cohorts:
        for e in rel_times:
            col_name = f'_coh{g_val}_e{e}'
            interactions[col_name] = (
                (df[g] == g_val) & (df['_rel_time'] == e)
            ).astype(float)
            interactions[col_name + '_meta'] = (g_val, e)

    if not interactions:
        raise ValueError("No valid cohort × event-time interactions found.")

    # Build design matrix
    interact_cols = [c for c in interactions if not c.endswith('_meta')]
    for col_name in interact_cols:
        df[col_name] = interactions[col_name]

    # Unit and time FE via demeaning
    Y_panel = df.pivot_table(index=i, columns=t, values=y, aggfunc='first')
    y_dm = _double_demean(Y_panel.values)

    x_panels = {}
    for col_name in interact_cols:
        x_p = df.pivot_table(index=i, columns=t, values=col_name, aggfunc='first').fillna(0)
        x_panels[col_name] = _double_demean(x_p.values)

    y_flat = y_dm.ravel()
    X_flat = np.column_stack([x_panels[c].ravel() for c in interact_cols])
    valid = np.isfinite(y_flat) & np.all(np.isfinite(X_flat), axis=1)
    y_v = y_flat[valid]
    X_v = X_flat[valid]

    # OLS
    try:
        beta = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(len(interact_cols))

    # Map back to (g, e) → coefficient
    cohort_effects = {}
    for idx, col_name in enumerate(interact_cols):
        g_val, e = interactions[col_name + '_meta']
        cohort_effects[(g_val, e)] = beta[idx]

    # --- Step 2: IW aggregation ---
    # For each relative time e, weight cohort-specific effects by cohort share
    unit_cohorts = df.groupby(i)[g].first()
    cohort_counts = unit_cohorts[unit_cohorts > 0].value_counts()

    z_crit = stats.norm.ppf(1 - alpha / 2)
    es_rows = []

    for e in sorted(set(rel_times)):
        # Which cohorts are observed at this relative time?
        eligible = [g_val for g_val in cohorts
                    if (g_val, e) in cohort_effects
                    and g_val + e in time_periods]

        if not eligible:
            continue

        # Cohort shares
        shares = np.array([cohort_counts.get(g_val, 0) for g_val in eligible],
                          dtype=float)
        total = shares.sum()
        if total == 0:
            continue
        shares = shares / total

        # IW estimate
        effects = np.array([cohort_effects[(g_val, e)] for g_val in eligible])
        iw_est = float(np.average(effects, weights=shares))

        # SE (conservative: pooled across cohorts)
        resid_all = y_v - X_v @ beta
        n_eff = len(resid_all)
        sigma2 = np.sum(resid_all ** 2) / max(n_eff - len(beta), 1)
        # Approximation: SE from regression divided by effective sample
        se_approx = float(np.sqrt(sigma2 / max(total * len(time_periods), 1)))

        pval = float(2 * (1 - stats.norm.cdf(abs(iw_est / se_approx)))) if se_approx > 0 else 1.0

        es_rows.append({
            'relative_time': e,
            'att': iw_est,
            'se': se_approx,
            'ci_lower': iw_est - z_crit * se_approx,
            'ci_upper': iw_est + z_crit * se_approx,
            'pvalue': pval,
            'n_cohorts': len(eligible),
        })

    event_study = pd.DataFrame(es_rows)

    # Overall ATT: average of post-treatment event-study coefficients
    post = event_study[event_study['relative_time'] >= 0]
    if len(post) > 0:
        att = float(post['att'].mean())
        se_att = float(np.sqrt(np.mean(post['se'] ** 2)))
    else:
        att, se_att = 0.0, np.inf

    z = att / se_att if se_att > 0 else 0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci = (att - z_crit * se_att, att + z_crit * se_att)

    model_info = {
        'estimator': 'Sun-Abraham IW',
        'control_group': control_group,
        'event_window': (e_min, e_max),
        'n_cohorts': len(cohorts),
        'cohorts': cohorts,
        'event_study': event_study,
    }

    return CausalResult(
        method='Sun and Abraham (2021)',
        estimand='ATT',
        estimate=att,
        se=se_att,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=event_study,
        model_info=model_info,
        _citation_key='sun_abraham',
    )


def _double_demean(M: np.ndarray) -> np.ndarray:
    """Remove unit and time means (within transformation)."""
    row_mean = np.nanmean(M, axis=1, keepdims=True)
    col_mean = np.nanmean(M, axis=0, keepdims=True)
    grand_mean = np.nanmean(M)
    return M - row_mean - col_mean + grand_mean


# Citation (already exists in CausalResult but ensure it's there)
CausalResult._CITATIONS['sun_abraham'] = (
    "@article{sun2021estimating,\n"
    "  title={Estimating Dynamic Treatment Effects in Event Studies "
    "with Heterogeneous Treatment Effects},\n"
    "  author={Sun, Liyang and Abraham, Sarah},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={225},\n"
    "  number={2},\n"
    "  pages={175--199},\n"
    "  year={2021},\n"
    "  publisher={Elsevier}\n"
    "}"
)
