"""
Sun & Abraham (2021) interaction-weighted event-study estimator.

Fits a *saturated* regression that interacts every cohort dummy with
every relative-time dummy, then aggregates the interaction coefficients
across cohorts using empirical cohort shares to deliver the IW
estimator δ̂^IW_ℓ that is robust to heterogeneous treatment effects
(Sun & Abraham 2021, Theorem 1 / Corollary 1).

Standard errors are computed from the classical OLS sandwich:

    Var(β̂) = (X'X)⁻¹  ( Σ_c  X_c' u_c u_c' X_c )  (X'X)⁻¹

clustered at the unit (or a user-supplied) level.  The IW estimator is
a linear combination of β̂, so Var(δ̂^IW_ℓ) = w_ℓ' Var(β̂) w_ℓ by the
delta method (shares treated as estimated but converging at parametric
rate, following SA 2021 eq. (18)).

References
----------
Sun, L. and Abraham, S. (2021).
    "Estimating Dynamic Treatment Effects in Event Studies with
     Heterogeneous Treatment Effects."
    *Journal of Econometrics*, 225(2), 175-199. [@sun2021estimating]
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

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
    Sun & Abraham (2021) interaction-weighted event-study estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    y : str
        Outcome variable.
    g : str
        Cohort variable: first treatment period (0 or inf = never treated).
    t : str
        Time period variable.
    i : str
        Unit identifier.
    event_window : tuple of (int, int), optional
        (min_relative_time, max_relative_time).
        Default: observed range in the data.
    control_group : str, default 'nevertreated'
        ``'nevertreated'`` or ``'lastcohort'``.  When ``'lastcohort'``,
        the latest treated cohort is used as the reference and dropped
        from the IW aggregation.
    covariates : list of str, optional
        Additional controls (time-varying; added linearly).
    cluster : str, optional
        Cluster variable for SEs. Default: clusters on ``i``.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``.detail`` is the event-study table (IW ATT by relative time
        with cluster-robust SE and 1−α CI).  ``.estimate`` / ``.se``
        are the simple post-treatment average and its delta-method SE.
    """
    df = data.copy()

    for col in [y, g, t, i]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
    if covariates:
        for c in covariates:
            if c not in df.columns:
                raise ValueError(f"Covariate '{c}' not found")
    if control_group not in ('nevertreated', 'lastcohort'):
        raise ValueError(
            f"control_group must be 'nevertreated' or 'lastcohort', "
            f"got {control_group!r}"
        )

    df[g] = df[g].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    time_periods = sorted(df[t].unique())
    t_max = max(time_periods)
    cohorts_all = sorted([v for v in df[g].unique() if v > 0 and v <= t_max])

    if not cohorts_all:
        raise ValueError("No treated cohorts found in the data.")

    # Reference cohort: never-treated (g=0) OR last cohort.
    if control_group == 'lastcohort':
        ref_cohort = max(cohorts_all)
        cohorts = [c for c in cohorts_all if c != ref_cohort]
        ref_mask = (df[g] == ref_cohort)
    else:
        cohorts = cohorts_all
        ref_mask = (df[g] == 0)

    if not cohorts:
        raise ValueError("No non-reference cohorts available for estimation.")
    has_ref = bool(ref_mask.any())
    if not has_ref:
        raise ValueError(
            f"Reference group is empty (control_group={control_group!r})."
        )

    cluster_col = cluster or i

    # Relative time (NaN for reference observations)
    df['_rel_time'] = np.where(df[g] > 0, df[t] - df[g], np.nan)

    if event_window is None:
        rel_obs = df.loc[df[g] > 0, '_rel_time'].dropna()
        e_min = int(rel_obs.min())
        e_max = int(rel_obs.max())
    else:
        e_min, e_max = int(event_window[0]), int(event_window[1])

    # Reference relative time = -1 (CS-SA standard).
    rel_times = [e for e in range(e_min, e_max + 1) if e != -1]

    # ----- Saturated design matrix: 1(G=g) × 1(e=ℓ) -----
    interact_meta: List[Tuple[int, int]] = []  # (g, e) per column
    X_cols: List[np.ndarray] = []
    for g_val in cohorts:
        in_cohort = (df[g] == g_val).values
        for e in rel_times:
            X_cols.append(
                (in_cohort & (df['_rel_time'].values == e)).astype(float)
            )
            interact_meta.append((g_val, e))

    X_int = np.column_stack(X_cols)

    # Unit + time FE via two-way within transformation ("within" projection).
    # Build the panel of y and X, demean, then flatten.
    unit_idx = pd.Categorical(df[i])
    time_idx = pd.Categorical(df[t])
    y_dm = _two_way_demean(df[y].values.astype(float), unit_idx, time_idx)
    X_dm = np.column_stack([
        _two_way_demean(X_int[:, k], unit_idx, time_idx)
        for k in range(X_int.shape[1])
    ])

    if covariates:
        for c in covariates:
            X_dm = np.column_stack([
                X_dm,
                _two_way_demean(df[c].values.astype(float), unit_idx, time_idx),
            ])

    valid = np.isfinite(y_dm) & np.all(np.isfinite(X_dm), axis=1)
    y_v = y_dm[valid]
    X_v = X_dm[valid]
    cluster_v = df.loc[valid, cluster_col].values
    k_int = len(interact_meta)

    # ----- OLS with ridge safety -----
    XtX = X_v.T @ X_v
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(X_v.shape[1]))
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X_v.T @ y_v)

    # ----- Cluster-robust sandwich SE (Liang-Zeger) -----
    u = y_v - X_v @ beta
    Xu = X_v * u[:, None]
    clusters = pd.Series(cluster_v)
    Xu_sum = np.zeros_like(XtX)
    for _, idx in clusters.groupby(clusters).indices.items():
        s = Xu[idx].sum(axis=0)
        Xu_sum += np.outer(s, s)
    n_clust = clusters.nunique()
    n, k = X_v.shape
    df_adj = (n_clust / max(n_clust - 1, 1)) * ((n - 1) / max(n - k, 1))
    V_beta = df_adj * XtX_inv @ Xu_sum @ XtX_inv

    # Slice to interaction block (drop covariate rows/cols for IW weights).
    V_int = V_beta[:k_int, :k_int]
    beta_int = beta[:k_int]

    # ----- IW aggregation at each relative time -----
    unit_cohorts = df.groupby(i)[g].first()
    cohort_counts = unit_cohorts[unit_cohorts > 0].value_counts()
    z_crit = stats.norm.ppf(1 - alpha / 2)

    es_rows = []
    for e in sorted(set(rel_times)):
        eligible = [
            g_val for g_val in cohorts
            if (g_val, e) in {m for m in interact_meta}
            and (g_val + e) in time_periods
        ]
        if not eligible:
            continue

        shares = np.array([cohort_counts.get(g_val, 0) for g_val in eligible],
                          dtype=float)
        if shares.sum() <= 0:
            continue
        shares = shares / shares.sum()

        # Selection vector w of length k_int picking out (g, e) positions.
        w = np.zeros(k_int)
        for share, g_val in zip(shares, eligible):
            idx = interact_meta.index((g_val, e))
            w[idx] = share

        est_e = float(w @ beta_int)
        se_e = float(np.sqrt(max(w @ V_int @ w, 0.0)))
        pval = (float(2 * (1 - stats.norm.cdf(abs(est_e / se_e))))
                if se_e > 0 else 1.0)

        es_rows.append({
            'relative_time': e,
            'att': est_e,
            'se': se_e,
            'ci_lower': est_e - z_crit * se_e,
            'ci_upper': est_e + z_crit * se_e,
            'pvalue': pval,
            'n_cohorts': len(eligible),
        })

    event_study = pd.DataFrame(es_rows)

    # ----- Overall post-treatment ATT via a single linear combination -----
    post = event_study[event_study['relative_time'] >= 0]
    if len(post) > 0:
        # Build block weight across post-event cells.
        W = np.zeros(k_int)
        w_total = 0.0
        for e in post['relative_time']:
            eligible = [
                g_val for g_val in cohorts
                if (g_val, e) in set(interact_meta)
                and (g_val + e) in time_periods
            ]
            if not eligible:
                continue
            shares = np.array(
                [cohort_counts.get(g_val, 0) for g_val in eligible],
                dtype=float,
            )
            if shares.sum() <= 0:
                continue
            shares = shares / shares.sum()
            for share, g_val in zip(shares, eligible):
                W[interact_meta.index((g_val, e))] += share
            w_total += 1.0
        if w_total > 0:
            W = W / w_total
        att = float(W @ beta_int)
        se_att = float(np.sqrt(max(W @ V_int @ W, 0.0)))
    else:
        att, se_att = 0.0, np.inf

    z = att / se_att if se_att > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci = (att - z_crit * se_att, att + z_crit * se_att)

    model_info = {
        'estimator': 'Sun-Abraham IW',
        'control_group': control_group,
        'event_window': (e_min, e_max),
        'n_cohorts': len(cohorts),
        'cohorts': cohorts,
        'event_study': event_study,
        'se_type': f'cluster-robust on {cluster_col}',
        'n_clusters': int(n_clust),
        'n_coeffs': int(k_int),
    }

    _result = CausalResult(
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
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.did.sun_abraham",
            params={
                "y": y, "g": g, "t": t, "i": i,
                "event_window": list(event_window) if event_window else None,
                "control_group": control_group,
                "covariates": list(covariates) if covariates else None,
                "cluster": cluster, "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# Helpers
# ======================================================================

def _two_way_demean(
    x: np.ndarray,
    unit_idx: pd.Categorical,
    time_idx: pd.Categorical,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> np.ndarray:
    """Iterative within-transformation for unbalanced two-way FE.

    Falls back to the identity transformation on a single-unit / single-period
    sample.  Converges in a handful of passes on well-behaved panels.
    """
    x = x.astype(float).copy()
    n_units = len(unit_idx.categories)
    n_times = len(time_idx.categories)
    if n_units <= 1 or n_times <= 1:
        return x - np.nanmean(x)

    u_codes = unit_idx.codes
    t_codes = time_idx.codes

    for _ in range(max_iter):
        u_mean = np.bincount(u_codes, weights=x, minlength=n_units) / \
                 np.bincount(u_codes, minlength=n_units).clip(min=1)
        x = x - u_mean[u_codes]
        t_mean = np.bincount(t_codes, weights=x, minlength=n_times) / \
                 np.bincount(t_codes, minlength=n_times).clip(min=1)
        x = x - t_mean[t_codes]
        if (np.nanmax(np.abs(u_mean)) < tol and
                np.nanmax(np.abs(t_mean)) < tol):
            break
    return x


# ----------------------------------------------------------------------
# Citation (redundant-safe registration)
# ----------------------------------------------------------------------
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
