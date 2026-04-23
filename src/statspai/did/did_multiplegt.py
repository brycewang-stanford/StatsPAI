"""
de Chaisemartin & D'Haultfoeuille (2020) DID estimator.

Estimates the effect of a binary treatment that can switch on AND off,
robust to heterogeneous treatment effects across groups and time periods.
Unlike Callaway & Sant'Anna (which assumes staggered adoption / no
treatment reversal), this estimator handles general treatment paths
where units can enter and exit treatment.

The DID_M estimator computes a weighted average of group-time DID
estimates, where "switchers" (units whose treatment changes between
consecutive periods) are compared to "stayers" (units whose treatment
remains constant).

References
----------
de Chaisemartin, C. and D'Haultfoeuille, X. (2020).
"Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects."
*American Economic Review*, 110(9), 2964-2996. [@dechaisemartin2020fixed]

de Chaisemartin, C. and D'Haultfoeuille, X. (2022).
"Two-way fixed effects and differences-in-differences with heterogeneous
treatment effects: A survey."  *The Econometrics Journal*, 26(3), C1-C30. [@dechaisemartin2022fixed]

de Chaisemartin, C. and D'Haultfoeuille, X. (2024).
"Difference-in-Differences Estimators of Intertemporal Treatment Effects."
*Review of Economics and Statistics*, forthcoming.  (Joint placebo test
and average cumulative effect, Section 3.) [@dechaisemartin2024difference]
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def did_multiplegt(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]] = None,
    placebo: int = 0,
    dynamic: int = 0,
    cluster: Optional[str] = None,
    n_boot: int = 100,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    de Chaisemartin & D'Haultfoeuille (2020) DID estimator.

    Estimates the effect of a binary treatment that can switch on AND off,
    robust to heterogeneous treatment effects across groups and time.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format (one row per unit-period).
    y : str
        Outcome variable name.
    group : str
        Group (unit) identifier.
    time : str
        Time period variable.
    treatment : str
        Binary treatment indicator (0/1). Unlike Callaway-Sant'Anna, this
        is the *current* treatment status, not the first-treatment period.
        Units may switch treatment on and off.
    controls : list of str, optional
        Control variables. When provided, first-differences of the outcome
        are residualized on first-differences of the controls.
    placebo : int, default 0
        Number of placebo (pre-treatment) tests.  Placebo *l* checks
        whether switchers at *t* already differed from stayers between
        *t-l-1* and *t-l*.
    dynamic : int, default 0
        Number of dynamic (post-treatment) effect horizons to estimate.
        Dynamic effect *l* measures the long difference Y_{t+l} - Y_{t-1}
        for switchers at *t* relative to stayers.
    cluster : str, optional
        Variable for cluster bootstrap.  Defaults to ``group``.
    n_boot : int, default 100
        Number of bootstrap replications for standard errors.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Result object with ``.summary()`` and ``.plot()`` methods.
        The ``detail`` DataFrame contains all (group, time)-level DID
        estimates; ``model_info`` stores event-study coefficients
        (placebo and dynamic effects) for plotting.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.did_multiplegt(
    ...     data=df, y="wage", group="county", time="year",
    ...     treatment="treated", placebo=2, dynamic=3,
    ...     cluster="state", n_boot=200, seed=42,
    ... )
    >>> result.summary()
    >>> result.plot()

    Notes
    -----
    The estimator proceeds as follows for each consecutive pair of
    periods (t-1, t):

    1. **Switchers**: units whose treatment status changed between t-1
       and t.
    2. **Stayers**: units whose treatment status did *not* change.
    3. DID_{g,t} = mean(DeltaY_switchers) - mean(DeltaY_stayers),
       where DeltaY = Y_t - Y_{t-1}.
    4. DID_M = weighted average of DID_{g,t} with weights proportional
       to the number of switchers in each cell.

    Standard errors are computed via cluster bootstrap (resampling
    clusters with replacement).
    """
    # ── Validate inputs ──────────────────────────────────────────── #
    df = data.copy()
    _required = [y, group, time, treatment]
    for col in _required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if controls is not None:
        for col in controls:
            if col not in df.columns:
                raise ValueError(f"Control column '{col}' not found in data.")

    cluster_var = cluster if cluster is not None else group

    # ── Sort and compute main estimate ───────────────────────────── #
    df = df.sort_values([group, time]).reset_index(drop=True)

    main = _estimate_did_m(df, y, group, time, treatment, controls)

    # ── Placebo effects ──────────────────────────────────────────── #
    placebo_results = []
    for l in range(1, placebo + 1):
        plac = _estimate_placebo(df, y, group, time, treatment, controls, lag=l)
        placebo_results.append(plac)

    # ── Dynamic effects ──────────────────────────────────────────── #
    dynamic_results = []
    for l in range(0, dynamic + 1):
        dyn = _estimate_dynamic(df, y, group, time, treatment, controls, horizon=l)
        dynamic_results.append(dyn)

    # ── Bootstrap standard errors ────────────────────────────────── #
    rng = np.random.default_rng(seed)
    clusters = df[cluster_var].unique()
    n_clusters = len(clusters)

    boot_main = np.full(n_boot, np.nan)
    boot_placebo = np.full((n_boot, max(placebo, 0)), np.nan) if placebo > 0 else None
    boot_dynamic = np.full((n_boot, dynamic + 1), np.nan) if dynamic >= 0 else None

    for b in range(n_boot):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        # Build bootstrap sample preserving panel structure
        frames = []
        for j, c in enumerate(sampled):
            chunk = df[df[cluster_var] == c].copy()
            # Relabel group to avoid collisions from sampling with replacement
            chunk[group] = chunk[group].astype(str) + f"_b{j}"
            frames.append(chunk)
        bdf = pd.concat(frames, ignore_index=True)

        b_main = _estimate_did_m(bdf, y, group, time, treatment, controls)
        boot_main[b] = b_main['did_m']

        if placebo > 0:
            for l in range(1, placebo + 1):
                plac = _estimate_placebo(bdf, y, group, time, treatment, controls, lag=l)
                boot_placebo[b, l - 1] = plac['estimate']

        if dynamic >= 0:
            for l in range(0, dynamic + 1):
                dyn = _estimate_dynamic(bdf, y, group, time, treatment, controls, horizon=l)
                boot_dynamic[b, l] = dyn['estimate']

    # ── Compute SEs, p-values, CIs ───────────────────────────────── #
    se_main = np.nanstd(boot_main, ddof=1)
    z_main = main['did_m'] / se_main if se_main > 0 else 0.0
    p_main = 2 * (1 - stats.norm.cdf(abs(z_main)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_main = (main['did_m'] - z_crit * se_main,
               main['did_m'] + z_crit * se_main)

    # Placebo SEs
    placebo_out = []
    for l in range(placebo):
        est = placebo_results[l]['estimate']
        se = np.nanstd(boot_placebo[:, l], ddof=1) if boot_placebo is not None else 0.0
        z = est / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        ci = (est - z_crit * se, est + z_crit * se)
        placebo_out.append({
            'lag': -(l + 1), 'estimate': est, 'se': se,
            'pvalue': p, 'ci_lower': ci[0], 'ci_upper': ci[1],
        })

    # Dynamic SEs
    dynamic_out = []
    for l in range(dynamic + 1):
        est = dynamic_results[l]['estimate']
        se = np.nanstd(boot_dynamic[:, l], ddof=1) if boot_dynamic is not None else 0.0
        z = est / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        ci = (est - z_crit * se, est + z_crit * se)
        dynamic_out.append({
            'horizon': l, 'estimate': est, 'se': se,
            'pvalue': p, 'ci_lower': ci[0], 'ci_upper': ci[1],
        })

    # ── Build detail DataFrame ───────────────────────────────────── #
    detail_rows = main.get('cell_estimates', [])
    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    # ── Event-study DataFrame for plotting ───────────────────────── #
    # CausalResult.summary() and .plot() expect columns:
    #   relative_time, att, se, pvalue, ci_lower, ci_upper
    es_rows = []
    for p_row in placebo_out:
        es_rows.append({
            'relative_time': p_row['lag'],
            'att': p_row['estimate'],
            'se': p_row['se'],
            'pvalue': p_row['pvalue'],
            'ci_lower': p_row['ci_lower'],
            'ci_upper': p_row['ci_upper'],
            'type': 'placebo',
        })
    for d_row in dynamic_out:
        es_rows.append({
            'relative_time': d_row['horizon'],
            'att': d_row['estimate'],
            'se': d_row['se'],
            'pvalue': d_row['pvalue'],
            'ci_lower': d_row['ci_lower'],
            'ci_upper': d_row['ci_upper'],
            'type': 'dynamic',
        })
    es_df = pd.DataFrame(es_rows) if es_rows else pd.DataFrame()

    # ── Joint placebo test + avg cumulative dynamic effect ───────── #
    joint_placebo = _joint_placebo_test(placebo_results, boot_placebo)
    avg_cumulative = _avg_cumulative_effect(
        dynamic_results, boot_dynamic, alpha,
    )

    # ── Assemble model_info ──────────────────────────────────────── #
    model_info: Dict[str, Any] = {
        'estimator': 'de Chaisemartin-D\'Haultfoeuille (2020)',
        'n_switching_cells': main['n_switching_cells'],
        'n_switchers': main['n_switchers'],
        'n_boot': n_boot,
        'seed': seed,
        'cluster_var': cluster_var,
        'controls': controls,
        'placebo': placebo_out,
        'dynamic': dynamic_out,
        'event_study': es_df,
        'joint_placebo_test': joint_placebo,
        'avg_cumulative_effect': avg_cumulative,
    }

    return CausalResult(
        method="de Chaisemartin-D'Haultfoeuille (2020)",
        estimand='ATT',
        estimate=float(main['did_m']),
        se=float(se_main),
        pvalue=float(p_main),
        ci=ci_main,
        alpha=alpha,
        n_obs=len(data),
        detail=detail_df,
        model_info=model_info,
        _citation_key='did_multiplegt',
    )


# ======================================================================
# Internal helpers
# ======================================================================

def _residualize(dy: np.ndarray, dX: np.ndarray) -> np.ndarray:
    """Residualize dy on dX via OLS."""
    if dX.shape[1] == 0:
        return dy
    # Add intercept
    ones = np.ones((dX.shape[0], 1))
    X = np.hstack([ones, dX])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        return dy - X @ beta
    except np.linalg.LinAlgError:
        return dy


def _estimate_did_m(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Core DID_M estimator: weighted average of cell-level DID estimates.

    For each period t, identify switchers and stayers, compute DID_gt.
    """
    periods = sorted(df[time].unique())
    cell_estimates = []
    total_switchers = 0

    for idx in range(1, len(periods)):
        t_prev, t_curr = periods[idx - 1], periods[idx]

        df_prev = df[df[time] == t_prev][[group, y, treatment] + (controls or [])].copy()
        df_curr = df[df[time] == t_curr][[group, y, treatment] + (controls or [])].copy()

        df_prev = df_prev.rename(columns={y: f'{y}_prev', treatment: f'{treatment}_prev'})
        if controls:
            df_prev = df_prev.rename(columns={c: f'{c}_prev' for c in controls})

        merged = df_curr.merge(df_prev, on=group, how='inner')
        if merged.empty:
            continue

        merged['_switched'] = merged[treatment] != merged[f'{treatment}_prev']
        merged['_dy'] = merged[y] - merged[f'{y}_prev']

        switchers = merged[merged['_switched']]
        stayers = merged[~merged['_switched']]

        n_switch = len(switchers)
        if n_switch == 0 or len(stayers) == 0:
            continue

        dy_switch = switchers['_dy'].values.copy()
        dy_stay = stayers['_dy'].values.copy()

        # Residualize on controls if provided
        if controls:
            dX_switch = np.column_stack([
                switchers[c].values - switchers[f'{c}_prev'].values
                for c in controls
            ])
            dX_stay = np.column_stack([
                stayers[c].values - stayers[f'{c}_prev'].values
                for c in controls
            ])
            # Pool for consistent residualization
            dX_all = np.vstack([dX_switch, dX_stay])
            dy_all = np.concatenate([dy_switch, dy_stay])
            resid = _residualize(dy_all, dX_all)
            dy_switch = resid[:n_switch]
            dy_stay = resid[n_switch:]

        # Determine sign: if switchers turned ON → positive effect expected
        # If switchers turned OFF → flip sign so estimate is effect of treatment
        turned_on = (switchers[treatment].values == 1).mean() > 0.5
        sign = 1.0 if turned_on else -1.0

        did_gt = sign * (np.mean(dy_switch) - np.mean(dy_stay))

        cell_estimates.append({
            'time': t_curr,
            'time_prev': t_prev,
            'n_switchers': n_switch,
            'n_stayers': len(stayers),
            'did_gt': did_gt,
            'mean_dy_switch': float(np.mean(dy_switch)),
            'mean_dy_stay': float(np.mean(dy_stay)),
            'direction': 'on' if turned_on else 'off',
        })
        total_switchers += n_switch

    if total_switchers == 0:
        return {
            'did_m': 0.0,
            'n_switching_cells': 0,
            'n_switchers': 0,
            'cell_estimates': [],
        }

    # Weighted average
    did_m = sum(
        ce['did_gt'] * ce['n_switchers'] / total_switchers
        for ce in cell_estimates
    )

    return {
        'did_m': float(did_m),
        'n_switching_cells': len(cell_estimates),
        'n_switchers': total_switchers,
        'cell_estimates': cell_estimates,
    }


def _estimate_placebo(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    lag: int,
) -> Dict[str, Any]:
    """
    Placebo test at lag *l*.

    For switchers at time t, check outcome changes between t-l-1 and t-l
    (pre-treatment periods) relative to stayers.
    """
    periods = sorted(df[time].unique())
    period_idx = {p: i for i, p in enumerate(periods)}

    estimates = []
    weights = []

    for idx in range(1, len(periods)):
        t_prev, t_curr = periods[idx - 1], periods[idx]

        df_prev = df[df[time] == t_prev][[group, treatment]].copy()
        df_curr = df[df[time] == t_curr][[group, treatment]].copy()
        df_prev_ren = df_prev.rename(columns={treatment: f'{treatment}_prev'})
        merged_t = df_curr.merge(df_prev_ren, on=group, how='inner')
        if merged_t.empty:
            continue

        merged_t['_switched'] = merged_t[treatment] != merged_t[f'{treatment}_prev']
        switch_groups = set(merged_t[merged_t['_switched']][group].unique())
        stay_groups = set(merged_t[~merged_t['_switched']][group].unique())

        if not switch_groups or not stay_groups:
            continue

        # Look back to periods t-lag-1 and t-lag
        curr_idx = period_idx[t_curr]
        plac_end_idx = curr_idx - lag
        plac_start_idx = plac_end_idx - 1

        if plac_start_idx < 0 or plac_end_idx < 0 or plac_end_idx >= len(periods):
            continue

        p_start = periods[plac_start_idx]
        p_end = periods[plac_end_idx]

        df_p_start = df[df[time] == p_start][[group, y]].copy()
        df_p_end = df[df[time] == p_end][[group, y]].copy()

        df_p_start = df_p_start.rename(columns={y: f'{y}_start'})
        merged_p = df_p_end.merge(df_p_start, on=group, how='inner')
        if merged_p.empty:
            continue

        merged_p['_dy'] = merged_p[y] - merged_p[f'{y}_start']

        dy_switch = merged_p[merged_p[group].isin(switch_groups)]['_dy'].values
        dy_stay = merged_p[merged_p[group].isin(stay_groups)]['_dy'].values

        if len(dy_switch) == 0 or len(dy_stay) == 0:
            continue

        est = np.mean(dy_switch) - np.mean(dy_stay)
        n_sw = len(dy_switch)
        estimates.append(est)
        weights.append(n_sw)

    if not estimates:
        return {'estimate': 0.0, 'n_cells': 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {'estimate': float(weighted_est), 'n_cells': len(estimates)}


def _estimate_dynamic(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    horizon: int,
) -> Dict[str, Any]:
    """
    Dynamic effect at horizon *l*.

    For switchers at time t, compute Y_{t+l} - Y_{t-1} relative to
    stayers' same long difference.
    """
    periods = sorted(df[time].unique())
    period_idx = {p: i for i, p in enumerate(periods)}

    estimates = []
    weights = []

    for idx in range(1, len(periods)):
        t_prev, t_curr = periods[idx - 1], periods[idx]

        df_prev = df[df[time] == t_prev][[group, y, treatment]].copy()
        df_curr = df[df[time] == t_curr][[group, treatment]].copy()
        df_prev_ren = df_prev.rename(columns={
            treatment: f'{treatment}_prev', y: f'{y}_base',
        })
        merged_t = df_curr.merge(df_prev_ren, on=group, how='inner')
        if merged_t.empty:
            continue

        merged_t['_switched'] = merged_t[treatment] != merged_t[f'{treatment}_prev']
        switch_groups = set(merged_t[merged_t['_switched']][group].unique())
        stay_groups = set(merged_t[~merged_t['_switched']][group].unique())

        if not switch_groups or not stay_groups:
            continue

        # Outcome at t + horizon
        curr_idx = period_idx[t_curr]
        future_idx = curr_idx + horizon

        if future_idx >= len(periods):
            continue

        t_future = periods[future_idx]
        df_future = df[df[time] == t_future][[group, y]].copy()
        df_future = df_future.rename(columns={y: f'{y}_future'})

        # Base outcome at t-1
        df_base = df[df[time] == t_prev][[group, y]].copy()
        df_base = df_base.rename(columns={y: f'{y}_base2'})

        merged_ld = df_future.merge(df_base, on=group, how='inner')
        if merged_ld.empty:
            continue

        merged_ld['_ldy'] = merged_ld[f'{y}_future'] - merged_ld[f'{y}_base2']

        ldy_switch = merged_ld[merged_ld[group].isin(switch_groups)]['_ldy'].values
        ldy_stay = merged_ld[merged_ld[group].isin(stay_groups)]['_ldy'].values

        if len(ldy_switch) == 0 or len(ldy_stay) == 0:
            continue

        # Determine sign from switching direction
        turned_on_mask = merged_t[merged_t['_switched']][treatment].values == 1
        turned_on = turned_on_mask.mean() > 0.5
        sign = 1.0 if turned_on else -1.0

        est = sign * (np.mean(ldy_switch) - np.mean(ldy_stay))
        n_sw = len(ldy_switch)
        estimates.append(est)
        weights.append(n_sw)

    if not estimates:
        return {'estimate': 0.0, 'n_cells': 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {'estimate': float(weighted_est), 'n_cells': len(estimates)}


# ======================================================================
# Joint inference
# ======================================================================

def _joint_placebo_test(
    placebo_results: List[Dict[str, Any]],
    boot_placebo: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Joint Wald test of H0: all placebo lags equal zero.

    Uses the bootstrap draws (cluster-level) to estimate the covariance
    across lags, so the test accounts for correlation between consecutive
    pre-periods.  Follows dCDH (2024, §3.3) "joint placebo" recommendation.
    """
    if not placebo_results or boot_placebo is None:
        return None

    est = np.array([r['estimate'] for r in placebo_results], dtype=float)
    boot = np.asarray(boot_placebo, dtype=float)  # (n_boot, k)
    # Drop all-NaN columns / rows before covariance.
    mask_cols = ~np.all(np.isnan(boot), axis=0)
    if not mask_cols.any():
        return None
    est = est[mask_cols]
    boot = boot[:, mask_cols]
    valid_rows = ~np.any(np.isnan(boot), axis=1)
    if valid_rows.sum() < boot.shape[1] + 1:
        return None
    boot = boot[valid_rows]

    cov = np.cov(boot, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    k = est.shape[0]
    cov_reg = cov + np.eye(k) * 1e-10

    try:
        W = float(est @ np.linalg.solve(cov_reg, est))
    except np.linalg.LinAlgError:
        W = float(est @ np.linalg.pinv(cov_reg) @ est)

    pval = float(1 - stats.chi2.cdf(W, k))
    return {'statistic': W, 'df': int(k), 'pvalue': pval}


def _avg_cumulative_effect(
    dynamic_results: List[Dict[str, Any]],
    boot_dynamic: Optional[np.ndarray],
    alpha: float,
) -> Optional[Dict[str, Any]]:
    """Average cumulative dynamic effect: mean of dynamic[0..L].

    Bootstrap SE is computed over the per-draw average across horizons,
    preserving the cross-horizon covariance (dCDH 2024 §3.4).
    """
    if not dynamic_results or boot_dynamic is None:
        return None
    est_vec = np.array([r['estimate'] for r in dynamic_results], dtype=float)
    avg_est = float(np.mean(est_vec))

    boot = np.asarray(boot_dynamic, dtype=float)
    per_draw_avg = np.nanmean(boot, axis=1)
    per_draw_avg = per_draw_avg[np.isfinite(per_draw_avg)]
    if per_draw_avg.size < 2:
        return {
            'estimate': avg_est, 'se': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan,
            'pvalue': np.nan,
            'n_horizons': len(est_vec),
        }
    se = float(np.std(per_draw_avg, ddof=1))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = avg_est / se if se > 0 else 0.0
    return {
        'estimate': avg_est,
        'se': se,
        'ci_lower': avg_est - z_crit * se,
        'ci_upper': avg_est + z_crit * se,
        'pvalue': float(2 * (1 - stats.norm.cdf(abs(z)))),
        'n_horizons': int(len(est_vec)),
    }
