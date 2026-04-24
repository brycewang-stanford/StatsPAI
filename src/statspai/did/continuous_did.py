"""
Difference-in-Differences with Continuous Treatment (heuristic modes).

Provides three heuristic DiD estimators for settings where treatment
intensity varies continuously across units:

- ``method='twfe'`` — TWFE OLS with a ``dose × post`` interaction.
- ``method='att_gt'`` — dose-quantile binned 2×2 DID versus the ``dose=0``
  arm (or the lowest-dose quantile as fallback), with bootstrap SE.
- ``method='dose_response'`` — local-linear regression of
  ``ΔY = Y_post − Y_pre`` on baseline dose, reporting the average
  derivative.

**Not** the Callaway, Goodman-Bacon & Sant'Anna (2024) ATT(d|g,t) /
ACRT(d|g,t) estimator with strong parallel trends and influence-function
variance — a paper-faithful ``method='cgs'`` is tracked in
``docs/rfc/continuous_did_cgs.md`` and will land in a later sprint. The
current ``method='att_gt'`` is a dose-bin heuristic, not the CGS 2024
group-time estimand.

References
----------
Callaway, B., Goodman-Bacon, A. & Sant'Anna, P.H.C. (2024).
"Difference-in-Differences with a Continuous Treatment."
[@callaway2024difference] — the target of the forthcoming ``method='cgs'``.

de Chaisemartin, C. & D'Haultfœuille, X. (2018).
"Fuzzy Differences-in-Differences." [@dechaisemartin2018fuzzy].
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from ..core.results import CausalResult


def continuous_did(
    data: pd.DataFrame,
    y: str,
    dose: str,
    time: str,
    id: str,
    post: str = None,
    t_pre: int = None,
    t_post: int = None,
    method: str = "att_gt",
    n_quantiles: int = 5,
    controls: List[str] = None,
    cluster: str = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = None,
) -> CausalResult:
    """
    Difference-in-Differences with continuous treatment (heuristic).

    Three heuristic DiD estimators for continuous-dose panels. This is
    **not** a faithful implementation of Callaway, Goodman-Bacon &
    Sant'Anna (2024); ``method='att_gt'`` is a dose-bin 2×2 DID rollup,
    not the CGS ATT(d|g,t) group-time estimand. A paper-faithful
    ``method='cgs'`` is tracked in ``docs/rfc/continuous_did_cgs.md``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    y : str
        Outcome variable.
    dose : str
        Continuous treatment/dose variable.
    time : str
        Time period variable.
    id : str
        Unit identifier.
    post : str, optional
        Binary post-treatment indicator. If None, inferred from t_pre/t_post.
    t_pre : int, optional
        Last pre-treatment period.
    t_post : int, optional
        First post-treatment period.
    method : str, default 'att_gt'
        Estimation method:

        - ``'att_gt'``: Dose-quantile 2×2 DID rollup (heuristic).
        - ``'twfe'``: TWFE OLS with ``dose × post`` interaction.
        - ``'dose_response'``: local-linear regression of ΔY on baseline dose.
        - ``'cgs'``: Callaway-Goodman-Bacon-Sant'Anna (2024) ATT(d|g,t) /
          ACRT(d|g,t) MVP with ``[待核验]`` markers on paper formulas;
          see ``docs/rfc/continuous_did_cgs.md``. **Not yet reference-
          parity** with R ``contdid``; OR only, no DR/IPW, bootstrap SE.
    n_quantiles : int, default 5
        Number of dose quantiles for discretization.
    controls : list of str, optional
        Control variables.
    cluster : str, optional
        Cluster variable for SE.
    n_boot : int, default 500
        Bootstrap replications for SE.
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> # Estimate effect of varying training hours on wages
    >>> result = sp.continuous_did(
    ...     df, y='wage', dose='training_hours',
    ...     time='year', id='worker_id', t_pre=2019, t_post=2020,
    ... )
    >>> print(result.summary())
    """
    rng = np.random.default_rng(seed)
    df = data.copy()

    # Determine pre/post
    if post is None:
        if t_pre is not None and t_post is not None:
            df['_post'] = (df[time] >= t_post).astype(int)
        else:
            times = sorted(df[time].unique())
            mid = len(times) // 2
            df['_post'] = (df[time] >= times[mid]).astype(int)
        post_col = '_post'
    else:
        post_col = post

    if method == 'twfe':
        return _continuous_did_twfe(df, y, dose, time, id, post_col, controls, cluster, alpha)
    elif method == 'cgs':
        return _continuous_did_cgs(
            df, y=y, dose=dose, time=time, unit=id,
            t_pre=t_pre, t_post=t_post, controls=controls,
            n_boot=n_boot, alpha=alpha, rng=rng,
        )
    elif method == 'dose_response':
        return _continuous_did_dose_response(df, y, dose, time, id, post_col,
                                              controls, n_quantiles, n_boot, alpha, rng)
    else:
        return _continuous_did_att_gt(df, y, dose, time, id, post_col,
                                      controls, n_quantiles, n_boot, alpha, rng)


def _continuous_did_twfe(df, y, dose, time, id, post, controls, cluster, alpha):
    """TWFE approach: y_it = α_i + λ_t + β·dose_i·post_t + ε_it"""
    # Create interaction
    df['_dose_post'] = df[dose] * df[post]

    # Demean (within transformation for FE)
    panel = df.set_index([id, time])
    y_data = panel[y].values.astype(float)

    # Unit means
    unit_means = df.groupby(id)[y].transform('mean').values
    time_means = df.groupby(time)[y].transform('mean').values
    grand_mean = df[y].mean()

    y_demean = y_data - unit_means - time_means + grand_mean

    # Demean dose_post similarly
    dp = df['_dose_post'].values.astype(float)
    dp_unit = df.groupby(id)['_dose_post'].transform('mean').values
    dp_time = df.groupby(time)['_dose_post'].transform('mean').values
    dp_grand = df['_dose_post'].mean()
    dp_demean = dp - dp_unit - dp_time + dp_grand

    # Controls
    if controls:
        X_parts = [dp_demean.reshape(-1, 1)]
        for c in controls:
            cv = df[c].values.astype(float)
            cv_u = df.groupby(id)[c].transform('mean').values
            cv_t = df.groupby(time)[c].transform('mean').values
            cv_g = df[c].mean()
            X_parts.append((cv - cv_u - cv_t + cv_g).reshape(-1, 1))
        X = np.column_stack(X_parts)
    else:
        X = dp_demean.reshape(-1, 1)

    # OLS on demeaned data
    valid = np.isfinite(y_demean) & np.all(np.isfinite(X), axis=1)
    X_v, y_v = X[valid], y_demean[valid]

    try:
        beta = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
        resid = y_v - X_v @ beta
        n, k = X_v.shape

        # Clustered SE
        if cluster:
            clusters = df.loc[valid, cluster if cluster != id else id].values
            unique_clusters = np.unique(clusters)
            n_cl = len(unique_clusters)
            XtX_inv = np.linalg.inv(X_v.T @ X_v)
            meat = np.zeros((k, k))
            for cl in unique_clusters:
                cl_mask = clusters == cl
                score = X_v[cl_mask].T @ resid[cl_mask]
                meat += np.outer(score, score)
            correction = n_cl / (n_cl - 1) * (n - 1) / (n - k)
            var_cov = correction * XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = np.sum(resid**2) / (n - k)
            var_cov = sigma2 * np.linalg.inv(X_v.T @ X_v)

        tau = beta[0]
        se = np.sqrt(var_cov[0, 0])
    except np.linalg.LinAlgError:
        tau, se = np.nan, np.nan

    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_val = 2 * (1 - stats.norm.cdf(abs(tau / se))) if se > 0 else np.nan

    return CausalResult(
        method='Continuous DID (TWFE)',
        estimand='Dose-response coefficient',
        estimate=tau,
        se=se,
        pvalue=p_val,
        ci=(tau - z_crit * se, tau + z_crit * se),
        alpha=alpha,
        n_obs=int(valid.sum()),
        model_info={
            'dose_variable': dose,
            'n_units': df[id].nunique(),
            'n_periods': df[time].nunique(),
        },
    )


def _continuous_did_att_gt(df, y, dose, time, id, post, controls,
                            n_quantiles, n_boot, alpha, rng):
    """
    Group-time ATT approach: discretize dose into quantiles,
    estimate ATT for each dose group vs untreated.
    """
    # Get dose at baseline (pre-period)
    pre_data = df[df[post] == 0]
    dose_baseline = pre_data.groupby(id)[dose].mean()

    # Define dose groups (quantiles)
    quantile_edges = np.quantile(dose_baseline[dose_baseline > 0].values,
                                  np.linspace(0, 1, n_quantiles + 1))
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 2:
        quantile_edges = np.array([dose_baseline.min(), dose_baseline.max()])

    dose_groups = pd.cut(dose_baseline, bins=quantile_edges, labels=False,
                          include_lowest=True)

    # Untreated group: dose = 0
    untreated_ids = dose_baseline[dose_baseline == 0].index

    results_rows = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for g in sorted(dose_groups.dropna().unique()):
        group_ids = dose_groups[dose_groups == g].index
        if len(group_ids) < 2:
            continue

        # Compute DID for this dose group vs untreated
        group_data = df[df[id].isin(group_ids)]
        control_data = df[df[id].isin(untreated_ids)]

        if len(control_data) == 0:
            # Use lowest dose group as control
            lowest_g = dose_groups.dropna().min()
            if g == lowest_g:
                continue
            control_ids = dose_groups[dose_groups == lowest_g].index
            control_data = df[df[id].isin(control_ids)]

        # DID: E[Y|post=1,group] - E[Y|post=0,group] - (E[Y|post=1,ctrl] - E[Y|post=0,ctrl])
        y_g_post = group_data.loc[group_data[post] == 1, y].mean()
        y_g_pre = group_data.loc[group_data[post] == 0, y].mean()
        y_c_post = control_data.loc[control_data[post] == 1, y].mean()
        y_c_pre = control_data.loc[control_data[post] == 0, y].mean()

        att = (y_g_post - y_g_pre) - (y_c_post - y_c_pre)

        # Bootstrap SE
        boot_atts = np.empty(n_boot)
        all_ids = np.concatenate([group_ids.values, untreated_ids.values])
        for b in range(n_boot):
            boot_ids = rng.choice(all_ids, size=len(all_ids), replace=True)
            boot_df = df[df[id].isin(boot_ids)]
            boot_group = boot_df[boot_df[id].isin(group_ids)]
            boot_ctrl = boot_df[boot_df[id].isin(untreated_ids)]

            if len(boot_group) == 0 or len(boot_ctrl) == 0:
                boot_atts[b] = np.nan
                continue

            b_g_post = boot_group.loc[boot_group[post] == 1, y].mean()
            b_g_pre = boot_group.loc[boot_group[post] == 0, y].mean()
            b_c_post = boot_ctrl.loc[boot_ctrl[post] == 1, y].mean()
            b_c_pre = boot_ctrl.loc[boot_ctrl[post] == 0, y].mean()
            boot_atts[b] = (b_g_post - b_g_pre) - (b_c_post - b_c_pre)

        se = np.nanstd(boot_atts, ddof=1)
        dose_midpoint = dose_baseline[group_ids].mean()

        results_rows.append({
            'dose_group': int(g),
            'dose_midpoint': dose_midpoint,
            'att': att,
            'se': se,
            'ci_lower': att - z_crit * se,
            'ci_upper': att + z_crit * se,
            'p_value': 2 * (1 - stats.norm.cdf(abs(att / se))) if se > 0 else np.nan,
            'n_treated': len(group_ids),
            'n_control': len(untreated_ids),
        })

    results_df = pd.DataFrame(results_rows)

    # Aggregate: dose-weighted average
    if len(results_df) > 0:
        weights = results_df['n_treated'].values / results_df['n_treated'].sum()
        pooled_att = np.sum(weights * results_df['att'].values)
        pooled_se = np.sqrt(np.sum(weights**2 * results_df['se'].values**2))
    else:
        pooled_att, pooled_se = np.nan, np.nan

    p_val = 2 * (1 - stats.norm.cdf(abs(pooled_att / pooled_se))) if pooled_se > 0 else np.nan

    return CausalResult(
        method='Continuous DID (dose-bin heuristic)',
        estimand='Sample-weighted mean of dose-bin 2x2 DIDs (not CGS 2024 ATT(d|g,t))',
        estimate=pooled_att,
        se=pooled_se,
        pvalue=p_val,
        ci=(pooled_att - z_crit * pooled_se, pooled_att + z_crit * pooled_se),
        alpha=alpha,
        n_obs=len(df),
        detail=results_df if len(results_df) > 0 else None,
        model_info={
            'dose_variable': dose,
            'n_dose_groups': len(results_df),
            'n_units': df[id].nunique(),
        },
    )


def _continuous_did_dose_response(df, y, dose, time, id, post,
                                   controls, n_quantiles, n_boot, alpha, rng):
    """
    Estimate a full dose-response curve in DID framework.
    Uses local linear regression of the DID estimand on dose.
    """
    # Compute unit-level DID: ΔY_i = Y_i,post - Y_i,pre
    pre_y = df[df[post] == 0].groupby(id)[y].mean()
    post_y = df[df[post] == 1].groupby(id)[y].mean()
    common_ids = pre_y.index.intersection(post_y.index)

    delta_y = post_y[common_ids] - pre_y[common_ids]
    dose_vals = df[df[post] == 0].groupby(id)[dose].mean()[common_ids]

    # Local linear regression of ΔY on dose
    from ..nonparametric.lpoly import lpoly as _lpoly

    # Create temporary df for lpoly
    temp_df = pd.DataFrame({'delta_y': delta_y.values, 'dose': dose_vals.values})
    temp_df = temp_df.dropna()

    try:
        lp_result = _lpoly(temp_df, y='delta_y', x='dose', degree=1, n_grid=50)
    except Exception:
        # Fallback: simple linear
        from scipy.stats import linregress
        slope, intercept, _, p_val, se_slope = linregress(dose_vals.dropna(), delta_y.dropna())

        z_crit = stats.norm.ppf(1 - alpha / 2)
        return CausalResult(
            method='Continuous DID (Dose-Response)',
            estimand='Average marginal effect',
            estimate=slope,
            se=se_slope,
            pvalue=p_val,
            ci=(slope - z_crit * se_slope, slope + z_crit * se_slope),
            alpha=alpha,
            n_obs=len(temp_df),
        )

    # Use slope at different dose levels as the treatment effect
    # Average derivative as summary measure
    avg_effect = np.nanmean(np.diff(lp_result.fitted) / np.diff(lp_result.grid))
    avg_se = np.nanmean(lp_result.se)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_val = 2 * (1 - stats.norm.cdf(abs(avg_effect / avg_se))) if avg_se > 0 else np.nan

    return CausalResult(
        method='Continuous DID (Dose-Response)',
        estimand='Average marginal effect',
        estimate=avg_effect,
        se=avg_se,
        pvalue=p_val,
        ci=(avg_effect - z_crit * avg_se, avg_effect + z_crit * avg_se),
        alpha=alpha,
        n_obs=len(temp_df),
        model_info={
            'dose_response_grid': lp_result.grid.tolist(),
            'dose_response_fitted': lp_result.fitted.tolist(),
            'n_units': len(common_ids),
        },
    )


# ----------------------------------------------------------------------
# method='cgs' — Callaway-Goodman-Bacon-Sant'Anna (2024) MVP
# ----------------------------------------------------------------------

def _continuous_did_cgs(
    df: pd.DataFrame,
    *,
    y: str,
    dose: str,
    time: str,
    unit: str,
    t_pre: Optional[int],
    t_post: Optional[int],
    controls: Optional[List[str]],
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> CausalResult:
    """Callaway-Goodman-Bacon-Sant'Anna (2024) continuous DiD — MVP.

    **This is an MVP, not paper-parity.** See
    ``docs/rfc/continuous_did_cgs.md`` for the full design + test plan.
    All paper-specific identification formulas are flagged [待核验]
    below.

    MVP identification [待核验 — CGS 2024 §3-§4]
    -------------------------------------------
    Assumes a 2-period design (pre, post) with continuous dose d ∈ [0, ∞)
    on the treated arm and d = 0 on the untreated arm. Under strong
    parallel trends:

        ATT(d | post) ≈ E[Y_post − Y_pre | D = d]
                         − E[Y_post − Y_pre | D = 0]

    ATT(d) is estimated as a function of d by local-linear smoother over
    the unit-level long differences. ACRT(d) = d ATT(d) / dd is the
    derivative, approximated by first-differencing the smoother.

    MVP limitations
    ---------------
    - Only 2-period (pre, post) designs. Full CGS group-time ATT(d|g,t)
      across multiple cohorts / periods is not in this MVP.
    - OR only (no IPW / DR / influence-function variance). SE via
      pairs bootstrap.
    - No explicit strong-parallel-trends diagnostic yet.
    """
    # 2-period design: pre is everything with time < t_post (or == t_pre);
    # post is everything with time >= t_post (or == t_post if given).
    times = sorted(df[time].unique())
    if t_pre is None:
        t_pre_val = times[0]
    else:
        t_pre_val = t_pre
    if t_post is None:
        t_post_val = times[-1]
    else:
        t_post_val = t_post

    pre = df[df[time] == t_pre_val]
    post = df[df[time] == t_post_val]
    merged = post[[unit, y, dose] + (controls or [])].merge(
        pre[[unit, y]].rename(columns={y: f"{y}_pre"}),
        on=unit, how="inner",
    )
    if merged.empty:
        return CausalResult(
            method="Continuous DID (CGS 2024 MVP) [待核验]",
            estimand="ATT(d) averaged over treated dose support",
            estimate=np.nan, se=np.nan, pvalue=np.nan,
            ci=(np.nan, np.nan), alpha=alpha, n_obs=0,
        )
    merged["_dy"] = merged[y] - merged[f"{y}_pre"]

    untreated = merged[merged[dose] == 0]
    treated = merged[merged[dose] > 0]

    if len(untreated) < 2:
        return CausalResult(
            method="Continuous DID (CGS 2024 MVP) [待核验]",
            estimand="ATT(d) averaged over treated dose support",
            estimate=np.nan, se=np.nan, pvalue=np.nan,
            ci=(np.nan, np.nan), alpha=alpha, n_obs=int(len(merged)),
            model_info={
                "warning": "No dose=0 control units; paper requires never-treated.",
            },
        )
    if len(treated) < 3:
        return CausalResult(
            method="Continuous DID (CGS 2024 MVP) [待核验]",
            estimand="ATT(d) averaged over treated dose support",
            estimate=np.nan, se=np.nan, pvalue=np.nan,
            ci=(np.nan, np.nan), alpha=alpha, n_obs=int(len(merged)),
            model_info={
                "warning": "Too few treated units with dose>0 to fit ATT(d).",
            },
        )

    # Control baseline: average ΔY among dose=0 units.
    control_dy = untreated["_dy"].mean()

    # ATT(d) on the treated support: local-linear smoother of (Δy - control_dy)
    # on d. Use the Nadaraya-Watson-ish average for a robust MVP.
    treated_sorted = treated.sort_values(dose).reset_index(drop=True)
    treated_sorted["_att_raw"] = treated_sorted["_dy"] - control_dy

    # Grid over the treated dose support (quartile-spaced for evenness)
    n_grid = 50
    d_vals = treated_sorted[dose].values.astype(float)
    d_lo, d_hi = np.percentile(d_vals, [5, 95])
    grid = np.linspace(d_lo, d_hi, n_grid)

    att_curve = _local_linear_curve(
        x=d_vals,
        y=treated_sorted["_att_raw"].values.astype(float),
        grid=grid,
    )

    # ACRT(d) = derivative of att_curve
    acrt_curve = np.gradient(att_curve, grid)

    # Summary scalars
    att_overall = float(np.nanmean(treated_sorted["_att_raw"].values))
    acrt_overall = float(np.nanmean(acrt_curve))

    # Bootstrap SE for ATT_overall and ACRT_overall
    boot_att = np.full(n_boot, np.nan)
    boot_acrt = np.full(n_boot, np.nan)
    n_merged = len(merged)
    for b in range(n_boot):
        idx = rng.integers(0, n_merged, size=n_merged)
        sample = merged.iloc[idx]
        u_b = sample[sample[dose] == 0]
        t_b = sample[sample[dose] > 0]
        if len(u_b) < 2 or len(t_b) < 3:
            continue
        ctrl_b = u_b["_dy"].mean()
        ta_b = t_b["_dy"].values - ctrl_b
        boot_att[b] = float(np.nanmean(ta_b))
        # ACRT: fit local linear on bootstrap sample and take mean slope
        d_b = t_b[dose].values.astype(float)
        order = np.argsort(d_b)
        try:
            curve_b = _local_linear_curve(
                x=d_b[order], y=ta_b[order], grid=grid,
            )
            boot_acrt[b] = float(np.nanmean(np.gradient(curve_b, grid)))
        except Exception:
            continue

    se_att = float(np.nanstd(boot_att, ddof=1))
    se_acrt = float(np.nanstd(boot_acrt, ddof=1))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    p_att = (float(2 * (1 - stats.norm.cdf(abs(att_overall / se_att))))
             if (se_att > 0 and np.isfinite(se_att)) else np.nan)
    ci_att = (att_overall - z_crit * se_att,
              att_overall + z_crit * se_att) if se_att > 0 else (np.nan, np.nan)

    detail_df = pd.DataFrame({
        "dose": grid,
        "att_d": att_curve,
        "acrt_d": acrt_curve,
    })

    return CausalResult(
        method="Continuous DID (CGS 2024 MVP) [待核验 — OR only, 2-period design]",
        estimand="ATT(d) averaged over treated support",
        estimate=att_overall,
        se=se_att,
        pvalue=p_att,
        ci=ci_att,
        alpha=alpha,
        n_obs=int(len(merged)),
        detail=detail_df,
        model_info={
            "acrt_overall": acrt_overall,
            "acrt_se": se_acrt,
            "n_treated": int(len(treated)),
            "n_control_d0": int(len(untreated)),
            "grid_range": (float(d_lo), float(d_hi)),
            "warning": (
                "MVP: OR only, 2-period design. Full CGS (2024) ATT(d|g,t) "
                "across cohorts + IPW/DR + analytical IF variance pending. "
                "See docs/rfc/continuous_did_cgs.md for the roadmap."
            ),
        },
        _citation_key="callaway2024difference",
    )


def _local_linear_curve(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Local-linear smoother on x, y evaluated at grid points.

    Gaussian kernel, Silverman rule-of-thumb bandwidth by default. Used
    for the ATT(d) curve in the CGS MVP.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if bandwidth is None:
        n = len(x)
        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        scale = min(std, iqr / 1.349) if iqr > 0 else std
        bandwidth = 1.06 * scale * n ** (-0.2) if scale > 0 else 0.1

    out = np.empty(len(grid))
    for i, g in enumerate(grid):
        u = (x - g) / bandwidth
        w = np.exp(-0.5 * u * u)
        if w.sum() < 1e-10:
            out[i] = np.nan
            continue
        # Weighted local linear: fit y = a + b*(x - g) with weights w.
        X = np.column_stack([np.ones_like(x), x - g])
        W = w[:, None]
        try:
            XtWX = X.T @ (W * X)
            XtWy = X.T @ (W[:, 0] * y)
            beta = np.linalg.solve(XtWX, XtWy)
            out[i] = float(beta[0])
        except np.linalg.LinAlgError:
            out[i] = float(np.average(y, weights=w)) if w.sum() > 0 else np.nan
    return out
