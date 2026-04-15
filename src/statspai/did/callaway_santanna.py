"""
Callaway & Sant'Anna (2021) estimator for staggered DID.

Estimates group-time average treatment effects ATT(g,t) under staggered
treatment adoption, with proper handling of heterogeneous treatment effects
that invalidate standard TWFE estimators.

Supports three estimation approaches:
- Doubly Robust (DR) — default, combines outcome regression and IPW
- Inverse Probability Weighting (IPW)
- Outcome Regression (REG)

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021).
"Difference-in-Differences with Multiple Time Periods."
*Journal of Econometrics*, 225(2), 200-230.

Sant'Anna, P.H.C. and Zhao, J. (2020).
"Doubly Robust Difference-in-Differences Estimators."
*Journal of Econometrics*, 219(1), 101-122.
"""

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def callaway_santanna(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    i: str,
    x: Optional[List[str]] = None,
    estimator: str = 'dr',
    control_group: str = 'nevertreated',
    base_period: str = 'universal',
    anticipation: int = 0,
    alpha: float = 0.05,
    panel: bool = True,
) -> CausalResult:
    """
    Callaway & Sant'Anna (2021) estimator for staggered DID.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    y : str
        Outcome variable name.
    g : str
        Group variable: first period of treatment (0 = never treated).
    t : str
        Time period variable.
    i : str
        Unit identifier variable.
    x : list of str, optional
        Covariate names for conditional parallel trends.
    estimator : str, default 'dr'
        Estimation method: 'dr' (doubly robust), 'ipw', or 'reg'.
    control_group : str, default 'nevertreated'
        Comparison group: 'nevertreated' or 'notyettreated'.
    base_period : str, default 'universal'
        Base period: 'universal' (always g-1) or 'varying'.
    anticipation : int, default 0
        Number of pre-treatment periods over which units may anticipate
        the treatment.  Shifts the base period back by ``anticipation``
        periods and drops the (g, t) pairs with ``t > g - anticipation - 1``
        but ``t < g`` from the post-treatment set.  See Callaway &
        Sant'Anna (2021), Section 3.2.
    alpha : float, default 0.05
        Significance level.
    panel : bool, default True
        If ``True`` (default), treat the data as a balanced panel and
        estimate ATT(g, t) via within-unit first differences.
        If ``False``, treat the data as *repeated cross-sections* —
        observations are not matched across time.  In RCS mode the
        estimator is the unconditional 2×2 cell-mean DID per (g, t)
        pair (CS2021 §3.2, eqn 2.4, RCS version).  The covariate-free
        ``estimator='reg'`` path is the only one currently supported
        for RCS; IPW / DR can be added later.

    Returns
    -------
    CausalResult
        Results with group-time ATTs, event study coefficients,
        pre-trend test, and all standard CausalResult methods.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> # Create staggered panel data
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for unit in range(90):
    ...     g_val = [4, 6, 0][unit // 30]  # 3 cohorts
    ...     for period in range(1, 9):
    ...         te = max(0, period - g_val + 1) if g_val > 0 else 0
    ...         rows.append({'i': unit, 't': period, 'y': te + rng.normal(),
    ...                      'g': g_val})
    >>> df = pd.DataFrame(rows)
    >>> result = callaway_santanna(df, y='y', g='g', t='t', i='i')
    >>> result.estimate > 0
    True
    """
    if estimator not in ('dr', 'ipw', 'reg'):
        raise ValueError(f"estimator must be 'dr', 'ipw', or 'reg', got '{estimator}'")
    if control_group not in ('nevertreated', 'notyettreated'):
        raise ValueError(
            f"control_group must be 'nevertreated' or 'notyettreated', "
            f"got '{control_group}'"
        )
    if anticipation < 0:
        raise ValueError(f"anticipation must be >= 0, got {anticipation}")

    # ---- Repeated cross-sections branch --------------------------------
    if not panel:
        if estimator != 'reg':
            raise NotImplementedError(
                "panel=False currently only supports estimator='reg' "
                "(unconditional / covariate-adjusted 2×2 cell-mean DID).  "
                "IPW / DR for RCS are planned for a future release."
            )
        if control_group != 'nevertreated':
            raise NotImplementedError(
                "panel=False currently requires "
                "control_group='nevertreated'."
            )
        return _callaway_santanna_rcs(
            data=data, y=y, g=g, t=t, x=x,
            base_period=base_period, anticipation=anticipation,
            alpha=alpha,
        )

    # 1. Prepare panel data
    y_wide, unit_info, time_periods, cohorts, n_units = _prepare_panel(
        data, y, g, t, i, x
    )

    if not cohorts:
        raise ValueError("No treatment cohorts found. Check group variable encoding.")

    # 2. Determine (g, t, base) estimation triples
    gt_pairs = _get_gt_pairs(cohorts, time_periods, base_period, anticipation)
    if not gt_pairs:
        raise ValueError("No valid (group, time) pairs to estimate.")

    # 3. Estimate ATT(g,t) for each pair
    gt_results: List[Dict[str, Any]] = []
    inf_funcs_list: List[np.ndarray] = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for g_val, t_val, base_val in gt_pairs:
        att, se, inf_func = _estimate_single_att(
            y_wide, unit_info, g_val, t_val, base_val,
            g, x, estimator, control_group, n_units,
        )

        pval = 2 * (1 - stats.norm.cdf(abs(att / se))) if se > 0 else 1.0

        gt_results.append({
            'group': g_val,
            'time': t_val,
            'att': att,
            'se': se,
            'ci_lower': att - z_crit * se,
            'ci_upper': att + z_crit * se,
            'pvalue': pval,
            'relative_time': t_val - g_val,
        })
        inf_funcs_list.append(inf_func)

    detail = pd.DataFrame(gt_results)

    # Stack influence functions: (n_units, n_gt_pairs)
    inf_matrix = np.column_stack(inf_funcs_list) if inf_funcs_list else None

    # 4. Cohort sizes (for weighting)
    cohort_sizes = unit_info[g].value_counts()

    # 5. Simple aggregation (post-treatment)
    post_mask = detail['relative_time'] >= 0
    agg_est, agg_se, agg_pval, agg_ci = _aggregate_simple(
        detail[post_mask],
        inf_matrix[:, post_mask.values] if inf_matrix is not None else None,
        cohort_sizes, n_units, alpha,
    )

    # 6. Event study aggregation
    event_study = _aggregate_event_study(
        detail, inf_matrix, cohort_sizes, n_units, alpha,
    )

    # 7. Pre-trend test
    pretrend = _pretrend_test(detail, inf_matrix, n_units)

    # 8. Build result
    model_info: Dict[str, Any] = {
        'estimator': estimator.upper(),
        'control_group': control_group,
        'base_period': base_period,
        'anticipation': anticipation,
        'n_units': n_units,
        'n_periods': len(time_periods),
        'n_cohorts': len(cohorts),
        'cohorts': cohorts,
        'event_study': event_study,
        'pretrend_test': pretrend,
        'cohort_sizes': cohort_sizes,
    }

    return CausalResult(
        method="Callaway and Sant'Anna (2021)",
        estimand='ATT',
        estimate=agg_est,
        se=agg_se,
        pvalue=agg_pval,
        ci=agg_ci,
        alpha=alpha,
        n_obs=len(data),
        detail=detail,
        model_info=model_info,
        _influence_funcs=inf_matrix,
        _citation_key='callaway_santanna',
    )


# ======================================================================
# Data preparation
# ======================================================================

def _prepare_panel(
    data: pd.DataFrame,
    y: str, g: str, t: str, i: str,
    x: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, list, list, int]:
    """Validate and reshape panel data to wide format.

    Returns
    -------
    y_wide : pd.DataFrame
        Outcomes pivoted to (units × time periods).
    unit_info : pd.DataFrame
        Unit-level info indexed by unit id (group, covariates).
    time_periods : list
        Sorted unique time periods.
    cohorts : list
        Sorted treatment cohorts (excluding never-treated).
    n_units : int
    """
    for col in (y, g, t, i):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    if x:
        for col in x:
            if col not in data.columns:
                raise ValueError(f"Covariate '{col}' not found in data")

    # Pivot outcome to wide: rows = units, columns = time periods
    y_wide = data.pivot_table(index=i, columns=t, values=y, aggfunc='first')

    # Unit-level info (first occurrence per unit)
    info_cols = [g] + (x or [])
    unit_info = data.groupby(i)[info_cols].first()

    # Replace NaN / inf in group variable with 0 (never-treated)
    unit_info[g] = unit_info[g].fillna(0).replace([np.inf, -np.inf], 0)
    # Ensure integer type for group
    unit_info[g] = unit_info[g].astype(int)

    time_periods = sorted(data[t].unique())
    max_time = max(time_periods)

    g_values = sorted(unit_info[g].unique())
    # Cohorts: groups that actually get treated within the sample period
    cohorts = [v for v in g_values if v > 0 and v <= max_time]

    n_units = len(unit_info)
    return y_wide, unit_info, time_periods, cohorts, n_units


def _get_gt_pairs(
    cohorts: list,
    time_periods: list,
    base_period: str,
    anticipation: int = 0,
) -> List[Tuple[int, int, int]]:
    """Determine all (g, t, base) triples to estimate.

    With ``anticipation = δ > 0`` the base period for cohort g is shifted
    from g − 1 to g − 1 − δ (CS2021, Section 3.2). For the 'varying' base
    scheme the per-t base is similarly shifted to ``t − 1 − δ`` when t
    falls in the pre-treatment region.
    """
    pairs = []
    for g_val in cohorts:
        pre_cutoff = g_val - 1 - anticipation
        available_pre = [tp for tp in time_periods if tp <= pre_cutoff]
        if not available_pre:
            continue
        universal_base = max(available_pre)

        for t_val in time_periods:
            if t_val == universal_base:
                continue  # skip the base period itself

            if base_period == 'varying' and t_val < g_val:
                pre_of_t = [tp for tp in time_periods if tp <= t_val - 1 - anticipation]
                if not pre_of_t:
                    continue
                this_base = max(pre_of_t)
            else:
                this_base = universal_base

            pairs.append((g_val, t_val, this_base))

    return pairs


# ======================================================================
# ATT(g,t) estimators
# ======================================================================

def _estimate_single_att(
    y_wide: pd.DataFrame,
    unit_info: pd.DataFrame,
    g_val: int,
    t_val: int,
    base_val: int,
    g_col: str,
    x_cols: Optional[List[str]],
    estimator: str,
    control_group: str,
    n_total: int,
) -> Tuple[float, float, np.ndarray]:
    """Estimate a single ATT(g,t) and return (att, se, influence_func)."""

    g_series = unit_info[g_col]

    # Treatment indicator: units first treated at g_val
    is_treated = (g_series == g_val)

    # Control indicator
    if control_group == 'nevertreated':
        is_control = (g_series == 0)
    else:  # notyettreated
        is_control = (g_series == 0) | (g_series > t_val)

    # Outcome change ΔY = Y_t - Y_base
    if t_val not in y_wide.columns or base_val not in y_wide.columns:
        return 0.0, np.inf, np.zeros(n_total)

    dy = y_wide[t_val] - y_wide[base_val]

    # Valid units: in a relevant group AND observed in both periods
    relevant = (is_treated | is_control) & dy.notna()
    n_rel = relevant.sum()
    if n_rel < 5:
        return 0.0, np.inf, np.zeros(n_total)

    dy_sub = dy[relevant].values
    d_sub = is_treated[relevant].values.astype(float)
    n1 = d_sub.sum()
    n0 = n_rel - n1

    if n1 < 1 or n0 < 1:
        return 0.0, np.inf, np.zeros(n_total)

    # Covariates
    x_sub = None
    if x_cols:
        x_sub = unit_info.loc[relevant, x_cols].values.astype(float)
        # Drop covariates with zero variance
        var = np.var(x_sub, axis=0)
        if np.any(var < 1e-12):
            keep = var >= 1e-12
            if keep.sum() == 0:
                x_sub = None
            else:
                x_sub = x_sub[:, keep]

    # Dispatch
    if estimator == 'dr':
        att, se, inf_local = _dr_att(dy_sub, d_sub, x_sub)
    elif estimator == 'ipw':
        att, se, inf_local = _ipw_att(dy_sub, d_sub, x_sub)
    else:  # reg
        att, se, inf_local = _reg_att(dy_sub, d_sub, x_sub)

    # Map local influence function to full-sample vector
    inf_full = np.zeros(n_total)
    relevant_idx = np.where(relevant.values)[0]
    inf_full[relevant_idx] = inf_local

    return att, se, inf_full


# ------------------------------------------------------------------
# Doubly Robust
# ------------------------------------------------------------------

def _dr_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
) -> Tuple[float, float, np.ndarray]:
    """Doubly robust ATT(g,t) estimator (Sant'Anna & Zhao 2020)."""
    n = len(dy)
    c = 1 - d

    # --- Propensity score ---
    pscore = _estimate_pscore(d, x, n)

    # --- Outcome regression ---
    m_hat = _estimate_outcome_reg(dy, c, x, n)

    # --- DR weights ---
    p_d = np.mean(d)
    w1 = d / p_d if p_d > 0 else np.zeros(n)

    ipw_raw = pscore * c / (1 - pscore)
    ipw_denom = np.mean(ipw_raw)
    w0 = ipw_raw / ipw_denom if ipw_denom > 1e-12 else np.zeros(n)

    # ATT
    att = float(np.mean((w1 - w0) * (dy - m_hat)))

    # Influence function (treating nuisance as known — asymptotically equivalent)
    inf_func = (w1 - w0) * (dy - m_hat) - att * w1
    se = float(np.sqrt(np.mean(inf_func ** 2) / n))

    return att, se, inf_func


# ------------------------------------------------------------------
# IPW
# ------------------------------------------------------------------

def _ipw_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
) -> Tuple[float, float, np.ndarray]:
    """IPW ATT(g,t) estimator."""
    n = len(dy)
    c = 1 - d

    pscore = _estimate_pscore(d, x, n)

    p_d = np.mean(d)
    w1 = d / p_d if p_d > 0 else np.zeros(n)

    ipw_raw = pscore * c / (1 - pscore)
    ipw_denom = np.mean(ipw_raw)
    w0 = ipw_raw / ipw_denom if ipw_denom > 1e-12 else np.zeros(n)

    att = float(np.mean((w1 - w0) * dy))

    inf_func = (w1 - w0) * dy - att * w1
    se = float(np.sqrt(np.mean(inf_func ** 2) / n))

    return att, se, inf_func


# ------------------------------------------------------------------
# Outcome regression
# ------------------------------------------------------------------

def _reg_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
) -> Tuple[float, float, np.ndarray]:
    """Outcome regression ATT(g,t) estimator."""
    n = len(dy)
    c = 1 - d

    m_hat = _estimate_outcome_reg(dy, c, x, n)

    p_d = np.mean(d)
    w1 = d / p_d if p_d > 0 else np.zeros(n)

    att = float(np.mean(w1 * (dy - m_hat)))

    inf_func = w1 * (dy - m_hat) - att * w1
    se = float(np.sqrt(np.mean(inf_func ** 2) / n))

    return att, se, inf_func


# ======================================================================
# Nuisance estimators
# ======================================================================

def _estimate_pscore(
    d: np.ndarray,
    x: Optional[np.ndarray],
    n: int,
) -> np.ndarray:
    """Estimate propensity score P(D=1 | X) via logit.

    Uses statsmodels (core dep) — no sklearn needed.
    Falls back to unconditional probability if logit fails or no covariates.
    """
    p_d = np.mean(d)
    if x is None or x.shape[1] == 0:
        return np.full(n, p_d)

    try:
        import statsmodels.api as sm
        x_const = sm.add_constant(x)
        logit = sm.Logit(d, x_const)
        result = logit.fit(disp=0, maxiter=500, warn_convergence=False)
        pscore = result.predict(x_const)
    except Exception:
        pscore = np.full(n, p_d)

    return np.clip(pscore, 1e-6, 1 - 1e-6)


def _estimate_outcome_reg(
    dy: np.ndarray,
    c: np.ndarray,
    x: Optional[np.ndarray],
    n: int,
) -> np.ndarray:
    """Estimate E[ΔY | X, D=0] via OLS on the control group."""
    c_mask = c.astype(bool)
    c_count = c_mask.sum()

    if x is None or x.shape[1] == 0 or c_count < 2:
        m0 = np.mean(dy[c_mask]) if c_count > 0 else 0.0
        return np.full(n, m0)

    k = x.shape[1]
    if c_count <= k + 1:
        # Not enough control obs for regression
        return np.full(n, np.mean(dy[c_mask]))

    try:
        import statsmodels.api as sm
        x_const = sm.add_constant(x[c_mask])
        ols = sm.OLS(dy[c_mask], x_const)
        result = ols.fit()
        m_hat = result.predict(sm.add_constant(x))
    except Exception:
        m_hat = np.full(n, np.mean(dy[c_mask]))

    return m_hat


# ======================================================================
# Aggregation
# ======================================================================

def _aggregate_simple(
    post_detail: pd.DataFrame,
    post_inf: Optional[np.ndarray],
    cohort_sizes: pd.Series,
    n_total: int,
    alpha: float,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Simple aggregation: group-size-weighted average of post-treatment ATTs."""
    if len(post_detail) == 0:
        return 0.0, np.inf, 1.0, (np.nan, np.nan)

    weights = post_detail['group'].map(cohort_sizes).values.astype(float)
    weights = weights / weights.sum()

    att_agg = float(np.average(post_detail['att'].values, weights=weights))

    if post_inf is not None and post_inf.shape[1] > 0:
        inf_agg = post_inf @ weights
        se_agg = float(np.sqrt(np.mean(inf_agg ** 2) / n_total))
    else:
        se_agg = float(np.sqrt(np.average(post_detail['se'].values ** 2,
                                          weights=weights)))

    z = att_agg / se_agg if se_agg > 0 else 0
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att_agg - z_crit * se_agg, att_agg + z_crit * se_agg)

    return att_agg, se_agg, pval, ci


def _aggregate_event_study(
    detail: pd.DataFrame,
    inf_matrix: Optional[np.ndarray],
    cohort_sizes: pd.Series,
    n_total: int,
    alpha: float,
) -> pd.DataFrame:
    """Event study aggregation: average ATT by relative time e = t − g."""
    relative_times = sorted(detail['relative_time'].unique())
    z_crit = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for e in relative_times:
        mask = detail['relative_time'] == e
        sub = detail[mask]
        if len(sub) == 0:
            continue

        weights = sub['group'].map(cohort_sizes).values.astype(float)
        w_sum = weights.sum()
        if w_sum == 0:
            continue
        weights = weights / w_sum

        att_e = float(np.average(sub['att'].values, weights=weights))

        if inf_matrix is not None:
            col_idx = np.where(mask.values)[0]
            inf_e = inf_matrix[:, col_idx] @ weights
            se_e = float(np.sqrt(np.mean(inf_e ** 2) / n_total))
        else:
            se_e = float(np.sqrt(np.average(sub['se'].values ** 2,
                                            weights=weights)))

        pval = float(2 * (1 - stats.norm.cdf(abs(att_e / se_e)))) if se_e > 0 else 1.0

        rows.append({
            'relative_time': e,
            'att': att_e,
            'se': se_e,
            'ci_lower': att_e - z_crit * se_e,
            'ci_upper': att_e + z_crit * se_e,
            'pvalue': pval,
        })

    return pd.DataFrame(rows)


# ======================================================================
# Pre-trend test
# ======================================================================

def _pretrend_test(
    detail: pd.DataFrame,
    inf_matrix: Optional[np.ndarray],
    n_total: int,
) -> Dict[str, Any]:
    """Joint Wald test for H0: all pre-treatment ATT(g,t) = 0."""
    pre_mask = detail['relative_time'] < 0
    pre = detail[pre_mask]

    if len(pre) == 0:
        return {'statistic': np.nan, 'df': 0, 'pvalue': np.nan}

    theta = pre['att'].values
    k = len(theta)

    if inf_matrix is not None:
        col_idx = np.where(pre_mask.values)[0]
        inf_pre = inf_matrix[:, col_idx]
        # Variance-covariance: V = (1/n²) IF' IF
        V = inf_pre.T @ inf_pre / (n_total ** 2)
    else:
        V = np.diag(pre['se'].values ** 2)

    # Regularise for numerical stability
    V += np.eye(k) * 1e-10

    try:
        V_inv = np.linalg.inv(V)
        W = float(theta @ V_inv @ theta)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)
        W = float(theta @ V_inv @ theta)

    pvalue = float(1 - stats.chi2.cdf(W, k))

    return {'statistic': W, 'df': k, 'pvalue': pvalue}


# ======================================================================
# Repeated cross-sections (panel=False) branch
# ======================================================================

def _callaway_santanna_rcs(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    base_period: str,
    anticipation: int,
    alpha: float,
    x: Optional[List[str]] = None,
) -> CausalResult:
    """Unconditional (or regression-adjusted) 2×2 cell-mean DID for RCS.

    For each (g, t) pair with base period b:

        ATT(g, t) = (Ȳ_{g,t} - Ȳ_{g,b}) - (Ȳ_{c,t} - Ȳ_{c,b})

    where c = never-treated cohort.  Observation-level influence
    functions are assembled as

        ψ_i =  1{G_i=g, T_i=t}  (Y_i - Ȳ_{g,t}) / p_{g,t}
            -  1{G_i=g, T_i=b}  (Y_i - Ȳ_{g,b}) / p_{g,b}
            -  1{G_i=c, T_i=t}  (Y_i - Ȳ_{c,t}) / p_{c,t}
            +  1{G_i=c, T_i=b}  (Y_i - Ȳ_{c,b}) / p_{c,b}

    with ``p_{g,t} = #{i: G_i=g, T_i=t} / n``.  SE(ATT) is the sample
    variance of ψ divided by ``n``, matching CS2021 eqn (2.4) for RCS.
    """
    df = data.copy()
    for col in (y, g, t):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")
    if x:
        for col in x:
            if col not in df.columns:
                raise ValueError(f"Covariate '{col}' not found in data")

    df[g] = df[g].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    drop_cols = [y, t] + (list(x) if x else [])
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    n_obs = len(df)
    if n_obs == 0:
        raise ValueError("No observations after dropping NaNs.")

    # Covariate adjustment: residualise Y on X using the never-treated
    # pool with period fixed effects. Plug-in influence functions treat
    # β̂ as known (asymptotically valid; see Sant'Anna & Zhao 2020).
    y_series = df[y].astype(float).to_numpy().copy()
    covariate_info: Optional[Dict[str, Any]] = None
    if x:
        y_series = _rcs_residualise_on_controls(y_series, df, g, t, x)
        covariate_info = {
            'covariates': list(x),
            'approach': 'residualisation on never-treated with period FEs',
        }

    time_periods = sorted(df[t].unique())
    t_max = max(time_periods)
    cohorts = sorted([v for v in df[g].unique() if v > 0 and v <= t_max])
    if not cohorts:
        raise ValueError("No treatment cohorts found.")

    gt_pairs = _get_gt_pairs(cohorts, time_periods, base_period, anticipation)
    if not gt_pairs:
        raise ValueError("No valid (group, time) pairs to estimate.")

    y_arr = y_series  # possibly residualised
    g_arr = df[g].values
    t_arr = df[t].values

    gt_results: List[Dict[str, Any]] = []
    inf_funcs_list: List[np.ndarray] = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for g_val, t_val, base_val in gt_pairs:
        att, se, inf_func = _estimate_single_att_rcs(
            y_arr, g_arr, t_arr,
            g_val=g_val, t_val=t_val, base_val=base_val,
            n_obs=n_obs,
        )
        pval = (float(2 * (1 - stats.norm.cdf(abs(att / se))))
                if se > 0 else 1.0)
        gt_results.append({
            'group': g_val,
            'time': t_val,
            'att': att,
            'se': se,
            'ci_lower': att - z_crit * se,
            'ci_upper': att + z_crit * se,
            'pvalue': pval,
            'relative_time': t_val - g_val,
        })
        inf_funcs_list.append(inf_func)

    detail = pd.DataFrame(gt_results)
    inf_matrix = np.column_stack(inf_funcs_list) if inf_funcs_list else None

    # Cohort sizes for aggregation weights — use observation counts.
    cohort_sizes = pd.Series(
        {g_val: int((g_arr == g_val).sum()) for g_val in cohorts}
    )

    post_mask = detail['relative_time'] >= 0
    agg_est, agg_se, agg_pval, agg_ci = _aggregate_simple(
        detail[post_mask],
        inf_matrix[:, post_mask.values] if inf_matrix is not None else None,
        cohort_sizes, n_obs, alpha,
    )
    event_study = _aggregate_event_study(
        detail, inf_matrix, cohort_sizes, n_obs, alpha,
    )
    pretrend = _pretrend_test(detail, inf_matrix, n_obs)

    model_info: Dict[str, Any] = {
        'estimator': 'REG (RCS)' + (' + covariates' if x else ''),
        'control_group': 'nevertreated',
        'base_period': base_period,
        'anticipation': anticipation,
        'panel': False,
        'n_units': n_obs,              # treated as "n" for aggte bootstrap
        'n_obs': n_obs,
        'n_periods': len(time_periods),
        'n_cohorts': len(cohorts),
        'cohorts': cohorts,
        'event_study': event_study,
        'pretrend_test': pretrend,
        'cohort_sizes': cohort_sizes,
    }
    if covariate_info is not None:
        model_info.update(covariate_info)

    return CausalResult(
        method="Callaway and Sant'Anna (2021) — repeated cross-sections",
        estimand='ATT',
        estimate=agg_est,
        se=agg_se,
        pvalue=agg_pval,
        ci=agg_ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _influence_funcs=inf_matrix,
        _citation_key='callaway_santanna',
    )


def _estimate_single_att_rcs(
    y_arr: np.ndarray,
    g_arr: np.ndarray,
    t_arr: np.ndarray,
    g_val: int,
    t_val: int,
    base_val: int,
    n_obs: int,
) -> Tuple[float, float, np.ndarray]:
    """Observation-level 2×2 cell-mean DID + influence function."""
    m_gt = (g_arr == g_val) & (t_arr == t_val)
    m_gb = (g_arr == g_val) & (t_arr == base_val)
    m_ct = (g_arr == 0) & (t_arr == t_val)
    m_cb = (g_arr == 0) & (t_arr == base_val)

    # Any empty cell kills the estimator for this (g, t).
    for m in (m_gt, m_gb, m_ct, m_cb):
        if m.sum() < 2:
            return 0.0, np.inf, np.zeros(n_obs)

    mu_gt = y_arr[m_gt].mean()
    mu_gb = y_arr[m_gb].mean()
    mu_ct = y_arr[m_ct].mean()
    mu_cb = y_arr[m_cb].mean()

    att = float((mu_gt - mu_gb) - (mu_ct - mu_cb))

    p_gt = m_gt.sum() / n_obs
    p_gb = m_gb.sum() / n_obs
    p_ct = m_ct.sum() / n_obs
    p_cb = m_cb.sum() / n_obs

    inf = np.zeros(n_obs)
    inf[m_gt] += (y_arr[m_gt] - mu_gt) / p_gt
    inf[m_gb] += -(y_arr[m_gb] - mu_gb) / p_gb
    inf[m_ct] += -(y_arr[m_ct] - mu_ct) / p_ct
    inf[m_cb] += (y_arr[m_cb] - mu_cb) / p_cb

    # SE from sample variance of the influence function.
    se = float(np.sqrt(np.mean(inf ** 2) / n_obs))
    return att, se, inf


def _rcs_residualise_on_controls(
    y_arr: np.ndarray,
    df: pd.DataFrame,
    g_col: str,
    t_col: str,
    x_cols: List[str],
) -> np.ndarray:
    """Fit Y = Xβ + period-FE on never-treated observations; return
    Y − X'β̂ for every observation (treated + control).

    The period FEs absorb the unconditional time pattern in the control
    group, so after residualisation the remaining cross-period mean
    movement in the control cells is zero and the RCS DID reduces to a
    covariate-adjusted comparison, matching the "outcome regression"
    flavour of Sant'Anna & Zhao (2020) adapted to repeated cross-sections.
    Influence functions downstream treat β̂ as known; asymptotically
    negligible at √n.
    """
    y_arr = np.asarray(y_arr, dtype=float)
    g_arr = df[g_col].values
    t_arr = df[t_col].values
    x_mat = df[x_cols].to_numpy(dtype=float)

    # Keep only observations with finite covariates; the Y slot is
    # already clean from the upstream dropna.
    control = (g_arr == 0)
    if control.sum() < x_mat.shape[1] + 2:
        # Not enough controls to fit; return untouched Y.
        return y_arr

    # Build the control design: covariates + period dummies.
    periods = sorted(np.unique(t_arr[control]))
    X_ctrl = x_mat[control]
    t_ctrl = t_arr[control]
    period_dummies_ctrl = np.column_stack([
        (t_ctrl == p).astype(float) for p in periods
    ])
    design_ctrl = np.column_stack([X_ctrl, period_dummies_ctrl])

    y_ctrl = y_arr[control]
    try:
        beta, *_ = np.linalg.lstsq(design_ctrl, y_ctrl, rcond=None)
    except np.linalg.LinAlgError:
        return y_arr

    beta_x = beta[:x_mat.shape[1]]
    # For residualisation of every observation (including treated), we
    # only subtract the X contribution.  The period FE absorbs only the
    # control group's period mean, which is exactly what we want to
    # leave inside Y for the treated cell.
    return y_arr - x_mat @ beta_x
