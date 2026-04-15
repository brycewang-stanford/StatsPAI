"""
Modern staggered DID estimators: Wooldridge (2021), DR-DID, and TWFE decomposition.

Implements three cutting-edge methods for DID with staggered treatment adoption:

1. **wooldridge_did()** — Wooldridge (2021) extended TWFE with cohort × time interactions.
   Shows that a properly saturated TWFE regression recovers valid ATT even with
   heterogeneous treatment effects, without specialised estimators.

2. **drdid()** — Sant'Anna & Zhao (2020) doubly robust DID for 2×2 designs with
   covariates.  Combines outcome regression and inverse probability weighting,
   consistent if *either* model is correctly specified.

3. **twfe_decomposition()** — Enhanced Goodman-Bacon (2021) decomposition with
   de Chaisemartin–D'Haultfoeuille (2020) weights diagnostic.

References
----------
Wooldridge, J.M. (2021).
    "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and
     Difference-in-Differences Estimators."
    Working paper, Michigan State University.

Sant'Anna, P.H.C. and Zhao, J. (2020).
    "Doubly Robust Difference-in-Differences Estimators."
    *Journal of Econometrics*, 219(1), 101–122.

Goodman-Bacon, A. (2021).
    "Difference-in-Differences with Variation in Treatment Timing."
    *Journal of Econometrics*, 225(2), 254–277.

de Chaisemartin, C. and D'Haultfoeuille, X. (2020).
    "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects."
    *American Economic Review*, 110(9), 2964–2996.
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ═══════════════════════════════════════════════════════════════════════
#  Helper: cluster-robust OLS
# ═══════════════════════════════════════════════════════════════════════

def _ols_fit(
    X: np.ndarray,
    y: np.ndarray,
    cluster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with optional cluster-robust (CR1) standard errors.

    Returns (beta, se, vcov).
    """
    n, k = X.shape
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    if cluster is not None:
        unique_cl = np.unique(cluster)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for c in unique_cl:
            idx = cluster == c
            score = (X[idx] * resid[idx, np.newaxis]).sum(axis=0)
            meat += np.outer(score, score)
        correction = (n_cl / (n_cl - 1)) * ((n - 1) / (n - k))
        vcov = correction * XtX_inv @ meat @ XtX_inv
    else:
        # HC1 robust
        weights = (n / (n - k)) * resid ** 2
        meat = X.T @ (X * weights[:, np.newaxis])
        vcov = XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return beta, se, vcov


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════════════════
#  1. Wooldridge (2021) Extended TWFE
# ═══════════════════════════════════════════════════════════════════════

def wooldridge_did(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Wooldridge (2021) extended TWFE estimator for staggered DID.

    Estimates a properly saturated TWFE regression with cohort x post
    interactions, recovering cohort-specific ATTs and an overall
    cohort-weighted ATT that is valid even under heterogeneous treatment
    effects.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset (long format, one row per unit-period).
    y : str
        Outcome variable.
    group : str
        Unit identifier (e.g. county, individual).
    time : str
        Time period variable (integer-valued).
    first_treat : str
        Column indicating when the unit is first treated.
        Use ``np.nan`` (or 0) for never-treated units.
    controls : list of str, optional
        Time-varying covariates to include.
    cluster : str, optional
        Cluster variable for standard errors.  Defaults to *group*
        (unit-level clustering).
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        ``estimate`` is the cohort-size-weighted ATT.
        ``detail`` DataFrame contains cohort-specific ATTs.
        ``model_info`` contains event-study coefficients.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True)
    >>> result = sp.wooldridge_did(df, y='y', group='unit',
    ...                           time='period', first_treat='first_treat')
    >>> result.summary()
    """
    df = data.copy()

    # ── Normalise first_treat ────────────────────────────────────────
    ft = df[first_treat].copy()
    # Treat 0 and NaN as never-treated → sentinel
    ft = ft.replace(0, np.nan)
    df["_ft"] = ft

    periods = sorted(df[time].unique())
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())

    if len(cohorts) == 0:
        raise ValueError("No treated cohorts found. Check 'first_treat' column.")

    # ── Unit and time FE via demeaning ──────────────────────────────
    # Within-group (unit) demeaning
    df["_y"] = df[y].astype(float)
    unit_mean = df.groupby(group)["_y"].transform("mean")
    time_mean = df.groupby(time)["_y"].transform("mean")
    grand_mean = df["_y"].mean()
    df["_y_dm"] = df["_y"] - unit_mean - time_mean + grand_mean

    # ── Build cohort × post interaction dummies ─────────────────────
    interaction_cols: List[str] = []
    for g in cohorts:
        col = f"_coh{int(g)}_post"
        df[col] = ((df["_ft"] == g) & (df[time] >= g)).astype(float)
        interaction_cols.append(col)

    # ── Also build cohort × relative-time dummies for event study ───
    # H7 fix: build event dummies for BOTH leads and lags. Omit
    # rel = -1 as the reference category so the design is identified.
    # Post-only consumers can still filter via etwfe_emfx(include_leads=False).
    event_cols: List[str] = []
    rel_times = set()
    for g in cohorts:
        mask_g = df["_ft"] == g
        for t_val in periods:
            rel = int(t_val - g)
            if rel == -1:
                continue  # reference / omitted period
            col = (f"_coh{int(g)}_rel{rel}" if rel >= 0
                   else f"_coh{int(g)}_rel_neg{abs(rel)}")
            df[col] = ((mask_g) & (df[time] == t_val)).astype(float)
            event_cols.append(col)
            rel_times.add(rel)

    # ── Demean interactions (same FE projection) ────────────────────
    for col in interaction_cols + event_cols:
        u_m = df.groupby(group)[col].transform("mean")
        t_m = df.groupby(time)[col].transform("mean")
        g_m = df[col].mean()
        df[f"{col}_dm"] = df[col] - u_m - t_m + g_m

    # ── Demean controls ─────────────────────────────────────────────
    ctrl_dm_cols: List[str] = []
    if controls:
        for c in controls:
            df[f"_ctrl_{c}"] = df[c].astype(float)
            u_m = df.groupby(group)[f"_ctrl_{c}"].transform("mean")
            t_m = df.groupby(time)[f"_ctrl_{c}"].transform("mean")
            g_m = df[f"_ctrl_{c}"].mean()
            df[f"_ctrl_{c}_dm"] = df[f"_ctrl_{c}"] - u_m - t_m + g_m
            ctrl_dm_cols.append(f"_ctrl_{c}_dm")

    # ── Drop NaN rows ───────────────────────────────────────────────
    keep_cols = (
        ["_y_dm"]
        + [f"{c}_dm" for c in interaction_cols]
        + ctrl_dm_cols
    )
    valid = df[keep_cols].notna().all(axis=1)
    df_valid = df.loc[valid].reset_index(drop=True)

    # ── OLS on demeaned data ────────────────────────────────────────
    y_vec = df_valid["_y_dm"].values
    X_cols = [f"{c}_dm" for c in interaction_cols] + ctrl_dm_cols
    X = df_valid[X_cols].values

    if X.shape[1] == 0:
        raise ValueError("No cohort × post interactions could be created.")

    # Add constant (absorbed into demeaning but keep for numerical stability)
    X = np.column_stack([np.ones(len(y_vec)), X])
    col_names = ["const"] + X_cols

    cl_arr = None
    if cluster is not None:
        cl_arr = df_valid[cluster].values
    else:
        cl_arr = df_valid[group].values  # default: cluster at unit level

    beta, se, vcov = _ols_fit(X, y_vec, cluster=cl_arr)

    # ── Extract cohort-specific ATTs ────────────────────────────────
    n_obs = len(y_vec)
    df_resid = n_obs - X.shape[1]
    cohort_results = []
    cohort_sizes = []
    cohort_atts = []
    cohort_ses = []

    for i, g in enumerate(cohorts):
        idx = i + 1  # skip constant
        att_g = float(beta[idx])
        se_g = float(se[idx])
        t_g = att_g / se_g if se_g > 0 else np.nan
        p_g = float(2 * (1 - stats.t.cdf(abs(t_g), max(df_resid, 1))))
        n_g = int((df_valid["_ft"] == g).sum())
        cohort_results.append({
            "cohort": int(g),
            "att": att_g,
            "se": se_g,
            "tstat": t_g,
            "pvalue": p_g,
            "n_obs": n_g,
        })
        cohort_sizes.append(n_g)
        cohort_atts.append(att_g)
        cohort_ses.append(se_g)

    detail = pd.DataFrame(cohort_results)

    # ── Aggregate ATT (cohort-size weighted) ────────────────────────
    sizes = np.array(cohort_sizes, dtype=float)
    atts = np.array(cohort_atts)
    ses = np.array(cohort_ses)

    weights = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(len(sizes)) / len(sizes)
    att_overall = float(weights @ atts)

    # Delta-method SE for weighted average (assuming independent cohort estimates)
    # Var(sum w_g * att_g) = sum w_g^2 * Var(att_g) + cross terms from vcov
    # Use the full vcov for cohort coefficients
    cohort_vcov = vcov[1:1 + len(cohorts), 1:1 + len(cohorts)]
    att_se_overall = float(np.sqrt(weights @ cohort_vcov @ weights))

    t_overall = att_overall / att_se_overall if att_se_overall > 0 else np.nan
    p_overall = float(2 * (1 - stats.t.cdf(abs(t_overall), max(df_resid, 1))))
    t_crit = stats.t.ppf(1 - alpha / 2, max(df_resid, 1))
    ci = (att_overall - t_crit * att_se_overall, att_overall + t_crit * att_se_overall)

    # ── Event study (relative-time) coefficients ────────────────────
    # Run a separate regression with event-time dummies
    event_study_df = None
    event_vcov = None  # H1 fix: preserve full vcov for proper aggregation SE
    if len(event_cols) > 0:
        ev_X_cols = [f"{c}_dm" for c in event_cols] + ctrl_dm_cols
        ev_valid_cols = ["_y_dm"] + ev_X_cols
        ev_mask = df_valid[ev_valid_cols].notna().all(axis=1)
        if ev_mask.sum() > len(ev_X_cols) + 2:
            ev_y = df_valid.loc[ev_mask, "_y_dm"].values
            ev_X = df_valid.loc[ev_mask, ev_X_cols].values
            ev_X = np.column_stack([np.ones(len(ev_y)), ev_X])
            ev_cl = cl_arr[ev_mask.values] if cl_arr is not None else None
            ev_beta, ev_se, ev_vcov_full = _ols_fit(ev_X, ev_y, cluster=ev_cl)

            ev_rows = []
            for j, col in enumerate(event_cols):
                # Parse name: _coh{g}_rel{k}  or  _coh{g}_rel_neg{k}
                stripped = col.replace("_coh", "")
                if "_rel_neg" in stripped:
                    coh_part, rel_part = stripped.split("_rel_neg")
                    rel_val = -int(rel_part)
                else:
                    coh_part, rel_part = stripped.split("_rel")
                    rel_val = int(rel_part)
                coh_val = int(coh_part)
                idx_j = j + 1
                ev_rows.append({
                    "cohort": coh_val,
                    "rel_time": rel_val,
                    "estimate": float(ev_beta[idx_j]),
                    "se": float(ev_se[idx_j]),
                    "_vcov_idx": idx_j,
                })
            event_study_df = pd.DataFrame(ev_rows)
            # Keep only the event-study coefficient submatrix of vcov
            n_event = len(event_cols)
            event_vcov = ev_vcov_full[1:1 + n_event, 1:1 + n_event]

    # ── Model info ──────────────────────────────────────────────────
    model_info: Dict[str, Any] = {
        "n_cohorts": len(cohorts),
        "cohorts": [int(g) for g in cohorts],
        "n_periods": len(periods),
        "n_units": df[group].nunique(),
        "controls": controls or [],
        "cluster_var": cluster or group,
        "n_clusters": len(np.unique(cl_arr)) if cl_arr is not None else None,
        "cohort_weights": {int(g): float(w) for g, w in zip(cohorts, weights)},
    }
    if event_study_df is not None:
        model_info["event_study"] = event_study_df
        model_info["event_vcov"] = event_vcov  # H1: proper aggregation SE

    return CausalResult(
        method="Wooldridge (2021) Extended TWFE",
        estimand="ATT",
        estimate=att_overall,
        se=att_se_overall,
        pvalue=p_overall,
        ci=ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _citation_key="wooldridge_twfe",
    )


def etwfe(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    xvar: Optional[Any] = None,  # Union[str, List[str]]
    panel: bool = True,
    cgroup: str = "notyet",
) -> CausalResult:
    """
    Extended Two-Way Fixed Effects (ETWFE) — Wooldridge (2021).

    Explicit API matching the R package ``etwfe`` (McDermott, 2023).
    This is an alias for :func:`wooldridge_did`; both estimate the same
    saturated TWFE regression with cohort × post interactions that
    recovers valid ATT under heterogeneous treatment effects.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset (long format).
    y : str
        Outcome variable.
    group : str
        Unit identifier.
    time : str
        Time period variable.
    first_treat : str
        Column with first-treatment period; NaN or 0 for never-treated.
    controls : list of str, optional
        Time-varying covariates.
    cluster : str, optional
        Cluster variable for SE (defaults to ``group``).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        Cohort-size-weighted ATT with cohort-level detail and event-study
        coefficients in ``model_info['event_study']``.

    Notes
    -----
    Naming map to the R ``etwfe`` package:

    ============================  ========================================
    R ``etwfe`` argument          ``sp.etwfe`` argument
    ============================  ========================================
    ``fml = y ~ 1``               ``y='y'``
    ``tvar = time``               ``time='time'``
    ``gvar = first_treat``        ``first_treat='first_treat'``
    ``ivar = unit``               ``group='unit'``
    ``xvar`` (covariate het.)     ``xvar='x1'`` or ``xvar=['x1','x2']``
    ``vcov = ~cluster``           ``cluster='cluster'``
    ============================  ========================================

    For aggregated marginal effects (R ``emfx`` equivalents), combine with
    :func:`statspai.did.aggte` on a Callaway–Sant'Anna object, or inspect
    ``result.model_info['event_study']`` directly.

    References
    ----------
    Wooldridge, J.M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak
    Regression, and Difference-in-Differences Estimators."

    McDermott, G. (2023). ``etwfe``: Extended Two-Way Fixed Effects.
    https://grantmcdermott.com/etwfe/

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True)
    >>> res = sp.etwfe(df, y='y', group='unit',
    ...                time='period', first_treat='first_treat')
    >>> res.summary()

    See Also
    --------
    wooldridge_did : Identical estimator; this is a naming alias.
    callaway_santanna : CS (2021) group-time ATT estimator.
    aggte : Aggregation of group-time ATTs (event/group/calendar/simple).
    """
    # Validate cgroup
    if cgroup not in ("notyet", "nevertreated"):
        raise ValueError(
            f"cgroup must be 'notyet' or 'nevertreated'; got {cgroup!r}"
        )

    # Normalise xvar to a list (or None)
    xvar_list: Optional[List[str]] = None
    if xvar is not None:
        xvar_list = [xvar] if isinstance(xvar, str) else list(xvar)
        # C1/C2: fail fast on missing or constant xvars
        for xv in xvar_list:
            if xv not in data.columns:
                raise KeyError(f"xvar {xv!r} not found in data.columns")
            col = pd.to_numeric(data[xv], errors="coerce")
            finite = col.dropna()
            if len(finite) < 2:
                raise ValueError(
                    f"xvar {xv!r} has fewer than 2 non-NaN rows "
                    f"(found {len(finite)}); cannot estimate heterogeneity."
                )
            if float(finite.std()) < 1e-12:
                raise ValueError(
                    f"xvar {xv!r} is (near-)constant — no heterogeneity "
                    "slope can be identified. Drop it or choose another column."
                )

    # C3: explicit guard for the unimplemented combination
    if not panel and cgroup == "nevertreated":
        raise NotImplementedError(
            "cgroup='nevertreated' with panel=False is not yet supported. "
            "Use either panel=True + cgroup='nevertreated', or "
            "panel=False + cgroup='notyet'."
        )

    # Dispatch to the right implementation.
    if not panel:
        # Repeated cross-section: no unit FE, replace with cohort dummies.
        return _etwfe_repeated_cs(
            data=data, y=y, time=time, first_treat=first_treat,
            xvar=xvar_list, controls=controls, cluster=cluster,
            alpha=alpha, cgroup=cgroup,
        )
    if cgroup == "nevertreated":
        # Per-cohort regressions restricted to cohort + never-treated.
        return _etwfe_never_only(
            data=data, y=y, group=group, time=time, first_treat=first_treat,
            xvar=xvar_list, controls=controls, cluster=cluster, alpha=alpha,
        )
    if xvar_list is None:
        return wooldridge_did(
            data=data, y=y, group=group, time=time, first_treat=first_treat,
            controls=controls, cluster=cluster, alpha=alpha,
        )
    return _etwfe_with_xvar(
        data=data, y=y, group=group, time=time, first_treat=first_treat,
        xvar=xvar_list, controls=controls, cluster=cluster, alpha=alpha,
    )


def _etwfe_with_xvar(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    xvar: List[str],
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """ETWFE with covariate-moderated heterogeneity (R etwfe's ``xvar``).

    Supports single or multiple covariates. For each cohort ``g`` and
    each xvar ``x_j``, adds an interaction between the cohort × post
    dummy and ``(x_j - mean(x_j))``. The main cohort coefficient is
    ATT(g) evaluated at the sample mean of every xvar; each slope
    measures how ATT(g) shifts per unit of ``x_j``.
    """
    df = data.copy()
    ft = df[first_treat].replace(0, np.nan)
    df["_ft"] = ft

    periods = sorted(df[time].unique())
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())
    if len(cohorts) == 0:
        raise ValueError("No treated cohorts found. Check 'first_treat' column.")

    # Demean outcome
    df["_y"] = df[y].astype(float)
    unit_mean = df.groupby(group)["_y"].transform("mean")
    time_mean = df.groupby(time)["_y"].transform("mean")
    grand_mean = df["_y"].mean()
    df["_y_dm"] = df["_y"] - unit_mean - time_mean + grand_mean

    # Center every xvar by its grand mean so baseline ATT is evaluated
    # at (x1=mean, x2=mean, …).
    xc_cols: List[str] = []
    x_centers: Dict[str, float] = {}
    for x in xvar:
        raw = df[x].astype(float)
        ctr = float(raw.mean())
        x_centers[x] = ctr
        df[f"_xc_{x}"] = raw - ctr
        xc_cols.append(f"_xc_{x}")

    # Build cohort × post dummies AND cohort × post × xc_j interactions.
    base_cols: List[str] = []
    slope_cols_by_cohort: Dict[int, List[str]] = {}
    for g in cohorts:
        mask_post = (df["_ft"] == g) & (df[time] >= g)
        b = f"_coh{int(g)}_post"
        df[b] = mask_post.astype(float)
        base_cols.append(b)
        slope_cols_by_cohort[int(g)] = []
        for x in xvar:
            s = f"_coh{int(g)}_post_x_{x}"
            df[s] = df[b] * df[f"_xc_{x}"]
            slope_cols_by_cohort[int(g)].append(s)

    all_slope = [c for v in slope_cols_by_cohort.values() for c in v]
    all_inter = base_cols + all_slope

    # H4 fix: explicit name-to-index map so slope lookups never rely on
    # implicit ordering. Column 0 is the constant; columns [1:] are
    # X_cols = [f"{c}_dm" for c in all_inter] + ctrl_dm_cols in order.
    coef_index: Dict[str, int] = {"_const": 0}
    for i, col in enumerate(all_inter, start=1):
        coef_index[col] = i

    # Demean interactions via FE projection
    for col in all_inter:
        u_m = df.groupby(group)[col].transform("mean")
        t_m = df.groupby(time)[col].transform("mean")
        g_m = df[col].mean()
        df[f"{col}_dm"] = df[col] - u_m - t_m + g_m

    # Demean controls
    ctrl_dm_cols: List[str] = []
    if controls:
        for c in controls:
            df[f"_ctrl_{c}"] = df[c].astype(float)
            u_m = df.groupby(group)[f"_ctrl_{c}"].transform("mean")
            t_m = df.groupby(time)[f"_ctrl_{c}"].transform("mean")
            g_m = df[f"_ctrl_{c}"].mean()
            df[f"_ctrl_{c}_dm"] = df[f"_ctrl_{c}"] - u_m - t_m + g_m
            ctrl_dm_cols.append(f"_ctrl_{c}_dm")

    keep = ["_y_dm"] + [f"{c}_dm" for c in all_inter] + ctrl_dm_cols
    valid = df[keep].notna().all(axis=1)
    dfv = df.loc[valid].reset_index(drop=True)

    y_vec = dfv["_y_dm"].values
    X_cols = [f"{c}_dm" for c in all_inter] + ctrl_dm_cols
    X = dfv[X_cols].values
    X = np.column_stack([np.ones(len(y_vec)), X])

    cl_arr = dfv[cluster].values if cluster else dfv[group].values
    beta, se, vcov = _ols_fit(X, y_vec, cluster=cl_arr)
    n_obs = len(y_vec)
    df_resid = max(n_obs - X.shape[1], 1)

    k = len(cohorts)
    p_x = len(xvar)  # retained for the single-xvar backward-compat block

    cohort_results = []
    for g in cohorts:
        # H4 fix: look up baseline and slope indices by coefficient name
        # rather than by arithmetic position, so future column-order
        # changes cannot silently mis-attribute.
        b_idx = coef_index[f"_coh{int(g)}_post"]
        att = float(beta[b_idx])
        att_se = float(se[b_idx])
        p = float(2 * (1 - stats.t.cdf(abs(att / att_se) if att_se > 0 else 0,
                                       df_resid)))
        row: Dict[str, Any] = {
            "cohort": int(g),
            "att_at_xmean": att, "att_se": att_se, "att_pvalue": p,
        }
        for x in xvar:
            s_idx = coef_index[f"_coh{int(g)}_post_x_{x}"]
            slope = float(beta[s_idx])
            slope_se = float(se[s_idx])
            p_s = float(2 * (1 - stats.t.cdf(
                abs(slope / slope_se) if slope_se > 0 else 0, df_resid)))
            row[f"slope_{x}"] = slope
            row[f"slope_{x}_se"] = slope_se
            row[f"slope_{x}_pvalue"] = p_s
        # Backward-compat: if exactly one xvar, alias to the older name
        if p_x == 1:
            only = xvar[0]
            row["slope_wrt_x"] = row[f"slope_{only}"]
            row["slope_se"] = row[f"slope_{only}_se"]
            row["slope_pvalue"] = row[f"slope_{only}_pvalue"]
        row["n_obs"] = int((dfv["_ft"] == g).sum())
        cohort_results.append(row)

    detail = pd.DataFrame(cohort_results)
    sizes = detail["n_obs"].values.astype(float)
    weights_vec = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(k) / k
    att_overall = float(weights_vec @ detail["att_at_xmean"].values)
    # H4 fix: extract baseline-coefficient vcov by explicit index lookup
    base_idx = np.array([coef_index[f"_coh{int(g)}_post"] for g in cohorts])
    base_vcov = vcov[np.ix_(base_idx, base_idx)]
    att_se_overall = float(np.sqrt(weights_vec @ base_vcov @ weights_vec))
    t_stat = att_overall / att_se_overall if att_se_overall > 0 else np.nan
    p_overall = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid)))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (att_overall - t_crit * att_se_overall, att_overall + t_crit * att_se_overall)

    model_info = {
        "n_cohorts": k,
        "cohorts": [int(g) for g in cohorts],
        "n_periods": len(periods),
        "n_units": df[group].nunique(),
        "controls": controls or [],
        "cluster_var": cluster or group,
        "xvar": list(xvar),
        "xvar_means": x_centers,
        "heterogeneity": ("ATT(g) = baseline(g) + Σ_j slope_j(g) * "
                          "(x_j - mean(x_j))"),
    }

    x_label = ", ".join(f"{x}={x_centers[x]:.4g}" for x in xvar)
    return CausalResult(
        method="Wooldridge (2021) ETWFE with covariate heterogeneity",
        estimand=f"ATT at [{x_label}] (sample means)",
        estimate=att_overall,
        se=att_se_overall,
        pvalue=p_overall,
        ci=ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _citation_key="wooldridge_twfe",
    )


# ═══════════════════════════════════════════════════════════════════════
#  1a. ETWFE — repeated cross-section (ivar=NULL in R etwfe)
# ═══════════════════════════════════════════════════════════════════════

def _etwfe_repeated_cs(
    data: pd.DataFrame,
    y: str,
    time: str,
    first_treat: str,
    xvar: Optional[List[str]] = None,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    cgroup: str = "notyet",
) -> CausalResult:
    """ETWFE for repeated cross-sections (no unit fixed effects).

    Replaces unit FE with cohort (first-treatment) dummies plus time
    dummies. Matches R ``etwfe(ivar=NULL)`` semantics.
    """
    df = data.copy()
    ft = df[first_treat].replace(0, np.nan)
    df["_ft"] = ft

    periods = sorted(df[time].unique())
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())
    if len(cohorts) == 0:
        raise ValueError("No treated cohorts found. Check 'first_treat' column.")

    # cgroup='nevertreated' is guarded at the dispatcher level —
    # combining it with panel=False is currently a NotImplementedError.

    df["_y"] = df[y].astype(float)

    # Cohort dummies (leave never-treated as baseline), time dummies
    coh_dummies: List[str] = []
    for g in cohorts:
        col = f"_CG_{int(g)}"
        df[col] = (df["_ft"] == g).astype(float)
        coh_dummies.append(col)
    time_dummies: List[str] = []
    for tt in periods[1:]:  # first period as baseline
        col = f"_T_{int(tt)}"
        df[col] = (df[time] == tt).astype(float)
        time_dummies.append(col)

    # Cohort × post interactions
    base_cols: List[str] = []
    for g in cohorts:
        col = f"_coh{int(g)}_post"
        df[col] = ((df["_ft"] == g) & (df[time] >= g)).astype(float)
        base_cols.append(col)

    # Optional xvar slopes
    slope_cols: List[str] = []
    x_centers: Dict[str, float] = {}
    xvar = xvar or []
    for x in xvar:
        raw = df[x].astype(float)
        ctr = float(raw.mean())
        x_centers[x] = ctr
        df[f"_xc_{x}"] = raw - ctr
    for g in cohorts:
        for x in xvar:
            col = f"_coh{int(g)}_post_x_{x}"
            df[col] = df[f"_coh{int(g)}_post"] * df[f"_xc_{x}"]
            slope_cols.append(col)

    ctrl_cols: List[str] = []
    if controls:
        for c in controls:
            df[f"_ctrl_{c}"] = df[c].astype(float)
            ctrl_cols.append(f"_ctrl_{c}")

    design_cols = coh_dummies + time_dummies + base_cols + slope_cols + ctrl_cols
    keep = ["_y"] + design_cols
    valid = df[keep].notna().all(axis=1)
    dfv = df.loc[valid].reset_index(drop=True)

    y_vec = dfv["_y"].values
    X = np.column_stack([np.ones(len(y_vec))] + [dfv[c].values for c in design_cols])
    cl_arr = dfv[cluster].values if cluster else None

    # H6 fix: repeated-CS design is more collinearity-prone than the
    # within-demeaned panel path. Warn the user loudly when the design
    # matrix is rank-deficient; results are still produced via pinv.
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        import warnings as _warnings
        deficit = X.shape[1] - rank
        _warnings.warn(
            f"etwfe(panel=False): design matrix is rank-deficient by "
            f"{deficit} column(s) (rank={rank}, ncol={X.shape[1]}). "
            "Falling back to pseudoinverse; some coefficients are "
            "arbitrary linear combinations. Consider dropping collinear "
            "controls, shortening the event window, or using panel=True.",
            RuntimeWarning, stacklevel=3,
        )

    beta, se, vcov = _ols_fit(X, y_vec, cluster=cl_arr)
    n_obs = len(y_vec)
    df_resid = max(n_obs - X.shape[1], 1)

    # Extract ATT(g) coefficients — their index in X:
    # 0: const, [cohort dummies], [time dummies], [base cols], [slopes], [ctrl]
    base_start = 1 + len(coh_dummies) + len(time_dummies)
    k = len(cohorts)
    cohort_rows = []
    for i, g in enumerate(cohorts):
        idx = base_start + i
        att = float(beta[idx])
        s_ = float(se[idx])
        p = float(2 * (1 - stats.t.cdf(abs(att / s_) if s_ > 0 else 0, df_resid)))
        n_g = int((dfv["_ft"] == g).sum())
        cohort_rows.append({"cohort": int(g), "att": att, "se": s_,
                            "tstat": att / s_ if s_ > 0 else np.nan,
                            "pvalue": p, "n_obs": n_g})
    detail = pd.DataFrame(cohort_rows)
    sizes = detail["n_obs"].values.astype(float)
    w = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(k) / k
    att_overall = float(w @ detail["att"].values)
    base_vcov = vcov[base_start:base_start + k, base_start:base_start + k]
    att_se = float(np.sqrt(w @ base_vcov @ w))
    t_stat = att_overall / att_se if att_se > 0 else np.nan
    p_overall = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid)))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (att_overall - t_crit * att_se, att_overall + t_crit * att_se)

    return CausalResult(
        method="Wooldridge (2021) ETWFE — repeated cross-section",
        estimand="Overall ATT (no unit FE)",
        estimate=att_overall, se=att_se, pvalue=p_overall, ci=ci,
        alpha=alpha, n_obs=n_obs, detail=detail,
        model_info={
            "n_cohorts": k, "cohorts": [int(g) for g in cohorts],
            "n_periods": len(periods), "panel": False,
            "controls": controls or [],
            "xvar": list(xvar), "xvar_means": x_centers,
            "cgroup": cgroup,
        },
        _citation_key="wooldridge_twfe",
    )


# ═══════════════════════════════════════════════════════════════════════
#  1b. ETWFE — never-treated-only control (cgroup='nevertreated')
# ═══════════════════════════════════════════════════════════════════════

def _etwfe_never_only(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    xvar: Optional[List[str]] = None,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """ETWFE where each cohort is identified against never-treated only.

    Runs a separate ETWFE regression per cohort, each using only
    (units in cohort g) ∪ (never-treated units). Combines cohort ATTs
    with cohort-size weighting. Matches R ``etwfe(cgroup='never')``.

    Notes
    -----
    Per-cohort regressions each run on a different subset (cohort g +
    never-treated), so the cluster small-sample correction
    ``(n_cl/(n_cl-1))`` is evaluated cohort-by-cohort. Cohort SEs may
    therefore be slightly larger than a single full-sample regression
    would produce. The aggregated SE assumes cross-cohort independence,
    which is exact under this per-cohort design.
    """
    # H3 fix: compute the first-treat series locally rather than
    # writing a helper column back to the outer frame. Prevents
    # accidental column leakage when callers re-use `data`.
    df = data.copy()
    ft_local = df[first_treat].replace(0, np.nan)
    cohorts = sorted(ft_local.dropna().unique())
    if len(cohorts) == 0:
        raise ValueError("No treated cohorts found. Check 'first_treat' column.")
    never_ids = df.loc[ft_local.isna(), group].unique()
    if len(never_ids) == 0:
        raise ValueError(
            "cgroup='nevertreated' requires at least one never-treated "
            "unit (first_treat NaN / 0), but none were found."
        )

    rows: List[Dict[str, Any]] = []
    ses: List[float] = []
    for g in cohorts:
        coh_ids = df.loc[ft_local == g, group].unique()
        keep = np.concatenate([coh_ids, never_ids])
        sub = df.loc[df[group].isin(keep)].copy()
        if xvar:
            r = _etwfe_with_xvar(
                sub, y=y, group=group, time=time, first_treat=first_treat,
                xvar=xvar, controls=controls, cluster=cluster, alpha=alpha,
            )
        else:
            r = wooldridge_did(
                sub, y=y, group=group, time=time, first_treat=first_treat,
                controls=controls, cluster=cluster, alpha=alpha,
            )
        rows.append({
            "cohort": int(g),
            "att": float(r.estimate),
            "se": float(r.se),
            "pvalue": float(r.pvalue) if r.pvalue is not None else np.nan,
            "n_obs": int(r.n_obs),
        })
        ses.append(float(r.se))

    detail = pd.DataFrame(rows)
    sizes = detail["n_obs"].values.astype(float)
    w = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(len(rows)) / len(rows)
    att_overall = float(w @ detail["att"].values)
    # SE of a weighted sum of independent estimates (conservative — assumes
    # independent per-cohort regressions, which is exactly what we ran).
    att_se = float(np.sqrt(np.sum((w * np.array(ses)) ** 2)))
    t_stat = att_overall / att_se if att_se > 0 else np.nan
    df_resid = max(int(detail["n_obs"].sum()) - len(cohorts), 1)
    p_overall = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid)))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (att_overall - t_crit * att_se, att_overall + t_crit * att_se)

    return CausalResult(
        method="Wooldridge (2021) ETWFE — never-treated control",
        estimand="Cohort-size-weighted ATT",
        estimate=att_overall, se=att_se, pvalue=p_overall, ci=ci,
        alpha=alpha, n_obs=int(detail["n_obs"].sum()), detail=detail,
        model_info={
            "n_cohorts": len(cohorts),
            "cohorts": [int(g) for g in cohorts],
            "cgroup": "nevertreated",
            "controls": controls or [],
            "xvar": list(xvar) if xvar else None,
        },
        _citation_key="wooldridge_twfe",
    )


# ═══════════════════════════════════════════════════════════════════════
#  2. Doubly Robust DID — Sant'Anna & Zhao (2020)
# ═══════════════════════════════════════════════════════════════════════

def drdid(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    covariates: Optional[List[str]] = None,
    method: str = "imp",
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: Optional[int] = None,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Doubly Robust Difference-in-Differences (Sant'Anna & Zhao 2020).

    Combines outcome regression with inverse probability weighting for
    2×2 DID with covariates.  Consistent if *either* the outcome model
    *or* the propensity score model is correctly specified.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with one row per unit-period in 2×2 design.
    y : str
        Outcome variable.
    group : str
        Binary treatment-group indicator (1 = treated, 0 = control).
    time : str
        Binary time indicator (1 = post, 0 = pre).
    covariates : list of str, optional
        Covariate names.  If ``None``, runs a simple (un-adjusted) DID.
    method : str, default ``'imp'``
        ``'imp'`` for the improved estimator (locally efficient);
        ``'trad'`` for the traditional DR-DID.
    alpha : float, default 0.05
        Significance level.
    n_boot : int, default 500
        Number of bootstrap replications for inference.
    random_state : int, optional
        Seed for bootstrap reproducibility.

    Returns
    -------
    CausalResult
        ``estimate`` is the DR-DID ATT.
        ``detail`` contains influence-function diagnostics.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> G = rng.integers(0, 2, n)
    >>> T = rng.integers(0, 2, n)
    >>> x = rng.normal(0, 1, n)
    >>> y_val = 1 + 0.5*x + 2*G + 3*T + 4*G*T + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({'y': y_val, 'treated': G, 'post': T, 'x': x})
    >>> result = sp.drdid(df, y='y', group='treated', time='post',
    ...                   covariates=['x'])
    >>> abs(result.estimate - 4.0) < 1.0
    True
    """
    df = data.copy()
    rng = np.random.default_rng(random_state if random_state is not None else seed)

    # ── Validate 2×2 design ─────────────────────────────────────────
    g_vals = sorted(df[group].dropna().unique())
    t_vals = sorted(df[time].dropna().unique())
    if len(g_vals) != 2:
        raise ValueError(f"'{group}' must be binary, got values: {g_vals}")
    if len(t_vals) != 2:
        raise ValueError(f"'{time}' must be binary, got values: {t_vals}")

    G = (df[group] == g_vals[1]).astype(float).values
    T = (df[time] == t_vals[1]).astype(float).values
    Y = df[y].astype(float).values

    # Covariates
    if covariates and len(covariates) > 0:
        X = df[covariates].values.astype(float)
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
    else:
        X = np.ones((len(Y), 1))

    # Drop NaN rows
    valid = np.isfinite(Y)
    for j in range(X.shape[1]):
        valid &= np.isfinite(X[:, j])
    G, T, Y, X = G[valid], T[valid], Y[valid], X[valid]
    n = len(Y)

    def _estimate_att(G_b, T_b, Y_b, X_b):
        """Core DR-DID estimator for one sample."""
        n_b = len(Y_b)

        # Share treated
        p_hat = G_b.mean()
        if p_hat <= 0 or p_hat >= 1:
            return np.nan

        # ── Propensity score: P(G=1 | X) via logistic regression ────
        # Use IRLS for logistic regression (no sklearn dependency)
        ps = _logistic_fit(X_b, G_b)
        ps = np.clip(ps, 1e-6, 1 - 1e-6)

        # ── Outcome regression for controls: E[DeltaY | X, G=0] ────
        # Compute DeltaY for each unit that appears in both periods
        # In repeated cross-section / 2×2 stacked data, compute change
        # We treat the data as pooled; for controls in post vs pre:
        ctrl_post = (G_b == 0) & (T_b == 1)
        ctrl_pre = (G_b == 0) & (T_b == 0)

        # For the outcome model, regress Y on X separately for
        # control-post and control-pre
        if ctrl_post.sum() < X_b.shape[1] or ctrl_pre.sum() < X_b.shape[1]:
            # Not enough data; fall back to simple DID
            return (
                Y_b[(G_b == 1) & (T_b == 1)].mean()
                - Y_b[(G_b == 1) & (T_b == 0)].mean()
                - Y_b[(G_b == 0) & (T_b == 1)].mean()
                + Y_b[(G_b == 0) & (T_b == 0)].mean()
            )

        # OLS for E[Y|X, G=0, T=1]
        try:
            beta_post = np.linalg.lstsq(X_b[ctrl_post], Y_b[ctrl_post], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_post = np.linalg.pinv(X_b[ctrl_post]) @ Y_b[ctrl_post]

        # OLS for E[Y|X, G=0, T=0]
        try:
            beta_pre = np.linalg.lstsq(X_b[ctrl_pre], Y_b[ctrl_pre], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_pre = np.linalg.pinv(X_b[ctrl_pre]) @ Y_b[ctrl_pre]

        m1_x = X_b @ beta_post  # predicted E[Y|X, G=0, T=1]
        m0_x = X_b @ beta_pre   # predicted E[Y|X, G=0, T=0]
        delta_m = m1_x - m0_x   # predicted DeltaY for controls

        # ── DR-DID estimator ────────────────────────────────────────
        if method == "imp":
            # Improved (locally efficient) DR-DID
            # Weight construction
            w_treat_post = G_b * T_b
            w_treat_pre = G_b * (1 - T_b)
            w_ctrl_post = ps / (1 - ps) * (1 - G_b) * T_b
            w_ctrl_pre = ps / (1 - ps) * (1 - G_b) * (1 - T_b)

            # Normalise weights
            eta_1 = w_treat_post.mean()
            eta_0 = w_treat_pre.mean()
            gamma_1 = w_ctrl_post.mean()
            gamma_0 = w_ctrl_pre.mean()

            if eta_1 == 0 or eta_0 == 0:
                return np.nan

            att = (
                (w_treat_post * (Y_b - m1_x)).sum() / (w_treat_post.sum() + 1e-10)
                - (w_treat_pre * (Y_b - m0_x)).sum() / (w_treat_pre.sum() + 1e-10)
                - (w_ctrl_post * (Y_b - m1_x)).sum() / (w_ctrl_post.sum() + 1e-10)
                + (w_ctrl_pre * (Y_b - m0_x)).sum() / (w_ctrl_pre.sum() + 1e-10)
            )
        else:
            # Traditional DR-DID
            w1 = G_b / p_hat
            w0 = ps * (1 - G_b) / ((1 - ps) * p_hat)

            att_1 = (w1 * T_b * (Y_b - m1_x)).sum() / n_b
            att_0 = (w1 * (1 - T_b) * (Y_b - m0_x)).sum() / n_b
            ctrl_1 = (w0 * T_b * (Y_b - m1_x)).sum() / n_b
            ctrl_0 = (w0 * (1 - T_b) * (Y_b - m0_x)).sum() / n_b

            att = (att_1 - att_0) - (ctrl_1 - ctrl_0)

        return att

    # ── Point estimate ──────────────────────────────────────────────
    att_hat = _estimate_att(G, T, Y, X)

    # ── Bootstrap SE ────────────────────────────────────────────────
    boot_atts = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_atts[b] = _estimate_att(G[idx], T[idx], Y[idx], X[idx])

    boot_valid = boot_atts[np.isfinite(boot_atts)]
    att_se = float(np.std(boot_valid, ddof=1)) if len(boot_valid) > 1 else np.nan

    t_stat = att_hat / att_se if att_se > 0 else np.nan
    pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att_hat - z_crit * att_se, att_hat + z_crit * att_se)

    # ── Detail DataFrame ────────────────────────────────────────────
    detail = pd.DataFrame({
        "statistic": ["ATT", "SE (bootstrap)", "z-stat", "p-value",
                       "CI lower", "CI upper", "N boot valid"],
        "value": [att_hat, att_se, t_stat, pvalue, ci[0], ci[1], len(boot_valid)],
    })

    # ── Diagnostics ─────────────────────────────────────────────────
    ps_full = _logistic_fit(X, G)
    n_treated = int(G.sum())
    n_control = int((1 - G).sum())

    model_info: Dict[str, Any] = {
        "method": "improved" if method == "imp" else "traditional",
        "n_treated": n_treated,
        "n_control": n_control,
        "n_post": int(T.sum()),
        "n_pre": int((1 - T).sum()),
        "ps_mean_treated": float(ps_full[G == 1].mean()),
        "ps_mean_control": float(ps_full[G == 0].mean()),
        "n_boot": n_boot,
        "n_boot_valid": len(boot_valid),
        "covariates": covariates or [],
    }

    method_label = "Improved" if method == "imp" else "Traditional"
    return CausalResult(
        method=f"Doubly Robust DID ({method_label}, Sant'Anna & Zhao 2020)",
        estimand="ATT",
        estimate=att_hat,
        se=att_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="drdid",
    )


def _logistic_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Fit logistic regression via IRLS, return predicted probabilities."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = X @ beta
        z = np.clip(z, -20, 20)
        mu = 1.0 / (1.0 + np.exp(-z))
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        w = mu * (1 - mu)
        Xw = X * w[:, np.newaxis]
        try:
            H = np.linalg.inv(Xw.T @ X)
        except np.linalg.LinAlgError:
            H = np.linalg.pinv(Xw.T @ X)
        grad = X.T @ (y - mu)
        delta = H @ grad
        beta += delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    z = X @ beta
    z = np.clip(z, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


# ═══════════════════════════════════════════════════════════════════════
#  3. Enhanced TWFE Decomposition (Bacon + dCDH weights)
# ═══════════════════════════════════════════════════════════════════════

def twfe_decomposition(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    alpha: float = 0.05,
) -> CausalResult:
    """
    TWFE decomposition: Goodman-Bacon (2021) + de Chaisemartin–D'Haultfoeuille weights.

    Decomposes the standard two-way fixed effects estimator into all
    pairwise 2×2 DID comparisons, showing the weight and estimate for
    each.  Also computes de Chaisemartin–D'Haultfoeuille (2020) weights
    to diagnose whether *negative weights* are present.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    y : str
        Outcome variable.
    group : str
        Unit identifier.
    time : str
        Time period variable.
    first_treat : str
        Treatment timing column (NaN or 0 for never-treated).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``detail`` DataFrame has columns: ``type``, ``treated_cohort``,
        ``control_cohort``, ``estimate``, ``weight``, ``weighted_est``.
        ``model_info`` includes summary statistics and dCDH weights.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=8, staggered=True)
    >>> result = sp.twfe_decomposition(df, y='y', group='unit',
    ...                                time='period',
    ...                                first_treat='first_treat')
    >>> result.summary()
    """
    df = data.copy()

    ft = df[first_treat].copy()
    ft = ft.replace(0, np.nan)
    df["_ft"] = ft

    periods = sorted(df[time].unique())
    n_periods = len(periods)
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())
    has_never = df["_ft"].isna().any()

    # ── Standard TWFE estimate ──────────────────────────────────────
    # Unit and time demeaning, then regress on treatment indicator
    df["_treated"] = ((df["_ft"].notna()) & (df[time] >= df["_ft"])).astype(float)
    df["_y"] = df[y].astype(float)

    u_m = df.groupby(group)["_y"].transform("mean")
    t_m = df.groupby(time)["_y"].transform("mean")
    g_m = df["_y"].mean()
    y_dm = (df["_y"] - u_m - t_m + g_m).values

    u_m_d = df.groupby(group)["_treated"].transform("mean")
    t_m_d = df.groupby(time)["_treated"].transform("mean")
    g_m_d = df["_treated"].mean()
    d_dm = (df["_treated"] - u_m_d - t_m_d + g_m_d).values

    denom = d_dm @ d_dm
    twfe_beta = float(d_dm @ y_dm / denom) if denom > 0 else np.nan

    # ── Bacon decomposition ─────────────────────────────────────────
    # Enumerate all 2×2 comparisons
    comparisons: List[Dict[str, Any]] = []

    def _did_2x2_simple(df_sub, unit_col, time_col, y_col, g1_units, g2_units):
        """Simple 2x2 DID between two groups over their overlapping periods."""
        sub = df_sub[df_sub[unit_col].isin(set(g1_units) | set(g2_units))].copy()
        if len(sub) == 0:
            return np.nan, 0.0
        treat_mask = sub[unit_col].isin(set(g1_units))
        sub["_g"] = treat_mask.astype(float)
        # post = time >= treatment time of g1
        g1_ft = sub.loc[treat_mask, "_ft"].iloc[0] if treat_mask.any() else np.nan
        if np.isnan(g1_ft):
            return np.nan, 0.0
        sub["_post"] = (sub[time_col] >= g1_ft).astype(float)
        # Simple 2x2 DID
        yt = sub.groupby(["_g", "_post"])[y_col].mean()
        try:
            est = (yt[(1.0, 1.0)] - yt[(1.0, 0.0)]) - (yt[(0.0, 1.0)] - yt[(0.0, 0.0)])
        except KeyError:
            return np.nan, 0.0
        n_comp = len(sub[unit_col].unique())
        return float(est), n_comp

    # Type 1: Earlier vs Later treated
    for i, g_early in enumerate(cohorts):
        for g_late in cohorts[i + 1:]:
            early_units = df.loc[df["_ft"] == g_early, group].unique()
            late_units = df.loc[df["_ft"] == g_late, group].unique()
            est, n_comp = _did_2x2_simple(df, group, time, "_y", early_units, late_units)
            if not np.isnan(est):
                comparisons.append({
                    "type": "Earlier vs Later",
                    "treated_cohort": int(g_early),
                    "control_cohort": int(g_late),
                    "estimate": est,
                    "n_units": n_comp,
                })

    # Type 2: Later vs Earlier (forbidden — uses already-treated as control)
    for i, g_late in enumerate(cohorts):
        for g_early in cohorts[:i]:
            late_units = df.loc[df["_ft"] == g_late, group].unique()
            early_units = df.loc[df["_ft"] == g_early, group].unique()
            est, n_comp = _did_2x2_simple(df, group, time, "_y", late_units, early_units)
            if not np.isnan(est):
                comparisons.append({
                    "type": "Later vs Earlier",
                    "treated_cohort": int(g_late),
                    "control_cohort": int(g_early),
                    "estimate": est,
                    "n_units": n_comp,
                })

    # Type 3: Treated vs Never-treated
    if has_never:
        never_units = df.loc[df["_ft"].isna(), group].unique()
        for g in cohorts:
            g_units = df.loc[df["_ft"] == g, group].unique()
            est, n_comp = _did_2x2_simple(df, group, time, "_y", g_units, never_units)
            if not np.isnan(est):
                comparisons.append({
                    "type": "Treated vs Never",
                    "treated_cohort": int(g),
                    "control_cohort": "Never",
                    "estimate": est,
                    "n_units": n_comp,
                })

    if len(comparisons) == 0:
        raise ValueError("No valid 2×2 comparisons found. Check data structure.")

    comp_df = pd.DataFrame(comparisons)

    # Compute weights proportional to n_units × variance-of-treatment
    # Simplified: proportional to n_units (sample share)
    total_n = comp_df["n_units"].sum()
    comp_df["weight"] = comp_df["n_units"] / total_n
    # Re-normalise to sum to 1
    comp_df["weight"] = comp_df["weight"] / comp_df["weight"].sum()
    comp_df["weighted_est"] = comp_df["weight"] * comp_df["estimate"]

    # ── de Chaisemartin–D'Haultfoeuille weights ─────────────────────
    # Compute weights on each (g, t) cell in the TWFE regression.
    # dCDH show: beta_TWFE = sum_{g,t} w_{g,t} * ATT_{g,t}
    # where some w_{g,t} can be NEGATIVE.
    dcdh_rows: List[Dict[str, Any]] = []
    n_total = len(df)
    for g in cohorts:
        g_mask = df["_ft"] == g
        n_g = g_mask.sum()
        for t_val in periods:
            if t_val < g:
                continue  # only post-treatment cells
            t_mask = df[time] == t_val
            n_gt = (g_mask & t_mask).sum()
            if n_gt == 0:
                continue
            # Variance of treatment status in period t
            d_t = df.loc[t_mask, "_treated"].values
            var_d_t = np.var(d_t, ddof=0)
            if var_d_t == 0:
                continue
            # dCDH weight ∝ (n_gt / n_total) * (E[D|t] - E[D|g,t]) / Var(D|t)
            # Simplified formula
            e_d_t = d_t.mean()
            e_d_gt = df.loc[g_mask & t_mask, "_treated"].mean()
            w_gt = (n_gt / n_total) * (e_d_gt - e_d_t) / var_d_t
            dcdh_rows.append({
                "cohort": int(g),
                "period": int(t_val) if isinstance(t_val, (int, np.integer)) else t_val,
                "dcdh_weight": float(w_gt),
                "n_cell": int(n_gt),
            })

    dcdh_df = pd.DataFrame(dcdh_rows) if dcdh_rows else pd.DataFrame()

    n_negative = int((comp_df["weight"] < -1e-10).sum())
    bacon_att = float(comp_df["weighted_est"].sum())
    n_negative_dcdh = int((dcdh_df["dcdh_weight"] < -1e-10).sum()) if len(dcdh_df) > 0 else 0

    model_info: Dict[str, Any] = {
        "twfe_beta": twfe_beta,
        "bacon_att": bacon_att,
        "n_comparisons": len(comp_df),
        "n_negative_weights_bacon": n_negative,
        "n_negative_weights_dcdh": n_negative_dcdh,
        "n_cohorts": len(cohorts),
        "cohorts": [int(g) for g in cohorts],
        "has_never_treated": bool(has_never),
        "n_units": df[group].nunique(),
        "n_periods": n_periods,
    }
    if len(dcdh_df) > 0:
        model_info["dcdh_weights"] = dcdh_df

    # SE via simple approach: variation across comparisons
    if len(comp_df) > 1:
        att_se = float(np.sqrt(
            (comp_df["weight"] ** 2 * (comp_df["estimate"] - bacon_att) ** 2).sum()
        ))
    else:
        att_se = 0.0

    pvalue = float(2 * (1 - stats.norm.cdf(abs(bacon_att / att_se)))) if att_se > 0 else np.nan
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (bacon_att - z_crit * att_se, bacon_att + z_crit * att_se) if att_se > 0 else (np.nan, np.nan)

    return CausalResult(
        method="TWFE Decomposition (Bacon 2021 + dCDH 2020)",
        estimand="ATT (TWFE composite)",
        estimate=bacon_att,
        se=att_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(df),
        detail=comp_df,
        model_info=model_info,
        _citation_key="twfe_decomposition",
    )


# ═══════════════════════════════════════════════════════════════════════
#  4. etwfe_emfx — R etwfe-style marginal-effects aggregations
# ═══════════════════════════════════════════════════════════════════════

def etwfe_emfx(
    result: CausalResult,
    type: str = "simple",
    alpha: float = 0.05,
    include_leads: bool = False,
) -> CausalResult:
    """
    R ``etwfe::emfx``-style aggregated marginal effects for an ETWFE fit.

    Takes the result of :func:`etwfe` / :func:`wooldridge_did` and returns
    one of four aggregations used in applied work:

    ================  ========================================================
    ``type``          Aggregation
    ================  ========================================================
    ``'simple'``      Overall cohort-size-weighted ATT (same as ``result.estimate``).
    ``'group'``       ATT per treatment cohort ``g``.
    ``'event'``       ATT per event time ``e = t - g``, averaged across cohorts.
    ``'calendar'``    ATT per calendar time ``t``, averaged across cohorts for
                      which ``t >= g``.
    ================  ========================================================

    Parameters
    ----------
    result : CausalResult
        Output of :func:`etwfe` or :func:`wooldridge_did`.
    type : {'simple', 'group', 'event', 'calendar'}, default 'simple'
        Aggregation type.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    include_leads : bool, default False
        For ``type='event'`` and ``type='calendar'``, whether to include
        pre-treatment relative times (``rel_time < 0``) in the output.
        These coefficients identify pre-trends and are informative for
        parallel-trends inspection. Default ``False`` for backward
        compatibility with earlier versions; set ``True`` for full
        event-study output matching the R ``etwfe::emfx(type='event')``
        default. ``rel_time = -1`` is always the reference category
        and is excluded.

    Returns
    -------
    CausalResult
        ``estimate`` is the overall ATT (for ``type='simple'``) or the
        mean of the sub-aggregation (for the other types). ``detail``
        contains one row per group/event-time/calendar-time with
        (estimate, se, pvalue, ci_low, ci_high).

    Notes
    -----
    For ``'event'`` and ``'calendar'``, the reported SE treats the
    per-cohort coefficients as independent — a standard approximation
    that matches R etwfe's default under classical vcov. Cluster-robust
    or fully-general SEs require the full regression vcov, which can
    be requested via ``sp.wooldridge_did`` + the ``model_info`` matrix
    in a future release.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True)
    >>> fit = sp.etwfe(df, y='y', time='time',
    ...                first_treat='first_treat', group='unit')
    >>> evt = sp.etwfe_emfx(fit, type='event')
    >>> print(evt.detail)   # ATT by event time
    >>> grp = sp.etwfe_emfx(fit, type='group')
    >>> cal = sp.etwfe_emfx(fit, type='calendar')
    """
    valid = {"simple", "group", "event", "calendar"}
    if type not in valid:
        raise ValueError(f"type must be one of {sorted(valid)}; got {type!r}")

    if not isinstance(result.model_info, dict) or "cohorts" not in result.model_info:
        raise ValueError(
            "etwfe_emfx requires a result produced by sp.etwfe / "
            "sp.wooldridge_did — missing 'cohorts' in model_info."
        )

    mi = result.model_info
    cohorts = mi["cohorts"]
    cohort_weights = mi.get("cohort_weights", {})
    event_study = mi.get("event_study")

    # ── simple ──
    if type == "simple":
        # Just return a tidy one-row result
        df_resid = max(result.n_obs - len(cohorts), 1)
        t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
        est = float(result.estimate)
        se = float(result.se)
        ci = (est - t_crit * se, est + t_crit * se)
        detail = pd.DataFrame([{
            "aggregation": "simple",
            "estimate": est, "se": se,
            "pvalue": float(result.pvalue) if result.pvalue is not None else np.nan,
            "ci_low": ci[0], "ci_high": ci[1],
            "n_cohorts": len(cohorts),
        }])
        return CausalResult(
            method="ETWFE — simple aggregation (overall ATT)",
            estimand="Overall ATT",
            estimate=est, se=se,
            pvalue=float(result.pvalue) if result.pvalue is not None else np.nan,
            ci=ci, alpha=alpha, n_obs=int(result.n_obs),
            detail=detail,
            model_info={"type": "simple", "source_method": result.method},
            _citation_key="wooldridge_twfe",
        )

    # ── group ──
    if type == "group":
        det = result.detail.copy()
        df_resid = max(result.n_obs - len(cohorts), 1)
        t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
        if "att_at_xmean" in det.columns:
            est_col = "att_at_xmean"; se_col = "att_se"; p_col = "att_pvalue"
        else:
            est_col = "att"; se_col = "se"; p_col = "pvalue"
        rows = []
        for _, r in det.iterrows():
            est = float(r[est_col]); se = float(r[se_col])
            rows.append({
                "cohort": int(r["cohort"]),
                "estimate": est, "se": se,
                "pvalue": float(r[p_col]) if p_col in r.index else np.nan,
                "ci_low": est - t_crit * se,
                "ci_high": est + t_crit * se,
                "n_obs": int(r["n_obs"]),
            })
        out_det = pd.DataFrame(rows)
        # H2 fix: the group aggregation's headline == simple overall ATT.
        headline_est = float(result.estimate)
        headline_se = float(result.se)
        t_head = headline_est / headline_se if headline_se > 0 else np.nan
        p_head = float(2 * (1 - stats.t.cdf(abs(t_head), df_resid))) \
            if not np.isnan(t_head) else np.nan
        ci_head = (headline_est - t_crit * headline_se,
                   headline_est + t_crit * headline_se)
        return CausalResult(
            method="ETWFE — group aggregation (ATT per cohort)",
            estimand="ATT(g) per cohort",
            estimate=headline_est, se=headline_se,
            pvalue=p_head, ci=ci_head, alpha=alpha,
            n_obs=int(result.n_obs), detail=out_det,
            model_info={"type": "group", "source_method": result.method},
            _citation_key="wooldridge_twfe",
        )

    # ── event / calendar ──
    if event_study is None or len(event_study) == 0:
        raise ValueError(
            "type='event'/'calendar' requires event_study coefficients "
            "in result.model_info['event_study']."
        )
    es = event_study.copy()
    det = result.detail
    weight_by_cohort: Dict[int, float] = {}
    if "n_obs" in det.columns:
        weight_by_cohort = dict(zip(det["cohort"].astype(int),
                                    det["n_obs"].astype(float)))

    df_resid = max(result.n_obs - len(cohorts), 1)
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)

    # H7 fix: default still post-only to preserve prior behaviour, but
    # pre-treatment leads are available via include_leads=True for
    # pre-trend inspection (standard event-study practice).
    if not include_leads:
        es = es.loc[es["rel_time"] >= 0].copy()

    if type == "event":
        key_col = "rel_time"; label_col = "event_time"
    else:
        es["calendar_time"] = es["cohort"].astype(int) + es["rel_time"].astype(int)
        key_col = "calendar_time"; label_col = "calendar_time"

    # H1 fix: use the stored event-study vcov when available so SE is correct
    event_vcov = mi.get("event_vcov")
    has_vcov = (event_vcov is not None and "_vcov_idx" in es.columns)
    se_method = "vcov-based (delta method)" if has_vcov else \
        "independent-coefficient approximation (fallback — vcov unavailable)"

    rows = []
    for k, sub in es.groupby(key_col):
        w_raw = np.array([weight_by_cohort.get(int(c), 1.0)
                          for c in sub["cohort"].values], dtype=float)
        w = w_raw / w_raw.sum() if w_raw.sum() > 0 else \
            np.ones(len(w_raw)) / len(w_raw)
        est = float(np.sum(w * sub["estimate"].values))
        if has_vcov:
            # Build a weight vector over the full event-coefficient space
            # and compute sqrt(w' V w) using the right submatrix.
            idx = sub["_vcov_idx"].astype(int).values - 1  # 0-indexed in event_vcov
            V_sub = event_vcov[np.ix_(idx, idx)]
            se = float(np.sqrt(max(w @ V_sub @ w, 0.0)))
        else:
            se = float(np.sqrt(np.sum((w * sub["se"].values) ** 2)))
        t_stat = est / se if se > 0 else np.nan
        p = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid))) \
            if not np.isnan(t_stat) else np.nan
        rows.append({
            label_col: int(k),
            "estimate": est, "se": se, "pvalue": p,
            "ci_low": est - t_crit * se,
            "ci_high": est + t_crit * se,
            "n_cohorts_used": int(len(sub)),
        })
    out_det = pd.DataFrame(rows).sort_values(label_col).reset_index(drop=True)
    mean_est = float(out_det["estimate"].mean())

    return CausalResult(
        method=f"ETWFE — {type} aggregation",
        estimand=f"ATT by {label_col.replace('_', ' ')}",
        estimate=mean_est, se=np.nan,
        pvalue=np.nan, ci=(np.nan, np.nan), alpha=alpha,
        n_obs=int(result.n_obs), detail=out_det,
        model_info={"type": type, "source_method": result.method,
                    "se_method": se_method},
        _citation_key="wooldridge_twfe",
    )
