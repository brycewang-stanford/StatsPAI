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
    event_cols: List[str] = []
    rel_times = set()
    for g in cohorts:
        mask_g = df["_ft"] == g
        for t_val in periods:
            rel = int(t_val - g)
            if rel < 0:
                continue  # only post-treatment for basic spec
            col = f"_coh{int(g)}_rel{rel}"
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
    if len(event_cols) > 0:
        ev_X_cols = [f"{c}_dm" for c in event_cols] + ctrl_dm_cols
        ev_valid_cols = ["_y_dm"] + ev_X_cols
        ev_mask = df_valid[ev_valid_cols].notna().all(axis=1)
        if ev_mask.sum() > len(ev_X_cols) + 2:
            ev_y = df_valid.loc[ev_mask, "_y_dm"].values
            ev_X = df_valid.loc[ev_mask, ev_X_cols].values
            ev_X = np.column_stack([np.ones(len(ev_y)), ev_X])
            ev_cl = cl_arr[ev_mask.values] if cl_arr is not None else None
            ev_beta, ev_se, _ = _ols_fit(ev_X, ev_y, cluster=ev_cl)

            ev_rows = []
            for j, col in enumerate(event_cols):
                # Parse cohort and rel_time from column name
                parts = col.replace("_coh", "").replace("_rel", " ").split()
                coh_val = int(parts[0])
                rel_val = int(parts[1])
                idx_j = j + 1
                ev_rows.append({
                    "cohort": coh_val,
                    "rel_time": rel_val,
                    "estimate": float(ev_beta[idx_j]),
                    "se": float(ev_se[idx_j]),
                })
            event_study_df = pd.DataFrame(ev_rows)

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
