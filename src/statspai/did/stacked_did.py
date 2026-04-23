"""
Stacked DID estimator (Cengiz, Dube, Lindner & Zipperer, 2019).

Creates "stacked" datasets — one sub-experiment for each treatment cohort —
then estimates DID on the stacked data with cohort-specific unit and time
fixed effects. This avoids the negative-weighting problem of TWFE under
staggered treatment timing and heterogeneous effects.

References
----------
Cengiz, D., Dube, A., Lindner, A. and Zipperer, B. (2019).
"The Effect of Minimum Wages on Low-Wage Jobs."
*Quarterly Journal of Economics*, 134(3), 1405-1454. [@cengiz2019effect]
"""

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def stacked_did(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    window: Tuple[int, int] = (-5, 5),
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    never_treated_only: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Stacked DID estimator (Cengiz, Dube, Lindner & Zipperer, 2019).

    Constructs a stacked dataset with one sub-experiment per treatment
    cohort, then estimates event-study coefficients via TWFE on the
    stacked data with cohort-specific unit and time fixed effects.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable name.
    group : str
        Unit identifier column.
    time : str
        Time period column.
    first_treat : str
        Column indicating the period of first treatment.
        Use ``np.inf``, ``np.nan``, or ``0`` for never-treated units.
    window : tuple of (int, int), default (-5, 5)
        Event window (inclusive) around treatment.
        E.g. ``(-5, 5)`` keeps relative times -5 through 5.
    controls : list of str, optional
        Additional control covariates.
    cluster : str, optional
        Variable for cluster-robust standard errors.
        Defaults to ``group`` (unit-level clustering).
    never_treated_only : bool, default True
        If True, use only never-treated units as controls.
        If False, also include not-yet-treated units as controls.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Result object with ``.summary()``, ``.plot()`` (event study),
        and ``.cite()`` methods. Event study coefficients are stored
        in ``model_info['event_study']``.

    Examples
    --------
    >>> result = sp.stacked_did(
    ...     data=df, y='wage', group='county', time='year',
    ...     first_treat='first_treat', window=(-5, 5),
    ... )
    >>> result.summary()
    >>> result.plot()
    """
    # ── Input validation ─────────────────────────────────────────── #
    df = data.copy()
    required_cols = [y, group, time, first_treat]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if controls:
        for col in controls:
            if col not in df.columns:
                raise ValueError(f"Control column '{col}' not found in data.")

    if cluster is None:
        cluster = group

    if window[0] >= 0:
        raise ValueError("window[0] must be negative (pre-treatment periods).")
    if window[1] < 0:
        raise ValueError("window[1] must be non-negative (post-treatment periods).")

    # ── Normalize first_treat ────────────────────────────────────── #
    ft = df[first_treat].copy().astype(float)
    ft = ft.replace(0, np.inf)
    ft = ft.fillna(np.inf)
    df["_ft"] = ft

    # ── Step 1: Identify cohorts ─────────────────────────────────── #
    cohort_values = sorted(df.loc[np.isfinite(df["_ft"]), "_ft"].unique())
    if len(cohort_values) == 0:
        raise ValueError("No treated cohorts found. Check 'first_treat' column.")

    never_mask = np.isinf(df["_ft"])
    never_units = set(df.loc[never_mask, group].unique())

    # ── Step 2 & 3: Build sub-experiments and stack ──────────────── #
    stacked_frames = []

    for g in cohort_values:
        t_lo = g + window[0]
        t_hi = g + window[1]

        # Units in this cohort (treated at time g)
        cohort_units = set(df.loc[df["_ft"] == g, group].unique())

        # Control units
        if never_treated_only:
            ctrl_units = never_units
        else:
            # Not-yet-treated: units whose first_treat > t_hi
            nyt_mask = df["_ft"] > t_hi
            ctrl_units = never_units | set(df.loc[nyt_mask, group].unique())

        all_units = cohort_units | ctrl_units
        if len(ctrl_units) == 0:
            continue  # skip cohort if no controls available

        # Restrict to units and time window
        sub = df[
            df[group].isin(all_units)
            & (df[time].astype(float) >= t_lo)
            & (df[time].astype(float) <= t_hi)
        ].copy()

        if len(sub) == 0:
            continue

        sub["_cohort"] = g
        sub["_rel_time"] = sub[time].astype(float) - g
        sub["_treated_unit"] = sub[group].isin(cohort_units).astype(int)
        sub["_post"] = (sub["_rel_time"] >= 0).astype(int)

        stacked_frames.append(sub)

    if len(stacked_frames) == 0:
        raise ValueError(
            "No valid sub-experiments could be constructed. "
            "Check data coverage and window size."
        )

    stacked = pd.concat(stacked_frames, ignore_index=True)
    n_cohorts = len(stacked["_cohort"].unique())
    n_units = stacked[group].nunique()
    n_stacked = len(stacked)

    # ── Step 4: Estimate on stacked data ─────────────────────────── #
    # Create event-study dummies: D_k = 1(rel_time == k & treated_unit)
    # Exclude k = -1 as reference period
    rel_times = sorted(stacked["_rel_time"].unique())
    rel_times_est = [k for k in rel_times if k != -1]

    if len(rel_times_est) == 0:
        raise ValueError("Not enough relative time periods for estimation.")

    # Build treatment interaction dummies
    D_cols = []
    for k in rel_times_est:
        col_name = f"_D_{int(k)}"
        stacked[col_name] = (
            (stacked["_rel_time"] == k) & (stacked["_treated_unit"] == 1)
        ).astype(float)
        D_cols.append(col_name)

    # Add controls if specified
    x_cols = list(D_cols)
    if controls:
        x_cols = x_cols + controls

    # Create cohort-specific FE groups
    stacked["_unit_cohort"] = (
        stacked[group].astype(str) + "_" + stacked["_cohort"].astype(str)
    )
    stacked["_time_cohort"] = (
        stacked[time].astype(str) + "_" + stacked["_cohort"].astype(str)
    )

    # Within-group demeaning (Frisch-Waugh-Lovell for two-way FE)
    y_vec = stacked[y].values.astype(float)
    X_mat = stacked[x_cols].values.astype(float)

    uc_groups = stacked["_unit_cohort"].values
    tc_groups = stacked["_time_cohort"].values

    y_dm, X_dm = _twoway_demean(y_vec, X_mat, uc_groups, tc_groups)

    # OLS on demeaned data
    beta, residuals = _ols(X_dm, y_dm)

    # Map coefficients to event-study names
    es_betas = {}
    for idx, k in enumerate(rel_times_est):
        es_betas[k] = beta[idx]

    # ── Step 5: Cluster-robust standard errors ───────────────────── #
    cluster_ids = stacked[cluster].values
    se_vec = _cluster_robust_se(X_dm, residuals, cluster_ids)

    es_se = {}
    for idx, k in enumerate(rel_times_est):
        es_se[k] = se_vec[idx]

    # ── Step 6: Aggregate ATT (post-treatment periods) ───────────── #
    post_ks = [k for k in rel_times_est if k >= 0]
    if len(post_ks) > 0:
        att = np.mean([es_betas[k] for k in post_ks])
        # Delta method: ATT = mean of post betas → se = sqrt(sum of var / n^2)
        # Use cluster-robust vcov for joint inference
        V = _cluster_robust_vcov(X_dm, residuals, cluster_ids)
        post_indices = [rel_times_est.index(k) for k in post_ks]
        n_post = len(post_indices)
        w = np.zeros(len(rel_times_est))
        for pi in post_indices:
            w[pi] = 1.0 / n_post
        att_var = w @ V @ w
        att_se = np.sqrt(max(att_var, 0.0))
    else:
        att = 0.0
        att_se = 0.0

    z_crit = stats.norm.ppf(1 - alpha / 2)
    att_pval = float(2 * (1 - stats.norm.cdf(abs(att) / att_se))) if att_se > 0 else np.nan
    att_ci = (att - z_crit * att_se, att + z_crit * att_se)

    # ── Build event study detail DataFrame ───────────────────────── #
    all_ks = sorted(set(rel_times_est) | {-1})
    rows = []
    for k in all_ks:
        if k == -1:
            rows.append({
                "relative_time": int(k),
                "att": 0.0,
                "se": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "pvalue": np.nan,
            })
        else:
            b = es_betas[k]
            s = es_se[k]
            p = float(2 * (1 - stats.norm.cdf(abs(b) / s))) if s > 0 else np.nan
            rows.append({
                "relative_time": int(k),
                "att": b,
                "se": s,
                "ci_lower": b - z_crit * s,
                "ci_upper": b + z_crit * s,
                "pvalue": p,
            })

    detail = pd.DataFrame(rows)

    # ── Build model_info ─────────────────────────────────────────── #
    model_info = {
        "method_full": "Stacked DID (Cengiz, Dube, Lindner & Zipperer, 2019)",
        "n_cohorts": n_cohorts,
        "cohorts": sorted(cohort_values),
        "n_units": n_units,
        "n_stacked_obs": n_stacked,
        "window": window,
        "never_treated_only": never_treated_only,
        "cluster_var": cluster,
        "event_study": detail,
        "event_study_betas": es_betas,
        "event_study_se": es_se,
    }

    return CausalResult(
        method="Stacked DID (Cengiz et al. 2019)",
        estimand="ATT",
        estimate=float(att),
        se=float(att_se),
        pvalue=float(att_pval),
        ci=att_ci,
        alpha=alpha,
        n_obs=n_stacked,
        detail=detail,
        model_info=model_info,
        _citation_key="stacked_did",
    )


# ──────────────────────────────────────────────────────────────────── #
#  Private helpers
# ──────────────────────────────────────────────────────────────────── #


def _twoway_demean(
    y: np.ndarray,
    X: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple:
    """
    Iterative two-way demeaning (alternating projection) for Y and X.

    Returns demeaned (y_dm, X_dm).
    """
    n = len(y)
    k = X.shape[1]

    # Build group index arrays for fast lookup
    g1_map = {}
    for i, g in enumerate(group1):
        g1_map.setdefault(g, []).append(i)
    g2_map = {}
    for i, g in enumerate(group2):
        g2_map.setdefault(g, []).append(i)

    # Stack y and X for simultaneous demeaning
    Z = np.column_stack([y, X])  # (n, 1+k)

    for _ in range(max_iter):
        Z_old = Z.copy()

        # Demean by group1
        for indices in g1_map.values():
            idx = np.array(indices)
            Z[idx] -= Z[idx].mean(axis=0)

        # Demean by group2
        for indices in g2_map.values():
            idx = np.array(indices)
            Z[idx] -= Z[idx].mean(axis=0)

        # Check convergence
        if np.max(np.abs(Z - Z_old)) < tol:
            break

    return Z[:, 0], Z[:, 1:]


def _ols(X: np.ndarray, y: np.ndarray) -> tuple:
    """OLS regression. Returns (coefficients, residuals)."""
    if X.shape[1] == 0:
        return np.array([]), y.copy()

    # Use lstsq for numerical stability
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return beta, residuals


def _cluster_robust_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """
    Cluster-robust variance-covariance matrix (HC1-style with cluster correction).

    V = (X'X)^{-1} B (X'X)^{-1}
    where B = sum_g (X_g' e_g)(X_g' e_g)'
    with small-sample correction: G/(G-1) * N/(N-K).
    """
    n, k = X.shape
    if k == 0:
        return np.empty((0, 0))

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    # Build cluster score matrix
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    B = np.zeros((k, k))

    cluster_map = {}
    for i, c in enumerate(cluster_ids):
        cluster_map.setdefault(c, []).append(i)

    for c in unique_clusters:
        idx = np.array(cluster_map[c])
        Xg = X[idx]
        eg = residuals[idx]
        score = Xg.T @ eg  # (k,)
        B += np.outer(score, score)

    # Small-sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    V = correction * XtX_inv @ B @ XtX_inv

    return V


def _cluster_robust_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """Cluster-robust standard errors."""
    V = _cluster_robust_vcov(X, residuals, cluster_ids)
    if V.size == 0:
        return np.array([])
    return np.sqrt(np.maximum(np.diag(V), 0.0))
