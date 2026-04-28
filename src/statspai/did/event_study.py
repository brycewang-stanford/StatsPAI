"""
Traditional OLS event study (lead/lag) estimator.

Estimates dynamic treatment effects via relative-time dummies in a
two-way fixed effects framework.  Unlike the Callaway-Sant'Anna or
Sun-Abraham estimators (which correct for heterogeneous treatment timing),
this implements the **classic event study** that is standard in applied
economics when treatment timing is uniform or the researcher wants the
conventional specification.

Model
-----
Y_{it} = α_i + λ_t + Σ_{k≠-1} β_k · 1{t − g_i = k} + X_{it}'γ + ε_{it}

where g_i is unit i's treatment time, k indexes relative time, and
k = −1 is the omitted reference period.

References
----------
Freyaldenhoven, S., Hansen, C. and Shapiro, J.M. (2019).
"Pre-event Trends in the Panel Event-Study Design."
*American Economic Review*, 109(9), 3307-3338. [@freyaldenhoven2019event]

Roth, J. (2022).
"Pretest with Caution: Event-Study Estimates After Testing for Parallel
Trends." *American Economic Review: Insights*, 4(3), 305-322. [@roth2022pretest]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


def event_study(
    data: pd.DataFrame,
    y: str,
    treat_time: str,
    time: str,
    unit: str,
    window: Tuple[int, int] = (-4, 4),
    ref_period: int = -1,
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    weights: Optional[str] = None,
) -> CausalResult:
    """
    Traditional OLS event study with entity and time fixed effects.

    Generates relative-time dummies around the treatment date, omits a
    reference period (default: t = −1), and estimates with TWFE + optional
    clustering.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable.
    treat_time : str
        Column with each unit's treatment time (period when treatment
        starts).  Units never treated should have ``NaN`` or a value
        outside the data range.
    time : str
        Calendar time column (integer or datetime coercible to integer).
    unit : str
        Unit identifier column.
    window : (int, int), default (-4, 4)
        Relative time window (min_lag, max_lag). Periods outside this
        window are binned into the endpoints.
    ref_period : int, default -1
        Omitted reference period (relative time = ref_period).
    covariates : list of str, optional
        Additional time-varying controls.
    cluster : str, optional
        Cluster variable for standard errors (default: ``unit``).
    alpha : float, default 0.05
        Significance level.
    weights : str, optional
        Column name for analytical weights (e.g. population weights).
        Equivalent to Stata's ``[aweight=...]``.

    Returns
    -------
    CausalResult
        With event study estimates in ``model_info['event_study']``
        (DataFrame with columns: relative_time, estimate, se, ci_lower,
        ci_upper) and a pre-trend test in ``model_info['pretrend_test']``.

        Call ``result.event_study_plot()`` to visualize.

    Examples
    --------
    >>> result = sp.event_study(df, y='wage', treat_time='first_treat',
    ...                         time='year', unit='worker_id')
    >>> result.event_study_plot()

    >>> # Wider window with controls
    >>> result = sp.event_study(df, y='revenue', treat_time='policy_year',
    ...                         time='year', unit='firm_id',
    ...                         window=(-6, 6), covariates=['size', 'age'])
    """
    df = data.copy()
    min_lag, max_lag = window

    # --- Compute relative time ---
    df["__treat_time__"] = df[treat_time]
    df["__time__"] = df[time]
    df["__unit__"] = df[unit]

    # Convert time to numeric if needed
    if not np.issubdtype(df["__time__"].dtype, np.number):
        time_map = {t: i for i, t in enumerate(sorted(df["__time__"].unique()))}
        df["__time_num__"] = df["__time__"].map(time_map)
        if not np.issubdtype(df["__treat_time__"].dtype, np.number):
            df["__treat_time_num__"] = df["__treat_time__"].map(time_map)
        else:
            df["__treat_time_num__"] = df["__treat_time__"]
    else:
        df["__time_num__"] = df["__time__"].astype(float)
        df["__treat_time_num__"] = df["__treat_time__"].astype(float)

    # Relative time
    df["__rel_time__"] = df["__time_num__"] - df["__treat_time_num__"]

    # Never-treated units get NaN rel_time — they only contribute via FE
    never_treated = df["__treat_time_num__"].isna()

    # --- Bin endpoints ---
    df.loc[~never_treated, "__rel_time_binned__"] = df.loc[~never_treated, "__rel_time__"].clip(
        lower=min_lag, upper=max_lag
    )

    # --- Create dummies ---
    rel_periods = sorted(set(range(min_lag, max_lag + 1)) - {ref_period})
    for k in rel_periods:
        col = f"__rel_{k}__"
        df[col] = 0.0
        mask = (~never_treated) & (df["__rel_time_binned__"] == k)
        df.loc[mask, col] = 1.0

    # --- Build OLS with entity + time FE via demeaning ---
    dummy_cols = [f"__rel_{k}__" for k in rel_periods]
    cov_cols = covariates or []

    # Demean by entity and time (Frisch-Waugh for TWFE)
    all_y_x_cols = [y] + dummy_cols + cov_cols
    dropna_cols = all_y_x_cols + ["__unit__", "__time_num__"]
    if weights is not None:
        dropna_cols.append(weights)
    df_clean = df.dropna(subset=dropna_cols).copy()

    # Prepare weights array (before demeaning)
    if weights is not None:
        w_raw = df_clean[weights].values.astype(float)
        if np.any(w_raw < 0):
            raise ValueError(f"Weights column '{weights}' contains negative values.")
        n_clean = len(df_clean)
        w_arr = w_raw * (n_clean / w_raw.sum())
    else:
        w_arr = None

    Y, X_mat, col_names = _demean_twfe(
        df_clean, y, dummy_cols + cov_cols, "__unit__", "__time_num__",
        w=w_arr,
    )

    n, k = X_mat.shape

    # --- OLS (possibly weighted) ---
    if w_arr is not None:
        sqrt_w = np.sqrt(w_arr)
        Xw = X_mat * sqrt_w[:, np.newaxis]
        Yw = Y * sqrt_w
    else:
        Xw = X_mat
        Yw = Y

    try:
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)

    beta = XtX_inv @ Xw.T @ Yw
    resid = Y - X_mat @ beta  # residuals in original scale

    # --- Standard errors (clustered by default) ---
    cluster_var = cluster or unit
    cluster_ids = df_clean[cluster_var].values
    se = _cluster_se(Xw, resid, XtX_inv, cluster_ids,
                     w=w_arr)

    # --- Build event study table ---
    es_rows = []
    for i, k_val in enumerate(rel_periods):
        coef = float(beta[i])
        se_i = float(se[i])
        t_crit = sp_stats.norm.ppf(1 - alpha / 2)
        es_rows.append({
            "relative_time": k_val,
            "estimate": coef,
            "se": se_i,
            "ci_lower": coef - t_crit * se_i,
            "ci_upper": coef + t_crit * se_i,
            "pvalue": float(2 * (1 - sp_stats.norm.cdf(abs(coef / se_i)))) if se_i > 0 else 1.0,
        })

    # Add reference period (zero by definition)
    es_rows.append({
        "relative_time": ref_period,
        "estimate": 0.0,
        "se": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "pvalue": 1.0,
    })
    event_study_df = pd.DataFrame(es_rows).sort_values("relative_time").reset_index(drop=True)

    # --- Pre-trend test (joint F-test on pre-treatment coefficients) ---
    pre_indices = [i for i, k_val in enumerate(rel_periods) if k_val < 0]
    pretrend_result = _joint_f_test(beta, XtX_inv, pre_indices, resid, n, k,
                                     w=w_arr)

    # --- Overall ATT (average of post-treatment coefficients) ---
    post = event_study_df[event_study_df["relative_time"] >= 0]
    post_nonref = post[post["relative_time"] != ref_period]
    att = float(post_nonref["estimate"].mean()) if len(post_nonref) > 0 else 0.0
    att_se = float(np.sqrt(np.mean(post_nonref["se"] ** 2) / len(post_nonref))) if len(post_nonref) > 0 else 0.0
    att_p = float(2 * (1 - sp_stats.norm.cdf(abs(att / att_se)))) if att_se > 0 else 1.0

    n_clusters = len(np.unique(cluster_ids))

    _result = CausalResult(
        method="OLS Event Study (TWFE)",
        estimand="ATT",
        estimate=att,
        se=att_se,
        pvalue=att_p,
        ci=(att - sp_stats.norm.ppf(1 - alpha / 2) * att_se,
            att + sp_stats.norm.ppf(1 - alpha / 2) * att_se),
        alpha=alpha,
        n_obs=n,
        detail=event_study_df,
        model_info={
            "model_type": "DID Event Study",
            "event_study": event_study_df,
            "pretrend_test": pretrend_result,
            "ref_period": ref_period,
            "window": window,
            "n_clusters": n_clusters,
            "cluster_var": cluster_var,
            "weights": weights,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.did.event_study",
            params={
                "y": y, "treat_time": treat_time,
                "time": time, "unit": unit,
                "window": list(window),
                "ref_period": ref_period,
                "covariates": list(covariates) if covariates else None,
                "cluster": cluster, "alpha": alpha,
                "weights": weights,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _demean_twfe(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    unit_col: str,
    time_col: str,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Demean Y and X by entity and time means (within transformation).

    If *w* is provided, uses weighted means for demeaning (WLS-FE).
    """
    cols = [y_col] + x_cols
    data_mat = df[cols].values.astype(np.float64)

    # Entity means
    unit_ids = df[unit_col].values
    unique_units = np.unique(unit_ids)
    for u in unique_units:
        mask = unit_ids == u
        if w is not None:
            wm = w[mask]
            ws = wm.sum()
            if ws > 0:
                wmean = (wm[:, np.newaxis] * data_mat[mask]).sum(axis=0) / ws
            else:
                wmean = data_mat[mask].mean(axis=0)
            data_mat[mask] -= wmean
        else:
            data_mat[mask] -= data_mat[mask].mean(axis=0)

    # Time means (on already entity-demeaned data)
    time_ids = df[time_col].values
    unique_times = np.unique(time_ids)
    for t in unique_times:
        mask = time_ids == t
        if w is not None:
            wm = w[mask]
            ws = wm.sum()
            if ws > 0:
                wmean = (wm[:, np.newaxis] * data_mat[mask]).sum(axis=0) / ws
            else:
                wmean = data_mat[mask].mean(axis=0)
            data_mat[mask] -= wmean
        else:
            data_mat[mask] -= data_mat[mask].mean(axis=0)

    Y = data_mat[:, 0]
    X = data_mat[:, 1:]
    return Y, X, x_cols


def _cluster_se(
    X: np.ndarray,
    resid: np.ndarray,
    XtX_inv: np.ndarray,
    cluster_ids: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cluster-robust standard errors.

    *X* should already be the weighted design matrix (Xw) if weights are used.
    *resid* should be unweighted residuals; weighting is applied here via *w*.
    """
    n, k = X.shape
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        if w is not None:
            score_c = (X[mask] * (np.sqrt(w[mask]) * resid[mask])[:, None]).sum(axis=0)
        else:
            score_c = (X[mask] * resid[mask, None]).sum(axis=0)
        meat += np.outer(score_c, score_c)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov = correction * XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.maximum(np.diag(vcov), 0))


def _joint_f_test(
    beta: np.ndarray,
    XtX_inv: np.ndarray,
    indices: List[int],
    resid: np.ndarray,
    n: int,
    k: int,
    w: Optional[np.ndarray] = None,
) -> dict:
    """Joint F-test for subset of coefficients being zero."""
    if not indices:
        return {"statistic": 0.0, "pvalue": 1.0, "df": 0}

    q = len(indices)
    beta_sub = beta[np.array(indices)]

    # Submatrix of variance
    idx = np.array(indices)
    V_sub = XtX_inv[np.ix_(idx, idx)]
    if w is not None:
        sigma2 = np.sum(w * resid ** 2) / (n - k)
    else:
        sigma2 = np.sum(resid ** 2) / (n - k)

    try:
        V_inv = np.linalg.inv(sigma2 * V_sub)
        f_stat = float(beta_sub @ V_inv @ beta_sub / q)
    except np.linalg.LinAlgError:
        f_stat = 0.0

    pvalue = float(1 - sp_stats.f.cdf(f_stat, q, n - k))

    return {
        "statistic": round(f_stat, 4),
        "pvalue": round(pvalue, 4),
        "df": q,
        "test": "Joint F-test on pre-treatment coefficients",
    }
