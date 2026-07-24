"""
Borusyak, Jaravel & Spiess (2024) imputation estimator for staggered DID.

The imputation approach estimates unit and time fixed effects using only
untreated observations, imputes counterfactual outcomes for treated
observations, and computes treatment effects as the difference between
observed and imputed outcomes. This avoids the negative-weighting problem
of TWFE regressions under heterogeneous treatment effects.

References
----------
Borusyak, K., Jaravel, X. and Spiess, J. (2024).
"Revisiting Event-Study Designs: Robust and Efficient Estimation."
*Review of Economic Studies*, 91(6), 3253-3285. [@borusyak2024revisiting]
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import lsqr

from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult


def _didimp_cluster_bootstrap(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: str,
    n_boot: int,
    seed: int,
) -> float:
    """Pairs-cluster bootstrap of the overall ATT for ``did_imputation``.

    The analytic influence-function SE under-counts the variance from
    estimating the unit/time fixed effects on the untreated sample. Resampling
    whole clusters and re-running the full imputation estimator gives a valid
    standard error for the headline ATT. Returns the bootstrap SD of the ATT.
    """
    rng = np.random.default_rng(seed)
    clusters = pd.unique(data[cluster])
    n_g = len(clusters)
    rows_by_cluster = {c: data[data[cluster] == c] for c in clusters}
    ctrl = controls if controls else None

    n_boot_int = int(n_boot)
    boot = np.full(n_boot_int, np.nan, dtype=float)
    for b in range(n_boot_int):
        drawn = rng.choice(n_g, size=n_g, replace=True)
        parts = []
        for j, ci in enumerate(drawn):
            sub = rows_by_cluster[clusters[ci]].copy()
            sub[group] = sub[group].astype(str) + f"__b{j}"
            sub["__bcl"] = j
            parts.append(sub)
        bd = pd.concat(parts, ignore_index=True)
        try:
            r = did_imputation(
                bd,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                controls=ctrl,
                horizon=None,
                cluster="__bcl",
                vce="none",
            )
        except Exception:
            continue  # replicate stays NaN; bootstrap_se tracks the failure
        if np.isfinite(r.estimate):
            boot[b] = float(r.estimate)
    return _bootstrap_se(boot, label="did.imputation")


def did_imputation(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    horizon: Optional[List[int]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    vce: str = "analytic",
    n_boot: int = 199,
    boot_seed: int = 0,
) -> CausalResult:
    """
    Borusyak, Jaravel & Spiess (2024) imputation DID estimator.

    Estimates ATT by imputing counterfactual outcomes for treated
    observations using a TWFE model fit only on untreated data.

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
    controls : list of str, optional
        Additional control covariates.
    horizon : list of int, optional
        Relative time periods for event study estimates,
        e.g. ``list(range(-5, 6))``. If ``None``, reports only the
        overall ATT (no event study disaggregation).
    cluster : str, optional
        Variable for cluster-robust standard errors.
        Defaults to ``group`` (unit-level clustering).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    vce : {'analytic', 'bootstrap'}, default 'analytic'
        Standard-error mode for the overall ATT. ``'analytic'`` uses the
        influence-function cluster SE (fast) but under-counts the variance
        from estimating the unit/time fixed effects and is **anti-conservative**
        (≈0.87 coverage at a nominal 95% level); a ``UserWarning`` recommends
        ``'bootstrap'``. ``'bootstrap'`` resamples whole clusters and re-runs
        the full imputation estimator. Point estimates are identical either
        way; per-horizon event-study SEs are unaffected.
    n_boot : int, default 199
        Number of cluster-bootstrap replications when ``vce='bootstrap'``.
    boot_seed : int, default 0
        Seed for the cluster bootstrap (deterministic results).

    Returns
    -------
    CausalResult
        Result object with ``.summary()``, ``.plot()`` (event study),
        and ``.cite()`` methods. Event study coefficients are stored
        in ``model_info['event_study']``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for unit in range(40):
    ...     first = [4, 7, 0][unit % 3]  # cohorts 4, 7, never-treated (0)
    ...     for year in range(1, 9):
    ...         treated = first != 0 and year >= first
    ...         te = 2.0 * (year - first + 1) if treated else 0.0
    ...         rows.append({'county': unit, 'year': year,
    ...                      'wage': unit * 0.1 + year + te + rng.normal(),
    ...                      'first_treat': first})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.did_imputation(
    ...     data=df, y='wage', group='county', time='year',
    ...     first_treat='first_treat', horizon=list(range(-5, 6)),
    ... )
    >>> bool(result.estimate > 0)
    True
    >>> fig, ax = result.event_study_plot()

    Notes
    -----
    The algorithm proceeds in four steps:

    1. **Classify** observations as treated (t >= first_treat_i) or
       untreated (t < first_treat_i, or never-treated unit).
    2. **Estimate TWFE** on untreated observations only:
       Y_it = alpha_i + lambda_t + X_it'beta + eps_it.
    3. **Impute** counterfactual outcomes for treated observations:
       tau_hat_it = Y_it - (alpha_hat_i + lambda_hat_t + X_it'beta_hat).
    4. **Aggregate** into ATT or event-study ATT(k) and compute
       cluster-robust standard errors with a two-step adjustment.

    References
    ----------
    Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-study
    designs: Robust and efficient estimation. *Review of Economic Studies*.
    [@borusyak2024revisiting]
    """
    # ── Input validation ─────────────────────────────────────────── #
    df = data.copy()
    control_names = list(controls or [])
    for col in [y, group, time, first_treat]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    for col in control_names:
        if col not in df.columns:
            raise ValueError(f"Control column '{col}' not found in data.")

    if vce not in ("analytic", "bootstrap", "none"):
        raise ValueError(f"vce must be 'analytic', 'bootstrap', or 'none'; got {vce!r}")
    if cluster is None:
        cluster = group

    # ── Step 1: Identify treated / untreated observations ──────── #
    # Normalize first_treat: inf, NaN, 0 → never treated (use np.inf)
    ft = df[first_treat].copy().astype(float)
    ft = ft.replace(0, np.inf)
    ft = ft.fillna(np.inf)
    df["_ft"] = ft

    df["_treated_obs"] = df[time].astype(float) >= df["_ft"]
    df["_never_treated"] = np.isinf(df["_ft"])
    df["_untreated_obs"] = ~df["_treated_obs"]  # includes never-treated

    n_treated = df["_treated_obs"].sum()
    n_untreated = df["_untreated_obs"].sum()

    if n_treated == 0:
        raise ValueError("No treated observations found. Check 'first_treat' column.")
    if n_untreated == 0:
        raise ValueError("No untreated observations found. Need control observations.")

    # ── Step 2: Estimate TWFE on untreated observations ─────────── #
    # Encode unit and time as integer indices for FE
    unit_ids = df[group].unique()
    time_ids = sorted(df[time].unique())
    unit_map = {u: idx for idx, u in enumerate(unit_ids)}
    time_map = {t_val: idx for idx, t_val in enumerate(time_ids)}
    n_units = len(unit_ids)
    n_times = len(time_ids)

    df["_uid"] = df[group].map(unit_map)
    df["_tid"] = df[time].map(time_map)

    untreated = df[df["_untreated_obs"]].copy()

    has_controls = len(control_names) > 0
    uid_u = untreated["_uid"].values
    tid_u = untreated["_tid"].values

    unit_adj_count = np.bincount(uid_u, minlength=n_units).astype(float)
    time_resid_count = np.bincount(tid_u, minlength=n_times).astype(float)

    treated_mask = df["_treated_obs"].values
    treated_uids = np.unique(df.loc[treated_mask, "_uid"].values)
    treated_tids = np.unique(df.loc[treated_mask, "_tid"].values)
    missing_units = [
        unit_ids[int(ui)] for ui in treated_uids if unit_adj_count[int(ui)] <= 0
    ]
    missing_times = [
        time_ids[int(ti)] for ti in treated_tids if time_resid_count[int(ti)] <= 0
    ]
    if missing_units:
        preview = ", ".join(map(str, missing_units[:5]))
        raise ValueError(
            "BJS imputation needs at least one untreated observation for "
            "every treated unit to estimate its unit fixed effect. "
            f"Missing untreated history for unit(s): {preview}"
            + (" ..." if len(missing_units) > 5 else "")
        )
    if missing_times:
        preview = ", ".join(map(str, missing_times[:5]))
        raise ValueError(
            "BJS imputation needs at least one untreated observation in "
            "every treated time period to estimate its time fixed effect. "
            f"Missing untreated comparison period(s): {preview}"
            + (" ..." if len(missing_times) > 5 else "")
        )

    y0_hat, beta = _fit_untreated_twfe_sparse(
        df=df,
        untreated=untreated,
        y=y,
        controls=control_names if has_controls else None,
        uid_col="_uid",
        tid_col="_tid",
        n_units=n_units,
        n_times=n_times,
    )

    # These arguments are retained for the existing SE helper API.  The
    # helper only needs the untreated counts and residuals; fitted values
    # now come from the exact sparse TWFE solve above.
    alpha_hat = np.zeros(n_units)
    lambda_hat = np.zeros(n_times)

    # ── Step 3: Impute counterfactual for treated observations ── #
    y_all = df[y].values.astype(float)

    # Individual treatment effects for treated obs
    tau_hat = y_all - y0_hat  # defined for all obs; meaningful for treated

    df["_tau_hat"] = tau_hat
    df["_y0_hat"] = y0_hat

    # ── Relative time ──────────────────────────────────────────── #
    df["_rel_time"] = df[time].astype(float) - df["_ft"]
    # For never-treated, _rel_time will be -inf; that's fine

    # ── Step 4: Aggregate treatment effects ────────────────────── #
    treated_df = df[treated_mask].copy()

    # Overall ATT
    att = float(treated_df["_tau_hat"].mean())

    # ── Step 5: Standard errors ────────────────────────────────── #
    # Cluster-robust SEs with influence-function approach
    # Compute residuals on untreated for the FE model
    resid_u = np.zeros(len(df))
    resid_u[~treated_mask] = y_all[~treated_mask] - y0_hat[~treated_mask]

    se_att, psi_clusters = _cluster_se_imputation(
        df=df,
        tau_hat=tau_hat,
        treated_mask=treated_mask,
        resid_untreated=resid_u,
        cluster_col=cluster,
        uid_col="_uid",
        tid_col="_tid",
        alpha_hat=alpha_hat,
        lambda_hat=lambda_hat,
        unit_adj_count=unit_adj_count,
        time_resid_count=time_resid_count,
        n_units=n_units,
        n_times=n_times,
    )

    # Standard-error mode for the overall ATT. The analytic influence-function
    # SE under-counts the variance from estimating the unit/time fixed effects
    # (≈0.87 coverage at a nominal 95% level); 'bootstrap' resamples whole
    # clusters and re-runs the full imputation estimator. 'none' is internal
    # (used by the bootstrap to avoid recursion / the warning).
    if vce == "bootstrap":
        se_boot = _didimp_cluster_bootstrap(
            data, y, group, time, first_treat, controls, cluster, n_boot, boot_seed
        )
        if np.isfinite(se_boot):
            se_att = se_boot
    elif vce == "analytic":
        warnings.warn(
            "did_imputation: the default analytic standard error for the "
            "overall ATT under-counts the variance from estimating the "
            "unit/time fixed effects, so it is anti-conservative (~0.87 "
            "coverage at a nominal 95% level). For valid inference on the "
            "overall ATT pass vce='bootstrap' (a cluster bootstrap of the "
            "full imputation estimator).",
            UserWarning,
            stacklevel=2,
        )

    z_crit = stats.norm.ppf(1 - alpha / 2)
    pvalue_att = (
        float(2 * (1 - stats.norm.cdf(abs(att / se_att)))) if se_att > 0 else 1.0
    )
    ci_att = (att - z_crit * se_att, att + z_crit * se_att)

    # ── Event study (if horizon requested) ─────────────────────── #
    event_study_df = None
    pretrend_test = None

    if horizon is not None:
        es_rows = []
        pre_k_chi2_components = []

        # For event study, we need all obs of eventually-treated units
        # (including pre-treatment periods for placebo/pre-trend checks)
        eventually_treated = ~np.isinf(df["_ft"].values)
        rel_time_rounded = np.round(df["_rel_time"].values)

        for k in sorted(horizon):
            # Observations of eventually-treated units at relative time k
            mask_k = eventually_treated & (rel_time_rounded == k)
            n_k = int(mask_k.sum())
            if n_k == 0:
                continue

            att_k = float(tau_hat[mask_k].mean())

            # Cluster SE for this horizon
            se_k = _cluster_se_horizon(
                df=df,
                tau_hat=tau_hat,
                mask_k=mask_k,
                treated_mask=treated_mask,
                resid_untreated=resid_u,
                cluster_col=cluster,
                uid_col="_uid",
                tid_col="_tid",
                alpha_hat=alpha_hat,
                lambda_hat=lambda_hat,
                unit_adj_count=unit_adj_count,
                time_resid_count=time_resid_count,
                n_units=n_units,
                n_times=n_times,
            )

            pval_k = (
                float(2 * (1 - stats.norm.cdf(abs(att_k / se_k)))) if se_k > 0 else 1.0
            )

            es_rows.append(
                {
                    "relative_time": k,
                    "att": att_k,
                    "se": se_k,
                    "ci_lower": att_k - z_crit * se_k,
                    "ci_upper": att_k + z_crit * se_k,
                    "pvalue": pval_k,
                    "n_obs": n_k,
                }
            )

            # Collect pre-treatment for joint test
            if k < 0 and se_k > 0:
                pre_k_chi2_components.append((att_k, se_k))

        event_study_df = pd.DataFrame(es_rows)

        # Pre-trend joint test (Wald chi-squared)
        if len(pre_k_chi2_components) > 0:
            pre_atts = np.array([c[0] for c in pre_k_chi2_components])
            pre_ses = np.array([c[1] for c in pre_k_chi2_components])
            chi2_stat = float(np.sum((pre_atts / pre_ses) ** 2))
            df_chi2 = len(pre_k_chi2_components)
            chi2_pval = float(1 - stats.chi2.cdf(chi2_stat, df_chi2))
            pretrend_test = {
                "statistic": chi2_stat,
                "df": df_chi2,
                "pvalue": chi2_pval,
            }

    # ── Build model_info ───────────────────────────────────────── #
    model_info: Dict[str, Any] = {
        "estimator": "BJS Imputation",
        "n_treated_obs": int(n_treated),
        "n_control_obs": int(n_untreated),
        "n_units": int(n_units),
        "n_time_periods": int(n_times),
        "n_never_treated": int(df["_never_treated"].sum() // max(n_times, 1)),
        "cluster_var": cluster,
        "vce": vce,
    }

    if has_controls:
        model_info["controls"] = control_names
        model_info["beta_controls"] = dict(zip(control_names, beta.tolist()))

    if event_study_df is not None and len(event_study_df) > 0:
        model_info["event_study"] = event_study_df

    if pretrend_test is not None:
        model_info["pretrend_test"] = pretrend_test

    # ── Return CausalResult ────────────────────────────────────── #
    _result = CausalResult(
        method="Borusyak, Jaravel & Spiess (2024) Imputation Estimator",
        estimand="ATT",
        estimate=att,
        se=se_att,
        pvalue=pvalue_att,
        ci=ci_att,
        alpha=alpha,
        n_obs=len(data),
        detail=event_study_df,
        model_info=model_info,
        _citation_key="did_imputation",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.did_imputation",
            params={
                "y": y,
                "group": group,
                "time": time,
                "first_treat": first_treat,
                "controls": controls,
                "horizon": horizon,
                "cluster": cluster,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════


def _ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via least squares. Returns empty array if no regressors."""
    if X.shape[1] == 0:
        return np.array([])
    try:
        return np.asarray(np.linalg.lstsq(X, y, rcond=None)[0], dtype=float)
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])


def _fit_untreated_twfe_sparse(
    df: pd.DataFrame,
    untreated: pd.DataFrame,
    y: str,
    controls: Optional[List[str]],
    uid_col: str,
    tid_col: str,
    n_units: int,
    n_times: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit untreated-only TWFE by sparse least squares and predict all rows.

    The untreated sample in staggered DID is usually unbalanced: early
    cohorts contribute fewer untreated periods than late or never-treated
    cohorts.  A one-pass "unit mean + time mean - grand mean" transform is
    exact only on balanced panels.  BJS needs the actual least-squares
    projection on unit and time fixed effects, so we solve the dummy
    regression directly with a sparse design matrix.
    """
    controls = list(controls or [])
    y_u = untreated[y].values.astype(float)
    uid_u = untreated[uid_col].values.astype(int)
    tid_u = untreated[tid_col].values.astype(int)

    unit_seen = np.bincount(uid_u, minlength=n_units) > 0
    time_seen = np.bincount(tid_u, minlength=n_times) > 0
    if not unit_seen.any() or not time_seen.any():
        raise ValueError("No untreated observations available for BJS TWFE fit.")

    ref_unit = int(np.flatnonzero(unit_seen)[0])
    ref_time = int(np.flatnonzero(time_seen)[0])

    unit_cols = np.full(n_units, -1, dtype=int)
    next_col = 1  # intercept
    for u in range(n_units):
        if u != ref_unit:
            unit_cols[u] = next_col
            next_col += 1

    time_cols = np.full(n_times, -1, dtype=int)
    for t_idx in range(n_times):
        if t_idx != ref_time:
            time_cols[t_idx] = next_col
            next_col += 1

    n_fe_cols = next_col

    def _design(frame: pd.DataFrame) -> sparse.csr_matrix:
        n = len(frame)
        rows_parts: List[np.ndarray] = [np.arange(n, dtype=int)]
        cols_parts: List[np.ndarray] = [np.zeros(n, dtype=int)]
        data_parts: List[np.ndarray] = [np.ones(n, dtype=float)]

        uid = frame[uid_col].values.astype(int)
        ucols = unit_cols[uid]
        u_mask = ucols >= 0
        if u_mask.any():
            rows_parts.append(np.flatnonzero(u_mask))
            cols_parts.append(ucols[u_mask])
            data_parts.append(np.ones(int(u_mask.sum()), dtype=float))

        tid = frame[tid_col].values.astype(int)
        tcols = time_cols[tid]
        t_mask = tcols >= 0
        if t_mask.any():
            rows_parts.append(np.flatnonzero(t_mask))
            cols_parts.append(tcols[t_mask])
            data_parts.append(np.ones(int(t_mask.sum()), dtype=float))

        fixed = sparse.coo_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(rows_parts), np.concatenate(cols_parts)),
            ),
            shape=(n, n_fe_cols),
        ).tocsr()

        if not controls:
            return fixed

        x = sparse.csr_matrix(frame[controls].values.astype(float))
        return sparse.hstack([fixed, x], format="csr")

    X_u = _design(untreated)
    n_cols = X_u.shape[1]
    fit = lsqr(
        X_u,
        y_u,
        atol=1e-10,
        btol=1e-10,
        iter_lim=max(1000, 4 * n_cols),
    )
    coef = fit[0]
    y0_hat = np.asarray(_design(df) @ coef, dtype=float)
    beta = coef[-len(controls) :] if controls else np.array([])
    return y0_hat, np.asarray(beta, dtype=float)


def _cluster_se_imputation(
    df: pd.DataFrame,
    tau_hat: np.ndarray,
    treated_mask: np.ndarray,
    resid_untreated: np.ndarray,
    cluster_col: str,
    uid_col: str,
    tid_col: str,
    alpha_hat: np.ndarray,
    lambda_hat: np.ndarray,
    unit_adj_count: np.ndarray,
    time_resid_count: np.ndarray,
    n_units: int,
    n_times: int,
) -> Tuple[float, Dict]:
    """
    Cluster-robust SE for the overall ATT with two-step correction.

    The influence function for cluster c is:

        psi_c = (1/N1) * sum_{(i,t) in treated, i in c} [tau_hat_it - ATT]
              + adjustment for estimation error in alpha_hat, lambda_hat

    The adjustment term propagates the uncertainty from estimating FEs
    on untreated data into the treated-observation imputation.
    """
    N1 = treated_mask.sum()
    att = float(tau_hat[treated_mask].mean())

    clusters = df[cluster_col].values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    uid = df[uid_col].values
    tid = df[tid_col].values

    psi_values = np.zeros(n_clusters)

    for c_idx, c_val in enumerate(unique_clusters):
        c_mask = clusters == c_val

        # ── Direct term: treated obs in this cluster ──
        c_treated = c_mask & treated_mask
        if c_treated.any():
            direct = np.sum(tau_hat[c_treated] - att) / N1
        else:
            direct = 0.0

        # ── Adjustment term: untreated obs in this cluster ──
        # The estimation error in alpha_hat_i and lambda_hat_t affects
        # the imputation for treated obs.
        c_untreated = c_mask & (~treated_mask)
        adjustment = 0.0

        if c_untreated.any():
            uids_c = uid[c_untreated]
            tids_c = tid[c_untreated]
            resids_c = resid_untreated[c_untreated]

            # How many treated obs use each unit FE / time FE
            # from this cluster's untreated observations?
            for idx in range(len(resids_c)):
                u_i = uids_c[idx]
                t_i = tids_c[idx]
                eps_it = resids_c[idx]

                # Count how many treated obs share unit u_i
                n_treated_unit = np.sum(treated_mask & (uid == u_i))
                # Count how many treated obs share time t_i
                n_treated_time = np.sum(treated_mask & (tid == t_i))

                # Influence via unit FE
                if unit_adj_count[u_i] > 0:
                    adjustment += eps_it * n_treated_unit / (unit_adj_count[u_i] * N1)
                # Influence via time FE
                if time_resid_count[t_i] > 0:
                    adjustment += eps_it * n_treated_time / (time_resid_count[t_i] * N1)

        psi_values[c_idx] = direct + adjustment

    # Clustered variance: V = sum(psi_c^2)
    variance = float(np.sum(psi_values**2))
    se = float(np.sqrt(variance))

    # Small-sample correction: G/(G-1)
    if n_clusters > 1:
        se *= np.sqrt(n_clusters / (n_clusters - 1))

    return se, {c: psi_values[i] for i, c in enumerate(unique_clusters)}


def _cluster_se_horizon(
    df: pd.DataFrame,
    tau_hat: np.ndarray,
    mask_k: np.ndarray,
    treated_mask: np.ndarray,
    resid_untreated: np.ndarray,
    cluster_col: str,
    uid_col: str,
    tid_col: str,
    alpha_hat: np.ndarray,
    lambda_hat: np.ndarray,
    unit_adj_count: np.ndarray,
    time_resid_count: np.ndarray,
    n_units: int,
    n_times: int,
) -> float:
    """
    Cluster-robust SE for ATT at a specific horizon k.

    Same influence-function approach as the overall ATT but restricted
    to treated observations at relative time k.
    """
    N_k = mask_k.sum()
    if N_k == 0:
        return np.inf

    att_k = float(tau_hat[mask_k].mean())

    clusters = df[cluster_col].values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    uid = df[uid_col].values
    tid = df[tid_col].values

    psi_values = np.zeros(n_clusters)

    for c_idx, c_val in enumerate(unique_clusters):
        c_mask = clusters == c_val

        # Direct term
        c_k = c_mask & mask_k
        if c_k.any():
            direct = np.sum(tau_hat[c_k] - att_k) / N_k
        else:
            direct = 0.0

        # Adjustment term (untreated obs)
        c_untreated = c_mask & (~treated_mask)
        adjustment = 0.0

        if c_untreated.any():
            uids_c = uid[c_untreated]
            tids_c = tid[c_untreated]
            resids_c = resid_untreated[c_untreated]

            for idx in range(len(resids_c)):
                u_i = uids_c[idx]
                t_i = tids_c[idx]
                eps_it = resids_c[idx]

                # Count how many horizon-k treated obs share this unit/time
                n_k_unit = np.sum(mask_k & (uid == u_i))
                n_k_time = np.sum(mask_k & (tid == t_i))

                if unit_adj_count[u_i] > 0:
                    adjustment += eps_it * n_k_unit / (unit_adj_count[u_i] * N_k)
                if time_resid_count[t_i] > 0:
                    adjustment += eps_it * n_k_time / (time_resid_count[t_i] * N_k)

        psi_values[c_idx] = direct + adjustment

    variance = float(np.sum(psi_values**2))
    se = float(np.sqrt(variance))

    if n_clusters > 1:
        se *= np.sqrt(n_clusters / (n_clusters - 1))

    return se


# Register citation
CausalResult._CITATIONS["did_imputation"] = (
    "@article{borusyak2024revisiting,\n"
    "  title={Revisiting Event-Study Designs: Robust and Efficient Estimation},\n"
    "  author={Borusyak, Kirill and Jaravel, Xavier and Spiess, Jann},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={91},\n"
    "  number={6},\n"
    "  pages={3253--3285},\n"
    "  year={2024},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
