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
*Review of Economic Studies*, 91(6), 3253-3285.
"""

from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


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

    Returns
    -------
    CausalResult
        Result object with ``.summary()``, ``.plot()`` (event study),
        and ``.cite()`` methods. Event study coefficients are stored
        in ``model_info['event_study']``.

    Examples
    --------
    >>> result = sp.did_imputation(
    ...     data=df, y='wage', group='county', time='year',
    ...     first_treat='first_treat', horizon=list(range(-5, 6)),
    ... )
    >>> result.summary()
    >>> result.event_study_plot()

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
    """
    # ── Input validation ─────────────────────────────────────────── #
    df = data.copy()
    for col in [y, group, time, first_treat]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if controls:
        for col in controls:
            if col not in df.columns:
                raise ValueError(f"Control column '{col}' not found in data.")

    if cluster is None:
        cluster = group

    # ── Step 1: Identify treated / untreated observations ──────── #
    # Normalize first_treat: inf, NaN, 0 → never treated (use np.inf)
    ft = df[first_treat].copy().astype(float)
    ft = ft.replace(0, np.inf)
    ft = ft.fillna(np.inf)
    df['_ft'] = ft

    df['_treated_obs'] = df[time].astype(float) >= df['_ft']
    df['_never_treated'] = np.isinf(df['_ft'])
    df['_untreated_obs'] = ~df['_treated_obs']  # includes never-treated

    n_treated = df['_treated_obs'].sum()
    n_untreated = df['_untreated_obs'].sum()

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

    df['_uid'] = df[group].map(unit_map)
    df['_tid'] = df[time].map(time_map)

    untreated = df[df['_untreated_obs']].copy()

    # Outcome vector for untreated obs
    y_u = untreated[y].values.astype(float)

    # Controls matrix for untreated obs (if any)
    has_controls = controls is not None and len(controls) > 0
    if has_controls:
        X_u = untreated[controls].values.astype(float)
        X_all = df[controls].values.astype(float)
    else:
        X_u = np.empty((len(untreated), 0))
        X_all = np.empty((len(df), 0))

    n_controls = X_u.shape[1]

    # Solve TWFE via demeaning (Frisch-Waugh-Lovell):
    # Demean by unit and time among untreated, then regress residuals
    # to get beta for controls. Then recover FEs.

    uid_u = untreated['_uid'].values
    tid_u = untreated['_tid'].values

    # Compute unit means and time means among untreated
    unit_y_sum = np.zeros(n_units)
    unit_count = np.zeros(n_units)
    time_y_sum = np.zeros(n_times)
    time_count = np.zeros(n_times)

    for idx in range(len(y_u)):
        ui, ti = uid_u[idx], tid_u[idx]
        unit_y_sum[ui] += y_u[idx]
        unit_count[ui] += 1
        time_y_sum[ti] += y_u[idx]
        time_count[ti] += 1

    # Avoid division by zero for units/times not in untreated
    unit_count_safe = np.where(unit_count > 0, unit_count, 1)
    time_count_safe = np.where(time_count > 0, time_count, 1)

    unit_y_mean = unit_y_sum / unit_count_safe
    time_y_mean = time_y_sum / time_count_safe
    grand_y_mean = np.sum(y_u) / len(y_u) if len(y_u) > 0 else 0.0

    if has_controls:
        # Compute unit and time means for controls
        unit_x_sum = np.zeros((n_units, n_controls))
        time_x_sum = np.zeros((n_times, n_controls))

        for idx in range(len(y_u)):
            ui, ti = uid_u[idx], tid_u[idx]
            unit_x_sum[ui] += X_u[idx]
            time_x_sum[ti] += X_u[idx]

        unit_x_mean = unit_x_sum / unit_count_safe[:, None]
        time_x_mean = time_x_sum / time_count_safe[:, None]
        grand_x_mean = X_u.mean(axis=0)

        # Double-demean Y and X
        y_dm = np.empty_like(y_u)
        X_dm = np.empty_like(X_u)
        for idx in range(len(y_u)):
            ui, ti = uid_u[idx], tid_u[idx]
            y_dm[idx] = y_u[idx] - unit_y_mean[ui] - time_y_mean[ti] + grand_y_mean
            X_dm[idx] = X_u[idx] - unit_x_mean[ui] - time_x_mean[ti] + grand_x_mean

        # OLS on demeaned data
        beta = _ols_coef(X_dm, y_dm)
    else:
        beta = np.array([])

    # ── Recover fixed effects ──────────────────────────────────── #
    # alpha_i = mean_i(Y - X'beta) among untreated for unit i
    # lambda_t = mean_t(Y - X'beta) among untreated for time t - mean(alpha)

    if has_controls:
        y_adj_u = y_u - X_u @ beta
    else:
        y_adj_u = y_u.copy()

    # Unit FE: mean of adjusted Y for each unit in untreated
    unit_adj_sum = np.zeros(n_units)
    unit_adj_count = np.zeros(n_units)
    for idx in range(len(y_adj_u)):
        ui = uid_u[idx]
        unit_adj_sum[ui] += y_adj_u[idx]
        unit_adj_count[ui] += 1

    unit_adj_count_safe = np.where(unit_adj_count > 0, unit_adj_count, 1)
    alpha_hat = unit_adj_sum / unit_adj_count_safe  # unit FE

    # Time FE: mean of (adjusted Y - alpha_i) for each time period
    time_resid_sum = np.zeros(n_times)
    time_resid_count = np.zeros(n_times)
    for idx in range(len(y_adj_u)):
        ui, ti = uid_u[idx], tid_u[idx]
        time_resid_sum[ti] += y_adj_u[idx] - alpha_hat[ui]
        time_resid_count[ti] += 1

    time_resid_count_safe = np.where(time_resid_count > 0, time_resid_count, 1)
    lambda_hat = time_resid_sum / time_resid_count_safe  # time FE

    # ── Step 3: Impute counterfactual for treated observations ── #
    treated_mask = df['_treated_obs'].values
    uid_all = df['_uid'].values
    tid_all = df['_tid'].values
    y_all = df[y].values.astype(float)

    # Predicted Y(0) for ALL observations
    y0_hat = alpha_hat[uid_all] + lambda_hat[tid_all]
    if has_controls:
        y0_hat += X_all @ beta

    # Individual treatment effects for treated obs
    tau_hat = y_all - y0_hat  # defined for all obs; meaningful for treated

    df['_tau_hat'] = tau_hat
    df['_y0_hat'] = y0_hat

    # ── Relative time ──────────────────────────────────────────── #
    df['_rel_time'] = df[time].astype(float) - df['_ft']
    # For never-treated, _rel_time will be -inf; that's fine

    # ── Step 4: Aggregate treatment effects ────────────────────── #
    treated_df = df[treated_mask].copy()

    # Overall ATT
    att = float(treated_df['_tau_hat'].mean())

    # ── Step 5: Standard errors ────────────────────────────────── #
    # Cluster-robust SEs with influence-function approach
    cluster_ids = df[cluster].values
    treated_clusters = treated_df[cluster].values

    # Compute residuals on untreated for the FE model
    resid_u = np.zeros(len(df))
    resid_u[~treated_mask] = y_all[~treated_mask] - y0_hat[~treated_mask]

    se_att, psi_clusters = _cluster_se_imputation(
        df=df,
        tau_hat=tau_hat,
        treated_mask=treated_mask,
        resid_untreated=resid_u,
        cluster_col=cluster,
        uid_col='_uid',
        tid_col='_tid',
        alpha_hat=alpha_hat,
        lambda_hat=lambda_hat,
        unit_adj_count=unit_adj_count,
        time_resid_count=time_resid_count,
        n_units=n_units,
        n_times=n_times,
    )

    z_crit = stats.norm.ppf(1 - alpha / 2)
    pvalue_att = float(2 * (1 - stats.norm.cdf(abs(att / se_att)))) if se_att > 0 else 1.0
    ci_att = (att - z_crit * se_att, att + z_crit * se_att)

    # ── Event study (if horizon requested) ─────────────────────── #
    event_study_df = None
    pretrend_test = None

    if horizon is not None:
        es_rows = []
        pre_k_chi2_components = []

        # For event study, we need all obs of eventually-treated units
        # (including pre-treatment periods for placebo/pre-trend checks)
        eventually_treated = ~np.isinf(df['_ft'].values)
        rel_time_rounded = np.round(df['_rel_time'].values)

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
                uid_col='_uid',
                tid_col='_tid',
                alpha_hat=alpha_hat,
                lambda_hat=lambda_hat,
                unit_adj_count=unit_adj_count,
                time_resid_count=time_resid_count,
                n_units=n_units,
                n_times=n_times,
            )

            pval_k = float(2 * (1 - stats.norm.cdf(abs(att_k / se_k)))) if se_k > 0 else 1.0

            es_rows.append({
                'relative_time': k,
                'att': att_k,
                'se': se_k,
                'ci_lower': att_k - z_crit * se_k,
                'ci_upper': att_k + z_crit * se_k,
                'pvalue': pval_k,
                'n_obs': n_k,
            })

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
                'statistic': chi2_stat,
                'df': df_chi2,
                'pvalue': chi2_pval,
            }

    # ── Build model_info ───────────────────────────────────────── #
    model_info: Dict[str, Any] = {
        'estimator': 'BJS Imputation',
        'n_treated_obs': int(n_treated),
        'n_control_obs': int(n_untreated),
        'n_units': int(n_units),
        'n_time_periods': int(n_times),
        'n_never_treated': int(df['_never_treated'].sum() // max(n_times, 1)),
        'cluster_var': cluster,
    }

    if has_controls:
        model_info['controls'] = controls
        model_info['beta_controls'] = dict(zip(controls, beta.tolist()))

    if event_study_df is not None and len(event_study_df) > 0:
        model_info['event_study'] = event_study_df

    if pretrend_test is not None:
        model_info['pretrend_test'] = pretrend_test

    # ── Return CausalResult ────────────────────────────────────── #
    return CausalResult(
        method='Borusyak, Jaravel & Spiess (2024) Imputation Estimator',
        estimand='ATT',
        estimate=att,
        se=se_att,
        pvalue=pvalue_att,
        ci=ci_att,
        alpha=alpha,
        n_obs=len(data),
        detail=event_study_df,
        model_info=model_info,
        _citation_key='did_imputation',
    )


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════


def _ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via least squares. Returns empty array if no regressors."""
    if X.shape[1] == 0:
        return np.array([])
    try:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])


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
                    adjustment += eps_it * n_treated_unit / (
                        unit_adj_count[u_i] * N1
                    )
                # Influence via time FE
                if time_resid_count[t_i] > 0:
                    adjustment += eps_it * n_treated_time / (
                        time_resid_count[t_i] * N1
                    )

        psi_values[c_idx] = direct + adjustment

    # Clustered variance: V = sum(psi_c^2)
    variance = float(np.sum(psi_values ** 2))
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
                    adjustment += eps_it * n_k_unit / (
                        unit_adj_count[u_i] * N_k
                    )
                if time_resid_count[t_i] > 0:
                    adjustment += eps_it * n_k_time / (
                        time_resid_count[t_i] * N_k
                    )

        psi_values[c_idx] = direct + adjustment

    variance = float(np.sum(psi_values ** 2))
    se = float(np.sqrt(variance))

    if n_clusters > 1:
        se *= np.sqrt(n_clusters / (n_clusters - 1))

    return se


# Register citation
CausalResult._CITATIONS['did_imputation'] = (
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
