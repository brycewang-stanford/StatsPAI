"""
Generalized Synthetic Control Method (GSynth).

Uses an interactive fixed-effects (IFE) model — a factor model —
to impute the treated unit's counterfactual. Unlike classic SCM
which constructs a single weighted average, GSynth estimates latent
factors from control units, then projects the treated unit onto
those factors.

Model
-----
Y_{it} = α_i + δ_t + λ_i' f_t + X_{it} β + ε_{it}

where λ_i are unit-specific factor loadings, f_t are common time
factors, estimated via principal components on the control panel.

References
----------
Xu, Y. (2017).
"Generalized Synthetic Control Method: Causal Inference with
Interactive Fixed Effects Models."
*Political Analysis*, 25(1), 57-76.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def gsynth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    n_factors: Optional[int] = None,
    max_factors: int = 5,
    cv_folds: int = 5,
    placebo: bool = True,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Generalized Synthetic Control via interactive fixed effects.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period (inclusive).
    covariates : list of str, optional
        Additional time-varying covariates.
    n_factors : int, optional
        Number of latent factors. If None, selected by cross-validation.
    max_factors : int, default 5
        Maximum factors to try during CV.
    cv_folds : int, default 5
        Cross-validation folds for factor selection.
    placebo : bool, default True
        Run placebo inference.
    seed : int, optional
        Random seed.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> result = sp.gsynth(df, outcome='gdp', unit='state', time='year',
    ...                    treated_unit='California', treatment_time=1989)
    >>> print(result.summary())
    """
    rng = np.random.default_rng(seed)

    # --- Build panel ---
    pivot = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(pivot.columns.tolist())
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 3:
        raise ValueError("GSynth needs at least 3 pre-treatment periods")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period")

    # Separate treated and control
    donors = [u for u in pivot.index if u != treated_unit]
    Y0_pre = pivot.loc[donors, pre_times].values.astype(np.float64)   # (J, T0)
    Y0_post = pivot.loc[donors, post_times].values.astype(np.float64)  # (J, T1)
    Y1_pre = pivot.loc[treated_unit, pre_times].values.astype(np.float64)  # (T0,)
    Y1_post = pivot.loc[treated_unit, post_times].values.astype(np.float64)  # (T1,)

    J, T0 = Y0_pre.shape
    T1 = len(post_times)

    # --- Handle covariates ---
    beta_X = None
    if covariates:
        Y0_pre, Y1_pre, Y0_post, Y1_post, beta_X = _partial_out_covariates(
            data, outcome, unit, time, treated_unit, treatment_time,
            covariates, donors, pre_times, post_times,
        )

    # --- Demean (two-way fixed effects) ---
    Y0_pre_dm, row_means, col_means, grand_mean = _twoway_demean(Y0_pre)

    # --- Select number of factors ---
    if n_factors is None:
        n_factors = _select_factors_cv(Y0_pre_dm, max_factors, cv_folds, rng)
    n_factors = min(n_factors, min(J, T0) - 1)
    n_factors = max(n_factors, 1)

    # --- Extract factors via SVD on demeaned control panel ---
    U, S, Vt = np.linalg.svd(Y0_pre_dm, full_matrices=False)
    F_pre = Vt[:n_factors].T  # (T0, r): time factors
    L_control = U[:, :n_factors] * S[:n_factors]  # (J, r): control loadings

    # --- Estimate treated unit's loadings from pre-period ---
    # L_treated solves  (F_pre' F_pre) L_treated' = F_pre' Y1_pre_dm'
    Y1_pre_dm = Y1_pre - grand_mean - (Y1_pre.mean() - grand_mean)
    L_treated = np.linalg.lstsq(F_pre, Y1_pre_dm, rcond=None)[0]  # (r,)

    # --- Estimate factors for post-period from control data ---
    # F_post.T solves  (L_control' L_control) F_post.T = L_control' Y0_post_dm
    Y0_post_dm = Y0_post - row_means[:, np.newaxis]
    F_post = np.linalg.lstsq(L_control, Y0_post_dm, rcond=None)[0].T  # (T1, r)

    # --- Counterfactual for treated unit ---
    # Y1_hat = grand_mean + unit_FE + F_post @ L_treated
    unit_fe = Y1_pre.mean() - grand_mean
    Y1_hat_pre = grand_mean + unit_fe + F_pre @ L_treated
    Y1_hat_post = grand_mean + unit_fe + F_post @ L_treated

    # Treatment effects
    effects = Y1_post - Y1_hat_post
    att = float(np.mean(effects))
    pre_mspe = float(np.mean((Y1_pre - Y1_hat_pre) ** 2))

    # --- Placebo inference ---
    placebo_atts = []
    if placebo and J >= 2:
        for j in range(J):
            # Treat donor j as "treated"
            other_idx = [i for i in range(J) if i != j]
            Y_plac = Y0_pre[j]
            Y_plac_post = Y0_post[j]
            Y_ctrl_pre = Y0_pre[other_idx]
            Y_ctrl_post = Y0_post[other_idx]

            try:
                Y_dm, rm, cm, gm = _twoway_demean(Y_ctrl_pre)
                U_p, S_p, Vt_p = np.linalg.svd(Y_dm, full_matrices=False)
                F_p = Vt_p[:n_factors].T
                L_c = U_p[:, :n_factors] * S_p[:n_factors]

                Y_plac_dm = Y_plac - gm - (Y_plac.mean() - gm)
                L_p = np.linalg.lstsq(F_p, Y_plac_dm, rcond=None)[0]

                Y_ctrl_post_dm = Y_ctrl_post - rm[:, np.newaxis]
                F_post_p = np.linalg.lstsq(
                    L_c, Y_ctrl_post_dm, rcond=None
                )[0].T

                ue = Y_plac.mean() - gm
                hat = gm + ue + F_post_p @ L_p
                placebo_atts.append(float(np.mean(Y_plac_post - hat)))
            except Exception:
                continue

    if len(placebo_atts) > 0:
        se = float(np.std(placebo_atts, ddof=1))
        pvalue = float(np.mean(np.abs(placebo_atts) >= abs(att)))
        pvalue = max(pvalue, 1 / (len(placebo_atts) + 1))
    else:
        se = float(np.std(effects)) / max(np.sqrt(T1), 1)
        pvalue = np.nan

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z_crit * se, att + z_crit * se)

    # --- Results ---
    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y1_post,
        "counterfactual": Y1_hat_post,
        "effect": effects,
    })

    trajectory_df = pd.DataFrame({
        "time": all_times,
        "treated": np.concatenate([Y1_pre, Y1_post]),
        "synthetic": np.concatenate([Y1_hat_pre, Y1_hat_post]),
    })

    model_info = {
        "n_factors": n_factors,
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "pre_treatment_mspe": round(pre_mspe, 6),
        "pre_treatment_rmse": round(np.sqrt(pre_mspe), 6),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "factors_pre": F_pre,
        "factors_post": F_post,
        "loadings_treated": L_treated,
        "singular_values": S[:n_factors],
        "effects_by_period": effects_df,
        "trajectory": trajectory_df,
        "Y_synth": np.concatenate([Y1_hat_pre, Y1_hat_post]),
        "Y_treated": np.concatenate([Y1_pre, Y1_post]),
        "times": all_times,
    }

    if placebo_atts:
        model_info["placebo_atts"] = placebo_atts
        model_info["n_placebos"] = len(placebo_atts)

    return CausalResult(
        method="Generalized Synthetic Control (Xu 2017)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info=model_info,
        _citation_key="gsynth",
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _twoway_demean(Y: np.ndarray):
    """Remove row means, column means, and grand mean."""
    grand_mean = np.nanmean(Y)
    row_means = np.nanmean(Y, axis=1)
    col_means = np.nanmean(Y, axis=0)
    Y_dm = Y - row_means[:, np.newaxis] - col_means[np.newaxis, :] + grand_mean
    return Y_dm, row_means, col_means, grand_mean


def _select_factors_cv(
    Y_dm: np.ndarray,
    max_factors: int,
    n_folds: int,
    rng: np.random.Generator,
) -> int:
    """Select number of factors via cross-validation on the control panel."""
    J, T = Y_dm.shape
    max_factors = min(max_factors, min(J, T) - 1)
    if max_factors < 1:
        return 1

    # Random fold assignment for entries
    indices = list(range(J * T))
    rng.shuffle(indices)
    fold_size = len(indices) // n_folds

    best_r = 1
    best_mse = np.inf

    for r in range(1, max_factors + 1):
        mse_sum = 0.0
        for f in range(n_folds):
            start = f * fold_size
            end = start + fold_size if f < n_folds - 1 else len(indices)
            test_idx = set(indices[start:end])

            # Mask test entries
            Y_train = Y_dm.copy()
            for idx in test_idx:
                i, j = divmod(idx, T)
                Y_train[i, j] = 0.0

            # SVD reconstruction with r factors
            U, S, Vt = np.linalg.svd(Y_train, full_matrices=False)
            recon = (U[:, :r] * S[:r]) @ Vt[:r]

            # MSE on held-out entries
            fold_mse = 0.0
            for idx in test_idx:
                i, j = divmod(idx, T)
                fold_mse += (Y_dm[i, j] - recon[i, j]) ** 2
            mse_sum += fold_mse / len(test_idx)

        avg_mse = mse_sum / n_folds
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_r = r

    return best_r


def _partial_out_covariates(
    data, outcome, unit, time, treated_unit, treatment_time,
    covariates, donors, pre_times, post_times,
):
    """Partial out covariates via OLS, return residualised outcomes."""
    pre_ctrl = data[
        (data[unit].isin(donors)) & (data[time].isin(pre_times))
    ].copy()

    X = pre_ctrl[covariates].values
    y = pre_ctrl[outcome].values

    # OLS: y = X beta + e
    XtX = X.T @ X + 1e-8 * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ y)

    # Residualise all data
    data_res = data.copy()
    X_all = data_res[covariates].values
    data_res[outcome] = data_res[outcome] - X_all @ beta

    pivot = data_res.pivot_table(index=unit, columns=time, values=outcome)
    Y0_pre = pivot.loc[donors, pre_times].values.astype(np.float64)
    Y0_post = pivot.loc[donors, post_times].values.astype(np.float64)
    Y1_pre = pivot.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = pivot.loc[treated_unit, post_times].values.astype(np.float64)

    return Y0_pre, Y1_pre, Y0_post, Y1_post, beta


# Citation
CausalResult._CITATIONS["gsynth"] = (
    "@article{xu2017generalized,\n"
    "  title={Generalized Synthetic Control Method: Causal Inference\n"
    "  with Interactive Fixed Effects Models},\n"
    "  author={Xu, Yiqing},\n"
    "  journal={Political Analysis},\n"
    "  volume={25},\n"
    "  number={1},\n"
    "  pages={57--76},\n"
    "  year={2017},\n"
    "  publisher={Cambridge University Press}\n"
    "}"
)
