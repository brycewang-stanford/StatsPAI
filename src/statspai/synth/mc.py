"""
Matrix Completion for Synthetic Control (MC-SCM).

Treats the treated unit's post-treatment outcomes as missing entries in
a panel matrix Y (N x T) and imputes them via nuclear norm regularisation
(low-rank matrix completion).  The counterfactual is the imputed value;
the treatment effect is observed minus imputed.

Algorithm -- Soft-Impute / SVT
-------------------------------
1. Build Y matrix (units x time).  Mask treated unit's post-treatment entries.
2. Initialise M = row/column means.
3. Iterate:  M = SVT_lambda( P_obs(Y) + P_miss(M) )
   where SVT_lambda soft-thresholds singular values by lambda.
4. Counterfactual for treated = M[treated, post_periods].
5. Effects = Y_observed - M_imputed.

Cross-validation for lambda: hold out random entries from observed cells,
pick lambda minimising reconstruction error.

References
----------
Athey, S., Bayati, M., Doudchenko, N., Imbens, G. and Khosravi, A. (2021).
"Matrix Completion Methods for Causal Panel Data Models."
*Journal of the American Statistical Association*, 116(536), 1716-1730.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ====================================================================== #
#  Public API
# ====================================================================== #

def mc_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    lambda_reg: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    cv_folds: int = 5,
    alpha: float = 0.05,
    placebo: bool = True,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Matrix Completion Synthetic Control Method.

    Imputes the treated unit's post-treatment counterfactual by solving
    a nuclear-norm-penalised matrix completion problem on the full
    panel, following Athey et al. (2021).

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
        Time-varying covariates to partial out before matrix completion.
    lambda_reg : float, optional
        Nuclear norm penalty.  If ``None`` (default), selected
        automatically via cross-validation on observed entries.
    max_iter : int, default 500
        Maximum Soft-Impute iterations.
    tol : float, default 1e-6
        Convergence tolerance (relative change in Frobenius norm).
    cv_folds : int, default 5
        Number of CV folds for automatic lambda selection.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    placebo : bool, default True
        Run placebo (permutation) inference by treating each control
        unit as if it were treated.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    CausalResult
        With ``.estimate`` equal to the average post-treatment effect (ATT),
        period-level effects in ``detail``, and full diagnostics in
        ``model_info``.

    Notes
    -----
    The algorithm uses the Soft-Impute / Singular Value Thresholding (SVT)
    procedure.  At each iteration the current completion is projected onto
    observed entries, combined with the previous imputation at missing
    entries, then rank-reduced by soft-thresholding the singular values.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.mc_synth(df, outcome='gdp', unit='state', time='year',
    ...                      treated_unit='California',
    ...                      treatment_time=1989)
    >>> print(result.summary())
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    #  Build panel matrix
    # ------------------------------------------------------------------ #
    pivot = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(pivot.columns.tolist())
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period.")
    if treated_unit not in pivot.index:
        raise ValueError(f"treated_unit '{treated_unit}' not found in data.")

    donors = [u for u in pivot.index if u != treated_unit]
    all_units = list(pivot.index)
    treated_idx = all_units.index(treated_unit)
    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)
    T_total = len(all_times)

    # Full panel matrix (N x T)
    Y_full = pivot.loc[all_units, all_times].values.astype(np.float64)

    # ------------------------------------------------------------------ #
    #  Handle covariates: partial out via OLS on observed entries
    # ------------------------------------------------------------------ #
    if covariates:
        Y_full = _partial_out_covariates(
            data, outcome, unit, time, covariates,
            all_units, all_times, treated_unit, treatment_time,
        )

    # ------------------------------------------------------------------ #
    #  Observation mask:  1 = observed, 0 = missing (to impute)
    # ------------------------------------------------------------------ #
    obs_mask = np.ones_like(Y_full, dtype=bool)
    post_col_start = T0  # post-treatment columns start at index T0
    obs_mask[treated_idx, post_col_start:] = False

    Y_observed = Y_full[obs_mask]  # kept for reference

    # ------------------------------------------------------------------ #
    #  Auto-select lambda via CV on observed entries
    # ------------------------------------------------------------------ #
    if lambda_reg is None:
        lambda_reg = _cv_lambda(
            Y_full, obs_mask, cv_folds, max_iter, tol, rng,
        )

    # ------------------------------------------------------------------ #
    #  Soft-Impute
    # ------------------------------------------------------------------ #
    M = _soft_impute(Y_full, obs_mask, lambda_reg, max_iter, tol)

    # ------------------------------------------------------------------ #
    #  Extract counterfactual and effects
    # ------------------------------------------------------------------ #
    Y_treated_pre = Y_full[treated_idx, :T0]
    Y_treated_post = Y_full[treated_idx, T0:]
    Y_synth_pre = M[treated_idx, :T0]
    Y_synth_post = M[treated_idx, T0:]

    effects = Y_treated_post - Y_synth_post
    att = float(np.mean(effects))

    pre_residuals = Y_treated_pre - Y_synth_pre
    pre_rmspe = float(np.sqrt(np.mean(pre_residuals ** 2)))

    # Effective rank of completed matrix
    _, S_full, _ = np.linalg.svd(M, full_matrices=False)
    eff_rank = int(np.sum(S_full > 1e-10))

    # ------------------------------------------------------------------ #
    #  Placebo inference
    # ------------------------------------------------------------------ #
    placebo_atts: list[float] = []
    if placebo and J >= 2:
        for j_idx, donor_unit in enumerate(donors):
            d_idx = all_units.index(donor_unit)
            plac_mask = np.ones_like(Y_full, dtype=bool)
            plac_mask[d_idx, post_col_start:] = False

            try:
                M_plac = _soft_impute(
                    Y_full, plac_mask, lambda_reg, max_iter, tol,
                )
                plac_post = Y_full[d_idx, T0:]
                plac_synth = M_plac[d_idx, T0:]
                placebo_atts.append(float(np.mean(plac_post - plac_synth)))
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

    # ------------------------------------------------------------------ #
    #  Build output tables
    # ------------------------------------------------------------------ #
    gap_table = pd.DataFrame({
        "time": all_times,
        "treated": np.concatenate([Y_treated_pre, Y_treated_post]),
        "synthetic": np.concatenate([Y_synth_pre, Y_synth_post]),
        "gap": np.concatenate([pre_residuals, effects]),
        "post_treatment": [False] * T0 + [True] * T1,
    })

    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y_treated_post,
        "counterfactual": Y_synth_post,
        "effect": effects,
    })

    model_info: dict[str, Any] = {
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "rank": eff_rank,
        "lambda_reg": lambda_reg,
        "gap_table": gap_table,
        "Y_synth": np.concatenate([Y_synth_pre, Y_synth_post]),
        "Y_treated": np.concatenate([Y_treated_pre, Y_treated_post]),
        "times": all_times,
        "singular_values": S_full,
        "pre_treatment_rmspe": pre_rmspe,
        "effects_by_period": effects_df,
    }

    if placebo_atts:
        model_info["placebo_atts"] = placebo_atts
        model_info["n_placebos"] = len(placebo_atts)

    return CausalResult(
        method="Matrix Completion SCM (Athey et al. 2021)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info=model_info,
        _citation_key="mc_synth",
    )


# ====================================================================== #
#  Core algorithms
# ====================================================================== #

def _soft_impute(
    Y: np.ndarray,
    obs_mask: np.ndarray,
    lam: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Soft-Impute / SVT algorithm for matrix completion.

    Parameters
    ----------
    Y : ndarray (N, T)
        Panel matrix with all entries filled (missing entries will be
        ignored via *obs_mask*).
    obs_mask : ndarray of bool (N, T)
        True where entries are observed.
    lam : float
        Nuclear norm penalty (singular value threshold).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on relative Frobenius norm change.

    Returns
    -------
    M : ndarray (N, T)
        Completed matrix.
    """
    # Initialise with row and column means on observed entries
    M = _init_from_means(Y, obs_mask)

    for _ in range(max_iter):
        # Fill observed entries from Y, missing from current M
        Z = np.where(obs_mask, Y, M)

        # SVT: soft-threshold singular values
        M_new = _svt(Z, lam)

        # Convergence check
        diff = np.linalg.norm(M_new - M, "fro")
        norm_prev = np.linalg.norm(M, "fro")
        if norm_prev > 0 and diff / norm_prev < tol:
            M = M_new
            break
        M = M_new

    return M


def _svt(Z: np.ndarray, lam: float) -> np.ndarray:
    """Singular Value Thresholding: soft-threshold singular values by lam."""
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    S_thresh = np.maximum(S - lam, 0.0)
    return (U * S_thresh) @ Vt


def _init_from_means(Y: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Initialise missing entries with row + column means of observed data."""
    N, T = Y.shape
    M = Y.copy()

    # Compute row means and column means from observed entries only
    row_sums = np.where(obs_mask, Y, 0.0).sum(axis=1)
    row_counts = obs_mask.sum(axis=1).clip(min=1)
    row_means = row_sums / row_counts

    col_sums = np.where(obs_mask, Y, 0.0).sum(axis=0)
    col_counts = obs_mask.sum(axis=0).clip(min=1)
    col_means = col_sums / col_counts

    grand_mean = np.where(obs_mask, Y, 0.0).sum() / max(obs_mask.sum(), 1)

    # Fill missing entries: row_mean + col_mean - grand_mean
    for i in range(N):
        for j in range(T):
            if not obs_mask[i, j]:
                M[i, j] = row_means[i] + col_means[j] - grand_mean

    return M


# ====================================================================== #
#  Cross-validation for lambda
# ====================================================================== #

def _cv_lambda(
    Y: np.ndarray,
    obs_mask: np.ndarray,
    n_folds: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> float:
    """
    Select nuclear norm penalty via cross-validation on observed entries.

    Holds out random observed entries, runs Soft-Impute on the rest,
    and picks lambda minimising reconstruction MSE.
    """
    # Candidate lambdas: fraction of largest singular value
    Z_init = _init_from_means(Y, obs_mask)
    _, S0, _ = np.linalg.svd(np.where(obs_mask, Y, Z_init), full_matrices=False)
    s_max = S0[0] if len(S0) > 0 else 1.0

    lambdas = np.logspace(
        np.log10(max(s_max * 0.001, 1e-8)),
        np.log10(s_max * 0.5),
        num=15,
    )

    # Get indices of observed entries
    obs_rows, obs_cols = np.where(obs_mask)
    n_obs = len(obs_rows)
    perm = rng.permutation(n_obs)

    fold_size = n_obs // n_folds

    best_lam = float(lambdas[len(lambdas) // 2])
    best_mse = np.inf

    for lam in lambdas:
        mse_total = 0.0
        n_eval = 0

        for f in range(n_folds):
            start = f * fold_size
            end = start + fold_size if f < n_folds - 1 else n_obs
            test_perm = perm[start:end]

            # Build training mask: remove test entries
            train_mask = obs_mask.copy()
            for idx in test_perm:
                train_mask[obs_rows[idx], obs_cols[idx]] = False

            # Run Soft-Impute on training set
            try:
                M_cv = _soft_impute(Y, train_mask, lam, max_iter=100, tol=tol)
            except Exception:
                mse_total += 1e10
                n_eval += 1
                continue

            # MSE on held-out entries
            fold_mse = 0.0
            for idx in test_perm:
                r, c = obs_rows[idx], obs_cols[idx]
                fold_mse += (Y[r, c] - M_cv[r, c]) ** 2
            mse_total += fold_mse / len(test_perm)
            n_eval += 1

        avg_mse = mse_total / max(n_eval, 1)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_lam = float(lam)

    return best_lam


# ====================================================================== #
#  Covariate adjustment
# ====================================================================== #

def _partial_out_covariates(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    covariates: List[str],
    all_units: list,
    all_times: list,
    treated_unit: Any,
    treatment_time: Any,
) -> np.ndarray:
    """
    Partial out covariates via OLS on observed (control + pre) entries.

    Returns the residualised panel matrix (N x T) as a numpy array
    ordered by ``all_units`` x ``all_times``.
    """
    # Fit OLS on control observations + treated pre-treatment
    mask_ctrl = data[unit] != treated_unit
    mask_pre = data[time] < treatment_time
    fit_mask = mask_ctrl | mask_pre

    fit_data = data.loc[fit_mask]
    X = fit_data[covariates].values.astype(np.float64)
    y = fit_data[outcome].values.astype(np.float64)

    XtX = X.T @ X + 1e-8 * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ y)

    # Residualise the full dataset
    data_res = data.copy()
    X_all = data_res[covariates].values.astype(np.float64)
    data_res[outcome] = data_res[outcome].values - X_all @ beta

    pivot = data_res.pivot_table(index=unit, columns=time, values=outcome)
    return pivot.loc[all_units, all_times].values.astype(np.float64)


# ====================================================================== #
#  Citation
# ====================================================================== #

CausalResult._CITATIONS["mc_synth"] = (
    "@article{athey2021matrix,\n"
    "  title={Matrix Completion Methods for Causal Panel Data Models},\n"
    "  author={Athey, Susan and Bayati, Mohsen and Doudchenko, Nikolay\n"
    "          and Imbens, Guido and Khosravi, Azeem},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1716--1730},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
