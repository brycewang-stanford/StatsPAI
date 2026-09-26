"""
Matrix Completion for Synthetic Control (MC-SCM).

Treats the treated unit's post-treatment outcomes as missing entries in
a panel matrix Y (N x T) and imputes them via nuclear norm regularisation
(low-rank matrix completion).  The counterfactual is the imputed value;
the treatment effect is observed minus imputed.

Algorithm -- Soft-Impute / SVT with unpenalised fixed effects
--------------------------------------------------------------
1. Build Y matrix (units x time).  Mask treated unit's post-treatment entries.
2. Solve  min 1/2 ||P_obs(Y - a 1' - 1 b' - L)||^2 + lambda ||L||_*
   (``fixed_effects='two-way'``; see ``statspai.matrix_completion._core``)
   by soft-impute: fill missing cells with the current fit, double-centre,
   soft-threshold the singular values by lambda.
3. Counterfactual for treated = fit[treated, post_periods].
4. Effects = Y_observed - fit.

Cross-validation for lambda: hold out random entries from observed cells,
pick lambda minimising reconstruction error.

References
----------
[@athey2021matrix]
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from ..matrix_completion._core import _center, _check_fixed_effects, mc_nnm_fit
from ._core import placebo_rank_pvalue

# ====================================================================== #
#  Public API
# ====================================================================== #


@accepts_aliases(id="unit", y="outcome")
def mc_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    lambda_reg: Optional[float] = None,
    max_iter: int = 5000,
    tol: float = 1e-10,
    cv_folds: int = 5,
    alpha: float = 0.05,
    placebo: bool = True,
    seed: Optional[int] = None,
    fixed_effects: str = "two-way",
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
        Singular-value threshold on the ``1/2``-scaled squared loss.  The
        same minimiser is R ``MCPanel::mcnnm_fit(lambda_L = 2 * lambda /
        n_observed)`` and ``fect(method = "mc", CV = FALSE, lambda =
        lambda / (N * T))`` (both reported in ``model_info``).  If
        ``None`` (default), selected by this function's own random-entry
        K-fold cross-validation (not the ``MCPanel`` / ``fect`` CV rule).
    max_iter : int, default 5000
        Maximum Soft-Impute iterations (``RuntimeWarning`` if hit).
    tol : float, default 1e-10
        Convergence tolerance (relative change in Frobenius norm of the
        fitted matrix).
    cv_folds : int, default 5
        Number of CV folds for automatic lambda selection.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    placebo : bool, default True
        Run placebo (permutation) inference by treating each control
        unit as if it were treated.
    seed : int, optional
        Random seed for the CV fold assignment. With ``seed=None`` and
        ``lambda_reg=None`` the selected lambda (hence the estimate) can
        differ between calls.
    fixed_effects : {"two-way", "unit", "time", "none"}, default "two-way"
        Unpenalised additive effects fitted with the low-rank part.
        ``"two-way"`` is the Athey et al. (2021) estimator and the default
        of R ``MCPanel`` and ``fect``; ``"none"`` is pure soft-impute (the
        estimator of StatsPAI <= 1.28.0), which shrinks the outcome level
        towards zero.

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
    >>> df = sp.california_prop99()
    >>> result = sp.mc_synth(df, outcome='packspercapita', unit='state',
    ...                      time='year', treated_unit='California',
    ...                      treatment_time=1989, placebo=False, seed=0)
    >>> result.method, result.estimand
    ('Matrix Completion SCM (Athey et al. 2021)', 'ATT')
    >>> bool(result.estimate < 0)   # Prop 99 reduced cigarette sales
    True
    """
    rng = np.random.default_rng(seed)
    _check_fixed_effects(fixed_effects)

    # ------------------------------------------------------------------ #
    #  Build panel matrix
    # ------------------------------------------------------------------ #
    pivot = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(pivot.columns.tolist())
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")  # pragma: no cover
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period.")  # pragma: no cover
    if treated_unit not in pivot.index:
        raise ValueError(
            f"treated_unit '{treated_unit}' not found in data."
        )  # pragma: no cover

    donors = [u for u in pivot.index if u != treated_unit]
    all_units = list(pivot.index)
    treated_idx = all_units.index(treated_unit)
    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    # Full panel matrix (N x T)
    Y_full = pivot.loc[all_units, all_times].values.astype(np.float64)

    # ------------------------------------------------------------------ #
    #  Handle covariates: partial out via OLS on observed entries
    # ------------------------------------------------------------------ #
    if covariates:
        Y_full = _partial_out_covariates(
            data,
            outcome,
            unit,
            time,
            covariates,
            all_units,
            all_times,
            treated_unit,
            treatment_time,
        )

    # ------------------------------------------------------------------ #
    #  Observation mask:  1 = observed, 0 = missing (to impute)
    # ------------------------------------------------------------------ #
    obs_mask = np.isfinite(Y_full)
    post_col_start = T0  # post-treatment columns start at index T0
    obs_mask[treated_idx, post_col_start:] = False
    if not np.isfinite(Y_full[treated_idx, :T0]).all():
        raise MethodIncompatibility(
            "The treated unit has missing pre-treatment outcomes."
        )

    # ------------------------------------------------------------------ #
    #  Auto-select lambda via CV on observed entries
    # ------------------------------------------------------------------ #
    if lambda_reg is None:
        lambda_reg = _cv_lambda(
            Y_full,
            obs_mask,
            cv_folds,
            max_iter,
            tol,
            rng,
            fixed_effects,
        )
    lambda_reg = float(lambda_reg)

    # ------------------------------------------------------------------ #
    #  Soft-Impute
    # ------------------------------------------------------------------ #
    sol = _soft_impute(Y_full, obs_mask, lambda_reg, max_iter, tol, fixed_effects)
    M = sol["fit"]

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
    pre_rmspe = float(np.sqrt(np.mean(pre_residuals**2)))

    # Rank of the penalised low-rank component
    S_full = sol["singular_values"]
    eff_rank = int(np.sum(S_full > 1e-10))

    # ------------------------------------------------------------------ #
    #  Placebo inference
    # ------------------------------------------------------------------ #
    placebo_atts: list[float] = []
    if placebo and J >= 2:
        for j_idx, donor_unit in enumerate(donors):
            d_idx = all_units.index(donor_unit)
            plac_mask = np.isfinite(Y_full)
            plac_mask[d_idx, post_col_start:] = False
            if not np.isfinite(Y_full[d_idx]).all():
                continue  # donor with gaps: its placebo gap is undefined
            M_plac = _soft_impute(
                Y_full, plac_mask, lambda_reg, max_iter, tol, fixed_effects
            )["fit"]
            plac_post = Y_full[d_idx, T0:]
            plac_synth = M_plac[d_idx, T0:]
            placebo_atts.append(float(np.mean(plac_post - plac_synth)))

    if len(placebo_atts) > 0:
        se = float(np.std(placebo_atts, ddof=1))
        pvalue = placebo_rank_pvalue(abs(att), np.abs(placebo_atts))
    else:
        se = float(np.std(effects)) / max(np.sqrt(T1), 1)
        pvalue = np.nan

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z_crit * se, att + z_crit * se)

    # ------------------------------------------------------------------ #
    #  Build output tables
    # ------------------------------------------------------------------ #
    gap_table = pd.DataFrame(
        {
            "time": all_times,
            "treated": np.concatenate([Y_treated_pre, Y_treated_post]),
            "synthetic": np.concatenate([Y_synth_pre, Y_synth_post]),
            "gap": np.concatenate([pre_residuals, effects]),
            "post_treatment": [False] * T0 + [True] * T1,
        }
    )

    effects_df = pd.DataFrame(
        {
            "time": post_times,
            "treated": Y_treated_post,
            "counterfactual": Y_synth_post,
            "effect": effects,
        }
    )

    model_info: dict[str, Any] = {
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "rank": eff_rank,
        "lambda_reg": lambda_reg,
        "lambda_mcpanel": 2.0 * lambda_reg / int(obs_mask.sum()),
        "lambda_fect": lambda_reg / Y_full.size,
        "fixed_effects": fixed_effects,
        "converged": bool(sol["converged"]),
        "n_iter": int(sol["n_iter"]),
        "completed_matrix": M,
        "low_rank_matrix": sol["L"],
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
    fixed_effects: str = "two-way",
    warn: bool = True,
) -> dict:
    """Nuclear-norm completion via the shared ``matrix_completion`` solver.

    Starts from the row + column means of the observed entries.
    """
    return mc_nnm_fit(
        Y,
        obs_mask,
        lam,
        fixed_effects=fixed_effects,
        max_iter=max_iter,
        tol=tol,
        init=_init_from_means(np.where(obs_mask, Y, 0.0), obs_mask),
        warn=warn,
    )


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
    fixed_effects: str = "two-way",
) -> float:
    """
    Select nuclear norm penalty via cross-validation on observed entries.

    Holds out random observed entries, runs Soft-Impute on the rest,
    and picks lambda minimising reconstruction MSE.
    """
    # Candidate lambdas: fraction of the largest singular value of the
    # mean-filled matrix after removing the (unpenalised) fixed effects
    Y = np.where(obs_mask, Y, 0.0)
    Z_init = _init_from_means(Y, obs_mask)
    _, S0, _ = np.linalg.svd(
        _center(np.where(obs_mask, Y, Z_init), fixed_effects), full_matrices=False
    )
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

            # Run Soft-Impute on training set (capped at 100 iterations)
            M_cv = _soft_impute(
                Y, train_mask, lam, 100, tol, fixed_effects, warn=False
            )["fit"]

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
    return np.asarray(pivot.loc[all_units, all_times].values.astype(np.float64))


# ====================================================================== #
#  Citation
# ====================================================================== #

CausalResult._CITATIONS["mc_synth"] = (
    "@article{athey2021matrix,\n"
    "  title={Matrix Completion Methods for Causal Panel Data Models},\n"
    "  author={Athey, Susan and Bayati, Mohsen and Doudchenko, Nikolay\n"
    "          and Imbens, Guido and Khosravi, Khashayar},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1716--1730},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
