"""
Augmented Synthetic Control Method (ASCM).

Combines the synthetic control estimator with an outcome model (ridge
regression) to correct for imperfect pre-treatment fit — reducing bias
when the standard SCM cannot perfectly match the treated unit.

Model
-----
τ̂_ascm = τ̂_scm + (Y₀_post - X₀_post β̂) ⊤ γ̂

where γ̂ are the SCM weights, β̂ is a ridge estimator on pre-treatment
donor data, and the correction term adjusts for remaining pre-treatment
imbalance.

References
----------
Ben-Michael, E., Feller, A. and Rothstein, J. (2021).
"The Augmented Synthetic Control Method."
*Journal of the American Statistical Association*, 116(536), 1789-1803.

Ben-Michael, E., Feller, A. and Rothstein, J. (2022).
"Synthetic Controls with Staggered Adoption."
*Journal of the Royal Statistical Society: Series B*, 84(2), 351-381.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize

from ..core.results import CausalResult


def augsynth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit,
    treatment_time,
    covariates: Optional[List[str]] = None,
    ridge_lambda: Optional[float] = None,
    placebo: bool = True,
    alpha: float = 0.05,
    **kwargs,
) -> CausalResult:
    """
    Augmented Synthetic Control Method (Ben-Michael, Feller & Rothstein 2021).

    Fits a standard SCM then adds a ridge-outcome-model bias correction.
    Per-period correction is

        bias(t) = m̂_t(X1_pre) − Σ_j γ_j m̂_t(X_j,pre),

    where m̂_t is a ridge regression of donor post-period outcomes on
    donor pre-period outcomes. Collapses to standard SCM when
    ``ridge_lambda → ∞`` and to pure outcome-model imputation when
    ``ridge_lambda → 0``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    outcome, unit, time : str
        Column names.
    treated_unit, treatment_time : scalar
        Treated-unit identifier and first treatment period.
    covariates : list of str, optional
        Additional predictors (currently informational; main adjustment
        comes from pre-treatment outcomes).
    ridge_lambda : float, optional
        Ridge penalty. When ``None``, selected by leave-one-donor-out CV.
    placebo : bool, default True
        Run in-space placebo permutation tests for SE / p-value.
    alpha : float, default 0.05
        Significance level.
    **kwargs
        Ignored — accepted for dispatcher compatibility.

    Returns
    -------
    CausalResult
        ``detail`` has one row per post-treatment period with columns
        ``time, treated, counterfactual, effect``. ``model_info`` includes
        ``pre_rmspe, post_rmspe, weights, ridge_lambda, n_donors,
        n_pre_periods, n_post_periods, placebo_distribution``.

    References
    ----------
    Ben-Michael, E., Feller, A. and Rothstein, J. (2021). "The Augmented
    Synthetic Control Method." *JASA*, 116(536), 1789-1803.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.synth.california_tobacco()
    >>> result = sp.augsynth(df, outcome='cigsale', unit='state', time='year',
    ...                       treated_unit='California', treatment_time=1989)
    >>> print(result.summary())
    """
    # --- Reshape to wide format ---
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period")

    # Treated and donor matrices
    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)  # (J, T0)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    # --- Step 1: Standard SCM weights ---
    gamma = _scm_weights(Y1_pre, Y0_pre)

    # --- Step 2: Ridge outcome model ---
    # Fit ridge of donor post-outcomes on donor pre-outcomes:
    #   Y0_post = X β + ε,  X = Y0_pre (J × T0),  β ∈ R^{T0 × T1}
    # Closed-form:  β = (X'X + λI)^{-1} X' Y0_post.
    if ridge_lambda is None:
        ridge_lambda = _cv_ridge_lambda_bias(Y0_pre, Y0_post)

    beta = _ridge_post_coef(Y0_pre, Y0_post, ridge_lambda)   # (T0, T1)

    # --- Step 3: Augmented estimate (Ben-Michael et al. 2021 Eq. 3) ---
    Y1_hat_scm_pre = Y0_pre.T @ gamma    # (T0,)
    Y1_hat_scm_post = Y0_post.T @ gamma  # (T1,)

    pre_residual_scm = Y1_pre - Y1_hat_scm_pre           # (T0,)
    pre_rmspe = float(np.sqrt(np.mean(pre_residual_scm ** 2)))

    # Per-period bias correction:
    #   bias(t) = m̂_t(X1_pre) − Σ_j γ_j m̂_t(X_j,pre)
    #          = (Y1_pre − Y0_pre'γ) @ β_t
    #          = pre_residual_scm @ β[:, t]
    bias_per_period = pre_residual_scm @ beta            # (T1,)
    Y1_hat_aug_post = Y1_hat_scm_post + bias_per_period  # (T1,)

    effects = Y1_post - Y1_hat_aug_post
    att = float(np.mean(effects))

    # --- Inference via placebo permutation ---
    if placebo:
        placebo_effects = []
        for j in range(J):
            other_idx = [i for i in range(J) if i != j]
            Y_plac_pre = Y0_pre[j]
            Y_plac_post = Y0_post[j]
            Y_others_pre = Y0_pre[other_idx]
            Y_others_post = Y0_post[other_idx]

            g_plac = _scm_weights(Y_plac_pre, Y_others_pre)
            plac_pre_hat = Y_others_pre.T @ g_plac
            plac_post_hat = Y_others_post.T @ g_plac

            beta_plac = _ridge_post_coef(
                Y_others_pre, Y_others_post, ridge_lambda
            )
            plac_residual = Y_plac_pre - plac_pre_hat
            plac_bias = plac_residual @ beta_plac

            plac_eff = float(np.mean(
                Y_plac_post - plac_post_hat - plac_bias
            ))
            placebo_effects.append(plac_eff)

        placebo_effects = np.array(placebo_effects)
        se = float(np.std(placebo_effects, ddof=1))
        pvalue = float(np.mean(np.abs(placebo_effects) >= abs(att)))
        pvalue = max(pvalue, 1 / (J + 1))

        t_crit = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (att - t_crit * se, att + t_crit * se)
    else:
        placebo_effects = np.array([])
        se = float("nan")
        pvalue = float("nan")
        ci = (float("nan"), float("nan"))

    # Build period-level results
    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y1_post,
        "counterfactual": Y1_hat_aug_post,
        "effect": effects,
    })

    return CausalResult(
        method="Augmented Synthetic Control (ASCM)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info={
            "model_type": "Synthetic Control (Augmented)",
            "pre_rmspe": pre_rmspe,
            "post_rmspe": float(np.sqrt(np.mean(effects ** 2))),
            "weights": dict(zip(donors, gamma)),
            "n_donors": J,
            "n_pre_periods": T0,
            "n_post_periods": T1,
            "ridge_lambda": ridge_lambda,
            "effects_by_period": effects_df,
            "placebo_distribution": placebo_effects,
        },
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _scm_weights(Y1_pre: np.ndarray, Y0_pre: np.ndarray) -> np.ndarray:
    """
    Standard SCM: find non-negative weights that minimise
    ||Y1_pre - Y0_pre' γ||² subject to sum(γ) = 1, γ >= 0.
    """
    from ._core import solve_simplex_weights
    return solve_simplex_weights(Y1_pre, Y0_pre.T)


def _ridge_post_coef(
    Y0_pre: np.ndarray,
    Y0_post: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Ridge regression coefficients for the outcome model used in the
    Ben-Michael et al. (2021) ASCM bias correction.

    Fits β so that Y0_post ≈ Y0_pre @ β using Tikhonov-regularised OLS,
    with X ≡ Y0_pre treated as a (J, T0) design matrix and Y ≡ Y0_post
    treated as a (J, T1) multi-output target.

    Closed-form: β = (X'X + λ I_{T0})^{-1} X' Y0_post.

    Parameters
    ----------
    Y0_pre : (J, T0) donor pre-treatment outcomes.
    Y0_post : (J, T1) donor post-treatment outcomes.
    lam : non-negative ridge penalty.

    Returns
    -------
    beta : (T0, T1) coefficient matrix.
    """
    T0 = Y0_pre.shape[1]
    A = Y0_pre.T @ Y0_pre + lam * np.eye(T0)
    rhs = Y0_pre.T @ Y0_post
    return np.linalg.solve(A, rhs)


def _cv_ridge_lambda_bias(
    Y0_pre: np.ndarray,
    Y0_post: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
) -> float:
    """
    Leave-one-donor-out CV to pick the ridge penalty for the ASCM
    outcome model m̂: Y0_pre → Y0_post.
    """
    if lambdas is None:
        lambdas = np.logspace(-3, 3, 20)

    J = Y0_pre.shape[0]
    best_lam = 1.0
    best_mse = np.inf

    for lam in lambdas:
        mse = 0.0
        for j in range(J):
            idx = [i for i in range(J) if i != j]
            X_tr = Y0_pre[idx]
            Y_tr = Y0_post[idx]
            try:
                beta = _ridge_post_coef(X_tr, Y_tr, lam)
                pred = Y0_pre[j] @ beta      # (T1,)
                mse += float(np.mean((Y0_post[j] - pred) ** 2))
            except np.linalg.LinAlgError:
                mse += 1e10
        mse /= J
        if mse < best_mse:
            best_mse = mse
            best_lam = float(lam)

    return best_lam
