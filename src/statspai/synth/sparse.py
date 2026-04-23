"""
Sparse Synthetic Control Method.

Implements the L1-penalised (LASSO) approach to synthetic control,
inspired by Microsoft's SparseSC package.  Produces sparse donor
weight vectors — only a handful of donors receive non-zero weight —
which improves interpretability and can reduce over-fitting in panels
with many donors.

Three modes
-----------
* **lasso** (default) — L1-penalised weights, no sum-to-one constraint.
* **constrained_lasso** — L1 penalty + simplex constraints (>= 0, sum = 1).
* **joint** — Joint optimisation of feature weights *V* and donor
  weights *W* (the full SparseSC objective).

References
----------
Amjad, M., Shah, D. and Shen, D. (2018).
"Robust Synthetic Control." *Journal of Machine Learning Research*,
19(22), 1-51.

Microsoft SparseSC (2019). github.com/microsoft/SparseSC.

Doudchenko, N. and Imbens, G.W. (2016).
"Balancing, Regression, Difference-in-Differences and Synthetic
Control Methods: A Synthesis." NBER Working Paper 22791. [@doudchenko2016balancing]
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sparse_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    mode: str = "lasso",
    lambda_w: Optional[float] = None,
    lambda_v: Optional[float] = None,
    covariates: Optional[List[str]] = None,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Sparse Synthetic Control estimator.

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
    mode : {'lasso', 'constrained_lasso', 'joint'}, default 'lasso'
        * ``'lasso'`` — L1-penalised weights, no sum-to-one constraint.
        * ``'constrained_lasso'`` — L1 + non-negativity + sum-to-one.
        * ``'joint'`` — Joint V and W optimisation (full SparseSC).
    lambda_w : float or None
        L1 penalty on donor weights.  ``None`` selects via cross-validation.
    lambda_v : float or None
        L1 penalty on feature weights (``'joint'`` mode only).
        ``None`` selects via cross-validation.
    covariates : list of str, optional
        Additional covariates to append to the pre-treatment outcome
        matrix before weight estimation.
    placebo : bool, default True
        Run in-space placebo permutation for inference.
    alpha : float, default 0.05
        Significance level for confidence interval.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.sparse_synth(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     mode='lasso',
    ... )
    >>> result.summary()
    """
    _VALID_MODES = ("lasso", "constrained_lasso", "joint")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"mode must be one of {_VALID_MODES}, got {mode!r}"
        )

    # ------------------------------------------------------------------
    # 1. Reshape panel
    # ------------------------------------------------------------------
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period.")
    if treated_unit not in panel.index:
        raise ValueError(
            f"treated_unit {treated_unit!r} not found in '{unit}' column."
        )

    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    if len(donors) < 2:
        raise ValueError("Need at least 2 donor units.")

    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)  # (J, T_pre)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)  # (J, T_post)

    # Drop donors with NaN in pre-treatment
    valid_mask = ~np.any(np.isnan(Y0_pre), axis=1)
    if valid_mask.sum() < 2:
        raise ValueError("Fewer than 2 valid donor units (after dropping NaN).")
    Y0_pre = Y0_pre[valid_mask]
    Y0_post = Y0_post[valid_mask]
    donors = [donors[i] for i in range(len(donors)) if valid_mask[i]]
    J = len(donors)

    # ------------------------------------------------------------------
    # 2. Optional covariates — append to the pre-treatment matrix
    # ------------------------------------------------------------------
    if covariates:
        cov_panel = data.pivot_table(index=unit, columns=time, values=covariates)
        # Average covariates over pre-treatment for each unit
        cov_treated = np.array([
            data.loc[
                (data[unit] == treated_unit) & (data[time] < treatment_time),
                covariates,
            ].mean().values
        ]).ravel().astype(np.float64)
        cov_donors = np.array([
            data.loc[
                (data[unit] == d) & (data[time] < treatment_time),
                covariates,
            ].mean().values
            for d in donors
        ]).astype(np.float64)  # (J, K)
        Y1_pre = np.concatenate([Y1_pre, cov_treated])
        Y0_pre = np.hstack([Y0_pre, cov_donors])

    # ------------------------------------------------------------------
    # 3. Select lambda via CV if not provided
    # ------------------------------------------------------------------
    cv_results: Optional[Dict[str, Any]] = None

    if lambda_w is None:
        lambda_w, cv_results = _cv_lambda(Y0_pre, mode=mode)

    if mode == "joint" and lambda_v is None:
        lambda_v, cv_v_results = _cv_lambda(
            Y0_pre, mode="lasso",  # reuse lasso CV for V
        )
        if cv_results is not None:
            cv_results["lambda_v_curve"] = cv_v_results
    elif mode == "joint" and lambda_v is None:
        lambda_v = lambda_w * 0.1  # fallback

    # ------------------------------------------------------------------
    # 4. Solve for weights
    # ------------------------------------------------------------------
    if mode == "lasso":
        weights = _lasso_weights(Y1_pre, Y0_pre, lambda_w)
    elif mode == "constrained_lasso":
        weights = _constrained_lasso_weights(Y1_pre, Y0_pre, lambda_w)
    elif mode == "joint":
        weights, feature_weights = _joint_optimization(
            Y0_pre, Y1_pre, lambda_w, lambda_v,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # ------------------------------------------------------------------
    # 5. Construct synthetic control and compute effects
    # ------------------------------------------------------------------
    Y1_synth_pre = Y0_pre[:, :len(pre_times)].T @ weights   # (T_pre,)
    Y1_synth_post = Y0_post.T @ weights                      # (T_post,)
    Y_synth = np.concatenate([Y1_synth_pre, Y1_synth_post])
    Y_treated = np.concatenate([
        panel.loc[treated_unit, pre_times].values.astype(np.float64),
        Y1_post,
    ])
    gap = Y_treated - Y_synth
    gap_pre = gap[:len(pre_times)]
    gap_post = gap[len(pre_times):]
    att = float(np.mean(gap_post))
    pre_rmspe = float(np.sqrt(np.mean(gap_pre ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap_post ** 2)))

    # ------------------------------------------------------------------
    # 6. Placebo inference
    # ------------------------------------------------------------------
    placebo_atts: List[float] = []
    placebo_pre_mspes: List[float] = []

    if placebo and J >= 2:
        for j in range(J):
            y_p = Y0_pre[j]               # this donor becomes "treated"
            idx = [k for k in range(J) if k != j]
            Y_d_pre = Y0_pre[idx]
            Y_d_post = Y0_post[idx]

            try:
                if mode == "lasso":
                    w_p = _lasso_weights(y_p, Y_d_pre, lambda_w)
                elif mode == "constrained_lasso":
                    w_p = _constrained_lasso_weights(y_p, Y_d_pre, lambda_w)
                else:  # joint
                    w_p, _ = _joint_optimization(
                        Y_d_pre, y_p, lambda_w, lambda_v,
                    )

                synth_pre_p = Y_d_pre[:, :len(pre_times)].T @ w_p
                synth_post_p = Y_d_post.T @ w_p
                actual_pre_p = Y0_pre[j, :len(pre_times)]
                actual_post_p = Y0_post[j]

                gap_pre_p = actual_pre_p - synth_pre_p
                gap_post_p = actual_post_p - synth_post_p
                placebo_atts.append(float(np.mean(gap_post_p)))
                placebo_pre_mspes.append(float(np.mean(gap_pre_p ** 2)))
            except Exception:
                continue

    if len(placebo_atts) > 0:
        pre_mspe_treated = float(np.mean(gap_pre ** 2))
        post_mspe_treated = float(np.mean(gap_post ** 2))
        ratio_treated = (
            post_mspe_treated / pre_mspe_treated
            if pre_mspe_treated > 1e-10 else np.inf
        )
        placebo_ratios = np.array([
            (a ** 2) / m if m > 1e-10 else 0.0
            for a, m in zip(placebo_atts, placebo_pre_mspes)
        ])
        pvalue = float(np.mean(placebo_ratios >= ratio_treated))
        pvalue = max(pvalue, 1.0 / (len(placebo_ratios) + 1))
        se = float(np.std(placebo_atts, ddof=0))
    else:
        pvalue = np.nan
        se = float(np.std(gap_post, ddof=1)) / max(np.sqrt(len(gap_post)), 1)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z_crit * se, att + z_crit * se)

    # ------------------------------------------------------------------
    # 7. Effects by period
    # ------------------------------------------------------------------
    times_all = np.array(all_times)
    effects_df = pd.DataFrame({
        "time": times_all,
        "treated": Y_treated,
        "synthetic": Y_synth,
        "effect": gap,
        "post_treatment": np.array(
            [False] * len(pre_times) + [True] * len(post_times)
        ),
    })

    # ------------------------------------------------------------------
    # 8. Weight summary
    # ------------------------------------------------------------------
    weight_dict = {donors[j]: float(weights[j]) for j in range(J)}
    n_nonzero = int(np.sum(np.abs(weights) > 1e-6))
    sparsity_ratio = float(1.0 - n_nonzero / J) if J > 0 else 0.0

    weight_df = (
        pd.DataFrame({
            "unit": donors,
            "weight": weights,
        })
        .sort_values("weight", ascending=False, key=abs)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 9. Assemble model_info
    # ------------------------------------------------------------------
    model_info: Dict[str, Any] = {
        "mode": mode,
        "lambda_w": lambda_w,
        "lambda_v": lambda_v if mode == "joint" else None,
        "n_donors": J,
        "n_nonzero_weights": n_nonzero,
        "sparsity_ratio": round(sparsity_ratio, 4),
        "n_pre_periods": len(pre_times),
        "n_post_periods": len(post_times),
        "pre_rmspe": round(pre_rmspe, 6),
        "post_rmspe": round(post_rmspe, 6),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "weights": weight_df,
        "weight_dict": weight_dict,
        "effects_by_period": effects_df,
        "gap_table": effects_df,
        "Y_synth": Y_synth,
        "Y_treated": Y_treated,
        "times": times_all,
    }

    if cv_results is not None:
        model_info["cv_results"] = cv_results

    if mode == "joint":
        model_info["feature_weights"] = feature_weights

    if placebo_atts:
        model_info["placebo_atts"] = placebo_atts
        model_info["n_placebos"] = len(placebo_atts)

    mode_labels = {
        "lasso": "Sparse SCM (LASSO)",
        "constrained_lasso": "Sparse SCM (Constrained LASSO)",
        "joint": "Sparse SCM (Joint V-W Optimisation)",
    }

    return CausalResult(
        method=mode_labels[mode],
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=weight_df,
        model_info=model_info,
        _citation_key="sparse_synth",
    )


# ---------------------------------------------------------------------------
# Internal: weight solvers
# ---------------------------------------------------------------------------


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Soft-thresholding (proximal) operator for LASSO."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lambda_pen: float,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Coordinate descent solver for LASSO regression.

    Solves  min_w  0.5 * ||y - X @ w||^2  +  lambda_pen * ||w||_1

    Parameters
    ----------
    X : (T, J) donor matrix (columns = donors).
    y : (T,) treated unit outcome vector.
    lambda_pen : L1 penalty.
    max_iter : maximum iterations.
    tol : convergence tolerance.

    Returns
    -------
    w : (J,) weight vector.
    """
    T, J = X.shape
    w = np.zeros(J, dtype=np.float64)
    r = y.copy()  # residual

    # Precompute column norms
    col_norm_sq = np.sum(X ** 2, axis=0)  # (J,)

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(J):
            if col_norm_sq[j] < 1e-12:
                continue
            # Partial residual: add back j-th contribution
            r += X[:, j] * w[j]
            # Univariate soft-thresholding
            rho_j = X[:, j] @ r
            w[j] = float(_soft_threshold(
                np.array([rho_j / col_norm_sq[j]]),
                lambda_pen / col_norm_sq[j],
            )[0])
            # Update residual
            r -= X[:, j] * w[j]

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def _lasso_weights(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    lambda_w: float,
) -> np.ndarray:
    """
    L1-penalised donor weights without sum-to-one or non-negativity.

    Solves  min_w  0.5 * ||Y1_pre - Y0_pre.T @ w||^2  +  lambda_w * ||w||_1

    Parameters
    ----------
    Y1_pre : (P,) pre-treatment outcomes for treated unit
             (P = T_pre + n_covariates if covariates appended).
    Y0_pre : (J, P) pre-treatment outcomes for donors.
    lambda_w : L1 penalty.

    Returns
    -------
    w : (J,) sparse weight vector.
    """
    # X has shape (P, J), y has shape (P,)
    X = Y0_pre.T
    y = Y1_pre
    return _coordinate_descent(X, y, lambda_w)


def _constrained_lasso_weights(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    lambda_w: float,
) -> np.ndarray:
    """
    L1-penalised weights with simplex constraints (w >= 0, sum(w) = 1).

    Uses SLSQP to solve the constrained problem:
        min_w  0.5 * ||Y1_pre - Y0_pre.T @ w||^2  +  lambda_w * ||w||_1
        s.t.   w_j >= 0,  sum(w) = 1

    Since w >= 0, ||w||_1 = sum(w) = 1, so the L1 penalty is constant
    and acts only through the regularisation path: larger lambda_w
    pushes the solution towards the simplex corner (sparse).
    We keep the penalty in the objective to produce different solutions
    along the regularisation path (the constant shifts the level but
    the gradient still matters for SLSQP).

    Parameters
    ----------
    Y1_pre : (P,) pre-treatment outcomes for treated unit.
    Y0_pre : (J, P) pre-treatment outcomes for donors.
    lambda_w : L1 penalty.

    Returns
    -------
    w : (J,) sparse, non-negative weight vector summing to 1.
    """
    J = Y0_pre.shape[0]
    X = Y0_pre.T  # (P, J)
    y = Y1_pre    # (P,)

    def objective(w: np.ndarray) -> float:
        r = y - X @ w
        return 0.5 * float(r @ r) + lambda_w * float(np.sum(w))

    def gradient(w: np.ndarray) -> np.ndarray:
        return -X.T @ (y - X @ w) + lambda_w * np.ones(J)

    w0 = np.ones(J) / J
    res = optimize.minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * J,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return res.x


def _joint_optimization(
    Y0_pre: np.ndarray,
    Y1_pre: np.ndarray,
    lambda_w: float,
    lambda_v: float,
    max_outer_iter: int = 20,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Joint optimisation of feature weights V and donor weights W.

    Alternating minimisation:
        1. Fix V, solve for w via LASSO.
        2. Fix w, solve for V via LASSO.

    The objective is:
        min_{V,w}  ||V * (Y1_pre - Y0_pre.T @ w)||^2
                   + lambda_w * ||w||_1  + lambda_v * ||V||_1

    Parameters
    ----------
    Y0_pre : (J, P) donor pre-treatment matrix.
    Y1_pre : (P,) treated unit pre-treatment vector.
    lambda_w : L1 penalty on donor weights.
    lambda_v : L1 penalty on feature weights.
    max_outer_iter : max alternating iterations.
    tol : convergence tolerance.

    Returns
    -------
    w : (J,) donor weight vector.
    V : (P,) feature weight vector.
    """
    P = Y1_pre.shape[0]
    J = Y0_pre.shape[0]

    # Initialise V uniformly
    V = np.ones(P, dtype=np.float64)
    w = np.zeros(J, dtype=np.float64)

    for _ in range(max_outer_iter):
        w_old = w.copy()
        V_old = V.copy()

        # --- Step 1: Fix V, solve for w ---
        # Rewrite as LASSO: min_w 0.5 ||diag(V)*(y - X*w)||^2 + lambda_w*||w||_1
        # = min_w 0.5 ||V_diag*y - V_diag*X*w||^2 + lambda_w*||w||_1
        V_diag_y = V * Y1_pre
        V_diag_X = (V[:, np.newaxis] * Y0_pre.T)  # (P, J)
        w = _coordinate_descent(V_diag_X, V_diag_y, lambda_w)

        # --- Step 2: Fix w, solve for V ---
        # residuals per feature dimension
        resid = Y1_pre - Y0_pre.T @ w  # (P,)
        # min_V 0.5 * sum_p (V_p * resid_p)^2 + lambda_v * ||V||_1
        # = min_V 0.5 * sum_p V_p^2 * resid_p^2 + lambda_v * sum_p |V_p|
        # Separable per p: V_p* = soft_threshold(0, lambda_v / resid_p^2)
        # But this is degenerate — V=0 is always a solution.
        # Instead: solve min_V ||V * resid||^2 + lambda_v ||V||_1  s.t. ||V||_2=1
        # Proximal gradient on the sphere:
        resid_sq = resid ** 2
        # Closed form for separable: V_p = max(0, 1 - lambda_v / (2 * resid_p^2))
        # then normalise.
        raw_V = np.maximum(resid_sq - lambda_v * 0.5, 0.0)
        v_norm = np.linalg.norm(raw_V)
        if v_norm > 1e-10:
            V = raw_V / v_norm
        else:
            V = np.ones(P) / np.sqrt(P)

        # Check convergence
        delta_w = np.max(np.abs(w - w_old))
        delta_v = np.max(np.abs(V - V_old))
        if delta_w < tol and delta_v < tol:
            break

    return w, V


# ---------------------------------------------------------------------------
# Internal: cross-validation for lambda selection
# ---------------------------------------------------------------------------


def _cv_lambda(
    Y0_pre: np.ndarray,
    mode: str = "lasso",
    n_lambdas: int = 20,
    lambda_min_ratio: float = 1e-4,
) -> Tuple[float, Dict[str, Any]]:
    """
    Leave-one-donor-out cross-validation for lambda selection.

    For each donor j, treat it as the "treated" unit, estimate weights
    from the remaining donors, and measure prediction MSE.  Average
    over donors and pick the lambda that minimises CV-MSE.

    Parameters
    ----------
    Y0_pre : (J, P) donor pre-treatment matrix.
    mode : weight solver mode.
    n_lambdas : number of lambda candidates.
    lambda_min_ratio : smallest lambda as fraction of lambda_max.

    Returns
    -------
    best_lambda : float
    cv_info : dict with 'lambdas', 'mean_mse', 'se_mse'.
    """
    J, P = Y0_pre.shape

    # Compute lambda_max: the smallest lambda that zeroes out all weights.
    # For LASSO: lambda_max = max_j |X_j^T y| / T  (approx.)
    # Use a representative target (mean of donors).
    y_mean = Y0_pre.mean(axis=0)
    X_all = Y0_pre.T  # (P, J)
    lambda_max = float(np.max(np.abs(X_all.T @ y_mean))) / P
    lambda_max = max(lambda_max, 1e-3)
    lambdas = np.logspace(
        np.log10(lambda_max),
        np.log10(lambda_max * lambda_min_ratio),
        n_lambdas,
    )

    mse_matrix = np.full((J, n_lambdas), np.nan)

    for j in range(J):
        y_j = Y0_pre[j]
        idx = [k for k in range(J) if k != j]
        Y_d = Y0_pre[idx]

        for li, lam in enumerate(lambdas):
            try:
                if mode in ("lasso", "joint"):
                    w = _lasso_weights(y_j, Y_d, lam)
                else:
                    w = _constrained_lasso_weights(y_j, Y_d, lam)

                y_hat = Y_d.T @ w
                mse_matrix[j, li] = float(np.mean((y_j - y_hat) ** 2))
            except Exception:
                continue

    mean_mse = np.nanmean(mse_matrix, axis=0)
    se_mse = np.nanstd(mse_matrix, axis=0) / np.sqrt(
        np.sum(~np.isnan(mse_matrix), axis=0).clip(1)
    )

    # One-SE rule: pick the largest lambda within 1 SE of the minimum
    best_idx = int(np.nanargmin(mean_mse))
    threshold = mean_mse[best_idx] + se_mse[best_idx]
    # Walk from largest lambda (index 0) toward smallest
    chosen_idx = best_idx
    for i in range(best_idx):
        if mean_mse[i] <= threshold:
            chosen_idx = i
            break

    best_lambda = float(lambdas[chosen_idx])

    cv_info = {
        "lambdas": lambdas.tolist(),
        "mean_mse": mean_mse.tolist(),
        "se_mse": se_mse.tolist(),
        "best_idx": chosen_idx,
        "min_mse_idx": best_idx,
        "one_se_rule": chosen_idx != best_idx,
    }

    return best_lambda, cv_info
