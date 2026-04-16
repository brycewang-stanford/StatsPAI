"""
Penalized Synthetic Control Method (Abadie & L'Hour 2021).

Implements the penalized SCM estimator from:

    Abadie, A. and L'Hour, J. (2021).
    "A Penalized Synthetic Control Estimator for Disaggregated Data."
    *Journal of the American Statistical Association*, 116(536), 1817-1834.

The key innovation over standard ridge SCM is a **pairwise discrepancy
penalty**: donors that are far from the treated unit in covariate space
receive small weights even if they help fit the pre-treatment outcome.

Optimization problem:

    min_w  ||Y1_pre - Y0_pre.T @ w||^2  +  lambda * sum_j w_j * d(X1, Xj)^2
    s.t.   w >= 0,  sum(w) = 1

Three penalty variants are supported:

* ``pairwise`` (default) — Abadie & L'Hour original.
* ``max_dev`` — penalise maximum pairwise deviation.
* ``l1_pairwise`` — L1 version of pairwise distances.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def penalized_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    lambda_pen: Optional[float] = None,
    penalty_type: str = "pairwise",
    predictors: Optional[List[str]] = None,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Penalized Synthetic Control estimator (Abadie & L'Hour 2021).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data with columns for *unit*, *time*, *outcome*,
        and optionally covariates / predictors.
    outcome : str
        Name of the outcome column.
    unit : str
        Name of the unit identifier column.
    time : str
        Name of the time period column.
    treated_unit
        Identifier of the treated unit.
    treatment_time
        First treatment period (inclusive).
    covariates : list of str, optional
        Covariate columns used **only** for the pairwise distance penalty.
        When ``None`` the pre-treatment outcome values are used as the
        covariate vector for distance computation.
    lambda_pen : float, optional
        Penalty parameter.  ``None`` (default) triggers automatic
        selection via rolling-origin cross-validation on pre-treatment
        periods.
    penalty_type : {'pairwise', 'max_dev', 'l1_pairwise'}, default 'pairwise'
        Penalty functional form.

        * ``'pairwise'`` — ``sum_j w_j * ||X1 - Xj||^2``
          (Abadie & L'Hour original).
        * ``'max_dev'`` — ``max_j { w_j * ||X1 - Xj||^2 }``.
        * ``'l1_pairwise'`` — ``sum_j w_j * ||X1 - Xj||_1``.
    predictors : list of str, optional
        Columns whose pre-treatment *averages* are appended to the
        covariate vector for distance computation.
    placebo : bool, default True
        Run in-space placebo permutation tests.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        With ``detail`` set to the effects-by-period DataFrame.

    References
    ----------
    Abadie, A. and L'Hour, J. (2021). "A Penalized Synthetic Control
    Estimator for Disaggregated Data." *Journal of the American
    Statistical Association*, 116(536), 1817-1834.
    """
    _VALID_PENALTIES = ("pairwise", "max_dev", "l1_pairwise")
    if penalty_type not in _VALID_PENALTIES:
        raise ValueError(
            f"penalty_type must be one of {_VALID_PENALTIES}, "
            f"got {penalty_type!r}"
        )

    # ------------------------------------------------------------------
    # 1. Reshape panel
    # ------------------------------------------------------------------
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError(
            f"Need at least 2 pre-treatment periods, got {len(pre_times)}."
        )
    if len(post_times) == 0:
        raise ValueError("No post-treatment periods found.")
    if treated_unit not in panel.index:
        raise ValueError(f"treated_unit {treated_unit!r} not in data.")

    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    if len(donors) < 2:
        raise ValueError("Need at least 2 donor units.")

    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)   # (J, T0)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    # ------------------------------------------------------------------
    # 2. Covariate matrix for distance computation
    # ------------------------------------------------------------------
    X_treated, X_donors = _build_covariate_matrix(
        data, panel, outcome, unit, time, treated_unit, donors,
        pre_times, covariates, predictors,
    )

    # ------------------------------------------------------------------
    # 3. Pairwise distances
    # ------------------------------------------------------------------
    distances = _compute_pairwise_distances(
        X_treated, X_donors, penalty_type,
    )

    # ------------------------------------------------------------------
    # 4. Select lambda via CV if not provided
    # ------------------------------------------------------------------
    cv_results: Optional[Dict[str, Any]] = None
    if lambda_pen is None:
        lambda_pen, cv_results = _cv_lambda(
            Y1_pre, Y0_pre, distances, penalty_type,
        )

    # ------------------------------------------------------------------
    # 5. Solve for penalized weights
    # ------------------------------------------------------------------
    weights = _penalized_weights(
        Y1_pre, Y0_pre, distances, lambda_pen, penalty_type,
    )

    # ------------------------------------------------------------------
    # 6. Compute effects
    # ------------------------------------------------------------------
    Y_synth_pre = Y0_pre.T @ weights  # (T0,)
    Y_synth_post = Y0_post.T @ weights  # (T1,)

    gap_pre = Y1_pre - Y_synth_pre
    gap_post = Y1_post - Y_synth_post
    att = float(np.mean(gap_post))

    pre_rmspe = float(np.sqrt(np.mean(gap_pre ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap_post ** 2)))
    ratio_treated = post_rmspe / pre_rmspe if pre_rmspe > 1e-10 else np.inf

    # Effects by period
    effects_df = pd.DataFrame({
        "time": list(pre_times) + list(post_times),
        "treated": np.concatenate([Y1_pre, Y1_post]),
        "synthetic": np.concatenate([Y_synth_pre, Y_synth_post]),
        "effect": np.concatenate([gap_pre, gap_post]),
        "post_treatment": [False] * len(pre_times) + [True] * len(post_times),
    })

    # ------------------------------------------------------------------
    # 7. Placebo inference
    # ------------------------------------------------------------------
    se: float
    pvalue: float
    placebo_info: Dict[str, Any] = {}

    if placebo and len(donors) >= 2:
        placebo_info = _run_placebos(
            panel, donors, treated_unit, pre_times, post_times,
            data, outcome, unit, time, covariates, predictors,
            lambda_pen, penalty_type,
        )
        placebo_atts = placebo_info["atts"]
        placebo_ratios = np.array(placebo_info["ratios"])

        pvalue = float(np.mean(placebo_ratios >= ratio_treated))
        pvalue = max(pvalue, 1.0 / (len(placebo_ratios) + 1))
        se = float(np.std(placebo_atts)) if len(placebo_atts) > 1 else 0.0
    else:
        pvalue = np.nan
        se = float(np.std(gap_post)) / max(np.sqrt(len(gap_post)), 1.0)

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci = (att - z_crit * se, att + z_crit * se)

    # ------------------------------------------------------------------
    # 8. Build weight / distance dicts
    # ------------------------------------------------------------------
    weight_dict = {
        d: float(w) for d, w in zip(donors, weights) if w > 1e-6
    }
    distance_dict = {d: float(v) for d, v in zip(donors, distances)}

    # ------------------------------------------------------------------
    # 9. model_info
    # ------------------------------------------------------------------
    model_info: Dict[str, Any] = {
        "weights": weight_dict,
        "pairwise_distances": distance_dict,
        "lambda_pen": lambda_pen,
        "cv_results": cv_results,
        "pre_rmspe": round(pre_rmspe, 6),
        "post_rmspe": round(post_rmspe, 6),
        "effects_by_period": effects_df,
        "n_donors": len(donors),
        "n_pre_periods": len(pre_times),
        "n_post_periods": len(post_times),
        "penalty_type": penalty_type,
        "treated_unit": treated_unit,
        "treatment_time": treatment_time,
        "donor_units": donors,
        "Y_treated": np.concatenate([Y1_pre, Y1_post]),
        "Y_synth": np.concatenate([Y_synth_pre, Y_synth_post]),
        "times": all_times,
        "treated_ratio": ratio_treated,
    }

    if placebo_info:
        model_info["placebo_atts"] = placebo_info["atts"]
        model_info["placebo_ratios"] = placebo_info["ratios"]
        model_info["placebo_gaps"] = placebo_info["gaps"]
        model_info["placebo_units"] = placebo_info["units"]
        model_info["n_placebos"] = len(placebo_info["atts"])

    return CausalResult(
        method="Penalized Synthetic Control (Abadie & L'Hour 2021)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info=model_info,
        _citation_key="penscm",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_covariate_matrix(
    data: pd.DataFrame,
    panel: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    donors: List[Any],
    pre_times: List[Any],
    covariates: Optional[List[str]],
    predictors: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build covariate vectors for treated and each donor.

    Returns (X_treated, X_donors) where X_treated is shape (K,) and
    X_donors is shape (J, K).
    """
    parts_treated: List[np.ndarray] = []
    parts_donors: List[np.ndarray] = []

    if covariates is not None and len(covariates) > 0:
        # Use pre-treatment averages of each covariate
        pre_data = data[data[time] < data[time].max()]  # fallback
        pre_data = data[data[time].isin(pre_times)]

        for cov in covariates:
            if cov not in data.columns:
                raise ValueError(f"Covariate {cov!r} not found in data.")
            cov_means = pre_data.groupby(unit)[cov].mean()
            parts_treated.append(np.array([cov_means.loc[treated_unit]]))
            parts_donors.append(
                np.array([cov_means.loc[d] for d in donors]).reshape(-1, 1)
            )
    else:
        # Default: use pre-treatment outcome values as covariates
        parts_treated.append(
            panel.loc[treated_unit, pre_times].values.astype(np.float64)
        )
        parts_donors.append(
            panel.loc[donors, pre_times].values.astype(np.float64)
        )

    if predictors is not None and len(predictors) > 0:
        pre_data = data[data[time].isin(pre_times)]
        for pred in predictors:
            if pred not in data.columns:
                raise ValueError(f"Predictor {pred!r} not found in data.")
            pred_means = pre_data.groupby(unit)[pred].mean()
            parts_treated.append(np.array([pred_means.loc[treated_unit]]))
            parts_donors.append(
                np.array([pred_means.loc[d] for d in donors]).reshape(-1, 1)
            )

    X_treated = np.concatenate(parts_treated).astype(np.float64)
    X_donors = np.hstack(parts_donors).astype(np.float64) if len(parts_donors) > 1 \
        else parts_donors[0].astype(np.float64)

    # Ensure X_donors is 2-D (J, K)
    if X_donors.ndim == 1:
        X_donors = X_donors.reshape(-1, 1)

    return X_treated, X_donors


def _compute_pairwise_distances(
    X_treated: np.ndarray,
    X_donors: np.ndarray,
    penalty_type: str,
) -> np.ndarray:
    """Compute distances between treated unit and each donor.

    Parameters
    ----------
    X_treated : (K,) array
    X_donors : (J, K) array
    penalty_type : str

    Returns
    -------
    distances : (J,) array of non-negative distances.
    """
    diff = X_donors - X_treated[np.newaxis, :]  # (J, K)

    if penalty_type in ("pairwise", "max_dev"):
        # Squared L2 distance
        distances = np.sum(diff ** 2, axis=1)
    elif penalty_type == "l1_pairwise":
        # L1 distance
        distances = np.sum(np.abs(diff), axis=1)
    else:
        distances = np.sum(diff ** 2, axis=1)

    return distances


def _penalized_weights(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    distances: np.ndarray,
    lambda_pen: float,
    penalty_type: str,
) -> np.ndarray:
    """Solve the penalized SCM quadratic program.

    Parameters
    ----------
    Y1_pre : (T0,) treated pre-treatment outcomes
    Y0_pre : (J, T0) donor pre-treatment outcomes
    distances : (J,) pairwise distances
    lambda_pen : float, penalty parameter
    penalty_type : str

    Returns
    -------
    w : (J,) optimal weights, non-negative, sum to 1.
    """
    J = Y0_pre.shape[0]

    if penalty_type == "max_dev":
        return _penalized_weights_max_dev(
            Y1_pre, Y0_pre, distances, lambda_pen,
        )

    # ----- Standard QP for pairwise / l1_pairwise -----
    # Objective:  ||Y1 - Y0.T w||^2  +  lambda * d.T w
    #           = w.T (Y0 Y0.T) w  - 2 (Y0 Y1).T w + const  + lambda d.T w
    #
    # scipy.optimize.minimize with SLSQP handles this well for moderate J.

    # Quadratic term: H = Y0 @ Y0.T  (J x J)
    H = Y0_pre @ Y0_pre.T  # (J, J)

    # Linear term: c = -Y0 @ Y1 + 0.5 * lambda * d
    c = -Y0_pre @ Y1_pre + 0.5 * lambda_pen * distances

    def objective(w: np.ndarray) -> float:
        return float(w @ H @ w + 2.0 * c @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * H @ w + 2.0 * c

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * J
    w0 = np.ones(J) / J

    result = optimize.minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = result.x
    # Project onto simplex (numerical cleanup)
    w = np.maximum(w, 0.0)
    w /= w.sum() if w.sum() > 0 else 1.0
    return w


def _penalized_weights_max_dev(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    distances: np.ndarray,
    lambda_pen: float,
) -> np.ndarray:
    """Solve the max-deviation penalized SCM problem.

    Objective: ||Y1 - Y0.T w||^2 + lambda * max_j { w_j * d_j }

    We reformulate with an auxiliary variable t >= w_j * d_j for all j:

        min  ||Y1 - Y0.T w||^2 + lambda * t
        s.t. t >= w_j * d_j   for all j
             w >= 0, sum(w) = 1

    This is a QP with linear inequality constraints, solved via SLSQP
    on the joint variable z = [w; t].
    """
    J = Y0_pre.shape[0]
    H = Y0_pre @ Y0_pre.T  # (J, J)
    q = -Y0_pre @ Y1_pre   # (J,)

    def objective(z: np.ndarray) -> float:
        w, t = z[:J], z[J]
        return float(w @ H @ w + 2.0 * q @ w + lambda_pen * t)

    def gradient(z: np.ndarray) -> np.ndarray:
        w = z[:J]
        g = np.zeros(J + 1)
        g[:J] = 2.0 * H @ w + 2.0 * q
        g[J] = lambda_pen
        return g

    constraints: List[Dict[str, Any]] = [
        {"type": "eq", "fun": lambda z: np.sum(z[:J]) - 1.0},
    ]
    # t >= w_j * d_j  =>  t - w_j * d_j >= 0
    for j in range(J):
        dj = distances[j]
        constraints.append({
            "type": "ineq",
            "fun": lambda z, _j=j, _d=dj: z[J] - z[_j] * _d,
        })

    bounds = [(0.0, 1.0)] * J + [(0.0, None)]
    z0 = np.zeros(J + 1)
    z0[:J] = 1.0 / J
    z0[J] = np.max(distances) / J  # initial t

    result = optimize.minimize(
        objective,
        z0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = result.x[:J]
    w = np.maximum(w, 0.0)
    w /= w.sum() if w.sum() > 0 else 1.0
    return w


def _cv_lambda(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    distances: np.ndarray,
    penalty_type: str,
    n_folds: int = 5,
) -> Tuple[float, Dict[str, Any]]:
    """Select lambda via rolling-origin (expanding window) cross-validation.

    Splits pre-treatment periods into expanding training windows.  For
    each split, fit weights on the training window and evaluate
    prediction error on the next hold-out period(s).

    Parameters
    ----------
    Y1_pre : (T0,) treated pre-treatment outcomes
    Y0_pre : (J, T0) donor pre-treatment outcomes
    distances : (J,) pairwise distances
    penalty_type : str
    n_folds : int, default 5
        Number of expanding-window splits.

    Returns
    -------
    best_lambda : float
    cv_results : dict with 'lambdas', 'cv_errors', 'best_lambda'
    """
    T0 = len(Y1_pre)

    # Lambda grid: heuristic range from 0 to scale of data
    outcome_var = float(np.var(Y1_pre)) if np.var(Y1_pre) > 0 else 1.0
    dist_mean = float(np.mean(distances)) if np.mean(distances) > 0 else 1.0
    scale = outcome_var / max(dist_mean, 1e-10)

    lambdas = np.concatenate([
        [0.0],
        np.logspace(-4, 2, 20) * scale,
    ])

    # Rolling-origin splits: train on [0, split_end), validate on [split_end, split_end + h)
    min_train = max(3, T0 // 3)
    h = max(1, T0 // (n_folds + 2))  # hold-out window size
    splits: List[Tuple[int, int]] = []
    for k in range(n_folds):
        train_end = min_train + k * h
        val_end = min(train_end + h, T0)
        if train_end >= T0 or val_end > T0:
            break
        splits.append((train_end, val_end))

    if len(splits) == 0:
        # Fallback: single split at 2/3 mark
        split_pt = max(2, 2 * T0 // 3)
        splits = [(split_pt, T0)]

    cv_errors = np.zeros(len(lambdas))

    for lam_idx, lam in enumerate(lambdas):
        fold_errors: List[float] = []
        for train_end, val_end in splits:
            Y1_train = Y1_pre[:train_end]
            Y0_train = Y0_pre[:, :train_end]
            Y1_val = Y1_pre[train_end:val_end]
            Y0_val = Y0_pre[:, train_end:val_end]

            w = _penalized_weights(
                Y1_train, Y0_train, distances, lam, penalty_type,
            )
            Y_hat = Y0_val.T @ w
            fold_errors.append(float(np.mean((Y1_val - Y_hat) ** 2)))

        cv_errors[lam_idx] = float(np.mean(fold_errors))

    best_idx = int(np.argmin(cv_errors))
    best_lambda = float(lambdas[best_idx])

    cv_results = {
        "lambdas": lambdas.tolist(),
        "cv_errors": cv_errors.tolist(),
        "best_lambda": best_lambda,
        "n_splits": len(splits),
    }

    return best_lambda, cv_results


def _run_placebos(
    panel: pd.DataFrame,
    donors: List[Any],
    treated_unit: Any,
    pre_times: List[Any],
    post_times: List[Any],
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    covariates: Optional[List[str]],
    predictors: Optional[List[str]],
    lambda_pen: float,
    penalty_type: str,
) -> Dict[str, Any]:
    """In-space placebo permutation tests.

    For each donor, pretend it is the treated unit and run the penalized
    SCM with all remaining units (including the actual treated) as donors.

    Returns
    -------
    dict with keys: atts, ratios, gaps, units
    """
    atts: List[float] = []
    ratios: List[float] = []
    gap_trajectories: List[np.ndarray] = []
    placebo_units: List[Any] = []

    all_units = [treated_unit] + list(donors)

    for placebo_unit in donors:
        p_donors = [u for u in all_units if u != placebo_unit]

        Y1_pre_p = panel.loc[placebo_unit, pre_times].values.astype(np.float64)
        Y1_post_p = panel.loc[placebo_unit, post_times].values.astype(np.float64)
        Y0_pre_p = panel.loc[p_donors, pre_times].values.astype(np.float64)
        Y0_post_p = panel.loc[p_donors, post_times].values.astype(np.float64)

        # Build covariate matrix for this placebo
        X_treated_p, X_donors_p = _build_covariate_matrix(
            data, panel, outcome, unit, time, placebo_unit, p_donors,
            pre_times, covariates, predictors,
        )
        distances_p = _compute_pairwise_distances(
            X_treated_p, X_donors_p, penalty_type,
        )

        try:
            w_p = _penalized_weights(
                Y1_pre_p, Y0_pre_p, distances_p, lambda_pen, penalty_type,
            )
        except Exception:
            continue

        Y_synth_pre_p = Y0_pre_p.T @ w_p
        Y_synth_post_p = Y0_post_p.T @ w_p

        gap_pre_p = Y1_pre_p - Y_synth_pre_p
        gap_post_p = Y1_post_p - Y_synth_post_p

        pre_rmspe_p = float(np.sqrt(np.mean(gap_pre_p ** 2)))
        post_rmspe_p = float(np.sqrt(np.mean(gap_post_p ** 2)))

        ratio_p = post_rmspe_p / pre_rmspe_p if pre_rmspe_p > 1e-10 else np.inf

        atts.append(float(np.mean(gap_post_p)))
        ratios.append(ratio_p)
        gap_trajectories.append(np.concatenate([gap_pre_p, gap_post_p]))
        placebo_units.append(placebo_unit)

    gaps_array = (
        np.column_stack(gap_trajectories)
        if gap_trajectories
        else np.empty((len(pre_times) + len(post_times), 0))
    )

    return {
        "atts": atts,
        "ratios": ratios,
        "gaps": gaps_array,
        "units": placebo_units,
    }
