"""
Augmented Synthetic Control Method (ASCM).

Combines synthetic-control weights with a ridge correction to reduce
pre-treatment imbalance when a simplex SCM alone cannot perfectly match
the treated unit.

Model
-----
    w_aug = w_scm + (x1 - X0' w_scm)' (X0'X0 + λI)^(-1) X0'
    τ̂_ascm = mean(Y1_post - Y0_post' w_aug)

where pre-treatment outcomes are centered by the control mean before
the SCM and ridge steps. This matches the no-covariate
``augsynth::augsynth(..., progfunc="Ridge", scm=TRUE)`` convention used
by the Track-A parity harness.

References
----------
Ben-Michael, E., Feller, A. and Rothstein, J. (2021).
"The Augmented Synthetic Control Method."
*Journal of the American Statistical Association*, 116(536), 1789-1803. [@benmichael2021augmented]

Ben-Michael, E., Feller, A. and Rothstein, J. (2022).
"Synthetic Controls with Staggered Adoption."
*Journal of the Royal Statistical Society: Series B*, 84(2), 351-381. [@benmichael2022synthetic]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


def augsynth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    ridge_lambda: Optional[float] = None,
    placebo: bool = True,
    alpha: float = 0.05,
    backend: str = "native",
    **kwargs: Any,
) -> CausalResult:
    """
    Augmented Synthetic Control Method (Ben-Michael, Feller & Rothstein 2021).

    Fits SCM weights on control-mean-centered pre-treatment outcomes and
    then applies the ridge-augmented weight correction used by
    ``augsynth::augsynth``:

        w_aug = w_scm + (x1 - X0' w_scm)' (X0'X0 + λI)^(-1) X0'.

    The augmented weights are applied directly to the full donor
    trajectory and may be negative. Large ``ridge_lambda`` approaches
    centered SCM; smaller values allow more extrapolation.

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
        Ridge penalty. When ``None``, selected by time-holdout CV
        matching ``augsynth``'s one-standard-error rule.
    placebo : bool, default True
        Run in-space placebo permutation tests for SE / p-value.
    alpha : float, default 0.05
        Significance level.
    backend : {'native', 'augsynth', 'r'}, default 'native'
        ``'native'`` uses StatsPAI's Python ridge-augmented SCM
        implementation. ``'augsynth'``/``'r'`` delegates the point
        estimate and pre-period RMSPE to the R ``augsynth`` package
        through ``Rscript`` using ``progfunc='Ridge'`` and ``scm=TRUE``.
        The R backend is intended for exact reference-package parity;
        the native path remains the dependency-light default.
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
    Synthetic Control Method." *JASA*, 116(536), 1789-1803. [@benmichael2021augmented]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_tobacco()
    >>> result = sp.augsynth(df, outcome='cigsale', unit='state', time='year',
    ...                       treated_unit='California', treatment_time=1989)
    >>> bool(result.estimate < 0)  # Prop 99 lowered cigarette sales
    True
    """
    backend_norm = backend.lower().replace("-", "_")
    if backend_norm in {"augsynth", "r", "augsynth_r"}:
        if covariates:
            raise NotImplementedError(
                "The augsynth R reference backend currently supports the "
                "outcome/treatment specification used in the parity harness; "
                "use backend='native' for covariates."
            )
        if ridge_lambda is not None:
            raise NotImplementedError(
                "The augsynth R reference backend uses augsynth::augsynth's "
                "own Ridge regularisation convention; use backend='native' "
                "for an explicit ridge_lambda."
            )
        return _augsynth_r_backend(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            alpha=alpha,
        )
    if backend_norm != "native":
        raise ValueError("Unknown backend. Use 'native', 'augsynth', or 'r'.")

    # --- Input validation (unified with classic SCM contract) ---
    required_cols = [outcome, unit, time]
    if covariates:
        required_cols = required_cols + list(covariates)
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    if treated_unit not in data[unit].values:
        raise ValueError(f"Treated unit '{treated_unit}' not found in '{unit}' column")

    # --- Reshape to wide format ---
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        from statspai.exceptions import DataInsufficient

        raise DataInsufficient(
            "Need at least 2 pre-treatment periods",
            recovery_hint=(
                "Augmented SCM fits an outcome bridge on pre-periods — "
                "needs at least 2. Use sp.did if you only have 1 pre period."
            ),
            diagnostics={"n_pre_periods": int(len(pre_times))},
            alternative_functions=["sp.did"],
        )
    if len(post_times) < 1:
        from statspai.exceptions import DataInsufficient

        raise DataInsufficient(
            "Need at least 1 post-treatment period",
            recovery_hint=("Verify treatment_time is within the panel window."),
            diagnostics={"n_post_periods": int(len(post_times))},
            alternative_functions=[],
        )

    # Treated and donor matrices
    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)  # (J, T0)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    # --- Step 1: R-augsynth-compatible augmented weights ---
    #
    # ``augsynth::augsynth(..., progfunc="Ridge", scm=TRUE)`` centers
    # pre-treatment outcomes by the control mean, solves SCM on the
    # centered pre-period matrix, then adds a ridge correction to the
    # weights themselves.  The final counterfactual is the full donor
    # trajectory multiplied by these augmented weights, which may be
    # negative.  This is deliberately different from a post-on-pre
    # outcome-model imputation.
    weight_fit = _fit_ridge_augmented_weights(
        Y1_pre,
        Y0_pre,
        ridge_lambda=ridge_lambda,
    )
    gamma = weight_fit["weights"]
    syn_weights = weight_fit["synthetic_weights"]
    ridge_lambda = float(weight_fit["ridge_lambda"])

    # --- Step 2: Counterfactual trajectory using augmented weights ---
    Y1_hat_scm_pre = Y0_pre.T @ syn_weights
    Y1_hat_aug_pre = Y0_pre.T @ gamma
    Y1_hat_aug_post = Y0_post.T @ gamma

    pre_residual_aug = Y1_pre - Y1_hat_aug_pre
    pre_residual_scm = Y1_pre - Y1_hat_scm_pre
    pre_rmspe = float(np.sqrt(np.mean(pre_residual_aug**2)))

    effects = Y1_post - Y1_hat_aug_post
    att = float(np.mean(effects))

    # --- Inference via placebo permutation ---
    placebo_effects: Any
    if placebo:
        placebo_effects = []
        for j in range(J):
            other_idx = [i for i in range(J) if i != j]
            Y_plac_pre = Y0_pre[j]
            Y_plac_post = Y0_post[j]
            Y_others_pre = Y0_pre[other_idx]
            Y_others_post = Y0_post[other_idx]

            plac_fit = _fit_ridge_augmented_weights(
                Y_plac_pre,
                Y_others_pre,
                ridge_lambda=ridge_lambda,
            )
            plac_weights = plac_fit["weights"]
            plac_hat = Y_others_post.T @ plac_weights
            plac_eff = float(np.mean(Y_plac_post - plac_hat))
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
    effects_df = pd.DataFrame(
        {
            "time": post_times,
            "treated": Y1_post,
            "counterfactual": Y1_hat_aug_post,
            "effect": effects,
        }
    )

    # Unified gap table (full trajectory, matches classic SCM contract)
    all_times_arr = np.array(pre_times + post_times)
    Y1_all = np.concatenate([Y1_pre, Y1_post])
    Y1_hat_all = np.concatenate([Y1_hat_aug_pre, Y1_hat_aug_post])
    gap_all = Y1_all - Y1_hat_all
    gap_df = pd.DataFrame(
        {
            "time": all_times_arr,
            "treated": Y1_all,
            "synthetic": Y1_hat_all,
            "gap": gap_all,
            "post_treatment": np.concatenate(
                [
                    np.zeros(T0, dtype=bool),
                    np.ones(T1, dtype=bool),
                ]
            ),
        }
    )

    weight_df = (
        pd.DataFrame({"unit": donors, "weight": gamma})
        .assign(abs_weight=lambda df: np.abs(df["weight"]))
        .sort_values("abs_weight", ascending=False)
        .drop(columns=["abs_weight"])
        .reset_index(drop=True)
    )
    weight_df = weight_df[np.abs(weight_df["weight"]) > 1e-6]

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
            "pre_treatment_rmse": pre_rmspe,
            "pre_treatment_mspe": pre_rmspe**2,
            "scm_pre_treatment_rmse": float(np.sqrt(np.mean(pre_residual_scm**2))),
            "post_rmspe": float(np.sqrt(np.mean(effects**2))),
            "weights": weight_df,
            "weights_dict": dict(zip(donors, np.asarray(gamma))),
            "synthetic_weights": dict(zip(donors, np.asarray(syn_weights))),
            "augmented_weights_can_be_negative": True,
            "n_donors": J,
            "n_pre_periods": T0,
            "n_post_periods": T1,
            "treatment_time": treatment_time,
            "treated_unit": treated_unit,
            "ridge_lambda": ridge_lambda,
            "effects_by_period": effects_df,
            "gap_table": gap_df,
            "times": all_times_arr,
            "Y_synth": Y1_hat_all,
            "Y_treated": Y1_all,
            "placebo_distribution": placebo_effects,
            "n_placebos": len(placebo_effects),
        },
    )


def _find_rscript() -> str:
    """Return a usable Rscript executable, including common macOS paths."""
    candidates = [
        shutil.which("Rscript"),
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(
        "The augsynth backend requires Rscript plus the R packages "
        "'augsynth' and 'jsonlite'. Install R or use backend='native'."
    )


def _augsynth_r_backend(
    *,
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    alpha: float,
) -> CausalResult:
    """Delegate ASCM estimation to R ``augsynth`` for parity."""
    if not pd.api.types.is_numeric_dtype(data[time]):
        raise TypeError(
            "The augsynth R backend requires a numeric time column so it "
            "can reproduce the reference package's pre/post split."
        )

    unit_col = "statspai_unit"
    time_col = "statspai_time"
    outcome_col = "statspai_outcome"
    treated_col = "statspai_treated"
    target_col = "statspai_target"
    panel_df = pd.DataFrame(
        {
            unit_col: data[unit],
            time_col: data[time],
            outcome_col: data[outcome],
        }
    )
    panel_df[target_col] = (data[unit] == treated_unit).astype(int)
    panel_df[treated_col] = (
        (data[unit] == treated_unit) & (data[time] >= treatment_time)
    ).astype(int)
    panel_df = panel_df.sort_values([unit_col, time_col]).reset_index(drop=True)

    r_script = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("expected 3 arguments: input output treatment_time")
}
input_path <- args[[1]]
output_path <- args[[2]]
treatment_time <- as.numeric(args[[3]])

suppressPackageStartupMessages({
  library(augsynth)
  library(jsonlite)
})

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
fit <- augsynth::augsynth(
  form = statspai_outcome ~ statspai_treated,
  unit = statspai_unit,
  time = statspai_time,
  data = df,
  progfunc = "Ridge",
  scm = TRUE
)

sm <- summary(fit)
att_est <- as.numeric(sm$average_att$Estimate)

synth_traj <- predict(fit)
yrs <- as.numeric(names(synth_traj))
treated_rows <- df[df$statspai_target == 1, ]
treated_y <- treated_rows$statspai_outcome[match(yrs, treated_rows$statspai_time)]
pre_idx <- yrs < treatment_time
pre_residuals <- treated_y[pre_idx] - synth_traj[pre_idx]
pre_rmspe <- sqrt(mean(pre_residuals^2))

payload <- list(
  estimate = att_est,
  pre_rmspe = as.numeric(pre_rmspe),
  n_obs = nrow(df),
  n_units = length(unique(df$statspai_unit)),
  n_pre_periods = sum(sort(unique(df$statspai_time)) < treatment_time),
  n_post_periods = sum(sort(unique(df$statspai_time)) >= treatment_time)
)
jsonlite::write_json(
  payload,
  output_path,
  auto_unbox = TRUE,
  null = "null",
  na = "null",
  digits = 16
)
"""

    rscript = _find_rscript()
    with tempfile.TemporaryDirectory(prefix="statspai_augsynth_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "panel.csv"
        script_path = tmp_path / "augsynth_backend.R"
        out_path = tmp_path / "result.json"
        panel_df.to_csv(data_path, index=False)
        script_path.write_text(r_script, encoding="utf-8")

        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(data_path),
                str(out_path),
                str(float(treatment_time)),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R augsynth backend failed. stderr:\n" f"{proc.stderr.strip()}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    pre_rmspe = float(payload["pre_rmspe"])
    ci = (float("nan"), float("nan"))
    model_info = {
        "model_type": "Synthetic Control (Augmented)",
        "backend": "augsynth",
        "r_package": "augsynth",
        "progfunc": "Ridge",
        "scm": True,
        "pre_rmspe": pre_rmspe,
        "pre_treatment_rmse": pre_rmspe,
        "pre_treatment_mspe": pre_rmspe**2,
        "n_donors": int(payload["n_units"]) - 1,
        "n_pre_periods": int(payload["n_pre_periods"]),
        "n_post_periods": int(payload["n_post_periods"]),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "ridge_lambda": None,
    }

    return CausalResult(
        method="Augmented Synthetic Control (R augsynth bridge backend)",
        estimand="ATT",
        estimate=float(payload["estimate"]),
        se=float("nan"),
        pvalue=float("nan"),
        ci=ci,
        alpha=alpha,
        n_obs=int(payload["n_obs"]),
        detail=None,
        model_info=model_info,
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


def _solve_ridge_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a ridge normal equation with a pseudo-inverse fallback."""
    try:
        return np.asarray(np.linalg.solve(a, b))
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.pinv(a) @ b)


def _fit_ridge_augmented_weights(
    target_pre: np.ndarray,
    control_pre: np.ndarray,
    *,
    ridge_lambda: Optional[float] = None,
) -> dict[str, np.ndarray | float]:
    """
    Fit R ``augsynth``-style ridge-augmented SCM weights.

    The installed R package centers pre-treatment outcomes by the
    control-unit mean, obtains simplex SCM weights on the centered
    matrix, then applies a ridge correction to the weights:

        w_aug = w_scm + (x1 - X0' w_scm)' (X0'X0 + lambda I)^(-1) X0'.

    The resulting weights need not be non-negative.  They are then used
    directly on the raw control trajectory for both pre- and post-period
    counterfactuals.
    """
    target_pre = np.asarray(target_pre, dtype=np.float64).ravel()
    control_pre = np.asarray(control_pre, dtype=np.float64)
    control_mean = control_pre.mean(axis=0)
    x_c = control_pre - control_mean
    x_1 = target_pre - control_mean

    syn = _scm_weights(x_1, x_c)
    if ridge_lambda is None:
        ridge_lambda = _cv_ridge_lambda_augsynth(x_c, x_1)
    ridge_lambda = float(ridge_lambda)

    gram = x_c.T @ x_c + ridge_lambda * np.eye(x_c.shape[1])
    imbalance = x_1 - x_c.T @ syn
    ridge_weights = imbalance @ _solve_ridge_system(gram, x_c.T)
    weights = syn + ridge_weights
    return {
        "weights": np.asarray(weights, dtype=np.float64),
        "synthetic_weights": np.asarray(syn, dtype=np.float64),
        "ridge_lambda": ridge_lambda,
    }


def _cv_ridge_lambda_augsynth(
    x_c: np.ndarray,
    x_1: np.ndarray,
    *,
    lambda_min_ratio: float = 1e-8,
    n_lambda: int = 20,
    holdout_length: int = 1,
    min_1se: bool = True,
) -> float:
    """Time-holdout lambda CV matching ``augsynth``'s ridge path."""
    _, singular_values, _ = np.linalg.svd(x_c, full_matrices=False)
    if singular_values.size == 0:
        return 1.0
    lambda_max = float(singular_values[0] ** 2)
    if not np.isfinite(lambda_max) or lambda_max <= 0:
        return 1.0
    scaler = float(lambda_min_ratio ** (1.0 / n_lambda))
    lambdas = lambda_max * scaler ** np.arange(n_lambda + 1)

    n_periods = x_c.shape[1]
    n_holdouts = n_periods - holdout_length
    if n_holdouts <= 1:
        return float(lambdas[0])

    errors = np.zeros((n_holdouts, len(lambdas)), dtype=np.float64)
    for i in range(n_holdouts):
        holdout = np.arange(i, i + holdout_length)
        train_mask = np.ones(n_periods, dtype=bool)
        train_mask[holdout] = False
        x_0_train = x_c[:, train_mask]
        x_1_train = x_1[train_mask]
        x_0_val = x_c[:, holdout]
        x_1_val = x_1[holdout]

        syn = _scm_weights(x_1_train, x_0_train)
        imbalance = x_1_train - x_0_train.T @ syn
        base_gram = x_0_train.T @ x_0_train
        for j, lam in enumerate(lambdas):
            gram = base_gram + float(lam) * np.eye(x_0_train.shape[1])
            ridge_weights = imbalance @ _solve_ridge_system(gram, x_0_train.T)
            aug_weights = syn + ridge_weights
            val_error = x_1_val - x_0_val.T @ aug_weights
            errors[i, j] = float(val_error @ val_error)

    lambda_errors = errors.mean(axis=0)
    lambda_errors_se = errors.std(axis=0, ddof=1) / np.sqrt(errors.shape[0])
    min_idx = int(np.argmin(lambda_errors))
    if not min_1se:
        return float(lambdas[min_idx])
    threshold = lambda_errors[min_idx] + lambda_errors_se[min_idx]
    candidates = lambdas[lambda_errors <= threshold]
    if candidates.size == 0:
        return float(lambdas[min_idx])
    return float(np.max(candidates))
