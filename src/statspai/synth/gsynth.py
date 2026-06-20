"""
Generalized Synthetic Control Method (GSynth).

Uses an interactive fixed-effects (IFE) model — a factor model —
to impute the treated unit's counterfactual. Unlike classic SCM
which constructs a single weighted average, GSynth estimates latent
factors from never-treated control units, then projects the treated
unit onto those factors using the pre-treatment period.

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
*Political Analysis*, 25(1), 57–76. [@xu2017generalized]
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
    backend: str = "native",
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
    backend : {'native', 'gsynth', 'r'}, default 'native'
        ``'native'`` uses StatsPAI's Python interactive fixed-effects
        implementation. ``'gsynth'``/``'r'`` delegates to the R
        ``gsynth`` package through ``Rscript`` using the Track-A
        reference specification ``force='two-way'``, ``CV=TRUE``,
        ``r=c(0, max_factors)``, and ``se=FALSE``. The R backend is
        intended for exact reference-package parity; the native path
        remains the dependency-light default.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.gsynth(df, outcome='packspercapita', unit='state',
    ...                    time='year', treated_unit='California',
    ...                    treatment_time=1989, placebo=False, seed=0)
    >>> bool(result.estimate is not None)
    True
    """
    backend_norm = backend.lower().replace("-", "_")
    if backend_norm in {"gsynth", "r", "gsynth_r"}:
        if covariates:
            raise NotImplementedError(
                "The gsynth R reference backend currently supports the "
                "outcome/treatment specification used in the parity harness; "
                "use backend='native' for covariates."
            )
        if n_factors is not None:
            raise NotImplementedError(
                "The gsynth R reference backend uses gsynth::gsynth's CV "
                "factor selection; use backend='native' for an explicit "
                "n_factors."
            )
        return _gsynth_r_backend(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            max_factors=max_factors,
            seed=seed,
            alpha=alpha,
        )
    if backend_norm != "native":
        raise ValueError("Unknown backend. Use 'native', 'gsynth', or 'r'.")

    rng = np.random.default_rng(seed)

    # --- Build panel ---
    pivot = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(pivot.columns.tolist())
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 3:
        from statspai.exceptions import DataInsufficient

        raise DataInsufficient(
            "GSynth needs at least 3 pre-treatment periods",
            recovery_hint=(
                "Interactive-fixed-effects identification requires ≥ 3 "
                "pre-periods. Use sp.synth(method='classic') for 2 pre-"
                "periods, or sp.did for 1."
            ),
            diagnostics={"n_pre_periods": int(len(pre_times))},
            alternative_functions=["sp.synth", "sp.did"],
        )
    if len(post_times) < 1:
        from statspai.exceptions import DataInsufficient

        raise DataInsufficient(
            "Need at least 1 post-treatment period",
            recovery_hint="Verify treatment_time is within the panel window.",
            diagnostics={"n_post_periods": int(len(post_times))},
            alternative_functions=[],
        )

    # Separate treated and control
    donors = [u for u in pivot.index if u != treated_unit]
    Y0_pre = pivot.loc[donors, pre_times].values.astype(np.float64)  # (J, T0)
    Y0_post = pivot.loc[donors, post_times].values.astype(np.float64)  # (J, T1)
    Y1_pre = pivot.loc[treated_unit, pre_times].values.astype(np.float64)  # (T0,)
    Y1_post = pivot.loc[treated_unit, post_times].values.astype(np.float64)  # (T1,)

    J, T0 = Y0_pre.shape
    T1 = len(post_times)

    # --- Handle covariates ---
    beta_X = None
    if covariates:
        Y0_pre, Y1_pre, Y0_post, Y1_post, beta_X = _partial_out_covariates(
            data,
            outcome,
            unit,
            time,
            treated_unit,
            treatment_time,
            covariates,
            donors,
            pre_times,
            post_times,
        )

    Y0_all = np.concatenate([Y0_pre, Y0_post], axis=1)
    Y1_all = np.concatenate([Y1_pre, Y1_post])
    n_periods = Y0_all.shape[1]

    # --- Select number of factors ---
    Y0_all_dm, _, _, _ = _twoway_demean(Y0_all)
    if n_factors is None:
        n_factors = _select_factors_cv(Y0_all_dm, max_factors, cv_folds, rng)
    max_rank = max(0, min(J, n_periods) - 1)
    n_factors = max(0, min(int(n_factors), max_rank))

    # --- Fit the gsynth/fect convention on never-treated controls ---
    fit = _fit_control_factor_model(Y0_all, Y1_pre, T0, n_factors)
    Y1_hat_all = np.asarray(fit["treated_counterfactual"])
    Y1_hat_pre = Y1_hat_all[:T0]
    Y1_hat_post = Y1_hat_all[T0:]
    F_all = np.asarray(fit["factors"])
    F_pre = F_all[:T0]
    F_post = F_all[T0:]
    L_control = fit["control_loadings"]
    L_treated = fit["treated_loadings"]
    S = fit["singular_values"]

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
                Y_ctrl_all = np.concatenate([Y_ctrl_pre, Y_ctrl_post], axis=1)
                max_rank_p = max(0, min(Y_ctrl_all.shape) - 1)
                r_p = min(n_factors, max_rank_p)
                plac_fit = _fit_control_factor_model(Y_ctrl_all, Y_plac, T0, r_p)
                hat_post = np.asarray(plac_fit["treated_counterfactual"])[T0:]
                placebo_atts.append(float(np.mean(Y_plac_post - hat_post)))
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
    effects_df = pd.DataFrame(
        {
            "time": post_times,
            "treated": Y1_post,
            "counterfactual": Y1_hat_post,
            "effect": effects,
        }
    )

    trajectory_df = pd.DataFrame(
        {
            "time": all_times,
            "treated": np.concatenate([Y1_pre, Y1_post]),
            "synthetic": np.concatenate([Y1_hat_pre, Y1_hat_post]),
        }
    )

    model_info = {
        "backend": "native",
        "native_convention": "gsynth/fect two-way FE on full never-treated controls",
        "n_factors": n_factors,
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "pre_treatment_mspe": pre_mspe,
        "pre_treatment_rmse": float(np.sqrt(pre_mspe)),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "control_unit_fe": fit["control_unit_fe"],
        "treated_unit_fe": fit["treated_unit_fe"],
        "time_fe": fit["time_fe"],
        "grand_mean": fit["grand_mean"],
        "factors_pre": F_pre,
        "factors_post": F_post,
        "loadings_treated": L_treated,
        "loadings_control": L_control,
        "singular_values": S,
        "effects_by_period": effects_df,
        "trajectory": trajectory_df,
        "Y_synth": np.concatenate([Y1_hat_pre, Y1_hat_post]),
        "Y_treated": Y1_all,
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
        "The gsynth backend requires Rscript plus the R packages "
        "'gsynth' and 'jsonlite'. Install R or use backend='native'."
    )


def _gsynth_r_backend(
    *,
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    max_factors: int,
    seed: Optional[int],
    alpha: float,
) -> CausalResult:
    """Delegate IFE estimation to R ``gsynth`` for parity."""
    if not pd.api.types.is_numeric_dtype(data[time]):
        raise TypeError(
            "The gsynth R backend requires a numeric time column so it "
            "can reproduce the reference package's pre/post split."
        )

    unit_col = "statspai_unit"
    time_col = "statspai_time"
    outcome_col = "statspai_outcome"
    treated_col = "statspai_treated"
    panel_df = pd.DataFrame(
        {
            unit_col: data[unit],
            time_col: data[time],
            outcome_col: data[outcome],
        }
    )
    panel_df[treated_col] = (
        (data[unit] == treated_unit) & (data[time] >= treatment_time)
    ).astype(int)
    panel_df = panel_df.sort_values([unit_col, time_col]).reset_index(drop=True)

    r_script = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("expected 5 arguments: input output treatment_time max_factors seed")
}
input_path <- args[[1]]
output_path <- args[[2]]
treatment_time <- as.numeric(args[[3]])
max_factors <- as.integer(args[[4]])
seed <- suppressWarnings(as.integer(args[[5]]))

suppressPackageStartupMessages({
  library(gsynth)
  library(jsonlite)
})

if (!is.na(seed)) {
  set.seed(seed)
}

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
fit <- gsynth::gsynth(
  formula = statspai_outcome ~ statspai_treated,
  data = df,
  index = c("statspai_unit", "statspai_time"),
  force = "two-way",
  CV = TRUE,
  r = c(0, max_factors),
  se = FALSE,
  inference = "parametric",
  nboots = 50
)

payload <- list(
  estimate = as.numeric(fit$att.avg),
  n_factors = as.numeric(fit$r.cv),
  pre_rmse = as.numeric(fit$rmse),
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
    with tempfile.TemporaryDirectory(prefix="statspai_gsynth_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "panel.csv"
        script_path = tmp_path / "gsynth_backend.R"
        out_path = tmp_path / "result.json"
        panel_df.to_csv(data_path, index=False)
        script_path.write_text(r_script, encoding="utf-8")

        seed_arg = "NA" if seed is None else str(int(seed))
        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(data_path),
                str(out_path),
                str(float(treatment_time)),
                str(int(max_factors)),
                seed_arg,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R gsynth backend failed. stderr:\n" f"{proc.stderr.strip()}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    pre_rmse = float(payload["pre_rmse"])
    model_info = {
        "backend": "gsynth",
        "r_package": "gsynth",
        "force": "two-way",
        "CV": True,
        "n_factors": int(payload["n_factors"]),
        "n_donors": int(payload["n_units"]) - 1,
        "n_pre_periods": int(payload["n_pre_periods"]),
        "n_post_periods": int(payload["n_post_periods"]),
        "pre_treatment_rmse": pre_rmse,
        "pre_treatment_mspe": pre_rmse**2,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
    }

    return CausalResult(
        method="Generalized Synthetic Control (R gsynth bridge backend)",
        estimand="ATT",
        estimate=float(payload["estimate"]),
        se=float("nan"),
        pvalue=float("nan"),
        ci=(float("nan"), float("nan")),
        alpha=alpha,
        n_obs=int(payload["n_obs"]),
        detail=None,
        model_info=model_info,
        _citation_key="gsynth",
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #


def _twoway_demean(Y: np.ndarray) -> tuple[Any, Any, Any, Any]:
    """Remove row means, column means, and grand mean."""
    grand_mean = np.nanmean(Y)
    row_means = np.nanmean(Y, axis=1)
    col_means = np.nanmean(Y, axis=0)
    Y_dm = Y - row_means[:, np.newaxis] - col_means[np.newaxis, :] + grand_mean
    return Y_dm, row_means, col_means, grand_mean


def _fit_control_factor_model(
    Y0_all: np.ndarray,
    Y1_pre: np.ndarray,
    n_pre_periods: int,
    n_factors: int,
) -> dict[str, np.ndarray | float]:
    """Fit the ``gsynth``/``fect`` never-treated control factor convention.

    For ``method='gsynth'`` with ``force='two-way'``, the R implementation
    estimates the factor path on the complete never-treated control panel,
    then projects the treated unit onto those full-panel factors using only
    its pre-treatment outcomes. This helper implements that convention for
    the balanced-panel native path.
    """
    Y0_all = np.asarray(Y0_all, dtype=np.float64)
    Y1_pre = np.asarray(Y1_pre, dtype=np.float64)
    n_periods = Y0_all.shape[1]
    n_factors = max(0, min(int(n_factors), min(Y0_all.shape) - 1))

    Y0_dm, row_means, col_means, grand_mean = _twoway_demean(Y0_all)
    control_unit_fe = row_means - grand_mean
    time_fe = col_means - grand_mean

    if n_factors > 0:
        U, S_full, Vt = np.linalg.svd(Y0_dm, full_matrices=False)
        factors = Vt[:n_factors].T
        control_loadings = U[:, :n_factors] * S_full[:n_factors]
        singular_values = S_full[:n_factors]

        design_pre = np.column_stack(
            [
                np.ones(n_pre_periods),
                factors[:n_pre_periods],
            ]
        )
        target_pre = Y1_pre - grand_mean - time_fe[:n_pre_periods]
        coef = np.linalg.lstsq(design_pre, target_pre, rcond=None)[0]
        treated_unit_fe = float(coef[0])
        treated_loadings = np.asarray(coef[1:], dtype=np.float64)
        low_rank = factors @ treated_loadings
    else:
        factors = np.empty((n_periods, 0), dtype=np.float64)
        control_loadings = np.empty((Y0_all.shape[0], 0), dtype=np.float64)
        singular_values = np.empty(0, dtype=np.float64)
        target_pre = Y1_pre - grand_mean - time_fe[:n_pre_periods]
        treated_unit_fe = float(np.mean(target_pre))
        treated_loadings = np.empty(0, dtype=np.float64)
        low_rank = np.zeros(n_periods, dtype=np.float64)

    treated_counterfactual = grand_mean + treated_unit_fe + time_fe + low_rank

    return {
        "treated_counterfactual": treated_counterfactual,
        "factors": factors,
        "control_loadings": control_loadings,
        "treated_loadings": treated_loadings,
        "singular_values": singular_values,
        "control_unit_fe": control_unit_fe,
        "treated_unit_fe": treated_unit_fe,
        "time_fe": time_fe,
        "grand_mean": float(grand_mean),
    }


def _select_factors_cv(
    Y_dm: np.ndarray,
    max_factors: int,
    n_folds: int,
    rng: np.random.Generator,
) -> int:
    """Select number of factors via cross-validation on the control panel."""
    J, T = Y_dm.shape
    max_factors = min(max_factors, min(J, T) - 1)
    if max_factors < 0:
        return 0

    # Random fold assignment for entries
    indices = list(range(J * T))
    rng.shuffle(indices)
    n_folds = max(1, min(int(n_folds), len(indices)))
    fold_size = max(1, len(indices) // n_folds)

    best_r = 0
    best_mse = np.inf

    for r in range(0, max_factors + 1):
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

            # SVD reconstruction with r factors. r=0 is the two-way-FE
            # baseline, matching gsynth's candidate grid.
            if r == 0:
                recon = np.zeros_like(Y_train)
            else:
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
    data: Any,
    outcome: Any,
    unit: Any,
    time: Any,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Any,
    donors: Any,
    pre_times: Any,
    post_times: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Partial out covariates via OLS, return residualised outcomes."""
    pre_ctrl = data[(data[unit].isin(donors)) & (data[time].isin(pre_times))].copy()

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
