"""R ``augsynth`` bridge backend for ``sp.augsynth(backend='augsynth')``.

Delegates the single-treated-unit ``progfunc='Ridge', scm=TRUE`` fit to the
R package through ``Rscript``; kept out of ``augsynth.py`` so the native
estimator's module stays small.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.results import CausalResult


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
        _citation_key="augsynth",
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
