"""R ``HonestDiD`` backend of :func:`statspai.did.honest_did.honest_did`.

``backend='honestdid'`` / ``'r'`` hands the event-study coefficients and
their joint covariance to the R package through ``Rscript`` and reads the
sensitivity table back.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient
from ._flci import event_study_moments
from ._honest_common import (
    _RESTRICTION_LABEL,
    _coerce_l_vec,
    _coerce_m_grid,
    _extract_event_study,
    _require_alpha,
    _require_int,
    _require_string,
    _resolve_restriction,
    _window_filter,
)


def _honest_did_r_backend(
    result: CausalResult,
    e: int,
    m_grid: Optional[List[float]],
    method: str,
    alpha: float,
    honestdid_method: Optional[str] = None,
    l_vec: Optional[Any] = None,
    window: Optional[tuple] = None,
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    grid_points: int = 1000,
    bias_direction: Optional[str] = None,
    monotonicity_direction: Optional[str] = None,
) -> pd.DataFrame:
    """Run the R HonestDiD reference implementation for CI parity.

    ``window`` restricts the event study (table and covariance) before it is
    handed to R; ``l_vec`` goes to HonestDiD's ``l_vec`` as given.
    """
    e = _require_int(e, argument="e")
    method = _require_string(method, argument="method")
    alpha = _require_alpha(alpha)
    family, honestdid_method_arg, bias, mono = _resolve_restriction(
        method, honestdid_method, bias_direction, monotonicity_direction
    )
    method_norm = _RESTRICTION_LABEL[family]

    rscript = _find_rscript()
    if rscript is None:
        raise ImportError(
            "backend='honestdid' requires Rscript and the R package HonestDiD."
        )

    es = _extract_event_study(result)
    if window is not None:
        es = es[(es["relative_time"] >= window[0]) & (es["relative_time"] <= window[1])]
    moments = _window_filter(event_study_moments(result), window)
    pre = es[es["relative_time"] < 0].sort_values("relative_time")
    post = es[es["relative_time"] >= 0].sort_values("relative_time")

    lv: Optional[np.ndarray] = None
    if l_vec is not None:
        post_t = post["relative_time"].to_numpy(dtype=int)
        lv = _coerce_l_vec(l_vec, post_t)
        if moments is not None and len(moments[2]) == len(es):
            pm = moments[2] >= 0
            se_hat = float(np.sqrt(lv @ moments[1][np.ix_(pm, pm)] @ lv))
        else:
            se_hat = float(np.sqrt(lv**2 @ post["se"].to_numpy(dtype=float) ** 2))
        e = int(post_t[np.flatnonzero(lv != 0)].max())
    else:
        target_row = es[es["relative_time"] == e]
        if len(target_row) == 0:
            raise DataInsufficient(
                f"No event study estimate at relative time e={e}",
                recovery_hint="Choose an `e` value present in the event-study table.",
                diagnostics={"e": e, "available": es["relative_time"].tolist()},
            )
        se_hat = float(target_row["se"].iloc[0])
    m_grid = _coerce_m_grid(
        m_grid,
        se_hat=se_hat,
        relative_magnitude=family != "smoothness",
    )
    if len(pre) == 0:
        raise DataInsufficient(
            "backend='honestdid' requires at least one pre-treatment period.",
            recovery_hint=(
                "Use an event study with at least one pre-treatment " "coefficient."
            ),
            diagnostics={"backend": "honestdid"},
        )
    if len(post) == 0 or e not in set(post["relative_time"].astype(int)):
        raise DataInsufficient(
            "backend='honestdid' requires e to be a post-treatment period.",
            recovery_hint="Choose a post-treatment relative time for `e`.",
            diagnostics={"backend": "honestdid", "e": e},
        )

    r_code = r"""
suppressPackageStartupMessages({
  if (!requireNamespace("HonestDiD", quietly = TRUE)) {
    stop("R package 'HonestDiD' is not installed")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("R package 'jsonlite' is not installed")
  }
})
args <- commandArgs(trailingOnly = TRUE)
input <- args[[1]]
method <- args[[2]]
alpha <- as.numeric(args[[3]])
grid <- as.numeric(strsplit(args[[4]], ",", fixed = TRUE)[[1]])
target_e <- as.numeric(args[[5]])
honestdid_method <- args[[6]]
df <- utils::read.csv(input)
pre <- df[df$relative_time < 0, ]
post <- df[df$relative_time >= 0, ]
pre <- pre[order(pre$relative_time), ]
post <- post[order(post$relative_time), ]
if (nrow(pre) < 1 || nrow(post) < 1) {
  stop("HonestDiD backend requires pre and post event-study periods")
}
l_arg <- if (length(args) >= 8) args[[8]] else ""
if (nzchar(l_arg)) {
  l_given <- as.numeric(strsplit(l_arg, ",", fixed = TRUE)[[1]])
  if (length(l_given) != nrow(post)) {
    stop("l_vec must have one weight per post-treatment period")
  }
} else {
  post_idx <- which(post$relative_time == target_e)
  if (length(post_idx) != 1) {
    stop("target e must match exactly one post-treatment period")
  }
}
betahat <- c(pre$att, post$att)
ses <- c(pre$se, post$se)
# Full event-study covariance when StatsPAI could recover it from the
# influence functions; a diagonal built from the SEs otherwise. The
# diagonal discards the (large, positive) cross-period covariance and
# materially changes the confidence set, so it is a fallback, not a
# convention.
if (length(args) >= 7 && nzchar(args[[7]]) && file.exists(args[[7]])) {
  sigma <- as.matrix(utils::read.csv(args[[7]], header = FALSE))
  dimnames(sigma) <- NULL
} else {
  sigma <- diag(ses^2)
}
if (nzchar(l_arg)) {
  l_vec <- l_given
} else {
  l_vec <- rep(0, nrow(post))
  l_vec[post_idx] <- 1
}
l_vec <- matrix(l_vec, ncol = 1)
num_or_na <- function(i) {
  if (length(args) >= i && nzchar(args[[i]])) as.numeric(args[[i]]) else NA
}
grid_lb <- num_or_na(9)
grid_ub <- num_or_na(10)
grid_points <- num_or_na(11)
if (is.na(grid_points)) grid_points <- 1000
str_or_null <- function(i) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else NULL
}
bias_dir <- str_or_null(12)
mono_dir <- str_or_null(13)
if (method %in% c("relative_magnitude", "smoothness_relative_magnitude")) {
  sens <- suppressWarnings(
    HonestDiD::createSensitivityResults_relativeMagnitudes(
      betahat = betahat,
      sigma = sigma,
      numPrePeriods = nrow(pre),
      numPostPeriods = nrow(post),
      bound = if (method == "relative_magnitude") {
        "deviation from parallel trends"
      } else {
        "deviation from linear trend"
      },
      l_vec = l_vec,
      Mbarvec = grid,
      method = honestdid_method,
      monotonicityDirection = mono_dir,
      biasDirection = bias_dir,
      alpha = alpha,
      grid.lb = grid_lb,
      grid.ub = grid_ub,
      gridPoints = grid_points
    )
  )
  out <- data.frame(M = sens$Mbar, ci_lower = sens$lb, ci_upper = sens$ub)
} else {
  sens <- suppressWarnings(
    HonestDiD::createSensitivityResults(
      betahat = betahat,
      sigma = sigma,
      numPrePeriods = nrow(pre),
      numPostPeriods = nrow(post),
      l_vec = l_vec,
      Mvec = grid,
      method = honestdid_method,
      monotonicityDirection = mono_dir,
      biasDirection = bias_dir,
      alpha = alpha
    )
  )
  out <- data.frame(M = sens$M, ci_lower = sens$lb, ci_upper = sens$ub)
}
# To a file, not stdout: HonestDiD's solvers print to stdout ("Polishing not
# needed ..." from the C-F path), which corrupted the JSON read through 1.39.
cat(jsonlite::toJSON(out, dataframe = "rows", auto_unbox = TRUE,
                     digits = 16, null = "null"),
    file = file.path(dirname(input), "result.json"))
"""

    with tempfile.TemporaryDirectory(prefix="statspai-honestdid-") as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "event_study.csv"
        script_path = tmp_path / "run_honestdid.R"
        es[["relative_time", "att", "se"]].to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")

        # Hand R the full event-study covariance when we can recover it.
        # Feeding HonestDiD diag(se^2) silently drops the cross-period
        # covariance, which changes the confidence set materially.
        sigma_path = ""
        _moments = moments
        if _moments is not None:
            _beta_r, _sigma_r, _times_r = _moments
            _es_times = es["relative_time"].to_numpy()
            if len(_times_r) == len(_es_times) and set(_times_r) == set(_es_times):
                _order_r = np.concatenate(
                    [
                        np.where(_times_r < 0)[0][np.argsort(_times_r[_times_r < 0])],
                        np.where(_times_r >= 0)[0][np.argsort(_times_r[_times_r >= 0])],
                    ]
                )
                _sig_path = tmp_path / "sigma.csv"
                np.savetxt(
                    _sig_path,
                    _sigma_r[np.ix_(_order_r, _order_r)],
                    delimiter=",",
                )
                sigma_path = str(_sig_path)
        grid_arg = ",".join(f"{float(m):.17g}" for m in m_grid)
        method_arg = method_norm
        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(csv_path),
                method_arg,
                f"{alpha:.17g}",
                grid_arg,
                f"{float(e):.17g}",
                honestdid_method_arg,
                sigma_path,
                "" if lv is None else ",".join(f"{v:.17g}" for v in lv),
                "" if grid_lb is None else f"{float(grid_lb):.17g}",
                "" if grid_ub is None else f"{float(grid_ub):.17g}",
                str(_require_int(grid_points, argument="grid_points")),
                bias or "",
                mono or "",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        result_path = tmp_path / "result.json"
        result_text = (
            result_path.read_text(encoding="utf-8") if result_path.exists() else None
        )

    if proc.returncode != 0:
        raise ConvergenceFailure(
            "backend='honestdid' failed while running the R HonestDiD package: "
            f"{proc.stderr.strip() or proc.stdout.strip()}",
            recovery_hint=(
                "Check the local R HonestDiD/jsonlite installation or use "
                "backend='native'."
            ),
            diagnostics={"backend": "honestdid", "returncode": proc.returncode},
        )

    try:
        out = json.loads(result_text) if result_text is not None else None
    except json.JSONDecodeError:
        out = None
    if not isinstance(out, list):
        raise ConvergenceFailure(
            "backend='honestdid': R HonestDiD exited without error but did not "
            "return a readable sensitivity table. R stdout (tail): "
            f"{proc.stdout.strip()[-500:]!r}; stderr (tail): "
            f"{proc.stderr.strip()[-500:]!r}",
            recovery_hint=(
                "Run the same HonestDiD call in R to see its output, check the "
                "R HonestDiD/jsonlite installation, or use backend='native'."
            ),
            diagnostics={
                "backend": "honestdid",
                "honestdid_method": honestdid_method_arg,
                "returncode": proc.returncode,
            },
        )
    rows = []
    for item in out:
        ci_lower = float(item["ci_lower"])
        ci_upper = float(item["ci_upper"])
        rows.append(
            {
                "M": float(item["M"]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "rejects_zero": not (ci_lower <= 0 <= ci_upper),
            }
        )
    return pd.DataFrame(rows)


def _find_rscript() -> Optional[str]:
    """Return an Rscript executable, including the standard macOS R path."""
    candidate = shutil.which("Rscript")
    if candidate:
        return candidate
    for path in (
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
    ):
        if Path(path).exists():
            return path
    return None
