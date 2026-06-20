"""
Rambachan & Roth (2023) honest parallel trends sensitivity analysis.

Computes robust confidence sets for DID treatment effects under
violations of the parallel trends assumption. Instead of testing
whether pre-trends are exactly zero, this approach asks: "How large
could the parallel trends violation be, and would we still draw the
same conclusion?"

The key parameter M (or Delta) bounds the magnitude of possible
violations: |delta_t - delta_{t-1}| <= M for all t, where delta_t is
the deviation from parallel trends at time t.

References
----------
Rambachan, A. and Roth, J. (2023).
"A More Credible Approach to Parallel Trends."
*Review of Economic Studies*, 90(5), 2555-2591. [@rambachan2023more]

Roth, J. (2022).
"Pretest with Caution: Event-Study Estimates after Testing for
Parallel Trends."
*American Economic Review: Insights*, 4(3), 305-322. [@roth2022pretest]
"""

import json
import shutil
import subprocess
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any, Optional, List, Sequence, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility


def _require_string(value: Any, *, argument: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodIncompatibility(
            f"`{argument}` must be a non-empty string.",
            recovery_hint=f"Pass a supported string value for `{argument}`.",
            diagnostics={"argument": argument, "type": type(value).__name__},
        )
    return value


def _require_alpha(alpha: Any) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    out = float(alpha)
    if not np.isfinite(out) or not (0.0 < out < 1.0):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    return out


def _require_int(value: Any, *, argument: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{argument}` must be an integer.",
            recovery_hint=f"Pass an integer relative-time value for `{argument}`.",
            diagnostics={"argument": argument, "value": value},
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{argument}` must be an integer.",
            recovery_hint=f"Pass an integer relative-time value for `{argument}`.",
            diagnostics={"argument": argument, "type": type(value).__name__},
        ) from exc


def _coerce_m_grid(m_grid: Optional[Sequence[float]], *, se_hat: float) -> List[float]:
    if m_grid is None:
        sigma = float(se_hat)
        return [0.0, 0.5 * sigma, sigma, 1.5 * sigma, 2.0 * sigma, 3.0 * sigma]
    try:
        out = [float(m) for m in m_grid]
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "`m_grid` must be a sequence of finite non-negative numbers.",
            recovery_hint="Pass m_grid=[0, 0.5, 1.0] or leave it as None.",
            diagnostics={"argument": "m_grid", "type": type(m_grid).__name__},
        ) from exc
    if not out:
        raise DataInsufficient(
            "`m_grid` must contain at least one value.",
            recovery_hint="Pass at least one M value or leave m_grid as None.",
            diagnostics={"argument": "m_grid"},
        )
    if any((not np.isfinite(m)) or m < 0 for m in out):
        raise MethodIncompatibility(
            "`m_grid` must be a sequence of finite non-negative numbers.",
            recovery_hint="Remove NaN/Inf/negative M values from m_grid.",
            diagnostics={"argument": "m_grid", "values": out},
        )
    return out


def honest_did(
    result: CausalResult,
    e: int = 0,
    m_grid: Optional[List[float]] = None,
    method: str = "smoothness",
    alpha: float = 0.05,
    backend: str = "native",
    honestdid_method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rambachan & Roth (2023) sensitivity analysis for parallel trends.

    Given a DID result (from Callaway-Sant'Anna or Sun-Abraham),
    computes robust confidence intervals for the treatment effect at
    relative time ``e`` under different magnitudes of parallel trends
    violations.

    Parameters
    ----------
    result : CausalResult
        A fitted DID result that contains event study estimates in
        ``result.model_info['event_study']``.
    e : int, default 0
        Relative time period for which to compute robust CIs.
        0 = on-impact effect, 1 = one period after, etc.
    m_grid : list of float, optional
        Grid of M values (maximum violation magnitude) to evaluate.
        Default: [0, 0.5σ, 1σ, 1.5σ, 2σ] where σ is the SE at e.
    method : str, default 'smoothness'
        Type of restriction on violations:
        - ``'smoothness'``: |Δδ_t| ≤ M (bounded change in violation)
        - ``'relative_magnitude'``: |δ_post| ≤ M̄ × max|δ_pre|
    alpha : float, default 0.05
        Significance level.
    backend : {'native', 'honestdid', 'r'}, default 'native'
        ``'native'`` uses StatsPAI's dependency-light analytic
        intervals. ``'honestdid'``/``'r'`` delegates to the R
        ``HonestDiD`` package through ``Rscript`` and returns the
        reference package's finite-sample conditional intervals. The
        R backend is mainly useful when exact parity with
        ``HonestDiD::createSensitivityResults_relativeMagnitudes`` is
        required; the default native path remains available without an
        R installation.
    honestdid_method : {'C-LF', 'Conditional', 'FLCI', 'C-F'}, optional
        Solver method passed to the R ``HonestDiD`` backend. The default
        preserves HonestDiD defaults: ``'C-LF'`` for
        ``method='relative_magnitude'`` and ``'FLCI'`` for
        ``method='smoothness'``. Ignored by ``backend='native'``.

    Returns
    -------
    pd.DataFrame
        Columns: M, ci_lower, ci_upper, rejects_zero.
        Each row is a different M value.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []  # staggered panel: half treated at t=5, half never
    >>> for i in range(30):
    ...     g = 5 if i < 15 else 0
    ...     ui = rng.normal(0, 1)
    ...     for t in range(1, 9):
    ...         post = 1 if (g != 0 and t >= g) else 0
    ...         y = ui + 0.3 * t + 2.0 * post + rng.normal(0, 0.5)
    ...         rows.append({"unit": i, "time": t, "y": y, "g": g})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.sun_abraham(df, y='y', g='g', t='time', i='unit')
    >>> sensitivity = sp.honest_did(r, e=0)
    >>> list(sensitivity.columns)
    ['M', 'ci_lower', 'ci_upper', 'rejects_zero']
    >>> # If rejects_zero=True even at M=1 sigma, the result is robust

    Notes
    -----
    The method constructs the identified set for the treatment effect
    under the restriction that parallel trends violations are bounded:

    For ``method='smoothness'``:
        The post-treatment violation δ_post is bounded by extrapolating
        pre-trend violations. If pre-trends show a slope, the post
        period could continue that slope by at most M per period.

        The robust CI at level 1-α is:
        [θ̂ - bias_bound - z_{α/2} × SE,  θ̂ + bias_bound + z_{α/2} × SE]

        where bias_bound = M × (e + 1) is the worst-case bias under
        the smoothness restriction.

    For ``method='relative_magnitude'``:
        The post-treatment violation is bounded as a multiple of the
        largest pre-treatment violation observed. M̄ = 1 means post
        violations can be as large as the largest pre violation.

    See Rambachan & Roth (2023, *ReStud*), Section 2.

    References
    ----------
    Rambachan, A. and Roth, J. (2023). A more credible approach to parallel
    trends. *Review of Economic Studies*. [@rambachan2023more]
    """
    e = _require_int(e, argument="e")
    method = _require_string(method, argument="method")
    alpha = _require_alpha(alpha)
    backend = _require_string(backend, argument="backend")
    backend_norm = backend.lower().replace("-", "_")
    if backend_norm in {"r", "honestdid", "honest_did", "honestdid_r"}:
        return _honest_did_r_backend(
            result=result,
            e=e,
            m_grid=m_grid,
            method=method,
            alpha=alpha,
            honestdid_method=honestdid_method,
        )
    if backend_norm not in {"native", "statspai"}:
        raise MethodIncompatibility(
            "backend must be 'native', 'honestdid', or 'r'.",
            recovery_hint=(
                "Use backend='native' unless exact R HonestDiD parity is " "required."
            ),
            diagnostics={"backend": backend},
        )

    es = _extract_event_study(result)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Find the target period
    target_row = es[es["relative_time"] == e]
    if len(target_row) == 0:
        raise DataInsufficient(
            f"No event study estimate at relative time e={e}",
            recovery_hint="Choose an `e` value present in the event-study table.",
            diagnostics={"e": e, "available": es["relative_time"].tolist()},
        )

    theta_hat = float(target_row["att"].iloc[0])
    se_hat = float(target_row["se"].iloc[0])

    # Pre-treatment estimates (for calibrating M)
    pre = es[es["relative_time"] < 0].sort_values("relative_time")
    pre_atts = pre["att"].values

    # Default M grid: multiples of SE
    m_grid = _coerce_m_grid(m_grid, se_hat=se_hat)

    rows = []

    if method == "smoothness":
        # Smoothness bound: worst-case bias = M × (number of post periods from base)
        # For relative time e, the bias is bounded by M × (e + 1)
        # (each period can drift by at most M)
        n_drift = max(e + 1, 1)

        for M in m_grid:
            bias_bound = M * n_drift
            ci_lower = theta_hat - bias_bound - z_crit * se_hat
            ci_upper = theta_hat + bias_bound + z_crit * se_hat
            rejects = not (ci_lower <= 0 <= ci_upper)

            rows.append(
                {
                    "M": round(M, 6),
                    "ci_lower": round(ci_lower, 6),
                    "ci_upper": round(ci_upper, 6),
                    "rejects_zero": rejects,
                }
            )

    elif method == "relative_magnitude":
        # Relative magnitude: |δ_post| ≤ M̄ × max|δ_pre|
        if len(pre_atts) == 0:
            max_pre = 0
        else:
            max_pre = np.max(np.abs(pre_atts))

        for M_bar in m_grid:
            bias_bound = M_bar * max_pre if max_pre > 0 else M_bar * se_hat
            ci_lower = theta_hat - bias_bound - z_crit * se_hat
            ci_upper = theta_hat + bias_bound + z_crit * se_hat
            rejects = not (ci_lower <= 0 <= ci_upper)

            rows.append(
                {
                    "M": round(M_bar, 6),
                    "ci_lower": round(ci_lower, 6),
                    "ci_upper": round(ci_upper, 6),
                    "rejects_zero": rejects,
                }
            )

    else:
        raise MethodIncompatibility(
            f"method must be 'smoothness' or 'relative_magnitude', got '{method}'",
            recovery_hint="Use method='smoothness' or method='relative_magnitude'.",
            diagnostics={"method": method},
        )

    return pd.DataFrame(rows)


def _honest_did_r_backend(
    result: CausalResult,
    e: int,
    m_grid: Optional[List[float]],
    method: str,
    alpha: float,
    honestdid_method: Optional[str] = None,
) -> pd.DataFrame:
    """Run the R HonestDiD reference implementation for CI parity."""
    e = _require_int(e, argument="e")
    method = _require_string(method, argument="method")
    alpha = _require_alpha(alpha)
    method_norm = method.lower().replace("-", "_")
    if method_norm not in {"smoothness", "relative_magnitude", "relative_magnitudes"}:
        raise MethodIncompatibility(
            f"method must be 'smoothness' or 'relative_magnitude', got '{method}'",
            recovery_hint="Use method='smoothness' or method='relative_magnitude'.",
            diagnostics={"method": method},
        )
    if honestdid_method is None:
        honestdid_method_arg = (
            "C-LF"
            if method_norm in {"relative_magnitude", "relative_magnitudes"}
            else "FLCI"
        )
    else:
        honestdid_method = _require_string(
            honestdid_method,
            argument="honestdid_method",
        )
        method_aliases = {
            "c_lf": "C-LF",
            "c-lf": "C-LF",
            "clf": "C-LF",
            "conditional": "Conditional",
            "flci": "FLCI",
            "c_f": "C-F",
            "c-f": "C-F",
            "cf": "C-F",
        }
        key = honestdid_method.lower().replace(" ", "_")
        if key not in method_aliases:
            raise MethodIncompatibility(
                "honestdid_method must be one of 'C-LF', 'Conditional', "
                "'FLCI', or 'C-F'.",
                recovery_hint=(
                    "Use an HonestDiD solver method supported by the selected "
                    "restriction."
                ),
                diagnostics={"honestdid_method": honestdid_method},
            )
        honestdid_method_arg = method_aliases[key]
    if method_norm in {"relative_magnitude", "relative_magnitudes"} and (
        honestdid_method_arg not in {"C-LF", "Conditional"}
    ):
        raise MethodIncompatibility(
            "honestdid_method must be 'C-LF' or 'Conditional' for "
            "method='relative_magnitude'.",
            recovery_hint="Use honestdid_method='C-LF' or 'Conditional'.",
            diagnostics={
                "method": method_norm,
                "honestdid_method": honestdid_method_arg,
            },
        )

    rscript = _find_rscript()
    if rscript is None:
        raise ImportError(
            "backend='honestdid' requires Rscript and the R package HonestDiD."
        )

    es = _extract_event_study(result)
    target_row = es[es["relative_time"] == e]
    if len(target_row) == 0:
        raise DataInsufficient(
            f"No event study estimate at relative time e={e}",
            recovery_hint="Choose an `e` value present in the event-study table.",
            diagnostics={"e": e, "available": es["relative_time"].tolist()},
        )

    se_hat = float(target_row["se"].iloc[0])
    m_grid = _coerce_m_grid(m_grid, se_hat=se_hat)

    pre = es[es["relative_time"] < 0].sort_values("relative_time")
    post = es[es["relative_time"] >= 0].sort_values("relative_time")
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
post_idx <- which(post$relative_time == target_e)
if (length(post_idx) != 1) {
  stop("target e must match exactly one post-treatment period")
}
betahat <- c(pre$att, post$att)
ses <- c(pre$se, post$se)
sigma <- diag(ses^2)
l_vec <- rep(0, nrow(post))
l_vec[post_idx] <- 1
l_vec <- matrix(l_vec, ncol = 1)
if (method == "relative_magnitude") {
  sens <- suppressWarnings(
    HonestDiD::createSensitivityResults_relativeMagnitudes(
      betahat = betahat,
      sigma = sigma,
      numPrePeriods = nrow(pre),
      numPostPeriods = nrow(post),
      l_vec = l_vec,
      Mbarvec = grid,
      method = honestdid_method,
      alpha = alpha
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
      alpha = alpha
    )
  )
  out <- data.frame(M = sens$M, ci_lower = sens$lb, ci_upper = sens$ub)
}
cat(jsonlite::toJSON(out, dataframe = "rows", auto_unbox = TRUE,
                     digits = 16, null = "null"))
"""

    with tempfile.TemporaryDirectory(prefix="statspai-honestdid-") as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "event_study.csv"
        script_path = tmp_path / "run_honestdid.R"
        es[["relative_time", "att", "se"]].to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        grid_arg = ",".join(f"{float(m):.17g}" for m in m_grid)
        method_arg = (
            "relative_magnitude"
            if method_norm in {"relative_magnitude", "relative_magnitudes"}
            else "smoothness"
        )
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
            ],
            check=False,
            capture_output=True,
            text=True,
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

    out = json.loads(proc.stdout)
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


def breakdown_m(
    result: CausalResult,
    e: int = 0,
    method: str = "smoothness",
    alpha: float = 0.05,
) -> float:
    """
    Compute the breakdown value of M.

    The breakdown M* is the largest violation magnitude under which
    the treatment effect at relative time ``e`` remains statistically
    significant. Larger M* = more robust.

    Parameters
    ----------
    result : CausalResult
        DID result with event study.
    e : int, default 0
        Relative time period.
    method : str, default 'smoothness'
    alpha : float, default 0.05

    Returns
    -------
    float
        Breakdown value M*. The effect is significant for all M < M*.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []  # staggered panel: half treated at t=5, half never
    >>> for i in range(30):
    ...     g = 5 if i < 15 else 0
    ...     ui = rng.normal(0, 1)
    ...     for t in range(1, 9):
    ...         post = 1 if (g != 0 and t >= g) else 0
    ...         y = ui + 0.3 * t + 2.0 * post + rng.normal(0, 0.5)
    ...         rows.append({"unit": i, "time": t, "y": y, "g": g})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.sun_abraham(df, y='y', g='g', t='time', i='unit')
    >>> m_star = sp.breakdown_m(r, e=0)
    >>> bool(m_star >= 0)  # parallel trends can deviate by up to M* per period
    True

    Notes
    -----
    Formally, M* = sup{M : 0 ∉ CI(M)}.

    For the smoothness restriction with n_drift periods:
    M* = (|θ̂| - z_{α/2} × SE) / n_drift

    See Rambachan & Roth (2023, *ReStud*), Definition 2.
    """
    e = _require_int(e, argument="e")
    alpha = _require_alpha(alpha)
    method = _require_string(method, argument="method")
    if method not in {"smoothness", "relative_magnitude"}:
        raise MethodIncompatibility(
            f"method must be 'smoothness' or 'relative_magnitude', got '{method}'",
            recovery_hint="Use method='smoothness' or method='relative_magnitude'.",
            diagnostics={"method": method},
        )
    es = _extract_event_study(result)

    target = es[es["relative_time"] == e]
    if len(target) == 0:
        raise DataInsufficient(
            f"No estimate at relative time e={e}",
            recovery_hint="Choose an `e` value present in the event-study table.",
            diagnostics={"e": e, "available": es["relative_time"].tolist()},
        )

    theta = float(target["att"].iloc[0])
    se = float(target["se"].iloc[0])
    z_crit = stats.norm.ppf(1 - alpha / 2)

    n_drift = max(e + 1, 1)

    # M* such that |θ̂| - M* × n_drift - z × SE = 0
    m_star = (abs(theta) - z_crit * se) / n_drift
    return float(max(m_star, 0.0))


# ======================================================================
# Polymorphic event-study extractor
# ======================================================================


def _extract_event_study(result: CausalResult) -> pd.DataFrame:
    """Locate the event-study table inside a CausalResult.

    Accepts three flavours, all of which now surface the same shape:

    1. ``callaway_santanna()`` / ``sun_abraham()`` → event study lives in
       ``result.model_info['event_study']`` (legacy).
    2. ``aggte(cs, type='dynamic')`` → event study **is** the ``detail``
       frame (has ``relative_time`` + ``att`` + ``se``).
    3. ``did_multiplegt()`` → event study in ``model_info['event_study']``
       but placebos carry negative ``relative_time`` and dynamics
       non-negative ``relative_time`` — already compatible.

    Raises
    ------
    ValueError
        If no event study can be found.
    """
    info = result.model_info or {}
    es = cast(Optional[pd.DataFrame], info.get("event_study"))
    if es is None and getattr(result, "detail", None) is not None:
        det = result.detail
        if isinstance(det, pd.DataFrame) and {"relative_time", "att", "se"}.issubset(
            det.columns
        ):
            es = det
    if es is None:
        raise MethodIncompatibility(
            "Result does not expose an event-study table.  Supported "
            "inputs: callaway_santanna(), sun_abraham(), did_multiplegt(), "
            "or aggte(result, type='dynamic').",
            recovery_hint=(
                "Pass a DID/event-study result with relative_time, att, and se "
                "columns."
            ),
            diagnostics={"result_type": type(result).__name__},
        )
    # Defensive copy — callers mutate / filter it.
    es = es.copy()
    # If aggte produced uniform-band columns but no explicit 'type', ignore.
    required = {"relative_time", "att", "se"}
    missing = required - set(es.columns)
    if missing:
        raise MethodIncompatibility(
            f"Event-study table is missing required columns: {missing}",
            recovery_hint=(
                "Provide an event-study table with relative_time, att, and se "
                "columns."
            ),
            diagnostics={"missing_columns": sorted(missing)},
        )
    return es


# Citation
CausalResult._CITATIONS["honest_did"] = (
    "@article{rambachan2023more,\n"
    "  title={A More Credible Approach to Parallel Trends},\n"
    "  author={Rambachan, Ashesh and Roth, Jonathan},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={90},\n"
    "  number={5},\n"
    "  pages={2555--2591},\n"
    "  year={2023},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
