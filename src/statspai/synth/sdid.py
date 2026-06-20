"""
Synthetic Difference-in-Differences (SDID).

Python implementation of synthetic DID / SC / DID interfaces inspired
by Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021). Exact R
``synthdid`` numbers are available through ``backend="synthdid"``; the
native implementation is validation-tiered separately.

Three estimators share one optimisation framework:

* **SDID** – unit weights *and* time weights
* **SC**  – unit weights only (classic synthetic control)
* **DID** – uniform unit weights, uniform time weights

All three support *placebo*, *bootstrap*, and *jackknife* standard errors
and return :class:`~statspai.core.results.CausalResult`.

References
----------
Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W.
and Wager, S. (2021).
"Synthetic Difference-in-Differences."
*American Economic Review*, 111(12), 4088-4118. [@arkhangelsky2021synthetic]
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Any, Tuple, Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient

# ======================================================================
# Public API
# ======================================================================


def sdid(
    data: pd.DataFrame,
    outcome: Optional[str] = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    treated_unit: Any = None,
    treatment_time: Any = None,
    *,
    y: Optional[str] = None,
    treat_unit: Any = None,
    treat_time: Any = None,
    method: Literal["sdid", "sc", "did"] = "sdid",
    covariates: Optional[List[str]] = None,
    se_method: Literal["placebo", "bootstrap", "jackknife"] = "placebo",
    n_reps: int = 200,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    backend: Literal["native", "synthdid", "r"] = "native",
) -> CausalResult:
    """
    Synthetic Difference-in-Differences estimator (and SC / DID variants).

    Replicates the R ``synthdid`` package interface.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data in long format.
    outcome : str
        Outcome variable column. Alias ``y=`` accepted for R-style calls.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : scalar or list
        Treated unit(s). Alias ``treat_unit=`` accepted.
    treatment_time : scalar
        First treatment period (inclusive). Alias ``treat_time=`` accepted.
    method : {'sdid', 'sc', 'did'}, default 'sdid'
        * ``'sdid'`` — Synthetic DID (unit + time weights)
        * ``'sc'``   — Synthetic Control (unit weights only)
        * ``'did'``  — DID (uniform weights)
    covariates : list of str, optional
        Reserved for future covariate-adjusted extensions.
    se_method : {'placebo', 'bootstrap', 'jackknife'}, default 'placebo'
        Standard-error method (see Notes).
    n_reps : int, default 200
        Replications for placebo / bootstrap SE.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    backend : {'native', 'synthdid', 'r'}, default 'native'
        ``'native'`` uses StatsPAI's Python implementation.
        ``'synthdid'``/``'r'`` delegates to the R ``synthdid`` package
        through ``Rscript`` and returns the reference package's point
        estimate and ``synthdid_se`` standard error. The R backend is
        mainly for exact cross-language parity claims; the dependency-
        light native implementation remains the default.

    Returns
    -------
    CausalResult
        With ``model_info`` containing unit_weights, time_weights,
        Y_obs, Y_synth, control_units, pre_times, post_times, etc.

    Notes
    -----
    **SE methods** (matching R ``synthdid::vcov``):

    * *placebo* — reassign treatment to each control unit in turn.
    * *bootstrap* — resample control units with replacement.
    * *jackknife* — leave-one-control-unit-out.

    The SDID estimator is:

    .. math::
        \\hat{\\tau}_{sdid} = \\left(
            \\bar{Y}_{\\text{tr,post}}^{\\lambda}
          - \\hat{\\omega}' Y_{\\text{co,post}}^{\\lambda}
        \\right)
        - \\left(
            \\bar{Y}_{\\text{tr,pre}}^{\\lambda}
          - \\hat{\\omega}' Y_{\\text{co,pre}}^{\\lambda}
        \\right)

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.sdid(df, y='packspercapita', unit='state',
    ...                  time='year', treat_unit='California',
    ...                  treat_time=1989)
    >>> bool(result.estimate < 0)  # Prop. 99 reduced cigarette sales
    True
    >>> fig, ax = sp.synthdid_plot(result)  # synthdid-style trajectory plot
    >>> type(ax).__name__
    'Axes'

    Compare all three methods (``method='sc'`` and ``'did'``):

    >>> for m in ['sdid', 'sc', 'did']:  # doctest: +SKIP
    ...     r = sp.sdid(df, y='packspercapita', unit='state',  # doctest: +SKIP
    ...                 time='year', treat_unit='California',  # doctest: +SKIP
    ...                 treat_time=1989, method=m)  # doctest: +SKIP
    ...     print(f"{m:4s}: ATT = {r.estimate:.3f} (SE = {r.se:.3f})")  # doctest: +SKIP

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. and Wager, S.
    (2021). Synthetic difference-in-differences. *American Economic Review*.
    [@arkhangelsky2021synthetic]
    """
    # --- Resolve canonical/legacy parameter names ---------------------
    if outcome is None:
        outcome = y
    if treated_unit is None:
        treated_unit = treat_unit
    if treatment_time is None:
        treatment_time = treat_time

    if outcome is None:
        raise TypeError("sdid: provide `outcome` (or legacy alias `y`)")
    if unit is None or time is None:
        raise TypeError("sdid: `unit` and `time` are required")
    if treated_unit is None:
        raise TypeError("sdid: provide `treated_unit` (or legacy alias `treat_unit`)")
    if treatment_time is None:
        raise TypeError("sdid: provide `treatment_time` (or legacy alias `treat_time`)")

    # Internal variable names kept short for the math below.
    y = outcome
    treat_unit = treated_unit
    treat_time = treatment_time

    backend_norm = backend.lower().replace("-", "_")
    if backend_norm in ("synthdid", "r", "synthdid_r"):
        return _sdid_r_backend(
            data=data,
            outcome=y,
            unit=unit,
            time=time,
            treated_unit=treat_unit,
            treatment_time=treat_time,
            method=method,
            se_method=se_method,
            seed=seed,
            alpha=alpha,
        )
    if backend_norm != "native":
        raise ValueError("Unknown backend. Use 'native', 'synthdid', or 'r'.")

    rng = np.random.default_rng(seed)

    # --- Parse treated units ------------------------------------------
    if not isinstance(treat_unit, (list, tuple, np.ndarray)):
        treat_unit = [treat_unit]

    # --- Build panel matrix -------------------------------------------
    panel = data.pivot_table(
        index=unit,
        columns=time,
        values=y,
        aggfunc="first",
    )
    all_times = sorted(panel.columns.tolist())
    treated_mask = panel.index.isin(treat_unit)
    control_mask = ~treated_mask

    pre_times = [t for t in all_times if t < treat_time]
    post_times = [t for t in all_times if t >= treat_time]

    if len(pre_times) < 2:
        raise DataInsufficient("Need at least 2 pre-treatment periods.")
    if len(post_times) < 1:
        raise DataInsufficient("Need at least 1 post-treatment period.")

    n_tr = int(treated_mask.sum())
    n_co = int(control_mask.sum())
    T_pre = len(pre_times)
    T_post = len(post_times)

    Y_co_pre = panel.loc[control_mask, pre_times].values  # (N_co, T_pre)
    Y_co_post = panel.loc[control_mask, post_times].values  # (N_co, T_post)
    Y_tr_pre = panel.loc[treated_mask, pre_times].values  # (N_tr, T_pre)
    Y_tr_post = panel.loc[treated_mask, post_times].values  # (N_tr, T_post)

    # --- Solve weights ------------------------------------------------
    omega, lam = _compute_weights(
        Y_co_pre,
        Y_co_post,
        Y_tr_pre,
        method,
        n_co,
        T_pre,
        n_tr=n_tr,
    )

    # --- Point estimate -----------------------------------------------
    tau = _estimate_tau(
        Y_co_pre,
        Y_co_post,
        Y_tr_pre,
        Y_tr_post,
        omega,
        lam,
    )

    # --- Standard errors ----------------------------------------------
    if se_method == "placebo":
        se, tau_reps = _se_placebo(
            Y_co_pre,
            Y_co_post,
            Y_tr_pre,
            Y_tr_post,
            method,
            n_co,
            T_pre,
        )
    elif se_method == "bootstrap":
        se, tau_reps = _se_bootstrap(
            Y_co_pre,
            Y_co_post,
            Y_tr_pre,
            Y_tr_post,
            method,
            n_co,
            T_pre,
            n_reps,
            rng,
        )
    elif se_method == "jackknife":
        se, tau_reps = _se_jackknife(
            Y_co_pre,
            Y_co_post,
            Y_tr_pre,
            Y_tr_post,
            method,
            n_co,
            T_pre,
        )
    else:
        raise ValueError(f"Unknown se_method: {se_method!r}")

    z = tau / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (tau - z_crit * se, tau + z_crit * se)

    # --- Build synthetic trajectory for plotting ----------------------
    control_names = panel.index[control_mask].tolist()
    treated_names = panel.index[treated_mask].tolist()

    # Observed treated trajectory (average over treated units)
    Y_tr_all = panel.loc[treated_mask, all_times].values.mean(axis=0)  # (T,)
    # Synthetic trajectory
    Y_synth_pre = omega @ Y_co_pre  # (T_pre,)
    Y_synth_post = omega @ Y_co_post  # (T_post,)
    # Adjust intercept for SDID / DID
    if method in ("sdid", "did"):
        intercept = float(Y_tr_pre.mean(axis=0) @ lam) - float(omega @ (Y_co_pre @ lam))
        Y_synth_pre = Y_synth_pre + intercept
        Y_synth_post = Y_synth_post + intercept
    Y_synth_all = np.concatenate([Y_synth_pre, Y_synth_post])

    # Weight tables
    weight_df = (
        pd.DataFrame(
            {
                "unit": control_names,
                "weight": omega,
            }
        )
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    time_weight_series = pd.Series(lam, index=pre_times, name="time_weight")

    method_labels = {
        "sdid": "Synthetic Difference-in-Differences",
        "sc": "Synthetic Control",
        "did": "Difference-in-Differences",
    }

    model_info = {
        "estimator": method,
        "estimator_label": method_labels[method],
        "backend": "native",
        "validation_tier": "T2_native_reference_parity",
        "reference_backend": "synthdid",
        "validation_note": (
            "StatsPAI native SDID mirrors synthdid's no-covariate "
            "collapsed-form Frank-Wolfe weight solver and zeta scaling. "
            "The Prop. 99 Track A row is native Python reference parity; "
            "backend='synthdid' remains available for users who want to "
            "delegate directly to the R package."
        ),
        "n_treated": n_tr,
        "n_control": n_co,
        "T_pre": T_pre,
        "T_post": T_post,
        "treat_time": treat_time,
        "treated_units": treated_names,
        "control_units": control_names,
        "unit_weights": weight_df,
        "time_weights": time_weight_series,
        "se_method": se_method,
        "n_reps": n_reps if se_method in ("bootstrap",) else None,
        "Y_obs": pd.Series(Y_tr_all, index=all_times, name="treated"),
        "Y_synth": pd.Series(Y_synth_all, index=all_times, name="synthetic"),
        "pre_times": pre_times,
        "post_times": post_times,
        "all_times": all_times,
    }

    return CausalResult(
        method=f"{method_labels[method]} (Arkhangelsky et al. 2021)",
        estimand="ATT",
        estimate=float(tau),
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        model_info=model_info,
        _citation_key="sdid",
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
        "The synthdid backend requires Rscript plus the R packages "
        "'synthdid' and 'jsonlite'. Install R or use backend='native'."
    )


def _sdid_r_backend(
    *,
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    method: str,
    se_method: str,
    seed: Optional[int],
    alpha: float,
) -> CausalResult:
    """Delegate SDID/SC/DID estimation to R ``synthdid`` for parity."""
    method_norm = method.lower()
    if method_norm not in {"sdid", "sc", "did"}:
        raise ValueError("method must be one of 'sdid', 'sc', or 'did'")
    if se_method not in {"placebo", "bootstrap", "jackknife"}:
        raise ValueError(
            "se_method must be one of 'placebo', 'bootstrap', or 'jackknife'"
        )

    treated_units = (
        list(treated_unit)
        if isinstance(treated_unit, (list, tuple, np.ndarray, pd.Index))
        else [treated_unit]
    )
    unit_col = "statspai_unit"
    time_col = "statspai_time"
    outcome_col = "statspai_outcome"
    treatment_col = "statspai_treated"
    panel_df = pd.DataFrame(
        {
            unit_col: data[unit],
            time_col: data[time],
            outcome_col: data[outcome],
        }
    )
    panel_df[treatment_col] = (
        data[unit].isin(treated_units) & (data[time] >= treatment_time)
    ).astype(int)
    panel_df = panel_df.sort_values([unit_col, time_col]).reset_index(drop=True)

    r_script = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9) {
  stop("expected 9 arguments: input output outcome unit time treatment method se_method alpha_seed")
}
input_path <- args[[1]]
output_path <- args[[2]]
outcome_col <- args[[3]]
unit_col <- args[[4]]
time_col <- args[[5]]
treatment_col <- args[[6]]
method <- args[[7]]
se_method <- args[[8]]
alpha_seed <- strsplit(args[[9]], ":", fixed = TRUE)[[1]]
alpha <- as.numeric(alpha_seed[[1]])
seed <- suppressWarnings(as.integer(alpha_seed[[2]]))

suppressPackageStartupMessages({
  library(synthdid)
  library(jsonlite)
})

if (!is.na(seed)) {
  set.seed(seed)
}

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
panel <- do.call(
  synthdid::panel.matrices,
  list(
    panel = df,
    unit = unit_col,
    time = time_col,
    outcome = outcome_col,
    treatment = treatment_col
  )
)

estimate <- switch(
  method,
  sdid = synthdid::synthdid_estimate(Y = panel$Y, N0 = panel$N0, T0 = panel$T0),
  sc = synthdid::sc_estimate(Y = panel$Y, N0 = panel$N0, T0 = panel$T0),
  did = synthdid::did_estimate(Y = panel$Y, N0 = panel$N0, T0 = panel$T0),
  stop(paste("unknown method:", method))
)

se <- as.numeric(synthdid::synthdid_se(estimate, method = se_method))
tau <- as.numeric(estimate)
z <- qnorm(1 - alpha / 2)

payload <- list(
  estimate = tau,
  se = se,
  ci_lower = tau - z * se,
  ci_upper = tau + z * se,
  n_obs = nrow(df),
  n_units = nrow(panel$Y),
  n_times = ncol(panel$Y),
  N0 = panel$N0,
  T0 = panel$T0
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
    with tempfile.TemporaryDirectory(prefix="statspai_sdid_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "panel.csv"
        script_path = tmp_path / "sdid_backend.R"
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
                outcome_col,
                unit_col,
                time_col,
                treatment_col,
                method_norm,
                se_method,
                f"{float(alpha)}:{seed_arg}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R synthdid backend failed. stderr:\n" f"{proc.stderr.strip()}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    tau = float(payload["estimate"])
    se = float(payload["se"])
    pvalue = float(2 * (1 - stats.norm.cdf(abs(tau / se)))) if se > 0 else np.nan
    ci = (float(payload["ci_lower"]), float(payload["ci_upper"]))

    method_labels = {
        "sdid": "Synthetic Difference-in-Differences",
        "sc": "Synthetic Control",
        "did": "Difference-in-Differences",
    }
    n_control = int(payload["N0"])
    n_units = int(payload["n_units"])
    t_pre = int(payload["T0"])
    t_total = int(payload["n_times"])
    model_info = {
        "estimator": method_norm,
        "estimator_label": method_labels[method_norm],
        "backend": "synthdid",
        "r_package": "synthdid",
        "validation_tier": "reference_backend_bridge",
        "reference_backend": "synthdid",
        "validation_note": (
            "This result delegates to synthdid::synthdid_estimate/"
            "sc_estimate/did_estimate and synthdid::synthdid_se. It is useful "
            "for exact reference-package numbers, but it is not counted as "
            "native Python parity evidence because the reference backend is "
            "the estimator itself."
        ),
        "n_treated": n_units - n_control,
        "n_control": n_control,
        "T_pre": t_pre,
        "T_post": t_total - t_pre,
        "treat_time": treatment_time,
        "treated_units": treated_units,
        "se_method": se_method,
        "n_reps": None,
    }

    return CausalResult(
        method=f"{method_labels[method_norm]} (R synthdid reference backend)",
        estimand="ATT",
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(payload["n_obs"]),
        model_info=model_info,
        _citation_key="sdid",
    )


# ======================================================================
# Convenience wrappers (mirror R package names)
# ======================================================================


def synthdid_estimate(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treat_unit: Any,
    treat_time: Any,
    **kw: Any,
) -> CausalResult:
    """R-style alias: ``synthdid::synthdid_estimate``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.synthdid_estimate(
    ...     df, y="y", unit="unit", time="time",
    ...     treat_unit=0, treat_time=7, seed=42,
    ... )
    >>> round(res.estimate, 2)  # true effect = 3.0
    3.09
    """
    return sdid(data, y, unit, time, treat_unit, treat_time, method="sdid", **kw)


def sc_estimate(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treat_unit: Any,
    treat_time: Any,
    **kw: Any,
) -> CausalResult:
    """R-style alias: ``synthdid::sc_estimate``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.sc_estimate(
    ...     df, y="y", unit="unit", time="time",
    ...     treat_unit=0, treat_time=7, seed=42,
    ... )
    >>> round(res.estimate, 1)  # synthetic-control variant
    2.5
    """
    return sdid(data, y, unit, time, treat_unit, treat_time, method="sc", **kw)


def did_estimate(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treat_unit: Any,
    treat_time: Any,
    **kw: Any,
) -> CausalResult:
    """R-style alias: ``synthdid::did_estimate``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.did_estimate(
    ...     df, y="y", unit="unit", time="time",
    ...     treat_unit=0, treat_time=7, seed=42,
    ... )
    >>> round(res.estimate, 2)  # true effect = 3.0
    3.01
    """
    return sdid(data, y, unit, time, treat_unit, treat_time, method="did", **kw)


# ======================================================================
# Placebo analysis
# ======================================================================


def synthdid_placebo(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treat_unit: Any,
    treat_time: Any,
    method: Literal["sdid", "sc", "did"] = "sdid",
    **kw: Any,
) -> pd.DataFrame:
    """
    Run placebo estimates assigning treatment to each control unit.

    Replicates ``synthdid::synthdid_placebo``.

    Accepts the same arguments as :func:`sdid`, plus any extra keyword
    arguments.

    Returns
    -------
    pd.DataFrame
        One row per control unit with columns:
        ``unit``, ``estimate``, ``se``, ``pvalue``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> pl = sp.synthdid_placebo(
    ...     df, y="y", unit="unit", time="time",
    ...     treat_unit=0, treat_time=7, seed=42,
    ... )
    >>> list(pl.columns)
    ['unit', 'estimate', 'se', 'pvalue']
    >>> len(pl)  # one row per control unit
    11
    """
    if not isinstance(treat_unit, (list, tuple, np.ndarray)):
        treat_unit = [treat_unit]

    panel = data.pivot_table(index=unit, columns=time, values=y, aggfunc="first")
    control_units = [u for u in panel.index if u not in treat_unit]

    # Subset data to exclude the real treated units
    control_data = data[~data[unit].isin(treat_unit)]

    # Merge defaults with caller overrides
    placebo_kw: dict[str, Any] = {"n_reps": 50}
    placebo_kw.update(kw)

    rows: list[dict[str, Any]] = []
    for cu in control_units:
        try:
            r = sdid(
                control_data,
                y=y,
                unit=unit,
                time=time,
                treat_unit=cu,
                treat_time=treat_time,
                method=method,
                **placebo_kw,
            )
            rows.append(
                {
                    "unit": cu,
                    "estimate": r.estimate,
                    "se": r.se,
                    "pvalue": r.pvalue,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


# ======================================================================
# Plotting
# ======================================================================


def synthdid_plot(
    result: CausalResult,
    ax: Any = None,
    figsize: Tuple[float, float] = (10, 6),
    treated_color: str = "#2C3E50",
    synth_color: str = "#E74C3C",
    ci_alpha: float = 0.15,
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Plot observed vs synthetic trajectory.

    Replicates ``synthdid::plot.synthdid_estimate``.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`sdid`.
    ax : matplotlib Axes, optional
    figsize : tuple
    treated_color, synth_color : str
    ci_alpha : float
    title : str, optional

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.synthdid_estimate(
    ...     df, y="y", unit="unit", time="time",
    ...     treat_unit=0, treat_time=7, seed=42,
    ... )
    >>> fig, ax = sp.synthdid_plot(res)
    >>> type(ax).__name__
    'Axes'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    mi = result.model_info
    times = mi["all_times"]
    Y_obs = mi["Y_obs"].values
    Y_syn = mi["Y_synth"].values
    treat_time = mi["treat_time"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(times, Y_obs, color=treated_color, linewidth=2, label="Treated")
    ax.plot(
        times, Y_syn, color=synth_color, linewidth=2, linestyle="--", label="Synthetic"
    )

    # Shade post-treatment gap
    post_mask = np.array([t >= treat_time for t in times])
    ax.fill_between(
        np.array(times)[post_mask],
        Y_obs[post_mask],
        Y_syn[post_mask],
        alpha=ci_alpha,
        color=synth_color,
        label="Treatment effect",
    )

    ax.axvline(
        x=treat_time,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label="Treatment onset",
    )

    # Time weight shading (pre-period emphasis)
    if mi.get("estimator") in ("sdid",):
        tw = mi["time_weights"]
        pre_t = mi["pre_times"]
        max_w = tw.max() if tw.max() > 0 else 1
        for t_val, w_val in zip(pre_t, tw.values):
            ax.axvspan(
                t_val - 0.4, t_val + 0.4, alpha=0.08 * (w_val / max_w), color="blue"
            )

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Outcome", fontsize=11)
    label = mi.get("estimator_label", result.method)
    ax.set_title(title or f"{label}: ATT = {result.estimate:.3f}", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def synthdid_units_plot(
    result: CausalResult,
    top_n: int = 10,
    ax: Any = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[Any, Any]:
    """
    Horizontal bar chart of unit weight contributions.

    Replicates ``synthdid::synthdid_units_plot``.

    Parameters
    ----------
    result : CausalResult
    top_n : int
        Show the top-N donors by weight.
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.synthdid_estimate(
    ...     df, y="y", unit="unit", time="time",
    ...     treat_unit=0, treat_time=7, seed=42,
    ... )
    >>> fig, ax = sp.synthdid_units_plot(res)
    >>> type(ax).__name__
    'Axes'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    w = result.model_info["unit_weights"].head(top_n).copy()
    w = w[w["weight"] > 1e-6].sort_values("weight", ascending=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.barh(w["unit"].astype(str), w["weight"], color="#3498DB")
    ax.set_xlabel("Weight", fontsize=11)
    ax.set_title("Donor Unit Weights", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def synthdid_rmse_plot(
    result: CausalResult,
    ax: Any = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[Any, Any]:
    """
    Pre-treatment RMSE of treated vs synthetic trajectory.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.synthdid_estimate(
    ...     df, y="y", unit="unit", time="time",
    ...     treat_unit=0, treat_time=7, seed=42,
    ... )
    >>> fig, ax = sp.synthdid_rmse_plot(res)
    >>> type(ax).__name__
    'Axes'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    mi = result.model_info
    pre_t = mi["pre_times"]
    Y_obs_pre = mi["Y_obs"][pre_t].values
    Y_syn_pre = mi["Y_synth"][pre_t].values
    gap = Y_obs_pre - Y_syn_pre

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(range(len(pre_t)), gap**2, color="#95A5A6", alpha=0.7)
    ax.set_xticks(range(len(pre_t)))
    ax.set_xticklabels([str(t) for t in pre_t], rotation=45, fontsize=8)
    ax.set_ylabel("Squared Gap", fontsize=11)
    rmse = np.sqrt(np.mean(gap**2))
    ax.set_title(f"Pre-treatment Fit (RMSE = {rmse:.4f})", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# Example dataset
# ======================================================================


def california_prop99() -> pd.DataFrame:
    """
    California Proposition 99 tobacco control dataset.

    Returns a balanced panel of per-capita cigarette sales for 39 US states,
    1970-2000. California implemented Proposition 99 in 1989.

    This is the canonical ``synthdid`` example dataset.

    Returns
    -------
    pd.DataFrame
        Columns: ``state``, ``year``, ``packspercapita``, ``treated``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> list(df.columns)
    ['state', 'year', 'packspercapita', 'treated']
    >>> result = sp.sdid(df, y='packspercapita', unit='state',
    ...                  time='year', treat_unit='California',
    ...                  treat_time=1989)
    >>> bool(result.estimand == 'ATT')
    True
    """
    # Simulated data matching the structure of the R dataset.
    # 39 states × 31 years (1970-2000). California treated in 1989.
    rng = np.random.default_rng(99)

    states = [
        "Alabama",
        "Arkansas",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Georgia",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Mexico",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
        "California",
    ]

    years = list(range(1970, 2001))
    treat_year = 1989
    n_states = len(states)

    # Generate plausible smoking data
    base_levels = rng.uniform(60, 140, n_states)
    time_trend = -1.2  # national decline
    state_trends = rng.normal(0, 0.3, n_states)

    rows = []
    for i, st in enumerate(states):
        for yr in years:
            t = yr - 1970
            val = (
                base_levels[i] + time_trend * t + state_trends[i] * t + rng.normal(0, 3)
            )
            # California treatment effect (≈ -25 packs decline post-1989)
            if st == "California" and yr >= treat_year:
                val -= 25 * (1 - np.exp(-0.3 * (yr - treat_year + 1)))
            rows.append(
                {
                    "state": st,
                    "year": yr,
                    "packspercapita": max(val, 5),  # floor at 5
                    "treated": 1 if st == "California" and yr >= treat_year else 0,
                }
            )

    return pd.DataFrame(rows)


# ======================================================================
# Internal: Weight solvers
# ======================================================================


def _sdid_noise_level(Y_co_pre: np.ndarray) -> float:
    r"""Noise scale used by \pkg{synthdid}'s regularisation: the standard
    deviation of the control units' pre-treatment first differences
    (Arkhangelsky et al. 2021, eq. for $\hat\sigma$)."""
    if Y_co_pre.shape[1] < 2:
        return 1.0
    diffs = np.diff(Y_co_pre, axis=1).ravel()
    sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 1.0
    return sd if sd > 0 else 1.0


def _compute_weights(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
    method: str,
    n_co: int,
    T_pre: int,
    n_tr: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unit weights (omega) and time weights (lambda).

    The SDID/SC regularisation and optimisation mirrors ``synthdid``:
    collapse the panel to control rows plus the treated-average row, then
    solve the two simplex problems with the Frank-Wolfe routine used by
    ``synthdid:::sc.weight.fw``. This is intentionally not a generic SLSQP
    least-squares fit; the R routine's scaling, intercept handling, and
    sparsification materially affect the Prop. 99 benchmark.
    """
    T_post = int(Y_co_post.shape[1])

    sigma = _sdid_noise_level(Y_co_pre)
    zeta_omega = ((max(n_tr, 1) * max(T_post, 1)) ** 0.25) * sigma
    zeta_lambda = 1e-6 * sigma
    min_decrease = 1e-5 * sigma

    collapsed = _collapsed_form(Y_co_pre, Y_co_post, Y_tr_pre)

    if method == "did":
        omega = np.ones(n_co) / n_co
        lam = np.ones(T_pre) / T_pre
    elif method == "sc":
        # R synthdid::sc_estimate fixes pre-period lambda at zero and uses
        # an almost-unregularised no-intercept SC unit-weight problem.
        omega = _solve_unit_weights(
            collapsed[:, :T_pre].T,
            zeta=zeta_lambda,
            intercept=False,
            min_decrease=min_decrease,
        )
        lam = np.zeros(T_pre)
    else:  # sdid
        lam = _solve_time_weights(
            collapsed[:n_co, :],
            zeta=zeta_lambda,
            intercept=True,
            min_decrease=min_decrease,
        )
        omega = _solve_unit_weights(
            collapsed[:, :T_pre].T,
            zeta=zeta_omega,
            intercept=True,
            min_decrease=min_decrease,
        )

    return omega, lam


def _collapsed_form(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
) -> np.ndarray:
    """Mirror ``synthdid:::collapsed.form`` for the no-covariate case."""
    control_block = np.column_stack([Y_co_pre, Y_co_post.mean(axis=1)])
    # The final treated-post cell is not used by either weight problem.
    treated_row = np.r_[Y_tr_pre.mean(axis=0), np.nan]
    return np.vstack([control_block, treated_row])


def _estimate_tau(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
    Y_tr_post: np.ndarray,
    omega: np.ndarray,
    lam: np.ndarray,
) -> float:
    """
    Weighted DID estimator:
      τ̂ = (ȳ_tr_post - ω'Y_co_post_mean) - (ȳ_tr_pre_λ - ω'(Y_co_pre @ λ))
    """
    # Post-treatment: simple average over all post periods
    y_tr_post_mean = Y_tr_post.mean()
    y_co_post_omega = float(omega @ Y_co_post.mean(axis=1))

    # Pre-treatment: λ-weighted
    y_tr_pre_lam = float(Y_tr_pre.mean(axis=0) @ lam)
    y_co_pre_omega_lam = float(omega @ (Y_co_pre @ lam))

    return float(
        (y_tr_post_mean - y_co_post_omega) - (y_tr_pre_lam - y_co_pre_omega_lam)
    )


def _solve_unit_weights(
    collapsed_pre_transpose: np.ndarray,
    zeta: float,
    intercept: bool,
    min_decrease: float,
) -> np.ndarray:
    return _solve_synthdid_simplex(
        collapsed_pre_transpose,
        zeta=zeta,
        intercept=intercept,
        min_decrease=min_decrease,
    )


def _solve_time_weights(
    collapsed_controls: np.ndarray,
    zeta: float,
    intercept: bool,
    min_decrease: float,
) -> np.ndarray:
    return _solve_synthdid_simplex(
        collapsed_controls,
        zeta=zeta,
        intercept=intercept,
        min_decrease=min_decrease,
    )


def _solve_synthdid_simplex(
    Y: np.ndarray,
    *,
    zeta: float,
    intercept: bool,
    min_decrease: float,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
) -> np.ndarray:
    """Replicate ``synthdid:::sc.weight.fw`` plus default sparsification."""
    first = _sc_weight_fw(
        Y,
        zeta=zeta,
        intercept=intercept,
        min_decrease=min_decrease,
        max_iter=max_iter_pre_sparsify,
    )
    return _sc_weight_fw(
        Y,
        zeta=zeta,
        intercept=intercept,
        weights=_sparsify_function(first),
        min_decrease=min_decrease,
        max_iter=max_iter,
    )


def _sc_weight_fw(
    Y: np.ndarray,
    *,
    zeta: float,
    intercept: bool,
    weights: Optional[np.ndarray] = None,
    min_decrease: float,
    max_iter: int,
) -> np.ndarray:
    """Frank-Wolfe simplex solver matching ``synthdid:::sc.weight.fw``."""
    Y_work = np.asarray(Y, dtype=float).copy()
    n_rows, n_cols = Y_work.shape
    n_weights = n_cols - 1
    if weights is None:
        weights = np.ones(n_weights) / n_weights
    else:
        weights = np.asarray(weights, dtype=float).copy()

    if intercept:
        Y_work = Y_work - Y_work.mean(axis=0, keepdims=True)

    A = Y_work[:, :n_weights]
    b = Y_work[:, n_weights]
    eta = n_rows * float(zeta) ** 2
    vals: list[float] = []

    for _ in range(max_iter):
        ax = A @ weights
        half_grad = (ax - b) @ A + eta * weights
        vertex = int(np.argmin(half_grad))
        direction = -weights.copy()
        direction[vertex] += 1.0
        if np.all(direction == 0):
            break
        err_direction = A[:, vertex] - ax
        denom = float(np.sum(err_direction**2) + eta * np.sum(direction**2))
        step = 0.0 if denom <= 0 else -float(half_grad @ direction) / denom
        step = min(1.0, max(0.0, step))
        weights = weights + step * direction
        err = Y_work @ np.r_[weights, -1.0]
        vals.append(float(zeta**2 * np.sum(weights**2) + np.sum(err**2) / n_rows))
        if len(vals) >= 2 and vals[-2] - vals[-1] <= min_decrease**2:
            break

    return weights


def _sparsify_function(weights: np.ndarray) -> np.ndarray:
    """Mirror ``synthdid:::sparsify_function``."""
    sparse = np.asarray(weights, dtype=float).copy()
    cutoff = float(np.max(sparse)) / 4
    sparse[sparse <= cutoff] = 0.0
    total = float(np.sum(sparse))
    if total <= 0:
        return np.ones_like(sparse) / len(sparse)
    return sparse / total


# ======================================================================
# Internal: Standard error methods
# ======================================================================


def _se_placebo(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
    Y_tr_post: np.ndarray,
    method: str,
    n_co: int,
    T_pre: int,
) -> Tuple[float, np.ndarray]:
    """
    Deterministic placebo SE: assign treatment to each control unit in turn.

    This keeps the native default reproducible and fast. The R ``synthdid``
    package uses random placebo replications for ``synthdid_se``; the Track A
    comparison therefore treats the SE as a small Monte Carlo/convention
    tolerance while holding the ATT itself to strict reference parity.
    """
    taus: list[float] = []
    for i in range(n_co):
        Y_pl_tr_pre = Y_co_pre[i : i + 1, :]
        Y_pl_tr_post = Y_co_post[i : i + 1, :]
        idx = [j for j in range(n_co) if j != i]
        Y_pl_co_pre = Y_co_pre[idx, :]
        Y_pl_co_post = Y_co_post[idx, :]

        n_co_pl = len(idx)
        try:
            omega_pl, lam_pl = _compute_weights(
                Y_pl_co_pre,
                Y_pl_co_post,
                Y_pl_tr_pre,
                method,
                n_co_pl,
                T_pre,
            )
            tau_pl = _estimate_tau(
                Y_pl_co_pre,
                Y_pl_co_post,
                Y_pl_tr_pre,
                Y_pl_tr_post,
                omega_pl,
                lam_pl,
            )
            taus.append(tau_pl)
        except Exception:
            continue

    taus_arr = np.asarray(taus, dtype=float)
    se = float(np.std(taus_arr, ddof=1)) if len(taus_arr) > 1 else 0.0
    return se, taus_arr


def _se_bootstrap(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
    Y_tr_post: np.ndarray,
    method: str,
    n_co: int,
    T_pre: int,
    n_reps: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    """
    Bootstrap SE: resample control units with replacement.
    """
    taus = np.zeros(n_reps)
    for b in range(n_reps):
        idx = rng.choice(n_co, size=n_co, replace=True)
        Y_co_pre_b = Y_co_pre[idx]
        Y_co_post_b = Y_co_post[idx]

        try:
            omega_b, lam_b = _compute_weights(
                Y_co_pre_b,
                Y_co_post_b,
                Y_tr_pre,
                method,
                n_co,
                T_pre,
            )
            taus[b] = _estimate_tau(
                Y_co_pre_b,
                Y_co_post_b,
                Y_tr_pre,
                Y_tr_post,
                omega_b,
                lam_b,
            )
        except Exception:
            taus[b] = np.nan

    taus = taus[~np.isnan(taus)]
    se = float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0
    return se, taus


def _se_jackknife(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
    Y_tr_post: np.ndarray,
    method: str,
    n_co: int,
    T_pre: int,
) -> Tuple[float, np.ndarray]:
    """
    Jackknife SE: leave-one-control-unit-out.
    """
    taus: list[float] = []
    for i in range(n_co):
        idx = [j for j in range(n_co) if j != i]
        Y_co_pre_j = Y_co_pre[idx]
        Y_co_post_j = Y_co_post[idx]
        n_co_j = len(idx)

        try:
            omega_j, lam_j = _compute_weights(
                Y_co_pre_j,
                Y_co_post_j,
                Y_tr_pre,
                method,
                n_co_j,
                T_pre,
            )
            tau_j = _estimate_tau(
                Y_co_pre_j,
                Y_co_post_j,
                Y_tr_pre,
                Y_tr_post,
                omega_j,
                lam_j,
            )
            taus.append(tau_j)
        except Exception:
            continue

    taus_arr = np.asarray(taus, dtype=float)
    n = len(taus_arr)
    if n > 1:
        tau_bar = float(taus_arr.mean())
        se = float(np.sqrt((n - 1) / n * np.sum((taus_arr - tau_bar) ** 2)))
    else:
        se = 0.0
    return se, taus_arr


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS["sdid"] = (
    "@article{arkhangelsky2021synthetic,\n"
    "  title={Synthetic Difference-in-Differences},\n"
    "  author={Arkhangelsky, Dmitry and Athey, Susan and "
    "Hirshberg, David A. and Imbens, Guido W. and Wager, Stefan},\n"
    "  journal={American Economic Review},\n"
    "  volume={111},\n"
    "  number={12},\n"
    "  pages={4088--4118},\n"
    "  year={2021},\n"
    "  publisher={American Economic Association}\n"
    "}"
)
