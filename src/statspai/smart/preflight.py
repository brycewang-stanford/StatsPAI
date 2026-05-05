"""Method-specific pre-estimation diagnostics.

``sp.preflight(data, method, **kwargs)`` runs cheap, method-specific
shape and content checks BEFORE the agent commits to an expensive
estimator call. Different from the neighbours:

* :func:`statspai.smart.check_identification` — *design-level*
  diagnostics for an already-declared design (DID / RD / IV /
  observational). Heavier and broader.
* :func:`statspai.smart.assumption_audit` — heavyweight: re-runs
  statistical tests against the data after the model is fit.
* :func:`statspai.smart.audit` — read-only checklist of robustness
  evidence ON a fitted result.

``preflight`` answers: "if I call ``sp.{method}(data, ...)`` with these
arguments, will it work, and is the data the right shape?" — a quick
gate the agent can run first to avoid wasting tokens on bad calls.

Per-method check tables cover the curated agent-tool surface
(regress / did / callaway_santanna / rdrobust / ivreg / ebalance);
unknown methods get the universal sanity checks only (data is a
non-empty DataFrame, sample size sanity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd


# ====================================================================== #
#  Check primitive
# ====================================================================== #


CheckResult = Tuple[str, str, Dict[str, Any]]
"""``(status, message, evidence)`` — status in {passed, warning, failed}."""


@dataclass(frozen=True)
class _Check:
    name: str
    question: str
    fn: Callable[[pd.DataFrame, Dict[str, Any]], CheckResult]


def _passed(message: str = "OK", **evidence: Any) -> CheckResult:
    return ("passed", message, dict(evidence))


def _warning(message: str, **evidence: Any) -> CheckResult:
    return ("warning", message, dict(evidence))


def _failed(message: str, **evidence: Any) -> CheckResult:
    return ("failed", message, dict(evidence))


# ====================================================================== #
#  Universal checks (run for every method)
# ====================================================================== #


def _check_dataframe(data: pd.DataFrame, _kwargs: Dict[str, Any]
                      ) -> CheckResult:
    if not isinstance(data, pd.DataFrame):
        return _failed(
            f"data must be a pandas DataFrame; got {type(data).__name__}.",
            actual_type=type(data).__name__,
        )
    return _passed(n_rows=len(data), n_cols=len(data.columns))


def _check_non_empty(data: pd.DataFrame, _kwargs: Dict[str, Any]
                      ) -> CheckResult:
    n = len(data)
    if n == 0:
        return _failed("DataFrame is empty (0 rows).", n=0)
    return _passed(n=n)


def _make_check_min_n(threshold: int) -> Callable[
        [pd.DataFrame, Dict[str, Any]], CheckResult]:
    def _fn(data: pd.DataFrame, _kwargs: Dict[str, Any]) -> CheckResult:
        n = len(data)
        if n < threshold:
            return _warning(
                f"n = {n} below typical minimum {threshold} for this "
                "method; estimates may be high-variance.",
                n=n, threshold=threshold,
            )
        return _passed(n=n, threshold=threshold)
    return _fn


def _column_exists(data: pd.DataFrame, kwargs: Dict[str, Any],
                    arg_name: str) -> CheckResult:
    col = kwargs.get(arg_name)
    if col is None:
        return _failed(
            f"required argument {arg_name!r} not provided.",
            arg_name=arg_name,
        )
    if col not in data.columns:
        return _failed(
            f"column {col!r} (passed as {arg_name}={col!r}) not found "
            f"in DataFrame.",
            arg_name=arg_name, column=col,
            available=list(data.columns)[:20],
        )
    return _passed(column=col)


def _check_treat_binary(data: pd.DataFrame, kwargs: Dict[str, Any]
                         ) -> CheckResult:
    col = kwargs.get("treat") or kwargs.get("treatment")
    if col is None or col not in data.columns:
        return _failed("treat/treatment column not specified or missing.",
                        arg_name="treat")
    series = data[col]
    # Dtype gate first: a string column with values "0" / "1" (the
    # CSV-without-dtype-enforcement footgun) would otherwise reach
    # the values-check below and emit a misleading "will be coerced"
    # warning. Estimators don't auto-coerce strings — fail fast.
    if not (pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)):
        return _failed(
            f"treatment column {col!r} has dtype {series.dtype}; must "
            "be numeric (0/1) or boolean. String labels are not "
            "auto-coerced.",
            column=col, dtype=str(series.dtype),
        )
    vals = pd.unique(series.dropna())
    if len(vals) <= 1:
        return _failed(
            f"treatment {col!r} has only {len(vals)} unique value(s); "
            "needs 2 (binary 0/1).",
            column=col, n_unique=len(vals),
        )
    if len(vals) > 2:
        return _failed(
            f"treatment {col!r} has {len(vals)} unique values; this "
            "method requires binary (0/1). Use sp.callaway_santanna or "
            "sp.multi_treatment for non-binary cases.",
            column=col, n_unique=len(vals),
            unique_values=[v.item() if hasattr(v, 'item') else v
                            for v in list(vals)[:10]],
        )
    # Two unique values: confirm they're 0/1-like.
    sorted_vals = sorted(vals)
    if not (sorted_vals[0] in (0, False) and sorted_vals[1] in (1, True)):
        return _warning(
            f"treatment {col!r} is binary but values are "
            f"{sorted_vals!r}; will be coerced to 0/1.",
            column=col, values=list(sorted_vals),
        )
    return _passed(column=col, values=[0, 1])


def _check_time_has_two_periods(data: pd.DataFrame,
                                 kwargs: Dict[str, Any]) -> CheckResult:
    col = kwargs.get("time") or kwargs.get("t")
    if col is None or col not in data.columns:
        return _failed("time column not specified or missing.",
                        arg_name="time")
    n_periods = data[col].nunique(dropna=True)
    if n_periods < 2:
        return _failed(
            f"time column {col!r} has {n_periods} period(s); DID needs "
            "≥ 2 (pre + post).",
            column=col, n_periods=n_periods,
        )
    return _passed(column=col, n_periods=int(n_periods))


def _check_running_var_continuous(data: pd.DataFrame,
                                   kwargs: Dict[str, Any]) -> CheckResult:
    col = kwargs.get("x") or kwargs.get("running_var")
    if col is None or col not in data.columns:
        return _failed(
            "running variable (x / running_var) not specified or missing.",
            arg_name="x",
        )
    series = data[col]
    if not pd.api.types.is_numeric_dtype(series):
        return _failed(
            f"running variable {col!r} is not numeric (dtype={series.dtype}).",
            column=col, dtype=str(series.dtype),
        )
    n_unique = series.nunique(dropna=True)
    if n_unique < 30:
        return _warning(
            f"running variable {col!r} has only {n_unique} unique values; "
            "RD typically needs a continuous score (≥ 30). Consider "
            "discrete-RD (sp.rd_discrete) or check the data.",
            column=col, n_unique=int(n_unique),
        )
    return _passed(column=col, n_unique=int(n_unique))


def _check_id_column_for_staggered(data: pd.DataFrame,
                                    kwargs: Dict[str, Any]) -> CheckResult:
    col = kwargs.get("i") or kwargs.get("id") or kwargs.get("unit")
    if col is None:
        return _failed(
            "staggered DID requires a unit identifier (i / id / unit).",
            arg_name="i",
        )
    if col not in data.columns:
        return _failed(
            f"unit column {col!r} not found in DataFrame.",
            column=col,
        )
    n_units = data[col].nunique(dropna=True)
    if n_units < 5:
        return _warning(
            f"only {n_units} unique units; staggered DID inference is "
            "unstable with so few units.",
            column=col, n_units=int(n_units),
        )
    return _passed(column=col, n_units=int(n_units))


def _check_covariates_exist(data: pd.DataFrame, kwargs: Dict[str, Any]
                             ) -> CheckResult:
    covs = kwargs.get("covariates") or []
    if not isinstance(covs, (list, tuple)):
        return _failed(
            f"covariates must be a list of column names; got "
            f"{type(covs).__name__}.",
            actual_type=type(covs).__name__,
        )
    if not covs:
        return _warning(
            "no covariates supplied; effective sample is just "
            "outcome-on-treatment regression.",
            n_covariates=0,
        )
    missing = [c for c in covs if c not in data.columns]
    if missing:
        return _failed(
            f"covariate columns missing from DataFrame: {missing}",
            missing=missing,
        )
    return _passed(n_covariates=len(covs))


def _check_formula_columns(data: pd.DataFrame, kwargs: Dict[str, Any]
                            ) -> CheckResult:
    formula = kwargs.get("formula")
    if not formula:
        return _failed(
            "formula not provided.", arg_name="formula",
        )
    # Lightweight Wilkinson parse — pull bare-word identifiers out of
    # the formula and check each is a column. Doesn't need to be
    # perfect; patsy will give the precise error if we miss something.
    import re
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    # Drop common patsy / Wilkinson keywords.
    tokens -= {"C", "I", "Q", "Treatment", "Sum", "Diff", "Helmert",
               "Poly", "np", "log", "exp", "sqrt", "abs"}
    missing = [t for t in tokens if t not in data.columns]
    if missing:
        return _failed(
            f"formula references columns not in DataFrame: {missing}",
            missing=missing,
            available=list(data.columns)[:20],
        )
    return _passed(n_terms=len(tokens))


# ---------------------------------------------------------------------- #
#  IV-specific: first-stage strength gate
# ---------------------------------------------------------------------- #
#
# Stock & Yogo (2005) "Testing for weak instruments in linear IV
# regression" gives critical values for the first-stage F-statistic
# under a given desired bias of 2SLS relative to OLS.  For one
# endogenous variable, the 10%-maximum-size critical values are:
#
#     1 instrument:  16.38
#     2 instruments: 19.93
#     3 instruments: 22.30
#
# We use a soft layered gate: F < 10 is the long-standing rule of
# thumb (Staiger & Stock 1997) for "very weak"; F < 16.38 (1 endog, 1
# IV cutoff) is the Stock-Yogo 10% max-size band.  The preflight emits
# a `warning` with explicit recovery hints to switch to LIML or
# Anderson-Rubin inference, both of which `sp.iv` already supports.
# We also flag near-zero partial-R² as separate evidence of an
# almost-irrelevant instrument.

_STOCK_YOGO_F_CRIT_10PCT = 16.38   # 1 endog, 1 IV
_STAIGER_STOCK_F_RULE = 10.0       # rule of thumb


def _check_iv_first_stage_strength(data: pd.DataFrame,
                                    kwargs: Dict[str, Any]
                                    ) -> CheckResult:
    """First-stage F gate for IV preflight.

    Parses the Wilkinson IV formula, runs the first-stage OLS for each
    endogenous variable, and reports whether the partial F-statistic
    of the instruments clears the Staiger-Stock rule of thumb (F=10)
    and the Stock-Yogo 10% max-size critical value (F=16.38 for the
    one-instrument case).  Falls back to a passed verdict if the
    formula does not match an IV pattern (the universal column check
    will catch malformed formulas separately).
    """
    formula = kwargs.get("formula")
    if not isinstance(formula, str) or "~" not in formula:
        return _passed(skipped="formula missing or non-string")
    # Lazy import: parse_formula triggers a regression-module import
    # that we want to defer until preflight is actually run on an IV.
    try:
        from ..core.utils import parse_formula
    except Exception:
        return _passed(skipped="parse_formula unavailable")
    parsed = parse_formula(formula)
    endog = parsed.get("endogenous", []) or []
    instruments = parsed.get("instruments", []) or []
    exog = parsed.get("exogenous", []) or []
    if not endog or not instruments:
        # Not an IV formula — the formula_columns check + sp.ivreg
        # will diagnose this; preflight stays silent.
        return _passed(skipped="non-IV formula")

    needed = set(endog) | set(instruments) | set(exog)
    missing = [c for c in needed if c not in data.columns]
    if missing:
        return _passed(skipped="formula columns missing", missing=missing)

    import numpy as np
    sub = data[list(needed)].dropna()
    n = len(sub)
    if n < 30:
        return _passed(skipped="n<30; first-stage F unstable")

    results: Dict[str, Dict[str, float]] = {}
    weakest = None
    n_instr = len(instruments)
    for endog_var in endog:
        # First stage: endog_var ~ exog + instruments + intercept
        rhs_cols = list(exog) + list(instruments)
        # Filter out non-numeric columns to avoid a spurious failure
        # when the user has string-typed covariates that should be
        # passed through patsy in the actual fit.
        rhs_cols = [c for c in rhs_cols
                    if pd.api.types.is_numeric_dtype(sub[c])]
        if not rhs_cols:
            continue
        try:
            y = sub[endog_var].astype(float).to_numpy()
            X_full = np.column_stack(
                [np.ones(n)] + [sub[c].astype(float).to_numpy()
                                for c in rhs_cols]
            )
            X_restricted_cols = [c for c in rhs_cols if c not in instruments]
            X_restricted = np.column_stack(
                [np.ones(n)] + [sub[c].astype(float).to_numpy()
                                for c in X_restricted_cols]
            )
            # Use lstsq so a near-singular design degrades gracefully.
            beta_full, *_ = np.linalg.lstsq(X_full, y, rcond=None)
            beta_rest, *_ = np.linalg.lstsq(X_restricted, y, rcond=None)
            rss_full = float(np.sum((y - X_full @ beta_full) ** 2))
            rss_rest = float(np.sum((y - X_restricted @ beta_rest) ** 2))
            tss = float(np.sum((y - y.mean()) ** 2))
            df_num = n_instr
            df_denom = n - X_full.shape[1]
            if rss_full > 0 and df_denom > 0:
                f_stat = ((rss_rest - rss_full) / df_num) / (rss_full / df_denom)
            else:
                f_stat = float("nan")
            partial_r2 = (
                1.0 - rss_full / rss_rest if rss_rest > 0 else float("nan")
            )
            r2 = 1.0 - rss_full / tss if tss > 0 else float("nan")
        except (np.linalg.LinAlgError, ValueError):
            continue
        results[endog_var] = {
            "f_statistic": float(f_stat),
            "partial_r_squared": float(partial_r2),
            "r_squared": float(r2),
        }
        if weakest is None or f_stat < weakest[1]:
            weakest = (endog_var, float(f_stat))

    if not results:
        return _passed(skipped="first-stage F could not be computed")

    if weakest is None or not np.isfinite(weakest[1]):
        return _passed(first_stage=results)

    var, f = weakest
    evidence = {
        "first_stage": results,
        "weakest_endog": var,
        "weakest_F": f,
        "n_instruments": n_instr,
        "stock_yogo_F_10pct": _STOCK_YOGO_F_CRIT_10PCT,
        "staiger_stock_rule_F": _STAIGER_STOCK_F_RULE,
        "recovery_hints": [
            "Switch to method='liml' for Limited Information Maximum "
            "Likelihood (better small-sample behaviour under weak IV).",
            "Switch to inference='ar' (Anderson-Rubin) for tests "
            "robust to weak instruments.",
            "Use sp.anderson_rubin_ci(...) for an Anderson-Rubin "
            "confidence interval.",
        ],
    }
    if f < _STAIGER_STOCK_F_RULE:
        return _warning(
            f"Very weak first stage for '{var}': F = {f:.2f} < 10 "
            "(Staiger-Stock 1997 rule of thumb). 2SLS is biased "
            "toward OLS and HC1 SEs ignore the bias; switch to "
            "method='liml' or use sp.anderson_rubin_ci(...).",
            **evidence,
        )
    if f < _STOCK_YOGO_F_CRIT_10PCT:
        return _warning(
            f"Weak first stage for '{var}': F = {f:.2f} < 16.38 "
            "(Stock-Yogo 2005, 10% max size for 1 endog / 1 IV). "
            "Consider method='liml' or Anderson-Rubin inference.",
            **evidence,
        )
    return _passed(
        first_stage=results,
        weakest_endog=var,
        weakest_F=f,
        message=f"first-stage F = {f:.2f} clears Stock-Yogo 10% max size",
    )


# ====================================================================== #
#  Per-method check tables
# ====================================================================== #


_UNIVERSAL: Tuple[_Check, ...] = (
    _Check("data_is_dataframe",
           "Is `data` a pandas DataFrame?",
           _check_dataframe),
    _Check("data_non_empty",
           "Does the DataFrame have at least one row?",
           _check_non_empty),
)


_REGRESS_CHECKS: Tuple[_Check, ...] = _UNIVERSAL + (
    _Check("formula_columns_exist",
           "Are all formula identifiers present as columns?",
           _check_formula_columns),
    _Check("min_n_for_regression",
           "Is n above the typical minimum for OLS (30)?",
           _make_check_min_n(30)),
)


_DID_CHECKS: Tuple[_Check, ...] = _UNIVERSAL + (
    _Check("y_column_exists",
           "Outcome column exists?",
           lambda d, k: _column_exists(d, k, "y")),
    _Check("treat_column_exists",
           "Treatment column exists?",
           lambda d, k: _column_exists(d, k, "treat")),
    _Check("time_column_exists",
           "Time column exists?",
           lambda d, k: _column_exists(d, k, "time")),
    _Check("treat_is_binary",
           "Is the treatment column binary 0/1?",
           _check_treat_binary),
    _Check("time_has_two_periods",
           "Does the time column have at least 2 distinct periods?",
           _check_time_has_two_periods),
    _Check("min_n_for_did",
           "Is n above the typical minimum for DID (50)?",
           _make_check_min_n(50)),
)


_DID_STAGGERED_CHECKS: Tuple[_Check, ...] = _DID_CHECKS + (
    _Check("id_column_provided",
           "Is the unit identifier provided for staggered DID?",
           _check_id_column_for_staggered),
)


_RD_CHECKS: Tuple[_Check, ...] = _UNIVERSAL + (
    _Check("y_column_exists",
           "Outcome column exists?",
           lambda d, k: _column_exists(d, k, "y")),
    _Check("running_var_continuous",
           "Is the running variable numeric and reasonably continuous?",
           _check_running_var_continuous),
    _Check("min_n_for_rd",
           "Is n above the typical minimum for RD (500)?",
           _make_check_min_n(500)),
)


_IV_CHECKS: Tuple[_Check, ...] = _UNIVERSAL + (
    _Check("formula_columns_exist",
           "Are all formula identifiers present as columns?",
           _check_formula_columns),
    _Check("min_n_for_iv",
           "Is n above the typical minimum for IV (50)?",
           _make_check_min_n(50)),
    _Check("first_stage_strength",
           "Is the first-stage F-statistic above the Stock-Yogo "
           "10% max-size critical value (16.38 for 1 endog / 1 IV)?",
           _check_iv_first_stage_strength),
)


_MATCHING_CHECKS: Tuple[_Check, ...] = _UNIVERSAL + (
    _Check("y_column_exists",
           "Outcome column exists?",
           lambda d, k: _column_exists(d, k, "y")),
    _Check("treat_column_exists",
           "Treatment column exists?",
           lambda d, k: _column_exists(d, k, "treat")),
    _Check("treat_is_binary",
           "Is the treatment column binary 0/1?",
           _check_treat_binary),
    _Check("covariates_exist",
           "Are all covariate columns present in the DataFrame?",
           _check_covariates_exist),
    _Check("min_n_for_matching",
           "Is n above the typical minimum for matching (50)?",
           _make_check_min_n(50)),
)


# Method aliases mapped to their check tables.
_METHOD_TABLES: Dict[str, Tuple[_Check, ...]] = {
    "regress": _REGRESS_CHECKS,
    "ols": _REGRESS_CHECKS,
    "did": _DID_CHECKS,
    "did_2x2": _DID_CHECKS,
    "callaway_santanna": _DID_STAGGERED_CHECKS,
    "sun_abraham": _DID_STAGGERED_CHECKS,
    "did_imputation": _DID_STAGGERED_CHECKS,
    "did_analysis": _DID_STAGGERED_CHECKS,
    "rdrobust": _RD_CHECKS,
    "rd": _RD_CHECKS,
    "ivreg": _IV_CHECKS,
    "iv": _IV_CHECKS,
    "ebalance": _MATCHING_CHECKS,
    "match": _MATCHING_CHECKS,
    "psmatch": _MATCHING_CHECKS,
    "sbw": _MATCHING_CHECKS,
}


# ====================================================================== #
#  Public API
# ====================================================================== #


def preflight(data: pd.DataFrame, method: str,
              **kwargs: Any) -> Dict[str, Any]:
    """Method-specific pre-estimation diagnostics.

    Runs cheap, method-aware checks (column existence, data shape,
    treatment binarity, sample size) BEFORE the agent commits to an
    expensive estimator call. Use the verdict to decide whether to
    proceed, fix arguments, or pivot to a different method.

    Parameters
    ----------
    data : pandas.DataFrame
        Same DataFrame the agent plans to pass to ``sp.{method}(...)``.
    method : str
        Name of the StatsPAI estimator to pre-flight (e.g. ``"did"``,
        ``"rdrobust"``, ``"ivreg"``, ``"callaway_santanna"``,
        ``"ebalance"``). Unknown methods get only the universal
        DataFrame-shape sanity checks.
    **kwargs
        Estimator arguments — column names (``y``, ``treat``, ``time``,
        ``i``, ``running_var``), a Wilkinson ``formula``, a
        ``covariates`` list, etc. Passed through unchanged from what
        the agent intends to use; ``preflight`` doesn't run the
        estimator.

    Returns
    -------
    dict
        JSON-safe payload with keys:

        - ``method`` (str) — input method name (lower-cased)
        - ``verdict`` (str) — ``"PASS"`` / ``"WARN"`` / ``"FAIL"``
        - ``checks`` (list[dict]) — every check that ran, with
          ``name`` / ``question`` / ``status`` / ``message`` /
          ``evidence``
        - ``summary`` (dict) — count of ``passed`` / ``warning`` /
          ``failed``
        - ``n_obs`` (int) — sample size
        - ``known_method`` (bool) — whether the method has a
          dedicated check table (``False`` falls back to the universal
          checks only)

    Examples
    --------
    >>> df = pd.DataFrame({'y': [1, 2, 3, 4],
    ...                    'treated': [0, 1, 0, 1],
    ...                    't': [0, 0, 1, 1]})
    >>> sp.preflight(df, 'did', y='y', treat='treated', time='t')['verdict']
    'WARN'  # n=4 is below the typical-minimum threshold of 50

    See Also
    --------
    sp.check_identification :
        Design-level diagnostics for an already-declared design.
    sp.assumption_audit :
        Heavyweight: re-runs statistical tests after fitting.
    sp.audit :
        Read-only checklist of robustness evidence ON a fitted result.
    """
    method_key = (method or "").lower().strip()
    table = _METHOD_TABLES.get(method_key)
    known = table is not None
    if table is None:
        table = _UNIVERSAL

    # n_obs reported even when the data is not a DataFrame.
    n_obs = len(data) if isinstance(data, pd.DataFrame) else 0

    results: List[Dict[str, Any]] = []
    n_passed = n_warning = n_failed = 0

    for chk in table:
        try:
            status, message, evidence = chk.fn(data, kwargs)
        except Exception as e:
            # A check raising means the data is in a state none of the
            # checks anticipated — fail closed and surface the
            # exception for debugging.
            status = "failed"
            message = (f"check {chk.name!r} raised "
                        f"{type(e).__name__}: {e}")
            evidence = {"exception": type(e).__name__}

        if status == "passed":
            n_passed += 1
        elif status == "warning":
            n_warning += 1
        else:
            n_failed += 1

        results.append({
            "name": chk.name,
            "question": chk.question,
            "status": status,
            "message": message,
            "evidence": evidence,
        })

    if n_failed > 0:
        verdict = "FAIL"
    elif n_warning > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "method": method_key,
        "verdict": verdict,
        "checks": results,
        "summary": {
            "passed": n_passed,
            "warning": n_warning,
            "failed": n_failed,
            "n_total": len(results),
        },
        "n_obs": n_obs,
        "known_method": known,
    }


__all__ = ["preflight"]
