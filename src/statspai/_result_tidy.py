"""Generic broom-style ``tidy()`` / ``glance()`` for result objects.

``CausalResult`` and ``EconometricResults`` have their own. The ~360 domain
result classes that inherit :class:`statspai._result_serialize.ResultProtocolMixin`
get these defaults, so ``r.tidy()`` / ``r.glance()`` work on every result
(the 2026-10-19 audit found 346 result classes without them).

``tidy`` reads, in order: a coefficient vector (``params`` with standard
errors), a headline scalar (``estimate`` / ``att`` / ``ate`` ... with its
``se``), or a test (``statistic`` with ``pvalue``). A result with none of
these has no estimate table and ``tidy`` refuses with a typed error rather
than inventing one; ``glance`` (the scalar fields, one row) always works.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

TIDY_COLUMNS = [
    "term",
    "estimate",
    "std_error",
    "statistic",
    "p_value",
    "conf_low",
    "conf_high",
]

_ESTIMATE_NAMES = ("estimate", "att", "ate", "effect", "coef", "beta", "late")
_SE_NAMES = ("se", "std_error", "std_err", "stderr")
_P_NAMES = ("pvalue", "p_value", "pval")
_STAT_NAMES = ("statistic", "stat", "test_statistic", "t_stat", "z_stat")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.ravel()[0])
    return None


def _first_number(obj: Any, names: tuple) -> Optional[float]:
    for name in names:
        val = _number(getattr(obj, name, None))
        if val is not None:
            return val
    return None


def _as_series(values: Any, index: pd.Index) -> Optional[pd.Series]:
    if values is None:
        return None
    if isinstance(values, pd.Series):
        return values.reindex(index)
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size != len(index):
        return None
    return pd.Series(arr, index=index)


def _crit(conf_level: float) -> float:
    from scipy import stats

    return float(stats.norm.ppf(1 - (1 - conf_level) / 2))


def _vector_rows(obj: Any, conf_level: Optional[float]) -> Optional[pd.DataFrame]:
    params = getattr(obj, "params", None)
    if not isinstance(params, pd.Series) or params.empty:
        return None
    se = None
    for name in ("std_errors", "bse", "se", "std_err"):
        se = _as_series(getattr(obj, name, None), params.index)
        if se is not None:
            break
    if se is None:
        return None
    est = params.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = _as_series(getattr(obj, "tvalues", None), params.index)
        if stat is None:
            stat = est / se
    pv = _as_series(getattr(obj, "pvalues", None), params.index)
    if conf_level is not None:
        z = _crit(conf_level)
        lo, hi = est - z * se, est + z * se
    else:
        lo = _as_series(getattr(obj, "conf_int_lower", None), params.index)
        hi = _as_series(getattr(obj, "conf_int_upper", None), params.index)
    nan = pd.Series(np.nan, index=params.index)
    return pd.DataFrame(
        {
            "term": [str(t) for t in params.index],
            "estimate": est.to_numpy(),
            "std_error": se.to_numpy(dtype=float),
            "statistic": stat.to_numpy(dtype=float),
            "p_value": (pv if pv is not None else nan).to_numpy(dtype=float),
            "conf_low": (lo if lo is not None else nan).to_numpy(dtype=float),
            "conf_high": (hi if hi is not None else nan).to_numpy(dtype=float),
        }
    )


def _term_name(obj: Any) -> str:
    for name in ("estimand", "term", "method", "test", "name"):
        val = getattr(obj, name, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return type(obj).__name__


def _scalar_ci(obj: Any) -> tuple:
    for name in ("ci", "conf_int", "confint"):
        val = getattr(obj, name, None)
        if isinstance(val, (tuple, list, np.ndarray)) and len(val) == 2:
            lo, hi = _number(val[0]), _number(val[1])
            if lo is not None and hi is not None:
                return lo, hi
    lo = _first_number(obj, ("ci_lower", "ci_low", "conf_low", "lower"))
    hi = _first_number(obj, ("ci_upper", "ci_high", "conf_high", "upper"))
    return (
        lo if lo is not None else np.nan,
        hi if hi is not None else np.nan,
    )


def _scalar_row(obj: Any, conf_level: Optional[float]) -> Optional[pd.DataFrame]:
    est = _first_number(obj, _ESTIMATE_NAMES)
    stat = _first_number(obj, _STAT_NAMES)
    pv = _first_number(obj, _P_NAMES)
    if est is None and (stat is None or pv is None):
        return None
    se = _first_number(obj, _SE_NAMES)
    if est is not None and stat is None and se is not None and se > 0:
        stat = est / se
    if conf_level is not None and est is not None and se is not None:
        z = _crit(conf_level)
        lo, hi = est - z * se, est + z * se
    elif est is not None:
        lo, hi = _scalar_ci(obj)
    else:
        lo, hi = np.nan, np.nan
    row = {
        "term": _term_name(obj),
        "estimate": est if est is not None else np.nan,
        "std_error": se if se is not None else np.nan,
        "statistic": stat if stat is not None else np.nan,
        "p_value": pv if pv is not None else np.nan,
        "conf_low": lo,
        "conf_high": hi,
    }
    return pd.DataFrame([row], columns=TIDY_COLUMNS)


def tidy(obj: Any, conf_level: Optional[float] = None) -> pd.DataFrame:
    """Estimate table with the broom columns :data:`TIDY_COLUMNS`."""
    if conf_level is not None and not 0.0 < float(conf_level) < 1.0:
        from .exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"conf_level must be in (0, 1); got {conf_level!r}.",
            recovery_hint="Pass e.g. conf_level=0.95.",
        )
    for build in (_vector_rows, _scalar_row):
        frame = build(obj, conf_level)
        if frame is not None:
            return frame
    from .exceptions import MethodIncompatibility

    name = type(obj).__name__
    raise MethodIncompatibility(
        f"{name} has no estimate table: no coefficient vector, headline "
        "estimate or test statistic to put in tidy().",
        recovery_hint=(
            "Use result.glance() for its scalar fields, result.to_dict() for "
            "everything, or result.summary()."
        ),
        diagnostics={"result_type": name},
    )


def glance(obj: Any) -> pd.DataFrame:
    """One row of the result's scalar fields (numbers, flags, short text)."""
    from ._result_serialize import result_to_dict

    row: Dict[str, Any] = {}
    for key, val in result_to_dict(obj).items():
        if str(key).startswith("_") or val is None or isinstance(val, (list, dict)):
            continue
        if isinstance(val, str) and (
            "\n" in val or len(val) > 120 or val.startswith("<")
        ):
            continue  # rendered summaries and object reprs are not fields
        row[str(key)] = val
    columns: List[str] = list(row)
    return pd.DataFrame([row], columns=columns)
