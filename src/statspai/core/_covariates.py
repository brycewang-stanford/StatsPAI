"""Categorical covariates for the estimators that take a covariate list.

``covariates=['age', 'region']`` is read by most estimators as "these
columns, as numbers". That is wrong in two ways for a categorical column.
Text levels fail in numpy with ``could not convert string to float``, and a
``category`` column whose levels look like numbers is cast to those numbers
and enters the model as one linear term, although the dtype says it is not a
quantity.

:func:`expand_categorical_covariates` turns each categorical entry into
indicator columns (first level omitted) and leaves every numeric entry
alone, so a call with numeric covariates runs on the frame it was given.
:func:`expands_categorical_covariates` is the decorator form.
"""

from __future__ import annotations

import functools
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, cast

import pandas as pd

from ..exceptions import MethodIncompatibility

_F = TypeVar("_F", bound=Callable[..., Any])

_FACTOR_CALL = re.compile(r"^\s*(?:C\(\s*([^(),]+?)\s*\)|i\.(\w+))\s*$")


def _factor_column(entry: Any) -> Optional[str]:
    """Column named by ``C(col)`` or ``i.col``; ``None`` for a plain name."""
    if not isinstance(entry, str):
        return None
    m = _FACTOR_CALL.match(entry)
    if m is None:
        return None
    return m.group(1) or m.group(2)


def _is_categorical(col: pd.Series) -> bool:
    dtype = col.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if isinstance(dtype, pd.StringDtype):
        return True
    if pd.api.types.is_object_dtype(dtype):
        # An object column of numbers (ints next to None, Decimal) is a
        # number and is left to the estimator's own cast.
        return bool(pd.api.types.infer_dtype(col, skipna=True) == "string")
    return False


def _levels(col: pd.Series) -> List[Any]:
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(col.dropna().unique())
        return [lv for lv in col.cat.categories if lv in present]
    values = col.dropna().unique()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=repr)


def expand_categorical_covariates(
    data: pd.DataFrame,
    covariates: Sequence[Any],
    *,
    function: str,
) -> Tuple[pd.DataFrame, List[Any], Optional[Dict[str, Any]]]:
    """Replace categorical covariates by indicator columns.

    An entry is categorical when it is written ``C(col)`` or ``i.col``, or
    when the column has ``object``, string or ``category`` dtype. Its levels
    are sorted (a ``category`` column keeps its own order), the first is the
    omitted base, and each other level becomes a 0/1 column named
    ``col[T.level]``. A row with a missing value is missing in every
    indicator, so the estimator's own complete-case rule drops it. A
    ``bool`` column becomes 0/1.

    Returns
    -------
    data : pd.DataFrame
        ``data`` itself when nothing was expanded, else a copy with the
        indicator columns added.
    covariates : list
        The covariate list with each categorical entry replaced by its
        indicators, in place.
    info : dict or None
        ``{'levels': {col: [...]}, 'columns': {col: [...]}}`` for the
        expanded columns; ``None`` when nothing was expanded.
    """
    covariates = list(covariates)
    plan: List[Tuple[int, str, Any]] = []
    bools: List[str] = []
    for pos, entry in enumerate(covariates):
        named = _factor_column(entry)
        col = named if named is not None else entry
        if not isinstance(col, str) or col not in data.columns:
            if named is not None:
                raise MethodIncompatibility(
                    f"{function}: covariate {entry!r} names column {named!r}, "
                    "which is not in the data.",
                    diagnostics={"missing": [named]},
                )
            continue
        series = data[col]
        if named is not None or _is_categorical(series):
            plan.append((pos, col, entry))
        elif pd.api.types.is_bool_dtype(series.dtype):
            bools.append(col)
    if not plan and not bools:
        return data, covariates, None

    out = data.copy()
    for col in bools:
        out[col] = out[col].astype(float)

    levels_by_col: Dict[str, List[Any]] = {}
    columns_by_col: Dict[str, List[str]] = {}
    replacement: Dict[int, List[str]] = {}
    n = len(out)
    for pos, col, _entry in plan:
        series = data[col]
        levels = _levels(series)
        if len(levels) > max(n // 2, 1):
            raise MethodIncompatibility(
                f"{function}: covariate {col!r} has {len(levels)} distinct "
                f"levels in {n} rows, which looks like an identifier rather "
                "than a category.",
                recovery_hint=(
                    "Drop it from covariates, or pass it where the estimator "
                    "takes a unit, cluster or fixed-effect column."
                ),
                diagnostics={"column": col, "n_levels": len(levels)},
            )
        names: List[str] = []
        missing = series.isna()
        for level in levels[1:]:
            name = f"{col}[T.{level}]"
            dummy = (series == level).astype(float)
            out[name] = dummy.where(~missing)
            names.append(name)
        levels_by_col[col] = levels
        columns_by_col[col] = names
        replacement[pos] = names

    expanded: List[Any] = []
    for pos, entry in enumerate(covariates):
        expanded.extend(replacement.get(pos, [entry]))
    if not plan:
        return out, expanded, None
    info = {"levels": levels_by_col, "columns": columns_by_col}
    return out, expanded, info


def apply_covariate_expansion(
    data: pd.DataFrame, info: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """Add to ``data`` the indicator columns an earlier expansion created.

    For prediction on new rows: the levels are the ones seen at fit time, so
    the columns line up with the fitted model whatever levels the new rows
    happen to contain. A level not seen at fit time is an error.
    """
    if not info:
        return data
    out = data.copy()
    for col, levels in info["levels"].items():
        if col not in out.columns:
            continue
        series = out[col]
        unseen = set(series.dropna().unique()) - set(levels)
        if unseen:
            raise MethodIncompatibility(
                f"Covariate {col!r} has levels that were not in the fitted "
                f"data: {sorted(unseen, key=repr)[:5]}.",
                diagnostics={"column": col},
            )
        missing = series.isna()
        for level, name in zip(levels[1:], info["columns"][col]):
            out[name] = (series == level).astype(float).where(~missing)
    return out


def expands_categorical_covariates(
    *names: str,
) -> Callable[[_F], _F]:
    """Decorator: expand categorical entries of a covariate-list parameter.

    ``names`` are the parameter names to look at (default ``'covariates'``).
    The expansion is recorded in ``result.model_info['covariate_expansion']``
    when the result has a ``model_info`` dict. A call whose covariates are
    all numeric reaches the function unchanged.
    """
    params = names or ("covariates",)

    def decorate(fn: _F) -> _F:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError:
                return fn(*args, **kwargs)
            a = bound.arguments
            data = a.get("data")
            if not isinstance(data, pd.DataFrame):
                return fn(*args, **kwargs)
            merged: Dict[str, Any] = {"levels": {}, "columns": {}}
            changed = False
            for name in params:
                value = a.get(name)
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                data, new_value, info = expand_categorical_covariates(
                    data, value, function=fn.__name__
                )
                if new_value != list(value):
                    a[name] = new_value
                    changed = True
                if info is not None:
                    merged["levels"].update(info["levels"])
                    merged["columns"].update(info["columns"])
            if data is a.get("data") and not changed:
                return fn(*args, **kwargs)
            a["data"] = data
            result = fn(*bound.args, **bound.kwargs)
            model_info = getattr(result, "model_info", None)
            if merged["levels"] and isinstance(model_info, dict):
                model_info["covariate_expansion"] = merged
            return result

        return cast(_F, wrapper)

    return decorate
