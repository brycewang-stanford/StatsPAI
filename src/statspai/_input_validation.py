"""Package-level fail-loud input-validation primitives.

A recurring reliability defect across StatsPAI estimators is the pattern::

    df = data[[col_a, col_b, *covs]].dropna()
    ... linear algebra on df ...

which (a) surfaces a typo'd column name as a cryptic
``KeyError: "['x'] not in index"`` rather than a named, actionable error,
and (b) silently returns a NaN/garbage estimate when the frame is empty,
all-NaN, or has fewer complete rows than parameters — the "吞异常返回 NaN"
anti-pattern design principle 7 forbids.

These two helpers centralise the guard (CLAUDE.md §4: shared primitive, not
re-implemented per estimator).  Every failure raises
:class:`~statspai.exceptions.StatsPAIError` (the agent-native umbrella,
carrying a ``recovery_hint`` and ``diagnostics``); the too-few-rows case
uses :class:`~statspai.exceptions.DataInsufficient`, which is *also* a
:class:`ValueError`.  So ``except sp.StatsPAIError`` catches every case and
``except ValueError`` still catches the data-size failures.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from .exceptions import ColumnNotFound, DataInsufficient, StatsPAIError


def require_columns(
    data: pd.DataFrame, columns: Sequence[str], *, function: str
) -> None:
    """Validate that ``data`` is a frame carrying every required column.

    A wrong type raises :class:`~statspai.exceptions.StatsPAIError`; a missing
    column raises :class:`~statspai.exceptions.ColumnNotFound` (a
    :class:`~statspai.exceptions.DataInsufficient`, hence a ``ValueError``
    *and* a ``StatsPAIError``, with ``diagnostics["did_you_mean"]``) whose
    message contains the
    conventional ``"Missing columns"`` phrase plus the offending names — far
    more actionable than the bare ``KeyError: "['typo'] not in index"`` pandas
    would otherwise raise, while staying back-compatible with callers that
    ``except ValueError`` on the long-standing missing-column contract.
    """
    if not isinstance(data, pd.DataFrame):
        raise StatsPAIError(
            f"{function}: `data` must be a pandas DataFrame, got "
            f"{type(data).__name__}.",
            recovery_hint="Pass a pandas DataFrame whose columns include the "
            "variables named in the call.",
            diagnostics={"received_type": type(data).__name__},
        )
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            list(data.columns),
            message=f"{function}: Missing columns {missing} — not found in data.",
        )


def missing_named_columns(
    columns: Union[Mapping[str, Any], Sequence[Any]], available: Iterable[Any]
) -> Dict[str, Any]:
    """``{label: column}`` for every requested column absent from ``available``.

    ``columns`` is either ``{argument_label: column_or_columns}`` or a plain
    sequence of column names (each labelled by itself).  ``None`` values are
    skipped; a list / tuple value under ``label`` is expanded element-wise to
    ``label[i]`` so the label says which entry of the user's list is wrong.
    Unhashable values (an array passed where a name is also accepted) are
    not column names and are skipped.
    """
    avail = set(available)

    def _absent(col: Any) -> bool:
        # arrays / Series passed in place of a name are not column names
        try:
            return col is not None and col not in avail
        except TypeError:
            return False

    if isinstance(columns, Mapping):
        items = list(columns.items())
    else:
        items = [(str(c), c) for c in columns]
    missing: Dict[str, Any] = {}
    for label, value in items:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for i, col in enumerate(value):
                if _absent(col):
                    missing[f"{label}[{i}]"] = col
        elif _absent(value):
            missing[str(label)] = value
    return missing


def check_columns(
    data: pd.DataFrame,
    columns: Union[Mapping[str, Any], Sequence[Any]],
    *,
    function: Optional[str] = None,
    message: Optional[str] = None,
    recovery_hint: Optional[str] = None,
    extra_hint: Optional[str] = None,
    extra_diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    """Raise :class:`~statspai.exceptions.ColumnNotFound` for absent columns.

    The column-name guard every estimator needs, with ``did_you_mean``
    suggestions.  ``columns`` follows :func:`missing_named_columns`:
    ``{"y": y, "treat": treat, "covariates": covariates}`` labels each
    missing column with the argument the caller typed.

    Examples
    --------
    >>> import pandas as pd
    >>> from statspai._input_validation import check_columns
    >>> from statspai.exceptions import ColumnNotFound
    >>> df = pd.DataFrame({"wage": [1.0], "x1": [0.0]})
    >>> try:
    ...     check_columns(df, {"y": "wage", "covariates": ["x11"]})
    ... except ColumnNotFound as e:
    ...     print(e.diagnostics["did_you_mean"])
    {'covariates[0]': 'x1'}
    """
    missing = missing_named_columns(columns, data.columns)
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            list(data.columns),
            message=message,
            recovery_hint=recovery_hint,
            context=function,
            extra_hint=extra_hint,
            extra_diagnostics=extra_diagnostics,
        )


#: The one message every estimator gives for ``inf`` in the data, so the
#: same input is refused the same way whichever estimator reads it.
INFINITE_VALUES_HINT = (
    "Infinite values are not missing values, so they are neither dropped "
    "nor passed to the estimator. Fix the step that produced them (a log "
    "of zero, a division by zero), or recode them as missing with "
    "data.replace([np.inf, -np.inf], np.nan) to have those rows dropped "
    "like any other missing value."
)


def _infinite_count(values: Any) -> int:
    """Number of ``+inf`` / ``-inf`` entries in a column (NaN not counted)."""
    import numpy as np

    if isinstance(values, pd.Series):
        dtype = values.dtype
        if isinstance(dtype, np.dtype):
            if dtype.kind not in "fc":
                if dtype.kind != "O":
                    return 0  # integers, bools, datetimes hold no inf
                return int(
                    sum(
                        1
                        for v in values.to_numpy()
                        if isinstance(v, float) and np.isinf(v)
                    )
                )
            arr = values.to_numpy()
        else:
            if not pd.api.types.is_float_dtype(dtype):
                return 0
            arr = values.to_numpy(dtype=float, na_value=np.nan)
    else:
        arr = np.asarray(values)
        if arr.dtype.kind not in "fc":
            return 0
    if arr.size == 0 or np.isfinite(arr).all():
        return 0
    return int(np.isinf(arr).sum())


def infinite_values_by_column(
    data: pd.DataFrame, columns: Iterable[Any]
) -> Dict[str, int]:
    """``{column: number of ±inf values}`` for the named columns that hold any.

    Names that are ``None`` or not columns of ``data`` are skipped (the
    caller's column check reports those). Missing values are not counted:
    they are the caller's listwise-deletion business, ``inf`` is not.
    """
    out: Dict[str, int] = {}
    if not isinstance(data, pd.DataFrame):
        return out
    seen = set()
    for col in columns:
        try:
            if col is None or col in seen or col not in data.columns:
                continue
        except TypeError:  # an unhashable array passed in place of a name
            continue
        seen.add(col)
        series = data[col]
        if isinstance(series, pd.DataFrame):  # duplicated column name
            continue
        n_inf = _infinite_count(series)
        if n_inf:
            out[str(col)] = n_inf
    return out


def raise_infinite(
    counts: Mapping[str, int], *, function: str, n_rows: Optional[int] = None
) -> None:
    """Raise the shared refusal for ``counts`` (``{column: n_inf}``) if any."""
    if not counts:
        return
    from .exceptions import DataInsufficient

    total = int(sum(counts.values()))
    cols = ", ".join(f"{c!r} ({n})" for c, n in counts.items())
    diagnostics: Dict[str, Any] = {
        "infinite_by_column": dict(counts),
        "n_infinite": total,
    }
    if n_rows is not None:
        diagnostics["n_rows_infinite"] = int(n_rows)
    raise DataInsufficient(
        f"{function}: {total} infinite value(s) in the data -- by column: " f"{cols}.",
        recovery_hint=INFINITE_VALUES_HINT,
        diagnostics=diagnostics,
    )


def require_finite(
    data: pd.DataFrame, columns: Iterable[Any], *, function: str
) -> None:
    """Refuse ``+inf`` / ``-inf`` in the columns an estimator uses.

    The package-wide policy for non-finite values: a missing value (NaN) is
    handled by the estimator's own missing-data rule (usually listwise
    deletion with an ``AssumptionWarning``), while an infinite value raises
    :class:`~statspai.exceptions.DataInsufficient` naming each column and its
    count. ``inf`` almost always comes from a transformation that failed
    (``log(0)``, ``x / 0``), so dropping it silently would hide the error and
    propagating it returns a meaningless number.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from statspai._input_validation import require_finite
    >>> from statspai.exceptions import DataInsufficient
    >>> df = pd.DataFrame({"y": [1.0, np.inf, 2.0], "x": [0.0, 1.0, np.nan]})
    >>> try:
    ...     require_finite(df, ["y", "x"], function="demo")
    ... except DataInsufficient as e:
    ...     print(e.diagnostics["infinite_by_column"])
    {'y': 1}
    """
    counts = infinite_values_by_column(data, columns)
    if not counts:
        return
    import numpy as np

    hit = np.zeros(len(data), dtype=bool)
    for col in counts:
        values = pd.to_numeric(data[col], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        hit |= np.isinf(values)
    raise_infinite(counts, function=function, n_rows=int(hit.sum()))


def clean_frame(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    function: str,
    n_params: int,
) -> pd.DataFrame:
    """Validate columns, drop incomplete rows, enforce an identification floor.

    Returns the complete-case sub-frame (index reset).  Raises
    :class:`~statspai.exceptions.DataInsufficient` when ``n_params`` or fewer
    complete rows remain — the point below which the downstream linear algebra
    is under-determined and would otherwise emit a silent NaN estimate — and
    when a used column holds ``inf`` (:func:`require_finite`; infinite values
    are refused, not dropped as missing).
    """
    require_columns(data, columns, function=function)
    require_finite(data, columns, function=function)
    df = data[list(columns)].dropna().reset_index(drop=True)
    if len(df) <= n_params:
        raise DataInsufficient(
            f"{function}: only {len(df)} complete row(s) after dropping "
            f"missing values, but the model estimates {n_params} parameter(s) "
            "— need strictly more complete observations than parameters.",
            recovery_hint="Provide more non-missing observations or drop "
            "covariates.",
            diagnostics={"n_complete": int(len(df)), "n_params": int(n_params)},
        )
    return df
