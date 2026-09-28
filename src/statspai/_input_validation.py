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

from typing import Sequence

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
    is under-determined and would otherwise emit a silent NaN estimate.
    """
    require_columns(data, columns, function=function)
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
