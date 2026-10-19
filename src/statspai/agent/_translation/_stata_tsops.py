"""Stata time-series operators: ``L.x``, ``L2.x``, ``F.x``, ``D.x``, ``L(1/4).x``,
``L(1/3).D.x``.

Stata defines a lag by the time variable of ``tsset`` / ``xtset``, not by
the row above: ``L.x`` at time ``t`` is ``x`` at ``t - 1`` in the same panel,
and is missing when that period is absent. ``shift(1)`` on the rows gives a
different answer wherever a period is missing or the data are a panel, so
the operators are resolved here against the declared time variable.

A command line is rewritten so that every operator term becomes an ordinary
column (``L2.x`` -> ``x_L2``, ``D.x`` -> ``x_D1``, ``L(1/2).x`` -> ``x_L1
x_L2``); the columns are computed and added to the session's private copy
of the data. ``test`` / ``lincom`` lines are rewritten the same way so that
their restrictions name the same coefficients.

The operators themselves live in :mod:`statspai.utils.tsops` (the module
behind ``sp.xtset`` / ``sp.lag`` / ``sp.lead`` / ``sp.diff`` and ``L.x`` in
formulas); this module adapts them to the session, whose refusals are
:class:`StataExprError`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import StatsPAIError
from ...utils import tsops as _public
from ...utils.tsops import TS_TERM as _TERM  # noqa: F401  (used by callers)
from ...utils.tsops import apply_ts_ops as _apply  # noqa: F401
from ...utils.tsops import parse_ts_ops as _parse_ops  # noqa: F401
from ...utils.tsops import ts_column_name as _name  # noqa: F401
from ._stata_expr import StataExprError

__all__ = [
    "check_delta",
    "derived_column",
    "has_ts_operator",
    "rewrite_ts_operators",
]


def has_ts_operator(line: str) -> bool:
    return _TERM.search(line) is not None


class _Clock(_public.TimeClock):
    """:class:`statspai.utils.tsops.TimeClock`, refusing as the session does."""

    def __init__(
        self,
        data: pd.DataFrame,
        unit: Optional[str],
        time: str,
        delta: float = 1.0,
    ) -> None:
        try:
            super().__init__(data, unit, time, delta)
        except StatsPAIError as exc:
            raise StataExprError(exc.message) from None


def derived_column(
    data: pd.DataFrame,
    panel: Tuple[Optional[str], Optional[str]],
    var: str,
    ops: List[Tuple[str, int]],
    delta: float = 1.0,
) -> Optional[np.ndarray]:
    """``var`` under the operators ``ops`` (``L.x``), from ``data`` as it is
    now; ``None`` when ``var`` or the time declaration is gone."""
    unit, time = panel
    if time is None or var not in data.columns:
        return None
    values = data[var].to_numpy(dtype=float, na_value=np.nan)
    return _apply(_Clock(data, unit, time, delta), values, ops)


def check_delta(
    data: pd.DataFrame, unit: Optional[str], time: str, delta: float
) -> None:
    """``tsset t, delta(#)``: Stata refuses data with two periods closer
    than one step (r(451))."""
    try:
        _public.check_delta(data, unit, time, delta)
    except StatsPAIError as exc:
        raise StataExprError(exc.message) from None


def rewrite_ts_operators(
    line: str,
    data: Optional[pd.DataFrame],
    panel: Tuple[Optional[str], Optional[str]],
    made: Optional[Dict[str, tuple]] = None,
    delta: float = 1.0,
) -> Tuple[str, Dict[str, np.ndarray]]:
    """Replace operator terms by column names.

    Returns the rewritten line and the columns to add (name -> values), in
    the order they first appear. ``data`` may be ``None`` for a line that
    only names coefficients (``test``); no column is computed then.

    ``made`` records the columns this function computed (name -> (variable,
    operators)). A column listed there is computed again each time it is
    named: the variable, the sample or the time declaration may have
    changed since, and Stata reads ``L.x`` from the data as they are.
    """
    try:
        return _public.rewrite_ts_terms(line, data, panel, made, delta)
    except StatsPAIError as exc:
        raise StataExprError(exc.message) from None
