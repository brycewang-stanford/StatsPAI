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
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._stata_expr import StataExprError

__all__ = ["has_ts_operator", "rewrite_ts_operators"]

_OPS = r"(?:[LlFfDd]\d*)+"
_TERM = re.compile(
    rf"(?<![\w.])(?:(?P<ops>{_OPS})|(?P<kind>[LlFf])\((?P<lo>\d+)/(?P<hi>\d+)\)"
    rf"(?:\.?(?P<tail>{_OPS}))?)"
    r"\.(?P<var>[A-Za-z_]\w*)"
)
_ONE = re.compile(r"([LlFfDd])(\d*)")


def has_ts_operator(line: str) -> bool:
    return _TERM.search(line) is not None


def _parse_ops(ops: str) -> List[Tuple[str, int]]:
    """``'L2D'`` -> ``[('L', 2), ('D', 1)]``: applied right to left."""
    return [(k.upper(), int(n) if n else 1) for k, n in _ONE.findall(ops)]


def _name(var: str, ops: List[Tuple[str, int]]) -> str:
    return var + "_" + "".join(f"{k}{n}" for k, n in ops)


class _Clock:
    """Looks a series up at another period of the same panel."""

    def __init__(self, data: pd.DataFrame, unit: Optional[str], time: str) -> None:
        if time not in data.columns:
            raise StataExprError(f"the time variable {time!r} is not in the data")
        if unit is not None and unit not in data.columns:
            raise StataExprError(f"the panel variable {unit!r} is not in the data")
        col = data[time]
        if isinstance(col.dtype, pd.PeriodDtype):
            t = col.map(lambda p: p.ordinal if p is not pd.NaT else np.nan)
        elif pd.api.types.is_numeric_dtype(col):
            t = col
        else:
            raise StataExprError(
                f"the time variable {time!r} must be a number of periods or a "
                "pandas Period; a datetime has no unit step. Stata dates are "
                "integers"
            )
        t = t.to_numpy(dtype=float, na_value=np.nan)
        if np.isnan(t).any() or not np.all(t == np.round(t)):
            raise StataExprError(
                f"the time variable {time!r} has missing or non-integer values"
            )
        keys = pd.DataFrame({"u": 0 if unit is None else data[unit].to_numpy(), "t": t})
        if keys.duplicated().any():
            raise StataExprError(
                "repeated time values" + (" within a panel" if unit else "")
            )
        self._keys = keys

    def at(self, values: np.ndarray, offset: int) -> np.ndarray:
        """``values`` as observed ``offset`` periods later (negative: earlier)."""
        source = self._keys.assign(t=self._keys["t"] - offset, v=values)
        merged = self._keys.merge(source, how="left", on=["u", "t"])
        return np.asarray(merged["v"], dtype=float)


def _apply(clock: _Clock, values: np.ndarray, ops: List[Tuple[str, int]]) -> np.ndarray:
    out = values
    for kind, n in reversed(ops):
        if kind == "L":
            out = clock.at(out, -n)
        elif kind == "F":
            out = clock.at(out, n)
        else:  # D2 is the difference of the difference
            for _ in range(n):
                out = out - clock.at(out, -1)
    return out


def rewrite_ts_operators(
    line: str,
    data: Optional[pd.DataFrame],
    panel: Tuple[Optional[str], Optional[str]],
) -> Tuple[str, Dict[str, np.ndarray]]:
    """Replace operator terms by column names.

    Returns the rewritten line and the columns to add (name -> values), in
    the order they first appear. ``data`` may be ``None`` for a line that
    only names coefficients (``test``); no column is computed then.
    """
    unit, time = panel
    if time is None:
        raise StataExprError(
            "a time-series operator needs the time variable: put `tsset "
            "time` or `xtset id time` before it"
        )
    clock = None if data is None else _Clock(data, unit, time)
    new_columns: Dict[str, np.ndarray] = {}

    def column(var: str, ops: List[Tuple[str, int]]) -> Optional[str]:
        name = _name(var, ops)
        if data is None or clock is None:
            return name
        if var not in data.columns:
            # not a variable (a file name such as d.csv): leave the text alone
            return None
        source = data[var]
        if not pd.api.types.is_numeric_dtype(source):
            raise StataExprError(f"{var!r} is not numeric")
        if name in data.columns or name in new_columns:
            return name  # computed by an earlier line of the same session
        values = source.to_numpy(dtype=float, na_value=np.nan)
        new_columns[name] = _apply(clock, values, ops)
        return name

    def replace(m: "re.Match[str]") -> str:
        var = m.group("var")
        if m.group("ops") is not None:
            return column(var, _parse_ops(m.group("ops"))) or m.group(0)
        lo, hi = int(m.group("lo")), int(m.group("hi"))
        if hi < lo:
            raise StataExprError(f"empty lag list in {m.group(0)!r}")
        kind = m.group("kind").upper()
        # L(1/3).D.x: lags one to three of the first difference
        tail = _parse_ops(m.group("tail")) if m.group("tail") else []
        terms: List[str] = []
        for k in range(lo, hi + 1):
            ops = ([] if k == 0 else [(kind, k)]) + tail
            term = var if not ops else column(var, ops)
            if term is None:
                return m.group(0)
            terms.append(term)
        return " ".join(terms)

    return _TERM.sub(replace, line), new_columns
