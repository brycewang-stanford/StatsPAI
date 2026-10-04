"""``reshape wide`` / ``reshape long`` for the data steps of ``sp.stata``.

The two forms a do-file writes most: ``reshape wide stubs, i(id) j(t)`` and
``reshape long stubs, i(id) j(t)`` with a numeric ``j``. The column order and
the rows are Stata's: wide is ``i``, then one block of the stubs per value of
``j``, then the variables that are constant within ``i``; long is ``i``,
``j``, the stubs, then the rest, with a row for every ``i`` and every value
of ``j`` found in the names, missing where the wide data had no such column
value.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from ._stata_expr import StataExprError

if TYPE_CHECKING:
    from ._stata_datastep import DataSteps

__all__ = ["run_reshape"]


def _label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def run_reshape(steps: "DataSteps", varlist: List[str], options: dict) -> None:
    if not varlist or varlist[0] not in ("wide", "long"):
        raise StataExprError(
            "only `reshape wide stubs, i() j()` and `reshape long stubs, i() "
            "j()` are implemented"
        )
    direction, stubs = varlist[0], list(varlist[1:])
    i_vars = str(options.pop("i", "") or "").split()
    j_parts = str(options.pop("j", "") or "").split()
    if options:
        raise StataExprError(
            f"reshape: option(s) {sorted(options)} are not implemented"
        )
    if not stubs or not i_vars or len(j_parts) != 1:
        raise StataExprError("reshape needs the stub names, i(varlist) and j(varname)")
    j_var = j_parts[0]
    data = steps.data
    missing = [v for v in i_vars if v not in data.columns]
    if missing:
        raise StataExprError(f"reshape: variable(s) {missing} are not in the data")
    out = (
        _wide(data, stubs, i_vars, j_var)
        if direction == "wide"
        else _long(data, stubs, i_vars, j_var)
    )
    steps._own()
    steps.data = out.reset_index(drop=True)
    steps._float &= set(steps.data.columns)


def _wide(
    data: pd.DataFrame, stubs: List[str], i_vars: List[str], j_var: str
) -> pd.DataFrame:
    absent = [v for v in [*stubs, j_var] if v not in data.columns]
    if absent:
        raise StataExprError(f"reshape wide: variable(s) {absent} are not in the data")
    j = pd.to_numeric(data[j_var], errors="coerce")
    if j.isna().any():
        raise StataExprError(
            f"reshape wide: j({j_var}) has missing or non-numeric values"
        )
    if data.duplicated(subset=[*i_vars, j_var]).any():
        raise StataExprError(
            f"reshape wide: values of {j_var} are not unique within {i_vars}"
        )
    rest = [c for c in data.columns if c not in {*stubs, *i_vars, j_var}]
    varying = [
        c for c in rest if data.groupby(i_vars)[c].nunique(dropna=False).gt(1).any()
    ]
    if varying:
        raise StataExprError(
            f"reshape wide: variable(s) {varying} are not constant within "
            f"{i_vars}; list them as stubs or drop them"
        )
    levels = sorted(j.unique())
    base = data.groupby(i_vars, sort=True)[rest].first() if rest else None
    keyed = data.assign(**{j_var: j}).set_index([*i_vars, j_var])
    blocks = {}
    for level in levels:
        for stub in stubs:
            name = f"{stub}{_label(level)}"
            if name in data.columns:
                raise StataExprError(f"reshape wide: variable {name!r} already exists")
            blocks[name] = keyed[stub].xs(level, level=j_var)
    wide = pd.DataFrame(blocks)
    index = (
        base.index if base is not None else data.groupby(i_vars, sort=True).size().index
    )
    wide = wide.reindex(index)
    if base is not None:
        wide = pd.concat([wide, base], axis=1)
    return wide.reset_index()


def _long(
    data: pd.DataFrame, stubs: List[str], i_vars: List[str], j_var: str
) -> pd.DataFrame:
    if j_var in data.columns:
        raise StataExprError(f"reshape long: variable {j_var!r} already exists")
    if data.duplicated(subset=i_vars).any():
        raise StataExprError(f"reshape long: {i_vars} do not identify the rows")
    found = {}
    for stub in stubs:
        pattern = re.compile(rf"^{re.escape(stub)}(\d+)$")
        found[stub] = {
            int(m.group(1)): c for c in data.columns if (m := pattern.match(str(c)))
        }
        if not found[stub]:
            raise StataExprError(f"reshape long: no variable named {stub}<number>")
    levels = sorted({k for cols in found.values() for k in cols})
    wide_cols = {c for cols in found.values() for c in cols.values()}
    rest = [c for c in data.columns if c not in wide_cols and c not in i_vars]
    pieces = []
    for level in levels:
        piece = data[i_vars].copy()
        piece[j_var] = level
        for stub in stubs:
            col = found[stub].get(level)
            piece[stub] = data[col].to_numpy() if col is not None else np.nan
        for c in rest:
            piece[c] = data[c].to_numpy()
        pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)
    return long.sort_values([*i_vars, j_var], kind="mergesort")
