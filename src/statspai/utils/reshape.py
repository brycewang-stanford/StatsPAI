"""Stata's ``reshape wide`` / ``reshape long``.

The column order and the rows are Stata's: wide is ``i``, then one block of
the stubs per value of ``j``, then the variables that are constant within
``i``; long is ``i``, ``j``, the stubs, then the rest, with a row for every
``i`` and every value of ``j`` found in the names, missing where the wide
data had no such column.

This module is the one implementation: ``sp.reshape`` and the ``reshape``
data step of ``sp.stata`` both call :func:`reshape_wide_frame` /
:func:`reshape_long_frame`.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import ColumnNotFound, MethodIncompatibility

__all__ = ["reshape", "reshape_long_frame", "reshape_wide_frame"]


def _label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _refuse(message: str, hint: str) -> MethodIncompatibility:
    return MethodIncompatibility(message, recovery_hint=hint)


def reshape_wide_frame(
    data: pd.DataFrame,
    stubs: List[str],
    i_vars: List[str],
    j_var: str,
    string: bool = False,
) -> pd.DataFrame:
    """Long -> wide, Stata's ``reshape wide stubs, i(i_vars) j(j_var)``."""
    absent = [v for v in [*stubs, j_var] if v not in data.columns]
    if absent:
        raise ColumnNotFound(
            f"reshape wide: variable(s) {absent} are not in the data",
            recovery_hint="Name the stubs and j() as columns of the long data.",
            diagnostics={"missing_columns": absent},
        )
    if string:
        # j(name) string: the values are the suffixes as they are
        j = data[j_var].astype(object)
        if j.isna().any() or (j == "").any():
            raise _refuse(
                f"reshape wide: j({j_var}) has missing values",
                "Drop the rows with a missing j value first.",
            )
    else:
        j = pd.to_numeric(data[j_var], errors="coerce")
        if j.isna().any():
            raise _refuse(
                f"reshape wide: j({j_var}) has missing or non-numeric values",
                "Drop the rows with a missing j, or pass string=True for a " "text j.",
            )
    if data.duplicated(subset=[*i_vars, j_var]).any():
        raise _refuse(
            f"reshape wide: values of {j_var} are not unique within {i_vars}",
            "Each (i, j) pair must be one row; aggregate the duplicates "
            "first (sp.collapse).",
        )
    rest = [c for c in data.columns if c not in {*stubs, *i_vars, j_var}]
    varying = [
        c for c in rest if data.groupby(i_vars)[c].nunique(dropna=False).gt(1).any()
    ]
    if varying:
        raise _refuse(
            f"reshape wide: variable(s) {varying} are not constant within "
            f"{i_vars}; list them as stubs or drop them",
            "Add them to the stubs, or drop them before reshaping.",
        )
    levels = sorted(j.unique())
    base = data.groupby(i_vars, sort=True)[rest].first() if rest else None
    keyed = data.assign(**{j_var: j}).set_index([*i_vars, j_var])
    blocks = {}
    for level in levels:
        for stub in stubs:
            name = f"{stub}{level if string else _label(level)}"
            if name in data.columns:
                raise _refuse(
                    f"reshape wide: variable {name!r} already exists",
                    "Rename or drop that column first.",
                )
            blocks[name] = keyed[stub].xs(level, level=j_var)
    wide = pd.DataFrame(blocks)
    index = (
        base.index if base is not None else data.groupby(i_vars, sort=True).size().index
    )
    wide = wide.reindex(index)
    if base is not None:
        wide = pd.concat([wide, base], axis=1)
    return wide.reset_index()


def reshape_long_frame(
    data: pd.DataFrame,
    stubs: List[str],
    i_vars: List[str],
    j_var: str,
    string: bool = False,
) -> pd.DataFrame:
    """Wide -> long, Stata's ``reshape long stubs, i(i_vars) j(j_var)``."""
    if j_var in data.columns:
        raise _refuse(
            f"reshape long: variable {j_var!r} already exists",
            "Name a new variable in j().",
        )
    if data.duplicated(subset=i_vars).any():
        raise _refuse(
            f"reshape long: {i_vars} do not identify the rows",
            "i() must take a different value on every row of the wide data.",
        )
    found = {}
    for stub in stubs:
        suffix = ".+" if string else r"\d+"
        pattern = re.compile(rf"^{re.escape(stub)}({suffix})$")
        found[stub] = {
            (m.group(1) if string else int(m.group(1))): c
            for c in data.columns
            if (m := pattern.match(str(c)))
        }
        if not found[stub]:
            raise _refuse(
                f"reshape long: no variable named {stub}<number>",
                "The stubs are the names before the number (inc for inc80 "
                "inc81); pass string=True for text suffixes.",
            )
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


def _names(value: Union[str, Sequence[str], None], what: str) -> List[str]:
    if value is None:
        return []
    out = value.split() if isinstance(value, str) else [str(v) for v in value]
    return out


@accepts_aliases(i="id")
def reshape(
    data: pd.DataFrame,
    direction: str,
    stubs: Union[str, Sequence[str]],
    id: Union[str, Sequence[str]],
    j: str,
    string: bool = False,
) -> pd.DataFrame:
    """Reshape between long and wide, as Stata's ``reshape`` does.

    ``sp.reshape(df, "wide", stubs="inc", id="hid", j="year")`` is
    ``reshape wide inc, i(hid) j(year)``: one row per ``hid`` and the
    columns ``inc80``, ``inc81``, ... ``sp.reshape(df, "long", ...)`` goes
    back.

    Parameters
    ----------
    data : DataFrame
        Not modified.
    direction : {'wide', 'long'}
    stubs : str or list of str
        The variables that vary with ``j`` (wide: their names; long: the
        part of the wide names before the suffix).
    id : str or list of str
        Stata's ``i()``: the variable(s) that identify a wide row.
    j : str
        Stata's ``j()``: the variable whose values become the suffixes
        (wide), or the new variable that holds them (long).
    string : bool, default False
        Stata's ``string`` option: ``j`` takes text values. Without it the
        suffixes must be whole numbers.

    Returns
    -------
    DataFrame
        Wide: ``id``, then one block of the stubs per value of ``j`` in
        sorted order, then the variables that are constant within ``id``.
        Long: ``id``, ``j``, the stubs, then the rest, sorted by ``id`` and
        ``j``, with a row for every ``id`` and every suffix found (missing
        where the wide data had no such column). The index is a fresh
        ``RangeIndex``.

    Raises
    ------
    MethodIncompatibility
        As Stata refuses: ``j`` not unique within ``id`` (wide), a
        variable not listed as a stub that varies within ``id`` (wide), a
        column that the result would create already existing, ``id`` not
        identifying the wide rows (long), or no column matching a stub
        (long).

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> long = pd.DataFrame({"hid": [1, 1, 2, 2], "year": [80, 81, 80, 81],
    ...                      "inc": [5.0, 6.0, 7.0, 8.0], "sex": [0, 0, 1, 1]})
    >>> wide = sp.reshape(long, "wide", stubs="inc", id="hid", j="year")
    >>> list(wide.columns)
    ['hid', 'inc80', 'inc81', 'sex']
    >>> back = sp.reshape(wide, "long", stubs="inc", id="hid", j="year")
    >>> back[["hid", "year", "inc"]].values.tolist()
    [[1, 80, 5.0], [1, 81, 6.0], [2, 80, 7.0], [2, 81, 8.0]]

    References
    ----------
    StataCorp. Stata 18 Data Management Reference Manual, ``[D] reshape``.
    """
    how = str(direction).strip().lower()
    if how not in ("wide", "long"):
        raise MethodIncompatibility(
            f"sp.reshape: direction must be 'wide' or 'long', not {direction!r}",
            recovery_hint="Pass direction='wide' (long -> wide) or 'long'.",
        )
    stub_list = _names(stubs, "stubs")
    i_vars = _names(id, "id")
    if not stub_list or not i_vars or not isinstance(j, str) or not j.strip():
        raise MethodIncompatibility(
            "sp.reshape needs the stub names, id= (Stata's i()) and j=",
            recovery_hint="e.g. sp.reshape(df, 'wide', stubs='inc', id='hid', "
            "j='year').",
        )
    missing = [v for v in i_vars if v not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(
            {"id": missing[0]} if len(missing) == 1 else missing,
            list(data.columns),
            context="sp.reshape",
        )
    build = reshape_wide_frame if how == "wide" else reshape_long_frame
    return build(data, stub_list, i_vars, j.strip(), bool(string)).reset_index(
        drop=True
    )
