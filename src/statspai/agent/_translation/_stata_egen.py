"""``egen`` for the data steps of ``sp.stata``.

``egen [type] newvar = fcn(arguments) [if] [in] [, by(varlist) options]``,
also behind ``by varlist:`` / ``bysort varlist:``. Each function follows its
entry in ``[D] egen``; what the manual says about missing values is what
decides most of them, so it is repeated beside each one.

Statistics of an expression within the group (the whole data without
``by``), given to every selected row of the group:

``count``    the number of non-missing values (0, not missing, for none)
``mean`` ``median`` ``sd`` ``min`` ``max``
             over the non-missing values; missing when there are none
             (``sd``: fewer than two)
``total``    missing counts as zero, so a group with no value gets 0;
             with the option ``missing`` it gets missing
``pctile``   ``p(#)``, default 50, with the definition of ``summarize,
             detail``; ``iqr`` is ``pctile 75 - pctile 25``
``skew`` ``kurt``
             the moment ratios of ``summarize, detail``
``mad`` ``mdev``
             median absolute deviation from the median; mean absolute
             deviation from the mean
``mode``     the most frequent value; missing when several tie, unless
             ``minmode`` or ``maxmode`` picks one
``std``      ``(x - mean) / sd`` over the selected rows (no ``by``)

Row by row within the group:

``rank``     1 for the smallest value, ties sharing their average rank;
             ``field`` (1 + the number of higher values) and ``track``
             (1 + the number of lower values) count instead
``seq``      ``from()`` to ``to()`` in steps of 1, each value ``block()``
             times, starting again when ``to()`` is passed

Functions of a variable list:

``group``    1, 2, ... for the distinct combinations in sorted order;
             missing where a variable is missing, unless ``missing``
``tag``      1 for one row of each distinct combination (the first in the
             current order), 0 elsewhere and where a variable is missing
             (unless ``missing``); never missing
``rowtotal`` missing counts as zero (``missing``: all missing gives missing)
``rowmean`` ``rowmin`` ``rowmax`` ``rowsd``
             over the non-missing values; missing when there are none
``rownonmiss`` ``rowmiss``
             counts
``rowfirst`` ``rowlast``
             the first / last non-missing value in the order of the list
``anycount`` ``anymatch``
             how many of the variables equal one of ``values()``; whether
             any does
``cut``      ``at(#, #, ...)``: the left end of the interval ``[a_k,
             a_k+1)`` the value falls in (``icodes``: 0, 1, ...), missing
             outside the cut points

Rows not selected by ``if`` / ``in`` get missing (``tag``: 0). The result
is stored in single precision unless the line says ``double``, as in Stata;
counts, groups and tags are whole numbers and are exact either way.

Anything else (``rank`` with ``unique``, the string functions ``ends``
and ``concat``, ``fill``, the user-written ``egenmore`` functions) is
refused.

The computation is ``statspai.utils.egen.egen_compute``, the one
implementation behind the public ``sp.egen``; this module parses the line,
evaluates the expression (per group behind ``by g:``, where ``_n`` and
``_N`` count within the group), stores the result in single precision
unless the line says ``double``, and turns the taxonomy errors into
``StataExprError`` with the same message.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...exceptions import StatsPAIError
from ...utils.egen import (
    EGEN_FUNCTIONS,
    EGEN_GROUP_STATS,
    EGEN_NO_BY,
    EGEN_OPTIONS,
    EGEN_ROW,
    egen_codes,
    egen_compute,
)
from ._stata_expr import StataExprError, evaluate

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_datastep import DataSteps

__all__ = ["run_egen"]

_HEAD = re.compile(
    r"\s*egen\s+(?:(byte|int|long|float|double)\s+)?([A-Za-z_]\w*)\s*=\s*"
    r"([A-Za-z_]\w*)\s*\(",
    re.I,
)
_WHOLE = ("count", "group", "tag", "rownonmiss", "rowmiss", "anycount",
          "anymatch", "seq")  # fmt: skip
#: Stata's abbreviations of the seq() / anycount() options
_SHORT = {"f": "from", "t": "to", "b": "block", "v": "values"}
#: options that are flags (present or not)
_FLAGS = ("missing", "minmode", "maxmode", "field", "track", "icodes", "label")


def _balanced(text: str, start: int) -> int:
    """Index of the parenthesis closing the one just before ``start``."""
    depth, quote = 1, ""
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    raise StataExprError("egen: unbalanced parentheses")


def _varlist(spec: str, data: pd.DataFrame) -> List[str]:
    """Names, ``a-c`` ranges (in the order of the data) and ``x*`` patterns."""
    columns = [str(c) for c in data.columns]
    names: List[str] = []
    for token in spec.replace(",", " ").split():
        if "-" in token and token not in columns:
            first, _, last = token.partition("-")
            if first not in columns or last not in columns:
                raise StataExprError(f"egen: cannot read the range {token!r}")
            lo, hi = columns.index(first), columns.index(last)
            if lo > hi:
                raise StataExprError(f"egen: {token!r} runs backwards")
            names.extend(columns[lo : hi + 1])
        elif any(ch in token for ch in "*?"):
            hits = [c for c in columns if fnmatch.fnmatchcase(c, token)]
            if not hits:
                raise StataExprError(f"egen: no variable matches {token!r}")
            names.extend(hits)
        elif token in columns:
            names.append(token)
        else:
            raise StataExprError(f"egen: variable {token!r} is not in the data")
    if not names:
        raise StataExprError("egen: a variable list is expected")
    return names


def _values(spec: str) -> List[float]:
    """``values(1 3/5)`` / ``at(0, 10, 20)``: the numbers listed."""
    out: List[float] = []
    for token in spec.replace(",", " ").split():
        m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", token)
        n = re.fullmatch(r"(-?[\d.]+)\((-?[\d.]+)\)(-?[\d.]+)", token)
        try:
            if m:
                out.extend(
                    np.arange(float(m.group(1)), float(m.group(2)) + 0.5).tolist()
                )
            elif n:
                lo, step, hi = (float(n.group(i)) for i in (1, 2, 3))
                out.extend(np.arange(lo, hi + step / 2, step).tolist())
            else:
                out.append(float(token))
        except ValueError:
            raise StataExprError(
                f"egen: cannot read the number list {spec!r}"
            ) from None
    if not out:
        raise StataExprError("egen: a list of numbers is expected")
    return out


def run_egen(
    steps: "DataSteps", line: str, groups: Optional[List[np.ndarray]] = None
) -> None:
    """Run one ``egen`` line on ``steps.data``.

    ``groups`` are the row positions of each ``by`` group when the line was
    behind a ``by varlist:`` prefix (the data are then already sorted).
    """
    from ._stata_datastep import row_mask
    from ._stata_lexer import StataParseError
    from ._stata_lexer import parse as _parse

    head = _HEAD.match(line)
    if head is None:
        raise StataExprError("expected `egen [type] newvar = fcn(arguments)`")
    vtype, name, fcn = head.group(1), head.group(2), head.group(3).lower()
    close = _balanced(line, head.end())
    argument = line[head.end() : close].strip()
    try:
        tail = _parse("egen _x" + line[close + 1 :])
    except StataParseError as exc:
        raise StataExprError(f"egen: cannot read the qualifiers ({exc})") from exc
    if tail.varlist != ["_x"]:
        raise StataExprError("egen: text after the function is not understood")
    # Parsed option values: strings from the command line, then flags,
    # number lists and numbers once interpreted below.
    options: Dict[str, Any] = {str(k).lower(): v for k, v in dict(tail.options).items()}
    by_option = str(options.pop("by", "") or "").split()
    data = steps.data
    n = len(data)
    if name in data.columns:
        raise StataExprError(f"`egen`: variable {name!r} already exists")
    if vtype is not None and vtype.lower() not in ("float", "double"):
        if fcn not in _WHOLE:
            raise StataExprError(
                f"`egen {vtype}` of a statistic truncates in Stata; that is "
                "not implemented"
            )
    if fcn not in EGEN_FUNCTIONS:
        raise StataExprError(f"egen function {fcn}() is not implemented")
    if fcn in ("seq", "anycount", "anymatch"):
        options = {_SHORT.get(k, k): v for k, v in options.items()}
    extra = set(options) - EGEN_OPTIONS.get(fcn, frozenset())
    if extra:
        raise StataExprError(
            f"egen {fcn}(): option(s) {sorted(extra)} are not implemented"
        )
    if by_option and groups is not None:
        raise StataExprError("egen: `by` is given twice")
    if (by_option or groups is not None) and fcn in EGEN_NO_BY:
        raise StataExprError(f"egen {fcn}() may not be combined with by")
    unknown = [b for b in by_option if b not in data.columns]
    if unknown:
        raise StataExprError(f"egen: variable(s) {unknown} are not in the data")
    for flag in _FLAGS:
        if flag in options:
            options[flag] = True
    if fcn in ("anycount", "anymatch"):
        options["values"] = _values(str(options.get("values") or ""))
    elif fcn == "cut" and (options.get("at") is not None or not options.get("group")):
        options["at"] = _values(str(options.get("at") or ""))
    if "p" in options:
        options["p"] = float(options["p"] or 50.0)

    mask = row_mask(data, tail.if_cond, tail.in_range, steps.stored)
    code: Optional[np.ndarray] = None
    if groups is not None:
        code = np.zeros(n, dtype=int)
        for g, rows in enumerate(groups):
            code[rows] = g
    elif by_option:
        code = egen_codes(data[by_option], keep_missing=True)

    x: Optional[np.ndarray] = None
    frame: Optional[pd.DataFrame] = None
    if fcn == "seq":
        if argument:
            raise StataExprError("egen seq() takes no argument")
    elif fcn in ("group", "tag") or fcn in EGEN_ROW:
        frame = data[_varlist(argument, data)]
    elif fcn in EGEN_GROUP_STATS and groups is not None:
        if not argument:
            raise StataExprError(f"egen {fcn}() needs an expression")
        # behind `by g:` the expression sees one group at a time
        x = np.full(n, np.nan)
        for rows in groups:
            part = data.iloc[rows].reset_index(drop=True)
            got = evaluate(argument, part, steps.stored)
            if got.dtype == object:
                raise StataExprError(f"egen {fcn}() of a string expression")
            x[rows] = got
    else:
        if fcn in EGEN_GROUP_STATS and not argument:
            raise StataExprError(f"egen {fcn}() needs an expression")
        x = evaluate(argument, data, steps.stored)
        if fcn in EGEN_GROUP_STATS and x.dtype == object:
            raise StataExprError(f"egen {fcn}() of a string expression")
    try:
        value = egen_compute(fcn, n, mask, options, x=x, code=code, frame=frame)
    except StatsPAIError as exc:
        raise StataExprError(exc.message) from None

    double = (vtype or "").lower() == "double"
    steps.add_column(name, value, double=double)
