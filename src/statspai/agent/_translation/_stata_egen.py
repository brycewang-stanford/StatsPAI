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

Anything else (``cut`` with ``group()``, ``rank`` with ``unique``, the
string functions ``ends`` and ``concat``, ``fill``, the user-written
``egenmore`` functions) is refused.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._stata_expr import StataExprError, evaluate

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_datastep import DataSteps

__all__ = ["run_egen"]

_HEAD = re.compile(
    r"\s*egen\s+(?:(byte|int|long|float|double)\s+)?([A-Za-z_]\w*)\s*=\s*"
    r"([A-Za-z_]\w*)\s*\(",
    re.I,
)
_GROUP_STATS = (
    "count", "mean", "median", "sd", "min", "max", "total", "sum",
    "pctile", "iqr", "skew", "kurt", "mad", "mdev", "mode",
)  # fmt: skip
_ROW = (
    "rowtotal", "rowmean", "rowmin", "rowmax", "rowsd", "rownonmiss", "rowmiss",
    "rowfirst", "rowlast", "anycount", "anymatch", "cut",
)  # fmt: skip
_WHOLE = ("count", "group", "tag", "rownonmiss", "rowmiss", "anycount",
          "anymatch", "seq")  # fmt: skip


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


def _percentile(sorted_values: np.ndarray, p: float) -> float:
    from ._stata_session import stata_percentile

    return stata_percentile(sorted_values, p)


def _statistic(fcn: str, x: np.ndarray, options: Dict[str, Any]) -> float:
    """One group's statistic of the non-missing values ``x``."""
    n = x.size
    if fcn == "count":
        return float(n)
    if fcn in ("total", "sum"):
        if n == 0:
            return float("nan") if "missing" in options else 0.0
        return float(x.sum())
    if n == 0:
        return float("nan")
    if fcn == "mean":
        return float(x.mean())
    if fcn == "median":
        return _percentile(x, 50.0)
    if fcn == "min":
        return float(x.min())
    if fcn == "max":
        return float(x.max())
    if fcn == "sd":
        return float(x.std(ddof=1)) if n > 1 else float("nan")
    if fcn == "pctile":
        return _percentile(x, float(options.get("p") or 50.0))
    if fcn in ("skew", "kurt"):
        dev = x - x.mean()
        m2 = float(np.mean(dev**2))
        if not m2 > 0:
            return float("nan")
        power = 3 if fcn == "skew" else 4
        return float(np.mean(dev**power) / m2 ** (power / 2))
    if fcn == "mad":
        return _percentile(np.sort(np.abs(x - _percentile(x, 50.0))), 50.0)
    if fcn == "mdev":
        return float(np.mean(np.abs(x - x.mean())))
    if fcn == "mode":
        values, counts = np.unique(x, return_counts=True)
        modes = values[counts == counts.max()]
        if len(modes) == 1 or "minmode" in options:
            return float(modes[0])
        return float(modes[-1]) if "maxmode" in options else float("nan")
    return _percentile(x, 75.0) - _percentile(x, 25.0)  # iqr


def _combination_codes(frame: pd.DataFrame, keep_missing: bool) -> np.ndarray:
    """0, 1, ... for the distinct rows of ``frame`` in sorted order; -1 where
    a value is missing and missing values do not form groups."""
    if keep_missing:
        # a missing value sorts after every number, as in Stata
        return np.asarray(
            frame.groupby(list(frame.columns), sort=True, dropna=False).ngroup()
        )
    codes = frame.groupby(list(frame.columns), sort=True, dropna=True).ngroup()
    return np.asarray(codes.fillna(-1).astype(int))


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
    options = {str(k).lower(): v for k, v in dict(tail.options).items()}
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
    known = _GROUP_STATS + _ROW + ("group", "tag", "std", "rank", "seq")
    if fcn not in known:
        raise StataExprError(f"egen function {fcn}() is not implemented")
    allowed = {"pctile": {"p"}, "total": {"missing"}, "sum": {"missing"}}
    allowed.update({"group": {"missing"}, "tag": {"missing"}, "rowtotal": {"missing"}})
    allowed.update({"mode": {"minmode", "maxmode"}, "rank": {"field", "track"}})
    allowed.update({"seq": {"from", "f", "to", "t", "block", "b"}})
    allowed.update({"anycount": {"values", "v"}, "anymatch": {"values", "v"}})
    allowed.update({"cut": {"at", "icodes"}})
    extra = set(options) - allowed.get(fcn, set())
    if extra:
        raise StataExprError(
            f"egen {fcn}(): option(s) {sorted(extra)} are not implemented"
        )
    if by_option and groups is not None:
        raise StataExprError("egen: `by` is given twice")
    if (by_option or groups is not None) and fcn in _ROW + (
        "group",
        "tag",
        "std",
    ):  # noqa: E501
        raise StataExprError(f"egen {fcn}() may not be combined with by")
    unknown = [b for b in by_option if b not in data.columns]
    if unknown:
        raise StataExprError(f"egen: variable(s) {unknown} are not in the data")

    mask = row_mask(data, tail.if_cond, tail.in_range, steps.stored)
    value: Any = np.full(n, np.nan)

    if fcn in ("rank", "seq"):
        if groups is not None:
            code = np.zeros(n, dtype=int)
            for g, rows in enumerate(groups):
                code[rows] = g
        elif by_option:
            code = _combination_codes(data[by_option], keep_missing=True)
        else:
            code = np.zeros(n, dtype=int)
        if fcn == "seq":
            if argument:
                raise StataExprError("egen seq() takes no argument")
            start = float(options.get("from") or options.get("f") or 1)
            repeat = int(float(options.get("block") or options.get("b") or 1))
            stop = options.get("to") or options.get("t")
            for g in np.unique(code):
                rows = np.flatnonzero((code == g) & mask)
                steps_ = np.arange(len(rows)) // max(repeat, 1)
                if stop is not None:
                    span = int(float(stop) - start) + 1
                    if span < 1:
                        raise StataExprError("egen seq(): to() is below from()")
                    steps_ = steps_ % span
                value[rows] = start + steps_
        else:
            x = evaluate(argument, data, steps.stored)
            used = mask & ~np.isnan(x)
            for g in np.unique(code):
                rows = np.flatnonzero((code == g) & used)
                v = x[rows]
                lower = (v[:, None] > v[None, :]).sum(axis=1)
                higher = (v[:, None] < v[None, :]).sum(axis=1)
                if "field" in options:
                    value[rows] = 1.0 + higher
                elif "track" in options:
                    value[rows] = 1.0 + lower
                else:
                    ties = len(v) - lower - higher
                    value[rows] = lower + (ties + 1) / 2.0
    elif fcn in _GROUP_STATS:
        if not argument:
            raise StataExprError(f"egen {fcn}() needs an expression")
        if groups is not None:
            # behind `by g:` the expression sees one group at a time
            x = np.full(n, np.nan)
            code = np.zeros(n, dtype=int)
            for g, rows in enumerate(groups):
                part = data.iloc[rows].reset_index(drop=True)
                got = evaluate(argument, part, steps.stored)
                if got.dtype == object:
                    raise StataExprError(f"egen {fcn}() of a string expression")
                x[rows] = got
                code[rows] = g
        else:
            x = evaluate(argument, data, steps.stored)
            if x.dtype == object:
                raise StataExprError(f"egen {fcn}() of a string expression")
            if by_option:
                code = _combination_codes(data[by_option], keep_missing=True)
            else:
                code = np.zeros(n, dtype=int)
        used = mask & ~np.isnan(x)
        order = np.argsort(code, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(code[order]) != 0, True])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            rows = order[lo:hi]
            stat = _statistic(fcn, np.sort(x[rows][used[rows]]), options)
            value[rows] = stat
        value = np.where(mask, value, np.nan)
    elif fcn == "std":
        x = evaluate(argument, data, steps.stored)
        held = x[mask & ~np.isnan(x)]
        if held.size > 1 and held.std(ddof=1) > 0:
            value = np.where(mask, (x - held.mean()) / held.std(ddof=1), np.nan)
    elif fcn in ("group", "tag"):
        names = _varlist(argument, data)
        code = _combination_codes(data[names], keep_missing="missing" in options)
        selected = mask & (code >= 0)
        if fcn == "group":
            levels = np.unique(code[selected])
            rank = {int(c): i + 1 for i, c in enumerate(levels)}
            value = np.array(
                [rank[int(c)] if s else np.nan for c, s in zip(code, selected)],
                dtype=float,
            )
        else:
            value = np.zeros(n)
            positions = np.flatnonzero(selected)
            _, first = np.unique(code[positions], return_index=True)
            value[positions[first]] = 1.0
    else:  # row functions
        names = _varlist(argument, data)
        block = np.column_stack(
            [data[c].to_numpy(dtype=float, na_value=np.nan) for c in names]
        )
        present = ~np.isnan(block)
        k = present.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            if fcn == "rownonmiss":
                value = k.astype(float)
            elif fcn == "rowmiss":
                value = (block.shape[1] - k).astype(float)
            elif fcn in ("rowfirst", "rowlast"):
                order = block if fcn == "rowfirst" else block[:, ::-1]
                seen = ~np.isnan(order)
                first = seen.argmax(axis=1)
                value = np.where(
                    seen.any(axis=1), order[np.arange(len(order)), first], np.nan
                )
            elif fcn in ("anycount", "anymatch"):
                spec = str(options.get("values") or options.get("v") or "")
                wanted = _values(spec)
                hits = np.isin(block, wanted).sum(axis=1).astype(float)
                value = hits if fcn == "anycount" else (hits > 0).astype(float)
            elif fcn == "cut":
                if block.shape[1] != 1:
                    raise StataExprError("egen cut() takes one variable")
                at = np.array(_values(str(options.get("at") or "")), dtype=float)
                if len(at) < 2 or np.any(np.diff(at) <= 0):
                    raise StataExprError(
                        "egen cut() needs at(#, #, ...) in ascending order; "
                        "group() is not implemented"
                    )
                x = block[:, 0]
                where = np.searchsorted(at, x, side="right") - 1
                inside = present[:, 0] & (where >= 0) & (x < at[-1])
                where = np.clip(where, 0, len(at) - 1)
                picked = where.astype(float) if "icodes" in options else at[where]
                value = np.where(inside, picked, np.nan)
            elif fcn == "rowtotal":
                value = np.where(present, block, 0.0).sum(axis=1)
                if "missing" in options:
                    value = np.where(k > 0, value, np.nan)
            elif fcn == "rowmean":
                value = np.where(
                    k > 0, np.where(present, block, 0.0).sum(axis=1) / k, np.nan
                )
            elif fcn == "rowmin":
                value = np.where(
                    k > 0, np.where(present, block, np.inf).min(axis=1), np.nan
                )
            elif fcn == "rowmax":
                value = np.where(
                    k > 0, np.where(present, block, -np.inf).max(axis=1), np.nan
                )
            else:  # rowsd
                mean = np.where(present, block, 0.0).sum(axis=1) / np.maximum(k, 1)
                dev = np.where(present, block - mean[:, None], 0.0)
                value = np.where(
                    k > 1, np.sqrt((dev**2).sum(axis=1) / np.maximum(k - 1, 1)), np.nan
                )
        value = np.where(mask, value, np.nan)

    double = (vtype or "").lower() == "double"
    steps.add_column(name, value, double=double)
