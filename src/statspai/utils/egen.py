"""
Stata-style egen row-wise and rank helpers.

Ports the core ``pyegen`` convenience functions to StatsPAI so users
migrating from PyStataR keep a familiar API. All functions accept a
DataFrame plus a list of variables and return a pandas Series aligned
to the DataFrame's index.

Equivalent to Stata's::

    egen newvar = rowmean(var1 var2 var3)
    egen newvar = rowtotal(var1 var2 var3)
    egen newvar = rowmax(var1 var2 var3)
    egen newvar = rowmin(var1 var2 var3)
    egen newvar = rowsd(var1 var2 var3)
    egen newvar = rownonmiss(var1 var2 var3)
    egen newvar = rank(var), [by(group)] [field|track|unique]

:func:`egen` is Stata's ``egen`` itself (statistics by group, ranks,
sequences, groups and tags, row functions, ``cut``) with Stata's rules for
missing values and ``if``; it is the one implementation, which the
``egen`` data step of ``sp.stata`` also calls.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..exceptions import ColumnNotFound, MethodIncompatibility


def _select(
    data: pd.DataFrame, vars: Sequence[str], function: Optional[str] = None
) -> pd.DataFrame:
    check_columns(data, {"vars": list(vars)}, function=function)
    return data[list(vars)]


def rowmean(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise mean across ``vars``, ignoring NaN (Stata ``egen rowmean``).

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"math": [80.0, 90.0, np.nan],
    ...                    "read": [85.0, 95.0, 60.0],
    ...                    "sci":  [78.0, 88.0, 65.0]})
    >>> sp.rowmean(df, ["math", "read", "sci"]).tolist()
    [81.0, 91.0, 62.5]
    """
    return _select(data, vars, "sp.rowmean").mean(axis=1, skipna=True)


def rowtotal(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise sum across ``vars``, treating NaN as 0 (Stata ``egen rowtotal``).

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"math": [80.0, 90.0, np.nan],
    ...                    "read": [85.0, 95.0, 60.0],
    ...                    "sci":  [78.0, 88.0, 65.0]})
    >>> sp.rowtotal(df, ["math", "read", "sci"]).tolist()
    [243.0, 273.0, 125.0]
    """
    return _select(data, vars, "sp.rowtotal").sum(axis=1, skipna=True)


def rowmax(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise max across ``vars``, ignoring NaN.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"math": [80.0, 90.0, np.nan],
    ...                    "read": [85.0, 95.0, 60.0],
    ...                    "sci":  [78.0, 88.0, 65.0]})
    >>> sp.rowmax(df, ["math", "read", "sci"]).tolist()
    [85.0, 95.0, 65.0]
    """
    return _select(data, vars, "sp.rowmax").max(axis=1, skipna=True)


def rowmin(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise min across ``vars``, ignoring NaN.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"math": [80.0, 90.0, np.nan],
    ...                    "read": [85.0, 95.0, 60.0],
    ...                    "sci":  [78.0, 88.0, 65.0]})
    >>> sp.rowmin(df, ["math", "read", "sci"]).tolist()
    [78.0, 88.0, 60.0]
    """
    return _select(data, vars, "sp.rowmin").min(axis=1, skipna=True)


def rowsd(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise standard deviation across ``vars`` (Stata ``egen rowsd``).

    Uses the sample standard deviation (``ddof=1``), matching Stata.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({"q1": [2.0, 5.0],
    ...                    "q2": [4.0, 5.0],
    ...                    "q3": [6.0, 5.0]})
    >>> sp.rowsd(df, ["q1", "q2", "q3"]).tolist()
    [2.0, 0.0]
    """
    return _select(data, vars, "sp.rowsd").std(axis=1, skipna=True, ddof=1)


def rowcount(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise count of non-missing values (Stata ``egen rownonmiss``).

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"math": [80.0, 90.0, np.nan],
    ...                    "read": [85.0, 95.0, 60.0],
    ...                    "sci":  [78.0, 88.0, np.nan]})
    >>> sp.rowcount(df, ["math", "read", "sci"]).tolist()
    [3, 3, 1]
    """
    return _select(data, vars, "sp.rowcount").notna().sum(axis=1).astype(int)


def rank(
    data: pd.DataFrame,
    var: str,
    by: Optional[Union[str, Sequence[str]]] = None,
    method: str = "average",
    ascending: bool = True,
) -> pd.Series:
    """
    Rank a variable, optionally within groups.

    Equivalent to Stata's ``egen newvar = rank(var), by(group)`` with
    ``method`` / ``ascending`` mapping to Stata's tie-handling options
    (checked in Stata 18):

    - ``'average'``                    → Stata default
    - ``'min'``                        → ``track`` (1 + the number of lower values)
    - ``'min'`` with ``ascending=False`` → ``field`` (1 + the number of higher)
    - ``'first'``                      → ``unique`` (ties in row order)
    - ``'max'`` / ``'dense'`` have no Stata counterpart.

    ``sp.egen(df, "rank", var, rank_ties="field")`` is the same with
    Stata's ``if`` and missing-value handling.

    Parameters
    ----------
    data : pd.DataFrame
    var : str
        Column to rank.
    by : str or list of str, optional
        Group variables. If given, ranks are computed within each group.
    method : {'average', 'min', 'max', 'dense', 'first'}, default 'average'
    ascending : bool, default True

    Returns
    -------
    pd.Series aligned to ``data.index``.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({"grp": ["a", "a", "b", "b"],
    ...                    "score": [3.0, 1.0, 4.0, 2.0]})
    >>> sp.rank(df, "score").tolist()
    [3.0, 1.0, 4.0, 2.0]
    >>> sp.rank(df, "score", by="grp").tolist()
    [2.0, 1.0, 2.0, 1.0]
    """
    check_columns(data, {"var": var, "by": by}, function="sp.rank")

    if by is None:
        return data[var].rank(method=method, ascending=ascending)

    by_cols = [by] if isinstance(by, str) else list(by)

    return data.groupby(by_cols, dropna=False)[var].rank(
        method=method, ascending=ascending
    )


def outlier_indicator(
    data: pd.DataFrame,
    vars: Sequence[str],
    cuts: tuple = (1, 99),
    combined: bool = False,
) -> pd.DataFrame:
    """
    Flag outlier observations by percentile cutoffs.

    Returns 0/1 indicator columns for values outside ``[cuts[0], cuts[1]]``
    percentile bounds. Mirrors the ``pywinsor2.outlier_indicator`` API.

    Parameters
    ----------
    data : pd.DataFrame
    vars : list of str
        Columns to check.
    cuts : tuple of (float, float), default (1, 99)
        Lower and upper percentile cutoffs.
    combined : bool, default False
        If True, also add a column ``_outlier_any`` marking rows that are
        outliers on *any* variable.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with added ``{var}_outlier`` 0/1 columns.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    >>> out = sp.outlier_indicator(df, ["x"], cuts=(10, 90))
    >>> out["x_outlier"].tolist()
    [1, 0, 0, 0, 1]
    """
    df = data.copy()
    lo_pct, hi_pct = cuts
    flag_cols = []

    check_columns(df, {"vars": list(vars)}, function="sp.outlier_indicator")
    for v in vars:
        col = df[v].astype(float)
        valid = col.notna()
        if valid.any():
            lo = np.nanpercentile(col[valid], lo_pct)
            hi = np.nanpercentile(col[valid], hi_pct)
            flag = ((col < lo) | (col > hi)).astype(int)
        else:
            flag = pd.Series(0, index=col.index, dtype=int)
        flag[~valid] = 0
        name = f"{v}_outlier"
        df[name] = flag
        flag_cols.append(name)

    if combined and flag_cols:
        df["_outlier_any"] = (df[flag_cols].sum(axis=1) > 0).astype(int)

    return df


# ======================================================================
# egen: one implementation for sp.egen and the `egen` step of sp.stata
# ======================================================================

#: statistics of an expression within a group
EGEN_GROUP_STATS = (
    "count", "mean", "median", "sd", "min", "max", "total", "sum",
    "pctile", "iqr", "skew", "kurt", "mad", "mdev", "mode",
)  # fmt: skip
#: functions of a variable list, row by row
EGEN_ROW = (
    "rowtotal", "rowmean", "rowmin", "rowmax", "rowsd", "rownonmiss", "rowmiss",
    "rowfirst", "rowlast", "anycount", "anymatch", "cut",
)  # fmt: skip
EGEN_FUNCTIONS = EGEN_GROUP_STATS + EGEN_ROW + ("group", "tag", "std", "rank", "seq")
#: options (Stata's names) each function reads
EGEN_OPTIONS: Dict[str, frozenset] = {
    "pctile": frozenset({"p"}),
    "total": frozenset({"missing"}),
    "sum": frozenset({"missing"}),
    "group": frozenset({"missing"}),
    "tag": frozenset({"missing"}),
    "rowtotal": frozenset({"missing"}),
    "mode": frozenset({"minmode", "maxmode"}),
    "rank": frozenset({"field", "track"}),
    "seq": frozenset({"from", "to", "block"}),
    "anycount": frozenset({"values"}),
    "anymatch": frozenset({"values"}),
    "cut": frozenset({"at", "icodes", "group", "label"}),
}
#: functions that cannot be combined with by
EGEN_NO_BY = EGEN_ROW + ("group", "tag", "std")


def _refuse(message: str, hint: str) -> MethodIncompatibility:
    return MethodIncompatibility(message, recovery_hint=hint)


def stata_percentile(x: np.ndarray, p: float) -> float:
    """The ``p``-th percentile as ``summarize, detail`` and ``_pctile``
    define it: with ``P = n p / 100``, the mean of the ``P``-th and
    ``(P+1)``-th order statistics when ``P`` is an integer, otherwise the
    next order statistic above ``P``."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    pos = n * p / 100.0
    k = int(np.floor(pos + 1e-12))
    if abs(pos - k) < 1e-12:
        lo = x[max(k, 1) - 1]
        hi = x[min(k + 1, n) - 1]
        return float((lo + hi) / 2.0)
    return float(x[min(k + 1, n) - 1])


def _egen_statistic(fcn: str, x: np.ndarray, options: Dict[str, Any]) -> float:
    """One group's statistic of the sorted non-missing values ``x``."""
    n = x.size
    if fcn == "count":
        return float(n)
    if fcn in ("total", "sum"):
        if n == 0:
            return float("nan") if options.get("missing") else 0.0
        return float(x.sum())
    if n == 0:
        return float("nan")
    if fcn == "mean":
        return float(x.mean())
    if fcn == "median":
        return stata_percentile(x, 50.0)
    if fcn == "min":
        return float(x.min())
    if fcn == "max":
        return float(x.max())
    if fcn == "sd":
        return float(x.std(ddof=1)) if n > 1 else float("nan")
    if fcn == "pctile":
        p = options.get("p")
        return stata_percentile(x, 50.0 if p is None else float(p))
    if fcn in ("skew", "kurt"):
        dev = x - x.mean()
        m2 = float(np.mean(dev**2))
        if not m2 > 0:
            return float("nan")
        power = 3 if fcn == "skew" else 4
        return float(np.mean(dev**power) / m2 ** (power / 2))
    if fcn == "mad":
        return stata_percentile(np.sort(np.abs(x - stata_percentile(x, 50.0))), 50.0)
    if fcn == "mdev":
        return float(np.mean(np.abs(x - x.mean())))
    if fcn == "mode":
        values, counts = np.unique(x, return_counts=True)
        modes = values[counts == counts.max()]
        if len(modes) == 1 or options.get("minmode"):
            return float(modes[0])
        return float(modes[-1]) if options.get("maxmode") else float("nan")
    return stata_percentile(x, 75.0) - stata_percentile(x, 25.0)  # iqr


def egen_codes(frame: pd.DataFrame, keep_missing: bool) -> np.ndarray:
    """0, 1, ... for the distinct rows of ``frame`` in sorted order; -1 where
    a value is missing and missing values do not form groups."""
    frame = frame.copy()
    for name in frame.columns:
        col = frame[name]
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            # "" is the missing string: without `missing` it forms no group
            # (Stata 18: group(s) is missing and tag(s) 0 where s is ""),
            # with it "" is one group, sorted before the other strings
            text = col.astype(object).where(col.notna(), "")
            frame[name] = text if keep_missing else text.replace("", np.nan)
    if keep_missing:
        # a missing value sorts after every number, as in Stata
        return np.asarray(
            frame.groupby(list(frame.columns), sort=True, dropna=False).ngroup()
        )
    codes = frame.groupby(list(frame.columns), sort=True, dropna=True).ngroup()
    return np.asarray(codes.fillna(-1).astype(int))


def egen_compute(
    fcn: str,
    n: int,
    mask: np.ndarray,
    options: Dict[str, Any],
    *,
    x: Optional[np.ndarray] = None,
    code: Optional[np.ndarray] = None,
    frame: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """The values of ``egen fcn()`` on ``n`` rows (double precision).

    ``mask`` marks the rows ``if`` / ``in`` select; the others get missing
    (``tag``: 0). ``x`` is the evaluated expression (statistics, ``rank``,
    ``std``), ``code`` the group of each row (``by``), ``frame`` the
    variable list of ``group`` / ``tag`` and the row functions. ``options``
    use Stata's names: ``p``, ``missing``, ``minmode``, ``maxmode``,
    ``field``, ``track``, ``from``, ``to``, ``block``, ``values`` (a list
    of numbers), ``at`` (a list of numbers), ``group`` (an int), ``icodes``,
    ``label``.
    """
    value: Any = np.full(n, np.nan)
    if code is None:
        code = np.zeros(n, dtype=int)

    if fcn == "seq":
        start = float(options.get("from") or 1)
        repeat = int(float(options.get("block") or 1))
        stop = options.get("to")
        for g in np.unique(code):
            rows = np.flatnonzero((code == g) & mask)
            steps_ = np.arange(len(rows)) // max(repeat, 1)
            if stop is not None:
                span = int(float(stop) - start) + 1
                if span < 1:
                    raise _refuse(
                        "egen seq(): to() is below from()", "Give to() >= from()."
                    )
                steps_ = steps_ % span
            value[rows] = start + steps_
    elif fcn == "rank":
        assert x is not None
        used = mask & ~np.isnan(x)
        for g in np.unique(code):
            rows = np.flatnonzero((code == g) & used)
            v = x[rows]
            lower = (v[:, None] > v[None, :]).sum(axis=1)
            higher = (v[:, None] < v[None, :]).sum(axis=1)
            if options.get("field"):
                value[rows] = 1.0 + higher
            elif options.get("track"):
                value[rows] = 1.0 + lower
            else:
                ties = len(v) - lower - higher
                value[rows] = lower + (ties + 1) / 2.0
    elif fcn in EGEN_GROUP_STATS:
        assert x is not None
        used = mask & ~np.isnan(x)
        order = np.argsort(code, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(code[order]) != 0, True])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            rows = order[lo:hi]
            value[rows] = _egen_statistic(fcn, np.sort(x[rows][used[rows]]), options)
        value = np.where(mask, value, np.nan)
    elif fcn == "std":
        assert x is not None
        held = x[mask & ~np.isnan(x)]
        if held.size > 1 and held.std(ddof=1) > 0:
            value = np.where(mask, (x - held.mean()) / held.std(ddof=1), np.nan)
    elif fcn in ("group", "tag"):
        assert frame is not None
        gcode = egen_codes(frame, keep_missing=bool(options.get("missing")))
        selected = mask & (gcode >= 0)
        if fcn == "group":
            levels = np.unique(gcode[selected])
            rank_of = {int(c): i + 1 for i, c in enumerate(levels)}
            value = np.array(
                [rank_of[int(c)] if s else np.nan for c, s in zip(gcode, selected)],
                dtype=float,
            )
        else:
            value = np.zeros(n)
            positions = np.flatnonzero(selected)
            _, first = np.unique(gcode[positions], return_index=True)
            value[positions[first]] = 1.0
    else:  # row functions
        assert frame is not None
        value = _egen_row(fcn, frame, mask, options)
        value = np.where(mask, value, np.nan)
    return np.asarray(value, dtype=float)


def _egen_row(
    fcn: str, frame: pd.DataFrame, mask: np.ndarray, options: Dict[str, Any]
) -> np.ndarray:
    block = np.column_stack(
        [frame[c].to_numpy(dtype=float, na_value=np.nan) for c in frame.columns]
    )
    present = ~np.isnan(block)
    k = present.sum(axis=1)
    out: np.ndarray
    with np.errstate(invalid="ignore", divide="ignore"):
        if fcn == "rownonmiss":
            out = k.astype(float)
            return out
        if fcn == "rowmiss":
            out = (block.shape[1] - k).astype(float)
            return out
        if fcn in ("rowfirst", "rowlast"):
            order = block if fcn == "rowfirst" else block[:, ::-1]
            seen = ~np.isnan(order)
            first = seen.argmax(axis=1)
            return np.where(
                seen.any(axis=1), order[np.arange(len(order)), first], np.nan
            )
        if fcn in ("anycount", "anymatch"):
            wanted = list(options.get("values") or [])
            if not wanted:
                raise _refuse(
                    "egen: a list of numbers is expected",
                    "Give values(), e.g. values(1 3/5).",
                )
            hits = np.isin(block, wanted).sum(axis=1).astype(float)
            out = hits if fcn == "anycount" else (hits > 0).astype(float)
            return out
        if fcn == "cut":
            return _egen_cut(block, present, mask, options)
        if fcn == "rowtotal":
            value = np.where(present, block, 0.0).sum(axis=1)
            if options.get("missing"):
                value = np.where(k > 0, value, np.nan)
            out = value
            return out
        if fcn == "rowmean":
            return np.where(
                k > 0, np.where(present, block, 0.0).sum(axis=1) / k, np.nan
            )
        if fcn == "rowmin":
            return np.where(k > 0, np.where(present, block, np.inf).min(axis=1), np.nan)
        if fcn == "rowmax":
            return np.where(
                k > 0, np.where(present, block, -np.inf).max(axis=1), np.nan
            )
        # rowsd
        mean = np.where(present, block, 0.0).sum(axis=1) / np.maximum(k, 1)
        dev = np.where(present, block - mean[:, None], 0.0)
        return np.where(
            k > 1, np.sqrt((dev**2).sum(axis=1) / np.maximum(k - 1, 1)), np.nan
        )


def _egen_cut(
    block: np.ndarray, present: np.ndarray, mask: np.ndarray, options: Dict[str, Any]
) -> np.ndarray:
    if block.shape[1] != 1:
        raise _refuse("egen cut() takes one variable", "Give one variable to cut().")
    x = block[:, 0]
    codes = bool(options.get("icodes") or options.get("label"))
    if options.get("group"):
        # group(#): # groups of about equal frequency. The cuts are the
        # minimum, the # - 1 quantiles of `pctile x, nq(#)` and the maximum
        # plus one; the result is the group number from 0 (group() implies
        # icodes).
        if options.get("at"):
            raise _refuse(
                "egen cut(): at() and group() cannot be combined",
                "Give either at() or group().",
            )
        try:
            n_groups = int(float(str(options["group"])))
        except ValueError:
            n_groups = 0
        held = np.sort(x[present[:, 0] & mask])
        if n_groups < 2 or held.size == 0:
            raise _refuse(
                "egen cut(): group(#) needs an integer of 2 or more",
                "Give group(#) with # >= 2.",
            )
        inner = [
            stata_percentile(held, 100.0 * i / n_groups) for i in range(1, n_groups)
        ]
        at = np.array([held[0]] + inner + [held[-1] + 1.0])
        # ties in the quantiles leave empty groups, as in Stata
        at = np.maximum.accumulate(at)
        codes = True
    else:
        at = np.array(list(options.get("at") or []), dtype=float)
        if len(at) < 2 or np.any(np.diff(at) <= 0):
            raise _refuse(
                "egen cut() needs at(#, #, ...) in ascending order or group(#)",
                "Give at() with two or more increasing cut points, or group(#).",
            )
    where = np.searchsorted(at, x, side="right") - 1
    inside = present[:, 0] & (where >= 0) & (x < at[-1])
    where = np.clip(where, 0, len(at) - 1)
    picked = where.astype(float) if codes else at[where]
    return np.where(inside, picked, np.nan)


def egen(
    data: pd.DataFrame,
    fcn: str,
    x: Union[str, Sequence[str], pd.Series, np.ndarray, None] = None,
    *,
    by: Union[str, Sequence[str], None] = None,
    where: Union[pd.Series, np.ndarray, None] = None,
    p: Optional[float] = None,
    missing: bool = False,
    mode_ties: str = "missing",
    rank_ties: str = "average",
    start: float = 1.0,
    stop: Optional[float] = None,
    block: int = 1,
    values: Optional[Sequence[float]] = None,
    at: Optional[Sequence[float]] = None,
    n_groups: Optional[int] = None,
    icodes: bool = False,
) -> pd.Series:
    """Stata's ``egen``: a new variable from a function of the data.

    ``sp.egen(df, "mean", "wage", by="state")`` is ``egen m = mean(wage),
    by(state)``. The rules are those of ``[D] egen`` (checked against Stata
    18, ``tests/test_egen_public.py``): statistics skip missing values and a
    group with none gets missing (``count`` 0; ``total`` 0 unless
    ``missing=True``); rows not selected by ``where`` (Stata's ``if``) get
    missing (``tag`` 0); missing by-values form a group of their own. This
    is the implementation the ``egen`` step of ``sp.stata`` uses; there the
    result is stored in single precision unless the line says ``double``,
    here it is double.

    Parameters
    ----------
    data : pd.DataFrame
    fcn : str
        ``count mean median sd min max total sum pctile iqr skew kurt mad
        mdev mode`` (a statistic within the group), ``std``, ``rank``,
        ``seq``, ``group`` / ``tag`` (of a variable list), ``rowtotal
        rowmean rowmin rowmax rowsd rownonmiss rowmiss rowfirst rowlast
        anycount anymatch`` (of a variable list, row by row) or ``cut``.
    x : str, list of str, Series or array, optional
        The variable (statistics, ``std``, ``rank``, ``cut``) -- a column
        name or values aligned with the rows -- or the variable list
        (``group``, ``tag``, row functions). ``seq`` takes none.
    by : str or list of str, optional
        Groups (``by()``); not with ``std``, ``group``, ``tag`` or the row
        functions, as in Stata.
    where : bool Series or array, optional
        The rows to use (Stata's ``if``).
    p : float, optional
        Percentile for ``pctile`` (default 50).
    missing : bool, default False
        ``total`` / ``sum`` / ``rowtotal``: missing instead of 0 when there
        is no value; ``group`` / ``tag``: missing values form groups.
    mode_ties : {'missing', 'min', 'max'}, default 'missing'
        ``mode`` with several modes: missing, or ``minmode`` / ``maxmode``.
    rank_ties : {'average', 'field', 'track'}, default 'average'
        ``rank``: ties share their average rank; ``field`` counts 1 + the
        higher values (1 is the largest), ``track`` 1 + the lower values.
    start, stop, block : float, float, int
        ``seq``: from ``start`` (``from()``) to ``stop`` (``to()``, then
        again), each value ``block`` times.
    values : list of float, optional
        ``anycount`` / ``anymatch``: the values to look for.
    at : list of float, optional
        ``cut``: the cut points (the result is the left end of the interval
        ``[a_k, a_k+1)``; missing outside).
    n_groups : int, optional
        ``cut``: groups of about equal frequency (``group(#)``), coded 0, 1,
        ... .
    icodes : bool, default False
        ``cut``: code the intervals 0, 1, ... instead of their left ends.

    Returns
    -------
    pd.Series
        Aligned with ``data.index``, float64.

    Raises
    ------
    ColumnNotFound
        A named variable is not in ``data``.
    MethodIncompatibility
        An unknown function, an option the function does not take, ``by``
        with a function Stata refuses it for, a statistic of a text column,
        or bad ``cut`` / ``seq`` settings.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"g": [1, 1, 2, 2], "x": [1.0, np.nan, 3.0, 5.0]})
    >>> sp.egen(df, "mean", "x", by="g").tolist()
    [1.0, 1.0, 4.0, 4.0]
    >>> sp.egen(df, "count", "x", by="g").tolist()
    [1.0, 1.0, 2.0, 2.0]
    >>> sp.egen(df, "rank", "x").tolist()
    [1.0, nan, 2.0, 3.0]
    >>> sp.egen(df, "rowtotal", ["g", "x"]).tolist()
    [2.0, 1.0, 5.0, 7.0]

    References
    ----------
    StataCorp. Stata 18 Base Reference Manual, ``[D] egen``.
    """
    if not isinstance(fcn, str) or fcn.lower() not in EGEN_FUNCTIONS:
        raise _refuse(
            f"egen function {fcn}() is not implemented",
            f"Use one of: {', '.join(EGEN_FUNCTIONS)}.",
        )
    fcn = fcn.lower()
    n = len(data)
    options: Dict[str, Any] = {}
    given: List[str] = []

    def opt(stata_name: str, value: Any, default: Any) -> None:
        if value is not None and value is not False and value != default:
            options[stata_name] = value
            given.append(stata_name)

    opt("p", p, None)
    opt("missing", missing, False)
    if mode_ties not in ("missing", "min", "max"):
        raise _refuse(f"mode_ties={mode_ties!r}", "Use 'missing', 'min' or 'max'.")
    opt("minmode", mode_ties == "min", False)
    opt("maxmode", mode_ties == "max", False)
    if rank_ties not in ("average", "field", "track"):
        raise _refuse(f"rank_ties={rank_ties!r}", "Use 'average', 'field' or 'track'.")
    opt("field", rank_ties == "field", False)
    opt("track", rank_ties == "track", False)
    opt("from", float(start), 1.0)
    opt("to", stop, None)
    opt("block", int(block), 1)
    opt("values", list(values) if values is not None else None, None)
    opt("at", list(at) if at is not None else None, None)
    opt("group", n_groups, None)
    opt("icodes", icodes, False)
    extra = sorted(set(given) - EGEN_OPTIONS.get(fcn, frozenset()))
    if extra:
        raise _refuse(
            f"egen {fcn}(): option(s) {extra} are not implemented",
            f"egen {fcn}() takes {sorted(EGEN_OPTIONS.get(fcn, ())) or 'no options'}.",
        )
    if fcn == "seq":
        options.setdefault("from", 1.0)

    by_cols = [by] if isinstance(by, str) else list(by or [])
    if by_cols and fcn in EGEN_NO_BY:
        raise _refuse(
            f"egen {fcn} may not be combined with by",
            "Compute it on the whole data (or per group with groupby).",
        )
    check_columns(data, {"by": by_cols or None}, function="sp.egen")

    mask: np.ndarray
    if where is None:
        mask = np.ones(n, dtype=bool)
    else:
        w = np.asarray(where)
        if w.shape != (n,):
            raise _refuse(
                f"where= has {w.size} values for {n} rows",
                "Pass a boolean Series aligned with the data.",
            )
        mask = np.asarray(pd.Series(w).fillna(False).astype(bool))
    code = egen_codes(data[by_cols], keep_missing=True) if by_cols else None

    if fcn in ("group", "tag") or fcn in EGEN_ROW:
        names = [x] if isinstance(x, str) else list(x if x is not None else [])
        if not names or not all(isinstance(c, str) for c in names):
            raise _refuse(
                f"egen {fcn}() needs a variable list",
                f'Pass the column names, e.g. sp.egen(df, "{fcn}", ["a", "b"]).',
            )
        missing_cols = [c for c in names if c not in data.columns]
        if missing_cols:
            raise ColumnNotFound.from_missing(
                missing_cols, list(data.columns), context="sp.egen"
            )
        frame = data[names]
        if fcn in EGEN_ROW and any(
            not pd.api.types.is_numeric_dtype(frame[c]) for c in names
        ):
            raise _refuse(
                f"egen {fcn}() of a string variable",
                "Row functions take numeric variables.",
            )
        out = egen_compute(fcn, n, mask, options, frame=frame)
    elif fcn == "seq":
        if x is not None:
            raise _refuse("egen seq() takes no argument", "Drop x.")
        out = egen_compute(fcn, n, mask, options, code=code)
    else:
        if x is None:
            raise _refuse(
                f"egen {fcn}() needs an expression",
                f'Pass a column: sp.egen(df, "{fcn}", "x").',
            )
        if isinstance(x, str):
            if x not in data.columns:
                raise ColumnNotFound.from_missing(
                    {"x": x}, list(data.columns), context="sp.egen"
                )
            series = data[x]
        else:
            series = pd.Series(np.asarray(x))
            if len(series) != n:
                raise _refuse(
                    f"x has {len(series)} values for {n} rows",
                    "Pass values aligned with the data.",
                )
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(
            series
        ):
            if pd.api.types.is_bool_dtype(series):
                series = series.astype(float)
            else:
                raise _refuse(
                    f"egen {fcn}() of a string expression",
                    "Pass a numeric column.",
                )
        xv = series.to_numpy(dtype=float, na_value=np.nan)
        out = egen_compute(fcn, n, mask, options, x=xv, code=code)
    label = x if isinstance(x, str) else None
    return pd.Series(out, index=data.index, name=f"{fcn}({label})" if label else fcn)
