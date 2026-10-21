"""Panel / time-series declarations and Stata's lag operators.

``sp.xtset`` declares the panel and time variables of a DataFrame, as
Stata's ``xtset id time`` / ``tsset time`` do; ``sp.lag`` / ``sp.lead`` /
``sp.diff`` are Stata's ``L.`` / ``F.`` / ``D.`` operators.

Stata defines a lag by the time variable, not by the row above: ``L.x``
at time ``t`` is ``x`` at ``t - delta`` in the same panel, and is missing
when that period is absent ([U] 11.4.4 Time-series varlists). ``shift(1)``
on the rows gives a different answer wherever a period is missing, the
rows are unsorted, or the data are a panel.

This module is the one implementation of those operators: ``sp.stata``
(``L.x`` in a command line), the formula layer (``y ~ L.x``) and the
functions here all read the declaration and compute the lag through
:class:`TimeClock`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import ColumnNotFound, MethodIncompatibility

__all__ = [
    "PanelDeclaration",
    "TimeClock",
    "XTSET_ATTR",
    "apply_ts_ops",
    "check_delta",
    "diff",
    "get_xtset",
    "lag",
    "lead",
    "parse_ts_ops",
    "rewrite_ts_terms",
    "ts_column_name",
    "TS_TERM",
    "xtset",
]

#: key of ``DataFrame.attrs`` that holds the declaration of :func:`xtset`
XTSET_ATTR = "statspai_xtset"

_OPS = r"(?:[LlFfDd]\d*)+"
# stacked operators may be written together or each with its dot:
# LD.x, L.D.x and F.D.x (Stata 18 reads them all)
_CHAIN = rf"{_OPS}(?:\.{_OPS})*"
#: one operator term: ``L.x``, ``L2D.x``, ``L.D.x``, ``L(1/3).x``,
#: ``L(1/2).D.x``
TS_TERM = re.compile(
    rf"(?<![\w.])(?:(?P<ops>{_CHAIN})|(?P<kind>[LlFf])\((?P<lo>\d+)/(?P<hi>\d+)\)"
    rf"(?:\.?(?P<tail>{_CHAIN}))?)"
    r"\.(?P<var>[A-Za-z_]\w*)"
)
_ONE = re.compile(r"([LlFfDd])(\d*)")

_HINT_TIME = (
    "Give the time variable as whole numbers of periods (Stata dates are "
    "integers) or a pandas Period column."
)


def parse_ts_ops(ops: str) -> List[Tuple[str, int]]:
    """``'L2D'`` -> ``[('L', 2), ('D', 1)]``: applied right to left."""
    return [(k.upper(), int(n) if n else 1) for k, n in _ONE.findall(ops)]


def ts_column_name(var: str, ops: List[Tuple[str, int]]) -> str:
    """The column an operator term becomes: ``L2.x`` -> ``x_L2``."""
    return var + "_" + "".join(f"{k}{n}" for k, n in ops)


class TimeClock:
    """Looks a series up at another period of the same panel.

    Parameters
    ----------
    data : DataFrame
    unit : str or None
        The panel variable (``None``: one time series).
    time : str
        The time variable: whole numbers of periods or a pandas Period.
    delta : float, default 1
        The period of one step (``tsset t, delta(5)``: ``L.x`` at ``t`` is
        ``x`` at ``t - 5``).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit: Optional[str],
        time: str,
        delta: float = 1.0,
    ) -> None:
        self.delta = float(delta)
        if time not in data.columns:
            raise ColumnNotFound(
                f"the time variable {time!r} is not in the data",
                recovery_hint="Name a column of the data as the time variable.",
                diagnostics={"missing_columns": [time]},
            )
        if unit is not None and unit not in data.columns:
            raise ColumnNotFound(
                f"the panel variable {unit!r} is not in the data",
                recovery_hint="Name a column of the data as the panel variable.",
                diagnostics={"missing_columns": [unit]},
            )
        col = data[time]
        if isinstance(col.dtype, pd.PeriodDtype):
            t = col.map(lambda p: p.ordinal if p is not pd.NaT else np.nan)
        elif pd.api.types.is_numeric_dtype(col):
            t = col
        else:
            raise MethodIncompatibility(
                f"the time variable {time!r} must be a number of periods or a "
                "pandas Period; a datetime has no unit step. Stata dates are "
                "integers",
                recovery_hint="Convert it, e.g. df[t].dt.to_period('M') or "
                "df[t].dt.year.",
            )
        t = t.to_numpy(dtype=float, na_value=np.nan)
        if np.isnan(t).any() or not np.all(t == np.round(t)):
            raise MethodIncompatibility(
                f"the time variable {time!r} has missing or non-integer values",
                recovery_hint="Drop the rows with a missing time value. " + _HINT_TIME,
            )
        keys = pd.DataFrame({"u": 0 if unit is None else data[unit].to_numpy(), "t": t})
        if keys.duplicated().any():
            raise MethodIncompatibility(
                "repeated time values" + (" within a panel" if unit else ""),
                recovery_hint="Each panel may hold one row per period; "
                "aggregate the duplicates (sp.collapse) or name the panel "
                "variable.",
            )
        self._keys = keys

    def at(self, values: np.ndarray, offset: int) -> np.ndarray:
        """``values`` as observed ``offset`` periods later (negative: earlier)."""
        source = self._keys.assign(t=self._keys["t"] - offset * self.delta, v=values)
        merged = self._keys.merge(source, how="left", on=["u", "t"])
        return np.asarray(merged["v"], dtype=float)


def apply_ts_ops(
    clock: TimeClock, values: np.ndarray, ops: List[Tuple[str, int]]
) -> np.ndarray:
    """``values`` under the operators ``ops`` (right to left, as written)."""
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


def check_delta(
    data: pd.DataFrame, unit: Optional[str], time: str, delta: float
) -> None:
    """``tsset t, delta(#)``: Stata refuses data with two periods closer
    than one step (r(451))."""
    if not delta > 0:
        raise MethodIncompatibility(
            "delta() must be a positive number",
            recovery_hint="Pass the length of one period, e.g. delta=1.",
        )
    if time not in data.columns:
        return
    t = pd.to_numeric(data[time], errors="coerce")
    keys = data[unit] if unit is not None and unit in data.columns else 0
    frame = pd.DataFrame({"u": keys, "t": t}).dropna().sort_values(["u", "t"])
    gaps = frame.groupby("u")["t"].diff().to_numpy(dtype=float)
    if np.any((gaps > 0) & (gaps < delta)):
        raise MethodIncompatibility(
            "time values with period less than delta() found (Stata r(451))",
            recovery_hint="Use a delta no larger than the smallest step "
            "between two periods of a panel.",
        )


def rewrite_ts_terms(
    line: str,
    data: Optional[pd.DataFrame],
    panel: Tuple[Optional[str], Optional[str]],
    made: Optional[Dict[str, tuple]] = None,
    delta: float = 1.0,
    sep: str = " ",
) -> Tuple[str, Dict[str, np.ndarray]]:
    """Replace operator terms in ``line`` by column names.

    Returns the rewritten text and the columns to add (name -> values), in
    the order they first appear. ``data`` may be ``None`` for a line that
    only names coefficients; no column is computed then.

    ``made`` records the columns this function computed (name ->
    (variable, operators)). A column listed there is computed again each
    time it is named: the variable, the sample or the time declaration
    may have changed since. A column of ``data`` that already has the
    name and is not listed in ``made`` is used as it is. ``sep`` joins the
    columns of a lag list (``L(1/2).x``): a blank in a Stata varlist,
    ``" + "`` in a formula.
    """
    unit, time = panel
    if time is None:
        raise MethodIncompatibility(
            "a time-series operator needs the time variable: put `tsset "
            "time` or `xtset id time` before it",
            recovery_hint="Declare the panel with sp.xtset(data, id=..., "
            "time=...) or name time= in the call.",
        )
    clock = None if data is None else TimeClock(data, unit, time, delta)
    new_columns: Dict[str, np.ndarray] = {}

    def column(var: str, ops: List[Tuple[str, int]]) -> Optional[str]:
        name = ts_column_name(var, ops)
        if data is None or clock is None:
            return name
        if var not in data.columns:
            # not a variable (a file name such as d.csv): leave the text alone
            return None
        source = data[var]
        if not pd.api.types.is_numeric_dtype(source):
            raise MethodIncompatibility(
                f"{var!r} is not numeric",
                recovery_hint="Time-series operators apply to numeric "
                "variables only.",
            )
        if name in new_columns:
            return name
        if name in data.columns and (made is None or name not in made):
            return name  # a column of the caller's data with that name
        values = source.to_numpy(dtype=float, na_value=np.nan)
        new_columns[name] = apply_ts_ops(clock, values, ops)
        if made is not None:
            made[name] = (var, ops)
        return name

    def replace(m: "re.Match[str]") -> str:
        var = m.group("var")
        if m.group("ops") is not None:
            return column(var, parse_ts_ops(m.group("ops"))) or m.group(0)
        lo, hi = int(m.group("lo")), int(m.group("hi"))
        if hi < lo:
            raise MethodIncompatibility(
                f"empty lag list in {m.group(0)!r}",
                recovery_hint="Write the range low/high, e.g. L(1/3).x.",
            )
        kind = m.group("kind").upper()
        # L(1/3).D.x: lags one to three of the first difference
        tail = parse_ts_ops(m.group("tail")) if m.group("tail") else []
        terms: List[str] = []
        for k in range(lo, hi + 1):
            ops = ([] if k == 0 else [(kind, k)]) + tail
            term = var if not ops else column(var, ops)
            if term is None:
                return m.group(0)
            terms.append(term)
        return sep.join(terms)

    return TS_TERM.sub(replace, line), new_columns


# ---------------------------------------------------------------------------
# The public declaration and operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelDeclaration:
    """What ``sp.xtset`` declared, with Stata's description of the panel.

    Attributes
    ----------
    id : str or None
        The panel variable (``None`` after a time-series declaration).
    time : str or None
        The time variable (``None`` when only the panel is declared).
    delta : float
        The length of one period.
    balanced : str or None
        ``'strongly balanced'`` (every panel has the same periods),
        ``'weakly balanced'`` (the same number of periods, not the same
        ones) or ``'unbalanced'``, as Stata's ``r(balanced)``; ``None``
        without a panel or a time variable.
    gaps : bool
        Whether some panel skips a period (Stata ``r(gaps)``).
    n_units : int or None
        The number of panels.
    t_min, t_max : float or None
        The first and last period (``r(tmin)`` / ``r(tmax)``).
    """

    id: Optional[str]
    time: Optional[str]
    delta: float = 1.0
    balanced: Optional[str] = None
    gaps: bool = False
    n_units: Optional[int] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "time": self.time,
            "delta": self.delta,
            "balanced": self.balanced,
            "gaps": self.gaps,
            "n_units": self.n_units,
            "t_min": self.t_min,
            "t_max": self.t_max,
        }

    def summary(self) -> str:
        lines = []
        if self.id is not None:
            lines.append(
                f"Panel variable: {self.id}"
                + (f" ({self.balanced})" if self.balanced else "")
            )
        if self.time is not None:
            span = ""
            if self.t_min is not None:
                span = f", {self.t_min:g} to {self.t_max:g}"
                if self.gaps:
                    span += ", but with gaps"
            lines.append(f" Time variable: {self.time}{span}")
            unit = "unit" if self.delta == 1 else "units"
            lines.append(f"         Delta: {self.delta:g} {unit}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return self.summary()


def _describe(
    data: pd.DataFrame, id: Optional[str], time: Optional[str], delta: float
) -> PanelDeclaration:
    if time is None:
        n = int(data[id].nunique()) if id is not None else None
        return PanelDeclaration(id=id, time=None, delta=delta, n_units=n)
    # Stata's xtset accepts rows with a missing period (they take no part
    # in the description); a lag needs every period and refuses them
    observed = data
    if not isinstance(data[time].dtype, pd.PeriodDtype):
        observed = data.loc[pd.to_numeric(data[time], errors="coerce").notna()]
    clock = TimeClock(observed, id, time, delta)  # validates the time variable
    keys = clock._keys
    t_min, t_max = float(keys["t"].min()), float(keys["t"].max())
    steps = keys.sort_values(["u", "t"]).groupby("u")["t"].diff()
    gaps = bool((steps > delta).any())
    if id is None:
        return PanelDeclaration(
            id=None, time=time, delta=delta, gaps=gaps, t_min=t_min, t_max=t_max
        )
    sets = keys.groupby("u")["t"].apply(lambda s: tuple(sorted(s)))
    if sets.nunique() == 1:
        balanced = "strongly balanced"
    elif sets.map(len).nunique() == 1:
        balanced = "weakly balanced"
    else:
        balanced = "unbalanced"
    return PanelDeclaration(
        id=id,
        time=time,
        delta=delta,
        balanced=balanced,
        gaps=gaps,
        n_units=int(len(sets)),
        t_min=t_min,
        t_max=t_max,
    )


def xtset(
    data: pd.DataFrame,
    id: Optional[str] = None,
    time: Optional[str] = None,
    delta: float = 1.0,
) -> PanelDeclaration:
    """Declare the panel and time variables of a DataFrame (Stata ``xtset``).

    ``sp.xtset(df, id="firm", time="year")`` is Stata's ``xtset firm
    year``; ``sp.xtset(df, time="t")`` is ``tsset t``. The declaration is
    stored in ``data.attrs`` (in place, as Stata changes the dataset in
    memory) and is read by :func:`lag` / :func:`lead` / :func:`diff` and
    by formulas that use ``L.x`` / ``F.x`` / ``D.x``. Called with neither
    ``id`` nor ``time`` it returns the current declaration.

    Parameters
    ----------
    data : DataFrame
        The data. Only ``data.attrs`` is changed; the rows keep their
        order (Stata sorts by panel and time; nothing here depends on the
        order).
    id : str, optional
        The panel variable.
    time : str, optional
        The time variable: whole numbers of periods (Stata dates are
        integers) or a pandas ``Period`` column.
    delta : float, default 1
        The length of one period, Stata's ``delta()``: with ``delta=5``,
        ``L.x`` at ``t`` is ``x`` at ``t - 5``.

    Returns
    -------
    PanelDeclaration
        The variables, and Stata's description of the panel: balance
        (``r(balanced)``), whether some panel skips a period
        (``r(gaps)``), the number of panels, the first and last period.

    Raises
    ------
    MethodIncompatibility
        When the time variable has missing or non-integer values, repeats
        a period within a panel (Stata r(451)), is a datetime, or has two
        periods closer than ``delta`` (Stata r(451)); or when nothing has
        been declared and no variable is given.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"id": [1, 1, 1, 2, 2], "t": [1, 2, 4, 1, 2],
    ...                    "x": [1.0, 4.0, 9.0, 2.0, 3.0]})
    >>> decl = sp.xtset(df, id="id", time="t")
    >>> decl.balanced, decl.gaps
    ('unbalanced', True)
    >>> sp.lag(df, "x").tolist()   # t = 4 has no t = 3 before it
    [nan, 1.0, nan, nan, 2.0]

    References
    ----------
    StataCorp. Stata 18 Base Reference Manual, ``[XT] xtset`` and
    ``[TS] tsset``.
    """
    if id is None and time is None:
        current = get_xtset(data)
        if current is None:
            raise MethodIncompatibility(
                "no panel or time variable has been declared",
                recovery_hint="Call sp.xtset(data, id='unit', time='period').",
            )
        return _describe(data, current["id"], current["time"], current["delta"])
    for label, name in (("id", id), ("time", time)):
        if name is not None and name not in data.columns:
            raise ColumnNotFound.from_missing(
                {label: name}, list(data.columns), context="sp.xtset"
            )
    if time is not None:
        check_delta(data, id, time, float(delta))
    decl = _describe(data, id, time, float(delta))
    data.attrs[XTSET_ATTR] = {"id": id, "time": time, "delta": float(delta)}
    return decl


def get_xtset(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """The declaration ``sp.xtset`` stored on ``data`` (``None`` if none)."""
    found = getattr(data, "attrs", {}).get(XTSET_ATTR)
    return dict(found) if found else None


def _panel_of(
    data: pd.DataFrame,
    id: Optional[str],
    time: Optional[str],
    delta: Optional[float],
    function: str,
) -> Tuple[Optional[str], str, float]:
    declared = get_xtset(data)
    if time is None:
        if declared is None or declared.get("time") is None:
            raise MethodIncompatibility(
                f"sp.{function}: no time variable. A Stata lag is read from "
                "the time variable, not from the row above",
                recovery_hint="Pass time= (and id= for a panel), or declare "
                "them once with sp.xtset(data, id=..., time=...).",
            )
        time = declared["time"]
        if id is None:
            id = declared.get("id")
        if delta is None:
            delta = declared.get("delta", 1.0)
    for label, name in (("id", id), ("time", time)):
        if name is not None and name not in data.columns:
            raise ColumnNotFound.from_missing(
                {label: name}, list(data.columns), context=f"sp.{function}"
            )
    return id, time, float(1.0 if delta is None else delta)


def _operate(
    data: pd.DataFrame,
    var: Union[str, Sequence[str]],
    ops_of: Callable[[int, str], Tuple[str, List[Tuple[str, int]]]],
    orders: Union[int, Sequence[int]],
    id: Optional[str],
    time: Optional[str],
    delta: Optional[float],
    function: str,
) -> Union[pd.Series, pd.DataFrame]:
    unit, tvar, step = _panel_of(data, id, time, delta, function)
    names = [var] if isinstance(var, str) else list(var)
    missing = [v for v in names if v not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(
            missing, list(data.columns), context=f"sp.{function}"
        )
    many = not isinstance(orders, (int, np.integer))
    order_list = [int(k) for k in (list(orders) if many else [orders])]  # type: ignore
    clock = TimeClock(data, unit, tvar, step)
    out: Dict[str, np.ndarray] = {}
    for v in names:
        if not pd.api.types.is_numeric_dtype(data[v]):
            raise MethodIncompatibility(
                f"sp.{function}: {v!r} is not numeric",
                recovery_hint="Time-series operators apply to numeric "
                "variables only.",
            )
        values = data[v].to_numpy(dtype=float, na_value=np.nan)
        for k in order_list:
            label, ops = ops_of(k, v)
            out[label] = values.copy() if not ops else apply_ts_ops(clock, values, ops)
    if isinstance(var, str) and not many:
        ((label, values),) = out.items()
        return pd.Series(values, index=data.index, name=label)
    return pd.DataFrame(out, index=data.index)


def _check_order(k: Union[int, Sequence[int]], function: str, low: int) -> None:
    ks = [k] if isinstance(k, (int, np.integer)) else list(k)
    if not ks or any(not isinstance(j, (int, np.integer)) or j < low for j in ks):
        raise MethodIncompatibility(
            f"sp.{function}: the order must be an integer >= {low}",
            recovery_hint=f"Pass n={max(low, 1)} for the first "
            + ("difference" if function == "diff" else "period")
            + ", or a list such as [1, 2, 3].",
        )


def lag(
    data: pd.DataFrame,
    var: Union[str, Sequence[str]],
    n: Union[int, Sequence[int]] = 1,
    *,
    id: Optional[str] = None,
    time: Optional[str] = None,
    delta: Optional[float] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Stata's lag operator ``Ln.x``, read from the time variable.

    ``L.x`` at time ``t`` is ``x`` at ``t - delta`` in the same panel, and
    missing when that period is not in the data -- not the row above.

    Parameters
    ----------
    data : DataFrame
    var : str or list of str
        The variable(s) to lag.
    n : int or list of int, default 1
        The lag order; ``0`` returns the variable itself. A list such as
        ``[1, 2, 3]`` is Stata's ``L(1/3).x``.
    id, time : str, optional
        The panel and time variables. Default: the declaration of
        :func:`xtset`.
    delta : float, optional
        The length of one period (default: the declared one, else 1).

    Returns
    -------
    Series or DataFrame
        Aligned to ``data.index``, named ``L.x`` / ``L2.x`` (``x`` for
        order 0). One variable and one order give a Series.

    Raises
    ------
    MethodIncompatibility
        No time variable given or declared; a time variable with missing,
        non-integer or repeated values; a non-numeric variable.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"t": [1, 2, 3, 5], "x": [10.0, 12.0, 15.0, 19.0]})
    >>> sp.lag(df, "x", time="t").tolist()
    [nan, 10.0, 12.0, nan]
    >>> list(sp.lag(df, "x", n=[1, 2], time="t").columns)
    ['L.x', 'L2.x']

    References
    ----------
    StataCorp. Stata 18 User's Guide, ``[U] 11.4.4 Time-series varlists``.
    """
    _check_order(n, "lag", 0)

    def ops_of(k: int, v: str) -> Tuple[str, List[Tuple[str, int]]]:
        return (f"L{'' if k == 1 else k}.{v}" if k else v), ([("L", k)] if k else [])

    return _operate(data, var, ops_of, n, id, time, delta, "lag")


def lead(
    data: pd.DataFrame,
    var: Union[str, Sequence[str]],
    n: Union[int, Sequence[int]] = 1,
    *,
    id: Optional[str] = None,
    time: Optional[str] = None,
    delta: Optional[float] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Stata's lead operator ``Fn.x``, read from the time variable.

    ``F.x`` at time ``t`` is ``x`` at ``t + delta`` in the same panel, and
    missing when that period is not in the data.

    Parameters
    ----------
    data : DataFrame
    var : str or list of str
    n : int or list of int, default 1
        The lead order (``0``: the variable itself).
    id, time : str, optional
        Default: the declaration of :func:`xtset`.
    delta : float, optional

    Returns
    -------
    Series or DataFrame
        Named ``F.x`` / ``F2.x``.

    Raises
    ------
    MethodIncompatibility
        As :func:`lag`.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"t": [1, 2, 3, 5], "x": [10.0, 12.0, 15.0, 19.0]})
    >>> sp.lead(df, "x", time="t").tolist()
    [12.0, 15.0, nan, nan]
    """
    _check_order(n, "lead", 0)

    def ops_of(k: int, v: str) -> Tuple[str, List[Tuple[str, int]]]:
        return (f"F{'' if k == 1 else k}.{v}" if k else v), ([("F", k)] if k else [])

    return _operate(data, var, ops_of, n, id, time, delta, "lead")


def diff(
    data: pd.DataFrame,
    var: Union[str, Sequence[str]],
    n: Union[int, Sequence[int]] = 1,
    *,
    id: Optional[str] = None,
    time: Optional[str] = None,
    delta: Optional[float] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Stata's difference operator ``Dn.x``, read from the time variable.

    ``D.x`` is ``x - L.x``; ``D2.x`` is the difference of the difference
    (``x - 2 L.x + L2.x``), not ``x - L2.x``. Missing wherever a period it
    needs is absent.

    Parameters
    ----------
    data : DataFrame
    var : str or list of str
    n : int or list of int, default 1
        The order of the difference (``0``: the variable itself).
    id, time : str, optional
        Default: the declaration of :func:`xtset`.
    delta : float, optional

    Returns
    -------
    Series or DataFrame
        Named ``D.x`` / ``D2.x``.

    Raises
    ------
    MethodIncompatibility
        As :func:`lag`.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"t": [1, 2, 3, 5], "x": [10.0, 12.0, 15.0, 19.0]})
    >>> sp.diff(df, "x", time="t").tolist()
    [nan, 2.0, 3.0, nan]
    """
    _check_order(n, "diff", 0)

    def ops_of(k: int, v: str) -> Tuple[str, List[Tuple[str, int]]]:
        return (f"D{'' if k == 1 else k}.{v}" if k else v), ([("D", k)] if k else [])

    return _operate(data, var, ops_of, n, id, time, delta, "diff")
