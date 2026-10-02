"""Evaluate a Stata expression on a DataFrame, with Stata's missing-value rules.

``sp.stata`` uses this for ``if`` qualifiers and for ``generate`` /
``replace``. A pandas ``query`` would get the common case wrong without
saying so, because the two languages disagree about missing values:

* A numeric missing value is larger than every number, so ``x > 0`` is
  **true** where ``x`` is missing (pandas: false) and ``x < .`` is the usual
  way to write "x is observed".
* ``x == .`` is true where ``x`` is missing (``NaN == NaN`` is false).
* A missing value is "true" (it is not zero), so ``if x`` keeps it.
* Arithmetic on a missing value is missing, as is division by zero and the
  logarithm or square root of a negative number.

The grammar is a recursive-descent parser over a closed set of operators and
functions. Nothing is passed to ``eval``. Anything outside the set (a macro,
``e(sample)``, ``_b[x]``, a time-series operator, a string function, an
extended missing value) raises :class:`StataExprError`, and ``sp.stata``
refuses to run the command rather than guess.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["StataExprError", "evaluate", "sample_mask", "in_range_mask"]


class StataExprError(ValueError):
    """The expression uses something this evaluator does not implement."""


_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
      | (?P<str>"[^"]*")
      | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<op>==|!=|~=|>=|<=|[<>&|!~+\-*/^(),\[\]=.])
    )""",
    re.X,
)

Value = Any  # float | str | numpy array (float64 or object)


def _tokenise(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise StataExprError(f"cannot read {text[pos:pos + 12]!r}")
        kind = m.lastgroup or ""
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


# ---------------------------------------------------------------- helpers
def _is_str(v: Value) -> bool:
    if isinstance(v, str):
        return True
    return isinstance(v, np.ndarray) and v.dtype == object


def _num(v: Value, what: str) -> Any:
    if _is_str(v):
        raise StataExprError(f"{what} needs a number, got a string")
    return v


def _miss_high(v: Any) -> Any:
    """Missing sorts above every number."""
    return np.where(np.isnan(v), np.inf, v)


def _flag(cond: Any) -> Any:
    return np.asarray(cond, dtype=float)


def _truth(v: Value) -> Any:
    """Non-zero is true, and missing is not zero."""
    v = _num(v, "a logical operand")
    return np.asarray(v != 0) | np.isnan(v)


def _clean(v: Any) -> Any:
    """Infinity is not a Stata value: overflow and x/0 are missing."""
    v = np.asarray(v, dtype=float)
    return np.where(np.isfinite(v), v, np.nan)


def _compare(op: str, a: Value, b: Value) -> Any:
    if _is_str(a) or _is_str(b):
        if not (_is_str(a) and _is_str(b)):
            raise StataExprError("a string is compared with a number")
        if op not in ("==", "!="):
            raise StataExprError("strings can only be compared with == or !=")
        same = np.asarray(a == b, dtype=bool)
        return _flag(same if op == "==" else ~same)
    x, y = _miss_high(a), _miss_high(b)
    return _flag(
        {
            "==": lambda: x == y,
            "!=": lambda: x != y,
            ">": lambda: x > y,
            ">=": lambda: x >= y,
            "<": lambda: x < y,
            "<=": lambda: x <= y,
        }[op]()
    )


def _unary(fn: Callable[[Any], Any]) -> Callable[..., Any]:
    def call(*args: Value) -> Any:
        if len(args) != 1:
            raise StataExprError("this function takes one argument")
        with np.errstate(all="ignore"):
            return _clean(fn(np.asarray(_num(args[0], "a function"), dtype=float)))

    return call


def _f_round(*args: Value) -> Any:
    if len(args) not in (1, 2):
        raise StataExprError("round() takes one or two arguments")
    x = np.asarray(_num(args[0], "round()"), dtype=float)
    unit = 1.0 if len(args) == 1 else np.asarray(_num(args[1], "round()"), dtype=float)
    with np.errstate(all="ignore"):
        # halves round away from zero, unlike numpy's round-half-even
        scaled = x / unit
        return _clean(np.sign(scaled) * np.floor(np.abs(scaled) + 0.5) * unit)


def _extreme(reducer: Callable[..., Any]) -> Callable[..., Any]:
    def call(*args: Value) -> Any:
        if not args:
            raise StataExprError("min() / max() need an argument")
        cols = np.broadcast_arrays(
            *[np.asarray(_num(a, "min() / max()"), dtype=float) for a in args]
        )
        stacked = np.stack(cols)
        all_missing = np.isnan(stacked).all(axis=0)
        with np.errstate(all="ignore"):
            # missing arguments are ignored unless every one is missing
            out = reducer(np.where(all_missing, 0.0, stacked), axis=0)
        return np.where(all_missing, np.nan, out)

    return call


def _f_missing(*args: Value) -> Any:
    if not args:
        raise StataExprError("missing() needs an argument")
    flags = []
    for a in args:
        if _is_str(a):
            flags.append(np.asarray(a == "", dtype=bool))
        else:
            flags.append(np.isnan(np.asarray(a, dtype=float)))
    return _flag(np.logical_or.reduce(np.broadcast_arrays(*flags)))


def _f_inlist(*args: Value) -> Any:
    if len(args) < 2:
        raise StataExprError("inlist() needs a value and a list")
    hits = [_compare("==", args[0], other) != 0 for other in args[1:]]
    return _flag(np.logical_or.reduce(np.broadcast_arrays(*hits)))


def _f_inrange(*args: Value) -> Any:
    if len(args) != 3:
        raise StataExprError("inrange() takes three arguments")
    z, lo, hi = (np.asarray(_num(a, "inrange()"), dtype=float) for a in args)
    lo = np.where(np.isnan(lo), -np.inf, lo)
    hi = np.where(np.isnan(hi), np.inf, hi)
    with np.errstate(invalid="ignore"):
        inside = (z >= lo) & (z <= hi)
    return _flag(inside & ~np.isnan(z))


def _f_cond(*args: Value) -> Any:
    if len(args) not in (3, 4):
        raise StataExprError("cond() takes three or four arguments")
    test = np.asarray(_num(args[0], "cond()"), dtype=float)
    yes, no = args[1], args[2]
    if _is_str(yes) or _is_str(no):
        raise StataExprError("cond() with string results is not implemented")
    out = np.where(_truth(test), yes, no)
    if len(args) == 4:
        # with a fourth argument a missing test value is its own case
        out = np.where(np.isnan(test), _num(args[3], "cond()"), out)
    return np.asarray(out, dtype=float)


def _tail(dist: Any, n_shape: int, upper: bool) -> Callable[..., Any]:
    """Distribution functions whose leading arguments are degrees of freedom."""

    def call(*args: Value) -> Any:
        if len(args) != n_shape + 1:
            raise StataExprError(f"this function takes {n_shape + 1} arguments")
        vals = [
            np.asarray(_num(a, "a distribution function"), dtype=float) for a in args
        ]
        with np.errstate(all="ignore"):
            fn = dist.sf if upper else dist.cdf
            return _clean(fn(vals[-1], *vals[:-1]))

    return call


def _f_invttail(*args: Value) -> Any:
    if len(args) != 2:
        raise StataExprError("invttail() takes two arguments")
    dof, p = (np.asarray(_num(a, "invttail()"), dtype=float) for a in args)
    with np.errstate(all="ignore"):
        return _clean(stats.t.isf(p, dof))


def _f_mod(*args: Value) -> Any:
    if len(args) != 2:
        raise StataExprError("mod() takes two arguments")
    x, y = (np.asarray(_num(a, "mod()"), dtype=float) for a in args)
    with np.errstate(all="ignore"):
        return _clean(np.where(y > 0, x - y * np.floor(x / y), np.nan))


_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "ln": _unary(np.log),
    "log": _unary(np.log),
    "log10": _unary(np.log10),
    "exp": _unary(np.exp),
    "sqrt": _unary(np.sqrt),
    "abs": _unary(np.abs),
    "floor": _unary(np.floor),
    "ceil": _unary(np.ceil),
    "int": _unary(np.trunc),
    "trunc": _unary(np.trunc),
    "sign": _unary(np.sign),
    "normal": _unary(stats.norm.cdf),
    "normalden": _unary(stats.norm.pdf),
    "invnormal": _unary(stats.norm.ppf),
    "normprob": _unary(stats.norm.cdf),  # pre-Stata-7 name of normal()
    "chi2": _tail(stats.chi2, 1, upper=False),
    "chi2tail": _tail(stats.chi2, 1, upper=True),
    "chiprob": _tail(stats.chi2, 1, upper=True),  # old name of chi2tail()
    "ttail": _tail(stats.t, 1, upper=True),
    "invttail": _f_invttail,
    "F": _tail(stats.f, 2, upper=False),
    "Ftail": _tail(stats.f, 2, upper=True),
    "fprob": _tail(stats.f, 2, upper=True),  # old name of Ftail()
    "round": _f_round,
    "min": _extreme(np.nanmin),
    "max": _extreme(np.nanmax),
    "missing": _f_missing,
    "mi": _f_missing,
    "inlist": _f_inlist,
    "inrange": _f_inrange,
    "cond": _f_cond,
    "mod": _f_mod,
}


_DAILY = re.compile(r"\d{1,2}[A-Za-z]{3}\d{4}\Z")
_PERIODIC = re.compile(r"(\d{4})(?:([qQmM])(\d{1,2}))?\Z")


def _date_literal(text: str, column: pd.Series) -> Any:
    """A Stata date literal (``01jan1980``, ``1980q1``, ``1980m3``, ``1980``)
    as a value comparable with the time variable."""
    text = text.strip()
    if pd.api.types.is_datetime64_any_dtype(column):
        if not _DAILY.match(text):
            raise StataExprError(f"cannot read the date {text!r} (expected 01jan1980)")
        return pd.to_datetime(text, format="%d%b%Y")
    if isinstance(column.dtype, pd.PeriodDtype):
        m = _PERIODIC.match(text)
        if m is None:
            raise StataExprError(f"cannot read the period {text!r}")
        year, kind, num = m.group(1), (m.group(2) or "").lower(), m.group(3)
        stamp = f"{year}Q{num}" if kind == "q" else f"{year}-{int(num or 1):02d}"
        if kind == "q":
            return pd.Period(stamp, freq="Q").asfreq(column.dtype.freq, how="start")
        return pd.Period(stamp, freq="M").asfreq(column.dtype.freq, how="start")
    if pd.api.types.is_numeric_dtype(column):
        if _DAILY.match(text):
            # a Stata daily date counts days from 1 January 1960
            day = pd.to_datetime(text, format="%d%b%Y")
            return float((day - pd.Timestamp("1960-01-01")).days)
        try:
            return float(text)
        except ValueError:
            raise StataExprError(
                f"cannot compare the date {text!r} with a numeric time variable"
            ) from None
    raise StataExprError("the time variable is neither a date nor a number")


# ------------------------------------------------------------------ parser
class _Parser:
    def __init__(
        self,
        text: str,
        data: pd.DataFrame,
        stored: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.toks = _tokenise(text)
        self.i = 0
        self.data = data
        self.n = len(data)
        self.stored = stored or {}

    def _stored(self, kind: str, key: str, shown: str) -> float:
        table = self.stored.get(kind)
        if table is None:
            raise StataExprError(f"stored result {shown} is not available here")
        if key not in table:
            raise StataExprError(
                f"stored result {shown} is not set (available: "
                f"{', '.join(sorted(table)) or 'none'})"
            )
        return float(table[key])

    # token helpers
    def _peek(self) -> Tuple[str, str]:
        return self.toks[self.i] if self.i < len(self.toks) else ("end", "")

    def _take(self) -> Tuple[str, str]:
        tok = self._peek()
        self.i += 1
        return tok

    def _accept(self, *ops: str) -> Optional[str]:
        kind, val = self._peek()
        if kind == "op" and val in ops:
            self.i += 1
            return val
        return None

    def _expect(self, op: str) -> None:
        if self._accept(op) is None:
            raise StataExprError(f"expected {op!r} in {self.text!r}")

    # grammar, loosest binding first
    def parse(self) -> Value:
        value = self._or()
        if self._peek()[0] != "end":
            tok = self._peek()[1]
            if tok == "=":
                raise StataExprError("`=` is assignment; equality is `==`")
            raise StataExprError(f"unexpected {tok!r} in {self.text!r}")
        return value

    def _or(self) -> Value:
        left = self._and()
        while self._accept("|"):
            right = self._and()
            left = _flag(_truth(left) | _truth(right))
        return left

    def _and(self) -> Value:
        left = self._comparison()
        while self._accept("&"):
            right = self._comparison()
            left = _flag(_truth(left) & _truth(right))
        return left

    def _comparison(self) -> Value:
        left = self._additive()
        while True:
            op = self._accept("==", "!=", "~=", ">=", "<=", ">", "<")
            if op is None:
                return left
            left = _compare("!=" if op == "~=" else op, left, self._additive())

    def _additive(self) -> Value:
        left = self._multiplicative()
        while True:
            op = self._accept("+", "-")
            if op is None:
                return left
            right = self._multiplicative()
            if op == "+" and _is_str(left) and _is_str(right):
                raise StataExprError("string concatenation is not implemented")
            a, b = _num(left, "arithmetic"), _num(right, "arithmetic")
            with np.errstate(all="ignore"):
                left = _clean(a + b if op == "+" else a - b)

    def _multiplicative(self) -> Value:
        left = self._negation()
        while True:
            op = self._accept("*", "/")
            if op is None:
                return left
            right = self._negation()
            a, b = _num(left, "arithmetic"), _num(right, "arithmetic")
            with np.errstate(all="ignore"):
                left = _clean(np.multiply(a, b) if op == "*" else np.divide(a, b))

    def _negation(self) -> Value:
        if self._accept("-"):
            return _clean(-np.asarray(_num(self._negation(), "negation"), dtype=float))
        if self._accept("+"):
            return self._negation()
        return self._power()

    def _power(self) -> Value:
        # -2^2 is -(2^2), and 2^3^2 is (2^3)^2
        base = self._not()
        while self._accept("^"):
            exponent = self._signed_operand()
            with np.errstate(all="ignore"):
                base = _clean(
                    np.power(
                        np.asarray(_num(base, "^"), dtype=float),
                        np.asarray(_num(exponent, "^"), dtype=float),
                    )
                )
        return base

    def _signed_operand(self) -> Value:
        if self._accept("-"):
            operand = _num(self._signed_operand(), "negation")
            return _clean(-np.asarray(operand, dtype=float))
        return self._not()

    def _not(self) -> Value:
        if self._accept("!", "~"):
            return _flag(~_truth(self._not()))
        return self._primary()

    def _primary(self) -> Value:
        kind, val = self._take()
        if kind == "num":
            return float(val)
        if kind == "str":
            return val[1:-1]
        if kind == "op" and val == "(":
            inner = self._or()
            self._expect(")")
            return inner
        if kind == "op" and val == ".":
            nxt_kind, nxt = self._peek()
            if nxt_kind == "name" and len(nxt) == 1:
                raise StataExprError(f"extended missing value .{nxt}")
            return float("nan")
        if kind == "name":
            return self._name(val)
        raise StataExprError(f"unexpected {val!r} in {self.text!r}")

    def _name(self, name: str) -> Value:
        nxt = self._peek()
        if name in ("r", "e") and nxt == ("op", "(") and name not in _FUNCTIONS:
            # r(mean), e(N): results the previous command left behind
            self._expect("(")
            kind, key = self._take()
            if kind != "name":
                raise StataExprError(f"expected a name inside {name}()")
            self._expect(")")
            return self._stored(name, key, f"{name}({key})")
        if name in ("_b", "_se") and nxt == ("op", "["):
            self._expect("[")
            kind, key = self._take()
            if kind != "name":
                raise StataExprError(f"expected a coefficient name inside {name}[]")
            self._expect("]")
            return self._stored(name, key, f"{name}[{key}]")
        if name == "tin" and nxt == ("op", "("):
            return self._tin()
        if nxt == ("op", "("):
            return self._call(name)
        if nxt == ("op", "."):
            raise StataExprError(
                f"time-series operator or stored result {name}.… is not implemented"
            )
        if name == "_n":
            return np.arange(1.0, self.n + 1.0)
        if name == "_N":
            return float(self.n)
        if name == "_pi":
            return float(np.pi)
        if name not in self.data.columns and name in self.stored.get("scalars", {}):
            # a variable of that name would win, as in Stata
            return float(self.stored["scalars"][name])
        column = self._column(name)
        if self._accept("["):
            index = _num(self._or(), "a subscript")
            self._expect("]")
            return self._subscript(column, index)
        return column

    def _tin(self) -> Value:
        """``tin(d1, d2)``: the time variable is within the two dates."""
        self._expect("(")
        parts = [""]
        while True:
            kind, val = self._take()
            if kind == "end":
                raise StataExprError("unclosed tin(")
            if kind == "op" and val == ")":
                break
            if kind == "op" and val == ",":
                parts.append("")
            else:
                parts[-1] += val
        if len(parts) != 2:
            raise StataExprError("tin() takes two dates")
        time_var = self.stored.get("time_var")
        if not time_var or time_var not in self.data.columns:
            raise StataExprError("tin() needs the time variable of `tsset`")
        col = self.data[time_var]
        inside = np.ones(self.n, dtype=bool)
        if parts[0]:
            inside &= np.asarray(col >= _date_literal(parts[0], col))
        if parts[1]:
            inside &= np.asarray(col <= _date_literal(parts[1], col))
        return _flag(inside)

    def _call(self, name: str) -> Value:
        self._expect("(")
        fn = _FUNCTIONS.get(name)
        if fn is None:
            raise StataExprError(f"function {name}() is not implemented")
        args: List[Value] = []
        if not self._accept(")"):
            args.append(self._or())
            while self._accept(","):
                args.append(self._or())
            self._expect(")")
        return fn(*args)

    def _column(self, name: str) -> Value:
        if name not in self.data.columns:
            raise StataExprError(f"variable {name!r} is not in the data")
        col = self.data[name]
        if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            return col.to_numpy(dtype=float, na_value=np.nan)
        if pd.api.types.is_datetime64_any_dtype(col):
            raise StataExprError(
                f"{name!r} is a datetime column; Stata dates are day counts"
            )
        # a missing string is the empty string
        return col.astype(object).where(col.notna(), "").to_numpy(dtype=object)

    def _subscript(self, column: Value, index: Any) -> Value:
        idx = np.broadcast_to(np.asarray(index, dtype=float), (self.n,))
        valid = ~np.isnan(idx) & (idx >= 1) & (idx <= self.n) & (idx == np.floor(idx))
        pos = np.where(valid, idx, 1).astype(int) - 1
        col = np.asarray(column)
        if col.dtype == object:
            return np.where(valid, col[pos], "").astype(object)
        return np.where(valid, col[pos], np.nan)


def evaluate(
    expr: str,
    data: pd.DataFrame,
    stored: Optional[Dict[str, Any]] = None,
) -> Value:
    """Value of a Stata expression on every row of ``data``.

    ``stored`` holds what earlier commands left behind, by kind: ``"r"`` and
    ``"e"`` for ``r(name)`` / ``e(name)``, ``"_b"`` and ``"_se"`` for
    ``_b[x]`` / ``_se[x]``, and ``"scalars"`` for named scalars.

    Returns a float array of length ``len(data)`` (an object array for a
    string expression). Raises :class:`StataExprError` for anything outside
    the implemented grammar.
    """
    if not expr or not expr.strip():
        raise StataExprError("empty expression")
    value = _Parser(expr, data, stored).parse()
    if isinstance(value, str):
        return np.full(len(data), value, dtype=object)
    return np.broadcast_to(np.asarray(value), (len(data),)).copy()


def sample_mask(
    expr: str,
    data: pd.DataFrame,
    stored: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Rows an ``if`` qualifier keeps: where the expression is not zero."""
    value = evaluate(expr, data, stored)
    if value.dtype == object:
        raise StataExprError("an `if` condition must be numeric")
    return np.asarray(_truth(value), dtype=bool)


_IN_RANGE = re.compile(r"^\s*(-?\d+|f|l)\s*(?:/\s*(-?\d+|f|l))?\s*$", re.I)


def in_range_mask(spec: str, n: int) -> np.ndarray:
    """Rows an ``in`` range keeps: ``in 5``, ``in 1/100``, ``in -10/l``."""
    m = _IN_RANGE.match(spec or "")
    if m is None:
        raise StataExprError(f"cannot read the range `in {spec}`")

    def position(tok: str) -> int:
        low = tok.lower()
        if low == "f":
            return 1
        if low == "l":
            return n
        value = int(tok)
        return n + value + 1 if value < 0 else value

    first = position(m.group(1))
    last = position(m.group(2)) if m.group(2) else first
    if not (1 <= first <= last <= n):
        raise StataExprError(f"`in {spec}` is outside the {n} observations")
    mask = np.zeros(n, dtype=bool)
    mask[first - 1 : last] = True
    return mask
