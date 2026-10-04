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
from scipy import special, stats

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

_COEF_REF = re.compile(r"\b(_b|_se)\[\s*([^\]\"]*#[^\]\"]*?)\s*\]")
_POWER_TERM = re.compile(r"I\((\w+) ?\*\* ?(\d+)\)\Z")
_LEVEL_SPELLINGS = (
    re.compile(r"C\((\w+)(?:, Treatment\([^)]*\))?\)\[T\.(-?\d+)(?:\.0)?\]\Z"),
    re.compile(r"(\w+)::(-?\d+)(?:\.0)?\Z"),
    re.compile(r"(\w+)\[(-?\d+)(?:\.0)?\]\Z"),
)


def coefficient_key(name: str) -> str:
    """A coefficient name in one spelling, whoever wrote it.

    Stata writes ``1.d#3.t`` and ``1.d#c.x``; the estimators name the same
    columns ``C(d)[T.1]:C(t)[T.3]``, ``d::1.0:t::3.0`` or ``d[1.0]``. All of
    them map to the level-dot-name parts, sorted (the order of the factors
    in a product carries no meaning).
    """
    parts = []
    for part in re.split(r"#|(?<!:):(?!:)", name.strip()):
        part = part.strip()
        if part.startswith("c."):
            part = part[2:]
        power = _POWER_TERM.match(part)
        if power:
            # c.x#c.x is fitted as I(x ** 2)
            parts.extend([power.group(1)] * int(power.group(2)))
            continue
        for pattern in _LEVEL_SPELLINGS:
            m = pattern.match(part)
            if m:
                part = f"{m.group(2)}.{m.group(1)}"
                break
        parts.append(part)
    return "#".join(sorted(parts))


def _tokenise(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    pos = 0
    # _b[1.d#3.t] holds factor-variable notation, which is not an expression
    text = _COEF_REF.sub(r'\1["\2"]', text.rstrip())
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


def _f_normalden(*args: Value) -> Any:
    """``normalden(z)``, ``normalden(x, sd)`` or ``normalden(x, mean, sd)``."""
    if len(args) not in (1, 2, 3):
        raise StataExprError("normalden() takes one to three arguments")
    vals = [np.asarray(_num(a, "normalden()"), dtype=float) for a in args]
    x = vals[0]
    mean = vals[1] if len(args) == 3 else 0.0
    sd = vals[-1] if len(args) > 1 else 1.0
    with np.errstate(all="ignore"):
        return _clean(np.where(sd > 0, stats.norm.pdf(x, mean, sd), np.nan))


def _f_running_sum(*args: Value) -> Any:
    """``sum(x)``: the running sum down the rows; a missing value adds zero."""
    if len(args) != 1:
        raise StataExprError("sum() takes one argument")
    x = np.asarray(_num(args[0], "sum()"), dtype=float)
    if x.ndim == 0:
        raise StataExprError("sum() of a constant needs the data's rows")
    return np.cumsum(np.where(np.isnan(x), 0.0, x))


def _quantile(dist: Any, n_shape: int, upper: bool) -> Callable[..., Any]:
    """Inverse distribution functions: degrees of freedom first, then p."""

    def call(*args: Value) -> Any:
        if len(args) != n_shape + 1:
            raise StataExprError(f"this function takes {n_shape + 1} arguments")
        vals = [
            np.asarray(_num(a, "a distribution function"), dtype=float) for a in args
        ]
        with np.errstate(all="ignore"):
            fn = dist.isf if upper else dist.ppf
            return _clean(fn(vals[-1], *vals[:-1]))

    return call


def _density(dist: Any, n_shape: int) -> Callable[..., Any]:
    def call(*args: Value) -> Any:
        if len(args) != n_shape + 1:
            raise StataExprError(f"this function takes {n_shape + 1} arguments")
        vals = [np.asarray(_num(a, "a density function"), dtype=float) for a in args]
        with np.errstate(all="ignore"):
            return _clean(dist.pdf(vals[-1], *vals[:-1]))

    return call


# ---- dates: a Stata date is a count of periods since 1960 ---------------
_EPOCH = np.datetime64("1960-01-01", "D")


def _days(v: Value, what: str) -> Tuple[np.ndarray, np.ndarray]:
    """Daily dates as numpy days, with the mask of the usable ones."""
    x = np.asarray(_num(v, what), dtype=float)
    ok = np.isfinite(x)
    return _EPOCH + np.where(ok, np.floor(x), 0).astype("timedelta64[D]"), ok


def _date_part(part: str) -> Callable[..., Any]:
    def call(*args: Value) -> Any:
        if len(args) != 1:
            raise StataExprError(f"{part}() takes one argument")
        days, ok = _days(args[0], f"{part}()")
        months = days.astype("datetime64[M]")
        years = months.astype("datetime64[Y]")
        if part == "year":
            out = years.astype(int) + 1970
        elif part == "month":
            out = (months - years.astype("datetime64[M]")).astype(int) + 1
        elif part == "quarter":
            out = (months - years.astype("datetime64[M]")).astype(int) // 3 + 1
        elif part == "day":
            out = (days - months.astype("datetime64[D]")).astype(int) + 1
        elif part == "dow":  # 0 = Sunday; 1 January 1960 was a Friday
            out = np.mod((days - _EPOCH).astype(int) + 5, 7)
        elif part == "mofd":
            out = months.astype(int) - (1960 - 1970) * 12
        elif part == "qofd":
            out = np.floor_divide(months.astype(int) - (1960 - 1970) * 12, 3)
        else:  # yofd
            out = years.astype(int) + 1970
        return np.where(ok, out.astype(float), np.nan)

    return call


def _periods_to_days(per_year: int) -> Callable[..., Any]:
    """``dofm()`` / ``dofq()`` / ``dofy()``: first day of the period."""

    def call(*args: Value) -> Any:
        if len(args) != 1:
            raise StataExprError("this function takes one argument")
        x = np.asarray(_num(args[0], "a date function"), dtype=float)
        ok = np.isfinite(x)
        whole = np.where(ok, np.floor(x), 0).astype(int)
        if per_year == 1:  # a yearly date is the year itself
            months = (whole - 1960) * 12
        else:
            months = whole * (12 // per_year)
        first = (
            np.datetime64("1960-01", "M") + months.astype("timedelta64[M]")
        ).astype("datetime64[D]")
        return np.where(ok, (first - _EPOCH).astype(float), np.nan)

    return call


def _f_mdy(*args: Value) -> Any:
    if len(args) != 3:
        raise StataExprError("mdy() takes three arguments")
    m, d, y = np.broadcast_arrays(
        *[np.asarray(_num(a, "mdy()"), dtype=float) for a in args]
    )
    ok = np.isfinite(m) & np.isfinite(d) & np.isfinite(y)
    ok &= (m >= 1) & (m <= 12) & (d >= 1) & (d <= 31)
    months = np.where(ok, (y - 1960) * 12 + (m - 1), 0).astype(int)
    first = (np.datetime64("1960-01", "M") + months.astype("timedelta64[M]")).astype(
        "datetime64[D]"
    )
    day = first + np.where(ok, d - 1, 0).astype("timedelta64[D]")
    # 31 February is not a date: the day must stay inside its month
    ok &= day.astype("datetime64[M]") == first.astype("datetime64[M]")
    return np.where(ok, (day - _EPOCH).astype(float), np.nan)


_MONTH_NAMES = {
    name: i + 1
    for i, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun"]
        + ["jul", "aug", "sep", "oct", "nov", "dec"]
    )
}


def _parse_date_text(text: Any, order: str) -> float:
    """One string -> days since 01jan1960 under a three-letter mask."""
    if not isinstance(text, str):
        return np.nan
    text = text.strip()
    # digits and letters are fields of their own: 31jul2015, July 31, 2015
    parts = re.findall(r"\d+|[A-Za-z]+", text)
    if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 8:
        # run-together digits: the year is the four-digit field
        digits, parts, pos = parts[0], [], 0
        for field in order:
            width = 4 if field == "Y" else 2
            parts.append(digits[pos : pos + width])
            pos += width
    if len(parts) != 3:
        return np.nan
    field = dict(zip(order, parts))
    month = field["M"]
    if month.isdigit():
        m_num = int(month)
    else:
        m_num = _MONTH_NAMES.get(month[:3].lower(), 0)
    # a two-digit year needs a century in the mask, which is not read here
    if not (field["Y"].isdigit() and len(field["Y"]) == 4 and field["D"].isdigit()):
        return np.nan
    out = _f_mdy(float(m_num), float(field["D"]), float(field["Y"]))
    return float(np.asarray(out))


def _f_date(*args: Value) -> Any:
    """``date(s, mask)`` for the masks that are a permutation of D, M, Y."""
    if len(args) != 2 or not isinstance(args[1], str):
        raise StataExprError("date() takes a string and a literal mask")
    order = args[1].strip().upper()
    if sorted(order) != ["D", "M", "Y"]:
        raise StataExprError(
            f"date(): the mask {args[1]!r} is not implemented (only the "
            'orderings of D, M and Y, e.g. "YMD")'
        )
    source = args[0]
    if isinstance(source, str):
        return _parse_date_text(source, order)
    if not _is_str(source):
        raise StataExprError("date() needs a string, got a number")
    return np.array([_parse_date_text(v, order) for v in source], dtype=float)


def _f_periodic(per_year: int) -> Callable[..., Any]:
    """``ym(y, m)`` / ``yq(y, q)``."""

    def call(*args: Value) -> Any:
        if len(args) != 2:
            raise StataExprError("this function takes two arguments")
        y, sub = (np.asarray(_num(a, "a date function"), dtype=float) for a in args)
        with np.errstate(all="ignore"):
            out = (y - 1960) * per_year + sub - 1
            return _clean(np.where((sub >= 1) & (sub <= per_year), out, np.nan))

    return call


#: Functions that draw random numbers. Stata's generator (a 64-bit Mersenne
#: Twister with its own transformations) is not reproduced: the draws come
#: from numpy, seeded by ``set seed``, so a simulation has the same design
#: and a different sample.
_RANDOM = frozenset(
    {"rnormal", "runiform", "rchi2", "rt", "rbinomial", "rpoisson", "rexponential",
     "runiformint", "rbeta", "rgamma", "uniform"}
)  # fmt: skip


def _draw(name: str, args: List[Any], n: int, rng: np.random.Generator) -> Any:
    a = [np.asarray(_num(v, f"{name}()"), dtype=float) for v in args]

    def arity(*allowed: int) -> None:
        if len(a) not in allowed:
            raise StataExprError(
                f"{name}() takes {' or '.join(str(k) for k in allowed)} argument(s)"
            )

    with np.errstate(all="ignore"):
        if name == "rnormal":
            arity(0, 1, 2)
            mean = a[0] if a else 0.0
            sd = a[1] if len(a) == 2 else 1.0
            return _clean(mean + sd * rng.standard_normal(n))
        if name in ("runiform", "uniform"):
            arity(0, 2)
            lo, hi = (a[0], a[1]) if a else (0.0, 1.0)
            return _clean(lo + (hi - lo) * rng.random(n))
        if name == "runiformint":
            arity(2)
            return _clean(np.floor(a[0] + (a[1] - a[0] + 1) * rng.random(n)))
        if name == "rchi2":
            arity(1)
            return _clean(rng.chisquare(np.broadcast_to(a[0], (n,))))
        if name == "rt":
            arity(1)
            return _clean(rng.standard_t(np.broadcast_to(a[0], (n,))))
        if name == "rbinomial":
            arity(2)
            return _clean(
                rng.binomial(
                    np.broadcast_to(a[0], (n,)).astype(int), np.broadcast_to(a[1], (n,))
                ).astype(float)
            )
        if name == "rpoisson":
            arity(1)
            return _clean(rng.poisson(np.broadcast_to(a[0], (n,))).astype(float))
        if name == "rexponential":
            arity(1)
            return _clean(rng.exponential(np.broadcast_to(a[0], (n,))))
        if name == "rbeta":
            arity(2)
            return _clean(rng.beta(np.broadcast_to(a[0], (n,)), a[1]))
        if name == "rgamma":
            arity(2)
            return _clean(rng.gamma(np.broadcast_to(a[0], (n,)), a[1]))
    raise StataExprError(f"function {name}() is not implemented")


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
    "normalden": _f_normalden,
    "invnormal": _unary(stats.norm.ppf),
    "normprob": _unary(stats.norm.cdf),  # pre-Stata-7 name of normal()
    "invnorm": _unary(stats.norm.ppf),  # pre-Stata-10 name of invnormal()
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
    "sin": _unary(np.sin),
    "cos": _unary(np.cos),
    "tan": _unary(np.tan),
    "asin": _unary(np.arcsin),
    "acos": _unary(np.arccos),
    "atan": _unary(np.arctan),
    "sinh": _unary(np.sinh),
    "cosh": _unary(np.cosh),
    "tanh": _unary(np.tanh),
    "asinh": _unary(np.arcsinh),
    "acosh": _unary(np.arccosh),
    "atanh": _unary(np.arctanh),
    "lngamma": _unary(special.gammaln),
    "logit": _unary(special.logit),
    "invlogit": _unary(special.expit),
    "sum": _f_running_sum,
    "t": _tail(stats.t, 1, upper=False),
    "tden": _density(stats.t, 1),
    "chi2den": _density(stats.chi2, 1),
    "Fden": _density(stats.f, 2),
    "invt": _quantile(stats.t, 1, upper=False),
    "invchi2": _quantile(stats.chi2, 1, upper=False),
    "invchi2tail": _quantile(stats.chi2, 1, upper=True),
    "invF": _quantile(stats.f, 2, upper=False),
    "invFtail": _quantile(stats.f, 2, upper=True),
    "year": _date_part("year"),
    "month": _date_part("month"),
    "quarter": _date_part("quarter"),
    "day": _date_part("day"),
    "dow": _date_part("dow"),
    "mofd": _date_part("mofd"),
    "qofd": _date_part("qofd"),
    "yofd": _date_part("yofd"),
    "dofm": _periods_to_days(12),
    "dofq": _periods_to_days(4),
    "dofy": _periods_to_days(1),
    "mdy": _f_mdy,
    "date": _f_date,
    "daily": _f_date,
    "ym": _f_periodic(12),
    "yq": _f_periodic(4),
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


_DATE_LITERALS = frozenset({"tq", "tm", "td", "ty"})
_TS_PREFIX = re.compile(r"(?:[LlFfDd]\d*)+\Z")


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
            key = self._coefficient_name(name)
            return self._stored(name, key, f"{name}[{key}]")
        if name == "scalar" and nxt == ("op", "("):
            # scalar(name): the named scalar, even if a variable shares it
            self._expect("(")
            kind, key = self._take()
            self._expect(")")
            scalars = self.stored.get("scalars", {})
            if kind != "name" or key not in scalars:
                raise StataExprError(f"scalar {key!r} is not defined")
            return float(scalars[key])
        matrices = self.stored.get("matrices") or {}
        if name in ("rowsof", "colsof", "el") and nxt == ("op", "("):
            self._expect("(")
            kind, key = self._take()
            if kind != "name" or key not in matrices:
                raise StataExprError(f"matrix {key} not found")
            values = matrices[key]["values"]
            if name == "el":
                self._expect(",")
                i = _num(self._or(), "a row number")
                self._expect(",")
                j = _num(self._or(), "a column number")
                self._expect(")")
                return self._cell(key, values, i, j)
            self._expect(")")
            return float(values.shape[0 if name == "rowsof" else 1])
        if name in matrices and nxt == ("op", "[") and name not in self.data.columns:
            self._expect("[")
            i = _num(self._or(), "a row number")
            self._expect(",")
            j = _num(self._or(), "a column number")
            self._expect("]")
            return self._cell(name, matrices[name]["values"], i, j)
        if name == "tin" and nxt == ("op", "("):
            return self._tin()
        if name in _DATE_LITERALS and nxt == ("op", "("):
            return self._date_function(name)
        if nxt == ("op", "("):
            return self._call(name)
        if nxt == ("op", "."):
            if _TS_PREFIX.match(name):
                return self._ts_operator(name)
            raise StataExprError(f"{name}.… is not a time-series operator")
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

    def _cell(self, name: str, values: Any, i: Any, j: Any) -> float:
        """``A[i, j]``; a subscript outside the matrix is missing, as in
        Stata's ``el()``."""
        row, col = float(np.asarray(i).flat[0]), float(np.asarray(j).flat[0])
        if not (row == int(row) and col == int(col)):
            raise StataExprError(f"matrix {name}: subscripts must be whole numbers")
        if not (1 <= row <= values.shape[0] and 1 <= col <= values.shape[1]):
            return float("nan")
        return float(values[int(row) - 1, int(col) - 1])

    def _coefficient_name(self, kind: str) -> str:
        """The name inside ``_b[...]``; ``_b[L.y]`` is the coefficient the
        session stored as ``y_L1``."""
        text = ""
        while True:
            tok_kind, val = self._take()
            if tok_kind == "end":
                raise StataExprError(f"unclosed {kind}[")
            if tok_kind == "op" and val == "]":
                break
            text += val
        text = text.strip('"')  # _b["x"] is _b[x]
        if not text:
            raise StataExprError(f"expected a coefficient name inside {kind}[]")
        m = re.fullmatch(r"((?:[LlFfDd]\d*)+)\.([A-Za-z_]\w*)", text)
        if m:
            from ._stata_tsops import _name, _parse_ops

            return _name(m.group(2), _parse_ops(m.group(1)))
        held = self.stored.get(kind) or {}
        if text not in held and ("#" in text or re.match(r"\d+\.[A-Za-z_]", text)):
            key = coefficient_key(text)
            if key not in held and held:
                # the base level of a factor that is in the model: Stata
                # holds a zero for it (coefficient and standard error)
                def shape(k: str) -> List[str]:
                    return [re.sub(r"^-?\d+\.", "", p) for p in k.split("#")]

                if any(shape(k) == shape(key) and k != key for k in held):
                    held[key] = 0.0
            return key
        return text

    def _ts_operator(self, ops: str) -> Value:
        """``l.y`` / ``d.lny`` / ``L2.x`` inside an expression."""
        from ._stata_tsops import _apply, _Clock, _parse_ops

        self._expect(".")
        kind, var = self._take()
        if kind != "name":
            raise StataExprError(f"expected a variable after {ops}.")
        time_var = self.stored.get("time_var")
        if not time_var:
            raise StataExprError(
                f"the time-series operator {ops}.{var} needs the time variable: "
                "put `tsset time` or `xtset id time` before it"
            )
        clock = _Clock(self.data, self.stored.get("panel_var"), time_var)
        return _apply(
            clock, np.asarray(self._column(var), dtype=float), _parse_ops(ops)
        )

    def _date_function(self, name: str) -> Value:
        """``tq(1999q1)`` / ``tm(1999m3)`` / ``td(01jan1999)`` / ``ty(1999)``."""
        self._expect("(")
        text = ""
        while True:
            kind, val = self._take()
            if kind == "end":
                raise StataExprError(f"unclosed {name}(")
            if kind == "op" and val == ")":
                break
            text += val
        text = text.strip().lower()
        if name == "td":
            if not _DAILY.match(text):
                raise StataExprError(f"cannot read the date {text!r} in td()")
            day = pd.to_datetime(text, format="%d%b%Y")
            return float((day - pd.Timestamp("1960-01-01")).days)
        if name == "ty":
            if not re.fullmatch(r"\d{4}", text):
                raise StataExprError(f"cannot read the year {text!r} in ty()")
            return float(text)
        letter, per_year = ("q", 4) if name == "tq" else ("m", 12)
        m = re.fullmatch(rf"(\d{{4}}){letter}(\d{{1,2}})", text)
        if m is None or not 1 <= int(m.group(2)) <= per_year:
            raise StataExprError(f"cannot read the period {text!r} in {name}()")
        return float((int(m.group(1)) - 1960) * per_year + int(m.group(2)) - 1)

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
        if fn is None and name not in _RANDOM:
            raise StataExprError(f"function {name}() is not implemented")
        args: List[Value] = []
        if not self._accept(")"):
            args.append(self._or())
            while self._accept(","):
                args.append(self._or())
            self._expect(")")
        if fn is None:
            rng = self.stored.get("rng")
            if rng is None:
                raise StataExprError(
                    f"{name}() draws random numbers, which needs a session "
                    "(sp.stata), not a single translated line"
                )
            self.stored["random_draws"] = True
            return _draw(name, args, self.n, rng)
        return fn(*args)

    def _column(self, name: str) -> Value:
        if name not in self.data.columns:
            # Stata reads an unambiguous abbreviation of a variable name
            hits = [c for c in self.data.columns if str(c).startswith(name)]
            if len(hits) > 1:
                raise StataExprError(
                    f"{name!r} is an ambiguous abbreviation: it fits {hits[:4]}"
                )
            if not hits:
                raise StataExprError(f"variable {name!r} is not in the data")
            name = hits[0]
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
