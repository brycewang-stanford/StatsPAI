"""More functions of the Stata expression language.

``_stata_expr`` holds the grammar and the functions the first estimation
do-files needed. This module adds what a data-management do-file uses:
string functions, the grouping functions (``recode`` / ``irecode`` /
``autocode``), time-of-day arithmetic, the remaining calendar parts, the
discrete distributions and the storage-type helpers (``float`` / ``real`` /
``string``). Each follows the function's entry in the Stata manual ([FN]),
including what it returns for a missing argument.

Strings follow Stata's byte semantics: ``strlen("Müller")`` is 7 and
``substr`` counts bytes, while the ``u*`` functions count characters.

Importing the module adds the functions to the evaluator's table.
"""

from __future__ import annotations

import os
import platform
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import special, stats

from . import _stata_expr as _expr
from ._stata_expr import StataExprError, _clean, _is_str, _num

__all__ = ["stata_format", "system_value", "SYSTEM_STRINGS"]

Value = Any


# ----------------------------------------------------------- broadcasting
def _length(args: tuple) -> Optional[int]:
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim:
            return len(a)
    return None


def _cells(value: Value, n: Optional[int]) -> List[Any]:
    if isinstance(value, np.ndarray) and value.ndim:
        return list(value)
    return [value] * (n or 1)


def _elementwise(fn: Callable[..., Any], args: tuple, *, text: bool) -> Value:
    """``fn`` applied row by row; a scalar when every argument is one."""
    n = _length(args)
    columns = [_cells(a, n) for a in args]
    out = [fn(*row) for row in zip(*columns)]
    if n is None:
        return out[0] if text else float(out[0])
    if text:
        return np.array(out, dtype=object)
    return np.array(out, dtype=float)


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise StataExprError(f"{what} needs a string, got a number")
    return value


def _number(value: Any, what: str) -> float:
    if isinstance(value, str):
        raise StataExprError(f"{what} needs a number, got a string")
    return float(value)


def _arity(name: str, args: tuple, *allowed: int) -> None:
    if len(args) not in allowed:
        raise StataExprError(
            f"{name}() takes {' or '.join(str(k) for k in allowed)} argument(s)"
        )


def _string_function(
    name: str, fn: Callable[..., Any], *allowed: int, text: bool = True
) -> Callable[..., Any]:
    def call(*args: Value) -> Value:
        _arity(name, args, *allowed)
        return _elementwise(fn, args, text=text)

    return call


# ------------------------------------------------------------ byte strings
def _bytes(s: str) -> bytes:
    return s.encode("utf-8", "surrogateescape")


def _unbytes(b: bytes) -> str:
    return b.decode("utf-8", "surrogateescape")


def _substr(s: Any, start: Any, length: Any) -> str:
    raw = _bytes(_text(s, "substr()"))
    start, length = _number(start, "substr()"), _number(length, "substr()")
    if start != start or start == 0:
        return ""
    n = len(raw)
    first = int(start) if start > 0 else n + int(start) + 1
    if first < 1 or first > n:
        return ""
    count = n if length != length else int(length)
    if count <= 0:
        return ""
    return _unbytes(raw[first - 1 : first - 1 + count])


def _usubstr(s: Any, start: Any, length: Any) -> str:
    text = _text(s, "usubstr()")
    start, length = _number(start, "usubstr()"), _number(length, "usubstr()")
    if start != start or start == 0:
        return ""
    n = len(text)
    first = int(start) if start > 0 else n + int(start) + 1
    if first < 1 or first > n:
        return ""
    count = n if length != length else int(length)
    return text[first - 1 : first - 1 + count] if count > 0 else ""


def _strpos(s: Any, sub: Any) -> float:
    hay, needle = _bytes(_text(s, "strpos()")), _bytes(_text(sub, "strpos()"))
    return float(hay.find(needle) + 1) if needle else 0.0


def _strrpos(s: Any, sub: Any) -> float:
    hay, needle = _bytes(_text(s, "strrpos()")), _bytes(_text(sub, "strrpos()"))
    return float(hay.rfind(needle) + 1) if needle else 0.0


def _ustrpos(s: Any, sub: Any) -> float:
    needle = _text(sub, "ustrpos()")
    return float(_text(s, "ustrpos()").find(needle) + 1) if needle else 0.0


def _ascii_map(s: str, fn: Callable[[str], str]) -> str:
    return "".join(fn(ch) if ch.isascii() else ch for ch in s)


def _proper(s: Any) -> str:
    """First letter and every letter after a non-letter in upper case, the
    rest in lower case (ASCII letters only, as ``strproper``)."""
    out, after_letter = [], False
    for ch in _text(s, "strproper()"):
        if ch.isascii() and ch.isalpha():
            out.append(ch.lower() if after_letter else ch.upper())
            after_letter = True
        else:
            out.append(ch)
            after_letter = False
    return "".join(out)


def _subinstr(s: Any, old: Any, new: Any, count: Any) -> str:
    text, old, new = (_text(v, "subinstr()") for v in (s, old, new))
    count = _number(count, "subinstr()")
    if not old:
        return text
    return (
        text.replace(old, new)
        if count != count
        else text.replace(old, new, max(int(count), 0))
    )


def _subinword(s: Any, old: Any, new: Any, count: Any) -> str:
    text, old, new = (_text(v, "subinword()") for v in (s, old, new))
    count = _number(count, "subinword()")
    if not old:
        return text
    pattern = r"(?<!\S)" + re.escape(old) + r"(?!\S)"
    limit = 0 if count != count else max(int(count), 0)
    if count == count and limit == 0:
        return text
    return re.sub(pattern, lambda _m: new, text, count=limit)


def _word(s: Any, k: Any) -> str:
    words = _text(s, "word()").split()
    k = _number(k, "word()")
    if k != k or k == 0:
        return ""
    pos = int(k) - 1 if k > 0 else len(words) + int(k)
    return words[pos] if 0 <= pos < len(words) else ""


def _strmatch(s: Any, pattern: Any) -> float:
    regex = "".join(
        ".*" if ch == "*" else "." if ch == "?" else re.escape(ch)
        for ch in _text(pattern, "strmatch()")
    )
    return float(re.fullmatch(regex, _text(s, "strmatch()"), re.S) is not None)


def _itrim(s: Any) -> str:
    return re.sub(r" {2,}", " ", _text(s, "stritrim()"))


def _abbrev(s: Any, n: Any) -> str:
    text, n = _text(s, "abbrev()"), int(_number(n, "abbrev()"))
    n = max(5, min(n, 32))
    if len(text) <= n:
        return text
    return text[: n - 2] + "~" + text[-1]


def _strtoname(s: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", _text(s, "strtoname()"))[:32]
    return "_" + out[:31] if out[:1].isdigit() else out


#: POSIX character classes of Stata's (pre-Unicode) regular expressions
#: are those of Python; ``regexs`` reads the groups of the last match.
_LAST_MATCH: Dict[str, Any] = {"groups": None}


def _regexm(s: Any, pattern: Any) -> float:
    found = re.search(_text(pattern, "regexm()"), _text(s, "regexm()"))
    _LAST_MATCH["groups"] = found
    return float(found is not None)


def _f_regexm(*args: Value) -> Value:
    _arity("regexm", args, 2)
    n = _length(args)
    rows = list(zip(_cells(args[0], n), _cells(args[1], n)))
    found = [re.search(_text(p, "regexm()"), _text(s, "regexm()")) for s, p in rows]
    _LAST_MATCH["groups"] = found if n is not None else found[0]
    flags = [float(m is not None) for m in found]
    return np.array(flags, dtype=float) if n is not None else flags[0]


def _f_regexs(*args: Value) -> Value:
    _arity("regexs", args, 1)
    k = int(_number(args[0], "regexs()"))
    held = _LAST_MATCH["groups"]

    def group(m: Any) -> str:
        if m is None:
            return ""
        try:
            return m.group(k) or ""
        except IndexError:
            return ""

    if isinstance(held, list):
        return np.array([group(m) for m in held], dtype=object)
    return group(held)


def _regexr(s: Any, pattern: Any, new: Any) -> str:
    return re.sub(
        _text(pattern, "regexr()"),
        lambda _m: _text(new, "regexr()"),
        _text(s, "regexr()"),
        count=1,
    )


def _ustrregexm(s: Any, pattern: Any, *nocase: Any) -> float:
    flags = re.I if nocase and _number(nocase[0], "ustrregexm()") else 0
    return float(
        re.search(_text(pattern, "ustrregexm()"), _text(s, "ustrregexm()"), flags)
        is not None
    )


def _icu_replacement(new: str) -> str:
    # ICU writes a group as $1, Python as \1
    return re.sub(r"\$(\d)", r"\\\1", new.replace("\\", "\\\\"))


def _ustrregexrf(s: Any, pattern: Any, new: Any, *nocase: Any) -> str:
    flags = re.I if nocase and _number(nocase[0], "ustrregexrf()") else 0
    return re.sub(
        _text(pattern, "ustrregexrf()"),
        _icu_replacement(_text(new, "ustrregexrf()")),
        _text(s, "ustrregexrf()"),
        count=1,
        flags=flags,
    )


def _ustrregexra(s: Any, pattern: Any, new: Any, *nocase: Any) -> str:
    flags = re.I if nocase and _number(nocase[0], "ustrregexra()") else 0
    return re.sub(
        _text(pattern, "ustrregexra()"),
        _icu_replacement(_text(new, "ustrregexra()")),
        _text(s, "ustrregexra()"),
        flags=flags,
    )


# ------------------------------------------------------- numbers and text
_FORMAT = re.compile(r"%(-)?(0)?(\d+)?(?:\.(\d+))?([efgs])(c)?\Z")
_DATE_FORMAT = re.compile(r"%-?(\d+)?t([dcCwmqhy])(.*)\Z")
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
           "Nov", "Dec"]  # fmt: skip
_MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]  # fmt: skip
_DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
_DAYS_FULL = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday"]  # fmt: skip


def _commas(text: str) -> str:
    sign = "-" if text.startswith("-") else ""
    whole, dot, frac = text.lstrip("-").partition(".")
    whole = f"{int(whole):,}" if whole.isdigit() else whole
    return sign + whole + dot + frac


def _general(value: float, width: int, digits: Optional[int]) -> str:
    """``%w.dg``: with d = 0, as many significant digits as fit in w."""
    if value == 0:
        return "0"
    if digits:
        text = format(value, f".{digits}g")
    else:
        text = ""
        for sig in range(min(width, 17), 0, -1):
            text = format(value, f".{sig}g")
            if "e" in text:
                mantissa, _, exponent = text.partition("e")
                if "." in mantissa:
                    mantissa = mantissa.rstrip("0").rstrip(".")
                text = f"{mantissa}e{exponent[0]}{int(exponent[1:]):02d}"
            text = re.sub(r"^(-?)0\.", r"\1.", text)
            if len(text) <= width:
                break
    if "e" in text:
        mantissa, _, exponent = text.partition("e")
        text = f"{mantissa}e{exponent[0]}{int(exponent[1:]):02d}"
    return re.sub(r"^(-?)0\.", r"\1.", text)


def _civil(day: int) -> tuple:
    """Year, month, day, day of week of a Stata daily date."""
    date = np.datetime64("1960-01-01") + np.timedelta64(int(day), "D")
    year, month, dom = (int(v) for v in str(date).split("-"))
    return year, month, dom, (int(day) + 5) % 7  # 01jan1960 was a Friday


def _format_daily(day: float, detail: str, ms: float = 0.0) -> str:
    year, month, dom, dow = _civil(int(day))
    seconds = int(ms // 1000) if ms else 0
    hour, minute, second = seconds // 3600, seconds // 60 % 60, seconds % 60
    if not detail:
        return f"{dom:02d}{_MONTHS[month - 1].lower()}{year}"
    codes = [
        ("CCYY", f"{year:04d}"), ("CC", f"{year // 100:02d}"),
        ("YY", f"{year % 100:02d}"),
        ("Month", _MONTHS_FULL[month - 1]), ("Mon", _MONTHS[month - 1]),
        ("month", _MONTHS_FULL[month - 1].lower()),
        ("mon", _MONTHS[month - 1].lower()),
        ("NN", f"{month:02d}"), ("nn", str(month)),
        ("DD", f"{dom:02d}"), ("dd", str(dom)),
        ("Dayname", _DAYS_FULL[dow]), ("Day", _DAYS[dow]),
        ("dayname", _DAYS_FULL[dow].lower()), ("day", _DAYS[dow].lower()),
        ("JJJ", f"{int(_doy(day)):03d}"), ("jjj", str(int(_doy(day)))),
        ("HH", f"{hour:02d}"), ("hh", str(hour)),
        ("MM", f"{minute:02d}"), ("mm", str(minute)),
        ("SS", f"{second:02d}"), ("ss", str(second)),
        ("_", " "), ("!", ""),
    ]  # fmt: skip
    out, pos = [], 0
    while pos < len(detail):
        for code, shown in codes:
            if detail.startswith(code, pos):
                if code == "!":  # the next character, as it is
                    out.append(detail[pos + 1 : pos + 2])
                    pos += 2
                else:
                    out.append(shown)
                    pos += len(code)
                break
        else:
            out.append(detail[pos])
            pos += 1
    return "".join(out)


def stata_format(value: Any, fmt: str) -> str:
    """``value`` as ``display`` prints it under the format ``fmt``.

    Numeric formats ``%w.df`` / ``e`` / ``g`` (with ``c`` for commas and a
    leading ``-`` or ``0``), string formats and the date formats ``%td``
    and ``%tc`` with or without a detail string are covered; the output is
    not padded to the width, which only matters on a screen.
    """
    fmt = fmt.strip()
    if isinstance(value, str):
        return value
    value = float(value)
    if value != value:
        return "."
    dated = _DATE_FORMAT.match(fmt)
    if dated:
        unit, detail = dated.group(2), dated.group(3)
        if unit == "d":
            return _format_daily(value, detail)
        if unit in "cC":
            day, ms = divmod(value, 86_400_000.0)
            return _format_daily(day, detail or "DDmonCCYY_HH:MM:SS", ms)
        if unit == "y":
            return str(int(value))
        year = 1960 + int(value // {"m": 12, "q": 4, "h": 2, "w": 52}[unit])
        part = int(value % {"m": 12, "q": 4, "h": 2, "w": 52}[unit]) + 1
        return f"{year}{unit}{part}"
    m = _FORMAT.match(fmt)
    if m is None:
        raise StataExprError(f"the format {fmt!r} is not implemented")
    width = int(m.group(3) or 9)
    digits = int(m.group(4)) if m.group(4) is not None else None
    kind = m.group(5)
    if kind == "f":
        text = format(value, f".{digits or 0}f")
        if abs(value) < 1 and text.lstrip("-").startswith("0."):
            text = text.replace("0.", ".", 1)
    elif kind == "e":
        text = format(value, f".{digits if digits is not None else 6}e")
        mantissa, _, exponent = text.partition("e")
        text = f"{mantissa}e{exponent[0]}{int(exponent[1:]):02d}"
    elif kind == "g":
        text = _general(value, width, digits)
    else:
        raise StataExprError(f"the format {fmt!r} is for strings")
    return _commas(text) if m.group(6) else text


def _string(value: Any, *fmt: Any) -> str:
    if isinstance(value, str):
        raise StataExprError("string() needs a number")
    return stata_format(value, _text(fmt[0], "string()") if fmt else "%10.0g")


def _real(s: Any) -> float:
    text = _text(s, "real()").strip()
    if not re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text):
        return np.nan
    return float(text)


def _f_float(*args: Value) -> Value:
    _arity("float", args, 1)
    x = np.asarray(_num(args[0], "float()"), dtype=float)
    with np.errstate(over="ignore"):
        return _clean(x.astype(np.float32).astype(np.float64))


# ----------------------------------------------------------------- grouping
def _f_recode(*args: Value) -> Value:
    """``recode(x, x1, ..., xn)``: the first bound not below x, the last
    one above them all, and missing where x is missing."""
    if len(args) < 3:
        raise StataExprError("recode() needs a value and at least two bounds")
    x = np.asarray(_num(args[0], "recode()"), dtype=float)
    bounds = [np.asarray(_num(a, "recode()"), dtype=float) for a in args[1:]]
    out = np.broadcast_to(bounds[-1], np.shape(x)).astype(float).copy()
    with np.errstate(invalid="ignore"):
        for bound in reversed(bounds[:-1]):
            out = np.where(x <= bound, bound, out)
    return np.where(np.isnan(x), np.nan, out)


def _f_irecode(*args: Value) -> Value:
    """``irecode(x, x1, ..., xn)``: how many of the bounds x exceeds."""
    if len(args) < 2:
        raise StataExprError("irecode() needs a value and a bound")
    x = np.asarray(_num(args[0], "irecode()"), dtype=float)
    out = np.zeros(np.shape(x), dtype=float)
    with np.errstate(invalid="ignore"):
        for bound in args[1:]:
            out = out + (x > np.asarray(_num(bound, "irecode()"), dtype=float))
    return np.where(np.isnan(x), np.nan, out)


def _f_autocode(*args: Value) -> Value:
    """``autocode(x, n, x0, x1)``: ``recode`` on n equal intervals of
    (x0, x1), each coded by its upper bound."""
    _arity("autocode", args, 4)
    x = np.asarray(_num(args[0], "autocode()"), dtype=float)
    n, lo, hi = (float(np.asarray(_num(a, "autocode()")).flat[0]) for a in args[1:])
    if n != n or lo != lo or hi != hi or n < 1 or int(n) != n or lo >= hi:
        return np.full(np.shape(x), np.nan)
    step = (hi - lo) / n
    out = np.full(np.shape(x), hi, dtype=float)
    with np.errstate(invalid="ignore"):
        for k in range(int(n) - 1, 0, -1):
            bound = lo + k * step
            out = np.where(x <= bound, bound, out)
    return np.where(np.isnan(x), np.nan, out)


def _f_clip(*args: Value) -> Value:
    _arity("clip", args, 3)
    x, lo, hi = (np.asarray(_num(a, "clip()"), dtype=float) for a in args)
    with np.errstate(invalid="ignore"):
        out = np.where(x < lo, lo, np.where(x > hi, hi, x))
    return np.where(np.isnan(x), np.nan, out)


def _f_reldif(*args: Value) -> Value:
    _arity("reldif", args, 2)
    x, y = (np.asarray(_num(a, "reldif()"), dtype=float) for a in args)
    return _clean(np.abs(x - y) / (np.abs(y) + 1))


# ----------------------------------------------------------- distributions
def _numeric(name: str, fn: Callable[..., Any], *allowed: int) -> Callable[..., Any]:
    def call(*args: Value) -> Value:
        _arity(name, args, *allowed)
        cols = [np.asarray(_num(a, f"{name}()"), dtype=float) for a in args]
        with np.errstate(all="ignore"):
            return _clean(fn(*cols))

    return call


def _binomial(n: Any, k: Any, p: Any) -> Any:
    return stats.binom.cdf(np.floor(k), n, p)


def _binomialtail(n: Any, k: Any, p: Any) -> Any:
    return stats.binom.sf(np.ceil(k) - 1, n, p)


# -------------------------------------------------------------- date parts
def _daily(value: Any, what: str) -> np.ndarray:
    d = np.asarray(_num(value, what), dtype=float)
    return d


def _ymd(d: np.ndarray) -> tuple:
    ok = np.isfinite(d)
    days = np.where(ok, np.floor(d), 0).astype("int64")
    dates = np.datetime64("1960-01-01") + days.astype("timedelta64[D]")
    years = dates.astype("datetime64[Y]")
    months = dates.astype("datetime64[M]")
    year = years.astype(int) + 1970
    month = months.astype(int) % 12 + 1
    doy = (dates - years.astype("datetime64[D]")).astype(int) + 1
    return ok, year, month, doy


def _doy(d: Any) -> Any:
    arr = np.asarray(d, dtype=float)
    ok, _, _, doy = _ymd(arr)
    return np.where(ok, doy.astype(float), np.nan)


def _week(d: Any) -> Any:
    # week 1 starts on 1 January; the 52nd week takes the last days
    return np.minimum(np.floor((_doy(d) - 1) / 7) + 1, 52)


def _halfyear(d: Any) -> Any:
    arr = np.asarray(d, dtype=float)
    ok, _, month, _ = _ymd(arr)
    return np.where(ok, np.where(month <= 6, 1.0, 2.0), np.nan)


def _year_part(per_year: int, part: Callable[[np.ndarray], Any]) -> Callable:
    def call(d: Any) -> Any:
        arr = np.asarray(d, dtype=float)
        ok, year, _, _ = _ymd(arr)
        return np.where(ok, (year - 1960) * per_year + part(arr) - 1, np.nan)

    return call


def _from_periods(per_year: int, start_day: Callable[[Any], Any]) -> Callable:
    def call(p: Any) -> Any:
        arr = np.asarray(p, dtype=float)
        ok = np.isfinite(arr)
        whole = np.where(ok, np.floor(arr), 0)
        year = 1960 + np.floor(whole / per_year)
        index = whole - (year - 1960) * per_year  # 0-based within the year
        jan1 = (
            (year.astype("int64") - 1970)
            .astype("datetime64[Y]")
            .astype("datetime64[D]")
            - np.datetime64("1960-01-01")
        ).astype(float)
        return np.where(ok, jan1 + start_day((year, index)), np.nan)

    return call


def _half_start(pair: Any) -> Any:
    year, index = pair
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    return np.where(index == 0, 0.0, 181.0 + leap)


def _two_field(name: str, per_year: int) -> Callable[..., Any]:
    def fn(y: Any, part: Any) -> Any:
        return np.where((part >= 1) & (part <= per_year),
                        (y - 1960) * per_year + part - 1, np.nan)  # fmt: skip

    return _numeric(name, fn, 2)


_MS_DAY = 86_400_000.0


def _hms(h: Any, m: Any, s: Any) -> Any:
    ok = (h >= 0) & (h < 24) & (m >= 0) & (m < 60) & (s >= 0) & (s < 60)
    return np.where(ok, h * 3_600_000 + m * 60_000 + np.round(s * 1000), np.nan)


def _dhms(d: Any, h: Any, m: Any, s: Any) -> Any:
    return d * _MS_DAY + _hms(h, m, s)


def _mdyhms(mo: Any, d: Any, y: Any, h: Any, m: Any, s: Any) -> Any:
    days = np.asarray(_expr._f_mdy(mo, d, y), dtype=float)
    return _dhms(days, h, m, s)


def _within_day(ms: Any) -> Any:
    return ms - np.floor(ms / _MS_DAY) * _MS_DAY


_CLOCK_TOKEN = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+")


def _parse_clock(text: Any, mask: str) -> float:
    """``clock(s, mask)``: the mask lists the fields in their order (Y, M,
    D, h, m, s); a time without a date is on 01jan1960."""
    if not isinstance(text, str):
        return np.nan
    fields = re.findall(r"[YMDhms]", mask)
    parts = _CLOCK_TOKEN.findall(text)
    suffix = [p.lower() for p in parts if p.lower() in ("am", "pm")]
    parts = [p for p in parts if p.lower() not in ("am", "pm")]
    if len(parts) != len(fields) or len(set(fields)) != len(fields):
        return np.nan
    got = dict(zip(fields, parts))
    try:
        day = 0.0
        if any(f in got for f in "YMD"):
            if not all(f in got for f in "YMD") or len(got["Y"]) != 4:
                return np.nan
            month = got["M"]
            m_num = (int(month) if month.isdigit()
                     else _expr._MONTH_NAMES.get(month[:3].lower(), 0))  # fmt: skip
            day = float(
                np.asarray(_expr._f_mdy(float(m_num), float(got["D"]), float(got["Y"])))
            )
        hour = float(got.get("h", 0))
        if suffix:
            if not 1 <= hour <= 12:
                return np.nan
            hour = hour % 12 + (12 if suffix[0] == "pm" else 0)
        ms = float(
            np.asarray(_hms(hour, float(got.get("m", 0)), float(got.get("s", 0))))
        )
    except ValueError:
        return np.nan
    return day * _MS_DAY + ms


def _f_clock(*args: Value) -> Value:
    if len(args) != 2 or not isinstance(args[1], str):
        raise StataExprError("clock() takes a string and a literal mask")
    if re.search(r"\d", args[1]):
        raise StataExprError(
            f"clock(): the mask {args[1]!r} names a century for two-digit "
            "years, which is not implemented"
        )
    source = args[0]
    if isinstance(source, str):
        return _parse_clock(source, args[1])
    if not _is_str(source):
        raise StataExprError("clock() needs a string, got a number")
    return np.array([_parse_clock(v, args[1]) for v in source], dtype=float)


# ---------------------------------------------------------- system values
_LIMITS: Dict[str, float] = {
    "pi": float(np.pi),
    "maxbyte": 100.0, "minbyte": -127.0,
    "maxint": 32740.0, "minint": -32767.0,
    "maxlong": 2147483620.0, "minlong": -2147483647.0,
    "maxfloat": 1.7014117331926443e38, "minfloat": -1.7014117331926443e38,
    "maxdouble": 8.98846567431158e307, "mindouble": -8.98846567431158e307,
    "epsfloat": 2.0**-23, "epsdouble": 2.0**-52,
    "smallestdouble": 2.0**-1022,
    "level": 95.0, "maxstrvarlen": 2045.0,
    "stata_version": 18.0, "version": 18.0,
}  # fmt: skip

SYSTEM_STRINGS: Dict[str, Callable[[], str]] = {
    "os": lambda: {"Darwin": "MacOSX", "Windows": "Windows"}.get(
        platform.system(), "Unix"
    ),
    "pwd": os.getcwd,
    "dirsep": lambda: "/",
    "Mons": lambda: " ".join(_MONTHS),
    "Months": lambda: " ".join(_MONTHS_FULL),
    "Wdays": lambda: " ".join(_DAYS),
    "Weekdays": lambda: " ".join(_DAYS_FULL),
    "alpha": lambda: " ".join("abcdefghijklmnopqrstuvwxyz"),
    "ALPHA": lambda: " ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
}


def system_value(name: str, n_obs: int, n_vars: int) -> Any:
    """``c(name)``: a number, a string, or ``None`` for a value that only a
    running Stata has (the date, the seed, a path of its installation)."""
    if name in _LIMITS:
        return _LIMITS[name]
    if name == "N":
        return float(n_obs)
    if name == "k":
        return float(n_vars)
    if name in SYSTEM_STRINGS:
        return SYSTEM_STRINGS[name]()
    return None


# ----------------------------------------------------------------- the table
_S = _string_function
_NEW: Dict[str, Callable[..., Any]] = {
    # strings
    "strlen": _S("strlen", lambda s: float(len(_bytes(_text(s, "strlen()")))), 1,
                 text=False),
    "ustrlen": _S("ustrlen", lambda s: float(len(_text(s, "ustrlen()"))), 1,
                  text=False),
    "strlower": _S("strlower", lambda s: _ascii_map(_text(s, "strlower()"),
                                                    str.lower), 1),
    "strupper": _S("strupper", lambda s: _ascii_map(_text(s, "strupper()"),
                                                    str.upper), 1),
    "ustrlower": _S("ustrlower", lambda s, *_l: _text(s, "ustrlower()").lower(),
                    1, 2),
    "ustrupper": _S("ustrupper", lambda s, *_l: _text(s, "ustrupper()").upper(),
                    1, 2),
    "strproper": _S("strproper", _proper, 1),
    "strtrim": _S("strtrim", lambda s: _text(s, "strtrim()").strip(" "), 1),
    "strltrim": _S("strltrim", lambda s: _text(s, "strltrim()").lstrip(" "), 1),
    "strrtrim": _S("strrtrim", lambda s: _text(s, "strrtrim()").rstrip(" "), 1),
    "stritrim": _S("stritrim", _itrim, 1),
    "ustrtrim": _S("ustrtrim", lambda s: _text(s, "ustrtrim()").strip(), 1),
    "strpos": _S("strpos", _strpos, 2, text=False),
    "strrpos": _S("strrpos", _strrpos, 2, text=False),
    "ustrpos": _S("ustrpos", _ustrpos, 2, text=False),
    "substr": _S("substr", _substr, 3),
    "usubstr": _S("usubstr", _usubstr, 3),
    "subinstr": _S("subinstr", _subinstr, 4),
    "subinword": _S("subinword", _subinword, 4),
    "word": _S("word", _word, 2),
    "wordcount": _S("wordcount", lambda s: float(len(_text(s, "wordcount()").split())),
                    1, text=False),
    "strreverse": _S("strreverse", lambda s: _text(s, "strreverse()")[::-1], 1),
    "ustrreverse": _S("ustrreverse", lambda s: _text(s, "ustrreverse()")[::-1], 1),
    "strmatch": _S("strmatch", _strmatch, 2, text=False),
    "strtoname": _S("strtoname", _strtoname, 1),
    "abbrev": _S("abbrev", _abbrev, 2),
    "strdup": _S("strdup", lambda s, n: _text(s, "strdup()")
                 * max(int(_number(n, "strdup()")), 0), 2),
    "char": _S("char", lambda n: chr(int(_number(n, "char()"))), 1),
    "uchar": _S("uchar", lambda n: chr(int(_number(n, "uchar()"))), 1),
    "regexm": _f_regexm,
    "regexs": _f_regexs,
    "regexr": _S("regexr", _regexr, 3),
    "ustrregexm": _S("ustrregexm", _ustrregexm, 2, 3, text=False),
    "ustrregexrf": _S("ustrregexrf", _ustrregexrf, 3, 4),
    "ustrregexra": _S("ustrregexra", _ustrregexra, 3, 4),
    "real": _S("real", _real, 1, text=False),
    "string": _S("string", _string, 1, 2),
    "strofreal": _S("strofreal", _string, 1, 2),
    # storage and grouping
    "float": _f_float,
    "recode": _f_recode,
    "irecode": _f_irecode,
    "autocode": _f_autocode,
    "clip": _f_clip,
    "reldif": _f_reldif,
    # distributions and counting
    "comb": _numeric("comb", lambda n, k: special.comb(n, k), 2),
    "lnfactorial": _numeric("lnfactorial", lambda n: special.gammaln(n + 1), 1),
    "binomial": _numeric("binomial", _binomial, 3),
    "binomialp": _numeric("binomialp", lambda n, k, p: stats.binom.pmf(k, n, p), 3),
    "binomialtail": _numeric("binomialtail", _binomialtail, 3),
    "poisson": _numeric("poisson", lambda m, k: stats.poisson.cdf(np.floor(k), m), 2),
    "poissonp": _numeric("poissonp", lambda m, k: stats.poisson.pmf(k, m), 2),
    "poissontail": _numeric(
        "poissontail", lambda m, k: stats.poisson.sf(np.ceil(k) - 1, m), 2
    ),
    "gammap": _numeric("gammap", lambda a, x: special.gammainc(a, x), 2),
    "ibeta": _numeric("ibeta", lambda a, b, x: special.betainc(a, b, x), 3),
    "invFtail": _numeric("invFtail", lambda a, b, p: stats.f.isf(p, a, b), 3),
    "invF": _numeric("invF", lambda a, b, p: stats.f.ppf(p, a, b), 3),
    "F": _numeric("F", lambda a, b, f: stats.f.cdf(f, a, b), 3),
    "Ftail": _numeric("Ftail", lambda a, b, f: stats.f.sf(f, a, b), 3),
    "Fden": _numeric("Fden", lambda a, b, f: stats.f.pdf(f, a, b), 3),
    "lnnormal": _numeric("lnnormal", stats.norm.logcdf, 1),
    "lnnormalden": _numeric("lnnormalden", stats.norm.logpdf, 1),
    "digamma": _numeric("digamma", special.digamma, 1),
    "cloglog": _numeric("cloglog", lambda p: np.log(-np.log1p(-p)), 1),
    "invcloglog": _numeric("invcloglog", lambda x: -np.expm1(-np.exp(x)), 1),
    # calendar and clock
    "doy": _numeric("doy", _doy, 1),
    "week": _numeric("week", _week, 1),
    "halfyear": _numeric("halfyear", _halfyear, 1),
    "wofd": _numeric("wofd", _year_part(52, _week), 1),
    "hofd": _numeric("hofd", _year_part(2, _halfyear), 1),
    "dofw": _numeric("dofw", _from_periods(52, lambda pair: pair[1] * 7.0), 1),
    "dofh": _numeric("dofh", _from_periods(2, _half_start), 1),
    "yw": _two_field("yw", 52),
    "yh": _two_field("yh", 2),
    "hms": _numeric("hms", _hms, 3),
    "dhms": _numeric("dhms", _dhms, 4),
    "mdyhms": _numeric("mdyhms", _mdyhms, 6),
    "hh": _numeric("hh", lambda ms: np.floor(_within_day(ms) / 3_600_000), 1),
    "mm": _numeric("mm", lambda ms: np.floor(_within_day(ms) / 60_000) % 60, 1),
    "ss": _numeric("ss", lambda ms: (_within_day(ms) % 60_000) / 1000, 1),
    "hours": _numeric("hours", lambda ms: ms / 3_600_000, 1),
    "minutes": _numeric("minutes", lambda ms: ms / 60_000, 1),
    "seconds": _numeric("seconds", lambda ms: ms / 1000, 1),
    "msofhours": _numeric("msofhours", lambda h: h * 3_600_000, 1),
    "msofminutes": _numeric("msofminutes", lambda m: m * 60_000, 1),
    "msofseconds": _numeric("msofseconds", lambda s: s * 1000, 1),
    "dofc": _numeric("dofc", lambda ms: np.floor(ms / _MS_DAY), 1),
    "dofC": _numeric("dofC", lambda ms: np.floor(ms / _MS_DAY), 1),
    "cofd": _numeric("cofd", lambda d: d * _MS_DAY, 1),
    "Cofd": _numeric("Cofd", lambda d: d * _MS_DAY, 1),
    "clock": _f_clock,
    "Clock": _f_clock,
}  # fmt: skip

#: the names Stata used before version 13 (still accepted)
_OLD_NAMES = {
    "length": "strlen", "lower": "strlower", "upper": "strupper",
    "proper": "strproper", "trim": "strtrim", "ltrim": "strltrim",
    "rtrim": "strrtrim", "itrim": "stritrim", "index": "strpos",
    "reverse": "strreverse", "match": "strmatch",
}  # fmt: skip
for _old, _current in _OLD_NAMES.items():
    _NEW[_old] = _NEW[_current]
for _name, _fn in _NEW.items():
    _expr._FUNCTIONS.setdefault(_name, _fn)
