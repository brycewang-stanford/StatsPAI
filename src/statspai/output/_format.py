"""Canonical numeric / significance formatters for the output package.

Historically each backend (``regtable`` / ``esttab`` / ``modelsummary`` /
``outreg2``) carried its own ``_format_stars`` / ``_fmt_val`` / ``_fmt_int``
implementations. Most were byte-for-byte equivalent; a few had legitimate
semantic differences (input robustness, rounding, dict-based thresholds).

This module owns the **canonical** versions used by every backend whose
behavior is identical. Backends that need different semantics keep their
own helpers but justify the divergence in a comment.

Public API for the ``output`` package only — names are kept short and
underscore-prefixed-when-imported externally to discourage cross-package
use. Stability contract: signatures and outputs are part of the rendered
result and must not change without a CHANGELOG entry.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

__all__ = [
    "format_stars",
    "fmt_val",
    "fmt_int",
    "fmt_auto",
    "fmt_fixed",
    "sig_decimals",
    "value_decimals",
    "auto_decimals",
    "normalize_fmt",
    "is_missing",
    "AUTO",
    "AUTO_PREFIX",
    "AUTO_SIG_DIGITS",
    "SUBUNIT_DECIMALS",
    "MAX_AUTO_DECIMALS",
]

#: Sentinel requesting magnitude-adaptive precision.
AUTO = "auto"

#: Prefix for a *resolved* auto format, ``"auto:<decimals>"``. Produced by the
#: renderer once it has decided the precision for a coefficient/SE pair, and
#: consumed by :func:`fmt_val`. Keeping the resolved form a plain string means
#: the existing ``fmt``-threading (including ``to_dict`` round-trips) needs no
#: type changes.
AUTO_PREFIX = "auto:"

#: Significant digits targeted for values at or above 1.0. Three is the
#: working convention in applied-economics tables: ``1521`` prints as
#: ``1,521``, ``30.9`` keeps one decimal, ``3.96`` keeps two.
AUTO_SIG_DIGITS = 3

#: Decimals given to sub-unit values. Three is what published tables use for
#: elasticities, log-points and shares (``0.288``, ``0.051``, ``0.012``); the
#: floor lifts only when three would round the value away entirely.
SUBUNIT_DECIMALS = 3

#: Hard ceiling on auto-selected decimals, so a pathologically small standard
#: error cannot blow a column up to twenty decimal places.
MAX_AUTO_DECIMALS = 6


def is_missing(value: Any) -> bool:
    """Return ``True`` for ``None`` / NaN / pandas-NA scalars.

    Centralised so backends do not each reinvent the
    ``isinstance(..., float) and np.isnan(...)`` dance.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_stars(
    pvalue: float,
    levels: Tuple[float, ...] = (0.10, 0.05, 0.01),
) -> str:
    """Return significance stars (``"***"`` / ``"**"`` / ``"*"`` / ``""``).

    ``levels`` are the cutoffs **from least to most strict**. The default
    ``(0.10, 0.05, 0.01)`` yields ``*`` for ``p < 0.10``, ``**`` for
    ``p < 0.05``, ``***`` for ``p < 0.01`` — the convention shared by
    ``regtable`` / ``esttab`` / ``outreg2``.
    """
    if is_missing(pvalue) or pvalue < 0:
        return ""
    stars = ""
    for lev in sorted(levels, reverse=True):
        if pvalue < lev:
            stars += "*"
    return stars


def sig_decimals(
    value: Any,
    sig: int = AUTO_SIG_DIGITS,
    max_decimals: int = MAX_AUTO_DECIMALS,
) -> Optional[int]:
    """Decimals needed to show ``|value|`` with *sig* significant digits.

    ``589`` → ``0`` (prints ``589``), ``45.3`` → ``1``, ``0.104`` → ``3``,
    ``0.0104`` → ``4``. Never negative: the integer part is always shown in
    full, so ``1084`` → ``0`` rather than ``-1``.

    Returns ``None`` for missing / non-finite / zero input — callers decide
    the fallback, because "no information about scale" is not the same as
    "zero decimals".
    """
    if is_missing(value):
        return None
    try:
        av = abs(float(value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(av) or av == 0.0:
        return None
    return int(min(max(0, sig - 1 - math.floor(math.log10(av))), max_decimals))


def value_decimals(
    value: Any,
    sig: int = AUTO_SIG_DIGITS,
    max_decimals: int = MAX_AUTO_DECIMALS,
) -> Optional[int]:
    """Decimals a *single* value wants, following table convention.

    Two regimes, because published tables treat them differently:

    - ``|value| >= 1`` — *sig* significant digits, so ``1521`` → ``0``
      decimals, ``30.9`` → ``1``, ``3.96`` → ``2``.
    - ``|value| < 1`` — three decimals, the near-universal convention for
      sub-unit estimates (``0.288``, ``0.051``, ``0.012``). The floor lifts
      only when three decimals would round the value away entirely, so
      ``0.00042`` gets five rather than collapsing to ``0.000``.

    Returns ``None`` for missing / non-finite / zero input.
    """
    if is_missing(value):
        return None
    try:
        av = abs(float(value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(av) or av == 0.0:
        return None
    if av >= 1.0:
        return sig_decimals(av, sig=sig, max_decimals=max_decimals)
    # Two significant digits is the point at which a sub-unit value stops
    # being legible at SUBUNIT_DECIMALS; below that, extend.
    needed = sig_decimals(av, sig=2, max_decimals=max_decimals) or SUBUNIT_DECIMALS
    return int(min(max(SUBUNIT_DECIMALS, needed), max_decimals))


def auto_decimals(
    coef: Any = None,
    se: Any = None,
    sig: int = AUTO_SIG_DIGITS,
    max_decimals: int = MAX_AUTO_DECIMALS,
) -> int:
    """Decimals shared by one coefficient/standard-error **pair**.

    Takes the finer of what the two halves want and prints both at that
    single decimal place — the convention in published economics tables,
    where a row reads ``1,556 (589)`` or ``0.288 (0.146)``, never
    ``-5.22 (45.3)`` with the estimate and its own standard error
    disagreeing about precision.

    Taking the *maximum* rather than the standard error's requirement alone
    is what keeps ``1.071 (0.054)`` intact: the coefficient by itself would
    settle for two decimals and silently truncate the SE to ``0.05``.

    Falls back to whichever half is usable when the other is missing, zero,
    or non-finite (``se_type='t'`` columns, constrained parameters), and to
    :data:`SUBUNIT_DECIMALS` when neither is.
    """
    wants = [
        d
        for d in (
            value_decimals(coef, sig=sig, max_decimals=max_decimals),
            value_decimals(se, sig=sig, max_decimals=max_decimals),
        )
        if d is not None
    ]
    if not wants:
        return int(min(SUBUNIT_DECIMALS, max_decimals))
    return int(max(wants))


def fmt_fixed(
    value: Any,
    decimals: int,
    thousands: bool = True,
    sci_fallback: bool = False,
) -> str:
    """Render *value* at exactly *decimals* places, with thousands separators.

    Separate from ``"%.<n>f"`` because printf has no thousands-separator
    flag, and published tables group thousands (``1,556``, not ``1556``).

    ``sci_fallback`` switches to scientific notation when rounding would
    render a nonzero value as all zeros. It is off by default because in a
    coefficient/SE pair a coefficient that vanishes against its own standard
    error genuinely *is* zero at the displayed precision, and ``1.00e-08``
    beside ``(0.0500)`` reads as noise. :func:`fmt_auto` turns it on, having
    no such context to lean on.
    """
    if is_missing(value):
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(f):
        return ""
    d = max(0, int(decimals))
    out = f"{f:,.{d}f}" if thousands else f"{f:.{d}f}"
    if sci_fallback and f != 0.0 and not any(ch in "123456789" for ch in out):
        # Rounding annihilated a nonzero estimate. Printing "0.000000" would
        # read as an exact zero — the precise failure mode adaptive precision
        # exists to prevent — so escape rather than lie.
        return f"{f:.{max(0, AUTO_SIG_DIGITS - 1)}e}"
    return out


def fmt_auto(value: float) -> str:
    """Magnitude-adaptive formatting of a *single* value.

    Used where no coefficient/SE pair exists (summary-statistic cells,
    extra-SE brackets). Applies the same ladder as :func:`value_decimals`,
    so ``1521.1`` → ``1,521``, ``30.925`` → ``30.9``, ``0.2876`` →
    ``0.288``. Unlike the pre-1.22 ladder, which bottomed out at three
    decimals, ``0.00042`` → ``0.00042`` instead of collapsing to ``0.000``.

    Prefer :func:`auto_decimals` + :func:`fmt_fixed` when a standard error
    is available: pairing beats per-cell guessing.
    """
    if is_missing(value):
        return ""
    d = value_decimals(value)
    if d is None:
        # Exact zero: no scale to read off. Print ``0.000`` so the cell lines
        # up with its elasticity-magnitude neighbours instead of a bare "0".
        d = SUBUNIT_DECIMALS
    return fmt_fixed(value, d, sci_fallback=True)


def _auto_decimals_from_fmt(fmt: str) -> Optional[int]:
    """Parse ``"auto:<n>"`` → ``n``; return ``None`` when *fmt* is not one."""
    if not isinstance(fmt, str) or not fmt.startswith(AUTO_PREFIX):
        return None
    try:
        return max(0, int(fmt[len(AUTO_PREFIX) :]))
    except (TypeError, ValueError):
        return None


def fmt_val(value: Any, fmt: str = "%.4f") -> str:
    """Format a numeric value, returning ``""`` for missing / non-finite.

    ``fmt`` is a printf-style template (e.g. ``"%.3f"``), the sentinel
    ``"auto"`` (per-value adaptive precision, :func:`fmt_auto`), or a
    resolved ``"auto:<n>"`` produced by the renderer once it has picked the
    precision for a coefficient/SE pair.
    """
    if is_missing(value):
        return ""
    if fmt == AUTO:
        return fmt_auto(value)
    resolved = _auto_decimals_from_fmt(fmt)
    if resolved is not None:
        return fmt_fixed(value, resolved)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(f):
        return ""
    return fmt % f


def normalize_fmt(fmt: Any, param: str = "fmt") -> str:
    """Coerce user input into a format string :func:`fmt_val` understands.

    Accepts an ``int`` (``3`` → ``"%.3f"``) because that is what everyone
    reaches for first, the ``"auto"`` sentinel, or a printf template. Any
    other input raises here — naming the offending parameter — rather than
    surfacing as a ``TypeError`` from inside the renderer's string
    concatenation several frames down.
    """
    if isinstance(fmt, bool):  # bool is an int subclass; almost surely a typo
        raise MethodIncompatibility(
            f"{param}={fmt!r} is a bool. Pass a decimal count (e.g. "
            f"{param}=3), the string 'auto', or a printf template "
            f"such as '%.3f'.",
            recovery_hint=f"Use {param}=3 or {param}='auto'.",
            diagnostics={param: fmt},
        )
    if isinstance(fmt, (int, np.integer)):
        n = int(fmt)
        if n < 0 or n > 20:
            raise MethodIncompatibility(
                f"{param}={n} is outside the supported range 0-20 decimals.",
                recovery_hint=f"Use e.g. {param}=3.",
                diagnostics={param: n},
            )
        return f"%.{n}f"
    if isinstance(fmt, str):
        if fmt == AUTO or _auto_decimals_from_fmt(fmt) is not None:
            return fmt
        # Validate eagerly: a bad template must fail at the call site, not
        # halfway through rendering a 40-row table.
        try:
            fmt % 1.0
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"{param}={fmt!r} is not a valid printf float template " f"({exc}).",
                recovery_hint=(
                    f"Use {param}='%.3f' for fixed precision, {param}=3 for "
                    f"the same thing, or {param}='auto' for journal-style "
                    f"adaptive precision."
                ),
                diagnostics={param: fmt},
            ) from exc
        return fmt
    raise MethodIncompatibility(
        f"{param} must be an int, 'auto', or a printf template; got "
        f"{type(fmt).__name__}.",
        recovery_hint=f"Use {param}=3, {param}='auto', or {param}='%.3f'.",
        diagnostics={param: repr(fmt)},
    )


def fmt_int(value: Any) -> str:
    """Format an integer-valued cell with thousands separators.

    Floats are rounded to the nearest integer (mirrors how Stata / fixest
    print ``N`` derived from weighted observation counts). Returns ``""``
    for missing / non-numeric input.
    """
    if is_missing(value):
        return ""
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return ""
