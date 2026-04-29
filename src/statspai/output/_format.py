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

from typing import Any, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "format_stars",
    "fmt_val",
    "fmt_int",
    "fmt_auto",
    "is_missing",
]


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
    if is_missing(pvalue):
        return ""
    stars = ""
    for lev in sorted(levels, reverse=True):
        if pvalue < lev:
            stars += "*"
    return stars


def fmt_auto(value: float) -> str:
    """Magnitude-adaptive numeric formatting.

    Picks decimal precision per ``|value|`` so a single table can mix
    dollar-magnitude coefficients (e.g. ``1521``) and elasticity-magnitude
    coefficients (e.g. ``0.288``) without one side being rounded to zero.
    """
    if is_missing(value):
        return ""
    av = abs(float(value))
    if av >= 1000:
        return f"{value:,.0f}"
    if av >= 100:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.1f}"
    if av >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_val(value: Any, fmt: str = "%.4f") -> str:
    """Format a numeric value, returning ``""`` for missing / non-finite.

    ``fmt`` is a printf-style template (e.g. ``"%.3f"``); the sentinel
    ``"auto"`` switches to :func:`fmt_auto`.
    """
    if is_missing(value):
        return ""
    if fmt == "auto":
        return fmt_auto(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(f):
        return ""
    return fmt % f


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
