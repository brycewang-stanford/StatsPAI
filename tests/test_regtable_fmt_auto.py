"""Tests for ``fmt="auto"`` magnitude-adaptive precision in ``sp.regtable``.

Repro for the LaLonde QJE-table bug surfaced 2026-04-25:
when a single table mixes dollar-magnitude (~$1500) and
elasticity-magnitude (~0.3) coefficients, fixed ``fmt="%.0f"``
rounds the latter to ``0`` even though significance stars survive.
``fmt="auto"`` picks per-value precision so neither side is killed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output.estimates import _fmt_auto, _fmt_val


# ---------------------------------------------------------------------------
# 1. Unit tests for _fmt_auto bucketing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (1521.109, "1,521"),    # >= 1000 -> thousands separator, no decimal
        (-1521.109, "-1,521"),
        (590.769, "591"),       # >= 100 -> integer
        (30.925, "30.9"),       # >= 10 -> 1 decimal
        (3.955, "3.96"),        # >= 1 -> 2 decimals
        # NOTE: Python uses banker's rounding (round-half-to-even) on
        # IEEE-754 doubles. 0.2876 rounds to 0.288; 0.2875 actually
        # stores as 0.28749999... so round-half-to-even drops to 0.287.
        # The expected values below reflect Python's actual behavior.
        (0.2876, "0.288"),
        (-0.0106, "-0.011"),
        (0.0, "0.000"),         # zero
    ],
)
def test_fmt_auto_buckets(value, expected):
    assert _fmt_auto(value) == expected


def test_fmt_auto_handles_nan_and_none():
    assert _fmt_auto(float("nan")) == ""
    assert _fmt_auto(None) == ""


# ---------------------------------------------------------------------------
# 2. _fmt_val routing: "auto" delegates, others go through % formatting
# ---------------------------------------------------------------------------


def test_fmt_val_auto_routing():
    """fmt='auto' uses _fmt_auto; explicit C-style preserved unchanged."""
    assert _fmt_val(0.2876, "auto") == "0.288"
    assert _fmt_val(0.2876, "%.0f") == "0"          # legacy behavior preserved
    assert _fmt_val(0.2876, "%.3f") == "0.288"
    assert _fmt_val(1521.109, "auto") == "1,521"
    assert _fmt_val(1521.109, "%.0f") == "1521"


# ---------------------------------------------------------------------------
# 3. End-to-end: regtable with mixed-magnitude coefficients
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_magnitude_data():
    """Synthetic: y is dollar-scale; true coef on x_small is 0.3, on x_large is 5.

    Mirrors the LaLonde production-table problem: one regressor has an
    elasticity-magnitude coefficient, another has a per-unit coefficient
    in the single-digit range, and the intercept dominates. ``%.0f``
    fixed format would round x_small's coefficient to 0; ``"auto"``
    must keep three decimals so the digit survives.
    """
    rng = np.random.default_rng(42)
    n = 500
    x_small = rng.normal(0, 1, n)
    x_large = rng.normal(50, 10, n)
    eps = rng.normal(0, 1, n)
    y = 1500 + 0.3 * x_small + 5 * x_large + eps
    return pd.DataFrame({"y": y, "x_small": x_small, "x_large": x_large})


def test_regtable_fmt_auto_preserves_small_coefs(mixed_magnitude_data):
    """fmt='auto' must NOT round 0.X coefficients to 0."""
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    md = sp.regtable(m, fmt="auto").to_markdown()
    x_small_lines = [ln for ln in md.split("\n") if "x_small" in ln]
    assert x_small_lines, "x_small row not found in markdown output"
    # The coefficient cell must contain a "0.X" decimal pattern, not bare "0"
    line = x_small_lines[0]
    cell = line.split("|")[2].strip()  # markdown col 1 (after row label)
    assert "0." in cell or "." in cell, (
        f"fmt='auto' should keep decimals for ~0.3 coef; got {cell!r}"
    )


def test_regtable_fmt_pct0_kills_small_coefs(mixed_magnitude_data):
    """Regression test: fmt='%.0f' DOES kill small coefs (legacy behavior).

    This documents the precise bug pattern that ``fmt='auto'`` fixes —
    a small coefficient gets rounded to ``0`` while its significance
    stars survive, leaving readers staring at ``0***`` cells.
    """
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    md = sp.regtable(m, fmt="%.0f").to_markdown()
    x_small_lines = [ln for ln in md.split("\n") if "x_small" in ln]
    assert x_small_lines
    line = x_small_lines[0]
    cell = line.split("|")[2].strip()
    # Under %.0f the ~0.3 coefficient is killed: digit is "0" with stars,
    # no decimal point, no significant digits.
    assert cell.lstrip("-").startswith("0") and "." not in cell, (
        f"%.0f should round ~0.3 coef to 0/-0 (the bug being fixed); "
        f"got {cell!r}"
    )


def test_regtable_default_fmt_unchanged(mixed_magnitude_data):
    """Default fmt='%.3f' behavior is untouched by fmt='auto' addition."""
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    md = sp.regtable(m).to_markdown()  # no fmt arg -> default %.3f
    x_small_lines = [ln for ln in md.split("\n") if "x_small" in ln]
    assert x_small_lines
    cell = x_small_lines[0].split("|")[2].strip()
    # %.3f keeps 3 decimals on a ~0.3 coef
    assert "." in cell


# ---------------------------------------------------------------------------
# 4. modelsummary-style layer: fmt='auto' parity
# ---------------------------------------------------------------------------


def test_modelsummary_fmt_auto_parity(mixed_magnitude_data):
    """sp.modelsummary (R-style layer) must accept fmt='auto' too.

    ``_format_num`` in modelsummary.py and ``_fmt_val`` in estimates.py
    are independent code paths; both must honor ``fmt='auto'`` so users
    get consistent behavior across the two style layers.
    """
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    out = sp.modelsummary(m, fmt="auto", output="markdown")
    # Coerce to text whatever the renderer returns (str or DataFrame).
    text = out if isinstance(out, str) else str(out)
    assert text.strip(), "modelsummary returned empty output"
    # x_small ~ 0.3 should NOT be rounded to bare 0 under fmt='auto'
    x_small_segment = next(
        (ln for ln in text.split("\n") if "x_small" in ln), ""
    )
    assert x_small_segment, "x_small row not found in modelsummary output"
    assert "0." in x_small_segment or "." in x_small_segment, (
        f"fmt='auto' should keep decimals on ~0.3 coef in modelsummary "
        f"layer; got {x_small_segment!r}"
    )
