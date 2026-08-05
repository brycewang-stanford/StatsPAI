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
        (1521.109, "1,521"),  # >= 1000 -> thousands separator, no decimal
        (-1521.109, "-1,521"),
        (590.769, "591"),  # >= 100 -> integer
        (30.925, "30.9"),  # >= 10 -> 1 decimal
        (3.955, "3.96"),  # >= 1 -> 2 decimals
        # NOTE: Python uses banker's rounding (round-half-to-even) on
        # IEEE-754 doubles. 0.2876 rounds to 0.288; 0.2875 actually
        # stores as 0.28749999... so round-half-to-even drops to 0.287.
        # The expected values below reflect Python's actual behavior.
        (0.2876, "0.288"),
        (-0.0106, "-0.011"),
        # Three decimals is the sub-unit convention, but the floor lifts
        # rather than rounding a value away: pre-1.22 this printed "0.000".
        (0.00042, "0.00042"),
        (0.0, "0.000"),  # zero: no scale to read off
    ],
)
def test_fmt_auto_buckets(value, expected):
    assert _fmt_auto(value) == expected


def test_fmt_auto_handles_nan_and_none():
    assert _fmt_auto(float("nan")) == ""
    assert _fmt_auto(None) == ""


def test_fmt_auto_never_renders_a_nonzero_value_as_zero():
    """A value under the decimal ceiling escapes to scientific notation.

    Printing ``0.000000`` for a nonzero estimate is the exact failure mode
    adaptive precision exists to prevent, so the renderer must not do it at
    the bottom of its own range either.
    """
    out = _fmt_auto(1e-9)
    assert "e-" in out
    assert float(out) == pytest.approx(1e-9)


# ---------------------------------------------------------------------------
# 2. _fmt_val routing: "auto" delegates, others go through % formatting
# ---------------------------------------------------------------------------


def test_fmt_val_auto_routing():
    """fmt='auto' uses _fmt_auto; explicit C-style preserved unchanged."""
    assert _fmt_val(0.2876, "auto") == "0.288"
    assert _fmt_val(0.2876, "%.0f") == "0"  # legacy behavior preserved
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
    assert (
        "0." in cell or "." in cell
    ), f"fmt='auto' should keep decimals for ~0.3 coef; got {cell!r}"


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
        f"%.0f should round ~0.3 coef to 0/-0 (the bug being fixed); " f"got {cell!r}"
    )


def test_regtable_default_is_auto(mixed_magnitude_data):
    """The default is adaptive precision — identical to fmt='auto'."""
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    assert sp.regtable(m).to_markdown() == sp.regtable(m, fmt="auto").to_markdown()


def test_regtable_explicit_fmt_still_wins(mixed_magnitude_data):
    """An explicit printf template overrides the adaptive default verbatim."""
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    md = sp.regtable(m, fmt="%.4f").to_markdown()
    cell = [ln for ln in md.split("\n") if "x_small" in ln][0].split("|")[2].strip()
    assert len(cell.split(".")[1].rstrip("*")) == 4


def test_regtable_digits_alias(mixed_magnitude_data):
    """digits=N is shorthand for fmt='%.Nf'."""
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    assert sp.regtable(m, digits=2).to_text() == sp.regtable(m, fmt="%.2f").to_text()


def test_regtable_rejects_fmt_and_digits_together(mixed_magnitude_data):
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    with pytest.raises(sp.MethodIncompatibility, match="not both"):
        sp.regtable(m, fmt="%.2f", digits=2)


def test_regtable_rejects_unusable_fmt(mixed_magnitude_data):
    """A bad fmt fails at the call site, not mid-render."""
    m = sp.regress("y ~ x_small + x_large", data=mixed_magnitude_data)
    with pytest.raises(sp.MethodIncompatibility, match="fmt"):
        sp.regtable(m, fmt=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3b. Pairing: a coefficient and its SE share one decimal place
# ---------------------------------------------------------------------------


def _decimals_of(text: str) -> int:
    text = text.replace(",", "").strip()
    return len(text.split(".")[1]) if "." in text else 0


@pytest.fixture
def lalonde_like_data():
    """Dollar-scale outcome with regressors of three different magnitudes.

    Reproduces the shape of the LaLonde NSW earnings table: a treatment
    dummy with a ~$1,500 effect, an age coefficient in the single digits
    with a two-digit SE, and a lagged-earnings coefficient near zero.
    """
    rng = np.random.default_rng(11)
    n = 445
    df = pd.DataFrame(
        {
            "treat": rng.integers(0, 2, n),
            "age": rng.integers(17, 55, n),
            "re75": rng.normal(3000, 5000, n),
        }
    )
    df["re78"] = (
        1500 * df["treat"] - 5 * df["age"] + 0.3 * df["re75"] + rng.normal(0, 5000, n)
    )
    return df


def test_auto_pairs_coef_and_se_decimals(lalonde_like_data):
    """Every rendered row shows coef and SE at the same decimal place.

    The pre-1.22 ``fmt='auto'`` chose precision per *cell*, producing rows
    like ``-5.22 (45.3)`` where the estimate and its own standard error
    disagreed — not a convention any economics journal follows.
    """
    m = sp.regress("re78 ~ treat + age + re75", data=lalonde_like_data)
    lines = sp.regtable(m).to_markdown().split("\n")
    checked = 0
    for i, line in enumerate(lines):
        cell = line.split("|")[2].strip() if line.count("|") > 2 else ""
        if not cell or not any(c.isdigit() for c in cell):
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        se_cell = nxt.split("|")[2].strip() if nxt.count("|") > 2 else ""
        if not (se_cell.startswith("(") and se_cell.endswith(")")):
            continue
        checked += 1
        coef_dec = _decimals_of(cell.rstrip("*"))
        se_dec = _decimals_of(se_cell.strip("()"))
        assert coef_dec == se_dec, (
            f"coefficient {cell!r} and its SE {se_cell!r} disagree on "
            f"precision ({coef_dec} vs {se_dec} decimals)"
        )
    assert checked >= 3, f"expected several coef/SE pairs, checked {checked}"


def test_auto_keeps_row_aligned_across_models(lalonde_like_data):
    """One decimal count per row, shared by every model column."""
    m1 = sp.regress("re78 ~ treat", data=lalonde_like_data)
    m2 = sp.regress("re78 ~ treat + age + re75", data=lalonde_like_data)
    md = sp.regtable(m1, m2).to_markdown()
    row = next(ln for ln in md.split("\n") if ln.strip().startswith("| treat"))
    cells = [c.strip().rstrip("*") for c in row.split("|")[2:4]]
    decs = {_decimals_of(c) for c in cells if any(ch.isdigit() for ch in c)}
    assert len(decs) == 1, f"treat row has ragged precision across models: {cells}"


def test_se_fmt_can_break_the_pairing(lalonde_like_data):
    """se_fmt= is the deliberate escape hatch from paired precision."""
    m = sp.regress("re78 ~ treat + age", data=lalonde_like_data)
    lines = sp.regtable(m, fmt="%.0f", se_fmt="%.2f").to_markdown().split("\n")
    idx = [i for i, ln in enumerate(lines) if "treat" in ln][0]
    assert _decimals_of(lines[idx + 1].split("|")[2].strip().strip("()")) == 2


def test_stats_fmt_is_independent_of_fmt(lalonde_like_data):
    """R² / F follow stats_fmt, not fmt (they are on their own scale)."""
    m = sp.regress("re78 ~ treat + age", data=lalonde_like_data)
    txt = sp.regtable(m, fmt="%.1f", stats_fmt="%.4f").to_text()
    r2_line = next(ln for ln in txt.split("\n") if ln.strip().startswith("R²"))
    assert _decimals_of(r2_line.split()[-1]) == 4


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
    x_small_segment = next((ln for ln in text.split("\n") if "x_small" in ln), "")
    assert x_small_segment, "x_small row not found in modelsummary output"
    assert "0." in x_small_segment or "." in x_small_segment, (
        f"fmt='auto' should keep decimals on ~0.3 coef in modelsummary "
        f"layer; got {x_small_segment!r}"
    )
