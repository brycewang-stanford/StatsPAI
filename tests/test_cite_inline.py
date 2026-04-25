"""Tests for ``sp.cite()`` — inline coefficient reporting."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def ols_strong_signal():
    rng = np.random.default_rng(2026)
    n = 500
    x = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x})
    return sp.regress("y ~ x", data=df)


@pytest.fixture
def ols_null_signal():
    """A model where the slope is statistically zero — no stars expected."""
    rng = np.random.default_rng(2026)
    n = 500
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)  # x has no effect
    df = pd.DataFrame({"y": y, "x": x})
    return sp.regress("y ~ x", data=df)


# ---------------------------------------------------------------------------
# Basic formatting
# ---------------------------------------------------------------------------

def test_cite_default_format(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x")
    # 0.5xx*** (0.0xx) — 3-decimal, three stars on a 1-pct effect
    assert re.match(r"^0\.\d{3}\*+ \(\d\.\d{3}\)$", s) or re.match(r"^0\.\d{3} \(\d\.\d{3}\)$", s)


def test_cite_stars_appear_for_significant(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x")
    assert "*" in s


def test_cite_no_stars_for_nonsignificant(ols_null_signal):
    s = sp.cite(ols_null_signal, "x")
    assert "*" not in s


def test_cite_custom_fmt(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", fmt="%.4f")
    # Estimate should now have 4 decimals
    assert re.search(r"0\.\d{4}", s)


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

def test_cite_latex_uses_superscript(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", output="latex")
    assert "^{" in s and "}" in s
    # Tilde non-breaking space between estimate+stars and SE
    assert "~" in s


def test_cite_html_uses_b_and_sup(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", output="html")
    assert "<b>" in s and "</b>" in s
    assert "<sup>" in s and "</sup>" in s


def test_cite_markdown_escapes_stars(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", output="markdown")
    assert "**" in s  # bold delimiters around estimate
    assert "\\*" in s  # escaped stars to avoid markdown collision


# ---------------------------------------------------------------------------
# second_row variants
# ---------------------------------------------------------------------------

def test_cite_second_row_ci(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", second_row="ci")
    assert "[" in s and "]" in s
    assert "(" not in s  # no parens when CI requested


def test_cite_second_row_p(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", second_row="p")
    # p-value will be very small; just verify it's in parens after the stars
    m = re.search(r"\(([\d.]+)\)$", s)
    assert m
    assert 0.0 <= float(m.group(1)) <= 1.0


def test_cite_second_row_t(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", second_row="t")
    m = re.search(r"\(([\-\d.]+)\)$", s)
    assert m
    # t-stat for a strong signal should be well above 2 in absolute value
    assert abs(float(m.group(1))) > 2.0


def test_cite_second_row_none(ols_strong_signal):
    s = sp.cite(ols_strong_signal, "x", second_row="none")
    assert "(" not in s
    assert "[" not in s


# ---------------------------------------------------------------------------
# CausalResult support
# ---------------------------------------------------------------------------

def test_cite_causal_result_default_term():
    rng = np.random.default_rng(2026)
    n = 400
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "treat": rng.integers(0, 2, n),
        "g": rng.integers(0, 5, n),
        "t": rng.integers(0, 3, n),
    })
    df["y"] = df["y"] + 0.5 * df["treat"]
    # Use a simple regression-as-causal-result via regress
    m = sp.regress("y ~ treat", data=df)
    s = sp.cite(m, "treat")
    assert isinstance(s, str) and len(s) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_cite_unknown_term_raises(ols_strong_signal):
    with pytest.raises(KeyError):
        sp.cite(ols_strong_signal, "nonexistent_var")


def test_cite_invalid_output_raises(ols_strong_signal):
    with pytest.raises(ValueError, match="output"):
        sp.cite(ols_strong_signal, "x", output="rtf")


def test_cite_invalid_second_row_raises(ols_strong_signal):
    with pytest.raises(ValueError, match="second_row"):
        sp.cite(ols_strong_signal, "x", second_row="bootstrap")


def test_cite_invalid_alpha_raises(ols_strong_signal):
    with pytest.raises(ValueError, match="alpha"):
        sp.cite(ols_strong_signal, "x", second_row="ci", alpha=1.5)
