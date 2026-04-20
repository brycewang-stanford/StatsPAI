"""Regression tests for 0.9.3 bug fixes reported in production.

Covers:
1. `use_chinese()` auto-detects Linux-common Chinese fonts
   (Noto Sans/Serif CJK JP/TC/KR, WenQuanYi Zen Hei, Source Han Sans/Serif).
2. `RegtableResult` / `MeanComparisonResult` / `EstimateTableResult`
   do not auto-print during construction (no double-print in REPL/Jupyter).
3. `regtable(..., output="latex")` actually returns LaTeX from `str(result)`.
4. `did()` docstring spells out the `treat` semantics (first_treat period
   for staggered; 0/1 indicator for 2x2) with a runnable example.
"""

from __future__ import annotations

import contextlib
import io
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Bug 1 — use_chinese() on Linux with JP/TC/Zen Hei / Source Han fonts
# ---------------------------------------------------------------------------


def _fake_fontmanager(names):
    """Return an object mimicking matplotlib's fontManager with given names."""
    return SimpleNamespace(ttflist=[SimpleNamespace(name=n) for n in names])


def test_use_chinese_picks_noto_jp_on_linux():
    """Noto Sans CJK JP covers CJK; auto-detect must pick it up on Linux.

    Also pins the priority order: sans is preferred over serif in 'auto' mode.
    """
    from statspai.plots import themes
    import matplotlib.font_manager as _fm

    fake = _fake_fontmanager([
        "DejaVu Sans", "Liberation Serif",
        "Noto Sans CJK JP", "Noto Serif CJK JP",
        "WenQuanYi Zen Hei",
    ])
    with patch.object(_fm, "fontManager", fake):
        chosen = themes.use_chinese()
    # Sans-preferred in auto mode, and Noto Sans CJK JP is listed before
    # WenQuanYi Zen Hei in sans_priority — so we pin Noto wins.
    assert chosen == "Noto Sans CJK JP", (
        f"auto mode should prefer Noto Sans CJK JP (sans, higher priority) "
        f"when both are available; got {chosen!r}"
    )


def test_use_chinese_picks_wenquanyi_zen_hei():
    """WenQuanYi Zen Hei is the classic Linux free Chinese font."""
    from statspai.plots import themes
    import matplotlib.font_manager as _fm

    fake = _fake_fontmanager(["DejaVu Sans", "WenQuanYi Zen Hei"])
    with patch.object(_fm, "fontManager", fake):
        chosen = themes.use_chinese()
    assert chosen == "WenQuanYi Zen Hei"


def test_use_chinese_picks_source_han_sans():
    """Source Han Sans (Adobe/Google) is Linux/cloud-common."""
    from statspai.plots import themes
    import matplotlib.font_manager as _fm

    fake = _fake_fontmanager(["DejaVu Sans", "Source Han Sans"])
    with patch.object(_fm, "fontManager", fake):
        chosen = themes.use_chinese()
    assert chosen == "Source Han Sans"


def test_use_chinese_warns_when_no_chinese_font():
    """Still warns + returns '' when no CJK font is present."""
    from statspai.plots import themes
    import matplotlib.font_manager as _fm

    fake = _fake_fontmanager(["DejaVu Sans", "Liberation Serif"])
    with patch.object(_fm, "fontManager", fake):
        with pytest.warns(UserWarning):
            chosen = themes.use_chinese()
    assert chosen == ""


# ---------------------------------------------------------------------------
# Bug 2 — regtable / mean_comparison / esttab must NOT auto-print
# ---------------------------------------------------------------------------


@pytest.fixture
def two_ols():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1 + 2 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    from statspai import regress
    return regress("y ~ x1", data=df), regress("y ~ x1 + x2", data=df)


def _captured(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) capturing stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


def test_regtable_does_not_auto_print(two_ols):
    """regtable() must not write to stdout — double-print bug in 0.9.3."""
    from statspai import regtable
    m1, m2 = two_ols
    result, stdout = _captured(regtable, m1, m2)
    assert stdout == "", (
        "regtable() wrote to stdout; REPL/Jupyter users saw the table twice.\n"
        f"Got:\n{stdout[:400]}"
    )
    # And the returned object must render nicely on demand
    assert "x1" in str(result)


def test_mean_comparison_does_not_auto_print():
    from statspai import mean_comparison
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "treated": rng.integers(0, 2, 200),
        "age": rng.normal(30, 5, 200),
        "income": rng.normal(50, 10, 200),
    })
    result, stdout = _captured(
        mean_comparison, df, variables=["age", "income"], group="treated"
    )
    assert stdout == "", f"mean_comparison wrote to stdout:\n{stdout[:400]}"


def test_esttab_does_not_auto_print(two_ols):
    from statspai import esttab
    m1, m2 = two_ols
    result, stdout = _captured(esttab, m1, m2)
    assert stdout == "", f"esttab wrote to stdout:\n{stdout[:400]}"


# ---------------------------------------------------------------------------
# Bug 3 — output='latex' must be honoured by str(result)
# ---------------------------------------------------------------------------


def test_regtable_output_latex_changes_str(two_ols):
    """regtable(..., output='latex') should make str(result) return LaTeX."""
    from statspai import regtable
    m1, m2 = two_ols
    result = regtable(m1, m2, output="latex")
    out = str(result)
    assert "\\begin{tabular}" in out, (
        "output='latex' did not propagate to str(result). "
        f"Got first 200 chars:\n{out[:200]}"
    )
    assert "\\hline" in out


def test_regtable_output_markdown_changes_str(two_ols):
    from statspai import regtable
    m1, m2 = two_ols
    result = regtable(m1, m2, output="markdown")
    out = str(result)
    # Markdown tables use pipe separators
    assert out.count("|") > 5, f"Expected markdown pipes, got:\n{out[:300]}"


def test_regtable_output_html_changes_str(two_ols):
    from statspai import regtable
    m1, m2 = two_ols
    result = regtable(m1, m2, output="html")
    out = str(result)
    assert "<table" in out.lower(), f"Expected HTML table:\n{out[:300]}"


def test_regtable_default_output_is_text(two_ols):
    from statspai import regtable
    m1, m2 = two_ols
    result = regtable(m1, m2)
    out = str(result)
    # Default text uses unicode thick/thin rules
    assert ("\u2501" in out) or ("\u2500" in out) or ("x1" in out), out[:200]


def test_regtable_to_methods_still_work(two_ols):
    """The explicit to_xxx() methods must still work regardless of output."""
    from statspai import regtable
    m1, m2 = two_ols
    result = regtable(m1, m2, output="latex")
    # Explicit calls always return their format
    assert "<table" in result.to_html().lower()
    assert "\\begin{tabular}" in result.to_latex()
    assert "x1" in result.to_text()


def test_regtable_invalid_output_raises(two_ols):
    """Unknown output= should raise ValueError, not silently fall back."""
    from statspai import regtable
    m1, m2 = two_ols
    with pytest.raises(ValueError, match="output="):
        regtable(m1, m2, output="xml")


def test_mean_comparison_invalid_output_raises():
    from statspai import mean_comparison
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "treated": rng.integers(0, 2, 50),
        "age": rng.normal(30, 5, 50),
    })
    with pytest.raises(ValueError, match="output="):
        mean_comparison(df, variables=["age"], group="treated", output="xml")


# ---------------------------------------------------------------------------
# Bug 4 — did() docstring clarifies treat semantics
# ---------------------------------------------------------------------------


def test_did_docstring_clarifies_treat_semantics():
    """did() must document the first_treat vs 0/1 distinction with an example."""
    from statspai.did import did
    doc = did.__doc__ or ""
    low = doc.lower()
    # Key phrases we want the user to see when they call help(did)
    assert "never treated" in low or "never-treated" in low or "never_treated" in low
    assert "first" in low and "treatment" in low
    # And an example constructing first_treat from a 0/1 indicator
    assert "first_treat" in doc, (
        "did() docstring should include an explicit 'first_treat' example "
        "showing how to construct it from a 0/1 indicator."
    )
