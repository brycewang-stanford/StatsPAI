"""Tests for ``regtable(multi_se=...)`` — extra SE specs side-by-side."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def two_models():
    rng = np.random.default_rng(2026)
    n = 300
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x": rng.normal(0, 1, n)})
    return sp.regress("y ~ x", data=df), sp.regress("y ~ x", data=df, robust="hc3")


def _bootstrap_se_dict(coef_value=0.077, intercept_value=0.061):
    return {"x": coef_value, "Intercept": intercept_value}


def test_multi_se_basic(two_models):
    m1, m2 = two_models
    txt = sp.regtable(
        m1, m2,
        multi_se={"Bootstrap SE": [_bootstrap_se_dict(), _bootstrap_se_dict()]},
    ).to_text()
    # Bracket-wrapped values should appear under each coef
    assert "[0.077]" in txt
    assert "Bootstrap SE in […]" in txt


def test_multi_se_two_extra_specs_use_distinct_brackets(two_models):
    m1, m2 = two_models
    txt = sp.regtable(
        m1, m2,
        multi_se={
            "Bootstrap": [_bootstrap_se_dict(0.07), _bootstrap_se_dict(0.07)],
            "Wild-cluster": [_bootstrap_se_dict(0.08), _bootstrap_se_dict(0.08)],
        },
    ).to_text()
    # First extra uses [...], second uses {...}
    assert "[0.070]" in txt
    assert "{0.080}" in txt
    assert "Bootstrap in […]" in txt
    assert "Wild-cluster in {…}" in txt


def test_multi_se_accepts_pandas_series(two_models):
    m1, m2 = two_models
    s1 = pd.Series({"x": 0.044, "Intercept": 0.055})
    s2 = pd.Series({"x": 0.045, "Intercept": 0.056})
    txt = sp.regtable(
        m1, m2,
        multi_se={"Cluster SE (firm)": [s1, s2]},
    ).to_text()
    assert "[0.044]" in txt
    assert "[0.045]" in txt


def test_multi_se_wrong_length_raises(two_models):
    m1, m2 = two_models
    with pytest.raises(ValueError, match="2 models"):
        sp.regtable(
            m1, m2,
            multi_se={"Cluster SE": [_bootstrap_se_dict()]},  # only 1 entry, need 2
        )


def test_multi_se_invalid_entry_type_raises(two_models):
    m1, m2 = two_models
    with pytest.raises(TypeError, match="multi_se"):
        sp.regtable(
            m1, m2,
            multi_se={"Wild": [42, "junk"]},  # type: ignore[list-item]
        )


def test_multi_se_missing_var_yields_empty_cell(two_models):
    m1, m2 = two_models
    txt = sp.regtable(
        m1, m2,
        multi_se={"Bootstrap SE": [{"Intercept": 0.061}, {"Intercept": 0.061}]},
    ).to_text()
    # x row's bootstrap line should be blank rather than crashing or emitting NaN
    assert "[nan]" not in txt.lower()


def test_multi_se_appears_in_latex(two_models):
    m1, m2 = two_models
    tex = sp.regtable(
        m1, m2,
        multi_se={"Bootstrap SE": [_bootstrap_se_dict(), _bootstrap_se_dict()]},
    ).to_latex()
    assert "[0.077]" in tex
    assert "Bootstrap SE in [" in tex


def test_multi_se_appears_in_html(two_models):
    m1, m2 = two_models
    html = sp.regtable(
        m1, m2,
        multi_se={"Bootstrap SE": [_bootstrap_se_dict(), _bootstrap_se_dict()]},
    ).to_html()
    assert "[0.077]" in html
    assert "Bootstrap SE" in html


def test_multi_se_appears_in_dataframe(two_models):
    m1, m2 = two_models
    df = sp.regtable(
        m1, m2,
        multi_se={"Bootstrap SE": [_bootstrap_se_dict(), _bootstrap_se_dict()]},
    ).to_dataframe()
    flat = df.to_string()
    assert "[0.077]" in flat


def test_multi_se_appears_in_markdown(two_models):
    """Markdown rendering must keep the bracket characters visible.

    Regression: an earlier ``_MULTI_SE_BRACKETS`` definition included a
    ``("|", "|")`` fourth pair which collided with GFM pipe-table cell
    delimiters and broke the row. The current bracket set is Markdown-safe.
    """
    m1, m2 = two_models
    md = sp.regtable(
        m1, m2,
        multi_se={"Bootstrap SE": [_bootstrap_se_dict(), _bootstrap_se_dict()]},
    ).to_markdown()
    assert "[0.077]" in md
    assert "Bootstrap SE in […]" in md


def test_multi_se_four_specs_brackets_are_markdown_safe(two_models):
    """All four bracket pairs must coexist in a single markdown table."""
    m1, m2 = two_models
    extra = _bootstrap_se_dict()
    md = sp.regtable(
        m1, m2,
        multi_se={
            "S1": [extra, extra],
            "S2": [extra, extra],
            "S3": [extra, extra],
            "S4": [extra, extra],
        },
    ).to_markdown()
    # The four bracket pairs (in cycle order)
    for opener, closer in [("[", "]"), ("{", "}"), ("⟨", "⟩"), ("«", "»")]:
        assert f"{opener}0.077{closer}" in md, f"missing bracket pair {opener}{closer} in markdown"
    # No pipe-character leak from any bracket — would corrupt GFM tables
    # The only bare ``|`` characters in a well-formed pipe table are cell
    # delimiters, not data. Count rows of similar shape to verify.
    body_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    cell_counts = {ln.count("|") for ln in body_lines}
    assert len(cell_counts) <= 2, f"inconsistent column counts: {cell_counts}"


def test_multi_se_appears_in_excel(two_models, tmp_path):
    """Excel export must carry the multi_se footer notes (regression test)."""
    pytest.importorskip("openpyxl")
    m1, m2 = two_models
    out_file = tmp_path / "regtable.xlsx"
    sp.regtable(
        m1, m2,
        multi_se={"Bootstrap SE": [_bootstrap_se_dict(), _bootstrap_se_dict()]},
    ).to_excel(str(out_file))

    import openpyxl

    wb = openpyxl.load_workbook(str(out_file))
    ws = wb.active
    found = False
    for row in ws.iter_rows(values_only=True):
        for cell in row:
            if cell and isinstance(cell, str) and "Bootstrap SE in" in cell:
                found = True
                break
        if found:
            break
    assert found, "Excel export lost the Bootstrap SE footer note"


def test_multi_se_combined_with_template_and_repro(two_models):
    """Multi-SE composes with the journal-preset and repro footer features."""
    m1, m2 = two_models
    out = sp.regtable(
        m1, m2,
        template="qje",
        multi_se={"Bootstrap SE": [_bootstrap_se_dict(), _bootstrap_se_dict()]},
        repro=True,
    )
    txt = out.to_text()
    assert "Robust standard errors" in txt
    assert "Bootstrap SE" in txt
    assert "StatsPAI v" in txt
