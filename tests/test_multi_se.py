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
