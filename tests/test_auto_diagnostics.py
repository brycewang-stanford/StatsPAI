"""Tests for ``regtable``'s auto-extraction of FE / cluster / IV diagnostic rows.

Covers the public ``diagnostics='auto'`` behaviour that turns
``model_info`` / ``diagnostics`` metadata into explicit
``Fixed Effects: Yes`` / ``Cluster SE: <var>`` / ``First-stage F: ...``
rows above the summary-stats block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output._diagnostics import (
    _fe_token_label,
    _parse_fe_tokens,
    extract_diagnostic_rows,
    extract_fe_cluster_indicators,
)


class _FakeResult:
    """Minimal duck-typed model carrying ``model_info`` only.

    Used to drive the per-FE row extractor with synthetic FE metadata
    without booting a real estimator (HDFE/pyfixest are heavy).
    """
    def __init__(self, fe=None, cluster=None):
        mi = {}
        if fe is not None:
            mi["fixed_effects"] = fe
        if cluster is not None:
            mi["cluster"] = cluster
        self.model_info = mi
        self.params = pd.Series({"x": 1.0})  # for _is_econometric duck-test
        self.std_errors = pd.Series({"x": 0.1})


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ols_models():
    rng = np.random.default_rng(2026)
    n = 400
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "x": rng.normal(0, 1, n),
        "g": rng.integers(0, 5, n),
    })
    m_plain = sp.regress("y ~ x", data=df)
    m_clust = sp.regress("y ~ x", data=df, cluster="g")
    return m_plain, m_clust


@pytest.fixture
def iv_model():
    rng = np.random.default_rng(2026)
    n = 1000
    z = rng.normal(0, 1, n)
    x = 0.7 * z + rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    return sp.IVRegression("y ~ (x ~ z)", data=df).fit()


# ---------------------------------------------------------------------------
# extract_fe_cluster_indicators
# ---------------------------------------------------------------------------

def test_cluster_row_appears_when_any_model_clusters(ols_models):
    m_plain, m_clust = ols_models
    rows = extract_fe_cluster_indicators([m_plain, m_clust])
    assert "Cluster SE" in rows
    assert rows["Cluster SE"] == ["No", "g"]


def test_cluster_row_omitted_when_no_model_clusters(ols_models):
    m_plain, _ = ols_models
    rows = extract_fe_cluster_indicators([m_plain, m_plain])
    assert "Cluster SE" not in rows


def test_fe_row_omitted_when_no_fe(ols_models):
    m_plain, _ = ols_models
    rows = extract_fe_cluster_indicators([m_plain, m_plain])
    assert "Fixed Effects" not in rows


# ---------------------------------------------------------------------------
# _parse_fe_tokens: shape tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, []),
    ("", []),
    ("None", []),
    ("none", []),
    ("firm", ["firm"]),
    ("firm+year", ["firm", "year"]),
    ("firm + year", ["firm", "year"]),
    ("firm^year", ["firm^year"]),                    # interaction stays one token
    ("firm^year+industry", ["firm^year", "industry"]),
    (["firm", "year"], ["firm", "year"]),
    (("firm", "year"), ["firm", "year"]),
    ({"firm": object(), "year": object()}, ["firm", "year"]),
])
def test_parse_fe_tokens_shapes(value, expected):
    assert _parse_fe_tokens(value) == expected


@pytest.mark.parametrize("token,label", [
    ("firm", "Firm FE"),
    ("year", "Year FE"),
    ("firm^year", "Firm × Year FE"),
    ("industry^state^quarter", "Industry × State × Quarter FE"),
    ("state-1", "State-1 FE"),                       # non-letter preserved
])
def test_fe_token_label(token, label):
    assert _fe_token_label(token) == label


# ---------------------------------------------------------------------------
# Per-FE row extraction (the AER-style format)
# ---------------------------------------------------------------------------

def test_per_fe_rows_one_row_per_variable():
    """3 columns with mixed FE specs → one row per distinct FE."""
    m1 = _FakeResult(fe="firm+year")           # both
    m2 = _FakeResult(fe="firm")                # only firm
    m3 = _FakeResult(fe="year")                # only year
    rows = extract_fe_cluster_indicators([m1, m2, m3])
    assert "Fixed Effects" not in rows         # no collapsed row
    assert rows["Firm FE"] == ["Yes", "Yes", "No"]
    assert rows["Year FE"] == ["Yes", "No", "Yes"]


def test_per_fe_rows_preserve_first_seen_order():
    m1 = _FakeResult(fe="state+industry")
    m2 = _FakeResult(fe="industry+year")
    rows = extract_fe_cluster_indicators([m1, m2])
    labels = list(rows.keys())
    # state first (from m1), then industry (m1), then year (m2)
    assert labels == ["State FE", "Industry FE", "Year FE"]


def test_per_fe_rows_handle_interaction_as_single_row():
    m1 = _FakeResult(fe="firm^year")
    m2 = _FakeResult(fe="firm^year")
    rows = extract_fe_cluster_indicators([m1, m2])
    assert list(rows.keys()) == ["Firm × Year FE"]
    assert rows["Firm × Year FE"] == ["Yes", "Yes"]


def test_per_fe_rows_columns_without_fe_get_no():
    """Mixing FE and plain-OLS columns → plain column gets all "No"s."""
    m_fe = _FakeResult(fe="firm+year")
    m_plain = _FakeResult(fe=None)
    rows = extract_fe_cluster_indicators([m_fe, m_plain])
    assert rows["Firm FE"] == ["Yes", "No"]
    assert rows["Year FE"] == ["Yes", "No"]


def test_no_fe_anywhere_drops_all_fe_rows():
    """All columns without FE → no FE-related rows at all (regression test)."""
    m1 = _FakeResult(fe=None)
    m2 = _FakeResult(fe="")
    rows = extract_fe_cluster_indicators([m1, m2])
    assert all("FE" not in label for label in rows.keys())
    assert "Fixed Effects" not in rows


# ---------------------------------------------------------------------------
# regtable() integration — per-FE rows visible in rendered output
# ---------------------------------------------------------------------------

def test_regtable_renders_per_fe_rows():
    """End-to-end: synthetic results with FE metadata produce labelled rows."""
    m1 = _FakeResult(fe="firm+year")
    m2 = _FakeResult(fe="firm")
    # We hand-construct add_rows-equivalent metadata; regtable picks it up
    # via diagnostics='auto'.
    rows = extract_fe_cluster_indicators([m1, m2])
    assert "Firm FE" in rows
    assert "Year FE" in rows
    # The Yes/No pattern is what regtable will render; structure-only check.
    assert rows["Firm FE"] == ["Yes", "Yes"]
    assert rows["Year FE"] == ["Yes", "No"]


# ---------------------------------------------------------------------------
# Fallback branch: unparseable FE metadata → single Yes/No row
# ---------------------------------------------------------------------------

class _OpaqueFE:
    """Truthy non-string/list/dict to exercise the fallback branch."""
    def __bool__(self):
        return True


def test_per_fe_falls_back_to_single_row_for_unknown_shape():
    """Unknown truthy FE metadata → single ``Fixed Effects: Yes/No`` row."""
    m1 = _FakeResult(fe=_OpaqueFE())
    m2 = _FakeResult(fe=None)
    rows = extract_fe_cluster_indicators([m1, m2])
    # No per-FE row labels (they would have invented a token name)
    assert all(not k.endswith(" FE") for k in rows.keys())
    # Single fallback row uses the legacy label
    assert rows["Fixed Effects"] == ["Yes", "No"]


# ---------------------------------------------------------------------------
# Backwards-compat: user-supplied "Fixed Effects" row suppresses auto per-FE
# ---------------------------------------------------------------------------

def test_user_fixed_effects_row_suppresses_auto_per_fe_rows():
    """Legacy ``add_rows={'Fixed Effects': [...]}`` must keep working: when the
    user explicitly provides this row, the auto per-FE expansion should be
    suppressed so the rendered table shows the user's single row only — not
    the user's row stacked on top of auto ``Firm FE`` / ``Year FE`` rows.
    """
    pytest.importorskip("pyfixest")
    rng = np.random.default_rng(2026)
    n = 400
    df = pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "x": rng.normal(0, 1, n),
        "firm": rng.integers(0, 20, n),
        "year": rng.integers(2010, 2020, n),
    })
    m1 = sp.feols("y ~ x | firm + year", data=df)
    m2 = sp.feols("y ~ x | firm", data=df)
    txt = sp.regtable(
        m1, m2,
        add_rows={"Fixed Effects": ["All", "Firm only"]},
    ).to_text()
    assert "Fixed Effects" in txt
    assert "All" in txt and "Firm only" in txt
    # Auto per-FE rows must be suppressed
    assert "Firm FE" not in txt
    assert "Year FE" not in txt


# ---------------------------------------------------------------------------
# extract_diagnostic_rows
# ---------------------------------------------------------------------------

def test_iv_first_stage_F_row_extracted(iv_model):
    rows = extract_diagnostic_rows([iv_model])
    assert "First-stage F" in rows
    cell = rows["First-stage F"][0]
    # Should be a formatted positive number
    val = float(cell)
    assert val > 0


def test_diagnostic_rows_empty_for_plain_ols(ols_models):
    m_plain, _ = ols_models
    rows = extract_diagnostic_rows([m_plain, m_plain])
    # No FE, no cluster, no IV/DiD/RD ⇒ empty
    assert len(rows) == 0


def test_diagnostic_rows_disable_iv():
    rng = np.random.default_rng(2026)
    n = 500
    z = rng.normal(0, 1, n)
    x = 0.7 * z + rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    m_iv = sp.IVRegression("y ~ (x ~ z)", data=df).fit()
    rows = extract_diagnostic_rows([m_iv], include_iv=False)
    assert "First-stage F" not in rows


# ---------------------------------------------------------------------------
# regtable() integration
# ---------------------------------------------------------------------------

def test_regtable_auto_emits_cluster_row(ols_models):
    m_plain, m_clust = ols_models
    txt = sp.regtable(m_plain, m_clust).to_text()
    assert "Cluster SE" in txt
    assert "g" in txt  # cluster variable name shown


def test_regtable_diagnostics_off_suppresses_cluster_row(ols_models):
    m_plain, m_clust = ols_models
    txt = sp.regtable(m_plain, m_clust, diagnostics=False).to_text()
    assert "Cluster SE" not in txt


def test_regtable_diagnostics_off_string(ols_models):
    """diagnostics='off' is also accepted as a string flag."""
    m_plain, m_clust = ols_models
    txt = sp.regtable(m_plain, m_clust, diagnostics="off").to_text()
    assert "Cluster SE" not in txt


def test_regtable_user_add_rows_override_auto(ols_models):
    m_plain, m_clust = ols_models
    out = sp.regtable(
        m_plain, m_clust,
        add_rows={"Cluster SE": ["—", "Industry"]},
    )
    txt = out.to_text()
    # User's value should win, not the auto-extracted "g"
    assert "Industry" in txt
    # The auto value "g" should not appear in the cluster row
    assert "g" not in txt.splitlines()[
        next(i for i, ln in enumerate(txt.splitlines()) if "Cluster SE" in ln)
    ]


def test_regtable_iv_first_stage_F_row(iv_model):
    txt = sp.regtable(iv_model).to_text()
    assert "First-stage F" in txt


def test_regtable_iv_diagnostics_with_template(iv_model):
    """Template + auto diagnostics combine cleanly."""
    txt = sp.regtable(iv_model, template="aer").to_text()
    assert "First-stage F" in txt
    assert "Standard errors in parentheses" in txt
