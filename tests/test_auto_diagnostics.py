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
    extract_diagnostic_rows,
    extract_fe_cluster_indicators,
)


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
