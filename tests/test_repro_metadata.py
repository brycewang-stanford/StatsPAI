"""Tests for the reproducibility-metadata footer (``regtable(repro=...)``)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output._repro import build_repro_note


@pytest.fixture
def ols_models():
    rng = np.random.default_rng(2026)
    n = 200
    df = pd.DataFrame({"y": rng.normal(0, 1, n), "x": rng.normal(0, 1, n)})
    return sp.regress("y ~ x", data=df), sp.regress("y ~ x", data=df, robust="hc3")


@pytest.fixture
def small_df():
    rng = np.random.default_rng(2026)
    return pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.integers(0, 5, 50)})


# ---------------------------------------------------------------------------
# build_repro_note unit tests
# ---------------------------------------------------------------------------

def test_repro_note_default_includes_version_and_timestamp():
    s = build_repro_note()
    assert s.startswith("Reproducibility:")
    assert "StatsPAI v" in s
    # timestamp YYYY-MM-DD HH:MM
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", s)


def test_repro_note_with_seed():
    s = build_repro_note(seed=42, timestamp=False)
    assert "seed=42" in s


def test_repro_note_with_dataframe(small_df):
    s = build_repro_note(data=small_df, timestamp=False)
    assert "data 50×2" in s
    # SHA256 prefix (10 hex chars by default)
    assert re.search(r"SHA256:[0-9a-f]{10}\b", s)


def test_repro_note_dataframe_hash_is_deterministic(small_df):
    s1 = build_repro_note(data=small_df, timestamp=False, package_version=False)
    s2 = build_repro_note(data=small_df, timestamp=False, package_version=False)
    assert s1 == s2


def test_repro_note_dataframe_hash_changes_with_data(small_df):
    other = small_df.copy()
    other.loc[0, "a"] = 999.0
    h1 = build_repro_note(data=small_df, timestamp=False, package_version=False)
    h2 = build_repro_note(data=other, timestamp=False, package_version=False)
    assert h1 != h2


def test_repro_note_skip_all_components_yields_empty():
    s = build_repro_note(timestamp=False, package_version=False)
    assert s == ""


def test_repro_note_extra_appended():
    s = build_repro_note(timestamp=False, extra="commit:abcdef")
    assert "commit:abcdef" in s


def test_repro_note_python_version_optional():
    s = build_repro_note(timestamp=False, python_version=True)
    assert re.search(r"Python \d+\.\d+\.\d+", s)


def test_repro_note_ignores_non_dataframe():
    s = build_repro_note(data="not a frame", timestamp=False)
    # No data fingerprint should appear
    assert "SHA256" not in s
    assert "data " not in s


# ---------------------------------------------------------------------------
# regtable() integration
# ---------------------------------------------------------------------------

def test_regtable_repro_true_appends_footer(ols_models):
    m1, m2 = ols_models
    txt = sp.regtable(m1, m2, repro=True).to_text()
    assert "Reproducibility:" in txt
    assert "StatsPAI v" in txt


def test_regtable_repro_false_no_footer(ols_models):
    m1, m2 = ols_models
    txt = sp.regtable(m1, m2, repro=False).to_text()
    assert "Reproducibility:" not in txt


def test_regtable_repro_dict_records_seed_and_data_hash(ols_models, small_df):
    m1, m2 = ols_models
    txt = sp.regtable(m1, m2, repro={"seed": 42, "data": small_df}).to_text()
    assert "seed=42" in txt
    assert "SHA256:" in txt


def test_regtable_repro_in_latex_output(ols_models):
    m1, m2 = ols_models
    tex = sp.regtable(m1, m2, repro=True).to_latex()
    assert "Reproducibility:" in tex
    assert "StatsPAI v" in tex


def test_regtable_repro_in_html_output(ols_models):
    m1, m2 = ols_models
    html = sp.regtable(m1, m2, repro=True).to_html()
    assert "Reproducibility:" in html
