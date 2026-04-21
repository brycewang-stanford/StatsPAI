"""Tests for ``sp.causal_question`` DSL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def rct_data():
    rng = np.random.default_rng(7)
    n = 500
    treat = rng.binomial(1, 0.5, n)
    y = 5 + 1.5 * treat + rng.normal(0, 1, n)
    return pd.DataFrame({"treat": treat, "y": y})


@pytest.fixture
def confounded_data():
    rng = np.random.default_rng(13)
    n = 800
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-x))
    treat = rng.binomial(1, p)
    y = 2 + 1.0 * treat + 0.8 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"treat": treat, "y": y, "x": x})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_causal_question_construction():
    q = sp.causal_question(
        treatment="d",
        outcome="y",
        estimand="ATE",
        design="rct",
    )
    assert q.treatment == "d"
    assert q.outcome == "y"
    assert q.estimand == "ATE"


def test_causal_question_rejects_invalid_estimand():
    with pytest.raises(ValueError):
        sp.causal_question(
            treatment="d", outcome="y", estimand="NONSENSE",
        )


def test_causal_question_rejects_invalid_design():
    with pytest.raises(ValueError):
        sp.causal_question(
            treatment="d", outcome="y", design="unknown_thing",
        )


# ---------------------------------------------------------------------------
# identify()
# ---------------------------------------------------------------------------


def test_identify_rct_design():
    q = sp.causal_question(
        treatment="d", outcome="y", design="rct",
    )
    plan = q.identify()
    assert plan.estimator == "regress"
    assert "random" in plan.identification_story.lower()


def test_identify_iv_design_requires_instruments():
    q = sp.causal_question(
        treatment="d", outcome="y", design="iv",
    )
    plan = q.identify()
    assert plan.estimator == "iv"
    assert plan.warnings  # should warn about missing instruments


def test_identify_iv_design_with_instruments():
    q = sp.causal_question(
        treatment="d", outcome="y", design="iv",
        instruments=["z1"],
    )
    plan = q.identify()
    assert plan.estimator == "iv"
    assert plan.estimand == "LATE"
    assert "z1" in plan.identification_story


def test_identify_auto_design_with_instruments():
    q = sp.causal_question(
        treatment="d", outcome="y", design="auto",
        instruments=["z1"],
    )
    plan = q.identify()
    assert plan.estimator == "iv"


def test_identify_auto_design_with_running_variable():
    q = sp.causal_question(
        treatment="d", outcome="y", design="auto",
        running_variable="score", cutoff=50,
    )
    plan = q.identify()
    assert plan.estimator == "rdrobust"


def test_identify_auto_default_is_selection_on_observables():
    q = sp.causal_question(
        treatment="d", outcome="y", design="auto",
        covariates=["age", "sex"],
    )
    plan = q.identify()
    assert plan.estimator == "aipw"


# ---------------------------------------------------------------------------
# estimate()
# ---------------------------------------------------------------------------


def test_estimate_rct(rct_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="rct", data=rct_data,
    )
    r = q.estimate()
    # True ATE = 1.5
    assert abs(r.estimate - 1.5) < 0.3
    assert r.n == len(rct_data)
    assert r.estimator == "regress"


def test_estimate_aipw_confounded(confounded_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="selection_on_observables",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate()
    # True ATE = 1.0
    assert abs(r.estimate - 1.0) < 0.3
    assert r.estimator == "aipw"


# ---------------------------------------------------------------------------
# report()
# ---------------------------------------------------------------------------


def test_report_requires_estimate_first():
    q = sp.causal_question(treatment="d", outcome="y", design="rct")
    with pytest.raises(ValueError):
        q.report()


def test_report_markdown(rct_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="rct", data=rct_data,
    )
    q.estimate()
    md = q.report("markdown")
    assert "Causal Question" in md
    assert "Identification" in md
    assert "Estimate" in md


def test_report_text(rct_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="rct", data=rct_data,
    )
    q.estimate()
    txt = q.report("text")
    assert "Estimand" in txt


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


def test_question_to_dict():
    q = sp.causal_question(
        treatment="d", outcome="y",
        design="did", covariates=["x1", "x2"],
    )
    d = q.to_dict()
    assert d["treatment"] == "d"
    assert d["covariates"] == ["x1", "x2"]
