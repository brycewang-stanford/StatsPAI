"""Tests for DAG.recommend_estimator() and TargetTrialResult.to_paper()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# DAG -> estimator recommendation
# ---------------------------------------------------------------------------


def test_recommend_estimator_simple_backdoor():
    g = sp.dag("X -> Y; Z -> X; Z -> Y")
    rec = g.recommend_estimator("X", "Y")
    assert rec.estimator in ("regress",)
    assert rec.adjustment_set == {"Z"}
    assert "Z" in str(rec.sp_call)


def test_recommend_estimator_no_confounders():
    g = sp.dag("X -> Y")
    rec = g.recommend_estimator("X", "Y")
    # With no backdoor, OLS without controls is identifying
    assert rec.estimator == "regress"
    assert rec.adjustment_set == set()


def test_recommend_estimator_iv_when_unobserved_confounder():
    # U is latent common cause; Z is valid IV
    g = sp.dag("U -> X; U -> Y; X -> Y; Z -> X")
    # Manually mark U as latent by using the bidirected convention
    g2 = sp.dag("X -> Y; X <-> Y; Z -> X")
    rec = g2.recommend_estimator("X", "Y")
    # Should find IV or flag non-identifiable
    assert rec.estimator in ("iv", "identify", "front_door")


def test_recommend_estimator_detects_mediator():
    # X -> M -> Y, and we also adjust for a back-door Z
    g = sp.dag("X -> M; M -> Y; Z -> X; Z -> Y")
    rec = g.recommend_estimator("X", "Y")
    assert "M" in rec.mediators or any("mediate" in a for a in rec.alternatives)


def test_recommend_estimator_summary_runs():
    g = sp.dag("X -> Y; Z -> X; Z -> Y")
    rec = g.recommend_estimator("X", "Y")
    text = rec.summary()
    assert "Recommended" in text
    assert "Identification" in text


# ---------------------------------------------------------------------------
# Target trial to_paper()
# ---------------------------------------------------------------------------


@pytest.fixture
def tte_data_and_protocol():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(65, 8, n),
        "sex": rng.integers(0, 2, n),
        "treat": rng.binomial(1, 0.4, n),
        "outcome": rng.normal(0, 1, n),
    })
    # Apply a treatment effect
    df.loc[df["treat"] == 1, "outcome"] += 0.5
    protocol = sp.target_trial_protocol(
        eligibility="age >= 50",
        treatment_strategies=["statin", "no_statin"],
        assignment="observational emulation",
        time_zero="date of cohort entry",
        followup_end="min(event, 5y)",
        outcome="MI",
        causal_contrast="ITT",
        analysis_plan="IPW + pooled logistic",
        baseline_covariates=["age", "sex"],
    )
    return df, protocol


def test_target_trial_to_paper_markdown(tte_data_and_protocol):
    df, protocol = tte_data_and_protocol
    result = sp.target_trial_emulate(
        protocol=protocol,
        data=df,
        outcome_col="outcome",
        treatment_col="treat",
    )
    md = result.to_paper(fmt="markdown")
    assert "Target Trial Specification" in md
    assert "Eligibility" in md
    assert "Causal contrast" in md or "Causal" in md


def test_target_trial_to_paper_text(tte_data_and_protocol):
    df, protocol = tte_data_and_protocol
    result = sp.target_trial_emulate(
        protocol=protocol, data=df,
        outcome_col="outcome", treatment_col="treat",
    )
    text = result.to_paper(fmt="text")
    assert "Target Trial Emulation Report" in text
    assert "n eligible" in text


def test_target_trial_to_paper_latex(tte_data_and_protocol):
    df, protocol = tte_data_and_protocol
    result = sp.target_trial_emulate(
        protocol=protocol, data=df,
        outcome_col="outcome", treatment_col="treat",
    )
    tex = result.to_paper(fmt="latex")
    assert "\\begin{tabular}" in tex
    assert "tabular" in tex


def test_target_trial_to_paper_rejects_bad_fmt(tte_data_and_protocol):
    df, protocol = tte_data_and_protocol
    result = sp.target_trial_emulate(
        protocol=protocol, data=df,
        outcome_col="outcome", treatment_col="treat",
    )
    with pytest.raises(ValueError):
        result.to_paper(fmt="json")
