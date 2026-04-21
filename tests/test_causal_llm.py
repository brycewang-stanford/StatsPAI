"""Smoke tests for sp.causal_llm — LLM × Causal namespace."""

from __future__ import annotations

import statspai as sp


def test_llm_dag_propose_heuristic():
    res = sp.llm_dag_propose(
        variables=['age', 'education', 'wage', 'distance'],
        domain="labor economics",
    )
    assert isinstance(res, sp.LLMDAGProposal)
    # 'distance' should be classified as instrument
    assert res.roles['distance'] == 'instrument'
    assert res.roles['wage'] == 'outcome'
    # Should have at least one edge to wage
    edges_to_wage = [e for e in res.edges if e[1] == 'wage']
    assert len(edges_to_wage) >= 1
    # to_dag_string format check
    s = res.to_dag_string()
    assert " -> " in s


def test_llm_unobserved_confounders():
    res = sp.llm_unobserved_confounders(
        treatment="ACE inhibitor",
        outcome="cardiovascular events",
        domain="health",
        point_estimate_rr=1.5,
    )
    assert isinstance(res, sp.UnobservedConfounderProposal)
    assert len(res.candidates) > 0
    assert all(e > 0 for e in res.suggested_evalue_thresholds)
    assert "patient adherence" in res.candidates


def test_llm_sensitivity_priors():
    res = sp.llm_sensitivity_priors(
        treatment="job training",
        outcome="earnings",
        domain="labor",
    )
    assert isinstance(res, sp.SensitivityPriorProposal)
    assert 0 < res.rho_max < 1
    assert 0 < res.r2 < 1
    assert "ability" in res.rationale.lower() or "motivation" in res.rationale.lower()


def test_llm_unknown_domain_fallback():
    res = sp.llm_sensitivity_priors(
        treatment="x", outcome="y", domain="weatherology",
    )
    assert res.backend == "heuristic"
    assert "Generic" in res.rationale
