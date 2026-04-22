"""Tests for P1-A: LLM-DAG closed loop + constrained PC + DAG validate.

Covers the design's six P1-A acceptance tests plus a regression check
that unconstrained PC behaviour is identical to the prior contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.causal_discovery.pc import pc_algorithm
from statspai.causal_llm.llm_dag_loop import _normalize_oracle_output


RNG = np.random.default_rng(2026)


def _chain_dgp(n: int = 800, seed: int = 0):
    """X -> Y -> Z linear-Gaussian DGP."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    Y = 0.8 * X + 0.5 * rng.standard_normal(n)
    Z = 0.7 * Y + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


# --------------------------------------------------------------------- #
#  Constrained PC
# --------------------------------------------------------------------- #


def test_unconstrained_pc_unchanged():
    """Plain pc_algorithm with neither forbidden nor required behaves
    exactly as before."""
    df = _chain_dgp()
    r = pc_algorithm(df, variables=["X", "Y", "Z"])
    # On the chain, PC should detect skeleton {X-Y, Y-Z} and X _||_ Z | Y.
    assert ("X", "Y") in r["undirected_edges"] or ("X", "Y") in r["edges"]
    assert ("Y", "Z") in r["undirected_edges"] or ("Y", "Z") in r["edges"]
    assert ("X", "Z") not in r["undirected_edges"]
    assert ("X", "Z") not in r["edges"] and ("Z", "X") not in r["edges"]


def test_constrained_pc_respects_required():
    """A required edge survives even when CI rejects it."""
    df = _chain_dgp()
    r = pc_algorithm(df, variables=["X", "Y", "Z"], required=[("X", "Z")])
    assert ("X", "Z") in r["edges"], r["edges"]


def test_constrained_pc_respects_forbidden():
    """A forbidden edge does not appear regardless of skeleton evidence."""
    df = _chain_dgp()
    r = pc_algorithm(df, variables=["X", "Y", "Z"], forbidden=[("Y", "Z")])
    assert ("Y", "Z") not in r["edges"]
    assert ("Y", "Z") not in r["undirected_edges"]
    assert ("Z", "Y") not in r["undirected_edges"]


def test_constrained_pc_required_wins_over_forbidden():
    """When the same pair is in both lists, required wins."""
    df = _chain_dgp()
    r = pc_algorithm(
        df, variables=["X", "Y", "Z"],
        required=[("X", "Y")], forbidden=[("X", "Y")],
    )
    assert ("X", "Y") in r["edges"]


def test_constrained_pc_unknown_edge_silently_ignored():
    """Unknown variable in constraint list does not raise."""
    df = _chain_dgp()
    # 'Q' isn't in variables — should be ignored, not crash.
    r = pc_algorithm(
        df, variables=["X", "Y", "Z"],
        required=[("Q", "Y")], forbidden=[("Q", "Z")],
    )
    assert isinstance(r, dict)
    assert "edges" in r


# --------------------------------------------------------------------- #
#  llm_dag_constrained — closed loop
# --------------------------------------------------------------------- #


def test_loop_demotes_ci_rejected_edge():
    """LLM proposes a wrong edge; loop demotes it after CI test."""
    df = _chain_dgp()

    def oracle(vars_, desc):
        return [("X", "Y", 0.95), ("Y", "Z", 0.90), ("X", "Z", 0.85)]

    r = sp.llm_dag_constrained(
        df, variables=["X", "Y", "Z"], oracle=oracle, max_iter=3,
    )
    edges = set(r.final_edges)
    assert ("X", "Y") in edges
    assert ("Y", "Z") in edges
    # X -> Z should have been demoted.
    assert ("X", "Z") not in edges
    # Iteration log shows the demotion.
    assert any(it["demoted"] >= 1 for it in r.iteration_log)
    assert r.converged


def test_loop_converges_when_llm_correct():
    """If the LLM proposes only correct edges, the loop converges in
    iteration 0 (no demotions)."""
    df = _chain_dgp()

    def oracle(vars_, desc):
        return [("X", "Y", 0.99), ("Y", "Z", 0.99)]

    r = sp.llm_dag_constrained(
        df, variables=["X", "Y", "Z"], oracle=oracle, max_iter=5,
    )
    assert r.converged
    assert r.iteration_log[0]["demoted"] == 0
    assert ("X", "Y") in r.final_edges
    assert ("Y", "Z") in r.final_edges


def test_loop_works_without_oracle():
    """No oracle -> falls back to plain PC, returns sensible result."""
    df = _chain_dgp()
    r = sp.llm_dag_constrained(
        df, variables=["X", "Y", "Z"], oracle=None, max_iter=2,
    )
    assert isinstance(r.final_edges, list)
    # No LLM scores recorded.
    assert r.edge_confidence["llm_score"].isna().all()
    assert r.provenance["oracle_error"] is None or "no oracle" in str(
        r.provenance.get("oracle_error", ""))


def test_loop_provenance_records_threshold_and_proposal():
    df = _chain_dgp()

    def oracle(vars_, desc):
        return [("X", "Y", 0.8), ("X", "Z", 0.6)]  # second is mid-conf

    r = sp.llm_dag_constrained(
        df, variables=["X", "Y", "Z"], oracle=oracle,
        high_conf_threshold=0.7, max_iter=2,
    )
    assert r.provenance["high_conf_threshold"] == 0.7
    proposed = r.provenance["oracle_edges_proposed"]
    assert ("X", "Y", 0.8) in proposed
    # X->Z below threshold so it's a candidate, not a required edge.


def test_loop_to_dag_returns_dag_object():
    df = _chain_dgp()

    def oracle(vars_, desc):
        return [("X", "Y", 0.95), ("Y", "Z", 0.95)]

    r = sp.llm_dag_constrained(
        df, variables=["X", "Y", "Z"], oracle=oracle, max_iter=2,
    )
    g = r.to_dag()
    # Should be a DAG with X, Y, Z as nodes.
    assert {"X", "Y", "Z"} <= set(g.nodes)


def test_loop_respects_low_conf_forbidden_when_enabled():
    """forbid_low_conf=True drops low-conf edges from the skeleton."""
    df = _chain_dgp()

    def oracle(vars_, desc):
        return [("X", "Y", 0.95), ("Y", "Z", 0.10)]  # Y->Z is low-conf

    r = sp.llm_dag_constrained(
        df, variables=["X", "Y", "Z"], oracle=oracle,
        high_conf_threshold=0.7, low_conf_threshold=0.3,
        forbid_low_conf=True, max_iter=2,
    )
    # When forbid_low_conf=True, Y-Z (low-conf) should be forbidden.
    edges = set(r.final_edges)
    assert ("Y", "Z") not in edges
    assert ("Z", "Y") not in edges


# --------------------------------------------------------------------- #
#  llm_dag_validate
# --------------------------------------------------------------------- #


def test_validate_returns_per_edge_support():
    """Declared DAG with one wrong edge gets per-edge p-values."""
    df = _chain_dgp()
    g = sp.dag("X -> Y; Y -> Z; X -> Z")
    v = sp.llm_dag_validate(g, df, alpha=0.05)
    # X->Z should be unsupported when conditioning on Y (parent of Z).
    by_edge = {tuple(row["edge"]): row for _, row in v.edge_evidence.iterrows()}
    assert by_edge[("X", "Z")]["supported"] is False
    assert by_edge[("X", "Y")]["supported"] is True
    assert by_edge[("Y", "Z")]["supported"] is True
    assert v.n_unsupported == 1
    assert v.n_supported == 2


def test_validate_handles_empty_dag():
    """Empty DAG returns an empty result, not a crash."""
    df = _chain_dgp()
    g = sp.dag("")
    v = sp.llm_dag_validate(g, df)
    assert v.n_supported == 0
    assert v.n_unsupported == 0


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #


def test_normalize_oracle_output_handles_pairs():
    out = _normalize_oracle_output([("A", "B"), ("C", "D", 0.5)])
    assert out == [("A", "B", 1.0), ("C", "D", 0.5)]


def test_normalize_oracle_output_handles_proposal_object():
    """LLMDAGProposal-like object with .edges attribute."""
    class _FakeProposal:
        edges = [("A", "B", 0.7)]
    out = _normalize_oracle_output(_FakeProposal())
    assert out == [("A", "B", 0.7)]


def test_normalize_oracle_output_clips_confidence():
    out = _normalize_oracle_output([("A", "B", 1.5), ("C", "D", -0.2)])
    assert out[0][2] == 1.0
    assert out[1][2] == 0.0


# --------------------------------------------------------------------- #
#  Registry / agent surface
# --------------------------------------------------------------------- #


def test_llm_dag_constrained_registered():
    funcs = sp.list_functions()
    assert "llm_dag_constrained" in funcs
    assert "llm_dag_validate" in funcs


def test_llm_dag_constrained_agent_card_exposes_assumptions():
    spec = sp.describe_function("llm_dag_constrained")
    assert "assumptions" in spec
    assert any("Faithfulness" in a for a in spec["assumptions"])
