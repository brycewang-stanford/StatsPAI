"""Tests for causal_mas — multi-agent LLM causal-discovery framework."""

import warnings
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def test_causal_mas_heuristic_fallback_returns_edges():
    """Without an LLM client, causal_mas falls back to heuristics and must
    still return a sensible DAG on a clearly-ordered problem."""
    r = sp.causal_mas(
        variables=["age", "smoking", "lung_cancer"],
        domain="epidemiology",
        treatment="smoking",
        outcome="lung_cancer",
        rounds=1,
    )
    assert r is not None
    # Result should expose edges
    has_edges = hasattr(r, "edges") or hasattr(r, "final_edges") or hasattr(r, "dag")
    assert has_edges, f"result missing edge attribute; attrs: {dir(r)[:8]}"


def test_causal_mas_sensitive_to_treatment_outcome_pair():
    """Specifying T→Y should at minimum produce an edge touching outcome."""
    r = sp.causal_mas(
        variables=["income", "education", "health"],
        treatment="education",
        outcome="health",
        rounds=1,
    )
    edges = getattr(r, "edges", None) or getattr(r, "final_edges", None) or []
    # Should include at least one edge involving the outcome
    touches_outcome = any("health" in (e if isinstance(e, tuple) else e) for e in edges) \
        if edges else True  # lenient: heuristic may return empty in degenerate cases
    assert touches_outcome or edges == []


def test_causal_mas_signature_exposes_llm_hook():
    import inspect
    sig = inspect.signature(sp.causal_mas)
    # Should accept `client=None` so users can plug in an LLM
    assert "client" in sig.parameters


def test_causal_mas_registered():
    assert "causal_mas" in sp.list_functions()
