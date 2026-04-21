"""
Tests for sp.llm_causal_assess + sp.pairwise_causal_benchmark.
"""

from __future__ import annotations

import pandas as pd
import pytest

import statspai as sp


def _oracle_llm(prompt: str) -> str:
    """Toy LLM that returns 'yes' iff A is in {'smoking', 'education'}."""
    # Parse "{a} causally influence {b}" from the prompt
    lower = prompt.lower()
    # Look for the causal verbs
    if "smoking" in lower and "lung" in lower:
        return "yes, smoking causally influences lung cancer"
    if "education" in lower and "wage" in lower:
        return "Yes"
    if "horoscope" in lower:
        return "no, horoscopes do not cause outcomes"
    return "no"


def test_pairwise_causal_benchmark_oracle():
    gt = pd.DataFrame([
        {"A": "smoking", "B": "lung cancer", "a_causes_b": True},
        {"A": "education", "B": "wage", "a_causes_b": True},
        {"A": "horoscope", "B": "success", "a_causes_b": False},
        {"A": "shoe_size", "B": "iq", "a_causes_b": False},
    ])
    res = sp.pairwise_causal_benchmark(
        gt, llm_client=_oracle_llm, llm_identifier="oracle",
    )
    assert res.accuracy == 1.0
    assert res.recall_forward == 1.0
    assert len(res.per_pair) == 4
    assert "Pairwise" in res.summary()


def test_pairwise_benchmark_missing_columns():
    bad = pd.DataFrame({"X": ["a"], "Y": ["b"]})
    with pytest.raises(ValueError, match="Missing"):
        sp.pairwise_causal_benchmark(
            bad, llm_client=_oracle_llm,
        )


def test_llm_causal_assess_level1():
    def llm(q: str) -> str:
        if "smoking" in q.lower():
            return "The answer is yes"
        return "I don't know"
    items = pd.DataFrame([
        {"question": "Does smoking cause cancer?", "answer": "yes"},
        {"question": "Does wealth cause health?", "answer": "yes"},
    ])
    res = sp.llm_causal_assess(
        level1_items=items, llm_client=llm, llm_identifier="toy",
    )
    # Only the first item matches
    assert res.level1_accuracy == 0.5
    assert res.level2_accuracy is None
    assert res.llm_identifier == "toy"


def test_llm_causal_assess_requires_items():
    with pytest.raises(ValueError, match="At least one"):
        sp.llm_causal_assess(llm_client=lambda q: "x")


def test_llm_evaluator_in_registry():
    fns = set(sp.list_functions())
    assert "llm_causal_assess" in fns
    assert "pairwise_causal_benchmark" in fns
