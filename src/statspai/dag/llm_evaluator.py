"""
LLM causal-reasoning evaluator.

Benchmarks an LLM's causal ability along two axes identified by the
2024–2025 surveys (arXiv:2403.09606, 2409.09822, 2503.09326, 2509.00987):

* **Level-1 — knowledge retrieval**: given a pair of variables, can the
  LLM report the correct causal direction from its pre-trained
  knowledge? Evaluated on a user-supplied ground-truth dataset.
* **Level-2 — deductive reasoning**: given a DAG fragment and a do-calculus
  query, can the LLM identify the correct adjustment set or rule out
  non-identifiability?

Provides a common interface for any ``llm_client`` that returns a string
given a prompt. The evaluator itself is model-agnostic and contains no
prompt engineering — the caller owns the LLM specifics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "llm_causal_assess",
    "pairwise_causal_benchmark",
    "LLMCausalAssessResult",
    "PairwiseBenchmarkResult",
]


@dataclass
class LLMCausalAssessResult:
    """Output of :func:`llm_causal_assess`."""
    level1_accuracy: Optional[float]
    level2_accuracy: Optional[float]
    per_item: pd.DataFrame       # columns: id, level, question, truth, pred, correct
    llm_identifier: str

    def summary(self) -> str:
        lines = [
            "LLM Causal Reasoning Assessment",
            "=" * 60,
            f"  LLM                : {self.llm_identifier}",
            f"  # items (total)    : {len(self.per_item)}",
        ]
        if self.level1_accuracy is not None:
            n1 = (self.per_item["level"] == 1).sum()
            lines.append(f"  Level-1 accuracy   : {self.level1_accuracy:.4f}  (n={n1})")
        if self.level2_accuracy is not None:
            n2 = (self.per_item["level"] == 2).sum()
            lines.append(f"  Level-2 accuracy   : {self.level2_accuracy:.4f}  (n={n2})")
        return "\n".join(lines)


@dataclass
class PairwiseBenchmarkResult:
    """Output of :func:`pairwise_causal_benchmark`."""
    accuracy: float
    precision_forward: float
    recall_forward: float
    per_pair: pd.DataFrame

    def summary(self) -> str:
        return "\n".join([
            "Pairwise Causal Discovery Benchmark",
            "=" * 60,
            f"  Accuracy            : {self.accuracy:.4f}",
            f"  Precision (A -> B)  : {self.precision_forward:.4f}",
            f"  Recall (A -> B)     : {self.recall_forward:.4f}",
            f"  # pairs             : {len(self.per_pair)}",
        ])


def _parse_yes_no(text: str) -> Optional[bool]:
    """Extract a yes/no from a free-form LLM response."""
    lower = text.strip().lower()
    if lower.startswith("yes") or "->" in lower:
        return True
    if lower.startswith("no") or lower.startswith("false"):
        return False
    return None


def pairwise_causal_benchmark(
    ground_truth: pd.DataFrame,
    *,
    llm_client: Callable[[str], str],
    llm_identifier: str = "llm",
    pair_a_col: str = "A",
    pair_b_col: str = "B",
    truth_col: str = "a_causes_b",
    prompt_template: str = (
        "Does variable {a} causally influence variable {b}? Answer 'yes' or 'no'."
    ),
) -> PairwiseBenchmarkResult:
    """Benchmark an LLM on pairwise causal-direction identification.

    Parameters
    ----------
    ground_truth : DataFrame
        One row per pair with columns ``pair_a_col``, ``pair_b_col``, and
        a boolean ``truth_col`` indicating whether A causally influences B.
    llm_client : callable(str) -> str
        Function taking a prompt and returning a string.
    llm_identifier : str, default 'llm'
    prompt_template : str, default ``"Does variable {a} causally ..."``

    Returns
    -------
    PairwiseBenchmarkResult
    """
    required = {pair_a_col, pair_b_col, truth_col}
    missing = required - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    rows = []
    for _, row in ground_truth.iterrows():
        prompt = prompt_template.format(a=row[pair_a_col], b=row[pair_b_col])
        raw = llm_client(prompt)
        pred = _parse_yes_no(raw)
        truth = bool(row[truth_col])
        correct = (pred == truth) if pred is not None else False
        rows.append({
            "A": row[pair_a_col], "B": row[pair_b_col],
            "truth": truth, "pred": pred, "correct": correct,
            "raw_response": raw,
        })
    per_pair = pd.DataFrame(rows)
    acc = float(per_pair["correct"].mean())
    # Precision (forward): of predicted-forward, how many are truly-forward.
    pred_series = per_pair["pred"]
    forward_pred_mask = pred_series == True  # noqa: E712
    if forward_pred_mask.any():
        precision = float(
            (per_pair.loc[forward_pred_mask, "truth"] == True).mean()
        )
    else:
        precision = float("nan")
    # Recall (forward): of truly-forward, how many were predicted-forward.
    truly_forward = per_pair["truth"] == True  # noqa: E712
    if truly_forward.any():
        recall = float(
            (per_pair.loc[truly_forward, "pred"] == True).mean()
        )
    else:
        recall = float("nan")
    return PairwiseBenchmarkResult(
        accuracy=acc,
        precision_forward=precision,
        recall_forward=recall,
        per_pair=per_pair,
    )


def llm_causal_assess(
    level1_items: Optional[pd.DataFrame] = None,
    level2_items: Optional[pd.DataFrame] = None,
    *,
    llm_client: Callable[[str], str],
    llm_identifier: str = "llm",
) -> LLMCausalAssessResult:
    """Combined Level-1 + Level-2 LLM causal-reasoning assessment.

    Parameters
    ----------
    level1_items : DataFrame, optional
        Columns: ``question``, ``answer``. The LLM's response is
        marked correct if the target ``answer`` appears (case-insensitive)
        in the response.
    level2_items : DataFrame, optional
        Columns: ``question``, ``answer``. Level-2 questions ask the LLM
        to reason about a DAG fragment; answer checking uses substring
        match.
    llm_client, llm_identifier

    Returns
    -------
    LLMCausalAssessResult
    """
    rows: List[Dict[str, Any]] = []

    def _score(level: int, items: Optional[pd.DataFrame]):
        if items is None or len(items) == 0:
            return None
        if not {"question", "answer"} <= set(items.columns):
            raise ValueError(f"Level-{level} items need 'question' and 'answer' columns.")
        correct = 0
        for i, row in items.iterrows():
            raw = llm_client(row["question"])
            ans = str(row["answer"]).strip().lower()
            pred_correct = bool(ans and ans in raw.lower())
            rows.append({
                "id": i, "level": level,
                "question": row["question"],
                "truth": row["answer"],
                "pred": raw,
                "correct": pred_correct,
            })
            if pred_correct:
                correct += 1
        return correct / len(items)

    lvl1 = _score(1, level1_items)
    lvl2 = _score(2, level2_items)
    if lvl1 is None and lvl2 is None:
        raise ValueError(
            "At least one of level1_items / level2_items must be supplied."
        )
    return LLMCausalAssessResult(
        level1_accuracy=lvl1,
        level2_accuracy=lvl2,
        per_item=pd.DataFrame(rows),
        llm_identifier=llm_identifier,
    )
