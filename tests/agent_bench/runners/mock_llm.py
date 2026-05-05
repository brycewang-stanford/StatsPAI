"""Deterministic mock LLM for CausalAgentBench harness validation.

The real harness will call Anthropic / OpenAI APIs. The mock LLM
returns fake tool-call traces so the runner + grader pipeline can
be smoke-tested end-to-end without API cost. The mock is biased
toward the gold-correct estimator with cell-dependent noise so
the pipeline picks up sensible scores.

Behaviour parameters:
  * StatsPAI cells (C1, C2):       95% gold-correct, 100 tokens / call
  * Pythonic-stack cells (C3, C4): 75% gold-correct, 250 tokens / call
                                   + 5% hallucinated function name
  * R-via-MCP cells (C5, C6):      90% gold-correct, 350 tokens / call

These numbers are placeholders; only the harness's plumbing is
being validated.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GOLDS_PATH = HERE.parent / "golds" / "golds.json"


@dataclass
class MockTrace:
    cell: str
    prompt_id: str
    rep: int
    final_estimate: float | None
    final_estimator: str
    n_tokens_in: int
    n_tokens_out: int
    n_tool_calls: int
    hallucinated_call: bool
    transcript: str  # short redacted text


CELL_BEHAVIOUR = {
    # cell -> (gold_correct_prob, tokens_in, tokens_out, hallucination_prob)
    "C1": (0.95, 80, 50, 0.01),
    "C2": (0.93, 80, 60, 0.02),
    "C3": (0.75, 200, 150, 0.07),
    "C4": (0.72, 200, 160, 0.08),
    "C5": (0.90, 250, 200, 0.03),
    "C6": (0.88, 250, 210, 0.03),
}


def _seeded_rng(cell: str, prompt_id: str, rep: int) -> random.Random:
    """Stable per-(cell, prompt, rep) RNG so a re-run is bit-identical."""
    return random.Random(f"{cell}__{prompt_id}__{rep}".__hash__())


def _gold_value(gold: dict[str, Any]) -> float | None:
    """Extract a single comparable scalar from the gold record."""
    if "expected_values" in gold and gold["expected_values"]:
        # Pick the first numeric value in the dict.
        for v in gold["expected_values"].values():
            if isinstance(v, (int, float)):
                return float(v)
    if "expected_value_range" in gold and gold["expected_value_range"]:
        for v in gold["expected_value_range"].values():
            if isinstance(v, list) and len(v) == 2:
                return float((v[0] + v[1]) / 2)
    return None


def run_trial(cell: str, prompt: dict[str, Any], gold: dict[str, Any], rep: int) -> MockTrace:
    rng = _seeded_rng(cell, prompt["id"], rep)
    p_correct, t_in, t_out, p_halluc = CELL_BEHAVIOUR[cell]

    is_hallucinated = rng.random() < p_halluc
    is_correct = (not is_hallucinated) and (rng.random() < p_correct)

    expected_est = gold.get("expected_estimator")
    final_estimator = (
        expected_est if is_correct else
        f"hallucinated_{rng.choice(['fakefunc', 'imaginarymethod'])}"
        if is_hallucinated else
        rng.choice([expected_est, "ols", "regression"])
    )

    target = _gold_value(gold)
    if target is None:
        final_estimate = None
    elif is_correct:
        final_estimate = target * (1.0 + rng.gauss(0, 0.02))
    else:
        final_estimate = target * (1.0 + rng.gauss(0, 0.50))

    n_tool_calls = rng.randint(2, 6) if cell.startswith("C") else 4

    transcript = (
        f"[mock {cell} on {prompt['id']} rep={rep}]\n"
        f"<user>{prompt['question'][:140]}...</user>\n"
        f"<assistant>{'I will use ' + final_estimator if not is_hallucinated else 'Let me try ' + final_estimator}</assistant>\n"
        f"<tool>{final_estimator}(...)</tool>\n"
        f"<assistant>The estimate is {final_estimate}</assistant>\n"
    )

    return MockTrace(
        cell=cell, prompt_id=prompt["id"], rep=rep,
        final_estimate=final_estimate,
        final_estimator=final_estimator,
        n_tokens_in=t_in + rng.randint(-20, 30),
        n_tokens_out=t_out + rng.randint(-20, 30),
        n_tool_calls=n_tool_calls,
        hallucinated_call=is_hallucinated,
        transcript=transcript,
    )


def to_dict(t: MockTrace) -> dict[str, Any]:
    return asdict(t)
