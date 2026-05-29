"""The 3x2 factorial conditions C1..C6 (OSF pre-registration, Track D).

                | Claude            | GPT
  --------------+-------------------+-------------------
  StatsPAI+MCP  | C1                | C2
  Pythonic      | C3                | C4
  R via MCP     | C5                | C6

Each cell maps a *stack* (the tool surface available to the agent) to a
*provider* (the LLM family). ``oracle`` is an extra, non-pre-registered
reference cell: a deterministic StatsPAI pipeline with no LLM, used for
dry-runs, CI, and as an upper-bound calibration point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Condition:
    code: str
    stack: str       # "statspai" | "pythonic" | "r" | "oracle"
    provider: str    # "claude" | "gpt" | "none"
    label: str


CONDITIONS: Dict[str, Condition] = {
    "C1": Condition("C1", "statspai", "claude", "StatsPAI+MCP / Claude"),
    "C2": Condition("C2", "statspai", "gpt", "StatsPAI+MCP / GPT"),
    "C3": Condition("C3", "pythonic", "claude", "Pythonic stack / Claude"),
    "C4": Condition("C4", "pythonic", "gpt", "Pythonic stack / GPT"),
    "C5": Condition("C5", "r", "claude", "R via MCP / Claude"),
    "C6": Condition("C6", "r", "gpt", "R via MCP / GPT"),
    # Non-pre-registered reference cell (no LLM, deterministic).
    "oracle": Condition("oracle", "oracle", "none", "StatsPAI reference oracle (no LLM)"),
}

# Pythonic-stack package allow-list (no StatsPAI surface).
PYTHONIC_STACK: List[str] = [
    "statsmodels", "linearmodels", "doubleml", "grf-python",
    "econml", "causalml", "scipy",
]

# R-via-MCP package allow-list.
R_STACK: List[str] = [
    "MatchIt", "did", "fixest", "rdrobust", "Synth",
    "synthdid", "DoubleML", "grf", "HonestDiD",
]


def get(code: str) -> Condition:
    if code not in CONDITIONS:
        raise KeyError(
            f"Unknown condition '{code}'. Available: {list(CONDITIONS)}"
        )
    return CONDITIONS[code]
