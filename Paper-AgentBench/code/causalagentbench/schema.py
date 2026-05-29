"""Core data model for CausalAgentBench (JSS Track D).

The schema is a direct, code-level encoding of the OSF pre-registration
(``Paper-AgentBench/manuscript/notes/osf-preregistration.md``):

* a **Task** = research question + sandboxed data + a known gold answer +
  a grading rubric, tagged with a difficulty level (L1/L2/L3);
* a **Condition** = one of the six C1..C6 cells of the 3x2 factorial
  (StatsPAI / Pythonic / R)  x  (Claude / GPT);
* a **Trajectory** = everything an agent emitted on one trial;
* a **TrialResult** = the graded outcome of (task, condition, seed).

Nothing here imports an LLM SDK or statspai estimators, so the schema is
cheap to import and safe to use in pure-analysis contexts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Difficulty(str, Enum):
    """Pre-registered difficulty tiers (Track D task set)."""

    L1 = "L1"  # direct: the method is named in the prompt
    L2 = "L2"  # indirect: identification described, agent picks the method
    L3 = "L3"  # workflow: full pipeline incl. diagnostics + robustness


class Design(str, Enum):
    """Canonical identification strategies covered by the benchmark.

    Mirrors the designs frozen in the pre-registration / Track B
    (RCT, DiD incl. staggered, RD, IV, selection-on-observables).
    """

    RCT = "rct"
    DID = "did"
    STAGGERED_DID = "staggered_did"
    RD = "rd"
    IV = "iv"
    OBSERVATIONAL = "observational"  # conditional exogeneity / selection on observables


@dataclass
class Gold:
    """Frozen gold answer for a task.

    ``point_estimate`` is the reference number a trial is scored against
    (M1, the +/-5% success band). For synthetic tasks it is the *true*
    DGP effect (``source="dgp_true_effect"``); for canonical-dataset
    tasks it is the StatsPAI reference-pipeline estimate frozen at deposit
    time (``source="statspai_reference"``).
    """

    point_estimate: float
    design: Design
    estimand: str = "ATE"  # ATE / ATT / LATE — what the number means
    se: Optional[float] = None
    source: str = "dgp_true_effect"
    # Diagnostics a correct L3 workflow is expected to run (drives M6).
    required_diagnostics: List[str] = field(default_factory=list)
    # Methods judged correct for this task (drives M2). The first entry is
    # the canonical gold method; others are accepted equivalents.
    accepted_methods: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Task:
    """One benchmark prompt: question + data handle + gold + rubric."""

    task_id: str
    difficulty: Difficulty
    design: Design
    question: str
    # Column roles in the attached CSV (what the agent must analyse).
    roles: Dict[str, Any]  # e.g. {"y": "lwage", "treatment": "educ", "instrument": "nearc4"}
    gold: Gold
    # How the data is produced. Either a synthetic scenario (statspai DGP)
    # or a canonical dataset loader. Kept as a spec so tasks serialise
    # cleanly and re-materialise deterministically from a seed.
    data_spec: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["difficulty"] = self.difficulty.value
        d["design"] = self.design.value
        d["gold"]["design"] = self.gold.design.value
        return d


@dataclass
class AgentAction:
    """A single step in an agent's trajectory (tool call / code / message)."""

    kind: str  # "tool_call" | "code" | "message" | "final"
    name: str = ""
    payload: Any = None


@dataclass
class Trajectory:
    """Everything one agent emitted on one trial.

    The scorer reads ``final_estimate`` / ``final_method`` for the primary
    metrics and the action log for hallucination (M5) and diagnostic
    completeness (M6).
    """

    final_estimate: Optional[float] = None
    final_method: Optional[str] = None
    code: str = ""
    reported_diagnostics: List[str] = field(default_factory=list)
    called_functions: List[str] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    executed_ok: bool = False  # M3: code ran to completion w/o unhandled exception
    refused: bool = False  # agent declared the task unidentifiable / declined
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_s: float = 0.0
    error: str = ""
    raw_final: str = ""


@dataclass
class TrialResult:
    """Graded outcome of one (task, condition, seed) trial."""

    task_id: str
    condition: str  # "C1".."C6" or "oracle"
    seed: int
    difficulty: str
    design: str
    # Per-metric outcomes (see metrics.py for definitions).
    m1_success: Optional[bool] = None
    m2_method_correct: Optional[bool] = None
    m3_exec_ok: Optional[bool] = None
    m4_tokens: Optional[int] = None
    m5_hallucinated: Optional[bool] = None
    m6_diag_completeness: Optional[float] = None
    m7_estimate: Optional[float] = None  # for across-rep variance (M7)
    m8_wall_s: Optional[float] = None
    rel_error: Optional[float] = None
    refused: Optional[bool] = None
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
