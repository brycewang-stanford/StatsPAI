"""CausalAgentBench — behavioural benchmark for LLM agents doing causal inference.

A code-level implementation of JSS Track D (see the OSF pre-registration in
``StatsPAI/Paper-AgentBench/manuscript/notes/osf-preregistration.md``):
50 prompts x 6 conditions x 3 seeds = 900 trials of LLM agents performing
causal-inference tasks, graded on eight pre-registered metrics.

Built on StatsPAI's known-truth DGPs and canonical datasets, so gold answers
are *true* causal effects rather than other papers' estimates.

Quick start (no API key needed — runs the StatsPAI reference oracle):

>>> from causalagentbench import load_tasks, run_suite
>>> tasks = load_tasks(n_l1=3, n_l2=3, n_l3=3)
>>> results = run_suite(tasks, conditions=["oracle"], seeds=[0, 1, 2])
>>> from causalagentbench import summarize
>>> print(summarize(results))
"""

from __future__ import annotations

__version__ = "0.1.0"

from .schema import (
    Difficulty,
    Design,
    Gold,
    Task,
    Trajectory,
    TrialResult,
)
from .tasks import load_tasks, materialize
from .conditions import CONDITIONS, Condition, get as get_condition
from .scoring import grade
from .runner import run_suite, run_trial
from .stats import summarize, test_hypotheses

__all__ = [
    "Difficulty", "Design", "Gold", "Task", "Trajectory", "TrialResult",
    "load_tasks", "materialize",
    "CONDITIONS", "Condition", "get_condition",
    "grade", "run_suite", "run_trial",
    "summarize", "test_hypotheses",
]
