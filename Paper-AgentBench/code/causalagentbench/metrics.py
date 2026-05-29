"""The eight pre-registered metrics M1..M8 (OSF pre-registration, Track D).

Primary:   M1 task success, M2 method correctness, M3 code-exec success,
           M4 token efficiency.
Secondary: M5 hallucination, M6 diagnostic completeness, M7 reproducibility
           (across-rep estimate variance, computed at analysis time),
           M8 time-to-result.

Each function is pure: (Task, Trajectory) -> metric value. M7 is computed
across replications in ``stats.py``, not here.
"""

from __future__ import annotations

import functools
from typing import List, Optional

from .schema import Task, Trajectory

# Success band for M1: final estimate within +/-5% of gold (pre-registered).
SUCCESS_REL_TOL = 0.05


def relative_error(estimate: Optional[float], gold: float) -> Optional[float]:
    if estimate is None:
        return None
    denom = abs(gold) if abs(gold) > 1e-12 else 1.0
    return abs(estimate - gold) / denom


def m1_success(task: Task, traj: Trajectory) -> Optional[bool]:
    """Final point estimate within +/-5% of the gold answer."""
    if traj.refused:
        return False
    re = relative_error(traj.final_estimate, task.gold.point_estimate)
    return None if re is None else (re <= SUCCESS_REL_TOL)


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def m2_method_correct(task: Task, traj: Trajectory) -> Optional[bool]:
    """Reported method is in the gold rubric's accepted set.

    Matching is tolerant: a method counts as correct if any accepted alias
    is a normalised substring of (or equals) the reported method string.
    The staggered-DiD rubric deliberately excludes plain TWFE, so an agent
    that reaches for ``twfe`` on a staggered task scores M2 = False.
    """
    if not traj.final_method:
        return None
    reported = _norm(traj.final_method)
    # One-directional: an accepted alias must appear *within* the reported
    # method. The reverse direction (reported within alias) is unsafe — it
    # would let "twfe" match "etwfe" and defeat the staggered-DiD trap,
    # where plain two-way fixed effects is deliberately NOT accepted.
    for alias in task.gold.accepted_methods:
        a = _norm(alias)
        if a and a in reported:
            return True
    return False


def m3_exec_ok(task: Task, traj: Trajectory) -> bool:
    return bool(traj.executed_ok)


def m4_tokens(task: Task, traj: Trajectory) -> int:
    return int(traj.input_tokens + traj.output_tokens)


@functools.lru_cache(maxsize=1)
def _known_statspai_names() -> frozenset:
    try:
        import statspai as sp

        return frozenset(n for n in dir(sp) if not n.startswith("_"))
    except Exception:  # pragma: no cover
        return frozenset()


def m5_hallucinated(task: Task, traj: Trajectory) -> Optional[bool]:
    """True if the agent called a StatsPAI function that does not exist.

    Machine-checkable for the StatsPAI stack (C1/C2 and the oracle):
    a ``statspai.<name>`` call whose ``<name>`` is not an exported symbol
    is a hallucinated API. For non-StatsPAI stacks this returns None
    (the pre-registration routes those to manual review).
    """
    known = _known_statspai_names()
    saw_statspai = False
    for fn in traj.called_functions:
        if fn.startswith("statspai.") or fn.startswith("sp."):
            saw_statspai = True
            name = fn.split(".", 1)[1]
            if name not in known:
                return True
    return False if saw_statspai else None


def m6_diag_completeness(task: Task, traj: Trajectory) -> float:
    """Share of the design's required diagnostics the agent reported.

    Non-L3 tasks have no required diagnostics -> defined as 1.0.
    """
    required: List[str] = task.gold.required_diagnostics
    if not required:
        return 1.0
    reported = {_norm(d) for d in traj.reported_diagnostics}
    hit = sum(1 for d in required if _norm(d) in reported)
    return hit / len(required)


def m8_wall_s(task: Task, traj: Trajectory) -> float:
    return float(traj.wall_clock_s)
