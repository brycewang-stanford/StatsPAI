"""Grade one trajectory into a TrialResult by applying M1..M8."""

from __future__ import annotations

from . import metrics
from .schema import Task, TrialResult, Trajectory


def grade(task: Task, condition_code: str, seed: int, traj: Trajectory) -> TrialResult:
    return TrialResult(
        task_id=task.task_id,
        condition=condition_code,
        seed=seed,
        difficulty=task.difficulty.value,
        design=task.design.value,
        m1_success=metrics.m1_success(task, traj),
        m2_method_correct=metrics.m2_method_correct(task, traj),
        m3_exec_ok=metrics.m3_exec_ok(task, traj),
        m4_tokens=metrics.m4_tokens(task, traj),
        m5_hallucinated=metrics.m5_hallucinated(task, traj),
        m6_diag_completeness=metrics.m6_diag_completeness(task, traj),
        m7_estimate=traj.final_estimate,
        m8_wall_s=metrics.m8_wall_s(task, traj),
        rel_error=metrics.relative_error(traj.final_estimate, task.gold.point_estimate),
        refused=bool(traj.refused),
        error=traj.error,
    )
