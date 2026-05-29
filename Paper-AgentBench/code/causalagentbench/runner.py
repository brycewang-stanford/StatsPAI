"""Trial orchestration: run (task x condition x seed) cells and collect grades.

The full pre-registered run is 50 prompts x 6 conditions x 3 seeds = 900
trials. ``run_suite`` is generic over the task list, condition list, and
seed list, so a 3x1x3 dry-run and the full 50x6x3 run use the same path.
Results stream to a JSONL file (one TrialResult per line) so a long run is
crash-resumable and analysable incrementally.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .adapters import build_adapter
from .scoring import grade
from .schema import Task, TrialResult


def run_trial(task: Task, condition_code: str, seed: int,
              adapter=None, **adapter_kwargs) -> TrialResult:
    """Run and grade one trial."""
    adapter = adapter or build_adapter(condition_code, **adapter_kwargs)
    traj = adapter.run(task, seed=seed)
    return grade(task, condition_code, seed, traj)


def run_suite(
    tasks: List[Task],
    conditions: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    out_path: Optional[str] = None,
    progress: bool = True,
    **adapter_kwargs,
) -> List[TrialResult]:
    """Run the full grid and return the graded trials.

    Parameters
    ----------
    conditions : defaults to ``["oracle"]`` (no API key needed).
    seeds : defaults to ``[0, 1, 2]`` (the 3 pre-registered reps).
    out_path : if given, append each TrialResult as a JSONL line.
    """
    conditions = conditions or ["oracle"]
    seeds = seeds or [0, 1, 2]
    results: List[TrialResult] = []

    # Reuse one adapter per condition (LLM clients are expensive to build).
    adapters: Dict[str, object] = {}

    fh = None
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fh = open(out_path, "a", encoding="utf-8")

    total = len(tasks) * len(conditions) * len(seeds)
    n = 0
    try:
        for cond in conditions:
            if cond not in adapters:
                adapters[cond] = build_adapter(cond, **adapter_kwargs)
            adapter = adapters[cond]
            for task in tasks:
                for seed in seeds:
                    n += 1
                    res = run_trial(task, cond, seed, adapter=adapter)
                    results.append(res)
                    if fh:
                        fh.write(res.to_json() + "\n")
                        fh.flush()
                    if progress:
                        flag = "ok" if res.m1_success else ("·" if res.m1_success is None else "✗")
                        print(f"[{n:>4}/{total}] {cond:<7} {task.task_id:<20} "
                              f"m1={flag} method={res.m2_method_correct} "
                              f"est={res.m7_estimate}")
    finally:
        if fh:
            fh.close()
    return results


def load_results_jsonl(path: str) -> List[TrialResult]:
    """Re-read a JSONL results file into TrialResult objects."""
    out: List[TrialResult] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(TrialResult(**json.loads(line)))
    return out
