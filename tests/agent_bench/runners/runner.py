"""CausalAgentBench runner: (cell, task, rep) -> results/trials.jsonl.

Modes
-----
``--model oracle``  replays the reference solutions (``oracle_statspai.py``
                    for StatsPAI cells, ``oracle_other.py`` for the others,
                    where one exists) through the full loop. No API, no cost;
                    it checks the harness and the golds.
``--model api``     the production run. Refuses unless the pre-registration
                    lock verifies (``prereg.py``); the model for each cell
                    comes from the lock.

Usage::

    python runners/runner.py --model oracle --cells C1 --tasks L1
    python runners/runner.py --model api --cells C1,C2,C3,C4,C5,C6 --tasks all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent_loop import run_trial  # noqa: E402
from cells import CELLS  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
TASKS = ROOT / "prompts" / "prompts.json"


def select(tasks: List[Dict[str, Any]], spec: str) -> List[Dict[str, Any]]:
    if spec == "all":
        return tasks
    if spec in ("L1", "L2", "L3"):
        return [t for t in tasks if t["level"] == spec]
    ids = {s.strip() for s in spec.split(",") if s.strip()}
    return [t for t in tasks if t["id"] in ids]


def oracle_model(task_id: str, cell):
    from models import OracleModel

    if cell.toolset == "statspai":
        from oracle_statspai import H, SOLUTIONS

        if task_id not in SOLUTIONS:
            return None
        est, code = SOLUTIONS[task_id]
        return OracleModel(H + code, est, "run_python")
    from oracle_other import PYTHONIC, R_SOLUTIONS

    table = R_SOLUTIONS if cell.toolset == "r" else PYTHONIC
    if task_id not in table:
        return None
    est, code = table[task_id]
    return OracleModel(code, est, "run_r" if cell.toolset == "r" else "run_python")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["oracle", "api"], default="oracle")
    ap.add_argument("--cells", default="C1")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--out", default="trials.jsonl")
    args = ap.parse_args()

    tasks = select(json.loads(TASKS.read_text(encoding="utf-8"))["tasks"], args.tasks)
    cells = [CELLS[c.strip()] for c in args.cells.split(",") if c.strip()]
    settings: Dict[str, Any] = {"reps": 1, "max_turns": 20, "call_timeout_seconds": 300}
    lock = None
    if args.model == "api":
        from prereg import verify

        lock = verify()  # raises before any API call when the lock is absent or stale
        settings.update(lock["settings"])
    reps = args.reps if args.reps is not None else settings["reps"]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / args.out
    n = 0
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for cell in cells:
            for task in tasks:
                for rep in range(reps):
                    if args.model == "api":
                        from models import real_model

                        model = real_model(cell.family, lock)
                    else:
                        model = oracle_model(task["id"], cell)
                        if model is None:
                            continue
                    trial = run_trial(
                        task,
                        cell,
                        model,
                        rep=rep,
                        max_turns=settings["max_turns"],
                        call_timeout=settings["call_timeout_seconds"],
                    )
                    if lock is not None:
                        trial["osf_doi"] = lock["osf_doi"]
                    fh.write(json.dumps(trial, default=str) + "\n")
                    fh.flush()
                    n += 1
    print(f"{n} trials in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
