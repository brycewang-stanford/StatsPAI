"""CausalAgentBench v2 grader: per-trial scores and the H1-H5 tests.

Every metric is computed from the trial record alone; nothing reads free
text or asks a model to judge.

Per trial
---------
``submitted``         the agent called submit_answer.
``task_success``      the headline value is within tolerance of some
                      variant in golds.json (tolerance = max(rel * |gold|, abs)).
``method_correct``    the reported estimator is the estimator of some variant.
``values_correct``    share of the graded fields of the closest variant that
                      are within tolerance.
``required_complete`` every field the task lists as ``required`` is non-null.
``fabricated``        the headline was submitted but no tool output contains
                      a number within 1e-3 (relative) of it: a number the
                      agent did not compute.
``invented_api``      a tool output shows a call to something that does not
                      exist (missing attribute or function, unknown keyword
                      argument, package or module not installed -- except the
                      StatsPAI import the pythonic toolset blocks on purpose,
                      which is ``toolset_violation``), or a call to a tool
                      that is not offered.
``hallucination``     fabricated or invented_api.
``tokens``            every token processed over the trial: input, cache
                      reads, cache writes and output (the H4 measure).
``tokens_uncached``   the same without cache reads (secondary measure).

Hypotheses (prompts/_protocol.md)
---------------------------------
Cluster bootstrap with the prompt as cluster, B resamples, intervals at
1 - 0.05/5 (Bonferroni over H1-H5).

Usage::

    python runners/grader.py --in results/trials.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
TASKS = ROOT / "prompts" / "prompts.json"
GOLDS = ROOT / "golds" / "golds.json"

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_INVENTED = re.compile(
    r"has no attribute|could not find function|there is no package called|"
    r"unexpected keyword argument|unused argument|No module named|"
    r"object is not callable|cannot import name",
)
_BLOCKED = "No module named 'statspai' (not available in this toolset)"
STATSPAI = ("C1", "C2")
PYTHONIC = ("C3", "C4")
R_CELLS = ("C5", "C6")


def load_tasks() -> Dict[str, Dict[str, Any]]:
    return {t["id"]: t for t in json.loads(TASKS.read_text(encoding="utf-8"))["tasks"]}


def load_golds() -> Dict[str, Dict[str, Any]]:
    return {g["id"]: g for g in json.loads(GOLDS.read_text(encoding="utf-8"))["golds"]}


def _within(value: Any, gold: float, tol: Dict[str, float]) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return abs(v - gold) <= max(tol["rel"] * abs(gold), tol["abs"])


def _numbers_in(text: str) -> List[float]:
    out = []
    for m in _NUMBER.finditer(text or ""):
        try:
            out.append(float(m.group(0)))
        except ValueError:  # pragma: no cover - the regex only matches numbers
            pass
    return out


def _appears(value: Any, outputs: Iterable[str]) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    scale = max(abs(v), 1e-12)
    for text in outputs:
        for x in _numbers_in(text):
            if abs(x - v) <= 1e-3 * scale:
                return True
    return False


def score_trial(
    trial: Dict[str, Any], task: Dict[str, Any], gold: Dict[str, Any]
) -> Dict[str, Any]:
    sub = trial.get("submitted") or {}
    headline = gold["headline"]
    head_value = sub.get(headline)
    variants = gold["variants"]
    success = any(
        _within(head_value, v["values"][headline], v["tolerance"]) for v in variants
    )
    method_ok = sub.get("estimator") in {v["estimator"] for v in variants}

    def frac(v: Dict[str, Any]) -> float:
        hits = [_within(sub.get(k), x, v["tolerance"]) for k, x in v["values"].items()]
        return sum(hits) / len(hits)

    values_correct = max((frac(v) for v in variants), default=0.0) if sub else 0.0
    required = all(sub.get(k) is not None for k in gold.get("required", []))
    outputs = [
        c.get("output", "")
        for c in trial.get("tool_calls", [])
        if c["name"] != "submit_answer"
    ]
    fabricated = head_value is not None and not _appears(head_value, outputs)
    blocked = any(_BLOCKED in o for o in outputs)
    invented = any(c.get("unknown_tool") for c in trial.get("tool_calls", [])) or any(
        _INVENTED.search(o.replace(_BLOCKED, "")) for o in outputs
    )
    usage = trial.get("usage", {})
    return {
        "task_id": trial["task_id"],
        "level": task["level"],
        "cell": trial["cell"],
        "rep": trial.get("rep", 0),
        "model": trial.get("model", ""),
        "submitted": int(bool(trial.get("submitted"))),
        "task_success": int(success),
        "method_correct": int(method_ok),
        "values_correct": round(values_correct, 4),
        "required_complete": int(required),
        "fabricated": int(fabricated),
        "invented_api": int(invented),
        "hallucination": int(fabricated or invented),
        "toolset_violation": int(blocked),
        "tokens": sum(
            int(usage.get(k, 0))
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            )
        ),
        "tokens_uncached": sum(
            int(usage.get(k, 0))
            for k in ("input_tokens", "output_tokens", "cache_write_tokens")
        ),
        "turns": int(trial.get("turns", 0)),
        "tool_errors": sum(
            1 for c in trial.get("tool_calls", []) if not c.get("ok", True)
        ),
        "stop": trial.get("stop", ""),
    }


# ---------------------------------------------------------------------------
# Aggregation and the pre-registered tests
# ---------------------------------------------------------------------------


def _cluster_bootstrap(
    rows: Sequence[Dict[str, Any]],
    stat,
    *,
    B: int,
    seed: int,
    alpha: float,
) -> Dict[str, float]:
    prompts = sorted({r["task_id"] for r in rows})
    by_prompt: Dict[str, List[Dict[str, Any]]] = {p: [] for p in prompts}
    for r in rows:
        by_prompt[r["task_id"]].append(r)
    point = stat(rows)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        pick = rng.choice(len(prompts), size=len(prompts), replace=True)
        sample = [r for i in pick for r in by_prompt[prompts[i]]]
        value = stat(sample)
        if value is not None and math.isfinite(value):
            draws.append(value)
    lo, hi = (
        np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if draws else (np.nan, np.nan)
    )
    return {
        "estimate": point,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_prompts": len(prompts),
    }


def _rate(metric: str, cells: Sequence[str], levels: Optional[Sequence[str]] = None):
    def stat(rows):
        sel = [
            r[metric]
            for r in rows
            if r["cell"] in cells and (levels is None or r["level"] in levels)
        ]
        return float(np.mean(sel)) if sel else None

    return stat


def _diff(metric, a, b, levels=None):
    ra, rb = _rate(metric, a, levels), _rate(metric, b, levels)

    def stat(rows):
        x, y = ra(rows), rb(rows)
        return None if x is None or y is None else x - y

    return stat


def _token_ratio(a, b):
    def stat(rows):
        ta = [r["tokens"] for r in rows if r["cell"] in a]
        tb = [r["tokens"] for r in rows if r["cell"] in b]
        if not ta or not tb or np.mean(tb) == 0:
            return None
        return float(np.mean(ta) / np.mean(tb))

    return stat


def hypotheses(
    rows: Sequence[Dict[str, Any]], *, B: int = 10000, seed: int = 20261017
) -> Dict[str, Any]:
    alpha = 0.05 / 5
    boot = lambda stat: _cluster_bootstrap(
        rows, stat, B=B, seed=seed, alpha=alpha
    )  # noqa: E731
    out: Dict[str, Any] = {}
    h1 = {}
    for level, threshold in (("L1", 0.90), ("L2", 0.70), ("L3", 0.50)):
        r = boot(_rate("task_success", STATSPAI, [level]))
        r["threshold"] = threshold
        r["supported"] = bool(r["ci_lower"] >= threshold)
        h1[level] = r
    out["H1"] = h1
    r = boot(_diff("task_success", STATSPAI, PYTHONIC, ["L2", "L3"]))
    r["supported"] = bool(r["ci_lower"] >= 0.15)
    out["H2"] = r
    s = boot(_rate("hallucination", STATSPAI))
    p = boot(_rate("hallucination", PYTHONIC))
    out["H3"] = {
        "statspai": s,
        "pythonic": p,
        "supported": bool(s["ci_upper"] < 0.05 and p["ci_lower"] >= 0.15),
    }
    r = boot(_token_ratio(STATSPAI, PYTHONIC))
    r["supported"] = bool(r["ci_upper"] <= 0.60)
    out["H4"] = r
    d = boot(_diff("task_success", R_CELLS, STATSPAI))
    t = boot(_token_ratio(R_CELLS, STATSPAI))
    out["H5"] = {
        "success_difference": d,
        "token_ratio": t,
        # Equivalence on success within +/- 10 points, and the token claim.
        "supported": bool(
            d["ci_lower"] > -0.10 and d["ci_upper"] < 0.10 and t["ci_lower"] >= 1.5
        ),
    }
    out["alpha_per_hypothesis"] = alpha
    out["bootstrap"] = {"B": B, "seed": seed, "cluster": "prompt"}
    return out


def by_cell(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cells = sorted({r["cell"] for r in rows})
    metrics = (
        "submitted",
        "task_success",
        "method_correct",
        "values_correct",
        "required_complete",
        "hallucination",
        "fabricated",
        "invented_api",
        "tokens",
        "tokens_uncached",
        "turns",
    )
    table = []
    for c in cells:
        sel = [r for r in rows if r["cell"] == c]
        table.append(
            {
                "cell": c,
                "n": len(sel),
                **{m: float(np.mean([r[m] for r in sel])) for m in metrics},
            }
        )
    return table


def grade(trials: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks, golds = load_tasks(), load_golds()
    return [score_trial(t, tasks[t["task_id"]], golds[t["task_id"]]) for t in trials]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default=str(RESULTS / "trials.jsonl"))
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()
    trials = [
        json.loads(l)
        for l in Path(args.inp).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rows = grade(trials)
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"by_cell": by_cell(rows), "hypotheses": hypotheses(rows, B=args.B)}
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["by_cell"], indent=2))


if __name__ == "__main__":
    main()
