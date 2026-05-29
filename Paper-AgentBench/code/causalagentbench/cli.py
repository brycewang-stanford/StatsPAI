"""``cab`` command-line interface.

  cab tasks                       list the demonstration task pack
  cab run  [--conditions ...]     run the grid, write JSONL
  cab analyze RESULTS.jsonl       summary table + H1..H5

Examples
--------
  cab run --conditions oracle --n 3 3 3 --out runs/oracle.jsonl
  cab run --conditions C1 C3 --seeds 0 1 2 --out runs/c1c3.jsonl
  cab analyze runs/oracle.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import load_tasks, run_suite, summarize, test_hypotheses
from .runner import load_results_jsonl


def _cmd_tasks(args) -> int:
    tasks = load_tasks(n_l1=args.n[0], n_l2=args.n[1], n_l3=args.n[2])
    for t in tasks:
        print(f"{t.task_id:<24} {t.difficulty.value:<3} {t.design.value:<16} "
              f"gold={t.gold.point_estimate:<5} method={t.gold.accepted_methods[0]}")
    print(f"\n{len(tasks)} tasks "
          f"(L1={args.n[0]}, L2={args.n[1]}, L3={args.n[2]}).")
    return 0


def _cmd_run(args) -> int:
    tasks = load_tasks(n_l1=args.n[0], n_l2=args.n[1], n_l3=args.n[2])
    results = run_suite(
        tasks, conditions=args.conditions, seeds=args.seeds,
        out_path=args.out, progress=not args.quiet,
        model=args.model, execute=not args.no_exec,
    )
    print("\n" + summarize(results))
    if args.out:
        print(f"\nWrote {len(results)} trials -> {args.out}")
    return 0


def _cmd_analyze(args) -> int:
    results = load_results_jsonl(args.results)
    print(summarize(results))
    print("\n=== Hypotheses (H1..H5) ===")
    H = test_hypotheses(results, B=args.bootstrap)
    print(json.dumps(H, indent=2, default=str))
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cab", description="CausalAgentBench CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("tasks", help="list the task pack")
    pt.add_argument("--n", type=int, nargs=3, default=[12, 12, 6],
                    metavar=("L1", "L2", "L3"))
    pt.set_defaults(func=_cmd_tasks)

    pr = sub.add_parser("run", help="run the grid")
    pr.add_argument("--conditions", nargs="+", default=["oracle"])
    pr.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    pr.add_argument("--n", type=int, nargs=3, default=[12, 12, 6],
                    metavar=("L1", "L2", "L3"))
    pr.add_argument("--out", default=None, help="JSONL output path")
    pr.add_argument("--model", default=None, help="override LLM model id")
    pr.add_argument("--no-exec", action="store_true",
                    help="do not execute agent code (parse inline answer only)")
    pr.add_argument("--quiet", action="store_true")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="summarise a results JSONL")
    pa.add_argument("results")
    pa.add_argument("--bootstrap", type=int, default=9999)
    pa.set_defaults(func=_cmd_analyze)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
