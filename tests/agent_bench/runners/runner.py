"""CausalAgentBench main runner: iterate over (cell, prompt, rep)
and emit one trial record per combination to results/trials.jsonl.

Default behaviour: --mock (no API calls). To run real APIs, pass
--api anthropic|openai|both and set ANTHROPIC_API_KEY / OPENAI_API_KEY.
The real-API path is intentionally a TODO stub here; flipping
mock=True to mock=False is the one-line change. The real production
run is gated on the OSF pre-registration deposit (see prompts/
_protocol.md) and the budget approval logged in NEXT-STEPS §M.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mock_llm import run_trial as run_mock_trial, to_dict as mock_to_dict


HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE.parent / "prompts" / "prompts.json"
GOLDS_PATH = HERE.parent / "golds" / "golds.json"
RESULTS_DIR = HERE.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CELL_LABELS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true", default=True,
                    help="Use the deterministic mock LLM (default).")
    p.add_argument("--api", choices=["anthropic", "openai", "both", "none"],
                    default="none",
                    help="Real API. Off by default until OSF + budget approved.")
    p.add_argument("--cells", default="C1,C2,C3,C4,C5,C6",
                    help="Comma-separated cell IDs to run.")
    p.add_argument("--prompts", default="all",
                    help="'all' or one of L1, L2, L3, or a comma-separated "
                         "list of prompt IDs.")
    p.add_argument("--reps", type=int, default=3,
                    help="Reps per (cell, prompt) -- default 3.")
    p.add_argument("--out", type=str, default="trials.jsonl",
                    help="Output JSONL filename inside results/.")
    return p.parse_args()


def filter_prompts(prompts: list[dict], spec: str) -> list[dict]:
    if spec == "all":
        return prompts
    if spec in ("L1", "L2", "L3"):
        return [p for p in prompts if p["level"] == spec]
    ids = {s.strip() for s in spec.split(",") if s.strip()}
    return [p for p in prompts if p["id"] in ids]


def main() -> None:
    args = parse_args()
    if args.api != "none":
        raise NotImplementedError(
            "Real-API path is gated on OSF pre-registration + budget "
            "approval; see tests/agent_bench/prompts/_protocol.md and "
            "Paper-JSS/NEXT-STEPS.md §M. Run with --mock for now."
        )

    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    golds_list = json.loads(GOLDS_PATH.read_text(encoding="utf-8"))
    golds = {g["id"]: g for g in golds_list}

    selected = filter_prompts(prompts, args.prompts)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out_path = RESULTS_DIR / args.out

    print(f"Running {len(cells)} cells x {len(selected)} prompts x {args.reps} reps "
          f"= {len(cells) * len(selected) * args.reps} trials")
    if args.mock:
        print("Mode: MOCK LLM (no API calls)")
    print(f"Out:   {out_path}")
    print()

    t0 = time.time()
    n_done = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for cell in cells:
            for prompt in selected:
                gold = golds[prompt["id"]]
                for rep in range(args.reps):
                    if args.mock:
                        trial = run_mock_trial(cell, prompt, gold, rep)
                        fh.write(json.dumps(mock_to_dict(trial)) + "\n")
                    n_done += 1
                    if n_done % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n_done} trials  ({elapsed:.1f}s)")
    elapsed = time.time() - t0
    print(f"Done: {n_done} trials in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
