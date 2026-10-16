"""Cost ceiling for a CausalAgentBench production run (no API calls).

A trial is a multi-turn loop: every turn re-sends the system prompt, the
tool definitions and the growing history. The ceiling assumes

* ``--turns`` turns per trial (the loop's hard limit is 20);
* the tool definitions of each cell, measured here from the live tool lists
  (four characters per token);
* ``--history`` tokens added to the history per turn (code + output);
* everything but the newest turn served from the prompt cache.

Prices are per million tokens. The Anthropic defaults are the list prices
in the Claude API documentation as of 2026-10; OpenAI prices must be passed
in from the provider's pricing page -- this script does not guess them.
Re-check both immediately before the run and set a hard cap in the
providers' billing dashboards.

Usage::

    python runners/estimate_budget.py --openai-input 2.5 --openai-output 10 \\
        --openai-cached 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cells import CELLS, code_tool, submit_tool, system_prompt  # noqa: E402

#: Claude Opus 5.5 list prices (USD / 1M tokens): input, output, cache read.
ANTHROPIC = {"input": 4.00, "output": 20.00, "cached": 0.20, "cache_write_factor": 1.25}


def _tokens(obj) -> int:
    return len(json.dumps(obj)) // 4


def tool_tokens(cell) -> int:
    tools = [code_tool(cell), submit_tool(["a", "b"])]
    if cell.toolset == "statspai":
        from cells import mcp_tools

        tools += mcp_tools()
    return _tokens(tools) + _tokens(system_prompt(cell))


def trial_cost(
    prefix: int, turns: int, history: int, out_per_turn: int, price
) -> float:
    cost = 0.0
    for t in range(turns):
        cached = prefix + history * t
        fresh = history
        cost += (
            cached * price["cached"]
            + fresh * price["input"] * price["cache_write_factor"]
        )
        cost += out_per_turn * price["output"]
    return cost / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--history", type=int, default=1500)
    ap.add_argument("--output", type=int, default=800, help="output tokens per turn")
    ap.add_argument("--openai-input", type=float)
    ap.add_argument("--openai-output", type=float)
    ap.add_argument("--openai-cached", type=float)
    a = ap.parse_args()
    openai = None
    if a.openai_input and a.openai_output and a.openai_cached:
        openai = {
            "input": a.openai_input,
            "output": a.openai_output,
            "cached": a.openai_cached,
            "cache_write_factor": 1.0,
        }
    total = 0.0
    print(
        f"{'cell':5s} {'prefix tok':>10s} {'$/trial':>9s} {'trials':>7s} {'$ total':>9s}"
    )
    for cid, cell in CELLS.items():
        price = ANTHROPIC if cell.family == "anthropic" else openai
        prefix = tool_tokens(cell)
        n = a.tasks * a.reps
        if price is None:
            print(f"{cid:5s} {prefix:10d} {'(pass --openai-* prices)':>27s}")
            continue
        per = trial_cost(prefix, a.turns, a.history, a.output, price)
        total += per * n
        print(f"{cid:5s} {prefix:10d} {per:9.3f} {n:7d} {per * n:9.2f}")
    print(f"ceiling (priced cells): ${total:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
