"""Estimate the API budget for a CausalAgentBench production run.

Computes a token / cost ceiling from the prompt set and a fixed
per-trial allowance, *without* making any real API calls.  This is
the first script the user runs before flipping `runner.py` from
``--mock`` to ``--api``.

Usage::

    python tests/agent_bench/runners/estimate_budget.py
    python tests/agent_bench/runners/estimate_budget.py --reps 3 --cells C1,C2

The bundled price table is an illustrative planning snapshot for the
four LLM options the protocol pre-registers; it intentionally does
*not* authorise spending. Re-check the provider pricing pages and
update PRICE_TABLE immediately before any real API run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE.parent / "prompts" / "prompts.json"


# Fixed per-trial allowances (input + output tokens) chosen to be
# roomy enough to avoid truncation on the L3 "complete workflow"
# prompts that drive the largest tool-use traces.  These are the same
# allowances the protocol pre-registers.
BUDGET_INPUT_TOKENS_PER_TRIAL = 8_000
BUDGET_OUTPUT_TOKENS_PER_TRIAL = 4_000


# Illustrative list-price snapshot entered 2026-04-30 (USD per 1M
# tokens). Update these only by re-checking provider pricing pages.
@dataclass(frozen=True)
class ModelPrice:
    name: str
    input_per_mtok: float
    output_per_mtok: float


PRICE_TABLE = (
    # Anthropic Claude — cell C1 / C5 of the protocol
    ModelPrice("claude-sonnet-4", input_per_mtok=3.00, output_per_mtok=15.00),
    # Anthropic Claude Opus — premium cell variant
    ModelPrice("claude-opus-4", input_per_mtok=15.00, output_per_mtok=75.00),
    # OpenAI GPT-4 family — cell C2 / C4 / C6 of the protocol
    ModelPrice("gpt-4-turbo", input_per_mtok=10.00, output_per_mtok=30.00),
    ModelPrice("gpt-4o", input_per_mtok=5.00, output_per_mtok=15.00),
)


def estimate_cells(n_cells: int, n_prompts: int, reps: int) -> dict:
    n_trials = n_cells * n_prompts * reps
    in_tok = n_trials * BUDGET_INPUT_TOKENS_PER_TRIAL
    out_tok = n_trials * BUDGET_OUTPUT_TOKENS_PER_TRIAL
    return {
        "n_cells": n_cells,
        "n_prompts": n_prompts,
        "reps": reps,
        "n_trials": n_trials,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


def cost_for_model(stats: dict, model: ModelPrice) -> dict:
    in_cost = stats["input_tokens"] / 1_000_000 * model.input_per_mtok
    out_cost = stats["output_tokens"] / 1_000_000 * model.output_per_mtok
    return {
        "model": model.name,
        "input_usd": round(in_cost, 2),
        "output_usd": round(out_cost, 2),
        "total_usd": round(in_cost + out_cost, 2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cells", default="C1,C2,C3,C4,C5,C6", help="Comma-separated cell IDs."
    )
    p.add_argument(
        "--reps", type=int, default=3, help="Reps per (cell, prompt). Default 3."
    )
    p.add_argument("--prompts-path", type=Path, default=PROMPTS_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts_path.read_text(encoding="utf-8"))
    n_cells = len([c for c in args.cells.split(",") if c.strip()])
    stats = estimate_cells(n_cells, len(prompts), args.reps)

    print("=== CausalAgentBench budget estimate ===")
    print(f"  Cells:    {n_cells} ({args.cells})")
    print(f"  Prompts:  {stats['n_prompts']}")
    print(f"  Reps:     {stats['reps']}")
    print(f"  Trials:   {stats['n_trials']}")
    print(
        f"  Per-trial allowance: {BUDGET_INPUT_TOKENS_PER_TRIAL} in "
        f"+ {BUDGET_OUTPUT_TOKENS_PER_TRIAL} out tokens"
    )
    print(
        f"  Total tokens (worst case): "
        f"{stats['input_tokens']:,} in + {stats['output_tokens']:,} out "
        f"= {stats['total_tokens']:,}"
    )
    print()
    print("=== Cost ceiling per model (bundled price snapshot, USD) ===")
    print(f"  {'Model':<18}  {'Input':>9}  {'Output':>9}  {'Total':>9}")
    print(f"  {'-' * 18}  {'-' * 9}  {'-' * 9}  {'-' * 9}")
    for m in PRICE_TABLE:
        c = cost_for_model(stats, m)
        print(
            f"  {c['model']:<18}  ${c['input_usd']:>7.2f}  "
            f"${c['output_usd']:>7.2f}  ${c['total_usd']:>7.2f}"
        )
    print()
    print("Notes:")
    print("  - These are upper-bound ceilings from the bundled token")
    print("    allowance and price snapshot; actual usage depends on")
    print("    provider metering and observed smoke-test traces.")
    print("  - The protocol pre-registers two cells per language-model")
    print("    family (Anthropic + OpenAI); compute the final cap after")
    print("    selecting exact OSF-pinned model IDs.")
    print("  - Multiply by 2 if running both prompt-set seeds.")
    print("  - Re-check provider pricing pages before any real API run.")
    print()
    print("Before flipping `runner.py` to --api:")
    print("  1. Deposit prompts/_protocol.md on OSF, freeze the SHA-256")
    print("     hash of prompts.json in the deposit.")
    print("  2. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in the")
    print("     shell, *not* in any committed file.")
    print("  3. Run a 5-trial smoke test (--reps 1 --prompts L1)")
    print("     against the real API and confirm the trace looks sane.")
    print("  4. Then run the full 900-trial production sweep.")


if __name__ == "__main__":
    main()
