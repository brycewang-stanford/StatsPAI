"""CausalAgentBench quick-start (no API key required).

Runs the StatsPAI reference oracle over a small task pack and prints the
summary table. To run the real agent conditions, install the LLM extras
and set the API keys, then pass conditions=["C1", "C3"] etc.

    pip install -e .            # core (oracle path)
    pip install -e .[llm]       # + anthropic/openai for C1..C6
    export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...

    python examples/quickstart.py
"""

from __future__ import annotations

import warnings

import causalagentbench as cab

warnings.filterwarnings("ignore")


def main() -> None:
    tasks = cab.load_tasks(n_l1=4, n_l2=4, n_l3=4)
    print(f"Loaded {len(tasks)} tasks across {len({t.design for t in tasks})} designs.\n")

    results = cab.run_suite(tasks, conditions=["oracle"], seeds=[0, 1, 2],
                            progress=False)
    print(cab.summarize(results))

    print("\nHypotheses (oracle-only run — LLM cells absent are flagged):")
    H = cab.test_hypotheses(results, B=999)
    for k in ("H1", "H2", "H3", "H4", "H5"):
        ev = H[k].get("evaluable")
        print(f"  {k}: evaluable={ev}"
              + ("" if ev else f"  ({H[k].get('reason')})"))

    # To run real agents:
    #   results = cab.run_suite(tasks, conditions=["C1", "C3"], seeds=[0, 1, 2],
    #                           out_path="runs/c1c3.jsonl")
    #   print(cab.test_hypotheses(results))


if __name__ == "__main__":
    main()
