# `tests/agent_bench/` — CausalAgentBench harness

This directory holds the infrastructure for the JSS plan §5.5 /
manuscript §7 LLM-agent behavioural benchmark. The harness is
**fully built** but the production 900-trial run (6 cells × 50
prompts × 3 reps with Anthropic and OpenAI APIs) is gated on:

1. an OSF pre-registration deposit (the protocol is in
   `prompts/_protocol.md` and ready to upload),
2. an API budget commitment (~$300–600), and
3. a green light from the project lead.

Until those land, the harness runs end-to-end against a
deterministic mock LLM (`runners/mock_llm.py`) so the scoring,
result-aggregation, and statistical-test pipeline can be smoke-
tested. The mock-LLM run produces fake transcripts and a real
score table; flipping the harness to the real API is a one-line
change in `runners/runner.py`.

## Layout

```
tests/agent_bench/
├── README.md
├── prompts/
│   ├── _protocol.md        # OSF pre-registration text (frozen for upload)
│   └── prompts.json        # 50 prompts (L1 × 20, L2 × 20, L3 × 10)
├── golds/
│   └── golds.json          # gold answers + 5-dimensional rubrics
├── runners/
│   ├── mock_llm.py         # deterministic stub for harness smoke
│   ├── runner.py           # main loop: run all 900 trials
│   └── grader.py           # apply rubric + emit per-trial scores
├── sandbox/                # per-trial scratch directories
└── results/
    ├── trials.jsonl        # one line per trial: {prompt_id, cell, rep, output}
    ├── scores.csv          # per-trial 5-dimensional scores
    └── headline.md         # H1-H5 hypothesis test outcomes
```

## Pre-registered hypotheses

The OSF document freezes:
- **H1**: StatsPAI cells (C1, C2) achieve task success ≥ 90% on L1,
  ≥ 70% on L2, ≥ 50% on L3.
- **H2**: StatsPAI cells exceed Pythonic-stack cells (C3, C4) on
  L2–L3 task success by ≥ 15 percentage points.
- **H3**: StatsPAI hallucination rate < 5%; Pythonic-stack ≥ 15%.
- **H4**: StatsPAI uses ≤ 60% of the tokens of Pythonic-stack at
  fixed prompt+model.
- **H5**: R-via-MCP cells (C5, C6) match StatsPAI on task success
  but require ≥ 1.5× the tokens.

Statistical test: cluster bootstrap with prompt as the cluster,
α = 0.05, Bonferroni correction across H1–H5.

## Six experimental cells

| Cell | Toolset | Agent |
| --- | --- | --- |
| C1 | StatsPAI + MCP | Anthropic Claude (latest) |
| C2 | StatsPAI + MCP | OpenAI GPT (latest) |
| C3 | Pythonic stack (statsmodels, linearmodels, DoubleML, grf-python) | Claude |
| C4 | Pythonic stack | GPT |
| C5 | R via MCP (radian + Jupyter R) | Claude |
| C6 | R via MCP | GPT |

900 total trials = 50 prompts × 6 cells × 3 reps.

## How to run

Mock-LLM smoke (no API cost):

```bash
cd tests/agent_bench
python3 runners/runner.py --mock --cells C1,C3,C5 --prompts L1
python3 runners/grader.py
```

Full production run (after OSF + budget approval):

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
python3 runners/runner.py --cells C1,C2,C3,C4,C5,C6 --prompts all
python3 runners/grader.py
```
