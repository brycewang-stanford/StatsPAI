# CausalAgentBench OSF Pre-registration Protocol

**Status: ready for OSF deposit.** Freeze this document and upload
to `osf.io/<project>` *before* executing the first production trial.
The hash of this file + `prompts.json` + `golds.json` will be the
pre-registration fingerprint.

## Title

CausalAgentBench: behavioural evaluation of LLM agents on a
50-prompt applied-econometrics benchmark.

## Brief description

We evaluate whether LLM agents produce more accurate and more
efficient empirical analyses when given a unified, schema-first
toolset (StatsPAI) versus the prevailing fragmented Python and R
ecosystems. The benchmark consists of 50 causal-inference research
prompts spanning three difficulty levels and ten identification
strategies; each prompt is run in six experimental cells crossing
two language models (Anthropic Claude, OpenAI GPT) with three
toolsets (StatsPAI, Pythonic stack, R via MCP).

## Hypotheses (registered before any production trial)

- **H1 (within-cell threshold)**: StatsPAI cells (C1, C2) achieve
  task-success rates of at least 90% on L1 prompts, 70% on L2, and
  50% on L3.
- **H2 (between-toolset gap)**: StatsPAI cells exceed Pythonic-
  stack cells (C3, C4) on L2-L3 task-success by at least 15
  percentage points.
- **H3 (hallucination)**: StatsPAI cells have hallucination rate
  below 5%; Pythonic-stack cells exceed 15%.
- **H4 (token efficiency)**: StatsPAI cells use no more than 60%
  of the tokens used by Pythonic-stack cells at fixed prompt and
  language model.
- **H5 (R-via-MCP comparison)**: R-via-MCP cells (C5, C6) achieve
  task-success rates statistically indistinguishable from
  StatsPAI but use at least 1.5× as many tokens.

## Statistical test

Cluster bootstrap with prompt as the clustering unit, B = 10,000.
Two-sided p-values at α = 0.05 with Bonferroni correction across
H1-H5. The full pre-specified test code lives in
`tests/agent_bench/runners/grader.py::stat_tests` and will be
frozen at the time of OSF deposit.

## Experimental design

50 prompts × 6 cells × 3 reps = 900 trials.

Six cells:

| Cell | Toolset | Agent |
| --- | --- | --- |
| C1 | StatsPAI + MCP | Anthropic Claude (latest) |
| C2 | StatsPAI + MCP | OpenAI GPT (latest) |
| C3 | Pythonic stack (statsmodels, linearmodels, DoubleML, grf-python) | Claude |
| C4 | Pythonic stack | GPT |
| C5 | R via MCP (radian + Jupyter R) | Claude |
| C6 | R via MCP | GPT |

Three difficulty levels:

- **L1 (direct, n = 20)**: prompt names the method.
- **L2 (indirect, n = 20)**: prompt describes the structural
  problem, agent picks the method.
- **L3 (workflow, n = 10)**: prompt asks for a complete analysis
  with diagnostics and at least one robustness check.

## Sandboxing and reproducibility

Each trial runs in a per-trial sandbox with:

- the calibrated replica CSVs from `sp.datasets.*` (CSVs are
  shipped in `tests/agent_bench/sandbox/csvs/`);
- a fresh Python interpreter (or radian + Jupyter R) with the
  appropriate toolset's libraries pinned to the versions logged in
  `tests/agent_bench/sandbox/env.lock`;
- agent temperature = 0;
- seed = `hash((cell, prompt_id, rep))`;
- network egress disabled.

## Metrics

- M1 task_success — final estimate within ±5% of the gold answer
- M2 method_correct — agent picked the gold-rubric estimator
- M3 code_executes — generated code runs without unhandled
  exceptions
- M4 token_total — input + output tokens for the full trial
- M5 hallucination — agent attempted to call a non-existent function
- M6 diagnostic_complete — required diagnostics (parallel-trends
  test for DiD, density test for RD, first-stage F for IV,
  propensity-score overlap for PSM) appear in the agent's output
  (LLM-as-judge double-blind on a 20% sub-sample for inter-rater
  agreement)

## Deviations and amendments

Any deviation from this pre-registration during execution must be
recorded in `tests/agent_bench/prompts/_deviations.md` with a
timestamp and a justification. Deviations will be disclosed in the
final paper.

## Pre-registration checklist

- [ ] Freeze prompts.json hash
- [ ] Freeze golds.json hash
- [ ] Freeze grader.py at HEAD before first trial
- [ ] Deposit on OSF
- [ ] Record OSF DOI in NEXT-STEPS §M
- [ ] Verify API budget (target $300-600)
- [ ] Run 900 trials
- [ ] Lock results/trials.jsonl hash
- [ ] Generate headline.md on full data
- [ ] Update Paper-JSS/manuscript/sections/07-agent-eval.tex
