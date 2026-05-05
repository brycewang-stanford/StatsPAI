# CausalAgentBench production-run runbook

> Concise step-by-step for flipping `runner.py` from the deterministic
> mock LLM to the real Anthropic + OpenAI APIs.  This runbook does
> **not** authorise spending; it documents the guardrails the project
> commits to before the first production trial.

---

## Status as of 2026-05-05

The harness is feature-complete:

- 50 prompts (`prompts/prompts.json`, L1×20 + L2×20 + L3×10) covering
  the seven major identification strategies of applied econometrics.
- Gold answers, expected estimator, and grading rubric in
  `golds/golds.json`.
- 6 experimental cells (C1–C6 in a 3 toolset × 2 LLM-family factorial)
  fixed in the protocol.
- Deterministic mock LLM stub at `runners/mock_llm.py` that completes
  a 900-trial dry-run end-to-end in under one second.
- Grader at `runners/grader.py` emits the H1–H5 directional headline
  table on `results/trials.jsonl`.

What is **explicitly off** until the user authorises spending:

- The `--api {anthropic,openai,both}` path in `runners/runner.py`
  raises `NotImplementedError`.
- No API keys are read by the harness itself; `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` are only consulted on the production path.
- No model name is hard-coded.  The OSF deposit (see below) is the
  contract that pins a fixed model release per cell *before* trial 1.

---

## Step 1 — Estimate the bill

```bash
python tests/agent_bench/runners/estimate_budget.py
```

Worst-case ceilings at 900 trials (8 K input + 4 K output token
budget per trial; bundled planning price snapshot):

| Model            | Worst-case 900-trial cost |
| ---              | ---:                      |
| claude-sonnet-4  | $76                        |
| claude-opus-4    | $378                       |
| gpt-4o           | $90                        |
| gpt-4-turbo      | $180                       |

Before authorising any real API calls, update `PRICE_TABLE` from the
provider pricing pages and set a project-level hard cap in the
provider billing dashboards. The table above is a ceiling calculator,
not a spending authorisation.

---

## Step 2 — Deposit the OSF pre-registration

The protocol document is at `prompts/_protocol.md`.  Required steps
before trial 1:

1. Compute the SHA-256 hash of the frozen prompt set:
   ```bash
   shasum -a 256 tests/agent_bench/prompts/prompts.json
   ```
2. Open an OSF project, paste `_protocol.md` into the registration
   form, and pin the prompt-set hash + the prompts JSON file as
   attachments.
3. Choose the four hypotheses to pre-register (H1 task success,
   H2 PySpAI vs Pythonic stack delta, H3 hallucination rate, H4
   token efficiency, H5 R-via-MCP equivalence) — copy the language
   verbatim from §7.2 of the JSS draft.
4. Freeze the model releases:
   - Anthropic cell: copy the exact current model ID from the
     provider dashboard or model documentation at deposit time and
     write that string into the OSF deposit.
   - OpenAI cell: use the same convention; pin the exact documented
     model ID before trial 1 rather than relying on a moving alias.
5. Cite the OSF DOI in the JSS revision; do not change prompts,
   golds, or rubric after the deposit.

---

## Step 3 — Wire the API path

Open `runners/runner.py` and remove the
`raise NotImplementedError(...)` block at line 61–66.  Implement two
simple wrappers in `runners/api_llm.py`:

```python
# api_llm.py (new file the user authors after authorisation)
import os, anthropic, openai

ANTHROPIC = anthropic.Anthropic()
OPENAI = openai.OpenAI()

ANTHROPIC_MODEL = os.environ["STATSPAI_ANTHROPIC_MODEL"]
OPENAI_MODEL = os.environ["STATSPAI_OPENAI_MODEL"]

def run_anthropic_trial(cell, prompt, gold, rep):
    """Call ANTHROPIC.messages.create(...) with the StatsPAI MCP
    server attached as a tool; serialise tool-call trace; return
    a Trial dataclass shaped exactly like mock_llm.run_trial()."""
    ...

def run_openai_trial(cell, prompt, gold, rep):
    """Call OPENAI.chat.completions.create(...) with the StatsPAI
    MCP tools serialised via ``sp.function_schema(format='openai')``;
    return a Trial dataclass shaped exactly like mock_llm.run_trial()."""
    ...
```

The function-schema is already on `sp.function_schema(name, format=...)`
with the right OpenAI / Anthropic flavours, so the schema serialisation
is one call per estimator.  Rate-limiting and retries should use the
provider SDK's built-in exponential-back-off rather than rolling our
own.

---

## Step 4 — Smoke test before the full sweep

```bash
export ANTHROPIC_API_KEY=...   # do not commit
export OPENAI_API_KEY=...
export STATSPAI_ANTHROPIC_MODEL=...  # exact OSF-pinned model ID
export STATSPAI_OPENAI_MODEL=...     # exact OSF-pinned model ID

# 5 L1 trials, 1 rep, both APIs:
python tests/agent_bench/runners/runner.py \
    --api both --reps 1 --prompts L1 --cells C1,C2 \
    --out smoke.jsonl

# Inspect by eye — expect ~10 trials, all completing without error,
# tool calls visible in the trace.
python tests/agent_bench/runners/grader.py \
    --in smoke.jsonl \
    --scores-out smoke_scores.csv \
    --headline-out smoke_headline.md
```

If the smoke test produces sane traces and no API errors, proceed.

---

## Step 5 — Production run

```bash
# 900 trials, both APIs, full 6-cell × 50-prompt × 3-rep sweep:
python tests/agent_bench/runners/runner.py \
    --api both --cells C1,C2,C3,C4,C5,C6 --prompts all --reps 3 \
    --out trials.jsonl

# Score and emit the H1-H5 headline table:
python tests/agent_bench/runners/grader.py \
    --in trials.jsonl \
    --scores-out scores.csv \
    --headline-out headline.md
```

Expected wall-clock with normal API rate limits: **2–4 hours**.

---

## Step 6 — Update the JSS draft

After grading completes:

1. Replace the §7.4 *what we report in this draft* paragraph with
   the measured H1–H5 results.
2. Move the deterministic mock-LLM dry-run paragraph to a footnote.
3. Cite the OSF DOI in §7.2.
4. Append the post-hoc deviation log (any prompt that hit a rate
   limit, any cell that needed a re-run) verbatim from the OSF
   protocol's revision history.

---

## Hard guardrails

- **Never commit API keys.**  Use `.envrc` (direnv) or shell exports;
  the harness ignores any committed dotfiles.
- **Never run `--api` without first running `estimate_budget.py`** —
  the worst-case ceiling is the single most useful sanity check
  against a runaway loop.
- **Never edit prompts.json or golds.json after the OSF deposit.**
  If a prompt is genuinely broken, append a deviation note to the
  OSF revision and re-deposit; do not silently rewrite history.
- **Never run on a personal credit card.**  Request project-level
  Anthropic / OpenAI billing keys with explicit hard monthly caps.
