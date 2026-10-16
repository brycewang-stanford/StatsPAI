# CausalAgentBench v2: pre-registration protocol

**Status: draft for deposit.** This document, together with the SHA-256
fingerprints printed by `python runners/prereg.py fingerprint`, is the
pre-registration. Nothing below may change after the deposit; the runner
refuses to call any model API until `prompts/preregistration.lock.json`
records the registration DOI and the fingerprints still match.

## 1. Question

Do LLM agents produce correct applied-econometrics results more often, with
fewer fabricated numbers or invented functions, and at lower token cost,
when their toolset is StatsPAI (a Python library with an MCP server and
self-describing function schemas) than when it is the general Python
scientific stack or R?

## 2. Design

50 tasks x 6 cells x 3 repetitions = 900 trials.

| Cell | Toolset | Model family |
| --- | --- | --- |
| C1 | Python + StatsPAI, StatsPAI MCP tools | Anthropic |
| C2 | Python + StatsPAI, StatsPAI MCP tools | OpenAI |
| C3 | Python: pandas, numpy, scipy, statsmodels, linearmodels, scikit-learn, DoubleML (StatsPAI import blocked) | Anthropic |
| C4 | as C3 | OpenAI |
| C5 | R with the packages listed in `runners/cells.py` | Anthropic |
| C6 | as C5 | OpenAI |

The exact model releases and the Anthropic effort setting are recorded in
the lock at deposit time. No refusal fallback to another model is enabled.

### Tasks

`prompts/prompts.json` (version 2). Three levels:

* **L1 (20)**: the method is named ("Estimate ... by 2SLS ...").
* **L2 (20)**: the design is described, the method is not.
* **L3 (10)**: an open analysis with required diagnostics.

Every task lists the named values the agent must return (`report`) and the
one it is scored on (`headline`). Data files are in `data/`, fingerprinted
in `data/MANIFEST.json`. Several are simulated with known parameters
(`runners/build_data.py`); the others are the datasets distributed with
StatsPAI, some of which are simulated replicas. Results are scored against
the analysis of the file given, not against claims about original studies.

### Trial procedure

`runners/agent_loop.py`. The model receives a fixed system prompt
(`runners/cells.py`), the task text and its reporting instruction, and the
tools of its cell: one code-execution tool (`run_python` or `run_r`, a
persistent session in a fresh directory holding the data file, 300 s per
call, output truncated at 8,000 characters), `submit_answer` with the
task's report fields as typed arguments, and, in C1/C2, StatsPAI's MCP
tools. The trial ends at `submit_answer`, a turn without a tool call, a
refusal, or 20 turns. Sessions run as local processes on the experimenter's
machine; the task's data file is the only data supplied, and the tools
offer no web search.

### Gold answers

`golds/golds.json`, built by `runners/build_golds.py` from
`golds/reference_values.json`, which `golds/reference.R` computes on the
frozen files with canonical R implementations (package versions recorded in
the file). A task may have several acceptable variants (different valid
estimators, or implementation conventions such as matching tie rules), each
with its own value and tolerance:

* `EXACT`: relative 1% (absolute floor 1e-4) for deterministic estimators;
* `CONVENTION`: relative 5% where valid implementations differ by an
  optimiser, a tie rule or a cross-fitting split;
* wider task-specific bands for synthetic-control weights (15%).

StatsPAI's own solution (`runners/oracle_statspai.py`) was checked against
every gold before the deposit and is within tolerance on all 50 tasks.

## 3. Measures (per trial, `runners/grader.py`)

* **task_success** (primary): the headline value is within tolerance of some
  variant.
* **method_correct**: the reported estimator is that of some variant.
* **values_correct**: share of the closest variant's fields within tolerance.
* **required_complete** (L3): every required diagnostic field is non-null.
* **fabricated**: the headline was submitted but no tool output contains a
  number within relative 1e-3 of it.
* **invented_api**: a tool output shows a missing attribute or function, an
  unknown keyword argument, a missing module or package (other than the
  deliberately blocked StatsPAI import in C3/C4), or the agent called a tool
  that was not offered.
* **hallucination**: fabricated or invented_api.
* **tokens**: all tokens processed in the trial (input, cache reads, cache
  writes, output). **tokens_uncached** excludes cache reads (secondary).

## 4. Hypotheses and tests

Statistics are pooled over the two model families within a toolset. Tests
use a cluster bootstrap over tasks (B = 10,000, seed fixed in `runners/grader.py`) with
intervals at 1 - 0.05/5 (Bonferroni over H1-H5).

* **H1**: StatsPAI cells' task success is at least 90% on L1, 70% on L2 and
  50% on L3 (supported for a level when the interval's lower bound reaches
  the threshold).
* **H2**: on L2 and L3, StatsPAI cells' task success exceeds the Python-stack
  cells' by at least 15 points (lower bound >= 0.15).
* **H3**: StatsPAI cells' hallucination rate is below 5% (upper bound < 0.05)
  and the Python-stack cells' is at least 15% (lower bound >= 0.15).
* **H4**: StatsPAI cells use at most 60% of the tokens of the Python-stack
  cells (upper bound of the ratio of means <= 0.60).
* **H5**: R cells' task success is within 10 points of StatsPAI cells' (the
  interval of the difference inside (-0.10, 0.10)) and they use at least 1.5
  times the tokens (lower bound of the ratio >= 1.5).

All per-cell, per-level and per-family descriptive statistics are reported
regardless of the tests.

## 5. Exclusions and deviations

No trial is excluded. API errors are retried by the provider SDK; a trial
that still fails to complete is re-run once and the failure reported. Any
deviation from this protocol is reported with its reason.

## 6. Conflict of interest

The authors develop StatsPAI. The gold answers come from R reference
implementations, not from StatsPAI, and the scoring is fully automatic.
