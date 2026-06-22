# Agent Empirical Analysis Uplift Worklog

Start date: 2026-06-21

## Objective

Improve the root StatsPAI repo from the perspective of agent-driven
end-to-end empirical analysis automation, while staying out of the active JOSS
review lane and out of parallel-agent work already present in the tree.

## Scope

Included:

- Additive root-only tooling that helps an agent audit an empirical-analysis
  workflow before running estimators.
- A machine-readable example workflow spec.
- Focused tests for the new static audit.
- Local verification evidence.

Excluded:

- `Paper-JSS/`
- `CausalAgentBench/`
- `paper.md`
- `paper.bib`
- Release metadata, tags, GitHub releases, pushes, and package publishing.
- Any root path already dirty at baseline unless explicitly recorded here.

## Baseline

Root branch at start:

- `main...origin/main`

Pre-existing dirty root paths treated as not owned by this goal:

- `CHANGELOG.md`
- `MIGRATION.md`
- `README.md`
- `README_CN.md`
- `docs/index.md`
- `docs/reference/index.md`
- `docs/stats.md`
- `plans/2026-06-21-causal-inference-correctness-pass/WORKLOG.md`
- `plans/2026-06-21-causalpy-design-pass/`
- `plans/2026-06-21-causalpy-inspired-contracts/`
- `schemas/agent_cards.json`
- `schemas/functions.json`
- `schemas/index.json`
- `schemas/tools.json`
- `scripts/check_contract_inventory.py`
- `scripts/quality_gate.py`
- `src/statspai/__init__.py`
- `src/statspai/bayes/_base.py`
- `src/statspai/checks/`
- `src/statspai/core/effect_summary.py`
- `src/statspai/core/results.py`
- `src/statspai/did/did_imputation.py`
- `src/statspai/did/gardner_2s.py`
- `src/statspai/matching/match.py`
- `src/statspai/plots/__init__.py`
- `src/statspai/plots/counterfactual.py`
- `src/statspai/regression/iv.py`
- `src/statspai/schemas/agent_cards.json`
- `src/statspai/schemas/functions.json`
- `src/statspai/schemas/index.json`
- `src/statspai/schemas/tools.json`
- `src/statspai/smart/__init__.py`
- `src/statspai/smart/intake.py`
- `src/statspai/timeseries/its.py`
- `tests/test_causalpy_inspired_contracts.py`
- `tests/test_correctness_inference_fixes.py`
- `tests/test_counterfactual_plot.py`

Nested review-lane baseline:

- `Paper-JSS/` has pre-existing generated audit result diffs under
  `replication/results/`.
- `CausalAgentBench/` reports `main...origin/main` with no dirty paths.

Other active worktrees observed:

- `/Users/brycewang/Documents/GitHub/StatsPAI-improve-wt`
- `/Users/brycewang/Documents/GitHub/StatsPAI-wt-synth`
- `/Users/brycewang/Documents/GitHub/StatsPAI/.claude/worktrees/improve-correctness`

## Owned Files

- `scripts/agent_workflow_spec_audit.py`
- `tests/test_agent_workflow_spec_audit.py`
- `plans/2026-06-21-agent-empirical-analysis-uplift/WORKLOG.md`
- `plans/2026-06-21-agent-empirical-analysis-uplift/example_workflow_spec.json`

## Batch 1 - Empirical workflow spec audit

Status: implemented and focused verification passed.

Intent:

- Convert the existing narrative full-analysis skill into an executable
  preflight contract that can reject under-specified agent workflows before
  they produce results.
- Keep the audit static and dependency-free so it can run in a lean environment
  without optional estimator packages.

Implemented:

- Added `scripts/agent_workflow_spec_audit.py`.
- Added a passing DID example spec for data intake, identification, estimator
  selection, diagnostics, robustness, result export, reproducibility, and
  validation gates.
- Added focused tests covering text rendering, JSON shape, passing check mode,
  failing check mode, and DID-specific parallel-trends diagnostics.

Verification:

- `.venv/bin/python scripts/agent_workflow_spec_audit.py --check` passed with
  status `PASS`, score `100`, and zero issues for the bundled DID spec.
- `.venv/bin/python -m pytest -o addopts='' tests/test_agent_workflow_spec_audit.py`
  passed with 5 tests.
- `.venv/bin/python -m py_compile scripts/agent_workflow_spec_audit.py tests/test_agent_workflow_spec_audit.py`
  passed.
- `.venv/bin/python -m flake8 scripts/agent_workflow_spec_audit.py tests/test_agent_workflow_spec_audit.py --max-line-length=88 --ignore=E203,W503`
  passed.
- `git diff --check -- scripts/agent_workflow_spec_audit.py tests/test_agent_workflow_spec_audit.py plans/2026-06-21-agent-empirical-analysis-uplift/WORKLOG.md plans/2026-06-21-agent-empirical-analysis-uplift/example_workflow_spec.json`
  passed.
- Full `git diff --check` passed.

Python 3.9 note:

- `python3.9` was not available on this machine during the run. The new script
  avoids Python 3.10-only union syntax and optional dependencies; syntax and
  tests were verified with the repo `.venv` Python 3.10.20.

## Batch 2 - Agent execution loop and function-name guard

Status: implemented; verification pending.

Intent:

- Make the workflow spec audit check the *full agent loop*, not just the
  estimator list. A full empirical agent should declare preflight, fitting,
  result-handle reuse, result audit, robustness/sensitivity, export, and
  validation before producing a polished answer.
- Close a function-hallucination gap by checking declared `sp.*` estimator
  names against the offline `schemas/functions.json` bundle without importing
  StatsPAI or optional estimator dependencies.

Implemented:

- Added an `agent_execution` section to the bundled DID workflow spec with:
  `workflow_steps`, `result_handle_policy`, `handoff_artifacts`, and
  `stop_conditions`.
- Extended `scripts/agent_workflow_spec_audit.py` with:
  - required agent-execution fields,
  - full-loop step-group checks,
  - result-handle policy checks,
  - stop-condition checks,
  - optional offline public-function validation from `schemas/functions.json`.
- Added `scripts/quality_gate.py agent-workflow` and included it in
  `scripts/quality_gate.py all` so the empirical workflow spec gate can run
  with the rest of the repo's gradual quality gates.
- Added focused tests for:
  - missing/weak agent execution loops,
  - missing result-handle policy,
  - declared `sp.*` functions that do not exist in the offline schema bundle.
  - the new `quality_gate.py agent-workflow` subcommand.

Boundary notes:

- Did not touch `Paper-JSS/`, `CausalAgentBench/`, release/tag/PyPI state, or
  active `quasi/` files that appeared in the tree while this batch was running.

## Final Verification Snapshot

Status: passed for this root-only additive lane.

Commands:

- `.venv/bin/python scripts/quality_gate.py agent-workflow`
- `.venv/bin/python scripts/agent_workflow_spec_audit.py --check`
- `.venv/bin/python -m pytest -o addopts='' tests/test_agent_workflow_spec_audit.py -q`
- `.venv/bin/python -m py_compile scripts/agent_workflow_spec_audit.py scripts/quality_gate.py tests/test_agent_workflow_spec_audit.py`
- `.venv/bin/python -m flake8 scripts/agent_workflow_spec_audit.py scripts/quality_gate.py tests/test_agent_workflow_spec_audit.py --max-line-length=88 --ignore=E203,W503`
- `git diff --check`
- `git -C Paper-JSS status --short --branch`
- `git -C CausalAgentBench status --short --branch`

Results:

- The bundled example workflow spec passes with score `100`.
- `quality_gate.py agent-workflow` passes with observed `0` / baseline `0`.
- Focused pytest passed with 8 tests.
- Full whitespace check passed.
- `Paper-JSS/` still has the same style of generated audit result diffs under
  `replication/results/`; this goal did not modify that review lane.
- `CausalAgentBench/` still reports clean `main...origin/main`.
- Root remains dirty by design due to pre-existing and parallel-agent work.

Additional current dirty paths observed during verification and not owned by
this goal:

- `src/statspai/did/callaway_santanna.py`
- `tests/test_effect_summary.py`
- `src/statspai/quasi/__init__.py`
- `src/statspai/quasi/ancova.py`
- `tests/test_quasi_ancova_negd.py`
