# Status

## Batch 1 - Network anchors and quality-gate ratchets

Status: implemented and focused verification passed.

Intent:

- Add direct deterministic numeric anchors for the six network Tier-D worklist
  functions.
- Stop mypy configuration warnings from being counted as a successful 0-error
  run.
- Pin the dev mypy dependency below 2.0 while the package still declares
  Python 3.9 support.
- Ratchet the flake8 baseline to the current measured package count.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_network.py`
- `.venv/bin/python scripts/tierd_classify.py report`
- `.venv/bin/python scripts/quality_gate.py all`
- `git diff --check`

Results:

- Added closed-form network anchors for direct `sp.degree_centrality`,
  `sp.assortativity`, `sp.network_components`, `sp.katz_centrality`,
  `sp.bonacich_power`, and the `sp.centrality` dispatcher.
- `.venv/bin/python -m pytest -o addopts='' tests/test_network.py` passed
  with 48 tests.
- `.venv/bin/python scripts/tierd_classify.py report` now reports 0
  estimator-like Tier-D functions.
- Pinned dev `mypy` to `<2.0` while StatsPAI still declares Python 3.9
  support, and made `scripts/quality_gate.py` fail on mypy config warnings
  rather than counting them as a 0-error run.
- Added a regression test in `tests/test_import_budget.py` proving a mypy
  config warning with return code 0 still fails the quality gate.
- Installed `mypy 1.20.2` in the local venv to match the new dev constraint.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8
  `observed=1010 baseline=1010` and mypy `observed=1058 baseline=1058`.

## Batch 2 - Narrow silent-degradation ratchet

Status: implemented and focused verification passed.

Results:

- Replaced three `except Exception: pass` fallbacks with specific exception
  tuples in `src/statspai/smart/verify.py` and `src/statspai/smart/citations.py`.
- Lowered `tests/test_no_silent_degradation.py` `BARE_SWALLOW_MAX` from 7 to 4.
- Lowered `scripts/error_taxonomy_audit.py` `BROAD_EXCEPT_MAX` from 589 to 586.
- `.venv/bin/python -m pytest -o addopts='' tests/test_no_silent_degradation.py tests/test_workflow_degradations.py tests/test_stability.py::TestStabilityInHelpLayer`
  passed with 21 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed.
- `.venv/bin/python -m py_compile src/statspai/smart/verify.py src/statspai/smart/citations.py scripts/error_taxonomy_audit.py`
  passed.

## Parallel Dirt Not Owned Here

After Batch 1, these root files became dirty outside this lane and are not
edited or owned by this run:

- `CITATION.cff`
- `.github/workflows/parity-guards.yml`
- `.flake8`
- `README_CN.md`
- `docs/index.md`

They appear to be release-doc refreshes for v1.19.0 from the parallel agent.
The `.flake8` file was inspected because it can affect local lint behavior; it
mirrors the quality-gate 88-column settings and does not exclude `src/statspai`,
but it remains outside this run's owned-file set.
`pyproject.toml` is mixed ownership: this lane owns only the dev mypy
constraint change; the pytest default `addopts` coverage change was already in
the tree outside this lane. All pytest verification here passed
`-o addopts=''`, so the results do not rely on that unrelated hunk.

## Batch 3 - Low-risk lint cleanup

Status: implemented and focused verification passed.

Results:

- Removed two unused local variables in `src/statspai/censoring/ipcw.py` and
  `src/statspai/synth/mc.py`.
- Rewrapped the abstract `crossval` engine hook to avoid a one-line function
  body lint violation.
- Added a targeted `noqa` note on the deliberate article-alias late bind in
  `src/statspai/__init__.py`; the remaining `F811` entries are intentional
  shadowing for `sp.policy_tree` and `sp.dml` and still counted in the ratchet.
- Lowered `scripts/quality_gate.py` flake8 baseline from 1010 to 1006.
- `.venv/bin/python -m pytest -o addopts='' tests/test_target_trial.py tests/test_cross_validate.py::TestDegradation tests/test_article_aliases_round2.py tests/test_article_aliases.py`
  passed with 76 tests and 3 expected warnings.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8
  `observed=1006 baseline=1006` and mypy `observed=1058 baseline=1058`.
- `.venv/bin/python -m py_compile src/statspai/__init__.py src/statspai/censoring/ipcw.py src/statspai/crossval/_engines.py src/statspai/synth/mc.py`
  passed.

## Final Verification Snapshot

Status: passed on 2026-06-21.

Results:

- `.venv/bin/python -m pytest -o addopts='' tests/test_import_budget.py tests/test_network.py tests/test_no_silent_degradation.py tests/test_workflow_degradations.py tests/test_stability.py::TestStabilityInHelpLayer tests/test_target_trial.py tests/test_cross_validate.py::TestDegradation tests/test_article_aliases_round2.py tests/test_article_aliases.py`
  passed with 149 tests and 3 expected PSM warnings.
- `.venv/bin/python -m flake8 tests/test_import_budget.py --max-line-length=88 --ignore=E203,W503`
  passed.
- `.venv/bin/python scripts/tierd_classify.py report` passed and reports
  `1108` registered functions, evidence distribution `reference=128`,
  `anchored=608`, `weak=149`, `smoke=11`, `untested=212`, and `0`
  estimator-like Tier-D functions.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8
  `observed=1006 baseline=1006`, mypy `observed=1058 baseline=1058`,
  import-budget `observed=0 baseline=0`, and result/error protocol checks OK.
- `.venv/bin/python scripts/registry_stats.py --check` passed with `1108`
  functions across `83` submodules.
- `.venv/bin/python scripts/dump_schemas.py --check` passed.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `git diff --check` passed.
- `git -C Paper-JSS status --short --branch` and
  `git -C CausalAgentBench status --short --branch` both reported clean
  `main...origin/main`.
- Root `git status --short --branch` remains dirty by design: this lane owns
  the quality/test/package changes and the `plans/` directory; `.flake8`,
  `.github/workflows/parity-guards.yml`, `CITATION.cff`, `README_CN.md`, and
  `docs/index.md` are explicitly not owned here.

## Batch 4 - Stata migration contract hardening

Status: implemented; focused verification passed.

Intent:

- Improve the Stata-to-StatsPAI migration surface without touching estimator
  internals, parity artifacts, or JOSS/JSS review files.
- Cover official Stata command shapes that users paste during migration:
  `ivregress 2sls`, `ivregress liml`, `didregress`, `xtdidregress`, and
  `teffects ..., atet`.

Results:

- `ivregress 2sls ...` no longer treats `2sls` as the outcome variable; the
  translator now emits `method='2sls'`, preserves `robust` as `hc1`, and keeps
  `small` as an explicit finite-sample convention note.
- `ivregress liml ... vce(cluster firm)` now emits
  `sp.ivreg(..., method='liml', cluster='firm')` with a migration note.
- `didregress` and `xtdidregress` now translate to
  `sp.did(..., method='twfe')` and carry a note that Stata's official command
  uses a treatment-status indicator, while StatsPAI's staggered DID APIs need a
  first-treatment cohort column.
- `teffects ipw/aipw/nnmatch` translations now preserve `ATE` versus `ATET`
  in the emitted `estimand` argument.

Focused verification:

- `.venv/bin/python -m pytest -o addopts='' tests/test_translation.py` passed
  with 100 tests.
- `.venv/bin/python -m py_compile src/statspai/agent/_translation/_stata.py tests/test_translation.py`
  passed.
- `git diff --check` passed.

Final verification for this batch:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` passed with 1108
  registered functions and 0 estimator-like Tier-D functions.
- `.venv/bin/python scripts/registry_stats.py --check` passed with 1108
  functions across 83 submodules.
- `.venv/bin/python scripts/dump_schemas.py --check` passed; schemas remain in
  sync.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8
  `observed=999 baseline=999`, mypy `observed=1058 baseline=1058`, and the
  import-budget, agent-card, result-protocol, and error-taxonomy checks OK.
- `git -C Paper-JSS status --short --branch` and
  `git -C CausalAgentBench status --short --branch` both reported clean
  `main...origin/main`.
