# R/Stata Parity Roadmap - 2026-06-08

Constraint: the active JOSS review path stays first. This roadmap keeps parity
work in root validation assets (`tests/`, `scripts/`, `plans/`, reference
fixtures) and does not edit `paper.md`, `paper.bib`, `docs/joss_*`, or the
separate `Paper-JSS/` repository unless a reviewer response explicitly requires
it.

## Guardrails

- Treat `Paper-JSS/` as a separate manuscript repository. Root parity work can
  inform that paper later, but it should not mutate JSS generated artifacts
  during the JOSS review window.
- Separate fast, license-free gates from optional external runtime gates:
  committed JSON/fixture checks must run without R or Stata; live R/Stata runs
  are opt-in evidence refreshes.
- Never promote a method from `api_stable` to `validated` or `certified` unless
  the exact parity artifact and tolerance are committed.
- Convention gaps are acceptable when explicit: record the package command,
  estimator target, tolerance, and why the gap is a convention rather than a
  numerical bug.

## Month 1 - Freeze the Existing Evidence Surface

1. Lock Tier A fixtures.
   - Keep `scripts/tier_a_fixture_lock.py` in the fast pytest contract suite.
   - Refresh `tests/r_parity/TIER_A_FIXTURE_LOCK.json` only after reviewing the
     R/Stata script, input CSV, golden JSON, and rendered table diffs.

2. Add published numerical benchmarks that do not need external software.
   - Start with NIST StRD OLS fixtures for coefficient and standard-error
     accuracy.
   - Add similar static fixtures only when the source is public, stable, and
     has certified values.

3. Make the parity gap inventory actionable.
   - Use `statspai.validation.parity_gap_report(...)` and the rendered
     `parity_table_3way.md` as the source of truth.
   - Split gaps into: missing reference, documented convention gap, stale
     fixture, and true implementation defect.

4. Keep quick gates cheap.
   - Primary local gate:
     `.venv/bin/python -m pytest -o addopts='' tests/test_parity_harness_contract.py tests/reference_parity/test_nist_strd_ols.py`
   - Optional external gate when R/Stata are available:
     `.venv/bin/python -m pytest -o addopts='' tests/test_parity_runtime.py -m external_parity_runtime`

## Month 2 - Close High-Value R/Stata Gaps

1. Finish partially materialized modules.
   - Generate and review `50_xtabond_R.json` in an R-equipped environment.
   - Re-run the joined table generator after any new R/Stata artifact lands.

2. Prioritize high-review-salience estimators.
   - OLS/cluster/HDFE/PPMLHDFE: small-sample corrections, FE absorption, and
     multiway clustering should have explicit parity rows.
   - IV/weak-IV: first-stage diagnostics, LIML/AR/CLR conventions, and robust
     covariance choices should be pinned separately from point estimates.
   - RD/rddensity: bandwidth-selection differences should stay documented and
     same-bandwidth numerical parity should remain strict.
   - DID/synth: aggregation weights, control-group definitions, and random
     placebo conventions should be recorded in artifact metadata.

3. Add schema-level checks for new parity rows.
   - Every JSON row should include module id, statistic name, estimate, optional
     SE, reference backend, tolerance class, and convention note when relevant.
   - The comparator should fail on unclassified new gaps.

## Month 3 - Turn Parity Into Release Discipline

1. Wire evidence to API validation status.
   - `certified`: cross-language or published certified-value parity.
   - `validated`: known-DGP, reference-parity, or external package parity.
   - `api_stable`: usable API without enough numerical evidence to claim more.

2. Use the Tier D classifier as the backlog generator.
   - Run `scripts/tierd_classify.py report` before each parity sprint.
   - Convert smoke-only estimator tests into analytic known-DGP tests when no
     R/Stata reference exists.

3. Add a pre-release parity report.
   - Summarize locked Tier A files, live runtime status, open gaps, and
     validation-status changes.
   - Keep it separate from JOSS reviewer-facing prose until the facts are
     stable and relevant to a reviewer question.

4. Refresh external environments on a fixed cadence.
   - R: update `R_ENVIRONMENT.md` and `renv.lock` only after a full artifact
     comparison.
   - Stata: update `STATA_ENVIRONMENT.md` after running the full `.do` harness
     and reviewing changed `_Stata.json` files.

## First Three Implementation Targets

1. Land the Tier A fixture lock and contract test.
2. Land NIST StRD OLS coefficient/SE tests for the QR solver path.
3. Produce a parity-gap triage table from current artifacts and pick the next
   five modules by review value, not by ease.
