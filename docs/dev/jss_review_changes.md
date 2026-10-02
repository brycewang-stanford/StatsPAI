# JSS review period: changes to frozen manuscript artifacts

The JSS manuscript describes one tagged release (the `tag` in
`tests/jss_review_freeze.json`). Its parity, coverage, seed-study and timing
tables are re-tabulated from frozen artifacts, which that manifest hashes.
While the paper is under review, any change to one of those artifacts must be
recorded here; `tests/test_jss_review_freeze.py` fails until it is.

Rules:

1. **Do not regenerate a frozen artifact to make a table look better.** A
   change here is a correctness fix, a reference-environment change, or a
   new module, and it goes through CHANGELOG / MIGRATION as usual
   (⚠️ correctness fix if an output moves).
2. **One entry per change**, newest first, giving the date, the commit, the
   reason, and the effect on the paper: which table, listing or quoted
   number moves, and from what to what. "None" is a valid effect only if it
   was checked (for example, a module the paper does not print).
3. **List every changed path in backticks**, exactly as the test reports it,
   and **the SHA of every commit that changed it** (7+ hex characters, in
   backticks) in the same entry. Since 2026-09-28 the check is per commit:
   naming a path once no longer covers later commits to the same file.
   Merge commits that bring a frozen file in count as commits.
   A commit cannot name its own SHA, so the entry goes in a follow-up
   commit pushed together with it; the check runs at pre-push and in CI,
   never at pre-commit.
4. The manuscript itself is not edited during review. Recorded changes are
   folded into the next revision, which is re-anchored to a new release and
   re-frozen with `python scripts/jss_review_freeze.py --write --release X.Y.Z`.
5. When the paper is decided, set `"active": false` in the manifest.

## Entries

### 2026-10-03 — call traces re-recorded after `sp.ardl` was exported; two functions gained external-replication evidence

- **Commit.** `d6d3eb0b` added `sp.ardl` and `ARDLResult` (new exports in
  `src/statspai/__init__.py`), the `break_vars=` / `vce=` arguments of
  `sp.structural_break`, regenerated `docs/parity.md` and re-recorded
  `tests/r_parity/results/_implementation_trace.json` for modules 03 13 15
  24 25 26 27 53 65 66.
- **Reason.** Those ten modules import the package root, whose digest the
  trace binds. `src/statspai/timeseries/structural_break.py` is not on any
  Track A estimation path, and its new arguments default to the previous
  behaviour.
- **Effect on the paper.** None on any table. Only `exercised_sources`
  digests and `seconds` changed in the trace. In `docs/parity.md` the
  registered total is 1,280 (two new names) and the
  "external-replication (published numbers)" count goes from 3 to 5:
  `ardl` and `unitroot` are credited to
  `tests/external_parity/test_stock_watson_4e_ch15.py`, which checks them
  against the RATS output shipped with a textbook's replication files.
  That test needs the files on disk and is skipped otherwise, so it is
  evidence a reader can rerun only after downloading them. If the paper
  quotes the external-replication count, say what kind of evidence these
  two rows are.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `docs/parity.md`

### 2026-10-02 — call traces re-recorded after `sp.regress` gained HAC options; parity-index denominators moved

- **Commits.** `aad512aa` added `sp.ttest` (new exports in
  `src/statspai/__init__.py`) and regenerated `docs/parity.md`. `dd753900`
  added `sp.unitroot`, the `hac_lags=` / `hac_small=` arguments of
  `sp.regress` (`src/statspai/regression/ols.py`), regenerated
  `docs/parity.md` again and re-recorded
  `tests/r_parity/results/_implementation_trace.json` for modules 01 03 13
  14 15 24 25 26 27 51 53 54 55 56 65 66.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path; `regression/ols.py` and the package root are on
  the path of those sixteen. `aad512aa` was pushed without re-recording, so
  the trace check was stale on `main` between the two commits. The new
  `sp.regress` arguments default to the previous behaviour:
  `verify_reproduce_py.py --no-report 01_ols 14_ols_cluster 51_newey 53_cr2
  54_twoway_cluster 55_hc2_hc3 56_multiway_cluster` reports 7 reproduce,
  0 drift.
- **Effect on the paper.** None on any table. Only `exercised_sources`
  digests and `seconds` changed in the trace; no module's implementation
  classification moved. In `docs/parity.md` the registered-function
  denominators grew by the four new names (`ttest`, `TTestResult`,
  `unitroot`, `UnitRootResult`), which carry no cross-language evidence
  row: "unverified" 529 to 533 and estimator callables 736 to 738. One
  thing worth knowing at the next re-anchor: module `51_newey` passes
  against Stata with `rel_se` 1e-2 because Stata's `newey` scales the
  covariance by `N/(N-K)`. `sp.regress(..., robust="hac", hac_lags=4,
  hac_small=True)` now reproduces that Stata golden to 3e-16
  (`tests/test_regress_hac_options.py`), so the row could be moved to the
  strict budget by adding a Stata-convention row to the module. That is a
  change to a frozen artifact and was not made here.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `docs/parity.md`
### 2026-10-02 — call traces re-recorded for 16, 73 and 84 after the scope-map fields were added

- **Commit.** `f88b6bee`.
- **Reason.** `sp.did_imputation` and `sp.gardner_did` now record the
  configuration that `sp.validation_scope` reads (`y0_fe`, `y0_covariates`,
  `horizon_requested`; `covariates`) in `model_info`. The traces bind the
  SHA-256 of every source file on each module's estimation path, and
  `did/did_imputation.py` and `did/gardner_2s.py` are on the paths of
  Track A modules 16, 73 and 84.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; their committed `_py.json` results are
  byte-identical, so no estimate or standard error moved, and no module's
  implementation classification changed. The same commit regenerates the
  six option-level Stata fixtures under `tests/stata_parity/option_parity/`
  in double precision. Those are outside the Track A enumeration and are
  not read by any manuscript table.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-02 — release 1.35.0: parity tables carry the new version string

- **Commits.** `173f2bf8` (release 1.35.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** The appendix parity tables read "StatsPAI
  1.35.0" once the manuscript is re-anchored to this release. Everything
  the paper quotes from the registry and the parity index should be
  regenerated against the `v1.35.0` tag at the same time (see the entries
  below for what moved since 1.34.2).

### 2026-10-02 — call traces re-recorded for 12 and 76 after the pre-trend and placebo fixes

- **Commits.** `3958ab1c` changed `src/statspai/did/pretrends.py`
  (`sp.sensitivity_rr`), `src/statspai/synth/sdid.py`
  (`sp.synthdid_placebo(kind='time')`) and `src/statspai/registry.py`,
  and re-recorded `tests/r_parity/results/_implementation_trace.json`
  for modules 12 and 76.
- **Reason.** Correctness fix to `sp.sensitivity_rr` and a new option on
  the SDID placebo helper. Neither is called by a Track A module:
  `verify_reproduce_py.py --no-report 12_sdid 76_pretrends` reports 2
  reproduce, 0 drift.
- **Effect on the paper.** None on any table. If the text describes
  `sp.sensitivity_rr` intervals or quotes a breakdown `Mbar` from it, the
  numbers move (wider intervals; see `MIGRATION.md`). The honest-DiD rows
  of the paper use `sp.honest_did`, which is unchanged.

### 2026-10-02 — call traces re-recorded after the example-data loaders were labelled as simulated

- **Commits.** `8b3c53ab` changed `src/statspai/synth/datasets.py` and
  `src/statspai/synth/sdid.py` (docstrings, and `df.attrs['simulated']`
  on the returned frames) and re-recorded
  `tests/r_parity/results/_implementation_trace.json` for modules 07,
  12, 18 and 19, whose entry scripts load those datasets.
- **Reason.** The loaders did not say their rows are simulated. No row
  of any dataset changed: the modules' dumped CSVs and committed results
  are byte-identical, and `verify_reproduce_py.py --no-report` reports 89
  reproduce, 0 drift.
- **Effect on the paper.** None on any number. One thing to check in the
  text: wherever the manuscript calls the module 12 input "California
  Proposition 99", it is the simulated replica
  (`sp.datasets.california_prop99(simulated=True)`), on which both
  StatsPAI and R give -17.9, not the published -15.60. On the real panel
  StatsPAI gives -15.6038279 against R `synthdid` 0.0.9's -15.6038279
  (`tests/reference_parity/test_oct2026_fourth_pass.py`).

### 2026-10-02 — call traces re-recorded twice during the known-truth hardening pass; parity-index denominators moved

- **Commits.** `e364393e` and `06fe553d` re-recorded
  `tests/r_parity/results/_implementation_trace.json` and
  `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them: `badaed34` (loud fallbacks in 62 handlers, many on
  Track A estimation paths), `9d7c9854` (`src/statspai/rd/_cct_bandwidth.py`),
  `5adaff59` (`src/statspai/registry.py`), `6c56e516`
  (`src/statspai/dtr/q_learning.py`, a docstring note) and `c5577401`
  (`src/statspai/did/event_study.py`, `src/statspai/did/pretrends.py`).
- **Reason.** Correctness fixes found by probing estimators without
  parity evidence against simulated designs with a known answer. None of
  the fixed code paths is exercised with the options a Track A module
  uses: `python tests/r_parity/verify_reproduce_py.py --no-report`
  reports 89 reproduce, 0 drift, before and after.
- **Effect on the paper, traces.** None. Checked field by field against
  the version before `e364393e`: in 70 of 89 Track A modules and all 12
  original-data modules only `exercised_sources` digests and `seconds`
  differ; no `packages`, boundary-call package set, `rscript_launches` or
  `error` changed.
- **Effect on the paper, quoted counts.** `docs/parity.md` and
  `src/statspai/_parity_index.json` are not frozen artifacts, but the
  manuscript quotes their "honest denominators" through
  `\ParityEstimatorTotal` in `generated_claims.tex`. `27820fec` moved 84
  plots, bundled datasets, exporters, language-model helpers and
  catalogue listings out of the estimator denominator
  (`statspai._parity_taxonomy.NON_NUMERIC_CALLABLES`), and the new
  anchors in `tests/reference_parity/` added evidence. Estimator
  callables: 822 to 738. With any evidence: 572 to 694 (679 at
  `5630df21`, 694 after the anchors added later the same day). Cross-language:
  417, unchanged, so the cross-language share reads 56.5% where it read
  50.7%. Infrastructure: 131 to 215. All registered: 1274, unchanged.
  The next revision should regenerate `generated_claims.tex` and say in
  the text that the denominator was redefined, since the share rose
  without any new cross-language row.
- **Changed estimator defaults a reader could hit.** `sp.deepiv` now
  defaults to the paired-sample loss; `sp.event_study` drops event times
  with no treated observation; `sp.did_few_treated` reports the exact
  inversion interval. None is a Track A module configuration. Full list
  in `MIGRATION.md` under `oct2026-known-truth-fixes`.

### 2026-09-28 — call traces re-recorded after the agent-surface pass (W7–W10)

- **Commits.** `1e702f2` (result contract on every result class, MCP
  hardening, error taxonomy, discovery / schemas) changed files on Track A
  estimation paths — `src/statspai/core/results.py`,
  `src/statspai/_result_serialize.py`, `src/statspai/did/__init__.py`,
  `src/statspai/_input_validation.py` — which staled the traces;
  `71d0df1` re-recorded `tests/r_parity/results/_implementation_trace.json`
  and `tests/orig_parity/results/_implementation_trace.json`.
- **Reason.** No estimator's numbers changed: the edits add agent-facing
  methods (`to_dict(detail=)`, `violations`, `next_steps`, `result_card`),
  stop `next_steps()` printing, and raise `ColumnNotFound` (still a
  `MethodIncompatibility`) with a did-you-mean hint.
- **Effect on the paper.** None. Checked field by field against the
  previous commit: across all 89 Track A modules and the 12 original-data
  modules no `packages`, boundary-call package set, `rscript_launches` or
  `error` changed; only `exercised_sources` digests and `seconds` differ.
  Traced in a fresh `site-packages` venv (Python 3.11.15, pandas 2.3.3; a
  first pass under pandas 3.0.6 was discarded because the `50_xtabond`
  entry script references `pd.errors.SettingWithCopyWarning`, removed in
  pandas 3).

### 2026-09-28 — freeze paused: the paper has not been submitted yet

- **Reason.** The freeze was written at 1.32.0 on 2026-09-27 on the
  assumption that the JSS package had gone in. It has not; submission is
  planned for about two weeks later. A freeze is meant to start at
  submission, so `"active"` is set to `false` in
  `tests/jss_review_freeze.json` and both checks pass vacuously until then.
- **At submission.** Re-anchor to the release actually submitted and
  re-freeze with `python scripts/jss_review_freeze.py --write --release
  X.Y.Z`, which rewrites every hash and sets `"active": true` again. The
  entries below describe changes against the provisional 1.32.0 anchor;
  they stay as history and do not need to be carried into the new freeze.

### 2026-09-28 — call traces re-recorded after the routing commit changed `__init__.py`

- **Commits.** `e53532f1` (machine-readable estimator routing: exports
  `sp.route` / `sp.decision_guide` from `src/statspai/__init__.py`) went
  in without a re-trace; the ten Track A modules whose recorded path
  includes the package `__init__` (`03_hdfe`, `13_causal_forest`,
  `15_hdfe_cluster`, `24_coxph`, `25_lmm`, `26_glmm_logit`,
  `27_glmm_aghq`, `53_cr2`, `65_spatial`, `66_spatial_gmm`) were stale
  until the commit that carries this entry re-recorded
  `tests/r_parity/results/_implementation_trace.json`.
- **Reason.** Only the exported-name table of `src/statspai/__init__.py`
  changed (two new lazy exports); no estimator source moved. The
  original-data ledger (`tests/orig_parity/results/_implementation_trace.json`)
  does not bind `__init__.py` and was not stale.
- **Effect on the paper.** None. Checked field by field against `HEAD`
  with the same script as the entry below: across all 89 Track A modules
  and the 12 original-data modules no `packages`, boundary-call package
  set, `rscript_launches` or `error` changed; only the ten
  `exercised_sources` digests and `seconds` differ. Traced in the same
  `site-packages` venv (Python 3.11.15, pandas 2.3.3) as the entry below.

### 2026-09-28 — call traces re-recorded for sdid(treat=) and an es_inference docstring

- **Commits.** `80abadb7` (re-trace after `sp.sdid(treat=...)`, commit
  `46ec6369`) and `fb8c61ff` (re-trace of modules 10 and 21 after a
  docstring correction in `src/statspai/did/es_inference.py`). Neither
  recorded an entry here at the time; this one covers both.
- **Reason.** Both edited source files on Track A estimation paths, so the
  bound SHA-256 digests went stale. No estimator code path changed in
  `fb8c61ff`: the edit corrects `sp.uniform_bands`' stated Monte Carlo
  error (about 5e-3, not 1e-3).
- **Effect on the paper.** None. Checked field by field against
  `v1.32.0`: across all 89 Track A modules and the 12 original-data modules
  only `exercised_sources` digests and `seconds` differ, no classification
  moved, the 86 / 0 / 3 census and the original-data provenance marks are
  unchanged. No estimate, standard error or table cell is read from these
  files. (`80abadb7` also re-traced the original-data modules; the first
  version of this entry named only the Track A file, which the per-commit
  check added the same day caught.)
- **For the next revision (not frozen artifacts).** The same commits and
  `a089a6f0` add reference tests that pin the full joint event-study
  covariance behind `sp.event_study_vcov` against R `did`, `fixest`,
  `did2s`, `etwfe` and Stata `did_imputation`. The registry grade of
  `sp.event_study_vcov` moves from unverified to bit-exact, so
  `\ParityCrossLanguage` in `generated_claims.tex` (415 at 1.32.0) will
  read higher when regenerated; other functions added since 1.32.0 move
  the totals as well. The HonestDiD passage in Section 4 ("one extractor,
  `sp.event_study_vcov`") can then cite this cross-language evidence for
  the off-diagonal blocks, which Track A never compared.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json` (`80abadb7`,
    `fb8c61ff`)
  - `tests/orig_parity/results/_implementation_trace.json` (`80abadb7`)

### 2026-09-28 — call traces re-recorded after the agent-native audit fixes

- **Commit.** `c18b1183` (the agent-native audit commit; CHANGELOG
  "Unreleased": MCP tool-list profiles and discovery meta-tools,
  `search_functions` ranking, `sp.did` aliases / indicator-to-cohort,
  `sp.rd(cutoff=)` fix, nested diagnostics and `degradations` in the agent
  payload); re-traced every Track A module and every original-data module.
- **Reason.** `src/statspai/_aliases.py` (did-you-mean on `**kwargs`
  functions) and `src/statspai/core/results.py` (`to_dict` diagnostics /
  degradations, `to_json(detail=)`) sit on every module's estimation path,
  so every `exercised_sources` digest moved and both traces went stale
  (`tests/test_parity_implementation_provenance.py`,
  `tests/test_orig_parity_native_contract.py`). Neither change touches a
  numerical path: the alias wrapper only changes the exception raised for an
  unknown keyword, and the result changes only affect serialisation.
- **Effect on the paper.** None. The re-trace was checked field by field
  against the previous record: no module's `packages`, boundary-call
  package set, `rscript_launches` or `error` changed, so the native / port /
  third-party census of Section 5.3 and Appendix A and the original-data
  ledger's provenance marks are unchanged. What did change besides the
  digests: the recorded tracing environment (Python 3.10.20 → 3.11.15,
  scipy 1.15.3 → 1.17.1, scikit-learn 1.6.1 → 1.9.1; numpy 2.2.6 and
  pandas 2.3.3 as in the release trace), `seconds`, and a few boundary
  callee *names* inside the same package (scikit-learn 1.9 renamed
  `_base.predict` to `_base.MultiOutputLinearModel.predict`), because
  this run was made in a fresh venv rather than the release container. A
  first attempt in the container's `dist-packages` layout was discarded:
  the tracer keys third-party boundaries on `site-packages`, so it had
  silently reclassified every sklearn / statsmodels / pyfixest call as
  native. No estimate, standard error or table cell is read from these
  files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-09-27 — call traces re-recorded after the post-release merge

- **Commits.** The merge `4ad3a993` of `main` (upstream `a916c7bc`..`881fdf8c`: sdid
  refusal of staggered cohorts, LaTeX output precision, `did_had` bandwidth
  selectors, forest covariance attrs, and others) into the 1.32.0 release
  line; re-traced Track A modules 03 13 15 24 25 26 27 35 53 65 66 and
  original-data module 08.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path; the merged commits edited some of those files,
  and upstream had recorded its traces before the release's
  `__version__`-line normalisation.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; no module's implementation classification moved
  (checked field by field against `v1.32.0`), so the native / port /
  third-party census of Section 5.3 and Appendix A (86 / 0 / 3) and the
  original-data ledger's provenance marks are unchanged. No estimate,
  standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
