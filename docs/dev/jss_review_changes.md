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

### 2026-10-19 — did: bacon_decomposition(covariates=)

- **Commits.** `bcc645a0` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (module 20).
- **Reason.** `did/bacon.py` gained the `covariates=` branch (the
  controlled decomposition lives in `did/_bacon_controls.py`, which module
  20 does not execute).
- **Effect on the paper.** None. Module 20 was re-run and reproduces
  `tests/r_parity/results/20_bacon_py.json` byte for byte.

### 2026-10-19 — did: sp.panel_match (new export)

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 03, 13, 15,
  24, 25, 26, 27, 53, 65, 66) and
  `tests/orig_parity/results/_implementation_trace.json` (module 08);
  `5d355f04` re-recorded `tests/r_parity/results/_implementation_trace.json`
  (the same modules) after a rebase.
- **Reason.** One new import in `src/statspai/__init__.py`
  (`sp.panel_match`), which those modules execute. No estimation code on
  their path changed.
- **Effect on the paper.** None. Only the `src/statspai/__init__.py` hash
  and the timings changed in the traces; the committed result hashes are
  the same.

### 2026-10-19 — post-estimation after fixed-effects fits; wild bootstrap by estimator

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `0b4e384b` and `485d9e15` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (every Track A module)
  and `tests/orig_parity/results/_implementation_trace.json` (every
  original-data module).
- **Reason.** `core/results.py` (`fixef()`, FE-aware `predict`),
  `postestimation/_covariance.py`, `fast/fepois.py`, `fast/_feglm.py`,
  `fixest/_native.py` / `wrapper.py`, `inference/jackknife.py` and
  `__init__.py` (`sp.iv_wild_bootstrap`) are on the estimation paths.
  The changes add stored state and new code paths; no estimator's
  reported numbers move.
- **Effect on the paper.** None. Every Track A module re-run into scratch
  reproduces its committed `*_py.json` (89 of 89, worst 1.3e-10 relative)
  and every original-data module too (worst 1.6e-11), so no result file or
  table number changes.

### 2026-10-19 — survival: sp.stteffects, sp.rmst (new exports)

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 03, 13, 15,
  24, 25, 26, 27, 53, 65, 66) and
  `tests/orig_parity/results/_implementation_trace.json` (module 08);
  `5d355f04` re-recorded `tests/r_parity/results/_implementation_trace.json`
  (module 24) after `survival/__init__.py` began loading `sp.rmst` /
  `sp.stteffects` on first use.
- **Reason.** New exports in `src/statspai/__init__.py` (`sp.rmst`,
  `sp.stteffects`), which those modules execute, and a docstring /
  import-order change in `survival/models.py` / `survival/__init__.py`
  (module 24).
- **Effect on the paper.** None. Every listed module was re-run into
  scratch and compared with its committed result: largest relative
  difference 3e-10 (`25_lmm`, pre-existing last-digit jitter), inside the
  1e-9 reproducibility tolerance; the committed results were kept.

### 2026-10-19 — R argument-name aliases on the DiD entry points; aipw propensity in `model_info`

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 04, 17, 38,
  79, 83, 85) and `tests/orig_parity/results/_implementation_trace.json`
  (modules 02, 07).
- **Reason.** `@accepts_aliases` lines in `did/callaway_santanna.py`,
  `did/aggte.py`, `did/event_study.py` and `did/wooldridge_did.py` gained
  the R `did` / `DRDID` spellings; `inference/aipw.py` stores its unclipped
  propensity in `model_info` for `violations()`. No estimate changes.
- **Effect on the paper.** None. Each module was re-run into scratch and
  compared with its committed `*_py.json`: largest relative difference
  1.5e-11 (38_drdid), within the 1e-9 reproducibility tolerance.

### 2026-10-19 — survival: weighted / counting-process Cox and Kaplan-Meier

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (module 24).
- **Reason.** `survival/_cox_core.py` (case weights and `(start, stop]`
  risk sets in the kernel), `survival/models.py` (`sp.cox(weights=, start=,
  ssc=, weights_as=)`) and the new `survival/_cox_inputs.py` are on the
  `24_coxph` estimation path.
- **Effect on the paper.** None. Module 24 was re-run into scratch and
  compared with its committed `24_coxph_py.json`: largest relative
  difference 1e-15 (last-bit summation order), inside the 1e-9
  reproducibility tolerance, so the committed result was kept.

### 2026-10-19 — `sp.mediate` bootstraps the potential-outcome estimator

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (module 36).
- **Reason.** `mediation/mediate.py` is on module 36's estimation path; the
  default `inference='bootstrap'` now also runs for `interaction=True` and
  non-linear models (module 36 uses neither).
- **Effect on the paper.** None. The module was re-run into scratch and its
  `36_mediation_py.json` is byte-identical to the committed one.

### 2026-10-19 — treatment-effect audit fixes (binary-treatment check, matching, DML, TMLE, sensemakr)

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 04, 08, 11,
  22, 50, 71, 72, 79) and `tests/orig_parity/results/_implementation_trace.json`
  (modules 02, 04, 04b, 06, 07, 09, 10, 11).
- **Reason.** `core/_validate.py` (binary-treatment message),
  `dml/_base.py` (clipping warning), `tmle/tmle.py`, `inference/ipw.py`,
  `inference/aipw.py` (typed binary check), `matching/match.py`
  (`exact=` / `covariate_calipers=` / `mahvars=`) and
  `diagnostics/sensemakr.py` (result object) changed on those estimation
  paths (commits `6c367711`, `038bf891`).
- **Effect on the paper.** None. Each module was re-run into scratch and
  compared with its committed `*_py.json`: largest relative difference
  1.9e-13 (07_nhefs_ch13_gformula). `72_tmle_py.json` holds a key
  `epsilon` where the current script writes `epsilon_vec`; that mismatch
  predates these commits (it is on origin/main) and is not touched here.


### 2026-10-19 — shared staggered-DiD input checks

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 04, 05, 16,
  17, 38, 73, 75, 79, 84, 86) and
  `tests/orig_parity/results/_implementation_trace.json` (module 02).
- **Reason.** New `did/_validate.py` and `did/_sa_sample.py`, and edits to
  `callaway_santanna.py`, `aggte.py`, `sun_abraham.py`,
  `did_imputation.py`, `gardner_2s.py`, `stacked_did.py`,
  `wooldridge_did.py` and `synth/fect.py`, on those modules' estimation
  paths (cohort constancy, repeated rows, always-treated cohorts,
  anticipation options, result notes).
- **Effect on the paper.** None. Each module was re-run into scratch and
  compared with its committed `*_py.json`: 04, 05, 16, 17, 75, 84, 86 and
  original-data 02 byte-identical in every number; 38, 73 and 79 differ
  by at most 7.3e-11 relative (pre-existing last-digit jitter, inside the
  1e-9 reproducibility tolerance), so no result file or table number
  changes. None of the parity panels has an always-treated cohort or a
  varying cohort.

### 2026-10-19 — command-name hint in `sp.__getattr__`

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 03, 13, 15,
  24, 25, 26, 27, 53, 65, 66) and
  `tests/orig_parity/results/_implementation_trace.json` (module 08).
- **Reason.** `src/statspai/__init__.py` (on those modules' estimation
  paths) gained the Stata / R command-name hint in the AttributeError
  message (`sp.csdid` names `sp.callaway_santanna`); no estimator code
  changed.
- **Effect on the paper.** None. Each module was re-run into scratch and
  compared with its committed `*_py.json`: largest relative difference
  3.3e-10 (25_lmm), within the 1e-9 reproducibility tolerance, so no
  result file or table number changes.

### 2026-10-19 — `sp.rdrobust` no longer demeans covariates

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 06, 89) and
  `tests/orig_parity/results/_implementation_trace.json` (module 05).
- **Reason.** `rd/rdrobust.py` is on those modules' estimation paths; the
  parser stopped demeaning `covs`, which moved the fuzzy + covariates
  bandwidth off rdrobust's.
- **Effect on the paper.** None. No module uses covariates with a fuzzy
  design; each was re-run into scratch and its `*_py.json` is
  byte-identical to the committed one.

### 2026-10-18 — native multiple estimation and two-way clustered GLMs

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 03, 15, 47,
  53, 67).
- **Reason.** `fast/fepois.py`, `fast/_feglm.py`, `fixest/_native.py` and
  `fixest/wrapper.py` changed on those modules' estimation paths (two-way
  clustering for the GLMs, the `fixef_rm` default notice).
- **Effect on the paper.** None. Each module was re-run into scratch and
  compared with its committed `*_py.json`: largest relative difference
  3.6e-13 (47_ppmlhdfe_3fe), so no result file or table number changes.

### 2026-10-18 — native two-way clustering, `i()` terms and IV in `sp.feols`

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 03, 15, 47,
  53, 67).
- **Reason.** `fast/feols.py`, `fast/inference.py` and `fixest/_native.py`
  changed on those modules' estimation paths (multiway cluster codes, a
  shared cluster-meat helper).
- **Effect on the paper.** None. Each module was re-run into scratch and
  compared with its committed `*_py.json`: largest relative difference
  3.6e-13 (47_ppmlhdfe_3fe), so no result file or table number changes.

### 2026-10-18 — `sp.feols` / `sp.fepois` / `sp.feglm` on StatsPAI's own engine

- **Commits.** `bc3caa3b` changed
  `tests/r_parity/results/67_panel_glm_py.json` and re-recorded
  `tests/r_parity/results/_implementation_trace.json` (46 modules: 01-09,
  11, 13-17, 24, 30, 31, 34-36, 38, 39, 42, 47, 48, 51-59, 67, 71-75, 79,
  81, 84, 85, 89) and `tests/orig_parity/results/_implementation_trace.json`
  (01, 02, 03, 04, 04b, 05, 06, 07, 09, 10, 11).
- **Reason.** `sp.fepois` and `sp.feglm(family="logit")` now fit module 67
  on StatsPAI's native engine instead of pyfixest; the provenance decorator
  (`output/_lineage.py`) and `fast/*` changed on many estimation paths.
- **Effect on the paper.** Module 67 no longer calls pyfixest: in the Track A
  table (`parity_table_3way.tex`) its dagger goes and the caption's counts
  read 87 native and 2 third-party modules (were 86 and 3). Its fepois
  standard errors move by 5e-8 relative (x1 0.0449982127 to 0.0449982104);
  the gaps to R stay 9.1e-6 and 1.4e-5 (R's fixest stops IRLS at its 1e-8
  default) and to Stata 4.1e-7 to 4.07e-7. Logit rows: estimates unchanged at
  1e-15, SE gap to R 8.3e-6 to 8.6e-6, inside the 5e-5 budget. Every other
  re-traced module was re-run into scratch and compared with its committed
  `*_py.json`: largest relative difference 7.3e-11 (73_did2s), so no other
  result file or table number changes.

### 2026-10-18 — Rust kernels rebuilt; parity traces re-recorded

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 01, 03, 14,
  15, 47, 51, 53, 54, 55, 56, 68, 75) and
  `tests/orig_parity/results/_implementation_trace.json` (01, 04, 04b, 09).
- **Reason.** Source files on those modules' estimation paths changed:
  `core/_numba_kernels.py`, `fast/*` and `panel/hdfe.py` now load the
  optional Rust kernels through `statspai/_rust_backend.py`, and
  `fast.feols` / `fast.fepois` check for absorbed regressors. The parity
  environment has no Rust extension (the harnesses now set
  `STATSPAI_SKIP_RUST=1`), so the Python path is the NumPy / Numba one.
- **Effect on the paper.** None. Before re-recording, every affected module
  was re-run into a scratch directory and compared with its committed
  `*_py.json`: the largest relative difference was 1.6e-11 (01_card_original;
  1.1e-11 for 01_ols), inside the 1e-9 reproducibility tolerance, so no
  result file changed and no table number moves. Track C 01_hdfe's timed
  path changed too (`tests/perf/results/_timed_path.json` is not touched
  here); it is to be re-timed on main.

### 2026-10-14 — release 1.39.6: parity tables carry the new version string

- **Commits.** `b840bef4` (release 1.39.6) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None. The manuscript stays anchored to 1.39.1.
  If it is re-anchored later, the appendix parity tables read "StatsPAI
  1.39.6".

### 2026-10-14 — call traces re-recorded after the `sp.stata` round

- **Commits.** `e3e26858` re-recorded Track A module 11 in
  `tests/r_parity/results/_implementation_trace.json` and the 12
  original-data modules in
  `tests/orig_parity/results/_implementation_trace.json`.
- **Reason.** A docstring paragraph in `src/statspai/matching/match.py`
  (why the psmatch2-style default standard error depends on row order
  under ties), which changed the file's digest. No code on the
  estimation path changed.
- **Effect on the paper.** None. Module 11 re-derived with
  `python tests/r_parity/verify_reproduce_py.py 11_psm` reproduces its
  committed result; only digests and `seconds` differ in the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-14 — call traces re-recorded after the October review

- **Commits.** `e3e26858` re-recorded all 89 Track A modules in
  `tests/r_parity/results/_implementation_trace.json` and all 12
  original-data modules in
  `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in the same commit. `9264bdec` re-recorded
  original-data modules 06, 10 and 11 in
  `tests/orig_parity/results/_implementation_trace.json` after a type
  annotation in `src/statspai/inference/ipw.py` (no behaviour change).
- **Reason.** The review of October's work (CHANGELOG, Unreleased)
  edited files on the estimation path of most modules:
  `src/statspai/core/results.py` and `core/_validate.py` (shared by
  nearly all), `regression/ols.py`, `regression/iv.py`,
  `regression/glm.py`, `regression/logit_probit.py`,
  `regression/count.py`, `fast/feols.py`, `fast/fepois.py`,
  `panel/feols.py`, `panel/panel_reg.py`, `did/_core.py`,
  `did/sun_abraham.py`, `matching/_binary_fit.py`, `matching/match.py`,
  `inference/ipw.py`, `inference/aipw.py`,
  `inference/g_computation.py`. The edits add `weights_as=`, record the
  factor levels a fit saw, validate arguments, and change behaviour on
  inputs the parity modules do not use (integer time with gaps,
  separation, single-arm resamples, repeated index labels).
- **Effect on the paper.** None. Before re-recording, every Python-side
  result was re-derived on this tree with
  `python tests/r_parity/verify_reproduce_py.py`: 89 of 89 reproduce the
  committed `*_py.json` within 1e-9 (worst 1.3e-10, module 25), every
  fixture is byte-identical, and `python tests/orig_parity/compare_orig.py`
  rebuilds the original-data table from the committed results. No
  committed result file changed. In both trace files only source digests
  and `seconds` differ.
- **Track C.** `python scripts/trace_perf_path.py --check` reports all
  four timed modules stale (`fast/feols.py`, `fast/fepois.py`,
  `core/results.py`, `core/_validate.py`, `did/_core.py` are on timed
  paths). Modules 02 to 04 were already stale on main after `3d5dfe37`.
  The timings were not re-measured: other sessions were using the
  machine. They are to be re-measured with `tests/perf/run_when_idle.sh`
  before the next re-anchor.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-13 — call traces re-recorded after the sample-size alias in the base result class

- **Commits.** `e3e26858` re-recorded all 89 Track A modules in
  `tests/r_parity/results/_implementation_trace.json` and all 12
  original-data modules in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in the same commit: `src/statspai/core/results.py`.
- **Reason.** `EconometricResults` now fills `data_info["nobs"]` from
  `data_info["n_obs"]` when an estimator recorded only the latter
  (CHANGELOG, Unreleased).
- **Effect on the paper.** None. No estimate or standard error is
  computed differently; in both trace files only the digest of
  `src/statspai/core/results.py` and `seconds` differ, and no committed
  result file changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-12 — release 1.39.5: parity tables carry the new version string

- **Commits.** `d6b412a3` (release 1.39.5) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None. The manuscript stays anchored to 1.39.1.
  If it is re-anchored later, the appendix parity tables read "StatsPAI
  1.39.5".

### 2026-10-12 — call traces re-recorded after `sp.ctree` was exported

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03, 13,
  15, 24, 25, 26, 27, 53, 65 and 66 in
  `tests/r_parity/results/_implementation_trace.json` and of
  original-data module 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in the same commit: `src/statspai/__init__.py` (two new
  exported names, `ctree` and `CTreeResult`).
- **Reason.** Conditional inference trees, and the per-fold refit in the
  cross-validation of `sp.lasso_select` (CHANGELOG, Unreleased).
- **Effect on the paper.** None. The ten Track A modules were re-run first
  and reproduce their committed results exactly; the original-data module
  is byte-identical. In both trace files only the digest of
  `src/statspai/__init__.py` and `seconds` differ. `sp.lasso_select` is
  not on the estimation path of any parity module. The registry census
  moves to 1,566 registered functions across 90 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-12 — call traces re-recorded after `sp.blp_instruments` was exported

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03, 13,
  15, 24, 25, 26, 27, 53, 65 and 66 in
  `tests/r_parity/results/_implementation_trace.json` and of
  original-data module 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in the same commit: `src/statspai/__init__.py` (one new
  exported name).
- **Reason.** The logit and nested logit of Berry (1994) in `sp.blp`, and
  `sp.blp_instruments` (CHANGELOG, Unreleased).
- **Effect on the paper.** None. No estimation path of a Track A or
  original-data module changed; in both trace files only the digest of
  `src/statspai/__init__.py` and `seconds` differ. The registry census
  moves to 1,564 registered functions across 90 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-11 — release 1.39.4: parity tables carry the new version string

- **Commits.** `f3cefc88` (release 1.39.4) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None. The manuscript stays anchored to 1.39.1.
  If it is re-anchored later, the appendix parity tables read "StatsPAI
  1.39.4".

### 2026-10-11 — call traces re-recorded after the efficiency pass

- **Commits.** `339ce542` (the two trace files) and `a65007f4` (module 78
  again, after a handler in `did_multiplegt_dyn.py` was narrowed; the
  module reproduces exactly). The source changes
  behind it are `3fc4b846` (survival), `da110778` (quantile regression),
  `68ec0a88` (count and binary models, `survival/competing_risks.py`),
  `5a5e770e` (`did/_arp.py`), `5735df1e` (matching), `25bde4f3`
  (`event_study`, `lp_did`, `did_multiplegt_dyn`), `a9266495` (SDID) and
  `3190108b` (mixed models).
- **Reason.** A pass for speed (CHANGELOG, "Speed: estimators whose cost
  grew faster than the data") rewrote loops and allocations without
  changing an algorithm. The edited files are on the estimation path of
  21 Track A modules (11 12 21 24 25 26 27 37 40 42 47 48 57 58 61 63 64
  78 83 85 89) and of original-data modules 04 and 04b. The 21 were
  re-run first with `verify_reproduce_py.py`: every fixture is
  byte-identical and every statistic reproduces the committed
  `*_py.json`: 1.8e-15 at worst for 20 of them, and 3.3e-10 for
  `25_lmm` (the intercept standard error; 1.3e-10 on the estimates),
  inside the 1e-9 reproducibility tolerance. The mixed-model likelihood
  is now evaluated from per-group summaries, which rounds differently.
  The two original-data modules left their result files unchanged.
- **Effect on the paper.** None. No result file changed.
  `trace_perf_path.py --check` reports all four Track C modules current:
  no file on a timed path was edited.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-11 — call traces re-recorded for the GLM dispersion and documentation edits

- **Commits.** `5b4c3f6c` (`regression/glm.py`), `68442541` (docstrings
  in `forest/` and `matching/match.py`), `0198e449` (ledgers).
- **Reason.** `sp.glm` now estimates the dispersion of gamma and
  inverse-Gaussian fits by the Pearson statistic, as R and Stata do; the
  other two files changed in documentation only. They are on the
  estimation path of Track A modules 11, 13 and 24 and of the
  original-data modules, which were re-run first: all three reproduce
  the committed `*_py.json`, and all 12 original-data modules reproduce
  (worst relative difference 1.6e-11).
- **Effect on the paper.** None. No Track A module fits a gamma or
  inverse-Gaussian GLM, no result file changed, and no file on the Track
  C timed path changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-10 — call traces re-recorded after the matching / regression / forest fixes

- **Commits.** `54ec5feb` (forest), `769abd08` (matching), `228cc5fe`
  (regression and the JIVE variants in `iv/`), `54fcef66` (ledgers).
- **Reason.** A second pass over options and degenerate inputs off the
  main path (CHANGELOG, "Options and inputs off the main path") edited
  `regression/ols.py`, `regression/iv.py`, `regression/glm.py`,
  `regression/count.py`, `regression/logit_probit.py`, `matching/match.py`,
  `matching/ps_diagnostics.py` and others, which are on the estimation
  path of 28 Track A modules (01 02 11 13 14 24 35 37 41 42 44 45 46 47
  48 49 51 53 54 55 56 57 58 59 61 62 63 64) and of the original-data
  modules. All 28 were re-run first and reproduce the committed
  `*_py.json` (fixtures byte-identical, worst relative difference 5e-12);
  all 12 original-data modules reproduce (worst 1.6e-11, LaLonde exact).
- **Follow-up, same day.** The Linux spot check of these tests failed
  one case on Python 3.12 and 3.13 (`sp.regress(hac_panel=("g", "g"))`
  raised pandas' `TypeError`); the fix is in `regression/ols.py`, so
  modules 01 14 51 53 54 55 56 and the original-data set were re-run
  (all reproduce) and re-recorded again in the commit after this one.
- **Effect on the paper.** None. No result file changed and no file on
  the Track C timed path changed. The fixes are to calls the parity
  tables do not make: matching with weights, an ATE or a covariate
  distance; JIVE; a GLM with HAC errors; a single cluster; split-sample
  forest evaluation.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-10 — release 1.39.3: parity tables carry the new version string

- **Commits.** `51b857b4` (release 1.39.3) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None. The manuscript stays anchored to 1.39.1.
  If it is re-anchored later, the appendix parity tables read "StatsPAI
  1.39.3".

### 2026-10-10 — Track C re-timed after the coverage-pass fixes

- **Commit.** `c9ccbf09`.
- **Reason.** Four files on the timed paths were edited on 2026-10-10 by
  fixes found while closing the core coverage gap (`did/_core.py` and
  `synth/scm.py` in `e631e767`, `did/callaway_santanna.py` in `cdd4f719`,
  `synth/scm.py` again in `baa34df7`, `dml/_base.py` in `2710b368`), so
  `python scripts/trace_perf_path.py --check` reported `02_csdid`,
  `03_scm` and `04_dml` as stale. All four modules were re-timed with
  `tests/perf/run_when_idle.sh` on `b159761f` (run id `20261010T023643`;
  every step started under a load of 1.5, peak 1.44, none discarded) and
  the timed-path record was re-written on the same tree.
- **Effect on the paper.** The manuscript is anchored to v1.39.1 and is
  not edited. Against the timings committed for that anchor, no ratio
  moves by more than run to run noise. Ratio (reference time over
  StatsPAI time), new against old, at the three sizes of each module:
  1. **HDFE.** 1.82 / 0.85 / 0.61 against 1.80 / 0.85 / 0.64.
  2. **CS-DiD.** 1.38 / 2.09 / 2.60 against 1.39 / 1.94 / 2.60.
  3. **Classical SCM.** 1.04 / 1.07 / 3.49 against 1.02 / 1.08 / 3.48.
  4. **DML.** 1.17 / 1.10 / 1.07 against 1.17 / 1.11 / 1.07.
  The package-default SCM timing at 100 donors is 0.044 against 0.043
  seconds. If the paper is re-anchored to a release that contains these
  commits, the Track C table takes the new numbers; the text needs no
  change.
- **Paths.**
  - `tests/perf/results/01_hdfe_R.json`
  - `tests/perf/results/01_hdfe_py.json`
  - `tests/perf/results/02_csdid_R.json`
  - `tests/perf/results/02_csdid_py.json`
  - `tests/perf/results/03_scm_R.json`
  - `tests/perf/results/03_scm_py.json`
  - `tests/perf/results/04_dml_doubleml_py.json`
  - `tests/perf/results/04_dml_py.json`
  - `tests/perf/results/_timed_path.json`

### 2026-10-10 — call traces re-recorded for the zero-standard-error floor

- **Commit.** `4c54b9c5`.
- **Reason.** `rd/rdrobust.py` (the rbc bootstrap) and
  `did/calibrated_simulation.py` compare a standard error with 1e-10 of
  the outcome's size instead of with exactly zero; the nightly run failed
  on Linux because the old test was platform-dependent. `rdrobust.py` is
  on the estimation path of Track A modules 06 and 89 and of the
  original-data Lee module, so their traces were re-recorded. All three
  were re-run first and reproduce the committed `*_py.json` byte for
  byte.
- **Effect on the paper.** None. No result file changed, and no file on
  the Track C timed path changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-10 — call traces re-recorded after the coverage-pass fixes

- **Commits.** `e631e767`, `cdd4f719`, `baa34df7`, `b159761f`.
- **Reason.** Each commit edited StatsPAI source on the estimation path of
  some Track A or original-data modules (the fixes are listed in
  CHANGELOG under "Defects found while closing the core coverage gap" and
  "Teaching notebooks"), so the call-trace ledgers were re-recorded for
  the modules that execute an edited file. Before each re-record the
  affected modules were re-run and compared with the committed
  `*_py.json`. For `b159761f` (modules 05, 08, 19, 71, 75, 78, 86) all
  seven reproduce byte for byte. The session that made `e631e767`
  reported 16 of its 17 modules byte-identical and `73_did2s` at 1.6e-12;
  the checks behind `cdd4f719` and `baa34df7` were run by their authors
  and are not re-verified here.
- **Effect on the paper.** None. No result file changed; the ledgers bind
  source hashes only.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-08 — release 1.39.2: parity tables carry the new version string

- **Commits.** `4e8e561e` (release 1.39.2) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
  The release itself is one fix, to the search of `sp.mswitch` and
  `sp.mswitch_lrtest` (`35ee2cff`), which no frozen artifact exercises.
- **Effect on the paper.** None. The manuscript stays anchored to 1.39.1.
  If it is re-anchored later, the appendix parity tables read "StatsPAI
  1.39.2" and estimator callables with evidence read 936 of 938 in place
  of 935 of 938, `sp.mswitch_lrtest` having gained a grade.

### 2026-10-08 — release 1.39.1: parity tables carry the new version string

- **Commits.** `f775354d` (release 1.39.1) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** This is the release the manuscript is being
  anchored to, so the appendix parity tables read "StatsPAI 1.39.1" and
  `\StatsPAIVersion` is 1.39.1. The registry count is 1,562. The Track C
  timings were taken on the 1.39.0 tree two commits earlier; the timed-path
  record shows every file that ran inside a timed call unchanged in this
  release.

### 2026-10-08 — re-anchor checks: Track C re-timed, one Track B row refreshed, everything else re-derived

- **Commits.** `7c3f6d3d` re-timed the four Track C modules and re-recorded
  `tests/perf/results/_timed_path.json`. `2233e118` regenerated
  `tests/coverage_monte_carlo/results_b1000/coverage_robustness_b1000.json`.
- **Reason.** The manuscript is being moved from 1.34.2 to the 1.39 line
  for submission, so every frozen artifact was re-derived on this tree
  rather than assumed.
- **What was re-derived and did not change.** All 89 R modules reproduce
  their goldens at 1e-9 (`verify_reproduce.py`; module 81 needs
  `STATSPAI_DIDM_LIB`), all 85 Stata modules (`verify_reproduce_stata.py`),
  and the Python side of all 89 modules (`verify_reproduce_py.py`). The
  twelve original-data scripts reproduce to 1.6e-11. The Python side of
  both forest seed studies is identical except for the recorded version
  string. Twelve of the thirteen Track B coverage rows, the two other
  stress rows, the size and power sweep and the four mechanism experiments
  reproduce to the last printed digit. None of those files was rewritten.
- **Effect on the paper.**
  1. Track C, all four rows (`tab:track-c-perf`, the log-log figure and
     the prose of Section 6). At the largest size, against the 1.34.0
     archive: HDFE 0.121 against 0.075 seconds becomes 0.118 against 0.075,
     still `fixest` by 1.6x. Callaway-Sant'Anna 0.028 against 0.457 becomes
     0.030 against 0.079, so 16.1x becomes 2.6x; the reference side moved,
     with `did` 2.5.1 in place of 2.3.0. Classical SCM 264 against 4.6
     becomes 1.3 against 4.6, so `Synth` 58x faster becomes StatsPAI 3.5x
     faster; this is the exact inner solver of 1.36.0. DML 1.10x becomes
     1.07x. The package-default SCM call at 100 donors goes from 7.5 to
     0.04 seconds. The solver-agreement cell of the SCM row prints `3e-06`
     where it printed `0.000`. Section 6's interpretation paragraph, the
     sentence in Section 9 about replacing the inner solver, and the cover
     letter's "61 times faster" all had to be rewritten.
  2. Track B, one cell of `tab:track-b-monte-carlo`: the
     heterogeneous-timing Callaway-Sant'Anna stress row, 0.946 to 0.954.
     The committed value predates 1.31.0. Releases 1.31.0, 1.32.0, 1.33.0,
     1.34.0, 1.34.2, 1.35.0 and this tree were each run and all give 954 of
     1,000, with the same point estimates (median absolute bias identical
     to sixteen digits). So the 1.34.2 manuscript printed a number its own
     release did not produce. Inside the documented band either way.
  3. The timed-path record leaves out `src/statspai/__init__.py` from this
     commit on (reason in `scripts/trace_perf_path.py`). The paper's
     headline check reads the record and needs no change.
- **Paths.**
  - `tests/perf/results/01_hdfe_R.json`
  - `tests/perf/results/01_hdfe_py.json`
  - `tests/perf/results/02_csdid_R.json`
  - `tests/perf/results/02_csdid_py.json`
  - `tests/perf/results/03_scm_R.json`
  - `tests/perf/results/03_scm_py.json`
  - `tests/perf/results/04_dml_doubleml_py.json`
  - `tests/perf/results/04_dml_py.json`
  - `tests/perf/results/_timed_path.json`
  - `tests/coverage_monte_carlo/results_b1000/coverage_robustness_b1000.json`

### 2026-10-07 — call traces re-recorded after `sp.glmnet`; one original-data result regenerated

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24
  25 26 27 53 65 66 in `tests/r_parity/results/_implementation_trace.json`
  and of original-data modules 06 08 10 11 in
  `tests/orig_parity/results/_implementation_trace.json`, and regenerated
  `tests/orig_parity/results/10_nhefs_ch17_survival_py.json`. The source
  that staled the traces is in `cda79296` and `bc0ab40a`:
  `src/statspai/__init__.py` (new exports and name hints),
  `src/statspai/forest/causal_forest.py` (the default of
  `variable_importance`, which no module calls) and
  `src/statspai/inference/ipw.py` (the sandwich variance also covers
  Horvitz-Thompson and clipped weights; the bootstrap path the modules use
  is untouched).
- **Reason.** Second round of the pass recorded in
  `docs/dev/2026-10-07-yuksel-aydede-causal-ml-review.md`.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. All fourteen modules were rerun into a scratch directory first.
  Thirteen reproduce their committed `*_py.json`. Module 10 of the
  original-data ledger (NHEFS chapter 17) reproduces every parity row to
  1e-13, but two diagnostics in its `extra` block could not be reproduced:
  `ipw_pscore_min` / `ipw_pscore_max` were committed as 0.0445 / 0.7951 and
  the code gives 0.0510 / 0.7769, which is also what an independent
  `statsmodels` logit gives on the same 1,566 rows. A checkout of main
  before this change shows the same mismatch, so the stale values predate
  this work; the trace gate binds hashes and had not noticed. The file was
  regenerated. Those two numbers are not parity rows and are not printed in
  the manuscript. The registry census moves to 1,561 registered functions
  across 90 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/10_nhefs_ch17_survival_py.json`

### 2026-10-07 — call traces of the count-model modules re-recorded after the second Gelman-Vehtari round

Commit `ceb55c39` re-recorded the entries of Track A modules 37 42 47 58 63
64 in `tests/r_parity/results/_implementation_trace.json`. The source that
staled them is in `636d4b16`: `regression/count.py`, where `sp.nbreg` now
counts the fixed-effect indicators that survive the collinearity scan
instead of the number of levels minus one. The estimation is untouched.

Effect on the paper: none. The six modules were re-run on the new source
first and reproduce their committed results; in the trace file only
`exercised_sources` digests and `seconds` differ. The original-data ledger
was not stale and is unchanged.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after the Yuksel-Aydede pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 07
  08 13 15 19 24 25 26 27 52 53 65 66 71 in
  `tests/r_parity/results/_implementation_trace.json` and of original-data
  modules 03 07 08 in `tests/orig_parity/results/_implementation_trace.json`.
  The source that staled them is in `e193c47a`:
  `src/statspai/__init__.py` (new exports), `src/statspai/synth/scm.py`
  (the `gsynth` branch of the dispatcher passes a treatment column on),
  `src/statspai/synth/gsynth.py` (several treated units and covariates
  are routed to a new estimator; the single-unit, no-covariate path that
  module 19 runs is untouched), `src/statspai/dml/_base.py` and
  `double_ml.py` (`fold_indices` also reads scikit-learn splits),
  `src/statspai/forest/_grf_inference.py`, `forest_inference.py` and
  `causal_forest.py` (`subset=` for averages; categorical columns on the
  column-name interface), and `src/statspai/inference/aipw.py` (the
  clipping bound became the argument `trim`, default unchanged).
- **Reason.** The pass recorded in
  `docs/dev/2026-10-07-yuksel-aydede-causal-ml-review.md`.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. The modules whose estimation path changed (Track A 07 08 13 19
  24 52 71, original-data 03 07 08) were rerun into a scratch directory
  before re-recording: nine reproduce their committed `*_py.json` byte for
  byte, and the NHEFS chapter 13 module to 1.9e-13 in relative terms, inside
  the 1e-9 reproducibility tolerance, so its committed result stands. In
  both trace files only `exercised_sources` digests and `seconds` differ.
  Two things to fold in at the next re-anchor: the registry census moves to
  1,551 registered functions across 90 submodules, and the list of
  certified rows whose reference is a Python package
  (`PYTHON_REFERENCE_ROWS`) gains `dml_did` (reference: `doubleml`
  `DoubleMLDID` / `DoubleMLDIDCS`), so the manuscript sentence that
  enumerates those rows needs it.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after the Gelman-Vehtari pass (collinear regressors, formula idioms)

Commit `f6a8a215` re-recorded the entries of Track A modules 01 02 03 13 14
15 24 25 26 27 37 40 41 42 44 45 46 47 48 49 51 53 54 55 56 57 58 63 64 65
66 67 in `tests/r_parity/results/_implementation_trace.json` and the 12
modules of `tests/orig_parity/results/_implementation_trace.json`. The
source that staled them is in `c7e63214`: `core/utils.py` (the `y ~ .`
expansion and logical outcomes in `create_design_matrices`),
`core/_collinear.py` (new; the likelihood-based estimators omit a
dependent regressor before fitting), `regression/logit_probit.py`,
`glm.py`, `count.py`, `multinomial.py`, `tobit.py`, `quantile.py`, and
`src/statspai/__init__.py` (new exports). Recorded in
`docs/dev/2026-10-07-gelman-vehtari-active-statistics-review.md`.

Effect on the paper: none. Every Track A design is of full rank, where the
new step returns the design untouched. The 32 modules were re-run on the
new source first and all reproduce their committed results, worst relative
difference 2e-15; the five original-data modules that execute the changed
files reproduce to 2e-11 (last-digit jitter, files restored). In both
trace files only `exercised_sources` digests and `seconds` differ. The
registry census the paper quotes moves to 1,548 registered functions.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.callaway_santanna` gained `balance=`

Commit `dcb99f34`. `sp.callaway_santanna` takes `balance='pair' | 'full' |
'none'` for an unbalanced panel. The default is the rule it already
applied, so no result changes. The edit is in `did/callaway_santanna.py`,
which Track A modules 04 and 79 and the original-data module 02 execute,
so their source hashes in the traces changed.

Effect on the paper: none. The three modules run on balanced panels,
where the option is inert. They were run on the source before and after
the change and their result files are byte-identical; only the recorded
source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the weighted repeated-cross-section fix in `sp.aggte`

Commit `ada363aa`. On the repeated-cross-section route of
`sp.callaway_santanna`, the observation weights are now passed to the
aggregation, so the share-estimation term of an aggregated variance uses
weight mass. The edit is in `did/callaway_santanna.py`, which Track A
modules 04 and 79 and the original-data module 02 execute, so their
source hashes in the traces changed.

Effect on the paper: none. None of the three modules uses weights on
repeated cross-sections. They were run on the source before and after the
change and their result files are byte-identical; only the recorded
source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the weight diagnostic reached logit, probit, glm and iv

Commit `8cdfa8d8`. `sp.logit`, `sp.probit`, `sp.glm` and `sp.iv` now record
the Kish effective sizes of a weighted fit and warn when the weights are
dispersed. The edits are in `regression/logit_probit.py`,
`regression/glm.py` and `regression/iv.py`, which Track A modules 02, 35,
48, 57 and 59 and the original-data module 01 execute, so their source
hashes in the traces changed.

Effect on the paper: none. The six modules were run on the source before
and after the change and their result files are identical; against the
committed files they reproduce within 3e-11, inside the 1e-9
reproducibility tolerance, and the committed files are kept.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after two error classes changed in `sp.fect` and `sp.interflex`

Commit `2df681ff`. Four precondition errors (`method='ife'` without `r=`,
`method='mc'` without `lam=`, `estimator='kernel'` without `bw=`,
`vce='bootstrap'` with another estimator) now raise
`MethodIncompatibility` with a recovery hint instead of a bare
`ValueError`. The edits are in `synth/fect.py` and
`regression/interflex.py`, which Track A modules 86 and 87 execute, so
their source hashes in the trace changed.

Effect on the paper: none. The two modules were run on the source before
and after the change and their result files are byte-identical.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — traces of modules 24, 41 and 62 re-recorded for the second Wooldridge round

Commit `208af7fd`. `regression/tobit.py` and `regression/truncreg.py` gain a
`formula=` argument, and `survival/models.py` accepts the data first in
`sp.survreg`. Those files are on the estimation path of Track A modules
24, 41 and 62, whose traces were re-recorded.

**Effect on the paper.** None. The new argument is not used by the
modules and the fits with `y=` and `x=` are unchanged; only source
digests and `seconds` differ in the ledger, and no implementation
classification moves. No result file changes.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Wooldridge textbook pass

Commit `85cdb0c6`. The pass edits `core/results.py` (`predict`),
`core/utils.py` (the namespace formulas are evaluated in),
`regression/ols.py`, `regression/iv.py`, `regression/glm.py`,
`regression/logit_probit.py`, `regression/quantile.py`,
`panel/panel_reg.py`, `did/_core.py` and the `estat` modules. Those files
are on the estimation path of 72 Track A modules and of original-data
modules, so both ledgers were re-recorded on the committed tree.

**Effect on the paper.** None. Only `exercised_sources` digests and
`seconds` change; compared field by field with the previous ledgers, no
module's implementation classification moves, so the native / port /
third-party census and the provenance marks of the original-data ledger
are unchanged. No estimate, standard error or table cell is read from
these files. The estimates of the modules themselves are untouched: the
changes are to prediction, diagnostics and formula parsing, and the
Track A result files are byte-identical.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — original-data ledger refreshed; LaLonde PSM row regenerated

Commits `c725ba3d` (the result and the roll-up table) and `dfe32e16` (the trace). Eight original-data modules had traces that no longer
described the tree, after source changes that had re-recorded the R
ledger only: `regression/ols.py` (the few-treated diagnostic, `5ad164e1`),
`regression/iv.py`, `matching/match.py` (`39a8be49`), `synth/scm.py` and
`inference/ipw.py`. Every module was rerun on the current tree. Seven
reproduce their committed result within the 1e-9 reproducibility
tolerance (the largest move is 1.3e-10, the Card IV coefficient) and
keep the committed file. One does not: the tied-propensity-score fix of
`39a8be49` moves `psm_att` in module 04.

Effect on the paper: the original-data table's LaLonde row. `psm_att`
goes from 1963.43 to 1967.94 and its standard error from 1186.31 to
1192.17; the relative gap to R `MatchIt` (2006.86) goes from 0.022 to
0.019 and to the published 1794 from 0.094 to 0.097. No other row moves.

- `tests/orig_parity/results/04_lalonde_original_py.json`
- `tests/orig_parity/results/parity_table_orig.md`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after `sp.iv` accepted `small=False` for GMM

Commit `dfe32e16`. `regression/iv.py` gained the GMM branch of the
large-sample rescaling and a longer docstring; Track A modules 02, 35
and 59 execute that file, so their source hashes in the trace changed.

Effect on the paper: none. The three modules were run on the source
before and after the change and their result files are byte-identical.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — outcome-only classic SCM fits the raw pre-treatment path; module 52 regenerated

Commits `51f569ba` and `81314cbb`. `synth/scm.py`, on the estimation path
of Track A modules 07 and 52, changes a default: without `covariates=` or
`special_predictors=` the pre-treatment outcomes are no longer rescaled
period by period before the donor weights are solved
(`standardize_predictors=None`). `51f569ba` re-records the traces of both
modules. `81314cbb` regenerates the Python result of module 52, whose
fits are outcome-only, with the fixture lock, the parity tables and the
index that depend on it.

**Effect on the paper.** Module 07 uses special predictors and reproduces
its committed file exactly. Module 52: the average post-treatment gap,
the pre-treatment RMSE and the five donor weights are unchanged to 1e-14
(the design has a unique exact fit, which both problems share), so its
parity rows against R `Synth` and Stata `synth` are unchanged at
4.1e-10. Its placebo standard error, which no reference reports and no
parity row uses, moves from 2.159192 to 1.828665, because the placebo
fits of the donor units are not exact and follow the new default. In
`parity_table.md` that changes the StatsPAI SE cell of the
`avg_post_gap` row; in the three-way table the Stata column's worst
relative difference for module 52 reads 2e-15 instead of 9.2e-15. No
tolerance, status or tier moves. If the manuscript quotes the README's
Proposition 99 example, that number is now -19.51 (10.80) instead of
-19.76 (11.23).

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/r_parity/results/52_scm_unique_py.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`

### 2026-10-05 — call traces re-recorded after the weight diagnostic reached the fast and negative-binomial entry points

Commit `d7b67583`. `sp.fast.feols`, `sp.fast.fepois` and `sp.nbreg` now
record the Kish effective sizes of a weighted fit and warn when the
weights are dispersed. The edits are in `core/_agent_summary.py`,
`fast/feols.py`, `fast/fepois.py` and `regression/count.py`, which Track
A modules 03, 14, 15, 37, 42, 47, 53, 58, 63 and 64 execute, so their
source hashes in the trace changed.

Effect on the paper: none. The ten modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the few-treated diagnostic reached the fixed-effects entry points

Commit `5ad164e1`. `sp.panel`, `sp.hdfe_ols`, `sp.feols` and `sp.regress`
now record a 0/1 regressor that is ever 1 in only a few clusters and
warn. The edits are in `core/_agent_summary.py`, `regression/ols.py`,
`panel/panel_reg.py`, `panel/feols.py` and `fixest/wrapper.py`, which
Track A modules 01, 14, 35, 51, 53, 54, 55, 56, 67 and 69 execute, so
their source hashes in the trace changed.

Effect on the paper: none. The ten modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded for the third round of the Clarke textbook pass

Commit `ff889a42`. `matching/match.py` and `matching/_ai2016.py` (module 11)
gained the caliper condition of the Abadie-Imbens (2016) ATE variance, a
branch module 11, which estimates the ATT without a caliper, does not
execute. `forest/causal_forest.py` (modules 13 and 24) accepts `treat=` and
`covariates=` as other names for `d=` and `x=`; the decorator line changed
and nothing below it. The traces of modules 11, 13 and 24 were re-recorded.
The rest of the commit is in `agent/_translation/`, which no Track A
module executes.

**Effect on the paper.** None. Only the recorded source hashes move; no
result file changed and no module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after a docstring in `synth/scm.py` changed

Commit `57ccb23f`. `synth/scm.py` is on the estimation path of Track A
modules 07 and 52. Only its docstring changed: the description of the
default without predictors, which called the fit the simplex
least-squares estimator, now says that each pre-treatment period is
rescaled and how to obtain the least-squares weights. No executable line
changed. The traces of modules 07 and 52 were re-recorded on the tree of
that commit.

**Effect on the paper.** None. Only `exercised_sources` digests and
`seconds` changed in the trace, and no module's implementation
classification moved. The Python results of modules 07 and 52 were not
regenerated. No estimate, standard error or table cell is read from this
file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the last two commits of the Croissant textbook pass

Commits `912e97d9` and `623ca007`. Source files on the estimation path of
Track A modules changed: `regression/tobit.py` (module 41; `method='scls'`
is a new branch, the maximum likelihood path is unchanged),
`regression/logit_probit.py` (modules 48 and 57; fits also report a family
of goodness-of-fit measures), `regression/multinomial.py` (modules 44, 45,
46 and 49; fits store their per-observation log-likelihood), and
`src/statspai/__init__.py` (new exports `sp.etpoisson` and `sp.nlogit`),
which is on the path of modules 03, 13, 15, 24, 25, 26, 27, 53, 65 and 66
and of original-data module 08. The traces of all of these were
re-recorded.

**Effect on the paper.** None. Modules 41, 44, 45, 46, 48, 49 and 57 were
re-run with `tests/r_parity/verify_reproduce_py.py` and are byte-identical
to their committed files. No result file was regenerated and no parity row
changes. No module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
### 2026-10-04 — call traces re-recorded after a comment in `callaway_santanna.py` changed

Commit `6c00bcce`. One source file on the estimation path of Track A
modules 04 and 79 changed, `did/callaway_santanna.py`, and only in a
comment: the note above the default propensity-score trimming level said
Stata `csdid` inherits 0.995, and `csdid` 1.81 defaults to no trimming.
No executable line of the file changed. The same commit changes
`panel/feols.py` (a base level for a categorical regressor) and
`inference/g_computation.py`, which are on no Track A module's path. The
traces of modules 04 and 79 were re-recorded on the tree of that commit.

**Effect on the paper.** None. The Python results of both modules were
re-run on the new tree: module 04 reproduces its committed file exactly
and module 79 to 1.4e-15. Only `exercised_sources` digests and `seconds`
changed in the trace. No estimate, standard error or table cell is read
from this file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call trace of module 73 re-recorded after a docstring edit

Commit `75f9a73a`. A paragraph in the docstring of `sp.gardner_did`
(`did/gardner_2s.py`) moved under the `second_stage` parameter so that the
strict docs build parses it. Track A module 73 executes that file.

Effect on the paper: none. No code changed; only the recorded source hash
of module 73 moves.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call trace re-recorded after `sp.rdms` gained the cumulative-cutoff form

Commit `a002611b`. `sp.rdms` without `x2` estimates one score with
cumulative cutoffs. The code is in `rd/rdmulti.py`, which Track A module
89 executes.

Effect on the paper: none. Module 89 calls the two-score form with scalar
cutoffs, which is unchanged; it was re-run and its result file is
byte-identical. Only the recorded source hash moves.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the second round of the Clarke textbook pass

Commit `a4a892ff`. `matching/match.py` and `matching/_ai2016.py` (module
11) gained the Abadie-Imbens (2016) standard error for the ATE, a new
function and a new branch that module 11, which estimates the ATT, does
not execute. `inference/jackknife.py` (module 53) gained
`sp.wild_cluster_boot(confidence_set=)`, off by default. The traces of
modules 11 and 53 were re-recorded.

**Effect on the paper.** None. Only the recorded source hashes move; the
result files of both modules are untouched and no module's implementation
classification changed.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the general form of `sp.gardner_did` and `sp.drdid(est_method='all')`

Commit `87fd76ad`. Two source files on the estimation path of Track A
modules changed. `did/gardner_2s.py` (module 73) gains `treat=`, `fe=`
and `second_stage=`, which hand over to a new module before the existing
code is reached; a call with `first_treat=` runs the lines it ran before.
`did/wooldridge_did.py` (modules 17 and 38) gains
`sp.drdid(est_method='all')`, a loop over the existing estimators that
returns before the body when it is requested. The traces of modules 17,
38 and 73 were re-recorded on the tree of that commit.

**Effect on the paper.** None. The Python results of the three modules
were re-run on the new tree and on the tree before the change. Module 17
reproduces its committed file exactly. Modules 38 and 73 differ from
theirs by 1.5e-11 and 7.3e-11 in relative terms on both trees, so the
difference is the machine's and not the change's, and it is inside the
1e-9 reproducibility tolerance. Only `exercised_sources` digests and
`seconds` changed in the trace, and no module's implementation
classification moved. No estimate, standard error or table cell is read
from this file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.rdmcplot` and `sp.bitest` were exported

Commit `c1af9b25`. Two new public functions are imported in
`src/statspai/__init__.py`, which the modules that reach the package
through a lazy attribute execute: Track A 03, 13, 15, 24, 25, 26, 27, 53,
65, 66 and original-data module 08.

Effect on the paper: none on any number. The eleven modules were re-run
and their result files are byte-identical. The registered-function count
moves from 1,300 to 1,303, which the manuscript's inventory will pick up
at the next re-anchoring.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.best_linear_projection` gained `.attrs['vcov']`

Commit `6a2558e3`. The projection of forest scores now attaches the
coefficient covariance to the returned table. The edit is in
`forest/_grf_inference.py` and `forest/forest_tools.py`, which Track A
modules 13 and 24 execute.

Effect on the paper: none. No estimate or standard error is computed
differently; the result files of modules 13 and 24 are untouched, and
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.survreg` was rewritten

Commit `dbad2724`. `survival/models.py` is on the estimation path of
Track A module 24 (`sp.cox`). Only `sp.survreg` in that file changed (it
now honours `robust=` and `cluster=`, iterates to the optimum and supports
gamma frailty); `sp.cox` was not touched. The trace of module 24 was
re-recorded. The original-data trace was re-recorded in the same run
because the gate reported it stale; none of its modules has
`survival/models.py` on its path, so only digests moved there.

**Effect on the paper.** None. Module 24 was re-run with
`tests/r_parity/verify_reproduce_py.py` and is byte-identical to its
committed file. No module calls `sp.survreg`, so no parity row changes.
No module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — original-data call trace of module 02 re-recorded

Commit `424bdd41`. Commit `6c9fd021` edited `did/callaway_santanna.py`,
which the original-data module 02 (mpdta) executes, and left its trace
stale. The same commit gives `RLassoEffectsResult` the agent contract;
that class is on no parity module's path.

Effect on the paper: none. Module 02 was re-run and its result file is
byte-identical; only the recorded source hash moves.

- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call trace re-recorded after `sp.rdms` gained several boundary points

Commit `7f700ff5`. `sp.rdms` accepts lists for `cutoff1=` / `cutoff2=`
and an `xnorm=` column, returning one row per boundary point and a pooled
row. The code is in `rd/rdmulti.py`, which Track A module 89 executes.

Effect on the paper: none. Module 89 calls `sp.rdms` with scalar cutoffs,
a path that is unchanged; it was re-run and its result file is
byte-identical. Only the recorded source hash moves.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the Remix labs pass (time-varying covariates in Callaway-Sant'Anna, unidentified imputation leads)

Commit `6c9fd021`. Two source files on the estimation path of Track A
modules changed. `did/callaway_santanna.py` (modules 04 and 79) reads a
covariate that varies within unit in the earlier period of each ATT(g,t)
cell instead of in the unit's first row, and gains the improved doubly
robust estimator `estimator='drimp'`. `did/_bjs_pretrends.py` (module 84)
refuses lead coefficients that the design does not identify instead of
inverting a singular matrix. The traces of modules 04, 79 and 84 were
re-recorded on the tree of that commit.

**Effect on the paper.** None. The Python results of modules 04, 79 and 84
were re-run on the new tree, together with 16 and 17, whose estimators
share files with the change. Each reproduces its committed file; the
largest relative difference is 1.4e-15 (module 79). None of these modules
uses a time-varying covariate, and module 84 requests leads that are
identified, so the changed branches are not reached. Only
`exercised_sources` digests and `seconds` changed in the trace, and no
module's implementation classification moved. No estimate, standard error
or table cell is read from this file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the Croissant textbook pass (zero-modified counts, per-observation log-likelihoods, new exports)

Commits `e799808c`, `725a734a` and `d9bfd690`. Source files on the
estimation path of Track A modules changed: `regression/zeroinflated.py`
(modules 63 and 64; the logistic terms of the likelihood are evaluated in
an overflow-safe form, and the Vuong statistic in `diagnostics` compares
against a model fitted on its own), `regression/count.py` and
`regression/logit_probit.py` (modules 37, 42, 47, 48, 57 and 58; fits
store their per-observation log-likelihood), `regression/tobit.py` (module
41; the result keeps its design for `sp.cmtest`), and
`src/statspai/__init__.py` (new exports `sp.ivprobit`, `sp.ivtobit`,
`sp.ivpoisson`, `sp.cmtest`, `sp.vuong`), which is on the path of modules
03, 13, 15, 24, 25, 26, 27, 53, 65 and 66 and of original-data module 08.
The traces of all of these were re-recorded on the tree of `d9bfd690`.

**Effect on the paper.** None. The Python results of modules 37, 41, 42,
47, 48, 57, 58, 63 and 64 were re-run on the new tree with
`tests/r_parity/verify_reproduce_py.py`. Every one reproduces its
committed file, the largest relative difference being 8.9e-16. No result
file was regenerated and no parity row changes. The Vuong statistic that
changed value (see `CHANGELOG.md`) is not an output of modules 63 or 64.
No module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the Clarke textbook pass (matching scores, `sp.twfe_decomposition`, `sp.wild_cluster_boot`)

Commit `39a8be49`. Three source files on the estimation path of Track A
modules changed: `matching/match.py` and `matching/_ai2016.py` (module 11;
the propensity index is evaluated once per distinct covariate row, and a
probit score is available), `did/wooldridge_did.py` (modules 17 and 38;
only `sp.twfe_decomposition`, which neither module calls, was rewritten)
and `inference/jackknife.py` (module 53; `sp.wild_cluster_boot` gained
`h0=`). The traces of modules 11, 17, 38 and 53 were re-recorded.

**Effect on the paper.** None. The Python results of the four modules were
re-run on the new tree. Modules 17 and 53 are byte-identical. Module 11
moves in the sixteenth digit (621.7932847377471 to 621.7932847377473),
which is the row-wise evaluation of the same index. Module 38 differs from
its committed file in the twelfth digit, and does so identically with the
previous `wooldridge_did.py` put back, so that is this machine's numerical
libraries against the ones the file was produced with, not this commit.
Both are far inside the 1e-9 reproducibility tolerance, so the committed
result files were left as they are and no parity row changes. No module's
implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — module 22 (`sp.sensemakr`) standard error from a QR factor; traces re-recorded for the causal-ML textbook pass

Commit `b24b2777`. `sp.sensemakr` computed the treatment standard error
from `pinv(Z'Z)`, which costs eight digits on unscaled regressors. It now
uses the QR factor of `Z`. The same commit fixes classifier nuisances in
`sp.dml` PLR / PLIV, the IRM branch of `sp.dml_sensitivity`, and adds
`sp.lm_lin`; these touch `dml/_base.py`, `dml/plr.py`, `dml/irm.py`,
`dml/pliv.py`, `dml/double_ml.py`, `forest/forest_tools.py`,
`diagnostics/sensemakr.py` and `statspai/__init__.py`, which Track A
modules 03, 08, 13, 15, 22, 24, 25, 26, 27, 53, 65, 66, 71 and
original-data module 08 execute.

Effect on the paper: module 22 only, and only in the agreement columns.
The StatsPAI standard error of the treatment coefficient moves from
587.0203418 to 587.0203721 (R: 587.0203721), so its relative difference
from R falls from 5.2e-08 to 2.1e-15; `t_treat`, `rv_q` and `rv_qa` move
in the eighth digit toward R for the same reason. The point estimate, the
PASS verdict and the registered tolerance (1e-6) are unchanged. Modules
08, 13, 24 and 71 were rerun and their result files are byte-identical
(regressor nuisances and the forest paths are untouched); for the others
only the recorded source hashes move.

- `tests/r_parity/results/22_sensemakr_py.json`
- `tests/r_parity/results/parity_table.md`
- `tests/r_parity/results/parity_table.tex`
- `tests/r_parity/results/parity_table_3way.md`
- `tests/r_parity/results/parity_table_3way.tex`
- `tests/r_parity/TIER_A_FIXTURE_LOCK.json`

### 2026-10-04 — call traces re-recorded after `sp.rdrobust`'s density check moved to `sp.rddensity`

Commit `ba15e1b9`. The manipulation check that `sp.rdrobust` runs beside
the estimate now calls `sp.rddensity` instead of McCrary's binned test,
and the violation message in `core/_agent_summary.py` names the test.
`rd/rdrobust.py` is on the path of Track A modules 06 and 89 and of the
original-data module 05; `core/_agent_summary.py` on that of module 14.

Effect on the paper: none. The four modules were re-run and their result
files are byte-identical; the check sits beside the estimate and does not
enter it. Only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the `sp.rddensity` binomial windows and `sp.rdmc` inference

Commits `7082264e` and `276b2e33`. `sp.rddensity` gained options for the
windows of its binomial tests (`bino_w=` and four more), and
`sp.rdmc(cutoff_var=)` now reports robust bias-corrected intervals. The
edits are in `diagnostics/rddensity.py` and `rd/rdmulti.py`, which
Track A modules 09 and 89 execute, so their source hashes in the trace
changed. The default binomial table, the density test and `sp.rdms`
are untouched.

Effect on the paper: none. Modules 09 and 89 were re-run and their
result files are byte-identical; only the recorded source hashes move.
`sp.rdmc`, `sp.rdrandinf` and `sp.rdwinselect`, whose numbers did
change (see CHANGELOG, "Correctness: local randomization and multi-cutoff
RD"), have no Track A module. In the manuscript they appear only in the
function inventory, whose one-line descriptions are unchanged.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after two warning texts changed

Commit `98538177`. The unbalanced-panel warning and docstring of
`sp.callaway_santanna` now state what `allow_unbalanced_panel=True`
assumes (the result records it in `model_info`), and the `sp.rdrobust`
warning for a discrete running variable quotes the mass-points study.
The edits are in `did/callaway_santanna.py` and `rd/rdrobust.py`, which
Track A modules 04, 06, 79, 89 and the original-data modules 02 and 05
execute, so their source hashes in the traces changed.

Effect on the paper: none. The six modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after the weight diagnostic reached the fixed-effects and count entry points

Commit `7579f977`. `sp.panel`, `sp.hdfe_ols`, `sp.feols`, `sp.poisson` and
`sp.ppmlhdfe` now record the Kish effective sizes of a weighted fit and
warn when the weights are dispersed. The edits are in
`core/_agent_summary.py`, `panel/panel_reg.py`, `panel/feols.py`,
`regression/count.py` and `fixest/wrapper.py`, which Track A modules 14,
35, 37, 42, 47, 58, 67 and 69 execute, so their source hashes in the
trace changed.

Effect on the paper: none. The eight modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the `sp.rdrobust` refusal paths

Commit `4b3a65d4`. `sp.rdrobust(masspoints='off')` now raises
`NumericalInstability` where it used to crash on a complex number or
return a NaN interval, warns when it falls back to the legacy selector,
and warns when clusters coincide with the support points of the running
variable. The edits are in `rd/rdrobust.py` and `rd/_cct_bandwidth.py`,
which Track A modules 06, 88, 89 and the original-data module 05 execute,
so their source hashes in the traces changed.

Effect on the paper: none. The four modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.regress` gained the weight diagnostic

- **Commits.** `712d3bf2`.
- **Reason.** `sp.regress` now records `model_info['n_effective_weights']`
  on a weighted fit and warns when the weights are dispersed and the
  variance is the classical one, or when HC0 / HC1 / HC2 rest on a small
  Kish effective sample. The edits are in `src/statspai/regression/ols.py`
  and `src/statspai/core/_agent_summary.py`, on the estimation path of
  Track A modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of
  original-data modules 01, 04, 04b and 09.
- **Effect on the paper.** None. No Track A module is weighted; a field
  may be added and a warning raised on weighted fits, and nothing is
  computed differently. The Python result files of modules 01 and 14 are
  byte-identical on the parent tree and on the changed tree. Only
  `exercised_sources` digests and `seconds` changed in the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after the effective-cluster diagnostic reached the fixed-effects entry points

- **Commits.** `f030cf55`.
- **Reason.** `sp.panel`, `sp.hdfe_ols` and `sp.feols` now record
  `n_clusters_effective` and warn when many clusters are few in effect,
  through a helper shared with `sp.regress`. The edits are in
  `src/statspai/panel/panel_reg.py`, `src/statspai/panel/feols.py`,
  `src/statspai/fixest/wrapper.py`, `src/statspai/regression/ols.py` and
  `src/statspai/core/_agent_summary.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56, 67 and 69 and of original-data
  modules 01, 04, 04b and 09. Original-data module 08 was re-traced in
  the same commit: its record was stale against `src/statspai/__init__.py`
  on the main it was rebased onto.
- **Effect on the paper.** None. A field is added and a warning may be
  raised; nothing is computed differently. The Python result files of
  modules 14, 35, 67 and 69 are byte-identical on the parent tree and on
  the changed tree. Only `exercised_sources` digests and `seconds`
  changed in the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — release 1.38.0: parity tables carry the new version string

- **Commits.** `58748eb1` (release 1.38.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None while the manuscript stays anchored where
  it is. If it is re-anchored to this release, the appendix parity tables
  read "StatsPAI 1.38.0". The registry count stays at 1,294.

### 2026-10-04 — release 1.37.0: parity tables carry the new version string

- **Commits.** `db5fc782` (release 1.37.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None while the manuscript stays anchored where
  it is. If it is re-anchored to this release, the appendix parity tables
  read "StatsPAI 1.37.0", the registry count is 1,294, and Python 3.10 is
  the oldest supported version.

### 2026-10-03 — call traces re-recorded after `sp.label_values` and `sp.decode` were exported

- **Commits.** `dbe20e11`.
- **Reason.** Two label helpers were added to the public namespace, so
  `src/statspai/__init__.py` changed. That file is on the recorded
  estimation path of Track A modules 03, 13, 15, 24, 25, 26, 27, 53, 65
  and 66. The rest of the commit is label handling in `sp.read_data`,
  `sp.tab`, `sp.describe`, `sp.regtable(labels=)` and `sp.stata`, none of
  which a parity module calls.
- **Effect on the paper.** None. No estimator was touched and no Python
  result file changed; only the `exercised_sources` digest of
  `__init__.py` and `seconds` changed in the trace. The registered
  function count goes from 1,292 to 1,294, which matters only when the
  manuscript is next re-anchored.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.regress` gained the few-treated-clusters diagnostic

- **Commits.** `6c2d7e3a`.
- **Reason.** `sp.regress` now warns when a cluster-level 0/1 regressor has
  few clusters on one side and records `model_info['few_treated_clusters']`.
  The edits are in `src/statspai/regression/ols.py` and
  `src/statspai/core/_agent_summary.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of original-data
  modules 01, 04, 04b and 09.
- **Effect on the paper.** None. A field may be added and a warning
  raised; nothing is computed differently. The Python result files of
  modules 01 and 14 are byte-identical on the parent tree and on the
  changed tree. Only `exercised_sources` digests and `seconds` changed in
  the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 13 and 24 after the forest docstring splice was made version-independent

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` for Track A modules
  13 and 24.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path, and both modules execute
  `src/statspai/forest/regression_forests.py`. The same commit changed how
  that file splices the shared forest options into four docstrings at
  import (clean the docstring, then substitute), because Python 3.13 strips
  docstring indentation at compile time and the old indentation-sensitive
  match never fired there. No estimation code changed and no committed
  result file was regenerated or edited.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed for the two modules; the native / port / third-party
  census is unchanged. No estimate, standard error or table cell is read
  from this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after CR2 / CR3 / two-way clustering took the t(G - 1) reference

- **Commits.** `9a922b3d`.
- **Reason.** `sp.regress`, `sp.feols` and `sp.ivreg` with `vce='cr2'` /
  `'cr3'`, and `sp.regress(cluster=[a, b])`, now take p-values and
  intervals from t(G - 1) instead of the normal. The edits are in
  `src/statspai/regression/ols.py`, `src/statspai/regression/iv.py`,
  `src/statspai/fixest/wrapper.py`, `src/statspai/inference/jackknife.py`
  and `src/statspai/core/_agent_summary.py`, on the estimation path of
  Track A modules 01, 02, 14, 35, 51, 53, 54, 55, 56, 59 and 67 and of
  original-data modules 01, 04, 04b and 09.
- **Effect on the paper.** None on the tables. The Track A rows compare
  estimates and standard errors, which this change does not touch, and
  modules 53, 54 and 56 call the standalone helpers (`sp.cr2_se`,
  `sp.twoway_cluster`, `sp.multiway_cluster_vcov`), not the changed
  branches: the Python result files of 02, 14, 53, 54, 56, 59 and 67 are
  byte-identical on the parent tree and on the changed tree. Only
  `exercised_sources` digests and `seconds` changed in the traces. If the
  text states the reference distribution of the CR2 / CR3 options, it is
  now t(G - 1).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.regress` gained the effective-cluster diagnostic

- **Commits.** `99f78beb`.
- **Reason.** `sp.regress` now records `model_info['n_clusters_effective']`
  and warns when 30 or more clusters are fewer than 30 in effect. The
  edits are in `src/statspai/regression/ols.py` and
  `src/statspai/core/_agent_summary.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of original-data
  modules 01, 04, 04b and 09.
- **Effect on the paper.** None. A field is added and a warning may be
  raised; no estimate or standard error is computed differently. The
  Python result files of the clustered modules 14, 53 and 54 are
  byte-identical on the parent tree and on the changed tree. Only
  `exercised_sources` digests and `seconds` changed in the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after the few-cluster hint was reworded

- **Commits.** `e44aa48e`.
- **Reason.** The recovery hint attached to the few-cluster warning was
  replaced by one shared sentence
  (`statspai.core._agent_summary.FEW_CLUSTERS_HINT`). The edits are in
  `src/statspai/core/_agent_summary.py`, `src/statspai/regression/ols.py`
  and `src/statspai/panel/panel_reg.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of original-data
  modules 01, 04, 04b and 09. Only the text of a warning changed.
- **Effect on the paper.** None on any number: only `exercised_sources`
  digests and `seconds` changed in the traces. One thing to weigh at the
  next re-anchor: the new size study (`tests/reliability/`, not a frozen
  artifact) shows the wild cluster bootstrap over-rejecting with one
  dominant cluster and under-rejecting with two treated clusters. If the
  manuscript recommends it for few clusters without that qualification,
  the sentence should be qualified.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 17 and 38 after a docstring edit in the ETWFE module

- **Commits.** `794e1a9f`.
- **Reason.** The `controls` entry of the `sp.etwfe` docstring was
  rewritten in `src/statspai/did/wooldridge_did.py`, which is on the
  estimation path of modules 17 and 38. No executable line changed.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed in the trace.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 37, 42, 47 and 58 after weighted Poisson fits took their weights into the covariance

- **Commits.** `ddceeee9`.
- **Reason.** `sp.poisson(weights=)` and `sp.ppmlhdfe(weights=)` computed
  the covariance without the weights, and three count models reported an
  unweighted log-likelihood for a weighted fit. The edit is in
  `src/statspai/regression/count.py`, on the estimation path of the four
  modules.
- **Effect on the paper.** None. The four modules are unweighted, and
  without weights the new code multiplies by nothing: their Python
  result files are byte-identical on the parent tree and on the fixed
  tree. Only `exercised_sources` digests and `seconds` changed in the
  trace. The parity index gains one reference test
  (`tests/reference_parity/test_count_weights_stata_parity.py`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`

### 2026-10-03 — call traces re-recorded for 37, 42, 47 and 58 after the count models learned to drop missing rows

- **Commits.** `3917e872`.
- **Reason.** `sp.poisson`, `sp.nbreg` and `sp.ppmlhdfe` failed when a
  plain-column formula met a missing value; they now drop those rows.
  The edit is in `src/statspai/regression/count.py`, on the estimation
  path of the four modules.
- **Effect on the paper.** None. The four fixtures have no missing
  values, so the new step returns the frame untouched: the Python result
  files are byte-identical on the parent tree and on the fixed tree.
  Only `exercised_sources` digests and `seconds` changed in the trace.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 03, 15, 47 and 53 after `sp.fast.feols_jax` took the convention of `sp.fast.feols`

- **Commits.** `ca410f77`.
- **Reason.** The JAX backend kept the pre-1.31 small-sample count while
  reporting `ssc='fixest'`. The edit is in
  `src/statspai/fast/jax_feols.py` and `src/statspai/fast/_jax_fallback.py`,
  which the four modules import through `statspai.fast`.
- **Effect on the paper.** None on any table. No Track A module calls
  `sp.fast.feols_jax`; only `exercised_sources` digests and `seconds`
  changed in the trace. The Track C GPU benchmark times this function,
  and the change adds integer arithmetic outside the timed linear
  algebra, so the timings are not affected in any measurable way; they
  are re-measured at the next re-anchor in any case. One sentence to
  check at that point: wherever the manuscript says the JAX backend
  returns the same result as the NumPy backend, that was true of the
  coefficients and is now also true of the standard errors.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after docstring repairs that unblock the strict docs build

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` for Track A modules
  01, 02, 13, 14, 24, 35, 51, 53, 54, 55, 56 and 59, and
  `tests/orig_parity/results/_implementation_trace.json` for original-data
  modules 01, 04, 04b and 09.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path. The same commit edited four of those files,
  docstrings only: `src/statspai/regression/ols.py` and
  `src/statspai/regression/iv.py` (a References separator and the
  `**kwargs` options of `IVRegression.fit`),
  `src/statspai/forest/regression_forests.py` (indentation of the
  placeholder that is replaced by the shared forest options at import; the
  runtime docstrings were hashed before and after and are identical) and
  `src/statspai/forest/causal_forest.py` (a Returns section).
  `mkdocs build --strict` had aborted on these since 2026-09-28. No
  executable line changed and no committed result file was regenerated or
  edited.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; every other field of all 16 modules was compared with
  the previous file and is identical, so the native / port / third-party
  census and the original-data ledger's provenance marks are unchanged. No
  estimate, standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 03, 15, 47 and 53 after `sp.fast.fepois` took fixest's small-sample factors

- **Commits.** `01260115`.
- **Reason.** `sp.fast.fepois` gained `ssc=` (default `'fixest'`) and a
  weighted `hc1` score. The edit is in `src/statspai/fast/fepois.py`,
  which the four modules import through `statspai.fast`.
- **Effect on the paper.** None. No Track A module calls
  `sp.fast.fepois`: the Poisson modules (37, 47, 58, 67) run
  `sp.ppmlhdfe` / `sp.fepois`, which were not touched. The Python result
  files of 03, 15, 17, 37, 47, 58 and 67 are byte-identical on the parent
  tree and on the fixed tree. Only `exercised_sources` digests and
  `seconds` changed in the trace.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`

### 2026-10-03 — call traces re-recorded for 03, 15, 47 and 53 after the `sp.fast.feols` degrees-of-freedom count for one absorbed dimension

- **Commits.** `67492d1a`.
- **Reason.** With a single absorbed dimension `sp.fast.feols` used
  `n - p - (G - 1)` residual degrees of freedom where fixest and reghdfe
  use `n - p - G`; two clustered layouts were off by one as well. The fix
  is in `src/statspai/fast/feols.py`, on the estimation path of modules
  03, 15, 47 and 53.
- **Effect on the paper.** None. Those modules fit two-way models
  clustered on a key that nests one of the effects (or are unclustered
  two-way fits), the case that was already exact: their Python result
  files are byte-identical on the parent tree and on the fixed tree.
  Only `exercised_sources` digests and `seconds` changed in the trace.
  The Track C timings of `01_hdfe` were already marked stale by earlier
  commits and are re-measured at the next re-anchor; the fix changes one
  integer subtraction. The parity index gains one reference test
  (`tests/reference_parity/test_fast_feols_weights_fixest_parity.py`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`

### 2026-10-03 — call traces re-recorded after `sp.regress` kept its weights and its fitted rows under CR2, CR3 and two-way clustering

- **Commits.** `87a128eb`.
- **Reason.** `sp.regress(weights=)` with `vce='cr2'` / `'cr3'` or two-way
  clustering refit the model without the weights; with missing values in
  a formula variable those variances read their cluster keys from the
  first `n` rows; `sp.twoway_cluster` and `sp.cr2_se` ignored the weights
  of a weighted fit. The fixes are in `src/statspai/regression/ols.py`,
  `src/statspai/inference/jackknife.py` and
  `src/statspai/inference/twoway_cluster.py`, which are on the estimation
  path of Track A modules 01, 14, 51, 53, 54, 55 and 56 and of
  original-data modules 01, 04, 04b and 09.
- **Effect on the paper.** None. Every one of those modules is unweighted
  and runs on data with no missing values, which the fixes leave
  untouched. Checked by running modules 01, 54 and 56 on the source tree
  of the parent commit and on the fixed tree: the three result files are
  byte-identical between the two. (Both differ from the committed files
  in trailing floating-point digits, at most 3e-13, because this session
  runs with `STATSPAI_SKIP_RUST=1` under load; the committed files were
  kept.) Only `exercised_sources` digests and `seconds` changed in the
  traces; no implementation classification moved. The parity index gains
  one reference test
  (`tests/reference_parity/test_regress_vce_weights_stata_parity.py`),
  which is not a Track A module and feeds no table.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`
  - `docs/parity.md`

### 2026-10-03 — release 1.36.0: parity tables carry the new version string

- **Commits.** `0f70d29c` (release 1.36.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None while the manuscript stays anchored where
  it is. If it is re-anchored to this release, the appendix parity tables
  read "StatsPAI 1.36.0", and everything the paper quotes from the registry
  and the parity index should be regenerated against the `v1.36.0` tag at
  the same time (the entries below list what moved since 1.35.0).

### 2026-10-03 — call traces re-recorded for 35 and 69 after `sp.panel` named its default small-sample convention

- **Commits.** `eda627e1`.
- **Reason.** `sp.panel` without `ssc=` now writes
  `model_info['ssc'] = 'linearmodels'` and a description of that scaling,
  and the few-cluster warning mentions `ssc='stata'`. The edit is in
  `src/statspai/panel/panel_reg.py`, which is on the estimation path of
  modules 35 and 69, so their source digests moved.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; the classification of both modules is the same
  (35 third-party `linearmodels`, 69 native). No estimate or standard
  error moves: the default scaling is untouched, and
  `tests/reference_parity/test_panel_ssc_stata_parity.py` still holds
  the default to `linearmodels` at 1e-14.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — minimum-norm rule for non-unique synthetic-control weights; `52_scm_unique` recovers its truth to 2e-14

- **Commits.** `54ad1443`. The traces, parity index and fixture lock are
  re-recorded in the commit that follows it, which carries this entry.
- **Reason.** When the simplex least-squares minimiser is not unique (a
  treated unit inside the donors' hull, a predictor weighted to zero) the
  weights were SLSQP's choice from a uniform start. They are now the
  minimum-norm weights among the minimisers, computed exactly. The rule
  does not depend on the solver or the starting point, and a nested fit
  with its placebo fits takes seconds.
- **Effect on the paper.** `52_scm_unique` improves: `avg_post_gap`
  2.000000155 to 2.000000000 (relative gap to R 7.78e-08 to 4.14e-10) and
  the three donor weights to within 2e-14 of 0.5, 0.3, 0.2 (gaps to R
  1e-7 to 1e-9, which is R's own error). `07_scm`, `18_augsynth`,
  `19_gsynth`, `12_sdid` and original-data `03_basque_original` were
  rerun and their result files are byte-identical: their solutions are
  unique. No verdict, tier or implementation classification changes.
  Track C synthetic-control timings are now far out of date in the
  favourable direction; they are re-measured at the next re-anchor.
- **Paths.**
  - `tests/r_parity/results/52_scm_unique_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table.tex`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.write_data` was exported and the matched-frame helper changed

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` for Track A modules
  03, 13, 15, 24, 25, 26, 27, 53, 65 and 66, and
  `tests/orig_parity/results/_implementation_trace.json` for original-data
  modules 04, 04b and 08.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path. `a6465c4a` edited `src/statspai/__init__.py`
  (the `sp.write_data` export), which those Track A modules and module 08
  execute lazily while estimating, and earlier commits edited
  `src/statspai/matching/_matched_frame.py`, which original-data modules 04
  and 04b execute (`824d202d`, `7fc9ce22`, `ac01445f`; that helper builds
  the matched-sample frame, and the Track A side of those commits was
  re-traced in the entry below). This entry re-records traces only: no
  committed result file was regenerated or edited.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; every other field of all 13 modules was compared with
  the previous file and is identical, so the native / port / third-party
  census and the original-data ledger's provenance marks are unchanged. No
  estimate, standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 11, 35 and 69 after a matching bookkeeping fix and the panel vce check

- **Commits.** `ac01445f` (module 11), `25aeed06` (modules 35 and 69).
- **Reason.** `matching/_matched_frame.py` no longer divides by a zero
  weight sum when a bootstrap replicate of local linear matching has
  cancelling weights (module 11's estimation path), and
  `panel/panel_reg.py` now raises on a `vce=` value it does not
  implement instead of ignoring it (modules 35 and 69). The traces bind
  the SHA-256 of every source file executed.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; their committed `_py.json` results are
  byte-identical. Module 11 runs nearest-neighbour matching, not the
  local linear path that was fixed; modules 35 and 69 pass no `vce=`.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — exact inner solver for synthetic-control weights; three Track A result files and the Basque original-data file move below 4e-7

- **Commits.** `95d87260` (`sp.didregress`, optimized SDID covariates,
  `rddensity` conventional statistic and binomial tests, exact inner
  solver for simplex weights). The traces, schemas, parity index and
  fixture lock are re-recorded in the commit that follows it, which
  carries this entry.
- **Reason.** `synth/_core.py::solve_simplex_weights` solved problems
  with fewer rows than donors by SLSQP, which stops about 1e-5 short in
  the weights. When the minimiser is certified unique (optimality and
  affine independence of the tight donors) it is now solved exactly. Where
  the certificate fails SLSQP's choice is kept. A nested fit on the
  Proposition 99 data goes from 55 seconds to 2.
- **Effect on the paper.** Inside every registered budget; no verdict
  changes. `07_scm`: five values move, the largest
  `weight_Asturias` 0.01489185 to 0.01489185 (3.7e-7 relative; the
  printed table cell goes from 0.0148919 to 0.0148918), and the recorded
  winning start of the outer search is `regression` again in place of
  `dirichlet_3` (the two starts tie; the optimum is the same).
  `18_augsynth`:
  three values, at most 1.0e-7 (`att_augmented` relative gap to R 7.91e-06
  to 7.92e-06, `pre_rmspe` 3e-06 to 3.01e-06). `52_scm_unique`: the
  standard error of `avg_post_gap`, 1.1e-7. Original-data
  `03_basque_original`: estimate -0.89458856 to -0.89458854. Modules 12,
  19 and 09 were rerun and did not change. No implementation
  classification moved in either ledger. Track C synthetic-control timings
  were not re-measured; `scripts/trace_perf_path.py --check` decides at
  the next re-anchor whether they are stale.
- **Paths.**
  - `tests/r_parity/results/07_scm_py.json`
  - `tests/r_parity/results/18_augsynth_py.json`
  - `tests/r_parity/results/52_scm_unique_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`
  - `tests/orig_parity/results/03_basque_original_py.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 16, 73 and 84 after the bootstrap seed was recorded

- **Commit.** `3d9e056d`.
- **Reason.** `sp.did_imputation` and `sp.gardner_did` now write
  `boot_seed` into `model_info` when `vce='bootstrap'`, so the result
  card can report whether the bootstrap SE is reproducible. Both source
  files are on the estimation paths of Track A modules 16, 73 and 84.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; the committed `_py.json` results are
  byte-identical (the modules use the analytic variance, where the new
  field is not written).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after per-coefficient degrees of freedom in the result class

- **Commits.** `bcc3070c` (`sp.difference_in_means`, exact CR2 degrees of
  freedom, `sp.panel` formula terms, Bacon summary). The traces, schemas and
  parity index are re-recorded in the commit that follows it, which carries
  this entry.
- **Reason.** `core/results.py` gained `_coefficient_df`, which reads
  `data_info['df_by_coefficient']` when a fit records it. Only `sp.cr2_se`
  records it. The file is on the estimation path of every module, so both
  ledgers were re-recorded in full. `panel/panel_reg.py`, `did/bacon.py`,
  `iv/iv_diag.py` and `registry.py` changed in the same commit.
- **Effect on the paper.** None. Module `53_cr2` is the only one that
  calls `sp.cr2_se`, and it records the standard errors, which did not
  change (the degrees of freedom did). Without the new key the result class
  returns the scalar it returned before. No result file changed. No implementation classification moved
  in either ledger. The registered-function count goes from 1,280 to 1,281,
  which the paper does not track between anchors.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — parity index: `sp.ttest` gained a reference test

- **Commit.** `606fb6c3` added
  `tests/reference_parity/test_ttest_known_values.py` and regenerated
  `docs/parity.md`.
- **Reason.** The stability audit found `sp.ttest` stable with no
  parity-test evidence.
- **Effect on the paper.** None on any table. In `docs/parity.md`, `ttest`
  moves from "unverified" to analytical-only (T1).
- **Paths.**
  - `docs/parity.md`

### 2026-10-02 — pre-trend power integrated to 1e-7; traces re-recorded for 02, 10, 21, 59, 76 and the original-data ledger

- **Commits.** `826295bf` (Montiel Olea-Pflueger critical values, robust
  endogeneity tests, pre-trend power from coefficients and a covariance).
  The traces, schemas and parity index were re-recorded in the commit that follows it, which carries this entry.
- **Reason.** `sp.pretrends_power` integrates a multivariate-normal
  rectangle probability. SciPy's default absolute tolerance of 1e-5 left
  the power moving in the fifth digit between two calls, so the tolerance
  is now 1e-7 and the slope solver stops at that precision. The same commit
  touched `regression/iv.py` (diagnostics only) and `did/honest_did.py`
  (a refactor of the moments path), which are on the estimation path of
  the other four modules. The original-data ledger had also gone stale
  from the earlier textbook commits (`rd/_cct_bandwidth.py`,
  `rd/rdrobust.py`, `synth/_core.py`, `synth/scm.py`, `regression/iv.py`)
  and from `regression/ols.py` and `core/_vcov_spec.py` on `main`.
- **Effect on the paper.** Module 76 only, inside its registered 1e-3
  budget. The six Python values moved by 1e-6 to 1e-5 in absolute terms,
  for example `power_slope_0p02` 0.33164959 to 0.33164981 and
  `slope_for_power_0p5` 0.02791382 to 0.02791357. The headline relative
  gap to R `pretrends` goes from 3.98e-05 to 4.05e-05 at slope 0.02 and
  from 1.05e-05 to 8.57e-06 at slope 0.05. The verdict and the gap note
  ("rel < 1e-4") stand. Modules 02, 10, 21 and 59 were rerun and their
  result files are unchanged in the committed digits (02 and 59 differ
  from the committed files at 1e-12, which predates this commit and is
  below the 1e-9 reproducibility tolerance, so the files were left alone).
  No implementation classification moved in either ledger.
- **Paths.**
  - `tests/r_parity/results/76_pretrends_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-02 — call traces re-recorded for 17, 38 and 85 after the scope-map fields were added

- **Commit.** `d3f1b828`.
- **Reason.** `sp.event_study` and the linear `sp.etwfe` now record the
  configuration that `sp.validation_scope` reads (`n_adoption_dates`,
  `covariates`, `intensity`, `absorb`, `cluster_level`; `family`) in
  `model_info`. `did/event_study.py` and `did/wooldridge_did.py` are on the
  estimation paths of Track A modules 17, 38 and 85, whose traces bind the
  SHA-256 of every source file executed.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; their committed `_py.json` results are
  byte-identical and no module's implementation classification changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.regress(robust='ewc')`

- **Commit.** `503cb21e` added the `ewc` covariance kind to
  `src/statspai/regression/ols.py` and `src/statspai/core/_vcov_spec.py`,
  a rescaled F for it in `src/statspai/postestimation/hypothesis.py`, and
  re-recorded `tests/r_parity/results/_implementation_trace.json` for
  modules 01 02 07 14 18 37 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57
  58 59 61 62 63 64 67.
- **Reason.** Those 29 modules execute one of the two edited files. The
  new kind is opt-in and no existing branch changed:
  `verify_reproduce_py.py --no-report 01_ols 02_iv 14_ols_cluster 51_newey
  53_cr2 54_twoway_cluster 55_hc2_hc3 56_multiway_cluster` reports 8
  reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; no module's implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-02 — call traces re-recorded after the design-based textbook pass; three estimator fixes on Track A paths

- **Commits.** `fd2d0af3` (`sp.etwfe` with no untreated comparison),
  `58169a79` (RD pilot-bandwidth quantile), `d4a47bac`
  (`sp.match(method='nnmatch')`), `55654839`
  (`sp.mediate(inference='robust')`), `27ebf052` (LIML `kappa`, factor
  terms in IV formulas), `47d229a1` (`sp.synth(v_method='regression')`,
  `sp.rdrobust(scalepar=)`). The regenerated artefacts are in `6fdcfa8f`,
  `47d229a1` and `740fa734`: `docs/parity.md`, and
  `tests/r_parity/results/_implementation_trace.json` re-recorded for
  modules 02 03 06 07 13 15 17 18 19 24 25 26 27 35 36 38 52 53 59 65 66
  69 88 89.
- **Reason.** The traces bind the digests of every source file on each
  module's estimation path, and these commits edited
  `did/wooldridge_did.py`, `rd/_cct_bandwidth.py`, `rd/rdrobust.py`,
  `regression/iv.py`, `panel/panel_reg.py`, `mediation/mediate.py`,
  `synth/_core.py`, `synth/scm.py` and `matching/__init__.py`.
- **Effect on the paper.** No Track A number moved. Three of the commits
  are correctness fixes on code the Track A modules execute, so each was
  re-run and compared with the committed Python result:
  - `06_rd`, `88_rdbwselect`, `89_rdms`: byte-identical. The fix changes
    the pilot bandwidth only when `IQR / 1.349` of the running variable is
    below its standard deviation, which is not the case on those fixtures.
    The claim "data-driven bandwidths match `rdrobust`" was false on such
    data before the fix (5.6e-5 on the Lee House data) and holds to 6e-12
    now; if the paper states the scope of the RD rows, heavy-tailed
    running variables are now covered by
    `tests/reference_parity/test_rdbwselect_iqr_pilot.py`.
  - `59_liml`: moves at 1e-15 (not re-committed). The fixture's `kappa`
    is 1.00057, far enough from 1 that the old computation was accurate;
    the error grew as `kappa` approached 1.
  - `17_etwfe`, `38_drdid`: unchanged. The trim applies only when no unit
    is untreated in some period, and both fixtures have never-treated
    units.
  In `docs/parity.md` the "analytical-only (T1)" count goes from 325 to
  327 and "unverified" from 533 to 531: `from_stata` and `stata` are now
  credited to `tests/reference_parity/test_iv_stata_commands_parity.py`,
  which runs Stata command lines through them and compares with Stata 18
  output. Only `exercised_sources` digests and `seconds` changed in the
  trace; no module's implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `docs/parity.md`

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
  `5d355f04` re-recorded `tests/r_parity/results/_implementation_trace.json`
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

### 2026-10-05 — call traces re-recorded after the cluster-Lasso pass (Gaillac and L'Hour)

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 02 04 05 06 07 08 09 11 12 14 16 17 18 19 20 22 52 53 55 73 88) and
  `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled them are in the same commit:
  `src/statspai/datasets/_canonical.py` and
  `src/statspai/datasets/__init__.py` (the `nsw_dw` docstring and two
  `attrs` flags), `src/statspai/synth/_core.py` and
  `src/statspai/synth/scm.py` (a permutation interval added to
  `model_info`).
- **Reason.** A pass over the companion code of Gaillac and L'Hour's
  *Machine Learning for Econometrics*
  (`docs/dev/2026-10-05-gaillac-lhour-ml4econometrics-review.md`). None
  of the edits changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 07, 11,
  12, 18, 19, 22 and 52 reported 7 reproduce, 0 drift.
- **Effect on the paper.** None. Checked field by field against the
  version before `f1fa33c4`: in the 22 Track A modules and the 12
  original-data modules only `exercised_sources` digests and `seconds`
  differ; no `packages`, boundary-call package set, `rscript_launches` or
  `error` changed. No estimate, standard error or table cell is read
  from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the pass over *The Effect* (Huntington-Klein)

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 02 04 05 06 11 14 16 17 22 37 42 44 45 46 47 48 49 51 53 54 55 56 57
  58 63 64 67 78 79 83 84 85 89) and
  `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled them are in `96f33427`:
  `src/statspai/core/utils.py` (narrow integer columns are widened to
  `int64` before a formula is evaluated), `src/statspai/did/_core.py` (the
  same, for covariate formulas), `src/statspai/rd/rdrobust.py` (a check
  that the columns are numeric), `src/statspai/matching/match.py` (the
  coarsening of `method='cem'`) and `src/statspai/diagnostics/sensemakr.py`
  (group benchmarks).
- **Reason.** A replay of the Stata code of Huntington-Klein's *The
  Effect* (2nd ed.) against Stata 18
  (`docs/dev/2026-10-05-the-effect-2e-review.md`). None of the edits
  changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 34
  modules reported 34 reproduce, 0 drift (fixtures byte-identical, worst
  relative difference 0). No module squares a narrow integer column
  inside a formula, and none uses coarsened exact matching.
- **Effect on the paper.** None. Checked field by field against the
  version before `119ea880`: in the 34 Track A modules and the 12
  original-data modules only `exercised_sources` digests and `seconds`
  differ. No estimate, standard error or table cell is read from these
  files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the pass over the de Chaisemartin-D'Haultfoeuille DiD textbook

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 03 05 13 14 15 17 24 25 26 27 38 51 53 54 55 56 65 66 78 81 86) and
  `tests/orig_parity/results/_implementation_trace.json` (modules 01 04
  04b 08 09). The sources that staled them are in the same commit:
  `src/statspai/regression/ols.py` (`dfadjust=`; the CR2 / CR3 fits store
  their full covariance), `src/statspai/did/did_multiplegt_dyn.py`
  (non-binary treatments, analytic joint tests, `same_switchers`,
  `controls=`), `src/statspai/did/did_multiplegt.py` (non-binary
  treatments), `src/statspai/did/sun_abraham.py` (observation shares,
  `window_rule=`), `src/statspai/did/wooldridge_did.py` (a docstring),
  `src/statspai/synth/fect.py` (large panels, a convergence warning) and
  `src/statspai/__init__.py` (one export).
- **Reason.** A run of the four applications of de Chaisemartin and
  D'Haultfoeuille's DiD textbook against Stata 18
  (`docs/dev/2026-10-05-dcdh-did-textbook-review.md`). None of the edits
  changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 22
  modules reported 22 reproduce, 0 drift (fixtures byte-identical, worst
  relative difference 1.8e-12). The Track A fixtures of modules 05, 78 and
  81 are balanced panels with a binary treatment, where the corrected
  code paths coincide with the old ones.
- **Effect on the paper.** None. Checked field by field against the
  version before `b7b4166f`: in the 22 Track A modules and the 5
  original-data modules only `exercised_sources` digests and `seconds`
  differ. No estimate, standard error or table cell is read from these
  files.
- **Paths.**

### 2026-10-05 — module 43 (Heckman) standard errors, its budget, and the traces after the syllabus pass (Xu and Lan)

- **Commits.** `aab5c0ff` changed `tests/r_parity/results/43_heckman_py.json`
  (the three standard errors), `tests/r_parity/compare.py` (the
  `43_heckman` `rel_se` budget, `5e-4` to `1e-6`), the rendered
  `tests/r_parity/results/parity_table.md` and
  `tests/r_parity/results/parity_table_3way.md`, and
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`. It re-recorded
  `tests/r_parity/results/_implementation_trace.json` (the entries of
  these Track A modules differ from the previous commit:
  01 03 13 14 15 24 25 26 27 33 39 43 51 53 54 55 56 65 66)
  and `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled the traces are in the same commit:
  `src/statspai/regression/ols.py` (the collinearity scan works on blocks
  of the cross-product), `src/statspai/regression/heckman.py`,
  `src/statspai/timeseries/arima.py`, `src/statspai/timeseries/var.py`
  and `src/statspai/__init__.py` (six new exports and an import line
  re-sorted by the pre-commit hook).
- **Reason.** A syllabus audit of 徐小君、蓝嘉俊《因果推断计量经济学》
  against Stata 18
  (`docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`). The
  two-step Heckman variance took the expected information of the
  first-step probit; R `sampleSelection::heckit` and Stata
  `heckman, twostep` take the observed information. That was the whole of
  the gap module 43 carried inside its tolerance.
- **Effect on the paper.** Module 43 only. Its Python standard errors move
  in the sixth digit: intercept 0.14540320 to 0.14540375, slope
  0.05678104 to 0.05678103, inverse Mills ratio 0.21715296 to 0.21717166.
  The worst relative standard-error gap of the module, computed from the
  three committed result files, goes from 1.9e-5 (R) / 8.6e-5 (Stata) to
  7.9e-11 / 4.3e-9, so any table or sentence that quotes the module's SE
  gap or its `5e-4` budget is stale; the module now sits in the default
  `1e-6` tier. Point estimates, the headline metric (`rel_est`) and the
  PASS verdict are unchanged. No other module's numbers changed:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 01, 02,
  03, 14, 15, 33, 35, 39, 43, 51, 53, 54, 55, 56 and 59 reported 15
  reproduce, 0 drift. In both trace files only `exercised_sources`
  digests, `seconds` and module 43's result digest differ; no
  implementation classification moved. The registry census the paper
  quotes also moved with this commit: 1,306 to 1,312 registered
  functions.
- **Paths.**
  - `tests/r_parity/results/43_heckman_py.json`
  - `tests/r_parity/compare.py`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the second round of the syllabus pass

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 14 41 51 53 54 55 56) and
  `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled them are in the same commit:
  `src/statspai/regression/ols.py` (the bread matrix of the centred fit
  is assembled by index arrays instead of a double loop) and
  `src/statspai/regression/tobit.py` (`ll=None` means no lower limit).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`. Neither
  edit changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 01, 14,
  41, 51, 53, 54, 55 and 56 reported 8 reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` differ; no implementation classification moved. No estimate,
  standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded for every module: the SCM interval and `summary()` of infinite bounds

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (all 89 Track A
  modules) and `tests/orig_parity/results/_implementation_trace.json`
  (all 12). The sources that staled them are in the same commit:
  `src/statspai/core/results.py` (`CausalResult.summary()` prints an
  infinite interval end, a file on every estimation path),
  `src/statspai/synth/scm.py` (the `ci` of `sp.synth(method='classic')`
  is now the inversion of the placebo rank test) and
  `src/statspai/datasets/_canonical.py` / `src/statspai/datasets/__init__.py`
  (`nsw_dw(simulated=False)`).
- **Reason.** Decisions left open by the pass over Gaillac and L'Hour's
  *Machine Learning for Econometrics*
  (`docs/dev/2026-10-05-gaillac-lhour-ml4econometrics-review.md`).
- **Effect on the paper.** None on any frozen number: Track A modules
  07 and 52 compare weights, the estimate and fit statistics, not the
  interval, and `python tests/r_parity/verify_reproduce_py.py --no-report
  07_scm 52_scm_unique` reports 2 reproduce, 0 drift. Checked field by
  field against the previous traces: only `exercised_sources` digests and
  `seconds` differ in the 89 + 12 modules; no `packages`, boundary-call
  package set, `rscript_launches` or `error` changed. One thing to carry
  into the next revision of the manuscript: if the text describes the
  confidence interval that classic SCM prints, it is now the
  constant-effect inversion of the placebo test (Firpo and Possebom
  2018), not a normal interval around the placebo spread.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the second round of the dCDH textbook pass

- **Commits.** `e3e26858` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  10 21 78). The sources that staled them are in the same commit:
  `src/statspai/did/did_multiplegt_dyn.py` (the slope-estimation term of
  the variance under `controls=`; the regression behind that option is
  weighted and fitted within `trends_nonparam` cells) and
  `src/statspai/did/es_inference.py` (`sp.uniform_bands` reports a sup-t
  test and takes `terms=`).
- **Reason.** Open items of
  `docs/dev/2026-10-05-dcdh-did-textbook-review.md`. None of the edits
  changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on
  `10_honest_did`, `21_honest_relmags` and `78_multiplegt_dyn` reported
  3 reproduce, 0 drift (worst relative difference 0). Module 78 does not
  use `controls=`, and the event-study covariance modules 10 and 21 read
  is untouched.
- **Effect on the paper.** None. In the three modules only
  `exercised_sources` digests and `seconds` differ from the version
  before `c9a593ea`. No estimate, standard error or table cell is read from
  this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces of modules 68 and 75 re-recorded after `absorb_dof`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 68 and 75
  in `tests/r_parity/results/_implementation_trace.json`. The source that
  staled them is in the same commit: `src/statspai/panel/hdfe.py`
  (`sp.absorb_ols` takes `absorb_dof=`; the default, `'reghdfe'`, is
  the behaviour it had).
- **Reason.** `areg` is translated to the absorbing path
  (`docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`, third
  round). No default changed:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 03, 15,
  37, 68 and 75 reported 5 reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` differ; no implementation classification moved. No estimate,
  standard error or table cell is read from this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the chapter 1 functions of the syllabus pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in the same commit: `src/statspai/__init__.py`
  (four new exports: `prtest`, `sktest`, `swilk`, `ci`).
- **Reason.** Fourth round of the audit in
  `docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`. No
  estimator on a Track A path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves from
  1,312 to 1,316 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the *Causal Inference in R* pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `d5e62968`: `src/statspai/__init__.py` (six new
  exports), `src/statspai/inference/ipw.py` (estimands `'ATO'` and `'ATM'`
  added; the ATE / ATT / ATC branches are untouched) and
  `src/statspai/inference/g_computation.py` (estimand `'ATC'` added).
- **Reason.** Audit in
  `docs/dev/2026-10-05-barrett-causal-inference-in-r-review.md`. No
  estimator on a Track A path changed. The three correctness fixes of the
  pass (Crump trimming, weighted variances in the balance tables, minimal
  adjustment sets) are on no Track A or original-data path.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. Checked by rerunning the Python side of original-data
  modules 06, 07, 09, 10 and 11 and of Track A modules 03 and 53 on the new
  tree: 06, 11, 03 and 53 rewrite their result files byte for byte; 07, 09
  and 10 differ from the committed files in the 13th significant digit.
  Module 07 rerun with the code as it was before `d5e62968` gives the
  same digits as with it, so the difference is between this machine and
  the run that produced the committed files and does not come from this
  pass. The committed files were kept. The registry census the paper quotes
  moves from 1,316 to 1,322 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Facure notebooks pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 02 03
  04 05 07 08 11 13 15 16 17 24 25 26 27 35 36 38 50 52 53 59 65 66 70 71
  72 78 79 83 84 85 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them are in the same commit: the categorical-covariate
  wrapper (`src/statspai/core/_covariates.py`) on `sp.ipw`, `sp.aipw`,
  `sp.match`, `sp.dml`, `sp.tmle`, `sp.g_computation`, `sp.drdid` and
  `sp.callaway_santanna`; the date-typed time wrapper in
  `src/statspai/did/_core.py` on the four staggered estimators; the
  repeated-row check in `src/statspai/did/callaway_santanna.py`; the
  `inference=` validation in `src/statspai/synth/scm.py`; the bracket
  block in `src/statspai/regression/iv.py`; and five new exports in
  `src/statspai/__init__.py`.
- **Reason.** The pass over Facure's *Causal Inference in Python*
  (`docs/dev/2026-10-05-facure-causal-inference-in-python-review.md`).
  Both wrappers pass a call with numeric covariates and numeric time
  straight through.
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 32
  Track A modules reported 32 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census the paper quotes moves from 1,322 to 1,327 registered
  functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the *Linear Model and Extensions* pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 01 02
  03 04 05 07 13 14 15 16 17 18 24 25 26 27 35 37 40 41 42 43 44 45 46 47
  48 49 51 52 53 54 55 56 57 58 59 61 62 63 64 65 66 67 69 78 79 83 84 85
  in `tests/r_parity/results/_implementation_trace.json` and all 12
  modules of `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `8d9ba5a3`: `src/statspai/__init__.py`
  (six new exports), `src/statspai/core/utils.py` (`I(x^2)` read as a
  power), `src/statspai/core/_vcov_spec.py` and
  `src/statspai/regression/ols.py` (`robust='hc4'`; the collinearity
  check of models without an intercept), `src/statspai/regression/iv.py`,
  `src/statspai/regression/glm.py`, `src/statspai/regression/quantile.py`,
  `src/statspai/did/_core.py`, `src/statspai/panel/panel_reg.py`,
  `src/statspai/survival/models.py`.
- **Reason.** The audit in
  `docs/dev/2026-10-05-ding-linear-model-review.md`. No default on a
  Track A path changed: `python tests/r_parity/verify_reproduce_py.py
  --no-report` on the 50 modules reported 50 reproduce, 0 drift. The one
  correctness fix that touches a traced estimator, the Efron score
  residuals behind `sp.cox(robust=)`, does not reach module 24, which
  compares model-based standard errors.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves from
  1,327 to 1,333 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Kohler-Kreuter pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 01 02
  03 13 14 15 24 25 26 27 35 37 42 44 45 46 47 48 49 51 53 54 55 56 57 58
  63 64 65 66 67 in `tests/r_parity/results/_implementation_trace.json`
  and all 12 modules of
  `tests/orig_parity/results/_implementation_trace.json`. The sources that
  staled them are in `0a810f36`: in `src/statspai/core/utils.py` the
  formula helper reads the levels of `C(g)` from the rows that are used
  and accepts columns named like Python keywords, and
  `src/statspai/__init__.py` exports thirteen new names.
- **Reason.** The pass over Kohler, Kreuter and Haensch, *Data Analysis
  Using Stata* (`docs/dev/2026-10-05-kohler-kreuter-4e-review.md`). The
  two changes of the formula helper act on a factor level without
  complete rows and on a column named like a keyword.
  `python scripts/trace_parity_provenance.py` re-ran the 31 Track A
  modules and the 12 original-data modules and reported `ok` for each:
  every one reproduces its committed result.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census the paper quotes moves from 1,333 to 1,346 registered
  functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Dogan textbook pass

- **Commits.** `e3e26858` re-recorded every Track A module in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them are in the same commit: `src/statspai/core/results.py`
  (an `AttributeError` message that names where a statsmodels attribute
  lives), `src/statspai/__init__.py` (two new exports),
  `src/statspai/postestimation/hypothesis.py` (comma-separated
  restrictions, names matched without regard to blanks) and
  `src/statspai/timeseries/arima.py`.
- **Reason.** The pass over Dogan, *Introduction to Econometrics with
  Python* (`docs/dev/2026-10-05-dogan-python-econometrics-review.md`).
  The default `sp.arima` method now maximises the exact likelihood.
  Track A module `39_arima` calls `method="innovations_mle"`, which was
  exact before and is unchanged, so its row does not move.
  `python scripts/trace_parity_provenance.py` re-ran both ledgers and
  reported `ok` for each module: every one reproduces its committed
  result.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census the paper quotes moves from 1,346 to 1,348 registered
  functions. A sentence that describes the default `sp.arima` as exact
  maximum likelihood is true from this commit on, and was not before.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces of modules 24, 48, 57, 63 and 64 re-recorded after the second round of the *Linear Model and Extensions* pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 24 48 57
  63 64 in `tests/r_parity/results/_implementation_trace.json`. The
  source that staled them is in the same commit:
  `src/statspai/survival/models.py` (`sp.kaplan_meier(conf_type=)`
  deprecation of its default; Wald and score tests in `sp.cox`
  diagnostics), `src/statspai/regression/logit_probit.py`
  (`predict(what='confidence')`) and
  `src/statspai/regression/zeroinflated.py` (three more diagnostics
  keys).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-05-ding-linear-model-review.md`. No default on a
  Track A path changed: `python tests/r_parity/verify_reproduce_py.py
  --no-report` on the five modules reported 5 reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` differ; no implementation classification moved. No
  estimate, standard error or table cell is read from this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the second round of the Facure notebooks pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 05 08 11
  16 17 20 36 38 70 71 73 75 81 83 84 85 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `b4f05a76`: the categorical-covariate and date-typed
  time wrappers were put on the remaining DiD and matching estimators
  (`gardner_did`, `stacked_did`, `wooldridge_did`, `bacon_decomposition`,
  `lp_did`, `did_multiplegt`, `event_study`, `psmatch2`, `policy_tree`
  and others).
- **Reason.** Second round of
  `docs/dev/2026-10-05-facure-causal-inference-in-python-review.md`. Both
  wrappers pass a call with numeric covariates and numeric time straight
  through.
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 16
  Track A modules reported 16 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census is unchanged by this round.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the third round of the Facure notebooks pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 13 and
  24 in `tests/r_parity/results/_implementation_trace.json`. The source
  that staled them is in the same commit:
  `src/statspai/forest/_grf_family.py` and
  `src/statspai/forest/causal_forest.py` (categorical covariates become
  one indicator per level; numeric covariates take the path they took
  before).
- **Reason.** Third round of
  `docs/dev/2026-10-05-facure-causal-inference-in-python-review.md`.
  `python tests/r_parity/verify_reproduce_py.py --no-report 13_causal_forest
  24_coxph` reported 2 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; only `exercised_sources` digests and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Dogan follow-up

- **Commits.** `e3e26858` re-recorded every Track A module in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them are in the same commit: `src/statspai/__init__.py`
  (the message of the package-level `AttributeError`),
  `src/statspai/regression/iv.py` (IV results report formula terms under
  the names `sp.regress` gives them),
  `src/statspai/postestimation/hypothesis.py` and
  `src/statspai/postestimation/_design.py`.
- **Reason.** Follow-up to the pass over Dogan, *Introduction to
  Econometrics with Python*
  (`docs/dev/2026-10-05-dogan-python-econometrics-review.md`). The IV
  change relabels coefficients and leaves every array as it was.
  `python scripts/trace_parity_provenance.py` re-ran both ledgers and
  reported `ok` for each module: every one reproduces its committed
  result.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census is unchanged at 1,348. Where the paper prints an IV
  coefficient of a transformed or categorical term, its label is now the
  formula name (`np.log(x)`, `C(g)[T.2]`) and no longer `np.log[x]`,
  `g[2]`.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — Hansen's *Econometrics* pass: conditional logit row, call traces

- **Commits.** `8b2e429b` rewrote `tests/r_parity/results/46_clogit_py.json`.
  `faef6a04` regenerated the rollups that read it
  (`tests/r_parity/results/parity_table.md`, `parity_table.tex`,
  `parity_table_3way.md`, `parity_table_3way.tex`), refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`, and re-recorded the entries of
  Track A modules 02 03 13 15 24 25 26 27 33 35 44 45 46 49 53 59 65 66 69
  in `tests/r_parity/results/_implementation_trace.json` and all 12
  modules of `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `8b2e429b`: `src/statspai/__init__.py` (ten
  new exports), `src/statspai/regression/iv.py`,
  `src/statspai/regression/multinomial.py`, `src/statspai/panel/_cre.py`,
  `src/statspai/panel/xt_tools.py`, `src/statspai/timeseries/var.py`.
- **Reason.** Audit in `docs/dev/2026-10-05-hansen-econometrics-review.md`.
  `sp.clogit` now evaluates its likelihood without a loop over choice sets
  and finishes with Newton steps, so the estimate is at the optimum and no
  longer where BFGS stopped. 2SLS projects by QR and solves the second
  stage by least squares.
- **Effect on the paper.** One row moves. Module 46 (`clogit`,
  `beta_x`): the relative difference from R goes from 1.31e-08 to 2.04e-13
  for the estimate and from 2.68e-09 to 1.01e-13 for the standard error.
  The row was inside the 1e-6 budget before and after, so no evidence tier
  changes. `python tests/r_parity/verify_reproduce_py.py --no-report` on
  the other 18 modules reported 18 reproduce, 0 drift (run before the last
  rebase); module 02 (`iv`) moves by 5.2e-12, inside the 1e-9
  reproducibility tolerance, and its committed file was kept. In both
  trace files only `exercised_sources` digests, `seconds` and the result
  digest of module 46 differ; no implementation classification moved. The
  registry gains ten entries and one submodule (`multivariate`).
- **Paths.**
  - `tests/r_parity/results/46_clogit_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table.tex`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.gam` and `sp.conformal_regression`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 63 64 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `c41e6bfc`: `src/statspai/__init__.py` (three new
  exports: `gam`, `GAMResult`, `conformal_regression`) and
  `src/statspai/regression/zeroinflated.py` (the fitted-value vectors of
  `sp.zip_model` / `sp.zinb` move from `diagnostics` to `data_info`).
- **Reason.** Third round of the audit in
  `docs/dev/2026-10-05-ding-linear-model-review.md`. No estimator on a
  Track A path changed: `python tests/r_parity/verify_reproduce_py.py
  --no-report` on the 12 modules reported 12 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,361
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.xthtaylor`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `d8964f19`: `src/statspai/__init__.py` (one new
  export, `xthtaylor`).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-05-hansen-econometrics-review.md`. No estimator on a
  Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry gains one entry.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the discrete choice covariance

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 44 45 46 49
  in `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `46c3483a`: `src/statspai/regression/multinomial.py`
  (`sp.mlogit`, `sp.ologit`, `sp.oprobit` and `sp.clogit` now keep the
  covariance matrix they already computed).
- **Reason.** Third round of the audit in
  `docs/dev/2026-10-05-hansen-econometrics-review.md`.
- **Effect on the paper.** None. `python tests/r_parity/verify_reproduce_py.py
  --no-report` on 44, 45, 46 and 49 reported 4 reproduce, 0 drift, each
  byte for byte. In the trace files only `exercised_sources` digests and
  `seconds` differ. No registry entry was added.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Das *Causal Inference in R* pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  02 03 11 13 15 24 25 26 27 35 37 42 47 53 58 59 63 64 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of modules 01, 04,
  04b and 08 in `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `757bf198`: `src/statspai/__init__.py`
  (three new exports), `src/statspai/regression/iv.py`,
  `src/statspai/regression/count.py`, `src/statspai/matching/match.py` and
  `src/statspai/survival/models.py`.
- **Reason.** Audit in
  `docs/dev/2026-10-06-das-causal-inference-in-r-review.md`. `sp.iv` reads
  the two-part formula of `AER::ivreg`; `sp.poisson` reports a corrected
  overdispersion statistic among its diagnostics; `sp.match` counts tied
  matches and warns; `CoxResult.ph_test()` is the Grambsch-Therneau score
  test. None of these is on the path that produces a coefficient or a
  standard error.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved. The registry gains three entries
  (`power_ttest`, `esize`, `cor_test`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — Maitra's *Static and Dynamic Econometric Modelling* pass: call traces

- **Commits.** `e3e26858` re-recorded every entry of
  `tests/r_parity/results/_implementation_trace.json` (89 modules) and of
  `tests/orig_parity/results/_implementation_trace.json` (12 modules). The
  source that staled them is in `8e68b894`: `src/statspai/__init__.py` (three
  new exports), `src/statspai/core/results.py` (two pointers for
  statsmodels attribute names), `src/statspai/timeseries/arima.py`,
  `src/statspai/regression/quantile.py`, `src/statspai/inference/ttest.py`
  and `src/statspai/inference/basic_tests.py`.
- **Reason.** Audit in `docs/dev/2026-10-06-maitra-static-dynamic-review.md`.
  `sp.arima` uses the exact diffuse initialisation for differenced models
  and searches a badly scaled series on a standardised copy.
- **Effect on the paper.** None. `python tests/r_parity/verify_reproduce_py.py
  --no-report 39_arima 40_qreg`, the two Track A modules that run the
  changed estimators, reported 2 reproduce, 0 drift, both byte-identical:
  module 39 fits an undifferenced series of order one, which takes the
  unchanged path. No result file, parity table or fixture lock changed. In
  both trace files only `exercised_sources` digests and `seconds` differ;
  no implementation classification moved. The registry gains three entries
  (`chow_test`, `bds`, `rolling`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Gow and Ding pass

- **Commits.** `e3e26858` re-recorded the entries of 45 Track A modules in
  `tests/r_parity/results/_implementation_trace.json` and of 6 of the
  12 modules of `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `a07851ca`: `src/statspai/core/utils.py` (`factor(x)`
  read as `C(x)`; the formula rewriter is renamed `r_formula_idioms`, and
  its callers in `regression/ols.py`, `glm.py`, `iv.py`, `multinomial.py`,
  `panel/panel_reg.py`, `did/_core.py`, `postestimation/_design.py`,
  `survey/estimators.py` with it), `src/statspai/regression/ols.py`
  (`hac_panel=`; the two-way clustered covariance is stored and adjusted
  when it is not positive semi-definite), `src/statspai/regression/count.py`
  (Poisson starting values and the flat-deviance stop),
  `src/statspai/inference/jackknife.py`, `src/statspai/fixest/wrapper.py`
  (a warning) and `src/statspai/__init__.py` (four new exports).
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. Two of the changes
  alter numbers, and neither is reached by a Track A module: no module
  clusters in two dimensions on a covariance that is not positive
  semi-definite, and every Poisson fit on a Track A path converged before
  and converges to the same estimate now. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on the 45 modules
  reported 45 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,372
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Ness *Causal AI* pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 04 08 13 14 15 24 25 26 27 50 53 65 66 71 72 79 in
  `tests/r_parity/results/_implementation_trace.json` and of modules 02, 06,
  07, 08, 09, 10 and 11 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `159d0cb5`: `src/statspai/__init__.py` (four new
  exports), `src/statspai/core/_validate.py`,
  `src/statspai/core/_agent_summary.py`, `src/statspai/inference/ipw.py`,
  `src/statspai/inference/aipw.py`, `src/statspai/inference/g_computation.py`,
  `src/statspai/tmle/tmle.py` and `src/statspai/dml/_base.py`.
- **Reason.** Audit in `docs/dev/2026-10-06-ness-causal-ai-review.md`. The
  estimators named above convert the treatment column through one shared
  check that names a column of labels; `sp.ipw` also stores each arm's
  effective sample size and warns when it is small, and the violations
  report reads it. None of these is on the path that produces a
  coefficient or a standard error. The substantive fixes of the pass are
  in `dag/` and `causal_discovery/`, which no Track A module exercises.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved. The registry gains four entries
  (`bayes_net`, `BayesNet`, `refute`, `RefutationResult`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the forecasting toolkit and the `sp.arima` optimiser change

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 39 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of module(s)
  08 in `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `fdcf0a19`: `src/statspai/__init__.py` (19
  new exports, the forecasting functions and their result classes) and, for
  module 39 only, `src/statspai/timeseries/arima.py`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-hyndman-fpp-pythonic-review.md`. `sp.arima` with the
  default `method='statespace'` now starts its likelihood search twice and
  checks it with simplex searches, and estimates a differenced model on the
  differenced series. Module 39 runs `method='innovations_mle'`, which is
  unchanged: `python tests/r_parity/verify_reproduce_py.py --no-report` on
  Track A modules 03 13 15 24 25 26 27 39 53 65 66 reported 11 reproduce, 0
  drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. In the trace files `exercised_sources` digests and `seconds`
  differ, and for module 39 the count of attribute reads on the statsmodels
  result object moves from 11 to 8; its classification as served by
  statsmodels is unchanged. The registry census the paper quotes moves to
  1,395 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Bayesian econometrics pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `cefca309`: `src/statspai/__init__.py` (the
  exports of the new `statspai.mcmc` subpackage: `bayes_regress`,
  `bayes_mixed`, `bma`, `bayes_factor`, `savage_dickey`,
  `bayes_bootstrap`, seven chain diagnostics and their result classes).
- **Reason.** The audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed. The
  correctness fixes of that commit are in `sp.bayes_iv`,
  `sp.bayes_hte_iv` and `sp.bayes_fuzzy_rd`, which no parity module
  calls.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,413
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the dynamic panel fix (Hansen, Table 17.3)

- **Commits.** `e3e26858` re-recorded the entry of Track A module 50 in
  `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled module 50 is in `1879dce1`:
  `src/statspai/gmm/_dynpanel/_estimate.py`,
  `src/statspai/gmm/_dynpanel/_fit.py` and
  `src/statspai/gmm/arellano_bond.py`.
- **Reason.** Audit in `docs/dev/2026-10-05-hansen-econometrics-review.md`
  (findings 15 to 18). Three changes to dynamic panel GMM: which period
  dummies `time_dummies=True` keeps, a rank check before the weight matrix
  is inverted, and the generalized inverse used when it is singular. Each
  one acts only on a fit that has period dummies or a rank-deficient
  weight. Module 50 has neither.
- **Effect on the paper.** None. No committed result file changed; module
  50 rerun on the new code returns its two coefficients and standard
  errors unchanged to the sixteenth digit. In the Track A trace only the
  three `exercised_sources` digests of module 50 and its `seconds`
  differ. In the original-data trace only `seconds` differ. No
  implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call trace of module 39 re-recorded after the `sp.arima` likelihood change

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 39 in
  `tests/r_parity/results/_implementation_trace.json`. The source that
  staled them is in `2d82b3f8`: `src/statspai/timeseries/arima.py`.
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-06-hyndman-fpp-pythonic-review.md`. `sp.arima` with the
  default `method='statespace'` evaluates the likelihood of the
  differenced series by the innovations algorithm
  (`timeseries/_arma_core.py`) instead of statsmodels' Kalman filter; the
  two agree to 1e-8. Module 39 runs `method='innovations_mle'`, which is
  unchanged: `python tests/r_parity/verify_reproduce_py.py 39_arima
  --no-report` reported 1 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace file only `exercised_sources` digests and
  `seconds` differ. The registry census stays at 1,413 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.abnormal_returns` and the `sp.feols` covariance adjustment

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 67 in
  `tests/r_parity/results/_implementation_trace.json` and of 1 module(s) of
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `8abcf7f1`: `src/statspai/fixest/wrapper.py`
  (`sp.feols` sets the negative eigenvalues of a multiway clustered
  covariance to zero, as R's fixest does) and `src/statspai/__init__.py`
  (two new exports: `abnormal_returns`, `AbnormalReturnsResult`).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. The adjustment
  acts only on a multiway covariance that is not positive semi-definite,
  which no Track A module has: `python
  tests/r_parity/verify_reproduce_py.py --no-report` on the 11 modules
  reported 11 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,415
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — the Lalonde matching row after `sp.match` keeps tied matches by default

- **Commits.** `4d79e7b2` rewrote
  `tests/orig_parity/results/04_lalonde_original_py.json`. `b627761f`
  regenerated the rollup that reads it
  (`tests/orig_parity/results/parity_table_orig.md`) and re-recorded the
  entries of Track A module 11 in
  `tests/r_parity/results/_implementation_trace.json` and of modules 04 and
  04b in `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `4d79e7b2`: `src/statspai/matching/match.py`.
- **Reason.** Decision recorded in
  `docs/dev/2026-10-06-das-causal-inference-in-r-review.md`. With
  replacement, `sp.match` now keeps every control tied at the nearest
  distance (`ties='all'`) instead of the first in data order, so the
  estimate no longer depends on the order of the rows.
- **Effect on the paper.** One row of the original-data table moves.
  Module 04 (`psm_att`, Lalonde, 614 rows): the StatsPAI estimate goes from
  1967.94 to 1968.80 (the table prints 1968, now 1969) and its standard
  error from 1192.17 to 1181.66. The relative gaps to R `MatchIt` (0.019)
  and to the published value (0.097) are unchanged at the precision
  printed. The new value equals Stata 18
  `teffects psmatch (re78) (treat ..., logit), atet` (1968.799715855857) to
  the last digit. Module 04b (NSW with PSID-1) and Track A module 11
  (NSW-DW) have no exact ties and did not move:
  `python tests/r_parity/verify_reproduce_py.py --no-report 11_psm` reports
  the fixture byte-identical. In the Track A trace only the
  `exercised_sources` digest of module 11 differs.
- **Paths.**
  - `tests/orig_parity/results/04_lalonde_original_py.json`
  - `tests/orig_parity/results/parity_table_orig.md`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.threshold` (Hansen, chapter 23)

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `b6a1cdb9`: `src/statspai/__init__.py` (one new export,
  `threshold`).
- **Reason.** Audit in `docs/dev/2026-10-05-hansen-econometrics-review.md`
  (finding 19). `sp.threshold` is a new function in a new file,
  `src/statspai/regression/threshold.py`, which no parity module calls.
  Nothing on the path of an existing estimator changed.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only the `src/statspai/__init__.py` digest under
  `exercised_sources` and `seconds` differ; no implementation
  classification moved. The registry gains one entry.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.path_analysis`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of module 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `10eff6e9`: `src/statspai/__init__.py` (two new exports).
- **Reason.** A new function, `sp.path_analysis`, in a module no parity
  script calls.
- **Effect on the paper.** None. No committed result file changed; in both
  trace files only the digest of `src/statspai/__init__.py` and `seconds`
  differ. The registry gains two entries.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.bayes_ivreg`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `1e429eef`: `src/statspai/__init__.py` (one new
  export, `bayes_ivreg`).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,419
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.identify_counterfactual`

- **Commits.** `c3f1f752` holds the re-recorded entries of the Track A
  and original-data modules whose import path includes
  `src/statspai/__init__.py`, in
  `tests/r_parity/results/_implementation_trace.json` and
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `81821d0f`: two new exports in
  `src/statspai/__init__.py`.
- **Reason.** `sp.identify_counterfactual` (ID* and IDC*), the open item
  of `docs/dev/2026-10-06-ness-causal-ai-review.md`. The new module is
  `src/statspai/dag/counterfactual_id.py`, which no Track A module
  exercises.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved. The registry gains two entries
  (`identify_counterfactual`, `CounterfactualIdentification`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after a typing fix

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  04 05 06 13 16 17 20 24 44 45 49 50 73 75 77 78 79 80 83 84 85 88 89 in
  `tests/r_parity/results/_implementation_trace.json` and of two modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `fd124cce`: 45 return statements in 23 files, each
  wrapped in `np.asarray`.
- **Reason.** The mypy step of CI had been failing on main (848 errors
  against a ceiling of 827), which skipped the test run behind it.
  `np.asarray` is the identity on an array: `python
  tests/r_parity/verify_reproduce_py.py --no-report` on the 23 modules
  reported 23 reproduce, 0 drift.
- **Effect on the paper.** None; in both trace files only
  `exercised_sources` digests and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the last items of the forecasting audit

- **Commits.** `e3e26858` re-recorded the entries of Track A module(s) 03 13 15 24 25 26 27 39 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of original-data
  module(s) 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `122b028c`: `src/statspai/__init__.py` (five new exports:
  `ts_features`, `bootstrap_series`, `bagged_forecast`,
  `BaggedForecastResult`, `seasonal_dummies`) and, for module 39,
  `src/statspai/timeseries/arima.py` (new `boxcox=` and `biasadj=`
  arguments, unused by the module).
- **Reason.** Fourth round of the audit in
  `docs/dev/2026-10-06-hyndman-fpp-pythonic-review.md`. No estimator on a
  Track A or original-data path changed: `python
  tests/r_parity/verify_reproduce_py.py 39_arima --no-report` reported 1
  reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,426
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the feols IV diagnostics

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  [67] in
  `tests/r_parity/results/_implementation_trace.json` and of 0 module(s)
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `e497086c`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on those modules:
  Summary: 1 reproduce, 0 drift, 0 skipped/errored.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the matchit translation fix

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  [] in
  `tests/r_parity/results/_implementation_trace.json` and of 0 module(s)
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `d42a34b3`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on those modules:
  no Track A module was stale
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the FCI, mediation, hill-climbing and forest-tuning batch

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 08 11 13 15 24 25 26 27 36 53 65 66 70 71 in
  `tests/r_parity/results/_implementation_trace.json` and of modules
  04 04b 08 in `tests/orig_parity/results/_implementation_trace.json`.
  The source that staled them is in `284c81a9`.
- **Reason.** Four new functions (`hill_climb`, `bootstrap_edges`, `tune_causal_forest`, and the non-linear outcome models of `sp.mediate`), the completed `sp.fci`, and a higher cap on the MCP tool manifest; notes in `docs/dev/2026-10-06-das-causal-inference-in-r-review.md`. The feature commit is the one before the cited source commit. No parity script calls the changed estimation code.
- **Effect on the paper.** None. No committed result file changed; in both trace files only source digests and `seconds` differ. The registry gains three entries.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after backtick-quoted names in formulas

- **Commits.** `e3e26858` re-recorded the entries of every Track A and
  original-data module whose estimation path reads a formula, in
  `tests/r_parity/results/_implementation_trace.json` and
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `9c88e2d2`: `src/statspai/core/utils.py`,
  `src/statspai/regression/ols.py`, `src/statspai/regression/iv.py`,
  `src/statspai/regression/count.py` and `src/statspai/regression/gam.py`.
- **Reason.** Third round of
  `docs/dev/2026-10-06-ness-causal-ai-review.md`. A column name written
  between backticks is read as `Q("...")`; the IV parser no longer
  rewrites a hyphen inside a quoted name. A formula without backticks or
  quoted names takes the same path as before: the translation is skipped
  unless the formula contains a backtick.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after a docstring punctuation fix

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  [] in
  `tests/r_parity/results/_implementation_trace.json` and of 0 module(s)
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `1ee8048d`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on those modules:
  no Track A module was stale
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the latent-variable engine in sp.path_analysis

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  none in
  `tests/r_parity/results/_implementation_trace.json` and of modules
  nonein `tests/orig_parity/results/_implementation_trace.json`.
  The source that staled them is in `76566f98`.
- **Reason.** sp.path_analysis gained latent variables, mean structures and growth curves; the registry entry and the schema bundle changed with it. No Track A or original-data module calls the function.
- **Effect on the paper.** None. No estimate in any table changes; only source hashes on the traced call paths moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after sp.mmpc and sp.mmhc

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of modules
  08 in `tests/orig_parity/results/_implementation_trace.json`.
  The source that staled them is in `00f36f67`.
- **Reason.** Two functions were added to the registry (sp.mmpc, sp.mmhc), which changes registry.py and the package __init__ on every traced call path. No Track A or original-data module calls them.
- **Effect on the paper.** None. No estimate in any table changes; the registered-function count the paper quotes is pinned to the anchored release.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after missing='fiml' in sp.path_analysis

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  none in
  `tests/r_parity/results/_implementation_trace.json` and of modules
  nonein `tests/orig_parity/results/_implementation_trace.json`.
  The source that staled them is in `f70ee85f`.
- **Reason.** sp.path_analysis gained a missing= option; the registry entry and the schema bundle changed with it. No Track A or original-data module calls the function.
- **Effect on the paper.** None. No estimate in any table changes; only source hashes on the traced call paths moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the lmrob covariance option and the event-study rank tests

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  [] in
  `tests/r_parity/results/_implementation_trace.json` and of 0 module(s)
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `a014de33`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on those modules:
  no Track A module was stale
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.mprobit`, `sp.series` and the chapter 26 translations (Hansen)

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `29a44c6d`. The exercised files whose digests moved:
  `src/statspai/__init__.py`.
- **Reason.** Audit in `docs/dev/2026-10-05-hansen-econometrics-review.md`
  (findings 20 to 26). New functions in new files (`sp.mprobit`,
  `sp.series`), new arguments with unchanged defaults (`sp.nlogit`,
  `sp.xthtaylor`), a rotation method on the factor result, the covariance
  matrix kept by `sp.mixlogit`, and translator additions. In
  `postestimation/hypothesis.py` one pattern now also reads the name of a
  factor level nested in another (`C(t)[3]`), which matters only to
  `testparm` on a product of factors.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ
  (lines that are neither: 0); no implementation classification
  moved. The registry gains two entries (`mprobit`, `series`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after `sp.dlm`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `ea557463`: `src/statspai/__init__.py` (two new
  exports, `dlm` and `DLMResult`).
- **Reason.** Third round of the audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,435
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — R reference moved to did 2.5.1 and DRDID 1.3.0

- **Commits.** `4ca82f1c` regenerated the four Track A R goldens whose
  provenance records `did` or `DRDID`, under `did` 2.5.1 and `DRDID` 1.3.0
  (were 2.3.0 and 1.2.3), and refreshed the lock, the reproducibility
  report and the environment file.
- **Reason.** CRAN has carried `did` 2.5.x since June 2026. A reader who
  installs the reference today gets 2.5.1, and 2.5.0 fixed standard-error
  bugs that 2.3.0 had. All 89 R modules reproduce their goldens at 1e-9
  under the new versions.
- **Effect on the paper.** Four things, to fold in at the next re-anchor.
  1. The version strings. Section 8 and the architecture section name
     `did` 2.3.0 (`08-computational-details.tex`,
     `08-computational-details-compact.tex`, `02-architecture-compact.tex`,
     `main.md`, `main-zh.md`); they become 2.5.1.
  2. No Track A table cell moves. Modules 04, 38 and 80 are bit-identical.
     In module 79 the simulated p-value of the non-rejecting design goes
     from 0.99998 to 0.99997 on the R side (one draw in 100,000), so the
     Python-R gap on that row is 2e-5 instead of 1e-5. The appendix cell
     for module 79 is computed from the sixteen densities and stays at
     1.2e-15.
  3. **The Track C Callaway-Sant'Anna row must be re-timed.**
     `tests/perf/results/02_csdid_R.json` was measured with `did` 2.3.0 and
     is not touched by this change. `did` 2.5.0 made `att_gt` several times
     faster. An informal back-to-back run on a loaded machine gave 0.012,
     0.025 and 0.088 seconds under 2.5.1 against 0.047, 0.118 and 0.463
     under 2.3.0 (the archive has 0.045, 0.114 and 0.457). Against the
     archived Python timings that is roughly 1.6x, 2.3x and 3.1x, where
     Section 6 prints 5.9x at 5,000 observations and 16.1x at 125,000.
     The sentence that the gap widens with size needs rewriting as well.
     Re-time both legs with `tests/perf/run_when_idle.sh` on a quiet
     machine before quoting any number.
  4. The Section 6 remark that `did` cannot write `Inf` into an integer
     cohort column and silently drops the never-treated units is true of
     2.3.0 only. Under 2.5.1 an integer column gives the same twelve cells
     and the same estimate as a numeric one. The remark needs the version
     or the past tense.
- **Paths.**
  - `tests/r_parity/results/04_csdid_R.json`
  - `tests/r_parity/results/38_drdid_R.json`
  - `tests/r_parity/results/79_didff_R.json`
  - `tests/r_parity/results/80_contdid_R.json`

### 2026-10-07 — call traces re-recorded after sp.bayes_sur and sp.bayes_shrink

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and the 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `7188f202`: `src/statspai/__init__.py` (new
  exports).
- **Reason.** A further round of the audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,437
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the small never-treated group warning

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  04 79 in `tests/r_parity/results/_implementation_trace.json`, and
  `6c82c328` the entry of module 02 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in the first of the two.
- **Reason.** `sp.callaway_santanna` now warns when fewer than five
  never-treated units plus one per covariate serve as the comparison group,
  and `sp.cs_jackknife` warns when only its delete-one samples fall below
  that size. The edit is in `did/callaway_santanna.py`, which modules 04
  and 79 and the original-data module 02 execute.
- **Effect on the paper.** None on any estimate. The Track C
  Callaway-Sant'Anna timing was taken on earlier source;
  `scripts/trace_perf_path.py --check` already listed `02_csdid` as stale
  and this adds one more file to its list. That row has to be re-timed at
  the next re-anchor in any case (see the entry above).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after sp.stochvol and sp.bayes_arima

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and the 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `1b243a68`: `src/statspai/__init__.py` (new
  exports).
- **Reason.** A further round of the audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,439
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after the probit systems and sp.bayes_mixture

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and the 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `16214fdd`: `src/statspai/__init__.py` (new
  exports).
- **Reason.** A further round of the audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,442
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Neusser time-series pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 33 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and the 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `c3ac0134`: `src/statspai/__init__.py` (new
  exports) and, for module 33, `src/statspai/timeseries/var.py` (`sp.irf`
  gained `ci=`; the estimation in `sp.var` is untouched).
- **Reason.** The pass recorded in
  `docs/dev/2026-10-06-neusser-time-series-econometrics-review.md`. No
  estimator on a Track A or original-data path changed; module 33
  reproduces its committed result byte for byte.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,462
  registered functions across 89 submodules. The correctness fix of the
  same commit, the search in `sp.garch` for two or more lagged variances,
  is outside Track A (no Track A module fits a GARCH model).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after sp.gp_regress, sp.bart and sp.abc

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and the 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `54c3732d`: `src/statspai/__init__.py` (new
  exports).
- **Reason.** A further round of the audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,467
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call trace of module 39 re-recorded after a third start in sp.arima

- **Commits.** `3f9e2723` changed `src/statspai/timeseries/arima.py` (a
  final fit also searches from zero ARMA coefficients) and re-recorded the
  entry of Track A module 39 in
  `tests/r_parity/results/_implementation_trace.json`.
- **Reason.** An ARMA(5,2) on the data of Neusser's section 5.6 stopped
  3.1 log-likelihood points below R's `arima`
  (`docs/dev/2026-10-06-neusser-time-series-econometrics-review.md`).
- **Effect on the paper.** None. Module 39 was already at its maximum and
  reproduces its committed result byte for byte; in the trace file only
  the `exercised_sources` digest of `arima.py` and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-06 — call trace of module 33 re-recorded after the variance-decomposition bands

- **Commits.** `e3e26858` re-recorded the entry of Track A module 33 in
  `tests/r_parity/results/_implementation_trace.json`. The source that
  staled it is in `c3b02841`: `src/statspai/timeseries/var.py`
  (`VARResult.fevd` gained `ci=`; the estimation in `sp.var` is
  untouched).
- **Reason.** Second round of the pass recorded in
  `docs/dev/2026-10-06-neusser-time-series-econometrics-review.md`.
- **Effect on the paper.** None. Module 33 reproduces its committed result
  byte for byte; in the trace file only the `exercised_sources` digest of
  `var.py` and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-06 — LaLonde PSM row regenerated after the propensity fit was carried further; traces re-recorded

- **Commits.** `235598f2` changed the logit behind the matching estimators
  (`src/statspai/matching/match.py`, new `src/statspai/matching/_binary_fit.py`)
  and regenerated `tests/r_parity/results/11_psm_py.json` with it. The
  commit after it re-recorded the entries of Track A modules 03 11 13 15 24 25 26 27 43 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and the 12 modules of
  `tests/orig_parity/results/_implementation_trace.json`, refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json` and rewrote the two rendered
  tables.
- **Reason.** The pass recorded in
  `docs/dev/2026-10-06-qiu-jiaping-review.md`. The fit now runs on the
  standardised design and leaves out a redundant covariate. Module 11 has
  no redundant covariate, so its matches and its ATT are the same bytes.
  The fit is carried further (scaled gradient 2e-13, was 2e-9), which moves
  the one row that uses the fitted model's covariance.
- **Effect on the paper.** One number, in the ninth digit. `se_teffects_ai`
  of module 11 goes from 621.7932847377471 to 621.793287288746 (its
  components `c_V_c` from 74904.55949690941 to 74904.55669351933 and
  `d_V_d` from 8035.781492164042 to 8035.7818611619605). Stata's value is
  621.79324512017217. The relative gap moves from 6.4e-8 to 6.8e-8, inside
  the 1e-6 budget and inside what the note on that row attributes to
  Stata's stopping rule. `att_psm` is unchanged. The original-data modules
  04 and 04b reproduce their committed results byte for byte. The registry
  census the paper quotes moves to 1,470 registered functions across 89
  submodules.
- **Paths.**
  - `tests/r_parity/results/11_psm_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Rosenbaum pass (`sp.weighted_rank`, `sp.two_criteria_match` and eight more)

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `fe41236b`. The exercised file whose digest moved:
  `src/statspai/__init__.py` (sixteen new exports).
- **Reason.** Audit in `docs/dev/2026-10-06-rosenbaum-itos-review.md`. Ten
  new functions in new files, for matched observational designs and their
  sensitivity analysis. No estimator on a Track A or original-data path
  changed, and no existing function changed its output.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,486
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces of modules 11 and 43 re-recorded after the second round on Qiu Jiaping's textbook

- **Commits.** `b3e40777` changed two files on a Track A estimation path:
  `src/statspai/matching/match.py` (the stratification estimator takes its
  strata from a column on request) and `src/statspai/regression/heckman.py`
  (a two-step rho outside [-1, 1] is truncated; the selection equation is
  returned). The commit after it re-recorded the entries of modules 11 and
  43 in `tests/r_parity/results/_implementation_trace.json` and the 12
  modules of `tests/orig_parity/results/_implementation_trace.json`.
- **Reason.** The second round recorded in
  `docs/dev/2026-10-06-qiu-jiaping-review.md`.
- **Effect on the paper.** None. `11_psm_py.json` and `43_heckman_py.json`
  reproduce byte for byte (module 43's rho is inside the interval, so the
  truncation does not apply; module 11 does not stratify), and so do the
  original-data modules 04 and 04b. In both trace files only
  `exercised_sources` digests and `seconds` differ. No function was added,
  so the registry census does not move with this commit.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the third round of the Neusser pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 33 39 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and the 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `0f897085`: `src/statspai/__init__.py` (new
  exports); for module 33, `src/statspai/timeseries/var.py` and
  `irf_bands.py` (bands for the variance decomposition, the bias-corrected
  bootstrap); for module 39, `src/statspai/timeseries/arima.py` (a final
  fit also starts from the estimates of the nested models).
- **Reason.** Third round of the pass recorded in
  `docs/dev/2026-10-06-neusser-time-series-econometrics-review.md`.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. Modules 33 and 39 reproduce their committed results byte for
  byte: `sp.var`'s estimation is untouched, and module 39's model was
  already at its maximum. In both trace files only `exercised_sources`
  digests and `seconds` differ. The registry census the paper quotes moves
  to 1,490 registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the fourth round of the Neusser pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and the 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `ca468f13`: `src/statspai/__init__.py` (new
  exports).
- **Reason.** Fourth round of the pass recorded in
  `docs/dev/2026-10-06-neusser-time-series-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,494
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call trace of module 33 re-recorded after the bootstrap default

- **Commits.** `e3e26858` re-recorded the entry of Track A module 33 in
  `tests/r_parity/results/_implementation_trace.json`. The source that
  staled it is in `8813adf9`: `src/statspai/timeseries/var.py` and
  `irf_bands.py` (the default of `boot=` in `sp.irf` and
  `VARResult.fevd`; the estimation in `sp.var` is untouched).
- **Reason.** A decision delegated by Bryce, recorded in
  `docs/dev/2026-10-06-neusser-time-series-econometrics-review.md`.
- **Effect on the paper.** None. Module 33 compares point estimates and
  reproduces its committed result byte for byte; in the trace file only
  `exercised_sources` digests and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after the `sp.tmle` estimands and `sp.aipw(propensity=)`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03,
  13, 15, 24, 25, 26, 27, 53, 65, 66 and 72 in
  `tests/r_parity/results/_implementation_trace.json` and of original-data
  modules 07 and 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The sources that
  staled them are in the same commit: `src/statspai/tmle/tmle.py` (module
  72), `src/statspai/inference/aipw.py` (original-data module 07) and
  `src/statspai/__init__.py` (one new import, every other module listed).
- **Reason.** Textbook audit recorded in
  `docs/dev/2026-10-07-schuler-vanderlaan-review.md`: `sp.tmle` validates
  `estimand` and gains ATC, EY1, EY0, RR and OR; `sp.aipw` accepts a known
  propensity; `sp.causal_gap` is new.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. Module 72 (the ATE under `fluctuation='per_arm'`) and
  original-data module 07 were rerun and reproduce their committed results
  (last-digit jitter of 1e-13 in module 07, restored); in both trace files
  only `exercised_sources` digests and `seconds` differ. The registry
  census the paper quotes moves to 1,495 registered functions across 89
  submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after the design-of-experiments subpackage

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of module 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `f110d51e`: `src/statspai/__init__.py` (the lazy
  exports of the new `statspai.doe`).
- **Reason.** The pass recorded in
  `docs/dev/2026-10-07-joseph-experimental-design-review.md`. No
  estimator on a Track A or original-data path changed. The one existing
  function whose numbers can change, `sp.gp_regress`, is on neither path.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,517
  registered functions across 90 submodules.

### 2026-10-07 — call traces re-recorded after the pass over Wager's textbook

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and of module 08
  in `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is `src/statspai/__init__.py` (new exports), changed in
  `d8703987` and, to remove a duplicated block of names, in `a579b13a`
  itself.
- **Reason.** The pass recorded in
  `docs/dev/2026-10-07-wager-causal-inference-review.md`. No estimator on
  a Track A or original-data path changed. The functions that did change
  (`sp.network_exposure`, `sp.iv.mte`, the default final stage of the R-
  and DR-learners) are not on either ledger.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. Each of the eleven modules was rerun before the re-record and
  reproduces its committed result byte for byte; in both trace files only
  `exercised_sources` digests and `seconds` differ. The registry census
  the paper quotes moves to 1,525 registered functions across 90
  submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after `sp.rd_optimized`

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and of module 08
  in `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `90c0eb8f`: `src/statspai/__init__.py` (one new
  export).
- **Reason.** Second round of the pass recorded in
  `docs/dev/2026-10-07-wager-causal-inference-review.md`. The new
  estimator is in its own file, `src/statspai/rd/optimized.py`; no
  estimator on a Track A or original-data path changed, including
  `sp.rdrobust` and `sp.rd_honest`.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. Each of the eleven modules was rerun before the re-record and
  reproduces its committed result byte for byte; in both trace files only
  `exercised_sources` digests and `seconds` differ. The registry census
  the paper quotes moves to 1,552 registered functions across 90
  submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after `sp.tmle(se_method=)` and `sp.ipw(propensity=)`

- **Commits.** `e3e26858` re-recorded the entry of Track A module 72 in
  `tests/r_parity/results/_implementation_trace.json` and of original-data
  modules 06, 10 and 11 in
  `tests/orig_parity/results/_implementation_trace.json`. The sources that
  staled them are in the same commit: `src/statspai/tmle/tmle.py` (a
  bootstrap option and observation weights for the ATT / ATC; the ATE path
  is untouched) and `src/statspai/inference/ipw.py` (a known-propensity
  argument; the default path is untouched).
- **Reason.** Second round of the audit recorded in
  `docs/dev/2026-10-07-schuler-vanderlaan-review.md`.
- **Effect on the paper.** None. The four modules were rerun and reproduce
  their committed results (last-digit jitter of 1e-13 in original-data
  module 10, restored); in both trace files only `exercised_sources`
  digests and `seconds` differ.

### 2026-10-07 — call traces re-recorded after the second round on design of experiments

- **Commits.** `e3e26858` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of module 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in the same commit: `src/statspai/__init__.py` (the lazy
  export of `sp.factor_importance`).
- **Reason.** Second round of the pass recorded in
  `docs/dev/2026-10-07-joseph-experimental-design-review.md`. No
  estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,554
  registered functions across 90 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after the third round of the Wager pass

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and of module 08
  in `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `7510a33d`: `src/statspai/__init__.py` (new
  exports).
- **Reason.** Third round of the pass recorded in
  `docs/dev/2026-10-07-wager-causal-inference-review.md`
  (`sp.mdp_policy_value`, `sp.marginal_policy_effect`,
  `sp.contextual_bandit` and options on four existing functions). No
  estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. Each of the eleven modules was rerun before the re-record and
  reproduces its committed result byte for byte; in both trace files only
  `exercised_sources` digests and `seconds` differ. The registry census
  the paper quotes moves to 1,559 registered functions across 90
  submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after `sp.aipw(outcome_model=)` and the CV-TMLE bootstrap

- **Commits.** `e3e26858` re-recorded the entry of Track A module 72 in
  `tests/r_parity/results/_implementation_trace.json` and of original-data
  module 07 in `tests/orig_parity/results/_implementation_trace.json`.
  The sources that staled them are in the same commit:
  `src/statspai/tmle/tmle.py` (the bootstrap is no longer refused with
  `fold_indices`; no estimation path changes) and
  `src/statspai/inference/aipw.py` (logit and Poisson outcome models; the
  default linear path is untouched).
- **Reason.** Third round of the audit recorded in
  `docs/dev/2026-10-07-schuler-vanderlaan-review.md`.
- **Effect on the paper.** None. Both modules were rerun and reproduce
  their committed results; in both trace files only `exercised_sources`
  digests and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces of modules 13 and 24 re-recorded after a docstring fix

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 13 and 24 in
  `tests/r_parity/results/_implementation_trace.json`. The source that
  staled them is in the same commit: `src/statspai/forest/tuning.py`, a
  docstring in which two citation keys were separated by a space and are
  now separated by a comma.
- **Reason.** The strict documentation build on main was failing on
  adjacent citation keys in six files; only this one is on a Track A
  path.
- **Effect on the paper.** None. No code changed. Both modules were rerun
  and reproduce their committed results byte for byte; in the trace file
  only `exercised_sources` digests and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-07 — call trace of original-data module 07 re-recorded after the probit outcome model

- **Commits.** `e3e26858` re-recorded the entry of original-data module
  07 in `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled it is in the same commit: `src/statspai/inference/aipw.py`
  (a probit outcome model; the default linear path is untouched).
- **Reason.** Fourth round of the audit recorded in
  `docs/dev/2026-10-07-schuler-vanderlaan-review.md`.
- **Effect on the paper.** None. The module was rerun and reproduces its
  committed result; in the trace file only `exercised_sources` digests and
  `seconds` differ.
- **Paths.**
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — call traces re-recorded after `sp.ctmle` was exported

- **Commits.** `e3e26858` re-recorded the entries of Track A modules 03, 13,
  15, 24, 25, 26, 27, 53, 65, 66 and 72 in
  `tests/r_parity/results/_implementation_trace.json` and of
  original-data module 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them are in the same commit: `src/statspai/__init__.py`
  (one new import) and `src/statspai/tmle/__init__.py` (one new import;
  the existing ones reordered by isort).
- **Reason.** `sp.ctmle`, recorded in
  `docs/dev/2026-10-07-schuler-vanderlaan-review.md`.
- **Effect on the paper.** None. No estimation path changed; in both
  trace files only `exercised_sources` digests and `seconds` differ. The
  registry census the paper quotes moves to 1,562 registered functions
  across 90 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-07 — Track C re-timed on an idle machine, R reference did 2.5.1

- **Commits.** `6d9958f0` replaces every Track C timing and re-records
  `_timed_path.json`. Run id `20261007T170004`, measured on tree
  `53927dd3` (StatsPAI 1.38.0 plus later commits) with
  `tests/perf/run_when_idle.sh`: each step started at a 1-minute load
  below 1.5 and none peaked above 1.4.
- **Reason.** The R reference moved to `did` 2.5.1 (entry of 2026-10-06),
  whose `att_gt` is several times faster, and `trace_perf_path.py --check`
  had been listing modules 01 to 04 as stale against main.
- **Effect on the paper.** Section 6 and `track_c_perf.tex`. Seconds are
  medians, ratio is reference over StatsPAI.
  1. **Callaway-Sant'Anna.** R leg 0.045 / 0.114 / 0.457 to
     0.012 / 0.023 / 0.080; StatsPAI leg 0.0076 / 0.0110 / 0.0283 to
     0.0080 / 0.0117 / 0.0305. Ratio 5.91 / 10.40 / 16.15 to
     1.49 / 1.96 / 2.62. The text prints 5.9x at 5,000 observations and
     16.1x at 125,000 (0.028 against 0.457 seconds); both go. The gap
     still widens with size, by much less.
  2. **Classical SCM.** StatsPAI leg 3.08 / 91.06 / 264.02 to
     0.69 / 1.17 / 1.27; `Synth` 0.713 / 1.301 / 4.579 to
     0.693 / 1.283 / 4.358. Ratio 0.23 / 0.01 / 0.02 to
     1.01 / 1.09 / 3.42. The text says `Synth` is 4.3x faster at 20
     donors and 57.7x at 100 (4.6 against 264 seconds) and attributes
     99% of the time to the nested V search; none of that holds now. The
     two sides still solve the same task and their estimates agree to
     5e-5 or better. The cause was isolated afterwards by timing the
     50-donor task at each candidate commit (99.9 s at `v1.34.2`). It is
     the inner weight problem of the nested fit. `95d87260` solves it
     exactly when its minimiser is certified unique, where SLSQP iterated
     before (93.3 to 4.5 s), and `54ad1443` returns the least-norm weights
     exactly when it is not unique (4.5 to 1.6 s). The estimate is
     0.182686 at every commit from `v1.34.2` to main, so the search does
     the same work and reaches the same answer. The paragraph can be
     rewritten on that basis: the nested V search is still six starts,
     and its inner step is now a direct solve. The package-default timings in
     the second SCM table fall as well (7.465 to 0.043 seconds at 100
     donors with placebos).
  3. **HDFE.** Unchanged: ratio 1.80 / 0.85 / 0.62 to 1.81 / 0.84 / 0.62.
  4. **DML.** Unchanged: ratio 1.25 / 1.15 / 1.10 to 1.20 / 1.12 / 1.08.
- **Still stale.** After rebasing onto main, `trace_perf_path.py --check`
  lists `01_hdfe` because `src/statspai/__init__.py` changed (a function
  was added and 1.39.0 was cut after the measured tree). The file is on
  the timed path of that module only through the import. Modules 02 to
  04 are current.
- **Paths.**
  - `tests/perf/results/01_hdfe_R.json`
  - `tests/perf/results/01_hdfe_py.json`
  - `tests/perf/results/02_csdid_R.json`
  - `tests/perf/results/02_csdid_py.json`
  - `tests/perf/results/03_scm_R.json`
  - `tests/perf/results/03_scm_py.json`
  - `tests/perf/results/04_dml_doubleml_py.json`
  - `tests/perf/results/04_dml_py.json`
  - `tests/perf/results/_timed_path.json`
  - `tests/perf/results/perf_table.md`
  - `tests/perf/results/perf_table.tex`
  - `tests/perf/figures/track_c_loglog.pdf`
  - `tests/perf/figures/track_c_loglog.png`

### 2026-10-15 — matching SE under ties; option-parity round

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** This entry's commit (the matching change) and the
  option-parity merges up to `54996e2e`, which re-recorded call traces
  only.
- **Reason.** With `ties='all'` the Abadie-Imbens standard error took the
  same-arm neighbours in row order, so it moved when rows were shuffled.
  It now keeps every tied neighbour, which is Stata `psmatch2 , ai(1)
  ties` to the last digit (`tests/reference_parity/test_psm_ties_se_parity.py`).
- **Effect on the paper.** Standard errors only; no estimate moves.
  Track A module 11 (`se_abadie_imbens`, a side-specific row that does not
  join) goes from 643.35 to 640.41. Original-data module 04 (`psm_att`)
  goes from 1181.66 to 1178.21 and 04b from 981.65 to 978.13; the
  original-data table prints estimates only, so it does not change.
- **Paths.**
  - `tests/r_parity/results/11_psm_py.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/04_lalonde_original_py.json`
  - `tests/orig_parity/results/04b_nsw_psid_original_py.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-15 — honest_did FLCI centre at M = 0

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** This entry's commit.
- **Reason.** At `M = 0` the FLCI under `Delta^SD` is a closed form: the
  minimum-variance estimator unbiased under a linear trend through the
  reference period, plus or minus `z` standard deviations. The native
  solver ran its SLSQP program there too; it reached the half-length but
  not the centre, which on correlated covariances missed the closed form by
  up to 1.4e-3 of the half-length (HonestDiD's `BCdata_EventStudy`,
  `e = 0`; `tests/reference_parity/test_honest_did_options_R_parity.py`).
  `M = 0` now uses the closed form; `M > 0` is unchanged.
- **Effect on the paper.** Track A module 10 only, its `M = 0` rows: every
  `M = 0` bound moves by 2.19e-9 (`ci_lower_M_0` 0.301547089171849 to
  0.301547091360523, `ci_upper_M_0` 0.697024335022253 to
  0.697024337210905; the `shipped_` and `analytic_` copies of the same two
  numbers move identically; `breakdown_m_e0` moves by 1e-16). The headline
  `analytic_ci_*_M_0` gap to R's closed form falls from 2.2e-9 to 4.4e-16;
  module 10's headline maximum is still the `M > 0` rows' 1.4e-8, so the
  table entry, the PASS verdict and the 1e-6 budget are unchanged. The
  Stata column of the non-headline `ci_*_M_0` rows moves in the third digit
  (1.82e-6 to 1.83e-6, 7.89e-7 to 7.92e-7). Module 21 reproduces byte for
  byte.
- **Paths.**
  - `tests/r_parity/results/10_honest_did_py.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-17 — mixed-model observed information by Richardson extrapolation

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `506099db`.
- **Reason.** `multilevel/glmm.py::_numerical_oim_cov` inverted a single
  central-difference Hessian at h = 1e-4. Its rounding error,
  about `eps * |loglik| / h^2`, differs between platforms' libm: the melogit
  ICC standard error was 7e-7 from Stata `estat icc` on macOS and 1.3e-6
  (over the 1e-6 budget) on the new Windows CI gate. The Hessian is now the
  Richardson extrapolation `(4 H(h) - H(2h)) / 3` at h = 1e-3; the same ICC
  standard error agrees with Stata to 6e-10.
- **Effect on the paper.** Track A modules 26 and 27, standard errors only
  (estimates byte-identical). Module 26 (Laplace): `beta_intercept` se
  0.118062107501 to 0.118062108761, `beta_x1` 0.0636113492259 to
  0.0636113501831; the gap to R stays 1.21e-3 / 2.65e-3 (the registered
  Laplace convention difference). Module 27 (AGHQ): `beta_intercept` se
  0.118612132299 to 0.118612137396 (gap to lme4 3.92e-7 to 4.35e-7),
  `beta_x1` 0.0636049074172 to 0.063604907118 (2.09e-7 to 2.04e-7). Every
  row stays inside its budget; no verdict changes. Module 25 reproduces.
- **Paths.**
  - `tests/r_parity/results/26_glmm_logit_py.json`
  - `tests/r_parity/results/27_glmm_aghq_py.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-17 — fixture lock rehashed after a sort-key change

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `1d2e7d4b`.
- **Reason.** `scripts/tier_a_fixture_lock.py` sorted `Path` objects, which
  compare case-insensitively on Windows; it now sorts by POSIX string. The
  lock hashes its own script, so the lock was rewritten.
- **Effect on the paper.** None: no golden or result file changed, and the
  file order (hence the aggregate hash) is unchanged on Linux and macOS.
- **Paths.**
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`

### 2026-10-19 — trace re-record after the BJS covariance and ARP speed-up

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `2c111cc2`.
- **Reason.** `did/did_imputation.py` and `did/_bjs_pretrends.py` now build
  the lead-horizon cross-covariance and drop reghdfe singletons from the
  lead dof count; `did/_arp.py` enumerates dual vertices in one batch. All
  sit on the estimation paths of modules 10, 16, 21 and 84, so their traces
  were re-recorded.
- **Effect on the paper.** None. Modules 10, 16 and 21 reproduce their
  committed results exactly; module 84 differs in the last digit of the
  lead SEs (summation order of the meat, ~3e-16 relative) and the
  committed file was kept. Module 84's fixture has no singleton units, so
  the dof change does not apply to it.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record after the output-layer result fixes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `6881f126` (rebased from `466574f9`).
- **Reason.** `core/results.py` (`CausalResult.summary()` without a stored
  pre-trend test, `to_word` default precision, strict `plot(type=)`,
  `booktabs=`), `_result_serialize.py` (`tidy` / `glance` on the result
  protocol), `did/bacon.py` and `diagnostics/evalue.py` (results are now
  dict subclasses with the same keys) and `survival/models.py` (`plot`
  validates its type) sit on the estimation paths of 76 Track A modules and
  all 12 original-data modules, so both traces were re-recorded.
- **Effect on the paper.** None. Before re-recording, every affected module
  was re-run into a scratch directory: Track A 76 / 76 reproduce
  (`verify_reproduce_py.py`, fixtures byte-identical or embedded, worst
  relative difference 1.3e-10, inside the 1e-9 reproducibility floor);
  original data 12 / 12 reproduce (worst relative
  difference 1.6e-11, module 01). No result file changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record after the native Honest DiD restriction families

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `fa8d6630`.
- **Reason.** `did/honest_did.py` (new `bias_direction` /
  `monotonicity_direction` / `smoothness_relative_magnitude` dispatch) and
  `did/_arp.py` (conditional-FLCI hybrid first-stage edge rule) sit on the
  estimation paths of modules 10 and 21; their traces were re-recorded.
- **Effect on the paper.** None. Modules 10 and 21 reproduce their
  committed results exactly (the FLCI and the relative-magnitudes
  Conditional set do not touch the changed branches).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record after the in-place provenance hash

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `7a9f6769`.
- **Reason.** `output/_lineage.py` (hash the input frame's buffer instead
  of a `tobytes()` copy, same digest; `STATSPAI_PROVENANCE_HASH=0`
  opt-out) is executed by 43 Track A modules and 11 original-data
  modules; their traces were re-recorded. Only that file's hash changed
  in either ledger.
- **Effect on the paper.** None. The fingerprint is not an estimate and
  its value is unchanged (tested against the `tobytes()` digest), so no
  result file or table number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record for module 18 (augsynth extensions)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `3db1686c`.
- **Reason.** `synth/augsynth.py` (several treated units, `progfunc`,
  `scm`, `fixedeff`, covariates; the R bridge moved to
  `synth/_augsynth_r.py`) is on the estimation path of module 18; its trace
  was re-recorded.
- **Effect on the paper.** None. The default single-unit Ridge path is
  arithmetically unchanged and module 18 reproduces its committed result
  byte for byte.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record for module 12 (sdid block interface)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `89d36791`.
- **Reason.** `synth/sdid.py` (the `treated_unit=` interface hands
  covariates and `se_method='noinference'` to the `treat=` path) is on
  the estimation path of module 12; its trace was re-recorded.
- **Effect on the paper.** None. Module 12 (no covariates, placebo SE)
  does not reach the new branch and reproduces its committed result byte
  for byte.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record for the design guardrails

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `59ae2b20`.
- **Reason.** `regression/ols.py`, `fixest/wrapper.py`,
  `panel/panel_reg.py` and `core/_agent_summary.py` now call (or read)
  the shared design guardrails (`core/_design_guardrails.py`,
  `core/_spec_counter.py`), which run after the fit and write only
  diagnostics. The traces of 18 Track A modules (01 02 03 14 15 24 35 42
  48 51 53 54 55 56 57 58 67 69) and 4 original-data modules (01 04 04b
  09) were re-recorded.
- **Effect on the paper.** None. Every re-traced module was first re-run
  into a scratch directory and reproduces its committed `*_py.json`
  (all numbers within 1e-9, in fact identical); the guardrails never
  change an estimate or standard error.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record for module 86 (fect diagnostic tests)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `c8395801`.
- **Reason.** `synth/fect.py` (placebo / carryover / leave-one-period-out
  tests, the resampling loop moved to `synth/_fect_tests.py`) is on the
  estimation path of module 86; its trace was re-recorded.
- **Effect on the paper.** None. Module 86 is point-only and reproduces
  its committed result byte for byte; the new options are off by default.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for LDV margins and sp.heckprobit

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `bd370d02`, `afa268a2`.
- **Reason.** `regression/tobit.py` and `regression/heckman.py` (results
  now keep their ML parameters and covariance for `sp.margins`) are on the
  estimation paths of modules 41 and 43; the export of `sp.heckprobit` in
  `statspai/__init__.py` touches modules 03, 13, 15, 24-27, 53, 65, 66 and
  original-data module 08. Their traces were re-recorded.
- **Effect on the paper.** None. No estimate changes; the traced modules
  reproduce their committed results.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record for the IV / state / panel local projections

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `612557ee`.
- **Reason.** `timeseries/local_projections.py` (new arguments routed
  through `timeseries/_lp_dispatch.py`) and `timeseries/var.py`
  (`granger_causality(id=, time=)`) are on the estimation paths of
  modules 34 and 33; their traces were re-recorded.
- **Effect on the paper.** None. Both modules were re-run first and
  reproduce their committed `*_py.json` byte for byte; the new arguments
  are not reached by either module.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of sp.intreg

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `8081690a`.
- **Reason.** Exporting `sp.intreg` from `statspai/__init__.py` touches the
  traced paths of modules 03, 13, 15, 24-27, 53, 65, 66 and original-data
  module 08; their traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of sp.xtivreg

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `226d6394`.
- **Reason.** Registering `sp.xtivreg` in `statspai/__init__.py` touches the
  traced paths of modules 03, 13, 15, 24-27, 53, 65, 66 and original-data
  module 08; their traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**

### 2026-10-19 — 14_ols_cluster re-run, trace re-record for the HDFE batch

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `3475b888` (artifacts), following `79b0e100` (code);
  `1f8896f9` and `4dd0d132` re-record the same traces after rebasing
  onto newer main (`5c015c19`).
- **Reason.** `core/_numba_kernels.cluster_meat` now sums the cluster
  scores in one `bincount` pass instead of sorting the rows by cluster,
  and `inference/jackknife.two_way_vcov_ols` uses the same one-pass meat;
  `fast/feols.py` and `panel/hdfe.py` finish unconverged projections with
  a sparse direct solve (not triggered by any Track A module);
  `fixest/_native.py` / `fixest/wrapper.py` gained the negative binomial
  and the bias correction (not on any Track A row). Traces re-recorded:
  Track A 01 03 14 15 47 51 53 54 55 56 67 68 75; original data 01 04 04b 09.
- **Effect on the paper.** Every re-traced module was first re-run into a
  scratch directory. All reproduce their committed `*_py.json` to
  <= 1.6e-11 except `14_ols_cluster`, whose clustered standard errors
  moved by up to 1.1e-9 relative (the meat's summation order on a design
  with `year` in the thousands): intercept SE 5.065400492315133 ->
  5.0654004970504545, `treat` 0.012220360961010073 -> 0.012220360960616895,
  `year` 0.0025286050132379972 -> 0.002528605016004209 (R:
  5.06540050907329, 0.0122203609644141, 0.00252860502164486). Rel. errors
  against R stay below 3.4e-9, far inside the 1e-6 budget; the parity
  tables change only in those digits.
- **Paths.**
  - `tests/r_parity/results/14_ols_cluster_py.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of sp.xtpcse

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `63726d7d`.
- **Reason.** Registering `sp.xtpcse` in `statspai/__init__.py` touches the
### 2026-10-19 — trace re-records for the export of the data layer

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `157d0330`.
- **Reason.** Exporting `sp.xtset` / `sp.lag` / `sp.lead` / `sp.diff` /
  `sp.reshape` / `sp.collapse` from `statspai/__init__.py` touches the
  traced paths of modules 03, 13, 15, 24-27, 53, 65, 66 and original-data
  module 08; their traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for `design=` on the estimators

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `44aaf1d4` (on top of `11566d92`).
- **Reason.** `11566d92` added the `design=SurveyDesign` branch to
  `inference/ipw.py`, `inference/aipw.py`, `regression/ols.py`,
  `regression/logit_probit.py`, `tmle/`, `dml/`, `did/` (incl.
  `did/_core.py::influence_se_did`, `aggte.py`, `_flci.py`),
  `survival/models.py` and `msm/msm.py`. These files are on the estimation
  paths of Track A modules 01, 04, 05, 08, 10, 11, 14, 16, 17, 20, 21, 24,
  36, 48, 51, 53-57, 70-73, 75, 78, 79, 83-85 and original-data modules
  01, 02, 04, 04b, 06, 07, 09, 10, 11; their traces were re-recorded.
- **Effect on the paper.** None. Every one of those modules was re-run
  first and reproduces its committed `*_py.json` (relative 1e-9; the
  `design=None` paths are unchanged).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the panel wild-bootstrap fix

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `419a56bf9`, `3eedaaa3f`.
- **Reason.** `panel/panel_reg.py` (the wild bootstrap of `sp.panel(method=
  'fe', vce='wild')` now re-absorbs entity effects that are not nested in
  the clusters) is on the estimation paths of modules 35 and 69; their
  traces were re-recorded.
- **Effect on the paper.** None. Neither module uses `vce='wild'`; both
  reproduce their committed results.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
### 2026-10-19 — trace re-records for Stata syntax in formulas

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `70739f451`.
- **Reason.** `core/utils.py` (`r_formula_idioms`, `create_design_matrices`)
  now reads Stata factor-variable and time-series terms; it is on the
  traced path of Track A modules 01, 02, 14, 37, 42, 44-49, 51, 53-58, 63,
  64, 67 and original-data modules 01, 04, 04b, 09. The modules were re-run
  and their traces re-recorded.
- **Effect on the paper.** None; no estimate changes (formulas without
  Stata syntax take the same path).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for `finegray(weights=)` and `sar(weights=)`

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `1aaf5833` (on top of `ce35ca48`).
- **Reason.** `ce35ca48` added `weights=` to `survival/competing_risks.py`
  (Fine-Gray) and a typed refusal of `weights=` to
  `spatial/models/ml.py::sar`. These files are on the estimation paths of
  Track A modules 24, 65 and 66; their traces were re-recorded.
- **Effect on the paper.** None. The three modules were re-run first and
  reproduce their committed `*_py.json`; `weights=None` paths are
  unchanged.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of sp.mediate_paths

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `7bdfb8e7`.
- **Reason.** Exporting `sp.mediate_paths` from `statspai/__init__.py` touches
  the traced paths of modules 03 13 15 24 25 26 27 53 65 66 and original-data module 08; their
  traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of sp.weightit / sp.bal_tab

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `b18df32e`.
- **Reason.** Exporting `sp.weightit` / `sp.bal_tab` (and their result
  classes) from `statspai/__init__.py` touches the traced paths of modules
  03, 13, 15, 24-27, 53, 65, 66 and original-data module 08; their traces
  were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the xttobit / xtintreg / gologit exports

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `683ad352e`, `fe2bfe635`.
- **Reason.** Registering `sp.xttobit`, `sp.xtintreg` and `sp.gologit` in
  `statspai/__init__.py` touches the traced paths of modules 03, 13, 15,
  24-27, 53, 65, 66 and original-data module 08; their traces were
  re-recorded after the rebase.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records after sp.match(sampling_weights=)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `3ce03fe5`.
- **Reason.** Adding the `sampling_weights=` branch to
  `matching/match.py` touches the traced path of Track A module 11 and
  original-data modules 04 / 04b. The default path is unchanged: the three
  modules were re-run first and their committed results reproduce byte for
  byte; only the traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record after sp.mediate(weights=)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `78d1ee79`.
- **Reason.** sp.mediate(weights=) touches the traced path of Track A module(s) 36;
  the traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of the experiments tools

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `5a0aea8bc`.
- **Reason.** Exporting `sp.group_sequential` / `sp.msprt` /
  `sp.msprt_lambda` / `sp.cuped` (and their result classes) from
  `statspai/__init__.py` touches the traced paths of modules 03, 13, 15,
  24-27, 53, 65, 66 and original-data module 08; their traces were
  re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records after two-way DML and multi-action policy trees

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `9a670eb9`.
- **Reason.** `dml/_base.py` (two-way `cluster=`) and
  `policy_learning/policy_tree.py` (multi-action trees) sit on the traced
  paths of Track A modules 08, 70 and 71. Their one-way / binary paths are
  unchanged: the three modules were re-run first and their committed
  results reproduce byte for byte; only the traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of sp.bart_causal

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `2618f5cf`.
- **Reason.** Exporting `sp.bart_causal` / `BayesCausalResult` from
  `statspai/__init__.py` touches the traced paths of modules 03, 13, 15,
  24-27, 53, 65, 66 and original-data module 08; their traces were
  re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record after the sp.ida / sp.sdem / sp.gns exports

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `468ffc0d`.
- **Reason.** the sp.ida / sp.sdem / sp.gns exports touches the traced path of Track A module(s) 03 13 15 24 25 26 27 53 65 66;
  the traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-record after sp.mediate(design=)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `9179c25d`.
- **Reason.** sp.mediate(design=) touches the traced path of Track A module(s) 36;
  the traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — trace re-records for the export of sp.lmtp

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `85f3a0a4`.
- **Reason.** Exporting `sp.lmtp` from `statspai/__init__.py` touches
  the traced paths of modules 03, 13, 15, 24-27, 53, 65, 66 and
  original-data module 08; their traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-19 — original-data trace re-record after sp.aipw learners

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `2d6e2805`.
- **Reason.** `inference/aipw.py` (new `ps_learner=` /
  `outcome_learner=` arguments; the default path is unchanged) is on the
  traced path of original-data module 07. The module was re-run first:
  its committed results reproduce to the last digits (1e-13 relative,
  inside the 1e-9 reproducibility tolerance; the file was restored); only
  the trace was re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — trace re-records after CEM k2k and sp.l1_imbalance

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `e9c4827f`.
- **Reason.** The `k2k=` branch in `matching/match.py` and the export of
  `sp.l1_imbalance` from `statspai/__init__.py` touch the traced paths
  of Track A modules 03, 11, 13, 15, 24-27, 53, 65, 66 and original-data
  modules 04, 04b and 08. Modules 11, 04 and 04b were re-run first and
  their committed results reproduce byte for byte; only the traces were
  re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — module 72 result key renamed; trace and fixture lock

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `ba964a6b9`, `c3ed2441a`.
- **Reason.** The committed `72_tmle_py.json` named a metadata key
  `epsilon` while `72_tmle.py` writes `epsilon_vec`, so the committed file
  was not exactly reproducible from the script. The module was re-run on
  the no-Rust path: every value is identical (maximum relative difference
  0); only the key name changed. The trace and the Tier A fixture lock were
  re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/72_tmle_py.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`

### 2026-10-20 — traces re-recorded after the full-suite fixes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `2dab3646a`.
- **Reason.** The fixest fixed-effects part is no longer run through the
  Stata factor-variable translation (`core/utils.py`,
  `core/_stata_formula.py`, `fixest/wrapper.py`), and
  `sp.local_projections` records its call for `sp.vcov`
  (`timeseries/local_projections.py`). These files are on the traced
  paths of Track A modules 01, 02, 14, 34, 37, 42, 44-49, 51, 53-58, 63,
  64, 67 and original-data modules 01, 04, 04b, 09. Every one was re-run
  on the no-Rust path first and reproduces its committed result (Track A
  through `verify_reproduce_py.py`, worst relative difference 0;
  original data, worst 2.1e-12); only the traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — Track C re-timed on the Rust HDFE path

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `e6ecbde28`.
- **Reason.** All four timed paths were stale (`fast/`, `did/`,
  `core/results.py`, `output/_lineage.py`, `dml/` and the Rust crate had
  changed). Re-measured on 7568499a1 with `tests/perf/run_when_idle.sh`
  (peak load at most 1.47, no step repeated, one thread on both sides).
  statspai-hdfe 0.9.0 is a core dependency on wheel platforms, so the
  HDFE row now times the Rust kernels the default install runs;
  `_timed_path.json` records `rust_loaded`.
- **Effect on the paper.** Track C table (`track_c_perf.tex`) and the
  log-log figure. 01_hdfe, sp seconds (ratio to `fixest::feols`): n=1e4
  0.0022 -> 0.0007 (1.82 -> 7.08), 1e5 0.0130 -> 0.0035 (0.85 -> 3.18),
  1e6 0.1174 -> 0.0315 (0.61 -> 2.41); agreement unchanged (<= 2e-14).
  02_csdid ratios 1.38 / 2.09 / 2.60 -> 1.33 / 1.78 / 2.61. 03_scm ratios
  1.04 / 1.07 / 3.49 -> 1.01 / 1.04 / 3.59. 04_dml ratios 1.17 / 1.10 /
  1.07 -> 1.17 / 1.15 / 1.09. Any text saying StatsPAI is slower than
  fixest at large n no longer holds.
- **Paths.**
  - `tests/perf/results/01_hdfe_py.json`
  - `tests/perf/results/01_hdfe_R.json`
  - `tests/perf/results/02_csdid_py.json`
  - `tests/perf/results/02_csdid_R.json`
  - `tests/perf/results/03_scm_py.json`
  - `tests/perf/results/03_scm_R.json`
  - `tests/perf/results/04_dml_py.json`
  - `tests/perf/results/04_dml_doubleml_py.json`
  - `tests/perf/results/_timed_path.json`
  - `tests/perf/results/perf_table.md`
  - `tests/perf/results/perf_table.tex`
  - `tests/perf/figures/track_c_loglog.pdf`
  - `tests/perf/figures/track_c_loglog.png`

### 2026-10-20 — module 24 trace re-recorded after weighted ph_test

- **Commits.** `da89941d0` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (module 24).
- **Reason.** `CoxResult.ph_test` now runs the `cox.zph` score test on
  weighted and counting-process fits (`survival/models.py`,
  `survival/_cox_core.py`), and module 24 executes both files on its
  estimation path (the fit stores the proportional-hazards summary).
- **Effect on the paper.** None. Module 24 was re-run and its
  `24_coxph_py.json` differs from the committed one only in the last digit
  (about 1e-16, within the 1e-9 reproduction tolerance), so the committed
  result was kept; only the source hashes in the trace changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after sp.match(design=)

- **Commits.** `8e46917bb` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (module 11) and
  `tests/orig_parity/results/_implementation_trace.json` (04, 04b).
- **Reason.** `matching/match.py` gained the `design=` branch (the
  survey-design path lives in `matching/_sampling_weights.py`, which these
  modules do not execute); the three modules run `match.py`.
- **Effect on the paper.** None. Modules 11, 04 and 04b were re-run and
  their committed results are byte-identical; only the source hashes in the
  traces changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — trace 33 re-recorded after var.py docstring references

- **Commits.** `69dc2cfd9` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (module 33).
- **Reason.** `sp.granger_causality`'s docstring in `timeseries/var.py`
  gained a References section (Granger 1969, Dumitrescu-Hurlin 2012); no
  code changed.
- **Effect on the paper.** None. Module 33 was re-run and its committed
  result is byte-identical; only the source hash in the trace changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — original-data traces 06 / 10 / 11 after sp.ipw learners

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `f46ab3e16`.
- **Reason.** `inference/ipw.py` (new `ps_learner=` / `cross_fit=` /
  `n_folds=` arguments; the default path is unchanged) is on the traced
  path of original-data modules 06, 10 and 11. The modules were re-run
  first and their committed results reproduce exactly; only the traces
  were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the sp.egen export

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `f7882ad4c` (first written as `4b6310a70`, its sha before a rebase).
- **Reason.** `src/statspai/__init__.py` gained the `sp.egen` export; it is
  on the traced path of Track A modules 03, 13, 15, 24, 25, 26, 27, 53, 65,
  66 and original-data module 08. `verify_reproduce_py.py` re-ran the ten
  Track A modules first: fixtures byte-identical, results within 1.3e-10
  of the committed ones. Only the source hash in the traces changed.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the sp.ltmle_msm export

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `697addb60`.
- **Reason.** Exporting `sp.ltmle_msm` from `statspai/__init__.py`
  touches the traced paths of Track A modules 03, 13, 15, 24-27, 53, 65,
  66 and original-data module 08; no estimation code on those paths
  changed, so only the traces were re-recorded.
- **Effect on the paper.** None; no estimate changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after replicate-weight designs

- **Commits.** `57fee3046` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 01, 03, 04,
  10, 11, 13, 14, 15, 21, 24, 25, 26, 27, 48, 51, 53, 54, 55, 56, 57, 65,
  66, 72, 79) and `tests/orig_parity/results/_implementation_trace.json`
  (01, 02, 04, 04b, 06, 07, 08, 09, 10, 11).
- **Reason.** `design=` accepts replicate-weight designs: the regression,
  IPW / AIPW / TMLE, Cox, DiD and `survey/` sources these modules execute
  gained a guarded first statement (`design` given and a replicate design)
  or a refusal in `check_design`; the replicate machinery lives in new
  files these modules do not execute.
- **Effect on the paper.** None. Every affected module was re-run before
  re-recording; the committed results differ from the re-runs by at most
  3.3e-10 relative (module 25; all others <= 1.6e-11), inside the 1e-9
  reproduction tolerance, so they were kept. Only source hashes changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the sp.xtregar lazy export

- **Commits.** `1d6c1aa4d` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 03, 13, 15,
  24, 25, 26, 27, 53, 65, 66) and
  `tests/orig_parity/results/_implementation_trace.json` (08).
- **Reason.** `src/statspai/__init__.py` gained one `_register_lazy` line
  for `sp.xtregar`; these modules execute its lazy-attribute hook.
- **Effect on the paper.** None. The modules were re-run first; 13, 26,
  27, 53, 65, 66 and orig 08 are byte-identical, 03, 15, 24 and 25 differ
  only in the last digits (at most 4e-10 relative, inside the 1e-9
  reproducibility tolerance), so the committed results were kept and only
  the source hashes in the traces changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — honest_did split, breakdown_m restrictions, sp.xtevent_smpath

- **Commits.** `1f9873dce` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 03, 10, 13,
  15, 21, 24, 25, 26, 27, 53, 65, 66) and
  `tests/orig_parity/results/_implementation_trace.json` (08).
- **Reason.** `did/honest_did.py` and `did/_arp.py` were split by concern
  into `_honest_common.py` / `_honest_r.py` / `_honest_breakdown.py` /
  `_honest_moments.py` / `_arp_sets.py` (pure refactor), `sp.breakdown_m`
  gained the sign / shape / SDRM restrictions, and the smoothness
  test-inversion grid now recentres when it accepts nothing; modules 10 and
  21 execute those files. `src/statspai/__init__.py` gained the
  `sp.xtevent_smpath` export, executed by the other modules.
- **Effect on the paper.** None. Before re-recording, the Python side of
  modules 10, 16, 21 and 84 was re-run and writes byte-identical results to
  the pre-split tree, and `verify_reproduce_py.py` reproduces all twelve
  traced modules (fixtures byte-identical, worst relative difference
  1.3e-10, module 25). The recentred grid only changes rows that used to
  be empty (`NaN`), which no Track A module produces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — fixes from the re-probe of the 10-19/20 functions

- **Commits.** `78a696752` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 04, 05, 08,
  16, 18, 24, 36, 44, 45, 46, 49, 50, 53, 65, 66, 71, 72, 79, 84) and
  `tests/orig_parity/results/_implementation_trace.json` (02, 06, 07, 09,
  10, 11).
- **Reason.** Source files on those modules' estimation paths changed:
  input checks (`core/_validate.py`, `survival/models.py`,
  `survival/competing_risks.py`, `mediation/mediate.py`,
  `inference/jackknife.py`, `spatial/models/ml.py`), `n_obs` of
  `sun_abraham` / `did_imputation`, `augsynth(progfunc='none', scm=False)`,
  ordered-Categorical outcomes in `regression/multinomial.py`, and z
  inference for the spatial ML fits.
- **Effect on the paper.** None. Before re-recording, the Python side of
  every listed module was re-run: results are byte-identical or differ in
  the last digits (at most 1.9e-13 relative, inside the 1e-9
  reproducibility tolerance), so the committed results were kept. The
  spatial modules compare estimates and SEs, which the z change leaves as
  they were.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — orig trace 07 re-recorded after the learner helper

- **Commits.** `d439b60c7` re-recorded
  `tests/orig_parity/results/_implementation_trace.json` (module 07).
- **Reason.** `inference/aipw.py` now reads user learners through the
  shared `core/_learners.py` helper (SuperLearner as `ps_learner=`); module
  07 executes the file but not the changed function.
- **Effect on the paper.** None. Module 07 was re-run before re-recording:
  the output is identical on origin/main and on this tree, and both differ
  from the committed result by about 1e-13 relative (pre-existing, inside
  the 1e-9 reproducibility tolerance), so the committed result was kept.
- **Paths.**
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — trace 53 re-recorded after the bootcluster fix

- **Commits.** `aa7dd1063` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (module 53).
- **Reason.** `inference/jackknife.py` (`sp.wild_cluster_boot`) now hands
  `bootcluster=` to a helper that draws the weights for intersections when
  the bootstrap clusters cut across one-way error clusters, as boottest does;
  module 53 (CR2) executes the file but not that path.
- **Effect on the paper.** None. Module 53 was re-run before re-recording;
  its output is identical to the committed `53_cr2_py.json`.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — traces 08 / 13 / 24 / 71 re-recorded after the learner routing

- **Commits.** `ae7b038cf` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (modules 08, 13, 24,
  71).
- **Reason.** `dml/_base.py`, `dml/irm.py`, `dml/iivm.py`,
  `forest/_grf_fit.py` and `forest/causal_forest.py` (executed by those
  modules) now read classifier probabilities through `core/_learners.py`,
  so `sp.SuperLearner` works as a nuisance learner.
- **Effect on the paper.** None. The four modules were re-run before
  re-recording: 08, 13 and 71 reproduce their committed results byte for
  byte; 24 (Cox, which only imports the forest module) differs in the last
  digit (about 1e-15 relative, inside the 1e-9 reproducibility tolerance),
  so the committed result was kept.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — trace 10 re-record after the breakdown_m recovery hint

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `2a29aa53b`, `8a1f2d128`.
- **Reason.** `did/_honest_breakdown.py` (a recovery-hint string named a
  non-public function) is on the traced path of Track A module 10. The
  module was re-run first: its committed result reproduces byte for byte;
  only the trace was re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — traces 65 / 66 re-record after the `sp.sac` OIM fix

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `cff32842d`.
- **Reason.** `spatial/models/ml.py` (`sp.sac` rewritten), `durbin_error.py`
  and the new `_fiml.py` are imported on the traced path of Track A modules
  65 and 66. Both modules were re-run first: their committed results
  reproduce byte for byte (SAR / SEM / SDM / GMM code unchanged); only the
  trace was re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — traces 33 / 34 re-record after the clean-room local-projection rewrite

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `2746f026c`.
- **Reason.** The IV / state-dependent / panel local projections and the
  Dumitrescu-Hurlin test were rewritten clean-room (`d46afbb4a`):
  `timeseries/var.py` (module 33) and `timeseries/local_projections.py`
  changed, and `_lp_dispatch.py` on the module-34 path was replaced by
  `_lpx_route.py`. The direct and Cholesky paths that Track A exercises are
  the pre-extension code; `33_var.py` and `34_lp.py` were re-run first and
  reproduce their committed results byte for byte; only the trace was
  re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — trace 59 re-record after `sp.liml(robust='many_weak')`

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `d6b0f7e68`.
- **Reason.** `regression/advanced_iv.py` (new `robust='many_weak'`
  branch; the classical, robust and cluster paths are untouched) and the
  new `regression/_many_iv_se.py` are on the traced path of Track A module
  59. The module was re-run first: its committed result reproduces within
  the 1e-9 reproduction tolerance (last-digit differences, 1e-14), so the
  committed file was kept; only the trace was re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — trace 10 re-record after the `breakdown_m` RM search change

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `8cfa0055c`.
- **Reason.** `did/_honest_breakdown.py` (the relative-magnitude C-LF
  breakdown now scans for the first crossing) is on the traced path of
  Track A module 10, which reports only the smoothness breakdown. The
  module was re-run first: its committed result reproduces byte for byte;
  only the trace was re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the W5-defaults owner decisions

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `eda3f5b1e`.
- **Reason.** `core/_spec_counter.py` (public `sp.reset_spec_counter`),
  `did/sun_abraham.py` (`aggregation=` FutureWarning), `dml/_learners.py`
  (`'rlasso'` classifier alias), `forest/causal_forest.py` (`n_jobs`
  default) and `__init__.py` are on the traced paths of Track A modules 01
  03 05 08 13 14 15 24 25 26 27 35 51 53 54 55 56 65 66 67 71 and of the
  original-data modules 01 04 04b 08 09. Every module was re-run first: the
  committed results reproduce within 1e-9 (last-digit jitter in seven
  Track A and two original-data files, restored); only the traces were
  re-recorded. Module 05 passes `aggregation='fixest_att'` and no module
  uses the `'rlasso'` alias in a classifier role.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after new `sp` exports (mmqreg, two_stage_interference)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `94916caa5`.
- **Reason.** `src/statspai/__init__.py` gained `sp.mmqreg` (lazy) and
  `sp.two_stage_interference`; the file is on the traced path of Track A
  modules 03, 13, 15, 24, 25, 26, 27, 53, 65, 66 and original-data module
  08. The tracer re-ran the modules; their committed results are unchanged.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the DR-DiD clean-room rewrite (audit H2-H4)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `94f48be45` (rewrite), `e75853cd7` (traces).
- **Reason.** License provenance audit rows H2-H4: `did/_rcs.py`, the panel
  2x2 engines of `did/callaway_santanna.py`, `did/_core.py::weight_influence`
  and `did/aggte.py::_aggregated_influence` were re-implemented clean-room
  (new shared module `did/_dr_common.py`; spec in
  `docs/dev/cleanroom/did_drdid.md`). These files are on the traced paths of
  Track A modules 04 05 16 17 20 73 75 78 79 83 84 85 and original-data
  module 02. Every module was re-run first and reproduces its committed
  result (worst relative difference 3e-16 on 04 / 79 / orig 02, at most
  1.6e-12 elsewhere); only the traces were re-recorded.
- **Effect on the paper.** None; no number changes. Track C `02_csdid` was
  already stale before this change and remains to be re-timed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the E-value clean-room rewrite (audit H13)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `2969e6067` (rewrite), `7aea8e0e5` (traces).
- **Reason.** License provenance audit row H13: `diagnostics/evalue.py`
  (`evalue_rd` and the E-value threshold) was re-implemented clean-room
  (spec `docs/dev/cleanroom/evalue.md`). The file is on the traced paths of
  Track A module 23 and original-data module 11. Both were re-run first and
  reproduce their committed results (worst relative difference 5.7e-14 on
  23's `evalue_ci_rd`, whose value now equals R bit for bit; 0 on orig 11);
  only the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the RD clean-room rewrite

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `f7d83b89d`.
- **Reason.** Audit rows H5, H6, H7 and H12 rewritten clean-room:
  `rd/_cct_bandwidth.py` (+ new `rd/_cct_inference.py`, `rd/_cct_side.py`,
  `rd/_nn_variance.py`), `rd/_core.py` (`_spd_inverse`),
  `diagnostics/rddensity.py` (+ new `rd/_lpdensity.py`). They are on the
  traced paths of Track A modules 06, 09, 88, 89 and original-data module
  05. Every module was re-run first: the committed results reproduce within
  1.5e-10 (88), 2.1e-11 (09) and 3e-13 / 7e-12 (06, 89, orig 05), so the
  result files were restored; only the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-20 — traces re-recorded after the MEDIUM provenance clean-room rewrites

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `1792b7c81`.
- **Reason.** Comment / docstring rewording and bit-identical helper
  rewrites in `survey/`, `forest/_grf_family.py`, `forest/_grf_fit.py`,
  `diagnostics/sensemakr.py`, `decomposition/oaxaca.py`,
  `inference/jackknife.py`, `spatial/models/gmm.py`, and the clean-room
  psmatch2 kernel / llr / SE code in `matching/match.py` and
  `matching/_matched_frame.py`, are on the traced paths of Track A modules
  04 10 11 13 21 22 24 30 53 65 66 79 and original-data modules 02 04 04b.
  Every module was re-run on the rebased tree first: the committed results
  reproduce (worst 5.6e-16); only the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces re-recorded after the `reset_spec_counter(y=)` rename

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `8329bcd0a`.
- **Reason.** `core/_spec_counter.py` (argument of `sp.reset_spec_counter`
  renamed to `y=`, `outcome=` kept as an alias) is on the traced paths of
  Track A modules 01 14 35 51 53 54 55 56 67 and original-data modules 01
  04 04b 09. Every module was re-run first: the committed results reproduce
  within 2e-11 relative, so the result files were restored; only the traces
  were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces re-recorded after the AKM / KDE / GWR / HTZ clean-room rewrites (audit H19-H21, H23, H24)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `b1647b141` and `4eaf52b3c` (rewrites), `069d890c3` (traces).
- **Reason.** License provenance audit rows H19 (`fast/inference.py`, CR2
  HTZ test), H21 (`spatial/gwr/bandwidth.py`), H23 / H24
  (`regression/interflex.py`, `decomposition/_common.py`, now delegating
  to the new `core/_binned_kde.py`) were re-implemented clean-room (specs
  in `docs/dev/cleanroom/`). These files are on the traced paths of Track A
  modules 03, 15, 30, 31, 32, 47, 53, 65, 66, 67 and 87. All eleven were
  re-run first and reproduce their committed results (worst relative
  difference 5.5e-15); only the traces were re-recorded. H20
  (`bartik/_akm.py`) is not on a traced path.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-20 — DiD RCS / grid / bootstrap, AR tests, staggered SDID and score bootstrap clean-room; DR-RC SE default

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `f59df244a` (DiD clean-room and the delta-method default),
  `379f678c3` (dynamic-panel AR tests, FOD / `h=1` σ̂²), `ceab8ecf8`
  (staggered SDID inference), `cf5813fb4` (GLM score bootstrap),
  `2a1c9b877` (traces).
- **Reason.** License provenance audit rows H25-H27 and three DiD pieces
  (`did/callaway_santanna.py` grid and repeated-cross-section path,
  `did/_core.py::multiplier_bootstrap`, `did/wooldridge_did.py` improved
  panel DR) were re-implemented clean-room (specs in `docs/dev/cleanroom/`).
  The touched files are on the traced paths of Track A modules 04 05 16 17
  20 38 50 53 73 75 78 79 83 84 85 and original-data module 02. Every module
  was re-run first and reproduces its committed result within 1e-9; only the
  traces were re-recorded.
- **Effect on the paper.** No table number changes. Two statements to
  revisit at the next re-anchor: (1) the owner's decision makes the
  delta-method influence function the default for `drdid_rc` / `drdid_rc1`
  and `callaway_santanna(estimator='dr')` on repeated cross-sections; R
  `DRDID` / `did` and Stata `drdid` / `csdid` are reproduced only with
  `variance_convention='reference'`, so any sentence or table row that cites
  repeated-cross-section DR standard errors as T2 parity with R / Stata
  becomes T4 (reference divergence, Monte Carlo evidence in
  `tests/reference_parity/_fixtures/_drdid_rc_pre_regression_mc.py`); no
  Track A module uses this path. (2) Classical one-step dynamic-panel GMM
  after `transform='fod'` or `h=1` changes (σ̂² no longer halved), and
  classical AR statistics for system GMM / FOD now match xtabond2; Track A
  50 (one-step robust) is unaffected.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces re-recorded after the RD CR2 / CR3 and rddensity NaN fixes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `8026408c6`.
- **Reason.** `rd/_cct_side.py`, `rd/_cct_inference.py`,
  `rd/_cct_bandwidth.py`, `rd/rdrobust.py`, `rd/bandwidth.py` (CR2 / CR3
  cluster variances for `vce='hc2'` / `'hc3'` with clusters, as R
  rdrobust 4.0.0) and `diagnostics/rddensity.py` (NaN when the test is
  not computable) are on the traced paths of Track A modules 06 09 88 89
  and original-data module 05. None of them uses `hc2` / `hc3` with
  clusters or a non-computable density test. Every module was re-run
  first: the committed results reproduce within 1.4e-10 (88), 2.1e-11
  (09) and 3e-13 / 7e-12 (06, 89, orig 05), so the result files were
  restored; only the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces re-recorded after the radius `caliper_scale='sd'` fix in `sp.match`

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `b56926659` (fix), `5f6047f0c` (traces).
- **Reason.** `sp.match(method='radius')` now honours `caliper_scale='sd'`.
  `src/statspai/matching/match.py` is on the traced paths of Track A module
  11 and original-data modules 04 / 04b. None of them uses radius matching
  with an SD caliper; all three were re-run first and reproduce their
  committed results bit for bit. Only the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces re-recorded after the kernel / llr caliper warning in `sp.match`

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `f32eb3457` (warning), `31e0f81f3` (traces).
- **Reason.** `sp.match(method='kernel' | 'llr', caliper=...)` now warns
  that the caliper has no effect (numbers unchanged).
  `src/statspai/matching/match.py` is on the traced paths of Track A module
  11 and original-data modules 04 / 04b. All three were re-run first and
  reproduce their committed results bit for bit. Only the traces were
  re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces 06, 89 and orig 05 after the rdrobust recovery hint

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `26ce3ebf5`, `6acf85e28`.
- **Reason.** `src/statspai/rd/rdrobust.py` (a recovery-hint string and
  `alternative_functions` on the too-few-observations error) is on the
  traced path of Track A modules 06 and 89 and original-data module 05.
  All three were re-run first on the no-Rust path: their committed results
  reproduce within 7.3e-12 relative (06: 2.8e-13, 89: 7.2e-12, orig 05:
  2.8e-13), so the result files were kept and only the traces re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces 67 and 24 after the 1.40 default changes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `82fc04d7d` (defaults), `69b035a9a` (traces).
- **Reason.** The 1.40 defaults (`sp.feols` / `sp.fepois`
  `fixef_rm="singleton"`, `sp.kaplan_meier` `conf_type="log-log"`,
  `sp.bayes_regress` weakly informative prior) changed
  `src/statspai/fixest/wrapper.py` (traced path of Track A module 67) and
  `src/statspai/survival/models.py` (module 24). Module 67 now passes
  `fixef_rm="singleton"` to `sp.fepois` explicitly, the setting of its
  `fixest::fepois` reference. Both modules were re-run first on the
  no-Rust path: 67 reproduces its committed result bit for bit, 24 within
  1e-15 relative (last-digit jitter; the file was kept). Only the traces
  were re-recorded.
- **Effect on the paper.** None; no number changes. A reader who calls
  `sp.feols` / `sp.fepois` without `fixef_rm` on data with singletons now
  gets fixest's sample; module 67's Poisson fixture has none.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-21 — traces re-recorded after `sp.doctor` and the fetched gallery datasets

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `14c8461ff`.
- **Reason.** `src/statspai/datasets/__init__.py` and
  `src/statspai/datasets/_canonical.py` (new fetch-on-demand loaders,
  `mpdta(simulated=False)`, new `list_datasets` columns) and
  `src/statspai/__init__.py` (lazy `sp.doctor`) are imported on the traced
  paths of Track A modules 01 02 03 04 05 06 07 08 09 11 12 13 14 15 16 17
  18 19 20 22 24 25 26 27 53 55 65 66 73 88 and of the original-data
  modules 06-11. No estimator changed. Every module was re-run first: the
  committed results reproduce within 1e-9 (worst 1.4e-10 relative); only
  the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces 05 / 67 / 83 / 85 re-recorded after `parity=` and the feols cluster spellings

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `f2e119bc8`.
- **Reason.** `did/sun_abraham.py` and `did/event_study.py` (new `parity=`
  critical values, `did/_crit.py`) and `fixest/wrapper.py` (cluster
  `vcov` spellings, `core/_vcov_spellings.py`) are on the traced paths of
  Track A modules 05 67 83 85. Every module was re-run first: the
  committed results reproduce byte for byte; only the traces were
  re-recorded. No module passes `parity=`, so the defaults it reads are
  unchanged.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-21 — traces after the DiD API changes (swapped `group=`, aliases, result accessors)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `761b03983` (DiD API), `a4f10d3ee` (traces).
- **Reason.** `CausalResult` now mixes in `core/_result_access.py`
  (`.attgt` / `.event_study` / `.aggregations`), so
  `src/statspai/core/results.py` changed; it is on the traced path of 69
  Track A modules and all 12 original-data modules. The `did/` alias
  decorators and the argument-role checks (`did/_arg_roles.py`) touch
  modules 16, 17, 20, 38, 73, 75, 79, 82 and 84. Modules 04, 05, 16, 17,
  20, 38, 73, 75, 79, 82 and 84 were re-run first on the no-Rust path: 04,
  05, 16, 17, 20, 75 and 82 reproduce their committed results bit for
  bit; 38, 73, 79 and 84 differ in the last digits (about 1e-12
  relative), and main without these changes produces the same bytes, so
  the drift predates them and the files were kept. Only the traces were
  re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces 24, 65 and 66 after the strict-docs docstring fixes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `7bc4e4e78` (docstrings), `ce26425f4` (traces).
- **Reason.** `mkdocs build --strict` failed on free paragraphs inside the
  Parameters sections of `CoxResult.ph_test` and `gwr_bandwidth` and on
  `CoxResult.plot` documenting `type=`; the text moved to Notes /
  `**kwargs`. `src/statspai/survival/models.py` is on the traced path of
  Track A module 24, `src/statspai/spatial/gwr/bandwidth.py` on 65 and 66.
  All three were re-run on the no-Rust path first: 65 and 66 reproduce
  their committed results byte for byte, 24 within 1.4e-15 relative, so
  the result files were kept and only the traces re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
### 2026-10-21 — traces after the weighted Bacon, honest_did seed / breakdown and formula-integer changes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `ab6433bf9` (feature), `a600a87da` (traces).
- **Reason.** `core/utils.py` and the new `core/_formula_dtypes.py`
  (integer columns read by generated formula terms are cast to float64),
  `did/_core.py`, `regression/ols.py`, `regression/iv.py`,
  `regression/multinomial.py`, `panel/panel_reg.py`,
  `survey/estimators.py`, `postestimation/_design.py`, `did/bacon.py`,
  `did/honest_did.py`, `did/_honest_sets.py` and
  `did/_honest_breakdown.py` are on the traced paths of Track A modules 01
  02 04 05 10 14 16 17 20 21 35 37 42 44 45 46 47 48 49 51 53 54 55 56 57
  58 59 63 64 67 69 73 75 78 79 83 84 85 and of the original-data modules
  01 02 04 04b 09. All were re-run first on the no-Rust path into a
  staging directory: every committed result reproduces within 1e-9 (25
  modules bit for bit, among them 04, 05, 10, 17, 20, 21 and 85; 01
  2e-13, 02 5e-12, 73 2e-12, the rest below 1e-14; original-data 01 / 02
  / 09 in the last digits, 1e-11 at most), so the
  committed files were kept and only the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces of the formula-integer modules after narrowing an except

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `8ae8fdc03`.
- **Reason.** `core/_formula_dtypes.py` (added in `ab6433bf9`) caught
  `Exception` around the formula parse; it now catches patsy's parse
  errors only. The file is on the traced paths of Track A modules 01 14
  48 51 53 54 55 56 57 and of the original-data modules 01 04 04b 09. The
  change does not touch the estimation (those formulas parse), so the
  committed results stand; only the traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces re-recorded after the Callaway-Sant'Anna `parity=` changes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `086d3cfad`.
- **Reason.** `core/results.py` (the pre-test line of `summary()`),
  `did/callaway_santanna.py`, `did/aggte.py`, `did/_cs_rcs.py`,
  `did/report.py`, the new `did/_cs_conventions.py` and `registry.py` are on
  the traced paths of Track A modules 01 02 04 05 06 07 08 09 10 11 12 14 16
  17 18 19 21 24 28 29 35 36 37 38 40-59 61-67 71-79 81-87 89 and of all
  twelve original-data modules. Every module was re-run first: Track A 04
  and 79 and original-data 02 (the Callaway-Sant'Anna rows) are
  byte-identical to the `origin/main` code, and the rest reproduce the
  committed results within 1e-9 (worst 7.3e-11 relative, last-digit jitter
  that `origin/main` shows too); only the traces were re-recorded.
- **Effect on the paper.** None; no number changes. The default
  `callaway_santanna` keeps the universal base period and the Hotelling-F
  pre-test p-value; `parity=` is opt-in.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces after the mpdta docstring fix (strict docs)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `de972fcf8` (docstring), `2019529ce` (traces).
- **Reason.** The "Our replica preserves" list in `sp.datasets.mpdta`'s
  docstring sat inside its Parameters section and failed `mkdocs build
  --strict`; it moved to Notes. `src/statspai/datasets/_canonical.py` is
  on the traced paths of Track A modules 01 02 04 05 06 08 09 11 14 16 17
  20 22 53 55 73 88 and original-data modules 06-11. All were re-run on
  the no-Rust path first: the committed results reproduce within 1.4e-10
  relative (most bit for bit), so the result files were kept and only the
  traces re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-21 — traces after the feols integer-term fix

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `b1df36198` (fix), `0c0cc702b` (traces and fixture lock).
- **Reason.** `sp.feols` / `fepois` / `feglm` now cast integer columns read
  by `I(...)` terms to float64 (`fixest/wrapper.py`, new helper in
  `src/statspai/core/_formula_dtypes.py`). Those files are on the traced
  paths of Track A modules 01 14 48 51 53 54 55 56 57 67 and original-data
  modules 01 04 04b 09. All were re-run on the no-Rust path first: the
  committed results reproduce within 2e-11 relative, so the result files
  were kept and only the traces re-recorded. The same commit refreshes
  the Tier A fixture lock, stale since `82fc04d7d` edited
  `tests/r_parity/67_panel_glm.py`.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`

### 2026-10-21 — traces re-recorded after `crit_dist=` on the regression event studies

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `6685fc5da`.
- **Reason.** `did/_crit.py`, `did/event_study.py` and `did/sun_abraham.py`
  (new `crit_dist=` argument and the FutureWarning announcing the 1.41
  t(G-1) default) are on the traced paths of Track A modules 05, 83 and 85;
  the entry scripts 05 and 85 now pass `crit_dist="normal"`. All three
  modules were re-run first: `05_sunab_py.json`, `83_lpdid_py.json`,
  `85_twfe_event_study_py.json` and their CSVs are byte-identical; only the
  traces were re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-21 — traces after the DiD group guards, drdid `treat=` and `sp.did(crit_dist=)`

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `6e2cb7dd8` (group guards, drdid `treat=`, `parity=`
  forwarding), `94bada667` (`sp.did(crit_dist=)`, `_crit.py` warning
  attribution).
- **Reason.** `sp.drdid`'s treatment-group parameter is now `treat`
  (`group=` deprecated), and the unit-role swap guard of
  `src/statspai/did/_arg_roles.py` now also runs in `gardner_2s.py`,
  `stacked_did.py`, `wooldridge_did.py` (etwfe, wooldridge_did) and
  `did_multiplegt.py`; `did/_crit.py` changed only how the 1.41
  `FutureWarning` locates the caller. Those files are on the traced paths
  of Track A modules 05 16 17 38 73 75 81 84 85, and the entry script
  `tests/r_parity/38_drdid.py` now passes `treat=` (fixture lock
  refreshed). All nine were re-run on the no-Rust path first: the results
  are byte-identical to main's own reruns and reproduce the committed
  files within 7.3e-11 relative, so the result files were kept and only
  the traces re-recorded.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`

### 2026-10-21 — trace 81 after the did_multiplegt cell / controls fix

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `b65208241` (fix), `aeaa1ee15` (trace).
- **Reason.** `src/statspai/did/did_multiplegt.py` now collapses rows below
  the group level to weighted cells and uses Stata's stayer-cell regression
  for `controls=`. Track A 81 (one row per group-period, no controls) was
  re-run on the no-Rust path: the result equals what main's code gives
  (both differ from the committed file only in the 16th digit, an
  environment effect), so the result file was kept and only the trace
  re-recorded.
- **Effect on the paper.** None; no number in Track A changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-22 — trace 81 after did_multiplegt robust_dynamic / controls

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `cf9290845` (fix), `f61157249` (trace).
- **Reason.** `src/statspai/did/did_multiplegt.py` gains `robust_dynamic=`
  (Stata's first-time-switcher estimators, new `_dcdh_first_switch.py`),
  applies `controls=` to placebos and dynamic effects, and weighs
  disaggregated dynamic cells at the switching period. Track A 81 (default
  call, one row per group-period, no controls) was re-run on the no-Rust
  path: the result equals what the previous HEAD's code gives (both differ
  from the committed file only in the 16th digit), so the result file was
  kept and only the trace re-recorded.
- **Effect on the paper.** None; no number in Track A changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-22 — trace 81 after the did_multiplegt split / firstdiff_placebo

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `4feca435e` (split), `55a7fd849` (feature), `f80d5b171` (trace).
- **Reason.** `src/statspai/did/did_multiplegt.py` is split by concern
  into `_dcdh_cells.py`, `_dcdh_pairwise.py` and `_dcdh_inference.py`
  (pure move), and gains `firstdiff_placebo=` and
  `model_info['effect_average']` on the `robust_dynamic=True` path. Track A
  81 (default call) was re-run on the no-Rust path: the result equals what
  origin/main's code gives (both differ from the committed file only in
  the 16th digit), so the result file was kept and only the trace
  re-recorded.
- **Effect on the paper.** None; no number in Track A changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
### 2026-10-22 — traces after the personal-bankruptcy replication fixes

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `a8749a5a3` (guardrail), `1352dda7c` (dCDH / did_imputation),
  `366de8723` (did_summary / stacked_did), `3c6764a35` (breakdown and
  pre-test from moments, ddtiming-style Bacon), `66651bf91` (traces).
- **Reason.** Estimation-path sources changed:
  `src/statspai/core/_design_guardrails.py` (event-time dummies are no
  longer flagged as cohort-level assignment; warnings only),
  `did/did_imputation.py`, `did/did_multiplegt_dyn.py`, `did/stacked_did.py`,
  `did/bacon.py`, `did/pretrends.py`, `did/_honest_breakdown.py` (new
  options and columns, defaults unchanged). Track A 01 10 14 16 20 35 51 53
  54 55 56 67 75 76 78 84 were re-run on the no-Rust path and reproduce the
  committed results (worst relative 7.8e-16); original-data 01 04 04b 09
  are affected only through the guardrail. Only the traces are re-recorded.
- **Effect on the paper.** None; no number in Track A or the original-data
  ledger changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-22 — trace 67 after the feols singleton-mask fix

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `fdd595c30` (fix), `447cc2872` (trace).
- **Reason.** `src/statspai/fixest/wrapper.py` copies the singleton mask
  (pandas 3 read-only arrays). Track A 67 executes that file; it was re-run
  on the no-Rust path and reproduces its committed result byte for byte,
  so only the trace changes.
- **Effect on the paper.** None.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-22 — Track C re-timed on 1.40.0

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `c535db8d7`.
- **Reason.** The 1.40.0 batch (fixest defaults, DiD, DML, clean-room
  rewrites) changed files on all four timed paths. Re-measured on
  fd2a1b2b2 (the 1.40.0 release commit) with `tests/perf/run_when_idle.sh`
  (peak load at most 1.46, no step repeated, one thread on both sides,
  statspai-hdfe 0.9.1 loaded). None of the 46 timed-path source files nor
  the Rust crate differs between fd2a1b2b2 and main at `824218939`;
  `trace_perf_path.py --check` passes on the committed tree.
- **Effect on the paper.** Track C table (`track_c_perf.tex`) and the
  log-log figure. Seconds for StatsPAI (ratio to the reference): 01_hdfe
  n=1e6 0.0321 (2.37, was 2.41); 02_csdid n=125k 0.0345 (2.63, was 2.61);
  03_scm 100 donors 1.300 (3.55, was 3.59); 04_dml n=10k 0.0090 (0.99,
  was 1.09). No conclusion in the text changes; the DML row is now parity
  with DoubleML at n=10k rather than 9% faster.
- **Paths.**
  - `tests/perf/results/01_hdfe_py.json`
  - `tests/perf/results/01_hdfe_R.json`
  - `tests/perf/results/02_csdid_py.json`
  - `tests/perf/results/02_csdid_R.json`
  - `tests/perf/results/03_scm_py.json`
  - `tests/perf/results/03_scm_R.json`
  - `tests/perf/results/04_dml_py.json`
  - `tests/perf/results/04_dml_doubleml_py.json`
  - `tests/perf/results/_timed_path.json`
  - `tests/perf/results/perf_table.md`
  - `tests/perf/results/perf_table.tex`
  - `tests/perf/figures/track_c_loglog.pdf`
  - `tests/perf/figures/track_c_loglog.png`

### 2026-10-22 — traces 04 / 38 / 79 / orig 02 after the CS parity fix

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `c6c0b8c73` (fix), `f268a88ac` (traces).
- **Reason.** `src/statspai/did/callaway_santanna.py`, `src/statspai/did/aggte.py`,
  `src/statspai/did/_cs_conventions.py` and `src/statspai/did/_cs_rcs.py` changed
  (`parity=` on repeated cross-sections and unbalanced panels, csdid
  weighted counts). Default paths are unchanged: Track A 04 reproduces byte
  for byte; 38, 79 and original-data 02 within 1.5e-11 relative, an
  environment effect also seen with main's code, so the result files were
  kept and only the traces re-recorded. Track C 02_csdid is now stale
  (`trace_perf_path.py --check`), left for the next re-time.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-22 — release 1.40.1

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `21d386c6d` (release).
- **Reason.** Version strings in the parity tables and the fixture lock; the
  docstring-only edits (`versionadded:: 1.41.0` -> `1.40.1`) in
  `src/statspai/did/` made the traces of Track A 04 10 16 20 38 76 78 79
  81 84 and original-data 02 stale; they were re-recorded. No result file
  changes.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/parity_table.tex`
  - `tests/r_parity/results/parity_table_3way.tex`

### 2026-10-23 — traces after the GMM small-sample fix and `inference_rule()`

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `934889673`.
- **Reason.** `src/statspai/regression/iv.py` (the GMM `n/(n-k)` factor
  under the unadjusted weight matrix), `src/statspai/core/results.py` and
  `src/statspai/core/_agent_summary.py` (the inference-rule lines of
  `summary()`), `src/statspai/panel/panel_reg.py` (model test from the
  reported variance) and `src/statspai/postestimation/hypothesis.py`
  changed, so nearly every trace went stale and was re-recorded. Every
  Track A module was re-run into scratch first: 89 of 89 reproduce their
  committed `*_py.json` (worst 1.4e-10 relative, module 88), so no result
  file changed.
  No Track A module fits GMM through `sp.iv`.
- **Effect on the paper.** None; no number changes.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-24 — traces after `wmatrix=` and the 1.42 default announcements

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `ef844a880`, `b576aac85`.
- **Reason.** `src/statspai/regression/iv.py` (`wmatrix=`),
  `src/statspai/panel/_ssc.py` and `src/statspai/panel/panel_reg.py` (the
  `FutureWarning` for the 1.42 `ssc` default). Traces 02, 35, 59, 69 and the
  original-data ledger re-recorded; the four modules were re-run first and
  reproduce (worst 5.2e-12). Module 35 now emits the announcement warning;
  its numbers do not change until 1.42.
- **Effect on the paper.** None now. When the 1.42 default lands, the
  clustered rows of module 35 change (`ssc='stata'`), unless the module pins
  `ssc='linearmodels'`.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-25 — Track C CS-DiD re-timed on 1.40.1 (JSS re-anchor)

The freeze is inactive (not submitted); recorded for the re-anchor to 1.40.1.

- **Commits.** `70359a722`.
- **Reason.** The JSS manuscript moves to tag v1.40.1, whose CS-DiD files
  differ from the 1.40.0 tree the timings were measured on. Re-measured
  on a checkout of v1.40.1 with the idle-guarded runner (02_csdid only;
  peak load 1.48 / 1.33) and re-traced there; all four modules pass
  `trace_perf_path.py --check` on v1.40.1. On main the four are reported
  stale only because `core/results.py` / `core/_agent_summary.py` changed
  after the tag.
- **Effect on the paper.** Track C table and figure, CS-DiD row: 0.037 s
  against 0.083 at n=125k (2.2x, was 2.6x); 1.18x / 1.74x at the smaller
  sizes. Section 6 updated accordingly.
- **Paths.**
  - `tests/perf/results/02_csdid_py.json`
  - `tests/perf/results/02_csdid_R.json`
  - `tests/perf/results/_timed_path.json`
  - `tests/perf/results/perf_table.md`
  - `tests/perf/results/perf_table.tex`
  - `tests/perf/figures/track_c_loglog.pdf`
  - `tests/perf/figures/track_c_loglog.png`

### 2026-10-27 — trace ledgers re-recorded after the option scan of 18 modules

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `593bb19fb`, `7482f61e7`, `402c525e0`, `1f2480bd7`.
- **Reason.** The option scan (spatial, bartik, survival, epi, mendelian,
  qte, bounds, selection, postestimation, inference, robustness, survey,
  multilevel, gmm, metalearners, tmle, conformal_causal, policy_learning),
  the 1.41 default switches, and fixes in `core/results.py`,
  `postestimation/_covariance.py` and `panel/hdfe.py` changed source files
  on the estimation paths of most Track A and all original-data modules.
  Each module was re-run first: `verify_reproduce_py.py` reports 89
  reproduce, 0 drift, and the original-data results moved by at most 2e-11
  relative (restored). Module 70's script now passes
  `value_evaluation="in_sample"` (the `sp.policy_tree` default became
  cross-fitted in 1.41); its committed result reproduces byte for byte.
- **Effect on the paper.** None. No committed parity result changed.
  Track C is stale on main because `core/results.py` changed (it backs the
  v1.40.2 numbers; not re-timed).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-28 — trace ledgers re-recorded after the Econometrica 94(5) round

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `ed1670e25`.
- **Reason.** The Econometrica 94(5) round (`29f5739da`) added lazy
  exports and `__all__` entries to `src/statspai/__init__.py`, which is on
  ten Track A paths (03, 13, 15, 24, 25, 26, 27, 53, 65, 66) and on the
  original-data path of `08_nhefs_ch14_gestimation`. No estimation code
  changed. `verify_reproduce_py.py` on the ten modules reports 10
  reproduce, 0 drift (worst relative 1.7e-10); re-running module 08 left
  its result file byte-identical.
- **Effect on the paper.** None. No committed parity result changed. The
  new functions of that round are outside Track A.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-28 — trace ledgers re-recorded after the QJE gap pass

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `ce6a77604`, `c2d3165f1` (the merge with the Econometrica
  94(5) round, re-recorded on the merged tree).
- **Reason.** The pass added exports to `src/statspai/__init__.py`
  (`sp.fmm`, `sp.summary_index`, `sp.sharpened_qvalues`, `sp.shapley_r2`,
  `sp.krippendorff_alpha`, `sp.llm_code_responses`, `sp.iv_mediate`, result
  classes), the only file on the traced paths of Track A modules 03, 13,
  15, 24, 25, 26, 27, 53, 65, 66 and original-data module 08 that changed.
  `verify_reproduce_py.py` on those ten modules reports 10 reproduce, 0
  drift.
- **Effect on the paper.** None. No committed parity result changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-27 — trace ledgers re-recorded after option scan round 6

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `6022e8627`.
- **Reason.** Round 6 changed source files on Track A and original-data
  estimation paths: `core/results.py` (interval rescaling in `tidy` /
  `summary`, `predict` confidence df), `core/utils.py` (intercept
  spellings), `fast/feols.py`, `fast/event_study.py`, `fast/etable.py`
  (CR1 K, t(G-1) p values) and `inference/aipw.py` (PSU folds).
  `verify_reproduce_py.py` reports 89 reproduce, 0 drift, and no
  original-data result file changed.
- **Effect on the paper.** None. No committed parity result changed.
  Track C stays stale on main (`core/results.py` and `fast/*` also
  changed); the committed timings back v1.40.2.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-27 — trace ledgers re-recorded after option scan round 5

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `76c01ecc7`, `3fe717d60` (the second after a hint-text fix in
  `core/utils.py`).
- **Reason.** Round 5 (aipw / tmle / policy_tree / metalearner / conformal,
  spatial ML search and impacts, heckman and ipcw input checks,
  `panel/hdfe.py` absorb_ols warning, the 1.41 removals including
  `evalue(rare_outcome=)`, `core/utils.py` formula guards,
  `regression/iv.py` absorb t / p) changed source files on Track A and
  original-data estimation paths. `verify_reproduce_py.py` reports 89
  reproduce, 0 drift, and the original-data results reproduce unchanged.
- **Effect on the paper.** None. No committed parity result changed.
  Track C stays stale on main (`diagnostics/evalue.py` and
  `_article_aliases.py` also changed); the committed timings back v1.40.2.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-27 — release 1.41.0

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `029ebdde6` (release).
- **Reason.** Version strings in the parity tables and the fixture lock.
  No result file changes; the trace ledgers are current.
- **Effect on the paper.** None; no number changes. The paper stays
  anchored at v1.40.2.
- **Paths.**
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/parity_table.tex`
  - `tests/r_parity/results/parity_table_3way.tex`

### 2026-10-27 — Track C re-timed on 1.40.2 (JSS re-anchor); manifest re-hashed

The freeze is inactive (not submitted). The manifest
`tests/jss_review_freeze.json` was re-hashed on the v1.40.2 tag (with the
re-timed Track C files below in place) and set back to inactive, so at
submission the only step is `"active": true`.

- **Commits.** `9d5f2a2f4`.
- **Reason.** All four Track C modules were stale on v1.40.2
  (`core/results.py` and `core/_agent_summary.py` gained
  `inference_rule()` after 1.40.1, both on the timed paths). Re-timed on
  the tag with the idle-guarded runner (peak load 0.79–1.71) and re-traced
  there; `trace_perf_path.py --check` is OK for all four on v1.40.2.
- **Effect on the paper.** Section 6 and the Track C table / figure carry
  these numbers (Paper-JSS `2ba1890`): at the largest size HDFE 2.4x,
  CS-DiD 2.3x (was 2.2x), SCM 3.5x, DML 1.09x (was 0.99x; a tie under the
  1.5x rule).
- **Paths.**
  - `tests/perf/figures/track_c_loglog.pdf`
  - `tests/perf/figures/track_c_loglog.png`
  - `tests/perf/results/01_hdfe_R.json`
  - `tests/perf/results/01_hdfe_py.json`
  - `tests/perf/results/02_csdid_R.json`
  - `tests/perf/results/02_csdid_py.json`
  - `tests/perf/results/03_scm_R.json`
  - `tests/perf/results/03_scm_py.json`
  - `tests/perf/results/04_dml_doubleml_py.json`
  - `tests/perf/results/04_dml_py.json`
  - `tests/perf/results/_timed_path.json`
  - `tests/perf/results/perf_table.md`
  - `tests/perf/results/perf_table.tex`

### 2026-10-28 — trace ledgers re-recorded after the AER 116(10) batch

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `8ce4a31dc`.
- **Reason.** The AER 116(10) batch (`227691d7b`, derived files in
  `25b568688`) added lazy exports to `src/statspai/__init__.py` (ten Track A
  paths: 03, 13, 15, 24, 25, 26, 27, 53, 65, 66; original-data path of
  `08_nhefs_ch14_gestimation`), added `tvc=` / `texp=` to
  `src/statspai/survival/models.py` (24_coxph) and vectorised the
  nested-in-cluster check in `src/statspai/panel/hdfe.py` (68_demean_within,
  75_stacked). Default outputs of the touched estimators are bit-identical.
  `verify_reproduce_py.py` on the twelve modules reports 12 reproduce,
  0 drift; module 08 left its result file byte-identical.
- **Effect on the paper.** None. No committed parity result changed. The
  batch's ⚠️ fix to `sp.hdfe_ols` IV degrees of freedom is outside Track A
  (`panel/_hdfe_iv.py` is on no traced path) and outside Track C.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-28 — trace ledgers re-recorded for option scan round 7

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `00c1510f5`.
- **Reason.** Round 7 (agent tool layer, smart / workflow, the round-6 open
  items and the round-7 open items, rebased onto the AER 116(10) merge)
  changed files on traced paths: `src/statspai/core/results.py` (one
  `conf_int` rule across results), `src/statspai/core/_vcov_spec.py`
  (`vce(...)` wrapper, CRV1 `"f+h"`), `src/statspai/core/_collinear.py`,
  `src/statspai/regression/count.py`, `src/statspai/regression/logit_probit.py`,
  `src/statspai/regression/quantile.py`, `src/statspai/fast/inference.py`
  and `src/statspai/inference/ipw.py`. Every Track A module and every
  original-data module re-traced. `verify_reproduce_py.py` reports 89
  reproduce, 0 drift. The twelve original-data scripts, re-run, differ from
  their committed results only in the last digits (largest relative
  difference 1.6e-11); those files were restored.
- **Effect on the paper.** None. No committed parity result changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-28 — Track A trace re-recorded for 24_coxph (sp.cox convergence fix)

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commit.** `17807fc00`.
- **Reason.** `sp.cox` stopped one Newton step short when a step between
  ~1e-9 and ~1e-6 exhausted its line search. The fix changes
  `src/statspai/survival/models.py`, on the traced path of Track A module
  24. `verify_reproduce_py.py 24_coxph` reproduces (fixture byte-identical,
  worst relative difference 7.8e-16), so the committed result is unchanged.
- **Effect on the paper.** None. No committed parity result changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-28 — trace ledgers re-recorded after the Econometrica 94(5) follow-up

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `08236dd01`.
- **Reason.** The follow-up (`02c56a30d`) added the `rust_bus` loader to
  `src/statspai/datasets/__init__.py` and a lazy export to
  `src/statspai/__init__.py`. Between them these files are on 30 Track A
  paths (01-09, 11-20, 22, 24-27, 53, 55, 65, 66, 73, 88) and on the six
  NHEFS original-data paths (06-11). No estimation code changed.
  `verify_reproduce_py.py` on the 30 modules reports 30 reproduce, 0
  drift (worst relative 1.4e-10). Re-running the six NHEFS modules left
  four result files byte-identical; two (07, 09) moved by at most 1.9e-13
  and were restored. Recorded on top of `17807fc00` (the Cox Newton-step
  fix); `24_coxph` was re-verified on that tree (reproduces, 0 drift).
- **Effect on the paper.** None. No committed parity result changed. The
  functions of the follow-up are outside Track A.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`


### 2026-10-28 — Track A modules 90_eivreg and 91_biprobit added

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `ee4b0c796`.
- **Reason.** Two new Track A modules promote `sp.eivreg` and
  `sp.biprobit` from reference_parity frozen fixtures to three-way parity.
  `90_eivreg` agrees with Stata 18 `eivreg` and R `eivtools::eivreg`
  0.1-8 to 6.7e-15; `91_biprobit` agrees with Stata 18 `biprobit` to
  6.8e-12 and with R `GJRM` to 4.7e-7 (GJRM's bivariate normal CDF is
  accurate to about 1e-7). The full R, Python and Stata reproducibility
  runs that regenerated the three reports found 91 / 91 / 87 modules
  reproducing with 0 drift; no existing golden value changed. The trace
  ledger gains the two modules.
- **Effect on the paper.** None on the anchored tables. At the next
  re-anchor the Track A inventory grows from 89 to 91 modules (87 with a
  Stata reference) and the machine tier from 82 to 84; both rows join the
  appendix table as T2.
- **Paths.**
  - `tests/r_parity/results/90_eivreg_R.json`
  - `tests/r_parity/results/90_eivreg_py.json`
  - `tests/r_parity/results/91_biprobit_R.json`
  - `tests/r_parity/results/91_biprobit_py.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/stata_parity/results/90_eivreg_Stata.json`
  - `tests/stata_parity/results/91_biprobit_Stata.json`

### 2026-10-28 — trace ledgers re-recorded after the Econometrica 94(5) remaining items

The freeze is inactive (not submitted); recorded for the next re-anchor.

- **Commits.** `f2a102393`.
- **Reason.** The remaining-items round (`778ee2c34`) added the
  `china_shock` loader to `src/statspai/datasets/__init__.py` and the
  `private_info_bound` export to `src/statspai/__init__.py`, which are on
  31 Track A paths (01-09, 11-20, 22, 24-27, 53, 55, 65, 66, 73, 88, 90)
  and on the original-data paths 06-11. No estimation code on those
  paths changed. `verify_reproduce_py.py` on the 31 modules reports 31
  reproduce, 0 drift. Re-running the eleven original-data modules moved
  four result files by at most 1.6e-11 (last-digit jitter, restored);
  the rest were byte-identical.
- **Effect on the paper.** None. No committed parity result changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
