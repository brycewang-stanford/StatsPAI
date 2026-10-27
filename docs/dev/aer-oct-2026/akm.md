# AER 116(10) batch, line `akm`

Motivation: a two-step design (AKM worker effects on a large linked
employer-employee panel; then entrant outcomes on the mean standardized
worker effect of coworkers in the same firm x occupation x entry year,
excluding the focal entrants, with `absorb(occ#year firm#occ) cluster(firm)`).
This line adds the pieces StatsPAI was missing for that pipeline and closes a
silent drop in the Stata translator.

## Proposed CHANGELOG entry (Unreleased)

### Added

- `sp.akm(data, y, worker, firm, *, time=None, covariates=None,
  connected=None, variance='kss', leverage='auto', n_draws=200, seed=None,
  cluster=None)`: Abowd-Kramarz-Margolis worker and firm fixed effects
  (`abowd1999high`) on the largest connected set, the leave-one-observation-out
  connected set, or every row (`connected='all'`, effects normalised per
  connected set). Variance decomposition of the outcome (net of covariates and
  time indicators, partialled out with the full model) into var(worker),
  var(firm), 2cov and residual: plug-in, homoskedastic correction
  (`andrews2008high`) and the Kline-Saggio-Solvsten leave-out correction
  (`kline2020leave`), with exact leverages (dense firm-by-firm algebra, up to
  3000 firms by default) or seeded Rademacher random projections. Reports
  movers, connected-set sizes and dropped observations by reason. Returns
  `AKMResult` (`worker_effects`, `firm_effects`, `decomposition`, `shares`,
  `coefficients`, `sample`, `leverage`, `data`, `fixef()`,
  `felsdvreg_variables()`).
- `sp.leave_out_mean(data, value, group, *, exclude=None, weights=None)`:
  per-row mean over the other rows of the group (one or several key columns),
  with rows flagged by `exclude` kept out of every peer pool; rows without a
  peer are NaN with a `StatsPAIWarning` and counted in `.attrs`.
- `sp.two_step_bootstrap(step1, step2, data, *, cluster=None, n_boot=999,
  seed=None, alpha=0.05, idcluster=None)`: cluster (or row) bootstrap that
  re-runs a generated-regressor step and the model of interest on every
  resample; reports step 2's own standard errors and the ratio next to the
  bootstrap ones (built on the `sp.bootstrap` engine).
- `sp.hdfe_ols(...).fixef(absorb_constant=True)`: the absorbed fixed effects
  of an OLS fit, recovered by an exact sparse solve and normalised as Stata
  `reghdfe` saves them (`absorb_constant=False` gives reghdfe's saved
  variables, `_cons` in `.attrs['constant']`).
- `sp.from_stata` / `sp.stata`: `felsdvreg` (Stata Journal st0143,
  `cornelissen2008stata`) translates to `sp.akm(connected='all',
  variance='none')`; `sp.stata` creates its eight variables.

### Fixed

- `sp.from_stata('reghdfe y x, absorb(wfe=worker ffe=firm)')` and
  `absorb(..., savefe)` silently dropped the saved fixed effects (empty notes,
  `complete=True`). The translation now carries them: `python_code` stays
  the single fit call and the new `python_script` adds
  `fit.fixef(absorb_constant=False)` and the columns; `sp.stata` adds the variables (missing
  outside `e(sample)`, r(110) on an existing name, `__hdfe#__` for `savefe`),
  and the cases that cannot be recovered (saved slopes, the IV fit of
  `ivreghdfe`, weighted robust fits routed to `sp.feols`) are reported in
  `dropped` / `complete=False`.
- `sp.from_stata('ppmlhdfe y x, absorb(fe=id year)')` passed `fe=id` (and,
  with `, savefe`, the suboption) to `sp.ppmlhdfe` as column names while
  reporting `complete=True`; it now passes the columns and reports the saved
  effects as not translated.

No ⚠️ correctness entry: no existing estimator's numbers change.

MIGRATION: nothing (additive; `FEOLSResult` gains a method).

## Evidence

| Check | Reference | Result | Tolerance and why |
| --- | --- | --- | --- |
| leave-one-out connected set | pytwoway 0.3.21 `leave_out_observation` | same 1,727 rows, 346 workers, 30 firms, 241 movers | exact |
| plug-in var(alpha), var(psi), cov, var(eps), var(y); HO var(psi), cov, var(eps); KSS var(psi), var(eps); P_ii range; alpha+psi | pytwoway exact traces / leverages | max diff 4e-13 (effects 4e-11) | 1e-10: closed forms both sides, pytwoway effects from MINRES at 1e-10 |
| HO var(alpha) | pytwoway exact | ours = pytwoway - sigma2_ho (n_w - 1)/n to 1e-12 | pytwoway's exact trace adds the diagonal block `(W'W)^{-1}` as a 1-D vector (row broadcast), dropping exactly `(n_w-1)/n`; pinned as an identity |
| KSS var(alpha), cov; HO all | pytwoway Hutchinson traces, 4000 draws | within 0.6-2 MC SEs | 4 MC SEs (stored per-draw SDs / sqrt(4000)); S-level screen |
| P_ii, B_ii (three components) | dense `(X'X)^{-1}` of [D, F] in the test | 1e-12 / 1e-14 | rounding only |
| known truth, fixed design, 300 heteroskedastic error draws | analytic `E[plug-in] = theta + sum B_ii sigma_i^2`; `E[KSS] = theta`; `E[HO]` closed form | all within 4 MC SEs; the plug-in bias of var(psi) is > 30 MC SEs | 4 MC SEs of the 300-draw mean |
| JLA vs exact | sp.akm exact | KSS components within 10% of the size of the exact correction at 2000 draws | per-form relative sd sqrt(2/2000) ~ 3% |
| `hdfe_ols(...).fixef(absorb_constant=False)` | Stata 18 reghdfe 6.13.1 `absorb(w=worker f=firm)`, `+year`, `[aw=]`, `savefe` (all `tolerance(1e-14)`) | b to 1e-15, SE to 1e-12, `_cons` 2e-16, year effect 5e-11, worker/firm effects 1.2e-7 | 1e-6: reghdfe's recovered effects reproduce its own fitted values only to 3e-8 |
| felsdvreg translation | Stata felsdvreg 1.8.0 (st0143), with and without `normalize` | b 1e-15, SE 1e-12, N; peff/feff/xb/res equal to felsdvreg's float storage; mover, mnum, pobs, group identical | 2.4e-7 (half a float ulp below 4) |
| `two_step_bootstrap` | `sp.bootstrap` of step 2 (non-random step 1) | identical replicates | exact |
| `two_step_bootstrap` 2SLS by hand | `sp.ivreg` cluster SE (200 clusters) | 0.0354 vs 0.0346 (naive 0.0468) | 10%: MC sd of a B=999 bootstrap SE ~2.2%, plus O(1/G) |
| `leave_out_mean` | brute-force double loop; jackknife identity `(S - v_i)/(n_g - 1)` | equal | 1e-12 rounding |

Fixtures (all generated by committed scripts):
`tests/reference_parity/_fixtures/_generate_akm_twoway_data.py` ->
`akm_twoway.csv`; `_generate_akm_twoway_pytwoway.py` ->
`akm_twoway_pytwoway.json` (pytwoway 0.3.21 needs numpy 1.26 / pandas 2.1 /
scipy 1.11 / pyamg 5.0 in its throwaway env; MIT licence, so its source was
read to establish its conventions); `_generate_akm_twoway_Stata.do` ->
`akm_twoway_Stata.{csv,json}` (felsdvreg installed into a private adopath
under `c(tmpdir)`, never PLUS).

## Tests

- `tests/test_akm.py` (known truth, dense identities, JLA, edge cases),
  `tests/reference_parity/test_akm_pytwoway_parity.py`,
  `tests/reference_parity/test_leave_out_mean_identity.py`,
  `tests/reference_parity/test_two_step_bootstrap_known_answer.py`,
  `tests/test_stata_translation_akm.py` (Stata reghdfe / felsdvreg), plus a
  felsdvreg round trip in `tests/test_translation.py::TIER3_ROUND_TRIPS`.
- Related suites (every test file mentioning hdfe_ols / reghdfe / fixef /
  from_stata / sp.stata / bootstrap / ppmlhdfe / the registry): 17,300+ tests
  pass. Expected failures until the lead regenerates artefacts:
  `test_schema_export::test_committed_schemas_dir_is_in_sync`,
  `test_parity_index::{test_snapshot_matches_fresh_regeneration,
  test_public_parity_doc_is_in_sync}`,
  `test_source_failure_modes::test_table_is_current`.

## Open issues

- KSS inference (standard errors / confidence sets for the variance
  components, KSS Section 4-5) is not implemented; only point estimates.
- Leave-out is at the observation level. With person-year data and serially
  correlated errors within a worker-firm match the authors leave out whole
  matches (or collapse to match means with weights); neither is implemented,
  and the docstring/registry state the independence assumption.
- JLA: plain random projection of each quadratic form; the nonlinear
  `1/(1 - P_ii)` is not bias-corrected (pytwoway / LeaveOutTwoWay use a ratio
  `P/(P+M)` estimator plus a correction factor). A large-data run should
  compare a few draws counts; the exact path is the reference.
- `exact` keeps three dense J x J matrices (3000 firms = 216 MB); beyond that
  `auto` switches to JLA. A sparse exact path (only the worker-local blocks of
  `G`) would extend the exact range.
- Covariates are partialled out with the full model and the leverages are
  those of the two-way model on `y - x'b` (the authors' convention); the
  leave-out does not re-estimate `b`.
- The motivating paper's step 2 is a regression on a peer mean of *estimated*
  effects; `two_step_bootstrap` gives the bootstrap that re-estimates AKM on
  each resample, but at 260M observations that is a compute question, not
  covered by tests here.
- `sp.stata` places saved variables by index label; a non-unique index is
  refused (not silently misplaced).
- felsdvreg's `cons`, `robust`, `cluster()` (its small-sample factor counts
  every fixed effect), `feffse()`, `takegroup`, `grouponly`, `hat()/orig()`
  are reported as not translated.
- Generated artefacts not regenerated here (per the batch rules): schemas
  (`dump_schemas.py`), parity index, `docs/stats.md`, README counts,
  evidence inventory, source failure modes (`gen_source_failure_modes.py`:
  the new functions raise taxonomy errors with `recovery_hint`), trace
  ledgers. `panel/feols.py` and the translator files gained code; neither is
  on a Track A / orig trace path (`panel/hdfe.py` and `fixest/_fixef.py`,
  which are, were not edited).
