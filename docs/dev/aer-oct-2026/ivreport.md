# AER 116(10) batch, line `ivreport`

Motivating paper: Caprettini, Trew and Voth, "Fighting for Growth" (AER
116(10), Oct 2026). 887 coastal grid cells; IV of labour-saving machine
counts on log Navy recruits per capita with distance to the 12.5 m seabed
line as instrument; coefficients reported standardised with robust SEs;
Anderson-Rubin tests, tF intervals, Conley SEs and Ibragimov-Mueller checks.
StatsPAI already had AR / tF / `sp.weakrobust` / `sp.conley` /
`sp.iv(vce='conley')`. This line adds the missing reporting pieces.

## Proposed CHANGELOG entry (for `## [Unreleased]`)

### Added

- **`sp.standardize_coefs(result, data=None, *, variables=None,
  scale_y=True, weights=None)`.** Standardised (beta) coefficients of an
  OLS, 2SLS or fixed-effects fit. Each slope becomes `b_k * sd(x_k) /
  sd(y)` (Stata `regress, beta`); with an intercept the variables are also
  centred, so the result equals re-running the estimator on z-scored
  variables, which is how AER papers report standardised IV and FE
  coefficients. The SE and both interval ends of a slope are multiplied by
  the same factor and the covariance becomes `J V J'`; t statistics and
  p-values are unchanged under any variance (classical, HC, cluster,
  Conley). Works after `sp.regress` (incl. `weights=`), `sp.iv` /
  `sp.ivreg` (any method, `absorb=`), `sp.feols` and `sp.hdfe_ols`; with
  absorbed effects the sds are those of the raw columns (pass `data=`).
  Against Stata 18: `e(beta)` to 1e-13 (with and without `[aw=]`); regress,
  ivreg2 (robust; cluster small), reghdfe and ivreghdfe on z-scored
  variables, every coefficient and SE to 1e-13.
- **`sp.ibragimov_mueller(data, formula=None, *, cluster, coef,
  estimator='ols', h0=0, alpha=0.05, min_group_size=2, fit_kwargs=None)`.**
  Ibragimov-Mueller (2010, 2016) inference with few heterogeneous groups:
  the coefficient is estimated separately in each of the q groups
  (`sp.regress`, `sp.iv` or any callable) and the q estimates are t-tested
  with q - 1 degrees of freedom. Returns the estimate, SE, t, p, interval
  and the per-group estimates. Warns above alpha = 0.083, the level up to
  which the test is shown to be conservative under heterogeneous groups.
  Refs verified via the Crossref API and OpenAlex
  (`ibragimov2010tstatistic`, `ibragimov2016inference`).
- **`sp.conley_sensitivity(result, data, lat, lon, cutoffs, *, terms=None,
  kernel='uniform', distance='planar', alpha=0.05, inference='normal',
  method='ols', fit_kwargs=None)`.** Conley SEs, intervals and p-values of
  an OLS or 2SLS fit (or a formula) across distance cutoffs, as a long
  table with `.wide()`, `.summary()` and `.plot()`. Each row is
  `sp.conley` at that cutoff; no kernel is re-implemented. Defaults follow
  acreg (planar 111 km/degree distance, z statistics). Against Stata 18
  `acreg` at 25 / 75 / 150 / 300 km, OLS and 2SLS, uniform and Bartlett
  kernels: every SE to 6e-13; the uniform kernel's negative variances at
  300 km are missing in acreg and NaN here.
- **`sp.from_stata` / `sp.stata` translate `weakiv` and `acreg`, and
  `regress, beta`.** `weakiv ivreg2|ivregress 2sls ...` maps to
  `sp.weakrobust` with weakiv's LM-form statistics (AR / K / J / CLR to
  1e-9 against weakiv 2.4.07, i.i.d. / robust / cluster, one and two
  instruments); bare `weakiv` after an IV regression takes the model from
  that regression in `sp.stata`. `acreg ..., spatial latitude() longitude()
  dist()` maps to `sp.regress` / `sp.iv` (`vce='conley'`) or, with
  `pfe1()` / `pfe2()`, to `sp.hdfe_ols` with the Bartlett kernel and the
  panel options; forms with no single sp call are refused with the
  equivalent `sp.conley(...)` call in the message. `regress ..., beta` is
  no longer dropped as display-only: `sp.stata` returns the fit (Stata's
  `e(b)` is unstandardised too, and later commands use it) with Stata's
  `e(beta)` in `model_info['beta']`; `sp.from_stata` adds `report_code`
  (`sp.standardize_coefs(...)`) for the standardised table.

### ⚠️ Correctness

- ⚠️ **`sp.hdfe_ols(vce='conley', conley_kernel='bartlett')` without the
  panel options returned the uniform-kernel SEs.** The spatial-only path
  never read `conley_kernel`. It now applies the Bartlett kernel
  (acreg `..., pfe1() bartlett` to 1e-12; on the test fixture the SE of
  `x` moves from 0.05031 to 0.04171). An unknown kernel name now raises
  instead of being ignored. The uniform-kernel path is unchanged.

### Changed

- `sp.from_stata("regress ..., beta")`: `beta` is no longer listed in
  `ignored_display_options`; the payload carries `postprocess` and
  `report_code` (`sp.standardize_coefs(...)`).
- `tests/test_stata_translation_grammar.py`: the `("reg y x, beta",
  ["beta"])` display-option case is removed (beta is now translated).
- `tests/test_translation.py`: `TIER3_ROUND_TRIPS` gains a `weakiv` and an
  `acreg` round trip (required by the handler-coverage test).

## MIGRATION note

Only for the correctness fix: code that passed
`sp.hdfe_ols(..., vce='conley', conley_kernel='bartlett')` without
`conley_time` got uniform-kernel SEs through 1.41.x; it now gets Bartlett
SEs. To reproduce the old numbers pass `conley_kernel='uniform'`.

## Evidence summary

| Claim | Reference | Tolerance asserted | Observed |
| --- | --- | --- | --- |
| `standardize_coefs` = `e(beta)` | Stata 18 `regress, beta` (plain, `[aw=]`) | rel 1e-10 | 6e-14 |
| `standardize_coefs` = fit on z-scores, all coefs and SEs | regress robust / cluster, ivreg2 robust, ivreg2 cluster small, reghdfe cluster, ivreghdfe robust small | b abs 1e-10, SE rel 1e-9 | <= 4e-14 |
| `standardize_coefs` weighted OLS = weighted OLS on [aw] z-scores | analytic (Python) | abs 1e-12 / rel 1e-10 | rounding |
| slope t / p unchanged | identity (bit-equal) | `==` | exact |
| `ibragimov_mueller` | hand computation on noise-free groups | rel 1e-10 | exact |
| `ibragimov_mueller` size | seeded MC: q = 8, noise sd 0.5-7.5, cluster shocks, heteroskedastic, 1,500 reps | <= 0.05 + 3 MC SE (0.067) | 0.03 |
| `conley_sensitivity` | Stata acreg, 4 cutoffs x OLS / 2SLS x uniform / Bartlett | SE rel 1e-9 | <= 6e-13 |
| `conley_sensitivity` tiny cutoff | HC0 sandwich (identity) | rel 1e-10 | exact |
| `hdfe_ols` Bartlett + pfe | acreg `pfe1(fe) bartlett` | rel 1e-9 | ~1e-15 |
| `sp.conley(sp.iv(absorb=))` (the recipe in the acreg refusal) | acreg IV `pfe1(fe)` | rel 1e-9 | ~1e-14 |
| weakiv translation | weakiv 2.4.07 (5 specs incl. post-estimation form) | stats rel 1e-9; CLR p rel 1e-7 (Simpson rule) | 1e-14; 2e-9 |

Fixture: `tests/reference_parity/_fixtures/ivreport_fx.csv`
(`_generate_ivreport_data.py`), reference `ivreport_Stata.json`
(`_generate_ivreport_Stata.do`, Stata 18 MP, run from the repo root).

## Convention differences (documented, not bugs)

- weakiv confidence sets: weakiv interpolates on a grid (robust / cluster)
  or reports the closed-form Wald-type interval (i.i.d.); `sp.weakrobust`
  inverts the reported test exactly. Endpoints differ in the third or
  fourth digit; statistics agree to 1e-9. Already in iv.md.
- weakiv `small` and `md` change the statistics (Stata 18: cluster AR
  3.5185 vs 3.1984 with `small`); not translated, reported as dropped.
- acreg reports z statistics; `sp.hdfe_ols` refers the same Conley SEs to
  t(n - k). `sp.regress` / `sp.iv(vce='conley')` and
  `sp.conley_sensitivity` (default) use the normal like acreg.
- `sp.conley_sensitivity` defaults to `distance='planar'` (acreg, and
  `sp.iv(vce='conley')`), whereas `sp.conley`'s own default is haversine.

## Open issues

- **No `beta=` on `sp.regress`.** Adding it would touch
  `regression/ols.py` (Track A traced, core-frozen signature). The
  post-estimation function covers OLS, IV and HDFE uniformly and the
  translator uses it. Decide whether a `beta=` shortcut is still wanted.
- acreg forms refused (with the sp.conley recipe in the error): 2SLS with
  Bartlett / panel options / absorbed effects; OLS with Bartlett or panel
  options and no absorbed effect. The recipes for the cross-section forms
  are checked against acreg; the panel recipes rely on the existing
  `test_conley_acreg_spacetime_parity.py` (OLS) and are not checked for
  2SLS panels.
- acreg's `pfe1()` without `dropsingletons` keeps singleton groups (read
  from the ado's syntax; the fixture has none, so this is not exercised).
- `sp.ibragimov_mueller` has no Stata / R reference implementation to
  compare with (it is a t-test on group estimates); evidence is analytic
  plus a size simulation.
- `sp.conley_sensitivity` with `distance='planar'` builds an n x n matrix
  per cutoff; the 887-cell AER application is cheap, 20k+ rows are not.

## Files touched on parity-traced paths

- `src/statspai/__init__.py` (exports) is on both Track A trace ledgers
  (it is executed by every parity module), so both ledgers need
  re-recording at merge. No estimation file on a trace was edited:
  `regression/ols.py` and `regression/iv.py` are untouched;
  `panel/feols.py` (the Bartlett fix) and `inference/conley.py` are not on
  either ledger.
- Generated artifacts not regenerated here (per the batch rules): schemas,
  parity index, docs/stats.md, README counts, evidence inventory, source
  failure modes, trace ledgers.
