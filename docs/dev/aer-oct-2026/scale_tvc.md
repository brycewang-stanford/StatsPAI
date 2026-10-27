# AER 116(10) line `scale_tvc`: census-scale absorbed IV, and `stcox, tvc()`

Two items from the Oct 2026 AER batch, 2026-10-28.

- (A) Scale. Goldman, Gracie and Porter run `ivreghdfe married (tight = z) x, absorb(group#tract18#sex) cluster(county18)` on 15 to 23 million people.
- (B) `stcox ..., tvc() texp()`. Hegde, Ljungqvist and Raj ("Racial Promotion Gaps") need time-varying coefficients for time-to-promotion data.

No literature reference is added anywhere in this line.

## Proposed CHANGELOG entry (Unreleased)

### Added

- `sp.cox(tvc=, texp=)`: time-varying coefficients, Stata `stcox, tvc(varlist) texp(exp)`. Each `tvc` covariate `z` adds the regressor `z * g(t)`, reported as `tvc:z` (Stata's `tvc` equation). `texp` takes a Stata expression in `_t`, read by a closed-grammar parser with no `eval`: `_t`, `ln(_t)`, `_t^2`, `_t > 5`, `min(_t, 4)` and so on. It also takes a Python callable. Estimation is exact. Each row is split at the failure times of its stratum and fitted through the counting-process risk sets of `start=`. Efron and Breslow ties, weights, strata, delayed entry, and robust / clustered variances all work. The robust variance uses the input row as its unit, as Stata does. Coefficients and SEs match Stata 18 to 3.3e-7 relative on ten specifications and R `coxph(tt=)` to 2.5e-8 on six (`tests/reference_parity/test_cox_tvc_parity.py`). `ph_test()` refuses tvc fits, as R `cox.zph` refuses `tt()`. `design=` is refused. A split above 1e8 episodes raises an error rather than exhausting memory.
- `sp.from_stata` / `sp.stata` translate `stcox ..., tvc() texp()`. The `texp` expression passes through and is checked with the same parser. `texp()` without `tvc()` is refused, and so is a factor-variable `tvc()` (`tests/test_stata_translation_stcox_tvc.py`).
- `benchmarks/hdfe/iv_census_scale_bench.py`: a census-scale absorbed-2SLS benchmark (1M / 5M / 20M rows) of `sp.hdfe_ols` with and without Rust, R `fixest::feols` and Stata `ivreghdfe`.

### Changed

- `sp.hdfe_ols` with clusters is faster at scale, with bit-identical output. Two things changed. The check that a fixed effect is nested in the cluster was a Python loop over the rows, about 51 s at 20M rows on a loaded machine; it is now vectorised (1.6 s). The IV path's cluster sums use `np.bincount` (about 3x faster than `np.add.at`, same order of additions), and the cluster column is factorised once instead of six times. End to end at 20M rows: 68.6 s to 47.3 s (Rust path) and 62.4 s to 37.9 s (NumPy path).

### ⚠️ Correctness

- `sp.hdfe_ols("y ~ x | fe | d ~ z", cluster=c)` charged no degree of freedom for the swept-out constant when every absorbed fixed effect is nested in the cluster. `ivreghdfe` charges one (`e(sdofminus) = 1` with `e(df_a) = 0`). Clustered SEs, and the first-stage, KP and CD statistics' finite-sample factor, were too small by about 1/(2N) relative: 6.4e-4 at N = 785, 5e-7 at 1M, 1e-7 at 5M. The paper's own specification (`absorb(group#tract#sex)` with tracts nested in counties) is such a design. Partially nested and unnested designs were already right. Evidence: `tests/reference_parity/test_hdfe_iv_nested_sdof.py` covers four absorb designs against Stata 18 ivreghdfe 1.1.4 at 1e-7. At 1M and 5M census rows the SEs now equal Stata's to 8e-13, and fixest's to 3e-12 at 20M. MIGRATION.md needs a line.

## A. Census-scale absorbed IV

### DGP

The DGP is in the benchmark docstring. At 20M rows it has 70,175 tracts with lognormal sizes, nested in 3,051 counties (the clusters). The absorbed cell `grp x tract x sex` has 274,779 levels. The outcome rate is 3.0%. The first-stage clustered F is 934 at 1M, 4,377 at 5M and 18,510 at 20M. Every float is dyadic (`k/1024`), so all three CSV readers parse identical doubles.

### Results

Results are in `benchmarks/hdfe/results/iv_census_scale.json` (after) and `iv_census_scale_before.json` (HEAD code), with machine info. Machine: Apple M3, 8 cores, 24 GiB. Four threads were used for StatsPAI and fixest; Stata MP is licensed for 2 cores.

**The machine was shared and heavily loaded.** The 1-minute load average was 73 to 107 during the runs (other sessions' test suites). Treat the wall times as indicative only. The before and after runs were interleaved per size so that both saw similar load. Fit time excludes CSV loading; each engine first ran a 20k-row warm-up fit.

| rows | engine | fit (s) | peak RSS (GiB, process) |
| --- | --- | --- | --- |
| 1M | StatsPAI Rust, before / after | 2.39 / 1.26 | 0.71 / 0.72 |
| 1M | StatsPAI NumPy, before / after | 2.13 / 1.19 | 0.70 / 0.68 |
| 1M | fixest 0.14.0 (4 thr) | 3.21 | 0.68 |
| 1M | Stata ivreghdfe (2 cores) | 12.7 | 0.90 |
| 5M | StatsPAI Rust, before / after | 13.0 / 11.0 | 3.04 / 2.32 |
| 5M | StatsPAI NumPy, before / after | 13.0 / 9.1 | 2.96 / 2.98 |
| 5M | fixest | 7.9 | 1.89 |
| 5M | Stata ivreghdfe | 60.5 | 2.73 |
| 20M | StatsPAI Rust, before / after | 68.6 / 47.3 | 5.61 / 4.41 |
| 20M | StatsPAI NumPy, before / after | 62.4 / 37.9 | 6.66 / 6.78 |
| 20M | fixest | 39.4 | 6.31 |
| 20M | Stata | not run (est. ~4-5 min and ~11 GiB from the 5M run; skipped to keep memory safe on a shared machine) | |

How to read the table:

- Rust and NumPy give the same numbers, bit for bit. With a single absorbed dimension the demeaning is one closed-form group-mean sweep, so the Rust kernel does not enter the fit. Their timing differences here are load noise.
- Peak RSS differences between StatsPAI runs come mostly from the pyarrow CSV read: RSS after load is 2.9 to 4.7 GiB at 20M, from run to run. They do not come from the fit. Inside the fit the traced allocation peak is 239 bytes per row, about 30 doubles (tracemalloc, 1M rows). It was 279 before two copies were removed.

### Numerical agreement (relative to the StatsPAI Rust fit)

| rows | vs | coef | SE |
| --- | --- | --- | --- |
| 1M | Stata | ≤ 2.0e-15 | ≤ 7.8e-15 |
| 1M | fixest | ≤ 3.1e-14 | ≤ 6.4e-14 |
| 5M | Stata | ≤ 6.9e-13 | ≤ 8.1e-13 |
| 5M | fixest | ≤ 8.0e-13 | ≤ 9.1e-13 |
| 20M | fixest | ≤ 2.9e-12 | ≤ 3.1e-12 |
| all | NumPy path | 0 | 0 |

Before the sdof fix, the SEs differed from both Stata and fixest by exactly one degree of freedom: 5.0e-7 at 1M and 1.0e-7 at 5M (`iv_census_scale_before.json`). fixest (`fixef.rm = "singleton"`, default `ssc`) and ivreghdfe agree with each other.

### Profile and fixes

A cProfile at 5M rows (HEAD code) took 21 s in total under load. The main costs were:

- 6.2 s in `hdfe._codes_nested_in_cluster`, a per-row Python loop. It cannot exit early on a nested design.
- 1.8 s in `np.add.at` inside `_hdfe_iv._meat` (5 calls).
- 1.4 s in 13 `pd.factorize` calls, 6 of them on the cluster column.

After the fixes the same profile took 7.4 s. Microbenchmarks:

- Nested check at 20M rows: 51 s before, 1.6 s after.
- Cluster sums at 5M × 2: `add.at` 1.6 to 1.8 s, `bincount` 0.45 to 0.70 s.

There was no string interaction keys problem: interacted FEs were already integer-factorised (`_factorize_multi`). The FE is never densified.

Memory changes: `demean(copy=False)` on the freshly masked stack, freeing the unswept stack, and reusing `X_arr` as the exogenous block instead of rebuilding it.

### Bit-identity proof (default outputs)

1. Nine `sp.hdfe_ols` fits were captured before and after the performance edits, on both the Rust and the no-Rust paths. They were compared with `np.array_equal` on params, SE, full vcov, all `iv_diagnostics` and `cluster_info`. All were bit-identical. The fits covered:
   - census IV with cluster, robust and iid variances;
   - overidentified IV with nested and non-nested clusters;
   - two endogenous regressors;
   - OLS with a nested cluster, two-way clustering, and the census design.

   This comparison was made before the sdof fix, which changes the all-nested IV fits on purpose.
2. Three Track A modules have trace hashes on edited files: `24_coxph` (`survival/models.py`) and `68_demean_within` / `75_stacked` (`panel/hdfe.py`). Each was rerun with `STATSPAI_SKIP_RUST=1` on the HEAD source copy and on the modified source. The two `*_py.json` outputs are byte-identical. Both differ from the committed files by at most 9.2e-15 relative (environmental, within the 1e-9 reproducibility tolerance), so the committed files were restored with `git checkout`. Trace ledgers were not re-recorded, as instructed. They will need re-recording centrally because the hashes of `survival/models.py` and `panel/hdfe.py` changed. `orig_parity` has no module on these files.
3. Eight `sp.cox` fits without tvc were compared before and after on params, SE, all diagnostics, baseline hazard and `ph_test()`. All were bit-identical. The fits covered efron, breslow, robust with `ssc='stata'`, cluster, strata, pweights, start, and start with weights and cluster.

### Remaining cost (not fixed, by choice)

`Zf.T @ X` (`gemm` with two N×2 operands) takes 0.5 to 1.7 s at 5M in OpenBLAS, against 0.07 s for `Zf.T @ Zf` (`syrk`). Computing all the cross products from one stacked `S.T @ S` would remove most of it. That changes the BLAS kernel and so the last bits of every IV coefficient, so it was left for a decision. No Rust bottleneck was found: the single-FE demean is the NumPy/numba sweep.

## B. `sp.cox(tvc=, texp=)`

- New `survival/_cox_tvc.py` (the parser, `split_at_failures`, `MAX_EPISODES`), wired into `sp.cox` in `survival/models.py`. `ph_test()` refuses tvc fits.
- New signature: `sp.cox(..., design=None, tvc: str | list[str] | None = None, texp: str | Callable | None = None)`.
- `sp.cox` is auto-registered from its docstring. `sp.function_schema('cox')` already shows `tvc` and `texp`, and the registry-params drift test passes. No hand-written `ParamSpec` exists to extend.
- Result: the main coefficients keep their names and the tvc ones are `tvc:z`. `model_info['tvc' / 'texp']` is set, and `data_info['n_episodes']` alongside `nobs` (input rows).

### Evidence

Fixtures are in `tests/reference_parity/_fixtures/`:

- `cox_tvc_cont.csv`: 400 rows, continuous times.
- `cox_tvc_ties.csv`: 600 rows with 15 distinct integer times, strata, 40 clusters, pweights and delayed entry.

They come from `_generate_cox_tvc_data.py`. The reference scripts are `_generate_cox_tvc_Stata.do` (Stata 18, tightened ml tolerances, output `cox_tvc_Stata.csv`) and `_generate_cox_tvc_R.R` (survival 3.8-3, output `cox_tvc_R.json`).

| spec (Stata) | max rel b | max rel SE |
| --- | --- | --- |
| c1 breslow, `_t` | 2.4e-11 | 2.3e-12 |
| c2 efron, `ln(_t)` | 5.4e-11 | 8.2e-12 |
| c3 efron, `_t^2`, tvc var not in main, `vce(robust)` | 3.3e-7 | 9.8e-8 |
| c4 breslow, `min(_t, 4)` | 8.0e-11 | 7.4e-12 |
| t1 efron ties, `_t` | 1.3e-8 | 3.0e-9 |
| t2 breslow, `ln(_t)`, two tvc, `vce(robust)` | 3.8e-10 | 7.5e-11 |
| t3 efron, `strata(grp)`, `vce(cluster cl)` | 1.6e-8 | 6.3e-9 |
| t5 efron, `_t > 5` | 1.4e-7 | 3.5e-8 |
| t4 breslow, `[pw=w]` | 1.5e-8 | 6.5e-9 |
| t6 efron, `time0(entry)`, `ln(_t)` | 3.6e-11 | 4.3e-12 |

For c3, our log likelihood is the larger by 5.7e-10, so that gap is Stata's stopping point. R `coxph(tt=)` agrees on c1, c2, t1, t2, t3 and t6 to at most 2.5e-8 (c2, R's `eps`); the rest are at most 4.4e-13. R's robust (`cluster=id`) and clustered SEs equal Stata's divided by sqrt(600/599) and sqrt(40/39), which the test asserts.

Other tests:

- A hand split at the failure times, fitted with `start=` and `cluster=id`, equals the tvc fit to 1e-9.
- `texp='1'` reproduces the proportional coefficient.
- The callable form equals the string form bit for bit.
- The parser follows Stata precedence (`-_t^2 = -(t^2)`), and it refuses `x1*_t`, `a^b^c`, `eval(...)` and similar.
- A non-finite `g` at a failure time raises `DataInsufficient`. Misuse raises `MethodIncompatibility` with `recovery_hint`.
- `sp.stata` replays nine of the Stata specs, all within 1e-6.

## Tests run

- New tests: `test_cox_tvc_parity.py` (34 passed), `test_stata_translation_stcox_tvc.py` (14 passed) and `test_hdfe_iv_nested_sdof.py` (5 passed). The existing `test_hdfe_iv_ivreghdfe.py` still passes (10). That is 63 passed in total.
- Relevant suite: `grep -rlE 'cox|hdfe_ols|tvc|stcox|ivreghdfe|Absorber|_cluster_effective_fe_dof|absorb_ols' tests` (parity script trees excluded), plus `test_registry_param_drift`, `test_core_api_freeze`, `test_stata_translation_grammar`, `test_source_failure_modes`, `test_enum_seeds`, `test_rust_backend` and `test_hdfe_varying_slopes`. That is 134 files: 5032 passed, 68 skipped, 2 xfailed, 0 failed.
- Gates: `quality_gate.py flake8` passed. `error_taxonomy_audit.py --check` OK (6714 taxonomy raises, 1780 generic). `silent_except_audit.py --check` gave 155, the baseline. black and isort were applied to the changed files.

## Open items

1. Trace ledgers: `24_coxph`, `68_demean_within` and `75_stacked` need a central re-record (outputs proven byte-identical; see A).
2. The sdof fix changes defaults. It needs CHANGELOG ⚠️ and MIGRATION lines. It may need `scripts/gen_source_failure_modes.py`, schemas and evidence views regenerated centrally; `test_source_failure_modes.py` passed as-is.
3. Weighted Cox log likelihood: under `[pw=]`, `sp.cox`'s `Log-likelihood` is not Stata's `e(ll)` on the same data without tvc (-3664.24 vs -2348.95). Coefficients and SEs agree. This predates this line and was not investigated.
4. The tall-skinny `gemm` cost in `_hdfe_iv` (see "Remaining cost") needs a decision on whether a last-bit change is acceptable.
5. Stata at 20M rows was not run (memory safety on the shared machine). The wall times above should be re-taken on an idle machine before anyone quotes them.
6. A tvc split is O(rows × failure times inside each interval). Continuous-time data with many subjects can reach the 1e8 guard; grouping times on a coarser grid is the documented workaround. A risk-set formulation without splitting would remove the limit but is not implemented.
