# Campaign phase 3: time-series family

Worktree `pc-timeseries`. Every function below was run against its reference
on the same CSV bytes (`tests/reference_parity/_fixtures/ts_*.csv`, written by
`_fixtures/_generate_timeseries_data.py`). Reference numbers come from two
generator scripts that are committed alongside the fixtures:

- R: `tests/reference_parity/_generate_timeseries_R.R` writes `_fixtures/timeseries_R.json`
  (R 4.5.2, urca 1.3.4, aTSA 3.1.2.1, vars 1.6.1, strucchange 1.5.4, sandwich 3.1.1,
  rugarch 1.5.6, plm 2.6.7; the versions are stored in the JSON).
- Stata: `tests/reference_parity/_fixtures/_generate_timeseries_stata.do` writes
  `_fixtures/timeseries_Stata.json` (Stata 18 MP built-ins, plus SSC egranger 1.0.6
  and itsa 1.0.0 installed into the private `_fixtures/_ado_timeseries/`).
- Test: `tests/reference_parity/test_timeseries_R_parity.py` (69 tests, all pass, about 35 s).

## 1. Per-function results

Where a cell says "(stat)", that value is the test statistic. None of these functions reports a coefficient SE, except garch.

| function | reference (version) | class | max rel err, est / SE (observed) | test |
| --- | --- | --- | --- | --- |
| `johansen` | `urca::ca.jo` 1.3.4 (ecdet none/const/trend, K = 2, 3); Stata 18 `vecrank` (all 5 trend cases, lags 2, 3) | 2 (defects fixed), then 1 | eigenvalues, trace and max-eig: R 4.5e-14, Stata 8.2e-14; ca.jo eigenvectors 1e-8; critical-value table equals Stata `_vecgetcv` exactly | test_timeseries_R_parity.py |
| `engle_granger` | Stata `egranger` 1.0.6 (SSC, Schaffer); R `lm` + `urca::ur.df(type="none")`; `aTSA::coint.test` | 2, then 1 | Z(t): R 2.1e-13, Stata 2.0e-13; MacKinnon critical values vs egranger: 0 (exact) | same |
| `irf` | `vars::irf` 1.6.1 (ortho / simple / cumulative); Stata `irf create` after `var` and after `var, dfk` (oirf / irf / coirf) | 1 (options added) | R 2.6e-14; Stata 1.3e-14 | same |
| `granger_causality` | Stata `vargranger` after `var`, `var, small` and `var, small dfk` (all 9 rows); `vars::causality` (bivariate) | 1 (chi2 + joint test added) | Stata 4.1e-15; R F 3.9e-16 (vars' df2 is documented and reproduced) | same |
| `cusum_test` | `strucchange::efp(type="Rec-CUSUM")` + `sctest` 1.5.4; Stata `estat sbcusum` | 1 (statistic, p-value and exact boundary added; `alpha` was silently ignored) | path / stat / p vs R 7.1e-14; stat vs Stata 5.3e-15; boundary coefficient vs Stata 2.5e-6 (open, grade B, see §5) | same |
| `structural_break` | `strucchange::Fstats` / `breakpoints` 1.5.4; Stata `estat sbsingle` (swald) | 2 (grid off-by-one), 1 for sup-F and `method='global'` | F path, RSS-by-m and BIC-by-m vs R 5.9e-14; sup-Wald vs Stata 1.9e-15; break dates equal for every m | same |
| `its` | R `lm` + `sandwich::NeweyWest(lag=4, prewhite=FALSE, adjust=FALSE/TRUE)` 3.1.1; Stata `newey, lag(4)`; SSC `itsa` 1.0.0 | 1 (Stata small-sample option added) | coef and SE: R 7.9e-14, Stata newey 7.6e-14; itsa 1e-6 (glm2 IRLS, see test) | same |
| `garch` | Stata `arch, arch(1) garch(1)`, vce oim / opg / robust; `rugarch::ugarchfit` sGARCH(1,1) 1.5.6 | 2 (objective and optimiser fixed), then **aligned** | Stata: our log-likelihood at Stata's b equals e(ll) exactly, at our optimum 1.2e-13, b 8.2e-6, SE 1e-5 (robust after Stata's N/(N-1)). rugarch: log-likelihood at their parameters 8.5e-16; parameters 2e-5 (their optimiser); SE 0.7% (their numerical Hessian, §5) | same |
| `panel_unitroot` | `plm::purtest` 2.6.7 (levinlin, ips, madwu, invnormal, logit, Pm, hadri Hcons TRUE/FALSE; exo intercept/trend/none; dfcor TRUE/FALSE); Stata `xtunitroot` llc / ips / fisher dfuller / hadri (plus robust) | 2 (rewritten; LLC, IPS and Fisher were wrong), then 1 | every plm statistic 9.1e-15; Stata IPS, Fisher P/Z/Pm, Hadri both, LLC c and n 8.0e-15; Stata LLC with trend and Stata L* are T4 / documented, see §5 | same |
| `bvar` | Stata `bayes, minnfixedcovprior: var` (arcov, varcov, custom lambdas), 100,000 Gibbs draws | 2 (defect fixed), then **5 (T3)** | Sigma0 vs e(arcov)/e(varcov) 1e-10 (exact); all 63 posterior means within 4 MCSE; posterior SDs within 5·sd/√(2N) | same |

The earlier analytical-tier tests (`test_timeseries_parity.py`,
`test_engle_granger_parity.py`, `test_cusum_test_parity.py`,
`test_structural_break_parity.py`) still pass unchanged.

## 2. Defects

Each "before" number comes from `git archive HEAD` run on the same fixture bytes.

1. **`engle_granger`: statistic compared against the wrong critical values.** Found first-divergence: the Z(t) against `egranger` / `ur.df(type="none")`. The residual ADF regression included a constant (`trend='c'` appended one to the step-2 design). The MacKinnon Engle-Granger critical values are tabulated for step 2 **without** deterministic terms, because those terms belong in step 1.
   - Critical values were asymptotic three-decimal numbers, and wrong for N ≥ 3 (N=3 5%: −3.78, where the correct asymptotic value is −3.74066).
   - `trend='ct'` was silently ignored in both steps.
   - `alpha` was silently ignored (the decision was always made at 5%).
   - NaNs were not dropped.
   - Fix: step-2 ADF without deterministic terms; `trend` c/ct/ctt now enters step 1, as in `egranger` trend/qtrend; critical values are the MacKinnon (2010) response surface at T = n−1, transcribed from the paper by a parser and matched cell by cell against egranger's table; `alpha` ∈ {.01, .05, .10}.
   - Before → after on y1~y2, L=0: Z −9.6167 → **−9.6364** (egranger −9.6364); 5% critical value −3.34 → −3.3608.
   - With `trend='ct'`: Z −8.6168 → −8.4234; 5% critical value −3.34 → −3.8190.
   - Default output changes.
   - Also: statsmodels' `tau_2010s` has two typos against the paper (N=2 1% β₂ −33.527, where the paper has −22.527; N=3 5% β₁ −8.5632, where it has −8.5631). We do not use it.
2. **`johansen`: critical values.**
   - Max-eigenvalue 5% values for k−r = 3 and 4 were 21.12 and 27.42. Stata and Osterwald-Lenum have 20.97 and 27.07.
   - `trend='n'` and `trend='ct'` used the case-3 (unrestricted constant) table. For 'n', trace k−r=3 should be 24.31, not 29.68.
   - k=1 trace used 3.84 instead of 3.76.
   - `alpha` and an invalid `test` were silently ignored (the old code fell back to max-eig).
   - Fix: Stata's full 5-case × 11-row × 2-level table, with restricted-constant and restricted-trend cases added ('rc', 'rt', and Stata aliases); eigenvectors normalised the ca.jo way.
   - Statistics were already right: the gap was only in the decision, not the numbers.
3. **`garch`: objective and optimiser.**
   - The pre-sample variance was `var(eps)` (demeaned). Stata `arch0(xb)` and rugarch both use `mean(eps²)` at the current μ.
   - Nelder–Mead stopped short.
   - `forecast()` ignored every ARCH/GARCH lag beyond the first.
   - Fix: `presample='stata'` (default) or `'rugarch'`; analytic score, BFGS plus Newton polish (score < 1e-10); `vce='oim'|'opg'|'robust'`; `garch_loglik()` exposed.
   - Before → after log-likelihood: −2144.43397943 → **−2144.43398728** (Stata e(ll) −2144.43398728).
   - Before → after μ: 0.0629375 → 0.0629347 (Stata 0.0629347).
   - Default output changes (slightly).
4. **`panel_unitroot`: three of the four tests were not the named test.**
   - LLC was `sqrt(N)·mean(t_i) + sqrt(N)` ("mu*=-1, sigma*=1"). It had no long-run variance, no pooled regression and no adjustment table.
   - IPS used rounded asymptotic moments that did not depend on T or lags. With `trend='n'` it used invented moments.
   - Fisher combined **normal-CDF** p-values instead of MacKinnon.
   - Hadri silently treated `trend='n'` as a trend.
   - Fix: a full rewrite following Stata `_xturllc` / `_xturips` / `_xturfisher` / `_xturhadri` and plm's `purtest` source, with `convention='stata'|'plm'`, `dfcor` and `robust` options. Tables are checked cell by cell against plm. MacKinnon 1994 p-values are checked against plm's `padf` and statsmodels.
   - Before → after on `ts_panel.csv` (lags=1): LLC −7.80 → **−4.656** (xtunitroot −4.656); IPS −6.238 → **−6.109** (−6.109); Fisher P 239.1 → **117.09** (117.09).
   - On the random-walk panel, LLC went from −0.833 (p = 0.20) to −0.0016 (p = 0.50).
   - Default output changes (large).
5. **`bvar`: prior assembled from the wrong equation.**
   - One prior-variance vector was built in a loop over equations and overwritten each time, so every equation used the **last** equation's prior. The own-lag variance was applied to the wrong variable, and the estimates depended on column order: the gdp own-lag was 0.901 with columns (gdp, infl, rate) and 0.637 after reordering.
   - The prior variances were also implicitly scaled by s_k² a second time.
   - Fix: Litterman's original Minnesota prior with fixed Σ₀, in the formulation of Stata `bayes, minnfixedcovprior: var`: AR(p) variances with divisor n, constant prior variance s_i²(λ1λ4)², and the new λ3 (lag decay), λ4 and `sigma='ar'|'var'`. The posterior is solved in closed form as a system.
   - Before → after gdp own-lag: 0.901 → **0.6382** (Stata MCMC 0.6383 ± 0.00017). The result is now permutation-equivariant.
   - Default output changes (large).
6. **`cusum_test`: boundary.** `alpha` values other than .01/.05/.10 silently used 0.948. The boundary coefficient is now the exact root of the crossing probability (0.9478982 at 5%). The `statistic` and `p_value` keys were added. The default boundary array moves by 1e-4 relative.
7. **`structural_break` sup-F grid.** The candidate first-regime sizes stopped at n−h−1; strucchange's `Fstats(from=0.15)` includes n−h. The sup-F only changes when the maximum sits at the last point. `method='global'` was added (strucchange `breakpoints` dynamic programming plus BIC); the default `'bai-perron'` (break-at-a-time) is unchanged. The module docstring used to call the default equivalent to `breakpoints()`, and no longer does.

These are options added with defaults unchanged, not defects: `irf(cumulative=, sigma_df=)`, `granger_causality` chi2 / joint `causing=[...]`, `its(hac_small_sample=)`.

## 3. Proposed promotion records (`_FROZEN_PROMOTIONS`)

```python
"johansen": {
    "status": "bit-exact",
    "reference": "urca::ca.jo 1.3.4; Stata 18 vecrank",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "urca": "1.3.4", "Stata": "18"},
    "tolerance": "eigenvalues, trace & max-eigenvalue statistics 1e-11 rel vs ca.jo, 1e-10 vs vecrank (observed 8.2e-14); Osterwald-Lenum table equal to Stata _vecgetcv cell by cell",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "lags = ca.jo K - 1 = vecrank lags() - 1. trend 'c'/'rc'/'rt' vs ca.jo ecdet none/const/trend (K = 2, 3); all five vecrank trend() cases vs Stata. ca.jo ships a different critical-value table (not asserted); ours is Stata's.",
},
"engle_granger": {
    "status": "bit-exact",
    "reference": "egranger 1.0.6 (Stata SSC); urca::ur.df on lm residuals; aTSA::coint.test",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "urca": "1.3.4", "aTSA": "3.1.2.1", "Stata": "18", "egranger": "1.0.6"},
    "tolerance": "Z(t) and step-1 coefficients 1e-10 rel (observed 2.1e-13); MacKinnon (2010) critical values 1e-12 vs egranger",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "Six cases: 2 and 3 series, lags 0/1/2, trend c/ct/ctt (egranger trend/qtrend). Residual ADF without deterministic terms; critical values at T = n - 1.",
},
"irf": {
    "status": "bit-exact",
    "reference": "vars::irf 1.6.1; Stata 18 irf create",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "vars": "1.6.1", "Stata": "18"},
    "tolerance": "1e-10 rel vs vars, 1e-9 vs Stata irf file (observed 2.6e-14)",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "Orthogonalised, simple and cumulative responses. Residual covariance divisor T (Stata var) or T - m (vars::Psi, Stata var, dfk) via sigma_df.",
},
"granger_causality": {
    "status": "bit-exact",
    "reference": "Stata 18 vargranger; vars::causality 1.6.1",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "vars": "1.6.1", "Stata": "18"},
    "tolerance": "chi2 and F 1e-10 rel (observed 4.1e-15), p-values 1e-8",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "All nine vargranger rows (incl. ALL) after var, var small, var small dfk. vars::causality F equal; its df2 is the system K(T - m), reproduced.",
},
"cusum_test": {
    "status": "bit-exact",
    "reference": "strucchange::efp(type = 'Rec-CUSUM') + sctest 1.5.4; Stata 18 estat sbcusum",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "strucchange": "1.5.4", "Stata": "18"},
    "tolerance": "process, statistic and p-value 1e-10 rel (observed 7.1e-14); Stata boundary constants 3e-6",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "Stata's printed boundary coefficients differ from the root of strucchange's closed-form crossing probability by <= 2.5e-6 relative; mechanism not verified. The statistic agrees at 1e-10.",
},
"structural_break": {
    "status": "bit-exact",
    "reference": "strucchange::Fstats / breakpoints 1.5.4; Stata 18 estat sbsingle",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "strucchange": "1.5.4", "Stata": "18"},
    "tolerance": "F path, RSS and BIC by number of breaks 1e-10 rel (observed 5.9e-14); break dates equal",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "method='sup-f' vs Fstats(from = 0.15) and sbsingle swald (= k x sup-F); method='global' vs breakpoints(h = 0.15, breaks = 5). The default method='bai-perron' (break-at-a-time) and the simulated sup-F p-values are not covered.",
},
"its": {
    "status": "bit-exact",
    "reference": "lm + sandwich::NeweyWest 3.1.1; Stata 18 newey; itsa 1.0.0 (SSC)",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "sandwich": "3.1.1", "Stata": "18", "itsa": "1.0.0"},
    "tolerance": "coefficients and Newey-West SE 1e-10 rel (observed 7.9e-14); itsa 1e-6 (glm2 IRLS)",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "Bartlett L = 4, no prewhitening. Default = NeweyWest(adjust = FALSE); hac_small_sample=True = NeweyWest(adjust = TRUE) = Stata newey.",
},
"panel_unitroot": {
    "status": "bit-exact",
    "reference": "plm::purtest 2.6.7; Stata 18 xtunitroot",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "plm": "2.6.7", "Stata": "18"},
    "tolerance": "all statistics 1e-10 rel (observed 9.1e-15)",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "LLC, IPS W-t-bar, Fisher (P, Z, L*, Pm), Hadri (both variants); convention='plm' vs purtest (dfcor both ways), convention='stata' vs xtunitroot. Stata LLC with trend uses sigma* = .971 at T = 40 (plm .871); rebuilt from our quantities. Stata L* uses 5N+3 in its scale constant, plm 5N+4.",
},
"garch": {
    "status": "aligned",
    "reference": "Stata 18 arch; rugarch::ugarchfit 1.5.6",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "rugarch": "1.5.6", "Stata": "18"},
    "tolerance": "log-likelihood at the reference optimum 1e-12 (observed 8.5e-16); vs Stata: b 1e-5, SE 2e-5 (observed 8.2e-6 / 1e-5); vs rugarch: params 5e-4, SE at their parameters 1e-2",
    "sides": ["py", "R", "Stata"],
    "test": [
        "tests/reference_parity/test_timeseries_R_parity.py",
        "tests/reference_parity/_fixtures/timeseries_R.json",
        "tests/reference_parity/_fixtures/timeseries_Stata.json",
    ],
    "note": "Same objective on both sides (our log-likelihood at their parameters equals theirs). Parameter gaps are their optimiser's: both references stop below our optimum (Stata by 2.6e-10). Stata vce(robust) carries N/(N-1). rugarch SEs come from a second difference of the log-likelihood (.hessian2sided, step eps^(1/3)|x|) and are about 0.5% noisy; ours agree with Stata's to 4e-6 at the same parameters.",
},
```

`bvar` is **not** proposed for a cross-language grade. Its evidence is T3: our closed form agrees with Stata's Gibbs posterior within Monte-Carlo error, plus exact Σ₀ and identity tests. If the index gets a note field for T3 evidence, use:
"closed-form posterior vs Stata bayes, minnfixedcovprior: var (100,000 draws): means within 4 MCSE, SDs within 5 sd/sqrt(2N); Sigma0 exact."

## 4. Proposed CHANGELOG / MIGRATION

**⚠️ Correctness**

- `sp.panel_unitroot`: rewritten.
  - LLC was not the Levin-Lin-Chu statistic (no long-run variance, no pooled regression, no adjustment).
  - IPS used T-independent rounded moments.
  - Fisher combined normal p-values instead of MacKinnon (1994) p-values.
  - Now matches Stata `xtunitroot` (default `convention='stata'`) and `plm::purtest` (`convention='plm'`) to 1e-10.
  - IPS and Hadri with `trend='n'` now raise.
- `sp.engle_granger`:
  - The residual ADF regression no longer includes a constant.
  - `trend` ('c'/'ct'/'ctt') now acts on the first step.
  - Critical values are MacKinnon (2010) finite-sample.
  - `alpha` ∈ {0.01, 0.05, 0.10} is honoured.
  - Matches Stata `egranger` and `urca::ur.df`.
- `sp.johansen`: critical values corrected (max-eigenvalue k−r=3,4; separate tables for 'n' and 'ct'). The statistics were unchanged. Added the restricted cases `trend='rc'`/`'rt'` and Stata's trend names. `alpha` ∈ {0.05, 0.01}.
- `sp.bvar`: every equation used the last equation's Minnesota prior variances, so results depended on column order. Now follows the original Minnesota prior with fixed covariance of Stata `bayes, minnfixedcovprior: var`. New `lambda3`, `lambda4` and `sigma` parameters.
- `sp.garch`:
  - The pre-sample variance is `mean(eps²)` (Stata arch / rugarch), not `var(eps)`.
  - Converged optimiser.
  - `forecast()` uses all lags.
  - New `presample=` and `vce=` parameters, and `garch_loglik`.

**Added**

- `sp.irf(cumulative=, sigma_df=)`.
- `sp.granger_causality`: returns `chi2` / `chi2_p_value`; `causing` may be a list.
- `sp.its(hac_small_sample=)`.
- `sp.structural_break(method='global')`, and `sup_wald` / `sup_break` on sup-F results.
- `sp.cusum_test`: `statistic`, `p_value` and `boundary_coef` keys.

**Fixed**

- `sp.cusum_test`: boundary coefficient is the exact root for any `alpha`. Non-tabulated alpha was silently 5%.
- `sp.structural_break` sup-F grid: includes strucchange's last candidate point.

**MIGRATION rows**

| function | old default output | new default output | how to get close to the old number |
| --- | --- | --- | --- |
| `panel_unitroot` | not the named tests (LLC/IPS/Fisher) | xtunitroot | none; the old numbers were not legitimate |
| `engle_granger` | ADF with constant, asymptotic CVs | egranger | none (old statistic had no valid CVs) |
| `bvar` | order-dependent posterior | Stata minnfixedcovprior | none |
| `garch` | var(eps) presample, NM optimum | mean(eps²), converged | none; differences ~1e-5 |
| `johansen` | wrong CV for maxeig and n/ct | Stata table | statistics unchanged |
| `cusum_test` | boundary a = 0.948 | 0.9478982… | n/a (1e-4) |

## 5. Not closed / open items

- **Stata LLC with trend: T4, reference defect.**
  - `_xturllc.ado` model 3 has σ* = .971 at T = 40. plm has .871, and the column is otherwise monotone (.906, .871, .842).
  - The test rebuilds Stata's t* exactly from our quantities with Stata's value, so that cell is the only difference.
  - Separately, `_xturips.ado` trend-mean lag 8 at T = 60 is −2.204, where plm has −2.024 (neighbours −1.987 and −2.046). That cell is not exercised by the fixtures.
- **Stata Fisher L*: documented difference, correct value unverified.**
  - Stata `_xturfisher.ado` scales by k = 3(5N+3)/(π²N(5N+2)); plm uses 5N+4. Both use t(5N+4).
  - The test reproduces Stata's number from our p-values.
  - Which constant matches Choi (2001) was not verified: the paper was not checked. Open.
- **CUSUM boundary constants vs Stata: grade B.**
  - Stata's 1.142972691 / .9479006054 / .8499248005 differ from the root of strucchange's closed-form crossing probability by ≤ 2.5e-6 relative.
  - Hypothesis: Stata truncates the reflection series differently. Not verified.
  - The statistic agrees at 1e-15. The decision can differ only inside a 2.5e-6 band.
- **sup-F p-values.** Ours come from simulating the Andrews limit on a grid; strucchange and Stata use Hansen's (1997) approximation. They were not compared. Classed as a different approximation of the same asymptotic law.
- **`structural_break(method='bai-perron')` (default).** It adds breaks one at a time using the simulated sup-F. No package computes exactly this: `breakpoints()` is the global method, now available as `method='global'`. Left as is; class 6.
- **`bvar`: T3 only (class 5).** No reference has the closed form; Stata samples it.
- **rugarch SEs: T4.** Explained as noise from rugarch's numerical Hessian. Independent evidence: at identical parameters our Hessian agrees with Stata's to 4e-6, and with rugarch's only to about 0.7%.
- **egcm (R):** not installable from CRAN (archived). Not used.
- **Schemas.** New parameters change signatures, so `scripts/dump_schemas.py --check` reports `tools.json` and `functions.json` stale. The integrator needs to regenerate them. Registry drift tests pass.

## 6. `.gitignore`

```
tests/reference_parity/_fixtures/_ado_timeseries/
```

(The private Stata ado directory created by `_generate_timeseries_stata.do`, which installs egranger and itsa.)

## Existing tests edited

- `tests/test_correctness_inference_fixes.py`: the boundary is now exact (0.9479… rather than 0.948), and the key list has the three new keys.
- `tests/test_cov95_panel_misc.py`: IPS with `trend='n'` now raises.
- `tests/test_panel_cov_diagnostics.py`: IPS with trend at T = 8 and 2 lags is exactly identified and now raises, so the test passes `lags=1`.

Doctest outputs were updated in `cusum_test` and `granger_causality`.
