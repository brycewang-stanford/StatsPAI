# Chen Qiang, *Econometrics and Stata Applications* (2nd edition): what its do-files showed about StatsPAI

Started 2026-10-02. Worktree `.claude/worktrees/chenqiang-textbook`, branch
`wt/chenqiang-textbook`.

## What was examined

The programs and data that accompany 陈强《计量经济学及Stata应用》(第2版).
Eighteen chapter do-files (chapters 2 to 19), three small figure programs
and 44 datasets. The files sit in `改进建议-收集整理/陈强初计-2023/`, which
is gitignored. Neither the programs nor the data are redistributed.

The book is the standard undergraduate text in China. Its sequence is the
usual one up to chapter 14 (OLS, heteroskedasticity, serial correlation,
specification, IV, binary choice, panel, time series, unit roots and
cointegration). Chapters 15 to 19 are what set the second edition apart.
They cover matching, regression discontinuity, difference in differences,
synthetic control and the regression control method, the last two with
commands the author wrote himself (`synth2`, `rcm`).

| Chapters | Commands | Datasets |
| --- | --- | --- |
| 2 to 6 | data management, `regress`, `test`, `predict`, `simulate` | grilic, nerlove, cobb_douglas |
| 7 to 9 | `estat hettest / imtest / bgodfrey / dwatson / ic / ovtest / vif`, `newey`, `prais`, `corrgram`, weights | nerlove, icecream, consumption |
| 10 to 12 | `ivregress`, `estat overid / firststage / endogenous`, `hausman`, `logit`, `probit`, `margins`, `xtreg fe / re / be / mle`, `xttest0`, `xtoverid`, `xtserial`, `xtsum` | grilic, titanic, lin_1992 |
| 13, 14 | `var`, `varsoc`, `varwle`, `varlmar`, `varstable`, `vargranger`, `fcast`, `dfuller`, `vecrank`, `vec`, `veclmar`, `vecstable` | gdp_china, macro_swatson, nelson_plosser, mpyr |
| 15 to 19 | `teffects psmatch`, `tebalance`, `rdrobust`, `rddensity`, `reghdfe`, `synth`, `synth2`, `rcm` | nsw_dw, rdrobust_senate, cao_chen, synth_smoking, growth |

## Method

The do-files ship without output. They were run once in Stata 18 with a log
per chapter (`tests/external_parity/chen_qiang_2e_prepare.py` writes the
runner). Each log was then replayed through one `sp.stata` session by
`scripts/stata_log_replay.py`, which compares every number Stata printed
with what StatsPAI returns, to the precision Stata printed it.

The replay script learned to read more of a log along the way. It now
compares regression headers (N, F, R-squared, root MSE), `estat`, `margins`,
`hausman`, `corrgram`, the `var` and `vec` suites, `xtsum`, `xtserial`,
`xttest0`, `xtoverid`, `teffects`, `tebalance`, `rdrobust`, `rddensity`,
`synth` and `rcm`. It reads the `tsset` declaration and the value labels
stored in a `.dta` file, and marks numbers computed on simulated data as not
comparable.

State before and after, on the 18 logs.

| | Before | After |
| --- | --- | --- |
| Numbers compared and reproduced | 607 | 2,911 |
| Numbers different | 20 | 17, all documented below |
| Commands `sp.stata` refused | 230 | 3 (`synth2`) |
| Numbers with no counterpart | 122 | 0 |
| Numbers on simulated data (not comparable) | 0 | 33 |

The 20 differences before the work were all downstream of one refused
`append`. Chapters 18 and 19 could not be replayed at all at the start.

Run it with

```bash
python tests/external_parity/chen_qiang_2e_prepare.py <folder>   # then run _master.do in Stata
STATSPAI_CHENQIANG_DIR=<folder>/run pytest tests/external_parity/test_chen_qiang_2e_logs.py
```

Every estimator added or corrected here is also pinned without the textbook.
`tests/reference_parity/test_textbook_methods_stata_parity.py` compares 39
blocks with real Stata 18 output on four committed synthetic datasets
(`_fixtures/_generate_textbook_methods_stata.do` reads the same CSV bytes).

## What was wrong, and what changed

### Wrong numbers

| # | Finding | Status |
| --- | --- | --- |
| 1 | `sp.regress` reported the classical F statistic under a robust, clustered or HAC covariance. On the Nerlove regression it printed 437.9 where the robust Wald F is 177.19 | fixed |
| 2 | Without a constant, `sp.regress` reported the centred R-squared (negative on `lnw ~ s - 1`), an F that assumed an intercept and `df_model = K - 1` | fixed, uncentred as Stata and R |
| 3 | `AIC` and `BIC` of `sp.regress` and `sp.estat(..., 'ic')` left out the constant `N (log 2 pi + 1)`, so they disagreed with the reported log likelihood and with Stata and R | fixed |
| 4 | `sp.estat(..., 'white')` counted redundant terms as restrictions. With a dummy regressor its square is the dummy, so the degrees of freedom were too many | fixed |
| 5 | `sp.estat(..., 'vif')` returned the factors rounded to two decimals | fixed |
| 6 | `sp.estat(..., 'bgodfrey')` dropped the first observations, which is neither Stata's default nor R's | default is now `fill='zero'`, `fill='drop'` is Stata's `nomiss0` |
| 7 | The Breusch-Pagan LM test for random effects used the average panel length on unbalanced panels (277.2 where the Baltagi-Li statistic is 274.7) | fixed |
| 8 | After a robust or clustered IV fit `estat endogenous` was still the homoskedastic Wu-Hausman F | fixed on main while this work was in progress (`826295bf`); the Durbin score statistic was added here |
| 9 | The translation of `synth` ran the in-space placebo, which Stata's `synth` does not do and which takes minutes per unit with nested weights | `placebo=False` in the translation |

### Refused or missing

| # | Finding | Status |
| --- | --- | --- |
| 10 | `regress, noconstant` was refused | translated |
| 11 | A weight expression (`[aw=1/e2f]`), frequency weights and a weight clause after `if` were refused | the session evaluates the expression and expands frequency weights |
| 12 | `gsort`, `rename`, `tabulate` (with `generate()`), `collapse`, `ipolate`, `set obs`, `set seed`, `clear`, `drop _all` | run |
| 13 | `asinh()`, running `sum()`, date functions (`month`, `dofm`, `tq()` ...), densities and quantiles, random draws, time-series operators inside `generate` | evaluated |
| 14 | `e(rss)`, `e(mss)`, `e(rmse)`, `e(df_r)`, `e(df_m)`, `e(F)`, `e(ll)`; `predict, leverage`, `predict` after `logit` / `probit`, `predict ... if` | available |
| 15 | `estimates store / restore / table`, `esttab`, `bysort: summarize` | run |
| 16 | `estat` had no translation | `hettest`, `imtest, white`, `ovtest`, `bgodfrey`, `dwatson`, `vif`, `ic`, `overid`, `firststage`, `endogenous`, `classification` |
| 17 | `hausman` between two stored models | `sp.hausman` |
| 18 | `prais`, `corrgram`, `wntestq` | `sp.prais`, `sp.corrgram` |
| 19 | `varsoc`, `varwle`, `varlmar`, `varstable`, `vargranger`, `fcast compute`, `varbasic` | `sp.varsoc`, methods of the VAR result, `sp.estat` |
| 20 | `vecrank`, `vec`, `veclmar`, `vecstable` | `sp.johansen`, `sp.vec` |
| 21 | `xtreg, re / be / mle`, `xttest0`, `xtoverid`, `xtserial`, `xtsum`; the constant, `sigma_u`, `sigma_e`, `rho` and the three R-squared of `xtreg` | `sp.panel(method='mle')`, `sp.xtoverid`, `sp.xtserial`, `sp.xtsum`, session extras |
| 22 | `teffects psmatch` with `generate()`, `osample()`; `predict, ps`; `tebalance summarize` | run |
| 23 | `c.x#(c.z1-z9)` and `test c.x#c.z` | translated |
| 24 | `program ... end` with `simulate` | run, with numpy's random numbers |
| 25 | `rcm` | `sp.synth(method='rcm')` |
| 26 | The synthetic-control robustness helpers ignored the predictor specification of the fit they check | `covariates=`, `special_predictors=`, `v_method=` on `sp.synth_loo`, `sp.synth_time_placebo`, `sp.synth_rmspe_filter`, `sp.synth_donor_sensitivity` |

Some notes on individual items.

Item 1. The classical F is a ratio of sums of squares. Under
heteroskedasticity it tests nothing. Stata and R both report the Wald
statistic on the robust covariance, with `N - K` denominator degrees of
freedom, or `G - 1` with clusters. The Stock and Watson logs now check this
too, because the replay reads the header.

Item 6. Both reference implementations set the missing lagged residuals to
zero (Stata's default, R `lmtest::bgtest`). StatsPAI matched neither.

Item 21. `xtreg, mle` is the random-intercept model of `sp.mixed`. The
coefficients agreed. The standard errors did not, by up to 4%. `sp.mixed`
reports the GLS standard errors given the variance components. `xtreg, mle`
inverts the observed information of the full likelihood. Both are valid.
`sp.panel(method='mle')` computes the second analytically and agrees with
Stata to 2e-5, which is where Stata's own iterations stop.

Item 25. The regression control method predicts the treated unit from the
control units by an unrestricted regression, with the units chosen by best
subset and the size by AICc. The exact best subset is found by branch and
bound on the swept cross-product matrix. On the Hong Kong data with 24
control units and the placebo run for each of them it takes two seconds
where the Stata command needs twenty-five minutes. Selection tables,
coefficients, effects and placebo p-values agree with Stata on every printed
digit.

## Documented differences

Three places where the numbers differ and why. The opt-in test lists them and
fails on anything else.

1. `estat ovtest, rhs` when one regressor is the square of another (chapter
   9, `expr2 = expr^2`). Stata reports `F(11, 741) = 1.73`. It drops the
   original `expr2` from the augmented regression as collinear and then tests
   the power that replaced it, so the test is against the model without
   `expr2`. Reproduced: restricted model without `expr2`, 11 restrictions,
   `F = 1.7312`. The RESET test of the model that was fitted has 10
   restrictions and `F(10, 741) = 1.272`, which is what StatsPAI returns.
2. `pscore[match1]` after `teffects psmatch, gen(match)` (chapter 15). Stata
   lists the nearest neighbours in an order that is not the distance order.
   StatsPAI lists the nearest first. The mean of the first-listed neighbour's
   propensity score is 0.270529 in Stata and 0.270524 here. The matched set,
   the ATET and its standard error agree.
3. `synth, nested` (chapter 18). The predictor weights are a non-convex
   search. StatsPAI's solution has pre-treatment MSPE 3.086 where Stata's has
   3.227, so the donor weights differ (Utah 0.336 against 0.345). This is the
   non-uniqueness already recorded for the Basque data.

One more thing shows in chapter 15. `bysort treat: sum` sorts the data, and
Stata's sort is not stable, so the order of the treated rows after it is not
reproducible even inside Stata. Anything that reads a row by number
afterwards (`list in 1/2`) depends on that order.

## Where the textbook's practice has moved

This reads the book's chapters against what applied work expects in 2026 and
where StatsPAI stands. It is a judgement, not a citation list. The
references for each method are on the function's own docstring.

| Chapter | The book does | Common practice now | In StatsPAI |
| --- | --- | --- | --- |
| 5, 6 | Classical standard errors first, then robust | Robust by default. HC2 or HC3 in small samples | `sp.regress(robust='hc1' / 'hc2' / 'hc3')` |
| 7 | Test for heteroskedasticity (BP, White), then WLS or feasible GLS | No pretest. Robust standard errors always, and if GLS is used for efficiency, robust standard errors on top (the book's last line does this) | `sp.estat`, `weights=`, `robust=` |
| 8 | BG, Q and DW tests, Newey-West with a rule-of-thumb lag, Prais-Winsten | HAC with a larger bandwidth and fixed-b critical values. Feasible GLS only under strict exogeneity | `sp.regress(robust='ewc')`, `sp.regress(robust='hac', hac_lags=)`, `sp.prais` |
| 9 | Information criteria, RESET, VIF, leverage, Chow test, interpolation of missing values | The same diagnostics, plus sensitivity to an omitted variable and a specification curve. Multiple imputation in place of interpolation | `sp.sensemakr`, `sp.oster_bounds`, `sp.spec_curve`, `sp.imputation` |
| 10 | 2SLS, first-stage F above 10, overidentification, LIML, Hausman | Effective F, Anderson-Rubin confidence sets, tF critical values. The F above 10 rule does not control size with robust errors | `sp.iv_diag`, `sp.effective_f_test`, `sp.anderson_rubin_ci`, `sp.tF_adjustment` |
| 11 | Logit and probit with average marginal effects from `margins` | Unchanged. The book is already current here | `sp.logit`, `sp.margins` |
| 12 | FE, RE, the Hausman test, a robust version (`xtoverid`), two-way FE, a serial-correlation test | Fixed effects with standard errors clustered on the panel, without testing for serial correlation first. Wild cluster bootstrap with few clusters. Two-way FE is no longer the default when a policy starts at different dates | `sp.panel`, `sp.xtoverid`, `sp.wild_cluster_bootstrap`, `sp.callaway_santanna` |
| 13 | AR and ADL forecasts, VAR, orthogonalised impulse responses | Local projections next to the VAR | `sp.var`, `sp.irf`, `sp.local_projections` |
| 14 | ADF, Johansen, VECM | Unchanged as teaching material. DF-GLS has more power than ADF | `sp.unitroot(test='dfgls')`, `sp.johansen`, `sp.vec` |
| 15 | Propensity-score matching with balance tables | Doubly robust estimators, balancing weights, overlap weights, and a sensitivity analysis for unobserved confounding. Matching on the propensity score alone is rarely the last word | `sp.aipw`, `sp.ebalance`, `sp.overlap_weights`, `sp.dml`, `sp.rosenbaum_bounds` |
| 16 | `rdrobust` with several bandwidth selectors, covariates, a placebo outcome and the density test | Unchanged. Add donut and placebo-cutoff checks and a bandwidth sensitivity plot | `sp.rdrobust`, `sp.rddensity`, `sp.rdplacebo`, `sp.rdbwsensitivity` |
| 17 | Event study by two-way FE on an `asinh` outcome, with unit trends, and a joint pre-trend test | Four changes, below | see below |
| 18 | Synthetic control with nested weights, placebo ratios, leave-one-out | Prediction intervals or conformal inference, synthetic DiD and the augmented estimator as checks | `sp.synth(method='scpi' / 'sdid' / 'augmented')`, `sp.conformal_synth` |
| 19 | Regression control with best subset and AICc | Forward selection when there are many control units, and a placebo in time | `sp.synth(method='rcm', selection='forward', placebo_time=)` |

Chapter 17 is where practice has moved most since the book went to press.

- The outcome is `asinh` of a rate. With zeros in the data the estimated
  effect of an `asinh` or `log(1 + y)` outcome depends on the units of `y`.
  A Poisson regression with fixed effects does not (`sp.ppmlhdfe`), and the
  sensitivity to rescaling is worth showing either way.
- The joint test of the pre-period coefficients has little power against the
  trends that would matter. `sp.pretrends_power` says what it could have
  detected, and `sp.honest_did` gives intervals that stay valid under a
  bounded violation.
- Pointwise intervals on an event-study plot understate the uncertainty about
  the whole path. `sp.uniform_bands` gives sup-t bands.
- The book's application has one treatment date, so two-way FE is fine. With
  staggered dates it is not, and the textbook does not yet cover the
  heterogeneity-robust estimators (`sp.callaway_santanna`, `sp.sun_abraham`,
  `sp.did_imputation`, `sp.bacon_decomposition`).

## What was worth taking from the book

- The Monte Carlo demonstrations with `program` and `simulate` (chapters 6
  and 14). They are how the central limit theorem and the spurious regression
  are taught, and `sp.stata` now runs them.
- `estimates store` followed by `estimates table` / `esttab` as the way a
  results table is built. The session keeps the named models.
- The author's own `rcm`: model selection in two steps and placebo p-values
  period by period, both adopted in `sp.synth(method='rcm')`.
- `xtoverid` next to the classical Hausman test in the panel chapter.

## Open items

| Item | Why it matters | Size |
| --- | --- | --- |
| `synth2` is not translated | Four of the five commands of chapter 18. It needs period-by-period placebo p-values and a leave-one-out on donors with non-zero weight, on the fit's own predictor specification | medium |
| The nested predictor-weight search takes about 150 seconds on the smoking data and reports `converged=False` | Stata's `synth, nested` takes seconds. A placebo run over 38 states is hours | medium, touches a frozen parity module |
| `rcm` with `method(lasso)` and with covariates | the command's recommended setting for many control units | medium |
