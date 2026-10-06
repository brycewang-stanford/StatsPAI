# Gow and Ding, *Empirical Research in Accounting* (2024): review

Source: the online edition of Ian Gow and Tony Ding, *Empirical Research in
Accounting: Tools and Methods* (Chapman and Hall/CRC, 2024), its 27 R
programs (8,556 lines), the Quarto templates, and the source of the
companion package `farr` (MIT) with its 27 data sets.

The book is a course in the methods of capital-markets and financial
accounting research. It teaches regression and inference by simulation
(chapters 3 to 5), replicates the classics (Fama, Fisher, Jensen and Roll;
Ball and Brown; Beaver; post-earnings announcement drift; accruals;
earnings management, chapters 10 to 16), then turns to causal inference
(natural experiments, instrumental variables, panel data, regression
discontinuity, chapters 17 to 22) and to generalized linear models,
extreme values, matching and prediction (chapters 23 to 26).

Method. Twenty of the 27 programs query CRSP, Compustat or IBES through
WRDS and cannot be rerun without a subscription. What can be rerun was:
seven programs in full, and the simulated or packaged-data parts of the
rest. The deterministic part was run in R 4.5.2 to make an answer key
(`tests/external_parity/gow_ding_accounting_reference.R`, 53 result
blocks) and each number recomputed with the `sp.*` call a reader would
reach for. Where the two sides disagreed the first step was to find the
first point of divergence. Everything that changed was then tested a
second time on two committed synthetic files against both R and Stata 18,
so the evidence does not rest on data that are not ours to ship.

The WRDS chapters were read for the statistical calls they make, and those
calls were tried on other data. Their data work (linking tables, event
windows, fiscal-year arithmetic) is outside what StatsPAI does.

The book is recent, but its tools are a snapshot. It uses `plm::pmg` for
Fama-MacBeth and hand-rolled Newey-West on coefficient series; `lmrob` for
robust regression; `MatchIt` nearest-neighbour matching; `rdrobust`;
two-way fixed effects for staggered adoption, which it then criticises.
Those were treated as references to reproduce on request, not as defaults
to adopt, and the last section lists what a paper written now would do
differently.

## What was wrong in StatsPAI

1. **Two-way clustered standard errors from a matrix that is not a
   covariance matrix.** The extreme-values chapter clusters an accruals
   regression by firm and year: 8,850 firm-years, 21 years, 85 coefficients
   (year dummies interacted with three regressors). The inclusion-exclusion
   estimator `V_firm + V_year - V_firm×year` then has 46 negative
   eigenvalues. `sp.regress(cluster=['gvkey', 'fyear'])` reported the
   square roots of that matrix's diagonal, with a negative entry turned
   into a standard error of 0 by `np.sqrt(np.maximum(diag, 0))`, no
   warning, and `.vcov()` raising because the stored matrix belonged to the
   one-way fit. `fixest` sets the negative eigenvalues to zero, as Cameron,
   Gelbach and Miller (2011, section 2.3) propose and as `reghdfe` does,
   and its standard errors for the five coefficients of interest were 4 to
   19 percent larger than ours. We do the same now, warn, store the
   matrix, and count the negative eigenvalues in `diagnostics`.

   The first comparison after the fix was still 1 to 2.5 percent off. The
   eigenvalue adjustment is not invariant to a non-orthogonal change of
   parametrisation, and the two fits had different base years: R kept an
   empty factor level, 1995, as the base and dropped the 2015 terms as
   collinear, while StatsPAI takes the base from the estimation sample
   (1996). With the unused level dropped in R, the two agree to all twelve
   printed digits, on raw, winsorized and truncated data. The warning text
   and the migration note say that the adjusted matrix depends on the base
   category.

2. **`sp.poisson` failed on a quasi-separated model.** The whistleblower
   regressions of Call, Martin, Sharp and Wilde (2018) include `mobflag`,
   and every firm with `mobflag = 1` paid a penalty of zero. The Poisson
   likelihood then has no maximum in that coefficient. `glm` stops when the
   deviance stops changing and reports a large negative number.
   `sp.poisson` tested the coefficients only and kept stepping, one unit
   per iteration in that coefficient with the deviance flat to twelve
   digits, until the weighted cross-product matrix was singular and the
   routine died in an SVD with a message that said nothing about the
   cause. The iteration now also stops after two iterations of flat
   deviance, the fit warns and names the regressor
   (`model_info['separated_terms']`), and the coefficient and HC1 standard
   error of interest equal R's to 1e-9. `sp.glm` and `sp.ppmlhdfe` were
   not affected.

   The same routine started from a least-squares fit of `log(y)` with
   zeros set to 0.5. The second penalty variable has no separation and
   failed the same way. It now starts from `mu = y + 0.1`, as `glm.fit`
   does, with step-halving when the deviance rises, and that model
   converges in ten iterations.

3. **`sp.from_r` dropped clustering without saying so.** The book writes
   `feols(fml, ~ gvkey + fyear, data = d)`: the second positional argument
   is the covariance. The translator read keyword arguments only and
   returned `sp.feols(fml, data=df)` with empty notes. `vcov = ~ a + b`
   was noticed but not translated, and a multiway `cluster =` put only the
   first dimension in the code. All three now produce
   `vcov={'CRV1': 'a + b'}`.

4. **`factor(x)` in a formula raised `NameError` everywhere.** It is how R
   marks a categorical term and the book uses it throughout. `factor(x)`
   and `as.factor(x)` are now read as `C(x)` in every formula entry point,
   and the coefficient labels are the house ones.

## What was missing

New functions:

- **`sp.fama_macbeth`.** The statistical-inference chapter is built around
  the comparison of OLS, White, Newey-West, Fama-MacBeth and clustered
  standard errors on the simulated panel of Gow, Ormazabal and Taylor
  (2010). StatsPAI had every column but two.
- **`sp.regress(robust='hac', hac_panel=(unit, time))`**, the other
  missing column. `robust='hac'` read a stacked panel as one time series,
  pairing the last year of one firm with the first of the next.
- **`sp.robreg`.** Robust regression is the chapter's recommended response
  to extreme observations, after Leone, Minutti-Meza and Wasley (2019).
  StatsPAI had no M, S or MM estimator.
- **`sp.itcv`.** The impact threshold for a confounding variable is common
  in accounting papers and the chapter both computes and criticises it.
- **`sp.ndcg`**, the second evaluation measure of the fraud-prediction
  chapter (Bao et al. 2020).

Through existing entry points: `sp.winsor(trim=True)`; translations of
`pmg`, `lmrob`, `rlm`, `rdrobust`, `binom.test` and `linearHypothesis` in
`sp.from_r`, of `matchit(!d ~ x, caliper=)`, and of `xtfmb`, `robreg` and
panel `newey` in `sp.from_stata`.

## What already agreed

Compared and left alone: OLS, HC1 and one-way clustered standard errors
(1e-15); two-way clustering when the matrix is positive semi-definite
(1e-15); fixed effects through `sp.feols` with iid and two-way covariance;
a regression with 1,004 dummy coefficients; 2SLS with the Sargan,
first-stage F and Wu-Hausman statistics (`sp.estat` after `sp.ivreg`);
logit and Poisson coefficients, standard errors and average marginal
effects; Cook's distance; the exact binomial test; a Wald test of two
factor levels; winsorizing at the type-2 quantile; the rank AUC.

## Documented differences (not bugs)

- **Probit standard errors.** `sp.probit` uses the observed information
  (Stata), R's `glm` the expected. 0.4 percent apart at n = 1,000. Already
  recorded in the Ding review.
- **Fama-MacBeth inference.** `plm::pmg` prints normal p-values; `xtfmb`
  and `sp.fama_macbeth` use t(T - 1). With `lag()`, `xtfmb` sets the
  covariances between coefficients to zero; we keep them. Same standard
  errors.
- **`lmrob` versus `robreg` constants.** `lmrob` uses the rounded S
  constant 1.54764, `robreg` and `sp.robreg` solve for it (1.547645). The
  scale moves in the sixth digit. `tuning_s=1.54764` reproduces `lmrob`.
  `robreg` multiplies the sandwich by n / (n - k), `lmrob` does not
  (`small=`).
- **`rlm`.** It uses 0.6745 where the normal quantile is 0.67449, and
  stops at a residual change of 1e-4. `sp.robreg(method='m')` uses the
  same constant and iterates to 1e-10; against `rlm(acc = 1e-13)` the two
  agree to 1e-10.
- **The scale of `robreg m`.** `robreg` takes the median of the absolute
  LAD residuals that are not exactly zero. An L1 fit interpolates p
  observations, whose residuals are zero up to rounding; whether rounding
  leaves 0.0 or 4e-16 decides which order statistic `robreg` uses. We
  leave out the p smallest. On one file the two are equal, on the other
  they are neighbouring order statistics (0.08 percent apart).
- **ITCV of a non-significant coefficient.** The book's function divides
  by `1 - r#` in every case. Frank's threshold for an estimate short of
  significance, and `konfound`, divide by `1 + r#`. We follow the latter.
  `pkonfound` counts degrees of freedom as n - ncov - 2.
- **`winsor2, trim` on a float variable** also removes the two
  observations that sit exactly on the cut-offs, because it compares a
  float with a double held in a macro. On a double variable it keeps them,
  as R and we do.
- **pyfixest and a non-positive-semi-definite two-way covariance.**
  pyfixest does not apply the eigenvalue adjustment that R's `fixest`
  applies by default; a negative variance comes out as a missing standard
  error. `sp.feols` applies it to pyfixest's matrix (second round below),
  so `sp.feols` and R's `fixest` agree and bare pyfixest does not.

## One comparison that is not a parity claim

`lmrob`'s default standard errors (`.vcov.avar1`) and ours differ by 5e-7
to 8e-6 after the n / (n - k) factor is removed, with coefficients, scale
and weights equal to 1e-9. The estimator here is the sandwich of the
stacked estimating equations of the M step, the S step and the scale; it
equals Stata `robreg`'s covariance to 1e-8 for M, S and MM. Centring the
scale equation's score does not close the gap. The cause is not located,
so the tests bound the gap and do not call it agreement.

## Open items

1. The remaining gap to `lmrob`'s standard errors (above).
2. The S search is a random search and proves nothing about the global
   minimum. On 9,036 Compustat firm-years (a leverage ratio of 157, a
   market-to-book of 16,021) the scale has a second local minimum 0.4
   percent above the best one. Refining 5 candidates, as `robreg` does,
   found the better one for 6 of 8 seeds; refining 25, now the default,
   for 8 of 8, and that minimum is also `lmrob`'s. A user can compare
   `model_info['s_scale']` across seeds (smaller is better); nothing does
   it for them.
3. Closed in the second round: `sp.feols` applies the eigenvalue
   adjustment.
4. Closed in the third round: `sp.feols` IV results carry the
   first-stage F, Sargan and Wu-Hausman statistics.
5. Closed in the second round: `sp.abnormal_returns`.
6. Fama-French industry classifications, portfolio sorts and
   size-adjusted returns (`farr::get_ff_ind`, `get_size_rets_monthly`).
   Data utilities rather than estimators; not added.
7. AdaBoost and RUSBoost (`farr::rusboost`). scikit-learn and
   imbalanced-learn cover them; not added.
8. Closed in the third round. `sp.match` already reproduces MatchIt's
   pairs for every `m.order` (`tests/reference_parity/
   test_matching_r_parity.py`); what was wrong was the translation, which
   left the order out and so returned a different estimate (0.533 against
   0.542) without a note. `sp.from_r` now writes MatchIt's order into the
   call and six translated calls reproduce MatchIt 4.7.2 to 1e-15.
9. Closed in the third round: `robreg ..., eff()` is read.

## Second round

Bryce's answer to the two questions left open was "decide for me". Both
were done.

**`sp.feols` applies the eigenvalue adjustment.** pyfixest reports the
two-way covariance as computed, with a missing standard error where a
variance is negative. The wrapper now applies the same adjustment as
`sp.regress` to pyfixest's matrix and lets pyfixest recompute its
inference from it. On the accruals regression `sp.feols`, `sp.regress` and
R's `fixest` (same base year) agree to 1e-12.

**`sp.abnormal_returns`, the finance event study.** Chapters 10 to 14 are
event studies on CRSP returns, which we do not have, so the function was
built from the papers (Brown and Warner 1985; Patell 1976; Boehmer,
Musumeci and Poulsen 1991; Kolari and Pynnonen 2010) and checked against
Stata's `estudy` on synthetic returns: twelve securities, four models,
three windows, five tests. What the comparison found:

- Abnormal returns and CARs agree to 5e-6 in all four models. `estudy`
  works in single precision.
- The tests of the mean CAR are the same arithmetic. Fed `estudy`'s own
  standardised CARs, Patell and BMP reproduce to 1e-15 and the two
  Kolari-Pynnonen adjustments to 1e-9.
- The standard deviation of a CAR differs by a located rule. For the
  market model `estudy` divides the residual sum of squares by `n - 1`
  (the forecast-error variance has `n - 2`) and scales the market term by
  `(n - 1) / n`. Rebuilt from our numbers, its standard deviation matches
  to 5e-6 for the eleven securities with a complete estimation window. For
  the twelfth, which has one missing return, the rebuilt number is 6e-5
  off on the eleven-day window; how `estudy` counts the missing day was
  not worked out. For the market-adjusted and mean-adjusted models the
  standard deviations agree, and there the whole pipeline matches end to
  end to 5e-5.
- For the factor model `estudy`'s standard deviations are 1 to 6 percent
  below the forecast-error ones. Located in the third round: it uses
  `L * RSS / (n - 1)`, with no term for the error in the estimated
  coefficients. Rebuilt from our numbers it matches to 5e-6.
- `estudy` measures the cross-correlation behind the Kolari-Pynnonen
  adjustments by pairing residuals in event time. With different event
  dates those pairs are returns of different days. The default here pairs
  by calendar date, which is the correlation that clustering induces;
  `correlation='event'` gives `estudy`'s.
- `estudy`'s group CAAR is not the mean of its own security rows (3 to 7
  percent away on this file). It is not a portfolio of event-time average
  returns either. The rule was not identified, so the group row's level
  and its `Norm` test are not compared. Here the mean CAR is the mean of
  the CARs.

A unit test written for the function found one bug in it before it
shipped: an event within the estimation-window distance of the start of
the series produced a negative slice stop and took its estimation sample
from the end of the series.

Open from this round: `estudy`'s group row and its handling of a missing
estimation return; long-horizon buy-and-hold returns; rank and sign
tests. Two were tried in the third round and not added. The generalized
rank test of Kolari and Pynnonen (2011), written from the paper, came out
1.5 percent below `estudy`'s statistic on the market-adjusted model, where
the standardised CARs agree, and the difference was not located.
`estudy`'s Wilcoxon row reports a statistic of 96 for twelve securities
(the signed-rank sum cannot exceed 78) with a p-value of zero in every
window, so it is not a reference for anything.

Also in the third round: the `lmrob` standard errors were tried against
several variants of the scale-correction term (the S residual in place of
the MM one, a centred scale score, finite-sample factors on the term). No
scalar factor closes the gap, which stays open.

## What a paper written now would do differently

- The panel-data chapter estimates staggered adoption of the inevitable
  disclosure doctrine by two-way fixed effects and an event study on the
  same specification, and discusses why that is fragile. `sp.did` with
  `method='cs'`, `sp.sun_abraham` or `sp.did_imputation` estimate the
  effect without the forbidden comparisons, and `sp.bacon_decomposition`
  shows which comparisons the TWFE number is made of.
- With 21 year clusters, two-way clustering is at the edge of what the
  asymptotics support, which is what the indefinite covariance matrix was
  saying. `sp.wild_cluster_bootstrap` is the usual remedy.
- The RDD chapter's `rdrobust(..., masspoints = "off")` has a direct
  counterpart; `sp.rddensity` and `sp.rd_robustness_table` add the checks
  a referee now asks for.
- For sensitivity to omitted variables, `sp.sensemakr` (Cinelli and
  Hazlett) and `sp.oster_bounds` bound the coefficient itself where the
  ITCV describes one threshold.

## Evidence

- `tests/reference_parity/test_accounting_research_parity.py`: 42 tests
  on two committed synthetic files, R 4.5.2 (plm 2.6.7, sandwich 3.1.1,
  fixest 0.14.0, robustbase 0.99.7, MASS 7.3.65) and Stata 18 (xtfmb,
  newey, robreg, pkonfound, winsor2).
- `tests/reference_parity/test_event_study_returns_parity.py`: 45 tests
  against Stata 18 `estudy` on a committed synthetic returns file.
- `tests/external_parity/test_gow_ding_accounting.py`: 21 tests on the
  book's data, skipped without it.
- `tests/test_accounting_research_tools.py`: edge cases, refusals and the
  translations, executed.

Rerun: `export STATSPAI_GOW_DING_DIR=<empty folder>`;
`Rscript tests/external_parity/gow_ding_accounting_reference.R` (needs
`farr`); `pytest tests/external_parity/test_gow_ding_accounting.py`.
Committed fixtures: in `tests/reference_parity/_fixtures`,
`python _generate_accounting_research_data.py`,
`Rscript _generate_accounting_research_R.R`,
`do _generate_accounting_research_Stata.do`.
