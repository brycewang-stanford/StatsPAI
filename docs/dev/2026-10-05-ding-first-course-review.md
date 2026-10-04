# Ding, *A First Course in Causal Inference* (2024): review

Source: the book's replication files on Harvard Dataverse
(doi:10.7910/DVN/ZX3VEV), 28 R programs and 14 data files covering
randomized experiments (chapters 1 to 9), observational studies (11 to 19),
discontinuities (20, 24), instruments (21 to 25), principal strata (26),
mediation (27) and time-varying treatments (29).

Method. The programs write their estimators out by hand, so they were not
translated. The deterministic part of each one was rerun in R 4.5.2 to make
an answer key (`tests/external_parity/ding_first_course_reference.R`), and
each number was then recomputed with the `sp.*` function a user would reach
for. Simulations and Monte Carlo randomization tests were compared on their
observed statistics, not on their random draws. Where the two sides
disagreed the first step was to find the first point of divergence, as §5.1
of `CLAUDE.md` asks. The book is from 2023 to 2024 and its code predates
some current practice (no cross-fitting, grid-inverted intervals); those
choices were treated as conventions to reproduce on request, not as
defaults to adopt.

## What was wrong in StatsPAI

1. **`sp.ipw(estimand='ATT' or 'ATC', normalize=False)` was scaled by the
   share of the target group.** The Horvitz-Thompson sums were divided by
   `n` for every estimand. For the ATT the divisor is the number of
   treated, so the function returned `P(T=1) x ATT` (and `P(T=0) x ATC`).
   NHANES, chapter 13: -1.098 before, -1.992 after, which is the book's
   number. The default `normalize=True` and the ATE were never affected.
2. **The confidence interval of `sp.fisher_exact` was too short.** It was
   read off a 101-point grid spanning six standard deviations of the
   outcome, each point tested on its own 500 draws, with no refinement. On
   the LaLonde experiment it returned [999, 2589] for the effect on 1978
   earnings; effects of 1000 and 2600 have randomization p-values of 0.21
   and 0.20. With `controls` it also subtracted the unresidualized
   `tau_0 * D` from the residualized outcome. The interval is now the exact
   inversion of the test on the assignments that gave the p-value
   ([572, 3015] here, Neyman's is [479, 3109]) and is infinite, with a
   warning, when the design cannot reject at `alpha`.
3. **`sp.gformula_ice_fn` failed on its own documented input.** A flat
   list of confounders ("the same at every time") or a baseline covariate
   named again at a later time put the same column into the history twice.
   The point estimate went through; the default sandwich standard error
   raised `LinAlgError: Singular matrix`. Repeated names are now counted
   once.
4. **Lee and Zhang-Rubin bounds on a binary outcome collapsed without a
   word.** With `trimming='quantile'`, the default, every observation tied
   at the trimming quantile is kept. On the binary outcome of chapter 26
   that keeps everything and both bounds equal -0.045; the sharp bounds are
   [-0.176, -0.019], which `trimming='exact'` returns. The default is
   unchanged (it is the rule Stata's `leebounds` applies to continuous
   outcomes and is parity-tested), and `sp.lee_bounds`,
   `sp.principal_strat`, `sp.survivor_average_causal_effect` and
   `sp.attrition_bounds` now warn when ties stop it from trimming at least
   half of what it should.
5. **Principal scores under one-sided noncompliance.** When nobody in the
   control arm takes the treatment, `sp.principal_strat(method=
   'principal_score')` fitted a logit to an all-zero outcome on every
   bootstrap draw (thousands of perfect-separation warnings on JOBS II) and gave the empty always-taker stratum a share of 1e-4. A
   constant arm is now recognised, its stratum has share zero and nothing
   is fitted. The estimates move in the sixth digit.

## What was missing

- **`sp.rosenbaum_bounds(method='t')`**, the sensitivity analysis of the
  mean pair difference that chapter 19 runs with `sensitivitymw::senmw`.
  Upper-bound p-values agree to 2e-14 on the matched LaLonde pairs and on
  simulated pairs.
- **A heteroskedasticity-robust Anderson-Rubin test**,
  `sp.anderson_rubin_test(ar_vcov='HC0'..'HC3')`. The statistic was always
  the homoskedastic F unless `cluster=` was given; the `vcov` argument only
  ever affected the effective F printed next to it. The book's
  Fieller-Anderson-Rubin interval uses the robust variance. Statistic and
  p-value agree with `sandwich::vcovHC` on the reduced form to 1e-13 for
  one and two instruments; Card's interval is [0.0277, 0.2821] against the
  book's grid values [0.028, 0.282]. The default stays `'classic'`, which
  is parity-tested against `ivmodel::AR.test`.
- **Statistics for `sp.ri_test`**: `'rank_sum'` (the centred Wilcoxon rank
  sum), `'lin'` and `'lin_t'` (Lin's estimate and its HC2 t-ratio, the
  covariate-adjusted statistic chapter 8 recommends). `'t'` was already the
  unequal-variance t; its documentation now says so and says why it is the
  one to prefer. `sp.fisher_exact` gained `statistic='t'`.

## Agreement found, nothing to change

The last column is the relative tolerance the replication test enforces.
Quantities that pass through a logit are held to 1e-6 because R's `glm`
stops at 1e-8 in the deviance; the realised gaps elsewhere are several
orders smaller than the bound.

| chapter | computation | StatsPAI | held to |
| --- | --- | --- | --- |
| 1 | OLS on the LaLonde CPS sample | `sp.regress` | 1e-8 |
| 3 | pooled, Welch, rank-sum and KS statistics | `sp.ttest`, `sp.ri_test` | 1e-8 |
| 4 | Neyman variance; HC0, HC2, HC3 | `sp.difference_in_means`, `sp.regress` | 1e-8 |
| 5 | stratified and post-stratified estimators (Penn bonus, Chong et al.) | `sp.difference_in_means(blocks=)` | 1e-8 |
| 6, 9 | Lin's estimator, EHW and super-population variance | `sp.lm_lin` | 1e-7 (STAR is a float `.dta`), 1e-8 |
| 7 | Darwin's pairs, all 32,768 sign flips | `sp.ri_test(strata=)` | exact |
| 11 | Horvitz-Thompson and Hajek, four truncation levels | `sp.ipw` | 1e-6 |
| 12, 13 | outcome regression and doubly robust, ATE and ATT | `sp.g_computation(by_arm=True)`, `sp.aipw(cross_fit=False)` | 1e-8 (regression), 1e-6 |
| 15 | bias-adjusted matching, both LaLonde samples, estimate and SE | `sp.match(method='nnmatch', metric='ivariance', vce='iid')` | 1e-8 |
| 17 | E-values | `sp.evalue` | 1e-6 |
| 20, 24 | sharp and two fuzzy discontinuities, estimate, SE, bandwidths | `sp.rdrobust` | 1e-6 |
| 21, 24 | Wald estimator with the delta-method SE; local 2SLS with HC2 | `sp.ivreg(robust='hc2')` | 1e-8 |
| 23 | 2SLS on Card's data, HC0 | `sp.ivreg` | 1e-8 |
| 25 | inverse-variance weighting, fixed and multiplicative random effects | `sp.mr_ivw` | 1e-8 |
| 27 | Baron-Kenny direct and indirect effects | `sp.mediate(inference='delta')` | 1e-8 |

Two identities that the book derives show up numerically: Neyman's variance
estimator is the HC2 variance of the regression on the treatment dummy, and
the delta-method standard error of the Wald estimator is the HC2 standard
error of two-stage least squares.

## Documented differences, StatsPAI kept as is

- **MR-Egger.** The book regresses on the variants as coded. `sp.mr_egger`
  orients each variant to a positive exposure association first, as the
  method is defined; it equals the weighted regression on the oriented
  data to eight digits. Slope 0.452 against the book's 0.317.
- **Principal-score weighting.** The book's estimator divides the weighted
  control sum by the complier share of the treated arm, so it changes when
  a constant is added to the outcome. StatsPAI normalises the weights.
  JOBS II complier effect 0.0929 against 0.1695; both reproduce the
  intention-to-treat effect when recombined.
- **Propensity-score stratification.** `sp.match(method='stratify')` uses
  intervals closed on the left, R's `cut` on the right. Three NHANES units
  sit on a quintile boundary: -0.1174 against -0.1161. Right-closed strata
  passed to `sp.difference_in_means(blocks=)` give the book's number to
  1e-13.
- **Mediation standard errors.** The book plugs HC3 standard errors into
  Sobel's formula (0.00899); `inference='delta'` uses the classical ones
  (0.00901), as Stata's `paramed` does.
- **`Matching::Match` standard errors.** Its default assumes a constant
  outcome variance (`vce='iid'` here). The StatsPAI default is the
  two-neighbour heteroskedastic variance of `teffects nnmatch`, which is
  `Var.calc = 2` in `Match` and agrees with it to 1e-9.

## Left open

- **Rerandomization (chapter 6).** `sp.randomize(n_rerand=, rerand_threshold=)`
  keeps the best of `n_rerand` draws by a Mahalanobis distance that lacks
  the `n1 n0 / n` factor, so the threshold is not on the chi-square scale
  of Morgan and Rubin's criterion and there is no acceptance-probability
  argument. The fix is an `accept_prob` argument with the scaled distance.
- **Lin's estimator within strata (chapter 8).** `sp.lm_lin` has no
  `blocks=`; the book fits it stratum by stratum and combines.
- **Sensitivity parameters for the doubly robust estimator (chapter 18).**
  Not implemented.
- **Complier outcome means and the testable implications of the instrument
  assumptions for a binary outcome (chapter 22).** `sp.kitagawa_test`
  covers the test; the `IVbinary` table of stratum means has no single
  call.
- **Overflow warnings in bootstrap logits.** `core._glm_fit.safe_logit_fit`
  runs statsmodels' Newton steps on unscaled covariates; on JOBS II some
  bootstrap draws print `overflow encountered in exp`. Estimates are
  unaffected as far as the checks here go.
- **`sp.fisher_exact` summary** still prints "95% CI" whatever `alpha` is,
  and for statistics other than the difference in means the field holds
  quantiles of the null distribution, which are not a confidence interval.

## Tests

- `tests/reference_parity/test_ding_first_course_parity.py`: R references
  on simulated data, committed with the generator script and CSVs.
- `tests/test_ding_first_course_fixes.py`: the corrections, by brute-force
  inversion, known truth and coverage.
- `tests/external_parity/test_ding_first_course.py`: the book chapter by
  chapter. Needs `STATSPAI_DING_DIR`; skipped otherwise. To rerun:

  ```bash
  export STATSPAI_DING_DIR=/path/to/dataverse_ZX3VEV
  Rscript tests/external_parity/ding_first_course_reference.R
  pytest tests/external_parity/test_ding_first_course.py
  ```
