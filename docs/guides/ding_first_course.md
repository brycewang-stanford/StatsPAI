# Ding, *A First Course in Causal Inference*, in StatsPAI

Peng Ding's textbook (Chapman and Hall/CRC, 2024;
<https://doi.org/10.1201/9781003484080>) teaches causal inference from the
potential-outcomes side, starting from randomized experiments and ending
with instruments, principal strata and mediation. Its replication files are
28 short R programs. Nearly every estimator in them is written out by hand,
ten lines at a time, which is what makes the book a good check on a
package. If a function agrees with those ten lines, you know what it
computes.

This guide maps each chapter to the StatsPAI call that gives the same
number, says where StatsPAI makes a different choice and why, and lists
what a paper written today would add.

The files are at <https://doi.org/10.7910/DVN/ZX3VEV>. They are not
redistributed with StatsPAI. The examples below read them from `files/`.

## Chapter by chapter

| chapter | the book computes | StatsPAI |
| --- | --- | --- |
| 3 | Fisher randomization test with four statistics | `sp.ri_test(stat='diff_means' / 't' / 'rank_sum' / 'ks')` |
| 4 | Neyman's estimator and variance; HC0 to HC3 | `sp.difference_in_means`; `sp.regress(robust='hc2')` |
| 5 | stratified and post-stratified estimators | `sp.difference_in_means(blocks=)` |
| 6, 9 | Lin's estimator, with the super-population variance | `sp.lm_lin(superpopulation=True)` |
| 7 | matched pairs, exact sign-flip test | `sp.ri_test(strata=pair)`; `sp.difference_in_means(blocks=pair)` |
| 8 | studentized, covariate-adjusted randomization tests | `sp.ri_test(stat='lin_t', covariates=)` |
| 11 | propensity-score stratification, Horvitz-Thompson and Hajek weighting | `sp.match(method='stratify')`; `sp.ipw(normalize=, trim=)` |
| 12 | outcome regression and the doubly robust estimator | `sp.g_computation(by_arm=True)`; `sp.aipw(cross_fit=False)` |
| 13 | the same for the effect on the treated | `estimand='ATT'` in the three functions above |
| 15 | matching with bias adjustment | `sp.match(method='nnmatch', metric='ivariance', bias_adjust=True)` |
| 17 | E-values | `sp.evalue` |
| 19 | Rosenbaum's sensitivity analysis | `sp.rosenbaum_bounds(method='t')` |
| 20, 24 | sharp and fuzzy discontinuities | `sp.rdrobust(fuzzy=)` |
| 21 | Wald estimator, delta-method variance, Fieller-Anderson-Rubin interval | `sp.ivreg(robust='hc2')`; `sp.anderson_rubin_test(ar_vcov=)` |
| 23 | two-stage least squares on Card's data | `sp.ivreg`; `sp.anderson_rubin_test(exog=, ar_vcov='HC3')` |
| 25 | inverse-variance weighting, Egger regression | `sp.mr_ivw`; `sp.mr_egger` |
| 26 | bounds on the survivor average effect; principal scores | `sp.lee_bounds(trimming='exact')`; `sp.principal_strat(method='principal_score')` |
| 27 | Baron-Kenny with Sobel's variance | `sp.mediate(inference='delta')` |
| 29 | time-varying treatments | `sp.gformula_ice_fn`; `sp.msm` |

Chapter 29 has no data, only a simulation in which a regression on both
treatments and both covariates gets the effect of always treating wrong
(about 2 where the truth is 3) and the g-formula gets it right.
`sp.gformula_ice_fn` returns 3.03 with a standard error of 0.02 on 20,000
draws of that design.

Every row with data behind it is checked against the book's own R code in
`tests/external_parity/test_ding_first_course.py`. Closed-form quantities
agree to 1e-8 or better.

## Randomization tests

The book's advice in chapters 3 and 8 is to studentize. A randomization
test with the raw difference in means is exact for the sharp null of no
effect on any unit. It can over-reject the weaker null of a zero *average*
effect when the arms differ in size and in variance. The same test with the
difference divided by Neyman's standard error keeps the exactness and is
also valid in large samples for the weak null.

```python
import statspai as sp

sp.ri_test(df, y="re78", treat="treat", stat="t", n_perms=10_000, seed=1)
```

With covariates the studentized statistic is Lin's estimate over its HC2
standard error.

```python
sp.ri_test(df, y="re78", treat="treat", stat="lin_t",
           covariates=["age", "educ", "re74", "re75"], n_perms=10_000, seed=1)
```

`strata=` re-randomizes within blocks, and a block of two is a matched
pair. When the design has no more assignments than `n_perms` they are all
enumerated and the p-value is exact. Darwin's 15 pairs have 32,768 sign
flips, and the one-sided p-value of chapter 7 comes out as 0.02634 with no
simulation error.

`sp.fisher_exact` adds a confidence interval for a constant effect. It is
the set of effects the test does not reject, computed on the same
assignments as the p-value and exact for them.

## Weighting, and the two ways to normalise

Chapter 11 contrasts the Horvitz-Thompson estimator with the Hajek
estimator, which divides each weighted sum by the sum of its weights. The
Hajek form does not change when a constant is added to the outcome. The
Horvitz-Thompson form does, which is why the book's NHANES example gives
-1.52 for one and -0.16 for the other. `sp.ipw` defaults to Hajek;
`normalize=False` gives Horvitz-Thompson.

```python
sp.ipw(df, "BMI", "School_meal", covariates, normalize=True, trim=0.1)
sp.ipw(df, "BMI", "School_meal", covariates, estimand="ATT", normalize=False)
```

`trim=` truncates the estimated propensity score to `[trim, 1 - trim]`, as
the book does. It does not drop units.

## The doubly robust estimator without cross-fitting

The book fits a logit for the propensity score and a linear model for each
arm on the full sample and plugs them into the doubly robust formula.
`sp.aipw(cross_fit=False)` does exactly that and returns -0.0193 on the
NHANES data, the book's number. The default `cross_fit=True` is what you
want once the nuisance models are flexible.

## Matching

`Matching::Match` matches on the covariates scaled by their standard
deviations and, by default, assumes a constant outcome variance for the
standard error. In `sp.match(method='nnmatch')` those are
`metric='ivariance'` and `vce='iid'`. With them the estimate and the
standard error agree with `Match` to ten digits on both LaLonde samples.
The StatsPAI defaults are Stata's `teffects nnmatch`: Mahalanobis distance
and the heteroskedasticity-consistent variance with two neighbours
(`Var.calc = 2` in `Match`).

## Sensitivity analysis after matching

Chapter 19 asks how much hidden bias the matched LaLonde comparison could
absorb, using the test of the mean pair difference.

```python
res = sp.rosenbaum_bounds(treated, control, method="t",
                          gamma_grid=[1.0, 1.1, 1.2, 1.3])
res.pvalue_upper      # 0.0026, 0.0117, 0.0367, 0.0876
```

These are `sensitivitymw::senmw(method = "t")` to fourteen digits. The
default `method="wilcoxon"` uses ranks and is less sensitive to a few large
differences.

## Instruments and the Fieller-Anderson-Rubin interval

The Wald estimator's delta-method standard error in chapter 21 is the HC2
standard error of two-stage least squares, so `sp.ivreg(..., robust='hc2')`
returns it.

The book's weak-instrument interval inverts a test that uses the
heteroskedasticity-robust standard error of the reduced form.
`sp.anderson_rubin_test` computes the homoskedastic statistic unless told
otherwise, which is the textbook Anderson-Rubin F and agrees with
`ivmodel::AR.test`. Ask for the robust one with `ar_vcov`.

```python
sp.anderson_rubin_test(card, "lwage", "educ", ["nearc4"], exog=controls,
                       ar_vcov="HC3")["ar_ci"]      # (0.0277, 0.2821)
```

The homoskedastic interval is not valid under heteroskedasticity, weak
instruments or not. Use `ar_vcov` whenever you would have used a robust
standard error for the reduced form, and `cluster=` when the data are
clustered.

## Bounds when the outcome is binary

Chapter 26 bounds the survivor average effect in a trial where the outcome
is binary. The bounds trim a share of the survivors in one arm. On a binary
outcome most observations are tied at the trimming point, and the rule that
keeps every tied observation trims nothing. The two bounds then coincide.
`trimming='exact'` splits the ties and gives the sharp bounds of the book,
-0.176 and -0.019.

```python
sp.lee_bounds(df, "y", "z", "survived", trimming="exact")
```

StatsPAI warns when `trimming='quantile'`, the default that matches Stata's
`leebounds` on continuous outcomes, cannot trim because of ties.

## Where StatsPAI differs from the book's code

- **Egger regression.** `sp.mr_egger` first orients every variant so that
  its association with the exposure is positive, which is how the method is
  defined. The book regresses on the variants as coded. On its BMI and
  blood-pressure data 78 of 160 variants are flipped and the slope is 0.45
  rather than 0.32.
- **Principal scores.** The book's weighting estimator divides the weighted
  control sum by the complier share. `sp.principal_strat` normalises the
  weights, so the estimate does not depend on the level of the outcome. On
  the JOBS II data the complier effect is 0.093 normalised and 0.169 as in
  the book. Both add up to the same intention-to-treat effect.
- **Propensity-score stratification.** `sp.match(method='stratify')` puts
  a unit whose score equals a quintile boundary in the stratum above; R's
  `cut` puts it in the stratum below. Three NHANES units move and the
  estimate is -0.117 rather than -0.116. Cutting the score yourself and
  passing the strata to `sp.difference_in_means(blocks=)` reproduces the
  book to thirteen digits.
- **Fieller-Anderson-Rubin.** The book uses the normal distribution;
  StatsPAI refers the statistic to `F(1, n - k)`. The intervals agree to the
  third decimal on the book's examples.

## What a paper written today would add

- Cross-fitting once the nuisance models are flexible: `sp.aipw`,
  `sp.dml`.
- Overlap diagnostics before any weighting: `sp.overlap_plot`,
  `sp.trimming`, `sp.overlap_weights`.
- For instruments, the effective F and the tF critical value, which
  `sp.anderson_rubin_test` reports next to the interval.
- For discontinuities, the density test and bandwidth sensitivity:
  `sp.rddensity`, `sp.rdbwsensitivity`.
- For mediation, `sp.mediate(inference='robust')` and
  `sp.mediate_sensitivity`, since sequential ignorability cannot be tested.

## Run the check yourself

```bash
STATSPAI_DING_DIR=files Rscript tests/external_parity/ding_first_course_reference.R
STATSPAI_DING_DIR=files pytest tests/external_parity/test_ding_first_course.py
```

The R script reruns the deterministic part of the book's programs and
writes the answer key. The test recomputes each number with StatsPAI.
