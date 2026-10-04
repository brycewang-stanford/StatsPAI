# Croissant, *Microeconometrics with R*: what the book and `micsr` showed about StatsPAI

Started 2026-10-04. Worktree `.claude/worktrees/croissant-textbook`, branch
`wt/croissant-textbook`.

## What was examined

Yves Croissant, *Microeconometrics with R*, Chapman and Hall/CRC
(doi:10.1201/9781003100263; Crossref dates it 2024-12-11, the author cites
it as 2025 in the package DESCRIPTION, and that year is used here). The Quarto sources of the 14 chapters and the
companion R package `micsr` 0.1-5 sit in
`改进建议-收集整理/Croissant-Microeconometrics-R/`, which is gitignored. The
package is GPL. Nothing from its source or its datasets is in this
repository. It was run as a program and its printed numbers were compared.

The book has three parts. The first two (OLS, its properties, maximum
likelihood, non-spherical errors, endogeneity, treatment effects, spatial
models) cover ground that earlier textbook passes already went over. The
third part is where the book is distinctive, and where the work went.

| Chapter | Estimators and tests | R functions |
| --- | --- | --- |
| Binomial | LPM, probit, logit, IV probit, ordered models, score and conditional moment tests, pseudo R-squared | `micsr::binomreg`, `ivldv`, `ordreg`, `cmtest`, `rsq`, `scoretest` |
| Count | Poisson, NB1, NB2, log-normal mixing, hurdle, zero inflation, IV and GMM for an exponential mean, endogenous switching | `micsr::poisreg`, `expreg`, `escount`, `pscl::hurdle`, `zeroinfl` |
| Censored and truncated | tobit by ML, two-step, symmetrically trimmed least squares, IV tobit, sample selection, conditional moment tests | `micsr::tobit1`, `ivldv`, `cmtest`, `sampleSelection::selection` |
| Duration | Kaplan-Meier, Weibull and other AFT models, Cox, Weibull with gamma heterogeneity | `survival`, `micsr::weibreg` |
| Discrete choice | multinomial, conditional, nested and mixed logit | `mlogit` |

The book is from 2025 but several of its methods are older than the
current practice in their area. The treatment-effect chapter stops at
two-period difference in differences, propensity-score stratification
and the original synthetic control.
StatsPAI is ahead of the book there and nothing was taken from it.

## Method

1. The book's models were run in R on the datasets that ship with `micsr`
   (`mode_choice`, `trips`, `charitable`, `federiv`, `trade_protection`,
   `cigmales`, `housprod`) and on two simulated datasets for ordered and
   duration models.
2. The same data went through the StatsPAI counterpart where one existed.
3. Where the two differed, a third implementation decided
   (`MASS::polr`, `stats::glm` with a tight tolerance, `pscl`, a direct
   evaluation of the likelihood).
4. Where StatsPAI had no counterpart, the method was written from the
   papers and pinned to Stata 18 or to `micsr` on synthetic data that is
   committed with its generator.

## Existing estimators: what agreed and what did not

| Model | StatsPAI | Reference | Result |
| --- | --- | --- | --- |
| Probit, logit | `sp.probit`, `sp.logit` | `micsr::binomreg` | coefficients 1e-12, SE 6e-13 |
| Poisson | `sp.poisson` | `glm` (epsilon 1e-13) | 1e-11. `micsr::poisreg` is 3e-4 away from both |
| NB2 | `sp.nbreg` | `micsr::poisreg` | 6e-7, same log-likelihood |
| NB1 | `sp.nbreg(dispersion='constant')` | `micsr::poisreg` | 6e-5, log-likelihood equal to 8 digits |
| Tobit | `sp.tobit` | `micsr::tobit1` | coefficients and SE to 6 digits |
| Bivariate probit | `sp.biprobit` | `micsr::bivprobit` | coefficients equal. SE of rho 0.062 against 0.163 |
| Ordered logit and probit | `sp.ologit`, `sp.oprobit` | `micsr::ordreg`, `MASS::polr` | equal to `polr`. `ordreg` SE differ |
| Weibull AFT, Cox | `sp.survreg`, `sp.cox` | `survival` | 6 digits |
| Hurdle, ZIP, ZINB | `sp.hurdle`, `sp.zip_model`, `sp.zinb` | `pscl` | see finding 1 |

Two rows need a comment.

**Bivariate probit, standard error of rho.** `micsr::bivprobit` prints
0.163 on the `housprod` example, StatsPAI 0.062. The curvature of the
log-likelihood in rho alone, computed in R with `mvtnorm` at the common
estimate, gives 0.060. That is a lower bound for the standard error (it
holds the other parameters fixed) and it is consistent with 0.062, not
with 0.163. The source of the `micsr` number was not traced.

**Ordered models.** On simulated data `micsr::ordreg` gives standard
errors of 0.0490 and 0.0770 for the two slopes of an ordered logit.
`MASS::polr` gives 0.0514 and 0.0989, and StatsPAI gives the `polr`
numbers.

In both cases StatsPAI was left as it is.

## What was wrong, and what changed

| # | Finding | Status |
| --- | --- | --- |
| 1 | `sp.zip_model` and `sp.zinb` returned NaN for every standard error when one inflation coefficient was not identified (quasi-complete separation, as on `trips`). `1 / (1 + exp(-x))` overflowed inside the complex-step Hessian | fixed. All three zero-modified models now warn and name the coefficient |
| 2 | No probit or tobit with endogenous regressors (`ivprobit`, `ivtobit` in Stata, `micsr::ivldv`) | added `sp.ivprobit`, `sp.ivtobit` |
| 3 | No IV estimator for an exponential mean (`ivpoisson gmm`, `micsr::expreg`) | added `sp.ivpoisson` |
| 4 | No specification test for the normality and homoskedasticity that probit and tobit rest on (`micsr::cmtest`) | added `sp.cmtest` |
| 5 | `sp.from_stata` did not know `ivprobit`, `ivtobit`, `ivpoisson` | added |
| 6 | No general Vuong test (`micsr::ndvuong`, `pscl::vuong`); fitted models did not expose per-observation log-likelihoods | added `sp.vuong`; count, zero-modified, logit and probit fits carry `data_info['llobs']` |
| 7 | The Vuong statistic that `sp.zip_model` and `sp.zinb` report on their own was computed against a comparison model that was not at its maximum likelihood | fixed, correctness |

### 1. Zero-inflated models under separation

In the `trips` data the regressor `workschl` separates the zero part. Its
coefficient runs to several hundred and the likelihood is flat in it. R
`pscl::zeroinfl` reports a standard error of 2.4e4 for that coefficient
and ordinary ones for the other 19. StatsPAI returned NaN for all 20.

The per-observation likelihood used `1 / (1 + exp(-x))`. The overflow in
one row became a NaN in the complex-step Hessian and the inverse was NaN
everywhere. The likelihood now uses `log expit` and `logaddexp` forms that
are safe under a complex step. On `trips` the 19 identified standard
errors are within 0.5% of those of `pscl` (0.3% for the negative
binomial version) and the coefficients within 1e-4. StatsPAI's
log-likelihood is the higher of the two by 9e-7, so the remaining gap is
on the `pscl` side of the comparison and was not traced further.

`sp.hurdle` had the right numbers already. It returned a standard error
of 1.6e14 for the separated coefficient without a word. All three now
raise a `ConvergenceWarning` that names the coefficient.

### 2. `sp.ivprobit` and `sp.ivtobit`

The model is the one in Stata: a latent outcome equation, a linear
reduced form for each endogenous regressor, jointly normal errors.

- **Maximum likelihood.** The covariance of the reduced-form errors is
  parametrised by its Cholesky factor, and the outcome error by its
  loading on the standardised reduced-form errors. For the probit this
  keeps `Var(u) = 1` without a constraint. Stata's ancillary parameters
  (`/athrho2_1`, `/lnsigma2`, and so on) are then obtained by the delta
  method. Scores come from the complex-step method and the optimum from
  Newton steps, as in `sp.tobit`.
- **Two-step.** Newey's (1987) minimum chi-squared estimator, following
  the Methods and Formulas of the Stata manual. The Wald test of
  exogeneity is the test on the first-stage residuals in the
  control-function fit.

Evidence, on 14 blocks against Stata 18.

| Block | Coefficients | Standard errors |
| --- | --- | --- |
| Two-step, probit and tobit, 1 and 2 endogenous regressors, two-limit | 4e-13 to 2e-8 | 2e-12 to 8e-9 |
| ML, `vce(oim)`, `vce(robust)`, `vce(cluster)` | 9e-9 or better | 8e-8 or better |

The ML comparison needed one change on the Stata side. `ml` stops at
`nrtolerance(1e-5)`, which leaves coefficients up to 3e-6 from the
optimum. The do-file tightens the stopping rule. With Stata's default the
gap is 3e-6 and is the stopping rule alone. The log-likelihoods agree to
1e-8 either way.

`micsr::ivldv` was the prompt for this work but could not serve as a
reference. Its `minchisq` method stops with an error in version 0.1-5, and
its `twosteps` method is the Rivers-Vuong estimator, which StatsPAI does
not offer under that name (see open items).

### 3. `sp.ivpoisson`

GMM for `E[y | x] = exp(x'b)` with instruments. Two moment conditions.
The additive one is Stata's default. The multiplicative one is Mullahy
(1997), which the book uses for the cigarette example, and it is the valid
one when an omitted variable enters the exponential.

Twelve blocks against Stata 18 `ivpoisson gmm` agree to 2e-7 or better on
coefficients, standard errors and Hansen's J. Two conventions were read
off Stata's output and are asserted in the test.

- `wmatrix()` follows `vce()` when it is not given. `sp.ivpoisson` does
  the same, and takes `wmatrix=` when they should differ.
- The J statistic of one-step GMM is divided by the error variance. It is
  chi-squared only under homoskedastic moments. The result carries a note
  saying so.

### 4. `sp.cmtest`

Conditional moment tests (Newey 1985, Tauchen 1985) with the variance of
Skeels and Vella (1999, eq. 2.13). Every power of the unobserved error is
replaced by its expectation given what is observed, by the recursion for
truncated normal moments. The derivatives of the moments and the Hessian
come from the complex-step method, so the same 40 lines serve the probit
and the tobit with one or two limits.

| Model | Tests | Against `micsr::cmtest` |
| --- | --- | --- |
| Tobit | normality, heteroskedasticity, skewness, kurtosis, each in Hessian and outer-product form | 4e-11 or better |
| Probit | normality, heteroskedasticity | 8e-7 and 4e-8 |

The probit gap is `micsr::binomreg` stopping 1e-8 from the optimum. Both
sides' coefficients are compared in the test.

`micsr` also offers `test = "reset"` for the probit. Its moments, as the
book writes them, are the squared and cubed index times the generalised
residual, which are the normality moments. `micsr` prints a different
number for the two (3.22 against 4.70 on `mode_choice`) and the reason was
not found. `sp.cmtest` does not offer `reset`. The docstring says that a
rejection of normality in a probit cannot be told from a misspecified
index.

Under a true null the tobit tests reject 5% of the time (200 replications
in the test file).

### 5. `sp.vuong` and the statistic inside the zero-inflated models

`sp.vuong(model1, model2)` computes Vuong's (1989) statistic from the
per-observation log-likelihoods of two fits, with the AIC and BIC
corrections. Those log-likelihoods were not on the result objects before.
They now are, for Poisson, negative binomial, the three zero-modified
models, logit and probit, and each was checked against R's own density at
R's own estimate (4e-14 for Poisson, 7e-8 at worst).

Five model pairs agree with the statistic rebuilt from `pscl` fits to
6e-8, and with what `pscl::vuong` prints. One difference is deliberate.
`pscl::vuong` counts parameters with `length(coef())`, which does not
include the dispersion of a negative binomial. When exactly one of the
two models is negative binomial its corrections are off by one parameter.
StatsPAI counts every estimated parameter. The test reproduces the `pscl`
print from the `pscl` count, so the difference is pinned and explained.

Writing the reference exposed finding 7. `sp.zip_model` reported a Vuong
statistic of 13.42 on the test data where the test is 8.69, and `sp.zinb`
9.16 where it is 3.70. The Poisson (negative binomial) likelihood in the
comparison was evaluated at the count coefficients of the zero-inflated
fit. Those are not the maximum likelihood estimates of the plain model,
so the plain model's likelihood was understated. It is now fitted on its
own.

The docstrings also carry a warning that the book does not: the Vuong
statistic is not a valid test for zero inflation, because the plain model
sits on the boundary of the zero-inflated one (Wilson 2015). Stata 18
refuses the `vuong` option of `zip` with "Vuong test is not appropriate
for testing zero inflation" and computes it only under `forcevuong`. Under
that option it returns 8.6853869 and 3.6998482 on the test data, which is
what StatsPAI now returns (2e-10 and 1e-9 apart). The statistic is kept as
a description of fit.

The non-degenerate version of the test in `micsr::ndvuong` was not taken.
It needs simulated critical values and has no second implementation to
pin it to.

## Left open

Methods of the book that StatsPAI still lacks, in the order I would take
them.

| Item | Book | Reference to pin it to | Note |
| --- | --- | --- | --- |
| Non-degenerate Vuong test | `micsr::ndvuong` | `micsr` only | the classical test is done; `data_info['llobs']` is still missing on `sp.tobit`, `sp.ologit`, `sp.mlogit`, survival models |
| Tobit by a two-step method and by symmetrically trimmed least squares | `micsr::tobit1(method=)` | `micsr` | SCLS is the robust alternative `sp.cmtest` points to when it rejects |
| Endogenous switching and sample selection for counts | `micsr::escount` | Stata `etpoisson`, `heckpoisson` | |
| Weibull with gamma heterogeneity | `micsr::weibreg(mixing=TRUE)` | Stata `streg, frailty(gamma)` | |
| Poisson with log-normal mixing | `micsr::poisreg(mixing="lognorm")` | `micsr` | Gauss-Hermite quadrature |
| Rivers-Vuong two-step probit (2SCML) with its own standard errors | `micsr::ivldv(method="twosteps")` | `micsr` | the coefficients are already the control-function fit inside `sp.ivprobit` |
| Nested logit | `mlogit` | `mlogit`, Stata `nlogit` | |
| Pseudo R-squared family for binary models (McKelvey-Zavoina, Tjur, Estrella) | `micsr::rsq` | `micsr`, `DescTools` | only McFadden is reported today |
| `sp.cmtest` for weighted fits and for `sp.ivtobit` | | | |

Not taken, with the reason.

- **Propensity-score stratification** (`micsr::pscore`). The algorithm
  picks strata by repeated t-tests. Matching and
  weighting estimators with known properties are already in `sp.match`,
  `sp.ipw`, `sp.aipw`.
- **The generalised production function** (`micsr::zellner_revankar`). It appears in the book as an example of
  maximum likelihood, not as a tool.

## How to reproduce

```bash
# Stata 18 references (writes the CSV and the JSON next to the do-file)
cd tests/reference_parity/_fixtures
stata-mp -b do _generate_ivprobit_ivtobit_stata.do
stata-mp -b do _generate_ivpoisson_stata.do
Rscript _generate_cmtest_micsr.R          # needs micsr >= 0.1-5
Rscript _generate_vuong_pscl.R            # needs pscl, MASS

pytest tests/reference_parity/test_ivprobit_ivtobit_stata_parity.py \
       tests/reference_parity/test_ivpoisson_stata_parity.py \
       tests/reference_parity/test_cmtest_micsr_parity.py \
       tests/reference_parity/test_vuong_pscl_parity.py \
       tests/test_zeroinflated_separation.py -q
```
