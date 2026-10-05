# Ramirez-Hassan, *Introduction to Bayesian Econometrics* (2026): review

Source: the author's repositories, the book (`IntroductionBayesianEconometricsBook`,
14 chapters in R Markdown, about 18,000 lines of R) and its Shiny
application (`BSTApp`, GPL-3, with 28 data files). The chapters cover the
conjugate families (3), simulation and convergence diagnostics (4),
univariate regression models (6), multivariate models (7), time series
(8), longitudinal models (9), Bayesian model averaging and marginal
likelihoods (10), mixtures and splines (11), Bayesian machine learning
(12), causal inference (13) and approximate methods (14).

The book relies on `MCMCpack`, `bayesm`, `coda`, `BMA`, `dlm`, `stochvol`
and `bvartools`, and writes many samplers by hand. Several of those
packages are old; their conventions were treated as references to
reproduce on request, not as defaults to adopt. All of them are GPL.
Nothing was ported: the samplers here are written from the full
conditionals in the cited papers and compared with the packages as black
boxes.

## Where StatsPAI stood

`statspai.bayes` had ten PyMC estimators for causal designs and
`sp.bvar` had the Minnesota prior in closed form. There was no Bayesian
linear regression, no logit or probit, no convergence diagnostic that
takes a chain, no marginal likelihood, no model averaging outside
Mendelian randomisation. A reader of chapters 3 to 10 had nothing to call.
PyMC is also an optional dependency that the test environment does not
install, so the Bayesian estimators that did exist were rarely exercised.

## Method

MCMC has no "same digits" reference. Two correct samplers differ by Monte
Carlo error, and agreeing with another package to two digits says little.
The evidence was built in three layers.

1. **Deterministic functions against R, digit for digit.** Convergence
   diagnostics are functions of a chain; model averaging by BIC or under
   a g-prior is a function of the data. These are compared with `coda`,
   `BMA` and `BMS` on committed files at 1e-9.
2. **Every sampler against an exact posterior.** With two coefficients
   and at most one auxiliary parameter the posterior can be integrated on
   a grid. The integrand is written in the test from `scipy.stats`
   densities and shares no code with the sampler. Posterior means must be
   within four Monte Carlo standard errors of the exact means, standard
   deviations within five percent, and each marginal-likelihood estimator
   must reproduce the exact normalising constant. For the hierarchical
   models the fixed and random effects are integrated out analytically
   (Gaussian) or by Gauss-Hermite quadrature (logit, Poisson).
3. **The book's examples on its own data** against long runs of
   `MCMCpack` and `bayesm`, as a screen.

For the PyMC estimators an isolated environment was built (PyMC 5.25,
numba backend, since pytensor's C compilation fails on this machine) and
coverage was simulated under known truth.

## What was wrong in StatsPAI

1. **`sp.bayes_iv` and `sp.bayes_hte_iv` reported intervals that were too
   narrow under endogeneity.** The docstring promised a joint model of the
   first stage and the structural equation with an LKJ prior on the error
   correlation. The code ran OLS for the first stage, kept the residuals,
   and regressed the outcome on the treatment and those residuals. Given
   the residuals the coefficient on the treatment has variance
   `sigma_e^2 / (D_hat' D_hat)` with `sigma_e^2 = Var(eps | v)`, which is
   `(1 - corr(v, eps)^2)` times the structural error variance that 2SLS
   uses. The missing part is the uncertainty of the first stage. A numpy
   replica of the model shows the ratio `sqrt(1 - corr^2)` exactly (0.866,
   0.600, 0.312 at correlations 0.5, 0.8, 0.95) and coverage of 90, 76 and
   47 percent for a nominal 95. With PyMC on 60 samples of 400 observations
   at a correlation of 0.9: median posterior sd 0.44 of the 2SLS standard error, 35 of 60 intervals cover the true effect. The fix is one line: the
   residual in the outcome equation is `D - first_stage`, with
   `first_stage` the model's own expression. That is the Cholesky
   factorisation of the bivariate normal, so the model is now the joint
   one. After the change: ratio 1.02, 58 of 60 intervals cover.
2. **`prior_first_stage_sigma` was dead.** `sp.bayes_iv` accepted it,
   recorded it in `model_info`, and gave the first-stage coefficients the
   prior scale `prior_coef_sigma`.
3. **`sp.bayes_fuzzy_rd` assumed independent errors in the two reduced
   forms.** The variance of a ratio of two jumps has a covariance term,
   `v_Y - 2 LATE c_YD + LATE^2 v_D`, and the outcome's reduced-form error
   contains `LATE * eps_D`. With independent errors the posterior drops
   `c_YD`. A numpy replica gives a posterior sd between 0.95 and 1.23 of
   the correct one across signs of the effect and of the selection, and
   coverage between 93.5 and 98.5 percent. The outcome equation now has
   `lambda * (D - mu_D)` in its mean.
4. **`sp.bvar(...).summary()` printed coefficient rows as 0, 1, 2.**

Items 1 and 3 are the same mistake: a two-step Bayesian model in which
the first step's uncertainty, or its correlation with the second, never
reaches the posterior. `sp.bayes_mte` already offers `first_stage='joint'`
and documents the plug-in alternative; it was not changed.

## What was added

| function | what it does | evidence |
| --- | --- | --- |
| `sp.bayes_regress` | ten likelihoods, normal prior, Gibbs / data augmentation / random-walk Metropolis | exact posterior on a grid, all ten |
| `sp.bayes_mixed` | random intercepts and slopes; normal, logit, Poisson | exact posterior (analytic and Gauss-Hermite) |
| `sp.bma` | BIC with Occam's window; g-prior by enumeration or MC3 | `BMA`, `BMS` at 1e-9; brute force over 512 models |
| `sp.bayes_factor`, `sp.savage_dickey` | model comparison | exact identities under the conjugate prior |
| `sp.bayes_bootstrap` | Rubin's bootstrap | Rubin's variance of a mean |
| `sp.mcmc_summary`, `mcmc_ess`, `hpd_interval`, `geweke_diag`, `raftery_diag`, `heidel_diag`, `gelman_rubin` | diagnostics for any chain | `coda` at 1e-9 |

Design choices worth recording.

- One dispatcher, `sp.bayes_regress(model=)`, in line with `sp.synth` and
  `sp.dml`. A likelihood is a small class with `names`, `sample`, `to_u`
  and `log_kernel`. Mode finding, the Laplace and Gelfand-Dey marginal
  likelihoods and the Metropolis proposals all use `log_kernel`, so a new
  likelihood needs a sampler and one function.
- The default prior variance (1000) is the book's. It is not vague for a
  regressor on a small scale, and a silent informative prior is the
  easiest way to get a wrong Bayesian answer. After sampling, each
  coefficient's posterior is compared with its prior and a warning names
  the coefficients for which the default prior matters.
- Every fit computes the effective sample size and the split
  Gelman-Rubin factor and warns when they are poor.
- The identity default for the scale of the random-effects prior is the
  book's and MCMCpack's. It is kept, with the prior-share warning, because
  a data-dependent default would make the prior a function of the
  outcome. The warning names the scale the data suggest.
- The hierarchical logit and Poisson samplers add one Gibbs step that
  shifts a fixed effect and the matching random effects in opposite
  directions. The likelihood does not change under that shift and its
  full conditional is normal. Without it the intercept and the random
  intercepts mix slowly.
- `sp.bma` searches Occam's window by branch and bound on the criterion
  itself. `bicreg` takes the best 150 models of each size and filters
  them, so it can miss a model when that cap binds.

## Where the reference packages were not followed

- **`MCMCquantreg` fixes the scale of the asymmetric Laplace likelihood at
  one.** The posterior spread then depends on the units of the outcome.
  `sp.bayes_regress(model='quantile')` estimates the scale (Kozumi and
  Kobayashi 2011); `scale=1.0` gives the reference.
- **The book's ordered probit is a restricted model.** Section 6.6 passes
  a design without a constant to `bayesm::rordprobitGibbs`, whose first
  cutpoint is fixed at zero. With a constant, `rordprobitGibbs`
  and `sp.bayes_regress` agree to 0.04 of a posterior sd in slopes and
  cutpoints, and ours is within 0.02 of a standard error of maximum
  likelihood everywhere. The book's version is a different model: its
  coefficient on good self-rated health is 0.725 where maximum likelihood
  gives 0.032, 8.6 standard errors away. `sp.bayes_regress` always has
  free cutpoints.
- **`MCMChlogit` and `MCMChpoisson` carry an observation-level error.**
  `sp.bayes_mixed` fits the standard generalised linear mixed model.
- **`MCMChregress` understates the uncertainty of the fixed effects.** On
  the public-capital panel of section 9.1 (48 states, 17 years) it reports
  a posterior sd of 0.008 for the intercept. The model itself puts a floor
  under that number: with a state-effect variance of 0.106 the average of
  48 state effects has sd `sqrt(0.106 / 48) = 0.047`. Run as a black box,
  its draws of the fixed effects and of each state effect are serially
  uncorrelated and have the spread of the conditional distributions
  (`sigma / sqrt(n)`), so the direction "intercept up, every state effect
  down" is never explored. `sp.bayes_mixed` draws the fixed effects with
  the state effects integrated out (Chib and Carlin's point) and gives
  0.17 under the same prior. Independent evidence that this is the right
  number: the exact-posterior tests, and `lme4::lmer`, whose standard
  errors `sp.bayes_mixed` reproduces within 3 percent once the prior of
  the variance component is on the right scale (intercept 2.151 against
  2.149, sd 0.139 against a standard error of 0.136).
- **The book's prior for the variance of the state effects is not
  vague.** `r = 5, R = 1` is `InvWishart(5, 5)` on the variance of an
  effect in log points. It contributes 91 percent of the sum of squares in
  the full conditional, and both samplers return a variance of about
  0.106 where restricted maximum likelihood gives 0.0076. This is the
  same trap as a default prior variance on a badly scaled regressor.
  `sp.bayes_mixed` now reports the prior's share of each variance
  component and warns above 25 percent.
- **`bicreg` rounds R-squared.** Its BIC is computed from an R-squared
  with five decimals. Rounding ours the same way reproduces its BIC to
  1e-9, and its posterior probabilities applied to our per-model fits
  reproduce its averaged coefficients to 1e-8. Our own output is the
  unrounded one, which differs in the third digit.
- **`bic.glm` uses one dispersion for the gamma family**, the Pearson
  dispersion of the model with all candidates, for both BIC and standard
  errors. Found by backing the dispersion out of its BIC values. Followed.
- **`gelman.diag`'s multivariate factor** has `(1 + 1/p)` with `p` the
  number of parameters where Brooks and Gelman (1998) have `(1 + 1/m)`
  with `m` the number of chains. Ours follows the paper; the test rebuilds
  coda's number from the same eigenvalue.
- **`raftery.diag` prints the dependence factor to three significant
  digits.** Ours is unrounded.

## The book's examples

On the book's data, with the book's priors, against 60,000 to 300,000
draws of the reference sampler. "Gap" is the largest difference of
posterior means in units of the posterior sd; "sd ratio" is ours over the
reference's, smallest and largest across parameters.

| section | model, reference | n | gap | sd ratio | our min ESS / draws | our time |
| --- | --- | --- | --- | --- | --- | --- |
| 6.1 | linear, `MCMCregress` | 335 | 0.009 | 0.993 to 1.007 | 38,336 / 40,000 | 1 s |
| 6.8 | tobit, `MCMCtobit` | 335 | 0.014 | 0.990 to 1.001 | 15,793 / 40,000 | 3 s |
| 6.9 | median, `MCMCquantreg` | 335 | 0.017 | 0.995 to 1.009 | 10,188 / 40,000 | 2 s |
| 6.9 | 0.9 quantile, `MCMCquantreg` | 335 | 0.036 | 0.988 to 1.005 | 5,963 / 40,000 | 2 s |
| 6.3 | probit, `rbprobitGibbs` | 12,975 | 0.075 | 0.983 to 1.020 | 1,460 / 20,000 | 42 s |
| 6.2 | logit, `MCMClogit` | 12,975 | 0.043 | 0.995 to 1.015 | 2,738 / 20,000 (thin 5) | 42 s |
| 6.6 | ordered probit slopes, `rordprobitGibbs` with a constant | 12,975 | 0.023 | 0.992 to 1.012 | 15,326 / 20,000 | 104 s |
| 6.6 | ordered probit cutpoints | 12,975 | 0.040 | 1.006 to 1.009 | 9,185 / 20,000 | |
| 9.1 | hierarchical normal, `MCMChregress` | 816 | variance components within 5 percent; fixed effects see above | | | |

Times are on a heavily loaded laptop. The quantile rows use `scale=1.0`
to match the reference. With the scale estimated (0.38 at the median,
0.17 at the 0.9 quantile) the posterior sd of the national-team
coefficient is 0.099 and 0.118, against 0.168 and 0.298 with the scale
fixed at one: the fixed scale is not a neutral default.

The book reads section 6.1 as "playing for the national team raises
market value by `exp(0.85) - 1`, about 134 percent"; the posterior mean
here gives the same number.

## Dated material in the book

- Random-walk Metropolis with a hand-tuned scale is the book's sampler for
  logit and count models. It is kept here because it is transparent and
  checkable, with the proposal shaped by the posterior curvature. A
  Polya-Gamma Gibbs sampler would mix better for the logit.
- Geweke, Raftery-Lewis and Heidelberger-Welch are single-chain
  diagnostics from the 1980s and 1990s. They are implemented because the
  book and `coda` users expect them. The fit-time check uses the effective
  sample size and the split Gelman-Rubin factor.
- Marginal likelihoods under default "vague" priors are reported
  throughout the book. A Bayes factor moves with the prior variance
  without bound, and the docstrings say so.

## Open items

- Multinomial probit and logit, multivariate probit, SUR and IV by Gibbs
  (chapter 7). An IV sampler in NumPy would give `sp.bayes_iv` a path
  that does not need PyMC.
- State-space models, stochastic volatility (chapter 8). `sp.arima` and
  `sp.garch` are the frequentist counterparts; there is no Kalman
  filter / FFBS entry point.
- Dirichlet process mixtures, Bayesian splines (chapter 11); Bayesian
  lasso, stochastic search variable selection, BART, Gaussian processes
  (chapter 12); ABC, synthetic likelihood, INLA, variational Bayes
  (chapter 14).
- Chapter 13's Bayesian exponentially tilted empirical likelihood, general
  Bayes posteriors and doubly robust Bayesian inference have no
  counterpart.
- `sp.bayes_mte(first_stage='plug-in')` has the same structure as the old
  `bayes_iv`. It is documented as a choice, with `'joint'` available. Its
  default deserves a coverage study of its own.
- The other PyMC estimators (`bayes_did`, `bayes_rd`, `bayes_its`,
  `bayes_synth`, `bayes_dml`) were not put through a coverage simulation.
- No marginal likelihood for hierarchical models; `sp.bayes_regress` has
  no `weights=` or offset; `sp.stata` does not translate the `bayes:`
  prefix (Stata's default priors differ, so a translation would have to
  state them).
- PyMC tests do not run in the default environment. On this machine they
  need `PYTENSOR_FLAGS="cxx=,mode=NUMBA"`.

## Rerun recipe

```bash
# committed evidence, no book data needed
pytest tests/reference_parity/test_bayes_mcmc_parity.py \
       tests/reference_parity/test_bayes_regress_exact_posterior.py \
       tests/reference_parity/test_bayes_mixed_exact_posterior.py \
       tests/test_bayes_regress.py
# regenerate the R references (coda, BMA, BMS)
python tests/reference_parity/_fixtures/_generate_bayes_mcmc_data.py
Rscript tests/reference_parity/_fixtures/_generate_bayes_mcmc_R.R
# the book's examples (MCMCpack, bayesm; about half an hour)
export STATSPAI_RAMIREZ_HASSAN_DIR=<.../BSTApp/DataApp>
Rscript tests/external_parity/ramirez_hassan_bayes_reference.R
pytest tests/external_parity/test_ramirez_hassan_bayes.py
# PyMC estimators
PYTENSOR_FLAGS="cxx=,mode=NUMBA" pytest tests/test_bayes_iv.py \
    tests/test_bayes_hte_iv.py tests/test_bayes_fuzzy_rd.py
```
