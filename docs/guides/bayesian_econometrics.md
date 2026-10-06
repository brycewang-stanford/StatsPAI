# Bayesian econometrics without PyMC

StatsPAI has two Bayesian toolkits.

`statspai.bayes` holds the causal designs (difference in differences,
regression discontinuity, instrumental variables, synthetic control,
marginal treatment effects). They are written in PyMC and need the `bayes`
extra.

`statspai.mcmc` holds the models of a first course in Bayesian
econometrics. They are plain NumPy samplers and need nothing beyond the
core install. This guide covers them. It follows the chapters of
Ramirez-Hassan, *Introduction to Bayesian Econometrics* (2026), whose
examples use the R packages `MCMCpack`, `bayesm`, `coda` and `BMA`.

```python
import statspai as sp
```

## Chapter by chapter

| chapter | the book runs | StatsPAI |
| --- | --- | --- |
| 3 | conjugate normal / inverse-gamma regression, marginal likelihood, Bayes factor | `sp.bayes_regress(model='conjugate')`, `.log_marginal_likelihood()`, `sp.bayes_factor` |
| 4 | Gibbs, Metropolis-Hastings, convergence diagnostics | the samplers below; `sp.geweke_diag`, `sp.raftery_diag`, `sp.heidel_diag`, `sp.gelman_rubin`, `sp.mcmc_ess` |
| 6 | `MCMCregress` | `sp.bayes_regress(model='normal')` |
| 6 | `MCMClogit`, `rbprobitGibbs` | `model='logit'`, `model='probit'` |
| 6 | `rordprobitGibbs` | `model='oprobit'` |
| 6 | negative binomial by Metropolis-Hastings | `model='negbin'`, also `model='poisson'` |
| 6 | multinomial logit by Metropolis-Hastings | `model='mlogit'` |
| 6 | `MCMCtobit` | `model='tobit', lower=, upper=` |
| 6 | `MCMCquantreg` | `model='quantile', quantile=` |
| 6 | heteroskedastic errors by a scale mixture (exercise) | `model='t', dof=` |
| 6 | Bayesian bootstrap | `sp.bayes_bootstrap` |
| 8 | `dlm::dlmModReg`, `dlmMLE`, `dlmFilter`, `dlmSmooth`, `dlmGibbsDIG` | `sp.dlm("y ~ x", df)`, `method='mle'` or `'gibbs'` |
| 8 | Minnesota-prior VAR | `sp.bvar` |
| 7 | `MCMCregress` on several outcomes, `bayesm::rsurGibbs` | `sp.bayes_sur(["y1 ~ x", "y2 ~ z"], df)` |
| 7, 13 | `bayesm::rivGibbs` | `sp.bayes_ivreg("y ~ x + (d ~ z)", df)` |
| 9 | `MCMChregress`, hierarchical logit and Poisson | `sp.bayes_mixed(family='normal' / 'logit' / 'poisson')` |
| 10 | `BMA::bicreg`, `BMA::bic.glm` | `sp.bma(method='bic')` |
| 10 | g-prior model averaging, MC3 | `sp.bma(method='gprior')` |
| 10 | Savage-Dickey, Chib, Gelfand-Dey | `sp.savage_dickey`; `.log_marginal_likelihood(method=)` |
| 12 | Bayesian lasso, stochastic search variable selection | `sp.bayes_shrink(prior='lasso' / 'ssvs')` |
| 14 | variational Bayes for the linear model | `sp.bayes_regress(inference='vb')` |
| 13 | Bayesian IV, DiD, RD | `sp.bayes_iv`, `sp.bayes_did`, `sp.bayes_rd`, `sp.bayes_fuzzy_rd` (PyMC) |

Not covered yet: multinomial probit and logit, multivariate probit, SUR by
Gibbs, more than one endogenous regressor, stochastic volatility, Dirichlet process
mixtures, BART and Gaussian processes, approximate Bayesian computation
and variational Bayes. The frequentist counterparts of several of these
are in StatsPAI (`sp.mlogit`, `sp.sureg`, `sp.arima`, `sp.garch`).

## A regression

```python
fit = sp.bayes_regress(
    "lv ~ Perf + Age + Age2 + NatTeam + Goals + Exp + Exp2", players,
    prior_var=1000, sigma2_prior=(0.001, 0.001), draws=20000, seed=1,
)
print(fit.summary())
fit.prob("NatTeam > 0")          # posterior probability of a statement
fit.conf_int(kind="hpd")         # highest posterior density intervals
fit.predict(new, what="interval")
fit.diagnostics()                # Geweke, Heidelberger-Welch, Raftery-Lewis
fit.plot()                       # traces and densities
```

The table has the posterior mean and standard deviation, the Monte Carlo
standard error of the mean, the effective sample size, the credible
interval and the posterior probability that the parameter is positive.

`draws` is the number of draws kept. Each chain runs
`burnin + draws * thin` iterations. `chains=4` starts four chains from
dispersed values and adds the Gelman-Rubin factor to `fit.diagnostics()`.

## Priors

Coefficients have a normal prior, `prior_mean` and `prior_var`. The
variance can be one number, one per coefficient or a full matrix.

The default variance is 1000 (100 for logit, probit and ordered probit,
whose coefficients live on a bounded scale). That is vague only when the
regressors are on a moderate scale. A wage regression with income in
dollars, or a regressor measured in thousandths, can have a coefficient in
the hundreds, and then a standard deviation of 31.6 is an informative prior
that pulls the estimate to zero. StatsPAI checks this after sampling and
warns when the default prior is not vague for a coefficient. Set
`prior_var` or rescale the regressor.

The error variance has the prior `sigma2 ~ InvGamma(alpha0 / 2, delta0 /
2)`, given as `sigma2_prior=(alpha0, delta0)`. This is the convention of
the book and of `MCMCpack` (`c0`, `d0`).

Other models add one prior each.

| model | extra parameter | prior argument |
| --- | --- | --- |
| `negbin` | `alpha`, with variance `mu + alpha mu^2` | `size_prior=(shape, rate)` on `1 / alpha` |
| `quantile` | scale `sigma` of the asymmetric Laplace | `scale_prior=(n0, s0)`, or fix it with `scale=` |
| `oprobit` | cutpoints | `cut_prior_var` on the log distance between cutpoints |
| `t` | none, `dof` is fixed | |

## Things that differ from the R packages

**Quantile regression estimates the scale.** `MCMCpack::MCMCquantreg`
fixes the scale of the asymmetric Laplace likelihood at one. The posterior
spread then depends on the units of the outcome. Multiply `y` by ten and
the credible intervals do not grow by ten. `sp.bayes_regress` estimates the
scale by default (Kozumi and Kobayashi 2011). `scale=1.0` reproduces
`MCMCquantreg`.

**The ordered probit always has free cutpoints.** `bayesm::rordprobitGibbs`
fixes the first cutpoint at zero and expects a constant in the design. The
book's example passes a design without a constant, which forces the first
threshold to zero and moves every coefficient. `sp.bayes_regress` reports
the slopes and `J - 1` cutpoints, the parameterisation of `sp.oprobit` and
Stata, and cannot be given the restricted model by accident.

**The hierarchical logit and Poisson have no extra error term.**
`MCMChlogit` and `MCMChpoisson` add an observation-level normal error to
the linear index. `sp.bayes_mixed` fits the usual generalised linear mixed
model, the one `sp.melogit` and Stata's `melogit` fit by maximum
likelihood.

**The fixed effects of a hierarchical model have their marginal
spread.** On the public-capital panel `MCMChregress` reports a posterior
sd of 0.008 for the intercept. With 48 states and a state-effect variance
of 0.106 the intercept cannot be known better than
`sqrt(0.106 / 48) = 0.047`. `sp.bayes_mixed` reports 0.17 under the same
prior, and agrees with `lme4` under its own default prior.

**`sp.bma` finds every model inside Occam's window.** `bicreg` keeps the
best 150 models of each size and then applies the window. `sp.bma` runs an
exact branch and bound for the window itself. The two agree whenever
`bicreg`'s cap does not bind. `bicreg` also computes BIC from an R-squared
rounded to five decimals, so its probabilities agree with the exact ones
to about three digits.

**The multivariate Gelman-Rubin factor follows the paper.** `coda` puts the
number of parameters where Brooks and Gelman (1998) have the number of
chains. The univariate factors are identical.

## Comparing models

```python
small = sp.bayes_regress("y ~ x1", df, model="conjugate", prior_var=4)
large = sp.bayes_regress("y ~ x1 + x2", df, model="conjugate", prior_var=4)
sp.bayes_factor(small, large)
sp.savage_dickey(large, "x2")
```

Four estimators of the log marginal likelihood are available through
`fit.log_marginal_likelihood(method=)`.

| method | models | what it is |
| --- | --- | --- |
| `'exact'` | conjugate | closed form |
| `'chib'` | normal, probit | Chib (1995), from the Gibbs output |
| `'gelfand-dey'` | all | Gelfand and Dey (1994) with Geweke's truncated normal |
| `'laplace'` | all but quantile | Laplace approximation at the posterior mode |

A Bayes factor needs proper priors and depends on them. With a prior
variance of `c` on an extra coefficient, the Bayes factor in favour of the
smaller model grows like the square root of `c`. A "vague" default prior
therefore favours the smaller model by an arbitrary amount. Choose the
prior variance of the coefficients under test on purpose, and report how
the Bayes factor moves when it changes.

Under the conjugate prior the Savage-Dickey ratio compares with a
restricted model whose prior for `sigma2` has one more degree of freedom.
The docstring gives the exact statement.

## Model averaging

```python
out = sp.bma("y ~ x1 + x2 + x3 + x4 + x5 + x6", df)            # BIC
out = sp.bma("d ~ x1 + x2 + x3", df, family="binomial")        # logit
out = sp.bma("y ~ x1 + x2 + x3", df, method="gprior")          # g-prior
out = sp.bma("y ~ treat + x1 + x2 + x3", df, always=["treat"]) # controls only
out.table        # inclusion probabilities, averaged coefficients
out.models       # the models and their probabilities
```

`always=` keeps a term in every model, which is how to ask how an estimate
moves across sets of controls. A posterior inclusion probability is a
statement about prediction. It does not say that a control is a valid
adjustment variable.

## Panel data

```python
fit = sp.bayes_mixed("lgsp ~ lpcap + lpc + lemp + unemp", df, group="id")
fit = sp.bayes_mixed("y ~ x", df, group="id", random=["x"])        # random slope
fit = sp.bayes_mixed("d ~ x", df, group="id", family="logit")
fit.random_effects    # posterior mean and sd of every group's effects
fit.re_cov            # posterior mean of their covariance matrix
```

The random effects are assumed independent of the regressors. When that is
in doubt and the target is a within-unit effect, use `sp.panel(...,
method='fe')`.

The covariance of the random effects has the prior
`InvWishart(df, df * scale)`. The textbook and `MCMCpack` use an identity
scale, which says the random effects have a variance near one. For a log
outcome that is usually far too large. In the book's public-capital panel
(48 states, log gross state product) the prior `InvWishart(5, 5)` supplies
91 percent of the sum of squares behind the variance of the state
effects, and the posterior mean of that variance is 0.106 against a
restricted maximum likelihood estimate of 0.0076.

The default of `sp.bayes_mixed` is therefore different: `df = q + 1` and
`scale = 0.02 / df`. In the full conditional of the covariance matrix the
prior then adds 0.02 to the sum of squares of each random effect, which
is negligible on any ordinary scale. On the same panel the default gives
the `lme4` estimates and standard errors to two digits.
`re_prior=(q + 2, 1.0)` is the textbook prior.

Whatever the prior, `fit.model_info['re_prior_share']` is the share of
each variance component that comes from it, and a warning is raised above
25 percent. That happens with the default too when the effects are tiny
or the groups very few. Report how the variance components move when
`re_prior` changes.

## Instrumental variables

```python
fit = sp.bayes_ivreg("y ~ x + (d ~ z1 + z2)", df, seed=1)
fit.conf_int().loc["d"]          # the effect
fit.prob("rho > 0")              # is the regressor endogenous, and which way
fit.model_info["first_stage_F"]
```

`sp.bayes_ivreg` samples the joint normal model of the first stage and
the structural equation by Gibbs, with the formula syntax of `sp.ivreg`.
`rho` is the correlation of the two errors. Zero means the regressor is
exogenous, so its posterior is the Bayesian counterpart of a Hausman
test.

With a strong instrument the posterior of the effect is close to normal
around 2SLS with the 2SLS standard error. With a weak instrument it is
wide and skewed and depends on the priors, and the fit warns when the
first-stage F is below 10.

`sp.bayes_iv` fits the same model in PyMC, with half-normal priors on the
error scales. The two agree. `bayesm::rivGibbs` uses an identity scale for
the inverse-Wishart prior of the error covariance; the default here adds
0.02 to the sums of squared residuals, and `sigma_prior=(3, 1.0)` is the
`bayesm` prior.

## Time-varying coefficients

```python
fit = sp.dlm("y ~ x", df)                      # variances by maximum likelihood
fit = sp.dlm("y ~ x", df, method="gibbs", constant=["Intercept"], seed=1)
fit.smoothed       # coefficient paths given the whole sample, with intervals
fit.filtered       # given the data up to each date
fit.variances      # observation variance and one state variance per term
fit.forecast(4, new)
fit.plot()
```

Each coefficient follows a random walk. `"y ~ 1"` is the local level
model, and `constant=` holds a coefficient fixed. A state variance
estimated at zero says the data do not ask for that coefficient to move.
The filter, smoother and likelihood are those of the R package `dlm` and
agree with it to nine digits; `dlm::dlmLL` is the negative log likelihood
without the `2 pi` constant.

## Diagnostics for any chain

The diagnostic functions take any array or DataFrame of draws, including
draws produced elsewhere.

```python
sp.mcmc_summary(draws)     # mean, sd, naive and time-series SE, ESS, quantiles
sp.geweke_diag(draws)      # early mean against late mean
sp.heidel_diag(draws)      # stationarity, then precision of the mean
sp.raftery_diag(draws)     # how long a chain a quantile needs
sp.gelman_rubin([c1, c2])  # several chains, or one chain with split=True
sp.hpd_interval(draws)
```

They return the same numbers as `coda` to nine digits or more.

## What the evidence is

MCMC output cannot be compared digit for digit across programs. Two
correct samplers differ by Monte Carlo error. The checks are therefore of
three kinds.

Deterministic functions are compared with R on committed files. This
covers the diagnostics (`coda`), model averaging by BIC (`BMA`) and by
g-prior (`BMS`).

Every sampler is compared with the exact posterior of a small model. With
two coefficients and one auxiliary parameter the posterior can be
integrated on a grid without simulation. The posterior means must be
within four Monte Carlo standard errors of the exact ones, the standard
deviations within five percent, and the marginal likelihood estimators
must reproduce the exact normalising constant. The hierarchical models are
checked the same way, with the random effects integrated out analytically
or by Gauss-Hermite quadrature, and the IV sampler with the error
covariance integrated out in closed form.

The book's examples are rerun on its own data against long runs of
`MCMCpack` and `bayesm`. This is a screen, not a parity claim.
