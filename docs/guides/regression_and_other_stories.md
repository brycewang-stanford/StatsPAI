# Gelman, Hill and Vehtari, *Regression and Other Stories*, in StatsPAI

*Regression and Other Stories* (Cambridge University Press, 2020) teaches
regression as a workflow. Fit a model, simulate from it, check the
simulations against the data, compare models by how well they predict,
and only then read the coefficients. *Active Statistics* (Gelman and
Vehtari, 2024) is its companion of classroom stories, activities and
problems. The two books share one set of examples, the
[ROS-Examples](https://github.com/avehtari/ROS-Examples) repository,
written in R around `rstanarm::stan_glm`, `loo` and `arm`.

This guide maps that workflow to StatsPAI, says where StatsPAI follows a
different convention and why, and lists what has changed in practice
since the book was printed.

The book's data are not redistributed with StatsPAI. The examples below
read them from a local copy of ROS-Examples.

## The workflow in six calls

```python
import pandas as pd
import statspai as sp

kidiq = pd.read_csv("ROS-Examples/KidIQ/data/kidiq.csv")

fit = sp.bayes_regress("kid_score ~ mom_hs + mom_iq", kidiq,
                       prior="weakly_informative", chains=4,
                       draws=2000, burnin=1000, seed=1)
fit.summary()                      # posterior mean, sd, interval, ESS

sp.ppc(fit, stat="min", seed=1)    # can the model produce data like these?
sp.loo(fit)                        # expected log predictive density
sp.bayes_r2(fit); sp.loo_r2(fit)   # variance explained, in and out of sample

small = sp.bayes_regress("kid_score ~ mom_hs", kidiq,
                         prior="weakly_informative", chains=4,
                         draws=2000, burnin=1000, seed=1)
sp.loo_compare({"hs + iq": fit, "hs": small})
```

`sp.bayes_regress` is NumPy only. The fit above takes a fraction of a
second, against several seconds of compilation and sampling in Stan,
which matters when the activity is "fit it twenty times with different
fake data".

## Chapter by chapter

| chapters | the book runs | StatsPAI |
| --- | --- | --- |
| 4, 5 | simulation of sampling distributions, `quantile`, `mad` | NumPy; `sp.mad_sd` |
| 6 to 8 | `stan_glm(y ~ x)`, `lm` | `sp.bayes_regress(prior='weakly_informative')`; `sp.regress` |
| 9 | `posterior_linpred`, `posterior_epred`, `posterior_predict` | methods of the same names on the fit |
| 9 | informative priors, `prior = normal(...)` | `prior_mean=`, `prior_var=` |
| 10 | indicators, interactions, `factor(x)`, `y ~ .` | the same formulas; `factor()` and `.` are understood |
| 11 | residual plots, `pp_check`, `ppc_stat` | `sp.ppc(fit, stat=...)`, `.plot(kind='density')` |
| 11 | `bayes_R2`, `loo_R2` | `sp.bayes_r2`, `sp.loo_r2` |
| 11 | `loo`, `loo_compare`, `kfold` | `sp.loo`, `sp.loo_compare`, `sp.kfold` |
| 12 | logs, `arm::standardize`, rescaling by two standard deviations | formulas with `log()`; `sp.standardize` |
| 12 | horseshoe prior `hs()` on many predictors | `sp.bayes_shrink(prior='horseshoe', p0=, slab_scale=)`; `family='logit'` for a binary outcome |
| 13, 14 | `stan_glm(family = binomial)`, `invlogit` | `sp.bayes_regress(model='logit')`; `sp.logit`; `sp.invlogit` |
| 14 | average predictive comparisons | `sp.margins`, `sp.margins_at` |
| 14 | `binnedplot` | `sp.binned_residuals`, `sp.binned_residuals_plot` |
| 14 | separation and weakly informative priors | `ConvergenceWarning` from `sp.logit`; `prior='weakly_informative'` |
| 15 | Poisson and negative binomial with `offset`, overdispersion | `sp.bayes_regress(model='poisson' / 'negbin', exposure=)`; `sp.poisson`, `sp.nbreg`; `sp.glm(family='quasipoisson')` |
| 15 | `cbind(y, n - y) ~ x` | the same formula in `sp.glm(family='binomial')` and `sp.bayes_regress(model='logit')` |
| 15 | probit, ordered logit and probit (`stan_polr`), robit | `sp.probit`, `sp.ologit`, `sp.oprobit`, `sp.bayes_regress(model='ologit' / 'oprobit')`; `sp.glm(link='robit(4)')` |
| 15 | tobit, mixed discrete and continuous outcomes | `sp.tobit`, `sp.bayes_regress(model='tobit')`; two fits and `posterior_predict` |
| 16 | sample size, power, type S and type M errors | `sp.power_ttest`, `sp.power_two_proportions`; `sp.retrodesign` |
| 17 | poststratification | `sp.poststratify(fit, cells, count='N')` |
| 17 | missing data imputation | `sp.mice`, `sp.mi_estimate` |
| 18, 19 | randomized experiments, regression adjustment | `sp.regress`, `sp.lm_lin` |
| 20 | propensity scores, matching, weighting, balance | `sp.pscore`, `sp.match`, `sp.ipw`, `sp.balance_table`, `sp.svydesign(...).glm` |
| 21 | instrumental variables, regression discontinuity, fixed effects | `sp.ivreg`, `sp.rdrobust`, `sp.panel` |
| 22 | multilevel models (pointer to the next book) | `sp.mixed`, `sp.bayes_mixed` |

## Priors on the scale of the data

The book's first regression predicts the incumbent party's vote share
from economic growth, sixteen elections.

```python
hibbs = pd.read_csv("ROS-Examples/ElectionsEconomy/data/hibbs.dat", sep=r"\s+")
fit = sp.bayes_regress("vote ~ growth", hibbs, prior="weakly_informative",
                       chains=4, draws=5000, seed=1)
fit.draws.median()      # Intercept 46.3, growth 3.0
sp.mad_sd(fit.draws)    # 1.7 and 0.7, the numbers printed in the book
```

`prior="weakly_informative"` reproduces the default of `rstanarm`. Each
slope gets a normal prior with standard deviation `2.5 sd(y) / sd(x)`.
The intercept gets `N(mean(y), 2.5 sd(y))` *with the regressors
centred*, which is a correlated prior on the raw coefficients. The
residual standard deviation gets an exponential prior with rate
`1 / sd(y)`. For logit, probit, Poisson and negative binomial models
`sd(y)` is replaced by 1 and `mean(y)` by 0.

Under the fixed prior of `sp.bayes_regress` (variance 1000 on every
coefficient) the same fit puts the intercept at 46.1 and says so in a
warning. A prior of fixed scale is tight for a coefficient measured in
large units. The weakly informative prior does not depend on units.

The fixed prior is still what an unnamed `prior` gives, for one more
release. From StatsPAI 1.40 the weakly informative prior becomes the
default for the models that have one, and a `DeprecationWarning` says so
until then. Write `prior='vague'` to keep the old numbers or
`prior='weakly_informative'` to move now. A fit that sets `prior_mean` or
`prior_var` is not affected.

The prior scales are checked against `prior_summary()` of `rstanarm`
to nine digits. The sampler is checked against the exact posterior on a
grid.

## Predictive accuracy

`sp.loo` implements Pareto smoothed importance sampling as in Vehtari,
Simpson, Gelman, Yao and Gabry (2024). Given the same matrix of
pointwise log-likelihoods it returns the pointwise `elpd`, `p`, Pareto
`k` and effective sample sizes of R `loo` 2.9.0 to nine digits or
better. Three things differ from the book's text, all because the
method moved on after 2020.

- The threshold on Pareto `k` depends on the number of draws,
  `min(1 - 1 / log10(S), 0.7)`. It is 0.7 from about 2,200 draws on and
  lower for shorter chains. The book's fixed 0.7 is the long-chain limit.
- A difference in `elpd` below about 4 is small whatever its standard
  error. With fewer than about 100 observations the normal
  approximation behind `se_diff` is itself unreliable.
- `posterior_linpred(transform = TRUE)` is gone from `rstanarm`. The
  expected outcome is `posterior_epred`, and that is the name here.

`sp.loo` works on any draws-by-observations matrix, so it also serves a
PyMC trace or draws produced elsewhere. `sp.kfold` refits the model and
is the answer when Pareto `k` values are high or whole groups must be
held out together (`sp.kfold_split(n, k, groups=...)`).

On the conjugate normal model, where leave-one-out has a closed form,
`sp.loo` and `sp.kfold` both reproduce the exact pointwise values.

## What StatsPAI does differently, on purpose

**Collinear regressors.** The child care example of chapter 20 puts all
the race indicators and all the education indicators in the propensity
score model next to the constant. R drops one of each set and reports
`NA`. StatsPAI omits the later member of each dependent set, like Stata,
and names it in a note and in `model_info['omitted']`. Until this pass
that was true of `sp.regress` only. The likelihood-based estimators
returned coefficients of order 1e13 or standard errors of zero, often
silently. `sp.logit`, `sp.probit`, `sp.cloglog`, `sp.glm`, `sp.poisson`,
`sp.nbreg`, `sp.ologit`, `sp.oprobit`, `sp.mlogit`, `sp.tobit`,
`sp.qreg` and survey regression now share one omission step.

**Separation.** `sp.logit` and `sp.glm` warn when the linear index
separates the outcome. The earlier detector also required 99 percent of
fitted probabilities to be within 0.01 of the boundary and stayed silent
on small samples while the slope ran past 1,000.

**Observed and expected information.** For a link that is not canonical
(probit, cloglog, robit) `sp.glm` and `sp.probit` report standard errors
from the observed information, as Stata does. R `glm` uses the expected
information. On the wells data the two differ by 2 percent for probit
and 9 percent for cloglog. `sp.glm(..., information='expected')` gives R's
numbers.

**Negative binomial standard errors.** `sp.nbreg` inverts the joint
information of the coefficients and the dispersion (Stata, statsmodels).
`MASS::glm.nb` holds the dispersion fixed and uses expected information.
On the roaches data, which are very overdispersed, R's standard error
for `roach1` is 0.16 and the joint one is 0.25. The Bayesian fit gives
0.25.

**Design analysis with estimated standard errors.**
`sp.retrodesign(effect, se, dof=)` treats the test as a t-test, so
power and the sign error rate are noncentral-t probabilities and the
exaggeration ratio integrates over the estimated standard error. The
function printed in Gelman and Carlin (2014) shifts a central t instead.
With effect 0.5, standard error 1 and 20 degrees of freedom the
exaggeration ratio is 4.63 exactly and 5.18 by the shifted
approximation. A simulated t-test agrees with the first. Pass
`method='shifted'` for the published formula. With a known standard
error (no `dof`) there is one answer and it matches the R package
`retrodesign` to machine precision.

**Standardizing transformed terms.** `arm::standardize` rescales `x`
inside `log(x)`, which changes the model. `sp.standardize(df,
formula=...)` rescales the columns named in the formula and leaves
transformed terms alone. Create the transformed column first and
standardize that.

**Binned residuals where the outcome hardly varies.** The default band is
`2 sd / sqrt(n)` of the residuals in a bin, as in `arm`. In a bin where
every outcome is 1 the residuals are all tiny and nearly equal, so the
band collapses and the bin is flagged. `band='model'` uses the standard
error the fitted probabilities imply, `2 sqrt(sum p (1 - p)) / n`, which
does not collapse.

**Bayesian R-squared for counts.** `rstanarm::bayes_R2` stops at
Gaussian and binomial models. `sp.bayes_r2` applies the same definition,
variance of the fit over variance of the fit plus the residual variance
the model expects, to Poisson (`mu`) and negative binomial
(`mu + alpha mu^2`) fits.

## What the book leaves to Stan and is not here

- User-written Stan programs, such as the geometry-based golf putting
  model. `sp.nls` fits nonlinear least squares and `sp.abc` simulation
  models, but there is no general probabilistic programming layer in the
  NumPy samplers. The PyMC-based estimators are in `statspai.bayes`.
- `stan_gamm4` (splines inside a Bayesian fit). `sp.gam` is the
  penalized-likelihood counterpart.
- The prior of `stan_polr`, which is stated on the R-squared of the
  latent regression. `sp.bayes_regress(model='ologit')` puts a normal
  prior on the coefficients and on the log distances between cutpoints.
- NUTS. The logit, ordered logit, Poisson and negative binomial models
  are sampled by Metropolis with two proposals, a random walk and an
  independence proposal centred at the posterior mode. On the book's
  examples the effective sample size is a third to a half of the draws.
  A posterior far from normal (very few observations, near separation)
  gets less; the acceptance rate of the independence proposal is in
  `fit._extras['independence_accept']`.
- Shrinkage priors for count outcomes. `sp.bayes_shrink` covers Gaussian
  and binary (`family='logit'`) outcomes.

## Reproducing the checks

```bash
# reference values on a committed synthetic file (needs R with loo,
# rstanarm, arm, retrodesign)
Rscript tests/reference_parity/_fixtures/_generate_regression_stories_R.R
pytest tests/reference_parity/test_regression_stories_r_parity.py
pytest tests/reference_parity/test_bayes_workflow_exact_posterior.py

# the book's own data
export STATSPAI_ROS_DIR=/path/to/ROS-Examples
Rscript tests/external_parity/gelman_ros_reference.R
pytest tests/external_parity/test_gelman_ros_examples.py
```

## References

Gelman, A., Hill, J. and Vehtari, A. (2020). *Regression and Other
Stories*. Cambridge University Press. `gelman2020regression`

Gelman, A. and Vehtari, A. (2024). *Active Statistics: Stories, Games,
Problems, and Hands-on Demonstrations for Applied Regression and Causal
Inference*. Cambridge University Press. `gelman2024active`

The methods are cited by key in the docstrings: `vehtari2017practical`,
`vehtari2024pareto`, `watanabe2010asymptotic`, `zhang2009new`,
`gelman2019rsquared`, `gabry2019visualization`, `gelman2008weakly`,
`gelman2008scaling`, `gelman2014beyond`, `gelman2006data`,
`liu2004robit`, `carvalho2010horseshoe`, `piironen2017sparsity`,
`makalic2016simple`. `sp.bibtex(keys=[...])` returns the entries.
