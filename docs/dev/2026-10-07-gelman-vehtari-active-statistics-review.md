# Gelman and Vehtari, *Active Statistics* (2024), with *Regression and Other Stories* (2020): review

Source: `改进建议-收集整理/32-Gelman-Vehtari-ActiveStatistics/` (not in the
repository). It holds two checkouts. `ActiveStatistics/` is the web page of
the book of classroom stories, activities and problems; it carries no code
or data of its own and points to the second one. `ROS-Examples/` is the
code and data of *Regression and Other Stories* (Gelman, Hill and Vehtari),
93 R notebooks over some 70 example folders, which both books share.
So the audit ran on ROS-Examples and the syllabus of *Active Statistics*
was read as a list of what a student is expected to do with them.

Method. The notebooks fit almost everything with `rstanarm::stan_glm`
under its default weakly informative priors, summarise by posterior
simulation, and compare models with `loo`. Three kinds of check were made.

1. Thirty-three of the book's models were refitted by maximum likelihood in
   R 4.5.2 (`lm`, `glm`, `MASS::glm.nb`, `MASS::polr`, `AER::ivreg`) and in
   StatsPAI from the same formulas, written the way the book writes them.
   This is where the formula handling and the collinearity problems showed.
2. The Bayesian workflow was run in `rstanarm` 2.32.2 and `loo` 2.9.0. Its
   deterministic operators (PSIS given a log-likelihood matrix, the prior
   scales, Bayesian R-squared given draws) were compared digit for digit;
   posterior summaries, where two correct samplers differ by Monte Carlo
   error, were compared as a stochastic screen and each new sampler was
   checked against an exact posterior instead.
3. The causal chapters were run on the child care data against R `survey`
   and `Matching`.

`rstanarm`, `arm`, `loo` and `retrodesign` are GPL. They were installed in
a scratch library and run as black boxes; none of their source was read.
PSIS was written from Vehtari et al. (2024) and Zhang and Stephens (2009),
the default prior scales were pinned by running `prior_summary()` on
designs with binary, categorical and interaction columns, and the Bayesian
R-squared definitions by comparing candidate formulas with `bayes_R2()`.
The binned-residual rule is printed in the book's own notebook.

The book is from 2020. Where practice has moved since, the current state
was followed and the difference is recorded below under "What is dated".

## What was wrong in StatsPAI

1. **Perfectly collinear regressors.** The child care example of chapter
   20 writes the propensity score model with `black + hispanic + white`
   and `lths + hs + ltcoll + college` next to the constant. R drops
   `white` and `college`. `sp.logit` returned an arbitrary solution
   (intercept 1.76 where R has 6.35) with no warning. A sweep with three
   kinds of dependence (indicators summing to one, a duplicated column,
   an exact linear combination) over every likelihood-based entry point
   gave:

   | estimator | before |
   | --- | --- |
   | `sp.regress` | omits the dependent regressor with a note (already correct) |
   | `sp.logit`, `sp.cloglog` | arbitrary coefficients; SE 3.7e6, or "Hessian is singular; using pseudo-inverse", or nothing |
   | `sp.probit` | arbitrary coefficients, SE exactly 0, no warning |
   | `sp.glm` (binomial, Poisson) | coefficients of order 1e14, SE `nan`, "IRLS did not converge" |
   | `sp.poisson` | SE 1e6 or `nan` |
   | `sp.nbreg` | coefficients 6.7e13, no warning |
   | `sp.ologit`, `sp.oprobit` | SE exactly 0 or 8e6, no relevant warning |
   | `sp.mlogit` | all coefficients zero, "did not converge" |
   | `sp.tobit` | SE 7e4 or `nan`, no warning |
   | `sp.qreg` | SE 1e7, or `LinAlgError` |
   | `sp.svydesign(...).glm` | minimum-norm coefficients, SE `nan` for the intercept, treatment SE off by 17 percent |

   All of them now pass through `core/_collinear.py`: columns are scanned
   in the order the formula writes them (the constant first), a column
   with less than 1e-9 of its length left after projection on the kept
   ones is omitted, and the fit is announced as `note: white omitted
   because of collinearity (linear combination of 'Intercept', 'black',
   'hispanic')` and recorded in `model_info['omitted']`. On the child care
   data `sp.logit` now gives R's coefficients for the sixteen identified
   terms to 1e-6 and omits the same two, and the survey regression gives
   `svyglm`'s treatment coefficient and standard error to 1e-6. Prediction
   and `sp.margins` work on the reduced fit.

   The tolerance is not a rank tolerance on the whole design. `sp.regress`
   avoids one because the NIST designs are full rank with condition
   numbers near 1e15. Here each column is tested against the span of the
   kept ones after scaling to unit length, which leaves a regressor with a
   large mean and a small spread alone (year 2000 to 2009 next to a
   constant has 1.5e-3 of its length left).

2. **Separation went unannounced in `sp.logit` and `sp.glm`.** The
   detector asked for perfect classification *and* for 99 percent of the
   fitted probabilities to be within 0.01 of 0 or 1. Perfect
   classification by the sign of a linear index is complete separation by
   definition; the second condition only added a way to miss it. With 60
   observations and one point near the threshold `sp.logit` returned a
   slope of 1,412 silently and `sp.probit` warned. `sp.glm` had no check.
   The condition is now perfect classification, or fitted probabilities
   equal to 0 or 1 within ten machine epsilons (the event behind R's
   "fitted probabilities numerically 0 or 1 occurred"), in all three. The
   logistic function behind `sp.logit` also evaluated both branches of a
   `np.where` and overflowed with a `RuntimeWarning` on a large index; it
   is `scipy.special.expit` now.

3. **A logical outcome in the formula.** Chapter 15 writes the first part
   of the earnings model as `(earn > 0) ~ height + male`. The formula
   parser codes a true / false term as two columns. `sp.logit` and
   `sp.probit` raised `IndexError: boolean index did not match`,
   `sp.regress` "y has 3632 rows but X has 1816", `sp.glm` and
   `sp.bayes_regress` a broadcast error. `create_design_matrices` now
   keeps the `[True]` column as the outcome.

4. **`factor(vote) ~ value` in the ordered and multinomial models.** The
   spelling `MASS::polr` and `stan_polr` require. `sp.ologit` failed with
   "Ordered model requires J >= 3 categories, got 2" (the indicator
   expansion again) or `KeyError: ['C(vote)'] not in index`. A `factor()`,
   `as.factor()`, `ordered()` or `C()` around the outcome of a categorical
   model is now read as the bare column.

5. **`sp.bayes_regress(model='mlogit').predict(new_data)`** asked the new
   data for columns named `1:Intercept`, `1:x`, the stacked coefficient
   names. Found by the new test that evaluates every model's
   log-likelihood on new rows.

## What was missing

The book's workflow is fit, simulate, check, compare. StatsPAI had the
first step (`sp.bayes_regress`, eleven likelihoods, NumPy samplers) and
the marginal likelihood for comparison. The rest is new.

- **Priors on the scale of the data.** The book's first regression, vote
  share on growth with 16 elections, has an intercept near 46. Under the
  fixed default prior (variance 1000) the posterior mean was 46.12 with a
  warning that the prior is not vague; the least-squares and `stan_glm`
  answer is 46.3. `prior='weakly_informative'` is `rstanarm`'s default:
  `N(0, 2.5 sd(y) / sd(x))` on slopes, `N(mean(y), 2.5 sd(y))` on the
  intercept with centred regressors, `sigma ~ Exponential(1 / sd(y))`;
  `sd(y) = 1`, `mean(y) = 0` outside the Gaussian model; an
  Exponential(1) on the size of the negative binomial. The centred
  intercept prior is carried to the raw coefficients as a full prior
  covariance, so no sampler changed. The exponential prior on `sigma` is
  not conjugate: its full conditional is the flat-prior inverse gamma
  times `exp(-rate * sigma)`, drawn by proposing from the inverse gamma
  and accepting with probability `exp(-rate (sigma' - sigma))`, an exact
  independence Metropolis step (acceptance 0.92 on the election data).
  The fixed default was not changed (it would move existing results
  silently); its warning now points to the new option.
- **`offset=` / `exposure=`** for the logit, Poisson and negative
  binomial models (the roaches example).
- **Predictive draws and pointwise likelihood.** `posterior_linpred`,
  `posterior_epred`, `posterior_predict` and `log_lik` on every fit, for
  all eleven likelihoods and for `sp.bayes_shrink`.
- **`sp.loo`, `sp.waic`, `sp.kfold`, `sp.loo_compare`, `sp.loo_predict`,
  `sp.psis`, `sp.kfold_split`.**
- **`sp.ppc`, `sp.bayes_r2`, `sp.loo_r2`, `sp.mad_sd`.**
- **`sp.bayes_shrink(prior='horseshoe')`**, for the regression on many
  predictors of chapter 12. Makalic and Schmidt's Gibbs sampler, with
  `global_scale=` or Piironen and Vehtari's `p0 / (p - p0) / sqrt(n)`.
  It reports the mean shrinkage factor of each coefficient and the
  effective number of unshrunk ones. The book uses the regularized
  horseshoe, which is not conjugate and is not implemented.
- **`sp.retrodesign`**: type S and type M errors.
- **`sp.binned_residuals`, `sp.binned_residuals_plot`, `sp.standardize`,
  `sp.invlogit`**, the `arm` helpers the notebooks call most.
- **`sp.poststratify`.**
- **In `sp.glm`**: the robit link, `quasipoisson` / `quasibinomial`,
  `cbind(successes, failures) ~ x`.
- **`y ~ .`** in formulas (`expand_dot` in `core/utils.py`).

## Documented differences (not bugs)

- **Order of coefficients.** The formula parser puts categorical terms
  before numeric ones; R keeps the written order. Same numbers.
- **Observed and expected information.** `sp.glm` and `sp.probit` use the
  observed information, like Stata; R `glm` the expected. They coincide
  for canonical links. On the wells data the standard errors differ by
  2.3 percent (probit) and 8.7 percent (cloglog);
  `information='expected'` gives R's to 2e-5.
- **Negative binomial.** `sp.nbreg` inverts the joint observed
  information, the convention of Stata and statsmodels (statsmodels
  gives the same standard errors to four digits); `MASS::glm.nb`
  holds theta fixed and uses expected information. On the roaches data
  that is 0.2475 against 0.159 for the coefficient on `roach1`, a 56
  percent gap on a very overdispersed outcome. The posterior standard
  deviation under the weakly informative prior is 0.245.
- **`Matching::Match` on the child care propensity score** gives an ATT
  of 11.87; `sp.match` gives 11.37, which is what a brute-force
  nearest-neighbour search on the same score returns. `Match` treats
  distances within `distance.tolerance = 1e-5` as ties.
- **Monte Carlo standard error of `elpd`.** Derived here by the delta
  method on the self-normalised importance sampling estimate. `loo`
  reports a different approximation. They agree to within 0.8 percent on
  usable weights and both report it unavailable when a Pareto shape is
  over the threshold.
- **`loo_R2`.** `rstanarm` returns Bayesian bootstrap draws; their mean
  moves with the seed. `sp.loo_r2(...).estimate` is the plug-in value
  `1 - var(y - yhat_loo) / var(y)` and the draws are kept for the
  interval.
- **`arm::standardize` and transformed terms.** For `log(cnt + 1)` it
  rescales `cnt` inside the logarithm (`log(z.cnt + 1)`), which is a
  different model. `sp.standardize` does not touch transformed terms.
- **`polr` on one respondent of the storable-votes data** (20
  observations) fails with "attempt to find suitable starting values
  failed"; `sp.ologit` converges. The book fits it with `stan_polr` and a
  prior.

## R has the same hole at a tight tolerance

`glm.fit` decides which columns are aliased with a QR tolerance of
`min(1e-7, epsilon / 1000)`. The answer key for the child care model was
first written with `glm.control(epsilon = 1e-12)`, as the other models
are. At that setting the tolerance is 1e-15, the redundant indicators are
no longer detected, and R returns eighteen coefficients with "algorithm
did not converge". The reference script therefore fits that one model at
`epsilon = 1e-10`. The scan in `core/_collinear.py` does not depend on the
convergence tolerance of the fit.

## A reference that is not self-consistent

`retrodesign` 0.2.2, given degrees of freedom, computes power and the
type S rate from the noncentral t distribution and the exaggeration ratio
by simulating `effect + se * t`. The function printed in Gelman and Carlin
(2014) uses the shifted central t for all three. Neither is the
exaggeration ratio of a t-test, in which the estimate is normal and the
*standard error* is estimated. For effect 0.5, se 1, 20 degrees of
freedom:

| | power | type S | type M |
| --- | --- | --- | --- |
| Gelman and Carlin's function (shifted t) | 0.0730 | 0.1208 | 5.18 |
| `retrodesign` 0.2.2 | 0.0764 | 0.0970 | 5.01 (simulated) |
| exact for the t-test, `sp.retrodesign(dof=20)` | 0.0764 | 0.0970 | 4.63 |
| four million simulated t-tests | 0.0765 | 0.0970 | 4.63 |

`sp.retrodesign` integrates the known-variance expression over the
distribution of the estimated standard error (one-dimensional quadrature),
which is exact, and offers `method='shifted'` for the published formula.
With a known standard error all sources agree and the closed form matches
the package to 1e-15. This is a T4 row (reference disagreement) with
independent evidence, the simulation of the test itself
(`test_exact_exaggeration_ratio_is_what_a_simulated_t_test_gives`).

## What is dated in the book, and what was done instead

- **`posterior_linpred(transform = TRUE)`** was removed from `rstanarm`
  in favour of `posterior_epred`. The methods here carry the current
  names.
- **Pareto `k` threshold.** The book and Vehtari, Gelman and Gabry (2017)
  use 0.7. Since Vehtari et al. (2024) and `loo` 2.7 it is
  `min(1 - 1 / log10(S), 0.7)`: 0.62 for 400 draws, 0.7 from 2,200.
  `sp.loo` uses the current rule and reports the threshold.
- **Interpreting `elpd_diff`.** Current guidance adds that a difference
  under about 4 is small regardless of its standard error and that the
  normal approximation needs about 100 observations. Both are in the
  docstring of `sp.loo_compare`.
- **The horseshoe.** `hs()` in the book's student-grades example is the
  regularized horseshoe; see above.
- **`stan_glm(algorithm = 'optimizing')`** for speed. Not needed: the
  NumPy samplers fit the kid-IQ model with four chains in 0.15 seconds.

## Evidence

| claim | level | where |
| --- | --- | --- |
| PSIS-LOO, WAIC, comparison given a log-likelihood matrix = `loo` 2.9.0 (1e-9; three matrices incl. heavy tails and `r_eff`) | T2 | `tests/reference_parity/test_regression_stories_r_parity.py` |
| `sp.loo`, `sp.kfold` = exact leave-one-out of the conjugate normal model (pointwise, 0.02) | T1 | `test_bayes_workflow_exact_posterior.py` |
| weakly informative prior scales = `rstanarm::prior_summary` (1e-9) | T2 | parity file |
| samplers under the weak prior (normal, logit, probit, Poisson with exposure, negative binomial) = grid posterior | T1 | exact-posterior file |
| horseshoe Gibbs = exact posterior with one regressor, two global scales | T1 | exact-posterior file |
| Bayesian R-squared operator = `rstanarm::bayes_R2` on its own draws (1e-9) | T2 | parity file |
| pointwise log-likelihood of each model sums to the sampler's own likelihood; predictive draws have the model's moments | T1 | exact-posterior file |
| robit link, quasi families, grouped binomial, logical outcome, aliased regressors = R `glm` (1e-6) | T2 | parity file |
| binned residuals, rescaling, standardized refit = `arm` 1.15-3 (1e-8 or better) | T2 | parity file |
| `sp.retrodesign`, known variance = `retrodesign` 0.2.2 closed form (1e-9) | T2 | parity file |
| `sp.retrodesign(dof=)` exact for the t-test | T4 with simulation evidence | parity file |
| posterior medians and spreads on the book's data vs `rstanarm` long chains (0.1 sd, 10 percent) | S | `tests/external_parity/test_gelman_ros_examples.py` |
| omission of dependent regressors: each of twelve estimators returns the reduced fit | T1 | `tests/test_regression_stories_workflow.py` |

The reference fixture is `tests/reference_parity/_fixtures/
regression_stories_R.json`, written by
`_generate_regression_stories_R.R` on synthetic data it generates and
stores. The log-likelihood matrices follow a closed-form recipe with no
random numbers and are rebuilt, not stored, on the Python side (a
checksum guards the recipe).

One lesson from writing the horseshoe test. The first version integrated
the coefficient on a grid and disagreed with the sampler by eight Monte
Carlo standard errors at a small global scale. The sampler was right. A
small local scale puts a spike at zero that no grid in the coefficient
resolves. The coefficient is now integrated out analytically and the grid
runs over the variance and the scale only.

## Second round (same day; Bryce: "you decide, then complete the work left")

The first round ended with seven open items. Six are closed; the decision
that was Bryce's to make was delegated and is recorded first.

1. **The default prior of `sp.bayes_regress` will change, by the
   deprecation route.** Decision: yes. A prior of variance 1000 is not
   vague for a coefficient measured in large units, the book's first
   example shows it, and a default that needs a warning to be safe is the
   wrong default. But every stored result would move, so nothing changes
   in this release: when `prior` is omitted, the model has a weakly
   informative prior and neither `prior_mean` nor `prior_var` is given, a
   `DeprecationWarning` announces that 1.40 switches to
   `prior='weakly_informative'`. `prior='vague'` keeps the present
   numbers for good. The flip in 1.40 is one line (`prior_kind` when
   `prior is None`) plus MIGRATION and the test
   `test_default_prior_change_is_announced_only_where_it_applies`.
2. **Mixing of the Metropolis samplers.** The logit, Poisson, negative
   binomial and multinomial logit models used one random-walk proposal
   and returned an effective sample size near a tenth of the draws. Each
   iteration now picks, with probability 0.8, an independence proposal
   instead: multivariate Student-t with 5 degrees of freedom, centred at
   the posterior mode, scale 1.15 times the Laplace one
   (`_core.metropolis_mixture`). Both kernels leave the posterior
   invariant, so the mixture does; the heavy tails are what an
   independence sampler needs, and the random walk keeps the chain moving
   where the normal approximation is poor. Effective sample size per
   4,000 draws, before and after: wells logit 370 to 1,830, roaches
   negative binomial 440 to 1,260.
   The exact-posterior tests of all four models pass unchanged. The
   draws for a given seed are different numbers from before; they
   estimate the same posterior. `acceptance_rate` remains that of the
   random-walk moves (the tunable one); the other is in
   `_extras['independence_accept']`.
3. **`model='ologit'`**, the model of `stan_polr` and `sp.ologit`, with
   the reporting of the ordered probit (slopes and cutpoints, no
   intercept). No data augmentation: the whole vector is updated by the
   same mixed Metropolis kernel. Checked against a three-dimensional grid
   posterior, and on the pooled storable-votes data the posterior under
   the diffuse prior sits on the maximum-likelihood fit.
4. **Grouped binomial outcomes**: `trials=` or an outcome written
   `cbind(successes, failures)`, as the book's golf example does. The
   pointwise log-likelihood is the binomial log mass, predictions are
   counts, `posterior_epred` the success probability. Checked against a
   grid posterior. Bayesian R-squared is refused for grouped fits.
5. **The regularized horseshoe** (`slab_scale=`, `slab_df=`), the prior
   behind `rstanarm::hs()`. The slab breaks the conjugacy of Makalic and
   Schmidt's sampler, so the local scales, the global scale and the slab
   are updated by Metropolis on the log scale with steps tuned during
   burn-in. Two things were needed for the global scale to mix: a joint
   move that raises `tau` and lowers every `lam_j` by the same factor
   (the likelihood of the coefficients does not change along it, so only
   the priors decide), and five passes over the scales per draw of the
   coefficients. Effective sample size of `tau` on the 25-regressor
   example: 37 with one pass and no joint move, 106 with the joint move,
   490 with both. Checked against the exact posterior with one regressor
   at two global scales; that grid has to run far out in the scale,
   because under a slab a large scale costs no likelihood and the
   posterior keeps the half-Cauchy tail of the prior.
6. **`sp.binned_residuals(band='model')`**: the standard error the
   fitted binary model implies, which does not collapse where the outcome
   hardly varies. The default stays the `arm` band.
7. **`sp.nbreg` counts the fixed-effect indicators it estimates** after
   one is omitted for collinearity.

## Third round (same day, "ok continue"): the horseshoe for a binary outcome

`sp.bayes_shrink(prior='horseshoe', family='logit')`, plain or with a
slab. The coefficients are drawn by Polya-Gamma data augmentation (Polson,
Scott and Windle 2013): given `omega_i ~ PG(1, x_i' beta)` their full
conditional is normal with precision `X' Omega X` plus the prior
precisions, so the scale updates of the Gaussian sampler carry over with
the residual variance set to one. `mcmc/_polyagamma.py` is the exact
accept-reject sampler of that paper, written from the paper and vectorised
(400,000 draws in a quarter of a second); `p0` is calibrated with a latent
standard deviation of 2, as Piironen and Vehtari propose for the logit.

Evidence. The Polya-Gamma draws have the exact mean and variance at six
arguments from 0 to 40 and pass a two-sample test against the
sum-of-exponentials representation of the distribution. The sampler for
the model is checked against the exact posterior with one regressor at two
global scales. Two parameterisations of that reference failed before one
worked: a grid over (slope, scale) cannot resolve the spike at zero for a
small scale, and a grid over (slope / scale, scale) cannot resolve the
ridge of width 1 / scale for a large one. The scale has to be integrated
out of the prior by quadrature, which leaves a two-dimensional posterior
under the marginal horseshoe density, whose logarithmic pole at zero is
handled by a cubic grid. The regularized version is checked in the limit
of a very wide slab, where it must equal the plain one; its scale step is
the code already verified in the Gaussian case.

On a simulated sparse logit (300 observations, 30 regressors, 3 signals)
the root mean squared error of the coefficients is 0.11 against 0.23 for
maximum likelihood, and with 40 observations and 30 regressors, where the
outcome is separated and maximum likelihood diverges, the regularized
horseshoe returns a finite posterior that keeps the one real signal.

## Closing checks

The three lecture decks of *Active Statistics* (first semester, second
semester, one-semester version) were read for methods beyond the ROS
examples. They add none: linear, logistic, Poisson and overdispersed
Poisson regression, adjustment of a sample to a population, average
treatment effects from regressions with interactions, noncompliance and
missing data, all covered above. The decks' "regression with 21 data
points and 16 predictors" is the case the horseshoe is for.

The full test suite was run on main after the third round: 32,362
passed, and the one failure, a process-isolation timing test, passes on
its own (the machine was under heavy load from other sessions).

## Open items

- Shrinkage priors for count outcomes.
- The slab of the regularized horseshoe is in units of the residual
  standard deviation, which keeps the coefficient update conjugate.
  `rstanarm` scales it by `sd(y)`. The two differ by the factor
  `sigma / sd(y)`; no attempt was made to match `rstanarm` draws.
- User-written Stan models (golf putting geometry, the 2 x 2 restaurant
  example) have no counterpart outside PyMC. This is scope, not a gap to
  close.
- Flip the default prior in 1.40 (see 1 above).

## Rerun

```bash
# committed evidence (R with loo, rstanarm, arm, retrodesign, jsonlite)
Rscript tests/reference_parity/_fixtures/_generate_regression_stories_R.R
pytest tests/reference_parity/test_regression_stories_r_parity.py \
       tests/reference_parity/test_bayes_workflow_exact_posterior.py \
       tests/test_regression_stories_workflow.py

# the book's data
export STATSPAI_ROS_DIR=.../32-Gelman-Vehtari-ActiveStatistics/ROS-Examples
Rscript tests/external_parity/gelman_ros_reference.R      # about ten minutes
pytest tests/external_parity/test_gelman_ros_examples.py
```
