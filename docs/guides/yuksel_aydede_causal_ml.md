# Yuksel and Aydede, *Causal Inference and Machine Learning*, in StatsPAI

Mutlu Yuksel and Yigit Aydede's book (*Causal Inference and Machine
Learning: In Economics, Social, and Health Sciences*,
<https://www.causalmlbook.com>) is written for applied researchers who
know regression and want a path to machine-learning estimators. The first
half builds prediction tools in R. The second half is causal: experiments,
regression adjustment, matching, weighting, double machine learning,
meta-learners, causal forests, difference-in-differences and synthetic
control. This guide maps each causal chapter to the StatsPAI call that
gives the same number as the R package the book uses, and says where the
two differ and why.

The book and its code are not redistributed with StatsPAI. The numbers
quoted here come from the book's data-generating processes; the parity
tests are in `tests/reference_parity/test_yuksel_aydede_causal_ml_parity.py`
and `tests/reference_parity/test_dml_did_doubleml_parity.py`.

## Chapter by chapter

| chapter | the book computes (R) | StatsPAI |
| --- | --- | --- |
| Randomized controlled trials | difference in means with Neyman's variance; `estimatr::lm_robust(se_type = "HC2")` | `sp.ttest(unequal=True)`; `sp.regress(robust='hc2')` |
| | regression with centred covariates and interactions | `sp.lm_lin` |
| | Fisher's randomization test | `sp.fisher_exact` |
| Selection on observables | regression adjustment, separate regressions by arm | `sp.regress`; `sp.lm_lin(vce='hc2')`; `sp.g_computation` |
| Heterogeneous treatment effects | interactions; effect in a subgroup; `margins` after a logit | `sp.regress` + `sp.lincom`; `sp.logit` + `sp.margins` |
| Matching | subclassification on quantiles | `sp.match(method='stratify', strata=)` |
| | `MatchIt` nearest neighbour, Mahalanobis, caliper | `sp.match(distance=, replace=False, m_order=, caliper=, caliper_scale='sd')` |
| | `MatchIt` optimal pairs on a supplied score | `sp.optimal_match(covariates=[score], metric='euclidean')` |
| | `MatchIt` full matching | `sp.match(method='full')` or `sp.full_match` |
| | balance tables and density plots | `sp.balance_table`, `sp.love_plot`, `sp.overlap_plot`; `fit.balance` |
| Inverse weighting and doubly robust estimation | Hajek and Horvitz-Thompson weights, trimming | `sp.ipw(normalize=, trim=, se_method='sandwich')`; `sp.ps_weights`; `sp.trimming` |
| | AIPW by hand, with and without cross-fitting | `sp.aipw(cross_fit=, trim=0)` |
| Penalized regression | `glmnet`, `cv.glmnet`: ridge, lasso, elastic net, adaptive lasso | `sp.glmnet(alpha=, penalty_factor=, foldid=)` |
| Double machine learning | partialling out with `cv.glmnet`, `hdm::rlasso`, trees, forests | `sp.dml(model='plr', ml_g=, ml_m=, fold_indices=)`; `sp.rlasso_effect` |
| Selection on unobservables and DML-IV | cross-fitted residuals, then 2SLS; first-stage F | `sp.dml(model='pliv', instrument=)` |
| Difference-in-differences and DML-DiD | Chang's (2020) score with cross-fitted learners | `sp.dml_did` |
| Meta-learners | S-, T- and X-learner, a different learner in each arm | `sp.metalearner(learner=, outcome_model=(control, treated))` |
| Causal trees and forests | `grf::causal_forest`, ATE and ATT, effects in a subset, variable importance, best linear projection, calibration test, RATE | `sp.causal_forest`; `cf.average_treatment_effect(subset=)`; `sp.variable_importance`; `sp.best_linear_projection`; `sp.test_calibration`; `sp.rate` |
| Synthetic control, synthetic DiD and RD | `Synth`, `gsynth`, `augsynth`, `synthdid`, `rdrobust` | `sp.synth`; `sp.gsynth(treat=)`; `sp.augsynth`; `sp.sdid`; `sp.rdrobust`, `sp.rdbwselect`, `sp.rdplot` |

The rest of the first half of the book (trees, random forests, boosting,
neural networks, classification metrics) is general machine learning.
StatsPAI takes any scikit-learn estimator as a nuisance learner and does
not reimplement them.

## Where the numbers agree, and to how many digits

On the book's designs, with the same data on both sides:

| step | reference | agreement |
| --- | --- | --- |
| Neyman SE, HC2, Lin estimator | `estimatr` | 1e-12 |
| subclassification | by hand in R | 1e-12 |
| nearest neighbour, Mahalanobis, caliper | `MatchIt` | 1e-12 |
| Hajek IPW; AIPW with `trim=0` | by hand in R | 1e-9 |
| PLR and PLIV with `rlasso`, same folds | `hdm` + the book's code | 1e-12 |
| DML-DiD, panel and repeated cross-sections | Python `DoubleML` | 1e-15 |
| T- and X-learner with per-arm learners | by hand in R | 1e-9 |
| subset average on a forest | `grf` (given its forest) | 1e-14 |
| generalized synthetic control, 0 to 3 factors, with covariates, staggered | `gsynth` | 1e-12 |
| elastic net path, coefficients, `lambda.min`, `lambda.1se` | `glmnet` | path 1e-10, coefficients 1e-6, same penalties selected |
| `rdrobust`, `rdbwselect` | `rdrobust` | 1e-9 |
| synthetic DiD, SC and DiD on Proposition 99 | `synthdid` | 1e-9 (point estimates) |
| augmented synthetic control, Kansas | `augsynth` | 1e-8 |

Causal forests cannot agree digit for digit, because the two engines draw
different random trees. On the book's simulated design the two give the
same ATE (−1.157 against −1.155 to −1.166 over three grf seeds), the same
five most important variables, the same calibration coefficients and the
same RATE within sampling error.

## What is different, and why

**`MatchIt`'s default is matching without replacement.** `sp.match`
matches with replacement and keeps ties, as Stata `teffects` does. To
reproduce `matchit(method = "nearest")`:

```python
sp.match(df, "earnings", "treat", covs, distance="propensity",
         replace=False, m_order="largest")
```

With `distance = "mahalanobis"`, `MatchIt` takes treated units in data
order: `m_order="data"`. Its caliper is in standard deviations of the
score: `caliper=0.1, caliper_scale="sd"`.

**Optimal matching is exact here.** `optmatch`, which `MatchIt` calls for
`method = "optimal"` and `method = "full"`, rounds distances to a
tolerance before solving. StatsPAI solves the assignment problem exactly,
so its total matched distance is never larger. On the book's data the full
matching totals are 2.6035 (StatsPAI) and 2.6733 (`MatchIt` default); with
`tol = 1e-9` passed to `matchit` the two totals agree to 1e-9. The matched
sets still differ in places, because the optimum is nearly flat, and the
estimates differ in the third digit. Given the same matched sets the
estimate and its standard error are identical.

**Full matching reports a standard error clustered on matched set.** It is
the one the `MatchIt` documentation prescribes: the weighted regression of
the outcome on the treatment with `sandwich::vcovCL(cluster = ~subclass)`.

```python
fit = sp.full_match(df, "earnings", "treat", covs, estimand="ATT")
fit.balance          # standardized differences before and after
fit.matched_data(df) # rows with subclass and weights
```

**AIPW clips propensity scores at 0.01 unless told not to.** The book's
formula uses the fitted scores as they are. `sp.aipw(..., trim=0)` does the
same and matches it. With the default `trim=0.01`, scores outside
[0.01, 0.99] are moved to the bound, the count is reported in
`model_info['n_propensity_clipped']`, and a warning says so. On the book's
design with 20,000 observations, 538 scores are clipped and the estimate
moves in the sixth digit; at 600 observations 14 are clipped and it moves
in the first. The book *drops* observations outside the bounds instead,
which changes the population the estimate refers to: do that explicitly
with `sp.trimming` before estimating.

**`glmnet` and `cv.glmnet` are `sp.glmnet`.** Same objective, same
standardisation, same penalty path, same cross-validation summaries. With
the same fold labels the two choose the same `lambda.min` and
`lambda.1se`.

```python
fit = sp.glmnet(df, "y", xs, alpha=0.5, foldid="fold")   # elastic net
fit.lambda_min, fit.lambda_1se
fit.coef("lambda.1se")
fit.predict(new, s="lambda.min")

# the adaptive lasso of the book: ridge first, then weights 1 / |b|
ridge = sp.glmnet(df, "y", xs, alpha=0, lambda_=0.1, cv=False)
ada = sp.glmnet(df, "y", xs, penalty_factor=1 / ridge.params.abs())
```

On five designs and three mixing weights the penalty path agrees to 1e-10
and stops at the same length, coefficients agree to 1e-6, and the
cross-validated error at every penalty to 1e-4. Four conventions that the
`glmnet` documentation does not spell out matter for that agreement, and
for reading `glmnet` output generally:

- A Gaussian outcome is scaled to unit variance before the penalty is
  applied. The lasso does not notice. The ridge does: in the units of the
  data its penalty is `lambda / sd(y)`, so `glmnet(alpha = 0)` is not the
  textbook ridge at the same `lambda`.
- The first point of a computed path is the fit at an infinite penalty.
  For the ridge that is the null model, whatever its label says.
- A computed path stops early, on a relative gain in deviance explained
  for a Gaussian outcome and an absolute one for a binomial.
- `cv.glmnet` does not refit the folds at the full-sample penalties. Each
  training set builds its own path, and the coefficients are interpolated
  linearly in `lambda`.

`sp.glmnet` solves each problem to a tighter tolerance than `glmnet`'s
default (`thresh = 1e-7`). A default R run is itself up to 6e-4 away from
R at `thresh = 1e-14` in the coefficients of the test designs, so expect
agreement with default R output to about three decimal places, and to six
once R is told to converge. With more predictors than rows and a penalty near
zero, `glmnet` at `thresh = 1e-14` is still 1e-3 from the minimiser; the
StatsPAI fit satisfies the subgradient conditions to 1e-7.

`sp.shrinkage` remains for comparing ridge, lasso and principal components
by cross-validated error, with the penalty stated on the residual sum of
squares: lasso `penalty = 2 n lambda sqrt((n - 1) / n)`, ridge
`penalty = n (lambda / sd_y) (n - 1) / n`.

**DML standard errors.** The book regresses the cross-fitted residuals
with `lm_robust(se_type = "HC3")`. `sp.dml` reports the standard error of
the orthogonal score, `sqrt(mean(psi^2) / J^2 / n)`, which is the HC0
standard error of the same regression without an intercept. The two differ
in the fourth digit at the book's sample size.

**HC2 and HC3 after 2SLS.** For these the leverage of an observation has
to be defined. `sp.iv` uses the leverage of the second-stage regression,
as `estimatr::iv_robust` and the `ivreg` package do. `AER::ivreg` with
`sandwich::vcovHC` uses the diagonal of `X (Xhat'X)^-1 Xhat'`, which can be
negative. On the book's DML-IV residuals the HC3 standard errors are
0.214736 and 0.214624. HC0 and HC1 are the same everywhere.

**DML-DiD.** The book codes Chang's score by hand, with the treated share
estimated on each training fold and observations with extreme propensity
scores dropped. `sp.dml_did` follows the `DoubleML` package: the share is
the full-sample one, scores are clipped rather than rows dropped, and the
weights are normalised by default. `in_sample_normalization=False` gives
Chang's score; the treated share then cancels out of the estimate.

```python
sp.dml_did(df, "dy", "treated", covs)                       # outcome change, one row per unit
sp.dml_did(long, "y", "treated", covs, time="year", id="id")  # two-period panel
sp.dml_did(rcs, "y", "treated", covs, time="post")           # repeated cross-sections
```

**Generalized synthetic control.** `sp.gsynth(treat='D')` takes a 0/1
treatment column, any number of treated units and staggered adoption, and
estimates covariate coefficients inside the interactive fixed effects
model, as Xu (2017) does. With the number of factors fixed it reproduces
`gsynth` to 1e-12. The number of factors is chosen by the leave-one-period-out
criterion of the paper; current versions of `gsynth` (which call `fect`)
use a more elaborate cross-validation, and the two can choose differently
on other data. On the package's example both choose 2. Standard errors
come from the parametric bootstrap of the paper and are random on both
sides.

```python
fit = sp.gsynth(df, "Y", "id", "time", treat="D", covariates=["X1", "X2"])
fit.detail                       # effect by period since adoption
fit.model_info["cv_table"]       # prediction error by number of factors
```

**Effects in a subset of a forest.** `average_treatment_effect(subset=)`
is grf's argument and gives grf's number. The book splits the sample at
the median of the forest's own out-of-bag predictions and compares the two
halves. That comparison describes the fit, but the halves were chosen by
the same data that estimate their effects, so the difference is biased
away from zero and its standard error does not account for the selection.
`sp.rate_split` and `sp.forest_policy_tree` do the comparison on held-out
units.

## Not covered

- Honest causal trees (`causalTree`). A single tree is superseded by the
  forest for estimation; for an interpretable rule use `sp.policy_tree`.
- Continuous-treatment DML-DiD (`causalweight::didcontDMLpanel`). See
  `sp.continuous_did` for a continuous dose without machine learners.
