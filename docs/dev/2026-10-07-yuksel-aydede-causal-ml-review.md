# Yuksel and Aydede, *Causal Inference and Machine Learning*: what the book needs and what StatsPAI had

*2026-10-07. Worktree `wt/yuksel-aydede`.*

## What was done

The material is the bookdown site of Mutlu Yuksel and Yigit Aydede,
*Causal Inference and Machine Learning: In Economics, Social, and Health
Sciences* (<https://www.causalmlbook.com>): 36 HTML chapters and the R code
of 27 of them, about 8,700 lines. Twelve chapters are causal. For each of
them the book's code was run in R 4.5 with the packages it names
(`estimatr` 2.0.0, `MatchIt` 4.7.2 with `optmatch` 0.10.8, `hdm` 0.3.2,
`glmnet` 4.1.10, `grf` 2.6.1, `gsynth` 1.4.0, `augsynth`, `synthdid`,
`Synth`, `rdrobust` 4.0.0), the simulated data written to CSV, and the same
analysis done with StatsPAI on those bytes. The DML-DiD chapter was
compared with Python `DoubleML` 0.11.3, which implements the estimator the
book codes by hand.

The other fifteen chapters teach prediction: bias and variance,
cross-validation, penalised regression, trees, forests, boosting, neural
networks, classification metrics, time-series forecasting with forests.
Those are scikit-learn's job. They were read for what a causal user would
want from them, and one thing came out of that: how to state a `glmnet`
penalty in `sp.shrinkage`'s terms.

The note that the material might be dated applies in two places. The book
installs `causalTree` from GitHub, which no longer builds against current
R here, and its single honest tree is superseded by the forest. And the
book's `gsynth` call now runs through `fect`, whose cross-validation is not
the one in Xu (2017). Neither changes what was needed from StatsPAI.

No GPL source was read. `gsynth` and `fect` are MIT; their R code was
consulted once, to find out which cross-validation the current version
runs. `MatchIt`, `optmatch`, `glmnet` and `grf` were used as black boxes.

## Results by chapter

| Chapter | R | StatsPAI | Outcome |
| --- | --- | --- | --- |
| Randomized controlled trials | `lm_robust(se_type="HC2")`, Neyman by hand | `sp.regress(robust='hc2')`, `sp.ttest(unequal=True)`, `sp.lm_lin` | Equal to 1e-12, three specifications. |
| | permutation test, 10,000 draws | `sp.fisher_exact` | p = 0.055 against 0.058; Monte Carlo SE 0.002. |
| Selection on observables | additive, interacted and centred-interacted regressions, HC2 | `sp.regress`, `sp.lm_lin(vce='hc2')` | Equal to 1e-15. |
| Heterogeneous treatment effects | interaction model; `margins` after a logit | `sp.regress` + `sp.lincom`; `sp.logit` + `sp.margins` | Equal to 1e-15; AME and SE equal to 6 digits printed. |
| Matching | `ntile` subclassification | `sp.match(method='stratify')` | Equal, ATT and ATE. |
| | `matchit(method="nearest")`, Mahalanobis, caliper | `sp.match(replace=False, m_order=...)` | Equal to 1e-12 once `MatchIt`'s defaults are spelled out (below). |
| | `matchit(method="optimal")` on a supplied score | `sp.optimal_match` | Ours has the smaller total distance (64.447 against 64.460). `optmatch` rounds distances. |
| | `matchit(method="full")` | none | **Missing.** Added `sp.full_match`. |
| Inverse weighting and doubly robust | Hajek weights with `svyglm` | `sp.ipw` | Point estimate equal. SE 0.25355 against 0.25361: ours accounts for the estimated score, `svyglm` takes the weights as given. |
| | AIPW by hand | `sp.aipw(cross_fit=False)` | 3.216425 against 3.216440. **Found the clipping bug.** With `trim=0`: equal to 3e-11. |
| Double machine learning | cross-fitted `rlasso`, then OLS of residuals | `sp.dml(model='plr')` | Equal to 1e-12 with the same folds. A list of `(train, test)` splits crashed: fixed. |
| | `hdm::rlassoEffect` | `sp.rlasso_effect` | Equal to 1e-14. |
| Selection on unobservables and DML-IV | cross-fitted residuals, `ivreg`, HC3 | `sp.dml(model='pliv')`; `sp.iv` | PLIV equal to 1e-12. HC3 after 2SLS differs from `AER` by 1e-5 and equals `estimatr` and `ivreg`: a convention (below). |
| Difference-in-differences and DML-DiD | Chang's score by hand; `causalweight` | none | **Missing.** Added `sp.dml_did`. |
| Meta-learners | S, T, X with a forest in one arm and `lm` in the other | `sp.metalearner` | One learner for both arms only. Added `(control, treated)` pairs; equal to R to 1e-13 with deterministic learners. |
| Causal trees and forests | `causal_forest` and its tools | `sp.causal_forest` and the same tools | Statistically the same (below). `subset=` missing and text columns refused on one of two interfaces: both fixed. |
| | `causalTree::honest.causalTree` | none | Not added (open items). |
| Synthetic DiD and RD | `Synth` on the Basque data | `sp.synth` | Not rerun: this pair is a documented reference disagreement (Track A 07, T4). |
| | `gsynth` with five treated units and two covariates | `sp.gsynth` took one unit | **Missing, and the covariate path was wrong.** Rewritten; equal to 1e-12. |
| | `augsynth` on Kansas, ridge | `sp.augsynth` | ATT −0.0400629 on both sides; same ridge penalty to 13 digits. |
| | `synthdid`, `sc_estimate`, `did_estimate` on Proposition 99 | `sp.sdid(method=)` | −15.6038, −19.6197, −27.3491 on both sides. Placebo SEs 9.1 to 10.3 against 9.2 to 9.8 over three seeds each. |
| | `rdbwselect`, `rdrobust` with `p = 1, 2` | `sp.rdbwselect`, `sp.rdrobust` | Equal in every printed digit. |

## Defects found

### `sp.gsynth(covariates=)` was not the generalized synthetic control

`_partial_out_covariates` regressed the outcome on the covariates alone,
with no intercept, on the pre-treatment rows of the control units, added
`1e-8` to the diagonal, and subtracted the fit from everybody. Xu (2017)
estimates the coefficients in the interactive fixed effects model of the
control group, net of unit effects, period effects and the factors. The
two agree only when the covariates are uncorrelated with all of those. In
the example data that ship with `gsynth` they are correlated by
construction.

| one treated unit of `simdata`, covariates `X1`, `X2` | effect |
| --- | --- |
| R `gsynth`, 2 factors | 5.7015 |
| StatsPAI before | 6.6750 |
| StatsPAI now | 5.7015 |
| R `gsynth`, 0 factors | 6.6222 |
| StatsPAI before | 7.0551 |
| StatsPAI now | 6.6222 |

No test caught it because every test of that path asserted only that the
estimate was finite. Track A module 19 compares `sp.gsynth` with R without
covariates, and that path is unchanged to the last digit.

The new estimator is `synth/_gsynth_multi.py`. The control-group fit
alternates least squares for the coefficients with, for the residual
matrix, the two-way within transformation followed by a rank-r truncation.
Given the coefficients that pair is the exact minimiser over additive
effects plus a rank-r matrix (the within transformation commutes with the
truncation because it cannot raise rank), so this is alternating least
squares on Bai's objective and converges to the same point as `fect`'s
`inter_fe`. With the convergence tolerance tightened on both sides:

| `simdata`, five treated units | max relative difference from `gsynth` |
| --- | --- |
| ATT, 0 to 3 factors, with covariates | 7e-14 |
| coefficients | 6e-13 |
| ATT by period | 7e-13 |
| ATT, 0 to 3 factors, no covariates | 1e-15 |
| staggered adoption (three adoption dates) | 2e-14 |

### `sp.aipw` clipped silently and reported that it had not

`_fit_propensity` returned `clip(p, 0.01, 0.99)`. The caller then counted
how many of the returned values lay outside [0.01, 0.99], found none, and
stored `n_propensity_clipped = 0`. The warning for the sandwich variance
was conditional on that count and could not fire. On the book's design 538
of 20,000 scores were clipped.

The clip itself is a defensible default and is kept. It is now counted,
announced, and adjustable (`trim`). On the 600-row fixture, where 14
scores are clipped, the estimate is 1.57 clipped and 0.68 unclipped: the
user needs to know which one they are looking at.

### Smaller ones

- `sp.dml(fold_indices=list_of_splits)` died in `np.asarray` with
  "inhomogeneous shape". Splits are now converted to labels when they are
  a partition with complementary training sets, and refused with a reason
  otherwise.
- `sp.causal_forest(data=, y=, d=, x=)` did `data[x].to_numpy()` and so
  refused a text column that `sp.causal_forest("y ~ d | x + region", data)`
  expanded. And a forest fitted either way on a categorical column could
  not predict from a frame holding that column. Both fixed with the
  existing `one_hot_covariates` / `one_hot_newdata`.

## What was added

**`sp.dml_did`** (`dml/did.py`). Two-period DiD with cross-fitted
learners. Panel and repeated cross-sections; `observational` and
`experimental` scores; with and without in-sample normalisation. Evidence:

- Python `DoubleML` (`DoubleMLDID`, `DoubleMLDIDCS`) with the same folds:
  two layouts, two scores, with and without normalisation, once with
  linear and logit learners and once with forests. Estimate and SE equal
  to 2e-15 in all 16. The committed fixture holds the eight with
  deterministic learners.
- Identity: without normalisation the estimate is
  `mean((D - m) / (1 - m) * (dY - g0)) / mean(D)`, Chang's estimator.
- Known truth: ATT 2 with a quadratic trend and propensity. In a run of
  300 replications at n = 600 the mean estimate was 2.000 and coverage
  0.92 with learners that include the square; regression DiD with the
  confounder entered linearly is off by more than 0.5. The committed test
  runs 200.

  The first version of that test used a random forest and failed. That is
  not a defect of the estimator. On the design now in the test the forest
  gives 2.69 at n = 600 (Monte Carlo SE 0.03), 2.26 at 5,000 and 2.12 at
  20,000: its bias in the two nuisances is large and falls slowly. The
  docstring says so with these numbers.

  It differs from the book's code in one place: the book estimates the
  treated share on each training fold; `DoubleML` and StatsPAI use the
  full sample. With normalisation off the share cancels; with it on the
  two are different (both valid) normalisations.

**`sp.full_match`** (`matching/full.py`), also `sp.match(method='full')`.
An optimal full matching is a minimum-cost edge cover of the bipartite
graph. A minimum-cost edge cover is a maximum-weight matching on the gains
`min_i + min_j - c_ij`, plus the cheapest edge of every vertex the matching
leaves uncovered. So the whole problem is one call to
`linear_sum_assignment`, and no network-flow library is needed. Zero
distances can leave an edge whose two ends are both covered elsewhere;
those are removed, which keeps the cost and makes every component a star.
Evidence:

- Enumeration of all edge covers on 40 problems with up to 3 x 4 units,
  half of them with tied and zero distances: equal to 2e-16.
- `optmatch` with `tol = 1e-7` on an 11-unit problem: equal.
- The book's design (500 treated, 1,500 controls): total 2.60346 against
  2.67327 for `MatchIt`'s default and 2.603461582 for `MatchIt` with
  `tol = 1e-9`. Ours is 2.603461581. 320 sets on both sides.
- Given `optmatch`'s sets, the ATT and ATE and their matched-set-clustered
  standard errors equal `lm(weights=)` + `sandwich::vcovCL` to 1e-14.

  What is *not* claimed: that the estimate equals `MatchIt`'s. The optimum
  is nearly flat and the sets are not unique at any tolerance `optmatch`
  can be run at. Estimates differ by 1e-3 in relative terms.

**`sp.gsynth(treat=)`**. Above.

**`average_treatment_effect(subset=)`**. grf's argument. grf's own forest
outputs were injected into a grf object, as in the existing clustered
operator fixture, and `subset = x1 > 0` evaluated for the four targets
with rows, clusters, and equalised cluster weights: equal to 5e-15.

**Per-arm learners in `sp.metalearner`**. `outcome_model=(control,
treated)` for T and X, `cate_model=(control, treated)` for X. The AIPW
scores behind `estimate` and `se` use the same pair.

## Differences that are conventions

- **`MatchIt` matches without replacement and in a fixed order.**
  `matchit(method="nearest")` is `replace=False, m_order="largest"` on the
  score; with a Mahalanobis distance it is `m_order="data"`; the caliper is
  in standard deviations. `sp.match` defaults to replacement with ties
  kept (Stata `teffects`). Each is reproduced exactly when asked for.
- **Standardised differences.** `MatchIt` divides by the treated group's
  standard deviation for the ATT; `sp.balance_table` divides by the pooled
  one. `FullMatchResult.balance` follows the estimand.
- **HC2 / HC3 after 2SLS.** `sp.iv` uses the leverage of the second-stage
  regression, as `estimatr::iv_robust` and the `ivreg` package do (equal to
  12 digits). `AER::ivreg` + `sandwich` uses the diagonal of
  `X (Xhat'X)^-1 Xhat'`, which was negative for some observations of the
  book's DML-IV design (range −0.0008 to 0.0024). HC0 and HC1 are common
  to all.
- **DML standard error.** The book reports HC3 from a regression of
  residual on residual with an intercept. `sp.dml` reports the score-based
  one, which is HC0 without an intercept. 0.023666 against 0.023659.
- **`glmnet` penalties.** Exact mapping to `sp.shrinkage` for lasso and
  ridge, tested to 1e-8 on predictions. `glmnet` scales the outcome before
  applying a ridge penalty, which is why a ridge `lambda` cannot be carried
  over without `sd(y)`.
- **Number of factors in `gsynth`.** Ours is the leave-one-period-out rule
  of the paper. `gsynth` 1.4.0 delegates to `fect`, whose cross-validation
  has several schemes (masking control observations in rolling windows
  among them) and a one-standard-error option. Which of them `gsynth`
  ends up running was not traced. Both choose 2 on `simdata`.

## Causal forests, for the record

On the book's simulated design (2,000 rows, 26 covariates, 4,000 trees),
two StatsPAI seeds against three grf seeds, all spaced by 100,000:

| | StatsPAI | grf |
| --- | --- | --- |
| ATE | −1.1569, −1.1596 | −1.1555, −1.1579, −1.1664 |
| its SE | 0.1367, 0.1364 | 0.1367, 0.1366, 0.1364 |
| ATT | −1.1540, −1.1613 | −1.1553, −1.1575, −1.1627 |
| RMSE of OOB CATE against truth | 1.2636, 1.2672 | 1.2578, 1.2619, 1.2648 |
| calibration: mean / differential | 0.999, 1.150 | 1.004, 1.154 |
| top three variables | 1, 6, 3 | 1, 6, 3 |
| AUTOC, forest trained on the other half | 1.49 (0.18) | 1.40 (0.18) |

This is a screen with a handful of seeds, not an equivalence test. It
found nothing.

The mean standard error of predictions at 101 test points was 0.42 for
StatsPAI and 0.37 to 0.40 for grf. Not pursued.

## Second round: the open items, decided

Bryce delegated the decisions on 2026-10-07. What was done with each.

1. **Elastic net with `glmnet`'s conventions: added as `sp.glmnet`.**
   A separate function rather than a method of `sp.shrinkage`, because
   the point is a different statement of the penalty and the two would
   contradict each other inside one signature. Gaussian and binomial,
   penalty factors, user-supplied penalties, `lambda.min` / `lambda.1se`.
   `glmnet` is GPL: nothing was read, and four conventions had to be
   found by experiment.

   | convention | how it was found |
   | --- | --- |
   | Gaussian outcome scaled to unit variance, so the ridge penalty in data units is `lambda / sd(y)` | the closed-form ridge reproduces `glmnet` only with that factor (1e-12) |
   | first point of a computed path is the fit at an infinite penalty | ridge coefficients of 2.5e-36 at the first penalty, 2.7e-3 when the same penalty is supplied |
   | path stops on a relative gain (Gaussian), an absolute gain (binomial), threshold 1e-5 | two low-signal designs where the rules stop at different lengths; all 17 path lengths tried are reproduced |
   | `cv.glmnet` lets each training set build its own path and interpolates linearly in `lambda` | refitting the folds at the full-sample penalties misses `cvm` by up to 7e-3; own path plus `predict(s=)` reproduces it to 4e-16 |

   The third and fourth are worth knowing when reading any `cv.glmnet`
   output: the cross-validated error at a penalty is not the error of the
   model fitted at that penalty.

   With 40 rows and 60 predictors and a penalty of 0.001, `glmnet` at
   `thresh = 1e-14` is 1e-3 from the minimiser in the coefficients (its
   convergence test is on the change per sweep, which is tiny when
   coordinate descent crawls). Ours stops on a tighter threshold and
   satisfies the subgradient conditions to 1e-7 there; that penalty is
   compared loosely and the KKT check is the evidence. This is a
   convergence gap in the reference, not a convention.

   Speed: pure NumPy coordinate descent on the Gram matrix. 2,000 rows,
   200 predictors, 10 folds: 1.7 s. It will be slow for thousands of
   predictors; that is what scikit-learn's Cython solver is for.
2. **`sp.ipw(se_method='sandwich')` for Horvitz-Thompson and clipped
   weights: added.** No package computes this exact variance (Stata
   `teffects ipw` is the Hajek estimator), so the reference is an
   independent implementation in the test: the stacked estimating
   equations differentiated numerically. Twelve combinations, 1e-10.
   Coverage of the Horvitz-Thompson interval in 300 replications is in the
   test as well.
3. **`CausalForest.variable_importance()`: the announced default change
   was made.** The warning dated the change to 1.33 and the version is
   1.38, so the buffer the deprecation policy asks for had long passed.
4. **Single-unit `sp.gsynth` without covariates: left alone.** Track A
   module 19 calls it with `n_factors=None`, so its committed result
   depends on the present rule for choosing the number of factors.
   Changing the rule would change a frozen artifact for no gain in
   correctness (both rules are cross-validations). It stays until the JSS
   paper is re-anchored; the docstring says which rule applies where.
5. **`sp.dml` default learners: left alone, with more evidence.** On the
   book's linear PLR design (n = 2,000, 10 covariates, 40 replications)
   the default gradient boosting gives a mean estimate of 1.945 for a true
   2.0 (Monte Carlo SE 0.011) and 95% intervals that cover 0.825; lasso
   and linear learners give 2.001 and 0.975. This is the known issue that
   `summary()` already prints a note about and that the Track B coverage
   table reports (0.88 at n = 500). The default is quoted in the JSS
   manuscript, so changing it is a re-anchoring decision, not a patch.
   When it is revisited: choosing between a linear and a boosted learner
   by out-of-fold loss would fix this design without giving up the
   non-linear ones.
6. **Honest causal tree: not added.** No reference implementation could
   be built on this machine, and CLAUDE.md asks for one or for an
   analytical test of something worth having; the forest and
   `sp.policy_tree` cover the uses.
7. **Continuous-treatment DML-DiD: not added.** One cell of the book, and
   `causalweight` would be the only reference.

## Rerun

```bash
# R references (needs the packages named in the script header)
Rscript tests/reference_parity/_fixtures/_generate_yuksel_aydede_causal_ml.R
# DoubleML reference
python tests/reference_parity/_fixtures/_generate_dml_did_doubleml.py
Rscript tests/reference_parity/_fixtures/_generate_glmnet.R
pytest tests/reference_parity/test_yuksel_aydede_causal_ml_parity.py \
       tests/reference_parity/test_dml_did_doubleml_parity.py \
       tests/reference_parity/test_glmnet_r_parity.py -q
```

`ivreg` must not be loaded in the same R session as `AER`: both register
methods for class `ivreg`, and `sandwich::vcovHC` then fails with
"hatvalues() could not be extracted".
