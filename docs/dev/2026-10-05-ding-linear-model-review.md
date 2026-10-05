# Ding, *Linear Model and Extensions* (2024): review

Source: the book's replication files on Harvard Dataverse, 24 R programs
and some twenty data files. They cover least squares and its inference
(chapters 1 to 12), model selection and shrinkage (13 to 15),
transformations and weights (16 to 19), binary, categorical and count
outcomes (20 to 24), generalized estimating equations (25), quantile
regression (26) and survival analysis (27).

Method. Unlike the same author's causal inference book, these programs
call packaged estimators (`lm`, `glm`, `MASS`, `gee`, `quantreg`,
`survival`), so every real-data result has a direct counterpart. The
deterministic part of each program was rerun in R 4.5.2 to make an answer
key (`tests/external_parity/ding_linear_model_reference.R`), and each number
was recomputed with the `sp.*` function a user would reach for. Chapters
that only simulate were compared on one stored draw. Where the two sides
disagreed the first step was to find the first point of divergence, as §5.1
of `CLAUDE.md` asks. Everything that changed was then tested a second time
on a committed synthetic file against both R and Stata 18, so the evidence
does not depend on data that cannot be redistributed.

The book's code is two or three years old and leans on a few packages that
have aged (`gee`, `leaps`, `KernSmooth`). Their conventions were treated as
references to reproduce on request, not as defaults to adopt.

## What was wrong in StatsPAI

1. **`sp.cox` robust and clustered standard errors under tied event
   times.** With the default `ties='efron'` the coefficients and the
   model-based standard errors were Efron's, but the score residuals behind
   `robust=` and `cluster=` were the Breslow ones. The code said so in a
   comment; the result did not. With no ties the two coincide, which is why
   the paired-eyes example of chapter 27 matched R to 1e-14 while the
   COMBINE trial, whose relapse times are whole days, was off by 2%. The
   residuals are now Efron's under Efron and Breslow's under Breslow. On the
   committed file (40 tied event times) all four of `robust`, `cluster`,
   `strata` and `ties='breslow'` match `survival::coxph` to 1e-14 and
   `stcox` to Stata's convergence tolerance.
2. **`sp.regress` refused a full-rank regression through the origin.**
   Goodman's ecological regression in chapter 19, `t ~ 0 + x + I(1 - x)`,
   raised "perfectly collinear (|correlation| = 1)". A correlation of one
   means `x_j = a + b x_i`, which is a linear dependence only when the
   constant is in the model. Without an intercept the check now uses the
   uncentred cosine, and a single constant column is accepted as the
   intercept it is.
3. **`I(x^2)` failed with a message about `xor`.** patsy hands the inside
   of `I()` to Python, where `^` is bitwise XOR. Anyone arriving from R
   writes `I(exper^2)`, as chapter 16 does. Inside `I()` the caret is now
   read as a power, in every formula entry point (regress, glm, logit,
   poisson, qreg, iv, panel, did, cox, margins, survey). Outside `I()` it
   keeps its formula meaning.
4. **A string-labelled outcome with a built formula lost a category.**
   `sp.mlogit("survival ~ x + C(g)")` with survival coded "1", "2-4", "5+"
   stopped with "requires J >= 3 categories, got 2": patsy had expanded the
   outcome into dummies and the first one was taken as the outcome. Plain
   column formulas were never affected.
5. **`sp.estat(result, "leverage")` built the n by n hat matrix.** At
   100,000 observations that is 80 GB. Only the diagonal is needed and only
   the diagonal is computed now. Same numbers.

## What was missing

Added through existing entry points:

- `sp.regress(robust='hc4')`, Cribari-Neto's estimator, the fifth column of
  the chapter 6 tables.
- `sp.glm(link='cauchit')`, the fourth binary link of chapter 20.
- `sp.qreg(weights=)` and `vce='ker'`, for the census regressions of
  chapter 26. The fit and every covariance follow `quantreg`.
- `sp.cox("time ~ a*b + C(site) + I(x^2)", ...)`. The formula used to
  accept bare column names only and failed with a `KeyError` otherwise.
- `sp.mlogit`, `sp.ologit` and `sp.oprobit` results have `.predict()`,
  returning one column of probabilities per category, in sample or on new
  data. There was no way to get them out of sample before.
- `sp.kaplan_meier(conf_type='log' | 'log-log')`, the interval of R
  `survfit` and of Stata `sts list`. The default stays `'plain'`.
- `sp.estat(result, "leverage")` also returns `dffits`, the leave-one-out
  residuals, `press`, and the half-width of the exact leave-one-out
  prediction interval (chapter 12).

New functions:

- `sp.gee`, generalized estimating equations (chapter 25). Gaussian,
  binomial, Poisson and gamma margins with independence, exchangeable,
  AR(1) or unstructured working correlation. Returns the sandwich and the
  model-based covariance.
- `sp.ridge`, ridge regression with the scaling, GCV score and plug-in
  penalties of `MASS::lm.ridge` (chapters 14 and 15).
- `sp.boxcox`, the Box-Cox profile likelihood, its maximiser, a profile
  interval and the three likelihood-ratio tests Stata prints (chapter 16).
- `sp.best_subset`, exact best-subset selection by branch and bound
  (chapter 13). `sp.stepwise` on the Boston data stops at 8 variables with
  a BIC 9 points worse than the best 11-variable model, which is the
  reason the function exists.

`sp.stata` and `sp.from_stata` translate `xtgee` and `boxcox` to the new
functions. The `xtgee` translation writes Stata's conventions into the call
(`corstr='exchangeable'`, `vce='model'`, `dof_correction=False`, and
`scale=1.0` for the binomial and Poisson families), so running the Stata
line reproduces Stata's coefficients and model-based standard errors.

## Documented differences (not bugs)

- **Observed versus expected information.** For a non-canonical link R's
  `glm` reports the expected information and Stata's `glm` the observed
  one. `sp.glm` follows Stata by default. On the 100 simulated points of
  chapter 20 the two standard errors differ by 4% under probit and 19%
  under cloglog; `information='expected'` reproduces R to 1e-7. The
  observed version was checked against a numerical Hessian of the log
  likelihood.
- **`vce='robust'` in likelihood models is HC0 times N/(N-1)**, as in
  Stata. R's `sandwich()` is the bare HC0, which `robust='hc0'` gives.
- **Clustered Cox and GEE.** `sp.cox(cluster=)` carries Stata's
  `G/(G-1)`; `survival::coxph(cluster=)` does not. `sp.gee` reports the
  bare sandwich, like R `gee`; Stata `xtgee, vce(robust)` multiplies it by
  `G/(G-1)`. Both factors are exact and both are tested.
- **Negative binomial standard errors.** `MASS::glm.nb` holds theta fixed
  and reports the expected information of the mean model. `sp.nbreg` uses
  the joint observed information, as Stata's `nbreg` and statsmodels do
  (identical to statsmodels on the chapter 24 draw). The two differ by 2 to
  4%. For the same reason `sandwich(glm.nb)` is not our robust variance.
- **AIC of a linear model.** R counts the error variance as a parameter,
  Stata does not. Ours is Stata's, so it is lower than R's by exactly 2.
  Compare AIC across model families with one convention.
- **Scale in GEE.** R `gee` estimates the scale for every family. Stata
  `xtgee` fixes it at one for binomial and Poisson. `sp.gee` estimates it;
  `scale=1` gives Stata's model-based standard errors. Coefficients and
  robust standard errors do not depend on the choice.
- **AR(1) in GEE.** The two references estimate the AR(1) parameter with
  different moments, and `sp.gee` has both. `corstr='ar1'` pools the
  products of adjacent Pearson residuals over all clusters; it matches
  Stata `xtgee, corr(ar 1)` in both of its divisor conventions.
  `corstr='ar-m'` is R `gee`'s `"AR-M"`: each cluster's mean adjacent
  product, summed, over each cluster's mean square, summed. It matches
  `gee` to 1e-12. On a balanced panel the two differ only by the
  degrees-of-freedom term; with unequal cluster sizes the second weights
  short clusters more (0.557 against 0.568 on the committed file, and
  coefficients up to 5% apart). The first round left this unexplained.
  It was pinned down in the third by running `gee` on balanced panels of
  three sizes, where the implied divisor is `(m - 1)(N - p) / m` exactly,
  and then on unbalanced ones. The package source was not read (GPL).
- **Local polynomial regression.** `KernSmooth::locpoly` bins the data onto
  401 points before smoothing. `sp.lpoly` does not bin and equals the exact
  local linear fit to 1e-13; the two differ by up to 0.1 at the ends of the
  range on the chapter 19 example.
- **Quantile regression at a non-unique solution.** On the STAR data
  (median regression on a percentile score with 79 school dummies) and at
  the census median R warns "solution may be nonunique". The two programs
  then return different vertices of the same optimal face. Coefficients
  differ in the third digit, the objective does not.
- **Optimiser tolerance in R.** `nnet::multinom`, `MASS::polr` and
  `pscl::zeroinfl` call `optim` and stop early. Ours agree with them to
  1e-4 to 1e-3 and reach a log likelihood at least as high.

## One slip in the book's code

Chapter 12 computes leave-one-out prediction intervals for the Boston data
with `p = 13`, the number of covariates, where the formulas need the 14
coefficients including the intercept. The leave-one-out variance and the t
quantile are each off by one degree of freedom. The printed limits are
within 0.1% of the exact ones. `sp.estat(..., "leverage")` returns the
exact ones and was checked against forty refits.

## Second round

Bryce asked for the open decisions to be made and the work continued.

- **Kaplan-Meier default.** Decided: move it to `'log-log'`, through the
  deprecation process. In this release a call without `conf_type=` still
  returns the plain interval and raises a `DeprecationWarning`; the
  default changes in 1.40. Every documented call now names its interval.
- **GAM.** Decided: not now. A penalised-spline GAM with automatic
  smoothness selection that can be held to `mgcv` is a project of its
  own, and nothing else in the package depends on it. It stays on the
  list below.
- `sp.logit` / `sp.probit` / `sp.cloglog` results take
  `predict(data, what='confidence')`, like `sp.glm` results: probability,
  delta-method standard error, interval mapped through the link. Matches
  R's `predict(se.fit = TRUE)` to 1e-6.
- `sp.cox` reports the Wald and score tests. Both match `survival::coxph`
  (`summary.coxph` prints the Wald statistic rounded to two decimals; the
  reference is recomputed unrounded). Without ties the score test of a
  group indicator is the log-rank statistic to 1e-9.
- `sp.zip_model`, `sp.zinb` and `sp.hurdle` name their log likelihood,
  AIC and BIC like the other count models; the old keys stay.
- The Cook (1977) line in `diagnostics/estat.py` cites its own, verified
  entry (`cook1977detection`).

## Third round: the rest of the list

Bryce asked for what was left to be finished.

- **`sp.gam`** (chapter 16). Additive models with penalised cubic
  B-splines, one smoothing parameter per term, for Gaussian, binomial,
  Poisson and gamma responses. The basis, penalty, constraint and
  selection criteria are those of `mgcv::gam` with `s(x, bs = "ps")`, which
  makes it testable to the digit. At given smoothing parameters the
  coefficients, standard errors, effective degrees of freedom, criterion,
  predictions and term-wise curves match mgcv to 1e-9 in all three
  families. The selected parameters match to 1e-4 under both REML and
  GCV on curved outcomes. mgcv was used as a black box (`smoothCon` for
  the knots and penalty, `S.scale` for its rescaling); its source was not
  read.

  Two things came out of the comparison. First, the default. On the
  committed file, where every true effect is linear, GCV gives the first
  smooth 8.6 of its 9 degrees of freedom and UBRE does the same to the
  Poisson model; REML returns a straight line in both. mgcv's default is
  still `GCV.Cp` and its author's advice is REML, so `sp.gam` defaults to
  `method='reml'`; `method='gcv'` (with `gamma=`) reproduces mgcv's
  default. Second, the Poisson UBRE surface has two local minima. mgcv
  stops at 0.4320 and the grid-then-polish search here at 0.4295, the
  lower one, which mgcv confirms when handed those parameters. Where the
  two programs disagree on a GCV / UBRE choice, compare the criterion, not
  the curve.

  The solves go through the QR factor of the design stacked on the root of
  the penalty. With the normal equations a smooth pushed to a straight
  line (penalty of order 1e13) lost five digits.
- **`sp.conformal_regression`** (chapter 12). Split conformal, jackknife+
  and exact full conformal intervals for least squares, none of them by
  refitting in a loop. The full-conformal set is found exactly: with the
  new point appended, residuals are linear in its candidate outcome, so
  the set changes only at 2n crossings. It agrees with a brute-force grid
  to the grid step, the jackknife+ with n refits to 1e-12, and all three
  cover 90% on skewed heteroskedastic errors. `sp.conformal("regression",
  ...)` reaches it through the dispatcher.
- **`sp.stepwise(start=)`**. `method='both', start='full'` begins from the
  model with every candidate, as R's `step()` does. On Boston with BIC it
  reaches the best-subset optimum where the default start stops 8 points
  short. Both starts select what `step()` selects on the committed file.
- **Zero-inflated diagnostics.** The three n-length vectors of fitted
  values that `sp.zip_model`, `sp.zinb` and `sp.hurdle` kept in
  `diagnostics` are in `data_info`. The old keys still answer, with a
  `DeprecationWarning`, and are no longer listed when the dictionary is
  printed or exported. They go in 1.41.

## Open items

- **`sp.gam`** has univariate P-spline smooths only: no tensor products or
  smooths of two variables, no thin plate basis, no random-effect terms,
  and independence-based standard errors. Its bands are pointwise.
- **`sp.kaplan_meier`**: flip the default interval to `'log-log'` in 1.40.
- **Zero-inflated `diagnostics`**: remove the deprecated keys in 1.41.

## Evidence

- `tests/reference_parity/test_linear_model_extensions_parity.py`: 88
  tests on the committed synthetic file. R (`sandwich`, `MASS`, `leaps`,
  `gee`, `quantreg`, `survival`) to 1e-9 on closed forms and convex
  problems; Stata 18 (`xtgee`, `boxcox`, `stcox`, `sts`) to 1e-6, its own
  convergence tolerance. Also known-truth checks, brute-force checks of the
  subset search, and refits for the leave-one-out quantities.
- `tests/external_parity/test_ding_linear_model.py`: the book chapter by
  chapter, 38 tests, skipped without the data.

Rerun:

```bash
export STATSPAI_DING_LM_DIR=/path/to/unzipped/dataverse_files
Rscript tests/external_parity/ding_linear_model_reference.R   # slow: bootstrap, census
pytest tests/external_parity/test_ding_linear_model.py
Rscript tests/reference_parity/_fixtures/_generate_linear_model_extensions_R.R
# Stata, from tests/reference_parity/_fixtures:
#   do _generate_linear_model_extensions_Stata.do
pytest tests/reference_parity/test_linear_model_extensions_parity.py
```

The R key needs `car sandwich lmtest MASS glmnet leaps KernSmooth margins
nnet mlogit pscl gee quantreg survival timereg mlbench Matching mediation
foreign jsonlite`.
