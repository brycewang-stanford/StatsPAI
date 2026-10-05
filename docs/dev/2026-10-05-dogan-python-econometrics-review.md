# Dogan, *Introduction to Econometrics with Python*: what it showed about StatsPAI

2026-10-05. Worktree `.claude/worktrees/dogan-textbook`, branch
`wt/dogan-textbook`.

## What was examined

The book at <https://osmdogan.github.io/Python_Book> follows Stock &
Watson's *Introduction to Econometrics* chapter by chapter, with every
example written in Python for `statsmodels`, `linearmodels`,
`scikit-learn`, `rdrobust` and `arch`. The code extracted from its pages
(31 files, about 8,000 lines of code) and its 30 datasets sit in
`改进建议-收集整理/17-Dogan-IntroEconometricsPython/`, which is gitignored.

Two things make it useful after the Stock & Watson review of 2026-10-02.

- It covers chapters 8, 14, 16 and 17, whose replication files that review
  could not download, and it comes with their data.
- Its reference implementations are Python libraries that can be run
  here, so each number is compared with a live computation. The earlier
  review tested the Stata translator. This one tests what a Python user
  who knows `statsmodels` meets on arrival.

Chapters 0 to 3 teach Python, probability and statistics with `numpy` and
`scipy` and contain no estimator. Chapters 18 and 19 are theory.

## Method

For each chapter the book's computation was run twice on the book's data,
once with the library the book uses and once with the StatsPAI call a
user would write first, and the two were compared. Where the two
disagreed the difference was traced to a convention or to a defect.
Stata 18 was the third reference for ARIMA.

```bash
STATSPAI_DOGAN_DIR=<folder holding datasets/> \
  pytest tests/external_parity/test_dogan_python_econometrics.py
```

## What agreed

Agreement is relative difference in coefficients and standard errors.

| Chapter | Computation | Reference | Agreement |
| --- | --- | --- | --- |
| 4 to 7 | OLS, HC1, F tests, prediction | `statsmodels` | 1e-13 |
| 8 | Polynomials, logs, interactions, HC1 and HC3 | `statsmodels` | 1e-10 |
| 10 | Pooled, entity, two-way, random effects, clustered | `linearmodels` | 1e-6 as printed |
| 10 | LSDV with `C(state) + C(year)`, clustered | `statsmodels` | 1e-11 |
| 11 | Probit, logit, predictions, marginal effects at the mean and averaged | `statsmodels` | 1e-13 |
| 12 | TSLS, first-stage F, overidentification | `linearmodels` | 1e-6 as printed |
| 13 | Clustered OLS with school effects, difference-in-differences | `statsmodels`, `linearmodels` | 1e-12 |
| 13 | Sharp regression discontinuity | `rdrobust` | 4e-7 |
| 15 | AR and ADL with `.shift()` in the formula, HC1, Granger F | `statsmodels` | 1e-14 |
| 15 | ADF statistic and selected lag | `statsmodels` | identical |
| 16 | Distributed lags, Newey-West with 7 and 14 lags | `statsmodels` | 1e-14 |
| 17 | VAR coefficients, forecasts; Johansen trace statistics | `statsmodels` | identical as printed |
| 17 | Vector error correction model (`sp.vec`): loadings, cointegrating vector, short-run terms, log-likelihood | `statsmodels` `VECM` | 1e-6 as printed |
| 17 | GARCH(1,1) with robust errors | `arch` (book output) | third digit, see below |

## What was wrong, and what changed

| # | Finding | Kind | Status |
| --- | --- | --- | --- |
| 1 | `sp.arima` default was not the exact MLE. `enforce_stationarity=False` makes statsmodels start the state from a diffuse prior and drop the first `max(p, q + 1)` terms of the likelihood | correctness, silent | fixed |
| 2 | `sp.arima(auto=True)` skipped `(0, d, 0)`, scored larger models on fewer observations, and compared AICc across `d` | correctness, silent | fixed |
| 3 | `sp.arima(method='css')` returned exact ML | mislabel, silent | raises |
| 4 | `sp.test(res, "a = 0, b = 0")`, the `f_test` spelling, failed with "Unknown coefficient '0, b'" | usability | accepted |
| 5 | `sp.test(res, "I(x**2) = 0")` failed because the design names the term `I(x ** 2)` | usability | names match without regard to blanks |
| 6 | `sp.ivreg` refused the linearmodels formula `y ~ 1 + w + [x ~ z]` | usability | accepted (the Facure pass landed the same change first; its version is the one on main) |
| 7 | `res.bse`, `res.rsquared`, `res.resid`, `res.f_test` raised a bare `AttributeError` | usability | the message says where the value is |
| 8 | No ridge or principal-components prediction; no cross-validated MSPE (also open since the Stock & Watson review) | missing function | `sp.shrinkage` |
| 9 | IV results named terms `np.log[x]`, `I[x ** 2]`, `g[2]` where `sp.regress` names them `np.log(x)`, `I(x ** 2)`, `C(g)[T.2]`. `sp.regtable(ols, iv)` put one regressor on two rows, `sp.test(iv, "np.log(x) = 0")` failed, and Stata's `2.g` did not resolve | naming | IV results report the formula names |
| 10 | `predict()` and `sp.margins` on an IV fit with a transformed term could not rebuild the design (the formula still held the `(x ~ z)` block) | defect | the structural equation is rebuilt |
| 11 | `sp.vecm`, `sp.IV2SLS`, `sp.adfuller` raised a bare `AttributeError`, though `sp.vec`, `sp.ivreg`, `sp.unitroot` exist | usability | the message names the function; a near miss gets "did you mean" |
| 12 | The default `sp.arima` had no cross-language row | evidence | `tests/reference_parity/test_arima_default_stata_parity.py`, four models against Stata 18 on a committed series |

### Items 1 and 2

On the 223 quarters of GDP growth the book uses (1962:Q1 to 2017:Q3):

| | constant | AR(1) | log-likelihood |
| --- | --- | --- | --- |
| Stata 18 `arima y, ar(1)` | 2.9798 | 0.33656 | -564.99727 |
| `statsmodels` `ARIMA` | 2.97955 | 0.33653 | -564.99727 |
| `sp.arima` before | 2.94200 | 0.33552 | -562.06932 |
| `sp.arima` now | 2.97955 | 0.33653 | -564.99727 |

The earlier log-likelihood is higher because it is the likelihood of 222
observations. Track A module `39_arima` passes
`method="innovations_mle"`, which was exact all along, so the parity row
never exercised the default. Its note said the default "remains exact
state-space MLE". That was the assumption, and nothing tested it.

The consequence for order selection was larger than for the estimates. A
model with more states had more observations removed from its likelihood,
and AICc counted the gain as fit. With 100 observations of white noise
and `max_p = max_q = 2`, 150 replications:

| | chose `(0, 0, 0)` |
| --- | --- |
| before | 0 of 150 (it was not a candidate; MA(2) won 73 times) |
| likelihood fixed, search unchanged | 0 of 150 |
| now | 81 of 150 |

On a random walk the search now returns `(0, 1, 0)` 84 times in 150. It
could not return it before.

`d` is now chosen by KPSS tests on the successively differenced series
with the lag truncation of R's `forecast::ndiffs`. The statistic equals
`statsmodels.tsa.stattools.kpss` to 1e-10. This is not a full port of the
Hyndman-Khandakar stepwise search: every `(p, q)` up to the bounds is
fitted.

The independent check is `tests/test_arima_exact_likelihood.py`. It
writes the AR(1) likelihood in closed form, maximises it with scipy, and
compares both methods with it.

### Item 8

`sp.shrinkage(data, y, x, method='ridge' | 'lasso' | 'pcr' | 'ols')` on
the book's school data (1,966 districts in sample, 1,966 held out, 816
predictors built from 38 variables, their squares, cubes and
interactions):

| | chosen | 10-fold CV RMSPE | hold-out RMSPE | book |
| --- | --- | --- | --- | --- |
| OLS | | 77.0 | 64.1 | about 64 out of sample |
| Ridge | penalty 2,475 | 39.50 | 38.85 | 39.5 |
| Lasso | penalty 4,182, 61 nonzero | 39.67 | 39.09 | 39.7 |
| Principal components | 46 | 39.63 | 39.47 | 39.6 with 51 |

The fits agree with scikit-learn on the same standardised design: ridge
to 6e-13, principal components to 5e-12 (against `PCA(svd_solver="full")`),
the lasso to 7e-6 in the coefficients with the same 68 nonzero at the
book's penalty.

Three places where the book's numbers and these differ, all of them
conventions.

- The book standardises once and then splits into folds. `sp.shrinkage`
  standardises inside each training fold. The chosen tuning values move
  (ridge 1,413 on the book's coarse grid, 2,475 here on a finer one) and
  the RMSPE does not, because the cross-validation curve is flat there.
- The book's `PCA(n_components=51)` uses scikit-learn's randomised solver
  on a matrix this size, so its components are approximate. The first
  coefficient comes out as 0.229 against 0.231 from the exact
  decomposition.
- `LassoCV` holds scikit-learn's per-observation penalty fixed across
  folds. `sp.shrinkage` holds the penalty on the sum of squares fixed, as
  Stock & Watson write it, so the per-observation penalty is 11% larger
  inside a fold.

The data are read from `.dta` as single precision. Arithmetic on them in
pandas stays in single precision unless cast, and a reference computed
that way differs from the double-precision one at 1e-5.

## Documented differences from the book's libraries

None of these is a defect on either side. They are listed in
`docs/guides/migration-from-statsmodels.md` for users.

| Where | `statsmodels` / `linearmodels` / `arch` | StatsPAI |
| --- | --- | --- |
| Inference after robust errors | normal distribution | t with residual degrees of freedom |
| `cov_type="HC1"` on probit / logit | computes HC0 | `hc0` reproduces it; `hc1` applies `N/(N-K)`; `robust` is Stata's `N/(N-1)` |
| Clustered errors, `y ~ 1 + x + EntityEffects` | counts the intercept in `(N-1)/(N-K)` | no intercept to count; fourth digit |
| `IV2SLS(...).fit(cov_type="robust")` | HC0 | `robust="hc0"` |
| VAR residual covariance | divides by `T - Kp - 1` | divides by `T` (Stata) |
| Granger test in a VAR | F on system degrees of freedom `(2, 280)`, 4.88 | F on equation degrees of freedom `(2, 140)`, 5.06 |
| Johansen critical values | MacKinnon-Haug-Michelis | Osterwald-Lenum (Stata) |
| GARCH presample variance | exponentially weighted backcast | sample variance (Stata), or rugarch's rule |
| Engle-Granger | the book compares the residual ADF statistic with ordinary ADF critical values | `sp.engle_granger` uses critical values for estimated residuals |

Two slips in the book surfaced on the way. Chapter 12 calls
`.fit(vcov="HC1")` five times and chapter 15 `.fit(vcov_type='HC1')`
once. `statsmodels` ignores both keywords, so those fits carry classical
standard errors. And the Engle-Granger step compares with critical values
that are too small in absolute value.

## Open items

| Item | Why it matters | Size |
| --- | --- | --- |
| `sp.garch` has no Student-t likelihood (`arch_model(dist='t')`) | chapter 17 refits with t errors after the normality test rejects | medium |
| Panel results still name terms `np.log[x]` and `I[x ** 2]` | `sp.regtable(ols, fe)` splits the row. `sp.test` accepts either spelling. Relabelling the result broke `hausman_test()`, `compare()` and `f_test_effects()`, which refit from the stored column names, so the panel class needs its names separated from its columns first | medium |
| `sp.panel` has no time-effects-only method | `PanelOLS(... + TimeEffects)`; `sp.feols("y ~ x \| year")` covers it | small |
| `sp.structural_break` returns the break as a row position | the book reports a date; a `time=` argument would label it | small |
| `sp.acf` is Ackerberg-Caves-Frazer | a time-series user expects autocorrelations; `sp.corrgram` has them | naming, leave |
| Distributed-lag helper | cumulative multipliers are a reparameterisation the book builds by hand; `sp.regress` with `.shift()` and `hac_lags=` reproduces every number | not needed |
| Newey-West fixed-b critical values | carried over from the Stock & Watson review | medium |
| Track A `39_arima` could also run the default method | the Stata reference test above covers it outside Track A; a Track A row needs the R and Stata goldens regenerated | small |

## A correction to the first version of this note

The first version listed a missing `sp.vecm` as an open item. The model is
in the package as `sp.vec`, the Stata name. On the book's interest-rate
data `sp.vec(df, lags=3, rank=1, trend="rc")` gives the loadings
(-0.09455, 0.06884), the cointegrating vector (1, -1.00723, 1.57638) and
the log-likelihood -302.1175 that `statsmodels` `VECM(k_ar_diff=3,
coint_rank=1, deterministic="ci")` gives. Standard errors differ in the
third digit because `sp.vec` follows Stata's small-sample divisor. The
search that missed it looked for the name `vecm`, which is what finding 11
now answers.
