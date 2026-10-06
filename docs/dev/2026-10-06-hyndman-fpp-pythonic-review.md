# Review against *Forecasting: Principles and Practice, the Pythonic Way*

Date 2026-10-06. Worktree `.claude/worktrees/hyndman-fpp`, branch `wt/hyndman-fpp`.

## What was examined

Hyndman, Athanasopoulos, Garza, Challu, Mergenthaler and Olivares, *Forecasting:
Principles and Practice, the Pythonic Way* (OTexts, online edition updated 28
September 2026). The folder `改进建议-收集整理/23-Hyndman-FPP-Pythonic`
(gitignored) holds the code extracted from the 15 chapters, the rendered pages
with their printed output, and the 59 data files.

The book runs statsforecast, statsmodels, utilsforecast, hierarchicalforecast,
mlforecast, prophet and neuralforecast. Its methods come from R packages
written by the same first author, so there were two references for every
computation:

| Reference | Version | Role |
| --- | --- | --- |
| R `forecast` | 9.0.2 | canonical, written by the authors of the methods |
| R `stats` (`stl`, `decompose`, `Box.test`, `arima`) | 4.5.2 | canonical |
| R `hts` | 6.0.3 | canonical for reconciliation |
| statsforecast / utilsforecast | 2.0.1 / 0.2.10 | what the book prints |
| statsmodels | 0.14.6 | STL, VAR, Ljung-Box in the book |
| hierarchicalforecast | 1.5.3 | what the book prints in chapter 11 |

Each chapter was run three ways on the book's data: the book's library, R, and
the first `sp` call a reader would write. Chapters 3 to 13 were covered.
Chapters 14 (neural networks) and 15 (foundation models) are outside the scope
of the package. Chapter 6 (judgmental forecasts) has no computation.

## What `sp` had before

`sp.arima`, `sp.var`, `sp.varsoc`, `sp.unitroot`, `sp.corrgram`, `sp.ardl`,
`sp.garch`. On the book's data the existing functions were numerically right
where they could be called: KPSS (3.561 and 0.099 on the Google series), the
Ljung-Box column of `sp.corrgram`, VAR lag selection (5 by AIC, 1 by BIC) and
non-seasonal ARIMA estimates all matched. Everything else in the book had no
`sp` counterpart: exponential smoothing, the benchmark methods, decomposition,
accuracy measures, cross-validation, automatic seasonal ARIMA, forecasting a
regression with ARIMA errors, hierarchical reconciliation.

## Defects found in `sp.arima`

1. **Inferior local maxima, silently.** The fit ran one quasi-Newton search.
   Over 32 models on nine of the book's series it ended more than `1e-3`
   log-likelihood units below the maximum in 13. Examples: an MA(3) for
   Egyptian exports (-186.84 against -183.35), ARIMA(1,1,1) with drift for egg
   prices (58.17 against 59.00 in R), a cement model that stopped on the
   invertibility boundary. statsmodels reported convergence in each case.
2. **Fifty iterations.** The statsmodels default limit applied. On log gas
   production an ARIMA(3,1,3)(0,1,1)[4] stopped at 323.81; the maximum is
   326.47.
3. **AICc of seasonally differenced models.** The correction used `n - d`
   observations. It should use `n - d - D s`.
4. **No forecast from a model with regressors.** `forecast()` raised a
   statsmodels error and had no argument for future regressor values. The
   coefficients were named `x1`, `x2` whatever the columns were called.
5. **Automatic selection.** No seasonal orders, no choice of mean or drift, a
   full grid over `(p, q)` only. With a fixed seasonal part one monthly series
   took 31 seconds.

All five are fixed. See "What landed".

## What landed

New public functions, all in `timeseries/`:

| Function | Book chapter | Reference and evidence |
| --- | --- | --- |
| `sp.ets` | 8 | `forecast::ets`. Filter, likelihood, fitted values and analytic forecast variances equal R's at R's parameters to `1e-13` on 15 models. Estimates reach a likelihood at least as high as R's on every model tested. |
| `sp.simple_forecast` | 5 | `naive`, `snaive`, `rwf(drift=TRUE)` to `1e-14`. `meanf` to `1e-14` after swapping the quantile. |
| `sp.forecast_accuracy` | 5 | `forecast::accuracy` to `1e-14`. Winkler score, coverage and CRPS by hand. |
| `sp.tscv` | 5 | error matrix of `forecast::tsCV` to `1e-14`. |
| `sp.stl`, `sp.classical_decompose` | 3, 12 | `stats::stl`, `forecast::mstl`, `stats::decompose` to `1e-10`. `stlf(method="naive")` to `1e-15`. |
| `sp.ljungbox` | 5, 9 | `Box.test(fitdf=)` to `1e-14`. |
| `sp.ndiffs`, `sp.nsdiffs` | 9 | same answer as `forecast` on every series tried; seasonal strength to `1e-16`. |
| `sp.boxcox_lambda` | 3 | `BoxCox.lambda` to `1e-4`, which is R's stopping tolerance. |
| `sp.fourier_terms` | 7, 10, 12 | `forecast::fourier` to `1e-14`. |
| `sp.hierarchy`, `sp.reconcile` | 11 | `hts::MinT` and `combinef` to `1e-14`. |

Changes to `sp.arima`:

- Estimation is on the differenced series. This is the likelihood Stata's
  `arima` maximises and the one R's `arima` gives when handed the differenced
  series. It is also several times faster than carrying the differences as
  states. Forecasts against `forecast::Arima` now agree to `1e-6`.
- Every fit starts twice, from statsmodels' starting values and from
  conditional-sum-of-squares estimates as R's `CSS-ML` does. It is then
  checked by a simplex search from the better answer and one from the
  default start, each polished by a quasi-Newton run. The second start
  matters: for ARIMA(2,1,2)(0,1,1)[4] on log gas production every
  statsmodels optimiser, and R's `arima` on the differenced series, stop at
  319.40. `forecast::Arima` and `sp.arima` reach 323.47.
- `auto=True` follows Hyndman and Khandakar (2008). `period=` adds seasonal
  orders with `D` set by seasonal strength. A mean or drift is chosen by AICc.
  The search is stepwise by default and `stepwise=False` fits every model up to
  total order five. Models with a root of modulus below 1.01 are not returned. The
  paper says 1.001. R rejects a cement model whose smallest root is 1.0058. The three leading candidates are refitted thoroughly before one is
  returned.
- `forecast(level=(80, 95), exog=..., dof_adjust=...)`. Regressors keep their
  names.

On the fifteen book series compared, `auto=True` returns the model
`auto.arima(approximation=FALSE)` returns on every one: fifteen of fifteen
stepwise, and twelve of twelve with `stepwise=False` (the three monthly series
were not run that way in R either).

A parallel audit (Maitra, `docs/dev/2026-10-06-maitra-static-dynamic-review.md`,
commit `8e68b894`) changed `sp.arima` on the same day: exact diffuse
initialisation of the integration states and a search that does not depend on
the unit of measurement. The two were merged. Estimation follows this pass
(differenced series, two starts, simplex checks, Hyndman-Khandakar search).
The level model that produces fitted values, standard errors and forecasts
uses that pass's exact diffuse start, and a badly scaled series is searched on
a rescaled copy with the standard errors mapped back. Estimates, likelihood,
forecasts and standard errors are invariant to the unit. Both passes' tests
run on the merged code.

Second round, same day: the likelihood of the differenced series is evaluated
by the innovations algorithm in `timeseries/_arma_core.py` (numba). It returns
the number statsmodels' Kalman filter returns, to `1e-8`, about fifteen times
faster, and the level model is built only for the fit that is returned. The
book's three monthly series went from one to five minutes each to 2 to 7
seconds, where R takes 5 to 16 with the exact likelihood. The selections are
unchanged: 27 of 27 comparisons with `auto.arima` agree.
`tests/test_arma_innovations_likelihood.py` checks the likelihood against the
filter and against closed forms for AR(1) and MA(1).

Third round: `sp.reconcile(sd=)` reconciles normal prediction intervals
(section 11.6 of the book). The reconciled covariance is `S G W_h G' S'`, with
`W_h` built from the base forecasts' standard deviations and the correlations
of the residual covariance a MinT method estimated. On the book's tourism
hierarchy the interval widths equal hierarchicalforecast's `Normality` method
to `1e-14` for bottom-up, OLS and structural weights and to `2e-11` for
variance weights. For `mint_shrink` they differ by up to 2%, the covariance
convention of difference 7 below. A Monte Carlo test reconciles 200,000 normal
draws and recovers the reported standard deviations.

The same round fixed an import failure that was not this pass's: a dataclass in
`regression/gam.py` had `slice(0, 0)` as a default, which Python 3.11 refuses
(`slice` is unhashable before 3.12). `import statspai` raised on 3.11. The
repository venv is 3.10, where it does not show.

Tests:

- `tests/reference_parity/test_forecasting_r_parity.py`, 92 tests against
  committed R output on simulated series. Generators are
  `_fixtures/_generate_forecasting_data.py` and
  `_fixtures/_generate_forecasting_R.R`.
- `tests/test_forecasting_toolkit.py`, 52 tests of known truths, hand
  computations and error paths.
- `tests/external_parity/test_hyndman_fpp_pythonic.py`, 17 tests of numbers
  printed in the book. Opt-in with `STATSPAI_FPPPY_DIR`.

## Differences from the references, each with its reason

1. **R's forecast intervals for ETS(A,N,A) and ETS(M,N,A) are slightly off.**
   `forecast.ets` enters `gamma` in the forecast variance at lags `m - 1`,
   `2m - 1`, and so on. Table 6.2 of Hyndman, Koehler, Ord and Snyder (2008)
   has it at `m`, `2m`. Shifting the term by one lag reproduces R to `1e-13`.
   A simulation of 400,000 paths agrees with the table. `sp.ets` follows the
   table. The effect is small when `gamma` is small. Worth reporting upstream.
2. **R's `ets` optimiser stops early.** It runs one Nelder-Mead search capped
   at 2,000 iterations. `sp.ets` evaluates the same likelihood (R's own C code
   at our parameters returns our value to `1e-7`) and restarts the search. On
   24 fits to the book's series ours was never lower and was higher by more
   than 0.1 on thirteen. The largest gap is a monthly model with 17 parameters,
   113.6 against 99.96. Automatic selection then differs on three of eight
   series, where our choice has the lower AICc under R's own likelihood.
3. **statsforecast's ARIMA can stop at a local maximum.** For US leisure
   employment, ARIMA(2,1,0)(1,1,1)[12], the book's library gives a seasonal AR
   coefficient of -0.046 and a log likelihood of 390.49. R and `sp.arima` give
   0.3295 and 394.96. The book's printed ARIMA tables are therefore not
   reproduced digit for digit on seasonal models. The opt-in test pins the R
   values there.
4. **Likelihood of a differenced ARIMA model.** `forecast::Arima` keeps the
   differences as states with a diffuse prior. On a series differenced both
   regularly and seasonally its log likelihood is about `1.5e-2` away from the
   exact likelihood of the differenced series, and coefficients move in the
   fourth decimal. R's `arima` applied to the differenced series reproduces
   `sp.arima` to `1e-5`.
5. **Innovation variance in ARIMA forecasts.** `forecast::Arima` and `fable`
   divide the residual sum of squares by `n - k`. `stats::arima`, Stata and
   `sp.arima` use the maximum likelihood value. `forecast(dof_adjust=True)`
   gives the former.
6. **`meanf` uses a Student-t quantile.** The book's formula and `fable::MEAN`
   use the normal one, as `sp.simple_forecast` does.
7. **`mint_shrink` in hierarchicalforecast.** It centres the residual
   covariance and divides by `T - 1`. `hts` and `fabletools` do neither. On the
   tourism hierarchy the reconciled forecasts differ by up to `9e-3` relative.
   `sp.reconcile` follows `hts`.
8. **STL defaults.** `sp.stl` uses R's settings with a seasonal window of 11.
   statsmodels defaults to a window of 7, degree 1, five inner passes and no
   interpolation. The docstring gives the arguments that reproduce it.
9. **Robust STL.** After the default fifteen robustness iterations the Fortran
   and Cython implementations differ by up to `2e-5` on one simulated series.
   They agree to `1e-13` after two iterations and to `8e-11` on the book's
   employment series. Rounding decides which side of a weight threshold an
   observation falls on.
10. **`auto.arima` approximates by default.** For series longer than 150 or
    with period above 12 R ranks candidates by a conditional sum of squares.
    `sp.arima` always uses the exact likelihood. Comparisons were made with
    `approximation=FALSE`.

## Open items

| Item | Note |
| --- | --- |
| Top-down by forecast proportions, middle-out | Not implemented. |
| Box-Cox inside the forecasters | `forecast` and `fable` take `lambda=` and bias-adjust the back-transform. Here the transform is the caller's job. The docstring of `sp.boxcox_lambda` gives the adjustment. |
| Bagged forecasts (section 12.5) | The block bootstrap of STL remainders is not packaged. |
| Time series features (chapter 4) | Only the strength of trend and seasonality. The `tsfeatures` catalogue is not reproduced. |
| ETS with missing values | Refused. `forecast::ets` interpolates. |
| Prophet, neural networks, foundation models | Outside the scope of the package. |
| Regression helpers of chapter 7 | `sp.regress` with `predict(what="prediction")` covers the examples. A trend and seasonal dummy shorthand does not exist. |
| Report the `forecast.ets` variance issue upstream | Draft for Bryce to send. |

## Draft of an upstream report (for Bryce to send)

Repository `robjhyndman/forecast`. Checked on version 9.0.2 with R 4.5.2.

> **`forecast.ets`: prediction intervals of ETS(A,N,A) and ETS(M,N,A) are too
> wide at horizons that are multiples of the seasonal period**
>
> For the models with an additive season and no trend, the forecast standard
> deviation at `h = m, 2m, ...` is larger than the formula in Table 6.2 of
> Hyndman, Koehler, Ord and Snyder (2008) and than `simulate()` on the same
> fit. The numbers are reproduced exactly if `gamma` enters `c_j` at
> `j = m - 1, 2m - 1, ...` instead of `j = m, 2m, ...`. Models with a trend
> (AAA, AAdA, MAA) agree with the table.
>
> ```r
> library(forecast)
> set.seed(1)
> y <- ts(10 + rep(c(2, -1, 0.5, -1.5), 20) + cumsum(rnorm(80, 0, 0.3)) + rnorm(80),
>         frequency = 4)
> fit <- ets(y, model = "ANA", alpha = 0.3, gamma = 0.5)
> fc <- forecast(fit, h = 9, level = 95)
> sd_r <- as.numeric(fc$upper - fc$mean) / qnorm(0.975)
> a <- 0.3; g <- 0.5; m <- 4
> cj <- function(shift) a + g * (((1:8) + shift) %% m == 0)
> sd_table <- sqrt(fit$sigma2 * c(1, 1 + cumsum(cj(0)^2)))  # gamma at j = m, 2m
> sd_shift <- sqrt(fit$sigma2 * c(1, 1 + cumsum(cj(1)^2)))  # gamma at j = m-1, 2m-1
> set.seed(2)
> sims <- replicate(50000, simulate(fit, nsim = 9, future = TRUE))
> round(cbind(h = 1:9, forecast_ets = sd_r, table_6.2 = sd_table,
>             shifted = sd_shift, simulated = apply(sims, 1, sd)), 4)
> ```
>
> | h | `forecast.ets` | Table 6.2 | shifted | simulated |
> | --- | --- | --- | --- | --- |
> | 3 | 1.2928 | 1.2928 | 1.2928 | 1.2920 |
> | 4 | 1.6055 | 1.3412 | 1.6055 | 1.3465 |
> | 5 | 1.6447 | 1.6447 | 1.6447 | 1.6473 |
> | 8 | 1.9663 | 1.7571 | 1.9663 | 1.7702 |
> | 9 | 1.9985 | 1.9985 | 1.9985 | 2.0028 |
>
> With the usual small estimates of `gamma` the effect is in the fifth
> digit, which is probably why it has gone unnoticed.

## Rerun recipe

```bash
# committed fixtures (needs R with forecast and hts)
python tests/reference_parity/_fixtures/_generate_forecasting_data.py
Rscript tests/reference_parity/_fixtures/_generate_forecasting_R.R
pytest tests/reference_parity/test_forecasting_r_parity.py tests/test_forecasting_toolkit.py -q

# the book's own numbers
STATSPAI_FPPPY_DIR=<folder with the csv files> \
  pytest tests/external_parity/test_hyndman_fpp_pythonic.py -q
```

`forecasting_sp_ets.json` holds the parameters `sp.ets` estimated. The R script
passes them to `forecast:::pegelsresid.C`. If the ETS optimiser changes, rerun
both scripts in that order.
