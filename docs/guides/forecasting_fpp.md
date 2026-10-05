# Forecasting: the *Principles and Practice* workflow in StatsPAI

Hyndman and Athanasopoulos's *Forecasting: Principles and Practice* is the
standard text on business and economic forecasting. Its R edition runs on
`forecast` and `fable`, its Python edition on statsforecast, statsmodels and
hierarchicalforecast. This guide maps the book's workflow to `sp` and says
where the numbers agree with which library.

Every code block below runs as written. The series are simulated so that the
guide needs no download. The book's own data reproduce the same way; see
[the last section](#reproducing-the-books-numbers).

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(7)
idx = pd.period_range("2005Q1", periods=80, freq="Q")
t = np.arange(80)
season = np.tile([0.85, 1.05, 1.25, 0.85], 20)
trips = pd.Series((120 + 1.1 * t) * season * (1 + rng.normal(0, 0.03, 80)),
                  index=idx, name="trips")
train, test = trips.iloc[:64], trips.iloc[64:]
```

## The map

| Task | R (`forecast`) | Python edition of the book | StatsPAI |
| --- | --- | --- | --- |
| Box-Cox parameter | `BoxCox.lambda` | `coreforecast.boxcox_lambda` | `sp.boxcox_lambda` |
| STL, MSTL | `stl`, `mstl` | `statsmodels STL`, `statsforecast MSTL` | `sp.stl` |
| Classical decomposition | `decompose` | `seasonal_decompose` | `sp.classical_decompose` |
| Mean, naive, seasonal naive, drift | `meanf`, `naive`, `snaive`, `rwf` | `HistoricAverage`, `Naive`, `SeasonalNaive`, `RandomWalkWithDrift` | `sp.simple_forecast` |
| Residual autocorrelation | `Box.test`, `checkresiduals` | `acorr_ljungbox` | `sp.ljungbox` |
| Accuracy measures | `accuracy` | `utilsforecast evaluate` | `sp.forecast_accuracy` |
| Rolling-origin cross-validation | `tsCV` | `sf.cross_validation` | `sp.tscv` |
| Exponential smoothing | `ets` | `AutoETS` | `sp.ets` |
| Differencing orders | `ndiffs`, `nsdiffs` | `statsforecast.arima.ndiffs` | `sp.ndiffs`, `sp.nsdiffs` |
| ARIMA, automatic ARIMA | `Arima`, `auto.arima` | `ARIMA`, `AutoARIMA` | `sp.arima` |
| Regression with ARIMA errors | `Arima(xreg=)` | `AutoARIMA` with `X_df` | `sp.arima(exog=)` |
| Fourier terms | `fourier` | `utilsforecast fourier` | `sp.fourier_terms` |
| Hierarchies | `hts`, `aggregate_key` | `hierarchicalforecast aggregate` | `sp.hierarchy` |
| Reconciliation | `MinT`, `combinef`, `reconcile` | `HierarchicalReconciliation` | `sp.reconcile` |
| VAR | `vars::VAR` | `statsmodels VAR` | `sp.var`, `sp.varsoc` |

Every fitted model has `.forecast(horizon, level=(80, 95))`. It returns a table
with `forecast` and, for each level, `lower_<level>` and `upper_<level>`,
indexed by the forecast periods.

## Transformations and decomposition (chapter 3)

```python
lam = sp.boxcox_lambda(trips, period=4)          # Guerrero's method
dec = sp.stl(np.log(trips), period=4, robust=True)
dec.to_frame().head()                            # trend, seasonal, remainder
dec.strength                                     # {'trend': ..., 'seasonal': ...}
```

A `lambda` near zero says the seasonal swings grow with the level, so the
logarithm is decomposed. STL is additive.

`sp.stl` uses R's settings, with a seasonal window of 11 as in
`forecast::mstl`. Given the same arguments the components equal R's `stl` to
rounding error. statsmodels runs the same algorithm with other defaults. To
reproduce `STL(y, period=m).fit()` write

```python
sm_like = sp.stl(trips, 4, seasonal=7, seasonal_deg=1, inner_iter=5,
                 seasonal_jump=1, trend_jump=1, low_pass_jump=1)
```

Several seasonal periods give an MSTL decomposition with one seasonal
component each, for example `sp.stl(y, [48, 336])` on half-hourly data.

## Benchmarks, residual checks and accuracy (chapter 5)

```python
fcs = {
    "mean":   sp.simple_forecast(train, "mean").forecast(16),
    "naive":  sp.simple_forecast(train, "naive").forecast(16),
    "snaive": sp.simple_forecast(train, "snaive", period=4).forecast(16),
    "drift":  sp.simple_forecast(train, "drift").forecast(16),
}
acc = sp.forecast_accuracy(test, fcs, train=train, period=4)
acc[["RMSE", "MAE", "MAPE", "MASE", "winkler_95", "coverage_95"]]
```

`MASE` and `RMSSE` divide by the in-sample error of the seasonal naive
forecast. A value below one beats it. Because the forecast tables carry
intervals, the Winkler score and the empirical coverage come with them.
`crps=True` adds the continuous ranked probability score.

A model is adequate only if its residuals look like white noise.

```python
fit = sp.simple_forecast(train, "snaive", period=4)
sp.ljungbox(fit)                 # lags default to twice the period
```

Intervals can be bootstrapped from the residuals instead of assuming them
normal.

```python
fit.forecast(8, level=95, bootstrap=True)
```

Time series cross-validation refits at every forecast origin.

```python
cv = sp.tscv(trips, "snaive", horizon=4, initial=24, period=4)
cv.accuracy()                    # RMSE, MAE, MAPE by horizon
```

Any function of the training window works as a forecaster. It may return the
forecasts, a forecast table or a fitted model.

```python
cv_ets = sp.tscv(trips, lambda y, h: sp.ets(y, "MAM", period=4),
                 horizon=4, initial=40, step=4)
```

## Exponential smoothing (chapter 8)

```python
ses = sp.ets(train, "ANN")                    # simple exponential smoothing
holt = sp.ets(train, "AAN")                   # Holt's linear trend
damped = sp.ets(train, "AAdN", phi=0.9)       # damped, phi fixed
hw = sp.ets(train, "MAM", period=4)           # Holt-Winters multiplicative
auto = sp.ets(train, period=4)                # chosen by AICc
auto.model, auto.candidates.head(3)
auto.forecast(8, level=(80, 95))
auto.components().head()                      # level, slope, season, remainder
```

The three letters are the error, trend and seasonal component, each `N`
(none), `A` (additive) or `M` (multiplicative), with `Z` for an automatic
choice. A `d` after the trend letter damps it.

`sp.ets` evaluates the likelihood of `forecast::ets` exactly. At R's parameter
values it returns R's log likelihood, fitted values and prediction intervals to
rounding error. Its own estimates can differ, because R runs one Nelder-Mead
search capped at 2,000 iterations and `sp.ets` restarts the search. The
likelihood it reports is at least as high. With a seasonal period of 12 the
difference can be large enough to change which model AICc selects.

`beta` and `gamma` are on the state space scale used by R, where the trend
equation is `b_t = phi b_{t-1} + beta e_t`. The textbook's `beta*` equals
`beta / alpha`.

## ARIMA (chapter 9)

```python
sp.ndiffs(trips), sp.nsdiffs(trips, period=4)

fit = sp.arima(np.log(train), order=(1, 0, 0), seasonal_order=(0, 1, 1, 4),
               trend="c")
fit.summary()
sp.ljungbox(fit, lags=8)                      # df = lags - (p + q + P + Q)
fit.forecast(8, level=(80, 95))

auto = sp.arima(np.log(train), auto=True, period=4)
auto.order, auto.seasonal_order
auto.candidates.head()
```

`auto=True` is the algorithm of Hyndman and Khandakar. The seasonal difference
is set by the strength of seasonality, the regular difference by KPSS tests,
and the remaining orders and the constant by AICc. The search is stepwise.
`stepwise=False` fits every model up to total order five.

On the book's series it returns the model R's
`auto.arima(approximation=FALSE)` returns. R approximates the likelihood by
default on long or seasonal series and can then choose differently.

Three conventions to know when comparing numbers:

- The log likelihood is that of the differenced series, as in Stata and in R's
  `arima` applied to the differenced series. `forecast::Arima` keeps the
  differences as diffuse states, which moves the value by about `1e-2` and the
  coefficients in the fourth decimal on a doubly differenced series.
- `forecast::Arima` and `fable` report an innovation variance divided by
  `n - k` and build intervals on it. `fit.forecast(..., dof_adjust=True)` does
  the same. The default uses the maximum likelihood variance.
- A back-transformed forecast of a logged series is the median of the forecast
  distribution, not its mean.

## Dynamic regression and harmonics (chapter 10)

```python
df = pd.DataFrame({"y": np.log(trips.to_numpy())})
X = sp.fourier_terms(len(df), period=4, K=1)
df = pd.concat([df, X], axis=1)
df["trend"] = np.arange(1, len(df) + 1)

dyn = sp.arima("y", order=(1, 0, 0), exog=["trend", *X.columns], data=df)
dyn.params

future = sp.fourier_terms(8, period=4, K=1, start=len(df) + 1)
future.insert(0, "trend", np.arange(len(df) + 1, len(df) + 9))
dyn.forecast(8, level=95, exog=future)
```

A model with regressors forecasts conditionally on their future values. They
must be supplied, as a scenario or as forecasts of their own. The intervals do
not include their uncertainty.

## Hierarchical and grouped series (chapter 11)

```python
rows = []
for state, regions in {"A": ["a1", "a2"], "B": ["b1", "b2", "b3"]}.items():
    for region in regions:
        level = rng.uniform(20, 60)
        for q in range(48):
            rows.append({"country": "Total", "state": state, "region": region,
                         "quarter": q,
                         "trips": level + 0.2 * q + 3 * np.sin(np.pi * q / 2)
                                  + rng.normal(0, 2)})
long = pd.DataFrame(rows)

h = sp.hierarchy(long, [["country"], ["country", "state"],
                        ["country", "state", "region"]],
                 time="quarter", value="trips")
h.S.shape                                     # (8, 5): all series by bottom series

fits = {c: sp.ets(h.Y[c].iloc[:40], "ANA", period=4) for c in h.Y.columns}
base = pd.DataFrame({c: f.forecast(8)["forecast"].to_numpy()
                     for c, f in fits.items()})
resid = pd.DataFrame({c: f.response_residuals for c, f in fits.items()})

rec = sp.reconcile(base, h, method="mint_shrink", residuals=resid)
rec.forecasts.head()
```

Methods are `bottom_up`, `top_down`, `ols`, `wls_struct`, `wls_var`,
`mint_shrink` and `mint_cov`. The results equal `hts::MinT` and `combinef`.
hierarchicalforecast centres the residual covariance in `mint_shrink`, which
changes the third significant digit.

## What is not here

Prophet, neural networks and foundation models (sections 12.2, chapters 14 and
15) are outside the scope of the package. Intervals are not reconciled.
Bagged forecasts and the full `tsfeatures` catalogue are not packaged.

## Reproducing the book's numbers

Download `fpppy_data.zip` from the book's site and unpack it.

```bash
STATSPAI_FPPPY_DIR=/path/to/data \
  pytest tests/external_parity/test_hyndman_fpp_pythonic.py -q
```

The test file reproduces printed values from chapters 3, 5, 8, 9, 10, 11 and
12. Where statsforecast's optimiser stops short of the maximum, mostly on
seasonal ARIMA models, the test pins the value R's `forecast` gives and says
so.
