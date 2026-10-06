# Time series econometrics: from the correlogram to the Kalman filter

This guide follows the order of a graduate time-series course (the chapter
numbers are those of Neusser, *Time Series Econometrics*, Springer 2016)
and names the `sp` call for each step. Every function here has a reference
check against R or Stata on committed data; the table at the end says
which.

```python
import numpy as np
import pandas as pd
import statspai as sp
```

## One series

### Autocorrelation and the long-run variance (chapters 4 and 5)

```python
sp.corrgram(df, "growth", lags=12)        # ACF, PACF, Ljung-Box Q
sp.ljungbox(df["growth"], lags=[4, 8])    # portmanteau test alone
sp.lrvar(df, "growth")                    # sum of all autocovariances
```

The variance of a sample mean of an autocorrelated series is the long-run
variance divided by `T`, not the variance divided by `T`. `sp.lrvar` returns
both, with the kernel and the bandwidth it used. The default is the
Bartlett kernel with Andrews' data-driven bandwidth.
`kernel="qs", prewhite=1` is the choice of the `sandwich` package in R.

```python
fit = sp.arima(df["growth"], order=(1, 0, 3))
fit.summary()
fit.forecast(8)
sp.arima(df["growth"], auto=True, stepwise=False)   # search over orders
```

Likelihoods of mixed ARMA models have several maxima. When two orders give
nearly the same fit, prefer the smaller one, and do not read much into the
coefficients of a model whose AR and MA roots nearly cancel.

### The frequency domain and filters (chapter 6)

```python
spec = sp.periodogram(df, "growth", method="smoothed", spans=[5, 5])
spec.table.head()          # freq, cycle_length, spectrum, lower, upper
spec.plot()
sp.periodogram(df, "growth", method="ar")           # AR spectral estimate
sp.cumulative_periodogram_test(df, "resid")         # Bartlett's white-noise test
```

The raw periodogram (`method="raw"`) is not a consistent estimate of the
spectral density at any single frequency. Smooth it, or fit the AR
spectrum.

```python
hp = sp.tsfilter(df, "lgdp", method="hp", smooth=1600)
bk = sp.tsfilter(df, "lgdp", method="bk", low=6, high=32, K=12)
hp.trend, hp.cycle
hp.gain(np.linspace(0, np.pi, 200))    # what the filter does to each frequency
```

`method=` also takes `"cf"` (Christiano-Fitzgerald), `"bw"` (Butterworth)
and `"hamilton"` (the regression of `y[t + h]` on `p` lags). Three
cautions. The HP filter applied to data that are not seasonally adjusted
leaves the seasonal component in the cycle. The HP filter is two-sided, so
the last few cycle values are revised when new data arrive. And any of
these filters will produce a smooth "cycle" from a random walk; the gain
function shows which frequencies it lets through, not whether the data
have power there.

### Trends, unit roots and breaks (chapter 7)

```python
sp.unitroot(df, "lgdp", test="adf", trend="ct")     # also "pp", "kpss", "dfgls"
sp.zivot_andrews(df, "lgdp", model="both", lags=2)  # one break, date unknown
bn = sp.beveridge_nelson("lgdp", data=df)
bn.trend, bn.cycle, bn.long_run_multiplier
```

A break in the level or the slope of a trend looks like a unit root to a
test that does not allow for it. `sp.zivot_andrews` picks the break date
least favourable to the unit root and uses critical values that account
for the search. A rejection supports "stationary around a trend that
breaks once"; it does not date the break with any stated precision.

The Beveridge-Nelson trend is the long-horizon forecast of the series. Its
cycle depends on the AR order of the differenced series, sometimes a lot;
look at two or three orders before interpreting it.

### Volatility (chapter 8)

```python
sp.estat(ols_fit, "archlm", lags=5)                 # ARCH LM test
fit = sp.garch("ret", data=df, p=1, q=1, ar=1, dist="t")
fit.summary()
fit.forecast(10)           # conditional variance path
fit.value_at_risk(0.01)    # 1% quantile of the next observation
```

`p` counts lagged variances and `q` lagged squared innovations. `ar=`
adds autoregressive terms to the mean and `dist="t"` estimates the degrees
of freedom of a Student t innovation.

```python
sp.garch("ret", data=df, model="gjr")       # threshold GARCH
sp.garch("ret", data=df, model="egarch")    # exponential GARCH
```

Both let bad news move the variance by more than good news. In the
threshold model `gamma > 0` is that leverage effect; in EGARCH it is
`theta < 0`. Stata's `tarch` coefficient has the opposite sign of `gamma`
because Stata attaches it to positive shocks. A coefficient estimated at zero is
on the boundary of the parameter space, where the usual standard errors do
not apply; the fit warns, and the lower-order model is the one to report.

## Several series

### Cross-correlation (chapter 11)

```python
cc = sp.xcorr("gdp", "sentiment", data=df, lags=8, prewhiten="ar")
cc.table      # lag, xcorr, band, outside
cc.haugh      # portmanteau test of no relation at any lead or lag
```

At lag `h` the table holds the correlation of `gdp[t + h]` with
`sentiment[t]`, so a large value at a positive lag says that sentiment
moves first. Without `prewhiten=`, the autocorrelation of each series
leaks into every lag and the table cannot be read.

### VAR, structural VAR and their bands (chapters 12 to 15)

```python
sp.varsoc(df, maxlag=8)                 # lag order
fit = sp.var(df, lags=2)
fit.granger_table(); fit.stability(); fit.lm_test(4)
fit.forecast(8)

out = sp.irf(fit, periods=20, ci="asymptotic")        # delta method
out = sp.irf(fit, periods=20, ci="bootstrap", reps=1000, seed=1)
out["irf"]["x -> y"], out["lower"]["x -> y"], out["upper"]["x -> y"]
fit.fevd(20, ci="asymptotic")        # variance shares with standard errors
```

Orthogonalised responses depend on the order of the variables. The
bootstrap regenerates the series from resampled residuals and re-estimates
the VAR; `boot="hall"` reflects the percentile band about the estimate.

```python
nan = np.nan
bq = sp.svar(fit, long_run=[[nan, 0], [nan, nan]])    # Blanchard-Quah
bq.irf(40, cumulative=True, ci="bootstrap", reps=500, seed=1)
ab = sp.svar(fit, A=A_pattern, B=B_pattern)           # short-run AB model
sg = sp.svar(fit, sign={"demand": {"y": "+", "p": "+"}})
```

Bands of an AB or long-run model repeat the identification on every
bootstrap sample. Sign restrictions give a set of models, not one; their
`irf()` already reports the spread of that set, which is not a confidence
band.

### Cointegration (chapter 16)

```python
rank = sp.johansen(df, lags=1, trend="c")             # trace test
vec = sp.vec(df, lags=1, rank=2)
sp.johansen_lrtest(df, rank=2, beta_known=[1, 0, -1, 0])   # c - y stationary?
sp.johansen_lrtest(df, rank=2, beta=H)                     # beta = H phi
sp.johansen_lrtest(df, rank=2, loading=np.eye(4)[:, :3])   # weak exogeneity
```

`lags` counts lagged differences: Stata's `vecrank, lags(2)` and
`ca.jo(K = 2)` are `lags=1` here. The rank tests have non-standard
distributions. Once the rank is fixed, hypotheses about the cointegrating
vectors and the loadings are ordinary chi-squared tests, and they are
where the economics usually sits.

### State space models and the Kalman filter (chapter 17)

Write the model as `X[t] = F X[t-1] + V[t]`, `Y[t] = A + G X[t] + W[t]`
with `Var(V) = Q` and `Var(W) = R`.

```python
res = sp.kalman_filter(y, F=F, G=G, Q=Q, R=R)     # known matrices
res.states("smoothed"); res.forecast(8)

def build(theta):                                  # unknown parameters
    return {"F": theta[0], "G": 1.0,
            "Q": np.exp(theta[1]), "R": np.exp(theta[2])}

fit = sp.statespace(y, build, start=[0.5, 0.0, 0.0])
fit.table; fit.filter.states("smoothed")
```

Missing values in `y` are allowed anywhere, which is what makes mixed
frequencies easy: a quarterly state observed through an annual series is
an observation equation that is missing three quarters out of four.
Any matrix may vary over time (give it a leading time axis). For a
regression with slowly moving coefficients `sp.dlm` is shorter.

## What each function is checked against

| Function | Reference |
| --- | --- |
| `sp.corrgram`, `sp.ljungbox` | Stata `corrgram`, R `Box.test` |
| `sp.lrvar` | R `sandwich` |
| `sp.arima` | R `forecast`, Stata `arima` |
| `sp.periodogram` | R `spec.pgram`, `spec.ar`; Stata `pergram` |
| `sp.cumulative_periodogram_test` | Stata `wntestb` |
| `sp.tsfilter` | Stata `tsfilter`, statsmodels |
| `sp.unitroot` | statsmodels, Stata |
| `sp.zivot_andrews` | R `urca::ur.za` |
| `sp.beveridge_nelson` | its definition (no package found) |
| `sp.garch` | Stata `arch`, R `rugarch` |
| `sp.xcorr` | R `ccf`, Stata `xcorr` |
| `sp.var`, `sp.irf`, `sp.svar` | Stata `var`, `irf`, `svar`; R `vars` |
| `sp.johansen`, `sp.vec`, `sp.johansen_lrtest` | Stata `vecrank`, `vec`; R `urca` |
| `sp.kalman_filter`, `sp.statespace` | R `KFAS`, statsmodels |

The bootstrap bands are random and have no cross-language reference; they
are checked by coverage in simulations.
