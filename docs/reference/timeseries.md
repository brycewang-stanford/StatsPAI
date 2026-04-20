# Time Series

`statspai.timeseries` — classical and Bayesian time-series models,
cointegration tests, local projections, GARCH, and structural-break
detection.

## Univariate

```python
# ARIMA(p,d,q) with automatic order selection
m = sp.arima(y, order=(1,1,1), seasonal_order=(1,1,1,12))
m.forecast(steps=24); m.plot()

# GARCH family
m = sp.garch(ret, p=1, q=1, model='garch')          # 'garch'|'egarch'|'gjrgarch'
m.volatility; m.plot('conditional_volatility')
```

## Multivariate

```python
# Vector Autoregression
m = sp.var(df, columns=['gdp','infl','r'], lags=4)
m.impulse_response(shock='r', h=40, identification='cholesky')
m.variance_decomposition(h=40)
m.granger_causality(cause='r', effect='gdp')

# Bayesian VAR with Minnesota prior
m = sp.bvar(df, columns=['gdp','infl','r'], lags=4,
            prior='minnesota',
            lambda1=0.2, lambda2=0.5)
```

## Cointegration

```python
# Engle-Granger two-step
sp.cointegration(df[['y','x']], method='engle_granger')

# Johansen trace and max-eigenvalue
sp.cointegration(df[['y','x','z']], method='johansen', trend='c', k_ar_diff=2)

# Phillips-Ouliaris, Hansen
sp.cointegration(..., method='phillips_ouliaris')
```

## Local projections (Jordà 2005)

```python
sp.local_projections(
    df, y='gdp', shock='mp_shock',
    horizons=range(0, 21),
    controls=['infl_lag','r_lag'],
    se='driscoll_kraay',                            # HAC-panel SE
)
```

## Structural break

```python
sp.chow_test(y, x, break_point=t_star)              # known break
sp.quandt_andrews(y, x)                              # unknown break, sup-F
sp.bai_perron(y, x, n_breaks=3)                      # multiple breaks
```

## Result objects

```python
r.summary(); r.plot(); r.forecast(steps=10)
r.diagnostics()                       # Ljung-Box, Jarque-Bera, ARCH-LM
r.to_latex()
```
