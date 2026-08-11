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
m = sp.garch(ret, p=1, q=1)                         # GARCH(p, q)
m.volatility; m.plot('conditional_volatility')
```

## Multivariate

```python
# Vector Autoregression
m = sp.var(df, variables=['gdp','infl','r'], lags=4)
m.impulse_response(shock='r', h=40, identification='cholesky')
m.variance_decomposition(h=40)
m.granger_causality(cause='r', effect='gdp')

# Bayesian VAR with Minnesota prior
m = sp.bvar(df, lags=4,                     # Minnesota prior
            lambda1=0.2, lambda2=0.5)
```

## Cointegration

```python
# Engle-Granger two-step
sp.engle_granger(df, variables=['y', 'x'])

# Johansen trace and max-eigenvalue
sp.johansen(df, variables=['y', 'x', 'z'], trend='c', lags=2)
sp.johansen(df, variables=['y', 'x', 'z'], test='maxeig')
```

## Local projections (Jordà 2005)

```python
sp.local_projections(
    df, outcome='gdp', shock='mp_shock',
    horizons=20,
    controls=['infl_lag', 'r_lag'],
    auto_lag=False,                                 # controls are used verbatim
)

# Match lpirfs::lp_lin with a unit Cholesky shock.
sp.local_projections(
    df, outcome='gdp', shock='mp_shock',
    horizons=20,
    identification='lpirfs_cholesky',
    endog_order=['gdp', 'mp_shock'],
)
```

## Structural break

```python
sp.structural_break(df, y='y', x=['x'], method='chow')        # known break
sp.structural_break(df, y='y', x=['x'], method='sup-f')       # unknown, sup-F
sp.structural_break(df, y='y', x=['x'], method='bai-perron',  # multiple
                    max_breaks=3)
```

## Result objects

```python
r.summary(); r.plot(); r.forecast(steps=10)
r.diagnostics()                       # Ljung-Box, Jarque-Bera, ARCH-LM
r.to_latex()
```
