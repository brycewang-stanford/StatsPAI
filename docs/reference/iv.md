# Instrumental variables

`statspai.iv` — the unified IV namespace: a fixest-style formula front end
(`sp.iv` / `sp.ivreg`), a modern single-endogenous reporting bundle
(`sp.iv_diag`), a k-class / JIVE estimator panel (`sp.iv_compare`), and two
frontier estimators (`sp.kernel_iv`, `sp.continuous_iv_late`).

See also the decision guide: [Choosing an IV estimator](../guides/choosing_iv_estimator.md),
and the exhaustive auto-generated listing under
[Full API reference → iv](api/iv.md).

## The formula front end — `sp.iv` / `sp.ivreg`

IV models use the fixest convention: endogenous regressors and their
instruments go in parentheses, `(endog ~ instruments)`. Everything outside the
parentheses is exogenous.

```python
import statspai as sp

data = sp.datasets.card_1995()

# Card (1995) returns to schooling: educ instrumented by college proximity.
r = sp.ivreg(
    "lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa",
    data=data,
)
print(r.summary())

# Multiple excluded instruments + an exogenous control:
r = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df)

# IV with high-dimensional fixed effects absorbed (reghdfe / ivreghdfe style):
r = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb="firm + year")
```

`sp.iv` and `sp.ivreg` are the same estimator; `ivreg` is kept as the
Stata-flavoured alias. Two-stage least squares is the default.

## Modern reporting bundle — `sp.iv_diag`

For the common single-endogenous case, `sp.iv_diag` returns the full
post-2022 reporting standard in one object: the point estimate, first-stage
strength (effective F), weak-instrument-robust confidence intervals
(Anderson–Rubin and, optionally, conditional-likelihood-ratio and k-class
inversions), and bootstrap alternatives.

```python
res = sp.iv_diag(
    data, y="lwage", endog="educ", instruments="nearc4",
    exog=["exper", "expersq", "black", "south", "smsa"],
    cluster="region",
    include_clr_ci=True,   # conditional likelihood-ratio CI
    include_k_ci=True,     # k-class CI
)
res.summary()
```

!!! warning "Always check the first stage"
    A weak first stage biases 2SLS toward OLS and breaks conventional
    standard errors. `iv_diag` reports the effective F-statistic and
    weak-IV-robust intervals precisely so you do not have to interpret a 2SLS
    point estimate in the dark. As a rule of thumb, treat a first stage below
    the usual thresholds as a signal to report the Anderson–Rubin interval
    rather than the 2SLS CI.

## Comparing estimators — `sp.iv_compare`

With several (possibly many) instruments, 2SLS is biased in finite samples.
`sp.iv_compare` runs a panel of k-class and jackknife estimators side by side
so the many-instrument bias is visible:

```python
tbl = sp.iv_compare(
    "y ~ (d ~ z1 + z2 + z3) + x1", data=df,
    methods=("2sls", "liml", "fuller", "jive"),
)
tbl   # one row per estimator: coefficient, SE, CI
```

- **2SLS** — the workhorse; minimum-bias only with strong instruments.
- **LIML** — limited-information ML; better-centred under many instruments.
- **Fuller** — a finite-sample-adjusted LIML with finite moments.
- **JIVE** — jackknife IV; removes the own-observation bias of 2SLS.

## Frontier estimators

```python
# Nonparametric IV via kernel ridge in RKHS (Y on D instrumented by Z):
k = sp.kernel_iv(data, y="y", treat="d", instrument="z")

# LATE with a continuous instrument (quantile-bin Wald estimator):
late = sp.continuous_iv_late(data, y="y", treat="d", instrument="z")
```

## Identifying assumptions (agent-native)

Every IV estimator ships an agent card you can inspect before estimating:

```python
sp.agent_card("kernel_iv")["assumptions"]
# instrument relevance · exclusion restriction · exogeneity · (LATE) monotonicity
```

The core requirements are **relevance** (a non-zero first stage),
**exclusion** (the instrument affects the outcome only through the treatment),
**exogeneity / independence** of the instrument, and — for a LATE
interpretation — **monotonicity** (no defiers).

## Method-level API

The public top-level entry points used in examples and agent workflows are
documented here explicitly; the exhaustive namespace listing remains available
under [Full API reference -> iv](api/iv.md).

### `sp.iv(...)`

::: statspai.regression.iv.iv

### `sp.ivreg(...)`

::: statspai.regression.iv.ivreg

### `sp.kernel_iv(...)`

::: statspai.iv.kernel_iv

### `sp.continuous_iv_late(...)`

::: statspai.iv.continuous_iv_late

### `sp.iv_diag(...)`

::: statspai.iv.iv_diag

### `sp.iv_compare(...)`

::: statspai.iv.iv_compare
