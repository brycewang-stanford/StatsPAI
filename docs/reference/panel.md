# Panel data

`statspai.panel` — fixed/random effects, high-dimensional fixed effects
(reghdfe-style), interactive fixed effects (Bai 2009), panel FGLS, panel
limited-dependent models, panel unit-root tests, and panel utilities.

The exhaustive auto-generated listing is under
[Full API reference → panel](api/panel.md); this page is the guided tour.

## The dispatcher — `sp.panel`

`sp.panel` is a single entry point that dispatches to the estimator named by
`method=`:

```python
import statspai as sp

# Fixed effects (within estimator), clustered SEs:
r = sp.panel(df, "y ~ x1 + x2", entity="firm", time="year",
             method="fe", cluster="firm")
print(r.summary())

r = sp.panel(df, "y ~ x1 + x2", entity="firm", time="year", method="re")   # random effects
r = sp.panel(df, "y ~ x1 + x2", entity="firm", time="year", method="pooled")
```

| `method=` | Estimator |
| --- | --- |
| `"fe"` | One- or two-way fixed effects (within) |
| `"re"` | Random effects (GLS) |
| `"pooled"` | Pooled OLS |
| `"between"` | Between estimator |
| `"fd"` | First differences |

## High-dimensional fixed effects — `sp.hdfe_ols` / `sp.absorb_ols`

For many fixed-effect dimensions (firm × year × worker …), use the
reghdfe-style absorbing estimators, which demean iteratively instead of
building dummies:

```python
# Absorb firm and year FE; cluster by firm.
r = sp.hdfe_ols("y ~ x1 + x2 | firm + year", data=df, cluster="firm")

# Wild cluster bootstrap for few-cluster inference:
r = sp.hdfe_ols("y ~ x1 | firm", data=df, cluster="firm",
                wild=True, wild_n_boot=999, wild_seed=0)
```

`sp.Absorber` exposes the reusable demeaning operator, and `sp.demean`
returns the within-transformed array plus the singleton-keep mask, for callers
that want to drive the HDFE machinery directly. Singletons are dropped by
default (`drop_singletons=True`).

!!! tip "Performance"
    The HDFE path is backed by a Rust demeaning kernel when available and
    transparently falls back to NumPy / pyfixest otherwise — see
    [GPU acceleration](../guides/gpu_acceleration.md) and the `sp.fast.*`
    namespace.

## Interactive fixed effects — `sp.interactive_fe`

Bai (2009) interactive fixed effects allow a factor structure (unobserved
common shocks with heterogeneous loadings) instead of additive two-way FE:

```python
r = sp.interactive_fe(df, y="y", x=["x1", "x2"], id="firm", time="year",
                      n_factors=2)
```

## Other panel models

```python
# Feasible GLS for panels with heteroskedastic / correlated errors:
r = sp.panel_fgls(df, y="y", x=["x1", "x2"], id="firm", time="year")

# Panel limited-dependent models:
r = sp.panel_logit(df, y="d", x=["x1", "x2"], id="firm", time="year")
r = sp.panel_probit(df, y="d", x=["x1", "x2"], id="firm", time="year")

# Panel unit-root tests (Im–Pesaran–Shin default; also LLC, Fisher):
ur = sp.panel_unitroot(df, variable="gdp", id="country", time="year", test="ips")
```

## Comparing specifications — `sp.panel_compare`

Estimate the same model under several methods and line them up (a quick way
to see how sensitive a coefficient is to the FE/RE choice):

```python
tbl = sp.panel_compare(df, "y ~ x1 + x2", entity="firm", time="year",
                       methods=["fe", "re", "pooled"], cluster="firm")
tbl   # one column per method
```

A Hausman-style contrast between FE and RE is the classic use: if the FE and
RE coefficients diverge materially, the random-effects orthogonality
assumption is suspect and FE is the safer choice.

## Utilities

- `sp.balance_panel(df, entity, time)` — keep only units observed in every
  period (balanced panel).
- `sp.demean(...)` / `sp.Absorber(...)` — low-level within-transform building
  blocks shared by the HDFE estimators.
