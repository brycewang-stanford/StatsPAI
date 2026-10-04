# Regression Discontinuity

`statspai.rd` — 18+ RD estimators, diagnostics, and inference methods
across 14 modules (v0.9.1, ~10,300 LOC).

## Core estimation

```python
# Sharp / fuzzy / kink RD with bias-corrected robust inference
r = sp.rdrobust(df, y='earnings', x='score', c=0,
                fuzzy='treatment',                # optional — fuzzy RD
                deriv=1,                          # sharp=0, kink=1
                covs=['age', 'sex'],              # Calonico et al. 2019 adjustment
                bwselect='mserd',                 # or 'cct' for R rdrobust parity
                kernel='triangular',              # 'triangular'|'epanechnikov'|'uniform'
                vce='hc3',                        # 'hc0'–'hc3' | 'cluster'
)

# 2D / boundary RD
r = sp.rd2d(df, y='y', x1='lon', x2='lat', treatment='treated')

# Regression Kink Design (Card, Lee, Pei, Weber 2015)
r = sp.rkd(df, y='ui_benefits', x='earnings', c=cutoff)

# Intent-to-treat at running variable (RDIT)
r = sp.rdit(df, y='y', time='date', cutoff='2020-03-01')
```

## Honest inference and local randomisation

```python
# Armstrong-Kolesar honest CIs under smoothness bounds
sp.rd_honest(df, y='y', x='x', c=0, M=0.1, kernel='triangular')

# Local randomisation (Cattaneo-Titiunik-Vazquez-Bare)
win = sp.rdwinselect(df, x='x', c=0, covs=['z1', 'z2'], wobs=2)
wl, wr = win.attrs['recommended_window']   # endpoints on the score's scale
sp.rdrandinf(df, y='y', x='x', c=0, wl=wl, wr=wr)
sp.rdrandinf(df, y='y', x='x', c=0, wl=wl, wr=wr, fuzzy='d')  # ITT test
sp.rdsensitivity(df, y='y', x='x', c=0)    # sensitivity to window
```

`sp.rdwinselect` returns one row per nested window: the smallest balance
p-value across the covariates, the covariate attaining it, and a binomial
test of the split around the cutoff. The recommended window is the largest
one such that it and every window inside it reach `alpha` (0.15 by
default). `wl` and `wr` are the window's endpoints, so a placebo cutoff at
1 with half-width 0.75 is `c=1, wl=0.25, wr=1.75`.

`sp.rdrandinf` reports the randomization p-value, the large-sample one
(`model_info['pvalue_asymptotic']`), the power against `d`
(`model_info['power']`) and an interval from inverting the test (`ci=`
takes the grid of effects to test). With `p > 0` the randomization
p-value is built by permuting outcomes against the scores, which differs
from `rdlocrand`; the function's Notes give the reason.

## Multiple cutoffs and multiple scores

```python
# Each unit has its own cutoff, in a column
res = sp.rdmc(df, y='y', x='score', cutoff_var='cutoff')
res.summary()          # per cutoff, weighted, and pooled on the normalized score

# Two scores: several points on the boundary, plus the pooled estimate on
# the signed perpendicular distance to it
sp.rdms(df, y='y', x1='s1', x2='s2', treat='d',
        cutoff1=[0, 30, 0], cutoff2=[0, 0, 50], xnorm='dist')
```

Both report the conventional point estimate with the robust
bias-corrected standard error, interval and p-value, as `rdmulti` does.

## Diagnostics

```python
sp.rddensity(df, x='score', c=0)            # Cattaneo-Jansson-Ma density
sp.mccrary_test(df, x='score', c=0)         # McCrary legacy
sp.rdplot(df, y='y', x='x', c=0)            # binned-scatter RD plot
```

## Heterogeneous treatment effects

```python
sp.rdhte(df, y='y', x='x', z='group', c=0)     # by-subgroup CATE
sp.rd_forest(df, y='y', x='x', c=0, covs=[...])
sp.rd_boost(df, y='y', x='x', c=0, covs=[...])
sp.rd_lasso(df, y='y', x='x', c=0, covs=[...])
```

## External validity (Angrist-Rokkanen)

```python
sp.rd_extrapolate(df, y='y', x='x', c=0,
                  covs=['z1', 'z2'])           # conditioning on covariates
```

## Power analysis

```python
sp.rdpower(df, y='y', x='x', c=0, tau=1.5)
sp.rdsampsi(tau=1.5, target_power=0.80)   # design-stage: no data needed
```

## Single-call dashboard

```python
sp.rdsummary(df, y='earnings', x='score', c=0)
# Prints: CCT sharp/robust estimate, bandwidths, density test,
# placebo cutoffs, covariate balance, falsification checks,
# bias-corrected CI, honest CI, and one-figure diagnostic plot.
```

## Validation

97 RD tests pass. `rd/_core.py` consolidates kernel, weighted-least-
squares, and sandwich-variance primitives from 9 files into one 191-line
canonical module.
