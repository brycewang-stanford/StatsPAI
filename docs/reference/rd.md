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
                bwselect='mserd',                 # 'mserd'|'msetwo'|'certwo'|'cerrd'
                kernel='triangular',              # 'triangular'|'epanechnikov'|'uniform'
                vce='hc3',                        # 'hc0'–'hc3' | 'cluster'
)

# 2D / boundary RD
r = sp.rd2d(df, y='y', x1='lon', x2='lat', boundary_coords=...)

# Regression Kink Design (Card, Lee, Pei, Weber 2015)
r = sp.rkd(df, y='ui_benefits', x='earnings', c=cutoff)

# Intent-to-treat at running variable (RDIT)
r = sp.rdit(df, y='y', x='x', c=0, covs=['z'])
```

## Honest inference and local randomisation

```python
# Armstrong-Kolesar honest CIs under smoothness bounds
sp.rdhonest(df, y='y', x='x', c=0, M=0.1, kernel='triangular')

# Local randomisation (Cattaneo-Titiunik-Vazquez-Bare)
sp.rdrandinf(df, y='y', x='x', c=0, wl=-2, wr=2)
sp.rdwinselect(df, y='y', x='x', c=0)      # window selection
sp.rdsensitivity(df, y='y', x='x', c=0)    # sensitivity to window
```

## Diagnostics

```python
sp.cjm_density(df, x='score', c=0)          # Cattaneo-Jansson-Ma density
sp.mccrary_density(df, x='score', c=0)      # McCrary legacy
sp.rdplot(df, y='y', x='x', c=0)            # binned-scatter RD plot
```

## Heterogeneous treatment effects

```python
sp.rdhte(df, y='y', x='x', c=0, by='group')    # by-subgroup CATE
sp.rd_forest(df, y='y', x='x', c=0, covs=[...])
sp.rd_boost(df, y='y', x='x', c=0, covs=[...])
sp.rd_lasso(df, y='y', x='x', c=0, covs=[...])
```

## External validity (Angrist-Rokkanen)

```python
sp.rd_extrapolate(df, y='y', x='x', c=0,
                  conditioning=['z1','z2'])     # conditioning on covariates
```

## Power analysis

```python
sp.rdpower(df, y='y', x='x', c=0, tau=1.5)
sp.rdsampsi(df, y='y', x='x', c=0, tau=1.5, power=0.80)
```

## One-click dashboard

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
