# Stochastic Frontier Analysis

`statspai.frontier` — parametric stochastic frontier analysis with
Stata `frontier` / `xtfrontier` and R `frontier` / `sfaR` parity and
several extensions.

> **⚠️ Critical correctness fix (v0.9.3).** All prior versions carried a
> Jondrow-posterior sign error that systematically biased efficiency
> scores; the exponential path additionally returned NaN. **Re-run any
> prior frontier analyses.**

## Cross-sectional — `sp.frontier`

```python
r = sp.frontier(
    df, y='log_output', x=['log_k', 'log_l'],
    dist='half-normal',            # or 'exponential' | 'truncated-normal'
    cost=False,                    # production frontier
    usigma=['w1', 'w2'],           # heteroskedastic σ_u (CFG 1995)
    vsigma=['r1'],                 # heteroskedastic σ_v (Wang 2002)
    emean=['z1', 'z2'],            # BC95 inefficiency determinants
    vce='robust',                  # 'opg' | 'robust' | 'cluster' | 'bootstrap'
    cluster='firm_id',
)
r.summary()
r.efficiency(method='bc')          # Battese-Coelli 1988  E[exp(-u)|ε]
r.efficiency(method='jlms')        # Jondrow et al. 1982  exp(-E[u|ε])
r.efficiency_ci(alpha=0.05, B=500) # parametric-bootstrap CI
r.lr_test_no_inefficiency()        # Kodde-Palm mixed χ̄² one-sided LR
```

## Panel — `sp.xtfrontier`

```python
r = sp.xtfrontier(
    df, y='log_output', x=['log_k', 'log_l'], i='firm', t='year',
    model='ti',          # Pitt-Lee 1981 time-invariant
    # model='tvd',       # Battese-Coelli 1992 time-decay: u_it = exp(-η(t-T_i))·u_i
    # model='bc95',      # Battese-Coelli 1995 w/ z_it' δ
    # model='tfe',       # Greene 2005 true fixed effects
    bias_correct=True,   # Dhaene-Jochmans 2015 split-panel jackknife (TFE only)
    dist='truncated-normal',
)
```

## Advanced frontiers

| Function | Method |
| --- | --- |
| `sp.zisf(df, ...)` | Zero-Inefficiency SFA mixture (Kumbhakar-Parmeter-Tsionas 2013) |
| `sp.lcsf(df, ..., n_classes=2)` | Latent-Class SFA (Orea-Kumbhakar 2004 / Greene 2005) |
| `sp.malmquist(df, ..., period_col='year')` | Färe-Grosskopf-Lindgren-Roos 1994 Malmquist TFP: M = EC × TC |
| `sp.translog_design(df, x=[...])` | Cobb-Douglas → translog design matrix helper |
| `sp.metafrontier(group_fits)` | Cross-group metafrontier (Battese-Rao-O'Donnell 2004) |

## Post-estimation helpers

```python
sp.te_summary(r)                         # Stata-style TE descriptive table
sp.te_rank(r, with_ci=True, B=500)       # efficiency ranking + bootstrap CI
r.predict(new_data, type='frontier')     # frontier-only prediction
r.marginal_effects('emean')              # RH 1999 marginal effects
r.returns_to_scale()                     # Σ elasticities
```

## Validation

33 cross-sectional + panel tests cover parameter recovery for all three
distributions, cost vs production sign handling, heteroskedastic σ_u/σ_v,
BC95 determinants, LR tests, bootstrap CI structure, Pitt-Lee / BC92 /
BC95 / TFE / TRE panel recovery, and Monte-Carlo density-integration
($\int f(\epsilon)\,d\epsilon = 1$) kernel sanity checks for all three
distributions.
