# Multilevel / Mixed-Effects Models

`statspai.multilevel` — linear and generalised linear mixed models
with `lme4` / Stata `mixed` / `meglm` parity, rewritten in v0.9.3.

## Linear mixed model — `sp.mixed`

```python
r = sp.mixed(
    df, y='math_score', x=['ses', 'gender'],
    group='school',                            # two-level
    # group=['school', 'class'],               # three-level nested
    re_formula='1 + ses',                      # random intercept + random slope
    cov_type='unstructured',                   # new default; also 'diagonal', 'identity'
    method='reml',                             # or 'ml'
)
r.summary()
r.variance_components                          # dict of σ² estimates
r.ranef(conditional_se=True)                   # BLUPs with posterior SEs
r.r_squared()                                  # Nakagawa-Schielzeth marginal + conditional R²
r.predict(new_data, include_random=True)       # population or group-conditional
r.wald_test({'ses': 0, 'gender': 0})           # linear restriction test
r.plot(kind='caterpillar')                     # BLUP forest plot
```

## Generalised linear mixed models

```python
# Binomial / logit
sp.melogit(df, 'y', ['x1','x2'], group='school', nAGQ=7)

# Poisson / log
sp.mepoisson(df, 'count', ['x1'], group='region', offset='log_pop', nAGQ=11)

# Gamma / log (v0.9.3 new)
sp.megamma(df, 'cost', ['severity'], group='hospital', nAGQ=15)

# Negative binomial NB-2 (v0.9.3 new; alias family='negbin')
sp.menbreg(df, 'events', ['exposure'], group='county', nAGQ=7)

# Ordinal logit with K−1 reparameterised thresholds (v0.9.3 new)
sp.meologit(df, 'severity_cat', ['age','sex'], group='site', nAGQ=7)

# Generic GLMM dispatcher
sp.meglm(df, 'y', ['x'], group='g',
         family='binomial', link='logit', nAGQ=7)

r.odds_ratios()                # logit / ordinal
r.incidence_rate_ratios()      # Poisson / NegBin
r.predict(new_data, type='response' | 'linear')
```

### `nAGQ` — adaptive Gauss-Hermite quadrature

- `nAGQ=1` — Laplace approximation (verified to reduce to the exact
  Laplace formula within 1e-10).
- `nAGQ>1` — adaptive GHQ; matches Stata `intpoints(7)` and
  `lme4::glmer(nAGQ=7)` on small clusters with non-Gaussian outcomes.
- `nAGQ>1` is restricted to single-scalar random-effect models (the
  same restriction `lme4` imposes); multi-slope AGHQ is deferred
  because cost scales as `nAGQ^q`.

## Diagnostics

| Function | Purpose |
| --- | --- |
| `sp.icc(result)` | Intra-class correlation with delta-method logit-scale CI |
| `sp.lrtest(restricted, full)` | Likelihood-ratio test with Self-Liang mixed-$\bar\chi^2$ boundary correction when variance components are being tested |
| `result.aic()` / `result.bic()` | Information criteria with full normalisation constants (Poisson / Binomial) for cross-family comparability |

## Validation

- LMM fixed effects and variance components agree with
  `statsmodels.MixedLM` to four decimal places on random-intercept
  and unstructured random-slope specifications.
- Three-level nested variance components identified jointly and match
  `statsmodels.MixedLM(..., re_formula="1", vc_formula={...})` to
  two decimal places.
- GLMM parameter recovery tests confirm slope and random-intercept
  variance within expected sampling ranges.
- 53 tests pass (35 prior + 18 added in GLMM hardening:
  `TestAGHQ × 7`, `TestMEGamma × 3`, `TestMENegBin × 3`,
  `TestMEOLogit × 5`).
