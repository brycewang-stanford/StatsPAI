# Robustness workflow

A publication-quality causal inference result is not one estimate. It
is a *package*: the point estimate + a stack of robustness checks that
convince a reviewer the conclusion doesn't hinge on a single modelling
choice. StatsPAI's `.next_steps()` gives you a priority-ordered list.
This guide expands the full recipe for each design.

## The three-layer defence

```
Layer 1 — IDENTIFICATION: is my design valid?
  DID: parallel trends
  RD:  continuity, no manipulation
  IV:  relevance, exclusion, monotonicity
  SOO: unconfoundedness, overlap

Layer 2 — SPECIFICATION: does the result survive alternative choices?
  Bandwidth, polynomial order, covariate set, kernel, sample restriction

Layer 3 — SENSITIVITY: how much unobserved confounding would
kill the conclusion?
  E-value, Oster, sensemakr, Rambachan-Roth
```

## Layer 1 — Identification checks

### DID
```python
# Placebo pre-trends (per-period joint)
sp.pretrends_test(r)
sp.bjs_pretrend_joint(r)    # Borusyak-Jaravel-Spiess

# Honest inference under parallel-trends violations
sp.honest_did(r, max_M=0.5, method='smoothness')

# Decomposition: which 2x2 cells drive TWFE?
sp.bacon_decomposition(df, y='y', treat='d', time='t', id='i')
```

### RD
```python
# Density continuity (McCrary 2008 / Cattaneo-Jansson-Ma 2020)
sp.rddensity(df, x='running_var', c=0.0)
sp.mccrary_test(df, x='running_var', c=0.0)

# Covariate balance
sp.rdbalance(df, x='running_var', c=0.0, covariates=[...])

# Placebo cutoffs
sp.rdplacebo(df, y='y', x='running_var',
             true_cutoff=0.0, placebo_cutoffs=[-0.5, 0.5])
```

### IV
```python
# First-stage strength
r.diagnostics['First-stage F (d)']   # > 10 rule; > 30 comfortable

# Exclusion sensitivity
sp.plausibly_exogenous(df, y='y', d='d', z='z',
                       gamma_range=(-0.1, 0.1))

# Monotonicity (Imbens-Rubin / Kitagawa)
sp.kitagawa_test(df, y='y', d='d', z='z')

# Reduced form
sp.regress('y ~ z', data=df).tidy()
```

### Selection-on-observables
```python
# Overlap / common support
sp.overlap_plot(r)
sp.love_plot(r)
sp.ps_balance(r)

# E-value: how strong must unobserved U be to explain away effect?
sp.evalue(r)

# Oster delta bounds
sp.oster_bounds(r, delta=1.0)

# Sensemakr (Cinelli-Hazlett)
sp.sensemakr(r, benchmark_covariates=['X1'])
```

## Layer 2 — Specification robustness

### Specification curve / multiverse
```python
spec = sp.spec_curve(
    df, y='y', treat='d',
    covariate_sets=[
        ['X1'], ['X1', 'X2'], ['X1', 'X2', 'X3'],
        ['X1', 'X2', 'X3', 'X4'],
    ],
    estimators=['regress', 'ebalance', 'aipw'],
)
spec.plot()   # classic Simonsohn-Simmons-Nelson spec curve
```

### Bandwidth / kernel sensitivity (RD)
```python
sp.rdbwsensitivity(df, y='y', x='running_var', c=0.0)
```

### Subsample stability
```python
sp.robustness_report(r)
```
This runs: leave-one-cohort-out (DID), leave-one-unit-out (synth),
subsample reruns, and produces a single report.

## Layer 3 — Sensitivity to unobserved confounding

### E-value (VanderWeele-Ding 2017)
For any causal estimate on observational data, the **E-value** is the
minimum strength of association an unobserved confounder U would need
to have with *both* treatment and outcome (on the risk-ratio scale) to
fully explain away the observed effect.

```python
r = sp.dml(df, y='y', treat='d', covariates=[...])
ev = sp.evalue(r)
print(ev.evalue)          # e.g., 2.3 -> U needs RR > 2.3 with both
print(ev.evalue_ci)       # E-value for lower bound of CI
```

**Rule of thumb**: E > 2 is "reasonably robust"; E > 3 is strong; E < 1.5
is fragile (any of the measured covariates already has that level of
association).

### Oster (2019) coefficient stability bounds
```python
b = sp.oster_bounds(r, delta=1.0, r_max=1.3 * r.glance()['r_squared'].iloc[0])
print(b.lower, b.upper)
```
The interval `[lower, upper]` is where the true effect must lie if
unobservables are as important as observables (δ=1). If `0` is in this
interval, the effect is not robust.

### Sensemakr (Cinelli-Hazlett 2020)
```python
s = sp.sensemakr(r, benchmark_covariates=['X1', 'X2'])
s.summary()
s.plot()    # contour plot: partial R^2 of U with T and Y
```

### Bounds (Manski-style)
When no assumption feels credible, compute worst-case bounds:
```python
sp.bounds(df, y='y', treat='d', method='manski')
```

## Full robustness report in one call

```python
r = sp.callaway_santanna(df, y='y', g='g', t='t', i='i')
sp.robustness_report(r)   # prints the full layered check
```

Returns a dict with:
```
{
  'identification': {'pretrends_pvalue': 0.23, 'honest_did_M_breakdown': 0.34, ...},
  'specification':  {'spec_curve_sign_stability': 0.95, ...},
  'sensitivity':    {'evalue': 2.1, 'oster_upper': 0.42, 'oster_lower': 0.08, ...},
  'verdict':        'ROBUST',  # one of: ROBUST | MARGINAL | FRAGILE
}
```

## Reporting template

For the final paper, present:

1. Main table: point estimate, SE, 95% CI, N.
2. Identification check: one test per identifying assumption
   (pre-trend plot, McCrary plot, first-stage F).
3. Specification panel: point estimate varying key design choices
   (covariate sets, bandwidths, subsamples).
4. Sensitivity panel: E-value, Oster bounds, sensemakr contour.

`sp.regtable()` auto-formats this for LaTeX/Word:
```python
sp.regtable([r1, r2, r3],
            filename='table1.tex',
            robustness=True)
```
