# Smart Workflow

`statspai.smart` — estimator recommendation, comparison, assumption
auditing, and posterior verification.

## `sp.recommend`

```python
rec = sp.recommend(
    df,
    outcome='earnings',
    treatment='training',
    covariates=['age', 'educ', 'prior_earnings'],
    design='observational',          # 'rct' | 'observational' | 'did' | 'rd' | 'iv' | 'synth'
    verify=True,                     # run posterior verification (v0.9.3)
)
rec.summary()                        # ranked estimators with rationale
rec.recommended_method
rec.plot('verify_radar')             # stability breakdown per method
rec.to_latex()
```

## `sp.compare_estimators`

Run multiple estimators on the same data and show a coefficient-
stability forest:

```python
cmp = sp.compare_estimators(
    df, outcome='y', treatment='d', covariates=[...],
    methods=['ols', 'psm', 'dml', 'aipw', 'tmle', 'causal_forest'],
)
cmp.plot_forest()
cmp.table()
```

## `sp.assumption_audit`

One-call audit of the most common identification assumptions:

```python
audit = sp.assumption_audit(df, outcome='y', treatment='d', covariates=[...])
audit.overlap                        # propensity score overlap diagnostic
audit.covariate_balance              # Love plot of standardised diffs
audit.placebo_outcomes               # pre-treatment placebos
audit.instrument_strength            # first-stage F if IV specified
audit.parallel_trends                # pre-trend placebo if DID
audit.summary()
```

## `sp.verify` / `sp.verify_benchmark` (v0.9.3)

Posterior verification of any `sp.recommend()` output — aggregates
three signals into a `verify_score ∈ [0, 100]`:

```python
v = sp.verify(
    rec,
    n_boot=500,
    n_subsample=100,
    subsample_frac=0.8,
    n_placebo=20,
)

v.verify_score                       # 0–100 composite
v.components                         # dict: bootstrap / placebo / subsample
v.failures                           # methods that failed verification
v.plot('radar')                      # visual per-method breakdown
```

Calibration card: top-method `verify_score` is typically 85–95 on
clean DGPs (RD lower at ≈ 74 due to local-polynomial bootstrap
variance). `sp.verify_benchmark(...)` runs verify against synthetic
DGPs to calibrate what threshold constitutes "trust it".
