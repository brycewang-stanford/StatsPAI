# Sensitivity Analysis

Tools for probing the robustness of causal estimates to unobserved
confounding, model specification, and sampling variation.

## Unobserved confounding

```python
# Oster (2019) — coefficient stability and δ-bound
sp.oster(model_short, model_long, rmax=1.3 * R2_long)

# Cinelli & Hazlett (2020) sensemakr — RV, extreme scenarios, benchmarks
sp.sensemakr(df, y='y', treat='d', covariates=[...],
             benchmark_covariates=['educ', 'experience'],
             kd=[1, 2, 3])

# VanderWeele & Ding (2017) E-values
sp.e_value(rr=1.8, rr_ci_lb=1.3)

# Rosenbaum (2002) bounds — matched pairs
sp.rosenbaum_bounds(paired_diff, gamma_grid=[1.1, 1.25, 1.5, 2.0])

# Manski (1990) worst-case bounds
sp.manski_bounds(df, y='y', treat='d', covariates=[...])
```

## Specification curve analysis

```python
# Simonsohn-Simmons-Nelson (2020) spec curve
sc = sp.spec_curve(
    df, y='wage', treat='union',
    covariate_grid={'age':[None,'age'], 'edu':[None,'edu','edu+edu²']},
    fe_grid=['none','year','year+state'],
    sample_grid={'full':None, 'male':'sex==1'},
    se_grid=['hc3','cluster:state'],
)
sc.plot(kind='curve')               # full specification universe
sc.median_effect                    # across specs
sc.share_positive_significant       # share of specs with p<0.05 & +sign
```

## One-call robustness report

```python
report = sp.robustness_report(
    base_result,
    checks=['alt_specs','alt_samples','alt_se','placebos','leave_one_out'],
)
report.summary(); report.plot(); report.to_latex()
```

## Honest parallel-trends (DID)

```python
# Rambachan & Roth (2023) — relative-magnitude and smoothness restrictions
sp.honest_did(cs_result, type='relative_magnitude', Mbar_grid=[0.5,1,1.5,2])
sp.breakdown_m(cs_result)           # smallest M̄ that nullifies the effect
```

## Frontier-specific

```python
r.lr_test_no_inefficiency()         # Kodde-Palm mixed χ̄² one-sided LR
r.efficiency_ci(alpha=0.05, B=500)  # parametric-bootstrap unit CIs
```

## Posterior verification (v0.9.3)

```python
rec = sp.recommend(df, ...)
v = sp.verify(rec, B=500, subsample_frac=0.8)
v.verify_score                      # 0–100 stability score
v.components                        # bootstrap / placebo / subsample breakdown
```
