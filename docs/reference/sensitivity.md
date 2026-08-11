# Sensitivity Analysis

Tools for probing the robustness of causal estimates to unobserved
confounding, model specification, and sampling variation.

## Unobserved confounding

```python
# Oster (2019) — coefficient stability and δ-bound
sp.oster_bounds(df, y='y', treat='d', controls=[...], r_max=1.3 * r2_long)

# Cinelli & Hazlett (2020) sensemakr — RV, extreme scenarios, benchmarks
sp.sensemakr(df, y='y', treat='d', controls=['educ', 'experience', 'age'],
             benchmark=['educ', 'experience'])

# VanderWeele & Ding (2017) E-values
sp.evalue(estimate=1.8, ci=(1.3, 2.5), measure='RR')

# Rosenbaum (2002) bounds — matched pairs
sp.rosenbaum_bounds(paired_diff, gamma_grid=[1.1, 1.25, 1.5, 2.0])

# Manski (1990) worst-case bounds
sp.manski_bounds(df, y='y', treat='d', y_lower=0.0, y_upper=1.0)
```

## Specification curve analysis

```python
# Simonsohn-Simmons-Nelson (2020) spec curve
sc = sp.spec_curve(
    df, y='wage', x='union',
    controls=[[], ['age'], ['age', 'edu']],
    subsets={'full': None, 'male': 'sex == 1'},
    se_types=['hc3', 'cluster'], cluster_var='state',
)
sc.plot(kind='curve')               # full specification universe
sc.median_effect                    # across specs
sc.share_positive_significant       # share of specs with p<0.05 & +sign
```

## One-call robustness report

```python
report = sp.robustness_report(
    df, formula='y ~ d + x1 + x2', x='d',
    cluster_var='state',
    extra_controls=['x3'], drop_controls=['x2'],
    winsor_levels=[0.01, 0.05],
)
report.summary(); report.plot(); report.to_latex()
```

## Honest parallel-trends (DID)

```python
# Rambachan & Roth (2023) — relative-magnitude and smoothness restrictions
sp.honest_did(cs_result, method='relative_magnitude', m_grid=[0.5, 1, 1.5, 2])
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
v = sp.verify(rec, df, B=500, K_subsample=20)
v.verify_score                      # 0–100 stability score
v.components                        # bootstrap / placebo / subsample breakdown
```
