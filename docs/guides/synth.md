# Synthetic Control — 20 methods, one package

StatsPAI bundles the most comprehensive synthetic control toolkit
available in any language: 20 estimators, 6 inference strategies, plus
a research workflow of automatic comparison, power analysis,
sensitivity diagnostics, and one-click reports.

## Quick start

```python
import statspai as sp

# Load the California Proposition 99 dataset (Abadie et al. 2010 style)
df = sp.synth.california_tobacco()

# Classic SCM (Abadie, Diamond & Hainmueller 2010)
res = sp.synth(
    df, outcome='cigsale', unit='state', time='year',
    treated_unit='California', treatment_time=1989,
    method='classic',
)
print(res.summary())
```

`sp.synth()` is the unified dispatcher — switch `method=` to run any
of the 20 variants with the same API and return type.

## The 20 methods

| `method=` | Paper | When to use |
|---|---|---|
| `classic` | Abadie, Diamond & Hainmueller (2010) | Baseline. One treated unit, few pre-periods, good donor fit. |
| `penalized` / `ridge` | Doudchenko & Imbens (2016) | When classic weights are unstable; ridge improves extrapolation. |
| `demeaned` | Ferman & Pinto (2021) | When unit fixed effects are large; demean before matching. |
| `unconstrained` / `elastic_net` | Doudchenko & Imbens (2016) | When negative/unbounded weights are theoretically justified. |
| `augmented` / `ascm` | Ben-Michael, Feller & Rothstein (2021) | When no exact pre-treatment fit exists; bias-corrects classic SCM. |
| `sdid` | Arkhangelsky et al. (2021) | Combines DID & SCM; robust to effect heterogeneity. |
| `factor` / `gsynth` | Xu (2017) | Multiple treated units + interactive fixed effects. |
| `staggered` | Ben-Michael, Feller & Rothstein (2022) | Heterogeneous adoption times. |
| `mc` | Athey et al. (2021) | Many missing cells; matrix completion via nuclear norm. |
| `discos` | Gunsilius (2023) | When treatment affects the whole distribution, not just the mean. |
| `multi_outcome` | Sun (2023) | Multiple correlated outcomes; borrows strength across. |
| `scpi` | Cattaneo, Feng & Titiunik (2021) | Need honest prediction intervals, not placebo p-values. |
| `bayesian` | Vives & Martinez (2024) | Want full posterior; Dirichlet prior + MCMC credible intervals. |
| `bsts` / `causal_impact` | Brodersen et al. (2015) | Structural TS with level/trend; the Google CausalImpact model. |
| `penscm` | Abadie & L'Hour (2021) | Many similar donors; pairwise discrepancy penalty ensures interpretability. |
| `fdid` | Li (2024) | Want a *subset* of donors, not weights — simple to communicate. |
| `cluster` | Rho (2024) | Large donor pool; cluster first, then SCM within best cluster. |
| `sparse` / `lasso` | Amjad, Shah & Shen (2018) | High-dimensional donors; L1 picks a handful. |
| `kernel` | RKHS / MMD matching | Nonlinear donor relationships. |
| `kernel_ridge` | Kernel ridge regression | Nonlinear, unconstrained, regularised. |

## Inference strategies

| Strategy | Call | When |
|---|---|---|
| Placebo (in-space) | default | Always available; exchangeable donor pool. |
| Conformal | `method='conformal'` or `conformal_synth()` | Want finite-sample coverage guarantees (Chernozhukov et al. 2021). |
| Prediction intervals | `scpi()` | Account for parameter + residual + design uncertainty. |
| Bootstrap / jackknife | SDID internal | Inference for SDID. |
| Bayesian posterior | `bayesian_synth()` | Full credible intervals from MCMC. |
| BSTS posterior | `bsts_synth()` | Kalman filter / smoother posterior draws. |

## Research workflow

### Run all methods at once

```python
comp = sp.synth_compare(df, outcome='cigsale', unit='state', time='year',
                        treated_unit='California', treatment_time=1989)
comp.plot()                # overlay all counterfactuals
comp.summary_table()       # ATT, pre-RMSPE, p-value per method
best = sp.synth_recommend(comp)   # auto-pick best by pre-fit + robustness
```

### Power analysis before you commit

```python
power = sp.synth_power(df, outcome='cigsale', unit='state', time='year',
                       treated_unit='California', treatment_time=1989,
                       effect_sizes=[2, 5, 10, 15, 20], n_sims=500)
sp.synth_power_plot(power)

# What's the minimum detectable effect at 80% power?
mde = sp.synth_mde(df, outcome='cigsale', ..., target_power=0.80)
```

### Sensitivity suite

```python
sens = sp.synth_sensitivity(res)            # one-shot: LOO + time + donor + RMSPE
sp.synth_sensitivity_plot(sens)

# or individual components
sp.synth_loo(res)                # leave-one-donor-out
sp.synth_time_placebo(res)       # backdate the treatment time
sp.synth_donor_sensitivity(res)  # random donor-pool subsampling
sp.synth_rmspe_filter(res)       # drop high-residual placebos
```

### One-click publication report

```python
sp.synth_report(res, format='markdown')   # 'text' | 'markdown' | 'latex'
sp.synth_report_to_file(res, 'analysis.md')
```

Produces: abstract, method, weights, placebo distribution, sensitivity,
references — directly pasteable into a paper or appendix.

## Canonical datasets

StatsPAI ships the three canonical SCM datasets as realistic simulations
(same structure as the published data; use for teaching and unit tests):

```python
sp.synth.california_tobacco()      # Prop 99 (1989)
sp.synth.german_reunification()    # Abadie, Diamond & Hainmueller (2015)
sp.synth.basque_terrorism()        # Abadie & Gardeazabal (2003)
```

For real published data, use the `causaldata` R package or the authors'
replication files — StatsPAI does not redistribute copyrighted data.

## Choosing a method

If you are unsure, run `sp.synth_compare(...)` and inspect:

1. **Pre-RMSPE** — lower is better fit.
2. **Consistency across methods** — if classic/ASCM/SDID/MC agree, the
   answer is robust. If they disagree, inspect why.
3. **Placebo distribution** — is the treated-unit effect extreme relative
   to placebos?

Then run `sp.synth_sensitivity(best)` to stress-test the winner.

## References

- Abadie, Diamond & Hainmueller (2010). *JASA* 105(490).
- Abadie & L'Hour (2021). *JASA* 116(536).
- Amjad, Shah & Shen (2018). *JMLR* 19(22).
- Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021). *AER* 111(12).
- Athey, Bayati, Doudchenko, Imbens & Khosravi (2021). *JASA* 116(536).
- Ben-Michael, Feller & Rothstein (2021). *JASA* 116(536).
- Brodersen, Gallusser, Koehler, Remy & Scott (2015). *AOAS* 9(1).
- Cattaneo, Feng & Titiunik (2021). *JASA* 116(536).
- Chernozhukov, Wüthrich & Zhu (2021). *JASA* 116(536).
- Doudchenko & Imbens (2016). NBER WP 22791.
- Ferman & Pinto (2021). *Quantitative Economics* 12(4).
- Gunsilius (2023). *Econometrica* 91(3).
- Li (2024). Forward Difference-in-Differences.
- Rho (2024). Cluster Synthetic Control Methods.
- Sun (2023). *ReStat* forthcoming.
- Vives & Martinez (2024). *JCGS* forthcoming.
- Xu (2017). *Political Analysis* 25(1).
