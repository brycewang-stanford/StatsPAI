# Synthetic Control

`statspai.synth` — the most comprehensive SCM toolkit across Stata, R,
and Python (v0.9.0, 20 estimators + 6 inference strategies + full
research workflow).

## Unified dispatcher — `sp.synth`

```python
r = sp.synth(
    df, outcome='y', unit='state', time='year',
    treated_unit='California', treatment_time=1989,
    method='scm',           # 'scm' | 'sdid' | 'ascm' | 'bayesian' | 'bsts' |
                            # 'penscm' | 'fdid' | 'cluster' | 'sparse' |
                            # 'kernel' | 'kernel_ridge' | 'staggered' | …
    placebo=True,           # in-space placebos
    inference='placebo',    # 'placebo' | 'bootstrap' | 'jackknife' |
                            # 'conformal' | 'posterior' | 'agnostic'
)
r.summary(); r.plot(); r.to_latex()
```

## 20 estimators

| Family | Estimator | Reference |
| --- | --- | --- |
| Classical | `synth` | Abadie, Diamond & Hainmueller (2010, JASA) |
| Doubly robust | `sdid` | Arkhangelsky et al. (2021, AER) |
| Augmented | `ascm` | Ben-Michael, Feller & Rothstein (2021) |
| Bayesian | `bayesian_synth` | Dirichlet-prior MCMC |
| State space | `bsts_synth`, `causal_impact` | Brodersen et al. (2015) |
| Penalised | `penscm` | Abadie & L'Hour (2021) |
| Forward DID | `fdid` | Li (2023) |
| Cluster | `cluster_synth` | For large donor pools |
| Sparse | `sparse_synth` | LASSO-style weight selection |
| Kernel | `kernel_synth`, `kernel_ridge_synth` | Hazlett & Xu (2018) |
| Staggered | `staggered_synth` | Ben-Michael et al. (2022) |
| Multi-outcome | `multi_outcome_synth` | Athey et al. (2021) |
| Matrix completion | `mc_synth` | Athey et al. (2021, MC-NNM) |
| Generalised | `gsynth` | Xu (2017, Interactive FE) |
| Demeaned | `demeaned_synth` | Doudchenko & Imbens (2016) |
| Distributional | `discos` | Gunsilius (2023) |

## Research workflow

```python
# Compare all 20 on the same data
cmp = sp.synth_compare(df, ..., methods='all')
cmp.plot_forest()                               # ATT + CI across methods

# Auto-select based on fit + sensitivity
rec = sp.synth_recommend(df, ...)
rec.recommended_method, rec.reasons

# Power / MDE
p = sp.synth_power(df, ..., alphas=[0.05], effects=[1, 2, 3, 5])
mde = sp.synth_mde(df, ..., alpha=0.05, power=0.80)

# Sensitivity: leave-one-out donor, time placebo, pre-period length
s = sp.synth_sensitivity(df, ..., n_boot=1000)

# Report
sp.synth_report(rec, format='latex', save_to='report.tex')
```

## Canonical datasets

```python
df = sp.california_tobacco()        # Abadie-Diamond-Hainmueller 2010
df = sp.german_reunification()      # Abadie, Diamond & Hainmueller 2015
df = sp.basque_terrorism()          # Abadie & Gardeazabal 2003
```
