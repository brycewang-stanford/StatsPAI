# Double / Debiased Machine Learning

`statspai.dml` — Chernozhukov et al. (2018) double/debiased ML with
cross-fitted nuisance estimation and Neyman-orthogonal scores.

## Three model types

```python
# Partially Linear Regression (PLR): Y = Dθ + g(X) + U, E[U|D,X]=0
r = sp.dml(df, y='wage', treat='training',
           covariates=['age','edu','exp'],
           model='plr',
           ml_g='rf', ml_m='rf',         # nuisance learners
           n_folds=5, n_rep=1)

# Interactive Regression Model (IRM): binary treatment with Y = g₀(D,X) + U
r = sp.dml(df, y='y', treat='d_bin', covariates=[...],
           model='irm', trim=0.01)

# Partially Linear IV (PLIV) — new in v0.9.3
# Y = Dθ + g(X) + U, D = m(X) + V, Z = r(X) + ε
r = sp.dml(df, y='y', treat='d', covariates=[...],
           instrument='z',
           model='pliv',
           ml_g='rf', ml_m='rf', ml_r='rf')
```

## Learners

Either pass the name of a built-in learner (`'rf'`, `'lasso'`,
`'elastic_net'`, `'xgb'`, `'nn'`) or any sklearn-compatible estimator:

```python
from sklearn.ensemble import GradientBoostingRegressor
r = sp.dml(..., ml_g=GradientBoostingRegressor(n_estimators=200),
                ml_m=GradientBoostingRegressor(n_estimators=200))
```

## Cross-fitting

```python
r = sp.dml(..., n_folds=5, n_rep=10,
           cross_fit='standard')   # 'standard' | 'repeated'
```

Repeated cross-fitting averages over `n_rep` random fold assignments
to eliminate the dependency on a single split.

## Result attributes

```python
r.coef                 # θ̂ point estimate
r.se                   # influence-function standard error
r.ci(alpha=0.05)       # confidence interval
r.influence_function   # per-observation scores
r.summary()            # full table with reference
r.cite()               # Chernozhukov et al. 2018 BibTeX
```

## References

- Chernozhukov, V. et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal* 21(1).
- Chernozhukov, V. et al. (2018, §4.2). Partially Linear IV and
  the Neyman-orthogonal score used by `model='pliv'`.
