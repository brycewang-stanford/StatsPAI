# Double / Debiased Machine Learning

`statspai.dml` — Chernozhukov et al. (2018) double/debiased ML with
cross-fitted nuisance estimation and Neyman-orthogonal scores.

## Four model types

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

# Interactive IV Model (IIVM) — new in v0.9.6
# Binary D, binary Z → LATE (compliers) via the Wald ratio of two
# Neyman-orthogonal doubly-robust scores. See Chernozhukov et al.
# (2018) §5.
r = sp.dml(df, y='earnings', treat='college',
           covariates=['age', 'ability', 'parent_edu'],
           model='iivm', instrument='lottery_win')
r.estimand    # 'LATE'
```

!!! note "Multiple excluded instruments"
    StatsPAI's PLIV and IIVM implementations use a **scalar**
    instrument. For a problem with multiple excluded instruments,
    build a scalar first-stage index with
    `sp.scalar_iv_projection(data, treat, instruments=[...], covariates=[...])`
    and pass the resulting column name to `instrument=`. See
    `docs/ROADMAP.md` §2 for native multi-instrument support.

## Per-model classes

Each model family is also available as a dedicated class for users
who want fine-grained control or want to subclass the base
infrastructure:

```python
from statspai.dml import DoubleMLPLR, DoubleMLIRM, DoubleMLPLIV, DoubleMLIIVM

r = DoubleMLIIVM(data=df, y='y', treat='d', covariates=[...],
                 instrument='z').fit()
```

All four share `statspai.dml._DoubleMLBase`, which handles cross-
fitting, default learner selection (classifier vs regressor per
model), repeated-split aggregation, and `CausalResult` construction.

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
r = sp.dml(..., n_folds=5, n_rep=10)
```

With `n_rep > 1`, StatsPAI repeats the cross-fitting with different
random fold assignments and reports the **median** point estimate
together with the Chernozhukov et al. (2018, eq. 3.7) aggregated SE:

```text
σ̂² = median_r ( se_r² + (θ̂_r − θ̂_med)² )
```

which accounts for both within-rep nuisance variance and between-rep
dispersion of the point estimates — taking only `median(se_r)` would
under-cover whenever splits-randomness moves the point estimate.

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
- Chernozhukov, V. et al. (2018, §5). Interactive IV Model and the
  ratio-of-scores LATE estimator used by `model='iivm'`.
