# Causal ML: Forests, Meta-Learners, TMLE, Neural

Machine-learning-based heterogeneous treatment effect estimators.

## Causal forests

```python
r = sp.causal_forest(df, y='y', treat='d', covariates=[...],
                     n_estimators=2000,
                     honest=True,
                     min_samples_leaf=5)
r.cate(new_X)                  # conditional ATE for new units
r.variable_importance()
r.subgroup_test(by='age_bin')  # test heterogeneity
```

## Meta-learners

| Learner | Description | Reference |
| --- | --- | --- |
| `sp.metalearner(..., learner='s')` | Single-learner on `(X, D)` | Künzel et al. (2019) |
| `sp.metalearner(..., learner='t')` | Two-learner — separate treated/control | Künzel et al. (2019) |
| `sp.metalearner(..., learner='x')` | Cross-learner combining S and T | Künzel et al. (2019) |
| `sp.metalearner(..., learner='r')` | Residualised (Robinson-style) | Nie & Wager (2021) |
| `sp.metalearner(..., learner='dr')` | Doubly-robust — outcome + propensity | Kennedy (2023) |

```python
r = sp.metalearner(df, y='y', treat='d', covariates=[...], learner='x',
                   outcome_model=RandomForestRegressor())
r.cate(new_X)
```

Plus `sp.cate_eval(...)` for overlap, calibration, and CATE QQ plots.

## TMLE

```python
r = sp.tmle(df, y='y', treat='d', covariates=[...],
            outcome_library=['rf', 'lasso'],     # Super Learner ensemble
            propensity_library=['rf', 'logistic'])
r.ate, r.ci
```

## Neural causal models

```python
sp.tarnet(df, y='y', treat='d', covariates=[...],
          epochs=100, repr_layers=[200, 100])    # Shalit, Johansson, Sontag (2017)
sp.cfrnet(df, y='y', treat='d', covariates=[...],  # Counterfactual Regression Net
          ipm_weight=1.0)
sp.dragonnet(df, y='y', treat='d', covariates=[...],  # Shi, Blei, Veitch (2019)
             targeted_reg_weight=1.0)
sp.deepiv(df, y='y', treat='d', instruments='z',    # Hartford et al. (2017)
          covariates=[...])
```

## Causal discovery

```python
sp.notears(df, w_threshold=0.3, lambda1=0.1)        # Zheng et al. 2018
sp.pc_algorithm(df, alpha=0.05)                     # Spirtes-Glymour-Scheines
sp.lingam(df)                                       # Shimizu 2006
sp.ges(df)                                          # Chickering 2002
```

## Policy learning

```python
sp.policy_tree(df, y='y', treat='d', covariates=[...],
               depth=3)                              # Athey-Wager 2021
sp.policy_value(tree, df_test)
```

## Bayesian causal forests

```python
sp.bcf(df, y='y', treat='d', covariates=[...],
       n_trees_mu=200, n_trees_tau=50)               # Hahn, Murray, Carvalho 2020
```

## Conformal causal inference + matrix completion

```python
sp.conformal_cate(df, ...)                          # distribution-free CATE intervals
sp.mc_panel(df, ...)                                # Athey et al. 2021 MC-NNM
```
