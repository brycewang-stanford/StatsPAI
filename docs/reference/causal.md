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
| `sp.s_learner` | Single-learner on `(X, D)` | Künzel et al. (2019) |
| `sp.t_learner` | Two-learner — separate treated/control | Künzel et al. (2019) |
| `sp.x_learner` | Cross-learner combining S and T | Künzel et al. (2019) |
| `sp.r_learner` | Residualised (Robinson-style) | Nie & Wager (2021) |
| `sp.dr_learner` | Doubly-robust — combines outcome and propensity | Kennedy (2023) |

```python
r = sp.x_learner(df, y='y', treat='d', covariates=[...],
                 ml_model='rf')
r.cate(new_X)
```

Plus `sp.cate_diagnostics(r)` for overlap, calibration, and CATE QQ plots.

## TMLE

```python
r = sp.tmle(df, y='y', treat='d', covariates=[...],
            sl_library=['rf','lasso','xgb'])    # Super Learner ensemble
r.ate, r.ci
```

## Neural causal models

```python
sp.tarnet(df, y='y', treat='d', covariates=[...],
          epochs=100, hidden=[200,100])          # Shalit, Johansson, Sontag (2017)
sp.cfrnet(df, ..., imbalance='wass',            # Counterfactual Regression Net
          alpha=1.0)
sp.dragonnet(df, ..., targeted_regularization=True)  # Shi, Blei, Veitch (2019)
sp.deepiv(df, y='y', treat='d', instrument='z',     # Hartford et al. (2017)
          covariates=[...])
```

## Causal discovery

```python
sp.notears(df, threshold=0.3, lambda1=0.1)          # Zheng et al. 2018
sp.pc_algorithm(df, alpha=0.05)                     # Spirtes-Glymour-Scheines
sp.lingam(df)                                       # Shimizu 2006
sp.ges(df, score='bic')                             # Chickering 2002
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
       n_mcmc=2000, n_burn=1000)                     # Hahn, Murray, Carvalho 2020
```

## Conformal causal inference + matrix completion

```python
sp.conformal_cate(df, ...)                          # distribution-free CATE intervals
sp.mc_nnm(df, ...)                                  # Athey et al. 2021 MC-NNM
```
