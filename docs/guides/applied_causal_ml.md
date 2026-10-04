# *Applied Causal Inference Powered by ML and AI* in StatsPAI

Chernozhukov, Hansen, Kallus, Spindler and Syrgkanis wrote a textbook on
causal inference with machine learning [@chernozhukov2024applied]. It comes
with Python and R notebooks for every chapter. The Python notebooks were
rerun with StatsPAI in place of their hand-written estimators, next to the
reference packages the book itself uses (R `hdm`, `sensemakr`, `dagitty`,
`rdrobust`, and Python `DoubleML`). This guide records the result.

It has three parts.

1. A chapter map from what the notebooks build by hand to the StatsPAI call.
2. Notes on where the notebooks have aged.
3. What the comparison changed in StatsPAI.

The book's data are not redistributed with StatsPAI. The tests that pin the
fixes use simulated data.

## Chapter map

| Notebook group | What the notebook does | StatsPAI |
| --- | --- | --- |
| PM1 Linear regression, wage gap | OLS with HC3, partialling out, Oaxaca-Blinder | `sp.regress(robust='hc3')`, `sp.oaxaca` |
| PM1, PM2 Double lasso | `hdmpy.rlasso` wrapped in a class, residual-on-residual OLS | `sp.rlasso`, `sp.rlasso_effect(method='partialling out')` |
| PM2 Heterogeneous wage effects | one double lasso per interaction, covariance of the scores, joint band | `sp.rlasso_effects(...).conf_int(joint=True)` |
| PM3 ML prediction | scikit-learn learners | any scikit-learn estimator, passed to `sp.dml` |
| PM4 DML, 401(k), growth, guns | hand-written cross-fitting for PLR and IRM | `sp.dml(model='plr')`, `sp.dml(model='irm')`, `cluster=` |
| PM4, CM3 DAGs | `pgmpy`, `dagitty`, `dosearch` | `sp.dag`, `DAG.adjustment_sets`, `DAG.implied_independencies`, `sp.identify` |
| CM1 Experiments | two-sample, regression adjustment, interacted adjustment (Lin), Holm, risk ratio | `sp.difference_in_means`, `sp.regress`, `sp.lm_lin(superpopulation=True)`, `sp.holm`, `sp.relative_risk` |
| AC1 Sensitivity | `sensemakr`, DML bounds with standard errors | `sp.sensemakr`, `sp.dml_sensitivity` |
| AC1 Proxy controls | residualise, then just-identified IV | `sp.proximal` |
| AC2 DML with instruments | PLIV, interactive IV (LATE), Anderson-Rubin on a grid | `sp.dml(model='pliv')`, `sp.dml(model='iivm')`, `model_info['anderson_rubin']` |
| T CATE | doubly-robust scores, best linear predictor, group effects, forests, policy trees | `sp.best_linear_projection`, `sp.causal_forest`, `sp.metalearner`, `sp.cate_eval`, `sp.policy_tree` |
| T Difference-in-differences | ATT score on first differences with ML nuisances | `sp.dml(model='irm', score='ATTE')` on the differenced outcome, `sp.callaway_santanna` |
| T Regression discontinuity | `rdrobust`, covariates, ML adjustment | `sp.rdrobust(covs=)`, `sp.rd_flex` |

### Experiments

```python
fit = sp.lm_lin(df, "y", "treat", ["age", "female", "region"],
                superpopulation=True)
fit.estimate, fit.se
```

The regression includes every covariate, centred, and its interaction
with treatment. The default variance is that of R `estimatr::lm_lin`.
`superpopulation=True` adds the term for the estimated covariate means,
as the notebook does by hand.

### Double lasso

```python
import statspai as sp

fit = sp.rlasso_effect(W, y, d, method="partialling out", post=False)
fit.alpha, fit.se
```

On the book's wage and growth data the four combinations of method and
`post` agree with `hdm::rlassoEffect` to twelve digits. For many targets,
`sp.rlasso_effects` returns one result per column and a joint band:

```python
effects = sp.rlasso_effects(X, y, index=[0, 1, 2])
effects.conf_int(joint=True)   # all intervals cover at once
effects.vcov()                 # joint covariance of the estimates
```

### Double machine learning

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

fit = sp.dml(
    df, y="y", treat="d", covariates=covariates, model="plr",
    ml_g=RandomForestRegressor(min_samples_leaf=20),
    ml_m=RandomForestClassifier(min_samples_leaf=20),
)
```

A classifier may be used for a 0/1 treatment. Its predicted probability is
the nuisance, as in the book and in `DoubleML`. Pass `fold_indices=` to
reproduce a notebook's `KFold` split exactly, and `cluster=` for the
clustered standard errors of the Darfur and gun examples.

### Sensitivity to an unobserved confounder

```python
sens = sp.dml_sensitivity(fit, cf_y=0.04, cf_d=0.03)
sens.adjusted_estimate_low, sens.adjusted_estimate_high
sens.se_low, sens.se_high        # the bounds are estimates too
sens.ci_low, sens.ci_high
```

`sp.sensemakr` covers the regression version. Controls may be strings, and
`kd=[1, 2, 3]` asks about a confounder one, two and three times as strong
as the benchmark:

```python
out = sp.sensemakr(
    df, y="y", treat="d", controls=["x1", "x2", "region"],
    benchmark=["x2"], kd=[1, 2, 3],
)
out["benchmark_table"]
```

### Weak instruments

`sp.dml(model='pliv')` attaches a confidence set that stays valid when the
instrument is weak.

```python
iv = sp.dml(df, y="y", treat="d", instrument="z", covariates=covariates,
            model="pliv")
iv.model_info["anderson_rubin"]["intervals"]
```

The set is an interval when the first stage is significant at the chosen
level. Otherwise it is the whole line or the complement of an interval,
and that is the honest answer.

### Graphs

```python
g = sp.dag("D -> Y; X -> D; X -> Y; F -> X; F -> D", latent=["F"])
g.adjustment_sets("D", "Y")      # [{'X'}]; F is never offered
g.implied_independencies()       # what the graph says the data must satisfy
g.test_implications(df)          # partial correlations, Fisher z, Holm
sp.identify(g, "D", "Y").estimand
```

### Heterogeneous effects from a DML fit

```python
irm = sp.dml(df, y="y", treat="d", covariates=covariates, model="irm")
sp.best_linear_projection(irm, A=df[["x1"]])
```

The table is the best linear predictor of the conditional effect given
`A`. With group indicators in `A` it gives group average effects.

The R notebook on conditional effects plots the effect as a curve in one
covariate. Project on a basis in that covariate and use the covariance
of the coefficients:

```python
import numpy as np

A = df[["x1"]].assign(x1_sq=df["x1"] ** 2)
table = sp.best_linear_projection(irm, A=A)
grid = np.linspace(-2, 2, 41)
basis = np.column_stack([np.ones(41), grid, grid ** 2])
curve = basis @ table["coef"].to_numpy()
V = table.attrs["vcov"].to_numpy()
se = np.sqrt(np.einsum("ij,jk,ik->i", basis, V, basis))
```

The curve and its standard error equal `DoubleML`'s
`cate(basis).confint(grid_basis)`.

To judge a CATE estimate on held-out data, `sp.cate_eval` gives AUTOC,
Qini and the TOC curve with a band that covers the whole curve:

```python
ev = sp.cate_eval(cate_hat, Y, T, X=X)
ev.autoc, ev.autoc_ci
ev.toc_curve[["q", "toc", "band_lower", "band_upper"]]
```

The notebook takes the ranking thresholds from a separate sample.
`sp.cate_eval` ranks within the evaluation sample, as grf does, and its
standard errors account for that.

## Where the notebooks have aged

- **`hdmpy`** is cloned from GitHub in most notebooks. `sp.rlasso` is a
  port of `hdm` that is pinned against R and needs no clone.
- **`DoubleMLDID(trimming_threshold=)`** in the difference-in-differences
  notebook no longer runs on current `DoubleML`. The same ATT score is
  `sp.dml(model='irm', score='ATTE')` on the differenced outcome, which
  equals `DoubleMLIRM(score='ATTE')` to machine precision.
- **The Anderson-Rubin grid** in the weak-instrument notebook stops at 2,
  so the reported upper end of the set is the end of the grid.
- **`scikit-learn < 1.3`, `tensorflow < 2.16`** in `requirements.txt` are
  not needed for anything above.

## What the comparison changed in StatsPAI

Two estimates were wrong and are fixed. `sp.dml` with PLR or PLIV used a
classifier's predicted label in place of its predicted probability. And
`sp.dml_sensitivity` applied the PLR scaling to IRM fits. Both are
described in `MIGRATION.md` under `oct2026-causalml-textbook-fixes`.

One difference from a reference package is kept on purpose. `DoubleML`
0.11.3 reports the standard errors of the lower and the upper sensitivity
bound exchanged. StatsPAI attaches each to its own bound. The evidence is
a finite-difference check of the influence functions, and the book's own
function `dml_sensitivity_bounds` uses the same assignment.

The additions are listed in the changelog. They are the joint band for
`sp.rlasso_effects`, the bound standard errors, the Anderson-Rubin set,
`sp.best_linear_projection` on DML fits, `sp.lm_lin`, factor controls and benchmark
multiples in `sp.sensemakr`, and named latents and testable implications
in `sp.dag`.
