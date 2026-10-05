# When the interval can be trusted

Reference parity says StatsPAI reproduces Stata or R. It does not say the
95% interval covers 95% of the time on your data. This page collects what
the simulation studies under `tests/reliability/` found about that, as
rules you can act on. Each study fixed its design before the first run,
stores every cell, and has a test that recomputes part of it. The numbers
are in `docs/evidence_inventory.md`; the tables and their reading are in
`tests/reliability/README.md`.

A rule here is what one set of designs showed. Treat it as a reason to
look, and check the result's own warnings and `model_info` for the
diagnostic it names.

## Clusters

The count of clusters is not what matters. Two other numbers are, and
StatsPAI records both.

`n_clusters_effective` discounts unequal sizes. With 40 clusters of which
one holds half the sample, a cluster-robust t-test rejected a true null
36% of the time in a pooled regression and 22% with unit fixed effects.
In the pooled designs `vce='cr3'` stayed at 3% to 5%.

`n_clusters_effective_weights` discounts unequal weights. With unit-level
weights, the interval of a clustered fixed-effects regression covered 94%
with 82 clusters in effect, 91% with 23, 87% with 17 and 78% with 7. The
panels had 50 or 200 equal-sized clusters throughout.

```python
import statspai as sp

fit = sp.panel(df, "y ~ x", entity="id", time="t", method="fe",
               weights="w", cluster="id")
fit.model_info["n_clusters_effective"]          # by size
fit.model_info["n_clusters_effective_weights"]  # by weight
```

A wild cluster bootstrap helps when clusters are few and similar in size.
It does not help when only one or two clusters are treated, where it
almost never rejects.

### Few treated clusters

What can be few is the number of clusters that are ever treated, however
many clusters there are. With 40 units and a true effect of zero, a
unit-clustered two-way fixed-effects test rejected 75% of the time with
one treated unit, 31% with two, 10% with five and 6% with ten. StatsPAI
records such a regressor in `few_treated_clusters` and warns.

`sp.did_few_treated` is the remedy for one or two treated units, where
it rejected 3% and 7%. It is not a remedy beyond that. With five treated
units it rejected 10% and with ten 14%. Between three and nine treated
units no test here is at its nominal size, and the honest report says
so.

## Weights

Passing `weights=` and nothing else gives the classical weighted variance.
That is right when the weights are precisions (linear models) or
frequencies (Poisson). It is wrong for sampling or inverse-probability
weights. Under dispersed sampling weights the default interval covered
34% to 78% for OLS, 45% to 80% with fixed effects and 14% to 68% for
Poisson. The fast entry points, `sp.nbreg`, `sp.logit`, `sp.probit`,
`sp.glm` and `sp.iv` behave the same way (15% to 79%).

Ask for a robust variance when the weights are sampling weights.

```python
sp.regress("y ~ x", df, weights="w", robust="hc1")      # Stata [pw=w]
sp.poisson("y ~ x", df, weights="w", robust="robust")
```

A robust variance is itself short when a few observations carry most of
the weight. `n_effective_weights` is the Kish effective sample size.
Below about 100, HC1 covered 84% to 88% for OLS and Poisson alike. For
OLS `vce='hc3'` held 93% to 94% in the same designs.

## Unbalanced panels in staggered DiD

Estimators that compare a unit with itself are robust to cells that go
missing at random and to units that leave according to their level. The
default of `sp.callaway_santanna`, `sp.did_imputation` and two-way fixed
effects all covered 94% to 97% in both cases.

`sp.callaway_santanna(allow_unbalanced_panel=True)` keeps every observed
row by comparing group means. That needs each group's composition to be
stable over time. When treated units with a high level left the panel,
the estimate was biased by 40% of the effect and the interval covered
50% with 100 units and 3% with 400. Use it when cells are missing for
reasons unrelated to the unit, and keep the default otherwise.

Nothing repairs attrition on the outcome itself. When treated cells with
a low outcome went missing, every estimator was biased by 38% to 50% of
the effect.

## A discrete running variable in RD

`sp.rdrobust` held its coverage down to 20 support points on each side of
the cutoff. At 10 and 5 it covered 89% with 1,000 observations and 69%
with 4,000, because more data narrows the interval around a biased
value. `sp.rd_discrete` and `sp.rdrandinf` are built for that case.

Do not cluster on the support points of the running variable. That made
coverage worse at every level, 86% at 50 points a side and 22% at 5.

## The learner in DML

No first-stage learner was right under both kinds of confounding.

| confounding | OLS, Lasso | random forest | gradient boosting (default) | model averaging |
| --- | --- | --- | --- | --- |
| linear | 94% to 95% | 91% to 93% | 90% to 92% | 94% to 95% |
| nonlinear | 0% | 43% to 67% | 87% to 90% | 73% to 87% |

The reported standard error matched the spread of the estimates in every
design. The intervals failed because of bias in the nuisance fits, which
a standard error does not carry. The available check is to refit with a
different learner and see whether the estimate moves by more than its
standard error.

```python
a = sp.dml(df, y="y", treat="d", covariates=xs, model="plr")
b = sp.dml_model_averaging(df, y="y", treat="d", covariates=xs)
abs(a.estimate - b.estimate) / a.se
```

## Weak instruments and poor overlap

These two come from the Track B stress designs rather than the studies
above. 2SLS with a first-stage F near 3 covered 88%; use
`sp.anderson_rubin_ci` there. A causal forest with propensity scores
near 0 or 1 covered 90% on 300 replications; look at `sp.overlap_plot`
and trim before fitting.
