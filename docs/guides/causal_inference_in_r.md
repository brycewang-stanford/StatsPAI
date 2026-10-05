# *Causal Inference in R*, in StatsPAI

Malcolm Barrett, Lucy D'Agostino McGowan and Travis Gerke's book
(<https://www.r-causal.org>) teaches causal inference as a workflow. Draw
the DAG, fit a propensity score, weight for a population you can name,
check what the weights did, fit the outcome model with an honest variance,
then ask how wrong you could be. It leans on three small R packages by the
authors (`propensity`, `halfmoon`, `tipr`) and on `MatchIt`, `WeightIt` and
`marginaleffects`.

This guide follows the same steps with `import statspai as sp`. Every
number StatsPAI returns below is checked against the R package the book
uses for that step
(`tests/reference_parity/test_barrett_causal_inference_in_r_parity.py`, and
on the book's own data in
`tests/external_parity/test_barrett_causal_inference_in_r.py`).

The book is still being written. Its chapters on mediation, longitudinal
data, survival, doubly robust estimation, machine learning, instruments and
difference-in-differences are placeholders. For those topics see
`sp.mediate`, `sp.msm` and `sp.gformula_ice_fn`, `sp.aipw` and `sp.tmle`,
`sp.dml`, `sp.ivreg`, and the [DiD guide](choosing_did_estimator.md).

## From the book's functions to StatsPAI

| the book | StatsPAI |
| --- | --- |
| `glm(x ~ z, family = binomial())` then `augment()` | `sp.propensity_score(df, 'x', ['z'])` |
| `wt_ate()`, `wt_att()`, `wt_atu()`, `wt_atm()`, `wt_ato()` | `sp.ps_weights(ps, treat, 'ATE' / 'ATT' / 'ATC' / 'ATM' / 'ATO')` |
| `wt_ate(stabilize = TRUE)` | `sp.ps_weights(..., stabilize=True)` |
| `ps_trunc(method = "pctl", lower, upper)` | `sp.ps_weights(..., truncate=(lower, upper), truncate_scale='quantile')` |
| `ps_trim(method = "adaptive")` | `sp.trimming(df, treatment=, covariates=)` |
| `ess()`, `plot_ess()` | `sp.ess(w)`, `sp.ess(w, by=treat)` |
| `check_balance()`, `tidy_smd()`, `geom_love()` | `sp.balance_diagnostics()`, `sp.love_plot()` |
| `bal_energy()` | `sp.energy_distance()` |
| `check_model_auc(.weights =)` | `sp.auc(treat, ps, weights=)` |
| `geom_mirror_histogram()` | `sp.overlap_plot()` |
| `lm(y ~ x, weights = w)` then `ipw(ps_mod, outcome_mod)` | `sp.ipw(df, y, treat, covariates, estimand=, se_method='sandwich')` |
| `matchit()` then `lm()` with `vcovCL(cluster = ~subclass)` | `sp.match(df, y, treat, covariates)` |
| `avg_comparisons(model, variables = "x")` | `sp.contrast(fit, data=df, variable='x')` with `C(x)` in the formula |
| `avg_comparisons(newdata = filter(df, x == 1))` | `sp.contrast(..., subset='x == 1')` |
| `comparison = "lnratioavg"`, `"lnoravg"`, `transform = exp` | `sp.contrast(..., effect='ratio' / 'odds_ratio')` |
| `avg_predictions(variables = list(x = 0))` | `sp.margins_at(fit, data=df, at={'x': [0]})` |
| `lmw::lmw()` | `sp.implied_weights()` |
| `mice()` and Rubin's rules | `sp.mice()`, `sp.mi_estimate()` |
| `tipr::adjust_coef()`, `adjust_coef_with_binary()`, `adjust_rr()` ... | `sp.confounder_adjust(measure=)` |
| `tipr::tip_coef()`, `tip_with_binary()`, `tip_rr()` ... | `sp.confounder_tip(measure=)` |
| `tipr::e_value()` | `sp.evalue()` |
| `dagify()`, `adjustmentSets()` | `sp.dag()`, `.adjustment_sets(minimal=)` |
| `equivalentDAGs()`, `equivalenceClass()` | `.equivalent_dags()`, `.equivalence_class()` |
| `impliedConditionalIndependencies()`, `localTests()` | `.implied_independencies()`, `.test_implications(data)` |

## The workflow on one data set

The examples use the simulated data of the parity test: a binary exposure
`t`, three confounders, a continuous outcome `y` whose effect grows with
`x1`, and a binary outcome `yb`.

```python
import numpy as np
import pandas as pd
import statspai as sp

df = pd.read_csv("tests/reference_parity/_fixtures/barrett_ps_workflow.csv")
X = ["x1", "x2", "b"]
```

### 1. Weights for a population you can name

```python
ps = sp.propensity_score(df, "t", X)
w = {e: sp.ps_weights(ps, df["t"], e) for e in ["ATE", "ATT", "ATC", "ATM", "ATO"]}
pd.DataFrame({e: [sp.ess(v), v.max()] for e, v in w.items()},
             index=["ESS", "largest weight"]).round(1)
```

```text
                  ATE    ATT    ATC    ATM    ATO
ESS             333.8  275.3  234.5  345.3  355.6
largest weight   13.0    8.1   12.0    1.0    0.9
```

Five sets of weights answer five questions. ATE weights speak for all 500
units and pay for it with a weight of 13 on someone. Overlap weights (ATO)
never exceed one and keep the largest effective sample, but they describe
the units whose treatment was most in doubt. The choice is about the
question first and precision second. Chapter 10 of the book is the best
short treatment of this.

### 2. What the weights did

```python
bal = sp.balance_diagnostics(df, "t", X, weights=w["ATE"], ps=ps)
bal.table[["smd_raw", "smd_weighted", "variance_ratio_weighted", "ks_stat_weighted"]]
```

```text
          smd_raw  smd_weighted  variance_ratio_weighted  ks_stat_weighted
x1          1.016         0.130                    1.091             0.075
x2         -0.623        -0.067                    0.912             0.096
b           0.107         0.013                    1.008             0.006
```

`bal.summary_stats` also has the effective sample size of each group
(151.2 treated, 187.5 controls) and the energy distance, a single number
for the whole joint distribution. Here it falls from 0.486 to 0.018.

One more check from the book. A propensity score should separate the
groups before weighting and not after:

```python
sp.auc(df["t"], ps)                      # 0.806
sp.auc(df["t"], ps, weights=w["ATE"])    # 0.526
```

A weighted AUC near one half says the weights balanced whatever the score
was built from. It says nothing about variables left out of the score. For
the same reason, do not choose a propensity model by its unweighted AUC. A
variable that predicts treatment perfectly and has no effect on the
outcome raises the AUC and ruins the weights.

Standardized differences come in more than one convention. StatsPAI
divides the weighted mean difference by the weighted spread.
`sd_denom='unweighted'` keeps the spread of the original sample, which is
what R's `cobalt` does.

### 3. The weighted outcome model

```python
for e in ["ATE", "ATT", "ATC", "ATM", "ATO"]:
    r = sp.ipw(df, "y", "t", X, estimand=e, se_method="sandwich")
    print(e, round(r.estimate, 3), round(r.se, 3))
```

```text
ATE 2.038 0.161
ATT 2.536 0.151
ATC 1.530 0.242
ATM 1.836 0.128
ATO 1.866 0.125
```

The estimates differ because the effect differs across units, which is the
point of naming the estimand. The standard error is the M-estimation
variance that stacks the propensity model with the weighted means, so it
knows the weights were estimated. The book gets there in three steps
(bootstrap, a sandwich that ignores the propensity model, then
`propensity::ipw`); `se_method='sandwich'` is the last of them. R's
function multiplies the variance by `n / (n - 1)` and StatsPAI, like
Stata, does not.

### 4. G-computation

Fit an outcome model, predict everyone under both exposures, average the
difference. With the exposure entered as `C(t)`, `sp.contrast` does the
cloning, the averaging and the delta-method standard error:

```python
fit = sp.regress("y ~ C(t)*x1 + x2 + b", data=df)
sp.contrast(fit, data=df, variable="t")                    # ATE  1.835 (0.125)
sp.contrast(fit, data=df, variable="t", subset="t == 1")   # ATT  2.300 (0.134)
sp.contrast(fit, data=df, variable="t", subset="t == 0")   # ATC  1.405 (0.137)
```

Pass the full data and `subset=`. A frame that holds only the treated has
no untreated level to contrast with, and `sp.contrast` will say so.

For a binary outcome the same call gives the marginal risk difference,
risk ratio and odds ratio:

```python
logit = sp.logit("yb ~ C(t) + x1 + x2 + b", data=df)
sp.contrast(logit, data=df, variable="t")                        # 0.166 [0.066, 0.265]
sp.contrast(logit, data=df, variable="t", effect="ratio")        # 1.428 [1.148, 1.776]
sp.contrast(logit, data=df, variable="t", effect="odds_ratio")   # 1.956 [1.300, 2.944]
```

The coefficient of the same logit, exponentiated, is 2.083. That is a
conditional odds ratio and it is not the marginal one (1.956), even with
no confounding at all. Odds ratios are not collapsible; chapter 11 of the
book walks through why.

### 5. What a regression is doing

`y ~ t + x1 + x2 + b` also estimates a weighted difference of means. The
weights depend only on the exposure and the covariates:

```python
iw = sp.implied_weights(df, "t", X)
sp.ess(iw), int((iw < 0).sum())     # 359.4, 26
```

Twenty-six units enter with a negative weight. The regression is
extrapolating for them. The implied weights can go through every
diagnostic of step 2 (`sp.balance_diagnostics(weights=iw)`).

### 6. How wrong could it be?

Take the lower confidence limit of the overlap estimate, 1.62, and ask
what unmeasured confounder would move it to zero:

```python
sp.confounder_tip(1.62, confounder_outcome_effect=[1, 2, 3])
```

A confounder worth 2 units of the outcome would need a mean 0.81 standard
deviations higher among the exposed. Or state the confounder and see the
estimate move:

```python
sp.confounder_adjust([1.866, 1.621], confounder_outcome_effect=2,
                     exposed_prev=0.5, unexposed_prev=0.2)
# effect_adjusted: 1.266 and 1.021
```

`measure='rr'`, `'or'` and `'hr'` do the same for ratios.
`rare_outcome=False` applies the square-root (odds ratio) or VanderWeele
(hazard ratio) conversion when the outcome is common. `sp.evalue` and
`sp.sensemakr` answer the same question without asking you to describe the
confounder.

### 7. The DAG is an assumption too

```python
g = sp.dag("emm -> wait; close -> wait; season -> wait; temp -> wait; temp -> emm")
g.adjustment_sets("emm", "wait")                  # [{'temp'}]
g.adjustment_sets("emm", "wait", minimal=False)   # all four valid sets
g.equivalence_class()
# {'directed': [...], 'undirected': [('emm', 'temp')], 'n_dags': 2}
```

Two DAGs fit any data equally well here, and they differ in the direction
of the one edge that makes temperature a confounder. The data cannot
settle that; what you know about the world has to.
`g.test_implications(df)` tests the independencies the DAG does imply.

## Where StatsPAI differs from the book's packages

- **Standard errors of `ipw`.** Divisor `n` here, `n - 1` in
  `propensity::ipw`. Multiply by `sqrt(n / (n - 1))` to compare.
- **Standardized differences.** See step 2. `halfmoon` also uses the
  divisor `n` inside each group and `p (1 - p)` for binary covariates, so
  its raw differences are slightly larger.
- **AUC.** `sp.auc` is the Mann-Whitney probability and equals
  `pROC::auc`. `halfmoon::check_model_auc` can differ in the fourth
  decimal.
- **Trimming.** `sp.trimming` returns the rows to keep. Refit the
  propensity model on them (`ps_refit` in the book), since the trimmed
  sample is a new population.
- **Continuous exposures.** `sp.ps_weights(exposure='continuous',
  sigma=)` takes the residual standard deviation from you. The book
  averages leave-one-out values from `broom`.

## What a paper written today would add

The book's first sixteen chapters stay with parametric models. Once the
workflow is clear, the same estimands are available with cross-fitted,
doubly robust estimators that do not need the propensity or outcome model
to be right on its own: `sp.aipw`, `sp.tmle`, `sp.dml(model='irm')`. For
weights that balance by construction and skip the propensity model see
`sp.ebalance`, `sp.cbps` and `sp.sbw`.
