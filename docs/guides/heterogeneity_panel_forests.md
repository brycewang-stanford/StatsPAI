# Heterogeneous effects with panel data: causal forests with clusters, fixed effects and DiD

A causal forest estimates the conditional average treatment effect
`tau(x)`. On panel or grouped data three things can go wrong, and StatsPAI
has one tool for each:

| Your data | Risk | Use |
| --- | --- | --- |
| Cross-section, independent units | — | `sp.causal_forest(...)` |
| Several rows per unit / firm / school | honesty and SEs assume independent rows | `sp.causal_forest(..., clusters="unit")` |
| Panel, treatment switches within units, selection on unit characteristics | unit and period effects masquerade as heterogeneity | `sp.causal_forest(..., fe="twoway", id=..., time=...)`, then `sp.forest_group_effects` / `sp.cate_pretrend_test` |
| Staggered adoption, effects that change with exposure | pooled comparisons use already-treated units as controls | `sp.did_forest(...)` |

All three share one engine, an implementation of generalized random forests
[athey2019generalized; wager2018estimation].

## 1. Why a pooled or globally demeaned forest fails

Take `Y_it = alpha_i + gamma_t + tau(X_i) D_it + e_it`.

* **Pooled forest.** If adoption depends on `alpha_i`, the forest splits on
  covariates correlated with `alpha_i` and reports level differences as
  effect differences.
* **Global two-way demeaning, then a forest.** Unit demeaning is harmless,
  but subtracting period means subtracts the cross-sectional mean of
  `tau(X_j) D_jt` over *all* units `j`. The demeaned outcome no longer equals
  `tau(X_i)` times the demeaned treatment unless `tau` is constant, so the
  forest fits a misspecified model exactly when there is heterogeneity to find.
* **Node-level demeaning** [kattenberg2023causal] removes the effects inside
  each node and each leaf. Units in a node have similar `x`, so `tau` is
  nearly constant there and the within regression is correctly specified.

StatsPAI's own simulation (N = 300 units, T = 6, three adoption cohorts plus
never-treated, adoption selected on the unit effect, 4 seeds, estimates on
treated rows):

| | pooled forest (clusters = unit) | `fe="twoway"` |
| --- | --- | --- |
| bias, constant effect | +0.36 to +2.37 | -0.02 to +0.03 |
| bias, `tau = 1 + x1` | +0.34 to +2.36 | -0.06 to +0.02 |
| RMSE, `tau = 1 + x1` | 0.76 to 2.41 | 0.32 to 0.42 |
| heterogeneity test p (constant effect) | 0.00 to 0.91 | 0.06 to 0.86 |

## 2. Clustered forests

```python
import statspai as sp

cf = sp.causal_forest(
    data=df, y="wage", d="training", x=["age", "educ", "tenure"],
    clusters="firm",
)
cf.average_treatment_effect()          # AIPW ATE, cluster-robust SE
cf.average_treatment_effect("treated") # ATT
sp.calibration_test(cf)                # heterogeneity test, cluster-robust
cf.best_linear_projection()            # which covariates drive tau(x)
cf.effect_interval(X_new)              # pointwise CIs
```

With `clusters=` every tree draws whole clusters, honest halves are split by
cluster, user-supplied nuisance models are cross-fitted in folds that never
split a cluster, and every average, calibration test, BLP and RATE uses
cluster-robust variances. `equalize_cluster_weights=True` gives clusters equal
weight.

## 3. Causal forests with fixed effects

```python
cf = sp.causal_forest(
    data=panel, y="y", d="treated", x=["x1", "x2", "x3"],
    id="county", time="year", fe="twoway",
)
tau_hat = cf.predict()                 # out-of-bag CATE per row
cf.effect_variance()                   # little-bag variances
sp.calibration_test(cf)                # heterogeneity test
```

Requirements, all checked at fit time: `(unit, time)` pairs are unique,
the treatment varies within units, and `clusters` (default: the unit) nest
units. Unbalanced panels are fine.

### Which splitting rule

The default `split_rule="grf"` splits on the GRF gradient criterion applied
to the node-residualized data. `split_rule="cffe"` replaces it with the
criterion of [kattenberg2023causal] and the `causalfe` package
[aytug2026causalfe], which scores a candidate split by

```
n_L n_R / n^2 (tau_L - tau_R)^2,   tau = sum(D~ Y~) / sum(D~^2)
```

on the parent node's residualized outcome and treatment. Use it to
reproduce that implementation; everything downstream (honest leaves,
little-bag variances, imputation scores) is unchanged, and the imputation
ATT does not depend on the forest at all.

That criterion compares ratios computed in the candidate children, so it
needs the shallow, large-leaf trees it was designed with. On the causalfe
package's own simulation (15 replications, 400 trees) its CATE RMSE was
0.74 with StatsPAI's defaults (`min_samples_leaf=5`, no depth cap) and 0.53
with `min_samples_leaf=20, max_depth=4`; the GRF criterion gave 0.50 under
both. StatsPAI warns when `cffe` is combined with deep default trees.

```python
cf = sp.causal_forest(
    data=panel, y="y", d="treated", x=["x1", "x2", "x3"],
    id="county", time="year", fe="twoway",
    split_rule="cffe", min_samples_leaf=20, max_depth=4,
)
```

### Averages, groups and tests: imputation scores

A within-unit design has no propensity `E[D | X]`, so the doubly-robust
(AIPW) scores behind a pooled forest's averages do not exist. The FE forest's
own model supplies another unbiased signal. Fit the unit and period effects
on the **untreated** cells only (plus any time-varying covariates, see
`covariates` below) and impute each treated cell's untreated outcome
[borusyak2024revisiting]:

```
Gamma_it = Y_it - alpha_hat_i - gamma_hat_t        (treated cells)
```

`Gamma_it` is unbiased for that cell's effect under parallel trends and no
anticipation, whatever the heterogeneity: in `x`, in calendar time, or in
exposure. The forest becomes a *proxy* and `Gamma` the *signal*, the division
of labour of generic machine-learning inference [chernozhukov2025generic]:

```python
cf.average_treatment_effect("treated")    # imputation ATT
cf.best_linear_projection()               # Gamma on (1, X), treated cells
sp.calibration_test(cf)                   # Gamma on the OOB prediction
sp.calibrate_cate(cf)                     # de-attenuated predictions
sp.forest_group_effects(cf, by=...)       # group ATTs (see section 4)
```

* **The ATT is the imputation estimator.** On `mpdta` it equals
  `sp.did_imputation` to 1e-13, and with `variance="bjs"` so does its
  standard error (5e-15 relative), so it inherits that estimator's
  Stata / R parity.
* **Standard errors** come from the exact linear weights of the estimator
  (every quantity here is `v'y`), clustered by the forest's clusters.
  Treated residuals need a model of the heterogeneity. `variance="bjs"` uses
  cohort x event-time means (conservative). The default `variance="forest"`
  first subtracts the out-of-bag forest prediction, which never uses the
  unit's own data because trees draw whole units.
* **`covariates`.** The default `covariates="none"` is the pure two-way
  model, i.e. exactly `sp.did_imputation` without covariates (which spells
  the same argument `controls=`; that spelling is accepted here too).
  `covariates="auto"`
  adds the effect modifiers and controls that vary within units to the
  untreated model linearly (time-invariant ones are absorbed by the unit
  effect); a list of names selects some of them. If a time-varying covariate
  such as GDP drives the outcome, leaving it out gives the untreated outcome
  unit-specific trends and biases every average (section 4); a covariate
  that itself responds to the treatment must stay out. The identity with
  `sp.did_imputation` holds for `covariates="none"` (or covariates that do not
  vary within units).
* **Two-way only.** Imputation needs period ids, so it requires
  `fe="twoway"`; `fe="unit"` forests keep the within calibration regression
  and raise for the averages.
* **What stays unavailable.** `target_sample="all"`, `"control"` or
  `"overlap"` raise: parallel trends identifies effects on treated cells,
  and effects on untreated cells are extrapolations of `tau(x)` (predict them
  with `cf.effect(X_new)` after `sp.forest_support`, section 4). `sp.rate`
  is available but reads on the treated cells, which changes what it means
  (section 5).

Monte Carlo evidence (200 replications; N = 300 units, T = 8, cohorts 3, 5
and 7 plus never-treated, adoption selected on the unit effect; 500 trees):

| | effect `(1 + x1)(1 + 0.2 e)` | constant effect |
| --- | --- | --- |
| bias of the imputation ATT | -0.003 | -0.003 |
| bias of the mean forest prediction (`forest_plug_in`) | -0.177 | -0.001 |
| ATT coverage, `variance="forest"` / `"bjs"` | 97.5% / 100% | 94.5% / 94.0% |
| mean ATT standard error, `"forest"` / `"bjs"` | 0.093 / 0.140 | 0.085 / 0.082 |
| group ATT coverage (`x1 > 0` vs not), `"forest"` | 97.5% | 96.3% |
| calibration test rejects at 5% | 100% | 4.0% |
| RMSE of treated cells' predictions, raw / calibrated | 0.634 / 0.570 | 0.213 / 0.118 |

The mean forest prediction is shrunk toward zero when effects are
heterogeneous, which is why it is reported only for comparison. The
calibration slope is now a genuine de-attenuation factor. The 1.29.0
regression on globally within-transformed variables [aytug2026attenuated],
available as `method="within"`, tests for heterogeneity with the right size,
but its slope stayed at 0.88 to 1.07 while the predictions' slope on the
truth was 0.65 to 0.91, and rescaling by it did not reduce RMSE.

## 4. Worked example: a currency union on a dyadic trade panel

[aytug2026euro] estimates how the euro's trade effect varies across country
pairs: a pooled causal forest on pair-year data, then a causal forest with
fixed effects [kattenberg2023causal; aytug2026causalfe] to remove pair and
year effects, followed by pair- and country-level effects, counterfactual
effects for the countries that stayed out, and pre-trends by predicted-effect
group. `sp.datasets.currency_union_panel()` has the same layout with
**simulated** numbers and a known effect `tau_true`: 15 countries, 105 pairs,
1995-2015, 11 countries adopting in 1999 and one in 2001, pair effects that
rise with the latent "core" index that also drives adoption, and effects that
rise with pre-adoption trade intensity and size. (Country codes are generic;
nothing here is an estimate of the euro's effect.)

```python
import numpy as np
import statspai as sp

df = sp.datasets.currency_union_panel(seed=0)      # true ATT 0.133
x = ["pre_trade", "log_gdp_prod", "log_gdppc"]

pooled = sp.causal_forest(data=df, y="log_trade", d="euro", x=x,
                          clusters="pair", random_state=0)
pooled.average_treatment_effect("treated")          # 0.482 (se 0.041)

cf = sp.causal_forest(data=df, y="log_trade", d="euro", x=x,
                      id="pair", time="year", fe="twoway", random_state=0)
C = "auto"            # GDP varies within pairs and drives trade
att = cf.average_treatment_effect("treated", covariates=C)
# estimate 0.134, se 0.023, 95% CI [0.089, 0.179]; forest_plug_in 0.166;
# imputation_covariates ['log_gdp_prod', 'log_gdppc']
cf.average_treatment_effect("treated")["estimate"]   # covariates="none": 0.199
```

The pooled forest compares core pairs, which trade more *and* adopted, with
peripheral ones and more than triples the effect. The FE forest's imputation
ATT recovers it once GDP enters the untreated model; without it, pairs
growing faster look more affected and the estimate is 0.199. With the second wave of adopters
(`currency_union_panel(late_adopters=True)`: adoptions in 2007-2015 during a
common downturn, 378 pairs) the imputation ATT is 0.160 (se 0.012) against a
true 0.150; `tests/test_forest_fe_imputation.py` checks both designs.

**Which covariates drive the heterogeneity.**

```python
sp.calibration_test(cf, covariates=C)     # differential slope 0.63 (se 0.19), p = 5e-4
cf.best_linear_projection(covariates=C)   # pre_trade 0.060 (se 0.021); GDP terms ~0
```

**Pair, country and period effects.** `members` makes each country a group
that contains all of its pairs, the aggregation behind the country tables of
[aytug2026euro]; `scale="percent"` converts log points to `100 (exp(x) - 1)`:

```python
members = df[["country_i", "country_j"]].to_numpy()
by_country = sp.forest_group_effects(cf, members=members, scale="percent",
                                     covariates=C)
#        n_rows  estimate_pct  ci_low_pct  ci_high_pct  forest_mean_pct
# C07       185          32.5        20.2         46.0             26.6
# C05       185          21.0        10.7         32.3             25.2
# ...
# C06       185           6.0        -1.5         14.2             17.8
by_country.attrs["tests"]["equality_p"]             # 0.0008

sp.forest_group_effects(cf, by="cate_quantile", covariates=C)     # GATES
#       estimate     se  forest_mean
# Q1       0.067  0.031        0.069
# Q4       0.192  0.030        0.267    Q4 - Q1 = 0.125 (se 0.036)

period = np.where(df.year <= 2003, "1999-2003",
                  np.where(df.year <= 2008, "2004-2008", "2009-2015"))
sp.forest_group_effects(cf, by=period, covariates=C)  # 0.126, 0.126, 0.145
sp.forest_group_effects(cf, by=df["pair"].to_numpy(), covariates=C)  # per pair
```

Group estimates average imputation scores, so each country's interval is a
sampling interval for its average effect, not the spread of fitted values.
Countries appear in many pairs, so rows that share a country may be
correlated; `cluster="dyadic"` allows that [aronow2015cluster]. It is
justified as the number of countries grows: with 15 countries it warns, and
a group that loads on a single country can get a non-positive variance,
which is reported as `NaN` rather than as a zero-width interval.

**Counterfactual effects for countries that did not adopt.** A forest can
predict `tau(x)` for pairs that were never treated, but only as an
extrapolation from pairs that were. `sp.forest_support` says where that
extrapolation leaves the data:

```python
outs = df[(df.ever_euro == 0) & (df.year == 2010)]
sup = sp.forest_support(cf, outs[x])
sup.attrs["summary"]["share_supported"]             # 0.62
# mean predicted effect and share of supported pairs:
# C13  +3.6%  0.14 | C14  +8.2%  0.86 | C15  +10.0%  0.79
```

Only 14% of the pairs involving `C13` lie inside the covariate region of
pairs that switched, so its counterfactual is mostly an extrapolation. The
distance benchmark compares each reference row with rows of *other* units,
as a new unit is, so a pair's own years do not make the data look denser
than it is.

**Were high-effect pairs already on a different path?** A forest can rank
pairs by pre-existing trends rather than by effects. `sp.cate_pretrend_test`
sorts units by their mean OOB prediction and runs the pre-trend regression of
[borusyak2024revisiting] on untreated cells with group x lead indicators:

```python
pt = sp.cate_pretrend_test(cf, n_groups=2, leads=3, covariates=C)
pt["equal_across_groups"]      # chi2(3) = 3.13, p = 0.37
pt["joint_zero"]               # chi2(6) = 8.21, p = 0.22
```

In 100 replications of a staggered design like that of section 3
(N = 300, T = 7, cohorts 3, 4 and 6; two groups, two leads) the equality
test rejected 3% of the time without a pre-trend and 71% of the time when
high-`x1` treated units trended by 0.3 per period before adoption (the joint
test: 5% and 95%).

`time_effects="common"` (default) tests the comparison the imputation
scores use; `"by_group"` runs a separate event study per group, as in the
appendix of [aytug2026euro].

## 5. Was the ranking worth anything? RATE and split-sample evaluation

A forest that fits `tau(x)` well is not the same thing as a rule worth
targeting on. The rank-weighted average treatment effect
[yadlowsky2025evaluating] answers the second question directly: sort the
cells by the rule, and ask how much larger the effect is among the ones it
would have picked.

```
TOC(q) = ATT(top q by S) - ATT,
AUTOC  = int_0^1 TOC(q) dq,      QINI = int_0^1 q TOC(q) dq
```

`sp.rate` used to refuse a forest with fixed effects, because the
doubly-robust score it averages needs a propensity. The imputation scores of
section 3 need none, so the curve is read on the population they identify --
the **treated cells**. That makes it a *retrospective* targeting curve ("the
effect was this much larger among the pairs the rule would have prioritised")
rather than the population RATE a randomised design gives, and the returned
`estimand` says so.

The standard error is exact rather than assumed. Conditional on the ranking,
AUTOC and QINI are *linear* in the scores -- weight `(H_m - H_{j-1} - 1) / m`
on the `j`-th cell from the top -- and the imputation scores are themselves
linear in `y`, so composing the two makes RATE one more functional `v'y`,
with the same cluster- or dyad-robust variance as the ATT. The composed
weights annihilate the unit and period dummies to 1e-15 and put exactly zero
net weight on the treatment, RATE being a contrast rather than an effect;
both are asserted in `tests/test_forest_rate_fe.py`.

`variance` defaults to `"bjs"` here, unlike `sp.average_treatment_effect`,
and the reason is measured. `"forest"` nets the fitted heterogeneity out of
the residual, which estimates the variance of the RATE *of this sample*; the
ATT converges to its population value fast enough that the distinction does
not show, but RATE loads on the tail of the effect distribution and it does
(table below).

### Do not grade a rule on the sample that fitted it

`sp.rate` ranks the cells it also scores. For a pooled forest out-of-bag
predictions mostly handle that; for a forest with fixed effects they do not,
because every imputation score carries `-gamma_hat_t`, estimated from the
very periods the forest was trained on. The two stay correlated even though
no unit predicts itself, and the estimator acquires a **negative** bias.

`sp.rate_split` removes the overlap. The units -- or, with `members=`, the
nodes of a dyadic panel -- are split in two, a forest with the same
hyper-parameters is refitted on each half, the training half's forest ranks
the evaluation half's treated cells, and those cells' imputation scores come
from an untreated two-way model fitted on the evaluation half alone. Nothing
the rule saw enters the score it is graded on. For dyadic data the split is
by member, and flows between a training and an evaluation country are
dropped and counted, because they belong to neither half.

```python
sp.rate(cf)                    # diagnostic; warns that it is not a test
sp.rate(cf, priorities=other)  # valid, if `other` was fitted elsewhere
sp.rate_split(cf)              # valid, and does the split for you
```

The price is sample: each forest sees half the units, so the rule is noisier
than the one fitted on everything and the RATE it earns is a **lower bound**
on what the full-sample rule is worth. Below roughly 30 units (or members) a
side the function says so, because at that size the split is measuring
itself.

### One split is a draw, not an estimate

A single split fixes the answer to one partition, and with `random_state`
in reach it is easy to keep the partition that agrees with you — the
practice [chernozhukov2025generic] show invalidates inference. Their
variational estimation and inference (VEIN) is what `n_splits` does, and it
is the default:

* the estimate is the **median** over splits;
* the interval is the **median of the conditional intervals**, built at
  `1 - alpha/2` so that their median covers at `1 - alpha`;
* the p-value is **twice the median** conditional p-value.

`n_splits=21` costs about twelve seconds on a 150-unit panel, 100 (as they
use) about a minute, and `n_splits=1` reproduces one draw and warns.
`estimate_min` / `estimate_max` / `estimate_iqr` report how far the split
was moving the answer.

The trade panel of section 4 shows why this matters. Fifteen countries,
105 pairs, so a member split leaves seven or eight countries a side:

```python
members = df[["country_i", "country_j"]].to_numpy()

# The oracle ranking on the whole sample: the heterogeneity is real.
sp.rate(cf, priorities=df["tau_true"], cluster="dyadic",
        members=members, covariates=C)
# AUTOC 0.0809 (se 0.0310);  QINI 0.0222 (se 0.0111)

# Single splits, one seed each -- six different answers.
[sp.rate_split(cf, members=members, covariates=C,
               n_splits=1, random_state=s)["estimate"] for s in range(6)]
# [-0.075, -0.040, +0.037, -0.009, -0.003, +0.013]

# The default, aggregating 21 of them.
sp.rate_split(cf, members=members, covariates=C)
# estimate +0.0001, 95% CI [-0.0566, +0.0492], p 0.54
#   n_splits 20, n_splits_skipped 1, n_splits_without_interval 7
#   estimate_min -0.0752, estimate_max +0.0510
```

Seed 0 alone would have read as a **significant negative** AUTOC, 95% CI
[-0.133, -0.017]. VEIN reports what is actually there: nothing, on this
sample, with an interval that says so. Two of the bookkeeping fields earn
their place here — one partition left the evaluation half with no imputable
treated cell and was skipped as inadmissible (it would otherwise have
destroyed the whole call), and seven had a non-positive dyadic variance and
so contributed an estimate but no interval. Both are the estimator saying
fifteen countries is not enough, which is the honest reading.

Monte Carlo evidence (400 replications,
`tests/reliability/forest_split_evaluation.py`; N = 150 units, T = 8, staggered
adoption selected on the unit effect, 250 trees; `tau = 0.3 + b z` with `z`
standard normal and independent of adoption, so the population AUTOC is
`0.9032 b` and QINI `0.2821 b` exactly):

| `b = 0` (no heterogeneity: true RATE is 0 for *every* rule) | AUTOC | QINI |
| --- | --- | --- |
| `sp.rate`, ranked by the forest's own OOB predictions | **-0.0282**, rejects **16.0%** | -0.0061, rejects 12.2% |
| `sp.rate_split` (21 splits) | **-0.0002**, rejects **0.5%** | +0.0004, rejects 0.0% |
| `sp.rate_split`, `variance="forest"` | -0.0002, rejects 0.5% | +0.0004, rejects 0.0% |

| `b = 0.5` (population AUTOC 0.4516, QINI 0.1410) | AUTOC | QINI |
| --- | --- | --- |
| `sp.rate_split`, mean estimate | 0.383 | 0.128 |
| `sp.rate_split`, coverage of the ideal ranking's value, `"bjs"` / `"forest"` | 94.5% / 89.5% | 99.0% / 97.5% |
| `sp.rate_split` power at 5% | 100% | 100% |

The split estimate sits below the population value because that value
belongs to the ideal ranking, and a forest grown on half the units ranks
imperfectly: `sp.rate_split` estimates the RATE of the rule it fitted. The
rejection rate under the null is far below 5% because the aggregation over
21 splits is conservative by construction. Figures quoted for releases
through 1.39.3 (7.5% and 4.0% under the null) came from halves that were not
refitted the way the forest was, and are superseded by this table.

Read the first table as the reason `sp.rate_split` exists: on a design with
no heterogeneity whatsoever the reused ranking rejected the null more than
three times too often, and reported a negative AUTOC where the truth is
zero. The second is why `"bjs"` is the default and `"forest"` is not: each
is calibrated for a different estimand, and the population one is what a
reader takes away.

### From "is it worth targeting?" to "target like this"

`sp.rate_split` says whether *some* ranking pays. `sp.forest_policy_tree`
returns the rule — a depth-limited tree in the effect modifiers, the object
a programme could actually be written in — and prices it.

```python
res = sp.forest_policy_tree(cf, depth=2, cost=0.05, covariates=C,
                            members=members)
print(res["rules"])
# IF log_gdppc <= 10.4316:
#   IF pre_trade <= 19.1339:
#     DON'T TREAT (n=22, avg_benefit=-0.0815)
#   ELSE (pre_trade > 19.1339):
#     TREAT (n=78, avg_benefit=0.0995)
# ELSE (log_gdppc > 10.4316):
#   IF log_gdppc <= 10.4557:
#     DON'T TREAT (n=12, avg_benefit=-0.0321)
#   ELSE (log_gdppc > 10.4557):
#     TREAT (n=133, avg_benefit=0.0797)

res["value"]                # +0.0553, 95% CI [-0.0199, +0.1479]
res["value_treat_all"]      # ... of treating every cell (the ATT minus cost)
res["gain_over_treat_all"]  # -0.0033, 95% CI [-0.0277, +0.0067], p = 0.91
res["share_treated"]        # 0.92 -- the rule keeps almost the whole programme
res["diagnostics"]["split_stability"]
# {'root_covariate_counts': {'pre_trade': 11, 'log_gdp_prod': 5,
#                            'log_gdppc': 4},
#  'root_covariate_modal_share': 0.55, ...}
```

Read that the way section 5 read the RATE on the same panel: **the tree
always prints a rule, and here the evidence does not support it.** The gain
over treating everyone is -0.003 with an interval straddling zero, and the
root covariate changed across the splits — `pre_trade` in eleven of twenty,
`log_gdp_prod` in five, `log_gdppc` in four. A rule that cannot survive a
different half of the countries is not a rule this sample identified,
however confidently the tree prints it. (`cost=0.05` against an ATT of
0.134 makes targeting a real question here; `cost=0` would make treating
everyone optimal by construction.)

Three things this inherits from the design, and one it adds.

**`cost` is usually what makes the question real.** With `cost=0` and
effects positive everywhere, treating every cell is optimal and the tree
has nothing to find. Set `cost` in the outcome's units and the rule treats
where the effect clears it.

**The rule is retrospective.** The scores exist on treated cells, so the
question is "of the cells that were treated, which should have been" — not
whether an untreated unit should be treated, which is an extrapolation of
`tau(x)` that `average_treatment_effect` also refuses here. Run
`sp.forest_support` before carrying a rule to units the design never
switched.

**Fitted and priced on disjoint units, always.** This is the same trap as
section 5 and it bites harder, because a tree is chosen to maximise the
very quantity it is then graded on. On a design where *every rule is worth
exactly the same* — no heterogeneity, and `cost` equal to the constant
effect, so the true gain over treating everyone is exactly 0 (200
replications, N = 200 units, T = 8):

| | mean gain | claims a significant gain |
| --- | --- | --- |
| fitted and priced on the same cells | **+0.048** (own se 0.037) | **13.0%** |
| fitted and priced on disjoint halves (21 splits, 400 replications) | **+0.008** | **0.0%** |

and the split costs almost nothing when the heterogeneity is real: with
`tau = 0.3 + 0.8 z` and `cost = 0.3` the oracle gain is `0.8 phi(0) =
0.3192`, the split-sample estimate averaged **0.307** and was significant
in every one of 400 replications, and
the same-sample one **0.3316**. There is no flag to turn the split off.

**Aggregated over splits, like everything else here.** `n_splits` defaults
to 21 and the value, the treat-all value and the gain are each VEIN-medians
[chernozhukov2025generic]. A rule cannot be averaged, so the tree reported
is the one from the median-gain split, and
`diagnostics["split_stability"]` says how often each covariate was chosen at
the root and how far the thresholds and treated shares moved. *A tight
interval on the value of a rule whose root covariate changes from split to
split is not evidence for that rule.* Note that each feature is aggregated
on its own, so with `n_splits > 1` the three medians do not satisfy
`gain = value - value_treat_all`; that identity holds within a split, and
`n_splits=1` reports it directly.

**What it adds: the gain has a standard error.** Value, treat-all value and
the gain between them are three linear functionals of the same `y` from one
design, so the gain is reportable on its own rather than as two overlapping
intervals. Its estimate is exactly the difference; its variance is not the
difference of the other two, because the BJS centring weights the cohort ×
event-time mean by each functional's own `v^2` and the gain's weights
vanish on every cell the rule treats. Both readings are conservative
(0.063 and 0.096 against a Monte Carlo 0.047); the narrower is reported.

## 6. DiD causal forests for staggered adoption

```python
res = sp.did_forest(
    df, y="emp", id="county", time="year", cohort="first_treat",
    x=["pop", "income"], clusters="state",
    control_group="notyettreated",
)
res.event_study            # dynamic effects with influence-function SEs
res.overall                # ATT over post-treatment cells
res.pretrend_test          # joint Wald test of pre-period effects
res.att_gt                 # doubly-robust ATT(g, t) + heterogeneity test per cell
res.unit_cate              # each treated unit's CATE by event time
res.predict_cate(X, event_time=2)
res.forest(2004, 2006)     # the CausalForest behind one cell
```

For each cohort `g` and period `t`, the treated units (cohort `g`) and the
comparison units (never treated, plus not yet treated by `max(t, g - 1)`) form
one cross-section. The outcome is `Y_t - Y_{g-1}`. Under conditional parallel
trends its conditional mean contrast is `tau_{g,t}(x)`, so one honest causal
forest per cell estimates it: the DiD causal forest of
[gavrilova2025difference] applied to every Callaway-Sant'Anna group-time
comparison [callaway2021difference]. On a two-period block, removing unit and
period effects is the same long difference, which is the block construction of
[aytug2026fixed].

`ATT(g, t)` is the forest's doubly-robust ATT: the mean out-of-bag CATE over
cohort `g` plus an inverse-propensity-weighted residual correction. Each cell
has a per-unit influence function; aggregates sum them unit by unit (or
cluster by cluster), so units that appear in several cells are accounted for.
Cohort-size weights are treated as known.

Monte Carlo (200 replications; N = 800, T = 7, cohorts 4 and 6 plus
never-treated; adoption and the untreated trend both depend on `x1`; effect
`(1 + x1)(1 + 0.25 e)`; 300 trees per cell):

| quantity | 95% CI coverage |
| --- | --- |
| overall ATT | 97.5% |
| event study, e = 0 ... 3 | 95.5% to 98.5% |
| event study, e = -5 ... -2 (placebos) | 93.0% to 97.0% |
| pre-trend joint test, rejection at 5% | 8.0% |

A second run of 300 replications (fresh seeds) checked the pre-period
calibration directly: reported standard errors are 0.99 to 1.05 times the
Monte Carlo standard deviation of every placebo cell and event-study
coefficient, and the Wald statistic has mean 4.00 and variance 7.91 against
4 and 8 for its chi-squared(4) reference, rejecting in 3.3% of replications.
Pooled over both runs the pre-trend test rejects in 26 of 500 replications
(5.2%). Post-treatment intervals are slightly conservative.

The simulation design of [gavrilova2025difference] (workers in firms, firm
treatment, CATT 10 for `x1 = 1` and 1 for `x1 = 0`) is a known-truth test in
`tests/reference_parity/test_panel_forest_recovery.py`.

## 7. Evidence and parity

| What | Grade | Where |
| --- | --- | --- |
| Calibration test, AIPW ATE/ATT/ATC/overlap, BLP, `vcovCL` HC0-HC3, given a forest | exact vs grf 2.6.1 / sandwich (<= 8.5e-15) | `tests/reference_parity/test_grf_cluster_operator_parity.py` |
| Forest: CATE RMSE, pointwise variances, coverage, ATE vs grf (i.i.d. and clustered) | T3 (Monte Carlo) | `tests/reference_parity/test_grf_engine_statistical_parity.py` |
| Engine identities (forest weights, OOB, clusters, FE within transform) | exact | `tests/test_grf_engine.py` |
| FE forest and DiD forest recovery | T1 (known truth) | `tests/test_panel_causal_forest.py`, `tests/test_did_forest.py` |
| FE-forest imputation ATT = `sp.did_imputation` (estimate and BJS SE) | exact identity (1e-13 / 5e-15), hence the Stata / R parity of `did_imputation` | `tests/test_forest_fe_imputation.py` |
| Group, BLP and calibration estimates = OLS of imputation scores; dyadic variance = pairwise definition = `sp.dyadic_regression` | exact | `tests/test_forest_fe_imputation.py` |
| Pre-trend regression = dummy-variable OLS with CR1 | exact (1e-8) | `tests/test_forest_fe_imputation.py` |
| Imputation ATT / group coverage, calibration size and power | T1 (Monte Carlo, section 3) | `tests/reference_parity/test_fe_forest_imputation_recovery.py` |
| RATE = rank weights o imputation weights; weights annihilate the FE design and put zero net weight on D | exact (1e-15 / 1e-10) | `tests/test_forest_rate_fe.py` |
| RATE bias and coverage, reuse bias, `rate_split` size and power | T1 (Monte Carlo, section 5) | `tests/reference_parity/test_fe_forest_rate_recovery.py` |
| Policy-tree gain: same-sample bias, split-sample size and power, coverage of the fitted rule's true value | T1 (Monte Carlo, section 5) | `tests/reference_parity/test_fe_forest_policy_recovery.py` |

**Comparison with `causalfe`.** `causalfe` 0.3.2 [aytug2026causalfe] is the
Python implementation of [kattenberg2023causal] used in [aytug2026euro]. On
its own simulation design (50 replications) the two forests predict equally
well (CATE RMSE 0.476 vs 0.456, correlation 0.92 for both), but its pointwise
95% intervals cover 46% of treated cells against 93% here. With adoption
selected on the unit effect and effects that grow with exposure, the
two-way fixed-effects coefficient it reports as the ATE is biased by -0.18
(coverage 86%) while the imputation ATT is biased by -0.003 (coverage 96%).
This compares statistical properties of two stochastic estimators; it is not
parity. Scripts and numbers: `benchmarks/cross_library/panel_forest_causalfe/`.

The forest is not bit-identical to grf (independent random streams), so it is
never described as "aligned with R". Track A module 13 (AIPW ATE/ATT vs grf)
is inside its registered budget: estimates within 0.4%, standard errors
within 0.7%.

## 8. Mapping from R `grf`

| R | StatsPAI |
| --- | --- |
| `causal_forest(X, Y, W, clusters = g)` | `sp.causal_forest(Y=Y, T=W, X=X, clusters=g)` |
| `predict(cf)` (out-of-bag) | `cf.predict()` |
| `predict(cf, X.new, estimate.variance = TRUE)` | `cf.effect(X_new)`, `cf.effect_variance(X_new)` |
| `average_treatment_effect(cf, target.sample = "treated")` | `cf.average_treatment_effect("treated")` |
| `test_calibration(cf)` | `sp.calibration_test(cf)` |
| `best_linear_projection(cf, A)` | `cf.best_linear_projection(A)` |
| `rank_average_treatment_effect(cf, priorities)` | `sp.rate(cf)` (OOB priorities; see its docstring) |
| `split_frequencies(cf)` | `cf.split_frequencies()` |

Python `causalfe` (0.3.2):

| `causalfe` | StatsPAI |
| --- | --- |
| `CFFEForest(n_trees=, max_depth=, min_leaf=).fit(X, Y, D, unit, time)` | `sp.causal_forest(Y=Y, T=D, X=X, id=unit, time=time, fe="twoway", split_rule="cffe", n_estimators=, max_depth=, min_samples_leaf=)` |
| `predict(X)` | `cf.effect(X)`; `cf.predict()` for out-of-bag |
| `predict_interval(X)` | `cf.effect(X)`, `cf.effect_variance(X)` |
| `ate()` / `ate_interval()` (within-FE coefficient) | `cf.average_treatment_effect("treated")` (imputation ATT; see section 7) |
| `feature_importances()` | `cf.variable_importance()`, `cf.split_frequencies()` |

## References

[athey2019generalized; wager2018estimation; kattenberg2023causal;
gavrilova2025difference; callaway2021difference; chernozhukov2025generic;
aytug2026attenuated; aytug2026fixed; aytug2026causalfe; aytug2026euro;
borusyak2024revisiting; aronow2015cluster]
