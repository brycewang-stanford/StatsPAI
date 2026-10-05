# Facure, *Causal Inference in Python*, in StatsPAI

Matheus Facure's book (O'Reilly, 2023) teaches causal inference for people
who work with product and business data. Its eleven chapters go from A/B
tests to switchback experiments, and nearly every estimator is written out
with pandas, statsmodels, scikit-learn and cvxpy. This guide maps each
chapter to the StatsPAI call that gives the same number, and says where
StatsPAI makes a different choice.

The notebooks and data are at
<https://github.com/matheusfacure/causal-inference-in-python-code>. They
are not redistributed with StatsPAI. The examples below read the CSV files
from `data/`.

## Chapter by chapter

| chapter | the book computes | StatsPAI |
| --- | --- | --- |
| 2 | difference in means, its standard error and test; sample size | `sp.ttest(unequal=True)`; `sp.power('rct', effect_size=, sigma=, power_target=)` |
| 2 | normalized differences of covariates | `sp.balance_table`, `sp.balance_check` (two arms at a time) |
| 3 | d-separation, backdoor paths, bad controls | `sp.dag(...)`: `.d_separated`, `.adjustment_sets`, `.bad_controls`; `sp.dag_recommend_estimator` |
| 4 | regression adjustment, dummies, saturated models | `sp.regress` with `C()`, `I()`, `np.sqrt()`, `*`; `sp.margins` |
| 4 | fixed effects by de-meaning | `sp.feols('y ~ x \| group')` |
| 5 | propensity score, IPW, stabilised weights | `sp.propensity_score`; `sp.ipw(normalize=)` |
| 5 | matching on the propensity score | `sp.match(method='psm', estimand='ATE')` |
| 5 | doubly robust estimation | `sp.aipw` |
| 5 | continuous treatment | `sp.dose_response`; `sp.dml(model='plr')` |
| 6 | effect by quantile of a prediction, cumulative gain, its area | `sp.cate_gain_curve`; `sp.cate_eval` for a 0/1 treatment |
| 7 | T-, X- and S-learners | `sp.metalearner(learner='t' / 'x' / 's')`; `sp.predict_cate` |
| 7 | double machine learning for the ATE and the CATE | `sp.dml(model='plr')`; `sp.metalearner(learner='r')` |
| 8 | 2×2, two-way fixed effects, clustered and block-bootstrap inference | `sp.did`; `sp.feols(..., vcov={'CRV1': 'city'})`; `sp.bootstrap(cluster=)` |
| 8 | effect by period | `sp.event_study` |
| 8 | doubly robust difference-in-differences | `sp.drdid` |
| 8 | staggered adoption | `sp.did_imputation`, `sp.etwfe`, `sp.callaway_santanna`, `sp.sun_abraham` |
| 9 | synthetic control of several treated units | `sp.geolift` |
| 9 | debiasing and the t-test | `sp.synth(inference='ttest')`, `sp.synth_ttest` |
| 9 | synthetic difference-in-differences | `sp.sdid` |
| 10 | sample size for a geo experiment; which units to treat | `sp.power('cluster_rct', ...)`; `sp.synth_experimental_design(criterion='population')` |
| 10 | switchback experiments | `sp.switchback_design`, `sp.switchback` |
| 11 | Wald estimator, 2SLS, first stage | `sp.iv` (diagnostics in `.diagnostics`) |
| 11 | fuzzy discontinuity, bunching at the threshold | `sp.rdrobust(fuzzy=)`; `sp.rddensity` |

## Things the book's data will teach you about your own

### Categorical covariates

The management-training data of chapter 5 have three categorical
covariates. The book writes them as `C(role)` in a formula. In StatsPAI a
covariate list takes them in any of three forms, and all three build the
same indicator columns:

```python
import pandas as pd
import statspai as sp

df = pd.read_csv("data/management_training.csv")
cont = ["tenure", "last_engagement_score", "department_score"]

sp.aipw(df, y="engagement_score", treat="intervention",
        covariates=cont + ["C(n_of_reports)", "C(gender)", "C(role)"],
        cross_fit=False)                                   # 0.2712

as_cat = df.astype({c: "category" for c in ["n_of_reports", "gender", "role"]})
sp.aipw(as_cat, y="engagement_score", treat="intervention",
        covariates=cont + ["n_of_reports", "gender", "role"],
        cross_fit=False)                                   # the same
```

A text column is expanded too. A plain integer column is a number: it
enters as one linear term (0.2757 here), which is what you asked for.
The expansion is recorded in `result.model_info['covariate_expansion']`.

### A `treated` flag and a `post` flag

Chapter 8's data have one row per city and day, a 0/1 `treated` column and
a 0/1 `post` column. That is the 2×2 design:

```python
mkt = pd.read_csv("data/short_offline_mkt_south.csv", parse_dates=["date"])
res = sp.did(mkt, y="downloads", treat="treated", time="post", id="city")
res.estimate, res.se          # 0.6917, clustered on city
```

For anything staggered, the estimators want the first treated period of
each unit in the `treat` (or `g`, `first_treat`) column, not a flag. A 0/1
column that never changes within a unit is read as "first treated in period
1", and `sp.did` warns when it sees one next to a time variable with more
than two values.

### Dates

The staggered data of the same chapter keep `date` and `cohort` as dates,
with never-treated cities at `2100-01-01`. That works as it is:

```python
c = pd.read_csv("data/offline_mkt_staggered.csv", parse_dates=["date", "cohort"])
west = c[c["region"] == "W"]
res = sp.did_imputation(west, y="downloads", group="city",
                        time="date", first_treat="cohort")
res.estimate                                   # 2.2598, the book's 2.2598
res.model_info["calendar_time"]["periods"][:2]  # the date of period 1, 2, ...
```

Observed dates are numbered in order, a cohort date becomes the number of
the first observed date on or after it, and a missing cohort or one after
the last observed date means never treated. Event time therefore counts
observed periods. If the dates are not evenly spaced you get a warning.

### Several treated units

`sp.synth` takes one treated unit. Chapter 9 treats three cities and builds
one synthetic control for their average, which is `sp.geolift`:

```python
df = pd.read_csv("data/online_mkt.csv", parse_dates=["date"])
df["y"] = 100 * df["app_download"] / df["population"]
treated = ["sao_paulo", "porto_alegre", "joao_pessoa"]
start = pd.Timestamp("2022-05-01")

sp.geolift(df, outcome="y", geo="city", time="date",
           treated_geos=treated, treatment_time=start)            # placebo inference
sp.geolift(df, outcome="y", geo="city", time="date",
           treated_geos=treated, treatment_time=start,
           inference="ttest", alpha=0.1)                          # the book's t-test
sp.sdid(df, outcome="y", unit="city", time="date",
        treated_unit=treated, treatment_time=start)               # synthetic DiD
```

## What is new for this book

### A continuous treatment in the R-learner

Chapter 7 estimates how the effect of a discount varies across days with
the residual-on-residual regression of double machine learning.
`sp.metalearner(learner='r')` does this for a continuous treatment:

```python
sales = pd.read_csv("data/discount_data.csv")
train = sales[sales["day"] < "2018-01-01"]
test = sales[sales["day"] >= "2018-01-01"]

res = sp.metalearner(train, y="sales", treat="discounts",
                     covariates=["month", "weekday", "is_holiday",
                                 "competitors_price"],
                     learner="r")
res.estimate                       # the partially linear coefficient
cate_test = sp.predict_cate(res, test)
```

`cate` is the effect of one more unit of the treatment at each covariate
value, under an outcome that is linear in the treatment given the
covariates. `estimate` is the average of that function weighted by the
conditional variance of the treatment, which is the coefficient
`sp.dml(model='plr')` reports. The other learners compare a treated arm
with a control arm and stay binary.

### Evaluating a ranking

The test set of that chapter has randomized discounts, so the effect in any
group of days is the slope of sales on discounts in that group:

```python
curve = sp.cate_gain_curve(test.assign(cate=cate_test),
                           cate="cate", y="sales", treat="discounts")
curve.auc           # area under the normalized cumulative gain curve
curve.by_quantile   # effect within each decile of the prediction
curve.plot()

sp.cate_gain_curve(test.assign(cate=cate_test), cate="cate", y="sales",
                   treat="discounts", n_boot=500, cluster="rest_id",
                   seed=0).auc_ci      # bootstrap interval, restaurants resampled
```

A ranking that carries no information has an area near zero. The curve is
only meaningful on data where the treatment was randomized. For a 0/1
treatment on observational data use `sp.cate_eval`, which builds doubly
robust scores.

### Switchback experiments

A switchback experiment turns a treatment on and off for one unit over
time. Chapter 10 follows Bojinov, Simchi-Levi and Zhao (2023). Decide the
order of the carryover effect `m`, get the design, run it, analyse it:

```python
plan = sp.switchback_design(120, m=2, seed=0)   # columns: period, randomize, treat
# ... run the experiment with plan["treat"], record the outcome as plan["y"] ...
res = sp.switchback(plan, y="y", treat="treat", m=2, design="randomize")
res.estimate, res.ci, res.model_info["randomization_pvalue"]
```

The estimator weights each period whose last `m + 1` assignments were all
treated (or all control) by the inverse probability of that happening under
the design. It needs no model of the outcome. Under the optimal design the
standard error is the paper's conservative one, so the interval is wide by
construction; the randomization p-value tests the sharp null of no effect
at all and is exact. On the book's two data sets:

```python
every = pd.read_csv("data/sb_exp_every.csv")
sp.switchback(every, y="delivery_time", treat="d", m=2, design="every").estimate
# -7.4264

opt = pd.read_csv("data/sb_exp_opt.csv")
res = sp.switchback(opt, y="delivery_time", treat="d", m=2, design="rand_points")
res.estimate, res.ci        # -9.9210, (-18.49, -1.35)
```

The design you pass must be the one that generated the assignment. If the
assignment changes at a period that is not a randomization point of that
design, the function stops.

The first data set flips the coin every period, which is not the optimal
design for `m = 2`, and the paper has no variance estimator for it. If you
can bound the outcome, `outcome_bound=` gives the largest standard error
the estimator can have under that design and a Chebyshev interval:

```python
sp.switchback(every, y="delivery_time", treat="d", m=2, design="every",
              outcome_bound=14).ci        # (-43.8, 28.9): valid, and wide
```

That is the price of flipping too often. The optimal design exists to
avoid it.

### Choosing which cities to treat

The first half of chapter 10 asks which cities to treat so that the
experiment speaks for the whole market. `sp.synth_experimental_design`
with `criterion='population'` searches for a treated set whose weighted
average tracks the population average of the outcome before the
experiment, while the remaining cities can still build a synthetic control
for that same average:

```python
pre = pd.read_csv("data/online_mkt.csv").query("post == 0")
design = sp.synth_experimental_design(
    pre, unit="city", time="date", outcome="app_download", k=5,
    criterion="population", population_weights="population",
    n_search=1000, random_state=0)
design.selected              # the cities to treat
design.weights["treated"]    # their weights; design.weights["control"] for the rest
print(design.summary())
```

With few enough candidate sets (at most `n_search`) all of them are tried
and the answer is the optimum; `design.diagnostics['global_optimum']` says
which case you are in. Otherwise the best random set is improved by
exchanging one city at a time, and different seeds can end at different
sets of similar quality. The weights have no intercept; with units of very
different size, use a per-capita outcome.

## Where the numbers differ from the book

| chapter | quantity | book | StatsPAI | why |
| --- | --- | --- | --- | --- |
| 5 | doubly robust ATE | 0.271158 | 0.271163 | scikit-learn's logit stops at its default tolerance |
| 6 | area under the gain curve | 181.75 | 181.91 | the book's steps include one extra row |
| 9 | synthetic control ATT | 0.003327 | 0.003347 | cvxpy at default accuracy; weights not on the simplex to 1e-5 |
| 9 | synthetic DiD | 0.004086 | 0.003981 | the book omits the ridge penalty and the intercept of `synthdid` |
| 11 | 2SLS standard error | 80.529 | 80.537 | `n / (n - k)`, as in Stata `ivregress, small` and R `ivreg` |
| 11 | fuzzy discontinuity, global lines | 732.85 | 727.34 | 319 accounts sit exactly at the threshold; `rdrobust` counts them as above it |

`sp.rdrobust` without `h=` picks a bandwidth and fits local lines, which is
what a paper would report; the global fit above is only there to show where
the book's number comes from.

## What a paper written today would add

- Cross-fitting in the doubly robust estimators: `sp.aipw` does it by
  default, the book's version does not.
- For staggered adoption, an estimator that does not use already-treated
  units as controls, an event study with a uniform band
  (`sp.uniform_bands`) and a sensitivity analysis for parallel trends
  (`sp.honest_did`).
- With few treated units, inference that does not rely on their number:
  `sp.did_few_treated`. Chapter 8's first example has nine treated cities.
- For an instrument, a weak-instrument-robust interval:
  `sp.anderson_rubin_ci`.
- For a discontinuity, local polynomial estimation with robust
  bias-corrected inference (`sp.rdrobust`) and a density test
  (`sp.rddensity`) in place of global lines and a histogram.
