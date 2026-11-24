# Gow and Ding, *Empirical Research in Accounting*, in StatsPAI

Ian Gow and Tony Ding's book (Chapman and Hall/CRC, 2024; free online) is
a course in the methods of capital-markets and financial accounting
research. It teaches by simulation and by replicating published papers,
in R, with `fixest`, `plm`, `sandwich`, `robustbase`, `MatchIt`, `rdrobust`
and its own package `farr`.

This guide maps the statistical calls of each chapter to the StatsPAI call
that gives the same number, says where StatsPAI follows a different
convention, and lists what a paper written today would add. Most chapters
build their data from CRSP and Compustat through WRDS; StatsPAI starts
where the data frame is ready.

## Chapter by chapter

| chapter | the book runs | StatsPAI |
| --- | --- | --- |
| 3 Regression fundamentals | `lm(y ~ treat * post + factor(grade))` | `sp.regress("y ~ treat * post + factor(grade)", df)` |
| 3 | `feols(y ~ I(post * treat) \| grade + id)` | `sp.feols("y ~ I(post * treat) \| grade + id", df)` |
| 5 Statistical inference | `vcovHC(fm, type = "HC1")` | `sp.regress(..., robust="hc1")` |
| 5 | `plm::vcovNW` on a pooled panel | `sp.regress(..., robust="hac", hac_lags=1, hac_panel=("firm", "year"))` |
| 5 | `pmg(y ~ x, index = "year")` | `sp.fama_macbeth("y ~ x", df, time="year")` |
| 5 | `NeweyWest` on the yearly coefficients | `sp.fama_macbeth(..., lags=1)` |
| 5 | `feols(vcov = ~ firm)`, `~ year`, `~ year + firm` | `sp.regress(..., cluster="firm")`, `cluster=["year", "firm"]` |
| 10 to 14 Event studies | `farr::get_event_cum_rets`; market-adjusted returns around announcements | `sp.abnormal_returns(returns, events, market="mkt", event_window=(-1, 1))` |
| 15 Accruals | `ntile(x, 10)`; `linearHypothesis(fm, "d1 = d10")` | `pd.qcut`; `sp.test(fit, "C(d)[1] = C(d)[10]")` |
| 16 Earnings management | `binom.test(x, n, p)` | `sp.bitest(successes=x, n=n, p=p)` |
| 19 Natural experiments | DiD, post-only, change and ANCOVA estimators | `sp.regress` with the same formulas; `sp.ancova` |
| 20 Instrumental variables | `feols(y ~ 1 \| X ~ z1 + z2 + z3)` | `sp.ivreg("y ~ (X ~ z1 + z2 + z3)", df)` |
| 20 | `fitstat(iv, "ivf1")`, `iv$iv_sargan`, Wu-Hausman | `sp.estat(fit, "firststage")`, `"overid"`, `"endogenous"`, after `sp.ivreg` or `sp.feols` |
| 21 Panel data | `feols(y ~ post \| gvkey + year)` and event-time dummies | `sp.feols`; see "today" below |
| 22 Regression discontinuity | `rdrobust(y, x, c = 75, fuzzy = d, masspoints = "off")` | `sp.rdrobust(df, y=, x=, c=75, fuzzy=, masspoints="off")` |
| 22 | `rdplot` | `sp.rdplot` |
| 23 Beyond OLS | probit, logit, Poisson and their marginal effects | `sp.probit`, `sp.logit`, `sp.poisson`; `sp.margins(fit, df)` |
| 24 Extreme values | `farr::winsorize(x, 0.01)`, `truncate(x, 0.01)` | `sp.winsor(df, ["x"], cuts=(1, 99))`, `trim=True` |
| 24 | `cooks.distance(fm)` | `sp.estat(fit, "leverage")["cooks_d"]` |
| 24 | `lmrob(f, method = "MM", control = lmrob.control(tuning.psi = 3.4437))` | `sp.robreg(f, df)`; exact match with `tuning=3.4437, tuning_s=1.54764, small=False` |
| 24 | Poisson with `vcovHC(type = "HC1")` | `sp.poisson(f, df, robust="hc1")` |
| 24 | impact threshold for a confounding variable | `sp.itcv(fit, "wbflag")` |
| 25 Matching | `matchit(treat ~ x, caliper = 0.03)` | `sp.match(df, y=, treat=, covariates=, caliper=0.03, caliper_scale="sd", replace=False)` |
| 26 Prediction | `farr::auc`, `farr::ndcg(score, y, k = 0.01)` | `sp.auc(y, score)`, `sp.ndcg(y, score, k=0.01)` |
| 26 | `cv.glmnet(family = "binomial")` | `sp.rlassologit`; `sp.lasso_select` |

`sp.from_r` translates the calls in the right-hand column from the R
line, including `feols(fml, ~ a + b, data = d)` with its clustering:

```python
import statspai as sp

sp.from_r('feols(ta ~ big_n + cfo | sic2 + fyear, ~ gvkey + fyear, data = comp)')["python_code"]
# "sp.feols('ta ~ big_n + cfo | sic2 + fyear', data=df, vcov={'CRV1': 'gvkey + fyear'})"
```

## Seven standard errors for one coefficient

Chapter 5 fits `y ~ x` on a simulated panel with firm and year effects in
both regressor and error, and reports the coefficient with seven standard
errors. In StatsPAI:

```python
ols   = sp.regress("y ~ x", df)
white = sp.regress("y ~ x", df, robust="hc1")
nw    = sp.regress("y ~ x", df, robust="hac", hac_lags=1, hac_panel=("firm", "year"))
fm    = sp.fama_macbeth("y ~ x", df, time="year")
cl_i  = sp.regress("y ~ x", df, cluster="firm")
cl_t  = sp.regress("y ~ x", df, cluster="year")
cl_2  = sp.regress("y ~ x", df, cluster=["year", "firm"])
```

Each equals the book's column to 1e-15. Two things to know.

`robust="hac"` without `hac_panel` reads the rows as one time series. On
stacked panel data that pairs the last year of one firm with the first
year of the next. `hac_panel=` takes the lags within firms, at exact time
distances, in any row order.

Fama-MacBeth standard errors are robust to correlation across firms within
a year and not to a firm effect that persists across years. That is the
chapter's point (after Gow, Ormazabal and Taylor 2010), and `lags=` does
not rescue it. Cluster by firm and year when both kinds of dependence are
present.

## When two-way clustering warns

With few clusters in one dimension the two-way covariance matrix can fail
to be positive semi-definite. The accruals regression of chapter 24 has 21
years and 85 coefficients, and 46 of the matrix's eigenvalues are negative.
StatsPAI sets them to zero, as `fixest` and `reghdfe` do, and warns:

```
RuntimeWarning: Two-way clustered covariance was not positive
semi-definite (46 negative eigenvalues); they were set to zero ...
```

The adjusted standard errors depend on which year is the base category.
StatsPAI takes the base from the estimation sample; R keeps an unused
factor level as the base unless it is dropped, which is why the book's
numbers differ from `sp.regress` by a few percent while `fixest` with
`droplevels()` agrees to twelve digits. The warning is a sign that the
year dimension is too small for the asymptotics; a wild cluster bootstrap
(`sp.wild_cluster_bootstrap`) is the usual remedy.

`sp.feols` applies the same adjustment, with the same warning, so it and
`sp.regress(cluster=[a, b])` report the same standard errors. pyfixest
called directly does not: a negative variance there is a missing standard
error.

## Extreme values

The chapter compares doing nothing, winsorizing, truncating, dropping by
Cook's distance and robust regression, and recommends the last.

```python
rob = sp.robreg("ta ~ big_n + cfo + size + lev + mtb", comp)   # MM, 85% efficiency
rob.model_info["weights"]        # 1 = as in least squares, 0 = ignored
rob.model_info["n_zero_weight"]  # 915 of 9,036 on the book's data
```

Defaults follow Stata's `robreg mm`. To reproduce `lmrob` exactly pass
`tuning=4.685061` (or the chapter's 3.4437), `tuning_s=1.54764` and
`vce="avar1"`, which is `lmrob`'s own covariance;
`sp.from_r("lmrob(...)")` writes that call. The book then
feeds the robustness weights to a weighted regression with clustered
standard errors:

```python
comp["w"] = rob.model_info["weights"].reindex(comp.index).fillna(0)
sp.regress(formula, comp[comp["w"] > 0], weights="w", cluster=["gvkey", "fyear"])
```

Winsorizing uses the percentile definition of Stata's `winsor2` and R's
`quantile(type = 2)`, which are the same. Winsorize regressors, not the
outcome: selecting on the dependent variable biases the slope.

Winsorizing within year, as the book does with
`group_by(fyear) |> mutate(across(x, winsorize))`, is `by=`:

```python
sp.winsor(comp, ["ta", "cfo", "lev"], cuts=(1, 99), by="fyear", replace=True)
```

It agrees with `farr::winsorize` to the last digit, with two exceptions,
and each one raises a warning. An infinite value, say from a division by
zero, is left out of the cut-offs and left as it is; `farr::winsorize`
counts it in the quantile, which moves the cut-offs, and then pulls it in.
A row with a missing `fyear` is left alone, as in Stata's `winsor2, by()`;
`group_by()` makes the missing years a group of their own and winsorizes
it. Set infinite values to missing and drop rows without a year first, and
the two agree.

When the sign of an infinite value carries information, pass
`include_infinite=True` to do what `farr` does. `-inf` and `+inf` are then
the smallest and largest values of the variable: they count in the
percentile ranks and are pulled in to the cut-offs, or set to missing with
`trim=True`. NaN stays missing.

```python
sp.winsor(comp, ["mtb"], by="fyear", replace=True, include_infinite=True)
```

Stata and R compute the same percentile with different arithmetic, and at
a few uncommon percentiles they land one order statistic apart. With 100
values the 7th percentile is 7.5 in Stata and 8 in R. The default
`method="stata"` follows `winsor2`. Pass `method="r"` to follow
`quantile(type = 2)`, so that `cuts=(7, 93)` equals
`farr::winsorize(x, p_low = 0.07, p_high = 0.93)`. At 1, 2.5, 5, 10, 90,
95, 97.5 and 99 the two are the same.

A Poisson model can be quasi-separated. In the whistleblower regressions
every firm with `mobflag = 1` paid no penalty, so that coefficient has no
finite estimate. `sp.poisson` fits the rest, warns, and lists the
regressor in `model_info["separated_terms"]`; `sp.ppmlhdfe` drops the
separated observations instead.

## Event studies

Chapters 10 to 14 replicate Fama, Fisher, Jensen and Roll, Ball and Brown,
Beaver and the post-earnings announcement drift, each an event study on
CRSP returns. With returns in long format and a table of events:

```python
res = sp.abnormal_returns(
    returns, events,              # id, date, ret, mkt  |  id, event_date
    model="market", market="mkt",
    event_window=(-1, 1), estimation_window=(-250, -11),
)
res.events    # one row per event: CAR, standard error, t, p
res.aar       # average abnormal return by relative day, and its running sum
res.tests     # cross-sectional t, Patell, BMP, adjusted versions, rank tests
```

`model="market_adjusted"` is the book's own choice (return minus market
return, nothing estimated); `"mean_adjusted"` and `"factor"` are the other
two standard models. Day 0 is the first trading day on or after the event
date.

Which test to read. `patell` assumes the event leaves the variance of
returns unchanged, which earnings announcements do not (that is Beaver's
finding); `bmp` and the cross-sectional test allow for it. When event
dates cluster in calendar time, as with a regulatory change that hits
every firm on one day, abnormal returns are correlated across firms and
all three over-reject; `adj_patell` and `kp` correct for the average
correlation.

Two rows make no distributional assumption. `grank` is the generalized
rank test [@kolari2011nonparametric], which ranks each CAR against that
security's own estimation-window returns and holds up under event-induced
variance and thick tails. `sign_rank` is Wilcoxon's signed-rank test on
the CARs, for a median of zero. With few events its p-value cannot be
small: fifteen CARs of one sign give 0.0007 at best.

This is not `sp.event_study`, which draws the event-time coefficients of a
difference-in-differences design.

## The impact threshold for a confounding variable

```python
fit = sp.regress("ln_firmpenalty ~ wbflag + " + controls, cmsw)
out = sp.itcv(fit, "wbflag")
out["itcv"], out["r_threshold"]     # 0.092; correlations of 0.30 with y and x
out["benchmark_control"], out["benchmark"]   # 'lnmktcap', 0.091
out["percent_bias"], out["rir"]     # 52.6% of the estimate; 346 of 658 firms
```

An omitted variable would have to be correlated at 0.30 with both the
penalty and the whistleblower indicator, given the controls, to move the
estimate to the 5% threshold. The strongest observed control, firm size,
has an impact of 0.091, about the same. The chapter's caution applies:
the number describes how much confounding it would take, not whether it
is there. `sp.sensemakr` and `sp.oster_bounds` bound the coefficient
itself.

## The Python Polars edition

Ian Gow and Martien Lubberink are writing a Python edition of the book
(<https://iangow.github.io/era_pl_book/>) with Polars, statsmodels,
pyfixest, linearmodels, rdrobust and causalml, and the helper package
`era_pl` for its data. StatsPAI accepts the Polars frames directly. Each
call below gives the book's number to machine precision unless the
second column says otherwise.

| the book runs | StatsPAI | note |
| --- | --- | --- |
| `smf.ols(f, df).fit()`, `fit(cov_type="HC1")` | `sp.regress(f, df)`, `robust="hc1"` | |
| `cov_type="cluster"` with `use_correction`, `df_correction` | `sp.regress(f, df, cluster="firm")` | statsmodels reports normal p-values here, StatsPAI t(G - 1); `fit.normal_reference()` gives the normal ones |
| `PanelOLS(...).fit(cov_type="clustered", cluster_entity=True, cluster_time=True, group_debias=False)` | `sp.feols(f, df, vcov={"CRV1": "firm + year"}, ssc=pf.ssc(G_adj=False))` | equal to 1e-4: linearmodels scales by n / (n - k), fixest by (n - 1) / (n - k). The default two-way SE (fixest, Stata) is larger by sqrt(G / (G - 1)) |
| `FamaMacBeth.from_formula(...).fit(debiased=True)`; `cov_type="kernel", bandwidth=1` | `sp.fama_macbeth(f, df, time="year")`; `lags=1` | |
| `pf.feols(...)` with any `vcov=` and `ssc=` | `sp.feols` with the same arguments | two-way clustering on a design whose covariance is not positive semi-definite: StatsPAI applies the eigenvalue fix of R's fixest and warns, pyfixest does not, so the SEs differ (chapter 19's by-year figure) |
| `pf.feols("y ~ C(t, Treatment(reference=-1)) \| gvkey + year")` with `t = -inf` for never-treated firms | the same call | accepted since 2026-11-19 |
| `pf.feols("y ~ 1 \| X ~ z_1 + z_2 + z_3")`, `iv_sargan`, first-stage F | `sp.ivreg("y ~ (X ~ z_1 + z_2 + z_3)", df)`; `diagnostics` holds the Sargan and first-stage F | |
| `sm.Probit`, `sm.Logit`, `sm.Poisson`; average marginal effect | `sp.probit`, `sp.logit`, `sp.poisson`; `sp.margins(fit, data=df)` | |
| `smf.glm(f, family=Poisson()).fit(cov_type="HC1")` | `sp.poisson(f, df, robust="hc1")` | StatsPAI's HC1 includes n / (n - k), as R and Stata do. Chapter 24 multiplies statsmodels' matrix by that factor, chapter 23 does not |
| `sm.RLM(..., M=TukeyBiweight(c=3.4437))` | `sp.robreg(f, df, method="m", psi="bisquare", tuning=3.4437, vce="huber")` | coefficients equal to 3e-5: StatsPAI and R's `rlm` normalise the MAD by 0.6745, statsmodels by 0.674490 |
| `get_influence().cooks_distance` | `sp.influence_measures(sp.regress(f, df))["cooksd"]` | |
| `.era.winsorize()`, `.era.truncate()` | `df.pipe(sp.winsor, vars=cols, cuts=(1, 99), replace=True)`, `trim=True` | a Polars frame comes back as Polars with its other columns untouched; bit-identical on finite data with nulls, also with `by=` against `.over()` |
| the chapter 24 ITCV function | `sp.itcv(fit, "wbflag", alpha=a)` | |
| `rdrobust(y, x, c=75, fuzzy=d, masspoints="off")` | `sp.rdrobust(df, y=, x=, c=75, fuzzy=, masspoints="off")` | equal to R rdrobust 4.0. The Python rdrobust 2.1.1 port differs from R by about 1e-4 in the bandwidth when the running variable has ties |
| `NearestNeighborMatch(caliper=0.03 / sd, replace=False, shuffle=False)` on data sorted by the score | `sp.match(df, y=, treat=, covariates=, pscore="pscore", replace=False, caliper=0.03, caliper_scale="raw", m_order="data")` | the same matched set. With `shuffle=True` the order is random and differs between tools |
| `ETable(fms, drop=["fyear", "ppe"], signif_code=[0.01, 0.05, 0.10])` with t-statistics | `sp.regtable(*fms, drop=["fyear", "ppe"], se_type="t", fmt=3)` | `keep=` / `drop=` take exact names, globs or regular expressions |
| `rdplot(df, y=, x=, cutoff=75, masspoints="off")` | `sp.rdplot(df, y=, x=, c=75, masspoints="off")` | bins equal R `rdplot` |
| `stats.binomtest(k, n, p)` | `sp.bitest(successes=k, n=n, p=p)` | |

Loops that fit one formula group by group (a market model per security,
a Jones model per industry-year) should pass `coef_only=True`:
`sp.regress(f, g, coef_only=True).params`. It fits designs with no
residual degrees of freedom, and such fits are not counted by the
specification-search warning. A grouped Polars expression or
`np.linalg.lstsq` is still 10 to 50 times faster per fit, which matters
inside a simulation.

Three things to know when moving the code over. A Polars `Categorical`
takes its base level from the order of first appearance (in pyfixest and
statsmodels too); R sorts the levels, so fix the base with `pl.Enum([...])`
or `C(x, Treatment(reference=...))` where it matters. A text column on the
right-hand side is a factor in every StatsPAI estimator that can expand
it and is refused by the others; it is never fitted as a number. And
chapter 26's logit is scikit-learn's penalised fit, so `sp.logit` (the
maximum-likelihood fit of the R edition) gives slightly different scores.

In chapter 24 the Poisson regression of `firmpenalty` has no finite
coefficient on `mobflag`: the 13 cases with `mobflag = 1` all have a zero
penalty. statsmodels reports a large negative number without comment;
`sp.poisson` warns and lists the term in `model_info["separated_terms"]`.
The coefficient on `wbflag` is identified and the two agree to 2e-13.

## What a paper written today would add

- **Staggered adoption.** Chapter 21 estimates the effect of the inevitable
  disclosure doctrine by two-way fixed effects and discusses why the
  estimate is fragile. `sp.bacon_decomposition` shows which comparisons
  the number is made of; `sp.callaway_santanna`, `sp.sun_abraham` and
  `sp.did_imputation` estimate the effect without using already-treated
  firms as controls; `sp.honest_did` reports how far the conclusion
  survives violations of parallel trends.
- **Few clusters.** `sp.wild_cluster_bootstrap` when years or industries
  number in the tens.
- **Regression discontinuity.** `sp.rddensity` for manipulation of the
  running variable and `sp.rd_robustness_table` for bandwidth and
  specification checks, next to `sp.rdrobust`.
- **Matching.** `sp.balance_table` and `sp.ebalance` before trusting a
  matched comparison of Big Four and other auditors.

## Evidence

Every number in the table that can be computed without WRDS is checked in
`tests/external_parity/test_gow_ding_accounting.py` against R 4.5.2 on the
book's data, and the functions added for this book are tested on committed
synthetic data against R and Stata 18 in
`tests/reference_parity/test_accounting_research_parity.py`. What was
found, and what remains open, is in
`docs/dev/2026-10-06-gow-ding-accounting-review.md`.
