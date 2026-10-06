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
| 20 | `fitstat(iv, "ivf1")`, `iv$iv_sargan`, Wu-Hausman | `sp.estat(fit, "firststage")`, `"overid"`, `"endogenous"` |
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
`small=False`; `sp.from_r("lmrob(...)")` writes that call. The book then
feeds the robustness weights to a weighted regression with clustered
standard errors:

```python
comp["w"] = rob.model_info["weights"].reindex(comp.index).fillna(0)
sp.regress(formula, comp[comp["w"] > 0], weights="w", cluster=["gvkey", "fyear"])
```

Winsorizing uses the percentile definition of Stata's `winsor2` and R's
`quantile(type = 2)`, which are the same. Winsorize regressors, not the
outcome: selecting on the dependent variable biases the slope.

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
res.tests     # cross-sectional t, Patell, BMP, and the adjusted versions
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
