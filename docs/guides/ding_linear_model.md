# Ding, *Linear Model and Extensions*, in StatsPAI

Peng Ding's lecture notes (arXiv:2401.00649, first posted 2024-01-01) are
a second course in regression: least squares and its robust inference,
model selection and shrinkage, transformations, then binary, categorical
and count outcomes, clustered data, quantiles and survival times. Its
replication files are 24 R programs built on `lm`, `glm`, `MASS`, `gee`,
`quantreg` and `survival`.

This guide maps each chapter to the StatsPAI call that gives the same
number, says where StatsPAI follows a different convention and why, and
lists what a paper written today would do differently.

The files are on the Harvard Dataverse. They are not redistributed with
StatsPAI. The examples below read them from `files/`.

## Chapter by chapter

| chapter | the book runs | StatsPAI |
| --- | --- | --- |
| 1, 5 | `lm`, `glm(binomial)` | `sp.regress`, `sp.logit` |
| 5 | confidence and prediction intervals | `fit.predict(new, what='confidence' / 'prediction')` |
| 5, 8 | `linearHypothesis`, `anova(small, full)` | `sp.test(fit, ["x1=0", "x2=0"])`; `sp.lrtest(small, full)` |
| 6, 24 | `hccm` / `vcovHC`, HC0 to HC4 | `sp.regress(robust='hc0' ... 'hc4')` |
| 11 | `hatvalues`, `rstandard`, `rstudent`, `cooks.distance` | `sp.estat(fit, "leverage")` |
| 12 | leave-one-out prediction intervals | `sp.estat(fit, "leverage")["loo_residuals"]`, `["loo_interval_halfwidth"]`, `["press"]` |
| 13 | `leaps::regsubsets` | `sp.best_subset(df, y, x, criterion='bic')` |
| 14, 15 | `MASS::lm.ridge`, `glmnet` | `sp.ridge`; `sp.lasso_select` |
| 16 | `MASS::boxcox`; polynomial terms | `sp.boxcox`; `I(exper^2)` in any formula |
| 19 | `lm(weights=)`, feasible GLS, Goodman's regression | `sp.regress(weights=)`; `"t ~ 0 + x + I(1 - x)"` |
| 19 | `KernSmooth::locpoly` | `sp.lpoly(kernel='gaussian', degree=1)` |
| 20 | logit, probit, cloglog, cauchit | `sp.glm(family='binomial', link=...)` |
| 20 | `predict(se.fit=TRUE)`, `margins` | `sp.logit(...).predict(new, what='confidence')`; `sp.margins` |
| 21 | `nnet::multinom`, `MASS::polr`, `mlogit` | `sp.mlogit`, `sp.ologit`, `sp.clogit`; `.predict()` for probabilities |
| 22 | Poisson, `glm.nb`, `pscl::zeroinfl` | `sp.poisson`, `sp.nbreg`, `sp.zip_model`, `sp.zinb` |
| 24 | `sandwich` after `glm` | `robust='hc0'` in the same functions |
| 25 | `gee::gee` | `sp.gee(formula, df, id=, family=, corstr=)` |
| 26 | `quantreg::rq`, weights, clustered bootstrap | `sp.qreg(quantile=, weights=, vce=, cluster=)` |
| 27 | `survfit`, `survdiff`, `coxph` | `sp.kaplan_meier(conf_type='log')`, `sp.logrank_test`, `sp.cox` |

## A few of them in full

```python
import numpy as np
import pandas as pd
import statspai as sp

# Chapter 6: five heteroskedasticity-robust standard errors
lalonde = pd.read_csv("files/lalonde.txt", sep=r"\s+")
rhs = " + ".join(c for c in lalonde.columns if c != "re78")
table = pd.DataFrame(
    {kind: sp.regress(f"re78 ~ {rhs}", lalonde, robust=kind).tvalues
     for kind in ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hc4"]}
)

# Chapter 13: the best model of every size, then pick by BIC
boston = pd.read_csv("files/_statspai/BostonHousing.csv")
best = sp.best_subset(boston, "medv", [c for c in boston if c != "medv"])
best.history[["size", "rss", "bic"]]

# Chapter 16: which power of unemployment duration is closest to linear?
penn = pd.read_csv("files/pennbonus.txt", sep=r"\s+")
bc = sp.boxcox("duration ~ " + " + ".join(c for c in penn if c != "duration"), penn)
bc.lambda_, bc.ci        # 0.317, (0.291, 0.342): neither the log nor the level

# Chapter 25: a cluster-randomised trial, 107 villages
hyg = pd.read_csv("files/_statspai/hygaccess_analysis.csv")
fit = sp.gee("y ~ C(z)", hyg, id="vid", family="binomial", corstr="exchangeable")
fit.summary()
fit.model_info["se_model"]     # what you would report if you trusted the working correlation
fit.model_info["corr_alpha"]   # 0.063

# Chapter 26: weighted quantile regression, returns to schooling in 1980
census = pd.read_stata("files/census80.dta")
for tau in (0.1, 0.5, 0.9):
    q = sp.qreg(census, "logwk ~ educ + exper + exper2 + black",
                quantile=tau, weights="perwt", vce="ker")

# Chapter 27: Cox model with site strata and a robust variance
combine = pd.read_csv("files/combine_data.txt", sep="\t")
cox = sp.cox("futime ~ NALTREXONE*THERAPY + AGE + C(GENDER) + T0_PDA",
             combine, event="relapse", strata="site", robust="hc0")
```

## Where the numbers differ from R, and why

StatsPAI follows Stata where Stata and R disagree on a convention. Each
row below is exact, not approximate: the tests reproduce R's number from
ours by the stated rule.

| what | R | StatsPAI default | to get R's number |
| --- | --- | --- | --- |
| information matrix of a GLM with a non-canonical link | expected | observed (Stata `glm`) | `information='expected'` |
| `vce='robust'` in logit, Poisson, ... | HC0 | HC0 times N/(N-1) (Stata) | `robust='hc0'` |
| clustered Cox | bare sandwich | times G/(G-1) (Stata `stcox`) | multiply by `sqrt((G-1)/G)` |
| GEE sandwich | bare (R `gee`) | bare | Stata `xtgee` is ours times `sqrt(G/(G-1))` |
| GEE scale for binomial and Poisson | estimated | estimated | Stata fixes it at 1, `scale=1` |
| AIC of a linear model | counts the error variance | does not (Stata) | add 2 |
| negative binomial standard errors | theta held fixed (`glm.nb`) | joint observed information (Stata `nbreg`) | differ by 2 to 4% |
| Kaplan-Meier interval | symmetric for log S | symmetric for S until 1.40, then for log(-log S) as in Stata | `conf_type='log'` |

One difference is not a convention: `KernSmooth::locpoly` bins the data
before smoothing and is an approximation; `sp.lpoly` is exact.

For an AR(1) working correlation R and Stata estimate the parameter with
different moments. `sp.gee(corstr='ar1')` is Stata's `xtgee, corr(ar 1)`
and `sp.gee(corstr='ar-m')` is R `gee`'s `"AR-M"`. They coincide on a
balanced panel and differ when cluster sizes do.

## What a paper written today would add

- **Few clusters.** The GEE and clustered sandwiches assume many clusters.
  The mouse experiment of chapter 25 has 14. `sp.gee` warns below 40;
  `sp.wild_cluster_bootstrap` on the pooled model, or a CR2 variance with
  Satterthwaite degrees of freedom (`sp.regress(vce='cr2', cluster=,
  dfadjust=True)`), are the current answers for linear models.
- **Selection is not inference.** Chapter 13 shows Freedman's paradox by
  simulation. The t-statistics of a model chosen by `sp.best_subset` or
  `sp.stepwise` ignore the search. For a coefficient of interest with many
  candidate controls, use post-double-selection (`sp.rlasso_effect`) or
  double machine learning (`sp.dml`).
- **Transformed outcomes.** Chapter 16 picks a power by Box-Cox and
  chapter 6 takes `log(y + 1)`. Neither identifies an effect on the scale
  of `y`, and `log(y + 1)` depends on the units of `y`. For a non-negative
  outcome with zeros, Poisson regression with a robust variance
  (`sp.poisson(robust='hc0')` or `sp.ppmlhdfe`) estimates proportional
  effects on the mean under weaker assumptions.
- **Marginal effects, not coefficients.** For the logit, probit and count
  models report `sp.margins(fit, df)`. The coefficients of the four links
  of chapter 20 are on different scales and the average marginal effects
  are nearly identical.
- **Proportional hazards.** `sp.cox(...).ph_test()` runs the Schoenfeld
  residual test the chapter does not. When it fails, the hazard ratio is a
  weighted average that depends on the censoring pattern; restricted mean
  survival time or `sp.causal_survival_forest` give estimands that do not.
- **Quantile regression is about the conditional distribution.** The
  coefficients of chapter 26 are not effects on the unconditional quantiles
  of wages. For those use `sp.rifreg` or the distributional tools in
  `sp.qte`.

## Reproducing the check

```bash
export STATSPAI_DING_LM_DIR=/path/to/unzipped/dataverse_files
Rscript tests/external_parity/ding_linear_model_reference.R
pytest tests/external_parity/test_ding_linear_model.py
```

The same functions are tested without the book's data in
`tests/reference_parity/test_linear_model_extensions_parity.py`, against R
and Stata 18 on a synthetic file. What the pass found is written up in
`docs/dev/2026-10-05-ding-linear-model-review.md`.
