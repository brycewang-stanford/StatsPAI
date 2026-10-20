# Complex-survey analysis (NHANES, BRFSS, DHS, …)

> **Most population-health data are not simple random samples.** NHANES,
> BRFSS, DHS, MEPS and their kin use stratification, clustering, and
> unequal selection probabilities. Ignoring the design biases point
> estimates *and* understates standard errors. StatsPAI's `sp.survey`
> family mirrors R's `survey` package: declare the design once, then
> every estimator returns **design-corrected** standard errors, design
> effects (DEFF), and design degrees of freedom.

Scope note: `svymean` / `svytotal` / `svyglm`, calibration, and the
`design=` option of the estimators in section 4 reproduce R `survey`
(`svydesign` / `svyrecvar` / `svyglm` / `svycoxph`) to about 1e-14 on the
frozen fixtures under `tests/reference_parity/test_survey_*`. Read
`sp.describe_function(name)["limitations"]` for each estimator's scope.

---

## 1. Declare the design once

```python
import statspai as sp

design = sp.svydesign(
    data=nhanes,
    weights="WTMEC2YR",   # examination weight column
    strata="SDMVSTRA",    # stratification variable
    cluster="SDMVPSU",    # primary sampling unit
    nest=True,            # PSU ids are nested within strata
    # fpc="FPC",          # finite population correction, if available
)
```

Everything downstream takes this `design` object, so the design is
specified in exactly one place and cannot drift between estimates.

---

## 2. Means, totals, and proportions

```python
import statspai as sp

m = sp.svymean(["bmi", "sbp"], design)
print(m.summary())     # Estimate / Std.Err / t / p / 95% CI / DEFF per variable
print(m.estimate, m.std_error, m.deff, m.dof)

tot = sp.svytotal("had_visit", design)   # population total with design SE
```

The reported **DEFF** (design effect) tells you how much the complex
design inflated the variance relative to simple random sampling — a DEFF
of 1.5 means your effective sample size is two-thirds of the nominal
*n*. The **design degrees of freedom** (≈ #PSUs − #strata) drive the
t-based confidence intervals, which is why survey CIs are wider than
naïve ones.

---

## 3. Survey-weighted regression

```python
import statspai as sp

# Linear
fit = sp.svyglm("sbp ~ age + bmi + smoker", design, family="gaussian")
print(fit.summary())

# Logistic (design-based prevalence-odds model)
logit = sp.svyglm("hypertension ~ age + bmi + smoker",
                  design, family="binomial")
print(logit.summary())
```

Coefficients are population-level (weighted) associations; the standard
errors and CIs already account for stratification and clustering.

---

## 4. Causal and regression estimators with `design=`

The estimators below take the same `design` object. The fit uses the
design weights, and every standard error is the Taylor linearisation of
the estimator's influence function over strata, PSUs and the finite
population correction, as R `survey::svyrecvar` computes it; intervals use
the t distribution with the design degrees of freedom (#PSU - #strata),
as Stata `svy` does.

```python
import statspai as sp

design = sp.svydesign(df, weights="pw", strata="stratum", cluster="psu",
                      nest=True)

sp.regress("y ~ d + x1 + x2", df, design=design)        # = svyglm
sp.logit("yb ~ d + x1", df, design=design)              # = svyglm(quasibinomial)
sp.ipw(df, y="y", treat="d", covariates=["x1", "x2"], design=design)
sp.aipw(df, y="y", treat="d", covariates=["x1", "x2"], design=design)
sp.tmle(df, y="y", treat="d", covariates=["x1", "x2"], design=design)
sp.dml(df, y="y", treat="d", covariates=["x1", "x2"], design=design)
sp.cox("time ~ d + x1", data=df, event="status", design=design)  # = svycoxph
sp.kaplan_meier(df, duration="time", event="status", group="d",
                conf_type="log", design=design)
```

Pass the data frame the design was declared on as `data=` (the same rows):
the estimator reads its variables from it and the sampling structure from
the design. Rows dropped for a missing value contribute zero but keep their
PSU, which is domain estimation; do not filter the data first.

What each estimator linearises:

| estimator | influence function | reference |
| --- | --- | --- |
| `regress`, `logit`, `probit` | score rows with the inverse weighted information | R `svyglm` |
| `ipw`, `aipw(se_method='sandwich')` | stacked estimating equations, including the propensity (and outcome) model scores | R: Binder linearisation of the same system, written out by hand |
| `aipw` (default), `tmle`, `dml` | efficient influence function, nuisance fits held fixed | R `tmle` influence curves; DML on folds of whole PSUs |
| `callaway_santanna`, `did_2x2` | unit-level ATT(g, t) influence functions (panel); the DID regression | R `did` `inffunc`; `svyglm` |
| `msm` | outcome-model scores with the stabilised weights held fixed | R `ipw::ipwtm` + `svyglm` |
| `cox`, `kaplan_meier` | weighted score residuals; per-row influence on S(t) | R `svycoxph`; `resid(survfit)` |
| `ltmle` | the influence curve of the longitudinal TMLE | R `ltmle` influence curves |
| `match` (nearest neighbour, ATT / ATC) | the matched-sample regression `y ~ d`, matching weights fixed, matched rows a domain of the full design | MatchIt `s.weights` + `svyglm(y ~ d, subset(design, matched))` |

On panel data (`callaway_santanna`, `msm`) the unit is the sampled
element: its weight, stratum and PSU must be the same in every period.
`sp.dml` forms its cross-fitting folds over whole PSUs. With a
linearisation design, `design=` is refused with `se_method='bootstrap'`
and the multiplier bootstrap (a bootstrap of a complex design needs
replicate weights, section 4b); it is also refused with `weights=` /
`cluster=` (the design carries both). Functions without an influence
function refuse `design=` rather than approximate. `sp.did(design=)`
forwards the design to `did_2x2` / `callaway_santanna`.

## 4b. Replicate weights (BRR, Fay, jackknife, bootstrap)

Public-use files often ship replicate weights instead of strata and PSUs
(ACS, MEPS, CHIS, many national surveys). Declare them with
`sp.svrepdesign`, or build jackknife / BRR weights from a strata / PSU
design with `sp.as_svrepdesign`:

```python
rep = sp.svrepdesign(df, repweights="^repwt", weights="pw", type="BRR")
rep = sp.svrepdesign(df, repweights="^repwt", weights="pw", type="Fay", rho=0.5)
rep = sp.as_svrepdesign(design, type="JKn")    # = R as.svrepdesign

sp.svymean("y", rep)
sp.svyglm("yb ~ x1 + x2", rep, family="binomial")
sp.ipw(df, y="y", treat="d", covariates=["x1", "x2"], design=rep)
sp.callaway_santanna(panel, y="y", g="g", t="t", i="id", x=["x"], design=rep_panel)
```

The statistic is computed once with the full-sample weights and once with
each replicate's weights (rows of zero weight left out); its variance is
`scale * sum_r rscales_r (theta_r - c)^2` with `c` the mean of the
replicates, or the full-sample estimate with `mse=True`. `type` sets
`scale` / `rscales` / `mse` as R `svrepdesign` does (BRR `1/R`, Fay
`1/(R (1-rho)^2)`, JK1 `(R-1)/R`, JKn needs `rscales`, bootstrap `1/(R-1)`,
ACS / successive difference `4/R` with `mse=True`); the design df is the
rank of the replicate weights minus one. Accepted by `svymean`, `svytotal`,
`svyglm`, `regress`, `logit`, `probit`, `ipw`, `aipw`, `tmle`, `ltmle`,
`cox`, `did_2x2`, `callaway_santanna` and `sp.did`; the others refuse a
replicate design.

Stata users: `svyset [pw=w], brrweight(...)` / `fay()` / `jkrweight(...,
multiplier())` give the same standard errors (Stata reports #replicates - 1
as the df for BRR, where R and StatsPAI use the rank). Stata `svy
bootstrap` divides by R rather than R - 1: pass `scale=1/R` to reproduce
it. On Cox models with Efron ties a replicate here drops the deleted rows,
as Stata does; R `svycoxph` keeps them with a negligible weight, which
moves its SEs by a few parts in 10,000.

The shared helper is `statspai.survey.linearization.design_vcov`, which
takes per-row contributions `u_i` (with `theta_hat - theta ~= sum u_i`, the
weight inside) and returns their design covariance.

---

## 5. Calibration and raking

When your sample margins do not match known population totals (e.g.
census age × sex distributions), calibrate the weights so they do —
either by raking (iterative proportional fitting) or by
Deville–Särndal linear calibration:

```python
import statspai as sp

# Raking to known marginal distributions
cal = sp.rake(
    data=df,
    margins={
        "age_group": {"18-34": 0.30, "35-54": 0.40, "55+": 0.30},
        "sex": {"M": 0.49, "F": 0.51},
    },
    weight="base_weight",
)
calibrated_weights = cal  # feed back into sp.svydesign(weights=...)

# Linear calibration to known control totals
lin = sp.linear_calibration(df, totals={"age": 4.2e6, "income": 1.1e9},
                            weight="base_weight")
```

---

## 6. Why this matters for public health

Reporting an unweighted mean from NHANES, or a model SE that ignores the
PSU clustering, is one of the most common reproducibility errors in
applied public-health analysis. Declaring the design with `sp.svydesign`
and routing every estimate through the `sp.svy*` functions makes the
correction automatic and auditable.

Design-based survival analysis is `sp.cox(design=)` (R `svycoxph`) and
`sp.kaplan_meier(design=)`; see section 4.

## Where to next

- [Public health & epidemiology overview](public_health.md)
- [Epidemiological measures](epi_measures.md)
- [Survival analysis](survival_ph.md)
