# Complex-survey analysis (NHANES, BRFSS, DHS, …)

> **Most population-health data are not simple random samples.** NHANES,
> BRFSS, DHS, MEPS and their kin use stratification, clustering, and
> unequal selection probabilities. Ignoring the design biases point
> estimates *and* understates standard errors. StatsPAI's `sp.survey`
> family mirrors R's `survey` package: declare the design once, then
> every estimator returns **design-corrected** standard errors, design
> effects (DEFF), and design degrees of freedom.

Scope note: the survey estimators are API-stable and reproduce the
standard linearization (Taylor-series) variance, but are **not yet
parity-certified** against R's `survey::svyglm`. Validate before
publication and read `sp.describe_function(name)["limitations"]`.

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

## 4. Calibration and raking

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

## 5. Why this matters for public health

Reporting an unweighted mean from NHANES, or a model SE that ignores the
PSU clustering, is one of the most common reproducibility errors in
applied public-health analysis. Declaring the design with `sp.svydesign`
and routing every estimate through the `sp.svy*` functions makes the
correction automatic and auditable.

A known gap: design-based **survival** analysis (a survey-weighted Cox
model) is not yet wired into the survey family — for now, extract the
calibrated weights and pass them to a weighted analysis manually, and
validate against R's `survey::svycoxph`.

## Where to next

- [Public health & epidemiology overview](public_health.md)
- [Epidemiological measures](epi_measures.md)
- [Survival analysis](survival_ph.md)
