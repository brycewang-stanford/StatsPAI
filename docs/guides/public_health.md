# Public health & epidemiology with StatsPAI

> **StatsPAI is usable as an epidemiology and public-health toolkit, not
> only an econometrics one.** The same import that exposes
> difference-in-differences and synthetic control also exposes the
> association measures, standardization, g-methods, marginal structural
> models, target-trial emulation, survival models, and complex-survey
> estimators that epidemiologists and public-health researchers reach
> for. This guide is the map: it shows which study design points to
> which function, and links to the deeper family guides.

This page is **scope-honest**. StatsPAI's cross-language parity
certification (`sp.list_functions(validation_status="certified")`) is
currently anchored on econometrics benchmarks. Most of the
epidemiology surface below is API-stable but **not yet parity-certified
against R's `survival`, `epiR`, `gfoRmula`, `ipw`, `ltmle`,
`survey`, or `MendelianRandomization`**. Treat point estimates as
correct-by-construction and well tested, but **validate against your
reference package before publication**, and always read
`sp.describe_function(name)["limitations"]` first. See
[Stability tiers](stability.md).

---

## 1. Pick the function by study design

| Study design / question | Reach for | Family guide |
| --- | --- | --- |
| 2×2 table: risk/odds/rate of disease by exposure | `sp.odds_ratio`, `sp.relative_risk`, `sp.risk_difference`, `sp.incidence_rate_ratio`, `sp.number_needed_to_treat`, `sp.attributable_risk` | [Epidemiological measures](epi_measures.md) |
| Confounding by a categorical third variable | `sp.mantel_haenszel`, `sp.breslow_day_test` | [Epidemiological measures](epi_measures.md) |
| Compare rates across populations with different age structure | `sp.direct_standardize`, `sp.indirect_standardize` (SMR) | [Epidemiological measures](epi_measures.md) |
| Screening / diagnostic test accuracy | `sp.sensitivity_specificity`, `sp.diagnostic_test`, `sp.roc_curve`, `sp.cohen_kappa` | [Epidemiological measures](epi_measures.md) |
| Weighing the evidence for causation | `sp.bradford_hill` | [Epidemiological measures](epi_measures.md) |
| Point-treatment effect, confounding measured at baseline | `sp.g_computation`, `sp.ipw`, `sp.tmle`, `sp.aipw` | [G-methods for time-varying confounding](g_methods_ph.md) |
| **Time-varying** treatment with **time-varying confounders affected by past treatment** | `sp.msm`, `sp.gformula.ice`, `sp.gformula.mc`, `sp.ltmle` | [G-methods for time-varying confounding](g_methods_ph.md) |
| Observational data, but you want to reason like a trial | `sp.target_trial.protocol` + `sp.target_trial.emulate` + `sp.clone_censor_weight` | [Target-trial emulation](g_methods_ph.md) |
| Time-to-event outcome (mortality, relapse, time-to-diagnosis) | `sp.cox`, `sp.kaplan_meier`, `sp.survreg`, `sp.aft` | [Survival analysis](survival_ph.md) |
| Informative censoring / loss-to-follow-up | `sp.ipcw` | [Survival analysis](survival_ph.md) |
| Complex survey data (NHANES, BRFSS, DHS, …) | `sp.svydesign`, `sp.svymean`, `sp.svytotal`, `sp.svyglm`, `sp.rake` | [Complex-survey analysis](survey_ph.md) |
| Genetic instruments for a modifiable exposure | `sp.mendelian_randomization` and the MR family | [Mendelian randomization](mendelian_family.md) |
| Continuous exposure → dose-response curve | `sp.dose_response`, `sp.vcnet` | [G-methods for time-varying confounding](g_methods_ph.md) |
| Unmeasured-confounding sensitivity | `sp.evalue` | [Robustness workflow](robustness_workflow.md) |
| Power / sample size | `sp.power`, `sp.mde` | — |

---

## 2. The five-minute epidemiology tour

Every snippet below runs offline and is self-contained.

### Association measures from a 2×2 table

```python
import statspai as sp

# A cohort: 50 exposed cases / 950 exposed non-cases,
#           10 unexposed cases / 990 unexposed non-cases.
rr = sp.relative_risk(50, 950, 10, 990)
print(rr.estimate, rr.ci)          # RR = 5.0, with a log-binomial CI

orr = sp.odds_ratio(50, 20, 30, 40)
print(orr.estimate, orr.ci)        # OR = 3.33 (Woolf CI); Haldane-corrected on zero cells
```

### Confounder-adjusted association (Mantel–Haenszel)

```python
import numpy as np
import statspai as sp

# Two strata, each a 2x2 [[exposed-case, exposed-noncase],
#                         [unexp-case,   unexp-noncase]]
tables = np.array([[[10, 5], [8, 12]],
                   [[20, 15], [6, 9]]])
mh = sp.mantel_haenszel(tables, measure="OR")
print(mh.estimate)        # pooled OR adjusted for stratum
print(mh.homogeneity_p)   # Breslow–Day test for effect homogeneity
```

### Age-standardized rates

```python
import statspai as sp

std = sp.direct_standardize(
    events=[50, 60],            # events in each age band
    population=[1000, 2000],    # person-time / population in each band
    standard_weights=[0.4, 0.6] # reference (standard) population structure
)
print(std.rate, std.ci)
```

---

## 3. The modern causal-epidemiology core

Modern epidemiology's hardest problem is **time-varying confounding
affected by prior treatment** (Robins). Standard regression adjustment
is biased there; g-methods are the answer. StatsPAI ships the three
canonical g-methods plus the target-trial framework that ties them to a
protocol:

```python
import statspai as sp

# Parametric g-formula (iterative conditional expectation) under
# "always treat" vs "never treat" across two time points.
always = sp.gformula.ice(
    data=wide, id_col="id", time_col=None,
    treatment_cols=["A0", "A1"],
    confounder_cols=[["L0"], ["L1"]],
    outcome_col="Y", treatment_strategy=[1, 1],
)
never = sp.gformula.ice(
    data=wide, id_col="id", time_col=None,
    treatment_cols=["A0", "A1"],
    confounder_cols=[["L0"], ["L1"]],
    outcome_col="Y", treatment_strategy=[0, 0],
)
print(always.value - never.value)   # g-formula contrast
```

See the [g-methods family guide](g_methods_ph.md) for the marginal
structural model (`sp.msm`), longitudinal TMLE (`sp.ltmle`), and the
full target-trial workflow that protects against immortal-time bias.

---

## 4. Reporting standards

Public-health and clinical journals expect design-specific reporting
checklists. StatsPAI exposes structured reporting hooks rather than
free-form prose:

- **Target-trial protocol (the 7 protocol elements)** —
  `sp.target_trial.protocol(...).summary()` prints the eligibility,
  treatment strategies, assignment, time-zero, follow-up, outcome,
  causal contrast, and analysis plan as a structured block, and
  `sp.target_trial.to_paper(...)` renders a STROBE-style Methods &
  Results narrative [@hernan2016using].
- **Estimator citations** — mature estimators carry `.cite()` so the
  exact methodological reference lands in your bibliography.
- **Sensitivity to unmeasured confounding** — report an
  `sp.evalue(...)` E-value alongside the point estimate
  [@vanderweele2017sensitivity].

---

## 5. Honest limitations (read before you publish)

1. **Parity is not yet certified for most epi methods.** The numbers
   are well tested internally and recover known truths on simulated
   data, but cross-language certification against R's epidemiology
   packages is on the roadmap, not done. Re-run your headline estimate
   in your reference package.
2. **Positivity and sequential exchangeability are assumptions, not
   outputs.** G-methods identify causal effects only when treatment is
   (sequentially) unconfounded given the measured covariates and every
   covariate stratum has a chance of each treatment. Inspect weight
   distributions (`sp.msm` reports trimming) and think hard about
   unmeasured confounders.
3. **Competing risks are not yet first-class.** `sp.cox` /
   `sp.kaplan_meier` treat censoring as non-informative; there is no
   Fine–Gray subdistribution-hazard or cumulative-incidence-function
   estimator yet. For competing-risks settings, validate carefully and
   watch this space.

## Where to next

- [Epidemiological measures](epi_measures.md)
- [G-methods for time-varying confounding](g_methods_ph.md)
- [Survival analysis](survival_ph.md)
- [Complex-survey analysis](survey_ph.md)
- [Mendelian randomization](mendelian_family.md)
- [Stability tiers and validation status](stability.md)
