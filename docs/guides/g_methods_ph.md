# G-methods for time-varying confounding (public-health edition)

> **The signature problem of modern epidemiology.** When treatment
> changes over time and is driven by covariates that are themselves
> affected by past treatment, ordinary regression adjustment is biased
> in *both* directions — adjusting for the time-varying confounder
> blocks part of the causal path, not adjusting leaves confounding.
> Robins' **g-methods** solve this. StatsPAI ships all three —
> the parametric g-formula, marginal structural models via IPTW, and
> longitudinal TMLE — plus the **target-trial** framework that ties them
> to an explicit protocol [@robins1986new; @hernan2020causal].

This guide is scope-honest: these estimators are API-stable and recover
known truths on simulated data, but are **not yet parity-certified**
against R's `gfoRmula` / `ipw` / `ltmle`. Validate your headline number
in your reference package before publication, and read
`sp.describe_function(name)["limitations"]` first.

---

## 1. Point treatment first (confounding measured at baseline)

If treatment happens once and confounders are measured before it, you do
**not** need the full g-methods machinery — a g-computation, IPW, or
doubly-robust estimator is enough:

```python
import statspai as sp

# data has outcome y, binary treatment a, covariate(s) x
gc  = sp.g_computation(data, y="y", treat="a", covariates=["x"], seed=1)
ipw = sp.ipw(data, y="y", treat="a", covariates=["x"], seed=1)
dr  = sp.aipw(data, y="y", treat="a", covariates=["x"])   # doubly robust

print(gc.estimate, ipw.estimate, dr.estimate)   # all target the ATE
```

- `sp.g_computation` models the **outcome** (trust your outcome model).
- `sp.ipw` models the **treatment** (trust your propensity model);
  weights are normalized by default and `trim=` guards extreme weights.
- `sp.aipw` / `sp.tmle` are **doubly robust**: consistent if *either*
  model is right. Prefer these when you are unsure (see
  [Choosing an ML causal estimator](choosing_ml_causal_estimator.md)).

---

## 2. The time-varying case

Now suppose treatment `A` is measured at each visit, a time-varying
confounder `L` predicts both the next treatment and the outcome, **and**
`L` is itself affected by earlier treatment. This is exactly where
g-methods are required.

### 2a. Marginal structural model (`sp.msm`) — IPTW

```python
import statspai as sp

# Long format: one row per (id, time). `time_varying` lists the
# confounders measured at each visit.
msm = sp.msm(
    data=long, y="Y", treat="A",
    id="id", time="time",
    time_varying=["L"],
    exposure="cumulative",   # effect of cumulative treatment
    trim=0.01,               # truncate extreme stabilized weights
)
print(msm.summary())
```

`sp.msm` fits stabilized inverse-probability-of-treatment weights at
each visit, then a weighted outcome model — the canonical MSM of
Robins, Hernán & Brumback [@robins2000marginal; @cole2008constructing].
Inspect the weight trimming it reports: stabilized weights far from 1.0
are a positivity warning.

### 2b. Parametric g-formula (`sp.gformula.ice` / `gformula_mc`)

The g-formula evaluates the outcome under a *hypothetical intervention
strategy* applied at every time point, standardizing over the
time-varying confounder distribution at each step.

```python
import statspai as sp

# Iterative conditional expectation (ICE) g-formula, wide format:
# one row per subject, columns A0,L0,A1,L1,Y
always = sp.gformula.ice(
    data=wide, id_col="id", time_col=None,
    treatment_cols=["A0", "A1"],
    confounder_cols=[["L0"], ["L1"]],   # confounders measured before each A
    outcome_col="Y", treatment_strategy=[1, 1],   # always treat
)
never = sp.gformula.ice(
    data=wide, id_col="id", time_col=None,
    treatment_cols=["A0", "A1"],
    confounder_cols=[["L0"], ["L1"]],
    outcome_col="Y", treatment_strategy=[0, 0],   # never treat
)
print(f"g-formula contrast = {always.value - never.value:.3f}")
```

`sp.gformula_mc` provides the Monte-Carlo simulation estimator (the
classic parametric g-formula of Robins), which also supports
**dynamic** regimes (treat-when-L-crosses-a-threshold). Use ICE for
static "always vs never" contrasts and MC when the strategy depends on
evolving covariates.

### 2c. Longitudinal TMLE (`sp.ltmle`)

When you want a doubly-robust, machine-learning-friendly estimator for
the same longitudinal contrast, reach for `sp.ltmle`:

```python
import statspai as sp

res = sp.ltmle(
    data=long_or_wide, y="Y",
    treatments=["A0", "A1"],
    covariates_time=[["L0"], ["L1"]],
    baseline=["W"],
)
print(res.summary())
```

There is also `sp.ltmle_survival` for discrete-time survival outcomes
under dynamic regimes, and `sp.hal_tmle` for Highly Adaptive Lasso
nuisance learners.

---

## 3. Target-trial emulation

The target-trial framework forces you to write down the randomized trial
you *wish* you could run, then emulate it from observational data — the
single most effective guard against immortal-time bias and ill-defined
"time zero" [@hernan2016using].

### 3a. Specify the protocol (the 7 elements)

```python
import statspai as sp

protocol = sp.target_trial.protocol(
    eligibility="age >= 50 and no prior MI",
    treatment_strategies=["initiate statin", "no statin"],
    assignment="observational emulation",
    time_zero="date of eligibility",          # prevents immortal time
    followup_end="min(death, loss-to-follow-up, 5 years)",
    outcome="myocardial infarction",
    causal_contrast="per-protocol",
    analysis_plan="IPW for treatment + censoring",
    baseline_covariates=["age", "sex", "ldl"],
)
print(protocol.summary())     # structured 7-element protocol table
```

### 3b. Emulate and report

```python
result = sp.target_trial.emulate(
    protocol, data, outcome_col="mi", treatment_col="statin",
)
print(result.summary())
print(f"eligible: {result.n_eligible}  "
      f"excluded for immortal time: {result.n_excluded_immortal}")

# STROBE-style Methods & Results narrative for the manuscript
print(sp.target_trial.to_paper(result, fmt="target"))   # or "jama", "bmj"
```

### 3c. Per-protocol effects with clone-censor-weight

For per-protocol (sustained-strategy) effects, clone each subject into
every strategy, censor clones when they deviate, and reweight by inverse
probability of remaining uncensored:

```python
ccw = sp.clone_censor_weight(
    data=long, id_col="id", time_col="time", treatment_col="a",
    strategies={
        "always": lambda df: (df["a"] == 1).to_numpy(),
        "never":  lambda df: (df["a"] == 0).to_numpy(),
    },
    censor_covariates=["L"],
    stabilize=True,
)
```

Pair this with `sp.ipcw` (see [Survival analysis](survival_ph.md)) when
loss-to-follow-up is informative.

---

## 4. Continuous exposures → dose-response

When the exposure is continuous (dose, BMI, pollutant concentration),
estimate the whole causal dose-response curve rather than a single
contrast:

```python
import statspai as sp

drf = sp.dose_response(
    data, y="y", treat="dose", covariates=["x1", "x2"],
    n_dose_points=20,
)
# sp.vcnet provides a varying-coefficient alternative
```

---

## 5. Choosing among the g-methods

| You trust… | and your treatment is… | use |
| --- | --- | --- |
| the outcome model | point / static | `sp.g_computation` |
| the treatment model | point / static | `sp.ipw` |
| neither fully (be safe) | point / static | `sp.aipw`, `sp.tmle` |
| the outcome model | time-varying | `sp.gformula.ice` / `sp.gformula_mc` |
| the treatment model | time-varying | `sp.msm` |
| neither fully, time-varying | time-varying | `sp.ltmle` |
| a written protocol | time-varying | `sp.target_trial.*` |

All g-methods rely on **(sequential) exchangeability + positivity +
consistency**. They cannot test these assumptions — pair every estimate
with an `sp.evalue(...)` sensitivity analysis for unmeasured confounding
[@vanderweele2017sensitivity] and inspect weight distributions for
positivity violations.

## Where to next

- [Public health & epidemiology overview](public_health.md)
- [Survival analysis](survival_ph.md)
- [Choosing an ML causal estimator](choosing_ml_causal_estimator.md)
- [Robustness workflow](robustness_workflow.md)
