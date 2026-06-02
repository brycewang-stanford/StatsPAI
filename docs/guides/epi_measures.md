# Epidemiological measures — the `sp.epi` family

> **The primitives epidemiologists reach for first.** Before any model,
> public-health analysis starts with measures of association from
> contingency tables, rate standardization, and screening-test accuracy.
> StatsPAI ships these as top-level functions modelled after R's
> `epiR`, `epitools`, and `fmsb`, with the standard small-sample
> conventions (Woolf and exact CIs, Haldane–Anscombe continuity
> correction, exact Poisson rate intervals) built in
> [@rothman2008modern].

All functions return a small typed result with `.estimate`, `.ci`, and
(where defined) `.p_value`, so they compose with the rest of StatsPAI's
reporting layer.

---

## 1. Measures of association (2×2 tables)

The 2×2 layout throughout is `(a, b, c, d)` =
`(exposed cases, exposed non-cases, unexposed cases, unexposed non-cases)`.

```python
import statspai as sp

# Cohort study
rr = sp.relative_risk(50, 950, 10, 990)
print(f"RR = {rr.estimate:.2f}  95% CI {rr.ci}")
print(rr.risk_exposed, rr.risk_unexposed)

rd = sp.risk_difference(50, 950, 10, 990)
print(f"RD = {rd.estimate:.4f}  95% CI {rd.ci}")

nnt = sp.number_needed_to_treat(50, 950, 10, 990)
print(f"NNT = {nnt.estimate:.1f}")

# Case-control study (odds ratio)
orr = sp.odds_ratio(50, 20, 30, 40, method="woolf")
print(f"OR = {orr.estimate:.2f}  95% CI {orr.ci}")

# Population attributable fraction (Levin)
ar = sp.attributable_risk(50, 950, 10, 990)
print(ar.estimate)
```

**Person-time / rates.** When the denominator is person-time rather than
people, use the incidence-rate ratio with an exact Poisson interval:

```python
irr = sp.incidence_rate_ratio(
    events_exposed=30, pt_exposed=1000,
    events_unexposed=15, pt_unexposed=1200,
)
print(f"IRR = {irr.estimate:.2f}  95% CI {irr.ci}")
```

**Cross-sectional prevalence.** For prevalent (not incident) outcomes the
prevalence ratio is preferred over the OR; `sp.prevalence_ratio` uses the
modified-Poisson / log-binomial approach [@zou2004modified].

**Zero cells** are handled automatically with the Haldane–Anscombe 0.5
continuity correction on the log-scale standard error, so a single empty
cell does not blow up the interval.

---

## 2. Stratified analysis — Mantel–Haenszel

When a categorical variable confounds the exposure–outcome relationship,
pool stratum-specific tables with Mantel–Haenszel and test whether the
effect is homogeneous across strata [@mantel1959statistical]:

```python
import numpy as np
import statspai as sp

# K = 2 strata; each is a 2x2 [[a, b], [c, d]]
tables = np.array([
    [[10, 5], [8, 12]],
    [[20, 15], [6, 9]],
])

mh = sp.mantel_haenszel(tables, measure="OR")   # or measure="RR"
print(f"MH pooled OR = {mh.estimate:.3f}  95% CI {mh.ci}")
print(f"stratum-specific: {mh.stratum_estimates}")
print(f"Breslow–Day homogeneity p = {mh.homogeneity_p:.3f}")
```

A small homogeneity p-value warns that a single pooled estimate hides
effect-measure modification — report stratum-specific estimates instead.
`sp.breslow_day_test` exposes the test directly (with the Tarone
correction by default).

---

## 3. Rate standardization

Crude rates are not comparable across populations with different age
structures. Standardize directly (when you have stratum-specific rates in
the study population) or indirectly (SMR, when study strata are small):

```python
import statspai as sp

# Direct standardization to a reference population structure
std = sp.direct_standardize(
    events=[50, 60, 40],            # events per age band
    population=[1000, 2000, 1500],  # population/person-time per band
    standard_weights=[0.3, 0.45, 0.25],
)
print(f"Age-standardized rate = {std.rate:.5f}  95% CI {std.ci}")

# Indirect standardization → standardized mortality/morbidity ratio (SMR)
smr = sp.indirect_standardize(
    observed=120,
    events_reference=[50, 60, 40],
    population_reference=[1000, 2000, 1500],
    population_study=[800, 1500, 1000],
)
print(f"SMR = {smr.estimate:.3f}  95% CI {smr.ci}")
```

---

## 4. Screening and diagnostic-test accuracy

```python
import statspai as sp

# From a confusion matrix
acc = sp.sensitivity_specificity(tp=80, fn=20, fp=10, tn=90)
print(acc.summary())   # sensitivity, specificity, PPV, NPV, LR+/LR- with Wilson CIs

# From scores + labels
roc = sp.roc_curve(y_true, scores)
print(roc.auc)

# Inter-rater agreement
kappa = sp.cohen_kappa(rater_a, rater_b, weights="linear")
print(kappa.estimate)
```

---

## 5. Weighing the evidence — Bradford–Hill

`sp.bradford_hill` turns the nine Bradford–Hill viewpoints into a
structured, transparent scoring object — useful for causal-narrative
sections and for forcing each viewpoint to be argued explicitly rather
than asserted:

```python
import statspai as sp

bh = sp.bradford_hill(
    strength=1.0, temporality=1.0, consistency=0.5,
    biological_gradient=0.8, plausibility=0.7,
)
print(bh.summary())
```

It is a structuring aid, **not** a causal test — the score does not
establish causation, it documents how each viewpoint was judged.

---

## Where to next

- [Public health & epidemiology overview](public_health.md)
- [G-methods for time-varying confounding](g_methods_ph.md)
- [Survival analysis](survival_ph.md)
