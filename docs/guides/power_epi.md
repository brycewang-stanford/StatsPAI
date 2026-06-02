# Power & sample size for epidemiological designs

> **Design the study before you run it.** StatsPAI's power family already
> covers econometric designs (RCT, DiD, RD, IV, cluster RCT, OLS); this
> guide covers the three sample-size questions epidemiologists and
> clinical trialists ask most: a binary outcome compared across two arms,
> a time-to-event (log-rank) comparison, and an unmatched case-control
> study.

Every function returns a `PowerResult` (with `.power`, `.n`,
`.summary()`), accepts a scalar **or an array** for the sample-size
argument (for power curves), and solves for the required sample size when
you pass `power_target=` instead of `n`.

These calculators are validated by Monte-Carlo agreement with the test
they approximate and against the closed-form Schoenfeld events
requirement (see `tests/test_power_study_designs.py`).

---

## 1. Two proportions (cohort / RCT with a binary outcome)

```python
import statspai as sp

# Power for a fixed sample size
r = sp.power_two_proportions(n=200, p1=0.30, p2=0.50)
print(r.power)            # ≈ 0.84

# Solve for the total sample size that gives 80% power
r = sp.power_two_proportions(p1=0.30, p2=0.50, power_target=0.80)
print(r.n)                # smallest n with power >= 0.80

# Unequal allocation (twice as many controls) and a power curve
import numpy as np
curve = sp.power_two_proportions(n=np.arange(50, 500, 50),
                                 p1=0.30, p2=0.50, ratio=2.0)
print(curve.power)
```

`p1` is the reference (control) outcome probability, `p2` the comparison
arm; `ratio = n2/n1`.

---

## 2. Survival / log-rank (Schoenfeld)

Power for a time-to-event comparison depends on the number of **events**,
not just the sample size, so you supply the hazard ratio and the overall
probability that a subject is observed to have the event:

```python
import statspai as sp

# 80% power to detect HR = 0.5 needs ~66 events (Schoenfeld 1983)
r = sp.power_logrank(hazard_ratio=0.5, prob_event=1.0, power_target=0.80)
print(r.params["n_events"])     # ≈ 66

# If only 60% of subjects are expected to have the event during follow-up:
r = sp.power_logrank(hazard_ratio=0.7, prob_event=0.6, power_target=0.80)
print(r.n)                       # total sample size (= events / prob_event)
```

---

## 3. Unmatched case-control

Parameterised the way case-control studies are actually planned — by the
exposure **odds ratio** to detect and the exposure **prevalence among
controls** — with `ratio` controls per case:

```python
import statspai as sp

# Power with 150 cases, 1:1 controls, OR = 2, 30% control exposure
r = sp.power_case_control(n_cases=150, odds_ratio=2.0,
                          exposure_prevalence=0.30)
print(r.power)

# Number of cases for 80% power with 2 controls per case
r = sp.power_case_control(odds_ratio=2.0, exposure_prevalence=0.30,
                          ratio=2.0, power_target=0.80)
print(r.n)                       # required number of cases
```

The odds ratio and control exposure prevalence imply the case exposure
prevalence `p1 = OR·p0 / (1 + p0(OR−1))`, and power is computed as a
two-proportion comparison between cases and controls.

---

## Notes & limitations

- All three use the normal approximation; for very small samples or very
  rare outcomes, prefer an exact calculation and treat these as planning
  approximations.
- Stepped-wedge and other cluster-period designs are not yet covered here
  (cluster RCTs are available via `sp.power_cluster_rct`).

## Where to next

- [Competing risks](competing_risks.md)
- Survival analysis (`sp.cox`, `sp.kaplan_meier`, `sp.survreg`) — see the
  survival reference page.
