# Survival analysis for public health

> **Time-to-event is the native outcome of cohort epidemiology and
> clinical research.** Mortality, relapse, time-to-diagnosis,
> time-to-discharge — all are right-censored durations. StatsPAI ships
> the standard estimators (Kaplan–Meier, Cox proportional hazards,
> parametric AFT, frailty models) plus the causal-survival tools
> (inverse-probability-of-censoring weights, causal survival forests)
> that connect survival outcomes to the g-methods in the
> [g-methods guide](g_methods_ph.md).

Scope note: `sp.cox` is parity-tested against R's `survival::coxph` and
Stata's `stcox` (see `tests/r_parity/`, `tests/stata_parity/`). The
causal-survival surface (`sp.ipcw`, `sp.causal_survival_forest`,
`sp.synth_survival`) is API-stable but not yet cross-language certified
— validate before publication and read
`sp.describe_function(name)["limitations"]`.

---

## 1. Kaplan–Meier survival curves

```python
import statspai as sp

km = sp.kaplan_meier(data=df, duration="time", event="event", group="arm")
print(km.median_survival)     # median survival per group
print(km.survival_table)      # risk table: n.risk, n.event, S(t), CI
km.plot()                     # step survival curves with at-risk counts
```

`event` is the standard 1 = event observed, 0 = right-censored coding.
Pass `group=` to stratify; omit it for a single pooled curve.

---

## 2. Cox proportional hazards

```python
import statspai as sp

# Formula or explicit columns — both work
cox = sp.cox(data=df, duration="time", event="event", x=["arm", "age", "sex"],
             ties="efron")
print(cox.summary())
```

The reported coefficients are **log hazard ratios**; exponentiate for
the hazard ratio (pass `hazard_ratio=True` to print HRs directly). The
counting-process internals mean Cox accepts **time-varying covariates**
via `(start, stop, event)` rows, and you can add:

- `strata=` for a stratified baseline hazard,
- `robust="hc1"` or `cluster=` for robust / clustered standard errors,
- `ties="efron"` (default) / `"breslow"` for tie handling.

Always sanity-check the proportional-hazards assumption (Schoenfeld
residuals) before trusting a single HR over long follow-up.

---

## 3. Parametric survival (AFT) and frailty

When you need a fully parametric model — for extrapolation, or because
the accelerated-failure-time interpretation is more natural — use
`sp.survreg` / `sp.aft`:

```python
import statspai as sp

# Weibull / lognormal / loglogistic / exponential AFT
m = sp.survreg(data=df, duration="time", event="event", x=["arm", "age"],
               dist="weibull")
print(m.summary())

aft = sp.aft("time ~ arm + age", data=df, family="lognormal")
```

For clustered / shared-frailty survival (recurrent events, multi-centre
cohorts) the `sp.survival` module also exposes a frailty model.

---

## 4. Informative censoring — IPCW

When loss-to-follow-up depends on measured covariates, censoring is
**informative** and naïve KM/Cox are biased. Inverse-probability-of-
censoring weights restore the population that would have been observed
under complete follow-up [@robins2000marginal]:

```python
import statspai as sp

ipcw = sp.ipcw(
    data=df, time="time", event="event",
    censor_covariates=["age", "ldl", "on_treatment"],
    stabilize=True,
    method="pooled_logistic",
)
print(ipcw.summary_stats)     # weight distribution — watch for extremes
weights = ipcw.weights        # feed into a weighted outcome analysis
```

These weights compose directly with `sp.clone_censor_weight` for
per-protocol target-trial analyses (see
[g-methods](g_methods_ph.md)).

---

## 5. Causal survival effects

For heterogeneous treatment effects on a survival outcome under
unconfoundedness, `sp.causal_survival_forest` estimates a
restricted-mean-survival-time or survival-probability contrast that
varies with covariates; `sp.synth_survival` brings the synthetic-control
idea to time-to-event panel settings.

```python
import statspai as sp

csf = sp.causal_survival_forest(
    data=df, duration="time", event="event",
    treatment="arm", covariates=["age", "sex", "ldl"],
)
```

---

## 6. What is *not* here yet

- **Competing risks**: there is no Fine–Gray subdistribution-hazard
  model or cumulative-incidence-function / Aalen–Johansen estimator yet.
  In a setting with competing events (e.g. death from other causes),
  cause-specific Cox is available (model each cause with censoring on
  the others), but the subdistribution approach is on the roadmap.
- **Time-dependent ROC / net-benefit / decision-curve analysis**.

If your analysis depends on these, validate carefully and consider
cross-checking with R's `cmprsk` / `survival` in the interim.

## Where to next

- [Public health & epidemiology overview](public_health.md)
- [G-methods for time-varying confounding](g_methods_ph.md)
- [Complex-survey analysis](survey_ph.md)
