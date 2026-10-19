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

km = sp.kaplan_meier(data=df, duration="time", event="event", group="arm", conf_type="log-log")
print(km.median_survival)     # median survival per group
print(km.survival_table)      # risk table: n.risk, n.event, S(t), CI
km.plot()                     # step survival curves with at-risk counts
```

`event` is the standard 1 = event observed, 0 = right-censored coding.
Pass `group=` to stratify; omit it for a single pooled curve.

`weights=` gives weighted curves (with inverse-probability-of-treatment
weights, the confounder-adjusted curve of each arm), `start=` delayed
entry, and `cluster=` a subject id for repeated rows. The variance
follows R `survfit`: the infinitesimal jackknife (`se_method="ij"`) with
a cluster or non-integer weights, Greenwood otherwise;
`se_method="xie_liu"` is the variance `adjustedCurves` reports for IPTW
curves.

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
the hazard ratio (pass `hazard_ratio=True` to print HRs directly). You
can add:

- `strata=` for a stratified baseline hazard,
- `robust="hc0"` or `cluster=` for robust / clustered standard errors,
  with `ssc="r"` (no small-sample factor, as `coxph`) or `ssc="stata"`
  (`N/(N-1)` robust, `G/(G-1)` clustered, as `stcox`),
- `ties="efron"` (default) / `"breslow"` for tie handling,
- `start=` for counting-process rows `(start, stop]`: delayed entry and
  **time-varying covariates** split into intervals (R `Surv(start, stop,
  event)`, Stata `stset, time0()`); pass `cluster=<subject id>` with
  several rows per subject,
- `weights=` for case weights; with IPT or survey weights also
  `weights_as="sampling"`, which gives the robust variance R reports for
  such weights and Stata `stcox [pw=]` always uses.

```python
msm_cox = sp.cox(data=df, duration="time", event="event", x=["arm"],
                 weights="iptw", weights_as="sampling", ssc="r")
tv_cox = sp.cox(data=long, duration="tstop", start="tstart", event="event",
                x=["cd4", "arm"], cluster="id", ssc="r")
```

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
    data=df, time="time", event="event",
    treat="arm", covariates=["age", "sex", "ldl"],
    horizon=36, target="RMST",        # or target="survival_probability"
)
csf.ate, csf.se                       # doubly-robust average effect
```

Choose the horizon from the study design; the estimator weights by the
probability of staying uncensored up to it. See
[the GRF family guide](grf_forests.md) for the method, its evidence against
grf, and `sp.survival_forest` for conditional survival curves.

---

## 6. Average effects on survival time

**Restricted mean survival time.** `sp.rmst` integrates the
Kaplan–Meier curve of each group up to `tau` and reports the difference
and the RMST / RMTL ratios (R `survRM2::rmst2`). With IPT weights it is
the confounder-adjusted RMST difference, with the infinitesimal-jackknife
standard error (which holds the weights fixed):

```python
r = sp.rmst(df, duration="time", event="event", group="arm", tau=36,
            weights="iptw")
r.contrasts        # difference, RMST ratio, RMTL ratio
```

**Effects on mean survival time from parametric models.**
`sp.stteffects` is Stata's `stteffects`: regression adjustment
(`method="ra"`), weighted RA (`"wra"`), inverse-probability weighting
(`"ipw"`) and IPW regression adjustment (`"ipwra"`), for the ATE or the
ATET, with Weibull / exponential / log-normal outcome and censoring
models and the stacked-estimating-equation standard errors.

```python
te = sp.stteffects(df, duration="time", event="event", treat="arm",
                   covariates=["age", "sex"], method="ipwra")
te.detail          # ATE, POmean0, POmean1
```

For `ipwra` with `estimand="ATET"` StatsPAI averages over the treated;
Stata averages the same models over the whole sample (see
`docs/dev/modules/survival.md`).

**Dropout in a marginal structural model.** `sp.msm(censor=...)` adds
stabilized inverse-probability-of-censoring weights for person-period
data (R `ipw::ipwtm(type = "first")`).

---

## 7. What is *not* here yet

- **Competing risks** are covered by `sp.cuminc` (Aalen–Johansen
  cumulative incidence) and `sp.finegray`; weighted versions are not.
- `CoxResult.ph_test()` does not yet handle weighted or counting-process
  fits (it refuses them); use R `cox.zph` for those.
- **Time-dependent ROC / net-benefit / decision-curve analysis**.

If your analysis depends on these, validate carefully and consider
cross-checking with R's `cmprsk` / `survival` in the interim.

## Where to next

- [Public health & epidemiology overview](public_health.md)
- [G-methods for time-varying confounding](g_methods_ph.md)
- [Complex-survey analysis](survey_ph.md)
