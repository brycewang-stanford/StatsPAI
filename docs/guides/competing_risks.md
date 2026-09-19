# Competing risks — cumulative incidence & Fine-Gray

> **When a subject can fail from more than one cause, Kaplan-Meier lies.**
> Treating "death from other causes" as ordinary censoring makes the
> naïve `1 - KM` curve over-state the risk of the cause you care about,
> because it implicitly assumes the censored subjects could still have
> had the event. StatsPAI ships the two standard competing-risks tools:
> the **Aalen-Johansen cumulative incidence function** (`sp.cuminc`) for
> description and Gray's test, and the **Fine-Gray subdistribution
> hazards model** (`sp.finegray`) for regression
> [@aalen1978nonparametric; @gray1988class; @fine1999proportional].

**Event coding.** Throughout, the `event` column is an integer:
`0` = right-censored, and `1, 2, ...` = the competing causes. This
matches R's `cmprsk` and `survival` conventions.

Reference parity: the cumulative incidence, both of its variances, Gray's
test and every Fine-Gray number are pinned to R `cmprsk` (`cuminc`, `crr`)
and Stata (`stcompet`, `stcrreg`) on identical data with tied event and
censoring times, at 1e-10 or tighter
(`tests/reference_parity/test_survival_epi_R_parity.py`).

---

## 1. Why not Kaplan-Meier?

In a competing-risks setting the quantity you almost always want is the
**cumulative incidence function** (CIF) — the probability of failing from
cause *k* by time *t*, accounting for the fact that a competing event
removes a subject from ever experiencing cause *k*. The Aalen-Johansen
estimator weights each cause-specific increment by the overall (all-cause)
survival:

$$
\hat F_k(t) = \sum_{t_i \le t} \hat S(t_{i-1}) \, \frac{d_{ki}}{n_i}
$$

so that the CIFs of all causes plus the overall survival sum to one at
every time — a property `1 - KM_k` does not have.

---

## 2. Cumulative incidence — `sp.cuminc`

```python
import statspai as sp

ci = sp.cuminc(df, duration="time", event="status")
print(ci.summary())
print(ci.cif_table.head())   # group / cause / time / cif / se / ci_lower / ci_upper
ci.plot(cause=1)             # step CIF curve with the other causes overlaid
```

Each cause's CIF comes with a standard error and a confidence band. The
default `variance="delta"` is the Marubini-Valsecchi delta method (what
Stata's `stcompet` reports); `variance="gray"` is Gray's asymptotic
variance (the `var` of R `cmprsk::cuminc`). `conf_type="log-log"` gives
stcompet's bounds instead of the default linear band clipped to [0, 1]. To
read off the cumulative incidence at a specific horizon:

```python
ci.cif_at(time=5.0, cause=1)   # CIF and CI at t = 5
```

### Comparing groups — Gray's test

Pass `group=` to estimate per-group CIFs and get Gray's (1988) K-sample
test for equality of cumulative incidence, reported per cause:

```python
ci = sp.cuminc(df, duration="time", event="status", group="arm")
print(ci.gray_test[1])    # {'statistic', 'df', 'p_value'} for cause 1
ci.plot(cause=1)          # one CIF curve per arm
```

Gray's test targets the **subdistribution** hazard, so it answers the
clinically relevant question "do the groups differ in cumulative
incidence of cause 1?" rather than the cause-specific-hazard question.
The statistic is `cmprsk::cuminc`'s: subdistribution risk sets estimated
per group from the group's KM and CIF, and Gray's asymptotic covariance
(not a log-rank variance). `rho=` sets the `(1 - F(t-))^rho` weight.

---

## 3. Regression — `sp.finegray`

The Fine-Gray model puts covariates on the subdistribution hazard, so its
coefficients exponentiate to **subdistribution hazard ratios (sHR)** that
map monotonically to the cumulative incidence — a covariate with sHR > 1
*increases* the cause-of-interest CIF. (Cause-specific Cox coefficients do
not have this property, which is why they are easy to misread in a
competing-risks setting.)

```python
import statspai as sp

fg = sp.finegray(df, duration="time", event="status",
                 x=["treatment", "age", "stage"], cause=1)
print(fg.summary())
fg.tidy()        # term / coef / shr / std_err / z / p_value / shr_lower / shr_upper
```

Subjects who fail from a competing cause are kept in the risk set with
time-decaying inverse-probability-of-censoring weights
`Ĝ(t-)/Ĝ(T_i-)` (Fine & Gray 1999; the censoring KM at left limits, as
`cmprsk::crr` and `stcrreg` evaluate it); the weighted partial likelihood
is maximised by Newton-Raphson.

Standard errors default to Fine and Gray's sandwich (`vce="robust"`),
whose score residuals include the variability from estimating Ĝ — the
`var` of `cmprsk::crr`. `small_sample=True` multiplies it by `n/(n-1)`,
which reproduces Stata's `stcrreg`. `vce="model"` returns the inverse
information (`crr`'s `invinf`), which is not a valid variance for this
model and is kept only for comparison.

---

## 4. Cause-specific vs. subdistribution — which to report?

Both are legitimate and answer **different** questions:

| Question | Tool |
| --- | --- |
| "What is the probability of failing from cause 1 by time *t*?" | `sp.cuminc` (CIF) |
| "Does treatment change the **cumulative incidence** of cause 1?" | `sp.finegray` (sHR) or Gray's test |
| "Does treatment change the **rate** of cause 1 among those still event-free?" | cause-specific `sp.cox` (censor competing events) |

A common, defensible reporting choice is to present the CIF curves
descriptively and **both** a cause-specific Cox model and a Fine-Gray
model, since they decompose the effect on the rate vs. the effect on the
cumulative incidence.

---

## 5. Limitations (read before you publish)

- **No time-varying covariates** in `sp.finegray` yet (cmprsk's `cov2` /
  `tf`), and no `cengroup` (group-specific censoring distributions).
- **Gray's test is unstratified** (cmprsk's `strata=` is not exposed).

## Where to next

- [Survival analysis for public health](survival_ph.md)
- [G-methods for time-varying confounding](g_methods_ph.md)
