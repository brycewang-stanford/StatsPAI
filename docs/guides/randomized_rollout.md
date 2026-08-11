# Randomised rollouts: design-based DiD

Every other DiD estimator in StatsPAI identifies off **parallel trends**. This
page is about the case where you have something better: adoption *timing* was
randomly assigned. A policy lottery, a phased product launch where the order
was drawn, an RCT rolled out in waves.

When that is true, parallel trends is neither needed nor sufficient, and
imposing it throws away the randomisation you paid for.

---

## 1. The one question that decides everything

**Was the adoption date randomly assigned?**

No test on your data can answer this. It is a claim about how the data came to
be, and you either know it from the design or you do not. `sp.recommend`
therefore *surfaces* this branch but never picks it for you.

| Answer | Use | Identifying assumption |
| --- | --- | --- |
| Yes — timing was drawn | `sp.staggered_rollout` | random adoption timing |
| No — units chose, or were chosen | `sp.callaway_santanna` and friends (§2 of [choosing_did_estimator](choosing_did_estimator.md)) | parallel trends |

Getting this wrong is costly in both directions. Claiming design-based
inference on an observational rollout asserts something false about the world.
Imposing parallel trends on a genuinely randomised one is merely wasteful —
but the waste is real: the design-based standard error is smaller, and the
randomisation test needs no asymptotics at all.

> **This is not the same estimand as `sp.callaway_santanna`.** On `did::mpdta`
> — where timing was *not* randomised — `sp.staggered_rollout` returns −0.0471
> against CS's −0.0400. That gap is the estimand difference, not a bug. If you
> see numbers disagree, check which assumption each one is using before
> looking for an error.

---

## 2. The basic call

```python
import statspai as sp

res = sp.staggered_rollout(df, y="y", i="unit", t="time", g="first_treat")
res.summary()
```

`g` is the first-treated period. Never-treated units may be coded `0`, `NaN`
or `inf` — all three are accepted and mean the same thing.

> **In R this is a trap.** `staggered` requires `g = Inf` for never-treated
> units; pass `g = 0` and it silently reads them as a cohort treated before
> the sample. On mpdta that turns −0.047 into −0.370. StatsPAI normalises all
> three codings, so the trap is unreachable through the public API.

### Which estimand

```python
sp.staggered_rollout(df, ..., estimand="simple")     # per treated cell (default)
sp.staggered_rollout(df, ..., estimand="cohort")     # average within cohort first
sp.staggered_rollout(df, ..., estimand="calendar")   # average within period first
sp.staggered_rollout(df, ..., estimand="eventstudy", event_time=1)
```

### Which comparison group

```python
sp.staggered_cs(df, ...)   # every not-yet-treated cohort  (CS estimand)
sp.staggered_sa(df, ...)   # last-treated cohort only      (SA estimand)
```

Both are the *familiar* estimands with design-based inference — same weights
as `sp.callaway_santanna` / `sp.sun_abraham`, different standard errors.

---

## 3. Two standard errors, and which to report

R's `staggered` prints two, and so does StatsPAI:

| | what it is | when to use |
| --- | --- | --- |
| `se_type="neyman"` (default) | conservative bound; treats the fitted control weights as fixed | reporting, unless you have a reason |
| `se_type="adjusted"` | subtracts the part of the variance the randomisation identifies | matching R's primary `se`; tighter |

The adjusted SE is **never larger**, which is why the conservative one is the
default. Both are always available:

```python
res = sp.staggered_rollout(df, ...)
res.model_info["se_neyman"]     # conservative
res.model_info["se_adjusted"]   # tighter
```

How much do they differ? On mpdta's `simple` estimand, by 3e-6 relative —
nothing. On the `calendar` estimand, 1.4e-3. On a genuinely randomised panel,
up to ~2%. Say which one you report.

---

## 4. Randomisation inference

This is the payoff of a randomised design: inference with no asymptotics.

```python
res = sp.staggered_rollout(df, ..., fisher=True, n_fisher=1000, random_state=0)
res.model_info["fisher_pvalue"]
```

Adoption dates are permuted across units — exactly the null the design
licenses — and the studentised statistic recomputed on each draw. The p-value
is the share of draws whose |t| exceeds the observed one.

Because permuting adoption dates leaves every cohort *size* unchanged, the
weight matrices are built once and reused, so this is much cheaper than it
looks. Both p-values (conservative and adjusted) are reported.

---

## 5. Event studies

```python
res = sp.staggered_rollout(df, ..., estimand="eventstudy", event_time=[0, 1, 2])
res.detail          # one row per event time, with both SEs and a CI
res.model_info["vcov"]   # the JOINT covariance across event times
```

`.estimate` averages the requested non-negative event times, and its standard
error comes from that joint covariance — not from pretending the event times
are independent, which would understate it.

**A useful invariant**: at `event_time=-1` the outcome and control weights
coincide, so the estimate is mechanically zero. If it is not, something is
wrong with the weight construction.

---

## 6. The general control set

By default the efficient weights are chosen over a single control: the DiD
contrast at each cohort's last pre-period, `g − 1`. The general form uses
**every** pre-period:

```python
sp.staggered_rollout(df, ..., use_did_a0=False)
```

`beta` then becomes a vector rather than a scalar, and the estimator is weakly
more efficient — at the cost of estimating more nuisance weights. It requires
`efficient=True`: the plug-in fixes `beta = 1`, which is a single contrast and
has no meaning against a vector control set (R errors here too).

---

## 7. What *not* to run afterwards

`sp.did(method="staggered_rollout")` deliberately **skips** the Bacon
decomposition, the pre-trend test and honest-DiD sensitivity, and says so in
its step log. All three are about parallel trends:

- a **pre-trend test** neither supports nor threatens an assumption you are
  not making;
- **honest-DiD** relaxes parallel trends, which you did not assume;
- the **Bacon decomposition** diagnoses TWFE's forbidden comparisons and
  advises switching to Callaway-Sant'Anna — advice that does not apply here.

Reporting a passing pre-trend test next to a design-based estimate is not
extra reassurance. It is a category error, and it invites a reader to think
the estimate rests on an assumption it does not rest on.

What *does* belong: the randomisation test (§4), and the balance of covariates
across adoption cohorts, which is the thing randomisation actually promises.

---

## 8. Parity evidence

Every number above is pinned against R `staggered` 1.2.2:

| What | Agreement |
| --- | --- |
| Every estimand × `beta` × comparison group, both SEs | ≤1e-9 (achieved ~1e-16) |
| Every feasible event time + the joint covariance | ≤1e-9 |
| `staggered_cs` / `staggered_sa` | ≤1e-9 |
| General control set (`use_did_a0=False`) | ≤1e-9 |
| Randomisation test | pinned **draw-for-draw** on 40 R-generated permutations |

Three panels are used: `did::mpdta`, a randomised rollout with no never-treated
units (so `max(g)` is finite — a branch mpdta never reaches), and the same
design with the effect switched off (so the randomisation p-value lands in the
interior, where an error would show).

See `tests/reference_parity/test_staggered_extended_parity.py`, Track A module
`82_staggered`, and `tests/test_design_based_did.py` for the Monte-Carlo
coverage and calibration checks that no reference implementation can supply.

---

## References

- Roth, J. and Sant'Anna, P. H. C. (2023). "Efficient Estimation for Staggered
  Rollout Designs." *Journal of Political Economy Microeconomics*, 1(4),
  669–709. [`roth2023efficient`]
- Callaway, B. and Sant'Anna, P. H. C. (2021). *Journal of Econometrics*.
  [`callaway2021difference`]
- Sun, L. and Abraham, S. (2021). *Journal of Econometrics*.
  [`sun2021estimating`]
