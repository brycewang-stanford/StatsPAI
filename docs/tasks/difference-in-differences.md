---
description: Run difference-in-differences in Python, including staggered adoption with Callaway-Sant'Anna, a Goodman-Bacon decomposition and an event study, on real data.
---

# Difference-in-differences in Python

**Short answer.** With staggered adoption (units treated in different years),
do not stop at a two-way fixed-effects regression. In Python, StatsPAI runs the
whole sequence in a few calls: `sp.feols` for the TWFE benchmark,
`sp.bacon_decomposition` to see which comparisons drive it,
`sp.callaway_santanna` plus `sp.aggte` for a heterogeneity-robust ATT and
event study, and `sp.honest_did` for sensitivity to parallel-trends
violations. On the castle-doctrine panel used below, these numbers match
Stata (`reghdfe`, `bacondecomp`, `csdid`) and R (`fixest`, `did`); see the
[full replication](../guides/mixtape_castle_replication.md).

## When this applies

- **Estimand**: the average treatment effect on the treated (ATT), overall,
  by cohort or by time since treatment.
- **Identifying assumption**: parallel trends between treated cohorts and the
  comparison group (here, never-treated states), plus no anticipation.
- **Data**: a panel (or repeated cross-sections) with a unit id, a time
  variable, the outcome, and the first treated period of each unit.
- **Not the right tool** when treatment switches on and off (see
  `sp.did_multiplegt_dyn`), when there is no credible comparison group (see
  [synthetic control](../guides/synth.md)), or when you have one treated unit.

## Install

```bash
pip install statspai
```

## 1. The two-way fixed-effects benchmark

The data: 50 U.S. states, 2000–2010, with state "castle doctrine" laws adopted
in different years (Cheng and Hoekstra's study, as redistributed with the
*Mixtape*). The outcome is log homicides per 100,000.

```python
import statspai as sp

df = sp.datasets.castle_doctrine(event_time=True)   # adds the cohort column `gvar`

twfe = sp.feols("l_homicide ~ post | sid + year", data=df, cluster="sid")
print(twfe.summary())
```

```text
post 0.0694 0.0559 1.2424 0.2200 -0.0429 0.1817
```

StatsPAI warns while fitting: `post` switches on at five different dates, and
2.4% of the TWFE weight falls on comparisons that use already-treated states as
controls. That is the problem the next two steps address.

## 2. Which comparisons drive the TWFE number?

```python
bacon = sp.bacon_decomposition(df, y="l_homicide", treat="post", time="year", id="sid")
print(bacon["decomposition"].groupby("type")["weight"].sum().round(3))
```

```text
Earlier vs Later Treated 0.077
Later vs Earlier Treated 0.024
Treated vs Untreated 0.899
```

## 3. Heterogeneity-robust ATT (Callaway–Sant'Anna)

```python
cs = sp.callaway_santanna(df, y="l_homicide", g="gvar", t="year", i="sid",
                          control_group="nevertreated")
att = sp.aggte(cs, type="simple")
print(att.summary())
```

```text
ATT: 0.110 ***
Std. Error: (0.039)
[95% CI]: [0.034, 0.186]
```

The robust ATT (0.110) is about 60% larger than TWFE (0.069). These are
analytic standard errors with pointwise intervals. R's `did` defaults to a
multiplier bootstrap with uniform bands; pass `bstrap=True, cband=True` to
`sp.aggte` for those.

## 4. Event study

```python
es = sp.aggte(cs, type="dynamic")
print(es.detail[["relative_time", "att", "se"]].round(3).to_string(index=False))
es.plot()
```

```text
relative_time att se
...
0 0.097 0.040
1 0.112 0.049
```

## Check before you report

- **Pre-trends.** Read the leads in the event study. Here the earliest leads
  (−9, −7) rest on very few cohorts; test them jointly with
  `sp.pretrends_test` and report `sp.honest_did` bounds rather than a
  pass/fail verdict. See [sensitivity analysis](../guides/honest_did.md).
- **Cohort coding.** The adoption year is only partly treated. Coding the
  cohort as `effyear` or `effyear + 1` moves the ATT from 0.110 to 0.019, so
  report both or drop the adoption year (details in the
  [replication guide](../guides/mixtape_castle_replication.md)).
- **Estimator choice.** Sun–Abraham (`sp.sun_abraham`),
  Borusyak–Jaravel–Spiess imputation (`sp.bjs`) and Gardner's two-stage
  (`sp.gardner_did`) make different comparisons. The
  [DiD chooser](../guides/choosing_did_estimator.md) says which fits your data.

## Stata and R equivalents

| Step | Stata | R | StatsPAI |
| --- | --- | --- | --- |
| TWFE | `reghdfe y post, absorb(id year) vce(cluster id)` | `fixest::feols(y ~ post \| id + year, cluster = ~id)` | `sp.feols("y ~ post \| id + year", data=df, cluster="id")` |
| Decomposition | `bacondecomp` | `bacondecomp::bacon()` | `sp.bacon_decomposition()` |
| Callaway–Sant'Anna | `csdid`, `estat simple` | `did::att_gt()` + `aggte()` | `sp.callaway_santanna()` + `sp.aggte()` |
| Sensitivity | `honestdid` | `HonestDiD` | `sp.honest_did()` |

Check the evidence for any function with `sp.parity_status("callaway_santanna")`.

## Other Python options

`differences` and `csdid` implement Callaway–Sant'Anna, `pyfixest` offers
Gardner's `did2s`, local projections and saturated event studies, and
DoubleML has a DML-based staggered DiD. See the
[library comparison](python-econometrics-libraries.md).

---

*Tested with StatsPAI 1.43.1 on 2026-11-12. The code and numbers on this page
are run by `tests/test_docs_task_pages.py`.
[Report a problem](https://github.com/brycewang-stanford/StatsPAI/issues).*
