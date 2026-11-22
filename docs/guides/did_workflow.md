# The DiD protocol in nine steps

A difference-in-differences study in 2026 is a sequence of decisions and
checks, and a regression is one step of it. `sp.did_workflow` runs that
sequence on a panel and grades every step. This guide explains the nine
steps, how each status is decided, and how to move through them one at a
time.

The order follows two sources. Baker, Callaway, Cunningham, Goodman-Bacon
and Sant'Anna (2026, *JEL* 64(2), 498–557) [@baker2026difference] ask for
the target and the assumption to be fixed before a coefficient is read.
Roth, Sant'Anna, Bilinski and Poe (2023, *Journal of Econometrics* 235(2),
2218–2244) [@roth2023whats] give the practitioner checklist that the later
steps implement.

## One call

```python
import statspai as sp

df = sp.datasets.mpdta()        # simulated replica of the minimum-wage panel

wf = sp.did_workflow(
    df, y="lemp", id="countyreal", time="year", cohort="first_treat",
    target="overall",                  # which summary of ATT(g,t)
    control_group="nevertreated",      # the parallel-trends assumption
    cluster="countyreal",              # what the standard error treats as random
)
print(wf.summary())
```

```text
[ PASS  ] 1. Declare the target and the assumption
          Target: ATT averaged over all treated unit-periods. Assumption:
          parallel trends between each treated cohort and the never-treated
          units. Clustered on 'countyreal'.
[ PASS  ] 2. Audit the design
          500 units x 5 periods; 3 treated cohort(s) with 375 units, 125 never
          treated; 500 clusters on 'countyreal'.
[CAUTION] 3. Diagnose two-way fixed effects
          TWFE coefficient -0.03751; 26.7% of its weight is on comparisons
          that use already-treated units as controls. ...
[ PASS  ] 4. Estimate with a heterogeneity-robust estimator
          Callaway-Sant'Anna: ATT averaged over all treated unit-periods =
          -0.03298 (SE 0.007765, 95% CI [-0.04819, -0.01776], p = 2.17e-05).
[CAUTION] 5. Report the event-study path
          4 post-treatment and 3 pre-treatment event times; the simultaneous
          95% band (critical value 2.64 against 1.96 pointwise) excludes zero
          at 3 of 4 post-treatment event times. The cohorts behind the path
          change with the horizon: 3 at e = 0, 1 at e = 3. ...
[ PASS  ] 6. Test pre-trends and the power of that test
          Joint test of 3 pre-treatment coefficients: Wald chi2(3) = 1.99, p =
          0.574. It does not reject, which is not evidence that parallel
          trends holds. A linear trend this test detects only 80% of the time
          would bias the average post-treatment effect by 0.02933, 86.8% of
          the estimate (-0.03378).
[CAUTION] 7. Bound violations of parallel trends
          The sign of the average post-treatment effect survives post-
          treatment violations of parallel trends up to 0.268 times the
          largest violation between consecutive pre-treatment periods ...
[ PASS  ] 8. Vary the choices that were choices
          Across 6 alternative specification(s) the estimate ranges from
          -0.03511 to -0.0303 (headline -0.03298).
[ INFO  ] 9. Write down what was estimated
          14 items a reader needs to tell which number this is; 3 step(s) left
          a caveat the write-up owes.
Verdict   CAUTION
```

The workflow adds no estimator. Each number is the output of the public
function that the step names, so `wf.estimate` equals
`sp.aggte(sp.callaway_santanna(...), type="simple").estimate` to the last
bit. The path, the pre-trend test and the sensitivity analysis all come
from the estimator that produced the headline number.

## The nine steps

`sp.did_protocol()` returns this table without touching any data.

| # | Step | Question | Functions behind it |
| --- | --- | --- | --- |
| 1 | `declare` | Which effect, for whom, against which comparison group, over which window, and what is treated as random? | the arguments of the call |
| 2 | `design` | Is adoption absorbing, who is never treated, how many cohorts and clusters are there? | `sp.did_cluster_diagnostics` |
| 3 | `twfe` | How much of the TWFE coefficient comes from already-treated controls? | `sp.bacon_decomposition` |
| 4 | `estimate` | What is the declared target under the declared comparison group? | `sp.callaway_santanna` + `sp.aggte`, `sp.sun_abraham`, `sp.did_imputation` |
| 5 | `event_study` | What does the effect look like by time since treatment, and which cohorts identify each horizon? | `sp.aggte(type="dynamic")`, `sp.uniform_bands` |
| 6 | `pretrends` | Do the pre-treatment coefficients reject, and how large a trend could the test have missed? | `sp.pretrends_test`, `sp.pretrends_slope_for_power` |
| 7 | `sensitivity` | How large a violation does the conclusion survive? | `sp.honest_did`, `sp.breakdown_m` |
| 8 | `robustness` | Does the answer move with the estimator, the comparison group, the outcome scale or the standard error? | `sp.did_summary`, `sp.cs_jackknife`, `sp.functional_form_test` |
| 9 | `report` | What must the paper state? | `wf.report()`, `wf.checklist`, `wf.cite()` |

### 1. Declare

The comparison group is the identifying assumption, the window and the
weights are part of the estimand, and the clustering level decides what
the standard error treats as random. The step lists each choice with
whether it was declared or left to a default. A default for the target,
the comparison group or the clustering level returns `caution`, because a
choice made by omission still moves the number.

| Argument | Default | What it fixes |
| --- | --- | --- |
| `target` | `'overall'` | average over treated unit-periods; `'event'`, `'group'`, `'calendar'` are the other aggregations |
| `control_group` | `'nevertreated'` | parallel trends against never-treated units; `'notyettreated'` adds the not-yet-treated |
| `cluster` | the unit id | the level treatment is assigned at |
| `weights` | none | unweighted effects are for the average treated unit, weighted ones for the population the units represent |
| `anticipation` | 0 | periods before treatment in which units may already respond |
| `event_window` | all event times, at most five on each side | the horizons reported and, for `target='event'`, averaged over |
| `estimator` | `'cs'` | `'sa'` (Sun-Abraham) and `'bjs'` (imputation) are the alternatives |
| `absorb` | none | fixed effects beyond unit and period, such as `'state#year'`; parallel trends is then assumed within those cells, and the estimator is `'bjs'` |

### 2. Design

The audit reads the panel before anything is estimated. It reports units,
periods, cohorts with their sizes and pre-treatment histories,
never-treated and always-treated units, balance, duplicate rows, and the
cluster count graded against the simulation grid of Ulloa-Perez et al.
(2025) [@ulloaperez2025comparative].

The step returns `fail`, and estimation is not run, when the declared
target cannot be identified. That happens when treatment switches back
off, when a (unit, period) pair repeats, when no unit starts treatment
inside the sample, or when the declared comparison group is empty. The
action names the way out, for example `control_group='notyettreated'` or
`sp.did_multiplegt_dyn` for a treatment that reverses.

It returns `caution` for an unbalanced panel, always-treated units, fewer
than ten treated or comparison clusters, a cluster count at or below the
weakest cell of the reference grid, or a design with no second
pre-treatment period.

Treatment timing can be given two ways. `cohort=` is a column with each
unit's first treated period (`0` or `NaN` for never treated). `treat=` is a
0/1 path; the cohort is then the first treated period, and the audit
checks that the path never switches back.

### 3. TWFE

The Goodman-Bacon (2021) [@goodmanbacon2021difference] decomposition gives
the share of the two-way fixed effects coefficient that comes from
comparisons using already-treated units as controls. Any positive share is
a `caution`, and the TWFE coefficient is kept as a diagnostic next to the
estimate of step 4. With one adoption date there is nothing to diagnose and
the step is `info`.

### 4. Estimate

The declared target is estimated under the declared comparison group.
With `estimator='cs'` this is `sp.callaway_santanna` followed by the
`sp.aggte` type that matches `target`. With `'sa'` or `'bjs'` the overall
target is the estimator's own headline, and the event target is the
equal-weight average of the post-treatment path with the standard error
from the joint covariance of that path.

### 5. Event study

The path is reported with pointwise intervals and simultaneous (sup-t)
bands from `sp.uniform_bands`. The column `n_cohorts` counts the cohorts
observed at each event time. When that count changes along the
post-treatment path the step returns `caution`, because differences
between horizons then mix dynamics with a change in who is being averaged.
A window that every cohort reaches removes the problem.

### 6. Pre-trends

The joint test of the pre-treatment coefficients uses their full
covariance. A rejection is a `fail`. A non-rejection is never read as
support for parallel trends. Following Roth (2022) [@roth2022pretest], the
step reports the linear trend that the test would detect only 80 percent
of the time and the bias such a trend would put on the average
post-treatment effect. If that bias is as large as the estimate itself the
test could not have detected a problem of the size that matters, and the
step returns `caution`.

### 7. Sensitivity

Rambachan and Roth (2023) [@rambachan2023more] bounds are computed for the
equal-weight average of the post-treatment path. The step reports the
breakdown value under relative magnitudes (how many times the largest
pre-period violation the sign survives) and under the smoothness
restriction, plus the robust interval at `Mbar` of 0, 0.5, 1 and 2. The
step passes when the breakdown value is at least 1, the reference case in
which the post-treatment violation is as large as the largest one seen
before treatment. When the interval under exact parallel trends already
includes zero there is nothing to break down, and the step is `info`.

### 8. Robustness

Each row of `wf.robustness` changes one choice.

- The rest of the estimator family on one estimand (`sp.did_summary`).
- The other comparison group.
- The estimate without covariates, when covariates were given.
- A cluster-jackknife standard error. It refits once per cluster, so it
  runs unprompted only with 100 clusters or fewer; `jackknife=True` forces
  it.
- The functional-form test of Roth and Sant'Anna (2023)
  [@roth2023when], which asks whether parallel trends can hold for every
  monotone transformation of the outcome.

The step returns `caution` when an alternative changes the sign, leaves
the headline interval, or has an interval that includes zero while the
headline's does not. It also returns `caution` when the jackknife interval
includes zero where the analytic one does not, and when the
functional-form test rejects.

### 9. Report

`wf.checklist` lists what a paper has to state for the number to be
identifiable: target, assumption, sample, weights, anticipation, reference
period, window, estimator, standard-error construction, the estimate, the
pre-trend evidence, the sensitivity result, the robustness range and the
caveats still open. `wf.report()` returns the same content as one
paragraph, and `wf.cite()` returns the `paper.bib` keys of the methods that
were actually run (`sp.bibtex(wf.cite().split())` resolves them).

## Effects by subgroup

Heterogeneity is a design of its own. Baker et al. (2026, section 4.5)
[@baker2026difference] split the units into the cells of a discrete
characteristic and run the same DiD inside each cell. Every cell then has
its own treated and comparison units and its own parallel-trends
assumption.

```python
table = wf.heterogeneity("region")
table[["region", "share_of_treated", "estimate", "se"]]
table.attrs["equality_test"]
```

Each row is the workflow's headline on the rows of one cell. Nothing new
is estimated. `share_of_treated` is the cell's share of the treated units,
weighted when the workflow has weights, and
`attrs["share_weighted_average"]` averages the cell effects with those
shares. A cell without treated units, or without comparison units, is
listed with the reason and no number.

The test that the cell effects are equal is reported only when no cluster
spans two cells. Estimates from cells that share clusters are correlated,
and a test that treats them as independent would be wrong.

The characteristic must be constant within the unit. One that moves over
time may have been changed by the treatment; split on its pre-treatment
value. More than 20 cells is refused: bin the variable first.

## The eight steps of Baker et al.

The guide closes with a checklist. This is where the workflow answers
each item.

| Baker et al. (2026) | In the workflow |
| --- | --- |
| 1. Define the target parameter | Step 1: `target`, `weights`; the estimand is printed with every estimate |
| 2. State the identification assumptions and the evidence | Step 1 states the parallel-trends assumption the estimator imposes, with covariates and absorbed effects. Step 2 holds the covariate balance table in levels and changes. Step 6 holds the pre-trend test and its power; step 8 the test of sensitivity to the outcome scale |
| 3. Choose the estimation method | Step 4: `estimator`, and `estimator_options` for regression adjustment, weighting or the doubly robust form |
| 4. Discuss the sources of uncertainty | Step 1: `cluster`. Step 2 counts treated and comparison clusters. The report names what is treated as random |
| 5. Estimate | Steps 4 and 5: the summary effect, and the event study with pointwise and simultaneous bands |
| 6. Sensitivity analysis | Step 7: the largest violation of parallel trends the conclusion survives |
| 7. Heterogeneity analysis | `wf.heterogeneity(by)`, and `target='group'` or `'calendar'` for effects by cohort or date |
| 8. Keep learning | Each step's `action` says what to do when it fails |

Two points of wording follow the guide. The imputation estimator
(`estimator='bjs'`) fits unit and period effects on every untreated
observation. Its assumption is parallel trends between all groups in every
period, pre-treatment periods included. That is stronger than a comparison
with the not-yet-treated from the period before treatment, and step 1 says
which one is in force. The covariate balance table is reported and not
graded: the covariates are already conditioned on, and the table shows
what that conditioning has to absorb. The benchmark of 0.25 standard
deviations is the one the guide states, with its own caveat that no
threshold is universal.

## Reading the statuses

| Status | Meaning |
| --- | --- |
| `pass` | the step found nothing that qualifies the result |
| `caution` | the result stands, and the write-up owes the reader a caveat |
| `fail` | the step produced evidence against the design; a `fail` in step 2 also stops estimation |
| `info` | the step reports and does not grade |
| `skipped` | not run, with the reason in the finding |
| `pending` | not reached yet |

`wf.verdict` is the worst status across the steps. A status summarises the
evidence in `wf.step(key).evidence`. It is a reading aid and does not
replace that evidence.

The benchmarks behind the statuses are stated in the cited papers. `Mbar`
of 1 is the reference case of Rambachan and Roth. The pre-trend caution
compares the bias from a trend detected with 80 percent power to the
estimate, the comparison Roth recommends reporting. Ten clusters is the
count at which the staggered estimators in StatsPAI already warn.

## One step at a time

```python
wf = sp.did_workflow(df, y="lemp", id="countyreal", time="year",
                     cohort="first_treat", through=None)   # the plan
wf.steps[["step", "key", "status"]]

step = wf.advance()        # runs 'declare' and returns it
print(step.summary())

wf.run("design")           # run up to and including the design audit
wf.design["cohorts"]       # cohort sizes and pre-treatment histories

wf.run("pretrends")        # estimate, path, pre-trend test
wf.step("pretrends").evidence

wf.run()                   # everything that is still pending
```

`skip=` leaves out `'twfe'`, `'pretrends'`, `'sensitivity'` or
`'robustness'`. The other steps are what the later ones are computed from.

## What comes back

| Attribute | Content |
| --- | --- |
| `wf.steps` | one row per step: status, finding, action |
| `wf.step(key)` | the step with its `evidence` (numbers and tables) and `objects` (fitted results) |
| `wf.estimate`, `wf.se`, `wf.ci`, `wf.pvalue`, `wf.estimand` | the declared target |
| `wf.fit` | the estimator's own result, for example the group-time effects |
| `wf.event_study` | the path with pointwise and simultaneous intervals |
| `wf.event_study_result` | the result object to pass to `sp.honest_did` or `sp.uniform_bands` |
| `wf.sensitivity` | breakdown values and the relative-magnitude grid |
| `wf.robustness` | the headline next to every alternative |
| `wf.checklist`, `wf.report()`, `wf.cite()` | the reporting statement and references |
| `wf.plot()` | the path with both kinds of interval |
| `wf.to_markdown()`, `to_latex()`, `to_excel()`, `to_word()`, `to_dict()` | exports |
| `wf.degradations` | supporting checks that could not be computed, with the reason |

A supporting check that cannot be computed is left out, recorded in
`wf.degradations` and announced with a `WorkflowDegradedWarning`. The main
estimate is never optional: an error there is raised.

## For agents

`wf.to_dict(detail="minimal")` returns the verdict, the estimate and the
status of each step. `detail="standard"` adds findings, actions, the
declared specification and `next_steps`. `detail="agent"` adds the evidence
tables. `wf.violations()` lists the steps that failed or left a caveat in
the same shape as every other StatsPAI result, and `wf.next_steps()` gives
the next pending step or the actions still owed.

```python
wf = sp.did_workflow(df, y="y", id="unit", time="time", treat="d")
if wf.verdict == "fail":
    for v in wf.violations():
        print(v["test"], "->", v["recovery_hint"])
```

## Which entry point

| You want | Call |
| --- | --- |
| The graded protocol, start to finish or step by step | `sp.did_workflow` |
| One estimate | `sp.did(method=...)` or the estimator itself |
| A Callaway-Sant'Anna report with all four aggregations and covariate triangulation | `sp.cs_report` |
| The estimator family side by side on one estimand | `sp.did_summary` |
| What a fitted result pins down against the forward-engineering recipe | `sp.did_design_contract` |
| A quick bundle of estimate, path and sensitivity | `sp.did_analysis` |

Treatment that switches on and off is outside this protocol; use
`sp.did_multiplegt_dyn` or `sp.panel_match`. For repeated cross-sections
without a unit id use `sp.callaway_santanna(panel=False)`.

See also [Choosing a DID estimator](choosing_did_estimator.md) and
[Sensitivity analysis](honest_did.md).
