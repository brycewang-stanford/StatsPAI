# A staggered local treatment on a multi-level panel

Many applied DiD papers share one data shape. Outcomes are recorded for
units nested in places, such as species within counties or plants within
cities. Places adopt a treatment in different years. The regression is

```text
log y(i, t) = b * Post x Treat + unit effects
              + region-by-year effects + group-by-year effects + error
```

with standard errors clustered on the place. The paper then adds an event
study, a heterogeneity table, a spillover table and often an instrument.

This guide runs that design in StatsPAI. It shows where the two-way
regression can mislead, and which calls replace each table.

## Data

The example simulates 80 counties in 8 states over 20 years, with 6
species per county. A county that installs a facility keeps it. The true
effect is -0.4 log points, and -0.6 for two low-flying species, so the
average is -0.467. Each state has its own year shocks.

```python
import warnings

import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(7)
years = np.arange(2000, 2020)
n_county, n_species, n_state = 80, 6, 8
state = rng.integers(0, n_state, n_county)
xy = rng.uniform(0, 600, (n_county, 2))                 # km
first = np.where(rng.random(n_county) < 0.35,
                 rng.choice(np.arange(2006, 2016), n_county), 0)
state_year = rng.normal(0, 0.4, (n_state, len(years)))
species_year = rng.normal(0, 0.3, (n_species, len(years)))
rows = []
for c in range(n_county):
    for s in range(n_species):
        level = rng.normal()
        for k, t in enumerate(years):
            d = int(first[c] > 0 and t >= first[c])
            effect = -0.4 * d * (1.5 if s < 2 else 1.0)
            y = (level + state_year[state[c], k] + species_year[s, k]
                 + effect + rng.normal(0, 0.5))
            rows.append((c, s, t, state[c], first[c], d, int(s < 2),
                         xy[c, 0], xy[c, 1], y))
df = pd.DataFrame(rows, columns=["county", "species", "year", "state", "first_pv",
                                 "pv", "low_flying", "x_km", "y_km", "log_birds"])
df["unit"] = df["county"] * 100 + df["species"]
```

The unit is the county-species pair. `first_pv` is the year the county
first has a facility and 0 for a county that never has one.

## The two-way regression

```python
twfe = sp.feols("log_birds ~ pv | unit + state:year + species:year", df,
                vcov={"CRV1": "county"})
```

The coefficient is -0.455 (standard error 0.021). `sp.feols` also warns.
The treatment switches on at 10 different dates, and part of the
coefficient's weight sits on comparisons that use already-treated counties
as controls. With effects that grow over time those comparisons bias the
coefficient. The warning counts dates on the calendar even though the
regression absorbs `state:year` and has no year effect of its own.

## The workflow with absorbed effects

`sp.did_workflow` takes the extra fixed effects through `absorb=`.

```python
wf = sp.did_workflow(
    df, y="log_birds", id="unit", time="year", cohort="first_pv",
    cluster="county", target="overall",
    absorb=["state#year", "species#year"],
)
wf.summary()
```

Three things change with `absorb=`.

- The assumption in step 1 reads "parallel trends within cells of
  state#year, species#year". A shock common to a state in a year is no
  longer a violation.
- The estimator is Borusyak-Jaravel-Spiess imputation, the one of the
  three that takes fixed effects beyond unit and period. `estimator="cs"`
  and `"sa"` are refused.
- Step 8 reports the same estimator with unit and period effects only.

Here the estimate is -0.466 (standard error 0.018), against a truth of
-0.467. With unit and period effects only it is -0.376. The state shocks
did not difference out, because states adopt at different times.

Every number is the output of `sp.did_imputation(fe=[...])`. The workflow
adds no estimator.

## Long panels

This panel offers event times from -15 to 13. A report cannot use all of
them. The joint pre-test spreads its power over every lead, and the
sensitivity sets of step 7 become intractable. When `event_window` is not
given, steps 5 to 7 use the five event times nearest to treatment on each
side, and step 2 says so:

```text
Event times run from -15 to 13; with no event_window given, steps 5 to 7
report -5 to 5. Pass event_window= to report another range.
```

A panel with five or fewer event times a side is reported in full. An
explicit `event_window=(-8, 6)` is never changed.

## Heterogeneity

The paper's triple interaction asks whether one group loses more. Under
staggered adoption the interaction inherits the problems of the two-way
coefficient. Estimate the effect for each group with the same model:

```python
het = sp.did_imputation(
    df, y="log_birds", group="unit", time="year", first_treat="first_pv",
    cluster="county", fe=["unit", "state#year", "species#year"],
    hetby="low_flying",
)
het.model_info["hetby"]
```

The low-flying species lose 0.571 (truth 0.6) and the others 0.414
(truth 0.4).

The workflow offers the same split as a design of its own:

```python
wf.heterogeneity("low_flying")
```

It runs the whole declared analysis inside each cell, with that cell's
own comparison units: 0.558 for the low-flying species and 0.420 for the
others. The cells hold two thirds and one third of the treated units, and
their share-weighted average is -0.466, the headline. No test of equality
is reported here, because every county holds species of both kinds and the
two estimates share clusters.

## Spillovers

Birds cross county lines, so an untreated neighbour may not be a clean
control. `sp.spillover_did` splits untreated counties by distance to the
nearest treated one. Counties beyond the last ring are the clean controls.

```python
county = df.groupby(["county", "year"], as_index=False).agg(
    log_birds=("log_birds", "mean"), first_pv=("first_pv", "first"),
    x_km=("x_km", "first"), y_km=("y_km", "first"))
ring = sp.spillover_did(
    county, y="log_birds", unit="county", time="year", cohort="first_pv",
    coords=["x_km", "y_km"], ring_edges=(0.0, 100.0),
)
ring.summary()
```

The simulation has no spillover, and the ring effect is 0.050 (standard
error 0.102). The summary also warns that only 6 counties lie beyond 100
km of a facility. That is a property of the geography, and it would apply
to the paper's table as well.

## Event dates for untreated places

Some designs give every untreated place the event date of a matched or
nearby treated place, so that `Post` is defined for controls too.
`sp.assign_placebo_dates` draws such dates from the distribution of the
real ones. It reads a cohort of 0, an infinite value or a missing value as
never treated.

```python
dated = sp.assign_placebo_dates(
    county, id="county", event_time="first_pv", time="year", seed=0)
```

## An instrument

When adoption is instrumented, the fixed effects stay in the formula:

```python
sp.feols("log_birds ~ 1 | unit + state:year + species:year | pv ~ z_post",
         df, vcov={"CRV1": "county"})
```

The result reports the first-stage F and, when the instrument is weak,
replaces the conventional interval. The instrument inherits the staggered
problem of the two-way regression, so read it next to the workflow.

## Counts and logs

A log outcome with zeros needs a choice. `log(y + 1)` has no
scale-free interpretation. `sp.fepois` fits the multiplicative model on
the counts with the same fixed effects, and step 8 of the workflow tests
whether parallel trends depends on the scale of the outcome.

## What to report

1. The declared target, comparison group and clustering level.
2. The estimate from the workflow with the absorbed effects, next to the
   two-way coefficient.
3. The event-study path over a stated window, with the simultaneous band.
4. The pre-trend test and the trend it would miss.
5. The breakdown value from step 7.
6. Group effects from `hetby=`, and ring effects if spillovers are
   plausible.
