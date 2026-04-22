# Staggered Difference-in-Differences — Callaway & Sant'Anna (2021)

StatsPAI implements the Callaway–Sant'Anna estimator from first
principles, matching the R `did` package's core functionality while
adding new convenience layers on top.

## Basic usage

```python
import statspai as sp

cs = sp.callaway_santanna(
    df,
    y='earnings',        # outcome
    g='first_treat',     # first-treatment period (0 = never-treated)
    t='year',            # time period
    i='worker_id',       # unit id
    estimator='dr',      # 'dr' (default), 'ipw', or 'reg'
    control_group='nevertreated',  # or 'notyettreated'
    anticipation=0,      # periods of anticipation (CS2021 §3.2)
)

print(cs.summary())
cs.detail              # one row per (group, time) with ATT + pointwise CI
cs.model_info['event_study']   # event-study aggregation
cs.model_info['pretrend_test'] # joint Wald pre-trend test
```

## Aggregation with uniform bands

The raw `callaway_santanna()` result is a grid of ATT(g, t) estimates.
Collapse to a scalar or an event-study curve with `aggte()`, which
layers the Mammen (1993) multiplier bootstrap on top and returns
*simultaneous* confidence bands:

```python
es = sp.aggte(cs, type='dynamic',
              n_boot=500, random_state=0,
              balance_e=3)        # balance across cohorts for e ≤ 3

print(es.detail)
# relative_time  att  se  ci_lower  ci_upper  cband_lower  cband_upper ...
```

The `cband_lower` / `cband_upper` columns give a sup-t uniform band —
valid for simultaneous inference across the entire event window,
unlike the pointwise CI.

Other aggregation types:

| `type=` | Meaning |
| --- | --- |
| `'simple'` | cohort-share-weighted overall ATT |
| `'dynamic'` | event-study curve ATT(e) |
| `'group'` | per-cohort average ATT(g) |
| `'calendar'` | per-calendar-time ATT(t) |

## Repeated cross-sections

Pass `panel=False` when observations are not matched across time
(e.g. CPS pooled cross-sections).  The estimator switches to the
unconditional 2×2 cell-mean DID with observation-level influence
functions; downstream `aggte`, `cs_report`, `ggdid`, and `honest_did`
all work unchanged.

```python
cs_rcs = sp.callaway_santanna(
    survey_df,
    y='wage', g='first_treat', t='year', i='respondent_id',
    estimator='reg',         # only 'reg' supported in RCS mode
    x=['age', 'education'],  # optional covariate residualisation
    panel=False,
)
```

## Sensitivity — Rambachan & Roth (2023)

Every event-study result (from CS, SA, BJS, or `aggte`) feeds into
the Rambachan–Roth sensitivity framework:

```python
sens = sp.honest_did(es, e=2)     # robust CI at e=2 across an M grid
m_star = sp.breakdown_m(es, e=2)  # largest M* under which effect is significant
```

## One-call report

For a ready-to-publish summary — raw estimation + four aggregations
with uniform bands + pre-trend Wald + R-R breakdown M\* per post
event time — call [`cs_report()`](cs_report.md).

<!-- AGENT-BLOCK-START: callaway_santanna -->

## For Agents

**Pre-conditions**
- panel data with unit × time × outcome
- g column is integer: first-treated period or 0 for never-treated
- at least one never-treated or late-treated control group
- ≥ 2 pre-treatment periods per cohort

**Identifying assumptions**
- Parallel trends conditional on X (if covariates supplied)
- No anticipation (or adjust via anticipation= parameter)
- Overlap: positive propensity for each cohort
- SUTVA

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Pre-trend test on aggregated ATT(g,t) rejects | `AssumptionViolation` | Use sp.sensitivity_rr for honest CI, or add covariates for conditional parallel trends. | `sp.sensitivity_rr` |
| Cohort with only one unit — insufficient variation | `DataInsufficient` | Aggregate small cohorts or drop; check sp.diagnose_result. |  |
| All units treated at the same time (no staggering) | `MethodIncompatibility` | Fall back to 2x2 DID via sp.did(method='2x2'). | `sp.did` |

**Alternatives (ranked)**
- `sp.sun_abraham`
- `sp.did_imputation`
- `sp.sdid`
- `sp.did`

**Typical minimum N**: 50

<!-- AGENT-BLOCK-END -->
