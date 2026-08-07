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
layers the multiplier bootstrap on top (Rademacher weights, matching
the R `did` implementation) and returns *simultaneous* confidence
bands:

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

## Bootstrap inference at the ATT(g, t) level

R's `att_gt()` defaults to a multiplier bootstrap; Stata's `csdid`
offers it as `wboot`. The same inference surface is available directly
on `callaway_santanna()`:

```python
cs = sp.callaway_santanna(
    df, y='y', g='first_treat', t='year', i='id',
    bstrap=True,            # multiplier-bootstrap SEs (R: bstrap, Stata: wboot)
    biters=999,             # replications        (R: biters, Stata: reps())
    cband=True,             # uniform sup-t bands  (R: cband)
    random_state=42,
)

cs.detail[['group', 'time', 'att', 'se', 'cband_lower', 'cband_upper']]
cs.model_info['crit_val_uniform']   # sup-t critical value (> 1.96)
```

Option mapping:

| StatsPAI | R `did::att_gt` | Stata `csdid` |
| --- | --- | --- |
| `bstrap=True` | `bstrap=TRUE` (default in R) | `wboot` |
| `biters=999` | `biters=999` | `reps(999)` |
| `cband=True` | `cband=TRUE` | (default with `wboot`) |
| `clustervars=['id', 'state']` | `clustervars=c("id", "state")` | `cluster(state)` |
| `boot_weight_type='mammen'` | — (R draws Rademacher) | `wbtype(mammen)` |

Note StatsPAI defaults to `bstrap=False`, R `did` to `bstrap=TRUE`. The
two are different estimators of the same variance, so compare like with
like: StatsPAI's analytic path reproduces R's **analytic** standard
errors (`aggte(..., bstrap=FALSE)`) exactly, and `bstrap=True`
reproduces R's default bootstrap up to the draw. These are
influence-function standard errors, not a delta-method approximation —
the aggregation carries both the covariance between ATT(g, t) cells and
the sampling variability of the estimated cohort-share weights.

The default multiplier weights are Rademacher (±1) because that is what
R `did` actually draws (`BMisc::multiplier_bootstrap`), its Mammen
citation notwithstanding.

### Two-level clustering

`clustervars` mirrors R's `mboot` convention: the unit id is always
implied, at most one *additional* time-invariant variable is allowed,
and clustering requires the bootstrap (analytic SEs would silently
understate within-cluster dependence, so `clustervars` without
`bstrap=True` raises):

```python
cs = sp.callaway_santanna(
    df, y='y', g='first_treat', t='year', i='id',
    bstrap=True, clustervars=['id', 'state'], biters=999, random_state=42,
)
es = sp.aggte(cs, type='dynamic')   # inherits the clustering automatically
```

## Migrating from Stata `csdid`

The option names do **not** line up, and two of the mismatches change
your numbers silently. This table is the mapping.

| `csdid` | StatsPAI | Note |
| --- | --- | --- |
| `method(dripw)` | `estimator='dr'` | default both sides |
| `method(reg)` | `estimator='reg'` | |
| `method(stdipw)` | `estimator='ipw'` **or** `'stdipw'` | ⚠️ see below |
| `method(ipw)` | `estimator='ipw_abadie'` | ⚠️ see below |
| `wboot` | `bstrap=True` or `se_method='wboot'` | |
| `wboot(reps(999))` | `biters=999` | |
| `wboot(wtype(mammen))` | `boot_weight_type='mammen'` | csdid defaults to mammen, StatsPAI to rademacher — R `did` draws rademacher despite citing Mammen |
| `pointwise` | `cband=False` | csdid's default is *uniform*; StatsPAI's is pointwise |
| `long2` | `base_period='universal'` | StatsPAI's default |
| (csdid default gaps) | `base_period='varying'` | |
| `asinr` | `notyet_cutoff='period'` | StatsPAI's default |
| (csdid default) | `notyet_cutoff='cohort'` | |
| `notyet` | `control_group='notyettreated'` | |
| `pscoretrim(#)` | `pscore_trim=#` | both default 0.995 |
| `saverif(f)` | `sp.influence_functions(res, path=f)` | |
| `cluster(v)` | `clustervars=['v']` | requires `bstrap=True` |

### ⚠️ `ipw` means different things in the two ecosystems

StatsPAI follows **R `did`**, where `est_method='ipw'` dispatches to
`DRDID::std_ipw_did_panel` — the Hájek-*stabilized* estimator. Stata's
`method(ipw)` is Abadie (2005), which normalizes both arms by the same
`E[D]` and is a genuinely different estimator.

```python
# porting `csdid ..., method(ipw)`     -> estimator='ipw_abadie'
# porting `csdid ..., method(stdipw)`  -> estimator='ipw'  (or 'stdipw')
```

On `mpdta` the two differ by up to 2.4e-4 — small enough to look like
noise, large enough to change a marginal t-statistic. Both spellings are
pinned against Stata in
`tests/reference_parity/test_csdid_conventions_stata_parity.py`.

### ⚠️ `asinr` is a control-set convention, not a test

Despite the name, `asinr` does not test anything. It selects which date a
control must still be untreated at, for **pre-treatment** ATT(g,t) only:

- `notyet_cutoff='period'` — untreated as of `t` (R `did`, `csdid, asinr`)
- `notyet_cutoff='cohort'` — untreated as of `g` (`csdid`'s own default)

Post-treatment cells are identical either way. On `mpdta` the
pre-treatment placebos move in the third decimal, e.g. ATT(2007, 2004)
goes from 0.032971 to 0.033813.

## Influence-function export (`saverif` workflow)

Stata's `csdid, saverif()` saves the per-observation influence
functions so any custom aggregation can be computed later without
refitting. The StatsPAI equivalent:

```python
# Stage 1 — fit once, export the influence functions
cs = sp.callaway_santanna(df, y='y', g='first_treat', t='year', i='id')
sp.influence_functions(cs, path='cs_rif.csv')     # or .parquet

# Stage 2 — later / elsewhere: aggregate without the original data
es = sp.aggte_from_influence(
    'cs_rif.csv', type='dynamic',
    min_e=-4, max_e=8, bstrap=True, cband=True, random_state=0,
)
```

The export is self-contained (unit, cohort, (g, t) cell, ATT, influence
value, and the cluster label if the fit used `clustervars`), and the
round-trip is exact: `aggte_from_influence(influence_functions(cs), ...)`
reproduces `aggte(cs, ...)` to machine precision at the same seed.

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
- data is panel or repeated cross-section with a time column
- treat column is binary (0/1) for 2x2, or first-treatment-period (int) for staggered
- at least one pre-treatment period (≥ 2 periods for 2x2; ≥ 3 recommended for event study)
- for staggered designs: id column identifying units across time

**Identifying assumptions**
- Parallel trends conditional on X (if covariates supplied)
- No anticipation (or adjust via anticipation= parameter)
- Overlap: positive propensity for each cohort
- SUTVA
- Parallel trends: treated and control groups would have followed the same trajectory absent treatment
- No anticipation: outcomes in pre-treatment periods are unaffected by future treatment
- SUTVA: no spillovers between units
- For staggered / heterogeneous effects: use CS or SA — TWFE can produce negative weights (Goodman-Bacon)

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Pre-trend test on aggregated ATT(g,t) rejects | `AssumptionViolation` | Use sp.sensitivity_rr for honest CI, or add covariates for conditional parallel trends. | `sp.sensitivity_rr` |
| Cohort with only one unit — insufficient variation | `DataInsufficient` | Aggregate small cohorts or drop; check sp.diagnose_result. |  |
| All units treated at the same time (no staggering) | `MethodIncompatibility` | Fall back to 2x2 DID via sp.did(method='2x2'). | `sp.did` |
| Pre-trend joint test p < 0.05 (or underpowered at 0.10) | `AssumptionViolation` | Use sp.sensitivity_rr (Rambachan & Roth honest CI) or switch to sp.callaway_santanna. | `sp.sensitivity_rr` |
| Staggered treatment timing with TWFE method | `AssumptionWarning` | TWFE can give negative weights; use Callaway-Sant'Anna, Sun-Abraham, or BJS imputation. | `sp.callaway_santanna` |
| Pre-trend test underpowered (Roth 2022) | `AssumptionWarning` | Check sp.pretrends_power — if low, report honest CI via sp.sensitivity_rr. | `sp.sensitivity_rr` |
| Few clusters at unit level | `AssumptionWarning` | Use wild cluster bootstrap (sp.wild_cluster_bootstrap). | `sp.wild_cluster_bootstrap` |

**Alternatives (ranked)**
- `sp.sun_abraham`
- `sp.did_imputation`
- `sp.sdid`
- `sp.did`
- `sp.callaway_santanna`
- `sp.synth`

**Typical minimum N**: 50

<!-- AGENT-BLOCK-END -->
