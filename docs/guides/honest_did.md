# Sensitivity to parallel trends — Rambachan & Roth (2023)

Parallel trends is an assumption, not a fact.  Rambachan & Roth
(2023) answer the practical question: *how wrong could parallel
trends be before my post-treatment conclusion flips?*  StatsPAI wires
their tooling directly into every event-study object in the library.

## Functions

| Function | Purpose |
| --- | --- |
| `sp.honest_did(result, e, m_grid=None, method='smoothness')` | Robust CI at event time `e` across a grid of violation magnitudes `M` |
| `sp.breakdown_m(result, e, method='smoothness')` | Largest `M*` at which the effect remains significant |

Both functions are *polymorphic*: since 1.30 they accept any fitted
event study in the library — `callaway_santanna()` and
`aggte(type='dynamic')`, `event_study()`, `sun_abraham()`,
`gardner_did(event_study=True)`, `did_imputation(horizon=...)`,
`stacked_did()`, `lp_did()`, `did_multiplegt_dyn()`, and `etwfe()` (raw
fit or `etwfe_emfx(type='event')`).

The fixed-length interval is a function of the event-study vector **and
its joint covariance**, which `sp.event_study_vcov(result)` reads off
each estimator. Where that covariance is genuinely unavailable —
`did_imputation(pretrend_method='bjs')`, whose leads come from a
separate auxiliary regression — `honest_did` falls back to a
worst-case-bias interval, says so in a warning, and records which one it
returned in `result.attrs['interval']` (`'flci'` or
`'worst_case_bias'`).

> **⚠️ 1.30 correctness fix.** Handed a *raw* `callaway_santanna()` fit,
> earlier versions rebuilt that covariance with the estimated cohort
> shares held fixed. The diagonal matched the reported standard errors,
> so nothing looked wrong, but the off-diagonal blocks were up to 8
> percent from R `did`'s and the FLCI built on them was too narrow — on
> the `cs_report` demo panel the breakdown `M*` came out about three
> times too large. Re-run any HonestDiD sensitivity taken from a raw fit.

## Simultaneous bands on the same covariance

```python
band = sp.uniform_bands(fit, which='post')      # sup-t over the post path
band.attrs['crit_uniform'], band.attrs['crit_pointwise']
```

A 95 percent *pointwise* interval covers one horizon; a plot invites a
statement about the whole path, which needs the sup-t critical value —
the `1 - alpha` quantile of `max_j |Z_j|` with `Z ~ N(0, R)`. On a
five-horizon window it is typically 2.4 to 2.5 against 1.96, so the band
is 20 to 30 percent wider, and horizons that are individually
significant often stop being so. `which='pre'` does the same for a
visual pre-trend check, and `window=(lo, hi)` restricts the band to the
horizons a figure actually displays (the critical value depends on how
many are covered).

## Typical pipeline

```python
cs  = sp.callaway_santanna(df, y='y', g='g', t='t', i='id')
es  = sp.aggte(cs, type='dynamic', n_boot=500, random_state=0)
s   = sp.honest_did(es, e=1)          # robust CI table across M grid
mst = sp.breakdown_m(es, e=1)         # scalar breakdown value
```

## Restriction types

- `method='smoothness'` — bounds the change in the slope of the
  violation, $|\Delta^2 \delta_t| \le M$ (Δ^SD(M)).  The `M` parameter
  has units of the outcome per period.
- `method='relative_magnitude'` — post-period changes in the violation
  bounded as `M̄ × max|Δδ_pre|` (Δ^RM).  Useful when you want to express
  the sensitivity relative to the observed pre-trend violations.
- `method='smoothness_relative_magnitude'` — the same, on second
  differences (Δ^SDRM, HonestDiD's "deviation from linear trend"):
  post-period curvature at most `M̄` times the largest pre-period
  curvature. Needs two or more pre-periods.

Each can be combined with one extra restriction, as in HonestDiD:

- `bias_direction='positive'` / `'negative'` — the violation pushes the
  post-period estimate one way only (Δ^SDPB, Δ^RMNB, ...);
- `monotonicity_direction='increasing'` / `'decreasing'` — the violation
  is monotone in event time (Δ^SDI, Δ^SDRMD, ...).

`honestdid_method` picks the confidence set: `'FLCI'` (smoothness default),
`'C-F'` (the conditional-FLCI hybrid, smoothness default once a sign or
shape restriction is given, since the FLCI ignores it), `'Conditional'`
(the Andrews-Roth-Pakes conditional test) or `'C-LF'` (the conditional
least-favourable hybrid, default for the relative-magnitude families). All
run natively; `'Conditional'` accepts the same grid points as R HonestDiD
(`tests/reference_parity/test_honest_restrictions_R_parity.py`).

```python
sp.honest_did(es, e=0, method="smoothness", bias_direction="negative")
sp.honest_did(es, e=0, method="relative_magnitude",
              monotonicity_direction="increasing")
sp.honest_did(es, e=0, method="smoothness_relative_magnitude")
```

`sp.did_imputation` fits with BJS pre-trend leads now carry the joint
lead-horizon covariance (as Stata `did_imputation`'s `e(V)`), so they work
here too.

## One-shot via `cs_report()`

The [`cs_report`](cs_report.md) report card already runs
`breakdown_m` at every post-treatment event time and stores the
results in `rpt.breakdown`.  Column `robust_at_1_SE` flags whether
the effect survives a violation equal to one pointwise standard error.

<!-- AGENT-BLOCK-START: did -->

## For Agents

**Pre-conditions**
- data is panel or repeated cross-section with a time column
- treat column is binary (0/1) for 2x2, or first-treatment-period (int) for staggered
- at least one pre-treatment period (≥ 2 periods for 2x2; ≥ 3 recommended for event study)
- for staggered designs: id column identifying units across time

**Identifying assumptions**
- Parallel trends: treated and control groups would have followed the same trajectory absent treatment
- No anticipation: outcomes in pre-treatment periods are unaffected by future treatment
- SUTVA: no spillovers between units
- For staggered / heterogeneous effects: use CS or SA — TWFE can produce negative weights (Goodman-Bacon)

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Pre-trend joint test p < 0.05 (or underpowered at 0.10) | `AssumptionViolation` | Use sp.sensitivity_rr (Rambachan & Roth honest CI) or switch to sp.callaway_santanna. | `sp.sensitivity_rr` |
| Staggered treatment timing with TWFE method | `AssumptionWarning` | TWFE can give negative weights; use Callaway-Sant'Anna, Sun-Abraham, or BJS imputation. | `sp.callaway_santanna` |
| Pre-trend test underpowered (Roth 2022) | `AssumptionWarning` | Check sp.pretrends_power — if low, report honest CI via sp.sensitivity_rr. | `sp.sensitivity_rr` |
| Few clusters at unit level | `AssumptionWarning` | Use wild cluster bootstrap (sp.wild_cluster_bootstrap). | `sp.wild_cluster_bootstrap` |
| Few *treated* clusters (one or a handful) | `AssumptionWarning` | Cluster-robust SEs over-reject whatever the total cluster count; use sp.did_few_treated (Conley-Taber / Ferman-Pinto) or sp.cs_jackknife (CV3). | `sp.did_few_treated` |

**Alternatives (ranked)**
- `sp.callaway_santanna`
- `sp.sun_abraham`
- `sp.did_imputation`
- `sp.sdid`
- `sp.synth`

**Typical minimum N**: 50

<!-- AGENT-BLOCK-END -->
