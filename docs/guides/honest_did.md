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

Both functions are *polymorphic*: they accept

- a `callaway_santanna()` result (event study in `model_info`),
- a `sun_abraham()` result,
- or an `aggte(type='dynamic')` result (event study in `detail` with
  Mammen uniform bands).

## Typical pipeline

```python
cs  = sp.callaway_santanna(df, y='y', g='g', t='t', i='id')
es  = sp.aggte(cs, type='dynamic', n_boot=500, random_state=0)
s   = sp.honest_did(es, e=1)          # robust CI table across M grid
mst = sp.breakdown_m(es, e=1)         # scalar breakdown value
```

## Restriction types

- `method='smoothness'` — bounds $|\Delta \delta_t| \le M$ for every
  consecutive pair of periods (Δ^SD(M)).  The `M` parameter has units
  of the outcome per period.
- `method='relative_magnitude'` — post-period violation bounded as
  `M̄ × max|δ_pre|`.  Useful when you want to express the sensitivity
  relative to the magnitude of the observed pre-trend violations.

## One-shot via `cs_report()`

The [`cs_report`](cs_report.md) report card already runs
`breakdown_m` at every post-treatment event time and stores the
results in `rpt.breakdown`.  Column `robust_at_1_SE` flags whether
the effect survives a violation equal to one pointwise standard error.
