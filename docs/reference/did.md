# DiD module — API reference

## Core estimators

### `sp.callaway_santanna(data, y, g, t, i, ...)`

Callaway & Sant'Anna (2021) staggered DID.  See
[the guide](../guides/callaway_santanna.md) for an end-to-end
walkthrough.

Key arguments:

| Name | Default | Meaning |
| --- | --- | --- |
| `estimator` | `'dr'` | `'dr'` (Sant'Anna–Zhao doubly robust), `'ipw'`, or `'reg'` |
| `control_group` | `'nevertreated'` | Or `'notyettreated'` |
| `base_period` | `'universal'` | Or `'varying'` |
| `anticipation` | `0` | Periods of anticipation (CS2021 §3.2) |
| `panel` | `True` | `False` for repeated cross-sections |
| `x` | `None` | Covariate list |

### `sp.aggte(result, type='simple', ...)`

Unified aggregation with Mammen (1993) multiplier bootstrap.

`type` ∈ `{'simple', 'dynamic', 'group', 'calendar'}`.

Relevant arguments:

| Name | Default | Meaning |
| --- | --- | --- |
| `n_boot` | 1000 | Multiplier-bootstrap replications |
| `random_state` | `None` | RNG seed |
| `balance_e` | `None` | Balance cohorts across `e ∈ [0, balance_e]` |
| `min_e` / `max_e` | `-inf` / `inf` | Event-time window |
| `cband` | `True` | Attach uniform confidence band columns |

Result's `detail` frame carries `cband_lower` / `cband_upper` for
all aggregations except `simple`.

### `sp.cs_report(data_or_result, ..., save_to=None)`

One-call report card composing the full pipeline.  See
[the guide](../guides/cs_report.md).  Returns a `CSReport`
dataclass with `.plot()`, `.to_markdown()`, `.to_latex()`,
`.to_excel()`, and `.to_text()` methods.

### `sp.sun_abraham(data, y, g, t, i, ...)`

Sun & Abraham (2021) interaction-weighted event study with
Liang–Zeger cluster-robust sandwich SEs.

### `sp.did_imputation(data, ..., horizon=None)`

Borusyak, Jaravel & Spiess (2024) imputation estimator.

### `sp.did_multiplegt(data, ..., placebo=0, dynamic=0)`

de Chaisemartin–D'Haultfoeuille DID for switch-on-off treatments,
with dCDH 2024 joint placebo Wald and average cumulative effect.

## Sensitivity

### `sp.honest_did(result, e, m_grid=None, method='smoothness')`
### `sp.breakdown_m(result, e, method='smoothness')`

Rambachan & Roth (2023).  Accept any CausalResult carrying an event
study (CS, SA, BJS, or `aggte(dynamic)`).

### `sp.bjs_pretrend_joint(result, data, ..., n_boot=300, seed=None)`

Cluster-bootstrap joint Wald pre-trend test for BJS imputation
results.  Upgrades the default sum-of-z² test (which assumes
pre-period independence) to the full covariance-aware test.

## Visualisation

### `sp.ggdid(result, ax=None, ...)`

`aggte()` visualiser with uniform-band overlay.  Dispatches on
`result.model_info['aggregation']`.
