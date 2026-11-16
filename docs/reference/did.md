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
Also available through the unified entry point:

```python
sp.did(df, y="y", treat="first_treat", time="year", id="unit",
       method="bjs", horizon=list(range(-4, 5)))
```

`horizon` attaches the BJS event-study table to
`result.model_info["event_study"]`, so `result.plot()` and
`sp.honest_did(result, ...)` work the same way as CS/SA event studies.

### `sp.did_multiplegt(data, ..., placebo=0, dynamic=0)`

de Chaisemartin–D'Haultfoeuille DID for switch-on-off treatments,
with dCDH 2024 joint placebo Wald and average cumulative effect.

## Inference

### `sp.event_study_vcov(result, allow_diagonal=False)`

The joint covariance of an event study, read off whichever estimator
produced it (CS / `aggte`, `event_study`, `sun_abraham`, `gardner_did`,
`did_imputation`, `stacked_did`, `lp_did`, `did_multiplegt_dyn`, `etwfe`).
Returns `times`, `beta`, `vcov`, `joint` and `source`. When the fit has
no joint covariance it raises (since 1.44); `allow_diagonal=True` returns
`diag(se**2)` instead, flagged `joint=False`.

### `sp.uniform_bands(result, alpha=0.05, which='all', window=None)`

Sup-t simultaneous band over the covered event times — the object an
event-study *plot* needs, since a pointwise interval covers one horizon at
a time. Falls back to the conservative Sidak critical value when the
covariance is not joint.

### `sp.did_few_treated(data, y, id, time, treat, method='conley_taber')`

Conley–Taber (2011) and Ferman–Pinto (2019) inference for designs with one
or a handful of *treated* clusters, where the cluster-robust variance
over-rejects whatever the total cluster count. The placebo distribution is
built from the control groups and inverted; `method='ferman_pinto'`
additionally rescales it for the heteroskedasticity unequal group sizes
generate (needs `group_size=`).

### `sp.did_calibrated_simulation(data, y, id, time, cohort, estimators=..., effect=0.0)`

Scores candidate estimators on a placebo simulation calibrated to your own
panel: the estimated dynamic effect is removed with the imputation fit, the
adoption pattern is redrawn, a known effect is injected, and every candidate
is refit `n_sims` times. Returns bias, RMSE, coverage, rejection rate and the
ratio of mean reported SE to realised dispersion, each with a Monte Carlo
standard error, plus `.best('rmse')`. The redrawn assignment is random, so
parallel trends holds by construction and the table is about the estimators,
not the design. `stacked_did` and `lp_did` estimate over an event *window*,
which need not cover the same cells as the overall ATT — read a gap for them
as a possible estimand difference.

### `sp.cs_jackknife(data, y, g, time, id, type='simple')`

Delete-one-cluster (CV3) jackknife of a Callaway–Sant'Anna aggregate,
against R `didjack` and Stata `csdidjack`.

## Sensitivity

### `sp.honest_did(result, e, m_grid=None, method='smoothness')`
### `sp.breakdown_m(result, e, method='smoothness')`

Rambachan & Roth (2023).  Accept any CausalResult carrying an event study;
the fixed-length interval uses the joint covariance `sp.event_study_vcov`
recovers, and `result.attrs['interval']` records whether the returned
table is the FLCI or the worst-case-bias fallback.

### `sp.bjs_pretrend_joint(result, data, ..., n_boot=300, seed=None)`

Cluster-bootstrap joint Wald pre-trend test for BJS imputation
results.  Upgrades the default sum-of-z² test (which assumes
pre-period independence) to the full covariance-aware test.

## Visualisation

### `sp.ggdid(result, ax=None, ...)`

`aggte()` visualiser with uniform-band overlay.  Dispatches on
`result.model_info['aggregation']`.
