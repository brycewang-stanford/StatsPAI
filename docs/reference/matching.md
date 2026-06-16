# Matching and balancing

`statspai.matching` covers classical matching, balancing weights, diagnostics,
and Love plots behind a unified `sp.match(...)` dispatcher plus standalone
estimator functions for power users.

See also the decision guide:
[Choosing a matching estimator](../guides/choosing_matching_estimator.md), and
the exhaustive auto-generated listing under
[Full API reference -> matching](api/matching.md).

## Choosing an entry point

```python
import statspai as sp

# Default nearest-neighbour propensity-score matching.
r = sp.match(
    df,
    y="earnings",
    treat="training",
    covariates=["age", "education", "earnings_pre"],
)

# Balancing-weight estimators are available through method=...
r_ebal = sp.match(df, y="earnings", treat="training",
                  covariates=["age", "education"], method="ebalance")

# ...or as standalone functions with estimator-specific options.
w = sp.overlap_weights(df, treat="training",
                       covariates=["age", "education", "earnings_pre"])
diag = sp.balance_diagnostics(df, treat="training",
                              covariates=["age", "education"],
                              weights=w.weights)
```

## Estimator families

| Family | Functions | Typical use |
| --- | --- | --- |
| Classical matching | `sp.match(method="nearest" | "psm" | "mahalanobis" | "cem" | "stratify")` | Matched samples or subclassification with transparent design choices. |
| Stata `psmatch2` | `sp.psmatch2` | Stata-faithful supported PSM paths with matched-sample variables (`_weight`/`_support`; plus `_n1`/`_pdif` for nearest-neighbour), post-matching balance, common-support plot, and frequency-weighted PSM-DID. See the [PSM-DID guide](../guides/psm_did.md). |
| Entropy / CBPS / SBW | `sp.ebalance`, `sp.cbps`, `sp.sbw` | Direct covariate balance by reweighting. |
| Genetic matching | `sp.genmatch` | Automated balance search over covariate weights. |
| Overlap weights | `sp.overlap_weights` | ATE-style overlap-population estimands with stable weights. |
| Diagnostics | `sp.balance_diagnostics`, `sp.love_plot` | Standardised mean differences, variance ratios, and Love plots. |

## Method-level API

### `sp.match(...)`

::: statspai.match

### `sp.psmatch2(...)`

::: statspai.psmatch2

### `sp.ebalance(...)`

::: statspai.ebalance

### `sp.cbps(...)`

::: statspai.cbps

### `sp.genmatch(...)`

::: statspai.genmatch

### `sp.sbw(...)`

::: statspai.sbw

### `sp.overlap_weights(...)`

::: statspai.overlap_weights

### `sp.balance_diagnostics(...)`

::: statspai.balance_diagnostics

### `sp.love_plot(...)`

::: statspai.love_plot
