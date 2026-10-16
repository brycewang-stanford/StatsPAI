# Standard errors: one `vce=` grammar, and `sp.vcov` after the fit

Every StatsPAI estimator that reports standard errors takes them through one
keyword, `vce=`, written the way Stata writes `vce()`:

```python
import statspai as sp

sp.regress("y ~ x", data=df, vce="robust")            # HC1
sp.regress("y ~ x", data=df, vce="hc3")
sp.regress("y ~ x", data=df, vce="cluster state")
sp.regress("y ~ x", data=df, vce="cluster state year") # two-way
sp.feols("y ~ x | firm", data=df, vce="cluster state")
sp.sun_abraham(df, y="y", g="g", t="t", i="id", vce="cluster state")
sp.rdrobust(df, y="y", x="score", vce="cluster school")
```

`cluster=` still works everywhere; `vce="cluster state"` and
`cluster="state"` are the same call. The older spellings (`robust="hc1"`,
pyfixest's `vcov={"CRV1": "state"}`) are accepted too.

## What each estimator accepts

Estimators implement different variances, and a request an estimator does not
implement **raises**: it is never quietly replaced by another kind. The
schema says which kinds an estimator takes:

```python
sp.function_schema("sun_abraham")["parameters"]["properties"]["vce"]
# {'description': "... Accepted here: 'cluster <column>' ...
#   Omitting it gives the default: panel-unit clustered standard errors.",
#  'x-statspai-vce-kinds': ['cluster'], ...}
```

A few estimators keep a `vce` vocabulary of their own, because the reference
implementation has one: `rdrobust` (`'nn'`, `'hc0'`-`'hc3'`), the imputation
DiD (`'analytic'`, `'bootstrap'`), `mswitch` (`'oim'`, `'opg'`). They also read
`vce="cluster v"`, as Stata does.

The canonical spellings:

| request | meaning |
| --- | --- |
| `"ols"`, `"iid"`, `"nonrobust"` | model-based |
| `"robust"`, `"hc1"` | heteroskedasticity-robust (HC1 on least squares, the ML sandwich on likelihood models) |
| `"hc0"`, `"hc2"`, `"hc3"` | the other HC corrections |
| `"cluster v"`, `"cluster v w"` | one- or two-way cluster-robust (CR1) |
| `"cr2 v"`, `"cr3 v"` | Bell-McCaffrey / jackknife cluster corrections |
| `"hac"` | Newey-West (lags via `hac_lags=`) |
| `"bootstrap"`, `"bootstrap v"` | bootstrap, resampling rows or clusters `v` |
| `"wild"` with `cluster=` | wild cluster bootstrap |

## Changing the standard errors of a fit you already have

`sp.vcov(result, vce)` answers "what would my standard errors be under
another choice?" without rebuilding the call:

```python
fit = sp.regress("y ~ x + z", data=df, weights="w")
by_state = sp.vcov(fit, "cluster state")
```

It re-runs your own call with only `vce=` changed. The new standard errors
are therefore exactly what `sp.regress(..., weights="w", vce="cluster state")`
returns: weights, fixed effects, covariates, seeds and small-sample factors
included. `sp.vcov_compare` puts several requests side by side:

```python
sp.vcov_compare(fit, ["hc1", "hc3", "cluster state", "cluster state year"],
                terms=["x"])
```

| vce | term | estimate | se | ... | se_ratio | note |
| --- | --- | --- | --- | --- | --- | --- |
| (as fitted) | x | 2.75 | 0.068 | | 1.00 | |
| hc1 | x | 2.75 | 0.136 | | 2.02 | |
| ... | | | | | | |

A request the estimator cannot honour fills its row with `NaN` and says why
in `note`.

### Safeguards

* **Replay check.** Before switching `vce`, the recorded call is run once
  unchanged and must reproduce the result's estimates to a relative `1e-10`.
  If you edited the data after fitting, passed a different frame as `data=`,
  or the estimate depends on an unseeded random draw, `sp.vcov` refuses
  rather than reporting standard errors for a different fit.
* **The data.** A result holds its input frame through a weak reference, so
  it never keeps a large frame alive by itself. If the frame is gone, pass
  it again: `sp.vcov(fit, "cluster state", data=df)`.
* **Estimates that move.** For some estimators the cluster variable enters
  more than the variance: rows with a missing cluster value are dropped (as
  Stata does), cross-fitting folds and cluster-Lasso penalty loadings follow
  the clusters. `sp.vcov` then warns, and
  `result.model_info["vcov_refit"]["estimates_changed"]` is `True`.

`sp.regtable(..., vcov=...)` uses the same engine for each column.

## For agents

The MCP tool `vcov_from_result` takes a `result_id` and one request (returns
the new result, optionally as a new handle) or a list (returns the comparison
table). It re-runs the fit on the data handle the fit was made on.
