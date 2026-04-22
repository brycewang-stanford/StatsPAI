# `cs_report()` — the one-call report card

`sp.cs_report()` runs the full Callaway–Sant'Anna workflow under a
single random seed and bundles the outputs into a `CSReport`
dataclass that pretty-prints, plots, and exports.

## Minimal example

```python
import statspai as sp

rpt = sp.cs_report(
    df,
    y='y', g='g', t='t', i='id',
    n_boot=500,
    random_state=0,
    verbose=True,              # prints the report to stdout
)
```

## Structured fields

```python
rpt.overall      # dict: overall ATT, SE, CI, p
rpt.simple       # DataFrame: simple aggregation
rpt.dynamic      # DataFrame: event study with uniform bands
rpt.group        # DataFrame: per-cohort θ(g)
rpt.calendar     # DataFrame: per-calendar-time θ(t)
rpt.pretrend     # dict: χ² pre-trend Wald test
rpt.breakdown    # DataFrame: R-R breakdown M* per post event time
rpt.meta         # dict: run metadata (n_units, estimator, …)
```

## Export formats

```python
rpt.to_text()         # fixed-width ASCII
rpt.to_markdown()     # GitHub-flavoured Markdown (floatfmt configurable)
rpt.to_latex()        # booktabs LaTeX fragment (no jinja2 needed)
rpt.to_excel('out.xlsx')  # six-sheet workbook
rpt.plot()            # 2×2 summary figure via matplotlib
```

### One-call bundle

Pass `save_to='prefix'` to emit every format in one go:

```python
sp.cs_report(
    df, y='y', g='g', t='t', i='id',
    n_boot=500, random_state=0,
    save_to='~/studies/cs_v1',
)
# writes:
# ~/studies/cs_v1.txt   .md   .tex   .xlsx   .png
```

Missing parent directories are created on the fly; optional
dependencies (`openpyxl`, `matplotlib`) are skipped silently.

## From a pre-fitted result

Skip re-running estimation if you already have a
`callaway_santanna()` result:

```python
cs = sp.callaway_santanna(df, ...)
rpt = sp.cs_report(cs, n_boot=500, random_state=0)
```

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
