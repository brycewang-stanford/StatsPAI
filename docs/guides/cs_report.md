# `cs_report()` — the one-call report card

`sp.cs_report()` runs the eight-step **forward-engineering** workflow
that Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna (2026,
*JEL* 64(2), 498–557, §6) close their practitioner's guide with, under a
single random seed, and bundles the outputs into a `CSReport` dataclass
that pretty-prints, plots, and exports.

The guide's point is that estimation is step 5 of 8. A report that runs
only the estimator has skipped the parts that make the estimate
credible.

| Step | What the guide asks for | What `cs_report` does |
|------|-------------------------|-----------------------|
| 1 | Define the target parameter | ATT(g,t) blocks → simple / dynamic / group / calendar, ω-weighted if `weights=` |
| 2 | State the assumption **and generate evidence** | names the PT assumption; runs `sp.did_balance` (levels *and* ΔX), the Roth–Sant'Anna functional-form test, and an overlap check |
| 3 | Determine the estimation method | triangulates RA / IPW / DR side by side (the guide's Table 7) |
| 4 | Discuss sources of uncertainty | records the inference frame and bootstrap configuration |
| 5 | Estimate | the ATT and its aggregations |
| 6 | Sensitivity analysis | Rambachan–Roth breakdown M\* per post-treatment event time |
| 7 | Heterogeneity analysis | flagged as yours: the relevant partition is a policy question |
| 8 | Keep learning | points at `sp.recommend` / `sp.detect_design` if PT looks implausible |

```python
rpt.forward_engineering_checklist()   # the table above, as a DataFrame
```

## Minimal example

```python
import statspai as sp

rpt = sp.cs_report(
    df,
    y='y', g='g', t='t', i='id',
    x=['poverty'],             # enables balance + RA/IPW/DR triangulation
    weights='pop',             # omega: changes the estimand, not just precision
    n_boot=500,
    random_state=0,
    verbose=True,              # prints the report to stdout
)
```

Turn individual step-2/step-3 evidence off with `balance=False`,
`triangulate=False`, `functional_form=False` when you only want the
estimate. Anything that fails is recorded in `rpt.degradations` and
warned about — never silently skipped.

## Structured fields

```python
rpt.overall               # dict: overall ATT, SE, CI, p
rpt.balance               # DiDBalanceResult (levels + changes panels)
rpt.estimator_comparison  # DataFrame: RA / IPW / DR side by side
rpt.functional_form       # dict: Roth-Sant'Anna p-value
rpt.overlap               # dict: propensity trimming diagnostics
rpt.degradations          # list: any step that ran but did not complete
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
