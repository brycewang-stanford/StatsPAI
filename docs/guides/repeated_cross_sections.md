# Repeated cross-sections (`panel=False`)

Classic Callaway–Sant'Anna requires a balanced panel: every unit
observed in every period.  Many causal-inference datasets are
*repeated cross-sections* (RCS) — pooled surveys (CPS, ACS), rolling
polls, independent samples per period — where no within-unit first
difference is available.  StatsPAI's RCS path solves this with the
unconditional 2×2 cell-mean DID.

## Estimator

For each (g, t, base) triple with never-treated control c = 0:

$$
\widehat{\text{ATT}}(g, t)
  = (\bar{Y}_{g, t} - \bar{Y}_{g, b})
  - (\bar{Y}_{c, t} - \bar{Y}_{c, b})
$$

and observation-level influence functions

$$
\psi_i
  = \frac{\mathbf{1}\{G_i=g,\,T_i=t\}(Y_i - \bar{Y}_{g,t})}{p_{g,t}}
  - \frac{\mathbf{1}\{G_i=g,\,T_i=b\}(Y_i - \bar{Y}_{g,b})}{p_{g,b}}
  - \frac{\mathbf{1}\{G_i=c,\,T_i=t\}(Y_i - \bar{Y}_{c,t})}{p_{c,t}}
  + \frac{\mathbf{1}\{G_i=c,\,T_i=b\}(Y_i - \bar{Y}_{c,b})}{p_{c,b}}
$$

where $p_{\cdot,\cdot}$ is the empirical cell share.  The influence
functions are stacked into the same `inf_matrix` contract used by the
panel estimator, so `aggte`, `cs_report`, `ggdid`, and `honest_did`
work downstream without modification.

## Usage

```python
sp.callaway_santanna(
    df, y='y', g='first_treat', t='year', i='obs_id',
    estimator='reg',          # required for panel=False
    panel=False,
)
```

## Covariate adjustment

Add `x=[...]` for regression-adjusted RCS.  Y is residualised on X
using OLS fit on the never-treated pool (with period fixed effects);
the cell-mean DID then runs on the residualised outcome:

```python
sp.callaway_santanna(
    survey_df, y='wage', g='g', t='year', i='obs',
    estimator='reg', panel=False,
    x=['age', 'education', 'female'],
)
```

Influence functions treat β̂ as known (plug-in IF), which is
asymptotically valid at √n.  A fully-coupled Sant'Anna–Zhao (2020)
IF augmentation is planned.

## Scope of the current implementation

- `estimator` must be `'reg'`
- `control_group` must be `'nevertreated'`
- IPW / DR for RCS: planned for a future release

All other paths raise `NotImplementedError` with an actionable message.

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

**Alternatives (ranked)**
- `sp.callaway_santanna`
- `sp.sun_abraham`
- `sp.did_imputation`
- `sp.sdid`
- `sp.synth`

**Typical minimum N**: 50

<!-- AGENT-BLOCK-END -->
