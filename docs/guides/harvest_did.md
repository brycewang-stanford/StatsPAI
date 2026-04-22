# Harvesting DID + Event-Study Designs

> Abadie, Angrist, Frandsen & Pischke (NBER WP 34550, 2025).

## 1. What "harvesting" means

A staggered-adoption panel implicitly defines *many* valid
difference-in-differences contrasts: every pair of (cohort `g`, pre-
period `t₁`, post-period `t₂`) where `t₁` is before treatment and `t₂`
is after treatment, matched against a control cohort that is *not yet
treated* in either period, produces an unbiased 2×2 DID estimate under
parallel trends.

Harvesting extracts **all** such valid 2×2 estimates and combines them
into a single inverse-variance-weighted aggregate.  It is the natural
generalisation of Callaway–Sant'Anna (2021) ATT(g, t) and coincides
with their estimator when restricted to a single horizon.

## 2. API

```python
import statspai as sp

res = sp.harvest_did(
    data=df,
    unit='unit',
    time='time',
    outcome='y',
    treat='treated',          # binary indicator; cohort inferred
    # or: cohort='first_treat', never_value=0
    horizons=range(-3, 5),    # event-time windows (-3..4 by default)
    reference=-1,             # pre-treatment reference horizon
    alpha=0.05,
    weighting='precision',    # 'precision' | 'equal' | 'n_treated'
)
print(res.summary())
```

Returns a standard `CausalResult` with `estimate`, `se`, `ci`.  The
`model_info` payload contains:

- `event_study` — DataFrame with columns
  `relative_time, att, se, pvalue, n_comparisons`
- `pretrend_test` — Wald joint test of horizon `< 0`
- `n_comparisons` — total number of harvested 2×2 cells

The `detail` slot exposes the per-comparison table (cohort, horizon,
`t₁`, `t₂`, ATT, SE, donor counts) so you can audit every cell.

## 3. Staggered-adoption recipe

```python
import numpy as np, pandas as pd, statspai as sp

rng = np.random.default_rng(0)
n_units, n_periods = 120, 12
cohort = rng.choice([0, 5, 7, 9], size=n_units, p=[0.4, 0.2, 0.2, 0.2])
rows = []
for i in range(n_units):
    uf = rng.normal()
    g = cohort[i]
    for t in range(n_periods):
        D = 1 if (g > 0 and t >= g) else 0
        y = uf + 0.1 * t + 2.0 * D + rng.normal(0, 0.3)
        rows.append({'unit': i, 'time': t, 'y': y, 'treated': D})
df = pd.DataFrame(rows)

res = sp.harvest_did(
    df, unit='unit', time='time', outcome='y', treat='treated',
    horizons=range(-3, 5),
)
print(res.summary())
```

## 4. How it differs from its siblings

| Estimator                       | What it targets                                   | Who to pick when                           |
|---------------------------------|---------------------------------------------------|--------------------------------------------|
| `sp.callaway_santanna`          | ATT(g, t) building blocks + aggregation schemes   | You want a specific g×t heatmap            |
| `sp.sun_abraham`                | Interaction-weighted event-study coefficients     | You want a clean event-study plot          |
| `sp.did_imputation` (BJS 2024)  | Imputation-based estimator with uniform inference | You need finite-sample guarantees          |
| **`sp.harvest_did`**            | Every valid 2×2, inverse-variance averaged        | You want maximum efficiency across cells   |

All four converge to the same true ATT under homogeneous effects +
parallel trends; they differ mainly in their weighting of heterogeneity.

## 5. Clean-control rule

The key safety rule: a control cohort is valid for comparison
`(g, t₁, t₂)` only if it is **not yet treated at**
`max(t₁, t₂)`.  This guards both post-period horizons (`t₂ > t₁`) and
pre-period placebos (`t₂ < t₁`).

If you see the pretrend Wald test report `p < 0.05` on a DGP you
believe has parallel trends, double-check that the `reference` and
`horizons` combination doesn't straddle a cohort's treatment time in
an unexpected way.

## 6. Inference caveats

`harvest_did` treats **units within each cohort** as independent for
the per-comparison SE (unit-level cluster-robust), but it **ignores
cross-horizon covariance** when aggregating.  For strict finite-sample
inference, wrap the call in a unit-level bootstrap:

```python
from statspai.inference import bootstrap
stat = lambda d: sp.harvest_did(d, unit='unit', time='time',
                                 outcome='y', treat='treated').estimate
boot = bootstrap(df, stat, n_boot=500, cluster='unit', random_state=0)
```

## 7. References

- Abadie, Angrist, Frandsen & Pischke (NBER WP 34550, 2025).
  "Harvesting Differences-in-Differences and Event-Study Evidence."
- Callaway & Sant'Anna (2021).  "Difference-in-Differences with
  multiple time periods."  *JoE* 225.
- Baker, Callaway, Cunningham, Goodman-Bacon & Sant'Anna (2025).
  "Difference-in-Differences Designs: A Practitioner's Guide."
  arXiv:2503.13323.

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
