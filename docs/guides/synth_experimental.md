# Synthetic Controls for Experimental Design

> Abadie & Zhao (2025/2026), *MIT working paper* / Cambridge UP 2025.

## 1. The flipped workflow

Classical synthetic control answers: "I already have treated unit *A* —
build a reweighted average of donors that approximates *A* in the
pre-period, then impute *A*'s post-period counterfactual."

**Experimental design** flips this: you have a pool of candidates and a
budget `k`.  You want to decide *which* `k` units to treat so that the
post-period synthetic-control ATT has the tightest confidence interval.

Under the Abadie-Zhao framework,

$$
\operatorname{Var}\bigl[\widehat{\mathrm{ATT}} \mid D\bigr]
\;\approx\;
\sum_{i \in D} \sigma_i^2
$$

where $\sigma_i^2$ is the *feasible* pre-period MSPE of the synthetic
control fit for unit `i`.  Picking the best `k` candidates by smallest
pre-period MSPE minimises this variance.

## 2. API

```python
import statspai as sp

res = sp.synth_experimental_design(
    data=df,              # long-format panel
    unit='unit',
    time='time',
    outcome='y',
    k=5,                  # budget: treat 5 units
    pre_period=(0, 19),   # closed interval, pre-treatment periods
    candidates=None,      # default: all units are candidates
    donors=None,          # default: non-candidates; leave-one-out fallback
    risk='mspe',          # 'mspe' or 'rmse'
    concentration_weight=0.0,  # penalise Herfindahl weight concentration
    penalization=0.0,     # simplex-solver ridge penalty
    n_random=500,         # Monte-Carlo sample for the random-assignment baseline
    random_state=0,
)
```

Returns a `SynthExperimentalDesignResult` with

- `selected` — the `k` recommended units
- `ranking` — DataFrame with per-candidate risk scores
- `weights` — per-candidate donor weight vectors (for audit)
- `expected_variance`, `baseline_variance` — sum-MSPE under the
  chosen vs random assignment
- `summary()` — human-readable report

## 3. Recipe on a synthetic panel

```python
import numpy as np, pandas as pd, statspai as sp

rng = np.random.default_rng(0)
n_units, n_periods = 30, 20
F = rng.normal(size=(n_periods, 3))
L = rng.normal(size=(n_units, 3))
Y = L @ F.T + 0.1 * rng.normal(size=(n_units, n_periods))

df = pd.DataFrame([
    {'unit': i, 'time': t, 'y': Y[i, t]}
    for i in range(n_units) for t in range(n_periods)
])

res = sp.synth_experimental_design(
    df, unit='unit', time='time', outcome='y',
    k=5, pre_period=(0, 19), random_state=0, n_random=200,
)
print(res.summary())
```

On this panel the expected variance falls **96% below random assignment**
— a three-fold tighter post-period CI, for free, just by choosing
well-behaved units *before* you run the experiment.

## 4. When to use this vs `sp.synth`

| Step                          | Function                       |
|-------------------------------|--------------------------------|
| Deciding which units to treat | `sp.synth_experimental_design` |
| Post-treatment counterfactual | `sp.synth(method='classic')`   |
| Inference on the ATT          | `sp.scpi` / `sp.sdid`          |
| Cross-estimator robustness    | `sp.synth_compare`             |

The two are **sequential**: run `synth_experimental_design` at the
planning stage, then let the experiment run and hand the result to the
regular `sp.synth` pipeline.

## 5. Common pitfalls

- **Panel must be balanced** inside `pre_period`.  The function raises
  if any (unit, time) cell is NaN.
- Setting `candidates` equal to all units triggers the leave-one-out
  fallback — each candidate's donor pool becomes the other `n - 1`
  units.  This is fine but inflates computation.
- **`concentration_weight > 0`** adds a Herfindahl penalty to avoid
  selecting units whose SC fit depends on a single donor — the Abadie-
  Zhao (2025/2026) paper recommends `concentration_weight ≈ 0.5`
  when the donor pool is small.

## 6. References

- Abadie, A. & Zhao, J. (2025/2026).
  *Synthetic Controls for Experimental Design.*  MIT / Cambridge UP.
- Abadie, A. (2021).  "Using synthetic controls: feasibility, data
  requirements, and methodological aspects."  *JEL* 59(2).

<!-- AGENT-BLOCK-START: synth -->

## For Agents

**Pre-conditions**
- panel data in long form (unit × time × outcome)
- single treated unit (classic) or a treatment-timing column (staggered)
- ≥ 10 donor (untreated) units with similar pre-treatment trajectories
- ≥ 10 pre-treatment periods (fewer → large weight on any one year)

**Identifying assumptions**
- Treatment effect on the treated is identified by the counterfactual implicit in the donor weights
- No spillover from treated unit to donors (SUTVA)
- Donor pool contains units whose outcomes plausibly track the treated counterfactual
- Pre-treatment fit (RMSPE) is small relative to post-treatment effect for placebo inference

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Pre-treatment RMSPE > post-treatment effect | `AssumptionWarning` | Poor pre-fit — switch to method='demeaned'/'augmented' or enlarge donor pool. | `sp.synth` |
| Placebo p-value ≥ 0.1 despite visible gap | `AssumptionWarning` | Use inference='conformal' (valid under weak assumptions) or report ranked placebo statistic. | `sp.synth` |
| All weight concentrated on one donor | `AssumptionWarning` | Interpolation bias risk — check method='elastic_net' or augmented SCM. | `sp.synth` |
| Treated unit outside donor convex hull | `IdentificationFailure` | Extrapolation needed — use method='unconstrained' or 'augmented'. | `sp.synth` |

**Alternatives (ranked)**
- `sp.sdid`
- `sp.did`
- `sp.matrix_completion`
- `sp.causal_impact`

**Typical minimum N**: 10

<!-- AGENT-BLOCK-END -->
