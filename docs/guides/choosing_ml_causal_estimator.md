# Choosing an ML-based causal estimator

StatsPAI's `causal_question` DSL ships four ML-based estimators for the
selection-on-observables design: `dml`, `tmle`, `metalearner`, and
`causal_forest`. They all target the same parameter (population ATE
under conditional ignorability), but with different trade-offs in
estimand richness, IV support, and inference machinery. This guide is
a decision tree: read the first question, jump to the section it
sends you to, and stop when you have a recommendation.

## 0. TL;DR flowchart

```
Need heterogeneous effects tau(x), or just the population ATE?

  HETEROGENEOUS effects (CATE)
    Nonparametric tree-based -> sp.causal_question(..., design='causal_forest')
    Doubly-robust / R-loss   -> sp.causal_question(..., design='metalearner')

  POPULATION ATE only
    Have an instrument?
      YES -> sp.causal_question(..., design='dml', instruments=[...])  # LATE
      NO  -> Binary outcome AND want Super Learner?
              YES -> sp.causal_question(..., design='tmle')
              NO  -> sp.causal_question(..., design='dml')              # ATE
```

For all four, the dispatcher returns a scalar ATE summary
(point + SE + 95% CI). For `metalearner` and `causal_forest` the
per-unit CATEs live on `result.underlying`.

## 1. The four estimators

### `dml` — Double / Debiased ML [chernozhukov2018double]

Neyman-orthogonal moment with cross-fitted ML nuisance estimators.
Returns a scalar ATE (or LATE with an instrument). Auto-picks the
appropriate sub-model from your declarations:

| Treatment | Instruments | Auto-picked DML model | Estimand |
|-----------|-------------|------------------------|----------|
| Binary    | none        | `irm` (interactive)    | ATE      |
| Continuous| none        | `plr` (partially linear)| ATE     |
| Binary    | one binary  | `iivm` (interactive IV)| LATE     |
| Any       | one+        | `pliv` (partially linear IV) | LATE |

```python
import statspai as sp
q = sp.causal_question(
    treatment='trained', outcome='wage',
    design='dml',
    covariates=['age', 'edu', 'exp', 'tenure', 'industry'],
    data=df,
)
r = q.estimate()  # auto-picks model='irm'
print(r.summary())
```

### `tmle` — Targeted Maximum Likelihood [vanderlaan2006targeted]

Doubly robust + semiparametrically efficient under conditional
ignorability; the targeting step solves the efficient
influence-function score equation exactly. Uses Super Learner
internally for both outcome and propensity nuisance.

```python
q = sp.causal_question(
    treatment='trained', outcome='employed',  # binary outcome
    design='tmle',
    covariates=['age', 'edu', 'exp'],
    data=df,
)
r = q.estimate()
```

Supports `estimand='ATE'` or `'ATT'`. LATE / CATE are coerced to ATE
with a warning.

### `metalearner` — S/T/X/R/DR-Learner [kunzel2019metalearners; nie2021quasi]

Estimates `tau(x) = E[Y(1) - Y(0) | X=x]` via a chosen learner family.
The reported scalar is the population ATE (mean over units of the
estimated CATEs) with the AIPW (doubly robust) influence-function SE
— learner-independent. Per-unit CATEs are accessible on
`result.underlying.model_info['cate']`.

```python
q = sp.causal_question(
    treatment='trained', outcome='wage',
    design='metalearner', estimand='CATE',
    covariates=['age', 'edu', 'exp'],
    data=df,
)
r = q.estimate(learner='dr')              # default
cate = r.underlying.model_info['cate']    # per-unit tau(x_i)
```

### `causal_forest` — honest random forest [athey2019generalized; wager2018estimation]

Honest random-forest estimator of `tau(x)` with sub-sampled trees.
The reported scalar ATE point and SE come from the cross-fit AIPW
influence function [vanderlaan2003unified; chernozhukov2018double] —
B-independent and doubly robust, exactly the approach
`grf::average_treatment_effect` uses in R. Per-unit CATEs and
pointwise GRF intervals are available via `result.underlying.effect(X)`
and `result.underlying.effect_interval(X)`.

```python
q = sp.causal_question(
    treatment='trained', outcome='wage',
    design='causal_forest',
    covariates=['age', 'edu', 'exp'],
    data=df,
)
r = q.estimate(n_estimators=500, random_state=0)
cate = r.underlying.effect(df[['age', 'edu', 'exp']].to_numpy())
```

Binary treatment only — for continuous T, use `design='dml'`.

## 2. Comparison

| Property | `dml` | `tmle` | `metalearner` | `causal_forest` |
|----------|-------|--------|---------------|------------------|
| Population ATE point | ✓ | ✓ | ✓ (=mean of CATEs) | ✓ (via AIPW-IF) |
| Population ATE SE | Neyman-orth IF | EIF (Super Learner) | AIPW-IF | AIPW-IF |
| LATE via IV | ✓ (PLIV / IIVM) | ✗ | ✗ | ✗ |
| CATE function | ✗ | ✗ | ✓ | ✓ |
| Continuous treatment | ✓ (PLR) | ✗ | ✗ | ✗ |
| Binary outcome | ✓ | ✓ (Super Learner) | ✓ | ✓ |
| Doubly robust | ✓ | ✓ | ✓ (DR / R-Learner) | ✓ (via AIPW-IF) |

## 3. Decision tree

1. **Do you need heterogeneous effects τ(x), not just the population ATE?**
   - YES → go to step 2.
   - NO → go to step 3.

2. **Do you want a tree-based nonparametric CATE estimator?**
   - YES (and treatment is binary) → `design='causal_forest'`.
   - NO (or want a specific learner family — S/T/X/R/DR) →
     `design='metalearner'`.

3. **Do you have an instrument that satisfies relevance + exclusion?**
   - YES → `design='dml'`, `instruments=[...]`. The dispatcher picks
     IIVM (single binary Z → LATE) or PLIV (otherwise).
   - NO → go to step 4.

4. **Is your outcome binary AND do you want Super Learner nuisance?**
   - YES → `design='tmle'`.
   - NO → `design='dml'` (PLR for continuous T, IRM for binary T).

## 4. What the planner records

Every plan attached to a `CausalQuestion` records the assumptions you
must defend, plus warnings for any silent estimand coercion (e.g.
`design='dml'` with `estimand='CATE'` → coerced to ATE; for CATE use
`metalearner` or `causal_forest` instead).

```python
q = sp.causal_question(
    treatment='d', outcome='y', design='dml',
    estimand='CATE',          # mismatch with DML's scalar ATE target
    covariates=['x'], data=df,
)
plan = q.identify()
plan.estimand     # 'ATE' (coerced)
plan.warnings     # explains why
```

Same idea for the four reserved kwargs: passing `y=`, `treat=`,
`covariates=`, or `data=` to `q.estimate()` raises `TypeError` early
with a clear message — the dispatcher pulls these from the
`CausalQuestion` fields, never from kwargs.

## 5. References

All citations resolve to entries in `paper.bib`:

- `chernozhukov2018double` — DML, *Econometrics Journal* 21(1).
- `kunzel2019metalearners` — meta-learners, *PNAS* 116(10).
- `nie2021quasi` — R-learner / quasi-oracle, *Biometrika* 108(2).
- `wager2018estimation` — causal forest, *JASA* 113(523).
- `athey2019generalized` — generalised random forests, *Annals of
  Statistics* 47(2).
- `vanderlaan2006targeted` — TMLE, *International Journal of
  Biostatistics* 2(1).
- `vanderlaan2003unified` — efficient/AIPW estimating equations
  (Springer Series in Statistics, 2003).
