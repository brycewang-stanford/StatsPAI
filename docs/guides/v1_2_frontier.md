# v1.2 frontier estimators — doc-alignment sprint

StatsPAI v1.2 closes the remaining gaps between the *Causal-Inference
Method Family 万字剖析 v3* (2026-04-20) reference document and the public
API. This guide walks through every new estimator added in v1.2, when to
reach for it, and how it relates to the v1.0/v1.1 building blocks.

If you just want the one-liner, skip to [§ When to use which](#when-to-use-which).

---

## Staggered DID

### `sp.gardner_did` / `sp.did_2stage` — Gardner (2021) two-stage DID

The Stata `did2s` analogue.  Two-step regression that propagates Stage-1
uncertainty into Stage-2 inference:

```python
import statspai as sp
r = sp.gardner_did(
    df, y="wage", group="county", time="year", first_treat="first_treat",
    event_study=True, horizon=list(range(-5, 6)),
)
r.summary()
# r.model_info["event_study"]["coef"] is the event-study dict
```

**Why this one when you already have `sp.did_imputation`?** Gardner and BJS
target the same ATT, but Gardner's regression framing makes event-study
disaggregation, covariate interactions, and unbalanced panels trivially
extensible. On synthetic panels they agree to ~2%. Pick Gardner when you
want the event study or want to add interactions; pick BJS when you want
the efficiency proof and no customisation.

**Citation**: Gardner, J. (2021). arXiv:2207.05943. Butts & Gardner
(2022), *R Journal* 14(3).

### `sp.harvest_did` — MIT/NBER WP 34550 (2025) harvesting framework

Collects *every* valid 2×2 DID comparison implied by the staggered
panel and combines them via inverse-precision weighting. Treat as the
"one-call" stripped-down Callaway-Sant'Anna when you want a single
overall number with the right SE.

```python
r = sp.harvest_did(
    df, outcome="y", unit="id", time="t", cohort="first_treat",
)
# r.estimate, r.se, r.ci as usual
```

---

## Double ML

### `sp.dml_model_averaging` — Ahrens et al. (2025, *JAE*)

Standard `sp.dml` picks one nuisance learner and hopes it's correct.
Model averaging fits DML under a *set* of candidate learners
(`Lasso`, `Ridge`, `RandomForest`, `GBM` by default) and reports a
risk-weighted average of their θ estimates with a covariance-adjusted
SE:

```python
r = sp.dml_model_averaging(
    df, y="y", treat="d",
    covariates=[f"x{j}" for j in range(p)],
    weight_rule="inverse_risk",    # or "equal" / "single_best"
)
print(r.model_info["weights"])
print(r.model_info["theta_k"])
```

**Three weight rules:**

- `"inverse_risk"` (default) — w_k ∝ 1 / MSE_k
- `"equal"` — 1/K
- `"single_best"` — all mass on the lowest-risk candidate

Reach for this when your causal estimate swings a lot as you swap the
nuisance learner and you'd rather have a principled ensemble than a
coin flip. Citation: Ahrens, Hansen, Schaffer & Wiemann (2025),
DOI `10.1002/jae.3103`.

---

## Non-parametric IV

### `sp.kernel_iv` — Lob et al. (2025)

Kernel-smoothed IV regression with a **uniform** wild-bootstrap
confidence band over the structural function `h*(d) = E[Y | do(D=d)]`:

```python
r = sp.kernel_iv(df, y="y", treat="d", instrument="z", n_boot=200)
r.summary()                          # grid, point estimates, UCB
```

Use when the treatment effect is plausibly non-linear in `d` and a
point estimate is not enough — the uniform CI lets you reject
"effect is zero everywhere" without pointwise hacking. Citation:
arXiv:2511.21603.

### `sp.continuous_iv_late` — Xie et al. (2025)

LATE on the **maximal complier class** for continuous instruments,
via quantile-bin Wald estimators. Closer to the spirit of Angrist-Imbens
LATE than the binary-IV special case:

```python
r = sp.continuous_iv_late(
    df, y="y", treat="d", instrument="z", n_quantiles=5,
)
```

Citation: arXiv:2504.03063.

---

## TMLE

### `sp.hal_tmle` — Qian & van der Laan (2025)

TMLE with **Highly Adaptive Lasso** nuisance learners. HAL is a
non-parametric sieve estimator that approximates càdlàg functions of
bounded variation, giving more stable TMLE finite-sample coverage than
generic random-forest nuisance:

```python
r = sp.hal_tmle(
    df, y="y", treat="d", covariates=["x1","x2","x3","x4"],
    variant="delta",            # or "projection" for tangent-space shrinkage
    max_anchors_per_col=40,
)
```

On `n = 400` synthetic data with non-smooth heterogeneity, HAL-TMLE
recovers the ATE within ~3% where generic TMLE can drift 10-15% on the
same seed. Citation: arXiv:2506.17214.

---

## Synthetic control

### `sp.synth_survival` — Agarwal & Shah (2025)

Synthetic Survival Control: donor convex combination on the
complementary log-log (`cloglog`) scale matches the treated arm's
pre-treatment Kaplan-Meier curve, then projects forward and reports the
survival gap with a placebo-permutation uniform band.

```python
# df : long panel with one row per (unit, time) and a precomputed KM survival
r = sp.synth_survival(
    df, unit="arm", time="month",
    survival="km_est",       # Kaplan-Meier S_i(t)
    treated="tr",            # bool column or explicit unit name
    treat_time=6,
)
r.summary()                     # top-5 donor weights, post-gap, pre-RMSE
```

Citation: arXiv:2511.14133.

---

## RDD aliases (human-friendly names for existing methods)

The v3 document uses "geographic RD", "multi-cutoff RD", "boundary RD"
throughout — but the R/Stata conventions are `rdms`, `rdmc`, `rd2d`.
We now ship both:

| v3 document term     | R/Stata name   | New alias           |
|----------------------|----------------|---------------------|
| Multi-cutoff RD      | `rdmc`         | `sp.multi_cutoff_rd` |
| Geographic RD        | `rdms`         | `sp.geographic_rd`  |
| Boundary RD (2D)     | `rd2d`         | `sp.boundary_rd`    |
| Multi-score RD       | `rd_multi_score` | `sp.multi_score_rd` |

---

## Also new in v1.2 (maintainer's v3 work)

These arrived in the same release and are already wired into `sp.*`:

- `sp.shift_share_political` — Park & Xu (arXiv:2603.00135, 2026) Bartik
  IV specialised for political-science panel data with Rotemberg top-K
  + share-balance diagnostics.
- `sp.bcf_ordinal` — BCF for ordered (multi-level) treatments like
  dose. Extends Hahn-Murray-Carvalho (2020) to `T ∈ {0, 1, ..., K}`.
- `sp.bcf_factor_exposure` — BCF with factor-based exposure mapping for
  high-dimensional exposure vectors (diet, pollutants, polygenic).
- `sp.causal_mas` — Multi-agent LLM framework for causal discovery;
  runs proposer/critic/domain-expert loops over a variable set.
- `sp.evidence_without_injustice` (at `statspai.fairness.evidence_test`) —
  Counterfactual-fairness test for legal/admissibility contexts.
- `sp.causal_kalman` (at `statspai.assimilation.kalman`) — Bayesian
  assimilation of a stream of causal estimates with explicit
  process-variance modelling.

---

## When to use which

```
Need a staggered-DID ATT?
  +-- Want one overall number, small panel          -> sp.harvest_did
  +-- Want event study + covariate interactions     -> sp.gardner_did
  +-- Standard 4-design comparison + ATT(g,t)       -> sp.callaway_santanna
  +-- Robust to 2WFE decomposition problems         -> sp.did_imputation

Need DML but unsure which ML model?
  +-- Pick 4 models, let the data decide            -> sp.dml_model_averaging

Continuous instrument?
  +-- Want structural h*(d) with uniform CIs        -> sp.kernel_iv
  +-- Want LATE on maximal compliers                -> sp.continuous_iv_late

Complex nuisance + want semiparametric efficient?
  +-- Generic                                       -> sp.tmle
  +-- Non-smooth heterogeneity, small n             -> sp.hal_tmle

Survival outcome under one-treated synthetic control?
  +-- Kaplan-Meier donor matching                   -> sp.synth_survival

Geographic / multi-cutoff RD?
  +-- 1D running var, multiple cutoffs              -> sp.multi_cutoff_rd
  +-- Multi-score (eligibility by several rules)    -> sp.multi_score_rd
  +-- 2D running var (lat/long)                     -> sp.boundary_rd
```

Every method above is wired into `sp.list_functions()` /
`sp.describe_function()` / `sp.function_schema()` so LLM agents can
discover and call it without reading source.
