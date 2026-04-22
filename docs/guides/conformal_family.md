# Conformal causal inference — the full family

> **"Your CATE point estimate is a number. Your CATE prediction
> interval is a promise."** Conformal inference is the only framework
> that gives you **finite-sample** coverage guarantees on the individual
> treatment effect interval, with **no** distributional assumptions on
> the outcome. StatsPAI ships the 2021 Lei-Candès foundation plus the
> 2024-2026 frontier: covariate shift, density-based HDR, multi-stage,
> debiased ML, fairness, continuous treatment, and interference.

If the phrase "95% prediction interval" in your CATE output comes from
a bootstrap or a Normal approximation, you don't have a coverage
guarantee — you have a **conjecture about coverage**. Conformal
inference replaces the conjecture with a theorem: if calibration and
test data are exchangeable, empirical coverage hits the nominal level
in finite samples. The ten estimators below each loosen one
assumption of Lei-Candès 2021 to handle a real-world wrinkle.

Every function is at top level: `sp.conformal_cate`,
`sp.weighted_conformal_prediction`, `sp.conformal_counterfactual`,
`sp.conformal_ite_interval`, `sp.conformal_density_ite`,
`sp.conformal_ite_multidp`, `sp.conformal_debiased_ml`,
`sp.conformal_fair_ite`, `sp.conformal_continuous`,
`sp.conformal_interference`.

---

## The two things conformal inference promises

For a test point `x` and a target coverage `1 - α`:

1. **Marginal coverage** `P(τ(x) ∈ [L(x), U(x)]) ≥ 1 − α` — averaged
   over the random test point and calibration set, the interval
   contains the truth at least `1 − α` of the time.
2. **Finite-sample validity** — the guarantee holds for *any* sample
   size `n`, not asymptotically.

Conditional coverage (`P(τ(x) ∈ [L(x), U(x)] | X=x) ≥ 1 − α`) is
**not** promised and is impossible without further assumptions. When
someone says "CQR gives conditional coverage", they mean *better
conditional coverage than vanilla split conformal*, not literal
conditional validity.

---

## 1. Baseline — `sp.conformal_cate` (Lei-Candès 2021)

Split-conformal prediction intervals for the CATE, using separate
outcome models `μ_1(x)`, `μ_0(x)` and the combined calibration
quantile `q_1 + q_0`. Default outcome model is gradient boosting.

```python
r = sp.conformal_cate(
    data=df, y="y", treat="d",
    covariates=["x1", "x2", "x3"],
    alpha=0.05,
    calib_fraction=0.25,
    random_state=42,
)
cate        = r.model_info["cate"]         # point estimate per row
cate_lower  = r.model_info["cate_lower"]   # lower bound of PI
cate_upper  = r.model_info["cate_upper"]   # upper bound
width       = r.model_info["interval_width"]  # mean width
```

**When to use it**: observational or experimental data, binary
treatment, random calibration-test split is plausibly exchangeable,
you want one number of coverage per observation. Default.

Reference: Lei & Candès (2021), *JRSS-B* 83(5).

---

## 2. Covariate-shift-corrected counterfactuals — `sp.conformal_counterfactual`

The refinement that makes per-arm conformal work under **treatment
selection on observables**. The treated and control calibration sets
have different covariate distributions; `conformal_counterfactual`
re-weights each calibration set to the marginal-X distribution using
the propensity score as the Tibshirani-Barber-Candès-Ramdas (2019)
weight:

```python
r = sp.conformal_counterfactual(
    data=df, y="y", treat="d", covariates=["x1", "x2"],
    X_test=df[["x1", "x2"]].values,
    alpha=0.1,
    calib_frac=0.3,
)
r.summary()       # per-arm coverage target + mean band widths
r.to_frame()      # columns: Y1_lower, Y1_upper, Y0_lower, Y0_upper
```

**When to use it**: observational data with non-random treatment
assignment, and you want prediction intervals for the **counterfactual
potential outcomes** `Y(1) | X` and `Y(0) | X` (not just their
difference).

References: Lei & Candès (2021) Theorem 1; Tibshirani et al. (2019),
*NeurIPS*.

---

## 3. ITE intervals (nested bound) — `sp.conformal_ite_interval`

The Lei-Candès Eq. 3.4 nested bound for the individual treatment
effect `τ(x) = Y(1) − Y(0)`. Built from two per-arm conformal bands,
each at level `α/2`, so the combined band has finite-sample coverage
`≥ 1 − α` by a simple union bound:

```python
r = sp.conformal_ite_interval(
    data=df, y="y", treat="d", covariates=["x1", "x2"],
    alpha=0.1, calib_frac=0.3,
)
r.summary()
r.to_frame()      # columns: tau, tau_lower, tau_upper
```

**When to use it**: you want a **conservative** but finite-sample
valid ITE interval. "Conservative" here means "wider than necessary";
that's the price of not needing distributional assumptions. If the
interval is too wide to be useful, try `conformal_density_ite` or
`conformal_debiased_ml` next.

Reference: Lei & Candès (2021), Eq. 3.4.

---

## 4. Generic covariate-shift conformal — `sp.weighted_conformal_prediction`

Low-level primitive: split-conformal with arbitrary per-calibration
weights. Useful when you know the density ratio
`w(x) = f_test(x) / f_train(x)` from some external source:

```python
lower, upper, point = sp.weighted_conformal_prediction(
    X_train, y_train,
    X_calib, y_calib, X_test,
    weights_calib=w,             # likelihood-ratio weights
    model=my_sklearn_model,
    alpha=0.1,
)
```

**When to use it**: you're layering conformal on top of a custom
pipeline and need to inject weights. Most users don't need this
directly — `conformal_counterfactual` calls it internally.

Reference: Tibshirani, Barber, Candès, Ramdas (2019), *NeurIPS*.

---

## 5. Conditional-density ITE — `sp.conformal_density_ite` (2025)

When the counterfactual distribution is **heavy-tailed or multimodal**,
the mean-based conformal interval can be misleadingly wide and
off-center. `conformal_density_ite` builds a highest-density region
(HDR) from a KDE of the ITE-residual distribution, giving a sharper
(and potentially non-symmetric) interval:

```python
r = sp.conformal_density_ite(
    data=df, y="y", treat="d",
    covariates=["x1", "x2"],
    alpha=0.1,
    bandwidth=None,        # Silverman's rule; override with a float
    seed=0,
)
print(r.intervals)         # (n_test, 2)
print(r.point_estimate)    # ITE mean per row
```

**When to use it**: the counterfactual distribution is visibly
non-Gaussian. A classic tell: `r.intervals[:, 1] - r.intervals[:, 0]`
is noticeably narrower than what `conformal_ite_interval` produces,
while coverage holds on a held-out fold.

Citation: arXiv:2501.14933 (2025).

---

## 6. Multi-stage ITE (DTR) — `sp.conformal_ite_multidp` (2025)

Dynamic treatment regimes and multi-stage decisions: at stage `k`, the
treatment is chosen based on history through stage `k−1`, and the
final ITE accumulates across stages. Joint coverage uses a Bonferroni
correction `α/K` per stage:

```python
r = sp.conformal_ite_multidp(
    data=df,
    y_per_stage=["y1", "y2", "y3"],
    treat_per_stage=["d1", "d2", "d3"],
    history_per_stage=[
        ["x_base"],                       # stage 1 features
        ["x_base", "y1", "d1"],           # stage 2 features
        ["x_base", "y1", "d1", "y2", "d2"],
    ],
    alpha=0.1,
)
print(r.intervals_per_stage)     # K arrays of (n_test, 2)
print(r.cumulative_interval)     # sum across stages
```

**When to use it**: DTR, sequential RCTs (SMART), or any clinical /
business process where the `k`-th decision depends on the history of
previous outcomes and decisions. Bonferroni is conservative — if `K`
is large and you need tighter bands, consider per-stage recalibration
with a holdout test set.

Citation: arXiv:2512.08828 (2025).

---

## 7. Debiased ML conformal — `sp.conformal_debiased_ml` (2026)

Uses a **cross-fit AIPW** score as the non-conformity base rather than
raw outcome residuals. This gives the interval **double robustness**
— the coverage guarantee is preserved even when either the outcome
model or the propensity model is moderately misspecified, as long as
the product error rate is `o_p(n^{-1/2})`:

```python
r = sp.conformal_debiased_ml(
    data=df, y="y", treat="d",
    covariates=["x1", "x2"],
    alpha=0.1,
    n_folds=5,
    seed=0,
)
print(r.intervals)
print(r.point_estimate)
```

**When to use it**: observational data, you suspect misspecification
in one of your nuisance models, and you want the extra robustness
from AIPW residualization. The hidden cost is `n_folds` extra model
fits — fine for `n ≤ 100k`.

Citation: arXiv:2604.03772 (2026).

---

## 8. Counterfactual-fair conformal — `sp.conformal_fair_ite` (2025)

Adds a fairness constraint: **coverage must be the same across
protected subgroups** (race, sex, SES). Computes calibration quantiles
**per group** and reports both per-group and pooled intervals:

```python
r = sp.conformal_fair_ite(
    data=df, y="y", treat="d",
    covariates=["x1", "x2"],    # protected is auto-excluded from regression
    protected="race",
    alpha=0.1,
)
print(r.group_widths)               # dict: group → interval width
print(r.group_coverage_targets)     # dict: group → nominal coverage
```

**When to use it**: regulated applications (healthcare, credit,
hiring, criminal justice) where differential coverage across
protected groups is a fairness violation on top of a statistical one.
When a group has fewer than 5 calibration points, the estimator
**falls back to the conservative maximum quantile across well-covered
groups** — the per-group coverage guarantee is preserved, at the cost
of a wider interval for the underrepresented group. If *all* groups
are small, a `warnings.warn` is emitted and the per-group guarantee
does not hold.

Citation: arXiv:2510.08724 (2025).

---

## 9. Continuous-treatment dose response — `sp.conformal_continuous` (2024)

For continuous treatment, the target is the **dose-response curve**
`E[Y | T=t, X]` across a dose grid. Produces split-conformal bands at
each `t` on the grid:

```python
r = sp.conformal_continuous(
    data=train_df, y="y", treatment="dose",
    covariates=["x1", "x2"],
    test_data=test_df,
    dose_grid=np.linspace(0, 1, 21),
    alpha=0.1,
)
print(r.predictions)       # point + band at each test row's observed dose
print(r.dose_curves)       # full curve per test row across the grid
```

**When to use it**: dose-response analysis, e.g. drug dosing,
fertilizer application, time-in-app. Returns a point estimate + band
at each `(test_row, dose)` pair in the grid.

Citation: Kim, Jeong, Barber, Lee (arXiv:2407.03094, 2024).

---

## 10. Cluster-exchangeable conformal — `sp.conformal_interference` (2021/2025)

When units within a cluster **interfere** (spillover, networks),
unit-level exchangeability fails but **cluster-level** exchangeability
can still hold. Computes a cluster-averaged residual score and does
split-conformal at the cluster level:

```python
r = sp.conformal_interference(
    data=df, y="y", treatment="d", cluster="village",
    covariates=["x1", "x2"],
    test_clusters=["v101", "v102", "v103"],
    alpha=0.1,
)
print(r.predictions)        # cluster, prediction, lo, hi
print(r.cluster_scores)     # per-calibration-cluster mean |residual|
```

**When to use it**: SUTVA is violated within clusters but clusters
themselves are independent (villages, schools, firms). Requires `≥ 4`
non-test clusters — anything less and the calibration quantile is
meaningless.

References: Lei, Sesia & Candès (2021); systematic review at
arXiv:2509.21660 (2025) uses this as the recommended default under
interference.

---

## Decision guide

```
Point + interval for each unit's CATE, binary treatment
  → sp.conformal_cate

I want Y(1) | X and Y(0) | X bands separately
  → sp.conformal_counterfactual

I want τ(x) = Y(1) - Y(0) ITE interval
  → sp.conformal_ite_interval

---

My counterfactual distribution is heavy-tailed / bimodal
  → sp.conformal_density_ite

My decision problem has K sequential stages
  → sp.conformal_ite_multidp

My nuisance models might be misspecified
  → sp.conformal_debiased_ml

I need equal coverage across protected subgroups
  → sp.conformal_fair_ite

---

Treatment is continuous — I want a dose-response curve
  → sp.conformal_continuous

Units interfere within clusters (SUTVA violated inside clusters)
  → sp.conformal_interference

---

I have a custom pipeline and need weighted split-conformal
  → sp.weighted_conformal_prediction  (low-level primitive)
```

---

## Diagnostics every conformal analysis should report

1. **Empirical coverage on a held-out fold.** The theoretical
   guarantee is marginal, not conditional; always check empirically
   that coverage matches the nominal level on a fresh test split, not
   just on the calibration fold.
2. **Mean and median interval width.** The guarantee is on coverage;
   width is the cost you paid. If median width ≈ the full range of `Y`
   you are reporting `"anything is possible"`, which is correct but
   useless.
3. **Per-group coverage** (even without `conformal_fair_ite`). Even
   with marginal coverage nominally valid, subgroup coverage can drift
   badly. Report coverage conditional on key covariates.
4. **Calibration set size.** Marginal coverage requires
   `n_calib ≥ 1/α ≈ 10` for `α = 0.1`. If your calibration set is
   smaller (common when `conformal_fair_ite` strata thin out), the
   quantile `ceil((1-α)(n+1)) / n` clips to 1 and you get `[-∞, +∞]`
   intervals. Check `r.model_info['n_calib']` or the equivalent field.
5. **Propensity overlap.** For `conformal_counterfactual` and
   `conformal_ite_interval`, verify `ps ∈ [0.02, 0.98]`. The TBCR
   weight `1/g(X)` explodes near the boundaries, making the band
   arbitrarily wide.

---

## How to read disagreement

If `conformal_cate` and `conformal_ite_interval` disagree on coverage
or width, the usual culprit is that `conformal_cate` uses `q_1 + q_0`
(sum of per-arm quantiles) while `ite_interval` uses `Δ_1(x) + Δ_0(x)`
(sum of per-arm half-widths after splitting `α/2` between arms). The
latter is *always* at least as conservative as the former. If you
want the tightest finite-sample interval and have a well-specified
outcome model, use `conformal_cate`. If you want the most
defensible theoretical guarantee, use `conformal_ite_interval`.

If `conformal_cate` and `conformal_debiased_ml` give very different
widths: your outcome model is probably misspecified. The AIPW score
has smaller asymptotic variance, so narrower `debiased_ml` bands
while coverage holds are a *good* sign.

---

*Current for StatsPAI ≥ 1.5.0. All functions are registered;
`sp.describe_function("conformal_ite_interval")` returns the hand-
written schema. For dispatcher-style access see `sp.conformal(kind=...)`.*

<!-- AGENT-BLOCK-START: conformal -->

## For Agents

**Pre-conditions**
- calibration sample disjoint from training sample (auto-split or user-supplied)
- exchangeability between calibration and test distributions (weighted variants for covariate shift)
- for CATE / ITE variants: unconfoundedness + overlap on covariates
- ≥ 500 calibration observations for reliable finite-sample coverage at alpha ≤ 0.1

**Identifying assumptions**
- Exchangeability of calibration and test points (base case)
- For kind='weighted': known or estimable density ratio between calibration and test
- For kind='cate' / 'ite': selection-on-observables with correct propensity / outcome model
- For kind='interference': cluster-exchangeable exchangeability

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Calibration and test distributions differ (covariate shift) | `statspai.AssumptionViolation` | Use kind='weighted' with estimated density ratios. |  |
| Calibration set too small — intervals wide | `statspai.DataInsufficient` | Increase calibration sample or raise alpha; coverage gets loose below ~100. |  |
| Miscalibrated nuisance (propensity / outcome) for CATE/ITE | `statspai.AssumptionWarning` | Use kind='debiased' which orthogonalises via DML-style nuisance handling. |  |

**Alternatives (ranked)**
- `sp.conformal_cate`
- `sp.weighted_conformal_prediction`
- `sp.conformal_counterfactual`

**Typical minimum N**: 500

<!-- AGENT-BLOCK-END -->
