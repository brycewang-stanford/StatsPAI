# Mendelian Randomization — the full family

> **Your instrument comes from the germline.** Mendelian Randomization
> (MR) uses genetic variants — assumed random at conception — as
> instrumental variables to identify the causal effect of a modifiable
> exposure on a downstream outcome. StatsPAI ships a complete
> summary-statistic MR toolkit: four point estimators, six diagnostic
> tests, three multi-exposure extensions, and two plot helpers —
> byte-for-byte aligned with R's `MendelianRandomization` and
> `TwoSampleMR` packages.

MR has three identifying assumptions, traditionally called **IV1 / IV2 /
IV3**:

1. **IV1 (Relevance)**: the SNP is associated with the exposure.
2. **IV2 (Independence)**: the SNP is independent of confounders of
   exposure–outcome.
3. **IV3 (Exclusion)**: the SNP affects the outcome **only** through
   the exposure (no horizontal pleiotropy).

Failure of IV3 is the canonical threat. The suite below is organized
around "which IV3 violation are you worried about" and gives you the
right estimator + diagnostic combination.

Every function is at top level: `sp.mendelian_randomization`,
`sp.mr_ivw`, `sp.mr_egger`, `sp.mr_median`, `sp.mr_mode`,
`sp.mr_heterogeneity`, `sp.mr_pleiotropy_egger`, `sp.mr_leave_one_out`,
`sp.mr_steiger`, `sp.mr_presso`, `sp.mr_radial`, `sp.mr_f_statistic`,
`sp.mr_multivariable`, `sp.mr_mediation`, `sp.mr_bma`,
`sp.mr_funnel_plot`, `sp.mr_scatter_plot`.

---

## 1. The one-liner — `sp.mendelian_randomization`

The all-in-one dispatcher; runs IVW + MR-Egger + weighted median by
default and returns a unified `MRResult`:

```python
r = sp.mendelian_randomization(
    data=snp_df,
    beta_exposure="beta_x", se_exposure="se_x",
    beta_outcome="beta_y", se_outcome="se_y",
    exposure_name="BMI", outcome_name="T2D",
    methods=["ivw", "egger", "weighted_median"],  # add "penalized_median"
)
print(r.summary())
r.plot()
```

The summary block also returns the IVW Cochran Q heterogeneity test
and the MR-Egger intercept test for directional pleiotropy. For
anything more specific, call the sub-estimators directly.

Reference: Burgess et al. (2013), *Genet Epidemiol* 37(7).

---

## 2. Point estimators — four assumption profiles

### `sp.mr_ivw` — Inverse-Variance Weighted

The workhorse. Fixed-effects meta-analysis of SNP-specific Wald ratios
with weights `1/se_Y_i²`. Consistent under **no pleiotropy** (balanced
or unbalanced) and the strongest power of the four when IV3 holds.

```python
res = sp.mr_ivw(beta_x, beta_y, se_x, se_y, alpha=0.05)
print(res["estimate"], res["se"], res["Q"], res["I2"])
```

**When to use it**: your default. Always report it. If it disagrees
with the three robust estimators below, the disagreement *is* the
pleiotropy signal.

### `sp.mr_egger` — directional pleiotropy

Fits `β_Y = α + β * β_X`, where `α` is the average pleiotropic
intercept. Point estimate is consistent under the **InSIDE assumption**
(Instrument Strength Independent of Direct Effect): the pleiotropic
effects are uncorrelated with the SNP–exposure effects.

```python
res = sp.mr_egger(beta_x, beta_y, se_x, se_y, alpha=0.05)
print(res["estimate"], res["intercept"], res["intercept_p"])
```

**When to use it**: you suspect **directional** pleiotropy (all
invalid SNPs bias in the same direction). A non-zero Egger intercept
is the evidence. InSIDE is not testable — sensitivity is king.

Reference: Bowden et al. (2015), *IJE* 44(2).

### `sp.mr_median` — weighted / penalized median

Consistent when at least **50% of the weight** comes from valid
instruments (Bowden et al. 2016). The penalized variant down-weights
SNPs with large IVW residuals, making the estimator more robust to a
few strong outliers:

```python
res = sp.mr_median(beta_x, beta_y, se_x, se_y,
                    penalized=False, n_boot=1000, seed=0)
res_pen = sp.mr_median(beta_x, beta_y, se_x, se_y,
                       penalized=True, n_boot=1000, seed=0)
```

**When to use it**: you suspect some SNPs are invalid but at most
half the weight is contaminated. SE is via parametric bootstrap.

Reference: Bowden et al. (2016), *Genet Epidemiol* 40(4).

### `sp.mr_mode` — mode-based (ZEMPA)

Consistent under the **ZEro Modal Pleiotropy Assumption**: the modal
Wald ratio across SNPs equals the true causal effect, even if *most*
SNPs are invalid — as long as invalid SNPs don't cluster at a single
pleiotropic effect.

```python
res = sp.mr_mode(beta_x, beta_y, se_x, se_y,
                 method="weighted",     # or "simple"
                 n_boot=1000, seed=0)
```

**When to use it**: the most permissive of the four — the last line
of defense when IVW, Egger, and median all disagree. A weighted
Gaussian-kernel mode on the Wald-ratio distribution, bandwidth via
Silverman's rule.

Reference: Hartwig, Davey Smith & Bowden (2017), *IJE* 46(6).

---

## 3. Diagnostics — the six tests you always report

### `sp.mr_heterogeneity` — Cochran's Q / Rücker's Q'

Is there more between-SNP variation in Wald ratios than sampling error
alone would predict? If yes, **at least one** SNP is pleiotropic.

```python
res = sp.mr_heterogeneity(beta_x, beta_y, se_y, method="ivw")
# For Egger-based Rücker's Q':
res = sp.mr_heterogeneity(beta_x, beta_y, se_y, method="egger", se_exposure=se_x)
print(res.Q, res.Q_p, res.I2)
```

`I² > 25%` → moderate heterogeneity; `I² > 50%` → substantial. IVW is
still consistent but the reported SE becomes anti-conservative —
switch to a random-effects IVW or use one of the robust estimators.

### `sp.mr_pleiotropy_egger` — directional pleiotropy test

The MR-Egger intercept `α` with its `t(n-2)` p-value (matches R's
`MendelianRandomization` package convention; uses `t` not `z` because
`σ²` is plug-in estimated):

```python
res = sp.mr_pleiotropy_egger(beta_x, beta_y, se_y)
print(res.intercept, res.p_value)
```

`p < 0.05` → evidence of directional pleiotropy → trust `mr_egger` /
`mr_median` / `mr_mode` over `mr_ivw`.

### `sp.mr_leave_one_out` — per-SNP influence

Recompute IVW dropping each SNP in turn. Useful for spotting a single
driver SNP:

```python
res = sp.mr_leave_one_out(beta_x, beta_y, se_y,
                          snp_ids=rsid_list, alpha=0.05)
print(res.table)    # dropped_snp, estimate, se, ci_lower, ci_upper, p_value
```

Red flag: if dropping one SNP swings the point estimate by > 25%, that
SNP is doing most of the identification work — check it against
`PhenoScanner` / `Open Targets Genetics` for pleiotropic associations.

### `sp.mr_steiger` — directionality

Tests whether the instruments explain **more** variance in the
exposure than in the outcome. If not, the causal direction is
reversed:

```python
res = sp.mr_steiger(
    beta_exposure=beta_x, se_exposure=se_x, n_exposure=n_x,
    beta_outcome=beta_y, se_outcome=se_y, n_outcome=n_y,
    eaf=eaf_per_snp,          # optional; uses t² / (t² + n - 2) fallback
)
print(res.correct_direction, res.steiger_pvalue)
```

**Always run this first** — it's cheap and catches reverse-causation
cases where all downstream MR is meaningless.

Reference: Hemani et al. (2017), *PLOS Genetics* 13(11).

### `sp.mr_presso` — global + outlier-correction

The Verbanck et al. (2018) outlier hunter. Runs:
1. **Global test** — is total pleiotropy larger than expected under no
   pleiotropy? (simulated null distribution of RSS)
2. **Per-SNP outlier test** — which specific SNPs exceed the null?
3. **Outlier-corrected IVW** — refit dropping flagged SNPs.
4. **Distortion test** — is the corrected estimate significantly
   different from the raw estimate?

```python
res = sp.mr_presso(
    beta_x, beta_y, se_x, se_y,
    n_boot=1000, sig_threshold=0.05, seed=0,
)
print(res.summary())
print(res.outliers)                          # flagged SNP indices
print(res.outlier_corrected_estimate)        # re-IVW after dropping outliers
```

**When to use it**: always, if you have ≥ 10 SNPs. The global test p
is the most reliable "do I have a pleiotropy problem" signal.

Reference: Verbanck et al. (2018), *Nature Genetics* 50(5).

### `sp.mr_radial` — Bowden (2018) radial plot + Q contributions

Reparameterizes each SNP's Wald ratio as a coordinate in "radial"
space and flags SNPs whose individual χ² contribution to Cochran's Q
exceeds the Bonferroni threshold:

```python
res = sp.mr_radial(beta_x, beta_y, se_y, snp_ids=rsids)
print(res.outliers)             # SNPs to investigate
print(res.table.nlargest(5, "q_contribution"))
```

**When to use it**: complement to `mr_presso` — radial MR has better
Type I control at small `n_snps`. If both flag the same SNP, that SNP
is doing something suspicious.

Reference: Bowden et al. (2018), *IJE* 47(4).

---

## 4. Instrument strength — `sp.mr_f_statistic`

Per-SNP F-statistic using the summary-stat approximation
`F_i = (β_i / SE_i)²`. The Staiger-Stock (1997) rule of thumb says
each SNP needs `F ≥ 10` to avoid weak-instrument bias:

```python
res = sp.mr_f_statistic(beta_x, se_x, n_samples=N_exposure_GWAS)
print(res.f_mean, res.f_min, res.weak_instrument_risk)
```

`f_min < 10` → the weakest SNP is a weak instrument → MR estimates
can be biased toward the confounded observational estimate. Remedy:
drop the weak SNPs or use **LIML / GRAPPLE** (not currently in
StatsPAI — pre-filter at GWAS p-value threshold `5e-8` first).

Reference: Staiger & Stock (1997), *Econometrica* 65(3).

---

## 5. Multi-exposure extensions

### `sp.mr_multivariable` — MVMR (Sanderson et al. 2019)

Include multiple correlated exposures and estimate **direct** effects
(holding the others fixed):

```python
res = sp.mr_multivariable(
    snp_df,
    outcome="beta_y", outcome_se="se_y",
    exposures=["beta_bmi", "beta_ldl", "beta_sbp"],
)
print(res.direct_effect)            # direct α per exposure with SE + CI
print(res.conditional_f_stats)      # Sanderson-Windmeijer F per exposure
```

Any `conditional_F < 10` → that exposure's instruments are weak **given
the other exposures** — consider dropping it or finding stronger SNPs.

Reference: Sanderson, Davey Smith, Windmeijer, Bowden (2019), *IJE* 48(3).

### `sp.mr_mediation` — two-step MR

Decompose the total effect of exposure → outcome into a **direct**
path and an **indirect** path through a mediator:

```python
res = sp.mr_mediation(
    snp_df,
    beta_exposure="beta_x", se_exposure="se_x",
    beta_mediator="beta_m", se_mediator="se_m",
    beta_outcome="beta_y", se_outcome="se_y",
    exposure_name="BMI", mediator_name="Glucose", outcome_name="T2D",
)
print(res.direct_effect, res.indirect_effect, res.proportion_mediated)
```

Under the hood: step 1 IVW for the total effect, step 2 MVMR for the
direct effect, indirect = total − direct, delta-method SE.

Reference: Burgess et al. (2015), *IJE* 44(2).

### `sp.mr_bma` — Bayesian Model Averaging

You have many candidate correlated risk factors and want to know
**which ones** are causal for the outcome. MR-BMA iterates over all
2^k subsets and reports marginal inclusion probabilities:

```python
res = sp.mr_bma(
    snp_df,
    outcome="beta_y", outcome_se="se_y",
    exposures=[f"beta_x{j}" for j in range(10)],
    prior_inclusion=0.5,
)
print(res.marginal_inclusion.sort_values(ascending=False))
print(res.best_models.head(10))
```

Uses WLS-BIC as the model score. Exhaustive up to `k ≤ 14`; for
larger `k` use `max_model_size` to restrict subset size.

Reference: Zuber, Colijn, Staley & Burgess (2020), *Nature Comms* 11, 29.

---

## 6. Visualization — `sp.mr_scatter_plot` + `sp.mr_funnel_plot`

```python
ax = sp.mr_scatter_plot(beta_x, beta_y, se_x, se_y)
# β_Y vs β_X with IVW slope (red dashed) + MR-Egger line (green)

ax = sp.mr_funnel_plot(beta_x, beta_y, se_y, snp_ids=rsids)
# Wald ratio vs precision — asymmetry = directional pleiotropy
```

The funnel plot is the fastest visual diagnostic for pleiotropy:
symmetry around the IVW line = no directional pleiotropy; systematic
tilt = Egger-territory pleiotropy.

---

## Decision guide

```
Standard two-sample summary-stat MR, one exposure
  → sp.mendelian_randomization   (runs IVW + Egger + weighted median)

I suspect horizontal pleiotropy
  ├─ Directional (all invalid SNPs bias the same way)
  │    → sp.mr_egger + sp.mr_pleiotropy_egger
  ├─ Non-directional but < 50% of weight
  │    → sp.mr_median (optionally penalized)
  ├─ Majority of SNPs invalid, dispersed
  │    → sp.mr_mode
  └─ Unknown — just want outliers removed
       → sp.mr_presso  (outlier detection + correction)

Check causal direction (is exposure really upstream?)
  → sp.mr_steiger

Check instrument strength
  → sp.mr_f_statistic  (univariate)
  → sp.mr_multivariable's conditional_f_stats  (multivariable)

Multiple exposures
  ├─ Correlated, want direct effects
  │    → sp.mr_multivariable
  ├─ Exposure → mediator → outcome
  │    → sp.mr_mediation
  └─ Many candidates, which are causal?
       → sp.mr_bma

Visualize
  → sp.mr_scatter_plot + sp.mr_funnel_plot
```

---

## The four sanity checks every MR analysis should report

1. **`sp.mr_steiger`** — confirm the direction. If p > 0.05 or
   `r²_outcome > r²_exposure`, your MR is backwards; stop.
2. **`sp.mr_f_statistic`** — all SNPs with F ≥ 10? If no, weak-IV bias
   is likely; either drop or tighten the GWAS p-value threshold.
3. **`sp.mr_heterogeneity`** — I² < 25%? If no, run robust estimators
   (Egger / median / mode) and report the one closest to IVW.
4. **`sp.mr_presso`** — global test p > 0.05? If no, inspect flagged
   outliers and report both raw and outlier-corrected estimates.

Without all four, a reviewer will ask. Always run them.

---

## How to read disagreement

**IVW vs Egger**: if the slopes differ and the Egger intercept is
significant, **trust Egger** — IVW is biased by the directional
pleiotropy that Egger's intercept is modeling. If the slopes agree,
IVW is preferred (tighter SE).

**IVW vs Median**: if median is closer to zero than IVW, some SNPs
with large Wald ratios are driving IVW — run `leave_one_out` to find
them.

**Egger vs Mode**: both are robust, but Egger needs InSIDE (untestable)
and Mode needs ZEMPA (testable via Hartwig plot). If they agree,
you're in a good place. If they disagree, report both and defer the
interpretation to the biology.

**MVMR vs univariate MR**: if the univariate effect is large but the
MVMR direct effect is near zero, the exposure's apparent causal
effect was actually mediated by one of the other exposures in the
MVMR — valuable information for mechanism.

---

## Worked example: BMI → T2D

```python
import statspai as sp
import pandas as pd

snp_df = pd.DataFrame({
    "rsid":   ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6", "rs7", "rs8"],
    "beta_x": [ 0.10, 0.15, 0.12, 0.08, 0.20, 0.11, 0.07, 0.18],
    "se_x":   [0.02,  0.03, 0.02, 0.02, 0.04, 0.02, 0.02, 0.03],
    "beta_y": [ 0.40, 0.55, 0.48, 0.32, 0.80, 0.44, 0.28, 0.72],
    "se_y":   [0.08,  0.10, 0.09, 0.08, 0.15, 0.09, 0.08, 0.13],
})

# 1. Always start with direction
print(sp.mr_steiger(
    beta_exposure=snp_df["beta_x"].values,
    se_exposure=snp_df["se_x"].values,
    n_exposure=700_000,   # UK Biobank ≈ 700k
    beta_outcome=snp_df["beta_y"].values,
    se_outcome=snp_df["se_y"].values,
    n_outcome=200_000,    # DIAMANTE ≈ 200k
).summary())

# 2. Instrument strength
print(sp.mr_f_statistic(
    snp_df["beta_x"].values, snp_df["se_x"].values, n_samples=700_000,
).summary())

# 3. Main analysis
r = sp.mendelian_randomization(
    data=snp_df,
    beta_exposure="beta_x", se_exposure="se_x",
    beta_outcome="beta_y", se_outcome="se_y",
    exposure_name="BMI", outcome_name="T2D",
    methods=["ivw", "egger", "weighted_median"],
)
print(r.summary())

# 4. Outlier check
presso = sp.mr_presso(
    snp_df["beta_x"].values, snp_df["beta_y"].values,
    snp_df["se_x"].values, snp_df["se_y"].values,
    n_boot=2000, seed=0,
)
print(presso.summary())
```

IVW point estimate on this toy data ≈ 4.0 (scale of `β_Y / β_X`),
Egger intercept not significant, `I² ≈ 0%`, MR-PRESSO global p > 0.05
→ no pleiotropy detected → report IVW as the headline, with Egger and
median in the supplement.

---

*Current for StatsPAI ≥ 1.5.0. All functions are registered; use
`sp.describe_function("mr_presso")` or similar for schemas. For
dispatcher-style access see `sp.mr(method=...)`.*

<!-- AGENT-BLOCK-START: mr -->

## For Agents

**Pre-conditions**
- SNP-summary statistics for exposure and outcome aligned by SNP
- beta_exposure / beta_outcome / se_exposure / se_outcome arrays of equal length
- ≥ 10 genetic instruments for reliable IVW/median/mode; ≥ 20 for robust Egger intercept
- mvmr needs SNP × exposure associations matrix

**Identifying assumptions**
- Relevance: SNPs predict exposure (F-statistic ≥ 10 per SNP or set-F)
- Independence: SNPs ⊥ confounders of exposure-outcome
- Exclusion restriction: SNPs affect outcome only through exposure (InSIDE for Egger; ≥ 50% valid for median; modal for mode-based)
- Monotonicity when interpreting LATE on genetically-shifted subpopulation

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Egger intercept p < 0.05 — directional pleiotropy | `statspai.AssumptionViolation` | Use weighted-median or mode-based estimator; report Egger intercept + I² as pleiotropy diagnostic. | `sp.mr_median` |
| Q-statistic rejects homogeneity (Cochran's Q p < 0.05) | `statspai.AssumptionWarning` | Heterogeneity across SNPs — run sp.mr_presso to detect/remove outliers. | `sp.mr_presso` |
| Set-F < 10 (weak instruments in aggregate) | `statspai.AssumptionWarning` | Weak-IV bias in IVW — use debiased IVW or LAP-type estimator (sp.mr_lap). | `sp.mr_lap` |
| Steiger test flags reverse causation | `statspai.IdentificationFailure` | SNPs explain more outcome variance than exposure — direction of effect questionable. |  |

**Alternatives (ranked)**
- `sp.mr_ivw`
- `sp.mr_egger`
- `sp.mr_median`
- `sp.mr_presso`
- `sp.mr_multivariable`
- `sp.iv`

**Typical minimum N**: 10

<!-- AGENT-BLOCK-END -->
