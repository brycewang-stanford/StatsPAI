# Quantile & distributional treatment effects — the full family

> **"The ATE is misleading when treatment redistributes outcomes."**
> A policy can leave the mean unchanged while compressing or spreading
> the distribution — only QTE and distributional-TE catch that.
> StatsPAI v1.4 ships the full 2025-2026 frontier.

The Average Treatment Effect answers: "how much did the mean outcome
move?"  But for policy, clinical, and fairness work you usually care about
*who* got moved.  Did the bottom decile benefit?  Did treatment widen
inequality?  Those questions need quantile or distributional effects.

Every function below is at top level: `sp.qdid`, `sp.qte`,
`sp.distributional_te`, `sp.dist_iv`, `sp.kan_dlate`,
`sp.beyond_average_late`, `sp.qte_hd_panel`, `sp.cic`.

---

## The three levels of granularity

| Level                  | What it answers                                  | Estimator(s)                     |
|------------------------|--------------------------------------------------|----------------------------------|
| Average                | "Did the mean go up?"                            | `sp.did`, `sp.iv`, `sp.dml`      |
| Quantile (τ-th)         | "Did the bottom 10%? The median? The top 10%?"  | `sp.qte`, `sp.qdid`, `sp.cic`    |
| Whole distribution     | "How did the *shape* change?"                    | `sp.distributional_te`, `sp.dist_iv` |

---

## Conditional QTE under unconfoundedness — `sp.qte`

The default when you have observational data with `X` rich enough for
selection-on-observables:

```python
r = sp.qte(
    data=df, y="wage", treatment="job_program",
    controls=["age", "education", "experience"],
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    method="quantile_regression",   # or "ipw"
    n_boot=500,
)
r.plot()    # QTE curve with bootstrap uniform band
```

Reference: Firpo (2007), *Econometrica* 75(1).  IPW variant: Firpo (2007)
§4 with the propensity score.

---

## Quantile DiD — `sp.qdid`

Two-period, two-group DiD at each quantile.  Parallel-trends is replaced
by the **quantile parallel trends** assumption: "in the absence of
treatment, the τ-th quantile of the treated group would have moved the
same way as the τ-th quantile of the control group."

```python
r = sp.qdid(
    data=df, y="y", group="treated", time="post",
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_boot=500,
)
r.summary()     # QTE(τ) for each τ with placebo-based CI
```

Reference: Athey & Imbens (2006), *Econometrica* 74(2).

---

## Changes-in-changes (CiC) — `sp.cic`

Weaker than QDiD: CiC requires only that the **rank** of each unit within
its group's outcome distribution is stable over time in the absence of
treatment.  No parallel-trends.  Works when outcome distributions have
different shapes across groups.

```python
r = sp.cic(
    data=df, y="earnings", group="treated", time="post",
    quantiles=np.linspace(0.05, 0.95, 19),
    n_boot=500,
)
```

Reference: Athey & Imbens (2006), *Econometrica*.  The same paper proves
CiC nests QDiD when the distributions happen to share scale.

---

## Distributional TE (whole density/CDF) — `sp.distributional_te`

Instead of a QTE curve at a handful of quantiles, `distributional_te`
estimates the entire counterfactual CDF:

```python
r = sp.distributional_te(
    data=df, y="income", treatment="program",
    x=["age", "education"],
    method="ipw",           # or "regression"
    n_grid=200,
    n_boot=500,
)
r.plot_cdf()    # control CDF vs counterfactual-under-treatment CDF
r.plot_pdf()    # with bootstrap uniform band
```

**When to use it**: when you care about stochastic-dominance claims
("treatment first-order dominates control") or Lorenz-curve-style
inequality summaries.

---

## Distributional IV — `sp.dist_iv` (Sharma-Xue 2025)

IV at every quantile, with a uniform band over τ:

```python
r = sp.dist_iv(
    data=df, y="earnings", treat="schooling",
    instrument="compulsory_reform",
    covariates=["age", "region"],
    quantiles=np.linspace(0.1, 0.9, 17),
    n_boot=200,
)
```

**Identification**: extends Abadie (2002) QTE-under-compliance to a
continuous grid of τ with uniform-over-τ confidence coverage, which is
what you need to report "schooling raises wages at every quantile" or
"the top-decile return is twice the bottom-decile return."

Citation: Sharma & Xue (2025), arXiv:2502.07641.

---

## KAN D-LATE — `sp.kan_dlate` (Kennedy et al. 2025)

A Kolmogorov-Arnold-Network-powered variant of `dist_iv` that replaces
the linear bridge function with a flexible KAN surface.  Same call shape
as `dist_iv`, swap in when you suspect the return-to-schooling (say) is
strongly non-linear in `D`.

```python
r = sp.kan_dlate(
    data=df, y="y", treat="d", instrument="z",
    covariates=["x1", "x2"],
    quantiles=np.linspace(0.1, 0.9, 9),
)
```

Citation: Kennedy et al. (2025), arXiv:2506.12765.

---

## Beyond-average LATE — `sp.beyond_average_late` (Xie-Wu 2025)

The Angrist-Imbens LATE gives the treatment effect among *compliers*
(units whose treatment would flip if the instrument flipped).  But that's
an *average* over the complier subgroup.  Xie-Wu identify the **whole
distribution** of treatment effects among compliers:

```python
r = sp.beyond_average_late(
    data=df, y="y", treat="d", instrument="z",
    quantiles=np.linspace(0.1, 0.9, 9),
    n_boot=200,
)
```

**When it matters**: if the complier class has a heavy-tailed treatment-
effect distribution, the mean LATE hides heterogeneity that a decile or
quantile summary reveals.  Classic example: schooling returns vary
dramatically across compliers by ability.

Citation: Xie & Wu (2025), arXiv:2509.15594.

---

## High-dimensional panel QTE — `sp.qte_hd_panel` (Fan et al. 2025)

Panel data with many covariates?  `qte_hd_panel` fits an ℓ1-penalised
quantile regression per-τ with a Lasso nuisance step:

```python
r = sp.qte_hd_panel(
    data=df, y="y", treat="d",
    unit="county", time="year",
    covariates=[f"x{j}" for j in range(50)],
    quantiles=np.linspace(0.1, 0.9, 9),
    lasso_alpha=0.01,
)
```

**When to use**: many controls (`p ≥ 20` or so), panel DiD-flavoured
identification, and you care about the τ curve.  Honest post-Lasso
inference follows Belloni-Chernozhukov-Fernández-Val (2017).

Citation: Fan et al. (2025), arXiv:2504.00785.

---

## Decision guide

```
Cross-section + unconfoundedness:
  ATE only             → sp.dml / sp.aipw
  QTE at some τ        → sp.qte
  Full distribution    → sp.distributional_te

DiD design:
  Mean ATT             → sp.did
  QTE(τ) at each τ     → sp.qdid
  Rank-preserving      → sp.cic

IV design:
  LATE (mean)          → sp.iv
  Distributional LATE  → sp.dist_iv (linear) / sp.kan_dlate (KAN)
  Beyond-average LATE  → sp.beyond_average_late (distribution among compliers)

Panel with many controls:
  → sp.qte_hd_panel
```

---

## Sanity checks to report

1. **Monotonicity in τ** — the QTE curve should move smoothly with τ;
   wild oscillations signal overfitting or instability.
2. **Uniform band coverage** — report the Kolmogorov-Smirnov-style
   uniform band, not just pointwise CIs at each τ.
3. **Stochastic dominance test** — `sp.stochastic_dominance(r)` tells
   you if the treated-CDF first-order dominates control.
4. **Common-support diagnostic** — for IV/IPW-based QTE, verify the
   propensity overlap at *every* τ you report.

---

*Current for StatsPAI ≥ 1.5.0.  All functions are registered; inspect
with `sp.describe_function("beyond_average_late")`, etc.*
