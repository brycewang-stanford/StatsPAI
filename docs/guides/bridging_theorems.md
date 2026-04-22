# Bridging theorems — when two methods target the same estimand

> **Six 2025-2026 results proving that pairs of superficially different
> estimators identify the same causal quantity under distinct
> assumptions.** Running both and comparing is how you turn assumption
> uncertainty into data-visible diagnostics.

A bridging theorem takes two popular causal-inference methods and shows
that — when their identifying assumptions hold — they target the *same*
estimand. Empirically they should give numerically close point
estimates.  Disagreement is then a signal that at least one of the two
assumptions fails in your data.

This is the spiritual descendant of the classic "OLS = IV under no
endogeneity" sanity check, but lifted to the 2020s frontier of DiD,
synthetic controls, welfare maximisation, covariate balancing, and
proximal/surrogate long-term effects.

`sp.bridge(kind=..., **kwargs)` is the unified dispatcher. It runs both
paths, computes an agreement test on the difference, and returns a
doubly-robust combined estimate — plus a warning if the two paths
disagree beyond sampling error.

---

## The six bridges

| `kind`          | Theorem                                           | Reference                                                       |
|-----------------|---------------------------------------------------|-----------------------------------------------------------------|
| `did_sc`        | DiD ≡ Synthetic Control                           | Shi & Athey, arXiv:2503.11375 (2025)                            |
| `ewm_cate`      | EWM ≡ CATE-max policy                             | Ferman et al., arXiv:2510.26723 (2025)                          |
| `cb_ipw`        | Covariate Balancing ≡ IPW ≡ DR                    | Zhao & Percival, arXiv:2310.18563 v6 (2025)                     |
| `kink_rdd`      | Bunching ≡ Kink RDD first-order expansion         | Lu, Wang, Xie, arXiv:2404.09117 (2025)                          |
| `dr_calib`      | Doubly-robust ≡ outcome + Riesz joint calibration | Zhang et al., arXiv:2411.02771 (2025)                           |
| `surrogate_pci` | Long-term Surrogate Index ≡ PCI                   | Imbens, Kallus, Mao & Wang, JRSS-B 87(2) 2025; arXiv:2202.07234 |

Each bridge is importable at top level as `sp.bridge(kind="..." ...)` or
via the per-module function in `statspai.bridge.*`.

---

## How to read a `BridgeResult`

```python
>>> r = sp.bridge(kind="cb_ipw", data=df, y="y", treat="d",
...               covariates=["x1", "x2", "x3"])
>>> r.summary()
```

The key fields:

- `estimate_a`, `estimate_b` — the two path point estimates.
- `se_a`, `se_b` — their standard errors.
- `diff`, `diff_se`, `diff_p` — difference and formal test
  `H0: estimate_a - estimate_b = 0`.
- `estimate_dr`, `se_dr` — precision-weighted combination, the
  recommended final number **when the agreement test does not reject**.
- `path_a_name` / `path_b_name` — human-readable method names.
- `reference` — the citation for this bridge.

**Rule of thumb**: if `diff_p > 0.10`, trust the DR estimate. If
`diff_p < 0.05`, at least one identifying assumption is violated —
inspect the detail dict (`r.detail`) for method-specific diagnostics
before trusting either path.

---

## 1. DiD ≡ Synthetic Control — Shi-Athey (2025)

The result: whenever (a) the classical DiD parallel-trends assumption
*or* (b) unit-level synthetic-control weights identify the ATT, then
*both* methods recover the same quantity. The intuition is that the
DiD-weighted counterfactual is one specific convex combination of
control units (equal weights), and SC is another; any convex
combination is admissible.

```python
r = sp.bridge(
    kind="did_sc", data=df,
    y="gdp", unit="state", time="year",
    treated_unit="CA", treatment_time=1989,
)
print(f"DiD path   : {r.estimate_a:.3f} ± {r.se_a:.3f}")
print(f"SC path    : {r.estimate_b:.3f} ± {r.se_b:.3f}")
print(f"Agreement p: {r.diff_p:.3f}")
print(f"DR estimate: {r.estimate_dr:.3f}")
```

**When to use**: You have a single (or a handful of) treated units
adopted at a known time, and you want to hedge between
"pre-trends look parallel" and "my donor pool supports a good match."

**What disagreement tells you**: At least one of (parallel trends,
donor-pool matching) is violated. Inspect the SC weights (`r.detail`)
and the DiD residuals for a pre-trends placebo.

---

## 2. EWM ≡ CATE — Ferman et al. (2025)

Empirical Welfare Maximisation (the Kitagawa-Tetenov IPW welfare)
and policy optimisation via CATE-plug-in identify the same optimal
policy value under overlap + correct outcome/propensity modelling.

```python
r = sp.bridge(
    kind="ewm_cate", data=df,
    y="y", treat="d",
    covariates=["x1", "x2", "x3"],
)
```

**When to use**: You want to learn a treatment-assignment rule, not
just an ATE. Running both paths tells you whether your nuisance
models are internally consistent.

---

## 3. Covariate Balancing ≡ IPW ≡ DR — Zhao-Percival (2025 v6)

Under appropriate constraint choices, covariate balancing weights,
straight inverse-propensity weights, and augmented-IPW (doubly-robust)
all deliver the *same* point estimate of the ATE. What differs is
finite-sample efficiency and sensitivity to model misspecification.

```python
r = sp.bridge(
    kind="cb_ipw", data=df,
    y="y", treat="d",
    covariates=["x1", "x2", "x3"],
)
```

**Empirical check**: On a 400-obs synthetic DGP with true ATE = 1.5,
`cb_ipw` recovers 1.51 / 1.46 on the two paths with DR = 1.48 — the
two estimators agree within one SE.

---

## 4. Bunching ≡ Kink RDD — Lu-Wang-Xie (2025)

Saez (2010) showed that observed bunching at a kink point identifies
the behavioural elasticity. Lu-Wang-Xie prove that under standard
smoothness conditions the Saez bunching estimator is numerically equal
to the first-order expansion of a Kink-RDD slope-change estimator.
Running both tells you whether the identifying first-order conditions
are binding in your data.

```python
r = sp.bridge(
    kind="kink_rdd", data=df,
    y="hours_worked", running="taxable_income", cutoff=60_000,
    polynomial=2,
)
```

**Disagreement interpretation**: Large gap between `estimate_a` (kink
slope change) and `estimate_b` (bunching mass) means either the kink
is sharper than a first-order expansion captures (need higher-order
correction) or the bunching density is distorted by optimisation
frictions (anti-bunching, notches, etc.).

---

## 5. Doubly-robust via calibration — Zhang et al. (2025)

AIPW, TMLE, and DML all rely on **two** nuisance fits: an outcome
model (regression of Y on A, X) and a Riesz representer (here, the
propensity or its inverse). Zhang et al. show that finite-sample
double robustness is equivalent to *jointly calibrating* the two
fits — either via isotonic projection or via the self-tuned
influence-function residual.

```python
r = sp.bridge(
    kind="dr_calib", data=df,
    y="y", treat="d",
    covariates=["x1", "x2", "x3"],
)
```

**Why bother?** Without calibration, two reasonable ML fits can each
be miscalibrated in opposite ways and cancel out the AIPW's doubly-
robust guarantee. The bridge report makes the underlying agreement
explicit.

---

## 6. Long-term Surrogate Index ≡ Proximal Causal Inference — Imbens-Kallus-Mao-Wang (2025)

Surrogate indices (Athey-Chetty-Imbens-Kang, NBER WP 26463, 2019) use
short-term measurements as proxies for long-term outcomes. Imbens,
Kallus, Mao & Wang show that under a completeness condition, the
surrogate-index estimand is *identical* to a proximal-causal-inference
(PCI) estimand using the same short-term variables as proxies for an
unobserved confounder.

```python
r = sp.bridge(
    kind="surrogate_pci", data=df,
    long_term="revenue_24mo", short_term=["dau_90d", "retention_90d"],
    treat="feature_flag", covariates=["pre_dau", "pre_purch"],
)
```

**Why this matters**: You get two totally different identification
arguments for the same number. If they agree, you can trust the
long-term extrapolation; if they disagree, your surrogates are either
incomplete (miss relevant confounding) or mis-selected (the bridge
function is not identified).

---

## When to reach for bridges

1. **Before a PhD/prof audience**, to hedge across identification
   frameworks — "we report the DR-combined estimate; both paths agree."
2. **In a policy report**, to show the answer is robust to methodology
   choice.
3. **In internal A/B-testing tools**, to catch nuisance-model
   misspecification via cross-path disagreement.
4. **In replication studies**, to stress-test a prior paper's single
   path against an alternative.

Every bridge is wired into `sp.list_functions()`, so LLM agents can
discover them by searching e.g. "doubly robust", "synthetic control",
or "surrogate" — the tags in each spec make the cross-references
explicit.

```python
sp.search_functions("bridge")   # all six + the dispatcher
sp.describe_function("bridge")  # curated JSON schema
```

---

*This guide corresponds to v1.3+ of StatsPAI; the bridge module
`statspai.bridge` is stable and will not see breaking API changes in
1.x.*
