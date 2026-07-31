# Choosing a QTE estimator

> Quantile and distributional treatment effects in StatsPAI: which estimand
> you are asking for, which estimator delivers it, and what each one assumes.

Everything below uses `import statspai as sp`.

---

## 0. First: do you want a quantile effect at all?

A quantile treatment effect answers a question an average effect cannot:
**does treatment do the same thing to everyone?**

The trap is that the two are indistinguishable on the DGP people usually
reach for. If treatment adds a constant δ to every unit, then
`QTE(τ) = δ` at every τ — a flat line — and *any* estimator that returns a
number will look right. A quantile analysis only earns its keep when the
effect varies with τ.

So before choosing an estimator, decide whether you can test that:

```python
res = sp.qte(df, y="wage", treatment="program",
             quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
             method="firpo_qte", se="analytic")

res.test_no_effect()        # H0: QTE(τ) = 0 at EVERY τ
res.test_constant_effect()  # H0: QTE(τ) is the same at every τ
```

`test_constant_effect` is the one that matters. If you cannot reject it,
the ATE is an adequate summary and the quantile curve is decoration. These
tests need `se="analytic"`, which supplies the influence functions the
multiplier bootstrap runs on.

!!! warning "Pointwise intervals do not support curve-level claims"
    A row of 95% intervals covers each τ *separately*. Over 9 quantiles the
    probability that all of them cover at once is about **76%**, not 95%
    (measured; see `tests/reference_parity/test_qte_uniform_inference.py`).
    For statements like "the effect is positive across the distribution",
    use the simultaneous band on `res.ci_lower_uniform` /
    `res.ci_upper_uniform`. Its critical value is ~2.64 here against
    z = 1.96.

---

## 1. Decision tree

```text
Is treatment randomised or unconfounded given covariates?
├── YES ──────────────────────────────────────────────────────────┐
│                                                                 │
│   Do you want the effect on EVERYONE or on the TREATED?         │
│   ├── everyone  →  sp.qte(method="firpo_qte")                   │
│   └── treated   →  sp.qte(method="firpo_qtt")                   │
│                                                                 │
│   Want the whole counterfactual CDF, not just quantiles?        │
│   └── sp.distributional_te(method="ipw" | "dr")                 │
│                                                                 │
└── NO ───────────────────────────────────────────────────────────┐
                                                                  │
    Do you have a valid instrument?                               │
    ├── YES → sp.dist_iv  (complier QTE, Abadie κ weighting)      │
    │         sp.beyond_average_late  (same estimand, same core)  │
    │                                                             │
    └── NO → Do you have before/after on treated and controls?    │
        ├── YES → sp.qdid(method="cic")   ← prefer this           │
        │         sp.qdid(method="qdid")  ← only under a          │
        │                                    location shift       │
        │                                                         │
        └── Panel with many candidate controls?                   │
            └── sp.qte_hd_panel(method="canay")                   │
```

---

## 2. The estimators

### `sp.qte` — cross-section

| `method` | Estimand | Assumes |
| --- | --- | --- |
| `"firpo_qte"` *(default)* | **Unconditional QTE**: `F⁻¹_{Y(1)}(τ) − F⁻¹_{Y(0)}(τ)` | Unconfoundedness + overlap |
| `"firpo_qtt"` | **QTT**: the same contrast among the treated | Unconfoundedness + overlap |
| `"conditional_qr"` | **Conditional** QTE: the coefficient on `D` in a quantile regression of `Y` on `D + X` | Correct quantile model; **no causal reading without rank invariance** |
| `"distribution"` | **QTT** via an IPW counterfactual distribution | Unconfoundedness + overlap |

The distinction between the first and third rows is the one people get
wrong most often, and StatsPAI itself got it wrong until 1.21.0: the
`conditional_qr` coefficient is a within-covariate-cell statement. It does
**not** aggregate to an effect on the marginal distribution of the outcome.
If you want "the effect at the 25th percentile of the wage distribution",
you want `firpo_qte`.

Standard errors: `se="analytic"` (influence function, and the only path that
gives uniform bands and curve tests) or `se="bootstrap"`. The default
`se="auto"` picks analytic without covariates and bootstrap with them,
because the analytic form treats `p(X)` as known — which is conservative
when it is estimated.

### `sp.dist_iv` / `sp.beyond_average_late` — endogenous treatment

Complier QTE, `F⁻¹_{Y(1)|c}(τ) − F⁻¹_{Y(0)|c}(τ)`, by Abadie (2002, 2003)
κ weighting. With `covariates=` this is Frölich & Melly (2013). Both
functions share one core, so they agree to 1e-10.

Requires the standard LATE assumptions. A near-zero first stage triggers a
weak-instrument warning; every downstream quantity divides by the complier
share.

!!! danger "If you have results from ≤ 1.20.0"
    `sp.dist_iv` computed a *Wald ratio of quantiles*, which is inconsistent
    for any quantile estimand — quantiles are not linear, so the mean-Wald
    rescaling does not carry over. Old estimates were inflated by roughly
    `1/Δp`. Re-run; see [MIGRATION](../../MIGRATION.md#dist-iv-quantile-wald-ratio).

### `sp.qdid` — repeated cross-section / two-period panel

- `method="cic"` → delegates to `sp.cic`, Athey & Imbens (2006)
  changes-in-changes. **Prefer this.**
- `method="qdid"` → quantile DiD,
  `[Q₁₁(τ) − Q₁₀(τ)] − [Q₀₁(τ) − Q₀₀(τ)]`.

QDiD assumes the untreated outcome distribution shifts by the *same amount
at every rank* between periods. That is a strong restriction, and it is why
Athey & Imbens propose CiC instead — their paper criticises QDiD directly.
Use `qdid` only when a location shift is credible (or to show your result
does not hinge on the choice).

!!! note "`sp.qdid` is not CiC"
    Through 1.20.0 this function was labelled Athey & Imbens (2006)
    changes-in-changes. The numbers were always QDiD; only the attribution
    was wrong. `qte::MDiD` and `qte::ddid2` are not yet available.

### `sp.qte_hd_panel` — panel with many controls

Double-selection LASSO for the controls, then:

| `method` | Handles | Assumes |
| --- | --- | --- |
| `"canay"` *(default)* | Individual effects | Unit effect is a **pure location shift**; large `T` |
| `"dummy_fe"` | Individual effects | No location-shift restriction; incidental-parameter bias at small `T` |
| `"pooled"` | — | No individual effects at all |

Canay's assumption is not decorative: the estimator subtracts a single
`α̂ᵢ` estimated from a *mean* regression, so it presumes the unit effect
does not itself vary across quantiles. And it is a **large-`T`** estimator —
`α̂ᵢ` carries `O(T^{-1/2})` error. Measured on a scale-shift design, max
error runs 0.186 at T=20, 0.068 at T=50, 0.055 at T=200. Panels with mean
`T < 5` emit a warning. Cross-check against `"dummy_fe"` when `T` is short.

### `sp.distributional_te` — the whole counterfactual distribution

Returns the CDF difference across the outcome support plus QTEs, with KS and
Cramér-von Mises tests of "no distributional effect". `method="dr"` uses
distribution regression (Chernozhukov, Fernández-Val & Melly 2013) for the
conditional CDF and is doubly robust; `method="ipw"` needs only the
propensity score; `method="cic"` takes a four-cell group encoding.

---

## 3. Assumption cheat-sheet

| Estimator | Identification | Fails when |
| --- | --- | --- |
| `firpo_qte` / `firpo_qtt` | Unconfoundedness + overlap | Hidden confounding; `p(X)` near 0 or 1 |
| `conditional_qr` | Unconfoundedness + correct quantile model | You wanted an unconditional effect |
| `dist_iv` / `beyond_average_late` | Random assignment, exclusion, monotonicity, first stage | Weak instrument; defiers |
| `qdid` | Untreated distribution shifts by a constant at every rank | Any rank-varying trend |
| `cic` | Monotone, rank-invariant production function | Discrete outcomes (only bounds are identified) |
| `qte_hd_panel(canay)` | Unit effect is a pure location shift; large `T` | Unit effects that vary by quantile; short panels |

---

## 4. Reference alignment

The family is anchored to **R `qte` 1.3.1** and **`quantreg` 6.1** on the
`qte` package's own `lalonde.exp` / `lalonde.psid` / `lalonde.psid.panel`
data. Fixtures and generators live in
`tests/reference_parity/_fixtures/_generate_qte_firpo_R.R` and
`_generate_qte_panel_R.R`.

One thing worth knowing when you compare against R yourself: R's weighted
quantiles come from `BMisc::weighted_quantile`, which minimises the check
function with `stats::optimize` — a golden-section search on a
**piecewise-linear** objective. Between order statistics the objective has
plateaus where every point is a minimiser, so R's reported value there is an
artifact of the optimiser, not a functional of the data. (It returns
`-5.93e-06` for a quantile whose exact value is `0`.) StatsPAI solves the
same problem in closed form. The parity suite therefore certifies that our
solution attains a check-function value **no worse than R's at every
quantile** — a stronger claim than approximate numeric agreement — and
separately reports where the two coincide numerically.

---

## 5. Known gaps

- `qte::MDiD` and `qte::ddid2` are not implemented.
- Callaway & Li (2019) copula-based panel QTT (`qte::panel.qtet`) is not
  implemented; R reference values are already staged in
  `tests/reference_parity/_fixtures/qte_panel_R.json`.
- Uniform bands and curve-level tests are currently available on the Firpo
  estimators only, since they require influence functions.
- Analytic standard errors treat `p(X)` as known; use the bootstrap when the
  propensity model is estimated and you need the efficiency gain reflected.

See [`docs/rfc/qte_two_month_plan.md`](../rfc/qte_two_month_plan.md) for the
full work plan and its evidence.

---

## References

Abadie (2002, 2003); Athey & Imbens (2006); Belloni, Chernozhukov & Hansen
(2014); Canay (2011); Chernozhukov, Fernández-Val & Galichon (2010);
Chernozhukov, Fernández-Val & Melly (2013); Firpo (2007); Frölich & Melly
(2013); Koenker (2004); Koenker & Bassett (1978).

Canonical entries are in [`paper.bib`](https://github.com/brycewang-stanford/StatsPAI/blob/main/paper.bib).
