# Principal Stratification

Frangakis & Rubin (2002). Principal strata classify units by the
joint potential values of a **post-treatment** variable — most often
compliance type or survival status — and the resulting
**principal causal effects (PCEs)** are conditional on stratum
membership.

| Scenario | Post-treatment variable `S` |
| --- | --- |
| Encouragement design / noncompliance | Actual take-up |
| Survival / truncation-by-death | Alive-at-follow-up indicator |
| Employment after training | Employed indicator |
| Dropout / missingness mechanism | Response indicator |

StatsPAI ships two identification strategies via
`sp.principal_strat(..., method=...)`:

1. **`'monotonicity'`** — Angrist-Imbens-Rubin (1996) +
   Zhang-Rubin (2003) sharp bounds.
2. **`'principal_score'`** — Ding & Lu (2017) principal-score
   weighting (point identification under principal ignorability).

Plus a convenience wrapper `sp.survivor_average_causal_effect(...)`
for the classical truncation-by-death problem.

---

## Monotonicity method — complier LATE + SACE bounds

Under AIR monotonicity (`S(1) ≥ S(0)` almost surely, i.e. no
defiers), the three surviving strata have observable mixture
decompositions:

| Stratum | Proportion | Name |
| --- | --- | --- |
| `(S(0)=1, S(1)=1)` | `P(S=1 \| D=0)` | Always-taker / always-survivor |
| `(S(0)=0, S(1)=1)` | `P(S=1 \| D=1) - P(S=1 \| D=0)` | Complier / harmed |
| `(S(0)=0, S(1)=0)` | `P(S=0 \| D=1)` | Never-taker / never-survivor |

**Complier PCE (= LATE)** is point-identified by the Wald-type
ratio, and **the always-survivor SACE** has sharp Zhang-Rubin bounds.

```python
res = sp.principal_strat(
    df, y='wage', treat='treat', strata='employed',
    method='monotonicity',
    n_boot=500, seed=0,
)

res.strata_proportions         # dict: always / complier / never shares
res.effects                    # DataFrame — LATE row with SE / CI / p-value
res.bounds                     # DataFrame — SACE lower / upper bounds
print(res.summary())           # pretty-printed
```

Zhang-Rubin bounds on **E[Y(1) - Y(0) | always-survivor]** come from
worst/best-case slicing of the (D=1, S=1) cell, which is a mixture
of always-survivors (share `q = π_always / P(S=1|D=1)`) and
compliers. When `π_always ≈ 0` the bounds degenerate to `NaN`
(flagged explicitly).

---

## Principal-score method — point identification via covariates

Under **principal ignorability** (`Y(d) ⊥ stratum | X` within
`D=d`) + monotonicity, stratum membership is identified from the
observable cell probabilities `p11(X) = P(S=1|D=1,X)` and
`p10(X) = P(S=1|D=0,X)`:

```text
e_always(X)   = p10(X)
e_complier(X) = p11(X) - p10(X)
e_never(X)    = 1 - p11(X)
```

The stratum-specific ATEs are then estimated by covariate-weighted
slicing of the observed cells (Ding & Lu 2017).

```python
res = sp.principal_strat(
    df, y='wage', treat='treat', strata='employed',
    covariates=['age', 'edu', 'tenure'],
    method='principal_score',
    n_boot=500, seed=0,
)
res.effects                     # DataFrame: complier / always / never PCE
```

Returns per-stratum point estimate + bootstrap SE/CI/p-value for
all three strata.

**Monotonicity diagnostics**: when the fitted `p11(x) < p10(x)`
for more than 5 % of units, a `RuntimeWarning` fires and
`model_info` records:

```python
res.model_info['mono_violation_frac']     # fraction with p11<p10
res.model_info['mono_min_raw_complier']   # min(p11(x) - p10(x))
```

Principal ignorability is a strong assumption — pair this with
sensitivity analysis before publishing.

---

## Survivor Average Causal Effect wrapper

Classical truncation-by-death problem (Zhang-Rubin 2003):

```python
sace = sp.survivor_average_causal_effect(
    df, y='qol_score', treat='treat', survival='alive',
    alpha=0.05, n_boot=500, seed=0,
)
sace.model_info['sace_lower'], sace.model_info['sace_upper']
sace.ci    # Imbens-Manski-style union confidence interval
```

Returns a `CausalResult` with the midpoint of the sharp bounds as
the point estimate and the bound endpoints in `model_info`.
`pvalue` is `NaN` because the estimand is partially identified and
a point-null Wald test is not well-defined.

---

## Two-layer IV / encouragement design

The current release **does not** support passing both an
instrument *and* a take-up variable — that two-layer setup requires
a separate estimator that combines `IV + principal strata`. Use
`sp.dml(model='iivm')` for the classical AIR LATE in an
encouragement design. An explicit `instrument=` argument to
`sp.principal_strat` raises `NotImplementedError`.

---

## References

- Frangakis, C.E. & Rubin, D.B. (2002). Principal stratification in
  causal inference. *Biometrics*.
- Zhang, J.L. & Rubin, D.B. (2003). Estimation of causal effects
  via principal stratification when some outcomes are truncated by
  "death". *JEBS*.
- Angrist, J.D., Imbens, G.W. & Rubin, D.B. (1996). Identification
  of causal effects using instrumental variables. *JASA*.
- Ding, P. & Lu, J. (2017). Principal stratification analysis using
  principal scores. *JRSS-B*.
- Jo, B. & Stuart, E.A. (2009). On the use of propensity scores in
  principal causal effect estimation. *Stat. Med.*
