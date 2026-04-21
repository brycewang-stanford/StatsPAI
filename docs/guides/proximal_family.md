# Proximal causal inference — the full family

> **When you cannot observe the confounder**, Proximal Causal Inference (PCI) uses
> two *proxies* of the unobserved confounder to identify the ATE.  StatsPAI
> ships the base estimator plus four 2025-2026 frontier variants that are
> robust to bridge-function misspecification, policy-level interventions,
> bidirectional confounding, and short→long-term extrapolation.

This guide walks through the whole PCI family with a single running example
and tells you which variant to reach for in which situation.  Every function
below lives at top level: `sp.proximal`, `sp.fortified_pci`,
`sp.bidirectional_pci`, `sp.pci_mtp`, `sp.double_negative_control`,
`sp.proximal_surrogate_index`, `sp.select_pci_proxies`.

---

## The setup

You have outcome `Y`, treatment `D`, observed covariates `X`, and an
**unobserved** confounder `U`.  Back-door adjustment on `X` does not
identify the ATE because `U → D` and `U → Y` after conditioning on `X`.

Proximal identification (Miao-Geng-Tchetgen, 2018; Tchetgen et al., 2024)
works if you can find two proxies of `U`:

- **`Z`** (treatment-inducing confounding proxy) — depends on `U`, possibly
  on `D`; conditional on `(U, D, X)`, `Z` is independent of `Y`.
- **`W`** (outcome-inducing confounding proxy) — depends on `U`, possibly
  on `Y`; conditional on `(U, X)`, `W` is independent of `D`.

Plus: a *completeness* condition linking `(D, Z)` to `W` (roughly, `W`
carries enough information about `U` to predict it nonparametrically given
`(D, X)`).

Choosing proxies is the hard part — see `sp.select_pci_proxies` below.

---

## Canonical estimator: `sp.proximal`

Two-stage-least-squares on the outcome bridge.  Good default when bridges
are plausibly linear and you have one `Z` and one `W`.

```python
r = sp.proximal(
    data=df, y="Y", treat="D",
    proxy_z=["Z"], proxy_w=["W"],
    covariates=["X1", "X2"],
    bridge="linear",     # or "loglinear"
    n_boot=500,
)
r.summary()
```

Reference: Miao, Geng, Tchetgen Tchetgen (2018), *Biometrika* 105(4); Cui,
Pu, Miao, Kennedy, Tchetgen (2024), *JASA*.

---

## Fortified PCI — `sp.fortified_pci` (Yang & Schwartz, 2025)

Standard PCI fails hard when either bridge is misspecified.  **Fortified PCI**
adds a stability constraint: at each proxy choice, we fit both bridges with a
shared penalty that drives the solution toward an *overlap region* where both
moments are jointly satisfied.  Empirically this is the PCI variant most
robust to moderate bridge-function misspecification (Yang-Schwartz 2025
Tables 2-4).

```python
r = sp.fortified_pci(
    data=df, y="Y", treat="D",
    proxy_z=["Z"], proxy_w=["W"],
    covariates=["X1", "X2"],
)
```

**When to reach for it**: your point estimate from `sp.proximal` moves
noticeably when you add/remove a single covariate, or when `bridge="linear"`
vs `bridge="loglinear"` give very different answers.

Citation: Yang & Schwartz (2025), arXiv:2506.13152.

---

## Bidirectional PCI — `sp.bidirectional_pci` (Shi, Miao & Tchetgen, 2025)

Standard PCI solves the **outcome bridge** (regress `Y` on `D, W, X`) and
then inverts via `Z`.  Bidirectional PCI fits the outcome bridge *and* the
treatment bridge *simultaneously* in a single two-way regression, which:

1. Reduces finite-sample bias when either bridge is weakly identified.
2. Produces a natural doubly-robust GMM-style objective.
3. Makes the identification assumptions symmetric — useful when you cannot
   decide which proxy is `Z` and which is `W`.

```python
r = sp.bidirectional_pci(
    data=df, y="Y", treat="D",
    proxy_z=["Z"], proxy_w=["W"],
    covariates=["X1", "X2"],
)
```

Citation: Shi, Miao, Tchetgen Tchetgen (2025), arXiv:2507.13965.

---

## PCI for modified treatment policies — `sp.pci_mtp` (Park & Ying, 2025)

Suppose you do not want the ATE under a static "`D=1` vs `D=0`" contrast;
you want the effect of *shifting* the treatment distribution by some
amount `δ` — e.g., "raise the dose by 10% for everyone."  PCI-MTP
identifies that **modified treatment policy** effect under the same two-
proxy structure as base PCI.

```python
r = sp.pci_mtp(
    data=df, y="Y", treat="dose",
    proxy_z=["Z"], proxy_w=["W"],
    delta=0.10,                     # ← shift treatment by +10%
    covariates=["X1", "X2"],
)
```

**Why it matters**: in dose-response or continuous-treatment settings, the
"contrast between fixed levels" estimand is often uninteresting.  MTP
answers the policy-relevant question "what happens if we nudge treatment
a bit?" under unobserved confounding.

Citation: Park & Ying (2025), arXiv:2512.12038.

---

## Double negative controls — `sp.double_negative_control`

A simpler PCI special case used in epidemiology: one *negative control
exposure* (NCE, a treatment-like variable we know cannot affect `Y`) plus
one *negative control outcome* (NCO, an outcome-like variable we know
cannot be affected by `D`).  These together identify the treatment effect
under additive structural assumptions.

```python
r = sp.double_negative_control(
    data=df, y="Y", treat="D",
    nce="prior_dental_visits",       # placebo treatment
    nco="baseline_weight",           # placebo outcome
    covariates=["age", "sex"],
)
```

**Use case**: lower-tech than full PCI — good for initial analyses when
you have only one of each type of proxy.  References:
`sp.negative_control_outcome` and `sp.negative_control_exposure` for the
single-control versions.

---

## Long-term surrogate + PCI — `sp.proximal_surrogate_index` (Kallus-Mao, 2026)

You ran a randomised experiment for 3 months but care about the 2-year
outcome.  The classical solution (Athey-Chetty-Imbens 2020 surrogate index)
needs the surrogates to fully mediate the long-term effect — a strong
assumption.  Kallus-Mao show that combining the surrogate index with a PCI
layer on the **observational** data lets you drop the full-mediation
requirement: short-term surrogates play the role of `W`, observational
proxies play the role of `Z`, and the two together identify the long-term
ATE.

```python
r = sp.proximal_surrogate_index(
    experimental=df_exp,           # short-term RCT
    observational=df_obs,          # long-follow-up observational cohort
    treatment="feature_flag",
    surrogates=["dau_90d", "retention_90d"],   # → W
    proxies=["pre_dau", "pre_purchase"],        # → Z
    long_term_outcome="revenue_24mo",
    covariates=["country", "cohort"],
)
```

This is also available as the `surrogate_pci` bridge in `sp.bridge` — see
[Bridging theorems](bridging_theorems.md).

Citation: Kallus & Mao (2026), arXiv:2601.17712.

---

## Picking the proxies: `sp.select_pci_proxies`

When you have a list of *candidate* proxy variables, this helper scores
each on two PCI-relevant axes:

1. **`Z` score** — how strongly is the candidate predicted by `D` after
   partialling out `X`?  (Want this high for a good `Z`.)
2. **`W` score** — how strongly does the candidate predict `Y` after
   partialling out `D, X`?  (Want this high for a good `W`.)

It returns a ranked table of candidates so you can choose the top-scorers
for each role.

```python
ranks = sp.select_pci_proxies(
    data=df, y="Y", treat="D",
    candidates=["V1", "V2", "V3", "V4", "V5"],
    covariates=["X1", "X2"],
    top_k=2,
)
print(ranks.z_table)   # best Z candidates
print(ranks.w_table)   # best W candidates
```

---

## When to use which — decision guide

```
Got exactly one Z and one W, bridges look roughly linear
  → sp.proximal (default)

Bridges sensitive to specification choice
  → sp.fortified_pci

Unsure which proxy should play Z vs W
  → sp.bidirectional_pci

Continuous treatment / want "shift policy" effect
  → sp.pci_mtp(delta=...)

Only have a negative control exposure + negative control outcome
  → sp.double_negative_control

Want long-term effect from short-term experiment
  → sp.proximal_surrogate_index

Have a pile of candidate proxies, not sure which to use
  → sp.select_pci_proxies first, then the above
```

---

## Diagnostics every PCI analysis should report

1. **Bridge completeness**: can `Z` predict `W`? (If not, identification
   fails.)  Run
   `sp.regress("W ~ Z + D + X", data=df)` and check the F-stat.
2. **Proxy independence**: `Z ⊥ Y | D, U, X`.  You cannot test this
   directly (because `U` is unobserved), but you can test the weaker
   `Z ⊥ Y | D, X` and interpret a non-zero coefficient as suggestive of
   `U`-mediated dependence.
3. **Bridge stability**: rerun with `bridge="linear"` and `bridge="loglinear"`
   — substantial disagreement suggests misspecification.
4. **Sensitivity via `sp.bridge(kind="cb_ipw", ...)`** on the `(D, X)` pair —
   if the back-door-on-`X`-only estimate equals the PCI estimate, your
   unobserved confounder might not be doing much work in this sample.

---

*This guide is current for StatsPAI ≥ 1.4.2.  All functions are stable
and registered in `sp.list_functions()`.*
