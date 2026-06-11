# Mediation & causal pathways

A treatment effect estimate answers "does D move Y?". Mediation analysis
answers the follow-up question every referee asks next: "**through what
channel?**" StatsPAI ships a family of pathway estimators — natural
effects, interventional effects, four-way decomposition, front-door
adjustment, and the Gelbach regression decomposition — all behind
`import statspai as sp`. This guide tells you which one answers your
question, what each one assumes, and how badly things go wrong when the
assumptions fail.

## 0. TL;DR flowchart

```
Do you want a CAUSAL pathway decomposition (direct vs indirect)?
  NO, I just want to know which controls move my coefficient
      -> sp.gelbach (regression accounting, NOT causal mediation)
  YES -> Is the treatment-outcome relationship unconfounded
         (conditional on observed pre-treatment X)?
      NO, there is an unobserved D-Y confounder
          -> Does a mediator fully transmit D's effect, with no
             unobserved M-Y confounding? (front-door criterion)
              YES -> sp.front_door
              NO  -> mediation is not identified; see sp.proximal / sp.iv
      YES -> Is any mediator-outcome confounder itself AFFECTED by D?
              YES -> sp.mediate_interventional (natural effects are
                     not identified here)
              NO  -> sp.mediate  (+ ALWAYS sp.mediate_sensitivity)
                     Treatment-mediator interaction suspected?
                       -> sp.four_way_decomposition
```

## 1. The estimator map

| Function | Estimand | One-line description |
|---|---|---|
| `sp.mediate` | ACME / ADE (natural effects) | Imai–Keele–Tingley product-of-coefficients with bootstrap or delta inference [@imai2010general] |
| `sp.mediate_sensitivity` | ACME(ρ) | How much unobserved M–Y confounding kills the ACME [@imai2010identification] |
| `sp.mediate_interventional` | IIE / IDE | Interventional effects — survive treatment-induced mediator-outcome confounders [@vanderweele2014effect] |
| `sp.four_way_decomposition` | CDE + INT_ref + INT_med + PIE | Unifies mediation and interaction [@vanderweele2014unification] |
| `sp.mediation_decompose` | NDE / NIE | Linear nested-models decomposition with interaction term (decomposition family) |
| `sp.front_door` | ATE | Pearl's front-door formula — identification despite unobserved D–Y confounding [@pearl1995causal] |
| `sp.gelbach` | Coefficient-change accounting | "Which added controls explain the change in my coefficient?" [@gelbach2016covariates] |

Article-friendly aliases: `sp.mediation(df, y, d, m, X)` forwards to
`sp.mediate`, and `sp.frontdoor(df, y, d, m, X)` forwards to
`sp.front_door`, mapping `(d, m, X)` to `(treat, mediator, covariates)`.

**Terminology.** `sp.mediate` reports the **ACME** (average causal
mediation effect — the indirect path D → M → Y) and the **ADE** (average
direct effect — D → Y not through M), following Imai–Keele–Tingley
[@imai2010general]. The epidemiology literature calls the same two
quantities **NIE** (natural indirect effect) and **NDE** (natural direct
effect); `sp.mediation_decompose` uses those labels. They decompose the
total effect additively:

```
Total effect = ACME + ADE        (equivalently  TE = NIE + NDE)
Proportion mediated = ACME / Total
```

## 2. Sequential ignorability — read this before trusting any ACME

`sp.mediate` identifies natural effects under **sequential
ignorability** [@imai2010general; @imai2010identification]. Given
pre-treatment covariates X:

1. **Treatment ignorability**: `{Y(t', m), M(t)} ⊥ T | X` — no
   unobserved confounding of the treatment with either the mediator or
   the outcome. A randomized treatment buys you this part.
2. **Mediator ignorability**: `Y(t', m) ⊥ M(t) | T = t, X` — conditional
   on treatment and covariates, the mediator is as-good-as-randomized
   with respect to the outcome.

Part 2 is the killer. **Randomizing the treatment does not deliver it**
— the mediator is a post-treatment variable that units "choose", so any
unobserved variable affecting both M and Y (ability, health, motivation,
firm quality…) biases the ACME and ADE in either direction. Part 2 is
also **untestable from data**: no diagnostic, placebo, or pre-trend
check can detect a violation. Two further consequences:

- **You cannot fix part 2 by controlling for more variables** if any
  M–Y confounder is itself affected by the treatment. Conditioning on a
  treatment-induced confounder blocks part of the effect and opens
  collider paths; natural effects are simply **not identified** in that
  case. Use `sp.mediate_interventional` (§4) instead.
- Because the assumption is untestable, an ACME should never travel
  alone. Report `sp.mediate_sensitivity` (§7) with every mediation
  table, the same way you would report a pre-trend test with a DID.

Failure modes to watch for (cf. `sp.describe_function('mediate')`,
which tracks the same symptoms in the registry):

| Symptom | Remedy |
|---|---|
| Treatment-mediator interaction (decomposition no longer additive) | `sp.mediate_interventional` or `sp.four_way_decomposition` |
| Unknown sensitivity to unobserved M–Y confounder | Always report `sp.mediate_sensitivity` |
| Post-treatment confounder L of the M–Y relationship | `sp.mediate_interventional(..., tv_confounders=['l'])` |

Note: for the last row the registry entry currently suggests
`sp.four_way_decomposition`; this guide deliberately upgrades that to
`sp.mediate_interventional`, the estimator actually designed for
treatment-induced mediator–outcome confounders (§4).

## 3. Worked example: recovering a known direct/indirect split

Simulate a world where we *know* the answer: the effect of D on M is
1.0, the effect of M on Y is 1.5 (so true ACME = 1.5), and the direct
effect is 0.8 (true total = 2.3).

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(0)
n = 2000
x = rng.normal(size=n)                      # pre-treatment confounder
d = (rng.normal(size=n) + 0.5 * x > 0).astype(float)
m = 1.0 * d + 0.5 * x + rng.normal(size=n)            # D -> M (1.0)
y = 0.8 * d + 1.5 * m + 0.3 * x + rng.normal(size=n)  # direct 0.8, M -> Y 1.5
df = pd.DataFrame({"y": y, "d": d, "m": m, "x": x})

# Truth: ACME = 1.0 * 1.5 = 1.5, ADE = 0.8, total = 2.3
r = sp.mediate(df, y="y", treat="d", mediator="m",
               covariates=["x"], n_boot=500, seed=42)
print(r.detail.round(3))
#             effect  estimate     se  ci_lower  ci_upper  pvalue
# 0  ACME (indirect)     1.469  0.076     1.312     1.613   0.002
# 1     ADE (direct)     0.776  0.054     0.665     0.884   0.002
# 2     Total Effect     2.245  0.084     2.075     2.411   0.002
# 3   Prop. Mediated     0.654    NaN       NaN       NaN     NaN
```

All three estimates land on their true values (1.5 / 0.8 / 2.3) within
sampling error. How to read the result object:

```python
print(r.summary())                      # full formatted table
print(r.model_info["acme"])             # 1.469
print(r.model_info["ade"])              # 0.776
print(r.model_info["total_effect"])     # 2.245
print(r.model_info["prop_mediated"])    # 0.654  (= ACME / total)
print(r.model_info["ci_acme"])          # bootstrap percentile CI
print(r.model_info["n_boot_failed"])    # bootstrap health check
```

- `r.estimate` / `r.se` / `r.ci` / `r.pvalue` refer to the **ACME** (the
  headline estimand); the ADE and total live in `r.detail` and
  `r.model_info`.
- **Proportion mediated** is a ratio of estimates — it is reported
  without a SE and explodes when the total effect is near zero. Treat it
  as descriptive, not inferential.
- `n_boot_failed` / `boot_failure_rate` tell you how many bootstrap
  replicates failed to fit; a `RuntimeWarning` fires above 10%.

### Inference options

```python
# Delta-method (Sobel) SEs for the linear no-interaction model —
# matches Stata paramed's convention:
r_delta = sp.mediate(df, y="y", treat="d", mediator="m",
                     covariates=["x"], inference="delta")

# Wald p-values (2*(1-Phi(|z|))) instead of the default bootstrap
# sign-inversion p-value — aligns reporting with sp.aipw / sp.dml:
r_wald = sp.mediate(df, y="y", treat="d", mediator="m",
                    covariates=["x"], n_boot=500,
                    pvalue_method="wald", seed=42)
```

The default `inference='bootstrap'` follows the simulation path of R's
`mediation::mediate(..., boot=TRUE)`; `inference='delta'` is the
closed-form Sobel/delta method. The underlying point estimator is the
product method on two OLS models (`M ~ T + X`, `Y ~ T + M + X`) — if
you suspect a treatment-mediator interaction, the no-interaction model
is wrong and you should move to §5.

## 4. Interventional effects: when a confounder is caused by treatment

Suppose treatment D affects an intermediate variable L that confounds
the M–Y relationship (e.g., job training → employment status →
both skills and wages):

```
┌───────────┐
│           ▼
D ──► L ──► M ──► Y
│     └──────────►│
└─────────────────►
```

Natural effects are **not identified** here — no covariate strategy
works, because conditioning on L blocks part of the effect and opens
collider bias, while ignoring it leaves M–Y confounded.
**Interventional effects** [@vanderweele2014effect] fix counterfactual
mediator *distributions* instead of individual counterfactual values,
dropping the cross-world independence requirement:

- **IIE** (interventional indirect effect): shift M's post-treatment
  marginal distribution from its D=0 draw to its D=1 draw, holding D=1.
- **IDE** (interventional direct effect): change D from 0 to 1 while
  drawing M from its D=0 marginal.

```python
rng = np.random.default_rng(2)
n = 2000
x = rng.normal(size=n)
d = rng.binomial(1, 0.5, size=n).astype(float)
l = 0.7 * d + rng.normal(size=n)          # treatment-induced confounder
m = 0.8 * d + 0.5 * l + 0.4 * x + rng.normal(size=n)
y = 0.5 * d + 1.0 * m + 0.6 * l + 0.3 * x + rng.normal(size=n)
df2 = pd.DataFrame({"y": y, "d": d, "m": m, "l": l, "x": x})

# Natural-effects estimator, unaware of L: biased.
# (true effect of D on M's marginal = 0.8 + 0.7*0.5 = 1.15; times
#  M->Y = 1.0 gives a true indirect channel of 1.15)
r_nat = sp.mediate(df2, y="y", treat="d", mediator="m",
                   covariates=["x"], n_boot=300, seed=42)
print(round(r_nat.model_info["acme"], 3))   # 1.45  <- overstated

# Interventional estimator, L declared as tv_confounder: recovers it.
ri = sp.mediate_interventional(
    df2, y="y", treat="d", mediator="m",
    covariates=["x"], tv_confounders=["l"],
    n_mc=300, n_boot=200, seed=42,
)
print(ri.detail.round(3))
#                           effect  estimate     se  ci_lower  ci_upper  pvalue
# 0  IIE (interventional indirect)     1.171  0.059     1.065     1.290   0.005
# 1    IDE (interventional direct)     0.553  0.054     0.461     0.667   0.005
# 2                          Total     1.724  0.069     1.591     1.860   0.005
```

The IIE (1.17) is back at the truth (1.15) while the naive natural ACME
(1.45) absorbed the L-confounding.

**When to prefer interventional effects:**

- Any mediator-outcome confounder is plausibly *downstream* of
  treatment (very common with social/biomedical mediators measured
  after baseline).
- You care about the policy question "what if we shifted the mediator's
  distribution?" rather than the individual cross-world counterfactual.

**Honest caveats:**

- The current implementation hard-codes linear (OLS) mediator and
  outcome models with a Gaussian mediator draw; custom learners are not
  exposed.
- `tv_confounders` enter the outcome model as controls held at their
  observed distribution in **both** arms, so the pathway D → L → Y
  (not via M) is partialled out of the reported IDE and Total. The
  reported `Total = IIE + IDE` is therefore the effect transmitted
  through D and M only — it need not equal the overall ATE of D on Y
  when L itself carries part of the effect. Compare against
  `sp.regress('y ~ d', data=df2)` or `sp.g_computation` if you need the
  full total effect.

## 5. Mediation meets interaction: the four-way decomposition

When the effect of M on Y differs by treatment arm (a D×M interaction),
"direct vs indirect" is no longer a clean binary. VanderWeele's
four-way decomposition [@vanderweele2014unification] splits the total
effect into components attributable to *neither* mediation nor
interaction (CDE), *interaction only* (INT_ref), *both* (INT_med), and
*mediation only* (PIE):

```python
rng = np.random.default_rng(5)
n = 2000
x = rng.normal(size=n)
d = rng.binomial(1, 0.5, size=n).astype(float)
m = 0.9 * d + 0.4 * x + rng.normal(size=n)
# interaction: effect of M on Y differs by D
y = 0.5 * d + 0.7 * m + 0.5 * d * m + 0.3 * x + rng.normal(size=n)
df3 = pd.DataFrame({"y": y, "d": d, "m": m, "x": x})

fw = sp.four_way_decomposition(df3, y="y", treat="d", mediator="m",
                               covariates=["x"])
print(fw.summary())
# Total Effect    : +1.4931
# CDE(0)          : +0.4889  (32.7%)
# INT_ref         : +0.0202  (1.4%)
# INT_med         : +0.4072  (27.3%)
# PIE             : +0.5768  (38.6%)

# NDE/NIE-labelled cousin with the interaction term in the
# outcome model (decomposition family):
md = sp.mediation_decompose(df3, y="y", treatment="d", mediator="m",
                            covariates=["x"], inference="bootstrap",
                            n_boot=199)
print(round(md.nde, 3), round(md.nie, 3), round(md.propn_mediated, 3))
# 0.5  0.984  0.663
```

If `sp.mediate`'s ACME + ADE visibly fail to reproduce the total effect
you get from a direct regression, an interaction is the usual culprit —
switch to one of these two.

## 6. Front-door adjustment: identification despite unobserved confounding

Everything above assumed the D–Y relationship is unconfounded given
observables. The **front-door criterion** [@pearl1995causal;
@pearl2009causality] is the one classical case where a mediator
*rescues* you when that fails. Assumed DAG:

```
U ──┬──► D ──► M ──► Y
    │                ▲
    └────────────────┘        (U unobserved)
```

If (1) M fully transmits D's effect (no direct D → Y edge), (2) there
is no unobserved M–Y confounder (D blocks the back door into M), and
(3) P(M | D) has full support, then the ATE is identified by Pearl's
front-door formula even though U makes back-door adjustment impossible.

Check the criterion on your assumed DAG first — `<->` denotes an
unobserved common cause:

```python
g = sp.dag("D -> M; M -> Y; D <-> Y")
print(g.adjustment_sets("D", "Y"))   # []      <- back door is hopeless
print(g.frontdoor_sets("D", "Y"))    # [{'M'}] <- front door is open
```

Then estimate. With an unobserved U driving both D and Y, naive OLS is
badly biased while the front-door estimator recovers the truth:

```python
rng = np.random.default_rng(1)
n = 3000
u = rng.normal(size=n)                                # UNOBSERVED
d = (0.8 * u + rng.normal(size=n) > 0).astype(float)  # U -> D
m = 1.2 * d + rng.normal(size=n)                      # D -> M (no U arrow)
y = 1.0 * m + 1.5 * u + rng.normal(size=n)            # M -> Y, U -> Y
df4 = pd.DataFrame({"y": y, "d": d, "m": m})
# True ATE of D on Y: 1.2 * 1.0 = 1.2 (all routed through M)

naive = sp.regress("y ~ d", data=df4)
print(round(float(naive.params["d"]), 3))             # 2.751 <- garbage

fd = sp.front_door(df4, y="y", treat="d", mediator="m",
                   n_boot=200, n_mc=100, seed=42)
print(round(fd.estimate, 3), round(fd.se, 3))         # 1.185 0.049
```

Implementation notes:

- Binary D only (a `ValueError` tells you what to use instead for a
  continuous dose). M can be binary (closed-form sums) or continuous
  (Gaussian mediator model + Monte Carlo integration, `n_mc` draws).
- `integrate_by='marginal'` (default) follows Pearl's aggregate
  formulation; `'conditional'` follows the Fulcher et al. generalized
  front-door [@fulcher2020robust] with unit-specific M | D, X draws.
  The two coincide for binary M or when `covariates` is empty —
  `fd.model_info['integrate_by_effective']` records what actually ran.
- `covariates` must be **pre-treatment**; post-treatment controls
  re-open the back door. If the covariate-adjusted mediator model fails
  and silently degrades to the unadjusted marginal, a `RuntimeWarning`
  fires and `fd.model_info['mediator_model_degraded']` is set.

**The brutal honesty paragraph.** Assumptions (1) and (2) are pure DAG
assumptions — *nothing in the data can confirm them*. A small direct
D → Y leak or a lurking M–Y confounder propagates straight into the
estimate. The front-door design is most credible when the mediator is a
tightly-coupled mechanical channel (a dosage delivered, a document
processed, a pipeline stage) rather than a behavioral choice.

## 7. Sensitivity analysis: never ship an ACME without it

Because sequential ignorability part 2 is untestable, the Imai–Keele–
Yamamoto sensitivity analysis [@imai2010identification] asks: *how
correlated would the mediator-model and outcome-model errors have to be
(ρ ≠ 0 means an unobserved M–Y confounder) before the ACME crosses
zero?*

```python
rng = np.random.default_rng(3)
n = 2000
x = rng.normal(size=n)
d = rng.binomial(1, 0.5, size=n).astype(float)
m = 1.0 * d + 0.5 * x + rng.normal(size=n)
y = 0.4 * d + 0.6 * m + 0.3 * x + rng.normal(scale=2.0, size=n)
df5 = pd.DataFrame({"y": y, "d": d, "m": m, "x": x})

sens = sp.mediate_sensitivity(df5, y="y", treat="d", mediator="m",
                              covariates=["x"])
print(sens.summary())
# Baseline ACME (ρ=0) :  0.5075
# ρ at which ACME = 0  :  0.2585
# Interpretation: unobserved confounding with |ρ| > 0.26 would
# explain away the estimated mediation effect.

sens.plot()          # ACME(rho) curve with the robustness threshold
print(sens.rho_at_zero, sens.acme_at_zero)
```

Reading it: `rho_at_zero` is the mediation analogue of an E-value-style
breakdown point. A threshold near 0 means a whisper of unobserved M–Y
confounding flips your conclusion; if the curve never crosses zero on
[-0.9, 0.9], `rho_at_zero` is `None` and the ACME's sign is robust to
any single confounder of that form. There is no magic cutoff — report
the number and let readers compare ρ against the residual correlations
that observed covariates produce.

For the treatment-outcome (rather than mediator-outcome) confounding
axis, the generic tools in the
[robustness workflow guide](robustness_workflow.md) apply unchanged:
`sp.evalue_from_result`, `sp.oster_bounds`, `sp.sensemakr`.

## 8. Gelbach decomposition: the regression-accounting cousin

`sp.gelbach` [@gelbach2016covariates] answers a *different* question:
**"my coefficient on D moved when I added controls — which added
variable is responsible, and by how much?"** It is a conditional-
correlation accounting identity, not a causal pathway estimator:

```
beta_base - beta_full = sum_j  gamma_j * beta_full_j
```

where `gamma_j` comes from regressing added variable j on the base
specification. Its selling point over the informal "add controls one
group at a time and narrate the coefficient" ritual is that the
decomposition is **order-invariant** — every added variable is
evaluated against the *full* model, so the answer doesn't depend on the
sequence in which you happen to add columns.

```python
rng = np.random.default_rng(4)
n = 2000
educ = rng.normal(12, 2, size=n)
exper = 20 - 0.6 * educ + rng.normal(size=n)
union = (0.2 * educ + rng.normal(size=n) > 2.5).astype(float)
wage = 1.0 * educ + 0.3 * exper + 0.8 * union + rng.normal(size=n)
df6 = pd.DataFrame({"wage": wage, "educ": educ,
                    "exper": exper, "union": union})

g = sp.gelbach(df6, y="wage", base_x=["educ"],
               added_x=["exper", "union"])
g.summary()
#   Base coefficient on 'educ':  0.8843
#   Full coefficient on 'educ':  0.9975
#   Total change (base - full):  -0.1133
#   exper:  -0.1735***  (153.1% of the change)
#   union:  +0.0602***  (-53.1% of the change)
print(g.decomposition[["variable", "delta", "se"]].round(4))
```

Here `exper` and `union` push the education coefficient in *opposite*
directions — exactly the pattern that sequential one-at-a-time
narration gets wrong, and the kind of result `sp.gelbach` exists to
surface.

**Common misuse — do not confuse this with causal mediation:**

- A Gelbach `delta` is **not** an ACME. It tells you how much of a
  *coefficient change* is attributable to a control's correlation
  structure, under no causal assumptions about the control. Calling
  `delta_j / total_change` a "proportion mediated by j" smuggles in
  sequential ignorability without ever stating it.
- If an added variable is post-treatment (a mediator or collider of
  the variable of interest), the *full* model coefficient is itself a
  bad-control estimate — the decomposition faithfully accounts for a
  movement between two numbers, neither of which is the causal effect.
- Legitimate uses: omitted-variable-bias narratives ("the raw gap
  shrinks by X% once skills are added, and 80% of that shrinkage is the
  test-score block"), robustness sections, and Oaxaca-style descriptive
  work alongside `sp.oaxaca` [@oaxaca1973male].

Rule of thumb: `sp.gelbach` decomposes **estimates**; `sp.mediate`
decomposes **effects**. Pick by the noun in your research question.

## 9. Stata / R command mapping

| Task | Stata | R | StatsPAI |
|---|---|---|---|
| Natural effects, simulation/bootstrap inference | `medeff` (user-written `mediation` package) | `mediation::mediate(med.fit, out.fit, boot = TRUE)` | `sp.mediate(df, y, treat, mediator, covariates, inference='bootstrap')` |
| Natural effects, delta-method (Sobel) SEs | `paramed` | `mediation::mediate(..., boot = FALSE)` | `sp.mediate(..., inference='delta')` |
| Product-of-coefficients via SEM | `sem (y <- d m x) (m <- d x)` + `estat teffects` | `lavaan` with `a*b` defined parameter | `sp.mediate(..., inference='delta')` |
| Sensitivity to M–Y confounding (ρ) | — | `mediation::medsens` | `sp.mediate_sensitivity` |
| Interventional effects with treatment-induced confounder | — | — | `sp.mediate_interventional(..., tv_confounders=[...])` |
| Four-way mediation/interaction decomposition | `med4way` (user-written) | — | `sp.four_way_decomposition` |
| Front-door adjustment | — | — | `sp.front_door` |
| Coefficient-change decomposition | `b1x2` (Gelbach's command) | — | `sp.gelbach` |

Default conventions worth knowing when matching numbers across stacks:
`sp.mediate`'s bootstrap keeps the simulation path of R
`mediation::mediate(..., boot=TRUE)`, while `inference='delta'` matches
the linear no-interaction convention of Stata `paramed`. P-values
default to the bootstrap sign-inversion rule; pass
`pvalue_method='wald'` for the Wald convention used by `sp.aipw` /
`sp.dml`.

## 10. Reporting checklist

For a publication-quality mediation table, present:

1. Total effect, ADE, ACME (or IDE/IIE) with CIs and N — `r.detail`.
2. The identification claim, in words: which covariates make sequential
   ignorability (or the front-door criterion) plausible, and why no
   M–Y confounder is downstream of treatment.
3. The sensitivity curve: `sp.mediate_sensitivity(...).plot()` and the
   ρ-at-zero breakdown value.
4. An interaction check: run `sp.four_way_decomposition` — if INT_med
   is non-trivial, the two-component ACME/ADE decomposition is
   incomplete.
5. Citations from the result object itself: `r.cite()` emits verified
   BibTeX for the method actually used.

## References

- Natural effects & quasi-Bayesian inference: [@imai2010general]
- Sensitivity analysis (ρ): [@imai2010identification]
- Interventional effects under treatment-induced confounding:
  [@vanderweele2014effect]
- Four-way decomposition: [@vanderweele2014unification]
- Front-door criterion: [@pearl1995causal; @pearl2009causality];
  generalized front-door: [@fulcher2020robust]
- Coefficient-change decomposition: [@gelbach2016covariates]
- Classical stepwise mediation (historical baseline, superseded by the
  potential-outcomes framework above): [@baron1986moderator]
