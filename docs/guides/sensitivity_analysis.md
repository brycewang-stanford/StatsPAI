# Sensitivity analysis: which tool for which design

A point estimate with a small standard error answers exactly one
question: *given my identifying assumptions, what is the effect?* It
says nothing about the question every reviewer actually asks: *what if
the identifying assumptions are wrong — and by how much can they be
wrong before the conclusion dies?*

Sensitivity analysis answers that second question. StatsPAI ships the
major tools — `sp.sensemakr`, `sp.oster_bounds`, `sp.evalue`,
`sp.rosenbaum_bounds`, `sp.honest_did`, `sp.sensitivity_rr`,
`sp.weakrobust`, `sp.dml_sensitivity`, `sp.manski_bounds`,
`sp.lee_bounds` — but each one is tied to a specific research design
and a specific threatened assumption. Running the wrong one is not
conservative, it is meaningless: an E-value cannot rescue a DiD with
broken parallel trends, and `sp.honest_did` says nothing about omitted
variables in a cross-sectional regression. This guide is the decision
tree.

## Robustness checks vs. sensitivity analysis

The two are routinely conflated. They are different layers of the same
defence (see the [robustness workflow guide](robustness_workflow.md)):

| | Robustness check | Sensitivity analysis |
| --- | --- | --- |
| Question | Does the estimate survive *alternative specifications*? | How badly can an *untestable assumption* fail before the conclusion flips? |
| Varies | Things you can observe: covariate sets, bandwidths, subsamples, estimators | Things you cannot observe: unmeasured confounding, parallel-trends violations, selection |
| Output | A set of alternative point estimates | A breakdown value: "the conclusion survives violations up to *this* magnitude" |
| Tooling | `sp.spec_curve`, `sp.robustness_report`, `sp.rdbwsensitivity` | Everything in this guide |

A spec curve with 40 stable estimates is worthless against unmeasured
confounding — every one of those 40 specifications conditions on the
same observed covariates. Sensitivity analysis is the only honest
answer to "what about the variable you didn't measure?", and its
output is not a yes/no verdict but a *breakdown magnitude* that the
reader judges against domain knowledge.

## The decision tree

Start from your design, not from the tool:

```text
What is your design, and which assumption is under attack?

1. Selection on observables (regression / matching / weighting / AIPW)
   threat: an unmeasured confounder U
   ├── You fit an OLS-style regression and can name benchmark covariates
   │     -> sp.sensemakr          (partial-R^2 robustness value, Cinelli-Hazlett)
   ├── You want "how much would unobservables need to matter relative
   │   to the observables I added?"
   │     -> sp.oster_bounds       (coefficient-stability delta*, Oster)
   ├── You want a scale-free single number any referee understands
   │     -> sp.evalue / sp.evalue_from_result   (VanderWeele-Ding)
   └── You ran pair-matching (sp.psm / sp.match)
         -> sp.rosenbaum_bounds   (hidden-bias Gamma)

2. Difference-in-differences / event study
   threat: parallel trends fails post-treatment
   ├── ALWAYS pair the pre-trend test with its power:
   │     sp.pretrends_test + sp.pretrends_power   (Roth)
   └── Honest partial identification under bounded violations:
         sp.honest_did / sp.sensitivity_rr / sp.breakdown_m
         (Rambachan-Roth; see the honest DiD guide)

3. Instrumental variables
   threat: the instrument is weak (relevance, not exclusion)
   └── sp.weakrobust   (AR + CLR + K + effective F in one panel)
       sp.anderson_rubin_test / sp.anderson_rubin_ci for AR alone
       note: exclusion-restriction failure is NOT covered here —
       no statistic can test it; argue it, or bound it (sp.iv_bounds)

4. Double / debiased machine learning
   threat: unmeasured confounder survives the flexible nuisances
   └── sp.dml_sensitivity   (DML-flavoured omitted-variable bounds,
                             Chernozhukov-Cinelli-Newey-Sharma-Syrgkanis)

5. Missing, truncated, or selected outcomes
   threat: who you observe depends on treatment
   ├── Outcome bounded, no assumptions you trust
   │     -> sp.manski_bounds      (worst-case; add 'mtr'/'mts' to tighten)
   ├── Outcome observed only conditional on selection (employment,
   │   survival, attrition) and treatment shifts selection one way
   │     -> sp.lee_bounds         (trimming bounds)
   └── Outcomes/covariates missing not-at-random
         -> sp.horowitz_manski    (worst-case imputation bounds)
```

Quick reference:

| Design | Threatened assumption | Tool | Breakdown quantity |
| --- | --- | --- | --- |
| Regression / SOO | Unconfoundedness | `sp.sensemakr` | Robustness value RV (partial R²) |
| Regression / SOO | Unconfoundedness | `sp.oster_bounds` | δ\* (relative selection) |
| Any SOO estimate | Unconfoundedness | `sp.evalue` | E-value (risk-ratio scale) |
| Pair matching | Hidden bias in matching | `sp.rosenbaum_bounds` | Γ\* (odds of treatment) |
| DiD / event study | Parallel trends | `sp.honest_did`, `sp.sensitivity_rr` | Breakdown M / M̄ |
| DiD pre-test | Power of pre-test | `sp.pretrends_power` | Power against given violation |
| IV | Instrument relevance | `sp.weakrobust` | Effective F, AR/CLR CI |
| DML | Residual confounding | `sp.dml_sensitivity` | RV_q (confounder partial R² with DML residuals) |
| Bounded outcome | Point identification | `sp.manski_bounds` | Identified set |
| Selected outcome | Sample-selection ignorability | `sp.lee_bounds` | Trimming bounds |

The rest of this guide walks each branch with a runnable example and —
more importantly — tells you how to *read* the output.

## Branch 1 — Selection on observables

You estimated an effect from observational data by conditioning on
covariates (OLS, matching, weighting, AIPW). The untestable assumption
is unconfoundedness: no unmeasured variable drives both treatment and
outcome. Four tools, four parameterisations of the same threat.

All four examples below share this simulated dataset, which has a
genuine unobserved confounder `u` baked in:

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(42)
n = 2000
x1 = rng.normal(size=n)            # observed confounder
x2 = rng.normal(size=n)            # observed confounder
u  = rng.normal(size=n)            # UNOBSERVED confounder
d  = (0.5 * x1 + 0.3 * x2 + 0.4 * u + rng.normal(size=n) > 0).astype(int)
y  = 1.0 * d + 0.8 * x1 + 0.5 * x2 + 0.6 * u + rng.normal(size=n)
df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
```

The regression of `y` on `d`, `x1`, `x2` gives a biased estimate of
about 1.32 (truth: 1.00) because `u` is omitted. The sensitivity tools
cannot *detect* this bias — nothing can — but they quantify how strong
`u` would have to be to explain various amounts of it.

### 1a. Sensemakr (Cinelli-Hazlett)

```python
s = sp.sensemakr(df, y="y", treat="d", controls=["x1", "x2"],
                 benchmark=["x1"])
print(s["rv_q"])              # 0.400 — robustness value for the point estimate
print(s["rv_qa"])             # 0.374 — robustness value for significance
print(s["partial_r2_yd"])     # 0.211 — partial R^2 of treatment with outcome
print(s["interpretation"])
```

**How to read it.** `rv_q = 0.40` means: an unobserved confounder
would need to explain **more than 40% of the residual variance of both
the treatment and the outcome** to drive the estimate all the way to
zero. `rv_qa = 0.37` is the (lower) bar for merely wiping out
statistical significance. The `benchmark_table` entry translates this
into something a referee can judge: it shows the partial R² of an
observed covariate (`x1` here) and asks whether a confounder "as
strong as `x1`" — or 2×, 3× as strong — would breach the robustness
value. If the strongest covariate you measured explains 10% and the RV
is 40%, the confounder story needs a variable four times stronger than
anything you saw; that is an argument, not a proof.

### 1b. Oster coefficient-stability bounds

Oster's δ asks the question in movement terms: when you added
controls, how much did the coefficient move and how much did R² rise?
If adding the *observables* barely moved the coefficient while
explaining a lot of variance, the *unobservables* would have to be
disproportionately selected to undo the result.

```python
b = sp.oster_bounds(df, y="y", treat="d", controls=["x1", "x2"],
                    delta=1.0)
print(b["beta_short"])        # 2.00 — no controls
print(b["beta_long"])         # 1.32 — with controls
print(b["delta_for_zero"])    # 1.56 — delta* needed to zero the effect
print(b["identified_set"])    # (0.47, 1.32) under delta=1, R_max=1.3*R2
print(b["interpretation"])
```

**How to read it.** `delta_for_zero = 1.56` means unobservable
confounding would need to be **1.56 times as important as the
observable confounding you already controlled for** to explain the
entire effect. Oster's heuristic treats |δ\*| > 1 as robust — the
observables are usually chosen *because* they are the most important
confounders, so "unobservables matter even more" is a strong claim.
The `identified_set` is where the true effect lies if unobservables
are exactly as important as observables (δ = 1) and the hypothetical
full-controls R² is `r_max`: if 0 falls inside it, the result is
fragile. Note the trap in this example: the set (0.47, 1.32) excludes
zero — "robust" by Oster's standard — yet the truth (1.00) is well
below `beta_long`. Robust-to-zero is not the same as unbiased.

If you only have published regression output (no microdata), pass the
summary statistics directly: `sp.oster_bounds(beta_short=...,
r2_short=..., beta_long=..., r2_long=..., r_max=...)`.

### 1c. E-value (VanderWeele-Ding)

The E-value is the most portable of the four: it needs only the
estimate and its CI, works on risk ratios, odds ratios, hazard ratios,
and (after standardisation) linear coefficients, and is defined
*without* any modelling assumption on the confounder
(Ding-VanderWeele).

```python
# A linear (OLS) effect: pass the coefficient, its SE, and the outcome SD
ev = sp.evalue(estimate=0.6, se=0.12, sd=2.0, measure="OLS")
print(ev["evalue_estimate"])   # 1.96
print(ev["evalue_ci"])         # 1.64
print(ev["interpretation"])

# Ratio-scale estimates are passed directly
ev2 = sp.evalue(estimate=1.8, ci=(1.2, 2.7), measure="OR", rare=False)
print(ev2["evalue_estimate"], ev2["evalue_ci"])   # 2.02 1.42
```

**How to read it.** An E-value of 2.3 means, in plain language: *an
unmeasured confounder would have to be associated with both the
treatment and the outcome by a risk ratio of at least 2.3 each — above
and beyond the measured covariates — to fully explain away the
observed effect; anything weaker could shrink it but not erase it.*
The companion `evalue_ci` applies the same logic to the CI limit
nearest the null — the bar for destroying significance rather than the
point estimate. Calibrate against your data: if the strongest measured
covariate has RR ≈ 1.5 with the outcome, a confounder needing RR ≥ 2.3
on *both* arms is a demanding ask. Rule of thumb from the
[robustness workflow guide](robustness_workflow.md): E > 2 is
reasonably robust, E > 3 strong, E < 1.5 fragile.

For a fitted `CausalResult` (from `sp.dml`, `sp.psm`, `sp.aipw`, …)
use `sp.evalue_from_result(r)` — it standardises the estimate and
converts to the risk-ratio scale for you (shown in Branch 4 below).

### 1d. Rosenbaum bounds (matched designs)

If your design is pair matching, the natural sensitivity parameter is
Γ: how much could hidden bias multiply the *odds of treatment* within
a matched pair? At Γ = 1 matching is as-good-as-random; Γ = 2 means
one unit of an identical-looking pair can be twice as likely treated
because of something you did not match on (Rosenbaum).

```python
import numpy as np
import statspai as sp

rng = np.random.default_rng(7)
n = 200
treated = 0.35 + rng.normal(size=n)   # matched-pair treated outcomes
control = rng.normal(size=n)          # matched-pair control outcomes

rb = sp.rosenbaum_bounds(treated, control)   # default grid: 1.0–3.0, step 0.1
print(rb.summary())
print(rb.gamma_critical)   # 1.3
```

**How to read it.** The table reports, for each Γ, the worst-case
range of p-values consistent with that much hidden bias.
`gamma_critical` is the *smallest Γ at which the worst-case p-value
exceeds α* — that is, the result is already overturned **at** Γ\*, not
beyond it. Here the significant Wilcoxon result survives hidden bias
up to Γ = 1.2 (worst-case p = 0.040) and dies at Γ = 1.3 (worst-case
p = 0.103). Note that the reported Γ\* is grid-resolution-dependent —
a coarse grid like `gamma_grid=[1.0, 1.5, 2.0]` would round the
breakdown up to 1.5 and overstate the robustness — so keep the default
fine grid (or finer) when reporting. The larger Γ\* is, the more
hidden bias the finding tolerates; a Γ\* barely above 1 means even
mild unmatched heterogeneity could account for the entire result.

## Branch 2 — Difference-in-differences

The threatened assumption is parallel trends *after* treatment — which
no pre-trend test can verify, because the test only sees the pre
period. The modern recipe (Rambachan-Roth; Roth) is a three-part
package: pre-trend test, the test's *power*, and honest confidence
intervals under bounded violations. Full background lives in the
[honest DiD guide](honest_did.md); here is the sensitivity-analysis
view.

```python
import numpy as np
import pandas as pd
import statspai as sp

# Staggered-free panel: 30 units treated at t=5, 30 never treated
rng = np.random.default_rng(0)
rows = []
for i in range(60):
    alpha, g = rng.normal(), (5 if i < 30 else 0)
    for t in range(1, 9):
        treated = int(g != 0 and t >= g)
        rows.append((i, t, g, alpha + 0.3 * t + 1.5 * treated
                     + rng.normal(scale=0.5)))
df = pd.DataFrame(rows, columns=["i", "t", "g", "y"])

r = sp.callaway_santanna(df, y="y", g="g", t="t", i="i")

# (1) Pre-trend test ...
pt = sp.pretrends_test(r)
print(pt["pvalue"], pt["reject"])      # 0.358 False

# (2) ... ALWAYS paired with its power
pw = sp.pretrends_power(r)
print(round(pw["power"], 2))           # 0.15
print(pw["warning"])

# (3) Honest CIs under smoothness violations of magnitude M
hd = sp.honest_did(r, m_grid=[0.0, 0.1, 0.2, 0.5])
print(hd)

# Breakdown M in one number
print(sp.breakdown_m(r))               # 1.269

# Relative-magnitudes version (Mbar): violation post <= Mbar x worst pre
sr = sp.sensitivity_rr(r, Mbar=[0.0, 0.5, 1.0])
print(sr)
```

**How to read it.**

- `pretrends_test` not rejecting (p = 0.36) is *necessary, not
  sufficient*. The power calculation is what makes it honest: power
  0.15 means that even if a violation as large as the hypothesised
  slope existed, this test would catch it only 15% of the time — a
  flat pre-trend plot from an underpowered test is close to
  uninformative (Roth). Report both numbers, always.
- `sp.honest_did` re-computes the CI allowing post-treatment trend
  violations up to magnitude `M` per period (smoothness flavour). The
  table's `rejects_zero` column shows where the conclusion survives.
  The **breakdown M** (1.27 here, vs. an ATT of 1.73) is the headline:
  parallel trends can fail by up to 1.27 outcome units per period
  before zero enters the CI. Judge that against the pre-period
  fluctuations you actually saw.
- `sp.sensitivity_rr` parameterises violations relative to the
  observed pre-trends instead: M̄ = 1 allows a post-treatment violation
  as large as the worst pre-treatment one. A breakdown M̄ of 0.5
  (as printed here) reads: *if parallel trends fails post-treatment by
  even half of the worst pre-treatment wobble, significance is gone.*
  Breakdown M̄ ≥ 1 is the comfortable zone — the design survives
  violations as bad as anything visible in the data.

## Branch 3 — Instrumental variables

For IV the sensitivity question that has a statistical answer is
**weak instruments**: when the first stage is weak, 2SLS t-tests
over-reject and Wald CIs undercover, sometimes catastrophically
(Staiger-Stock; Stock-Yogo). The repair is not a bigger F-statistic
but *identification-robust* inference: Anderson-Rubin, Kleibergen's K,
and Moreira's CLR remain valid at any instrument strength.

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(3)
n = 1000
z = rng.normal(size=n)
u = rng.normal(size=n)
d = 0.15 * z + 0.8 * u + rng.normal(size=n)   # weak-ish first stage
y = 0.5 * d + 0.8 * u + rng.normal(size=n)
df = pd.DataFrame({"y": y, "d": d, "z": z})

# Full weak-IV-robust panel: AR + CLR + K + effective F in one call
wr = sp.weakrobust(df, y="y", endog="d", instruments=["z"],
                   clr_simulations=2000)
print(wr.summary())

# Or the Anderson-Rubin test alone
ar = sp.anderson_rubin_test(df, y="y", endog="d", instruments=["z"], h0=0.0)
print(ar["effective_F"])        # 13.8
print(ar["ar_ci"])              # (0.153, 1.304)
print(ar["tF_critical_value"])  # 2.61 — use instead of 1.96
```

**How to read it.**

- The **effective F** (Montiel Olea-Pflueger) of 13.8 lands in the
  "moderate" zone: above the folk threshold of 10, but below the ~23
  needed for the naive t-test to behave. In that zone, report the
  robust intervals, not the Wald CI.
- The **AR 95% CI** [0.15, 1.30] is the set of effect values β₀ not
  rejected by the Anderson-Rubin test. It is wider than the 2SLS Wald
  CI — that width *is* the honest price of the weak instrument. With
  truly weak instruments the AR CI can be unbounded; that is the
  method telling you the data cannot pin the effect down, not a bug.
- The **tF critical value** (Lee-McCrary-Moreira-Porter) gives a third
  option: keep the 2SLS t-statistic but compare it to 2.61 instead of
  1.96, with the cutoff adapting to the observed first-stage F.
- What this branch can *not* do: test the **exclusion restriction**.
  No weak-IV statistic speaks to whether `z` affects `y` only through
  `d`. That assumption must be argued from design, or relaxed into
  partial identification via `sp.iv_bounds`.

## Branch 4 — Double / debiased machine learning

DML removes confounding from the covariates you feed it — flexibly,
but only those. The DML-flavoured omitted-variable framework
(Chernozhukov-Cinelli-Newey-Sharma-Syrgkanis) extends the sensemakr
logic to the partially linear / nonparametric setting: the sensitivity
parameters are the partial R² of the latent confounder with the
outcome (`cf_y`) and with the treatment/Riesz representer (`cf_d`).

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(1)
n = 1500
X = rng.normal(size=(n, 4))
d = 0.6 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(size=n)
y = 0.8 * d + X[:, 0] + 0.5 * X[:, 2] + rng.normal(size=n)
df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
df["d"], df["y"] = d, y

r = sp.dml(df, y="y", treat="d", covariates=["x1", "x2", "x3", "x4"],
           model="plr")

ds = sp.dml_sensitivity(r, cf_y=0.05, cf_d=0.05,
                        benchmark_covariates=["x1"])
print(ds.summary())

# E-value works on any CausalResult too
ev = sp.evalue_from_result(r)
print(round(ev["evalue_estimate"], 2))   # 3.52
```

**How to read it.** The summary mirrors sensemakr's grammar:

- **RV_q = 0.448**: a confounder needs partial R² of ~45% with both
  the outcome and treatment residuals to drag the estimate to zero;
  **RV_qa** is the lower bar for losing significance.
- The **bias bound** row is scenario analysis: *if* a confounder with
  `cf_y = cf_d = 0.05` existed, the estimate could move at most ±0.067
  — the adjusted range [0.72, 0.86] still excludes zero, so the
  conclusion survives that scenario.
- The **benchmark table** calibrates the scenario: it computes the
  `cf_y`/`cf_d` a confounder "as strong as `x1`" would have, given
  everything else in the model. If your hypothesised confounder is "a
  second variable like x1", read its row instead of inventing
  `cf_y`/`cf_d` from thin air.

## Branch 5 — Missing, truncated, or selected outcomes

When the *observability* of the outcome depends on treatment —
attrition, employment-conditional wages, survival — no reweighting
trick restores point identification without untestable assumptions.
The honest output is an identified **set**.

### 5a. Manski worst-case bounds

For an outcome with known logical bounds (here binary, so [0, 1]),
Manski's no-assumption bounds bracket the ATE using only the data plus
the bounds themselves.

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(5)
n = 800
d = rng.integers(0, 2, size=n)
y = (rng.uniform(size=n) < 0.4 + 0.2 * d).astype(float)
df = pd.DataFrame({"d": d, "y": y})

mb = sp.manski_bounds(df, y="y", treat="d", y_lower=0.0, y_upper=1.0,
                      assumption="none", n_bootstrap=100)
print(mb.model_info["lower_bound"], mb.model_info["upper_bound"])
# -0.41  0.59   (width exactly y_upper - y_lower = 1.0)

# Monotone treatment response tightens the lower bound to 0
mb_mtr = sp.manski_bounds(df, y="y", treat="d", y_lower=0.0, y_upper=1.0,
                          assumption="mtr", n_bootstrap=100)
print(mb_mtr.model_info["lower_bound"], mb_mtr.model_info["upper_bound"])
```

**How to read it.** The no-assumption bounds [−0.41, 0.59] always have
width `y_upper − y_lower` and always contain zero — by construction
they can never sign the effect. That is not a defect; it is the
honest statement of what the data alone say. Each added assumption
(`'mtr'`: treatment can't hurt anyone; `'mts'`: selection is monotone
in levels) buys narrower bounds at the price of a defensible-or-not
behavioural claim. Present the bounds as a ladder — none → MTR → MTS —
so the reader sees exactly which assumption delivers which conclusion.

### 5b. Lee trimming bounds

Lee bounds target the canonical "wages only observed for the
employed" problem: treatment shifts selection (employment), so
comparing observed outcomes mixes the treatment effect with a
composition change. The fix trims the excess-selected group at the
relevant quantiles.

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(5)
n = 1200
d = rng.integers(0, 2, size=n)
wage = 2.0 + 0.4 * d + rng.normal(scale=0.8, size=n)
employed = (rng.uniform(size=n) < (0.55 + 0.15 * d)).astype(int)
df = pd.DataFrame({"d": d, "employed": employed,
                   "wage": np.where(employed == 1, wage, np.nan)})

lb = sp.lee_bounds(df, y="wage", treat="d", selection="employed",
                   n_bootstrap=100)
print(lb.summary())
```

**How to read it.** The example reports bounds [0.15, 0.73] for the
wage effect, with a trimming fraction of ~22%: treatment raised
employment by enough that 22% of the treated-and-employed have no
control counterpart, and the bounds come from deleting the top vs.
bottom 22% of their wage distribution. The estimand is the effect for
**always-selected units only** (people employed either way) — Lee
bounds cannot speak to marginal entrants. The identifying assumption
is monotonicity: treatment moves selection in one direction for
everyone. If treatment plausibly destroys some jobs while creating
others, the bounds are invalid.

For outcomes (or covariates) that are missing not-at-random rather
than selection-truncated, `sp.horowitz_manski(df, y=..., treatment=...,
covariates=[...], y_lower=..., y_upper=...)` returns the
worst-case-imputation analogue (Horowitz-Manski).

## One call for the common case

For a quick first pass on any fitted result, every `CausalResult` and
regression result exposes a `.sensitivity()` method (alias
`sp.unified_sensitivity`) that runs whatever applies — E-value and a
breakdown-bias calculation always, plus an Oster component when you
supply the needed inputs (`r2_treated`, `r2_controlled`,
`beta_uncontrolled`), a Rosenbaum component when the result exposes
`matched_pairs` outcome arrays, and a sensemakr component when you
pass the raw estimation data (`data=`, `y=`, `treat=`, `controls=`,
since result objects do not carry the data) — and collects them in
one dashboard. Anything it could not run is recorded in `dash.notes`
rather than silently dropped:

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(42)
n = 2000
x1, x2, u = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
d = (0.5 * x1 + 0.3 * x2 + 0.4 * u + rng.normal(size=n) > 0).astype(int)
y = 1.0 * d + 0.8 * x1 + 0.5 * x2 + 0.6 * u + rng.normal(size=n)
df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})

r = sp.regress("y ~ d + x1 + x2", data=df)
dash = r.sensitivity()                 # = sp.unified_sensitivity(r)
print(dash.e_value_point, dash.e_value_ci)   # 1.61 1.40
print(dash.breakdown)                  # bias needed to flip the conclusion
```

Treat the dashboard as triage, not as the final deliverable: for the
paper, run the design-specific tool from the decision tree with
explicit benchmarks, and check `dash.notes` for any component that was
skipped.

## What these tools can NOT tell you

Sensitivity analysis is the most honest part of the causal toolkit
precisely because its limits are sharp. Do not let a good-looking
robustness value write checks the design cannot cash:

1. **None of these tools detects confounding.** The sensitivity
   parameters (partial R², δ, Γ, M, cf_y/cf_d) are *not estimable from
   the data* — that is what makes the confounder unobserved. The tools
   answer "how strong would it need to be?", never "how strong is
   it?". The Oster example in Branch 1 shows a result that passes the
   δ > 1 bar while still being biased by 30%.
2. **A large breakdown value is an argument, not a proof.** "A
   confounder would need partial R² > 40%" persuades only if you can
   argue no such variable exists. That argument lives in your
   institutional knowledge, not in the output.
3. **Benchmarking assumes unobservables resemble observables.**
   Oster's δ and the sensemakr / DML benchmark tables calibrate
   against covariates you chose *because* they were the important
   ones. A qualitatively different confounder (a policy shock, a
   genetic factor) is not bound by that calibration.
4. **Each tool covers exactly one assumption.** An E-value of 3 says
   nothing about SUTVA violations, measurement error in the treatment,
   selection into the sample, or p-hacking across specifications.
   `sp.honest_did` covers parallel trends, not anticipation or
   spillovers. `sp.weakrobust` covers relevance, not exclusion.
5. **Passing a pre-trend test does not establish parallel trends** —
   the post-treatment counterfactual trend is unobservable by
   definition, and low-power pre-tests pass too easily (Roth). That is
   why Branch 2 insists on `sp.pretrends_power` plus honest CIs rather
   than the test alone.
6. **Wide Manski bounds are information, not failure.** If the
   no-assumption bounds straddle zero, the data genuinely do not sign
   the effect without further assumptions; reporting a point estimate
   instead just hides the assumption doing the work.
7. **Sensitivity to confounding is not sensitivity to specification.**
   Run the [robustness workflow](robustness_workflow.md) (spec curve,
   placebo tests, subsample stability) *as well as* — never instead
   of — the tools in this guide. The two layers fail independently.

## Reporting template

For the sensitivity panel of a paper:

1. **Name the threatened assumption** for your design (the decision
   tree's branch), and the tool that parameterises it.
2. **Report the breakdown value** next to the point estimate: RV and
   benchmark comparison (sensemakr), δ\* and identified set (Oster),
   E-value for estimate and CI, Γ\*, breakdown M / M̄, effective F with
   AR CI, or the bounds ladder.
3. **Calibrate it**: compare the breakdown value against an observed
   quantity (strongest measured covariate, worst pre-trend deviation)
   so the reader can judge plausibility.
4. **State the residual exposure** explicitly — which assumptions the
   reported analysis does *not* defend (item 4 above). Reviewers trust
   papers that say this unprompted.

## Cross-references

- [Honest DiD guide](honest_did.md) — full treatment of
  `sp.honest_did`, `sp.sensitivity_rr`, and event-study workflows.
- [Robustness workflow](robustness_workflow.md) — the three-layer
  defence; this guide is Layer 3 expanded.
- [Choosing a DID estimator](choosing_did_estimator.md) and
  [Choosing an IV estimator](choosing_iv_estimator.md) — pick the
  estimator before stress-testing it.

## References

All citations resolve to verified entries in `paper.bib`:
sensemakr [@cinelli2020making]; Oster bounds [@oster2019unobservable];
E-value [@vanderweele2017sensitivity; @ding2016sensitivity];
Rosenbaum bounds [@rosenbaum2002observational];
honest DiD [@rambachan2023more]; pre-test power [@roth2022pretest];
weak-IV-robust inference [@anderson1949estimation;
@kleibergen2002pivotal; @moreira2003conditional; @olea2013robust;
@lee2022valid; @staiger1997instrumental; @stock2005testing];
DML sensitivity [@chernozhukov2022long];
Manski bounds [@manski1990nonparametric]; Lee bounds [@lee2009training];
Horowitz-Manski bounds [@horowitz2000nonparametric].
