# Cross-engine validation — `sp.cross_validate`

> Estimate one model with several **independent** engines and trust the number
> only when they agree.

## Why this exists

Scott Cunningham — a prominent voice in the difference-in-differences
literature and author of *Causal Inference: The Mixtape* — ran a now-widely-
discussed experiment: he asked an AI agent to estimate the **same**
Callaway–Sant'Anna DiD with six different packages (two each in Python, R and
Stata) and found the point estimates frequently *did not match*. His
prescription is simple and hard to argue with:

> When you use an AI to run a causal-inference model, require it to estimate the
> same specification with **at least two independent packages**, and only trust
> the result when they agree.

That discipline is easy to state and tedious to do by hand — different
languages, different formula syntaxes, different default variance estimators.
`sp.cross_validate` turns it into a single call that humans and agents share.

## The one call

```python
import statspai as sp

cv = sp.cross_validate(
    df, "iv",
    y="wage", endog=["educ"], instruments=["qob"], covariates=["age"],
    engines=["statspai", "R::fixest", "pyfixest"],
)
print(cv.summary())
```

```text
========================================================================
Cross-engine validation — estimand 'iv', focal term 'educ'
========================================================================
Engine              Estimate     Std.Err                95% CI    status
------------------------------------------------------------------------
statspai             0.13420     0.02610      [0.0830, 0.1854]        ok
R::fixest            0.13420     0.02610      [0.0830, 0.1854]        ok
pyfixest             0.13420     0.02610      [0.0830, 0.1854]        ok
------------------------------------------------------------------------
VERDICT: ✓ AGREE   (3/3 engines ran)
  Independent engines reproduce the estimate within tolerance — the number
  is engine-robust.
  max relative coef difference: 1.7e-15   (reference: statspai)
```

## Supported engines and estimands

| Engine | How it runs | Estimands |
| --- | --- | --- |
| `statspai` | native | ols, feols, iv, poisson, dml, did |
| `pyfixest` | in-process | ols, feols, iv, poisson |
| `linearmodels` | in-process | ols, feols, iv |
| `doubleml` | in-process | dml |
| `R::fixest` | `Rscript` subprocess | ols, feols, iv, poisson |
| `R::did` | `Rscript` subprocess | did (Callaway–Sant'Anna) |
| `Stata` | batch `do` subprocess | ols, feols, iv |

`engines="auto"` (the default) runs every backend that is **installed and
applicable** to the estimand. Naming engines explicitly is stricter: a backend
you ask for that is missing is reported as `unavailable` (and recorded as a
degradation) rather than silently skipped — failures stay loud.

Estimand aliases are forgiving: `reg`/`regress`→`ols`, `fe`/`twfe`→`feols`,
`2sls`/`tsls`→`iv`, `ppml`→`poisson`.

## How the verdict is decided

`cross_validate` reconciles the focal coefficient (the treatment, or the first
endogenous regressor for IV, or whatever you pass as `term=`) across the engines
that ran:

| Verdict | Meaning |
| --- | --- |
| **AGREE** | Coefficients (and, where comparable, standard errors) match within tolerance. The number is engine-robust. |
| **PARTIAL** | Point estimates agree but standard errors do not — usually a default-variance-estimator mismatch. Align `vcov=` across engines. |
| **DISAGREE** | Coefficients differ beyond tolerance. The result is implementation-sensitive; reconcile the specification before trusting it. |
| **INSUFFICIENT** | Fewer than two engines produced an estimate. Cross-validation could not run — install another backend. |

For agents, `cv.to_dict(detail="agent")` includes two guard fields:
`engine_status_counts` and `can_claim_cross_engine_agreement`. The latter is
true only when the verdict is `AGREE` and at least two engines actually ran.
If it is false, do not write "cross-engine validated" in a table note or paper
draft; report the unavailable engines and rerun after installing another
backend.

### Tolerance is honest about *why*

Not every method *should* match to fifteen digits, and pretending otherwise
would either hide real disagreement or cry wolf. `cross_validate` picks a
tolerance regime from the estimand and tells you the rationale:

- **exact mode** (OLS, IV, fixed-effects): closed-form algebra — two correct
  implementations differ only by floating-point and degrees-of-freedom noise.
  Coefficients are compared at `rtol=1e-6`.
- **iterative mode** (Poisson / GLM): convergence tolerances legitimately
  differ across libraries, so coefficients are compared at `rtol=1e-4`.
- **statistical mode** (DML, causal forests): the estimator embeds randomness
  (cross-fitting, RNG). Two correct runs need *not* match to many digits; they
  are judged on a standard-error scale (`|Δβ| ≤ 0.25·max SE`).

Override per call when you know better:

```python
cv = sp.cross_validate(df, "dml", y="y", treatment="d", covariates=["x1", "x2"],
                       tol={"se_band": 0.5})   # looser band for noisy learners
```

## What "agreement" does and does not certify

Cross-engine agreement certifies **reproducibility**, not **truth**. If two
engines both fit a two-way fixed-effects DiD with a contaminated comparison,
they will agree on the *same biased number*. `cross_validate` answers "did I
implement the estimator I think I did, the same way the rest of the field
would?" — a different and complementary question from "is this estimator
identified?", which is what `sp.audit_result` / `sp.honest_did` /
`sp.sensitivity` address. Use both.

## Worked example: Cunningham's experiment, reproduced

Cunningham's experiment was specifically about Callaway–Sant'Anna staggered
DiD. `cross_validate` reproduces it directly with the `did` estimand: it
reconciles StatsPAI's `sp.callaway_santanna` overall ATT against R's `did`
package (`att_gt` + `aggte(type="simple")`) — the canonical implementation.

```python
import statspai as sp

# A staggered panel with y, a cohort column g (first-treatment period,
# 0 = never treated), a time column t and a unit id i.
cv = sp.cross_validate(
    mpdta, "did",
    y="lemp", g="first.treat", t="year", i="countyreal",
    engines=["statspai", "R::did"],
)
print(cv.verdict)            # AGREE — overall ATT matches to ~1e-15
```

On the canonical Callaway–Sant'Anna `mpdta`, StatsPAI and R's `did` agree on
the overall ATT to machine precision. Two notes worth their own paragraph,
because they *are* Cunningham's point:

- **CS-DID reconciles the overall (`type="simple"`) ATT**, the focal term
  `"ATT"`. Different aggregations (group, dynamic/event-study) are different
  estimands; `cross_validate` pins the simple aggregate so both engines answer
  the same question.
- **Watch the column types.** R's `did::att_gt` returns a *different* estimate
  for an integer-typed cohort column than for a numeric one (a real
  cross-package fragility). `cross_validate`'s `R::did` adapter coerces the
  cohort / time / id columns to numeric so the comparison is apples-to-apples —
  exactly the kind of silent discrepancy hand-rolled cross-checks miss.

For the *design* (parallel trends, sensitivity) keep using
`sp.callaway_santanna` + `sp.honest_did`; `cross_validate` answers the separate
"do the packages agree on the number?" question.

## Where it sits next to the alternatives

| Tool | Question it answers |
| --- | --- |
| `sp.cross_validate` | Do **independent engines / languages** reproduce this number? |
| `sp.compare_estimators` | Do **different identification strategies** (OLS vs IPW vs DML…) agree? |
| `sp.verify` | Is this estimate **stable** under resampling (bootstrap / placebo / subsample)? |
| `sp.replicate` | Does StatsPAI reproduce a **published paper's** headline number? |

`cross_validate` is the only one of the four that leaves the Python process —
it is the layer that actually shells out to R's `fixest` and Stata, the engines
an econometrician would reach for, and orchestrates them around one estimand.

## For agents

`cross_validate` is a registered, MCP-exposed tool. The result serialises for
agents:

```python
cv.to_dict(detail="agent")
# {
#   "verdict": "AGREE",
#   "engines": [{"engine": "statspai", "coef": ..., "status": "ok"}, ...],
#   "engine_status_counts": {"ok": 3},
#   "can_claim_cross_engine_agreement": true,
#   "agreement": {"max_rel_coef_diff": 1.7e-15, "policy": {...}, ...},
#   "next_steps": ["Estimate is engine-robust; safe to report. ..."],
#   "provenance": {"data": {...}, "statspai": "1.18.0",
#                  "R::fixest": "R 4.5.2", ...},
#   "degradations": [...],   # engines asked for but unavailable
# }
```

`provenance` records the exact engine versions that ran. If the input frame was
created by `sp.from_worldbank`, `sp.from_fred`, or `sp.from_sdmx`, it also
records the normalized data-source metadata under `provenance["data"]`.

## Installing more engines

Cross-validation is only as strong as the number of independent engines you can
muster:

```bash
pip install pyfixest linearmodels doubleml   # in-process Python engines
# R cross-check: install R, then in R:  install.packages(c("fixest","jsonlite"))
# Stata cross-check: put stata-mp / stata-se on PATH (or set STATSPAI_STATA_BIN)
```
