# Choosing an RD estimator

## 0. TL;DR flowchart

```
Is the treatment deterministic at the cutoff (P(D=1|X>=c)=1)?
  YES -> SHARP RD
  NO  -> Does the cutoff shift treatment PROBABILITY?
          YES -> FUZZY RD (Wald ratio)
          NO  -> RD is not identified; consider bunching/DiD

What is the running variable behaviour at the cutoff?
  Continuous density          -> Standard local polynomial (sp.rdrobust)
  Discrete (time-based)        -> RDiT (sp.rdit)
  Kink (derivative jump)       -> RKD (sp.rkd)
  Two running variables        -> sp.rd2d
  Multiple cutoffs             -> sp.rdmulti
  Randomisation (near cutoff)  -> Local randomization (sp.rdrandinf)
```

## 1. The default: sharp RD with CCT-robust CI

```python
r = sp.rdrobust(df, y='y', x='running_var', c=0.0,
                kernel='triangular', bwselect='mserd')
r.summary()
```

This is the Calonico-Cattaneo-Titiunik (2014) procedure:
- Triangular kernel + MSE-optimal bandwidth
- Bias-corrected point estimate
- Robust standard errors accounting for bias correction

**Do not use naive local-linear regression** — it underestimates
standard errors by ignoring bias.

## 2. Fuzzy RD

```python
r = sp.rdrobust(df, y='y', x='running_var', c=0.0, fuzzy='treatment_var')
```

Fuzzy RD identifies a LATE for compliers. Also report:
- First-stage jump in treatment probability (`r.model_info['first_stage']`)
- Kitagawa test for instrument validity (`sp.kitagawa_test`)

## 3. Decision tree for method variants

| Situation                                   | Method                          |
|---------------------------------------------|---------------------------------|
| Standard continuous-x sharp RD              | `sp.rdrobust`                   |
| Standard fuzzy RD                           | `sp.rdrobust(..., fuzzy='d')`   |
| Discrete running variable (e.g., date)      | `sp.rdit`                       |
| Kink design (slope jump, not level)         | `sp.rkd`                        |
| Two-dimensional cutoff                      | `sp.rd2d`                       |
| Multiple cutoffs (school-district boundaries)| `sp.rdmulti`                    |
| Near-cutoff local randomization             | `sp.rdrandinf`                  |
| Heterogeneous effects                       | `sp.rdhte`, `sp.rd_forest`      |
| ML-based extrapolation beyond cutoff        | `sp.rd_extrapolate`             |
| Honest inference (Armstrong-Kolesar)        | `sp.rd_honest`                  |
| Manipulation / bunching at cutoff           | `sp.bunching` + `sp.rddensity`  |

## 4. Mandatory diagnostics

Every RD paper must report these. StatsPAI packages them all:

```python
# 1. Density continuity (no manipulation)
sp.rddensity(df, x='running_var', c=0.0)
sp.mccrary_test(df, x='running_var', c=0.0)

# 2. Covariate balance across the cutoff
sp.rdbalance(df, x='running_var', c=0.0, covariates=[...])

# 3. Placebo cutoffs
sp.rdplacebo(df, y='y', x='running_var',
             true_cutoff=0.0, placebo_cutoffs=[-0.5, 0.5])

# 4. Bandwidth sensitivity
sp.rdbwsensitivity(df, y='y', x='running_var', c=0.0)

# 5. Power
sp.rdpower(df, y='y', x='running_var', c=0.0, tau=[0.1, 0.5, 1.0])
```

Or in one call:
```python
r = sp.rdrobust(df, y='y', x='running_var', c=0.0)
r.next_steps()  # prints the priority-ordered checklist
```

## 5. Bandwidth selection

`bwselect='mserd'` (default) is MSE-optimal and RD-specific. StatsPAI
implements the full Calonico–Cattaneo–Titiunik three-stage cascade
natively, and all ten of R's selector names are accepted:

| `bwselect`                | When to use                                     |
|---------------------------|-------------------------------------------------|
| `'mserd'`                 | Default — MSE-optimal, common bandwidth         |
| `'msetwo'`                | MSE-optimal, separate left/right                |
| `'msesum'`                | MSE-optimal for the sum of the two one-sided estimands |
| `'msecomb1'`/`'msecomb2'` | min / median of the MSE variants                |
| `'cerrd'`                 | Coverage-error-rate optimal (better CI coverage)|
| `'certwo'`, `'cersum'`    | CER-optimal, separate / sum                     |
| `'cercomb1'`/`'cercomb2'` | min / median of the CER variants                |
| Fixed `h=`                | Specified by you (for robustness checks)        |

Rule of thumb: use `mserd` for point estimates, run `cerrd` as a
robustness check for CI coverage.

The default path matches R `rdrobust` 4.0.0 to ~1e-12 on `h`, `b`, both
coefficients and both standard errors, so `bwselect='cct'` (which delegates
to the official Python port and needs the `statspai[rd-cct]` extra) is no
longer needed for parity. It remains available as an independent check.

**`cluster=` changes the bandwidth, not just the standard error.** The
cascade's variance term is a sandwich, so clustering propagates into `h`
and `b`. This surprises people who expect clustering to be an
inference-only switch, but it matches R — and R makes one further
substitution silently, which StatsPAI reproduces: passing `cluster=` with
the default `vce='nn'` promotes the variance to `cr1`, whose residuals are
`hc1`'s rather than nearest-neighbour ones. Nearest-neighbour differencing
removes exactly the within-cluster correlation a clustered variance exists
to capture, so the two must not be combined; pairing them understates the
SE by roughly 10x.

## 5b. Variance estimator

`vce=` mirrors R's argument of the same name:

| `vce`            | Meaning                                             |
|------------------|-----------------------------------------------------|
| `'nn'` (default) | Nearest-neighbour residuals, `nnmatch=3`            |
| `'hc0'`–`'hc3'`  | Regression residuals with the usual HC corrections  |

R's `cr1`/`cr2`/`cr3` are requested by passing `cluster=` rather than by
name.

## 6. Polynomial order

Default `p=1` (local linear). Gelman & Imbens (2019) argue strongly
against high-order polynomials. Report `p=2` (local quadratic) as a
sensitivity check, not as the preferred specification. `p>=3` is
almost never justified.

## 7. Reading the output

```python
r = sp.rdrobust(df, y='y', x='x', c=0.0)
r.estimate         # Point estimate (bias-corrected)
r.se               # Robust SE
r.ci               # Robust CI
r.model_info['bandwidth_h']    # Chosen bandwidth h
r.model_info['bandwidth_b']    # Bias-correction bandwidth b
r.model_info['n_effective_left'], ['n_effective_right']  # Obs used
r.tidy()           # Includes conventional, bias-corrected, robust rows
r.glance()         # Nobs, bandwidth, kernel, estimator
r.plot()           # Falls back to coefplot; use sp.rdplot for binscatter
```

## 8. When NOT to use RD

- **No clear discontinuity**: check `sp.rdplot` first; if the plot
  doesn't show a jump, there's nothing to estimate.
- **Bunching at cutoff**: McCrary/RD density tests will flag this.
  Use `sp.bunching` instead.
- **Running variable is choice variable**: identification fails.
  Use IV or DiD.

<!-- AGENT-BLOCK-START: rdrobust -->

## For Agents

**Pre-conditions**
- running variable x is continuous with support on both sides of c
- treatment assignment is determined by the cutoff c (sharp) or probabilistically at c (fuzzy)
- sufficient mass of observations within the optimal bandwidth

**Identifying assumptions**
- Continuity of potential outcomes in x at c (Hahn, Todd, van der Klaauw 2001)
- No manipulation of x at c (McCrary density test)
- Local randomization only in a neighborhood of c — extrapolation away from c is not identified
- Covariate balance at c (optional but recommended)

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| McCrary density test p < 0.05 | `AssumptionViolation` | Use donut-hole RD (donut=<δ>) or partial-identification bounds. | `sp.rdrobust` |
| Covariate imbalance at cutoff (sp.rdbalance rejects) | `AssumptionViolation` | Include covariates as controls, narrow bandwidth, or report as caveat. |  |
| Effect unstable across bandwidth halvings | `AssumptionWarning` | Report sp.rdbwsensitivity and sp.rd_honest (Armstrong-Kolesár honest CI). | `sp.rd_honest` |
| Placebo cutoffs show significant 'effects' | `AssumptionViolation` | The RD signal is noise; seek an alternative identification strategy. | `sp.manski_bounds` |

**Alternatives (ranked)**
- `sp.rd_honest`
- `sp.rdrbounds`
- `sp.bounds`

**Typical minimum N**: 500

<!-- AGENT-BLOCK-END -->
