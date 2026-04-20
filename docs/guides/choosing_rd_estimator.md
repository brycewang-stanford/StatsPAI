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

`bwselect='mserd'` (default) is MSE-optimal and RD-specific. Other
options:

| `bwselect`   | When to use                                          |
|--------------|------------------------------------------------------|
| `'mserd'`    | Default — MSE-optimal, common bandwidth              |
| `'msetwo'`   | Different bandwidths on each side of cutoff          |
| `'cerrd'`    | Coverage-error-rate optimal (better for CI coverage) |
| `'certwo'`   | CER-optimal, two bandwidths                          |
| Fixed `h=`   | Specified by you (for robustness checks)             |

Rule of thumb: use `mserd` for point estimates, run `cerrd` as a
robustness check for CI coverage.

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
