# PSM-DID and the Stata `psmatch2` workflow

This guide shows how to reproduce the supported Stata propensity-score-matching
pipeline — including the `psmatch2`-style post-matching variables
(`_weight`, `_support`, `_n1`, `_nn`, `_pdif`, …) — and then run a
frequency-weighted **PSM-DID** in StatsPAI.

For the pinned Stata 18 `psmatch2` paths (nearest-neighbour, Epanechnikov
kernel, and radius matching), `sp.psmatch2` is numerically faithful to Leuven
& Sianesi (2003): on the same data the ATT/SE and emitted matched-sample
variables match the reference fixtures (`tests/reference_parity/
test_psmatch2_parity.py`).

---

## 1. Why a dedicated front door?

`sp.match(method='psm')` and `sp.psm(...)` estimate the matched ATT, but the
classic empirical workflow needs more than a point estimate: it needs the
**per-observation matched-sample variables** that Stata writes back into the
dataset, so you can

1. run a **post-matching balance test**,
2. draw a **post-matching propensity-score density** on the *matched* sample,
3. select the matched sample and run a **frequency-weighted PSM-DID**.

Those variables are now produced automatically.

---

## 2. The matched-sample variables

Every nearest-neighbour run attaches a `matched_data` frame carrying the
Stata `psmatch2` columns:

| Column         | Meaning                                                        |
| -------------- | ------------------------------------------------------------- |
| `_id`          | running observation id over the estimation sample             |
| `_treated`     | treatment indicator (1/0)                                     |
| `_pscore`      | estimated propensity score                                    |
| `_support`     | common-support flag (1 on / 0 off)                           |
| `_weight`      | frequency weight (treated-on-support = 1; a reused control accumulates its `1/k` shares; outside the matched sample = missing) |
| `_n1` … `_nk`  | `_id` of the 1st … k-th matched control (treated rows)        |
| `_nn`          | number of matched controls (0 on control rows, like psmatch2) |
| `_pdif`        | \|Δ propensity score\| to the **nearest** match               |
| `_y`           | mean outcome of the matched control(s) (treated rows)         |

```python
import statspai as sp

df = sp.cps_wage()
m = sp.psmatch2(df, treat='union', outcome='log_wage',
                covariates=['education', 'experience', 'tenure'])
m.matched_data[['_pscore', '_treated', '_weight', '_n1', '_pdif']].head()
```

`sp.match` / `sp.psm` expose the same frame on `result.matched_data`.

---

## 2b. Matching methods and standard errors

`sp.psmatch2` reproduces the three psmatch2 matching algorithms:

```python
# nearest-neighbour (default), k = 1
m = sp.psmatch2(df, treat='union', outcome='log_wage', covariates=X)

# kernel matching (Epanechnikov, bandwidth 0.06 — Stata defaults)
m = sp.psmatch2(df, treat='union', outcome='log_wage', covariates=X,
                method='kernel', kernel='epan', bwidth=0.06)

# radius matching (= uniform kernel within the caliper)
m = sp.psmatch2(df, treat='union', outcome='log_wage', covariates=X,
                method='radius', caliper=0.05)
```

Kernel and radius match each treated unit to *all* controls within the
bandwidth, weighted by a kernel of the propensity-score distance, so they
produce `_weight` and `_y` but not the discrete-neighbour columns
(`_n1`/`_nn`/`_pdif`) — exactly like Stata.

**Standard errors.** Three estimators, all matched against Stata 18:

```python
m = sp.psmatch2(df, treat='d', outcome='y', covariates=X)            # se='psmatch2' (default)
m = sp.psmatch2(df, treat='d', outcome='y', covariates=X, ai=1)      # Abadie-Imbens robust, ai(1)
m = sp.psmatch2(df, treat='d', outcome='y', covariates=X, ai=2)      # ai(2): 2 within-arm matches
```

- `se='psmatch2'` (default) — Stata's homoskedastic analytic ATT SE
  `sqrt(var1/N1 + var0·Σw²/N1²)`, *digit for digit*.
- `ai=J` / `se='abadie_imbens'` — the Abadie–Imbens (2006)
  heteroskedasticity-robust SE (`psmatch2 , ai(J)`), which estimates
  `σ²(X)` from each unit's `J` nearest same-arm neighbours — reproduced to
  machine precision.
- `se='ai'` — the simple matched-pair SE.

The nearest-neighbour SE, the AI-robust SE, and the radius ATT/SE match
Stata 18 to machine precision; the smooth Epanechnikov kernel ATT matches to
~1e-8 (bounded by the independent logit propensity-score estimate, not the
matching algorithm).

`m.result.model_info` records the migration contract explicitly:
`propensity_model="logit"`, `estimand_scope="ATT"`, `outcome_status`, and
`att_defined`. When `outcome` is omitted, `sp.psmatch2` still returns
`matched_data` and `_weight` for downstream PSM-DID, but the cross-sectional
ATT is intentionally `NaN` and `att_defined=False`.

> **Local-linear (`llr`) matching is not provided.** Stata routes its default
> `psmatch2 ... llr` through the `lpoly` command, whose bandwidth and boundary
> handling are not bit-reproducible; rather than ship an estimator that
> disagrees with Stata, StatsPAI omits it. Use kernel matching with a small
> bandwidth for a comparable local estimator.

## 3. Post-matching balance (the `pstest` analogue)

```python
bal = m.balance()
print(bal.summary())
```

`smd_raw` is the standardized mean difference **before** matching; the
`smd_weighted` column is the **after**-matching SMD computed on the matched
sample with the `_weight` frequency weights — exactly what Stata `pstest`
reports.

---

## 4. Common-support / propensity-score plot

```python
fig, ax = m.psplot()          # matched controls reweighted by _weight
```

The control density uses the matching weights, so it reflects the matched
sample rather than the raw donor pool; the raw control density is overlaid
as a dashed line for comparison.

Impose common support (Stata's `, common`) with:

```python
m = sp.psmatch2(df, treat='union', outcome='log_wage',
                covariates=['education', 'experience', 'tenure'],
                common_support='minmax')
```

Off-support treated units are then dropped from matching, the ATT is taken
over the on-support treated, and `_support == 0` flags the trimmed rows.

---

## 5. PSM-DID: frequency-weighted difference-in-differences

The canonical Stata recipe

```stata
psmatch2 d x1 x2, neighbor(1)             // produces _weight, _support
* merge _weight onto the panel by id, then
reg y i.treat##i.post [fweight=_weight] if _support==1
```

becomes, in StatsPAI:

```python
# 1. match on a baseline cross-section (one row per unit; outcome optional)
m = sp.psmatch2(baseline, treat='d', covariates=['x1', 'x2'], neighbor=1)

# 2. weighted DiD on the panel — _weight is merged in by id
did = m.psm_did(panel, id='id', y='y', time='time', treat_time=1, treat='d')
print(did.summary())          # did.estimate is the DiD (treat × post) effect
```

`psm_did` merges `_weight` (and `_support`) onto the panel by `id`, keeps the
matched sample, builds the `treat × post` interaction, and fits the weighted
regression with `sp.feols`. Add unit/time fixed effects (the main effects
they absorb are dropped automatically) and clustered SEs:

```python
did = m.psm_did(panel, id='id', y='y', time='time', treat_time=1, treat='d',
                fixed_effects=['id', 'time'], cluster='id')
# fitted model: y ~ _did | id + time
```

Pass `post=<column>` directly instead of `time` + `treat_time` if you already
have a post-period indicator, and `weight='none'` to run the matched-sample
DiD unweighted.

---

## 6. Stata → StatsPAI cheat sheet

| Stata                                            | StatsPAI                                            |
| ------------------------------------------------ | --------------------------------------------------- |
| `psmatch2 d x, out(y) n(1) logit`                | `sp.psmatch2(df, treat='d', outcome='y', covariates=['x'], neighbor=1)` |
| `psmatch2 d x, out(y) kernel bw(0.06)`           | `... method='kernel', kernel='epan', bwidth=0.06`   |
| `psmatch2 d x, out(y) radius caliper(0.05)`      | `... method='radius', caliper=0.05`                 |
| default `r(seatt)` analytic SE                    | `... se='psmatch2'` (default)                        |
| `psmatch2 d x, out(y) ai(2)`                      | `... ai=2` (Abadie-Imbens robust SE)                |
| `psmatch2 d x, out(y) common`                    | `... common_support='minmax'`                       |
| `psmatch2 d x` without `outcome()`                 | matched-frame only; ATT undefined (`att_defined=False`) |
| `_weight`, `_support`; nearest-neighbour `_n1`, `_nn`, `_pdif` | columns on `m.matched_data`                         |
| `pstest x, both`                                 | `m.balance()`                                       |
| `psgraph` / kdensity of `_pscore`               | `m.psplot()`                                         |
| `reg y i.d##i.post [fw=_weight] if _support==1`  | `m.psm_did(panel, id='id', y='y', post='post')`     |

---

## See also

- [Choosing a matching estimator](choosing_matching_estimator.md)
- [Migrating from R to StatsPAI](migration-from-r.md)

## References

- Leuven, E. and Sianesi, B. (2003). *PSMATCH2: Stata module to perform full
  Mahalanobis and propensity score matching, common support graphing, and
  covariate imbalance testing.* Statistical Software Components S432001,
  Boston College Department of Economics.
- Rosenbaum, P.R. and Rubin, D.B. (1983). The central role of the propensity
  score in observational studies for causal effects. *Biometrika*, 70(1),
  41–55.
- Heckman, J.J., Ichimura, H. and Todd, P.E. (1997). Matching as an
  econometric evaluation estimator: Evidence from evaluating a job training
  programme. *Review of Economic Studies*, 64(4), 605–654.
