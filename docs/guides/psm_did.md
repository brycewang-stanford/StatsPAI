# PSM-DID and the Stata `psmatch2` workflow

This guide shows how to reproduce the supported Stata propensity-score-matching
pipeline — including the `psmatch2`-style post-matching variables
(`_weight`, `_support`, `_n1`, `_nn`, `_pdif`, …) — and then run a
weighted **PSM-DID** in StatsPAI.

For the pinned Stata 18 `psmatch2` paths — nearest-neighbour, Epanechnikov
kernel, radius, local linear regression, and the Mahalanobis metric —
`sp.psmatch2` is numerically faithful to Leuven & Sianesi (2003): on the same
data the ATT/SE and the emitted matched-sample variables match the reference
fixtures in `tests/reference_parity/` (`test_psmatch2_parity.py`,
`test_psmatch2_llr_parity.py`, `test_pstest_parity.py`,
`test_psmdid_weight_parity.py`).

---

## 1. Why a dedicated front door?

`sp.match(method='psm')` and `sp.psm(...)` estimate the matched ATT, but the
classic empirical workflow needs more than a point estimate: it needs the
**per-observation matched-sample variables** that Stata writes back into the
dataset, so you can

1. run a **post-matching balance test**,
2. draw a **post-matching propensity-score density** on the *matched* sample,
3. select the matched sample and run a **weighted PSM-DID**.

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

`sp.psmatch2` reproduces the psmatch2 matching algorithms:

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

**Standard errors.** Four estimators; the first three are pinned against
Stata 18:

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
- `se='ai'` — the simple matched-pair SE. **Anti-conservative** under
  matching with replacement (it ignores the extra variance from reusing
  controls); prefer `ai=J` for nearest-neighbour inference.
- `se='bootstrap'` — see below. Not a Stata psmatch2 option, and the only
  inference available for `method='llr'`.

The nearest-neighbour SE, the AI-robust SE, and the radius ATT/SE match
Stata 18 to machine precision; the smooth Epanechnikov kernel ATT matches to
~1e-8 (bounded by the independent logit propensity-score estimate, not the
matching algorithm).

`m.result.model_info` records the migration contract explicitly:
`propensity_model="logit"`, `estimand_scope="ATT"`, `outcome_status`, and
`att_defined`. When `outcome` is omitted, `sp.psmatch2` still returns
`matched_data` and `_weight` for downstream PSM-DID, but the cross-sectional
ATT is intentionally `NaN` and `att_defined=False`.

### Local linear regression (`llr`) matching

```python
m = sp.psmatch2(df, treat='d', outcome='y', covariates=X,
                method='llr', kernel='tricube', bwidth=0.5)
```

Each treated unit's counterfactual is the intercept of a kernel-weighted
degree-1 regression of the control outcome on the propensity gap
(Heckman, Ichimura & Todd 1997). Matches Stata to ~4e-11 relative across the
tricube, biweight, normal and uniform kernels.

Two things about Stata's `llr` you need to know before comparing numbers:

> **`psmatch2 ..., llr` with the default kernel is not local linear
> regression.** `epan` is psmatch2's default kernel for `llr`, and for that
> combination `psmatch2.ado` rewrites the request as *nearest-neighbour
> matching on an `lpoly`-smoothed outcome*. Only a non-Epanechnikov kernel
> reaches psmatch2's own LLR routine. On the reference fixture the two give
> 0.0322 and −0.0275 respectively. StatsPAI always runs genuine LLR and warns
> when you pass `kernel='epan'`; pass `kernel='tricube'` to reproduce Stata's
> LLR numbers.
>
> **Stata reports no standard error for LLR** (`seatt = .`), because local
> linear weights can be negative and the analytic formula assumes they are
> not. StatsPAI defaults `method='llr'` to `se='bootstrap'` instead, so you
> get inference where Stata gives you none. `se='psmatch2'` is refused for
> `llr` rather than silently returning a number the formula cannot support.

`method='spline'` is **not** implemented: Stata delegates it to the separate
SSC package `-spline-` and reports no SE for it either, so there is no
reference path to align against.

### Bootstrap standard errors

```python
m = sp.psmatch2(df, treat='d', outcome='y', covariates=X,
                method='kernel', se='bootstrap',
                bootstrap_reps=999, bootstrap_seed=42)
```

The bootstrap resamples units with replacement *within treatment arm* and
**re-estimates the propensity score in every replication**. That is the point:
every analytic matching SE conditions on the fitted score and so ignores the
uncertainty the score itself contributes. Pass `bootstrap_seed` for
reproducible numbers; `model_info` records how many replications succeeded and
the bootstrap bias.

> Abadie & Imbens (2008) prove the nonparametric bootstrap is **not** valid
> for nearest-neighbour matching with a fixed number of matches. StatsPAI
> warns if you ask for it there. It is sound for the smooth kernel-class
> estimators (`kernel`, `radius`, `llr`), which is where it is the default.

## 3. Post-matching balance

There are two balance tables, and they are deliberately different.

### `m.pstest()` — Stata's table, digit for digit

```python
t = m.pstest()
print(t.summary())            # the two blocks pstest prints
t.table                       # per-covariate rows
t.summary_stats['matched']    # Ps R2, LR chi2, MeanBias, Rubin's B and R
```

Use this to check a port against a printed `pstest` table. It reproduces
`pstest x1 x2, both` to 1e-14 on the per-covariate rows and 1e-9 on the
summary block, including Rubin's B and R and their `B < 25`, `R ∈ [0.5, 2]`
flags (Rubin 2001).

### `m.balance()` — StatsPAI's diagnostics

```python
bal = m.balance()
print(bal.summary())
```

Reports weighted SMD, variance ratios, KS statistics and effective sample
size.

> **These two do not agree, by design.** `pstest` keeps the **unmatched**
> pooled standard deviation in the denominator of the post-matching bias, so
> its "before" and "after" rows are directly comparable. `balance()` uses the
> matched-sample SD, the convention most non-Stata packages follow. On the
> reference fixture the post-matching figure for `x1` is 13.91 under `pstest`
> and 14.73 under `balance()` — same data, same weights, different (and both
> defensible) conventions. Quote whichever you like, but do not present one as
> the other.

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

## 5. PSM-DID: weighted difference-in-differences

The canonical Stata recipe

```stata
psmatch2 d x1 x2, neighbor(1)             // produces _weight, _support
* merge _weight onto the panel by id, then
reg y i.treat##i.post [aweight=_weight] if _support==1
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
have a post-period indicator.

### 5a. `aweight` or `fweight`? It changes your standard error

Stata's `aweight` and `fweight` give the **same coefficient** and **different
standard errors**, because `fweight` treats a control reused *w* times as *w*
independent observations:

| `weight=` | Stata equivalent | residual df | on the reference fixture |
| --------- | ---------------- | ----------- | ------------------------ |
| `'aweight'` (default) | `[aweight=_weight]` | `n_rows − k` | SE **0.250051** |
| `'fweight'` | `[fweight=_weight]` | `Σw − k` | SE **0.214797** |
| `'none'` | no weights | `n_rows − k` | SE 0.266448 |

The DiD coefficient is 1.551163 in all three cases. The `fweight` interval is
about 14% narrower *purely from the degrees of freedom*.

**The default is `'aweight'`, and that is the recommendation.** A control that
gets matched three times is not three independent draws, so the `fweight`
degrees of freedom overstate how much information the matched sample carries.
Use `'fweight'` when you specifically need to reproduce the textbook Stata
line, not because it looks more precise.

```python
did = m.psm_did(panel, id='id', y='y', post='post', weight='fweight')
```

`'fweight'` requires **integer** weights, exactly as Stata does. Matching with
`k > 1` neighbours splits each treated unit's weight into `1/k` shares, so
`_weight` is fractional and `weight='fweight'` raises with a pointer back to
`'aweight'`.

> **Changed in 1.22.** Before 1.22, `weight='fweight'` computed `aweight`
> numbers while this guide advertised the `[fweight=]` line — the standard
> error did not match the recipe it claimed to implement. The default moved
> from `'fweight'` to `'aweight'`, which is numerically identical to the old
> default, so results from a default call are unchanged. An explicit
> `weight='fweight'` now genuinely computes Stata's `fweight` degrees of
> freedom. See `tests/reference_parity/test_psmdid_weight_parity.py`.

---

## 6. Stata → StatsPAI cheat sheet

| Stata                                            | StatsPAI                                            |
| ------------------------------------------------ | --------------------------------------------------- |
| `psmatch2 d x, out(y) n(1) logit`                | `sp.psmatch2(df, treat='d', outcome='y', covariates=['x'], neighbor=1)` |
| `psmatch2 d x, out(y) kernel bw(0.06)`           | `... method='kernel', kernel='epan', bwidth=0.06`   |
| `psmatch2 d x, out(y) radius caliper(0.05)`      | `... method='radius', caliper=0.05`                 |
| `psmatch2 d x, out(y) llr kerneltype(tricube)`   | `... method='llr', kernel='tricube'`                |
| `psmatch2 d, out(y) mahalanobis(x)`              | `... method='mahalanobis'`                          |
| `psmatch2 d x, out(y) spline`                    | *not implemented* — use `method='llr'`              |
| default `r(seatt)` analytic SE                    | `... se='psmatch2'` (default)                        |
| `psmatch2 d x, out(y) ai(2)`                      | `... ai=2` (Abadie-Imbens robust SE)                |
| `bootstrap: psmatch2 ...`                         | `... se='bootstrap', bootstrap_reps=999`            |
| `psmatch2 d x, out(y) common`                    | `... common_support='minmax'`                       |
| `psmatch2 d x` without `outcome()`                 | matched-frame only; ATT undefined (`att_defined=False`) |
| `_weight`, `_support`; nearest-neighbour `_n1`, `_nn`, `_pdif` | columns on `m.matched_data`                         |
| `pstest x, both`                                 | `m.pstest()` — Stata's exact table                  |
| (no Stata equivalent)                            | `m.balance()` — StatsPAI's own diagnostics          |
| `psgraph` / kdensity of `_pscore`               | `m.psplot()`                                         |
| `reg y i.d##i.post [aw=_weight] if _support==1`  | `m.psm_did(panel, id='id', y='y', post='post')`     |
| `reg y i.d##i.post [fw=_weight] if _support==1`  | `... weight='fweight'` (see §5a — different SE)     |

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
