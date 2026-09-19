# did_synth / `didvar` sub-cluster — continuous dose, time-varying covariates, distributional DiD

Cluster tag `didvar`. Worktree `.claude/worktrees/pc-did-synth`. All changes uncommitted.

Files created:

- `tests/reference_parity/_generate_did_synth_didvar_data.py` (fixed-seed panel writer)
- `tests/reference_parity/_generate_did_synth_didvar_R.R` (R reference generator)
- `tests/reference_parity/_fixtures/did_synth_didvar_{contdose,contdose_unbal,tvc,mpdta}.csv`
- `tests/reference_parity/_fixtures/did_synth_didvar_R.json`
- `tests/reference_parity/test_did_synth_didvar_parity.py` (32 tests)

Files changed (src): `src/statspai/did/continuous_did.py`, `src/statspai/did/timevarying_covariates.py`.
`src/statspai/did/functional_form.py` was **not** changed (no defect found).

## 0. Where the three names resolve

| `sp.` name | resolves to | notes |
| --- | --- | --- |
| `continuous_did` | `statspai.did.continuous_did.continuous_did` | A **different code path** from `sp.cgs_continuous_did` (`did/cgs_continuous.py`, Track A 80 vs `contdid::cont_did`). Its modes are heuristics (`att_gt` = dose-bin 2x2 rollup, `twfe`, `dose_response`) plus the deprecated `cgs` MVP; none computes the CGS ATT(d\|g,t) / ACRT estimand, so `contdid` is not a reference for it. The only mode with a canonical reference is `twfe` (a plain TWFE with a `dose x post` regressor → `fixest::feols`). |
| `distributional_did` | `statspai.did.functional_form.distributional_did` | Docstring claims `didFF::distDD` (Roth & Sant'Anna's package). Confirmed by reading `didFF` source: `distDD(...)` is `didFF(..., distDD = TRUE)`. Not a Callaway–Li QTT, so `qte::panel.qtet` / `QDiD` / `CiC` compute a different estimand (effect on quantiles, not on bin probabilities) and were not used. |
| `did_timevarying_covariates` | `statspai.did.timevarying_covariates.did_timevarying_covariates` | Claims Caetano, Callaway, Payne & Rodrigues (bib key `caetano2022difference`, already in `paper.bib`). Canonical code: Callaway's `ptetools::pte_default(d_outcome = TRUE, xformula = ~X, est_method = "reg")` (CRAN 1.0.1, sole author/maintainer Brantly Callaway per its DESCRIPTION). `did::att_gt(est_method = "reg")` computes the same cells (its panel 2x2 cells read covariates from the base period, verified from `did:::compute.att_gt` / `get_wide_data` source) and serves as a second reference. |

## 1. Per-function table

| function | reference (version) | outcome class | max rel err est / SE | test file |
| --- | --- | --- | --- | --- |
| `continuous_did(method="twfe")` | `fixest::feols(y ~ dp \| id + time)`, vcov iid / `~id` / `~region`, with and without a control, balanced + unbalanced (fixest 0.14.0, R 4.5.2) | **2** (defect fixed) → 1 | 1.1e-15 / 2.2e-15 | `tests/reference_parity/test_did_synth_didvar_parity.py` |
| `continuous_did(method="att_gt")` | none (dose-bin heuristic, no package computes it) | **2** (SE defects fixed) + **6** | reference-free: point = analytic 2x2 DID to 1e-12; bootstrap SE within 5% of analytic 2x2 SE (B=4000) | same |
| `continuous_did(method="dose_response")` | none (heuristic) | **2** (SE defect fixed) + **6** | reference-free: exact recovery of a linear dose response (1e-10), bootstrap SE ~0 | same |
| `did_timevarying_covariates` | `ptetools::pte_default` 1.0.1 and `did::att_gt` 2.3.0 (+ `DRDID` 1.2.3), `est_method = "reg"`, never-treated | **2** (wrong estimator fixed) → 1 for point estimates; SE **5** (T3, not compared) | ATT(g,t) 3.8e-15, overall (group) 1.3e-15, simple 4.4e-16 / SE not compared | same |
| `distributional_did` | `didFF::distDD` 0.1.0 (on `did 2.3.0`) | **1** | est 2.6e-12 abs (rel ≤ 1e-10); SE 2.5e-10 rel (ipw/dr propensity cases), ≤ 1e-14 otherwise | same (+ existing `test_functional_form_extended_parity.py`) |

`distributional_did` was already pinned against `distDD` on 4 configurations in
`test_functional_form_extended_parity.py` (index grade was still `analytical-only`
because no promotion record existed). This sub-cluster adds 9 more configurations
reaching the pass-through arguments that had no reference coverage: `x` under
`estimator = dr / reg / ipw`, `control_group = "notyettreated"`, `aggregation =
"dynamic" / "calendar"`, `min_e / max_e`, `balance_e`, `binpoints`, and weights +
covariates + not-yet-treated together. All agree; no silently ignored argument found.

## 2. Defects found

### D1 `continuous_did(method="twfe")` — iid SE ignored absorbed fixed effects (⚠️ correctness, default output)

- **How found**: fixest `vcov = "iid"` SE 0.0472121 vs StatsPAI 0.0429124 on the balanced panel (-9.1%). Point estimate agreed to 1e-15, so the first divergence was the variance scale: `sigma2 = RSS / (n - k)` with `k = 1` slope, although `N + T - 1` unit/period parameters were absorbed.
- **Fix**: `K = k + (N + T - c)` (`c` = connected components of the unit–period graph), i.e. fixest/reghdfe's count.
- **Before → after** (balanced, `did_synth_didvar_contdose.csv`, default `cluster=None`): SE 0.042912363 → 0.047212137 (fixest 0.0472121368402392). Default output changed.

### D2 `continuous_did(method="twfe")` — clustered SE small-sample factor missed non-nested FE (⚠️ correctness, small)

- **How found**: `vcov = ~id` SE 0.0498495 vs 0.0496411 (-0.42%), with the point estimate exact. The factor `(n-1)/(n-k)` omitted the period effects, which are not nested in the cluster.
- **Fix**: `K = k + did._core.fe_dof_not_nested(df, [id, time], cluster)` — the helper already used by `event_study` / `sun_abraham` / `wooldridge_did` (fixest `ssc(fixef.K = "nested")`).
- **Before → after** (balanced, `cluster="id"`): 0.049641066 → 0.049849497 (fixest 0.049849497105422).

### D3 `continuous_did(method="twfe")` — unbalanced panels: wrong point estimate (⚠️ correctness)

- **How found**: on `did_synth_didvar_contdose_unbal.csv` (~6% of rows removed) the slope was off by 1.3e-3 relative (8.6e-3 with a control). Cause: the within transform was the one-pass `y - ybar_i - ybar_t + ybar`, which is exact only on balanced panels.
- **Fix**: absorb both effects with `statspai.fast.demean` (alternating projections, `tol=1e-14`, `drop_singletons=False`, `backend="numpy"`), warn if it does not converge.
- **Before → after** (unbalanced, default): estimate 0.403496461 → 0.402987816 (fixest 0.402987816424948); SE 0.044333346 → 0.049373238 (fixest 0.0493732382338185). Balanced-panel estimates are unchanged (bit-identical up to 1e-15).

### D4 `continuous_did(method="att_gt")` — bootstrap discarded draw multiplicity (⚠️ correctness, SE)

- **How found**: reading the source against the brief's "silently wrong bootstrap" class, then confirmed by the analytic check `test_att_gt_single_bin_matches_analytic_2x2`: the draw was `df[df[id].isin(boot_ids)]`, which keeps each sampled unit **once** however often it was drawn (≈63% of units, no multiplicity) — a subsample, not a bootstrap. SE understated.
- **Fix**: per-unit sums/counts of pre and post outcomes; each draw is a vector of multiplicities (`np.bincount`) and every bin DID is a multiplicity-weighted ratio — identical to row means on the resampled panel.

### D5 `continuous_did(method="att_gt")` — pooled SE assumed independent bins (⚠️ correctness, SE)

- All bins share the same zero-dose comparison arm; the pooled SE was `sqrt(sum w^2 se_b^2)`. Now every bin and the pooled estimate are recomputed on the same draw and the pooled SE is the SD of the pooled replicates. `test_att_gt_pooled_se_reflects_shared_control` asserts pooled SE > 1.2 × the independence formula on the fixture panel.
- **Before → after** (D4 + D5, `did_synth_didvar_contdose.csv`, `n_boot=500, seed=0`): pooled SE 0.10639 → 0.18500; unbalanced 0.11190 → 0.18991. Point estimates unchanged (1.3020016, 1.3210114).

### D6 `continuous_did(method="att_gt")` — fallback comparison arm never bootstrapped

- When no unit has dose 0 the lowest bin becomes the comparison arm, but the bootstrap still resampled the (empty) zero-dose set, so every replicate was NaN and SE = NaN. Fixed as a by-product of D4 (the bootstrap uses the actual comparison arm); `test_att_gt_fallback_control_has_finite_se`. `model_info["control_arm"]` records which arm was used.

### D7 `continuous_did(method="att_gt" | "dose_response")` — `controls` silently ignored

- Both heuristics accepted `controls` and never used them. Now a `UserWarning` says so (not an error, to avoid breaking callers); docstring updated.

### D8 `continuous_did(method="dose_response")` — SE was not the SE of the estimate (⚠️ correctness, SE); swallowed exception

- The headline is the grid-average slope of the local-linear fit, but its "SE" was `nanmean` of the **pointwise SEs of the fitted level** — a different quantity in different units. Now a unit bootstrap of the average-slope statistic (grid and bandwidth held at full-sample values). The pointwise SEs are kept in `model_info["dose_response_pointwise_se"]`.
- The `except Exception:` fallback to `linregress` silently changed the estimator; it now catches only `ValueError` / `LinAlgError`, emits a `RuntimeWarning`, and records `model_info["fallback"]`.
- **Before → after** (`did_synth_didvar_contdose.csv`, `n_boot=500, seed=0`): estimate 0.370570 (unchanged), SE 0.243084 → 0.154392. Unbalanced: SE 0.255191 → 0.146908.
- Cost: ~3 s at the default `n_boot=500` on 120 units.

### D9 `did_timevarying_covariates` — wrong estimator under the right name (⚠️ correctness, default output)

- **How found**: ATT(g,t) cells vs `ptetools` / `did` differed by up to 0.58 (e.g. (4,4): 1.2426 vs 1.8238). Two causes, both in `_compute_att_gt` / `_attach_baseline_covariates`:
  1. never-treated units' "frozen" covariates were taken at each unit's **median period** for every cohort, not at the cohort's `g-1`, so treated and comparison units were adjusted on covariates from different periods;
  2. the adjustment was a pooled OLS `dY ~ D + X` (common slope) and ATT = coefficient on `D`, which is not the ATT under effect heterogeneity. The reference is outcome regression fitted on the comparison units only (`DRDID::reg_did_panel`): `ATT = mean_treated(dY - X beta_controls)`.
- **Fix**: rewrite `_compute_att_gt` as the `X_{g+offset}` OR estimator for both groups; drop the median-period baseline.
- Aggregation: the headline previously used treated-count cell weights ("simple"). Added `aggregation={"group","simple"}`, default `"group"` = `ptetools` overall ATT and `did::aggte(type="group")`, following the method author's implementation; `"simple"` reproduces `did::aggte(type="simple")`. Both are in `model_info` (`att_group`, `att_simple`).
- Bootstrap: the blanket `except Exception: continue` now catches only `LinAlgError`.
- **Before → after** (`did_synth_didvar_tvc.csv`, x1 + x2, default): estimate 1.273390 → 1.257705 (ptetools 1.25770523749721); SE (n_boot=200, seed=0) 0.11274 → 0.14566. Cells: e.g. ATT(4,4) 1.242550 → 1.823768; ATT(6,6) 0.524275 → 0.244804.

### Upstream observations (not StatsPAI defects; recorded in the R generator)

- `ptetools` 1.0.1: `two_by_two_subset` does `subset(data, G == g | ...)`; if the user's cohort column is literally named `g`, non-standard evaluation resolves `g` to that column, every cohort is kept in every cell and `control_group` is silently lost. The generator hands ptetools a copy with the column renamed.
- `did` 2.3.0: with an **integer** cohort column, recoding `g == 0` to `Inf` in place yields `NA`; never-treated units are dropped and the last cohort is coerced to "never treated" (warning only). The generator stores `g` as double.
- `didFF` 0.1.0 `distDD`: crashes ("arguments imply differing number of rows") whenever a bin's influence function is dropped (here `balance_e = 1`). Pinned against distDD's recipe re-run in R, after first showing that recipe reproduces distDD's own `dynamic` output to 1e-12.

## 3. Proposed promotion records (`scripts/build_parity_index.py::_FROZEN_PROMOTIONS`)

```python
    "continuous_did": {
        "status": "bit-exact",
        "reference": "fixest::feols(y ~ dose:post | id + time) (method='twfe')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "fixest": "0.14.0",
        },
        "tolerance": "slope & SE 1e-9 rel (observed 2.2e-15), iid / ~id / ~region, balanced + unbalanced",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_didvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_didvar_R.json",
        ],
        "note": (
            "Grade covers method='twfe' only. iid SE divides by n-K with K "
            "counting every absorbed unit/period parameter; clustered SE uses "
            "G/(G-1)(n-1)/(n-K) with fixest's nested-FE rule. The default "
            "method='att_gt' (dose-bin 2x2 rollup) and 'dose_response' are "
            "heuristics with no package reference: pinned by an analytic 2x2 "
            "SE and exact linear-slope recovery instead. Not the CGS "
            "ATT(d|g,t) estimator -- that is sp.cgs_continuous_did (contdid). "
            "Regenerate via _generate_did_synth_didvar_R.R."
        ),
    },
    "did_timevarying_covariates": {
        "status": "bit-exact",
        "reference": "ptetools::pte_default(d_outcome=TRUE, est_method='reg') and did::att_gt(est_method='reg')",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "ptetools": "1.0.1",
            "did": "2.3.0",
            "DRDID": "1.2.3",
        },
        "tolerance": "ATT(g,t) and overall ATT 1e-9 rel (observed 3.8e-15)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_didvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_didvar_R.json",
        ],
        "note": (
            "X_{g-1} outcome-regression estimator, never-treated comparison: "
            "every post-period ATT(g,t), the group-aggregated overall ATT "
            "(ptetools overall / did aggte type='group') and the simple "
            "aggregate (did aggte type='simple'), with two covariates and "
            "one. ptetools and did agree with each other to ~1e-14. Point "
            "estimates only: the SE is a unit bootstrap here and a "
            "multiplier bootstrap in ptetools (T3), not compared."
        ),
    },
    "distributional_did": {
        "status": "bit-exact",
        "reference": "didFF::distDD 0.1.0 (Roth & Sant'Anna)",
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "didFF": "0.1.0",
            "did": "2.3.0",
        },
        "tolerance": "per-bin effect & SE 1e-9 (observed est 2.6e-12 abs, SE 2.5e-10 rel)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_didvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_didvar_R.json",
            "tests/reference_parity/test_functional_form_extended_parity.py",
            "tests/reference_parity/_fixtures/didff_extended_reference.json",
        ],
        "note": (
            "Fourteen configurations on did::mpdta: nbins / binpoints / "
            "discrete binning, weights, covariates under dr / reg / ipw, "
            "not-yet-treated comparisons, simple / group / dynamic / "
            "calendar aggregation and the dynamic event window. distDD "
            "itself crashes when a bin's influence function is degenerate "
            "(balance_e=1 here); that case is pinned against distDD's recipe "
            "re-run step by step in R, which first reproduces distDD's own "
            "dynamic output to 1e-12. Largest gaps are the propensity-score "
            "cases (logit iterated to tolerance on both sides)."
        ),
    },
```

## 4. Proposed CHANGELOG / MIGRATION

CHANGELOG — ⚠️ Correctness:

- `sp.continuous_did(method="twfe")`: the iid standard error now divides by `n - K` with `K` counting the absorbed unit and period effects (it used `n - 1`, understating the SE by ~9% on a 120 × 6 panel); the clustered SE's `(n-1)/(n-K)` factor now counts period effects not nested in the cluster (fixest `ssc(fixef.K="nested")`); and unit/period effects are absorbed by alternating projections, so the slope is correct on unbalanced panels (the one-pass double-demeaning was exact only on balanced ones). Now matches `fixest::feols` to 1e-15.
- `sp.continuous_did(method="att_gt")`: the bootstrap kept each drawn unit once regardless of how often it was drawn, and the pooled SE assumed the dose bins were independent although they share one comparison arm. Both fixed (multiplicity-weighted draws; bins and pooled estimate bootstrapped jointly). Pooled SE rises (0.106 → 0.185 on the test panel); point estimates unchanged. With no zero-dose units the SE was NaN; now finite.
- `sp.continuous_did(method="dose_response")`: the reported SE was the average pointwise SE of the fitted *level*, not an SE of the average slope it accompanies; it is now a unit bootstrap of the average slope. The `linregress` fallback now warns instead of switching silently.
- `sp.did_timevarying_covariates`: comparison units' covariates were frozen at their median period instead of the cohort's `g-1`, and the adjustment was a pooled `dY ~ D + X` regression rather than outcome regression on the comparison units. Rewritten as the `X_{g-1}` outcome-regression ATT(g,t) of `ptetools::pte_default(d_outcome=TRUE, est_method="reg")` / `did::att_gt(est_method="reg")`, matched to 1e-14. The headline aggregate is now the group aggregation (ptetools overall ATT); new `aggregation="simple"` gives the treated-count-weighted aggregate.

CHANGELOG — Added:

- `sp.did_timevarying_covariates(aggregation="group"|"simple")`; `model_info` carries both `att_group` and `att_simple`.
- `continuous_did` results: `model_info["dof_K"]`, `["control_arm"]`, `["dose_response_pointwise_se"]`, `["se_method"]`.
- Reference parity file `tests/reference_parity/test_did_synth_didvar_parity.py` (fixest, ptetools, did, didFF).

CHANGELOG — Fixed:

- `continuous_did(method="att_gt"|"dose_response")` warn that `controls` are ignored (previously silently dropped).

MIGRATION rows (default-output changes):

| function | what changed | old → new (fixture panel) | how to get the old number |
| --- | --- | --- | --- |
| `sp.continuous_did(method="twfe")` | iid / clustered SE df; unbalanced-panel slope | iid SE 0.042912 → 0.047212; unbalanced slope 0.403496 → 0.402988 | not reachable; old values were not a documented quantity |
| `sp.continuous_did(method="att_gt")` (default) | bootstrap SEs | pooled SE 0.10639 → 0.18500 | not reachable (old bootstrap was a subsample) |
| `sp.continuous_did(method="dose_response")` | SE definition | 0.24308 → 0.15439 | old number = `mean(model_info["dose_response_pointwise_se"])` |
| `sp.did_timevarying_covariates` | estimator and default aggregation | 1.273390 → 1.257705 | not reachable (old cells used mismatched covariate periods); `aggregation="simple"` gives the treated-count-weighted aggregate of the corrected cells |

Registry / schema (integrator, forbidden for me): `src/statspai/registry.py` entry for `did_timevarying_covariates` needs `ParamSpec("aggregation", "str", False, "group", ...)` and its description ("Aggregates via cohort-size weights", "[待核验]") updated; the `continuous_did` signature is unchanged. Schema dump will drift for `did_timevarying_covariates`.

## 5. Not closed

- `continuous_did(method="att_gt")` and `method="dose_response"`: class 6. These are StatsPAI heuristics; no package computes the dose-quantile 2x2 rollup or the grid-average local-linear slope. The CGS estimand is in `sp.cgs_continuous_did` (Track A 80). `contdid` 0.1.1 targets ATT(d)/ACRT(d), a different estimand; not used. Evidence is reference-free (analytic 2x2 SE within MC error; exact linear-slope recovery).
- `continuous_did(method="twfe")` p-values / CIs use the normal; fixest uses t(n-K) (iid) / t(G-1) (cluster). Documented convention (class 3); only estimates and SEs are compared.
- `continuous_did(method="cgs")`: deprecated MVP, superseded by `cgs_continuous_did`; not examined.
- `did_timevarying_covariates` SE: class 5 (T3). Unit bootstrap vs `ptetools` multiplier bootstrap; not compared. An analytic influence-function SE (IFs are available from `DRDID::reg_did_panel`) would make it deterministic and pinnable; not done.
- `did_timevarying_covariates` scope still limited to OR + never-treated (ptetools also offers `est_method="dr"`, not-yet-treated, `d_covs_formula`, `lagged_outcome_cov`).

## 6. .gitignore

None needed for this sub-cluster (no private ado dir or R library created; `_ado_did_synth/` and `_rlib_did_synth_scpi/` belong to other sub-clusters).
