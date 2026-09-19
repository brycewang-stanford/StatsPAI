# did_synth / `synthvar`: `sp.staggered_synth`, `sp.robust_synth`, `sp.demeaned_synth`, `sp.discos`

Cluster tag `synthvar`. Worktree `pc-did-synth`; nothing committed.

Reference fixture: `tests/reference_parity/_fixtures/did_synth_synthvar_R.json`, written by
`tests/reference_parity/_generate_did_synth_synthvar_R.R` (R 4.5.2) from the CSVs written by
`tests/reference_parity/_generate_did_synth_synthvar_data.py` (fixed seeds):
`_fixtures/did_synth_synthvar_{single,stag,micro}.csv`. Test:
`tests/reference_parity/test_did_synth_synthvar_parity.py` (31 tests).

## 1. Per-function table

| function | reference (version) | outcome class | max rel err (est / SE) | test file |
| --- | --- | --- | --- | --- |
| `sp.staggered_synth` | `augsynth::multisynth` 0.2.0 (Ben-Michael, Feller & Rothstein), OSQP 1.0.0 at eps 1e-12 | **2** (wrong estimator under the right name + contaminated donors; rewritten, then T2) | ATT 5.6e-12, per-unit/cohort ATT 2.6e-11, weights 2.1e-10 abs, nu / imbalance norms ~1e-11; jackknife SE 1.8e-11 (6 configurations: nu 0 / 0.5 / auto, fixedeff on/off, time_cohort, n_lags, lambda, R defaults) | `tests/reference_parity/test_did_synth_synthvar_parity.py` |
| `sp.robust_synth` | `scpi::scest(w.constr=list(name="ols"))` + `scdata(constant=TRUE)` 4.0.1; `stats::lm`; `glmnet` 4.1.10 | **2** (elastic-net path defect + placebo p-value defect, fixed; then T2 on the regression paths) | OLS weights 8.4e-13, intercept 2.8e-15, fitted path 2.6e-16; ridge / lasso / EN vs glmnet: weights 3.4e-11 abs, intercept 4.7e-12. SE: placebo spread, no reference | same |
| `sp.demeaned_synth` (`variant="demeaned"`) | `augsynth::augsynth(progfunc="None", fixedeff=TRUE)` 0.2.0 | **2** (placebo p-value defect fixed; point estimate was already right) → aligned | ATT 1.2e-9, gap path 8.2e-9 abs, weights ≤1e-8 abs. The limit is augsynth's `synth_qp`, which hard-codes OSQP eps 1e-8. SE: placebo spread, no reference | same |
| `sp.demeaned_synth` (`variant="detrended"`) | none found | **6** (reference-free identity only) | exact-fit DGP recovers the effect to 1e-6 | same |
| `sp.discos`, individual-level data, `method="quantile"` | `DiSCos::DiSCo(mixture=FALSE)` 0.1.4 (pracma 2.4.6 → quadprog 1.5.8) | **2** (wrong estimator under the right name; new Gunsilius path, then T2) | weights 4.9e-13 abs, counterfactual quantile functions 1.9e-12 abs, DiSCoTEA quantile differences 1.9e-12 abs, estimate 1.8e-14; `simplex=TRUE` about 1e-14 | same |
| `sp.discos`, `method="mixture"` | `DiSCos::DiSCo(mixture=TRUE)` (CVXR 1.8.2 / SCS), and the same LP by GLPK (Rglpk 0.6.5.1 via CVXR) | **2** → T2 vs GLPK; aligned vs DiSCo (named mechanism: SCS eps 1e-6) | LP weights vs GLPK 2.8e-16; vs DiSCo's SCS weights 1.6e-6 per period / 7.1e-7 averaged; counterfactual quantiles and quantile effects exact (0 / 2.2e-16) | same |
| `sp.discos` permutation test | `DiSCos:::DiSCo_per` | **4** (reference defect, with the mechanism reproduced) | StatsPAI placebo distances = one-line-patched DiSCo_per_iter 8.3e-12; zeroing the column in StatsPAI's own pieces reproduces unpatched R 1e-9; p-value equal (1/6) on this data | same |
| `sp.discos`, aggregate panels (1 row per unit-period) | none (the estimator is undefined there) | **6** | now warns and labels `model_info["estimator"]="time_series_quantiles_fallback"` | same |

Reference-free assertions: exact-fit DGPs for `demeaned_synth` (both variants), `staggered_synth` and `discos` (treated unit = copy of a donor, so w = e_j and zero effect). Also the continuity of the elastic net at `l1 → 0`, and placebo-statistic consistency for the MSPE-ratio p-values: donor k's placebo ratio must equal the ratio from a direct fit with k treated.

## 2. Defects found

### D1. `staggered_synth`: donors were contaminated, and the estimator was not Ben-Michael et al. (wrong estimator under the right name)

- **What.**
  - The method label and docstring claim "partially pooled SCM (Ben-Michael, Feller & Rothstein 2022)". The code fitted an independent per-unit SCM on calendar pre-periods. It had no `nu`, no event-time alignment and no `fixedeff`.
  - `method="pooled"` fitted the cohort average, which is multisynth `time_cohort=TRUE` with `nu=0`. That is not partial pooling.
  - **Bug:** the donor pool for cohort g was "never-treated + units adopting after g". The effect window ran to the end of the panel, so later adopters' *treated* outcomes entered earlier cohorts' counterfactuals.
  - The ATT weighted units by their number of post-periods. multisynth's `Average` is the unit-level (`n1`-weighted) mean of each unit's window-average effect.
- **How found.** I read `multisynth`, `multisynth_formatted`, `multisynth_qp`, `make_Pmat`, `make_qvec`, `get_eligible_donors`, `fit_feff`, `predict.multisynth` and `jackknife_se_multi` from augsynth 0.2.0's namespace. The first divergence is the donor set (`get_eligible_donors`: `trt > n_leads + grps[j]`), and after that the objective.
- **Fix.** Rewrote the estimator as multisynth's QP:
  - Parameters: `nu` (float or `"auto"`, multisynth's heuristic), `fixedeff`, `n_leads`, `n_lags`, `penalization` (= multisynth `lambda`), `method` (`"separate"`/`"pooled"` = `time_cohort`), and `se_method="jackknife"` (= `summary(inf_type="jackknife")`).
  - The QP is solved exactly by a primal active-set method (`_active_set_qp`), which replaces SLSQP: SLSQP took 3.6 s per fit.
  - New `model_info` keys: `event_study`, `weights` (unit × group), `nu`, `global_l2`, `ind_l2`, `jackknife_atts`.
  - Validation is now explicit: non-absorbing treatment, unbalanced panels, first-period adopters and no eligible donor all raise.
  - Defaults keep StatsPAI's old choices where they are not bugs: `nu=0` (separate), `fixedeff=False`, and a full window to the end of the panel (`n_leads=None`), which restricts donors to never-treated units.
- **Default output changed: ⚠️ correctness fix.** On `did_synth_synthvar_stag.csv`:
  - `method="separate"`: 2.664881 → 2.597125. Unit 1's ATT was 3.5145 with a cohort-13 donor; it is now 3.6477.
  - `method="pooled"`: 2.597259 → 2.526639.
  - The true full-window ATT is 2.375. `fixedeff=True` gives 2.594, and `fixedeff=True, nu="auto"` gives 2.608. This is one noisy draw, not a bias study; the fixture's 6-period window has truth 1.625 and multisynth configurations there give 1.72–1.83 (Python = R).
  - The placebo SE is unchanged here (1.0980).
- **Semantics change (non-default):** `penalization` is now multisynth's `lambda` in the scale of the normalised objective. Before, it was a ridge weight on the raw per-unit SSR.

### D2. `robust_synth(variant="elastic_net", l1_penalty>0)`: the intercept was penalised and l1 was doubled

- **What.**
  - The ridge path left the intercept unpenalised, but the coordinate-descent path (used whenever `l1_penalty > 0`) penalised the intercept column with both L1 and L2.
  - It also soft-thresholded at `l1` for an objective documented as `RSS + l2‖β‖² + l1‖β‖₁`, which needs a threshold of `l1/2`.
  - Its convergence test was an absolute 1e-8 with a stray `break  # pragma: no cover`.
- **How found:** vs `glmnet`. Also by continuity: at `l2=5`, `l1_penalty=1e-12` moved the intercept from 20.7388 to **0.5346** and the ATT from 3.0336 to **1.9978**. The fixed code gives 20.7388 / 3.0336 for both.
- **Fix.**
  - The intercept is profiled out by centring. CD uses the partial-residual update `S(x'r, l1/2)/(x'x + l2)` with a relative 1e-13 stopping rule and warns if it does not converge.
  - Ridge / OLS use `lstsq`, so `l2=0` is plain OLS.
  - Now matches glmnet to 1e-11 (the λ/α mapping is in the generator).
- **Default output changed:** no. The default is `l1_penalty=0`. Output changes only for users who set `l1_penalty>0`: **⚠️ correctness fix** for that path.

### D3. `demeaned_synth` and `robust_synth`: the placebo p-value compared different statistics

- **What.** The treated unit's statistic was `mean(post gap²)/pre MSPE`. Each placebo's was `mean(post gap)²/pre MSPE`, the square of the mean. By Jensen the treated statistic is inflated, so the p-value is anti-conservative. A perfect pre-fit also gave the treated unit `inf` but a placebo `0`.
- **How found:** reading the code, then a null DGP. Removing the effect from the single-treated panel, HEAD gave the minimum attainable p = 1/9 for both functions.
- **Fix.** Both functions use the same post/pre MSPE ratio via `_mspe_ratio`. The treated ratio and the placebo ratios are stored as `model_info["mspe_ratio"]` / `["placebo_mspe_ratios"]`. A test checks that donor k's placebo ratio equals the ratio from a direct fit with k treated.
- **Default output changed: ⚠️ correctness fix (p-value only).**
  - Null DGP: `demeaned_synth` p 0.111 → 0.222; `robust_synth` p 0.111 → 0.444.
  - With the true effect both stay at 0.111.
  - California Prop 99: both stay at 1/39. Point estimates are unchanged apart from solver noise: demeaned −17.6396418 vs −17.6396418, relative difference 4e-10.

### D4. `discos`: wrong estimator under the right name (individual-level data was averaged away)

- **What.** `sp.discos` pivoted with `pivot_table` (mean), so individual-level data (the setting of Gunsilius 2023 and of R `DiSCos`) was **silently averaged to cell means**. It then treated each unit's *time series* of means as its "distribution". It never computed a cross-sectional quantile function.
  - Its `"mixture"` was a simplex-constrained quantile regression, which is R's `mixture=FALSE, simplex=TRUE`.
  - Its `"quantile"` was unconstrained OLS without the adding-up constraint.
  - Neither matches R's definitions.
- **How found:** from `DiSCo`, `DiSCo_iter`, `DiSCo_weights_reg`, `DiSCo_mixture(_solve)`, `DiSCo_per(_iter)` and `DiSCoTEA` printed from the DiSCos 0.1.4 namespace.
- **Fix: new individual-level path** (`_discos_micro`), used whenever some unit-period has more than one row:
  - Per-pre-period weights, averaged: type-7 quantile regression with `sum=1`, `w≤1`, and `w≥0` iff `simplex`. This is the problem `pracma::lsqlincon → quadprog` solves; StatsPAI uses an exact active set in null-space least-squares form.
  - Or the CDF-mixture L1 LP, solved with HiGHS.
  - Counterfactual quantile functions on `0, 1/G, …, 1` for every period. In mixture mode they come from DiSCo's CDF inversion (`≥ τ − 1e-5` on the grid).
  - DiSCoTEA's quantile-range table without R's 4-decimal rounding (`quantile_effect_summary`).
  - The permutation test (`model_info["permutation"]`).
- **Randomness.** R draws the quantile nodes and the CDF grid at random. StatsPAI's defaults are deterministic: `M` mid-points, and `G` equispaced points on R's range. The new keyword arguments `q_nodes=` / `cdf_grid=` take R's draws. The generator replays R's L'Ecuyer stream call for call (`runif(G)`, then `runif(M)` per period; then `runif(M)` per (control, pre-period) for the permutation) and verifies the replay against DiSCo's own weights.
- **Defaults on individual-level data follow R:** `method="quantile"`, `simplex=False`, `M=1000`, `G=1000`.
- **Aggregate panels.** The old heuristic is kept as a fallback because `sp.synth(method="discos")`, `synth_compare` and `synth_report` call it on state-year panels. It now emits a `UserWarning` saying it is not the Gunsilius estimator, and `model_info["estimator"]` names it.
- **Default output changed: ⚠️ correctness fix** for individual-level data (`did_synth_synthvar_micro.csv`, true average effect 0.9):
  - `method="mixture"`: 0.921300 → 0.967711
  - `method="quantile"`: 1.249059 → 0.924513
  - Weights are entirely different (HEAD `quantile` weights did not sum to 1).
- **Aggregate panels, also changed:** the fallback's `_mixture_weights` used finite-difference SLSQP, which ran to its iteration cap on this badly scaled objective (~30 s for Prop 99) and stopped short of the optimum. It is now the exact active set: California −23.0638600 → −23.0638480 (5e-7), 30 s → 0.7 s.
- **Unchanged contract:** at least 2 pre-periods are still required. R accepts 1; the existing test `test_discos_insufficient_pre_raises` pins StatsPAI's rule.

### D5. `DiSCos` 0.1.4 permutation test bug (reference defect, T4; not copied)

- **What.** `DiSCo_per_iter` builds each placebo's donor pool with the original treated unit first. It fits the weights on that pool but fills the pool's quantile matrix only from column 2:

  ```r
  perc.q[[i]] <- matrix(0, ncol = length(c_df[[i]]), nrow = length(evgrid))
  perc.q[[i]][, j + 1] <- c_df.q[[i]][, keepcon[j]]
  ```

  Column 1 stays zero, so every placebo counterfactual drops a donor that carries weight (quantile mode only; the mixture mode uses the full CDF matrix).
- **Evidence.**
  1. StatsPAI's placebo Wasserstein distances equal a copy of `DiSCo_per_iter` in the generator with that one column filled, run on R's replayed nodes, at 8.3e-12.
  2. Zeroing the column in StatsPAI's own building blocks reproduces R's unpatched distances at 1e-9.
  3. Patched and unpatched differ by up to 99.6%.
- The p-value (rank / (J+1)) happens to be the same on this data (1/6). This should be reported upstream.

### D6. Minor

- **Silently ignored argument:** `covariates=` in `robust_synth` / `demeaned_synth` was ignored. It now raises `NotImplementedError`.
- **Rounded returned values:** `model_info["pre_treatment_mspe"/"pre_treatment_rmse"]` (robust, demeaned) and `["pre_rmsqe"]` (discos) were `round(…, 6)`. They are now full precision.
- **Mislabelled estimator:** `robust_synth(variant="penalized")` was labelled "Penalized SCM (Abadie & L'Hour 2021)". It is simplex + ridge, and the L1 term is constant on the simplex. It is relabelled "Ridge-penalized simplex SCM", and the module docstring says what Abadie–L'Hour is and points to `sp.synth(method="penalized")`. The Abadie–L'Hour citation was removed from this module; `[@abadie2021penalized]` is still in `paper.bib`.
- **Duplicated solvers:** the local simplex solvers in `demeaned.py` / `robust.py` were removed in favour of `_core.solve_simplex_weights` (demeaned ATT moves ~1e-9). `staggered._solve_weights` is kept as an alias of that function, because `tests/test_cov95_synth_r4_staggered.py` tests it directly (see §5).
- **Swallowed exceptions:** the bare `except Exception: continue` handlers in the placebo loops were narrowed to `ValueError` / `LinAlgError`.

## 3. Proposed promotion records (`_FROZEN_PROMOTIONS`)

```python
    "staggered_synth": {
        "status": "bit-exact",
        "reference": (
            "augsynth::multisynth 0.2.0 (Ben-Michael, Feller & Rothstein; "
            "partially pooled SCM for staggered adoption)"
        ),
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "augsynth": "0.2.0",
            "osqp": "1.0.0",
        },
        "tolerance": (
            "ATT, per-unit / per-cohort ATT, event-time ATT and jackknife SE "
            "1e-8 rel (observed <= 2.6e-11); weights 1e-8 abs (observed "
            "2.1e-10); nu 1e-8, imbalance norms 1e-6 rel"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": (
            "Six multisynth configurations on one staggered panel: nu = 0 / "
            "0.5 / auto, fixedeff on and off, time_cohort = TRUE, n_lags = 5 "
            "with lambda = 0.1, and multisynth's own defaults; jackknife SE "
            "via summary(inf_type = 'jackknife'). multisynth run with OSQP at "
            "eps 1e-12; StatsPAI solves the same QP exactly by active set. "
            "The placebo SE (StatsPAI's default se_method) has no reference. "
            "Regenerate via _generate_did_synth_synthvar_R.R."
        ),
    },
    "robust_synth": {
        "status": "bit-exact",
        "reference": (
            "scpi::scest(w.constr = list(name = 'ols')) with scdata(constant = "
            "TRUE) and stats::lm (unconstrained SC with intercept); glmnet "
            "(ridge / lasso / elastic net, unpenalised intercept)"
        ),
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "scpi": "4.0.1",
            "glmnet": "4.1.10",
        },
        "tolerance": (
            "OLS weights / intercept / fitted path 1e-10 rel (observed "
            "8.4e-13); penalised paths 1e-8 rel, weights atol 1e-10 "
            "(observed 3.4e-11 abs, glmnet's coordinate-descent stop)"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": (
            "Covers variant='unconstrained' (l2 = 0 and the default 0.01) and "
            "'elastic_net'. glmnet is compared after scaling y to unit (1/n) "
            "SD with n*lambda*(1-alpha) = l2 and 2*n*lambda*alpha = l1/sd(y). "
            "variant='penalized' (simplex + ridge) and the placebo SE have no "
            "reference. No canonical package implements Doudchenko & Imbens' "
            "CV-tuned estimator end to end; the penalty is user-set here."
        ),
    },
    "demeaned_synth": {
        "status": "aligned",
        "reference": (
            "augsynth::augsynth(progfunc = 'None', fixedeff = TRUE) 0.2.0 "
            "(de-meaned SCM)"
        ),
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "augsynth": "0.2.0",
            "osqp": "1.0.0",
        },
        "tolerance": (
            "gap path and ATT 1e-7 rel (observed 1.2e-9); weights 1e-8 abs"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": (
            "variant='demeaned' only. Limited by augsynth's synth_qp, which "
            "hard-codes OSQP at eps_abs = eps_rel = 1e-8 (R's zero weights "
            "come back as +-1e-9). variant='detrended' has no reference and "
            "is covered by an exact-fit identity."
        ),
    },
    "discos": {
        "status": "bit-exact",
        "reference": (
            "DiSCos::DiSCo 0.1.4 (Gunsilius distributional synthetic "
            "controls), mixture = FALSE; mixture = TRUE vs GLPK on DiSCo's LP"
        ),
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "DiSCos": "0.1.4",
            "pracma": "2.4.6",
            "quadprog": "1.5.8",
            "CVXR": "1.8.2",
            "Rglpk": "0.6.5.1",
        },
        "tolerance": (
            "quantile weights 1e-11 abs (observed 4.9e-13), counterfactual "
            "quantile functions 1e-10 rel, quantile effects 1e-10 abs "
            "(observed 1.9e-12); mixture LP weights vs GLPK 1e-12 abs, vs "
            "DiSCo's SCS solution 5e-6 abs (observed 7.1e-7)"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_synthvar_parity.py",
            "tests/reference_parity/_fixtures/did_synth_synthvar_R.json",
        ],
        "note": (
            "Individual-level data only (the aggregate-panel fallback is a "
            "StatsPAI heuristic and warns). R's random quantile nodes / CDF "
            "grids are replayed from its L'Ecuyer stream and passed via "
            "q_nodes / cdf_grid. Permutation test: T4 -- DiSCos 0.1.4's "
            "DiSCo_per_iter leaves the original treated unit's quantile "
            "column at zero; StatsPAI matches the one-line-patched function "
            "(8.3e-12) and reproduces the unpatched numbers when it zeroes "
            "that column (1e-9)."
        ),
    },
```

`_check_external_evidence`: each listed test loads `_fixtures/did_synth_synthvar_R.json`.

## 4. Proposed CHANGELOG / MIGRATION lines

**CHANGELOG**

- *Added*
  - `sp.staggered_synth(nu=, fixedeff=, n_leads=, n_lags=, se_method="jackknife")`: Ben-Michael, Feller & Rothstein's partially pooled SCM, matching `augsynth::multisynth` 0.2.0 at ~1e-11 (ATT, weights, event-time effects, jackknife SE). Also new `model_info["event_study"]`.
  - `sp.discos` on individual-level data: Gunsilius' distributional SC matching `DiSCos::DiSCo` 0.1.4 (quantile weights, counterfactual quantile functions, CDF-mixture weights). New arguments `M=`, `simplex=`, `q_nodes=`, `cdf_grid=`, plus DiSCo's permutation test.
  - `model_info["mspe_ratio"]` / `["placebo_mspe_ratios"]` in `demeaned_synth` / `robust_synth`.
  - Parity suite `tests/reference_parity/test_did_synth_synthvar_parity.py`, covering augsynth, multisynth, scpi, lm, glmnet, DiSCos and GLPK.
- *⚠️ Correctness*
  - `sp.staggered_synth`: donors were never-treated plus *not-yet-treated at adoption*, while effects ran to the end of the panel, so later adopters' treated outcomes entered earlier cohorts' counterfactuals. Donors now follow multisynth: untreated through the effect window. The ATT is now multisynth's unit-level average, where it used to be weighted by post-period count. Default output changes (fixture panel: separate 2.6649 → 2.5971, pooled 2.5973 → 2.5266).
  - `sp.discos`: individual-level data used to be averaged to unit-period means and then handled as a time series. It now runs the Gunsilius estimator (fixture: `quantile` 1.2491 → 0.9245, `mixture` 0.9213 → 0.9677; truth 0.9). The aggregate-panel fallback now warns.
  - `sp.demeaned_synth`, `sp.robust_synth`: placebo p-values compared the treated unit's post/pre MSPE ratio with the placebos' *squared mean* post gap over pre MSPE, which was anti-conservative. On a null DGP p went from 0.111 → 0.222 (demeaned) and 0.111 → 0.444 (robust). Point estimates are unchanged.
  - `sp.robust_synth(variant="elastic_net", l1_penalty>0)`: the intercept was penalised and the L1 threshold doubled. At `l1_penalty=1e-12` the intercept jumped from 20.74 to 0.53. It now matches glmnet.
- *Fixed*
  - `covariates=` was silently ignored by `robust_synth` / `demeaned_synth` and now raises.
  - `model_info` pre-fit MSPE / RMSE / RMSQE are no longer rounded to 6 decimals.
  - `robust_synth(variant="penalized")` is no longer labelled Abadie & L'Hour (2021).
  - The `discos` aggregate-panel weights are now solved exactly: about 40× faster, and the estimate moves by 5e-7.

**MIGRATION** (rows)

| function | change | before → after | how to get the old number |
| --- | --- | --- | --- |
| `sp.staggered_synth` (default) | donor pool excludes units treated inside the effect window; ATT = unit-level mean | 2.6649 → 2.5971 (fixture) | not reproducible (contaminated donors) |
| `sp.staggered_synth(penalization=)` | now multisynth's `lambda` on the normalised objective | — | none (the old scale had no reference) |
| `sp.discos` on individual-level data | Gunsilius estimator; default `method` is `"quantile"` there | 1.2491 → 0.9245 (fixture) | aggregate the data to unit-period means first (then the fallback runs, with a warning) |
| `sp.demeaned_synth` / `sp.robust_synth` p-value | consistent MSPE ratio | 0.111 → 0.222 / 0.444 (null DGP) | not reproducible (the old statistic was inconsistent) |
| `sp.robust_synth(l1_penalty>0)` | unpenalised intercept, `l1` = coefficient of `‖w‖₁` in `RSS + l2‖w‖² + l1‖w‖₁` | intercept 0.53 → 20.74 at `l2=5, l1=1e-12` | old `l1` ≈ new `l1/2`, except that the intercept is no longer penalised |
| `sp.robust_synth` / `sp.demeaned_synth(covariates=...)` | raises `NotImplementedError` | — | drop the argument (it was ignored) |

## 5. Not closed / caveats

- **`demeaned_synth(variant="detrended")`: class 6.** None of the packages checked computes an SCM on unit-detrended outcomes: augsynth (`fixedeff` de-means only), scpi (`cointegrated.data` / `constant` are different), synthdid and gsynth (factor model). It is covered only by an exact-fit identity.
- **Placebo SEs** of `staggered_synth` (default `se_method`), `robust_synth`, `demeaned_synth` and `discos` are StatsPAI heuristics (the spread of placebo effects) with no reference.
  - The staggered `jackknife` SE is matched to multisynth.
  - multisynth's default wild bootstrap and DiSCos' bootstrap CIs are T3 (random) and were not attempted.
- **`discos(method="mixture")` permutation** was not compared: DiSCo's placebo LPs go through SCS, and quantile mode carries the D5 bug.
- **Doudchenko & Imbens end to end** (penalty chosen by CV across control units) has no canonical package. `robust_synth` takes the penalty as given, so only the solver paths are T2.
- **`_core.solve_simplex_weights`** (shared, not mine) uses SLSQP with an *absolute* `ftol=1e-12`. On badly scaled objectives (quantile SSR ~1e5) it runs to `maxiter` and stops ~1e-5 short. The old discos fallback hit exactly this. Suggest scaling the objective, or using an exact active set like `discos._eq_bounded_lsq` / `staggered._active_set_qp`. Every caller of `solve_simplex_weights` is exposed.
- **Existing test that targets a removed internal:** `tests/test_cov95_synth_r4_staggered.py::test_solve_weights_*` calls `staggered._solve_weights`. The estimator no longer uses it, so it is kept only as an alias of `_core.solve_simplex_weights`. The integrator may delete the alias together with those two tests.
- **Signature changes:** `staggered_synth` and `discos` gained keyword arguments. `schemas/` and `src/statspai/schemas/` need `python scripts/dump_schemas.py` (not run, per the concurrency rules).
- **File size:** `synth/discos.py` is now ~1590 lines (limit ~800). Suggested split: the individual-level estimator into `synth/_disco_core.py`, and the plotting / tests into their own file.
- **Aggregate-panel `discos`** remains reachable through `sp.synth(method="discos")`, `synth_compare` and `synth_report` (files I do not own) and warns every time. Consider deprecating it, or routing those callers to individual-level data only.
- **Upstream reports worth filing:**
  - DiSCos: the `DiSCo_per_iter` zero column (D5).
  - augsynth: `multisynth_qp` default OSQP eps 1e-4 is loose, so `multisynth` defaults give ~1e-4 weights. This is not a defect, but users should pass `eps_abs` / `eps_rel`.

## 6. `.gitignore`

Nothing new. The generator reuses `tests/reference_parity/_fixtures/_rlib_did_synth_scpi/` (CVXR 1.9.2 + highs for scpi 4.0.1), which the scpi cluster already flagged as needing a `.gitignore` line. The generator installs nothing; it runs scpi in a separate `Rscript` subprocess so that DiSCos keeps CVXR 1.8.2. No Stata side. `ssc describe`, run 2026-09-18 in Stata 18 MP batch:

- `disco`, `discos`, `augsynth`, `synth_runner`: rc 601 (not on SSC).
- `multisynth` (rc 0): L. B. Friedel's module (2026-04-24). It applies `synth` independently to each treated unit, so it is neither Ben-Michael et al.'s partially pooled QP nor its donor / event-time rules; it computes a different estimand.
- `allsynth` (rc 0): a `synth` wrapper for bias correction and *stacked* SC (Dube & Zipperer), a different estimand.
- `scul` (rc 0): LASSO SC with a CV-chosen penalty, not the fixed-penalty ridge / EN problem.
- `sdid`, `synth2` (rc 0): not these estimators.

None is a canonical implementation of the four functions, so `STATA_SKIP_REASON`-style entries should cite these facts.
