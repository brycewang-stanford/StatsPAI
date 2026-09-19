# did_synth / `mc`: `sp.mc_panel`, `sp.mc_synth`

Cluster tag `mc`. Worktree `pc-did-synth`; nothing committed.

## 1. Per-function table

| function | reference (version) | outcome class | max rel err (est / fit / SE) | test file |
| --- | --- | --- | --- | --- |
| `sp.mc_panel` | `MCPanel::mcnnm_fit` (GitHub susanathey/MCPanel @ `6b2706fd7c35f3266048ceb22a7e9a61ae1774da`, pkg version 0.0), `fect::fect(method="mc", CV=FALSE)` 2.4.1, `gsynth(estimator="mc")` 1.4.0; R 4.5.2 | **2** (defects fixed, then T2 bit-exact at fixed lambda) | ATT 2.1e-13, fitted matrix 2.6e-13 (all 4 FE modes x 2 lambdas vs MCPanel; two-way vs fect and gsynth); SE not compared (unit bootstrap, no reference computes it) | `tests/reference_parity/test_did_synth_mc_parity.py` |
| `sp.mc_synth` | same (MCPanel two-way and no-FE; fect two-way) | **2** (defects fixed, then T2 bit-exact at fixed lambda) | ATT 6.0e-13, fitted matrix 5.2e-14; SE not compared (placebo spread) | same |

The two are **different front ends on one solver**. Before this change each had its own copy of
soft-impute (`MCPanel._soft_impute` in `mc_panel.py`, `_soft_impute` in `synth/mc.py`, with different starting values, stopping rules and default tolerances). Both now call `statspai/matrix_completion/_core.py::mc_nnm_fit`. `test_mc_synth_and_mc_panel_share_the_solver` checks that the same mask and lambda give the same fit (1e-10). Both functions still own their inputs (staggered `treat` vs single `treated_unit`/`treatment_time`), how lambda is chosen by default (heuristic vs their own CV) and their inference (unit bootstrap vs placebo spread). `sp.synth(method="mc")` dispatches to `mc_synth`, and `sp.matrix_completion` is an alias for `mc_panel`. `sp.fect(method="mc")` (`synth/fect.py`, which I do not own) is a third, independent port of fect and was not touched.

**One minimiser, three lambda scales.** This is in `_core.py` and checked in the fixture:
StatsPAI `lambda_reg` = θ is the SVT threshold on `½‖P_O(Y−F)‖² + θ‖L‖_*`. MCPanel `lambda_L = 2θ/|O|`: its `update_L` calls `SVT(..., lambda_L*train_size/2)` on the `(1/|O|)`-scaled loss. fect `lambda = θ/(NT)`: `panel_FE` thresholds `svd(E/(T*N))` and rescales by `T*N`. fect two-way does EM with exact double-centring, while MCPanel does coordinate descent over (u, v, L). They reach the same minimiser: centring is an orthogonal projection and `‖PLQ‖_* ≤ ‖L‖_*`, so the complete-data minimiser is `SVT(CZC)`. The fixture confirms this: fect and MCPanel agree with each other to about 1e-14.

## 2. Defects found

1. **Fixed effects were never estimated (wrong estimator under the right name).** Both docstrings claim the Athey, Bayati, Doudchenko, Imbens and Khosravi (2021) estimator. Both references default to unpenalised unit and time effects: MCPanel `to_estimate_u = to_estimate_v = 1`, fect `force = "two-way"`. StatsPAI fitted pure soft-impute with no intercept. The nuclear norm therefore also penalised the outcome level and the additive effects, which pulls the counterfactual towards 0.
   - **How found:** the first divergence was already in the fitted matrix. StatsPAI at θ=8 matched MCPanel only with `to_estimate_u=v=0`, at 8e-5 because of defect 2. It could not match any MCPanel or fect two-way run.
   - **Fix:** new `fixed_effects={"two-way","unit","time","none"}` argument, defaulting to `"two-way"`. `"none"` keeps the old estimator reachable, since it is a legitimate MCPanel configuration (u=v=0).
   - **Default output changes: ⚠️ correctness fix.** On the fixture panel (true ATT 2.344):
     - `mc_panel` default: 2.64068 → 2.37710
     - `mc_panel(lambda_reg=8)`: 2.93426 → 2.38462 (MCPanel two-way gives 2.38462)
     - `mc_synth(seed=0)` default: 1.66626 → 1.32783
     - `mc_synth(lambda_reg=8)`: 1.96107 → 1.49082
     - California Prop 99, `mc_synth(placebo=False, seed=0)`: −13.119 → −17.980
2. **Convergence tolerance too loose, and non-convergence was silent.**
   - The old stopping rules were `tol=1e-5` / `max_iter=1000` (`mc_panel`) and `1e-6` / `500` (`mc_synth`), on relative change. They left about 8e-5 relative error in the ATT: old `mc_panel(lambda=8, "none")` gave 2.934260, the MCPanel minimiser is 2.934034. `mc_synth` had about 4e-5 error.
   - Hitting `max_iter` returned an unconverged fit without any warning.
   - **Fix:** defaults are now `tol=1e-10`, `max_iter=5000`, and both run in about 90 iterations on the fixture. A `RuntimeWarning` fires when the point-estimate fit hits the cap. For the bootstrap it fires only if more than 5% of reps hit the cap; the count is always stored in `model_info`. `model_info` now also carries `converged` and `n_iter`.
3. **Swallowed exceptions in `mc_synth`.** The placebo loop and the CV loop had `except Exception: continue` and `mse_total += 1e10`, which violates §3.7. Both are removed. Donors with gaps are now skipped explicitly, and NaN cells are treated as unobserved: before, an unbalanced panel sent NaN straight into the SVD.
4. **Docstring and code disagreed on the default lambda in `mc_panel`.** The docstring said `sigma*sqrt(n)`; the code uses `sd(Y_control)*sqrt(max(N,T))/10`. The docstring now states the code's formula and says that no reference uses this heuristic. The value itself is unchanged.
5. **Wrong author in the `mc_synth` citation string** ("Khosravi, Azeem"; the module docstring had "Khosravi, A."). Crossref for DOI 10.1080/01621459.2021.1891924 and the MCPanel DESCRIPTION both give Khashayar Khosravi, which matches the `paper.bib` key `athey2021matrix`. The citation string is fixed, and the module docstring now cites only `[@athey2021matrix]`.
6. **Minor.**
   - `mc_panel` returned `pvalue = 0.0` when `se == 0`; it now returns NaN.
   - `model_info["completed_matrix"]` is still the imputed Y(0) matrix, now fixed effects plus low-rank part.
   - New `model_info` keys: `low_rank_matrix`, `fixed_effects_matrix`, `lambda_mcpanel`, `lambda_fect`, `fixed_effects`.
   - `effective_rank` is now the rank of the low-rank component. Before, it was the rank of the whole completed matrix, which was the same thing when there were no fixed effects.

`tests/reference_parity/test_matrix_completion_parity.py`: anchors A to C test pure completion of a rank-2 matrix with no additive effects, so they now pass `fixed_effects="none"`. The docstring says so. All 7 tests pass.

## 3. Proposed promotion records (`_FROZEN_PROMOTIONS`)

```python
    "mc_panel": {
        "status": "bit-exact",
        "reference": (
            "MCPanel::mcnnm_fit (Athey, Bayati, Doudchenko, Imbens & Khosravi; "
            "github.com/susanathey/MCPanel) and fect::fect(method = \"mc\")"
        ),
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "MCPanel": "0.0 @ 6b2706fd7c35f3266048ceb22a7e9a61ae1774da",
            "fect": "2.4.1",
            "gsynth": "1.4.0",
        },
        "tolerance": (
            "ATT and fitted untreated matrix 1e-9 rel at fixed lambda "
            "(observed <= 2.6e-13), four fixed-effect modes x two lambdas"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_mc_parity.py",
            "tests/reference_parity/_fixtures/did_synth_mc_R.json",
        ],
        "note": (
            "Fixed lambda: StatsPAI theta = MCPanel lambda_L * |O| / 2 = fect "
            "lambda * N * T (same minimiser). fixed_effects two-way/unit/time/"
            "none = MCPanel (to_estimate_u, to_estimate_v). References run past "
            "their default stopping rules (MCPanel rel_tol = 0, 3000 sweeps; "
            "fect tol 1e-15). The bootstrap SE and the heuristic default lambda "
            "have no reference and are not compared. Regenerate via "
            "_generate_did_synth_mc_R.R."
        ),
    },
    "mc_synth": {
        "status": "bit-exact",
        "reference": (
            "MCPanel::mcnnm_fit (Athey, Bayati, Doudchenko, Imbens & Khosravi; "
            "github.com/susanathey/MCPanel) and fect::fect(method = \"mc\")"
        ),
        "reference_versions": {
            "R": "R version 4.5.2 (2025-10-31)",
            "MCPanel": "0.0 @ 6b2706fd7c35f3266048ceb22a7e9a61ae1774da",
            "fect": "2.4.1",
        },
        "tolerance": (
            "ATT and fitted untreated matrix 1e-9 rel at fixed lambda "
            "(observed <= 6.0e-13), two-way and no-FE x two lambdas"
        ),
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_mc_parity.py",
            "tests/reference_parity/_fixtures/did_synth_mc_R.json",
        ],
        "note": (
            "Single treated unit (unit 40 from period 21) masked in the same "
            "40 x 25 panel; same shared solver as mc_panel. The placebo-spread "
            "SE and the default K-fold CV lambda (own random folds) have no "
            "reference and are not compared. Regenerate via "
            "_generate_did_synth_mc_R.R."
        ),
    },
```

## 4. CHANGELOG / MIGRATION

**Added**
- `sp.mc_panel` / `sp.mc_synth`: new `fixed_effects=` argument (`"two-way"` by default, or `"unit"`, `"time"`, `"none"`). New `model_info` keys: `lambda_mcpanel`, `lambda_fect`, `low_rank_matrix`, `fixed_effects_matrix`, `converged`, `n_iter`. Both functions now share the solver in `statspai.matrix_completion._core`.
- Reference parity for `mc_panel` / `mc_synth` against R `MCPanel::mcnnm_fit` and `fect(method="mc")` / `gsynth(estimator="mc")` at fixed lambda: ATT and completed matrix agree to 6e-13 (`tests/reference_parity/test_did_synth_mc_parity.py`).

**⚠️ Correctness**
- `sp.mc_panel` / `sp.mc_synth` / `sp.matrix_completion` / `sp.synth(method="mc")` now estimate the Athey et al. (2021) matrix-completion estimator with unpenalised unit and time fixed effects. This is the default of the authors' `MCPanel` and of `fect`. Before, they fitted pure soft-impute with no intercept, so the nuclear norm shrank the outcome level towards zero. On the parity panel (true ATT 2.34) the `mc_panel` ATT at lambda 8 moves from 2.934 to 2.385. `fixed_effects="none"` keeps the old estimator.
- Default convergence tolerance tightened: `tol` goes from 1e-5 to 1e-10 in `mc_panel` and from 1e-6 to 1e-10 in `mc_synth`, and `max_iter` from 1000 or 500 to 5000. The old stopping rule left about 1e-4 relative error in the ATT. Hitting `max_iter` now raises `RuntimeWarning` instead of returning silently.

**Fixed**
- `sp.mc_synth`: removed the swallowed `except Exception` in the placebo and CV loops. NaN cells in unbalanced panels are now treated as unobserved instead of reaching the SVD. The citation string now gives the correct author (Khashayar Khosravi).
- `sp.mc_panel`: the docstring now states the actual default-lambda heuristic. `pvalue` is NaN, not 0, when the bootstrap SE is 0.

**MIGRATION rows**

| function | change | old behaviour |
| --- | --- | --- |
| `sp.mc_panel` (and `sp.matrix_completion`) | default `fixed_effects="two-way"`; `tol` 1e-5 → 1e-10, `max_iter` 1000 → 5000 | `fixed_effects="none", tol=1e-5, max_iter=1000` (the old start value was zeros; this does not matter at convergence) |
| `sp.mc_synth` (and `sp.synth(method="mc")`) | default `fixed_effects="two-way"`; `tol` 1e-6 → 1e-10, `max_iter` 500 → 5000; the CV lambda grid is now built from the fixed-effect-centred matrix | `fixed_effects="none", tol=1e-6, max_iter=500` |

## 5. Not closed / not compared

- **SEs (class 6).** `mc_panel` SE is a unit bootstrap at fixed lambda, and `mc_synth` SE is the standard deviation of placebo ATTs. MCPanel reports no SE. `fect(se=TRUE)` runs its own bootstrap, which re-selects nothing at fixed lambda but uses R's RNG and a different resampling scheme, so at best it would be T3. I did not attempt a Monte Carlo T3 comparison.
- **Data-driven lambda (class 6 / open).**
  - `mc_panel` default: `sd(Y_control)*sqrt(max(N,T))/10` is a StatsPAI heuristic that no reference uses.
  - `mc_synth` default: 5-fold CV on randomly held-out observed cells over a 15-point log grid of `s_max × [1e-3, 0.5]`. It depends on `seed`, and with `seed=None` it is not reproducible.
  - `MCPanel::mcnnm_cv` uses its own `create_folds` (`cv_ratio=0.8`, C++ RNG) and a 100-point path from `lambda_L_max`.
  - fect CV uses `cv.method="rolling"` and a `1se` rule.
  - The grids, rules and fold RNGs all differ, so the CV-selected lambdas are not comparable. Matching `mcnnm_cv` would mean adding a `lambda_reg="cv_mcpanel"` mode and replicating its C++ fold generator; I did not attempt that.
- **`mc_synth(covariates=...)` (class 6).** It partials out a pooled OLS without an intercept and with a 1e-8 ridge. That is not MCPanel's `mcnnm_wc` covariate model. It was not compared and not changed.
- **fect `force="none"`** keeps an unpenalised grand mean. No StatsPAI mode reproduces it, so it was not compared (documented in `_core.py`).

## 6. .gitignore and other integrator notes

- **No new ado or rlib dirs.** MCPanel was installed into the default R user library from a scratch clone. The GitHub HEAD bundles a 2017 `src/Eigen` that fails to compile with current clang (`Transpositions.h: no member named 'derived'`). The workaround is `rm -rf src/Eigen` before `R CMD INSTALL`, so RcppEigen's Eigen is used. This is documented at the top of `_generate_did_synth_mc_R.R` and in the fixture `meta`. MCPanel is GPL-2, but it is only a test-time reference and not a dependency.
- **Schema drift.** The `mc_panel` / `mc_synth` signatures changed (new `fixed_effects`, new `tol` / `max_iter` defaults), so `python scripts/dump_schemas.py` has to be re-run by the integrator. The registry derives these parameters from the signature, and the `test_registry_*` / `test_agent_schema` tests pass. `test_registry_param_drift` currently fails on `did_timevarying_covariates: ['aggregation']`, which is another line's work and not this change.
- **Doctest in a file I do not own.** `src/statspai/_article_aliases.py::matrix_completion` expects `2.86` and now gets `2.1`; the DGP's true effect is 2.0. Please change the expected value to `2.1`. The `psm` / `rdd` doctests in that file and the `mc_synth` doctest (`print(result.summary())` with no expected output) already failed at HEAD.
- **REFERENCES.md** (not mine) has a `matrix_completion` section that cites `mc_panel.py` line numbers, which are now stale. Worth adding a pointer to the new test.

## Files

Created:
- `src/statspai/matrix_completion/_core.py`
- `tests/reference_parity/_generate_did_synth_mc_data.py`
- `tests/reference_parity/_fixtures/did_synth_mc_panel.csv`
- `tests/reference_parity/_generate_did_synth_mc_R.R`
- `tests/reference_parity/_fixtures/did_synth_mc_R.json`
- `tests/reference_parity/test_did_synth_mc_parity.py` (25 tests)

Changed:
- `src/statspai/matrix_completion/mc_panel.py`
- `src/statspai/synth/mc.py`
- `tests/reference_parity/test_matrix_completion_parity.py`

Tests run (all green):
- new parity file, 25 passed
- `test_matrix_completion_parity.py`, 7 passed
- the mc-related subset of `test_synth_advanced`, `test_cov95_synth_variants`, `test_conformal_bcf_bunching_mc`, `test_article_aliases_round2`, `test_methods_appendix`, `test_result_conveniences`, `test_registry_negative_guidance` and `test_cov95_did_r4_triple_diff`, 46 passed
- registry and schema contract tests: all pass except the `did_timevarying_covariates` failure noted above
