# Campaign phase 3: treatment-effects family (`teffects`)

Worktree `.claude/worktrees/pc-teffects`, all changes uncommitted.

Shared data: `tests/reference_parity/_fixtures/_generate_teffects_data.py`
writes `teffects_{cs,wide,wide_cens,long,surv}.csv` (seed 20260918, `%.17g`).
References: `tests/reference_parity/_generate_teffects_R.R` writes
`_fixtures/teffects_R.json`; `tests/reference_parity/_fixtures/_generate_teffects_stata.do`
writes `_fixtures/teffects_Stata.json`. Test:
`tests/reference_parity/test_teffects_R_parity.py` (51 tests, about 7 s).

R 4.5.2: AIPW 0.6.9.3, SuperLearner 2.0.40, tmle 2.1.1, ipw 1.3.0, geepack 1.3.13,
sandwich 3.1.1, ltmle 1.3.0, DTRreg 2.4, CMAverse 0.1.0 (GitHub BS1125/CMAverse),
geex 1.1.1, survival 3.8.3, AER 1.2.16. Stata 18 MP: `teffects`, `leebounds` 1.5 (SSC),
`tebounds` 1.8 (SJ15-2 st0386), `med4way` (GitHub anddis/med4way), `doseresponse` (SSC).

## 1. Per-function results

Outcome classes: 1 = matches the reference within tolerance (T2); 2 = defect fixed, then 1;
3 = documented convention difference; 4 = the reference is wrong or not unique;
5 = stochastic (T3); 6 = no canonical reference, or the reference computes a different estimand.

| function | reference | class | max rel err est / SE | test |
| --- | --- | --- | --- | --- |
| `aipw` | Stata `teffects aipw` (ATE, POmeans); R `AIPW::AIPW` 0.6.9.3 `stratified_fit(k_split=1)` | 2 → 1 (ATE); 4 for R's ATT | 4.4e-16 / 6.7e-16 (Stata, `se_method='sandwich'`); 2.2e-16 / 1.4e-14 (R, `'influence'`) | `test_aipw_*` |
| `multi_treatment` | Stata `teffects aipw` with a multivalued treatment (mlogit) | 2 → 1 | 2.2e-16 / 7.2e-15 | `test_multi_treatment_*` |
| `stabilized_weights` | R `ipw::ipwtm(type="all")`: binomial, and gaussian through geeglm | 1 (binary); 3 → 1 (gaussian, `density_sd='ml'`) | 1.5e-12 (binary), 3.3e-15 (gaussian) | `test_stabilized_weights_*` |
| `msm` | `ipwtm` + `lm`/`glm(quasibinomial)` + `sandwich::vcovCL(HC1)`; Stata `regress`/`logit [pw], vce(cluster)` | 1 (R); 3 (Stata logit SE factor) | 1.1e-13 / 1.5e-13 (R); 2.8e-9 / 4.5e-9 (Stata) | `test_msm_*` |
| `ipcw` | R `survival::coxph(ties="breslow")` + `basehaz` (only for `method='cox_ph'`) | 2 → 1 (`cox_ph`); 6 (`pooled_logistic`) | 2.0e-11 (weights) | `test_ipcw_*` |
| `tmle` | R `tmle::tmle` 2.1.1, end to end with glm learners (`cvQinit=FALSE`) | 3 → 1 (ATE, `q_bound=5e-4`); 6 (ATT) | 1.6e-9 / 6.5e-10 | `test_tmle_*` |
| `ltmle` | R `ltmle` 1.3-0 (glm, `variance.method="ic"`), with and without censoring | 2 → 1 | 1.3e-13 / 5.0e-13 (no censoring); 5.2e-10 / 4.4e-10 (censored) | `test_ltmle_*` |
| `gformula_ice_fn` | base-R `lm()` ICE + `geex::m_estimate` sandwich | 2 → 1 (SE); 6 (no package fits a linear ICE to a continuous end-of-follow-up outcome) | 2.2e-16 / 7.1e-12 | `test_ice_*` |
| `gformula_mc` | identity: converges to `gformula_ice_fn` with linear models | 5 | within 4 × MC SE | `test_mc_gformula_converges_to_ice_with_linear_models` |
| `policy_tree` | already Track A 70 (`policytree`) | (unchanged) | - | - |
| `policy_value` | closed form `mean(Gamma[i, pi_i])` | 6 | identity | `test_policy_value_is_mean_reward_of_chosen_action` |
| `lee_bounds` | Stata `leebounds` 1.5, as shipped and with the thresholds stored exactly | 2 → 1; 4 (shipped lower bound) | upper 2.2e-16 / 2.7e-15; exact trimming 6.7e-16 / 2.0e-15 | `test_lee_bounds_*` |
| `manski_bounds` | Stata `tebounds` (worst case, MTS, MTS+MTR, `erates(0)`) | 2 → 1 | 5.6e-17 abs | `test_manski_*` |
| `horowitz_manski` | `tebounds` worst case (same estimand, by identity) | 2 → 1 | 5.6e-17 abs | `test_horowitz_manski_equals_unconditional_worst_case` |
| `g_estimation` | R `DTRreg(method="gest")` 2.4 | 2 → 1 | 3.3e-15 (psi) | `test_g_estimation_*` |
| `four_way_decomposition` | Stata `med4way`; R `CMAverse::cmest(model="rb", "paramfunc")` | 1 (point); SE newly added | 5.3e-15 / 1.8e-15; SE 1.7e-15 (CMAverse), var 6.6e-10 (med4way) | `test_four_way_*` |
| `mediate_interventional` | R `CMAverse::cmest(model="gformula", postc=)` (`rpnde`, `rpnie`) and `rb`/paramfunc | 2 → 1 | 8.9e-16 (with `tv_confounders`), 2.3e-15 (without) | `test_interventional_effects_match_cmaverse` |
| `principal_strat` | R `AER::ivreg` (Wald LATE); SACE bounds = Lee bounds | 2 → 1 | 6.0e-15 | `test_principal_strat_*`, `test_sace_*` |
| `survivor_average_causal_effect` | Stata `leebounds` (identity with Lee bounds) | 2 → 1 | 2.2e-16 (upper) | `test_sace_equals_lee_bounds_and_uses_missing_outcomes` |
| `dose_response` | Stata `doseresponse` (Hirano-Imbens GPS) with the matching sklearn models | 1 (option path); 3 vs R `causaldrf` | 5.1e-10 (DRF) | `test_dose_response_hirano_imbens_matches_doseresponse` |

Counts: 13 functions reach class 1, 10 of them after a defect fix (12 fixes in all,
counting `principal_strat` and `survivor_average_causal_effect` separately); 1 is class 5
(`gformula_mc`); 1 is class 6 (`policy_value`). Partial class 6: the default
`pooled_logistic` path of `ipcw` and `tmle`'s ATT. Partial class 4: R `AIPW`'s ATT and the
lower bound from shipped Stata `leebounds`. `policy_tree` already had Track A coverage.

## 2. Defects

Unless a line says otherwise, "before" and "after" numbers come from the shared fixtures,
computed by running `git archive HEAD` against the patched tree.

1. **`ltmle`: wrong algorithm, binary ATE doubled** (default output changes).
   - What was wrong: pseudo-outcomes were fit by linear regression. Only the current
     treatment was set to the regime. Binary outcomes used a linearised update on
     `logit(Y)`, with `Y ∈ {0,1}` clipped to `1e-6`, so the update worked on ±13.8.
   - First divergence: binary ATE 0.757 against 0.373 from R.
   - Fix: rewrote the recursion to follow `ltmle`: quasi-binomial Q fits on the full
     history, predictions with every A set to the regime, Q bounded to `[1e-4, 0.9999]`,
     an intercept-only weighted fluctuation, and cumulative g bounded by `gbounds`.
     Censoring nodes are supported.
   - `propensity_bounds` now defaults to `(0.01, 1.0)` and bounds the cumulative
     probability, as `ltmle`'s `gbounds` does. It used to be `(0.01, 0.99)` per step.
   - Binary ATE / SE: 0.75744 / 0.03477 → 0.37280 / 0.04434 (R: 0.37280 / 0.04434).
   - Continuous ATE / SE: 1.95265 / 0.09166 → 2.04340 / 0.10320 (R: 2.04340 / 0.10320).
2. **`gformula_ice_fn`: made-up SE** (default output changes).
   - With `bootstrap=0` it reported `sd(Y)/sqrt(n)`, the SE of the observed outcome mean.
   - Now: the analytic sandwich of the stacked sequential OLS equations, checked against
     `geex`. SE 0.05909 → 0.06830 (geex 0.06830; a 400-draw bootstrap gives 0.0680).
   - The point estimate does not change. With OLS at every step and nested histories,
     leaving the past treatments at their observed values is algebraically the same as
     setting them to the strategy. Noted in the docstring, no code change.
3. **`aipw` ATT SE** (default output changes).
   - What was wrong: the influence function left out the `-tau D / p` term that the ratio
     estimator needs.
   - Fix: add the term. Checked against Python DoubleML's ATTE score on the same nuisances
     (point to 1e-16; SE equal up to n/(n−1)).
   - SE with the default cross-fit: 0.10180 → 0.08987.
   - Also: `aipw` had no way to turn off cross-fitting, so it could never reproduce
     `teffects`. Added `cross_fit` and `se_method` (see §4).
4. **`multi_treatment`** (default output changes). Three problems:
   - The "multinomial logit" GPS was sklearn `LogisticRegression()` at `C=1`, which is
     L2-penalised and not the MLE.
   - The bootstrap refit a different outcome model (50 trees) from the point estimate
     (100 trees).
   - Arms with ≤ 2 units silently got a zero outcome model.
   - Fix: a Newton-Raphson mlogit MLE; one `_point` function shared by the estimate and
     the bootstrap; a loud error for tiny arms.
   - New `outcome_model='linear'` and `se_method='influence'|'sandwich'` reproduce Stata.
   - Default ATE(1 vs 0): 1.163538 → 1.163558; bootstrap SE (B=50): 0.08920 → 0.09146.
5. **`lee_bounds`: wrong trimming count** (default output changes).
   - Kept `floor((1−p) n)` observations. That is one fewer than Lee's sample-quantile
     rule, and it matches no reference.
   - Now: `trimming='quantile'` (Lee's rule) by default, and `trimming='exact'`
     (fractional trimming at ties, which is how `leebounds` is written).
   - New `se_method='analytic'` is Lee's Prop. 3 variance as `leebounds` computes it.
   - `covariates=` used to be ignored silently; it is still ignored, but now with a
     `UserWarning`.
   - Bounds: (0.71763, 2.61867) → (0.72453, 2.61202).
6. **`survivor_average_causal_effect` / `principal_strat` SACE: bounds collapsed**
   (default output changes).
   - Under truncation by death (Y missing where S=0, which is the case these functions
     exist for), a blanket `dropna()` removed every non-survivor. That made P(S=1|D)=1,
     and the bounds collapsed to the naive survivor contrast.
   - SACE on `ys`: (1.68442, 1.68442) → (0.72453, 2.61202). This now equals `lee_bounds`
     exactly.
   - The trimming also used `round(q n)`; it now uses the shared Lee helper.
7. **`principal_strat` "Complier (LATE)"** (default output changes).
   - What was reported was the Imbens-Rubin complier mean of Y(1), not a LATE.
   - Now: E[Y(1)|c] − E[Y(0)|c], which is the Wald ratio. Empty cells no longer produce
     NaN.
   - 6.39543 → 7.35517 (AER `ivreg`: 7.35517).
8. **`manski_bounds(assumption='mts')`** (default output changes for `'mts'`).
   - `'mts'` silently imposed MTR as well. When the naive difference was negative it
     swapped the endpoints instead of reporting an empty set.
   - Now: `'mts'` is MTS alone, `'mts_mtr'` is the joint assumption, and a refuted joint
     assumption raises.
   - `'mts'`: [0, 0.32812] → [−0.336, 0.32812] (`tebounds` MTS positive: [−0.336, 0.32812]).
9. **`horowitz_manski`: strata dropped** (conflict-prone file `bounds/partial_id.py`).
   - Strata with one arm missing were skipped, which dropped their probability mass and
     made the bounds too narrow.
   - The docstring claimed conditioning tightens the bounds. For the ATE's worst-case
     bounds it cannot; they are identical.
   - Constructed case: [−0.332, 0.629] → [−0.337, 0.663], which equals the unconditional
     bounds.
   - `tests/test_article_aliases.py`'s bootstrap snapshot moves: both SEs are now
     0.016099, because the bound width is constant.
10. **`mediate_interventional` with `tv_confounders`** (default output changes).
    - It plugged in the marginal mean of L for both arms, which deleted the D→L→Y path.
    - Now: L ~ D + covariates, integrated at the intervened D.
    - IDE: 0.57441 → 1.10616; total: 0.81801 → 1.34976 (CMAverse `rpnde`/`te`: 1.10616 / 1.34976).
11. **`g_estimation`** (default output changes).
    - `propensity_covariates` was ignored silently, and the propensity was always a
      linear-probability fit.
    - Now: DTRreg's g-estimating equations with `propensity_model='logit'` (default) or
      `'linear'`. `'linear'` reproduces the old numbers.
    - psi on the fixture: (1.05636, 0.94856) → (1.05524, 0.94731) = DTRreg.
    - On NHEFS the new default gives 3.4611485591, which is What If Program 14.2's
      logistic g-estimate (R side of `tests/orig_parity/08`: 3.4611485612). The old
      linear value was 3.4626.
12. **`ipcw`** (default output changes).
    - `method='cox_ph'` crashed with a broadcasting error for more than one covariate.
      It also accumulated the Breslow hazard in data order rather than time order.
    - Censored rows got the same nonzero weight as observed rows. The code read
      `np.where(d == 1, w, w)`, even though the docstring says they get 0.
    - Truncation clipped the zeros upward.
    - Now: a Newton Cox fit with the Breslow `S_C(t−|x)`, censored rows weighted 0, and
      summary statistics computed over the uncensored rows.
    - Old censored-row weights were ≥ 0.77; they are now 0.

## 3. Proposed promotion records (`scripts/build_parity_index.py`, `_FROZEN_PROMOTIONS`)

```python
_TEFFECTS_TEST = [
    "tests/reference_parity/test_teffects_R_parity.py",
    "tests/reference_parity/_fixtures/teffects_R.json",
    "tests/reference_parity/_fixtures/teffects_Stata.json",
]
_TEFFECTS_R = {"R": "R version 4.5.2 (2025-10-31)"}

TEFFECTS_PROMOTIONS = {
    "aipw": {
        "status": "bit-exact",
        "reference": "Stata teffects aipw (ATE, POmeans); R AIPW::AIPW 0.6.9.3 stratified_fit(k_split = 1)",
        "reference_versions": {**_TEFFECTS_R, "AIPW": "0.6.9.3", "SuperLearner": "2.0.40", "Stata": "18"},
        "tolerance": "1e-10 rel on ATE, potential-outcome means and SEs (observed <= 1.4e-14)",
        "sides": ["py", "R", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": (
            "Full-sample nuisances (cross_fit=False). Stata's robust SE is the stacked "
            "M-estimation sandwich (se_method='sandwich'); R AIPW reports sd(EIF)/sqrt(n) "
            "(se_method='influence'). AIPW's stratified_fit() omits Q.model = FALSE for the "
            "propensity, so the R side forces SL.glm to binomial. R AIPW's ATT divides its "
            "control term by P(A = 0); the test rebuilds that number from the same nuisances, "
            "and the DR ATT is cross-checked against DoubleML's ATTE score."
        ),
    },
    "multi_treatment": {
        "status": "bit-exact",
        "reference": "Stata teffects aipw with a multivalued treatment (mlogit propensity)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-10 rel on both contrasts, potential-outcome means and sandwich SEs",
        "sides": ["py", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": "outcome_model='linear', se_method='sandwich'. The default GBM outcome model is not pinned.",
    },
    "stabilized_weights": {
        "status": "bit-exact",
        "reference": "ipw::ipwtm 1.3.0 (type = 'all'; binomial logit; gaussian via geepack::geeglm)",
        "reference_versions": {**_TEFFECTS_R, "ipw": "1.3.0", "geepack": "1.3.13"},
        "tolerance": "1e-10 rel on every weight (observed 1.5e-12)",
        "sides": ["py", "R"],
        "test": _TEFFECTS_TEST,
        "note": "Gaussian treatment: density_sd='ml' (geeglm's dispersion is RSS / N; the default divides by N - k).",
    },
    "msm": {
        "status": "bit-exact",
        "reference": "ipw::ipwtm + lm / glm(quasibinomial) + sandwich::vcovCL(type = 'HC1')",
        "reference_versions": {**_TEFFECTS_R, "ipw": "1.3.0", "sandwich": "3.1.1"},
        "tolerance": "coefficients and cluster SEs 1e-9 rel (observed 1.5e-13); Stata regress / logit [pw] 1e-7",
        "sides": ["py", "R", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": (
            "Cumulative, ever and current exposures; gaussian and binomial. Stata's logit "
            "cluster SE omits (N-1)/(N-k), which the test applies explicitly; the Stata side "
            "is at 1e-7 because its logits stop at Stata's default tolerance."
        ),
    },
    "ipcw": {
        "status": "bit-exact",
        "reference": "survival::coxph(ties = 'breslow') + basehaz(centered = FALSE)",
        "reference_versions": {**_TEFFECTS_R, "survival": "3.8.3"},
        "tolerance": "1e-9 rel on every weight (observed 2.0e-11)",
        "sides": ["py", "R"],
        "test": _TEFFECTS_TEST,
        "note": "method='cox_ph' only. The default method='pooled_logistic' is a complete-case IPW of the observed indicator and has no reference.",
    },
    "ltmle": {
        "status": "bit-exact",
        "reference": "ltmle::ltmle 1.3-0 (glm, gbounds c(0.01, 1), variance.method = 'ic')",
        "reference_versions": {**_TEFFECTS_R, "ltmle": "1.3.0"},
        "tolerance": "psi1 / psi0 / ATE / SE 1e-9 rel without censoring (observed 5e-13); 1e-8 with censoring nodes (observed 5e-10)",
        "sides": ["py", "R"],
        "test": _TEFFECTS_TEST,
        "note": "Binary and continuous outcomes; the censored rows are limited by R glm's default deviance tolerance.",
    },
    "gformula_ice_fn": {
        "status": "bit-exact",
        "reference": "base-R lm() ICE with geex::m_estimate 1.1.1 sandwich",
        "reference_versions": {**_TEFFECTS_R, "geex": "1.1.1"},
        "tolerance": "point 1e-10 rel, sandwich SE 1e-9 rel (observed 7.1e-12)",
        "sides": ["py", "R"],
        "test": _TEFFECTS_TEST,
        "note": "No package fits a linear ICE to a continuous end-of-follow-up outcome (gfoRmulaICE targets time-to-event risk); the SE is geex on the stacked estimating equations.",
    },
    "lee_bounds": {
        "status": "bit-exact",
        "reference": "Stata leebounds 1.5 (Tauchmann), vce(analytic)",
        "reference_versions": {"Stata": "18", "leebounds": "1.5 (2013-07-17, Tauchmann)"},
        "tolerance": "1e-10 rel on bounds and analytic variances",
        "sides": ["py", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": (
            "trimming='quantile' matches the shipped upper bound; trimming='exact' matches both "
            "bounds once the thresholds are held exactly (%21x). As shipped, leebounds keeps its "
            "threshold in a 15-digit local macro, so its tie branch never runs and its lower "
            "bound drops the quantile observation. The test rebuilds that number from the rounded "
            "threshold (reference defect, T4 for that one number)."
        ),
    },
    "survivor_average_causal_effect": {
        "status": "bit-exact",
        "reference": "Stata leebounds 1.5 (Zhang-Rubin SACE bounds = Lee bounds under monotonicity)",
        "reference_versions": {"Stata": "18", "leebounds": "1.5 (2013-07-17, Tauchmann)"},
        "tolerance": "1e-10 rel (upper bound; both bounds equal sp.lee_bounds)",
        "sides": ["py", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": "Outcome missing for non-survivors, as under truncation by death.",
    },
    "manski_bounds": {
        "status": "bit-exact",
        "reference": "Stata tebounds 1.8 (SJ15-2 st0386), erates(0)",
        "reference_versions": {"Stata": "18", "tebounds": "1.8 (SJ15-2 st0386)"},
        "tolerance": "1e-12 rel / 1e-15 abs on both bounds",
        "sides": ["py", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": "Worst case, MTS (positive selection), MTS+MTR; binary outcome in [0, 1].",
    },
    "horowitz_manski": {
        "status": "bit-exact",
        "reference": "Stata tebounds 1.8 worst-case bounds (identical estimand for ATE worst-case bounds)",
        "reference_versions": {"Stata": "18", "tebounds": "1.8 (SJ15-2 st0386)"},
        "tolerance": "1e-12 rel / 1e-15 abs on both bounds",
        "sides": ["py", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": "Averaging stratum worst-case bounds is the unconditional bound exactly; also asserted with a one-arm stratum.",
    },
    "g_estimation": {
        "status": "bit-exact",
        "reference": "DTRreg::DTRreg 2.4 (method = 'gest', treat.type = 'bin', weight = 'none')",
        "reference_versions": {**_TEFFECTS_R, "DTRreg": "2.4"},
        "tolerance": "1e-10 rel on each stage psi (observed 4.6e-15)",
        "sides": ["py", "R"],
        "test": _TEFFECTS_TEST,
        "note": "Constant blips, logit propensity (default propensity_model); with and without separate propensity_covariates.",
    },
    "four_way_decomposition": {
        "status": "bit-exact",
        "reference": "Stata med4way (yreg/mreg linear); CMAverse::cmest 0.1.0 (rb, paramfunc, delta)",
        "reference_versions": {**_TEFFECTS_R, "CMAverse": "0.1.0", "Stata": "18"},
        "tolerance": "components 1e-12 rel; delta SEs 1e-12 rel vs CMAverse (vcov='ols'); variances 5e-9 rel vs med4way (vcov='ml')",
        "sides": ["py", "R", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": "med4way fits the linear models by ML (ml maximize), so its residual variances stop at ml's tolerance (observed 6.6e-10).",
    },
    "mediate_interventional": {
        "status": "bit-exact",
        "reference": "CMAverse::cmest 0.1.0 (gformula with postc: rpnde / rpnie / te; rb paramfunc without)",
        "reference_versions": {**_TEFFECTS_R, "CMAverse": "0.1.0"},
        "tolerance": "IIE / IDE / total 1e-9 rel (observed 2.3e-15)",
        "sides": ["py", "R"],
        "test": _TEFFECTS_TEST,
        "note": "Point estimates only; SEs are bootstrap on both sides.",
    },
    "principal_strat": {
        "status": "bit-exact",
        "reference": "AER::ivreg 1.2.16 (Wald LATE); SACE bounds via Stata leebounds",
        "reference_versions": {**_TEFFECTS_R, "AER": "1.2.16"},
        "tolerance": "1e-10 rel on the monotonicity complier LATE",
        "sides": ["py", "R"],
        "test": _TEFFECTS_TEST,
        "note": "method='monotonicity'. The principal-score path is not pinned.",
    },
    "dose_response": {
        "status": "bit-exact",
        "reference": "Stata doseresponse / gpscore (Hirano-Imbens normal GPS, quadratic T and GPS with interaction)",
        "reference_versions": {"Stata": "18", "doseresponse": "SSC"},
        "tolerance": "1e-9 rel on the dose-response function at 5 doses (observed 5.1e-10)",
        "sides": ["py", "Stata"],
        "test": _TEFFECTS_TEST,
        "note": "treatment_model=LinearRegression(), outcome_model=PolynomialFeatures(2)+LinearRegression(); the default GBM path is not pinned. R causaldrf::hi_est uses the n-p sigma (convention gap).",
    },
}
```

`tmle` already has Track A module 72, which covers the targeting step with shared
nuisances. The new end-to-end rows (glm learners, `q_bound=5e-4`, observed 1.6e-9 / 6.5e-10)
can go in its `additional_tests` list: `tests/reference_parity/test_teffects_R_parity.py`.

## 4. CHANGELOG / MIGRATION

**Added**
- `sp.aipw(cross_fit=, se_method=)`: `cross_fit=False` fits the nuisances on the full sample.
  `se_method='sandwich'` reproduces Stata `teffects aipw`'s robust SE.
  `model_info['potential_outcome_means'/'_se']`.
- `sp.multi_treatment(outcome_model='linear', se_method='influence'|'sandwich')`: with a
  multivalued treatment this is Stata `teffects aipw`.
- `sp.stabilized_weights(density_sd=)`, `sp.msm(density_sd=)`: `'ml'` reproduces
  `ipw::ipwtm(family="gaussian")`.
- `sp.tmle(q_bound=)`: 5e-4 reproduces `tmle::tmle`.
- `sp.lee_bounds(se_method='analytic', trimming='quantile'|'exact')`; `trimming=` on
  `sp.principal_strat` and `sp.survivor_average_causal_effect`.
- `sp.g_estimation(propensity_model='logit'|'linear')`.
- `sp.four_way_decomposition(vcov='ols'|'ml')`, with delta-method SEs in `result.se`
  (CMAverse / med4way).
- `sp.manski_bounds(assumption='mts_mtr')`.
- `sp.ltmle` supports censoring nodes consistently with R `ltmle`.

**⚠️ Correctness fixes** (each recomputes published numbers):
- `sp.ltmle`: re-implemented as R `ltmle`'s sequential-regression TMLE. The old binary
  targeting step was invalid: ATE 0.757 against 0.373 on a test fixture. Recompute all
  `ltmle` results. `propensity_bounds` now bounds the cumulative g, with default
  `(0.01, 1.0)`.
- `sp.gformula_ice_fn(bootstrap=0)`: the SE was `sd(Y)/sqrt(n)`; it is now the M-estimation
  sandwich.
- `sp.aipw(estimand='ATT')`: the SE was missing the ratio term.
- `sp.principal_strat(method='monotonicity')`: the "Complier (LATE)" row reported
  E[Y(1)|complier]; it now reports the LATE (Wald).
- `sp.survivor_average_causal_effect` / `principal_strat` SACE: outcomes missing for
  non-survivors were dropped with their rows, which collapsed the bounds.
- `sp.lee_bounds`: trimming kept one observation too few; it now follows Lee's
  sample-quantile rule.
- `sp.manski_bounds(assumption='mts')` imposed MTR as well; that combination is now
  `'mts_mtr'`.
- `sp.horowitz_manski`: strata missing a treatment arm were dropped.
- `sp.mediate_interventional(tv_confounders=)`: the exposure → confounder → outcome path
  was omitted from the direct effect.
- `sp.multi_treatment`: the propensity was L2-penalised; the bootstrap refit a different
  outcome model.
- `sp.g_estimation`: `propensity_covariates` was ignored. The default propensity is now
  logistic (DTRreg; What If Program 14.2).
- `sp.ipcw`: censored rows now get weight 0, as documented. `method='cox_ph'` crashed or
  accumulated the hazard out of time order.

**MIGRATION rows**

| function | what moves | old → new (fixture) | how to get the old number |
| --- | --- | --- | --- |
| `ltmle` | ATE, SE, `propensity_bounds` meaning | binary ATE 0.757 → 0.373 | not reachable (invalid) |
| `gformula_ice_fn` | SE when `bootstrap=0` | 0.0591 → 0.0683 | not reachable (not a SE) |
| `aipw(estimand='ATT')` | SE | 0.1018 → 0.0899 | not reachable |
| `principal_strat` | "Complier (LATE)" | 6.395 → 7.355 | E[Y(1)\|c] = `(mu_11 p11 − mu_01 p01)/pi_c` |
| `survivor_average_causal_effect` | bounds with NaN outcomes | (1.684, 1.684) → (0.725, 2.612) | not reachable |
| `lee_bounds` | bounds (≤ one observation of mass) | (0.7176, 2.6187) → (0.7245, 2.6120) | not reachable; `trimming='exact'` for fractional |
| `manski_bounds('mts')` | lower bound | 0 → worst-case lower | `assumption='mts_mtr'` |
| `horowitz_manski` | bounds when a stratum lacks an arm | narrower → unconditional | not reachable |
| `mediate_interventional` | IDE / total with `tv_confounders` | 0.574 → 1.106 | not reachable |
| `multi_treatment` | ATEs (GPS penalty) and bootstrap SE | 1.163538 → 1.163558 | not reachable |
| `g_estimation` | psi (default propensity) | 1.0564 → 1.0552 | `propensity_model='linear'` |
| `ipcw` | censored-row weights, summary stats | ≥ 0.77 → 0 | none needed (weights for observed rows unchanged) |

## 5. Not closed / partial

- `tmle(estimand='ATT')`: `tmle::tmle`'s ATT is an iterated TMLE (`oneStepATT`, which
  updates g too). Ours applies the ATE-style fluctuation and then evaluates the DR ATT
  estimating equation. These are different estimators: 1.58179 against R's 1.58767 on the
  fixture. Class 6. Not changed.
- `ipcw(method='pooled_logistic')` (the default): it is a logistic model of the observed
  indicator with no time structure, so no IPCW package computes it. It is documented as
  complete-case IPW. Person-time IPCW with cumulative products is not implemented here
  (`sp.target_trial` has its own).
- `aipw` with the default `cross_fit=True`: the folds are drawn by
  `rng.choice(n_folds, n)`, so fold sizes are unequal. No reference uses this scheme. Left
  as is.
- `dose_response` against R `causaldrf::hi_est`: `hi_est` uses the `n − p` residual SD,
  while ours and Stata's `gpscore` use ML. Exposing the SD convention needs an edit to
  `dose_response/gps.py`, which another line is changing. Left for that line. The default
  GBM models have no reference.
- `gformula_mc`: T3 only (converges to the ICE value). Stata `gformula` and R
  `gfoRmula` are also Monte Carlo.
- `policy_value`: a closed-form mean; no package exposes it as a function.
- `principal_strat(method='principal_score')` and the `instrument=` path are not pinned.
  The candidate R package `PStrata` is Bayesian.
- `survivor_average_causal_effect` still reports `se` = half the bound width (a stand-in,
  not a standard error). I flagged it but did not change it.
- Pre-existing inaccuracies in the registry (I did not edit the registry): `aipw`'s
  `estimand` choices list `"ATC"`, which the function rejects, and its `seed` default is
  listed as `None`, while the signature default is 42.

## 6. Integration notes

- `.gitignore`: `tests/reference_parity/_fixtures/_ado_teffects/` and
  `tests/reference_parity/_fixtures/_ado_teffects_patched/`. These are private Stata ado
  directories created by the do-file. Stata's `.log` files are already covered by `*.log`.
- `src/statspai/registry.py`: `tests/test_registry.py::test_recent_handwritten_specs_match_callable_signature`
  currently fails. The integrator needs to add these specs:
  - to `aipw`: `ParamSpec("cross_fit", "bool", False, True, "Cross-fit nuisances; False = full-sample (teffects aipw)")`
    and `ParamSpec("se_method", "str", False, "influence", "'influence' or 'sandwich' (Stata teffects)", ["influence", "sandwich"])`;
  - to `principal_strat`: `ParamSpec("trimming", "str", False, "quantile", "SACE trimming rule", ["quantile", "exact"])`.

  The other signature changes (`multi_treatment`, `msm`, `stabilized_weights`, `tmle`,
  `lee_bounds`, `g_estimation`, `four_way_decomposition`,
  `survivor_average_causal_effect`, and `ltmle`'s default) need `scripts/dump_schemas.py`
  plus the registry param-drift baseline.
- `tests/orig_parity/08_nhefs_ch14_gestimation.py`: I updated the narrative only. Its
  `results/*_py.json` must be regenerated. `snmm_psi` becomes 3.46115, which now matches
  the R side's logistic value; the linear value moves to `propensity_model='linear'`. The
  rollup should be regenerated too.
- Conflict-prone file: `src/statspai/bounds/partial_id.py`. I changed only the
  `horowitz_manski` docstring and the stratum loop in `_hm_point`. I did not touch
  `dose_response/gps.py`.
- Tests run and passing: `tests/reference_parity/test_teffects_R_parity.py` plus every
  reference_parity and unit test file that touches these functions (about 1,350 tests).
  The 6 failures seen along the way were fixed; they came from tests that pinned the
  defects: the `ipcw` positivity tests, the `horowitz_manski` bootstrap snapshot, and the
  sklearn cold-import budget, which had been broken by an eager import. The only
  remaining failure is the registry spec test above.
