# Campaign phase 3 — family `panel_glmm`

Worktree `wt/pc-panel-glmm`. All changes uncommitted. R 4.5.2, Stata 18 MP.

Fixture generators (run in this order, from `tests/reference_parity/_fixtures/`):

1. `python _generate_panel_glmm_data.py` — writes `panel_glmm_data.csv`,
   `panel_count_data.csv`, `panel_ife_data.csv`, `gmm_nl_data.csv`,
   `absorb_ols_data.csv`, `mixlogit_data.csv` (fixed seeds, `%.17g`).
2. `Rscript _generate_panel_glmm_R.R` → `panel_glmm_R.json`
   (lme4 2.0.1, glmmTMB 1.1.14 / TMB 1.9.25, ordinal 2025.12.29,
   performance 0.16.0, psych 2.6.5, pglm 0.2.4, phtt 3.1.2 from the CRAN
   archive, gmm 1.9.1).
3. `stata-mp -b do _generate_panel_glmm_stata.do` → `panel_glmm_Stata.json`
   (official `me*`, `mixed`, `estat icc`, `lrtest`, `xtnbreg`, `gmm`; SSC
   `regife`, `reghdfe`, `ftools`, `require`, `mixlogit` in the private ado
   directory `_fixtures/_ado_panel_glmm/`). `general_gmm_data.csv` is the
   existing fixture of `_generate_general_gmm_R.R`.

Tests: the seven new files pass (111); the related unit suites
(`test_multilevel.py`, `test_count_panel_nbreg.py`, HDFE / panel / registry
contract tests) plus all of `tests/reference_parity/` pass (3904 passed, 2
pre-existing xfails):

```
tests/reference_parity/test_panel_glmm_parity.py
tests/reference_parity/test_panel_icc_lrtest_parity.py
tests/reference_parity/test_panel_xtnbreg_parity.py
tests/reference_parity/test_panel_ife_parity.py
tests/reference_parity/test_panel_gmm_stata_parity.py
tests/reference_parity/test_panel_absorb_ols_parity.py
tests/reference_parity/test_panel_mixlogit_parity.py
```

## 1. Per-function table

Outcome classes: 1 aligned/bit-exact, 2 defect fixed then aligned,
3 documented convention difference (option added), 4 reference wrong /
non-unique, 5 stochastic, 6 no reference / different estimand.

| function | reference | class | max rel err est / SE | test |
| --- | --- | --- | --- | --- |
| `mepoisson` | Stata 18 `mepoisson, intmethod(laplace)` and `mcaghermite intpoints(7)`; lme4 2.0.1 `glmer(nAGQ=1/7)` | 2 (Laplace optimiser stop) + 4 (Stata AGHQ short stop) | Laplace vs Stata 9.0e-11 / 1.2e-7; AGHQ-7 vs lme4 4.4e-8 / 9.3e-8; objective identity at Stata's θ 6.6e-12 | `test_panel_glmm_parity.py` |
| `menbreg` | Stata `menbreg` laplace / mcaghermite(7); glmmTMB 1.1.14 `nbinom2` (AD-Newton finished); lme4 `glmer.nb` for `curvature='expected'` | 2 (curvature) + 4 (Stata SE) | Laplace vs Stata 7.5e-8 est; vs glmmTMB 1.7e-11 / 2.3e-7; AGHQ-7 vs Stata 1.0e-6 / 9.7e-7; `expected` vs glmer.nb 7.3e-7 est | same |
| `megamma` | Stata `meglm, family(gamma) link(log)` laplace / mcaghermite(7); glmmTMB `Gamma(log)` | 2 (curvature) + 4 (Stata SE, AGHQ short stop) | Laplace vs Stata 2.2e-7 est; vs glmmTMB 6.9e-11 / 3.8e-8 | same |
| `meglm` | Stata `meglm` (gaussian, binomial with `binomial()`), lme4 `lmer(REML=FALSE)`, `glmer` | 2 (Gaussian σ² fixed at 1) | gaussian vs Stata 4.2e-9 / 2.8e-7; binomial-trials vs Stata 2.4e-9 / 6.6e-8 | same |
| `meologit` | Stata `meologit` laplace / mcaghermite(7); ordinal `clmm(nAGQ=1/7)` | 2 (curvature, conditional SEs, no cut SEs) | AGHQ-7 vs Stata 3.8e-7 / 1.3e-6 (5.7e-7 at Stata's θ); Laplace vs Stata 1.2e-6 est; objective identity vs Stata and clmm ≤ 1.4e-11 | same |
| `icc` | Stata `estat icc` after `mixed` (ML, REML), `melogit`, `meologit`; `performance::icc`; `psych::ICC` (ANOVA identity) | 2 (heuristic SE, silent NaN for GLMMs) | vs Stata ≤ 2.4e-7 est / 7e-7 SE / 1.2e-6 CI; vs performance ≤ 1.4e-7 | `test_panel_icc_lrtest_parity.py` |
| `lrtest` | Stata `lrtest`; R `anova()` on lme4 ML fits | 1 (+3: boundary default) | chi2 ≤ 1e-9, df exact, p ≤ 1e-8 (with `boundary=False`) | same |
| `mixed` (touched, already Track A) | Stata `mixed` e(k); R `AIC()` | 2 (`n_params` double count; REML/ML optimiser stop) | AIC vs R 1e-12; ICC ANOVA identity 7e-11 after fix | same |
| `xtnbreg` | Stata `xtnbreg, fe` / `re`; R `pglm(negbin, within/random)` 0.2.4 | 2 (wrong estimator under the right name) | vs Stata FE 2.6e-8 (Stata score 3e-7) / ≤1e-10; RE ≤ 1e-10; vs pglm ≤ 2.7e-8 | `test_panel_xtnbreg_parity.py` |
| `interactive_fe` | Stata `regife ..., noconstant` (SSC, Gomez, 2026-03-30); R `phtt::Eup` 3.1.2 | 2 (SE formula) + 3 (dof) + 4 (phtt σ̂²) | slopes ≤ 2e-11 vs both; SE vs regife (dof='regife') ≤ 1e-9; phtt SE reconstructed ≤ 1e-9 | `test_panel_ife_parity.py` |
| `gmm` | Stata `gmm` (linear, exponential-mean IV; twostep / igmm / onestep); R `gmm::gmm` 1.9.1 (nonlinear) | 2 (nonlinear stop) + 3 (two-step sandwich weight; onestep J) | vs Stata est ≤ 1.2e-12, SE 5e-14 (linear) / 2.5e-7 (nonlinear, Stata numerical Jacobian), J ≤ 4e-13; vs R ≤ 1.8e-7 | `test_panel_gmm_stata_parity.py` (+ existing `test_general_gmm_parity.py`) |
| `absorb_ols` | fixest 0.14.0 / reghdfe via Track A 03 / 15 goldens; Stata `reghdfe` (weights, singletons, 2-way cluster) | 1 (+3: multi-way small-sample factor) | coef ≤ 2e-15; SE iid ≤ 8e-15, clustered ≤ 5.6e-11 | `test_panel_absorb_ols_parity.py` |
| `mixlogit` | Stata `mixlogit` 1.4.0 (SSC, Hole), identical Halton draws | 2 (sd sign, robust factor, ε bias) + deterministic, not T3 | est ≤ 1e-7, SE ≤ 2.2e-7, log L 5e-13 | `test_panel_mixlogit_parity.py` |
| `panel_compare` | — | 6 (infrastructure: a formatted table of `sp.panel` fits) | n/a | loud-failure fix only |

## 2. Defects

1. **`meglm(family='gaussian')` fixed the residual variance at 1.**
   First divergence: SE of x1 0.0428 vs Stata/lmer 0.0335, point estimates
   off in the 3rd digit. The Gaussian family now carries `log σ²` as a
   dispersion parameter with the full normal density; with an identity link
   the Laplace approximation is exact, so the fit equals `sp.mixed(method='ml')`
   (asserted, 1e-8) and Stata `meglm` (4.2e-9 / 2.8e-7).
   Default output changed: yes (⚠️).
2. **GLMM Laplace optimum 2.7e-4 short.** `meglm` stopped at L-BFGS-B's
   relative-function-change criterion (`ftol=1e-8` × |ℓ|≈10³). Before →
   after for `mepoisson` Laplace `_cons`: 0.3451384 → 0.3450937699 (Stata
   0.34509377003). Fixed with a Newton finish on the numerical gradient and
   Hessian (`_newton_polish`), the inner mode now solved to machine precision
   (one extra quadratically-convergent step). Default output changed: yes,
   at the 1e-4 level (⚠️).
3. **Non-canonical families used the Fisher curvature in the Laplace
   approximation and AGHQ scaling** (NB-2, gamma, ordinal logit). That is
   lme4's PIRLS convention; Stata, glmmTMB and clmm use the observed
   curvature (the Laplace approximation proper). Gamma `_cons` 0.6196 →
   0.6372 (Stata 0.63715030, glmmTMB 0.63715031); NB `_cons` 0.4480 →
   0.4728; ologit x1 0.6816 → 0.6865. New default `curvature='observed'`;
   `curvature='expected'` keeps the old approximation and is pinned against
   `glmer.nb` (objective identity 1.6e-10, estimates 7.3e-7). The gamma
   class docstring claimed the observed information "would lose
   definiteness when y < μ" — false (y/(μφ) > 0); corrected.
   Default output changed: yes (⚠️).
4. **`meologit` SEs** were the variance-components-fixed conditional
   information (the formula `meglm` had already dropped in 1.24.0 for
   understating SEs) and there were no threshold SEs. Now inverse numerical
   OIM over (β, thresholds, covariance) with delta-method `thresholds_se`
   (new `MEGLMResult.thresholds_se` field). Default SE output changed (⚠️).
5. **GLMM module docstring** said Laplace is "Stata meglm default" (Stata's
   default is `mvaghermite` with 7 points) and described the old
   conditional-information SE. Corrected; a Notes section maps each Stata
   `intmethod` to `nAGQ`.
6. **`sp.icc`**: (a) returned NaN silently for every GLMM (looked for a
   `var(Residual)` key); now the latent ICC σ²_u/(σ²_u + π²/3) for
   melogit / meologit, a clear error for count/gamma GLMMs (Stata's
   `estat icc` is not available there either -- checked: "requested action
   not valid" after `mepoisson` and `menbreg`) and for random-slope models.
   (b) SE/CI were a heuristic (var(log σ²_u) ≈ 2/n_groups, no covariance)
   with a "n_groups < 30" warning; now the delta method on the observed
   information of the variance parameters and a logit-scale Wald CI, as
   `estat icc`. Before → after (mixed ML): SE heuristic → 0.0533785 (Stata
   0.0533785). (c) three-level fits raised KeyError / would have used the
   wrong denominator; now the Stata level ICCs (SE NaN with a warning).
   Default output changed: yes (⚠️ SE / CI).
7. **`MixedResult.n_params` double-counted the residual variance**
   (`_n_cov_params` already includes it): AIC off by +2, BIC by log n
   versus Stata `e(k)` / R `AIC()`. Fixed in `lmm.py` (one line) — flagged
   because `lmm.py` is being edited on another line. Default AIC/BIC changed
   (⚠️).
8. **`sp.mixed` REML/ML optimum ~1e-6 short** (same mechanism as 2): the
   balanced-design REML ICC missed the closed-form ANOVA value by 1.04e-6;
   after the Newton finish 7e-11. Also in `lmm.py` (flagged). Track A 25_lmm
   SE vs R improves 4.3e-7 → 6.2e-9. Default output changed at 1e-6 (⚠️,
   minor).
9. **`sp.xtnbreg` computed different estimators than `xtnbreg`.**
   `model="fe"` fitted an unconditional NB-2 with entity dummies while
   recording `stata_equivalent = "xtnbreg, fe"`; `model="re"` fitted the
   normal random-intercept NB-2 GLMM. Stata's (and pglm's) `xtnbreg` is the
   Hausman–Hall–Griliches model: conditional FE likelihood (intercept
   identified, all-zero / singleton panels dropped) and beta-distributed
   dispersion RE. Implemented in `regression/_xtnbreg_hhg.py` (analytic
   scores, Newton). On `panel_count_data.csv`, z1: 0.418 (old fe) → 0.3783114
   (Stata 0.3783114). Old estimators kept as `model="ufe"` and
   `model="normal_re"`. Default output changed: yes (⚠️).
10. **`interactive_fe` SEs were not Bai's.** The regressors were projected
    with M_F only; Bai's D0 uses Z = M_Λ X M_F. xa SE (homoskedastic)
    0.03194 → 0.03304 (exact dof) / 0.0331716 (`dof='regife'`, = Stata
    regife). Also: `method='pca'` was silently identical to `'iterative'`
    (now DeprecationWarning), units with missing cells were dropped silently
    (now RuntimeWarning), final residuals used stale factors, default tol
    1e-6 → 1e-10, cluster SEs now carry the CR1 factor. Default SE output
    changed (⚠️).
11. **`sp.gmm` nonlinear moments stopped ~1e-8 short and reported
    `converged=False`** (BFGS precision loss on a finite-difference
    gradient). Gauss–Newton finish added (Stata `gmm`'s own algorithm):
    estimates now 1.2e-12 from Stata. Default output changed at 1e-8.
12. **`sp.mixlogit`**: reported negative SDs (sign not identified); a
    hidden `+1e-6` on the Cholesky diagonal and `+1e-8` on SDs perturbed the
    model away from the reported parameters; `H + 1e-8 I` ridge on the
    Hessian; robust variance lacked Stata's N/(N−1); `converged=False` at the
    optimum; the module docstring claimed agreement with Stata/mlogit "to
    rtol < 1e-3" that had never been measured. All fixed; new options
    `halton_burn`, `halton_shift`, `small_sample`. With Stata's draws the
    comparison is deterministic (log L equal to 5e-13). Default output
    changed (sign of SDs; robust SE ×sqrt(N/(N−1)); ε removal) (⚠️).
13. **`panel_compare` swallowed exceptions** into an "error" cell. Now
    `record_degradation` (WorkflowDegradedWarning), per CLAUDE.md §3.7.

## 3. Convention differences / reference problems (not defects)

* **GLMM Stata SEs, non-canonical Laplace** (NB-2 4.5e-4, gamma 5.5e-3,
  ordinal 5e-5) and some AGHQ fits (Poisson 2.0e-4, gamma 4.1e-4): Stata's
  reported `e(V)` is not the Hessian of its own objective at its own
  estimates — our Hessian evaluated at Stata's θ still differs by the same
  amount, while it agrees with glmmTMB's AD Hessian (2.3e-7 / 3.8e-8) and is
  step-size stable to 1e-7. For AGHQ the Stata SEs are reproduced (3.7e-6
  gamma, 3.6e-7 NB) by the Hessian of the quadrature with the adaptive
  nodes held fixed; Poisson AGHQ only to 7.7e-5 — grade B (mechanism
  partly reconstructed). For Laplace NB/gamma/ologit: grade C (not
  reconstructed). Not asserted.
* **Stata AGHQ short stops**: `mepoisson`/`menbreg`/`meglm gamma`
  `mcaghermite` estimates sit 3e-5 / 1e-6 / 1e-5 from the optimum of
  Stata's own objective (objective identity 6.6e-12 at Stata's θ; ours
  strictly larger). T4 with evidence; Poisson AGHQ pinned vs lme4 4.4e-8.
* **lme4 default `tolPwrss = 1e-7`** leaves Laplace fits ~2.7e-4 from the
  optimum (Poisson here; the same in Track A module 26, whose 2e-4 budget
  this explains). With `tolPwrss = 1e-13` lme4 agrees with us to 2.4e-8.
* **lme4 `glmer.nb` / `lmer` vcov** are θ-conditional (4e-4 / 1.2e-4 from
  the full OIM); `sp.mixed` follows lme4 / Stata `mixed` (conditional),
  `sp.meglm(gaussian)` follows Stata `meglm` (full OIM).
* **`sp.gmm` two-step robust VCE**: default re-estimates S⁻¹ at the final
  β̂ (R `gmm`); Stata keeps the estimation weight (documented in its Methods
  and formulas). New `sandwich_weight='estimation'` reproduces Stata to
  5e-14. Onestep J: Stata recomputes an unadjusted-weight statistic; not
  compared.
* **`absorb_ols` multi-way clustering**: default per-term G/(G−1)
  (`sandwich::vcovCL`, `sp.multiway_cluster_vcov`); new `cluster_df='min'`
  reproduces reghdfe (5.6e-11).
* **`interactive_fe` dof**: `dof='regife'` counts r(N+T) absorbed
  parameters as regife/reghdfe do; the default counts r(N+T−r).
* **phtt `sig2.hat`** demeans residuals unit by unit although the model has
  no unit effect — not copied; reconstructed in the test.
* **Stata `regife` failure mode (reference bug)**: without the `require`
  package, reghdfe errors with r(9) and regife silently posts the pooled-OLS
  starting values as its estimate (the IFE estimate only in `e(bend)`).
  Worth reporting upstream.
* **`lrtest` default** applies the χ̄² boundary mixture; Stata/R report the
  naive χ²(df) (`boundary=False`). The previous warning claimed the
  0.5(χ²_{df−1}+χ²_df) formula was only a bound even in the single-added-
  effect unstructured case, where it is exact; the warning now fires only
  when it is a bound.

## 4. Proposed promotion records (`scripts/build_parity_index.py`, `_FROZEN_PROMOTIONS`)

```python
    "mepoisson": {
        "status": "bit-exact",
        "reference": "Stata 18 mepoisson, intmethod(laplace) / intmethod(mcaghermite) intpoints(7); lme4::glmer(nAGQ = 1 / 7)",
        "reference_versions": {"Stata": "18 MP", "R": "R version 4.5.2 (2025-10-31)", "lme4": "2.0.1"},
        "tolerance": "objective identity at Stata's estimates rtol 1e-11 (observed 6.6e-12); Laplace estimates / SEs vs Stata rtol 1e-6 (observed 9.0e-11 / 1.2e-7); AGHQ-7 vs lme4 rtol 1e-6 (observed 4.4e-8 / 9.3e-8)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Laplace = Stata intmethod(laplace); nAGQ = k = intmethod(mcaghermite) intpoints(k). Stata's default mvaghermite is a different rule and is not compared. Stata's mcaghermite estimate stops 2.9e-5 short of the optimum of its own objective (same function to 6.6e-12, ours strictly larger); lme4 needs tolPwrss = 1e-13 (its default 1e-7 leaves the Laplace fit 2.7e-4 away). The 1.28.x fix finishing the optimum with Newton steps moved the Laplace estimate by 1.3e-4.",
    },
    "menbreg": {
        "status": "bit-exact",
        "reference": "glmmTMB 1.1.14 nbinom2 (Laplace, AD Hessian); Stata 18 menbreg intmethod(laplace) / mcaghermite(7)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "glmmTMB": "1.1.14", "TMB": "1.9.25", "lme4": "2.0.1", "Stata": "18 MP"},
        "tolerance": "vs glmmTMB estimates / SEs rtol 1e-6 (observed 1.7e-11 / 2.3e-7); vs Stata Laplace estimates rtol 1e-6 (observed 7.5e-8); objective identity at Stata's and glmmTMB's estimates rtol 1e-11",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Laplace with the observed curvature (curvature='observed', the default since 1.28.x; Stata, glmmTMB). curvature='expected' is lme4's PIRLS Laplace and is pinned against glmer.nb (objective 1.6e-10, estimates 7.3e-7). Stata's Laplace SEs are not the Hessian of its own objective (4.5e-4) and are not asserted; glmmTMB's AD Hessian agrees with ours to 2.3e-7.",
    },
    "megamma": {
        "status": "bit-exact",
        "reference": "glmmTMB 1.1.14 Gamma(link = 'log') (Laplace, AD Hessian); Stata 18 meglm, family(gamma) link(log) intmethod(laplace)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "glmmTMB": "1.1.14", "TMB": "1.9.25", "Stata": "18 MP"},
        "tolerance": "vs glmmTMB estimates / SEs rtol 1e-6 (observed 6.9e-11 / 3.8e-8); vs Stata Laplace estimates rtol 1e-6 (observed 2.2e-7); objective identity rtol 1e-11",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Before 1.28.x the Laplace used the Fisher curvature 1/phi (lme4's convention), a different approximation: _cons 0.6196 vs 0.6372. Stata's gamma SEs are not the Hessian of its own objective (5.5e-3 Laplace) and are not asserted.",
    },
    "meglm": {
        "status": "bit-exact",
        "reference": "Stata 18 meglm (gaussian; binomial with binomial()); lme4::lmer(REML = FALSE); lme4::glmer",
        "reference_versions": {"Stata": "18 MP", "R": "R version 4.5.2 (2025-10-31)", "lme4": "2.0.1"},
        "tolerance": "gaussian vs Stata estimates / SEs rtol 1e-6 (observed 4.2e-9 / 2.8e-7), vs lmer ML estimates 6.1e-11; binomial-trials vs Stata 2.4e-9 / 6.6e-8",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Before 1.28.x family='gaussian' held the residual variance at 1 (x1 SE 0.0428 vs 0.0335); it is now estimated and the fit equals sp.mixed(method='ml'). SEs are the full OIM (Stata meglm); lmer's vcov is theta-conditional (1.2e-4 away) like sp.mixed / Stata mixed.",
    },
    "meologit": {
        "status": "bit-exact",
        "reference": "Stata 18 meologit intmethod(mcaghermite) intpoints(7) / intmethod(laplace); ordinal::clmm",
        "reference_versions": {"Stata": "18 MP", "R": "R version 4.5.2 (2025-10-31)", "ordinal": "2025.12.29"},
        "tolerance": "AGHQ-7 vs Stata estimates rtol 1e-6 (observed 3.8e-7), SEs incl. cutpoints rtol 2e-6 (observed 1.3e-6; 5.7e-7 at Stata's estimates); objective identity vs Stata and clmm rtol 1e-11 (observed <= 1.4e-11)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_glmm_parity.py"],
        "note": "Observed-curvature Laplace / AGHQ since 1.28.x (was Fisher); SEs are now the full OIM with delta-method threshold SEs (was the conditional information, no cutpoint SEs). clmm stops with gradient 2e-3; Stata's Laplace-ologit SEs are not the Hessian of its own objective (5e-5); neither is asserted.",
    },
    "icc": {
        "status": "bit-exact",
        "reference": "Stata 18 estat icc after mixed (ML, REML) and melogit; performance::icc; psych::ICC (balanced ANOVA identity)",
        "reference_versions": {"Stata": "18 MP", "R": "R version 4.5.2 (2025-10-31)", "performance": "0.16.0", "psych": "2.6.5"},
        "tolerance": "estimate / SE rtol 1e-6, logit-scale CI rtol 2e-6 (observed <= 2.4e-7 / 7e-7 / 1.2e-6); balanced REML ICC = ANOVA ICC(1) rtol 1e-9",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_icc_lrtest_parity.py"],
        "note": "Before 1.28.x the SE was a heuristic (var(log s2_u) ~ 2/n_groups, no covariance) and every GLMM returned NaN silently. Now the delta method on the observed information of the variance parameters and a logit-scale Wald CI, latent residual variance pi^2/3 after melogit / meologit.",
    },
    "lrtest": {
        "status": "bit-exact",
        "reference": "Stata 18 lrtest; R anova() on lme4 ML fits",
        "reference_versions": {"Stata": "18 MP", "R": "R version 4.5.2 (2025-10-31)", "lme4": "2.0.1"},
        "tolerance": "chi2 rtol 1e-6, df exact, p rtol 1e-5 (observed chi2 <= 1e-9)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_icc_lrtest_parity.py"],
        "note": "boundary=False reproduces Stata / R (naive chi2(df)); the default applies the chibar2 mixture, exact (Stram-Lee) for one added random effect under an unstructured covariance. df = difference in e(k); MixedResult.n_params no longer double-counts the residual variance.",
    },
    "xtnbreg": {
        "status": "bit-exact",
        "reference": "Stata 18 xtnbreg, fe / re (Hausman-Hall-Griliches); pglm::pglm(family = negbin, model = 'within' / 'random')",
        "reference_versions": {"Stata": "18 MP", "R": "R version 4.5.2 (2025-10-31)", "pglm": "0.2.4"},
        "tolerance": "coefficients / SEs rtol 1e-7 (observed <= 2.7e-8: Stata fe's score at its own estimate is 3e-7, pglm's gradtol), log-likelihood rtol 1e-12",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_xtnbreg_parity.py"],
        "note": "Before 1.28.x model='fe' fitted an unconditional dummy-variable NB-2 labelled 'xtnbreg, fe' and model='re' the normal random-intercept NB-2 GLMM; both remain as model='ufe' / 'normal_re'. Conditional FE drops all-zero and singleton panels as Stata does (N = 228 of 240).",
    },
    "interactive_fe": {
        "status": "bit-exact",
        "reference": "Stata regife (SSC, Gomez) ..., noconstant; R phtt::Eup(additive.effects = 'none')",
        "reference_versions": {"Stata": "18 MP", "regife": "2026-03-30 SSC", "R": "R version 4.5.2 (2025-10-31)", "phtt": "3.1.2"},
        "tolerance": "slopes rtol 1e-9 vs both (observed <= 2e-11); SEs rtol 1e-9 vs regife with dof='regife' (homoskedastic and cluster), phtt SE reconstructed rtol 1e-9",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_ife_parity.py"],
        "note": "SEs are Bai's D0 with Z = M_Lambda X M_F since 1.28.x (was M_F X only). Default dof counts r(N+T-r) absorbed parameters; regife/reghdfe count r(N+T). phtt's sig2.hat demeans residuals by unit and is not copied.",
    },
    "gmm": {
        "status": "bit-exact",
        "reference": "Stata 18 gmm (linear and exponential-mean IV; twostep, igmm, onestep); R gmm::gmm",
        "reference_versions": {"Stata": "18 MP", "R": "R version 4.5.2 (2025-10-31)", "gmm": "1.9.1"},
        "tolerance": "vs Stata estimates rtol 1e-10 (observed 1.2e-12), SEs 1e-9 linear / 1e-6 nonlinear (Stata's numerical Jacobian; observed 2.5e-7), J rtol 1e-10; vs R rtol 1e-6 (observed 1.8e-7)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_gmm_stata_parity.py", "tests/reference_parity/test_general_gmm_parity.py"],
        "note": "Stata's two-step robust VCE keeps the estimation weight: sandwich_weight='estimation'; the default re-estimates S^-1 at the final estimate (R gmm), a documented 1/n-order difference (2.6e-6 linear, 4.4e-5 nonlinear here). Onestep J differs by construction (Stata recomputes an unadjusted weight). Nonlinear fits now end with Gauss-Newton steps.",
    },
    "absorb_ols": {
        "status": "bit-exact",
        "reference": "fixest::feols 0.14.0 and Stata reghdfe (Track A 03_hdfe / 15_hdfe_cluster goldens); Stata reghdfe with aweights, singleton dropping, two-way clustering",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "fixest": "0.14.0", "Stata": "18 MP"},
        "tolerance": "coefficients rtol 1e-12 (observed 2e-15), iid SEs 1e-12 (observed 8e-15), clustered SEs 1e-10 (observed 5.6e-11)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_panel_absorb_ols_parity.py"],
        "note": "Called directly on the committed Track A bytes (the modules go through sp.fast.feols / sp.hdfe_ols). Two-way clustering: cluster_df='min' reproduces reghdfe's G_min/(G_min-1) on every inclusion-exclusion term; the default per-term factor (sandwich::vcovCL) differs by 1.8e-4 / 1.2e-2 in variance here.",
    },
    "mixlogit": {
        "status": "bit-exact",
        "reference": "Stata mixlogit 1.4.0 (SSC, Hole), nrep(50) burn(15), on identical Halton draws",
        "reference_versions": {"Stata": "18 MP", "mixlogit": "1.4.0"},
        "tolerance": "means / SDs / Sigma / SEs rtol 1e-6 (observed <= 2.2e-7), log-likelihood rtol 1e-10 (observed 5e-13)",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_panel_mixlogit_parity.py"],
        "note": "Deterministic, not Monte-Carlo: n_draws=50, halton_burn=15, halton_shift=False builds Stata's draw matrix invnormal(halton(50, k, 1 + 15 + 50(n-1))), so both maximise the same simulated likelihood. oim (robust=False), robust (x N/(N-1)), lognormal ln(1) and corr (compared on Sigma) all covered. With the default shifted draws the two are different simulators.",
    },
```

(`panel_compare`: no record — infrastructure, class 6.)

## 5. Proposed CHANGELOG / MIGRATION

### Added
- `sp.meglm` / `sp.melogit` / `sp.mepoisson` / `sp.menbreg` / `sp.megamma` /
  `sp.meologit`: `curvature={'observed','expected'}`; `MEGLMResult.thresholds_se`.
- `sp.xtnbreg`: Hausman–Hall–Griliches conditional-FE (`model="fe"`) and
  beta-RE (`model="re"`) negative binomial = Stata `xtnbreg`; `model="ufe"`
  and `model="normal_re"` for the previous estimators.
- `sp.gmm(sandwich_weight=...)`, `sp.absorb_ols(cluster_df=...)`,
  `sp.interactive_fe(dof=...)`, `sp.mixlogit(halton_burn=, halton_shift=,
  small_sample=)`.

### ⚠️ Correctness
- `meglm(family='gaussian')` now estimates the residual variance (was fixed
  at 1).
- GLMM Laplace / AGHQ for NB-2, gamma, ordinal logit use the observed
  curvature (Stata, glmmTMB, clmm); the Fisher version is `curvature='expected'`.
- GLMM and `sp.mixed` optima finished with Newton steps (were ~1e-4 / 1e-6
  short).
- `meologit` SEs: full OIM (were variance-components-fixed); threshold SEs.
- `sp.xtnbreg(model="fe"/"re")` now computes Stata's `xtnbreg`.
- `sp.icc`: Stata `estat icc` SE / CI; latent ICC for melogit / meologit
  (was NaN).
- `MixedResult.n_params` / AIC / BIC: residual variance counted once.
- `sp.interactive_fe`: Bai's SE (M_Λ X M_F), CR1 cluster factor, tol 1e-10.
- `sp.mixlogit`: SDs reported positive, no ε perturbation of σ / Cholesky
  diagonal, robust variance × N/(N−1).

### Fixed
- `sp.gmm` nonlinear: Gauss–Newton finish; `converged` no longer False at
  the optimum. `sp.mixlogit` `converged` by score.
- `sp.lrtest` warns only when the χ̄² formula is a bound.
- `sp.panel_compare` records failing methods with `WorkflowDegradedWarning`.

### MIGRATION rows
| function | old default | new default | how to get the old number |
| --- | --- | --- | --- |
| `sp.xtnbreg(model="fe")` | unconditional NB-2 with dummies | HHG conditional FE (Stata `xtnbreg, fe`) | `model="ufe"` |
| `sp.xtnbreg(model="re")` | normal random-intercept NB-2 (`menbreg`) | HHG beta RE (Stata `xtnbreg, re`) | `model="normal_re"` |
| `menbreg` / `megamma` / `meologit` | Fisher-curvature Laplace | observed-curvature Laplace | `curvature="expected"` |
| `meglm(family="gaussian")` | σ² fixed at 1 | σ² estimated | — (old was not a Gaussian mixed model) |
| `meologit` SEs | conditional information | full OIM | — |
| `sp.icc` SE / CI | heuristic | Stata `estat icc` | — |
| `sp.interactive_fe` SE | M_F-only projection | Bai D0 (M_Λ X M_F), CR1 | — |
| `sp.mixlogit(robust=True)` | no N/(N−1) | × N/(N−1) | `small_sample=False` |
| `MixedResult.aic/bic` | +2 / +log n | Stata / R | — |

## 6. Could not close / open

- **Stata SEs of non-canonical Laplace GLMMs** (grade C) and Stata AGHQ SEs
  (grade B, fixed-node reconstruction to 3.6e-7–3.7e-6 for NB / gamma, 7.7e-5
  Poisson). Estimates and objective are T2; SEs certified against glmmTMB AD
  instead.
- **Stata's default `mvaghermite`** not implemented; recorded
  (`pois_mvagh7_default`) but not compared. Track A module 27 compares
  `sp.melogit(nAGQ=8)` (mode-curvature) against Stata's default
  `mvaghermite` — a different rule; with the Newton finish the Stata
  headline sits at 9.8e-7 against a 1e-6 budget. Proposal: switch
  `tests/stata_parity/27_glmm_aghq.do` to `intmethod(mcaghermite)`.
- **Track A module 26**: its 2e-4 budget is lme4's loose `tolPwrss`; with
  `glmerControl(tolPwrss = 1e-13)` in `26_glmm_logit.R` lme4 agrees with
  `sp.melogit` to 2.4e-8 (est) / 2.4e-6 (SE, lme4's finite-difference
  vcov). Proposal for the integrator (files under `tests/r_parity/` are
  off-limits here).
- **Track A py goldens 25 / 26 / 27 move** (optimiser finish): 25_lmm SE vs
  R 4.3e-7 → 6.2e-9; 26 vs Stata 8.4e-7 → 7e-9; 27 intercept vs R
  3.0e-8 → 1.7e-7 (lme4 golden at default tolPwrss; logLik equal to
  3.4e-12). Needs `verify_reproduce.py` (full) + `tier_a_fixture_lock.py`.
- `mixlogit` vs R `mlogit`: not attempted (mlogit's draw layout differs;
  Stata is the canonical implementation by Hole).
- `glmer.nb`'s reported logLik sits 1.6e-7 above the Fisher-Laplace at its
  own estimates — source not located (identity asserted at 1e-9).
- Three-level `sp.icc` has no SE (the three-level fit does not retain its
  likelihood blocks); warns.
- `interactive_fe` with additive effects (phtt `twoways`, regife
  `absorb()`) not covered.

## 7. Integrator notes

- `.gitignore`: add `tests/reference_parity/_fixtures/_ado_panel_glmm/`
  (private SSC ado dir: ftools, reghdfe, require, regife, mixlogit).
- `registry.py` `xtnbreg` spec: description is now wrong ("model='fe' uses
  explicit entity fixed effects...") and `model` choices should be
  `["fe", "re", "pooled", "ufe", "normal_re"]`. Suggested description:
  "Panel negative-binomial regression (Stata xtnbreg): model='fe' is the
  Hausman-Hall-Griliches conditional fixed-effects NB, model='re' the
  beta-dispersion random-effects NB; 'ufe' (dummy-variable NB-2) and
  'normal_re' (menbreg) are the other panel NB estimators."
- New public keyword arguments change signatures → `python
  scripts/dump_schemas.py`.
- `src/statspai/multilevel/lmm.py` touched (two small changes: `n_params`,
  Newton finish) although another line is editing it — merge carefully.
- New module `src/statspai/multilevel/_glmm_ri.py` (vectorised
  random-intercept kernel; GLMM fits ~10x faster) and
  `src/statspai/regression/_xtnbreg_hhg.py`.
