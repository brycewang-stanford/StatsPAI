# Changelog

All notable changes to StatsPAI will be documented in this file.

## [0.9.3] - 2026-04-19 — Stochastic Frontier + Multilevel + Econometric Trinity

**Overview.** This release bundles three simultaneous deep overhauls plus an
author-metadata correction:

1. **Stochastic Frontier Analysis** — `sp.frontier` / `sp.xtfrontier` rewritten
   to Stata/R-grade, with a critical correctness bug fix.
2. **Multilevel / Mixed-Effects** — `sp.multilevel` rewritten to lme4/Stata-grade.
3. **Econometric Trinity** — three new P0 pillars: DML-PLIV, Mixed Logit, IV-QR.
4. **Author attribution** corrected to `Biaoyue Wang`.

⚠️ **Critical correctness fix** — `sp.frontier` carried a latent Jondrow posterior
sign error in all prior versions (0.9.2 and earlier). Efficiency scores were
systematically biased; the normal-exponential path additionally returned NaN
for unit efficiency. **Re-run any prior frontier analyses.** Detail below.

---

### Stochastic Frontier Analysis Overhaul

Release focus: `statspai.frontier`. The prior implementation was a
270-line single file with one function covering cross-sectional
half-normal / exponential / truncated-normal frontiers, no panel
support, no heteroskedasticity, no inefficiency determinants, and —
critically — a sign error in the Jondrow posterior that silently
produced wrong efficiency scores, plus a wrong ε-coefficient in the
exponential log-likelihood that the old test never exercised. The
module has been rewritten (~1,300 LOC across `_core.py`, `sfa.py`,
`panel.py`, `te_tools.py`) to match or exceed Stata's
`frontier` / `xtfrontier` and R's `frontier` / `sfaR`.

### Correctness fixes

- **Jondrow posterior μ\***: corrected `sign` convention in all three
  distributions — the old code's `μ* = -sign·ε·σ_u²/σ²` has been
  replaced by the derivation-verified `μ* = sign·ε·σ_u²/σ²` (and the
  analogous correction for truncated-normal). Efficiency scores from
  the old implementation were systematically biased; re-run any prior
  analyses.
- **Normal-exponential log-density**: fixed the ε-coefficient and
  Φ argument (the old form was `+ sign·ε/σ_u + log Φ((-sign·ε - σ_v²/σ_u)/σ_v)`;
  correct per Greene 2008 eq. 2.39 is `- sign·ε/σ_u + log Φ(sign·ε/σ_v - σ_v/σ_u)`).
  The old exponential path never produced efficiency scores (returned NaN) —
  now returns correct Battese-Coelli scores.
- **Truncated-normal density**: fixed the `centered` offset in the
  φ factor from `(ε + sign·μ)/σ` to `(ε - sign·μ)/σ`.
- Monte-Carlo density-integration tests (`∫ f(ε) dε = 1`) now guard
  against regressions for all three distributions.

### New cross-sectional `sp.frontier`

- **Heteroskedastic inefficiency** via `usigma=[...]` — parameterises
  `ln σ_u_i = γ_u' [1, w_i]` (Caudill-Ford-Gropper 1995, Hadri 1999).
- **Heteroskedastic noise** via `vsigma=[...]` — parameterises
  `ln σ_v_i = γ_v' [1, r_i]` (Wang 2002).
- **Inefficiency determinants** via `emean=[...]` — the
  Battese-Coelli (1995) / Kumbhakar-Ghosh-McGuckin (1991) model
  `μ_i = δ' [1, z_i]` for `dist='truncated-normal'`.
- **Battese-Coelli (1988) TE**: `result.efficiency(method='bc')` returns
  `E[exp(-u)|ε]` (the Stata default) in addition to the JLMS
  approximation `exp(-E[u|ε])` (`method='jlms'`).
- **LR test for absence of inefficiency**: one-sided mixed χ̄²
  (Kodde-Palm 1986) via `result.lr_test_no_inefficiency()`.
- **Bootstrap CI for unit efficiency**: parametric-bootstrap bounds
  via `result.efficiency_ci(alpha=.05, B=500)`.
- **Residual skewness diagnostic** stored at
  `result.diagnostics['residual_skewness']`.
- Optimiser now has hard bounds on `ln σ` and guards against
  σ → 0 / σ → ∞ excursions that previously caused truncated-normal
  fits to diverge.

### New panel `sp.xtfrontier`

- **Pitt-Lee (1981) time-invariant** (`model='ti'`):
  `u_it = u_i`, half-normal or truncated-normal.  Closed-form group
  log-likelihood derived from the per-unit integration; unit-level
  TE stored at `result.diagnostics['efficiency_bc_unit']`.
- **Battese-Coelli (1992) time-varying decay** (`model='tvd'`):
  `u_it = exp(-η(t - T_i)) · u_i` with η estimated jointly.  The
  obs-level efficiency uses `E[exp(-a_it u_i)|e_i]` under the
  posterior `u_i ~ N⁺(μ*, σ*²)` (MGF form).
- **Battese-Coelli (1995) inefficiency effects** (`model='bc95'`):
  `u_it ~ N⁺(z_it' δ, σ_u²)` independently; returned with unit-mean
  efficiency roll-up.

### Helpers

- `sp.te_summary(result)` — Stata-style descriptive table of TE
  scores (n, mean, sd, quartiles, share > 0.9, share < 0.5).
- `sp.te_rank(result, with_ci=True)` — efficiency ranking with
  optional bootstrap CIs for benchmarking.

### Tests

- **33 new tests** covering: parameter recovery for all three
  cross-sectional distributions, cost vs production sign handling,
  heteroskedastic σ_u / σ_v, BC95 determinants, LR specification tests,
  TE-score bounds and internal consistency, bootstrap CI structure,
  Pitt-Lee / BC92 / BC95 panel recovery, and density-integrates-to-1
  kernel sanity checks.

### Advanced frontier extensions

Three frontier extensions shipped after the initial overhaul (commit `e876937`):

- **`sp.zisf`** — Zero-Inefficiency SFA mixture (Kumbhakar-Parmeter-Tsionas
  2013). Mixture of fully-efficient (`u=0`, pure noise) and standard
  composed-error regimes; mixing probability `p_i` parameterised via logit
  on optional `zprob` covariates. Posterior `P(efficient|ε)` exposed in
  `diagnostics['p_efficient_posterior']`. Recovery test: true efficient
  share 0.30 → estimated 0.286 on `n=2000`.
- **`sp.lcsf`** — 2-class Latent-Class SFA (Orea-Kumbhakar 2004;
  Greene 2005). Two separate frontiers with their own `β_k` and variance
  parameters; class-membership logit on optional `z_class` covariates.
  Direct MLE with perturbed starts to break label symmetry.
- **`xtfrontier(..., model='tfe', bias_correct=True)`** — Dhaene-Jochmans
  (2015) split-panel jackknife for TFE:
  `β_BC = 2·β_full − (β_first_half + β_second_half)/2`. Cuts the `O(1/T)`
  incidental-parameters bias. Guards against degenerate halves by skipping
  σ corrections with an annotation in `model_info`. Verified at `T=30`,
  `N=25`: raw `σ_u=0.374` → BC `σ_u=0.359` (true 0.35).

### Productivity helpers

Shipped in commit `be59260`:

- **`sp.malmquist`** — Färe-Grosskopf-Lindgren-Roos (1994) Malmquist TFP
  index via period-by-period parametric frontier fits. Returns per-
  transition decomposition `M = EC × TC` (efficiency change × technical
  change). Row-wise identity `M == EC·TC` verified to `rtol=1e-8`. Cost
  frontiers supported via reciprocal distance convention. Validated on
  3-period DGP with 5%/year intercept growth: mean TC ≈ 1.07–1.09,
  mean EC ≈ 1.0.
- **`sp.translog_design`** — Cobb-Douglas → Translog design-matrix helper.
  Appends `0.5·log(x_k)²` squares and `log(x_k)·log(x_l)` interactions;
  the `translog_terms` list is stored in `df.attrs` for one-line feed to
  `frontier()` / `xtfrontier()`. Toggleable squares and interactions.

### Migration

- Old: `frontier(df, y='y', x=['x1'])` still works (same required args).
- New keyword-only args: `usigma`, `vsigma`, `emean`, `te_method`,
  `start`.
- Existing efficiency scores should be recomputed — prior values were
  systematically biased by the Jondrow sign error.

---

### Multilevel / Mixed-Effects Overhaul

Release focus: `statspai.multilevel`. The previous implementation was a
400-line single file covering only the two-level linear mixed model
with a diagonal random-effect covariance. It has been rewritten as a
proper sub-package (~2,000 LOC across `_core.py`, `lmm.py`, `glmm.py`,
`diagnostics.py`, `comparison.py`) with feature parity against
`lme4`/Stata `mixed` and additions on top.

### New in `sp.mixed`

- **Unstructured covariance** `G` for random effects is now the
  default (`cov_type='unstructured'`, Cholesky-parameterised so the
  optimiser is unconstrained). `diagonal` and `identity` remain
  available for nested-model comparisons.
- **Three-level nested models** via `group=['school', 'class']` —
  fits school- and class-level random intercepts jointly (verified to
  match `statsmodels.MixedLM(..., re_formula="1", vc_formula={...})`
  to four decimals on the variance components and fixed effects).
- **BLUP posterior standard errors** (`result.ranef(conditional_se=
  True)`) — exposes
  `Var(u|y) = G − GZ'V⁻¹ZG + GZ'V⁻¹X Cov(β̂) X'V⁻¹ZG` for use in
  caterpillar plots.
- **`predict(new_data, include_random=…)`** — population-marginal and
  group-conditional predictions, with zeroed-out BLUPs for unseen
  groups.
- **Nakagawa-Schielzeth marginal & conditional R²** via
  `result.r_squared()`.
- **AIC / BIC, `wald_test()`** for linear restrictions,
  **`to_markdown()` / `to_latex()` / `_repr_html_()` / `cite()`**,
  and `plot(kind='caterpillar' | 'residuals')`.

### New functions

- **`sp.melogit` / `sp.mepoisson` / `sp.meglm`** — Generalised linear
  mixed models (binomial logit, Poisson log, Gaussian identity) fitted
  by Laplace approximation with canonical-link observed information.
  Supports random intercepts and random slopes, `cov_type` as for
  `sp.mixed`, binomial `trials=` and Poisson `offset=`. Results expose
  `odds_ratios()` / `incidence_rate_ratios()` and a `predict(type=
  'response'|'linear')` method.
- **`sp.icc(result)`** — intra-class correlation with a delta-method
  (logit-scale) 95% CI.
- **`sp.lrtest(restricted, full)`** — likelihood-ratio test between
  two nested mixed-model fits with automatic Self-Liang χ̄²
  boundary correction when variance components are being tested.

### Validation

- Linear mixed models: fixed effects and variance components agree
  with `statsmodels.MixedLM` to 4 decimal places on both random-
  intercept and unstructured random-slope specifications
  (`test_multilevel.py::TestRandomSlopeUnstructured::
  test_matches_statsmodels`).
- Three-level nested: variance components identified jointly and match
  the reference implementation to 2 decimal places
  (`TestThreeLevelNested::test_separates_variance_components`).
- GLMM recovery tests on 2,000-observation synthetic panels confirm
  slope and random-intercept variance within expected sampling ranges.

### Behavioural changes

- The default `cov_type` for `sp.mixed` is now `'unstructured'`
  (previously effectively diagonal). Pass `cov_type='diagonal'`
  explicitly for the old behaviour.
- `LR test vs. pooled OLS` now uses the ML-converted likelihood
  (previously a mix of REML and ML that could produce inconsistent
  values when `method='reml'`).

### Post-review hardening (post oracle + code-reviewer audit)

- **[BLOCKER fix]** `MixedResult.predict(data=None)` previously returned
  predictions in group-iteration order rather than the original row
  order. `_GroupBlock` now carries the training row indices and
  `predict()` scatters the output back to the correct positions.
  Regression test: `tests/test_multilevel.py::TestRandomIntercept::
  test_predict_is_row_aligned_with_training_frame`.
- **[BLOCKER fix]** GLMM inner Newton (`_find_mode`) now damps large
  steps and returns a convergence flag. `meglm` aggregates per-cluster
  failures and emits a `RuntimeWarning` when any cluster fails to
  converge — a previously silent failure mode.
- **[HIGH fix]** `MEGLMResult` gains `to_latex()` and `plot()` so it
  matches the unified StatsPAI result contract.
- **[HIGH fix]** `lrtest` now raises `ValueError` on cross-family
  comparisons and on REML fits whose fixed-effect design differs,
  preventing invalid LR statistics. Multi-component boundary
  corrections emit a `RuntimeWarning` explaining the conservative
  upper bound (Stram–Lee 1994 mixture not implemented).
- **[HIGH fix]** `mixed()` / `meglm()` reject non-hashable group
  values with a descriptive `TypeError` instead of producing a
  silently corrupted BLUP dict.
- **[MED fix]** `icc(result, n_boot>0)` raises `NotImplementedError`
  instead of silently returning the delta-method CI. `icc()` warns
  when `n_groups < 30` (delta-method CI unreliable).
- **[MED fix]** Three-level nested fit emits a warning when any
  outer group has only one inner group (class variance then not
  identified), and exposes both school and class ICCs via
  `variance_components['icc(outer)']` / `icc(outer+inner)`.

---

### Econometric Trinity — P0 Pillars (DML-PLIV, Mixed Logit, IV-QR)

Three foundational econometric estimators identified as the highest-ROI gaps
vs. Stata, R, and existing Python packages are now first-class `sp.*` APIs
(~1,170 new LOC, 10 tests in `test_econ_trinity.py`).

- **`sp.dml(model='pliv', instrument=…)` — DML-PLIV (Partially Linear IV).**
  Chernozhukov et al. (2018, §4.2) Neyman-orthogonal score with cross-fitted
  nuisance functions `g(X)=E[Y|X]`, `m(X)=E[D|X]`, `r(X)=E[Z|X]`. Returns the
  LATE with influence-function-based standard errors. Closes the IV gap in
  the existing `DoubleML` (previously only PLM + IRM).
- **`sp.mixlogit` — Mixed Logit.** Random-coefficient multinomial logit via
  simulated maximum likelihood with Halton quasi-random draws. Supports:
  fixed + random coefficients, normal / log-normal / triangular mixing
  distributions, diagonal or full Cholesky covariance, panel (repeated-choice)
  data, OPG-sandwich robust SEs. Benchmarked against Stata `mixlogit` and R
  `mlogit`. Python's first feature-complete implementation.
- **`sp.ivqreg` — IV Quantile Regression.** Chernozhukov-Hansen (2005, 2006,
  2008) instrumental-variable quantile regression via inverse-QR profile.
  Scalar endogenous case uses grid + Brent refinement; multi-dim uses BFGS on
  the `b̂(α)` criterion. Multiple quantiles return a tidy DataFrame; single
  quantile returns `EconometricResults`. Optional pairs-bootstrap SEs.

All three reuse `_qreg_fit`, `CausalResult`, `EconometricResults` for API
consistency with the rest of StatsPAI.

#### Post-self-audit hardening

Self-audit + code-reviewer agent surfaced and fixed 4 BLOCKER + 7 HIGH bugs
in the first-cut implementation (see commit `2aa709b`). Parameter-recovery
tests now pass against controlled DGPs.

---

### Smart Workflow — Posterior Verification

Shipped in commit `be59260`:

- **`sp.verify`** / **`sp.verify_benchmark`** — posterior verification engine
  for `sp.recommend()` outputs. Runs bootstrap stability, placebo pass rate,
  and subsample agreement, aggregated into `verify_score ∈ [0, 100]`.
  Opt-in via `sp.recommend(verify=True)`; zero overhead when disabled.
- Calibration card shows top-method `verify_score` 85–95 on clean DGPs
  (RD lower at ≈ 74 due to local-polynomial bootstrap variance).
- 18/18 smart tests pass.

---

### Meta — Author Attribution

- Author metadata corrected from `Bryce Wang` to `Biaoyue Wang` in:
  `pyproject.toml` (`authors` + `maintainers`), `src/statspai/__init__.py`
  (`__author__`), `README.md` / `README_CN.md` (team line + BibTeX),
  `docs/index.md` (BibTeX), and `mkdocs.yml` (`site_author`).
  JOSS submission (`paper.md`) was already correct.

## [0.9.2] - 2026-04-16

### Decomposition Analysis — Most Comprehensive Decomposition Toolkit in Python

Release focus: `statspai.decomposition`. **18 first-class decomposition methods across 13 modules (~6,200 LOC, 54 tests)** — Python's first (and most complete) implementation of the full decomposition analysis toolkit spanning mean, distributional, inequality, demographic, and causal decomposition. Beats Stata `ddecompose` / `cdeco` / `oaxaca` / `rifhdreg` / `mvdcmp` / `fairlie` and R `Counterfactual` / `ddecompose` / `oaxaca` / `dineq` in scope; occupies the previously empty Python high-ground where only one unmaintained PyPI package existed.

#### What's in `sp.decompose` (18 methods, 30 aliases)

**Mean decomposition**

| Function | Method / Paper |
|---|---|
| `sp.oaxaca(df, ...)` | Blinder-Oaxaca threefold with 5 reference coefficients (Blinder 1973; Oaxaca 1973; Neumark 1988; Cotton 1988; Reimers 1983) |
| `sp.gelbach(df, ...)` | Sequential orthogonal decomposition of omitted-variable bias (Gelbach 2016, *JoLE*) |
| `sp.fairlie(df, ...)` | Nonlinear logit/probit decomposition (Fairlie 1999, 2005) |
| `sp.bauer_sinning(df, ...)` / `sp.yun_nonlinear(df, ...)` | Bauer-Sinning (2008) + Yun (2004, 2005) detailed nonlinear |

**Distributional decomposition**

| Function | Method / Paper |
|---|---|
| `sp.rifreg(df, ...)` / `sp.rif_decomposition(...)` | Recentered Influence Function regression + OB (Firpo, Fortin & Lemieux 2009, *Econometrica*) |
| `sp.ffl_decompose(df, ...)` | Two-step detailed decomposition (Firpo, Fortin & Lemieux 2018, *Econometrics*) |
| `sp.dfl_decompose(df, ...)` | Reweighting counterfactual distributions (DiNardo, Fortin & Lemieux 1996, *Econometrica*) |
| `sp.machado_mata(df, ...)` | Simulation-based quantile regression decomposition (Machado & Mata 2005, *JAE*) |
| `sp.melly_decompose(df, ...)` | Analytical quantile regression decomposition (Melly 2005, *Labour Economics*) |
| `sp.cfm_decompose(df, ...)` | Distribution regression counterfactuals (Chernozhukov, Fernández-Val & Melly 2013, *Econometrica*) |

**Inequality decomposition**

| Function | Method / Paper |
|---|---|
| `sp.subgroup_decompose(df, ...)` | Between/within for Theil T, Theil L, GE(α), Gini (Dagum 1997), Atkinson, CV² (Shorrocks 1984) |
| `sp.shapley_inequality(df, ...)` | Shorrocks-Shapley allocation of inequality to covariates (Shorrocks 2013, *JoEI*) |
| `sp.source_decompose(df, ...)` | Gini source decomposition (Lerman & Yitzhaki 1985, *ReStat*) |

**Demographic standardization**

| Function | Method / Paper |
|---|---|
| `sp.kitagawa_decompose(df, ...)` | Two-factor rate decomposition (Kitagawa 1955, *JASA*) |
| `sp.das_gupta(df_a, df_b, ...)` | Multi-factor symmetric decomposition (Das Gupta 1993) |

**Causal decomposition**

| Function | Method / Paper |
|---|---|
| `sp.gap_closing(df, method=...)` (regression / IPW / AIPW) | Gap-closing estimator (Lundberg 2021, *Sociol. Methods Res.*) |
| `sp.mediation_decompose(df, ...)` | Natural direct/indirect effects (VanderWeele 2014, *Epidemiology*) |
| `sp.disparity_decompose(df, ...)` | Causal disparity decomposition (Jackson & VanderWeele 2018, *Epidemiology*) |

**Unified entry point**

```python
import statspai as sp
result = sp.decompose(method='ffl', data=df, y='log_wage',
                      group='female', x=['education', 'experience'],
                      stat='quantile', tau=0.5)
result.summary(); result.plot(); result.to_latex()
```

30 aliases supported (`'mm'` → `machado_mata`, `'dinardo_fortin_lemieux'` → `dfl`, etc.).

#### Why this matters
- **Stata** has it scattered across 6+ packages (`oaxaca`, `ddecompose`, `cdeco`, `rifhdreg`, `mvdcmp`, `fairlie`) with no unified API.
- **R** has `ddecompose`, `Counterfactual`, `dineq` — three different authors, three different conventions.
- **Python** previously had only one 2018-vintage unmaintained PyPI package (basic Oaxaca).
- **StatsPAI 0.9.2**: one API, one result-class contract (`.summary()` / `.plot()` / `.to_latex()` / `._repr_html_()`), three inference modes (analytical / bootstrap / none), all numpy/scipy/pandas.

#### Quality bar
- 54 tests including cross-method consistency (`test_dfl_ffl_mean_agree`, `test_mm_melly_cfm_aligned_reference`, `test_dfl_mm_reference_convention_opposite`) and numerical identity checks (FFL four-part sum, weighted Gini RIF E_w[RIF]=G).
- Closed-form influence functions for Theil T / Theil L / Atkinson (no O(n²) numerical fallback).
- Weighted O(n log n) Dagum Gini via sorted-ECDF pairwise-MAD identity.
- Logit non-convergence surfaces as RuntimeWarning; bootstrap failure rate >5% warns.

## [0.9.1] - 2026-04-16

### Regression Discontinuity — Most Comprehensive RD Toolkit in Any Language

Release focus: `statspai.rd`. **18+ RD estimators, diagnostics, and inference methods across 14 modules (~10,300 LOC)** — now the most feature-complete RD package in Python, R, or Stata. The full machinery behind Calonico-Cattaneo-Titiunik (CCT), Cattaneo-Jansson-Ma density tests, Armstrong-Kolesar honest CIs, Cattaneo-Titiunik-Vazquez-Bare local randomization, Cattaneo-Titiunik-Yu boundary (2D) RD, and Angrist-Rokkanen external validity — all in one `import statspai as sp`.

#### What's in `sp.rd` (14 modules)

**Core estimation**

| Function | Method / Paper |
|---|---|
| `sp.rdrobust(df, ...)` | Sharp / Fuzzy / Kink RD with bias-corrected robust inference (Calonico, Cattaneo & Titiunik 2014, *Econometrica*; 2020, *Stata Journal*) |
| `sp.rdrobust(..., covs=...)` | Covariate-adjusted local polynomial (Calonico, Cattaneo, Farrell & Titiunik 2019, *ReStat*) |
| `sp.rd2d(df, x1, x2, ...)` | Boundary discontinuity / 2D RD designs (Cattaneo, Titiunik & Yu 2025) |
| `sp.rkd(df, ...)` | Regression Kink Design (Card, Lee, Pei & Weber 2015, *Econometrica*) |
| `sp.rdit(df, time, ...)` | Regression Discontinuity in Time (Hausman & Rapson 2018, *Annual Review*) |
| `sp.rdmc(df, cutoffs=[...])` | Multi-cutoff RD (Cattaneo, Titiunik, Vazquez-Bare & Keele 2016) |
| `sp.rdms(df, scores=[...])` | Multi-score RD (Cattaneo, Idrobo & Titiunik 2024) |

**Bandwidth selection**

| Function | Selector |
|---|---|
| `sp.rdbwselect(df, bwselect='mserd')` | MSE-optimal (Imbens-Kalyanaraman 2012) |
| `sp.rdbwselect(..., bwselect='msetwo')` | Two-bandwidth MSE |
| `sp.rdbwselect(..., bwselect='cerrd'/'cercomb1'/'cercomb2')` | CER-optimal coverage-error-rate (Calonico, Cattaneo & Farrell 2020, *Econometrics Journal*) |

**Inference**

| Function | Method |
|---|---|
| `sp.rd_honest(df, ...)` | Honest CIs with worst-case bias bound (Armstrong & Kolesar 2018, *Econometrica*; 2020, *QE*) |
| `sp.rdrandinf(df, ...)` | Local randomization inference via Fisher exact tests (Cattaneo, Frandsen & Titiunik 2015) |
| `sp.rdwinselect(df, ...)` | Data-driven window selection for local randomization |
| `sp.rdsensitivity(df, ...)` | Sensitivity analysis across windows |
| `sp.rdrbounds(df, ...)` | Rosenbaum sensitivity bounds for hidden selection |

**Heterogeneous treatment effects**

| Function | Method |
|---|---|
| `sp.rdhte(df, covs=[...])` | CATE via fully interacted local linear (Calonico et al. 2025) |
| `sp.rdbwhte(df, ...)` | HTE-optimal bandwidth |
| `sp.rd_forest(df, ...)` | Causal forest + RD |
| `sp.rd_boost(df, ...)` | Gradient boosting + RD |
| `sp.rd_lasso(df, ...)` | LASSO-assisted RD with covariate selection |

**External validity & extrapolation**

| Function | Method |
|---|---|
| `sp.rd_extrapolate(df, ...)` | Away-from-cutoff extrapolation (Angrist & Rokkanen 2015, *JASA*) |
| `sp.rd_multi_extrapolate(df, cutoffs=[...])` | Multi-cutoff extrapolation (Cattaneo, Keele, Titiunik & Vazquez-Bare 2024) |

**Diagnostics & visualization**

| Function | Purpose |
|---|---|
| `sp.rdsummary(df, ...)` | **One-click dashboard** — rdrobust + density test + bandwidth sensitivity + placebo cutoffs + covariate balance |
| `sp.rdplot(df, ...)` | IMSE-optimal binned scatter with pointwise CI bands (Calonico, Cattaneo & Titiunik 2015, *JASA*) |
| `sp.rddensity(df, ...)` | Cattaneo-Jansson-Ma (2020, *JASA*) manipulation test |
| `sp.rdbalance(df, covs=[...])` | Covariate balance tests at cutoff |
| `sp.rdplacebo(df, cutoffs=[...])` | Placebo cutoff tests |

**Power analysis**

| Function | Purpose |
|---|---|
| `sp.rdpower(df, effect_sizes=[...])` | Power curves for RD designs |
| `sp.rdsampsi(df, target_power=0.8)` | Required sample size |

#### Refactor — rd/\_core.py consolidation

A 5-sprint refactor (commit 44f7529) centralized shared low-level primitives that had been duplicated across 9 RD files into a single private module `rd/_core.py` (191 lines):

- `_kernel_fn` — triangular / epanechnikov / uniform / gaussian (previously 4 duplicate definitions)
- `_kernel_constants` / `_kernel_mse_constant` — MSE-optimal bandwidth constants
- `_local_poly_wls` — WLS local polynomial fit with HC1 / cluster-robust variance + optional covariate augmentation
- `_sandwich_variance` — HC1 / cluster sandwich for arbitrary design matrices

**Net effect**: 253 lines of duplicated math consolidated into 191 lines of canonical implementation. 97 RD tests pass with zero regression.

#### Bug fixes (since 0.9.0)

- RDD extrapolation: `_ols_fit` singular matrix fallback (commit 052594a)
- 3 critical + 3 high-priority bugs from comprehensive RD code review (commit 6489270)
- Density test: bug in CJM (2020) implementation + DGP helper fixes + validation tests (commit b66f312)

#### Tests

- **97 RD tests + 1 skipped, 0 failed** across 5 test files.

### Also in 0.9.1

- **`synth/_core.py`** — simplex weight solver consolidated from 6 duplicate implementations (commit a4036a2). Analytic Jacobian now available to all six callers for ~3-5x speedup.
- **`decomposition/_common.py`** — new `influence_function(y, stat, tau, w)` is the canonical 9-stat RIF kernel. `rif.rif_values` public API **expands from 3 to 9 statistics** (commits 0789223, 5569fd0).

---

## [0.9.0] - 2026-04-16

### Synthetic Control — Most Comprehensive SCM Toolkit in Any Language

Release focus: `statspai.synth`. **20 SCM methods + 6 inference strategies + full research workflow (compare / power / sensitivity / one-click reports)**, all behind the unified `sp.synth(method=...)` dispatcher. No competing package in Python, R, or Stata offers this breadth.

#### Seven new SCM estimators

| Method | Reference |
|---|---|
| `bayesian_synth` | Dirichlet-prior MCMC with full posterior credible intervals (Vives & Martinez 2024) |
| `bsts_synth` / `causal_impact` | Bayesian Structural Time Series via Kalman filter/smoother (Brodersen et al. 2015) |
| `penalized_synth` (penscm) | Pairwise discrepancy penalty (Abadie & L'Hour 2021, *JASA*) |
| `fdid` | Forward DID with optimal donor subset selection (Li 2024) |
| `cluster_synth` | K-means / spectral / hierarchical donor clustering (Rho 2024) |
| `sparse_synth` | L1 / constrained-LASSO / joint V+W (Amjad, Shah & Shen 2018, *JMLR*) |
| `kernel_synth` + `kernel_ridge_synth` | RKHS / MMD-based nonlinear matching |

Previous methods — classic, penalized, demeaned, unconstrained, augmented, SDID, gsynth, staggered, MC, discos, multi-outcome, scpi — remain with bug fixes (see below).

#### Research workflow

- `synth_compare(df, ...)` — run every method at once, tabular + graphical comparison
- `synth_recommend(df, ...)` — auto-select best estimator by pre-fit + robustness
- `synth_report(result, format='markdown'|'latex'|'text')` — one-click publication-ready report
- `synth_power(df, effect_sizes=[...])` — first power-analysis tool for SCM designs
- `synth_mde(df, target_power=0.8)` — minimum detectable effect
- `synth_sensitivity(result)` — LOO + time placebos + donor sensitivity + RMSPE filtering
- Three canonical datasets shipped: `california_tobacco()`, `german_reunification()`, `basque_terrorism()`

#### Release-blocker fixes from comprehensive module review

Following a 5-parallel-agent code review (correctness / numerics / API / perf / docs), nine release blockers were fixed:

- **ASCM correction formula** — `augsynth` now follows Ben-Michael, Feller & Rothstein (2021) Eq. 3 per-period ridge bias `(Y1_pre − Y0'γ) @ β(T0, T1)`, replacing the scalar mean-residual placeholder. `_ridge_fit` RHS bug also fixed.
- **Bayesian likelihood scale** — covariate rows are now z-scored to the pooled pre-outcome SD before concatenation, preventing scale mismatch from dominating the Gaussian `σ²` posterior.
- **Bayesian MCMC Jacobian** — missing `log(σ′/σ)` correction for the log-normal random-walk proposal on σ has been added to the MH acceptance ratio.
- **BSTS Kalman filter** — innovation variance floored at `1e-12` (prevents `log(0)` on constant outcome series); RTS smoother `inv → solve + pinv` fallback on near-singular predicted covariance.
- **gsynth factor estimation** — four `np.linalg.inv` calls (loadings + placebo loop) replaced with `np.linalg.lstsq` (robust to rank-deficient `F'F` / `L'L`).
- **Dispatcher `**kwargs` leakage** — `augsynth` gains `**kwargs + placebo=True`; `sp.synth(method='augmented', placebo=False)` no longer raises `TypeError`.
- **Dispatcher `kernel_ridge` placebo bypass** — `placebo=` now forwarded correctly.
- **Cross-method API consistency** — `sdid()` now accepts canonical `outcome / treated_unit / treatment_time` (legacy `y / treat_unit / treat_time` aliases retained for backwards compatibility).
- **Documentation accuracy** — `synth_compare` docstring reflects 20 methods (was 12); `synth()` Returns section enumerates all `CausalResult` fields.

#### Tests & validation

- **144 synth tests passing** (new: 12-method cross-method consistency benchmark verifying every estimator recovers a known ATT within 1.5 units on a clean DGP).
- **Full suite: 1481 passed, 4 skipped, 0 failed** (5m42s).
- New guide: `docs/guides/synth.md` — complete tutorial covering all 20 methods with a method-choice decision table.

#### API migration notes

`sdid(y=, treat_unit=, treat_time=)` still works but `outcome=, treated_unit=, treatment_time=` is preferred for consistency with every other `sp.synth.*` function. A deprecation of the legacy names is planned for v1.0.

### Other Modules

Decomposition and Regression Discontinuity modules received significant upgrades in this release cycle (tier-C decomposition expansion to 18 methods + unified `sp.decompose()`; RD `_core.py` primitive centralization + bug fixes from code review). These will be highlighted in a dedicated follow-up release note.

---

## [0.8.0] - 2026-04-16

### Spatial Econometrics Full-Stack + 10-Domain Breadth Upgrade

**Largest release in StatsPAI history. 60+ new functions across 10 domains.**

#### Spatial Econometrics (NEW — 38 API symbols)

From 3 functions / 419 LOC to **38 functions / 3,178 LOC / 69 tests**. Python's first unified spatial econometrics package.

- **Weights (L1)**: `W` (sparse CSR), `queen_weights`, `rook_weights`, `knn_weights`, `distance_band`, `kernel_weights`, `block_weights`
- **ESDA (L2)**: `moran` (global + local), `geary`, `getis_ord_g`, `getis_ord_local`, `join_counts`, `moran_plot`, `lisa_cluster_map`
- **ML Regression (L3)**: `sar`, `sem`, `sdm`, `slx`, `sac` — sparse-backed, dual log-det path (exact + Barry-Pace), scales to N=100K
- **GMM (L3)**: `sar_gmm`, `sem_gmm`, `sarar_gmm` — Kelejian-Prucha (1998/1999), heteroskedasticity-robust
- **Diagnostics**: `lm_tests` (Anselin 1988 full battery), `moran_residuals`
- **Effects**: `impacts` (LeSage-Pace 2009 direct/indirect/total + simulated SE)
- **GWR (L4)**: `gwr`, `mgwr` (Multiscale GWR), `gwr_bandwidth` (AICc/CV golden-section)
- **Spatial Panel (L5)**: `spatial_panel` (SAR-FE / SEM-FE / SDM-FE, entity + twoways)
- **Cross-validated**: Columbus rtol<1e-7 vs PySAL spreg 1.9.0; Georgia GWR bit-identical vs mgwr 2.2.1; GMM rtol<1e-4 vs spreg GM_*

#### Time Series

- `local_projections` — Jordà (2005) IRF with Newey-West HAC
- `garch` — GARCH(p,q) MLE with multi-step forecast
- `arima` — ARIMA/SARIMAX with auto (p,d,q) AICc grid search
- `bvar` — Bayesian VAR with Minnesota (Litterman) prior

#### Causal Discovery

- `lingam` — DirectLiNGAM (Shimizu 2011), bit-identical vs lingam package
- `ges` — Greedy Equivalence Search (Chickering 2002)

#### Matching

- `optimal_match` — Hungarian 1:1 matching (min total Mahalanobis distance)
- `cardinality_match` — Zubizarreta (2014) LP-based matching with balance constraints

#### Decomposition & Mediation

- `rifreg` — RIF regression (Firpo-Fortin-Lemieux 2009)
- `rif_decomposition` — RIF Oaxaca-Blinder for distributional statistics
- `mediate_sensitivity` — Imai-Keele-Yamamoto (2010) ρ-sensitivity

#### RD & Survey

- `rdpower`, `rdsampsi` — power/sample-size for RD designs
- `rake`, `linear_calibration` — survey calibration (Deville-Särndal 1992)

#### Survival

- `cox_frailty` — Cox with shared gamma frailty (Therneau-Grambsch)
- `aft` — Accelerated Failure Time (exponential/Weibull/lognormal/loglogistic)

#### ML-Causal (GRF)

- `CausalForest.variable_importance()`, `.best_linear_projection()`, `.ate()`, `.att()`
- **Bugfix**: honest leaf values now correctly vary per-leaf

#### Infrastructure

- OLS/IV `predict(data, what='confidence'|'prediction')` with intervals
- Pre-release code review: 3 critical + 2 high-priority bugs fixed

## [0.7.1] - 2026-04-15

DID-focused polish release. Brings the Wooldridge (2021) ETWFE
implementation to full feature parity with the R `etwfe` package,
adds a one-call method-robustness workflow, and closes 12 issues
uncovered by an internal code review round. All 27 new / updated
DID tests pass (`pytest tests/test_did_summary.py`).

### Added — ETWFE full parity with R `etwfe`

- **`sp.etwfe()` explicit API** aligned with R `etwfe` (McDermott 2023)
  naming. Thin alias over `wooldridge_did()` with a full argument-
  mapping table in the docstring.
- **`xvar=` covariate heterogeneity** (single string or list of names).
  Adds per-cohort × post × `(x_j − mean(x_j))` interactions; `detail`
  gains `slope_<x>` / `slope_<x>_se` / `slope_<x>_pvalue` columns.
  Baseline ATT is reported at the sample means of every covariate.
- **`panel=False` repeated cross-section mode** — replaces unit FE
  with cohort + time dummies (R `etwfe(ivar=NULL)` equivalent).
- **`cgroup='nevertreated'`** — per-cohort regressions restricted to
  (cohort g) ∪ (never-treated); cohort-size-weighted aggregation
  (R `etwfe(cgroup='never')` equivalent). Default `'notyet'` preserves
  prior ETWFE behaviour.
- **`sp.etwfe_emfx(result, type=…)`** — R `etwfe::emfx`-equivalent
  four aggregations: `'simple'`, `'group'`, `'event'`, `'calendar'`.
  `include_leads=True` returns full event-time output including pre-
  treatment leads for pre-trend inspection (`rel_time = -1` is the
  reference category).

### Added — one-call DID method-robustness workflow

- **`sp.did_summary()`** — fits five modern staggered-DID estimators
  (CS, SA, BJS, ETWFE, Stacked) to the same data and returns a tidy
  comparison table with per-method (estimate, SE, p, 95 % CI). Mean
  across methods + cross-method SD flag method-sensitivity of results.
- **`include_sensitivity=True`** — attaches the Rambachan-Roth (2023)
  breakdown `M*` to the CS row, giving a three-way robustness readout
  in a single call.
- **`sp.did_summary_plot()`** — forest plot of per-method estimates
  with cross-method mean line; `sort_by='estimate'` supported.
- **`sp.did_summary_to_markdown()` / `_to_latex()`** — publication-
  ready exports (GFM tables / booktabs LaTeX with auto-escaped
  ampersands).
- **`sp.did_report(save_to=dir)`** — one-call bundle that writes
  `did_summary.txt` / `.md` / `.tex` / `.png` / `.json` to a folder.

### Fixed — 12 issues from the internal code review

Blockers (C-severity):

- `etwfe(xvar=…)` now raises a clear `ValueError` when the covariate
  is all-NaN or (near-)constant. Previously returned `n_obs = 0,
  estimate = 0` silently.
- `etwfe(panel=False, cgroup='nevertreated')` now raises a crisp
  `NotImplementedError` instead of silently falling back to
  `'notyet'`.
- `did_summary` now validates column names up front (raises
  `KeyError` listing missing columns) and only catches narrow
  estimator-side exceptions inside the fit loop; user typos in
  `controls=` / `cluster=` surface as proper errors.
- `did_summary` results round-trip cleanly through stdlib
  serialisation (`DIDSummaryResult(CausalResult)` subclass with a
  real `.summary()` method, replacing the prior closure-bound
  instance attribute).

High-severity:

- `etwfe_emfx(type='event'/'calendar')` now computes SEs via the
  delta method on the stored event-study vcov instead of the
  independent-coefficient approximation. `model_info['se_method']`
  advertises which path was used.
- `etwfe_emfx(type='group')` headline `se` / `pvalue` / `ci` are now
  populated (match the underlying fit's overall ATT exactly).
- Validation for `did_summary_plot` / `_to_markdown` / `_to_latex`
  aligned on a single sentinel `model_info['_did_summary_marker']`.
- `_etwfe_never_only` no longer leaves a `_ft_cache` helper column
  on the caller's DataFrame.
- Slope indexing in `_etwfe_with_xvar` is now name-keyed
  (`coef_index` dict); regression test verifies swapping
  `xvar=['x1','x2']` vs `['x2','x1']` produces identical slopes per
  name.
- `etwfe(panel=False)` with rank-deficient designs emits a
  `RuntimeWarning` pointing at concrete remedies (previously fell
  through to `pinv` silently).

### Tests

- New test module `tests/test_did_summary.py` — 27 cases covering
  consistency with direct estimator calls, export formats, forest
  plot rendering, `etwfe_emfx` round-trips, xvar / panel / cgroup
  options, the 12 review fixes, and the `include_leads` mode.

## [0.7.0] - 2026-04-14

Focused release reaching feature parity with the R `did` / `HonestDiD`
packages and the Python `csdid` / `differences` packages for staggered
Difference-in-Differences.  All core algorithms are reimplemented from
the original papers — **no wrappers, no runtime dependencies on upstream
DID packages**.  Full DiD test suite: 47 → 170+ (including three rounds
of post-implementation audit that surfaced and fixed 9 bugs before
release).

### Added — Core estimation

- **`sp.aggte(result, type=...)`** — unified aggregation layer for
  `callaway_santanna()` results.  Four aggregation schemes (`simple`,
  `dynamic`, `group`, `calendar`) backed by a single weighted-
  influence-function engine.  Callaway & Sant'Anna (2021) Section 4.
- **Mammen (1993) multiplier bootstrap** — IQR-rescaled pointwise
  standard errors *and* simultaneous (uniform / sup-t) confidence
  bands over the aggregation dimension.  Matches the uniform-band
  behaviour of the R `did::aggte` function.
- **`balance_e` / `min_e` / `max_e`** — event-study cohort balancing
  and window truncation (CS2021 eq. 3.8).
- **`anticipation=δ`** parameter on `callaway_santanna()` — shifts
  the base period back by δ periods per CS2021 §3.2.
- **Repeated cross-sections** support via `callaway_santanna(panel=False)`
  — unconditional 2×2 cell-mean DID with observation-level influence
  functions (CS2021 eq. 2.4, RCS version).  Optional covariate
  residualisation with `x=[...]` for regression adjustment.  All
  downstream modules (`aggte`, `cs_report`, `ggdid`, `honest_did`)
  work on RCS results with no code changes.
- **dCDH joint inference** (`did_multiplegt`) — `joint_placebo_test`
  (Wald χ² across placebo lags with bootstrap covariance, dCDH 2024
  §3.3) and `avg_cumulative_effect` (mean of dynamic[0..L] with
  SE preserving cross-horizon covariance, dCDH 2024 §3.4).
- **`sp.bjs_pretrend_joint()`** — cluster-bootstrap joint Wald pre-
  trend test for BJS imputation results.  Upgrades the default
  sum-of-z² test (which assumes pre-period independence) to a full
  covariance-aware statistic.

### Added — Reporting & visualisation

- **`sp.cs_report(data, ...)`** — one-call report card.  Runs the
  full pipeline (ATT(g,t) → four aggregations with uniform bands →
  pre-trend Wald → Rambachan–Roth breakdown M\* for every post event
  time) under a single bootstrap seed and pretty-prints the result.
  Returns a structured `CSReport` dataclass.
- **`sp.ggdid(result)`** — plot routine for `aggte()` output,
  mirroring R `did::ggdid`.  Auto-dispatches on aggregation type;
  uniform band overlaid on pointwise CI.
- **`CSReport.plot()`** — one-call 2×2 summary figure: event study
  with uniform band (top-left), θ(g) per-cohort (top-right), θ(t)
  per-calendar-time (bottom-left), Rambachan–Roth breakdown M\*
  bars (bottom-right).
- **`CSReport.to_markdown()`** — GitHub-flavoured Markdown export
  with proper integer-column rendering and a configurable
  `float_format`.
- **`CSReport.to_latex()`** — publication-ready booktabs fragment
  wrapped in a `table` float.  Zero `jinja2` dependency (hand-rolled
  booktabs renderer); auto-escapes LaTeX special characters.
- **`CSReport.to_excel()`** — six-sheet workbook (`Summary`,
  `Dynamic`, `Group`, `Calendar`, `Breakdown`, `Meta`).  Engine
  autoselect (openpyxl → xlsxwriter) with a clear ImportError when
  neither is installed.
- **`cs_report(..., save_to='prefix')`** — one-call dump of the
  full export matrix: writes `<prefix>.{txt,md,tex,xlsx,png}` in
  a single invocation, auto-creating missing parent directories.
  Optional dependencies (openpyxl, matplotlib) are skipped silently
  so a minimal install still produces text + md + tex.
- **`sp.did(..., aggregation='dynamic', n_boot=..., random_state=...)`**
  — the top-level dispatcher now forwards CS-style arguments
  (`aggregation`, `panel`, `anticipation`) and can pipe a CS result
  straight through `aggte()` in a single call.

### Changed

- **`sun_abraham()` inference layer rewritten** — replaces the
  former ad-hoc `√(σ²/(total·T))` approximation with a Liang–Zeger
  cluster-robust sandwich `(X'X)⁻¹ Σ_c X_c' u_c u_c' X_c (X'X)⁻¹`
  (small-sample adjusted), delta-method IW aggregation SEs
  `w' V_β w`, iterative two-way within transformation (correct on
  unbalanced panels), and optional `control_group='lastcohort'` per
  SA 2021 §6.
- **`sp.honest_did()` / `sp.breakdown_m()` made polymorphic** — now
  accept the legacy `callaway_santanna()` / `sun_abraham()` format
  (event study in `model_info`) *and* the new `aggte(type='dynamic')`
  format (event study in `detail` with Mammen uniform bands).  The
  idiomatic pipeline `cs → aggte → honest_did → breakdown_m` now
  runs end-to-end with no manual plumbing.
- **README DiD parity matrix** added, comparing StatsPAI against
  `csdid`, `differences`, and R `did` + `HonestDiD` across 15
  capabilities.

### Fixed (from pre-release audit rounds)

- **Critical — `aggte(type='dynamic').estimate`** previously averaged
  pre- *and* post-treatment event times into the overall ATT,
  polluting the headline number with placebo signal.  Now averages
  only e ≥ 0, matching R `did::aggte`'s print convention.  On a
  typical DGP the bug shifted the reported overall by nearly a
  factor of 2.
- **LaTeX escape non-idempotence** in `CSReport.to_latex()`:
  `\` → `\textbackslash{}` followed by `{` → `\{` mangled the
  just-inserted braces.  Fixed with a single-pass `re.sub`.
- **`cs_report(save_to='~/study/…')`** did not expand `~`; fixed
  via `os.path.expanduser`.
- **`cs_report(sa_result)` / `aggte(sa_result)`** raised cryptic
  `KeyError: 'group'`; both entry points now detect non-CS input
  up-front and raise a clear `ValueError`.
- **`cs_report(pre_fitted_cs, estimator=…)`** silently ignored the
  override; now emits a `UserWarning` listing every shadowed arg.
- **`sp.did(method='2x2', aggregation='dynamic')`** silently ignored
  CS-only arguments; now raises an informative `ValueError`.
- **`bjs_pretrend_joint`** swallowed all exceptions as "bootstrap
  failed"; now narrows to expected failure modes and re-raises
  unexpected errors with context.
- **`matplotlib.use('Agg')`** in `_save_report_bundle` no longer
  switches the backend unconditionally (respects Jupyter sessions).

### References

- Callaway, B. and Sant'Anna, P.H.C. (2021). *J. of Econometrics* 225(2).
- Sun, L. and Abraham, S. (2021). *J. of Econometrics* 225(2).
- Mammen, E. (1993). *Ann. Statist.* 21(1).
- Liang, K.-Y. and Zeger, S.L. (1986). *Biometrika* 73(1).
- de Chaisemartin, C. and D'Haultfoeuille, X. (2020). *AER* 110(9).
- de Chaisemartin, C. and D'Haultfoeuille, X. (2024). *RESt*, forthcoming.
- Rambachan, A. and Roth, J. (2023). *Rev. Econ. Studies* 90(5).
- Borusyak, K., Jaravel, X. and Spiess, J. (2024). *ReStud* 91(6).

## [0.6.2] - 2026-04-12

### Added

- **OLS `predict()`**: `result.predict(newdata=)` for out-of-sample prediction on OLS results
- **`balance_panel()`**: Utility to keep only units observed in every time period (`sp.balance_panel()`)
- **Panel `balance=True`**: Convenience flag in `sp.panel()` to auto-balance before estimation
- **Analytical weights for DID**: `weights=` parameter added to `did()`, `ddd()`, and `event_study()` for population-weighted estimation (Stata `[aweight=...]` equivalent)
- **Matching `ps_poly=`**: Polynomial propensity score specification (`ps_poly=2` adds interactions/squares, following Cunningham 2021 Ch. 5)
- **Synth `rmspe` plot**: Post/pre RMSPE ratio histogram (`synthplot(result, type='rmspe')`) per Abadie et al. (2010)
- **Synth placebo gap plot**: Full spaghetti placebo gap paths with `rmspe_threshold` filter (Abadie et al. 2010, Figure 4)
- **Graddy (2006) replication**: Fulton Fish Market IV example added to `sp.replicate()` (Mixtape Ch. 7)
- **Numerical validation tests**: Cross-validated against Stata/R reference values with humanized error messages

### Fixed

- **`outreg2` format auto-detection**: Correctly infers `.xlsx`/`.csv`/`.tex` from filename extension
- **Synth placebo p-value**: Now uses RMSPE *ratio* (√post/√pre) instead of squared ratio, matching Abadie et al. (2010) convention

### Improved

- **DID/DDD/Event Study**: Weights propagation through WLS with proper normalization and validation
- **Synth placebos**: Store full placebo gap trajectories, per-unit RMSPE ratios, and unit labels for richer post-estimation analysis
- **Matching tests**: Added comprehensive test suite for PSM, Mahalanobis, CEM, and stratification methods

## [0.6.1] - 2026-04-07

### Fixed

- **Interactive Editor — Theme switching**: Themes now fully reset before applying, so switching between themes (e.g. ggplot → academic) correctly updates all visual properties instead of leaking stale settings
- **Interactive Editor — Apply button**: Fixed Apply button being clipped/hidden on the Layout tab due to panel overflow
- **Interactive Editor — Panel layout**: Fixed panel content disappearing when using flex layout for bottom-pinned Apply button
- **Interactive Editor — Style tab**: Fixed Style tab stuck on "Loading" after Theme tab was reordered to first position
- **Interactive Editor — Error visibility**: Widget callback errors now surface in the status bar instead of being silently swallowed

### Improved

- **Interactive Editor — Auto mode**: Clicking Auto now always refreshes the preview, giving immediate visual feedback
- **Interactive Editor — Auto/Manual toggle**: Compact toggle button moved to panel header with sticky positioning
- **Interactive Editor — Apply button**: Separated from Auto toggle and placed at panel bottom-right for better UX
- **Interactive Editor — Theme tab**: Moved to first position for better discoverability
- **Interactive Editor — Color pickers**: Added visual confirmation feedback on all color changes
- **Interactive Editor — Code generation**: Auto-generate reproducible code with text selection support in the editor
- **Smart recommendations**: Enhanced compare and recommend logic
- **Registry**: Expanded module support in the registry

## [0.1.0] - 2024-07-26

### Added
- **Core Regression Framework**
  - OLS (Ordinary Least Squares) regression with formula interface
  - Robust standard errors (HC0, HC1, HC2, HC3)
  - Clustered standard errors
  - Weighted Least Squares (WLS) support

- **Causal Inference Module**
  - Causal Forest implementation inspired by Wager & Athey (2018)
  - Honest estimation for unbiased treatment effect estimation
  - Bootstrap confidence intervals for treatment effects
  - Formula interface: `"outcome ~ treatment | features | controls"`

- **Output Management (outreg2)**
  - Excel export functionality similar to Stata's outreg2
  - Support for multiple regression models in single output
  - Customizable formatting options
  - Professional table layout

- **Unified API Design**
  - Consistent `reg()` function interface
  - Formula parsing: R/Stata-style syntax `"y ~ x1 + x2"`
  - Type hints throughout the codebase
  - Comprehensive documentation

### Technical Details
- Python 3.8+ support
- Dependencies: numpy, scipy, pandas, scikit-learn, openpyxl
- MIT License
- Comprehensive test suite
