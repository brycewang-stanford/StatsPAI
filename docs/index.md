# StatsPAI

**The agent-native Python toolkit for causal inference and applied
econometrics.** One `import statspai as sp` exposes **800+ functions**
spanning classical regression, staggered DiD, regression discontinuity,
synthetic control, decomposition, stochastic frontier, multilevel /
mixed-effects, modern ML causal inference, the full three-school
(Econometrics / Epidemiology / ML) toolkit, 2025-2026 research-frontier
modules (bridging theorems, fairness, surrogates, PCMCI, TMLE survival,
etc.), and publication-ready output in Word / Excel / LaTeX / HTML.

> **Current release: v1.4.1 (2026-04-21)** — v3-frontier Sprint 3.
> Additive follow-up to v1.4.0: AKM shock-clustered SE for
> `sp.shift_share_political_panel(cluster='shock')` (Park-Xu 2026 §4.2,
> typically 3× tighter than unit-clustered SEs); Claude 4.5 / Opus 4.7
> **extended-thinking** support via
> `sp.causal_llm.anthropic_client(thinking_budget=N)`; a 10-check
> assimilation parity suite plus an 11-test MAS integration suite with
> 3 Claude thinking block-splitter tests; two new MkDocs guides
> (`shift_share_political_panel`, `causal_mas`). All v1.4.0 APIs
> stable; new surface is additive kwargs only. See the
> [changelog](changelog.md) for detail.

```python
import statspai as sp

# One-call DiD pipeline with sensitivity + export
rpt = sp.cs_report(data, y='y', g='g', t='t', i='id',
                   n_boot=500, random_state=0,
                   save_to='~/study/cs_v1')
```

## What's inside

### Release highlights (v0.9.17 → v1.4.1)

| Release | Focus | Headline |
| --- | --- | --- |
| **v1.4.1** | v3-frontier Sprint 3 (AKM SE + Claude thinking + test suites + docs) | `sp.shift_share_political_panel(cluster='shock')` — panel-extended Adão-Kolesár-Morales (2019) shock-cluster variance (Park-Xu 2026 §4.2); `sp.causal_llm.anthropic_client(thinking_budget=N)` — Claude 4.5 / Opus 4.7 extended-thinking API; 10-check assimilation parity suite + 11-test MAS integration suite with 3 Claude thinking block-splitter tests; two new MkDocs guides (`shift_share_political_panel`, `causal_mas`). Strictly additive over v1.4.0. |
| **v1.4.0** | v3-frontier Sprint 2 (extensions + LLM SDK + docs) | `shift_share_political_panel` (Park-Xu 2026 multi-period); real LLM adapters `openai_client` / `anthropic_client` / `echo_client` for Causal MAS; `particle_filter` backend for `assimilative_causal` (non-Gaussian / nonlinear); three new MkDocs guides (`synth_experimental`, `harvest_did`, `assimilative_ci`); 20 unused-import cleanups; CausalForest parity-test de-flake. |
| **v1.3.0** | v3-frontier sprint (Sprint 1 of 知识地图 v3) | 11 frontier methods: Abadie-Zhao inverse synthetic experimental design, CJM RBC bootstrap for `rdrobust`, Kwak-Pleasants evidence-without-injustice fairness test, JAMA/BMJ TARGET manuscript renderer, Borusyak harvest-DiD, Zorzetto ordinal / factor-exposure BCF, multi-agent causal discovery (`causal_mas`), Park-Xu political shift-share IV, state-space `causal_kalman`. 35 new tests, 869 registered functions, zero regressions. `tabulate` promoted to core dep. |
| **v1.0.1** | Post-review correctness + NEEDS_VERIFICATION closeout | All Critical / High / Medium findings from the independent code-review-expert pass on v1.0 frontier modules fixed and pinned by regression tests. Abadie κ-weighted complier QTE now implemented for `beyond_average_late`; `bridge.surrogate_pci` path B now a genuine dual-path (arm-specific counterfactual bridge), not OLS-tautology. 2 706+ tests passing. |
| **v1.0.0** | Research-frontier capstone | `sp.bridge` (6 bridging theorems), `sp.fairness`, `sp.surrogate`, `mr_multivariable`/`mr_mediation`/`mr_bma`, PCMCI/LPCMCI/DYNOTEARS, conformal frontiers (debiased/density/fair/multi-DP), proximal frontiers, sequential SDID, BCF longitudinal, LTMLE survival, ML bounds, JAMA/BMJ 2025 TARGET Statement 21-item reporting checklist. |
| **v0.9.17** | Three-school completion | `sp.epi` (OR/RR/MH/standardization/Bradford-Hill/ROC/kappa), `sp.longitudinal` (MSM/g-formula/IPW unified + safe regime DSL), `sp.question` (estimand-first DSL), full MR diagnostic suite, DAG `recommend_estimator()`, unified `result.sensitivity()`, `preregister()` + `load_preregister()`. |
| **v0.9.3** | Frontier + Multilevel + GLMM + Trinity | `sp.frontier` / `sp.xtfrontier` full Stata/R parity; `sp.zisf`, `sp.lcsf`, `sp.malmquist`; `sp.mixed` lme4-grade; GLMMs with AGHQ; `sp.dml(model='pliv')`, `sp.mixlogit`, `sp.ivqreg`; `sp.verify` posterior verification. |
| **v0.9.2** | Decomposition | 18 methods under `sp.decompose(method=...)`. |
| **v0.9.1** | Regression discontinuity | 18+ estimators across 14 modules. |
| **v0.9.0** | Synthetic control | 20 SCM estimators + 6 inference strategies. |

### Methodological coverage

**Regression & panel.** OLS / IV / panel / GLM; fixed-effect high-
dimensional estimation; GMM; quantile regression; instrumental-variable
quantile regression (`sp.ivqreg`); mixed logit (`sp.mixlogit`).

**Difference-in-differences (10+ variants).**
`sp.callaway_santanna` (DR/IPW/REG), `sp.aggte` with Mammen uniform
bands, `sp.sun_abraham`, `sp.bjs` (Borusyak-Jaravel-Spiess imputation),
`sp.dcdh` (de Chaisemartin-D'Haultfoeuille), `sp.etwfe`,
`sp.goodman_bacon`; sensitivity via `sp.honest_did`, `sp.breakdown_m`;
one-call `sp.cs_report` with Markdown / LaTeX / Excel export.

**Regression discontinuity (18+ estimators).**
`sp.rdrobust` (CCT sharp/fuzzy/kink with bias-corrected robust CI),
`sp.rd2d` (2D/boundary), `sp.rkd`, `sp.rdit`, multi-cutoff and
multi-score designs, `sp.rdhonest` (Armstrong-Kolesar), local
randomization (`sp.rdrandinf`, `sp.rdwinselect`, `sp.rdsensitivity`),
`sp.cjm_density`, ML-based CATE (`sp.rd_forest`, `sp.rd_boost`,
`sp.rd_lasso`), Angrist-Rokkanen extrapolation, `sp.rdpower`,
`sp.rdsampsi`, one-click `sp.rdsummary` dashboard.

**Synthetic control (20 estimators).**
Classical SCM, SDID, Augmented SCM (ASCM), Bayesian SCM (MCMC), BSTS
and CausalImpact (Kalman smoother), Penalized SCM (Abadie-L'Hour),
Forward-DID, cluster SCM, sparse (LASSO) SCM, kernel and kernel-ridge
SCM, staggered synthetic control, multi-outcome SCM; research workflow:
`sp.synth_compare`, `sp.synth_recommend`, `sp.synth_power`,
`sp.synth_mde`, `sp.synth_sensitivity`, `sp.synth_report`.

**Decomposition analysis (18 methods).**
Mean: `sp.oaxaca` (5 reference coefficients), `sp.gelbach`,
`sp.fairlie`, `sp.bauer_sinning`, `sp.yun_nonlinear`.
Distributional: `sp.rifreg`, `sp.ffl_decompose`, `sp.dfl_decompose`,
`sp.machado_mata`, `sp.melly_decompose`, `sp.cfm_decompose`.
Inequality: `sp.subgroup_decompose`, `sp.shapley_inequality`,
`sp.source_decompose`.
Demographic: `sp.kitagawa_decompose`, `sp.das_gupta`.
Causal: `sp.gap_closing`, `sp.mediation_decompose`,
`sp.disparity_decompose`. Unified entry: `sp.decompose(method=…)`.

**Stochastic frontier (v0.9.3).**
`sp.frontier` cross-sectional with half-normal / exponential /
truncated-normal, heteroskedastic `usigma` / `vsigma`, Battese-Coelli
(1995) determinants `emean`, Battese-Coelli (1988) TE and JLMS,
Kodde-Palm LR mixed-$\bar\chi^2$ test, bootstrap unit-efficiency CI.
`sp.xtfrontier` panel with Pitt-Lee (1981), BC92 time-decay, BC95, Greene
(2005) TFE/TRE with Dhaene-Jochmans (2015) jackknife. `sp.zisf`,
`sp.lcsf`, `sp.malmquist` (M = EC × TC), `sp.translog_design`.

**Multilevel / mixed-effects (v0.9.3).**
`sp.mixed` linear mixed models with unstructured G default, three-level
nested, BLUP posterior SEs, Nakagawa-Schielzeth $R^2$. GLMMs
(`sp.melogit`, `sp.mepoisson`, `sp.meglm`, `sp.megamma`, `sp.menbreg`,
`sp.meologit`) via Laplace or adaptive Gauss-Hermite quadrature
(`nAGQ>1` matches Stata `intpoints()` and R `lme4::glmer`).
`sp.icc` with delta-method CI; `sp.lrtest` with Self-Liang boundary
correction.

**Modern ML causal.**
Double/debiased ML (`sp.dml` with PLR / IRM / PLIV); causal forests;
meta-learners (S / T / X / R / DR); TMLE and Super Learner; neural
causal (TARNet, CFRNet, DragonNet); causal discovery (NOTEARS, PC,
LiNGAM, GES); policy trees; Bayesian causal forests; matrix
completion; conformal causal inference; dose-response; dynamic-
treatment regimes; interference / spillover.

**Spatial, time-series, survival, survey, bunching, Mendelian.**
Spatial econometrics (weights, ESDA, ML/GMM, GWR/MGWR, spatial panel);
time-series (ARIMA, VAR, BVAR, GARCH, cointegration, local
projections, structural break); survival (Cox, AFT, frailty); survey
calibration and complex-survey regression; bunching; Mendelian
randomization.

**Sensitivity analysis.**
Oster bounds; sensemakr; E-values; Rosenbaum bounds; Manski bounds;
`sp.spec_curve()` specification curve analysis;
`sp.robustness_report()` one-call battery.

### Smart Workflow

```python
# Recommend estimators + run posterior verification
rec  = sp.recommend(df, outcome='y', treatment='d', verify=True)
rec.summary()                # ranked estimators with verify_score
rec.plot('verify_radar')     # visual stability check
```

### Agent-native API

Every function is discoverable programmatically:

```python
sp.list_functions(category='did')        # enumerate methods
sp.describe_function('rdrobust')         # natural-language description
sp.function_schema('dml')                # JSON schema: args, types, returns
```

## Installation

```bash
pip install statspai                       # core
pip install 'statspai[plotting]'           # matplotlib + seaborn
pip install 'statspai[fixest]'             # pyfixest HDFE
pip install 'statspai[deepiv]'             # PyTorch (Deep IV, TARNet)
```

## Citation

If you use StatsPAI in research, please cite the underlying papers
implemented by each estimator (every result object carries a `.cite()`
method that returns the correct BibTeX entry) and this package:

```bibtex
@software{statspai,
  author  = {Wang, Biaoyue},
  title   = {StatsPAI: A Unified, Agent-Native Python Toolkit for
             Causal Inference and Applied Econometrics},
  year    = {2026},
  version = {1.4.1},
  url     = {https://github.com/brycewang-stanford/StatsPAI}
}
```

## Further reading

- [Changelog](changelog.md) — release history, including the critical
  frontier correctness fix in v0.9.3.
- [Choosing a DID estimator](guides/choosing_did_estimator.md) — how
  to pick between TWFE / CS / Sun-Abraham / BJS / multiple-groups DID.
- [Callaway–Sant'Anna staggered DID](guides/callaway_santanna.md) —
  end-to-end tutorial with `cs_report()` and honest sensitivity.
- [Synth guide](guides/synth.md) — synthetic control with inference
  and research workflow.
- [GitHub](https://github.com/brycewang-stanford/StatsPAI) —
  source, issues, and API reference.
