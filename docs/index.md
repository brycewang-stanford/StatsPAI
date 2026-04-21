# StatsPAI

**The agent-native Python toolkit for causal inference and applied
econometrics.** One `import statspai as sp` exposes **550+ functions**
spanning classical regression, ten+ DID variants, eighteen+ regression-
discontinuity estimators, twenty synthetic-control estimators, eighteen
decomposition methods, stochastic frontier analysis, multilevel/mixed-
effects models, modern ML causal inference, and publication-ready
output in Word / Excel / LaTeX / HTML.

> **Current release: v0.9.3 (2026-04-19)** — Stochastic Frontier +
> Multilevel + GLMM + Econometric Trinity. **⚠️ Critical correctness
> fix** in `sp.frontier`: all prior versions ($\leq 0.9.2$) carried a
> Jondrow-posterior sign error that biased efficiency scores and caused
> the exponential path to return NaN. **Re-run any prior frontier
> analyses.** See the [changelog](changelog.md) for full detail.

```python
import statspai as sp

# One-call DiD pipeline with sensitivity + export
rpt = sp.cs_report(data, y='y', g='g', t='t', i='id',
                   n_boot=500, random_state=0,
                   save_to='~/study/cs_v1')
```

## What's inside

### Release highlights (v0.9.0 → v0.9.3)

| Release | Focus | Headline |
| --- | --- | --- |
| **v0.9.3** | Frontier + Multilevel + GLMM + Trinity | `sp.frontier` / `sp.xtfrontier` full Stata/R parity; `sp.zisf`, `sp.lcsf`, `sp.malmquist`; `sp.mixed` lme4-grade with unstructured G, 3-level nested, BLUP SEs; GLMMs (`melogit`/`mepoisson`/`meglm`/`megamma`/`menbreg`/`meologit`) with AGHQ (`nAGQ>1`); `sp.dml(model='pliv')`, `sp.mixlogit`, `sp.ivqreg`; `sp.verify` posterior verification. |
| **v0.9.2** | Decomposition | 18 methods under `sp.decompose(method=...)` — Oaxaca/Gelbach/Fairlie/FFL/DFL/Machado-Mata/Melly/CFM/Theil/Atkinson/Dagum/Shapley/Kitagawa/Das-Gupta/gap-closing/mediation/disparity. |
| **v0.9.1** | Regression discontinuity | 18+ estimators across 14 modules — CCT sharp/fuzzy/kink, `rd2d`, RDIT, multi-cutoff, honest CIs, local randomization, CJM density tests, `rd_forest`/`rd_boost`/`rd_lasso`, Angrist-Rokkanen, `rdpower`/`rdsampsi`. |
| **v0.9.0** | Synthetic control | 20 SCM estimators + 6 inference strategies — SCM, SDID, ASCM, Bayesian SCM, BSTS/CausalImpact, PenSCM, Forward-DID, cluster, sparse, kernel, `synth_compare` / `synth_recommend` / `synth_power` / `synth_sensitivity`. |

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
  version = {0.9.3},
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
