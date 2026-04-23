---
title: 'StatsPAI: A Unified, Agent-Native Python Toolkit for Causal Inference and Applied Econometrics'
tags:
  - Python
  - causal inference
  - econometrics
  - difference-in-differences
  - regression discontinuity
  - synthetic control
  - stochastic frontier
  - mixed-effects models
  - decomposition analysis
  - machine learning
  - heterogeneous treatment effects
authors:
  - name: Biaoyue Wang
    orcid: 0000-0002-1828-2208
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: StatsPAI
    index: 1
  - name: Stanford REAP Program, Stanford University, United States
    index: 2
    ror: 00f54p054
date: 20 April 2026
bibliography: paper.bib
---

# Summary

`StatsPAI` is an open-source Python package providing a unified API for
causal inference and applied econometrics. A single import
(`import statspai as sp`) exposes **550+ public functions** spanning
classical econometric models, modern ML-based causal methods, stochastic
frontier analysis, multilevel mixed-effects models, decomposition
analysis, and publication-ready output generation. The package
consolidates functionality that previously required dozens of separate
R packages (`did`, `rdrobust`, `Synth`, `sfaR`, `lme4`, `oaxaca`,
`ddecompose`, `mlogit`, `grf`, …) or proprietary software such as
Stata (`frontier`, `xtfrontier`, `mixed`, `meglm`, `mixlogit`, `ivqreg`,
…) into one coherent library. Uniquely, `StatsPAI` is *agent-native*:
every function exposes structured result objects and machine-readable
schemas (`list_functions()`, `describe_function()`, `function_schema()`),
making it the first econometrics toolkit purpose-built for LLM-driven
research workflows while remaining fully ergonomic for human researchers.

# Statement of Need

Empirical researchers face a fragmented software landscape for causal
inference and applied econometrics. Stata offers a broad but proprietary
and closed ecosystem that lacks modern ML causal methods and exposes no
programmatic schema for AI agent integration. R scatters equivalent
functionality across twenty or more packages with incompatible APIs,
data structures, and output conventions. Python's existing libraries
occupy non-overlapping niches: `DoWhy` [@sharma2020dowhy] emphasises
causal graphs and assumption refutation; `EconML` [@econml] focuses on
ML-based heterogeneous treatment effects; `CausalML` [@chen2020causalml]
specialises in uplift modelling; `DoubleML` [@bach2022doubleml] provides
selected double/debiased ML estimators; `linearmodels` covers IV and
panel estimation. None provides the full empirical workflow---from OLS
and panel models through DID, RD, synthetic control, decomposition,
stochastic frontier, and mixed-effects models to publication tables in
Word, Excel, and LaTeX---in a single interface.

`StatsPAI` addresses this gap for applied researchers who need to move
fluidly between classical and modern methods, and for AI coding agents
that discover and invoke statistical functions through self-describing
schemas.

# State of the Field

Several Python packages address parts of the causal inference pipeline.
`DoWhy` [@sharma2020dowhy] provides a graph-based framework for causal
assumptions but does not implement econometric estimators such as DID,
RD, or IV. `EconML` [@econml] offers ML-based heterogeneous treatment
effect estimators (DML, causal forests, DR-Learner) yet omits classical
regression, panel methods, and publication output. `DoubleML`
[@bach2022doubleml] implements Chernozhukov et al.'s double/debiased ML
for partially linear and interactive regression models but not, until
`StatsPAI` 0.9.3, the partially linear IV variant. `CausalML`
[@chen2020causalml] targets uplift modelling for marketing applications.
On the R side, packages like `did` [@callaway2021difference],
`rdrobust` [@calonico2014robust], `Synth` [@abadie2010synthetic],
`sfaR`, `lme4` [@bates2015lme4], `oaxaca`, and `grf`
[@wager2018estimation] each excel in one method family but require
users to learn separate APIs and data structures. Stata remains the
most integrated platform for applied economists, yet it is
proprietary, lacks native ML causal methods, and offers no
programmatic schema for AI agent integration.

No existing package unifies classical econometric models, modern
ML-based causal estimators, stochastic frontier analysis,
multilevel/mixed-effects models, decomposition analysis, sensitivity
analysis, and publication-ready output within a single API---nor
exposes machine-readable function schemas for LLM-driven workflows.
`StatsPAI` fills this gap.

# Software Design

`StatsPAI` is organised into roughly fifty modular subpackages. All
functions return structured result objects inheriting from
`CausalResult` or `EconometricResults`, providing a consistent
interface: `.summary()`, `.plot()`, `.to_latex()`, `.to_docx()`, and
`.cite()`.

**Methodological coverage** includes:

- **Classical regression and panel:** OLS/IV/panel/GLM,
  fixed-effect high-dimensional estimation.
- **Difference-in-differences (10+ variants):** Callaway and Sant'Anna
  [@callaway2021difference], Sun and Abraham [@sun2021estimating],
  Borusyak--Jaravel--Spiess imputation, de Chaisemartin--D'Haultfoeuille,
  ETWFE, Goodman-Bacon decomposition [@goodmanbacon2021difference], and
  Rambachan--Roth honest-parallel-trends sensitivity.
- **Regression discontinuity (18+ estimators, v0.9.1):** sharp/fuzzy/kink
  RD with bias-corrected robust inference [@calonico2014robust];
  covariate-adjusted local polynomials; 2D / boundary RD (Cattaneo,
  Titiunik and Yu 2025); multi-cutoff and multi-score designs; honest
  confidence intervals (Armstrong--Kolesar); local randomisation
  (`rdrandinf`, `rdwinselect`, `rdsensitivity`); Cattaneo--Jansson--Ma
  density tests; Rosenbaum bounds; CATE via `rdhte` and ML variants
  (`rd_forest`, `rd_boost`, `rd_lasso`); Angrist--Rokkanen
  extrapolation; power analysis (`rdpower`, `rdsampsi`).
- **Synthetic control (20 estimators + 6 inference strategies, v0.9.0):**
  classical SCM [@abadie2010synthetic], SDID [@arkhangelsky2021synthetic],
  Augmented SCM (Ben-Michael et al. 2021), Bayesian SCM (Dirichlet
  MCMC), BSTS / CausalImpact (Kalman smoother), Penalised SCM
  (Abadie--L'Hour 2021), Forward DID, cluster SCM, sparse (LASSO) SCM,
  kernel SCM, kernel-ridge SCM, and a Research Workflow
  (`synth_compare`, `synth_recommend`, `synth_power`, `synth_mde`,
  `synth_sensitivity`, `synth_report`).
- **Decomposition analysis (18 methods, v0.9.2):** Blinder--Oaxaca with
  five reference-coefficient conventions (Neumark 1988, Cotton 1988,
  Reimers 1983), Gelbach (2016) sequential OVB, Fairlie (1999/2005)
  nonlinear, Bauer--Sinning (2008), Firpo--Fortin--Lemieux (2009, 2018)
  RIF and two-step, DiNardo--Fortin--Lemieux (1996) reweighting,
  Machado--Mata (2005), Melly (2005), Chernozhukov--Fernández-Val--Melly
  (2013), Theil/Atkinson/Dagum--Gini decompositions with closed-form
  influence functions, Shorrocks--Shapley allocation,
  Lerman--Yitzhaki source decomposition, Kitagawa (1955) and Das Gupta
  (1993) demographic standardisation, Lundberg (2021) gap-closing, and
  VanderWeele mediation/disparity decomposition.
- **Stochastic frontier analysis (v0.9.3):** cross-sectional `sp.frontier`
  with half-normal/exponential/truncated-normal, heteroskedastic
  inefficiency (Caudill--Ford--Gropper 1995) and noise (Wang 2002),
  Battese--Coelli (1995) inefficiency determinants, Battese--Coelli
  (1988) and JLMS technical efficiency, Kodde--Palm (1986) mixed-$\bar
  \chi^2$ LR test; panel `sp.xtfrontier` with Pitt--Lee (1981),
  Battese--Coelli (1992) time-decay, Battese--Coelli (1995),
  Greene (2005) true fixed/random effects, and Dhaene--Jochmans (2015)
  split-panel jackknife bias correction; Kumbhakar--Parmeter--Tsionas
  (2013) Zero-Inefficiency SFA (`sp.zisf`); Orea--Kumbhakar (2004) /
  Greene (2005) Latent-Class SFA (`sp.lcsf`); Färe--Grosskopf--Lindgren--
  Roos (1994) Malmquist TFP index (`sp.malmquist`); Cobb--Douglas →
  translog design helper (`sp.translog_design`).
- **Multilevel / mixed-effects (v0.9.3):** `sp.mixed` linear mixed models
  with unstructured random-effect covariance (default), three-level
  nested models, BLUP posterior standard errors, Nakagawa--Schielzeth
  marginal and conditional $R^2$; `sp.melogit` / `sp.mepoisson` /
  `sp.meglm` / `sp.megamma` / `sp.menbreg` / `sp.meologit` GLMMs fitted
  by Laplace approximation or adaptive Gauss--Hermite quadrature
  (AGHQ, `nAGQ>1`) matching Stata `intpoints()` and R `lme4::glmer`
  semantics; `sp.icc` intra-class correlation with delta-method CI;
  `sp.lrtest` with Self--Liang mixed-$\bar\chi^2$ boundary correction.
  A critical Jondrow-posterior sign error in all prior frontier
  implementations is fixed in 0.9.3; efficiency scores computed on
  any prior version should be re-estimated.
- **Modern ML causal inference:** double/debiased ML
  [@chernozhukov2018double] including the new partially linear IV
  variant `sp.dml(model='pliv')` (v0.9.3); causal forests
  [@wager2018estimation]; meta-learners S/T/X/R/DR
  [@kunzel2019metalearners]; TMLE [@vanderlaan2011targeted]; neural
  causal models (TARNet, CFRNet, DragonNet) [@shalit2017estimating;
  @shi2019adapting]; causal discovery (NOTEARS, PC algorithm, LiNGAM,
  GES) [@zheng2018dags]; policy trees [@athey2021policy]; Bayesian
  causal forests [@hahn2020bayesian]; matrix completion; conformal
  inference for causal effects; dose--response curves;
  dynamic-treatment regimes; interference and spillover.
- **Classical and modern econometrics beyond causal inference:**
  mixed-logit random-coefficient multinomial choice (`sp.mixlogit`,
  v0.9.3); instrumental-variable quantile regression
  (`sp.ivqreg`, Chernozhukov--Hansen 2005/06/08, v0.9.3); propensity
  score matching [@rosenbaum1983central]; matching estimators; spatial
  econometrics (weights, ESDA, ML/GMM, GWR, MGWR, spatial panel);
  time-series (ARIMA, VAR, BVAR, GARCH, cointegration, local
  projections, structural break); survival (Cox, AFT, frailty);
  survey calibration and complex-survey regression; bunching;
  Mendelian randomisation.
- **Sensitivity analysis:** Oster [@oster2019unobservable]; sensemakr
  [@cinelli2020making]; E-values [@vanderweele2017sensitivity];
  Rosenbaum bounds; Manski bounds; specification curve analysis
  [@simonsohn2020specification] via `sp.spec_curve()`; and
  `sp.robustness_report()` batteries.

**Unique features** include:

1. A **Smart Workflow Engine** (`sp.recommend()` suggests estimators
   given data and research questions; `sp.compare_estimators()` runs
   multiple methods on the same data; `sp.assumption_audit()` tests
   all assumptions in one call; `sp.verify()` / `sp.verify_benchmark()`
   aggregate bootstrap stability, placebo pass rate, and subsample
   agreement into a posterior `verify_score ∈ [0, 100]` for any
   recommendation, new in v0.9.3).
2. **Specification curve analysis** [@simonsohn2020specification] via
   `sp.spec_curve()` and automated robustness batteries via
   `sp.robustness_report()`.
3. **Publication-ready output** to Word, Excel, LaTeX, and HTML via
   `sp.modelsummary()` and `sp.outreg2()`; every result object
   supports `.to_latex()`, `.to_docx()`, and `.cite()`.
4. An **agent-native API** with `sp.function_schema()` returning JSON
   schemas for all 550+ functions, and `sp.list_functions()` /
   `sp.describe_function()` for discoverability.

The package is implemented in pure Python atop NumPy, SciPy, Pandas,
statsmodels, scikit-learn, and linearmodels, with optional PyTorch and
JAX backends. It supports Python $\geq$ 3.9 and is distributed via
PyPI under the MIT license.

# Validation

We validate `StatsPAI` through replication of published results on
real datasets, cross-validation against established reference
implementations, and Monte Carlo coverage studies.

**Card (1995).** Using the Wooldridge textbook dataset ($N = 3{,}010$),
we replicate the IV returns-to-schooling estimate from Angrist and
Pischke [@angrist2009mostly] Table 4.1.1. `StatsPAI` produces
$\hat\beta_{\text{OLS}} = 0.074$ (published: 0.075) and
$\hat\beta_{\text{IV}} = 0.132$ (published: 0.132), matching within
rounding precision.

**LaLonde (1986).** Using the exact Dehejia--Wahba NSW experimental
subsample ($N = 445$), the raw difference in means is $\$1{,}794$---an
exact match to the published benchmark. All causal estimators (OLS,
PSM, DML, AIPW) produce positive, economically meaningful estimates.

**Lee (2008).** The RD incumbency-advantage estimate is 0.062
(published: $\sim$0.08), consistent with modern bias-corrected
inference [@calonico2014robust]. The Cattaneo--Jansson--Ma density
test ($p = 0.90$) confirms no manipulation around the cutoff.

**Cross-validation against reference implementations.**
DML on Card (1995) yields 0.0741 in `StatsPAI` vs. 0.0749 in `EconML`
(difference: 0.0008). `sp.mixed` linear mixed-model fixed effects and
variance components agree with `statsmodels.MixedLM` to four decimal
places on random-intercept and unstructured random-slope
specifications. Stochastic frontier parameter recovery for the
half-normal, exponential, and truncated-normal distributions has been
verified to within Monte Carlo tolerance against known data-generating
processes; kernel-density integration tests
($\int f(\epsilon)\,d\epsilon = 1$) guard the three frontier
log-likelihoods against regressions.

**Monte Carlo coverage.** Simulations (200 replications) on built-in
data-generating processes show negligible mean bias ($< 0.01$) and
empirical 95\% confidence-interval coverage of 96.5--100\% for DID,
RD, and IV estimators. Frontier panel models (Pitt--Lee, BC92, BC95)
recover true variance components within the expected sampling range,
and the Dhaene--Jochmans split-panel jackknife reduces the
incidental-parameters bias in Greene (2005) true-fixed-effects
frontier from $\hat\sigma_u = 0.374$ to $\hat\sigma_u = 0.359$
(true value 0.350) at $T = 30$, $N = 25$.

**Test suite.** The package ships with approximately 1{,}600 unit and
integration tests spanning all subpackages; these are executed in
continuous integration across macOS, Linux, and Windows on Python
3.10--3.13.

# Research Impact Statement

`StatsPAI` lowers the barrier to rigorous causal inference by
consolidating methods that previously required proficiency across
multiple languages and packages. Graduate students and applied
researchers can now move from OLS through DML to causal forests
without switching toolkits, reducing both onboarding time and the
risk of implementation errors. The agent-native API enables a new
class of AI-assisted empirical workflows: LLM agents can discover,
invoke, and interpret statistical methods through structured
schemas, accelerating literature replication and robustness
analysis. Early adoption in Stanford REAP research projects has
demonstrated its utility for rapid policy-evaluation prototyping.
By unifying classical and modern methods under one roof,
`StatsPAI` facilitates direct comparison of estimators on the same
data, encouraging the methodological transparency increasingly
demanded by journals and funding agencies.

# AI Usage Disclosure

Portions of code documentation were generated with Claude
(Anthropic). All content was reviewed and validated by the author.
Statistical implementations were verified against published
references and benchmark datasets.

# Acknowledgements

The author thanks the Stanford REAP Program for institutional support
and the CoPaper.AI team for feedback. The author is grateful to the
developers of NumPy, SciPy, statsmodels, scikit-learn, linearmodels,
and PyTorch, whose foundational libraries StatsPAI builds upon.

# References
