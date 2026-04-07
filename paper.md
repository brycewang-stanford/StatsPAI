---
title: 'StatsPAI: A Unified, Agent-Native Python Toolkit for Causal Inference and Applied Econometrics'
tags:
  - Python
  - causal inference
  - econometrics
  - difference-in-differences
  - regression discontinuity
  - synthetic control
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
date: 6 April 2026
bibliography: paper.bib
---

# Summary

`StatsPAI` is an open-source Python package that provides a unified API for causal inference and applied econometrics. With a single import (`import statspai as sp`), researchers access over 390 functions spanning classical econometric models, modern ML-based causal methods, and publication-ready output generation. The package consolidates functionality that previously required dozens of separate R packages or proprietary software such as Stata into one coherent library. Uniquely, `StatsPAI` is *agent-native*: every function exposes structured result objects and machine-readable schemas (`list_functions()`, `describe_function()`, `function_schema()`), making it the first econometrics toolkit purpose-built for LLM-driven research workflows while remaining fully ergonomic for human researchers.

# Statement of Need

Empirical researchers face a fragmented software landscape for causal inference. Stata lacks modern ML causal methods; R scatters them across 20+ packages with incompatible APIs; Python's existing libraries occupy non-overlapping niches. `DoWhy` [@sharma2020dowhy] emphasizes causal graphs and assumption refutation. `EconML` [@econml] focuses on heterogeneous treatment effects. `CausalML` [@chen2020causalml] specializes in uplift modeling. None provides the full empirical workflow---from OLS and panel models through DML and causal forests to publication tables in Word, Excel, and LaTeX---in a single interface.

`StatsPAI` addresses this gap for applied researchers who need to move fluidly between classical and modern methods, and for AI coding agents that discover and invoke statistical functions through self-describing schemas.

# Software Design

`StatsPAI` is organized into modular subpackages. All functions return structured result objects inheriting from `CausalResult` or `EconometricResults`, providing a consistent interface: `.summary()`, `.plot()`, `.to_latex()`, `.to_docx()`, and `.cite()`.

**Methodological coverage** includes: OLS/IV/panel/GLM; 10 DID variants including Callaway and Sant'Anna [@callaway2021difference], Sun and Abraham [@sun2021estimating], and Goodman-Bacon decomposition [@goodman2021difference]; sharp/fuzzy/kink RD following Calonico et al. [@calonico2014robust]; synthetic control [@abadie2010synthetic] and SDID [@arkhangelsky2021synthetic]; propensity score matching [@rosenbaum1983central]; double/debiased ML [@chernozhukov2018double]; causal forests [@wager2018estimation]; meta-learners (S/T/X/R/DR) [@kunzel2019metalearners]; TMLE [@vanderlaan2011targeted]; neural causal models (TARNet, DragonNet) [@shalit2017estimating; @shi2019adapting]; causal discovery (NOTEARS, PC algorithm) [@zheng2018dags]; policy trees [@athey2021policy]; Bayesian causal forests [@hahn2020bayesian]; and sensitivity analysis via Oster bounds [@oster2019unobservable], sensemakr [@cinelli2020making], and E-values [@vanderweele2017sensitivity].

**Unique features** include: (1) a Smart Workflow Engine where `recommend()` suggests estimators given data and research questions, `compare_estimators()` runs multiple methods on the same data, and `assumption_audit()` tests all assumptions in one call; (2) specification curve analysis [@simonsohn2020specification] via `spec_curve()` and automated robustness batteries via `robustness_report()`; (3) publication-ready output to Word, Excel, LaTeX, and HTML via `modelsummary()` and `outreg2()`; and (4) an agent-native API with `function_schema()` returning JSON schemas for all 390+ functions.

The package is implemented in pure Python atop NumPy, SciPy, Pandas, statsmodels, scikit-learn, and linearmodels, with optional PyTorch and JAX backends. It supports Python $\geq$ 3.9 and is distributed via PyPI under the MIT license.

# Validation

We validate `StatsPAI` through replication of published results using real datasets and cross-validation against `EconML`.

**Card (1995).** Using the Wooldridge textbook dataset ($N = 3{,}010$), we replicate the IV returns-to-schooling estimate from Angrist and Pischke [@angrist2009mostly] Table 4.1.1. `StatsPAI` produces OLS $= 0.074$ (published: 0.075) and IV $= 0.132$ (published: 0.132), matching within rounding precision.

**LaLonde (1986).** Using the exact Dehejia--Wahba NSW experimental subsample ($N = 445$), the raw difference in means is \$1,794---an exact match to the published benchmark. All causal estimators (OLS, PSM, DML, AIPW) produce positive, economically meaningful estimates.

**Lee (2008).** The RD incumbency advantage estimate is 0.062 (published: $\sim$0.08), consistent with the use of modern bias-corrected inference [@calonico2014robust]. The McCrary density test ($p = 0.90$) confirms no manipulation.

**Cross-validation.** Running `StatsPAI` and `EconML`'s DML on the same Card (1995) data yields estimates of 0.0741 and 0.0749 respectively---a difference of 0.0008, confirming cross-package consistency.

Monte Carlo simulations (200 replications) on built-in DGPs show negligible mean bias ($< 0.01$) and empirical 95\% CI coverage of 96.5--100\% for DID, RD, and IV estimators.

# AI Usage Disclosure

Portions of code documentation were generated with Claude (Anthropic). All content was reviewed and validated by the author. Statistical implementations were verified against published references and benchmark datasets.

# Acknowledgements

The author thanks the Stanford REAP Program for institutional support and the CoPaper.AI team for feedback. The author is grateful to the developers of statsmodels, scikit-learn, linearmodels, and PyTorch.

# References
