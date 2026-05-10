---
title: 'StatsPAI: An Agent-Native Python Toolkit for Causal Inference and Applied Econometrics'
tags:
  - Python
  - causal inference
  - econometrics
  - policy evaluation
  - machine learning
  - reproducible research
authors:
  - name: Biaoyue Wang
    orcid: 0000-0002-1828-2208
    email: brycew6m@stanford.edu
    corresponding: true
    affiliation: "1, 2"
  - name: Scott Rozelle
    email: rozelle@stanford.edu
    affiliation: "1, 2"
affiliations:
  - name: Rural Education Action Program, Stanford Center on China's Economy and Institutions, Stanford University, United States
    index: 1
    ror: 00f54p054
  - name: StatsPAI Inc., United States
    index: 2
date: 9 May 2026
bibliography: paper.bib
---

# Summary

`StatsPAI` is an open-source Python package for causal inference and
applied econometrics. It gives empirical researchers a single interface
for estimating, diagnosing, comparing, and reporting models that are
usually spread across many specialized packages or proprietary
statistical environments. The package currently exposes more than 950
registered functions across more than 80 submodules, covering classical
regression, instrumental variables, panel data, difference-in-differences,
regression discontinuity, synthetic control, matching,
stochastic frontier analysis, mixed-effects models, decomposition
methods, sensitivity analysis, and modern machine-learning estimators
for heterogeneous treatment effects.

The package is designed for policy evaluation, social science research,
and other empirical workflows where researchers must move between
research design, estimation, diagnostics, robustness checks, and
publication tables. A common result contract gives users `.summary()`,
`.plot()`, `.to_latex()`, `.to_docx()`, and `.cite()` methods where
appropriate. `StatsPAI` is also agent-native: registered functions
expose machine-readable schemas and structured failure metadata so that
LLM-driven research assistants can discover estimators, choose among
alternatives, and surface assumptions without parsing free-form prose.
The source code is available at
[https://github.com/brycewang-stanford/statspai](https://github.com/brycewang-stanford/statspai).

# Statement of Need

Applied researchers face a fragmented software landscape. Stata offers
an integrated workflow, but it is proprietary and does not expose a
typed, machine-readable interface for AI-assisted analysis. R provides
excellent method-specific packages such as `did`, `rdrobust`, `Synth`,
`grf`, and `lme4`, but these packages use different APIs, object
systems, output conventions, and diagnostic workflows
[@callaway2021difference; @calonico2014robust; @abadie2010synthetic;
@wager2018estimation; @bates2015lme4]. Python has strong pieces of the
causal inference ecosystem, including `DoWhy` for graphical causal
models [@sharma2020dowhy], `EconML` for machine-learning treatment
effect estimation [@econml], `CausalML` for uplift modeling
[@chen2020causalml], and `DoubleML` for double/debiased machine
learning [@bach2022doubleml]. None of these tools, however, is intended
to cover the full applied-econometrics workflow from design diagnosis
through estimation, robustness, and publication output.

`StatsPAI` addresses this gap for graduate students, applied
economists, policy researchers, and data scientists who want a
Python-native workflow without giving up the breadth of Stata or the
methodological depth of R. Its goal is not to replace every specialized
implementation. Instead, it provides a coherent empirical workspace:
shared formula conventions, common result objects, consistent export
methods, citations attached to estimators, and validation metadata that
make the relationship between methods, assumptions, and evidence
explicit.

# State of the Field

Existing Python packages are strongest when they focus on a narrower
problem. `DoWhy` emphasizes identification, graphical assumptions, and
refutation; `EconML` and `CausalML` focus on heterogeneous effects and
uplift modeling; `DoubleML` implements orthogonal-score estimators for a
well-defined family of double machine-learning designs. These packages
are complementary to `StatsPAI`, and several ideas in `StatsPAI` follow
the same methodological literature, including double/debiased machine
learning [@chernozhukov2018double], causal forests
[@wager2018estimation], and meta-learners [@kunzel2019metalearners].

The build-versus-contribute case for `StatsPAI` is therefore about
scope and interface rather than a single estimator. Contributing one
more estimator to each existing project would still leave users with
many incompatible result classes, separate diagnostic conventions, and
no unified agent-facing schema. `StatsPAI` contributes an integration
layer with substantive statistical content: broad method coverage,
shared reporting, explicit estimator citations, stable/experimental API
metadata, cross-language parity checks, and an LLM-oriented registry
that can expose statistical tools safely to automated workflows.

# Software Design

`StatsPAI` is organized around method families and a registry layer.
Researchers can call focused functions, such as an IV or
regression-discontinuity estimator, or use higher-level dispatchers that select
among variants within a design family. The registry records function
names, parameters, examples, stability tiers, limitations, citations,
and schema information. This makes the package usable both from a
notebook and from external systems such as a Model Context Protocol
server.

The central design choice is a shared result interface. Estimators
return structured objects that store coefficients, uncertainty
estimates, diagnostics, fitted values, plots, and exporter hooks in
predictable locations. This reduces the switching cost between classical
econometric estimators and modern machine-learning estimators, and it
also makes validation easier because tests can compare common fields
across implementations.

The package is implemented mainly in Python on top of NumPy, SciPy,
Pandas, statsmodels, scikit-learn, and linearmodels. This keeps the
installation path familiar for Python users and supports Python 3.9 and
newer. Optional accelerator backends are used only where they materially
change the computation: PyTorch for neural causal estimators, JAX for
selected bootstrap and linear algebra workloads, and a Rust/PyO3 kernel
for high-dimensional fixed-effect and cluster-variance routines. This
keeps the default package inspectable while allowing heavy workloads to
use specialized backends when available. The package is distributed via
PyPI under the MIT license.

# Research Impact Statement

`StatsPAI` has a concrete validation and community-readiness dossier.
The repository includes Python tests, 36 R-parity modules,
21 Stata-parity modules, original-paper replay fixtures, and Monte Carlo
coverage checks. Published-reference anchors include Card-style
returns-to-schooling IV estimates, LaLonde/Dehejia-Wahba job-training benchmarks,
Lee-style regression discontinuity estimates, Callaway-Sant'Anna
difference-in-differences examples, and synthetic-control benchmarks.
The validation suite also includes a 1000-replication coverage run for
representative OLS, difference-in-differences, and strong-instrument IV
designs, with empirical coverage close to the nominal 95 percent level.

The near-term research impact is a more reproducible empirical workflow
for applied policy evaluation. Because methods share one interface,
researchers can compare estimators on the same data, export tables with
the same metadata, and record the citations and assumptions attached to
each analysis. Early use in Stanford REAP research workflows has shown
the value of the package for rapid policy-evaluation prototyping, while
the agent-native registry supports a second use case: AI-assisted
replication and robustness analysis in which statistical tools are
discovered and invoked through explicit schemas rather than informal
prompts.

# AI Usage Disclosure

Generative AI tools, including Claude and OpenAI/Codex, were used to
draft portions of the documentation, assist with code generation, and
revise this manuscript. The corresponding author reviewed AI-generated
text, checked citations and software claims against repository evidence,
and retained responsibility for the correctness of the package and this
paper.

# Acknowledgements

The authors thank the Stanford REAP research community and the
CoPaper.AI team for feedback on early workflows. The authors are also
grateful to the developers of NumPy, SciPy, Pandas, statsmodels,
scikit-learn, linearmodels, PyTorch, JAX, and the broader open-source
scientific Python ecosystem that `StatsPAI` builds upon.

# References
