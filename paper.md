---
title: 'StatsPAI: A Unified, Agent-Native Python Toolkit for Causal Inference and Applied Econometrics'
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
date: 26 July 2026
bibliography: paper.bib
---

# Summary

`StatsPAI` is an open-source Python package for causal inference and
applied econometrics. It gives empirical researchers a single interface
for estimating, diagnosing, comparing, and reporting models that are
otherwise spread across many specialized packages or proprietary
statistical environments. A single `import statspai as sp` reaches
estimators for the main families of applied work — regression and panel
models, instrumental variables, the modern difference-in-differences and
regression-discontinuity toolkits, synthetic control and matching, and
machine-learning estimators of heterogeneous treatment effects —
together with the diagnostics, robustness checks, and reporting that
surround them. The full catalogue of more than 1,100 registered
functions across more than 80 submodules is enumerated in the package
documentation rather than here.

Results from the mature estimators share a common reporting surface, so
the same calls produce a summary, a figure, a LaTeX or Word table, or a
citation. `StatsPAI` is also agent-native: every registered function
exposes a machine-readable schema — a structured description of its
arguments and outputs that a program can parse directly — together with
structured failure metadata. This lets LLM-driven research assistants
discover estimators, choose among alternatives, and surface a method's
assumptions without parsing free-form prose, the capability that most
distinguishes `StatsPAI` from a conventional estimator library. The
source code is available at
[https://github.com/brycewang-stanford/StatsPAI](https://github.com/brycewang-stanford/StatsPAI)
and archived on Zenodo [@wang2026statspai].

# Statement of Need

Applied researchers face a fragmented software landscape. Stata offers
an integrated workflow, but it is proprietary and does not expose a
typed, machine-readable interface for AI-assisted analysis. R provides
excellent method-specific packages such as `did`
[@callaway2021difference], `rdrobust` [@calonico2014robust], `Synth`
[@abadie2010synthetic], `grf` [@athey2019generalized], and `lme4`
[@bates2015lme4], but these packages use different APIs, object systems,
and output conventions. Python has strong pieces of the causal inference
ecosystem, including `DoWhy` for graphical causal models
[@sharma2020dowhy], `EconML` for machine-learning treatment effect
estimation [@econml], `CausalML` for uplift modeling [@chen2020causalml],
and `DoubleML` for double/debiased machine learning [@bach2022doubleml].
None of these tools, however, is intended to cover the full
applied-econometrics workflow from design diagnosis through estimation,
robustness, and publication output.

`StatsPAI` addresses this gap for graduate students, applied economists,
policy researchers, and data scientists who want a Python-native
workflow without giving up the breadth of Stata or the methodological
depth of R. Its goal is not to replace every specialized implementation,
but to provide a coherent empirical workspace: shared formula
conventions, compatible result surfaces for mature estimators, export
methods where supported, citations attached to estimators, and
validation metadata that make the relationship between methods,
assumptions, and evidence explicit.

# State of the Field

Existing Python packages are strongest when they focus on a narrower
problem: `DoWhy` emphasizes identification, graphical assumptions, and
refutation; `EconML` and `CausalML` focus on heterogeneous effects and
uplift modeling; `DoubleML` implements orthogonal-score estimators for a
well-defined family of double machine-learning designs. These packages
are complementary to `StatsPAI`, and several of its ideas follow the
same methodological literature, including double/debiased machine
learning [@chernozhukov2018double], causal forests
[@wager2018estimation], and meta-learners [@kunzel2019metalearners]. In
the high-dimensional and double machine-learning tradition specifically,
`StatsPAI` builds on established reference implementations rather than
around them: it provides a faithful Python port of the rigorous
(data-driven) Lasso and post-double-selection estimators of the `hdm`
package [@chernozhukov2016hdm; @belloni2012sparse; @belloni2014inference],
validated for numerical agreement with `hdm`, and orthogonal-score
estimators aligned with the `DoubleML` implementations in Python and R
[@bach2022doubleml; @bach2024doubleml]. Researchers can thus reproduce
the high-dimensional econometrics workflow inside the same
agent-addressable interface used for the rest of the package.

The general-purpose regression toolkits applied economists already rely
on — `statsmodels` and `linearmodels` in Python, or `fixest` in R — sit
at a different layer, supplying core regression machinery that `StatsPAI`
builds on rather than competes with. Its contribution is therefore the
integration layer itself, not a single better estimator: contributing one
more estimator to each existing project would still leave users with
incompatible result classes, separate diagnostic conventions, and no
unified agent-facing schema. `StatsPAI` earns a standalone existence by
being the layer that makes a heterogeneous collection of estimators —
classical and machine-learning, Python-native and R/Stata-aligned —
behave as one coherent, agent-addressable workspace, with broad method
coverage, shared reporting, explicit citations, stable/experimental API
metadata, and cross-language parity checks.

# Software Design

`StatsPAI` is organized around method families and a registry layer.
Researchers can call focused functions, such as an IV or
regression-discontinuity estimator, or use higher-level dispatchers that
select among variants within a design family. The registry records
function names, parameters, examples, stability tiers, limitations,
citations, and schema information, making the package usable both from a
notebook and from external systems such as a Model Context Protocol
server. In practice an assistant can query the registry for estimators
valid for a detected design, invoke one, read back structured diagnostics
and assumption violations, and decide the next step through typed schemas
rather than free-form prompts.

The central design choice is a shared result interface: estimators
return structured objects that store coefficients, uncertainty
estimates, diagnostics, fitted values, plots, and exporter hooks in
predictable locations. This lowers the switching cost between classical
econometric and modern machine-learning estimators, and makes validation
easier because tests can compare common fields across implementations.

The package is implemented mainly in Python on top of NumPy, SciPy,
Pandas, statsmodels, scikit-learn, and linearmodels, and supports Python
3.9 and newer. Optional accelerator backends are used only where they
materially change the computation: PyTorch for neural causal estimators,
JAX for selected bootstrap and linear-algebra workloads, and a Rust/PyO3
kernel for high-dimensional fixed-effect and cluster-variance routines.
It is distributed via PyPI under the MIT license.

These choices carry costs worth stating plainly. A shared result
interface means adding or upgrading an estimator is never purely local,
and the reporting and export hooks are guaranteed only for the mature
estimators — which is why auxiliary helpers advertise narrower
capabilities through the registry rather than claiming a uniformity they
do not have. Favouring breadth also trades against depth: for any single
design, a dedicated package may expose more edge-case options than
`StatsPAI` does today. Performance is likewise not the first priority of
the default install, which runs on a pure-Python NumPy/SciPy stack and
reaches for the PyTorch, JAX, or Rust backends only when present, holding
those accelerated paths to the same documented numerical tolerances as
the fallbacks. The Rust/PyO3 kernel in particular carries a
compiled-language build cost — platform-specific wheels or a Rust
toolchain, plus a separately maintained crate — contained by keeping it
optional, with transparent pure-Python fallback. We accept these costs
deliberately: for researchers who must compare estimators, switch
designs, and produce reproducible output within one project, a single
coherent, agent-addressable interface outweighs the loss of per-method
specialization.

# Research Impact Statement

`StatsPAI` ships a concrete validation and community-readiness dossier
built from two complementary tracks. The first is a cross-language parity
harness: `StatsPAI`, R, and Stata are run on the same input bytes and
their numerical output compared directly. The harness checks more than
sixty estimator modules against a reference R implementation, the large
majority also against Stata. Closed-form estimators
agree to machine precision; iterative and machine-learning estimators
agree within pre-registered, documented tolerances, and the few remaining
convention gaps are disclosed rather than hidden. Within this harness,
the high-dimensional methods reproduce the published worked examples of
the `hdm` package [@chernozhukov2016hdm] — its growth-convergence,
institutions-and-development, and gender-wage-gap applications — and the
double-machine-learning estimators are checked against `DoubleML`
[@bach2022doubleml; @bach2024doubleml] on shared data, an independent
check against those established reference implementations. The second
track calibrates the simulated teaching datasets in `sp.datasets` so that
the canonical estimator recovers values near well-known published
results — returns-to-schooling IV (Card), job-training
(LaLonde/Dehejia-Wahba), RD elections (Lee), multi-period
difference-in-differences (Callaway-Sant'Anna), and synthetic control;
because these datasets are simulated rather than the original study data,
exact numerical replication is deliberately not claimed. The suite also
includes a 1000-replication coverage run for representative OLS,
difference-in-differences, and strong-instrument IV designs, with
empirical coverage close to the nominal 95 percent level. A
reviewer-facing validation dossier and short reviewer guide ship with the
repository documentation.

The near-term impact is a more reproducible workflow for applied policy
evaluation: sharing one interface, researchers can compare estimators on
the same data, export tables with the same metadata, and record the
citations and assumptions attached to each analysis. `StatsPAI` is
currently being used in an ongoing working
paper connected to the Rural Education Action Program at Stanford
University, *Family contagion of screen time? Within-person evidence from
six waves in China* (Wang, Zhang, and Hou, in preparation), which relies
on the package for its panel and within-person estimation; no
peer-reviewed research article using the package has yet been published.
The impact claim therefore rests on three things a reviewer can check
directly: active use in an ongoing working paper, public distribution on
PyPI, and the reproducible validation materials and worked examples
bundled with the repository.

# AI Usage Disclosure

Generative AI tools, including Claude Code and OpenAI ChatGPT/Codex,
were used for code-generation assistance, refactoring suggestions, test
scaffolding, documentation drafting, and manuscript copy-editing. Exact
model identifiers were not retained for all exploratory sessions. Human
authors made the core design decisions; reviewed, edited, and checked
AI-assisted code and prose; and checked citations and software claims
against repository evidence. The authors will not use generative AI to
produce substantive responses to journal editors or reviewers. All authors
take responsibility for the correctness, originality, licensing, and
compliance of the package and this paper.

# Author Contributions

**Biaoyue Wang** conceived and designed the package, implemented the
estimators, registry, schema layer, and result objects, wrote the
documentation, tests, and validation suites, and led the drafting of
this paper. **Scott Rozelle** provided guidance on the package's design
direction and target research workflows, and contributed to the
writing, review, and revision of this paper. Both authors reviewed and
approved the final manuscript and take responsibility for the
correctness of the package and this paper.

# Acknowledgements

The authors thank the Stanford Rural Education Action Program (REAP)
research community and the CoPaper.AI team for feedback on early
workflows. StatsPAI Inc. is the legal entity associated with the
project, and CoPaper.AI is a commercial downstream product that may
call the MIT-licensed `StatsPAI` package; the `StatsPAI` package itself is
permanently open source under the MIT license. The authors are also
grateful to the developers of NumPy, SciPy, Pandas, statsmodels,
scikit-learn, linearmodels, PyTorch, JAX, and the broader open-source
scientific Python ecosystem that `StatsPAI` builds upon.

# References
