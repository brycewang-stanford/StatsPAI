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
date: 1 June 2026
bibliography: paper.bib
---

# Summary

`StatsPAI` is an open-source Python package for causal inference and
applied econometrics. It gives empirical researchers a single interface
for estimating, diagnosing, comparing, and reporting models that are
usually spread across many specialized packages or proprietary
statistical environments. A single `import statspai as sp` reaches estimators for the main
families of applied work — regression and panel models, instrumental
variables, the modern difference-in-differences and
regression-discontinuity toolkits, synthetic control and matching, and
machine-learning estimators of heterogeneous treatment effects —
together with the diagnostics, robustness checks, and reporting that
surround them. The full catalogue of more than 1,000 registered
functions across 81 submodules is enumerated in the package
documentation rather than here.

The package is designed for policy evaluation, social science and
public health research, and other empirical workflows where researchers
must move between research design, estimation, diagnostics, robustness
checks, and publication tables. Results from the mature estimators
share a common reporting surface, so the same calls produce a summary, a
figure, a LaTeX or Word table, or a citation across the estimators that
support them.

`StatsPAI` is also agent-native: every registered function exposes a
machine-readable schema — a structured description of its arguments and
outputs that a program can parse directly — together with structured
failure metadata. This lets LLM-driven research assistants discover
estimators, choose among alternatives, and surface the assumptions
behind a method without parsing free-form prose, the capability that
most distinguishes `StatsPAI` from a conventional estimator library.
The source code is available at
[https://github.com/brycewang-stanford/StatsPAI](https://github.com/brycewang-stanford/StatsPAI)
and archived on Zenodo [@wang2026statspai].

# Statement of Need

Applied researchers face a fragmented software landscape. Stata offers
an integrated workflow, but it is proprietary and does not expose a
typed, machine-readable interface for AI-assisted analysis. R provides
excellent method-specific packages such as `did`
[@callaway2021difference], `rdrobust` [@calonico2014robust], `Synth`
[@abadie2010synthetic], `grf` [@athey2019generalized], and `lme4`
[@bates2015lme4], but these packages use different APIs, object
systems, output conventions, and diagnostic workflows. Python has strong pieces of the
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
methodological depth of R. The goal of `StatsPAI` is not to replace every specialized
implementation. Instead, it provides a coherent empirical workspace:
shared formula conventions, compatible result surfaces for mature
estimators, export methods where supported, citations attached to
estimators, and validation metadata that make the relationship between
methods, assumptions, and evidence explicit.

# State of the Field

Existing Python packages are strongest when they focus on a narrower
problem. `DoWhy` emphasizes identification, graphical assumptions, and
refutation; `EconML` and `CausalML` focus on heterogeneous effects and
uplift modeling; `DoubleML` implements orthogonal-score estimators for a
well-defined family of double machine-learning designs. These packages
are complementary to `StatsPAI`, and several ideas in `StatsPAI` follow
the same methodological literature, including double/debiased machine
learning [@chernozhukov2018double], causal forests
[@wager2018estimation], and meta-learners [@kunzel2019metalearners]. In
the high-dimensional and double machine-learning tradition specifically,
`StatsPAI` builds on established reference implementations rather than
around them: it provides a faithful Python port of the rigorous
(data-driven) Lasso and post-double-selection estimators of the `hdm`
package [@chernozhukov2016hdm; @belloni2012sparse; @belloni2014inference],
validated for numerical agreement with `hdm`, and orthogonal-score
estimators aligned with the `DoubleML` implementations in Python and R
[@bach2022doubleml; @bach2024doubleml]. This lets researchers reproduce
the high-dimensional econometrics workflow inside the same
agent-addressable interface used for the rest of the package.
These comparisons are with packages that, like `StatsPAI`, operate at
the causal-inference and machine-learning layer. The general-purpose
regression toolkits applied economists already rely on — `statsmodels`
and `linearmodels` in Python, or `fixest` in R — sit at a different
layer: they supply core regression machinery, not the integrated
workflow spanning design, estimation, diagnostics, and reporting that
`StatsPAI` targets. `StatsPAI` depends on that stack — `statsmodels` and
`linearmodels` are part of its own foundation — while implementing the
causal-inference estimators it specializes in and, where an established
reference implementation exists, validating them for numerical
agreement against the corresponding R and Stata packages. A direct
head-to-head is therefore not the relevant test; the question is whether
the integration layer `StatsPAI` adds is worth maintaining as a separate
package.

The build-versus-contribute case for `StatsPAI` is therefore about
scope and interface rather than a single estimator. Contributing one
more estimator to each existing project would still leave users with
many incompatible result classes, separate diagnostic conventions, and
no unified agent-facing schema. `StatsPAI` contributes an integration
layer with substantive statistical content: broad method coverage,
shared reporting, explicit estimator citations, stable/experimental API
metadata, cross-language parity checks, and an LLM-oriented registry
that can expose statistical tools safely to automated workflows.
`StatsPAI` therefore earns a standalone existence not by contributing a
better single estimator, but by being the layer that makes a
heterogeneous collection of estimators — classical and
machine-learning, Python-native and R/Stata-aligned — behave as one
coherent, agent-addressable workspace. Folding that layer into any one
upstream project would not serve the researchers who must move across
all of them.

# Software Design

`StatsPAI` is organized around method families and a registry layer.
Researchers can call focused functions, such as an IV or
regression-discontinuity estimator, or use higher-level dispatchers that select
among variants within a design family. The registry records function
names, parameters, examples, stability tiers, limitations, citations,
and schema information. This makes the package usable both from a
notebook and from external systems such as a Model Context Protocol
server. In practice this lets an assistant query the registry for the
estimators valid for a detected design, invoke one, read back structured
diagnostics and assumption violations, and decide the next step — all
through typed schemas rather than free-form prompts.

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
newer versions of Python. Optional accelerator backends are used only
where they materially change the computation: PyTorch for neural causal
estimators, JAX for
selected bootstrap and linear algebra workloads, and a Rust/PyO3 kernel
for high-dimensional fixed-effect and cluster-variance routines. This
keeps the default package inspectable while allowing heavy workloads to
use specialized backends when available. The package is distributed via
PyPI under the MIT license.

These choices carry costs worth stating plainly. A shared result
interface means that adding or upgrading an estimator is never purely
local: each estimator must be mapped onto the common fields, and the
reporting and export hooks are guaranteed only for the mature
estimators, which is why auxiliary helpers advertise narrower
capabilities through the registry rather than claiming a uniformity they
do not have. Favouring breadth across method families also trades
against depth: for any single design, a dedicated package may expose
more edge-case options or finer tuning than `StatsPAI` does today.
Finally, performance is deliberately not the first priority of the
default install — the package runs on a pure-Python NumPy/SciPy stack
and reaches for PyTorch, JAX, or a Rust kernel only when they are
present, falling back transparently otherwise — which keeps the default
small and inspectable at the cost of leaving peak performance to an
optional, environment-dependent layer whose accelerated paths are held
to the same documented numerical tolerances as the fallbacks. The
Rust/PyO3 kernel in particular adds a compiled-language component to the
build: shipping it requires platform-specific wheels (or a Rust
toolchain to compile from source) and a separately maintained crate, a
maintenance cost we contain by keeping the kernel optional and
transparently falling back to the pure-Python implementation wherever
the compiled extension is absent. We accept
these costs deliberately: for researchers who must compare estimators,
switch designs, and produce reproducible output within one project, a
single coherent, agent-addressable interface outweighs the loss of
per-method specialization.

# Research Impact Statement

`StatsPAI` ships a concrete validation and community-readiness dossier
built from two complementary tracks. The first is a cross-language
parity harness: `StatsPAI`, R, and Stata are run on the same input data
and their numerical output is compared directly. The harness checks
more than sixty estimator modules against a reference R implementation
on identical input bytes, the large majority of which are also checked
against Stata. On closed-form estimators
the three languages agree to machine precision; iterative and
machine-learning estimators agree within pre-registered, documented
tolerances, and the few remaining convention gaps are disclosed rather
than hidden. The second track calibrates the simulated teaching
datasets bundled in `sp.datasets` so that the canonical estimator
recovers values in the neighbourhood of well-known published results:
returns-to-schooling IV (Card), job-training effects
(LaLonde/Dehejia-Wahba), regression-discontinuity elections (Lee),
multi-period difference-in-differences (Callaway-Sant'Anna), and
synthetic control. Because these datasets are simulated rather than the
original study data, exact numerical replication is deliberately not
claimed. The validation suite also includes a 1000-replication coverage
run for representative OLS, difference-in-differences, and
strong-instrument IV designs, with empirical coverage close to the
nominal 95 percent level. A reviewer-facing validation dossier and a
short reviewer guide are included in the repository documentation.

The near-term research impact is a more reproducible empirical workflow
for applied policy evaluation. Because methods share one interface,
researchers can compare estimators on the same data, export tables with
the same metadata, and record the citations and assumptions attached to
each analysis. `StatsPAI` is currently being used in an ongoing working
paper connected to the Rural Education Action Program at Stanford
University, *Family contagion of screen time? Within-person evidence
from six waves in China* (Wang, Zhang, and Hou, in preparation), which
relies on the package for its panel and within-person estimation; no
peer-reviewed research article using the package has yet been
published. At this stage, then, the impact claim rests on three things
a reviewer can check directly: active use in an ongoing working paper,
public distribution on PyPI, and the reproducible validation materials
and worked examples bundled with the repository. The agent-native registry
also supports AI-assisted replication and robustness analysis in which
statistical tools are discovered and invoked through explicit schemas
rather than informal prompts.

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
