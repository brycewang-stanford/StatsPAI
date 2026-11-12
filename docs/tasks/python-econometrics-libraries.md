---
description: A fair comparison of Python libraries for econometrics and causal inference - statsmodels, linearmodels, pyfixest, DoubleML, EconML, DoWhy, CausalML, rdrobust, csdid and StatsPAI - checked against each library's own documentation.
---

# Choosing a Python library for econometrics and causal inference

**Short answer.** No single package is best at everything. For a fixest-style
fixed-effects workflow, **pyfixest** is fast and mature. For panel models and
the fullest set of IV diagnostics, **linearmodels**. For general statistical
models and time series, **statsmodels**. For machine-learning causal effects,
**DoubleML** and **EconML**; for a graph-first workflow, **DoWhy**. For one
method, the dedicated ports (**rdrobust**, **csdid**, **differences**,
**pysyncon**). **StatsPAI** aims to cover the whole applied-econometrics
workflow under one API (regression through staggered DiD, RD, synthetic
control and DML), reports for each function whether its numbers have been
checked against Stata or R, and exposes everything to AI agents through
schemas and an MCP server. Its breadth is younger than the focused
libraries' track records, so check `sp.parity_status(...)` for the functions
you rely on.

## How this page was made

Every cell for another library comes from that library's own documentation,
README or PyPI page, checked on **2026-11-12** (versions in the table below).
"—" means we found no documentation of the feature, not that it is
impossible; for single-purpose packages it simply means the feature is outside
their scope. If a cell is wrong or out of date, please
[open an issue](https://github.com/brycewang-stanford/StatsPAI/issues) and we
will correct it.

## Capability matrix

| | OLS robust / cluster SE | Absorbed high-dim. FE | IV / 2SLS + weak-IV diagnostics | Panel FE / RE | Staggered DiD (robust) | RD, bias-corrected | Synthetic control | DML | Tables (LaTeX / Word) | MCP server / agent files |
|---|---|---|---|---|---|---|---|---|---|---|
| [statsmodels](https://www.statsmodels.org/stable/api.html) | yes | — | partial (IV in `sandbox`) | partial (mixed models) | — | — | — | — | partial (LaTeX / HTML) | — |
| [linearmodels](https://bashtage.github.io/linearmodels/) | yes | yes (`AbsorbingLS`) | yes (2SLS, LIML, GMM, CUE, JIVE; first-stage F, Shea R², Kleibergen–Paap) | yes (FE, RE, between, first difference; Hausman) | — | — | — | — | partial | — |
| [pyfixest](https://pyfixest.org/) | yes (incl. multiway, wild bootstrap) | yes | yes (first-stage F, effective F) | FE only | yes (Gardner `did2s`, local projections, saturated event study) | — | — | — | yes (`etable`) | agent skill + `llms.txt` |
| [DoubleML](https://docs.doubleml.org/stable/) | — | — | DML-IV (PLIV, IIVM) | DML panel model | yes, DML-based Callaway–Sant'Anna | yes, `RDFlex` via rdrobust | — | yes | — | — |
| [EconML](https://www.pywhy.org/EconML/) | — | — | ML / DML IV | `DynamicDML` | — | — | — | yes | partial | — |
| [DoWhy](https://www.pywhy.org/dowhy/) | — | — | partial (Wald) | — | — | partial (manual bandwidth) | — | via EconML | — | — |
| [CausalML](https://causalml.readthedocs.io/) | — | — | partial (`IVRegressor`, DRIV) | — | — | — | — | R-learner | — | — |
| [rdrobust](https://github.com/rdpackages/rdrobust) | — | — | — | — | — | yes | — | — | — | — |
| [csdid](https://github.com/d2cml-ai/csdid) / [differences](https://bernardodionisi.github.io/differences/latest/) | — | — | — | — | yes (Callaway–Sant'Anna) | — | — | — | — | — |
| [pysyncon](https://sdfordham.github.io/pysyncon/) | — | — | — | — | — | — | yes (Synth, robust, augmented, penalized) | — | — | — |
| **StatsPAI** | yes | yes | yes (first-stage F, effective F, Anderson–Rubin) | yes (FE, RE, Hausman) | yes (Callaway–Sant'Anna, Sun–Abraham, BJS imputation, Gardner, dCDH) | yes | yes (20 methods) | yes | yes (Word, Excel, LaTeX, Markdown) | MCP server + schemas + `llms.txt` |

Versions checked: statsmodels 0.15.0, linearmodels 7.1, pyfixest 0.70.0,
doubleml 0.11.4, econml 0.17.0, dowhy 0.14, causalml 0.18.0, rdrobust 2.1.1,
csdid 0.4.5, differences 0.3.0, pysyncon 1.7.0, StatsPAI 1.43.1. Licences:
rdrobust, differences and the Python `marginaleffects` are GPL-3.0; the others
here are MIT, BSD, Apache-2.0 or NCSA.

## What the StatsPAI column rests on

The StatsPAI cells are backed by its cross-language test suite. For example,
each of these functions matches its R reference on identical data within a
relative error of 1e-6 for estimates and standard errors:

```python
import statspai as sp

for name in ["feols", "ivreg", "panel", "callaway_santanna", "sun_abraham",
             "bjs", "rdrobust", "synth", "dml"]:
    s = sp.parity_status(name)
    print(f"{name:18s} {s['status']:10s} {s['reference']}")
```

```text
feols bit-exact fixest::feols
ivreg bit-exact AER::ivreg
callaway_santanna bit-exact did::att_gt + aggte
rdrobust bit-exact rdrobust::rdrobust
dml bit-exact DoubleML::DoubleMLPLR
```

Not every StatsPAI function has that evidence. `sp.parity_summary()` gives
the honest totals (as of 1.43.1: 576 functions compared against R or Stata,
452 checked against known truths or published results, 652 with no numerical
evidence yet), and the [evidence scorecard](../scorecard.md) reports interval
coverage and speed, including failures.

## How to choose

- **You only need fixed-effects regressions, fast**: pyfixest, or StatsPAI's
  `sp.feols` if you also need the rest of the workflow.
- **Panel econometrics and IV diagnostics**: linearmodels or StatsPAI.
- **Heterogeneous effects with machine learning**: EconML, DoubleML, or
  StatsPAI's `sp.dml` / `sp.causal_forest` / `sp.metalearner`.
- **A graph-first identification workflow**: DoWhy.
- **One method, the reference implementation**: rdrobust, csdid / differences,
  pysyncon.
- **A whole paper's empirical workflow in one API, Stata/R conventions, or an
  AI agent doing the analysis**: StatsPAI.

The libraries also work together: StatsPAI uses linearmodels internally for
parts of the panel module, can call pyfixest as an optional backend, and
`sp.cross_validate` re-runs a model in pyfixest, linearmodels or DoubleML to
compare the numbers.

---

*Checked on 2026-11-12 against each library's documentation. The StatsPAI
code on this page is run by `tests/test_docs_task_pages.py`.
[Report a correction](https://github.com/brycewang-stanford/StatsPAI/issues).*
