# StatsPAI: The Causal Inference & Econometrics Toolkit for Python

[![PyPI version](https://badge.fury.io/py/StatsPAI.svg)](https://pypi.org/project/StatsPAI/)
[![Python versions](https://img.shields.io/pypi/pyversions/StatsPAI.svg)](https://pypi.org/project/StatsPAI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/brycewang-stanford/statspai/blob/main/LICENSE)
[![Tests](https://github.com/brycewang-stanford/statspai/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/brycewang-stanford/statspai/actions)
[![Downloads](https://static.pepy.tech/badge/statspai/month)](https://pepy.tech/project/statspai)

StatsPAI is a Python package for causal inference and applied econometrics. It provides a unified, Stata-style API covering the complete empirical research workflow — from estimation to publication-ready tables in Word, Excel, and LaTeX.

It brings R's [Causal Inference Task View](https://cran.r-project.org/web/views/CausalInference.html) (fixest, did, rdrobust, gsynth, DoubleML, MatchIt, CausalImpact) into a single, consistent Python package.

> Built by the team behind [CoPaper.AI](https://copaper.ai) · Stanford REAP Program

---

## Main Features

**Regression Models:**

- Ordinary Least Squares with robust / clustered / HAC standard errors
- Instrumental Variables / Two-Stage Least Squares (2SLS), with first-stage F, Sargan, and Hausman tests
- Panel data: Fixed Effects, Random Effects, Between, First Differences (via linearmodels)
- High-dimensional Fixed Effects (via pyfixest)

**Causal Inference — Difference-in-Differences:**

- Classic 2x2 DID estimator
- Staggered DID with heterogeneous treatment effects (Callaway & Sant'Anna 2021)
- Event study plots and pre-trend tests

**Causal Inference — Regression Discontinuity:**

- Sharp and Fuzzy RD with local polynomial estimation
- MSE-optimal bandwidth selection (CCT 2014)
- Robust bias-corrected confidence intervals
- RD plots with binned scatter and polynomial fit

**Causal Inference — Matching:**

- Propensity Score Matching (logit-based PSM)
- Mahalanobis distance matching
- Coarsened Exact Matching (CEM)
- Balance diagnostics with standardized mean differences

**Causal Inference — Synthetic Control:**

- Abadie-Diamond-Hainmueller SCM
- Penalized / ridge SCM for many donors
- Placebo (permutation) inference with MSPE ratios
- Donor weight tables and gap plots

**Causal Inference — Machine Learning Methods:**

- Double/Debiased Machine Learning: Partially Linear (PLR) and Interactive (IRM) models with cross-fitting (Chernozhukov et al. 2018)
- Causal Forest for heterogeneous treatment effects (HTE)
- Compatible with any scikit-learn estimator as first-stage ML model

**Causal Inference — Other Methods:**

- Causal Impact: Bayesian structural time-series intervention analysis (Brodersen et al. 2015)
- Causal Mediation Analysis: ACME / ADE decomposition with bootstrap inference (Imai et al. 2010)
- Shift-Share / Bartik IV with Rotemberg weight diagnostics (GPSS 2020)

**Post-Estimation:**

- Marginal effects (AME / MEM) with delta-method standard errors, equivalent to Stata's `margins, dydx(*)`
- Wald test for linear restrictions, equivalent to Stata's `test`
- Linear combinations of coefficients with inference, equivalent to Stata's `lincom`

**Diagnostics:**

- Oster (2019) coefficient stability / selection-on-unobservables bounds
- McCrary (2008) density manipulation test for RD validity

**Publication-Quality Output:**

- Multi-model comparison tables (equivalent to R's `modelsummary` / Stata's `esttab`)
- Coefficient forest plots across models
- Summary statistics tables (equivalent to Stata's `tabstat`)
- Balance tables for matching / DID / RCT papers
- Cross-tabulation with chi-squared / Fisher's exact test (equivalent to Stata's `tab, chi2`)
- **Export to Word (.docx), Excel (.xlsx), LaTeX (.tex), HTML** — all tables, all formats
- Every result object has `.summary()`, `.plot()`, `.to_latex()`, `.to_docx()`, `.cite()`

---

## Installation

```bash
pip install statspai
```

With optional dependencies:

```bash
pip install statspai[plotting]    # matplotlib, seaborn
pip install statspai[fixest]      # pyfixest for high-dimensional FE
```

**Requirements:** Python >= 3.9

**Core dependencies:** NumPy, SciPy, Pandas, statsmodels, scikit-learn, linearmodels, patsy, openpyxl, python-docx

---

## Quick Example

```python
import statspai as sp

# --- Estimation ---
r1 = sp.regress("wage ~ education + experience", data=df, robust='hc1')
r2 = sp.ivreg("wage ~ (education ~ parent_edu) + experience", data=df)
r3 = sp.did(df, y='wage', treat='policy', time='year', id='worker')
r4 = sp.rdrobust(df, y='score', x='running_var', c=0)
r5 = sp.match(df, y='outcome', treat='treated', covariates=['age', 'edu'])
r6 = sp.dml(df, y='wage', treat='training', covariates=['age', 'edu', 'exp'])

# --- Post-estimation ---
me = sp.margins(r1, data=df)            # Marginal effects
sp.test(r1, "education = experience")   # Wald test: beta_edu = beta_exp?
sp.lincom(r1, "education + experience") # Linear combination

# --- Tables (to Word / Excel / LaTeX) ---
sp.modelsummary(r1, r2, output='table2.docx')
sp.outreg2(r1, r2, r3, filename='results.xlsx')
sp.sumstats(df, vars=['wage', 'education', 'age'], output='table1.docx')
sp.balance_table(df, treat='treated', covariates=['age', 'edu'], output='balance.docx')
sp.tab(df, 'treatment', 'outcome', output='crosstab.docx')
```

---

## API Summary

| Category | Functions | Description |
| --- | --- | --- |
| **Regression** | `regress`, `ivreg`, `panel`, `fixest.feols` | OLS, IV/2SLS, Panel (FE/RE/FD/BE), High-dimensional FE |
| **DID** | `did`, `did_2x2`, `callaway_santanna` | Classic 2x2, Staggered (C&S 2021), Event study |
| **RD** | `rdrobust`, `rdplot` | Sharp/Fuzzy RD, CCT robust inference, RD plots |
| **Matching** | `match` | PSM, CEM, Mahalanobis, Balance diagnostics |
| **Synth** | `synth` | Abadie SCM, Penalized SCM, Placebo inference |
| **ML Causal** | `dml`, `causal_forest` | Double ML (PLR/IRM), Causal Forest (HTE) |
| **Other Causal** | `causal_impact`, `mediate`, `bartik` | Intervention analysis, Mediation, Shift-share IV |
| **Post-estimation** | `margins`, `marginsplot`, `test`, `lincom` | Marginal effects, Wald tests, Linear combinations |
| **Diagnostics** | `oster_bounds`, `mccrary_test` | Coefficient stability, Density manipulation |
| **Tables** | `modelsummary`, `outreg2`, `sumstats`, `balance_table`, `tab` | Multi-model tables, Summary stats, Balance, Cross-tabs |
| **Plots** | `coefplot`, `marginsplot`, `rdplot`, `result.plot()` | Coefficient, Margins, RD, Event study plots |
| **Export** | `.to_docx()`, `.to_latex()`, `output='*.xlsx'` | Word, Excel, LaTeX, HTML — all tables, all formats |

All causal methods return a unified **`CausalResult`** object:

```python
result.estimate       # Point estimate
result.se             # Standard error
result.pvalue         # P-value
result.ci             # Confidence interval
result.summary()      # Formatted text summary
result.plot()         # Appropriate visualization
result.to_latex()     # LaTeX table
result.to_docx()      # Word document
result.cite()         # BibTeX citation for the method
```

---

## Comparison with Stata and R

| Task | Stata | R | StatsPAI |
| --- | --- | --- | --- |
| OLS with robust SE | `reg y x, r` | `feols(y ~ x, vcov="HC1")` | `sp.regress("y ~ x", robust='hc1')` |
| IV regression | `ivregress 2sls y (x = z)` | `feols(y ~ 1 \| x ~ z)` | `sp.ivreg("y ~ (x ~ z)")` |
| Staggered DID | `csdid y, ivar(id) time(t) gvar(g)` | `att_gt(y ~ 1, ...)` | `sp.did(df, y, treat, time, id)` |
| RD design | `rdrobust y x, c(0)` | `rdrobust(Y, X, c=0)` | `sp.rdrobust(df, y, x, c=0)` |
| PSM matching | `psmatch2 treat x1 x2` | `matchit(treat ~ x1+x2)` | `sp.match(df, y, treat, covs)` |
| Double ML | — | `DoubleML$new(...)` | `sp.dml(df, y, treat, covs)` |
| Marginal effects | `margins, dydx(*)` | `margins(model)` | `sp.margins(result, data=df)` |
| Wald test | `test x1 = x2` | `linearHypothesis(...)` | `sp.test(result, "x1 = x2")` |
| Export to Word | `outreg2 using r.doc, word` | `modelsummary(output="t.docx")` | `sp.outreg2(r, filename="r.docx")` |
| Summary stats | `tabstat y x, s(mean sd)` | `datasummary(...)` | `sp.sumstats(df, vars=[...])` |

---

## About

**StatsPAI Inc.** is the research infrastructure company behind [CoPaper.AI](https://copaper.ai) — the AI co-authoring platform for empirical research, born out of Stanford's [REAP](https://reap.fsi.stanford.edu/) program.

**CoPaper.AI** — Upload your data, set your research question, and produce a fully reproducible academic paper with code, tables, and formatted output. Powered by StatsPAI under the hood. [copaper.ai](https://copaper.ai)

**Team:**

- **Bryce Wang** — Founder. Economics, Finance, CS & AI. Stanford REAP.
- **Dr. Scott Rozelle** — Co-founder & Strategic Advisor. Stanford Senior Fellow, author of *Invisible China*.

---

## Contributing

```bash
git clone https://github.com/brycewang-stanford/statspai.git
cd statspai
pip install -e ".[dev,plotting,fixest]"
pytest
```

---

## Citation

```bibtex
@software{wang2025statspai,
  title={StatsPAI: The Causal Inference & Econometrics Toolkit for Python},
  author={Wang, Bryce},
  year={2025},
  url={https://github.com/brycewang-stanford/statspai},
  version={0.1.0}
}
```

## License

MIT License. See [LICENSE](LICENSE).

---

[GitHub](https://github.com/brycewang-stanford/statspai) · [PyPI](https://pypi.org/project/StatsPAI/) · [Documentation](https://statspai.readthedocs.io/) · [CoPaper.AI](https://copaper.ai)
