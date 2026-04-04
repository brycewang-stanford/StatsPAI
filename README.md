# StatsPAI: The Causal Inference & Econometrics Toolkit for Python

[![PyPI version](https://img.shields.io/pypi/v/StatsPAI.svg)](https://pypi.org/project/StatsPAI/)
[![Python versions](https://img.shields.io/pypi/pyversions/StatsPAI.svg)](https://pypi.org/project/StatsPAI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/brycewang-stanford/statspai/blob/main/LICENSE)
[![Tests](https://github.com/brycewang-stanford/statspai/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/brycewang-stanford/statspai/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/statspai?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/statspai)

StatsPAI is a unified Python package for causal inference and applied econometrics. One `import`, 150+ functions, covering the complete empirical research workflow — from classical econometrics to cutting-edge ML/AI causal methods to publication-ready tables in Word, Excel, and LaTeX.

It brings R's [Causal Inference Task View](https://cran.r-project.org/web/views/CausalInference.html) (fixest, did, rdrobust, gsynth, DoubleML, MatchIt, CausalImpact, ...) and Stata's core econometrics commands into a single, consistent Python API.

> Built by the team behind [CoPaper.AI](https://copaper.ai) · Stanford REAP Program

---

## Why StatsPAI?

| Pain point | Stata | R | StatsPAI |
| --- | --- | --- | --- |
| Scattered packages | One environment, but \$695+/yr license | 20+ packages with incompatible APIs | **One `import`, unified API** |
| Publication tables | `outreg2` (limited formats) | `modelsummary` (best-in-class) | **Word + Excel + LaTeX + HTML in every function** |
| Robustness checks | Manual re-runs | Manual re-runs | **`spec_curve()` + `robustness_report()` — one call** |
| Heterogeneity analysis | Manual subgroup splits + forest plots | Manual `lapply` + `ggplot` | **`subgroup_analysis()` with Wald test** |
| Modern ML causal | Limited (no DML, no causal forest) | Fragmented (DoubleML, grf, SuperLearner separate) | **DML, Causal Forest, Meta-Learners, TMLE, DeepIV** |
| Neural causal models | None | None | **TARNet, CFRNet, DragonNet** |
| Causal discovery | None | `pcalg` (complex API) | **`notears()`, `pc_algorithm()`** |
| Policy learning | None | `policytree` (standalone) | **`policy_tree()` + `policy_value()`** |
| Result objects | Inconsistent across commands | Inconsistent across packages | **Unified `CausalResult` with `.summary()`, `.plot()`, `.to_latex()`, `.cite()`** |

---

## Complete Feature List

### Regression Models

| Function | Description | Stata equivalent | R equivalent |
| --- | --- | --- | --- |
| `regress()` | OLS with robust/clustered/HAC SE | `reg y x, r` / `vce(cluster c)` | `fixest::feols()` |
| `ivreg()` | IV / 2SLS with first-stage diagnostics | `ivregress 2sls` | `fixest::feols()` with IV |
| `panel()` | Fixed Effects, Random Effects, Between, FD | `xtreg, fe` / `xtreg, re` | `plm::plm()` |
| `heckman()` | Heckman selection model | `heckman` | `sampleSelection::selection()` |
| `qreg()`, `sqreg()` | Quantile regression | `qreg` / `sqreg` | `quantreg::rq()` |
| `tobit()` | Censored regression (Tobit) | `tobit` | `censReg::censReg()` |
| `xtabond()` | Arellano-Bond dynamic panel GMM | `xtabond` | `plm::pgmm()` |

### Difference-in-Differences

| Function | Description | Reference |
| --- | --- | --- |
| `did()` | Auto-dispatching DID (2×2 or staggered) | — |
| `did_2x2()` | Classic two-group, two-period DID | — |
| `callaway_santanna()` | Staggered DID with heterogeneous effects | Callaway & Sant'Anna (2021) |
| `sun_abraham()` | Interaction-weighted event study | Sun & Abraham (2021) |
| `bacon_decomposition()` | TWFE decomposition diagnostic | Goodman-Bacon (2021) |
| `honest_did()` | Sensitivity to parallel trends violations | Rambachan & Roth (2023) |

### Regression Discontinuity

| Function | Description | Reference |
| --- | --- | --- |
| `rdrobust()` | Sharp/Fuzzy RD with robust bias-corrected inference | Calonico, Cattaneo & Titiunik (2014) |
| `rdplot()` | RD visualization with binned scatter | — |
| `rddensity()` | McCrary density manipulation test | McCrary (2008) |

### Matching & Reweighting

| Function | Description | Stata equivalent |
| --- | --- | --- |
| `match()` | PSM, Mahalanobis, CEM with balance diagnostics | `psmatch2` / `cem` |
| `ebalance()` | Entropy balancing | `ebalance` |

### Synthetic Control

| Function | Description | Reference |
| --- | --- | --- |
| `synth()` | Abadie-Diamond-Hainmueller SCM | Abadie et al. (2010) |
| `sdid()` | Synthetic Difference-in-Differences | Arkhangelsky et al. (2021) |
| Placebo inference, gap plots, weight tables, RMSE plots | — | — |

### Machine Learning Causal Inference

| Function | Description | Reference |
| --- | --- | --- |
| `dml()` | Double/Debiased ML (PLR + IRM) with cross-fitting | Chernozhukov et al. (2018) |
| `causal_forest()` | Causal Forest for heterogeneous treatment effects | Wager & Athey (2018) |
| `deepiv()` | Deep IV neural network approach | Hartford et al. (2017) |
| `metalearner()` | S/T/X/R/DR-Learner for CATE estimation | Kunzel et al. (2019), Kennedy (2023) |
| `tmle()` | Targeted Maximum Likelihood Estimation | van der Laan & Rose (2011) |
| `aipw()` | Augmented Inverse-Probability Weighting | — |

### Neural Causal Models

| Function | Description | Reference |
| --- | --- | --- |
| `tarnet()` | Treatment-Agnostic Representation Network | Shalit et al. (2017) |
| `cfrnet()` | Counterfactual Regression Network | Shalit et al. (2017) |
| `dragonnet()` | Dragon Neural Network for CATE | Shi et al. (2019) |

### Causal Discovery

| Function | Description | Reference |
| --- | --- | --- |
| `notears()` | DAG learning via continuous optimization | Zheng et al. (2018) |
| `pc_algorithm()` | Constraint-based causal graph learning | Spirtes et al. (2000) |

### Policy Learning

| Function | Description | Reference |
| --- | --- | --- |
| `policy_tree()` | Optimal treatment assignment rules | Athey & Wager (2021) |
| `policy_value()` | Policy value evaluation | — |

### Conformal & Bayesian Causal Inference

| Function | Description | Reference |
| --- | --- | --- |
| `conformal_cate()` | Distribution-free prediction intervals for ITE | Lei & Candes (2021) |
| `bcf()` | Bayesian Causal Forest (separate mu/tau) | Hahn, Murray & Carvalho (2020) |

### Dose-Response & Multi-valued Treatment

| Function | Description | Reference |
| --- | --- | --- |
| `dose_response()` | Continuous treatment dose-response curve (GPS) | Hirano & Imbens (2004) |
| `multi_treatment()` | Multi-valued treatment AIPW | Cattaneo (2010) |

### Bounds & Partial Identification

| Function | Description | Reference |
| --- | --- | --- |
| `lee_bounds()` | Sharp bounds under sample selection | Lee (2009) |
| `manski_bounds()` | Worst-case bounds (no assumption / MTR / MTS) | Manski (1990) |

### Interference & Spillover

| Function | Description | Reference |
| --- | --- | --- |
| `spillover()` | Direct + spillover + total effect decomposition | Hudgens & Halloran (2008) |

### Dynamic Treatment Regimes

| Function | Description | Reference |
| --- | --- | --- |
| `g_estimation()` | Multi-stage optimal DTR via G-estimation | Robins (2004) |

### Bunching & Tax Policy

| Function | Description | Reference |
| --- | --- | --- |
| `bunching()` | Kink/notch bunching estimator with elasticity | Kleven & Waseem (2013) |

### Matrix Completion (Panel)

| Function | Description | Reference |
| --- | --- | --- |
| `mc_panel()` | Causal panel data via nuclear-norm matrix completion | Athey et al. (2021) |

### Other Causal Methods

| Function | Description | Stata/R equivalent |
| --- | --- | --- |
| `causal_impact()` | Bayesian structural time-series | R `CausalImpact` |
| `mediate()` | Mediation analysis (ACME/ADE) | `medeff` / R `mediation` |
| `bartik()` | Shift-share IV with Rotemberg weights | `bartik_weight` |

### Post-Estimation

| Function | Description | Stata equivalent |
| --- | --- | --- |
| `margins()` | Average marginal effects (AME/MEM) | `margins, dydx(*)` |
| `marginsplot()` | Marginal effects visualization | `marginsplot` |
| `test()` | Wald test for linear restrictions | `test x1 = x2` |
| `lincom()` | Linear combinations with inference | `lincom x1 + x2` |

### Diagnostics & Sensitivity

| Function | Description | Reference |
| --- | --- | --- |
| `oster_bounds()` | Coefficient stability bounds | Oster (2019) |
| `sensemakr()` | Sensitivity to omitted variables | Cinelli & Hazlett (2020) |
| `mccrary_test()` | Density discontinuity test | McCrary (2008) |
| `hausman_test()` | FE vs RE specification test | Hausman (1978) |
| `anderson_rubin_test()` | Weak instrument robust inference | Anderson & Rubin (1949) |
| `evalue()` | E-value sensitivity to unmeasured confounding | VanderWeele & Ding (2017) |
| `het_test()` | Breusch-Pagan / White heteroskedasticity | — |
| `reset_test()` | Ramsey RESET specification test | — |
| `vif()` | Variance Inflation Factor | — |
| `diagnose()` | General model diagnostics | — |

### Robustness Analysis *(unique to StatsPAI)*

| Function | Description | R/Stata equivalent |
| --- | --- | --- |
| `spec_curve()` | Specification Curve / Multiverse Analysis | R `specr` (limited) / Stata: none |
| `robustness_report()` | Automated robustness battery (SE variants, winsorize, trim, add/drop controls, subsamples) | None |
| `subgroup_analysis()` | Heterogeneity analysis with forest plot + interaction Wald test | None (manual in both) |

### Inference Methods

| Function | Description |
| --- | --- |
| `wild_cluster_bootstrap()` | Wild cluster bootstrap (Cameron, Gelbach & Miller 2008) |
| `ri_test()` | Randomization inference / Fisher exact test |

### CATE Diagnostics (for Meta-Learners & Causal Forest)

| Function | Description |
| --- | --- |
| `cate_summary()`, `cate_by_group()` | CATE distribution summaries |
| `cate_plot()`, `cate_group_plot()` | CATE visualization |
| `gate_test()` | Group Average Treatment Effect test |
| `blp_test()` | Best Linear Projection test |
| `compare_metalearners()` | Compare S/T/X/R/DR-Learner estimates |

### Publication-Quality Output

| Function | Description | Formats |
| --- | --- | --- |
| `modelsummary()` | Multi-model comparison tables | Text, LaTeX, HTML, Word, Excel, DataFrame |
| `outreg2()` | Stata-style regression table export | Excel, LaTeX, Word |
| `sumstats()` | Summary statistics (Table 1) | Text, LaTeX, HTML, Word, Excel, DataFrame |
| `balance_table()` | Pre-treatment balance check | Text, LaTeX, HTML, Word, Excel, DataFrame |
| `tab()` | Cross-tabulation with chi-squared / Fisher | Text, LaTeX, Word, Excel, DataFrame |
| `coefplot()` | Coefficient forest plot across models | matplotlib Figure |
| `binscatter()` | Binned scatter with residualization | matplotlib Figure |
| `set_theme()` | Publication themes (`'academic'`, `'aea'`, `'minimal'`, `'cn_journal'`) | — |

Every result object has:

```python
result.summary()      # Formatted text summary
result.plot()         # Appropriate visualization
result.to_latex()     # LaTeX table
result.to_docx()      # Word document
result.cite()         # BibTeX citation for the method
```

### Utilities

| Function | Description | Stata equivalent |
| --- | --- | --- |
| `label_var()`, `label_vars()` | Variable labeling | `label var` |
| `describe()` | Data description | `describe` |
| `pwcorr()` | Pairwise correlation with significance stars | `pwcorr, star(.05)` |
| `winsor()` | Winsorization | `winsor2` |
| `read_data()` | Multi-format data reader | `use` / `import` |

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
r5 = sp.dml(df, y='wage', treat='training', covariates=['age', 'edu', 'exp'])
r6 = sp.causal_forest("y ~ treatment | x1 + x2 + x3", data=df)

# --- Post-estimation ---
sp.margins(r1, data=df)              # Marginal effects
sp.test(r1, "education = experience") # Wald test
sp.oster_bounds(df, y='wage', treat='education', controls=['experience'])

# --- Tables (to Word / Excel / LaTeX) ---
sp.modelsummary(r1, r2, output='table2.docx')
sp.outreg2(r1, r2, r3, filename='results.xlsx')
sp.sumstats(df, vars=['wage', 'education', 'age'], output='table1.docx')

# --- Robustness (unique to StatsPAI) ---
sp.spec_curve(df, y='wage', x='education',
              controls=[[], ['experience'], ['experience', 'female']],
              se_types=['nonrobust', 'hc1']).plot()

sp.robustness_report(df, formula="wage ~ education + experience",
                     x='education', extra_controls=['female'],
                     winsor_levels=[0.01, 0.05]).plot()

sp.subgroup_analysis(df, formula="wage ~ education + experience",
                     x='education',
                     by={'Gender': 'female', 'Region': 'region'}).plot()
```

---

## StatsPAI vs Stata vs R: Honest Comparison

### Where StatsPAI wins

| Advantage | Detail |
| --- | --- |
| **Unified API** | One package, one `import`, consistent `.summary()` / `.plot()` / `.to_latex()` across all methods. Stata requires paid add-ons; R requires 20+ packages with different interfaces. |
| **Modern ML causal methods** | DML, Causal Forest, Meta-Learners (S/T/X/R/DR), TMLE, DeepIV, TARNet/CFRNet/DragonNet, Policy Trees — all in one place. Stata has almost none of these. R has them scattered across incompatible packages. |
| **Robustness automation** | `spec_curve()`, `robustness_report()`, `subgroup_analysis()` — no manual re-running. Neither Stata nor R offers this out-of-the-box. |
| **Free & open source** | MIT license, \$0. Stata costs \$695–\$1,595/year. |
| **Python ecosystem** | Integrates naturally with pandas, scikit-learn, PyTorch, Jupyter, cloud pipelines. |
| **Auto-citations** | Every causal method has `.cite()` returning the correct BibTeX. Neither Stata nor R does this. |

### Where Stata still wins

| Advantage | Detail |
| --- | --- |
| **Battle-tested at scale** | 40+ years of production use in economics. Edge cases are well-handled. |
| **Speed on very large datasets** | Stata's compiled C backend is faster for simple OLS/FE on datasets with millions of rows. |
| **Survey data & complex designs** | `svy:` prefix, stratification, clustering — Stata's survey support is unmatched. |
| **Mature documentation** | Every command has a PDF manual with worked examples. Community is massive. |
| **Journal acceptance** | Referees in some fields trust Stata output by default. |

### Where R still wins

| Advantage | Detail |
| --- | --- |
| **Cutting-edge methods** | New econometric methods (e.g., `fixest`, `did2s`, `HonestDiD`) often appear in R first. |
| **`ggplot2` visualization** | R's grammar of graphics is more flexible than matplotlib for complex figures. |
| **`modelsummary`** | R's `modelsummary` is the gold standard for regression tables — StatsPAI's is close but not yet identical. |
| **CRAN quality control** | R packages go through peer review. Python packages vary in quality. |
| **Spatial econometrics** | `spdep`, `spatialreg` — R has a deeper spatial ecosystem. |

---

## API at a Glance

```text
150+ public functions/classes

Regression:     regress, ivreg, panel, heckman, qreg, sqreg, tobit, xtabond
DID:            did, did_2x2, callaway_santanna, sun_abraham, bacon_decomposition, honest_did
RD:             rdrobust, rdplot, rddensity
Matching:       match, ebalance
Synth:          synth, sdid
ML Causal:      dml, causal_forest, deepiv, metalearner, tmle, aipw
Neural:         tarnet, cfrnet, dragonnet
Discovery:      notears, pc_algorithm
Policy:         policy_tree, policy_value
Conformal/Bayes:conformal_cate, bcf
Dose-Response:  dose_response
Multi-Treat:    multi_treatment
Bounds:         lee_bounds, manski_bounds
Interference:   spillover
DTR:            g_estimation
Bunching:       bunching
Panel MC:       mc_panel
Other:          causal_impact, mediate, bartik
Post-est:       margins, marginsplot, test, lincom
Diagnostics:    oster_bounds, sensemakr, evalue, mccrary_test, hausman_test, het_test, reset_test, vif
Robustness:     spec_curve, robustness_report, subgroup_analysis
Inference:      wild_cluster_bootstrap, ri_test
Output:         modelsummary, outreg2, sumstats, balance_table, tab, coefplot, binscatter
```

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
  version={0.3.1}
}
```

## License

MIT License. See [LICENSE](LICENSE).

---

[GitHub](https://github.com/brycewang-stanford/statspai) · [PyPI](https://pypi.org/project/StatsPAI/) · [User Guide](https://github.com/brycewang-stanford/statspai#quick-example) · [CoPaper.AI](https://copaper.ai)
