# StatsPAI — Ecosystem & Code Statistics

> This page tracks StatsPAI's size and coverage against the broader statistical ecosystem, in the spirit of *evidence-based* positioning. Numbers marked **"measured"** are reproducible on a standard install; numbers marked **"estimated"** are extrapolated from public ecosystem statistics with the reasoning shown.
>
> *Last measured: 2026-04-21 on macOS arm64 against `statspai` 0.9.16, `statsmodels` 0.14.x, `linearmodels` (latest), Stata 18, R 4.5.2.*

---

## 1 · Cross-ecosystem lines-of-code

| Ecosystem / Project                  | Method     |  Files | Lines of code | Primary focus                      |
| ------------------------------------ | ---------- | -----: | ------------: | ---------------------------------- |
| **StatsPAI** `src/statspai/`         | measured   |    528 |   **188,244** | full-stack causal inference        |
| StatsPAI tests (`tests/`)            | measured   |    255 |        42,768 | —                                  |
| statsmodels 0.14.x                   | measured   |    948 |   **381,981** | GLM / time series / general        |
| linearmodels                         | measured   |    131 |        36,607 | panel / IV                         |
| **Python causal-inference subtotal** |            |  1,079 |   **418,588** |                                    |
| Stata 18 — official `.ado`           | measured   |  3,767 |       937,543 | command layer above closed kernel  |
| Stata 18 — official `.mata`          | measured   |    411 |       103,822 | Mata numerical layer               |
| **Stata official executable code**   | measured   |  4,178 | **1,041,365** | (+ 738,543 lines of `.sthlp` help text, not counted as code) |
| Stata SSC (third-party)              | estimated  |  ~3,500 pkgs | **2M – 4M** | community extensions; local sample (reghdfe + winsor2 + 50 others) = 33,296 LOC |
| R base interpreter (C + R + Fortran) | estimated  |      — |        ~1.5M | language itself                    |
| R base library (73 recommended pkgs) | measured   |    509 |        62,321 | shipped with R on this machine     |
| CRAN (~22,000 packages, 2026)        | estimated  |      — |  **80M–120M** (R-only; >200M incl. C/C++/Fortran) | main R package universe |
| Bioconductor (~2,300 packages)       | estimated  |      — |       30M–50M | bioinformatics                     |
| **R ecosystem total**                | estimated  |      — |    **≈ 150M+** |                                   |

**How to read this table**

LOC is a vanity metric in isolation — what matters is **coverage density** within a target domain. StatsPAI is deliberately scoped at causal inference and applied econometrics; it is **not** trying to match R's 150M+ lines because ~90% of CRAN is bioinformatics, visualization, text mining, and general-purpose ML that is out of scope. The relevant comparison is the coverage matrix in §3 below.

---

## 2 · StatsPAI per-module breakdown

Sorted by LOC. All numbers measured on `src/statspai/` at the commit listed in the timestamp above.

| Module              | LOC    | Files | Registered functions (`sp.*`) | Focus                         |
| ------------------- | -----: | ----: | ----------------------------: | ----------------------------- |
| `synth`             | 17,428 |    30 |                            50 | Synthetic Control family      |
| `did`               | 13,926 |    28 |                            53 | DiD + event-study             |
| `rd`                | 11,169 |    21 |                            47 | Regression Discontinuity      |
| `regression`        | 10,597 |    18 |                            36 | OLS / GLM / Logit / Probit    |
| `decomposition`     |  5,828 |    16 |                            29 | Oaxaca / RIF / inequality     |
| `diagnostics`       |  5,372 |    12 |                            20 | assumption checks             |
| `output`            |  5,072 |     8 |                            14 | Word / Excel / LaTeX / HTML   |
| `smart`             |  5,028 |    10 |                            19 | workflow orchestration        |
| `plots`             |  4,925 |     6 |                             — | interactive plot editor       |
| `iv`                |  4,815 |    15 |                             4 | 2SLS / LIML / GMM / Kernel IV |
| `panel`             |  4,511 |    12 |                            17 | FE / RE / Arellano-Bond       |
| `inference`         |  4,446 |    14 |                            22 | SE / bootstrap / wild boot    |
| `frontier`          |  3,951 |     8 |                            12 | SFA / xtfrontier              |
| `bayes`             |  3,823 |    10 |                            17 | Bayesian causal (PyMC)        |
| `multilevel`        |  3,719 |     8 |                            11 | melogit / mepoisson / ICC     |
| `spatial`           |  3,566 |    29 |                            35 | weights → ESDA → GWR → panel  |
| `matching`          |  3,491 |     9 |                            21 | PS / CEM / optimal            |
| `core`              |  3,050 |     6 |                             — | `CausalResult` + infra        |
| `dag`               |  2,921 |     9 |                            19 | DAG inference                 |
| `causal_discovery`  |  2,818 |    10 |                            20 | NOTEARS / PC / LiNGAM / GES   |
| `metalearners`      |  2,406 |     7 |                            21 | S/T/X/R/DR-Learner            |
| `robustness`        |  2,243 |     6 |                             — | spec-curve / sensitivity      |
| `bounds`            |  2,216 |     5 |                             — | Manski / Lee bounds           |
| `mendelian`         |  2,041 |     6 |                            28 | MR-IVW / Egger / PRESSO       |
| `timeseries`        |  1,981 |     9 |                            18 | BVAR / GARCH / LP             |
| `survival`          |  1,884 |     5 |                             8 | Cox / AFT / frailty           |
| `tmle`              |  1,857 |     6 |                             9 | TMLE / HAL-TMLE               |
| `conformal_causal`  |  1,846 |     9 |                            19 | conformal ITE / CATE          |
| `interference`      |  1,795 |    10 |                            18 | spillovers / network          |
| `neural_causal`     |  1,751 |     4 |                             — | TARNet / CFRNet / DragonNet   |
| `causal`            |  1,712 |     5 |                             6 | generic causal utilities      |
| `proximal`          |  1,463 |     8 |                             — | Proximal CI family            |
| `qte`               |  1,406 |     6 |                            12 | quantile / distributional TE  |
| `bcf`               |  1,252 |     5 |                             5 | Bayesian Causal Forest        |
| `bunching`          |  1,171 |     5 |                             8 | bunching estimators           |
| `dml`               |  1,104 |     8 |                             8 | Double/Debiased ML            |
| `mediation`         |    973 |     4 |                             4 | mediation + E-value           |
| `causal_rl`         |    749 |     5 |                             9 | causal reinforcement learning |
| `deepiv`            |    737 |     2 |                             — | Deep IV / KAN-DeepIV          |
| `policy_learning`   |    712 |     3 |                             2 | policy_tree / policy_value    |
| `matrix_completion` |    359 |     2 |                             — | matrix completion for panels  |
| *(other 38 modules)* | ~43K | ~100  |                      ~100     | datasets, utils, compat, etc. |
| **Total**           | **188,244** | **528** |                   **889**   |                               |

`iv`, `neural_causal`, `proximal`, `deepiv`, `matrix_completion`, `robustness`, and `bounds` expose most of their surface via **dispatchers** (`sp.iv(...)`, `sp.robustness_report(...)`, etc.) and shared kernels in `_core.py` — the registered-function count only reflects the top-level named entry points, not the internal algorithm count.

---

## 3 · Causal-inference coverage matrix (full)

Legend: 🏆 most complete across ecosystems · ✅ full coverage · ⚠️ partial / scattered / single algorithm · ❌ not available.

"Stata" = official + major SSC packages. "R" = CRAN. "sm+lm" = statsmodels + linearmodels.

| # | Method family                                                   | Stata | R | sm+lm | DoubleML | **StatsPAI** | StatsPAI entry points |
| --: | ------------------------------------------------------------ | :---: | :---: | :---: | :---: | :---: | --- |
|  1 | DiD — staggered (CS / SA / BJS / dCdH / Gardner / Wooldridge ET) + event-study + honest CIs (Rambachan-Roth) | ⚠️ | ✅ | ❌ | ❌ | 🏆 | `sp.callaway_santanna`, `sp.sun_abraham`, `sp.borusyak_jaravel_spiess`, `sp.dchd`, `sp.gardner_did`, `sp.wooldridge_did`, `sp.honest_did`, `sp.cs_report` |
|  2 | IV — classical (2SLS / LIML / GMM) + modern (Kernel IV / Deep IV / KAN-DeepIV) | ✅ classical only | ✅ classical only | ⚠️ classical (lm) | ⚠️ | 🏆 | `sp.ivreg`, `sp.kernel_iv`, `sp.deep_iv`, `sp.kan_deepiv` |
|  3 | RD — CCT sharp/fuzzy/kink + 2D / boundary + multi-cutoff + honest CIs + ML-CATE (18+ estimators) | ⚠️ (`rdrobust` SSC) | ✅ (`rdrobust`) | ❌ | ❌ | 🏆 | `sp.rdrobust`, `sp.rd2d`, `sp.rdhte`, `sp.rd_forest`, `sp.rd_boost`, `sp.rdrandinf`, `sp.rdpower`, `sp.rdsummary` |
|  4 | Synthetic Control (ADH / ASCM / gsynth / BSTS / Bayesian / PenSCM / FDID — 20 methods + 6 inference strategies) | ⚠️ (`synth` SSC) | ⚠️ (7 pkgs: Synth, gsynth, CausalImpact, MSCMT, …) | ❌ | ❌ | 🏆 | `sp.synth(method=...)`, `sp.synth_compare`, `sp.synth_recommend`, `sp.synth_power`, `sp.synth_sensitivity` |
|  5 | Matching — PS / CEM / optimal / cardinality / one-to-many     | ⚠️ (`psmatch2` SSC) | ✅ (`MatchIt`, `optmatch`) | ❌ | ❌ | ✅ | `sp.match`, `sp.cem`, `sp.optimal_match`, `sp.cardinality_match` |
|  6 | Double / Debiased ML                                          | ❌ | ✅ (`DoubleML`) | ❌ | ✅ | ✅ | `sp.dml(model=...)`, `sp.dml_model_averaging`, `sp.kernel_dml` |
|  7 | Meta-Learners (S/T/X/R/DR) + Causal Forest / GRF              | ❌ | ✅ (`grf`, `rlearner`) | ❌ | ❌ | ✅ | `sp.s_learner`, `sp.t_learner`, `sp.x_learner`, `sp.r_learner`, `sp.dr_learner`, `sp.causal_forest` |
|  8 | TMLE / HAL-TMLE                                               | ❌ | ✅ (`tmle`, `hal9001`) | ❌ | ❌ | ✅ | `sp.tmle`, `sp.hal_tmle`, `sp.ctmle` |
|  9 | Neural causal (TARNet / CFRNet / DragonNet / CEVAE)           | ❌ | ❌ | ❌ | ❌ | 🏆 | `sp.tarnet`, `sp.cfrnet`, `sp.dragonnet`, `sp.cevae` |
| 10 | Causal discovery (NOTEARS / PC / LiNGAM / GES + deep variants) | ❌ | ⚠️ (`pcalg`) | ❌ | ❌ | 🏆 | `sp.notears`, `sp.pc_algorithm`, `sp.lingam`, `sp.ges` |
| 11 | Proximal CI (fortified / bidirectional / MTP / double-negative-control / surrogate) | ❌ | ⚠️ (`pci` scattered) | ❌ | ❌ | 🏆 | `sp.proximal`, `sp.fortified_pci`, `sp.bidirectional_pci`, `sp.pci_mtp`, `sp.double_negative_control` |
| 12 | QTE / distributional TE / CiC / dist-IV / beyond-avg-LATE / HD panel | ⚠️ (`ivqreg`) | ⚠️ (`qte`, `Counterfactual`) | ❌ | ❌ | ✅ | `sp.qte`, `sp.qdid`, `sp.cic`, `sp.distributional_te`, `sp.dist_iv` |
| 13 | Mendelian randomization (IVW / Egger / median / mode / MR-PRESSO / MVMR / BMA) | ❌ | ✅ (`MendelianRandomization`, `TwoSampleMR`) | ❌ | ❌ | ✅ | `sp.mr_ivw`, `sp.mr_egger`, `sp.mr_median`, `sp.mr_presso`, `sp.mvmr`, `sp.mr_bma` |
| 14 | Conformal causal inference (ITE / CATE / density / dose-response / cluster / fair) | ❌ | ❌ | ❌ | ❌ | 🏆 | `sp.conformal_ite`, `sp.conformal_cate`, `sp.conformal_dose_response` |
| 15 | Bayesian causal (BCF / ordinal BCF / factor-exposure BCF)     | ❌ | ⚠️ (`bcf`) | ❌ | ❌ | ✅ | `sp.bcf`, `sp.bcf_ordinal`, `sp.bcf_factor_exposure` |
| 16 | Spatial econometrics (weights → ESDA → ML/GMM → GWR/MGWR → panel) | ❌ | ⚠️ (5 pkgs: spdep, spatialreg, sphet, splm, GWmodel) | ❌ | ❌ | 🏆 | 38 functions including `sp.sem`, `sp.sar`, `sp.gwr`, `sp.mgwr`, `sp.splm` |
| 17 | Policy learning / OPE                                         | ❌ | ⚠️ (`policytree`) | ❌ | ❌ | ✅ | `sp.policy_tree`, `sp.policy_value`, `sp.doubly_robust_ope` |
| 18 | Bunching estimation                                           | ⚠️ (`bunching` SSC) | ❌ | ❌ | ❌ | ✅ | `sp.bunching`, `sp.kink_bunching` |
| 19 | Interference / spillover (partial / network / cluster-RCT / HTE) | ❌ | ⚠️ (`interference`) | ❌ | ❌ | 🏆 | 18 functions including `sp.spillover`, `sp.cluster_rct`, `sp.hte_interference` |
| 20 | Matrix completion for panels                                  | ❌ | ⚠️ (`gsynth`) | ❌ | ❌ | ✅ | `sp.matrix_completion` |
| 21 | Causal MAS (multi-agent LLM causal discovery — arXiv:2509.00987) | ❌ | ❌ | ❌ | ❌ | 🏆 | `sp.causal_mas`, `sp.causal_llm.openai_client`, `sp.causal_llm.anthropic_client` |
| 22 | Publication tables (Word / Excel / LaTeX / HTML / Markdown)   | ⚠️ (`outreg2`) | ⚠️ (`modelsummary`) | ⚠️ | ❌ | 🏆 | Every estimator: `.to_word()` / `.to_excel()` / `.to_latex()` / `.to_html()` |
| 23 | Agent-native tool-calling schemas (`function_schema()`)        | ❌ | ❌ | ❌ | ❌ | 🏆 | `sp.list_functions()`, `sp.describe_function()`, `sp.function_schema()`, `sp.agent.mcp_server` |

---

## 4 · How to reproduce these numbers

```bash
# StatsPAI (Python)
find src/statspai -name '*.py' -exec wc -l {} + | tail -1
find tests        -name '*.py' -exec wc -l {} + | tail -1
python3 -c "import statspai as sp; print(len(sp.list_functions()))"

# statsmodels + linearmodels
python3 -c "import statsmodels, os; print(os.path.dirname(statsmodels.__file__))"
python3 -c "import linearmodels, os; print(os.path.dirname(linearmodels.__file__))"
find $(python3 -c "import statsmodels,os;print(os.path.dirname(statsmodels.__file__))") -name '*.py' -exec wc -l {} + | tail -1
find $(python3 -c "import linearmodels,os;print(os.path.dirname(linearmodels.__file__))") -name '*.py' -exec wc -l {} + | tail -1

# Stata (macOS default install path)
find /Applications/Stata/ado/base -name '*.ado'  | xargs wc -l | tail -1
find /Applications/Stata/ado/base -name '*.mata' | xargs wc -l | tail -1

# R (installed library only; base-interpreter source requires R-<version>.tar.gz)
find /Library/Frameworks/R.framework/Resources/library \( -name '*.R' -o -name '*.r' \) -exec wc -l {} + | tail -1
```

**Ecosystem-wide estimates** (SSC / CRAN / Bioconductor totals) are drawn from:

- SSC — [Boston College Statistical Software Components archive](https://ideas.repec.org/s/boc/bocode.html) package listing.
- CRAN — [METACRAN](https://www.r-pkg.org/) and [CRAN by topic](https://cran.r-project.org/web/views/) task views.
- Bioconductor — [Bioconductor 3.20 release statistics](https://bioconductor.org/news/).

The SSC sample used for extrapolation is the 52-file local install (`/Users/brycewang/Library/Application Support/Stata/ado/plus`, 33,296 LOC — includes `reghdfe`, `winsor2`, and others).

---

## 5 · Why we don't lead with "line-count wins"

Three reasons a naked LOC comparison is misleading for StatsPAI positioning:

1. **Vanity metric**: 188K vs R's 150M+ tells a reviewer *nothing* about capability per line. R's CRAN is ~90% out-of-scope (bioinformatics, visualization, text, general ML) — apples to oranges.
2. **Moving target**: CRAN and statsmodels grow every month. A headline number in the README rots quarterly unless regenerated by CI.
3. **Wrong axis**: StatsPAI's differentiator is *causal-inference depth in one API* — see §3. Most cells where we win are ❌ in Stata / R / statsmodels entirely, not "fewer lines".

The coverage matrix in §3 is the honest positioning. LOC in §1 is supporting evidence for "yes, there is real code behind these claims" — not a competitive boast.
