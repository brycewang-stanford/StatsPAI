# StatsPAI Ecosystem Gap Analysis & Roadmap to v1.0

**Date**: 2026-04-15
**Status**: Top-level design — anchor for subsequent sub-specs
**Baseline**: StatsPAI v0.7.0 (54 modules, ~85k LOC, 250+ public functions)

---

## 1. Purpose

This document is the top-level design anchor for a comprehensive upgrade of StatsPAI
to reach parity with — and in several dimensions exceed — the mainstream R and
Python ecosystems for applied econometrics and causal inference. It catalogues
the current capability, maps each domain to the reference ecosystem, quantifies
the gap, and decomposes the upgrade into ten independent sub-projects.

Each sub-project below will have its own spec (`YYYY-MM-DD-sp-NN-<topic>-design.md`)
and its own implementation plan before any code is written.

## 2. Reference Ecosystem (Benchmark)

### R (mainstream)
`stats`, `fixest`, `estimatr`, `did`, `didimputation`, `DIDmultiplegt`, `honestDiD`,
`rdrobust`, `rddensity`, `rdlocrand`, `rdpower`, `Synth`, `augsynth`, `synthdid`,
`gsynth`, `MatchIt`, `cobalt`, `WeightIt`, `ebal`, `plm`, `grf`, `DoubleML`,
`mediation`, `EValue`, `sensemakr`, `spatialreg`, `spdep`, `splm`, `sphet`,
`GWmodel`, `vars`, `urca`, `rugarch`, `bvartools`, `survival`, `flexsurv`,
`pcalg`, `bnlearn`, `MendelianRandomization`, `TwoSampleMR`, `policytree`,
`survey`, `frontier`, `mice`, `Amelia`, `randomizr`, `DeclareDesign`.

### Python (mainstream)
`statsmodels`, `linearmodels`, `pyfixest`, `EconML`, `CausalML`, `DoubleML`,
`dowhy`, `causal-learn`, `lingam`, `causalnex`, `PySAL` (`libpysal`, `esda`,
`spreg`, `mgwr`, `splot`), `pyblp`, `lifelines`, `scikit-survival`, `mapie`,
`differences`, `pysyncon`, `SparseSC`.

## 3. Gap Matrix (24 domains)

Legend: 🟢 parity / 🟡 usable but gap / 🟠 large gap / 🔴 missing or placeholder

| # | Domain | StatsPAI Current | R Benchmark | Python Benchmark | Gap | Priority |
|---|---|---|---|---|---|---|
| 1 | Linear / IV basics | `regress`, `iv`, `advanced_iv` | fixest, estimatr, ivreg | linearmodels, pyfixest | No OOS `predict()`; HDFE speed behind fixest | 🟡 **P0** |
| 2 | GLM / discrete choice | logit/probit/poisson/zip/mlogit/fracreg | glm, VGAM, mlogit | statsmodels | Some GLM families raise `NotImplementedError` | 🟡 **P1** |
| 3 | DID (static + staggered) | did/CS/SA/bacon/honest/sdid/DDD/imputation/stacked/continuous | did, didimputation, DIDmultiplegt, honestDiD, fixest::sunab | differences, pyfixest | 🟢 **industry-leading**; only missing explicit ETWFE (Wooldridge 2021) API | 🟢 **P2** |
| 4 | RD / RKD / RDD | rdrobust/rdplot/rdbw/rkd/rd_honest/rdmc/rdms | rdrobust, rddensity, **rdlocrand**, **rdpower**, rdmulti | rdrobust-py | Missing `rdlocrand` (local randomization) and `rdpower` (power) | 🟡 **P1** |
| 5 | Synthetic control | SCM/gsynth/sdid/augsynth/conformal_synth/staggered | Synth, augsynth, synthdid, gsynth, tidysynth | pysyncon, SparseSC | 🟢 parity | 🟢 **P3** |
| 6 | Matching / weighting | NN/CEM/ebal/PS | **MatchIt**, cobalt, WeightIt, Matching | causalinference | Missing genetic / optimal / cardinality matching; balance reports shallow | 🟠 **P1** |
| 7 | Panel | FE/RE/Within/Between/FD/GMM/FGLS/interactive_fe/unitroot | plm, fixest, panelvar | linearmodels | Missing CCE (Pesaran), panel cointegration, dynamic panel diagnostics | 🟡 **P1** |
| 8 | Inference / resampling | WCB/AIPW/Fisher/Conley/CR2/bootstrap | sandwich, fwildclusterboot, clubSandwich | wildboottest, pyfixest | 🟢 parity | 🟢 **P2** |
| 9 | ML-causal core | causal_forest / metalearners / DML / TMLE / conformal_cate | grf, DoubleML, tlverse | EconML, CausalML, DoubleML | Missing full GRF kernel API, IV forest, local-linear forest, orthogonal forest | 🟠 **P0** |
| 10 | Deep causal | TARNet / CFRNet / DragonNet / DeepIV | — | EconML deep, CausalML | Training stability issues; representation-balancing depth | 🟠 **P2** |
| 11 | Causal discovery | NOTEARS / PC | **pcalg, bnlearn** | causal-learn, **lingam**, causalnex, dowhy | Missing FCI / GES / LiNGAM family / GIES / non-linear NOTEARS / score metrics | 🔴 **P1** |
| 12 | Mediation | mediate (basic) | **mediation (Imai)**, cmaverse | CausalML mediation | Missing Imai ρ-sensitivity, time-varying mediators, sequential multi-mediator | 🟠 **P1** |
| 13 | IV specialties | bartik / shift_share / mendelian / liml / jive | AER, ivreg, **ivmodel**, **ShiftShareSE** | linearmodels | Missing Anderson-Rubin robust inference, Andrews-Stock-Sun 2024, post-LASSO IV | 🟡 **P1** |
| 14 | Sensitivity / bounds | Lee / Manski / Oster / E-value / sensemakr / breakdown | sensemakr, EValue, bounds, tipr | sensemakr-py | 🟢 parity | 🟢 **P2** |
| 15 | Power / design | RCT/DID/RD/IV/cluster | pwr, **DeclareDesign**, PowerUpR | statsmodels.power | Missing DeclareDesign-style declaration→diagnose workflow | 🟡 **P2** |
| 16 | **Spatial econometrics** | SAR/SEM/SDM (ML, dense matrices) | **spatialreg+spdep+splm+sphet+GWmodel** | **PySAL (spreg+esda+mgwr)** | 🔴 **Largest gap**: weight builders, Moran/LISA, GWR/MGWR, spatial panel/GMM, sparse solver | 🔴 **P0** |
| 17 | Time series | VAR / Granger / IRF / breaks / cointegration | **vars, urca, rugarch, bvartools, tsDyn** | statsmodels.tsa | Missing ARIMA/SARIMAX wrapper, state-space Kalman, BVAR, MSVAR, GARCH, **local projections (Jordà)** | 🟠 **P1** |
| 18 | Survival | Cox / KM / survreg / logrank | **survival, flexsurv, rms** | lifelines, scikit-survival | Missing frailty, AFT family, competing risks, time-varying covariates | 🟡 **P2** |
| 19 | Nonparametric | lpoly / kdensity | locfit, KernSmooth, np | KDEpy | Missing CV bandwidth, multivariate kernel, adaptive kernel | 🟡 **P2** |
| 20 | Decomposition | Oaxaca / Gelbach | oaxaca, dineq | — | Missing RIF decomposition (Firpo-Fortin-Lemieux), DFL density decomposition, Ñopo | 🟠 **P1** |
| 21 | Experimental design | randomize / balance / attrition / optimal_design | **randomizr, DeclareDesign, blockTools** | — | Missing block/stratified/covariate-adaptive/re-randomization | 🟠 **P1** |
| 22 | Survey sampling | svydesign / svymean / svytotal / svyglm | **survey (gold standard), srvyr** | samplics | Missing replication weights (BRR/JK), calibration (raking/post-strat), small-area | 🟠 **P1** |
| 23 | Structural | BLP | — | **pyblp** | Missing pyblp-depth parity (supply side, micro-BLP, elasticities); AIDS system missing | 🟠 **P2** |
| 24 | Missing data / MR | mice / mendelian(ivw/egger/median) | mice, Amelia, **TwoSampleMR, MR-PRESSO** | miceforest | MR missing PRESSO/RAPS/CAUSE/colocalization; MI missing Amelia EM | 🟠 **P2** |

## 4. Strategic Judgment

- **Three P0 fronts** are the main battlefield for this upgrade cycle:
  1. **Spatial econometrics (#16)** — largest absolute gap, highest ROI, natural entry point.
  2. **ML-causal deepening (#9)** — frontier battlefield; `causal_forest` shell exists but lacks `grf` breadth.
  3. **Infrastructure completion (#1, #2)** — unglamorous but touched every session.
- **Three 🟢 domains** already world-class (DID, SCM, sensitivity) — freeze, only minor polish.

## 5. Sub-Project Decomposition

| ID | Title | Scope | Size | Sprint |
|---|---|---|---|---|
| **SP-01** | Spatial econometrics full-stack | #16 | XL (4–6 wk) | **S1** |
| **SP-02** | GRF full parity | #9 | L (3–4 wk) | S2 |
| **SP-03** | Infrastructure completion | #1, #2 | M (2 wk) | S2 |
| **SP-04** | Time-series modernization | #17 | L (3–4 wk) | S3 |
| **SP-05** | Causal-discovery breadth | #11 | M (2–3 wk) | S3 |
| **SP-06** | Matching / weighting upgrade | #6 | M (2 wk) | S3 |
| **SP-07** | RD / IV robust inference | #4, #13 | M (2 wk) | S4 |
| **SP-08** | Decomposition + mediation depth | #12, #20 | M (2 wk) | S4 |
| **SP-09** | Experimental design / survey | #21, #22 | M (2–3 wk) | S4 |
| **SP-10** | Survival / nonparametric / structural | #18, #19, #23 | M (2–3 wk) | S5 |

**Total horizon**: 5 sprints, ~3–4 months, v1.0 closure.

## 6. Execution Principles (applies to every sub-project)

1. **Serial polish over parallel sprawl** — one sub-project closes before the next opens.
2. **Brainstorm → Write spec → Write plan → Implement → Verify → Merge** for each sub-project.
3. Every new public function must have: unified `CausalResult`/`EconometricResults` return, `.summary()` / `.to_latex()` / `.cite()`, NumPy-style docstring, ≥3 pytest cases, cross-validated against R or reference Python implementation (tolerance stated per module).
4. Import convention: `import statspai as sp`. All examples use `sp.xxx`.
5. Direct push to `main` (per user preference); no PRs unless explicitly requested.

## 7. First Sub-Project

**SP-01 (Spatial econometrics full-stack)** enters brainstorming immediately after
this document is committed. Its spec will live at
`specs/2026-04-15-sp-01-spatial-full-stack-design.md`.
