# `tests/stata_parity/` — cross-language parity harness against Stata

This directory is the **StatsPAI ↔ Stata** sibling of
[`tests/r_parity/`](../r_parity/): each module pair runs the same
calibrated replica on both sides, dumps a full-precision JSON
result, and `tests/r_parity/compare.py` joins the three sides
(StatsPAI, R, Stata) into a single 3-way Track A parity table for
the JSS Appendix B.

The harness is read by the same `compare.py` that drives the R
side — there is **one** comparator and **one** tolerance budget
(`compare.py::TOLERANCES`). Parity is a property of the estimator,
not of the reference language, so we deliberately do not register
a separate budget for the Stata comparison. Known Stata convention
gaps that exceed the shared headline budget are explicitly enumerated
in `compare.py::STATA_HEADLINE_GAP_EXCEPTIONS`.

## What's here

```
tests/stata_parity/
├── README.md
├── _common.do            # shared scaffolding: JSON writer (file-based, survives `mata clear`)
├── _quick_compare.py     # ad-hoc 3-way comparator while developing modules
├── NN_<method>.do        # one .do per module
├── logs/                 # Stata's per-run .smcl/.log + the JSON-row tmp files
└── results/
    └── NN_<method>_Stata.json   # full-precision results, joined by compare.py
```

Each `.do` file imports `../r_parity/data/NN_<name>.csv` (the same
bytes the R side reads), runs the canonical Stata reference, and
writes one row per parity statistic to
`results/NN_<name>_Stata.json` via the helpers in `_common.do`.

## Materialized Stata golden modules (81 of 87 Python modules)

| # | Method                       | StatsPAI                       | Stata reference                                              |
| --- | --- | --- | --- |
| 01 | OLS + HC1 SE                  | `sp.regress`                   | `regress, vce(robust)`                                       |
| 02 | 2SLS + HC1 SE                 | `sp.iv`                        | `ivregress 2sls, vce(robust) small`                          |
| 03 | HDFE 2-way FE                 | `sp.fast.feols`                | `reghdfe, absorb(...) vce(unadjusted)`                       |
| 04 | CS-DiD simple ATT             | `sp.callaway_santanna`         | `csdid + estat simple, method(reg)`                          |
| 05 | Sun-Abraham event study       | `sp.sun_abraham`               | `eventstudyinteract`                                         |
| 06 | RD CCT bias-corrected         | `sp.rdrobust`                  | `rdrobust`                                                   |
| 07 | Classical SCM                 | `sp.synth(method="classic")`   | `synth ..., trunit(...) trperiod(...) nested`                |
| 08 | DML PLR                       | `sp.dml`                       | audited Stata/Mata linear-nuisance DML2 bridge               |
| 09 | RD density (CJM)              | `sp.rddensity`                 | `rddensity`                                                  |
| 10 | Honest DiD bounds (FLCI)      | `sp.honest_did`                | `honestdid, b(...) vcov(...) numpre(...) mvec(...) delta(sd)`|
| 11 | PSM 1:1 NN                    | `sp.psm`                       | `teffects psmatch, atet nneighbor(1)`                        |
| 12 | Synthetic DiD                 | `sp.synth(method="sdid")`      | `sdid ..., vce(placebo)`                                     |
| 14 | OLS + cluster (CR1)           | `sp.regress(robust="cluster")` | `regress, vce(cluster ...)`                                  |
| 15 | HDFE + cluster                | `sp.fast.feols(vcov="cluster")`| `reghdfe, absorb(...) vce(cluster ...)`                      |
| 16 | BJS imputation                | `sp.bjs_pretrend_joint`        | `did_imputation, autosample`                                 |
| 17 | Wooldridge ETWFE              | `sp.wooldridge_did`            | `jwdid + estat simple`                                       |
| 20 | Goodman-Bacon decomposition   | `sp.bacon_decomposition`       | `bacondecomp, ddetail`                                       |
| 21 | Honest-DiD relative-mags      | `sp.honest_did(restriction="relative_magnitudes")` | `honestdid, ... delta(rm) method(Conditional) gridPoints(1000) grid_lb(-2) grid_ub(2)` |
| 22 | sensemakr robustness          | `sp.sensemakr`                 | `sensemakr depvar regs, treat(...) benchmark(...) kd(1) ky(1)` |
| 23 | E-value                       | `sp.evalue`                    | `evalue rr`                                                 |
| 24 | Cox proportional hazards      | `sp.survival.cox`              | `stcox`                                                     |
| 25 | Linear mixed model            | `sp.mixed`                     | `mixed ..., reml`                                            |
| 26 | GLMM logit (Laplace)          | `sp.melogit`                   | `melogit ..., intmethod(laplace)`                           |
| 27 | GLMM AGHQ (n=8)               | `sp.melogit(nAGQ=8)`           | `melogit ..., intpoints(8)`                                  |
| 28 | Stochastic frontier (cross-sec) | `sp.frontier`                | `frontier, distribution(hnormal)`                            |
| 29 | Panel SFA Pitt-Lee            | `sp.xtfrontier`                | `xtfrontier, ti`                                             |
| 30 | Blinder-Oaxaca decomposition  | `sp.oaxaca_blinder`            | `oaxaca`                                                     |
| 31 | DFL reweighting               | `sp.decompose("dfl")`          | audited Stata/Mata DFL reweighting bridge                    |
| 32 | RIF / UQR decomposition       | `sp.rif_decomposition`         | audited Stata/Mata RIF-Oaxaca bridge                         |
| 33 | VAR                           | `sp.var`                       | `var`                                                        |
| 34 | Local projections             | `sp.local_projections`         | horizon-by-horizon `regress`; `lpirf` recorded in extras     |
| 35 | Panel FE/RE/Hausman           | `sp.panel`                     | `xtreg, fe/re` + `hausman`                                   |
| 36 | Causal mediation              | `sp.mediation`                 | `paramed`                                                    |
| 37 | PPML + HDFE                   | `sp.ppmlhdfe`                  | `ppmlhdfe`                                                   |
| 38 | DR-DID (Sant'Anna-Zhao)       | `sp.drdid(method="imp")`       | `drdid y x, ivar(id) time(post) treatment(treated) drimp`    |
| 39 | ARIMA(2,0,0)                  | `sp.arima`                     | `arima`                                                      |
| 40 | Quantile reg (median)         | `sp.qreg`                      | `qreg`                                                       |
| 41 | Tobit (left-censored)         | `sp.tobit`                     | `tobit, ll(0)`                                               |
| 42 | Negative binomial             | `sp.nbreg`                     | `nbreg`                                                      |
| 43 | Heckman 2-step                | `sp.heckman`                   | `heckman, twostep`                                           |
| 44 | Multinomial logit             | `sp.mlogit`                    | `mlogit`                                                     |
| 45 | Ordered logit                 | `sp.ologit`                    | `ologit`                                                     |
| 46 | Conditional logit             | `sp.clogit`                    | `clogit, group(...)`                                         |
| 47 | PPML + 3-way HDFE             | `sp.ppmlhdfe`                  | `ppmlhdfe, absorb(origin dest year)`                         |
| 48 | Binary probit                 | `sp.probit`                    | `probit`                                                     |
| 49 | Ordered probit                | `sp.oprobit`                   | `oprobit`                                                    |
| 50 | Arellano-Bond GMM             | `sp.xtabond`                   | `xtabond`                                                    |
| 51 | Newey-West HAC OLS            | `sp.regress(robust="hac")`     | `newey`                                                      |
| 52 | Classical SCM unique solution | `sp.synth(method="classic")`   | `synth y y(0..19), trunit(6) trperiod(20)`                   |
| 53 | CR2 / CR3 cluster SE          | `sp.cr2_se` / `sp.fast.crve`   | audited Stata/Mata cluster-hat bridge                        |
| 54 | Two-way cluster SE            | `sp.twoway_cluster`            | audited Stata/Mata CGM bridge; `reghdfe` diagnostic row       |
| 55 | OLS + HC2 / HC3 SE            | `sp.regress(robust="hc2"/"hc3")` | `regress, vce(hc2)` / `regress, vce(hc3)`                  |
| 56 | Three-way cluster SE          | `sp.multiway_cluster_vcov`     | audited Stata/Mata CGM bridge; `reghdfe` diagnostic row       |
| 57 | Binary logit                  | `sp.logit`                     | `logit`                                                      |
| 58 | Poisson ML (no FE)            | `sp.poisson`                   | `poisson`                                                    |
| 59 | LIML k-class IV               | `sp.liml`                      | `ivregress liml, small`                                      |
| 60 | SUR one-step FGLS             | `sp.sureg`                     | `sureg`                                                      |
| 61 | Beta regression               | `sp.betareg`                   | `betareg, nrtolerance(1e-13)`                                |
| 62 | Truncated regression          | `sp.truncreg`                  | `truncreg, ll(0)`                                            |
| 63 | Zero-inflated Poisson         | `sp.zip_model`                 | `zip, inflate(...)`                                          |
| 64 | Zero-inflated NB              | `sp.zinb`                      | `zinb, inflate(...) nrtolerance(1e-13)`                      |
| 65 | Spatial ML (SAR/SEM/SDM)      | `sp.sar`, `sp.sem`, `sp.sdm`   | `spregress, ml dvarlag()/errorlag()/ivarlag()`                |
| 70 | Policy tree (depth 1)         | `sp.policy_tree(depth=1)`      | audited Mata exhaustive depth-1 welfare search                |
| 72 | TMLE targeting step           | `sp.tmle(fluctuation='per_arm')` | audited Mata bridge: `glm ..., offset(logit(QAW)) noconstant` |
| 68 | Within transformation         | `sp.demean(solver='map')`      | `bysort id: egen mean` + subtract                             |
| 69 | Balanced-panel filter         | `sp.balance_panel`             | distinct-period count per entity, keep the full ones          |
| 66 | Spatial GMM (SAR-2SLS)        | `sp.sar_gmm`                   | audited Mata GS2SLS bridge: `ivregress 2sls (Wy = WX), small` |
| 67 | Panel GLM (feglm/fepois)      | `sp.feglm`, `sp.fepois`        | `logit y x1 x2 i.id` / `ppmlhdfe, absorb(id)` (point-only)    |
| 71 | DML family (IRM/PLIV/IIVM)    | `sp.dml(model=...)`            | `ddml init interactive\|iv\|interactiveiv, foldvar(...)`      |
| 73 | Gardner two-stage DiD         | `sp.gardner_did`               | `did2s`                                                      |
| 74 | Changes-in-Changes            | `sp.cic`                       | `cic all ..., at(10(10)90)` (`discrete_ci` column)           |
| 75 | Stacked DiD                   | `sp.stacked_did`               | hand-built stack + `reghdfe`                                 |
| 76 | Pre-trends power (Roth 2022)  | `sp.pretrends_power`           | `pretrends` / `pretrends power`                              |
| 78 | dCDH intertemporal event study| `sp.did_multiplegt_dyn`        | `did_multiplegt_dyn, effects() placebo() cluster()`          |
| 81 | dCDH 2020 DID_M               | `sp.did_multiplegt`            | `did_multiplegt_old, placebo(1) breps(0)`                    |
| 82 | Design-based staggered rollout| `sp.staggered_rollout`         | `staggered, i() t() g() estimand(simple cohort calendar eventstudy)` |
| 83 | LP-DiD event study            | `sp.lp_did`                    | `lpdid, unit() time() treat() pre_window(2) post_window(3)`  |
| 84 | BJS pre-treatment leads       | `sp.did_imputation`            | `did_imputation y unit time g, pretrends(3) horizons(0/3) cluster(unit)` |
| 85 | Dynamic TWFE event study      | `sp.event_study`               | `reghdfe y <rel-time dummies>, absorb(unit time) vce(cluster unit)`      |
| 86 | fect counterfactual estimators | `sp.fect(method="fe"/"ife"/"mc")` | `fect Y, treat(D) unit(id) time(time) cov(X1 X2) method() force(two-way) tol(1e-12) maxiterations(20000)` (fect_stata from GitHub, local ado path) |
| 87 | interflex marginal effects     | `sp.interflex(estimator=...)`  | `interflex Y D X Z1, type(linear|binning|kernel) vce(robust) neval(5) nbins(3) bw(1)` (SSC) |

### Modules **without** a materialized Stata JSON

Thirteen of the 81 Python modules have no Stata artifact.
`compare.py::STATA_SKIP_REASON` records the exact reason and the 3-way table
prints it explicitly. Every reason was re-measured on 2026-08-06 against a
licensed Stata 18 runtime with SSC reachable, so none of them rests on a
stale "not installed here" claim.

**No Stata implementation exists** (verified 2026-08-06 --
`ssc describe` returns `r(601)` and the command does not resolve locally):

- **77 ddd** — the Ortiz-Villavicencio/Sant'Anna triple-difference estimator
  ships only as the R package `triplediff`.
- **79 didff** — the Roth-Sant'Anna functional-form test ships only as the R
  package `didFF` (GitHub, not CRAN).
- **80 contdid** — the CGS continuous-treatment estimator ships only as the R
  package `contdid` (GitHub, not CRAN).
- **70 policy_tree** — no Stata command, official or user-written, solves the
  Athey-Wager welfare objective over a supplied doubly-robust score matrix.
- **72 tmle** — `eltmle` wraps the same R package rather than providing an
  independent implementation, so a Stata row would re-measure the R reference
  through a shell rather than cross-validate it.

**Not an external estimator** — these two modules check exact identities
against a base-R recomputation, so there is nothing for a third language to
disagree about:

- **68 demean_within** — the within (mean-deviation) transformation.
- **69 balance_panel** — the `counts == n_periods` row filter.

**Estimator agrees, convention does not** — a Stata command exists and was
run, but its estimand or variance convention is not like-for-like:

- **67 panel_glm** — `logit y x1 x2 i.id` and `ppmlhdfe y x1 x2, absorb(id)`
  reproduce the `fixest::feglm` / `fepois` point estimates to rel `1.8e-9`
  and `1e-16`, but no `vce()` setting reproduces fixest's standard errors:
  `vce(robust)`, `vce(cluster id)` and `vce(unadjusted)` land 0.6%, 21% and
  4% away on the logit slopes and 24-42% away on the Poisson ones. The
  module's registered `rel_se` budget is `1e-6`, so a Stata SE column here
  would record a variance-convention argument rather than an estimator check.
  Module 37 already carries the clean `ppmlhdfe` bridge.
- **65 spatial** / **66 spatial_gmm** — `spregress` and `spregress, gs2sls`
  are the natural analogs of `spatialreg::lagsarlm`/`errorsarlm` and
  `stsls`/`GMerrorsar`, but they follow distinct ML and instrument/moment
  conventions.
- **18 augsynth** — local `allsynth` is a candidate bias-corrected SCM
  reference, but its ridge de-biaser rejects the Basque outcome-only fixture
  with 16 controls and 15 pre-period predictors because it requires at least
  `K + 2` control units. A feasible California probe also follows a distinct
  `allsynth` bias-correction convention rather than the R `augsynth` estimand.
- **19 gsynth** — Xu's `fect_stata` selects `r=1` and reports ATT `0.679854`
  under `fect`'s convention while the R/Python `gsynth` headline is
  `-0.324171`; an option grid over `force(two-way/unit/time/none)` does not
  recover the R convention.

**Runtime version** — the candidate reference exists but not in this runtime:

- **13 causal_forest** — Stata 19's official `cate` is the candidate
  causal-forest/AIPW reference; the verified runtime here is Stata 18 and
  `which cate` fails.

### Modules closed in the second 2026-08-06 pass

Three more, after the py<->Stata budget contract was added and the remaining
skip reasons were re-measured rather than re-read:

| Module | Stata reference | Worst py-Stata rel | Note |
| --- | --- | ---: | --- |
| 66 spatial_gmm | audited Mata GS2SLS bridge | 7.3e-16 | estimates *and* SEs; `spregress, gs2sls` uses a wider instrument set and lands 1.5e-2 away, so the do-file builds the documented `stsls(W2X=FALSE)` estimator directly |
| 72 tmle | audited Mata bridge | 1.9e-9 est / 1.4e-11 se | Stata ships no TMLE and `eltmle` wraps the same R package, so the do-file implements the published per-arm fluctuation |
| 70 policy_tree | audited Mata search | 1.4e-16 | **depth 1 only** — exact depth-2 needs policytree's incremental search and a heuristic would be worse evidence than an honest two-sided row |
| 68 demean_within | `bysort id: egen mean` | 8.8e-15 | the third implementation exposed a naming bug that had kept 3 of 10 statistics out of the py-R join too |
| 69 balance_panel | distinct-period count | 0 (exact) | |
| 65 spatial | `spregress, ml` | 5.1e-8 | all 14 SAR/SEM/SDM parameters; the old skip reason claimed a "distinct ML/estimand convention" and was simply wrong |
| 67 panel_glm | `logit i.id` / `ppmlhdfe` | 1.8e-9 | point estimates only; no `vce()` setting reproduces fixest's absorbed-FE GLM variance |

### Modules closed in the first 2026-08-06 pass

Seven modules that previously appeared in the list above now carry
materialized artifacts. Three of them had been skipped on the grounds that
the package was "not installed in the verified local runtime", which stopped
being true once `did_multiplegt_old`, `did_multiplegt_dyn` and `cic` were
checked; two more were skipped as "no Stata implementation" when in fact
`did2s` is on SSC and `pretrends` has a maintained Stata port.

| Module | Stata reference | Worst py-Stata rel | Note |
| --- | --- | ---: | --- |
| 78 multiplegt_dyn | `did_multiplegt_dyn` | 2.1e-15 | 10 joined rows across the absorbing and switch-off designs |
| 81 didm | `did_multiplegt_old` | 3.2e-15 | `dynamic_1` deliberately not joined; see the do-file header |
| 82 staggered | `staggered` (SSC) | 3.7e-15 | all 33 rows join, including both the Neyman and adjusted SEs |
| 85 twfe_event_study | `reghdfe` | 5.7e-14 | estimates and SEs three-way machine level since 1.24.0 (sp.event_study applies the same nested-K rule) |
| 86 fect | `fect` (fect_stata, GitHub) | 1e-9 (fe / mc headline), 1.5e-7 (ife headline) | all three outcome models on one staggered panel; fect_stata's own EM stopping rule leaves the ife fixed point at ~1e-7 while R/Python run the same iteration path to 1e-10 |
| 73 did2s | `did2s` | 2.4e-12 | Stata SE lands on R's, localising the SE gap to a StatsPAI default |
| 71 dml_family | `ddml` | 6.9e-7 | shared fold partition via `foldvar()`; PLIV gap is ddml's second-stage intercept |
| 75 stacked | hand-built stack + `reghdfe` | 7.1e-13 | three independent stack constructions agree; SEs differ by a constant dof factor |
| 76 pretrends | `pretrends` | 5.1e-4 | inside the registered 1e-3 budget; the closed-form LR row agrees to 1e-15 |
| 74 cic | `cic` (`discrete_ci`) | 6.0e-3 | 8 of 9 deciles bit-identical; `qte_50` and the ATT are a documented tie-break gap |

`08_dml`, `31_dfl`, `32_rif`, `53_cr2`, `54_twoway_cluster`, and
`56_multiway_cluster` are deliberately labelled audited Stata/Mata algorithm
bridges rather than packaged-command references: `08_dml` implements the
deterministic linear-nuisance DML2 PLR score rather than treating `ddml` as
canonical for the published DoubleML R algorithm, `31_dfl` implements the DFL
logit reweighting algebra directly, `32_rif` avoids a nonbaseline `rifhdreg`
install, `53_cr2` implements clubSandwich-style CR2/CR3 because Stata's
built-in clustered covariance is CR1, and `54`/`56` implement the
CGM/sandwich multiway-cluster convention directly while keeping `reghdfe`
SEs as diagnostic convention rows.
## Running

End-to-end run for a single module (assumes the matching
`tests/r_parity/NN_<name>.py` has already produced the CSV in
`tests/r_parity/data/`):

```bash
cd tests/stata_parity
/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp -b -q do 11_psm.do
python3 ../r_parity/compare.py
```

Run everything:

```bash
cd tests/stata_parity
for dofile in [0-9][0-9]_*.do; do
  /Applications/Stata/StataMP.app/Contents/MacOS/stata-mp -b -q do "${dofile}"
done
python3 ../r_parity/compare.py
```

The same critical Stata smoke path is available through pytest:

```bash
pytest tests/test_parity_runtime.py -m external_parity_runtime --no-cov
```

## Tier A fixture lock

The Stata-side `.do` files, shared helper, golden `_Stata.json` outputs,
environment notes, and reproduction report are included in
[`../r_parity/TIER_A_FIXTURE_LOCK.json`](../r_parity/TIER_A_FIXTURE_LOCK.json).
The fast contract suite verifies the lock without requiring a local
Stata license:

```bash
python scripts/tier_a_fixture_lock.py
pytest -o addopts='' tests/test_parity_harness_contract.py
```

After an intentional Stata fixture refresh, review the JSON/table diff
and then run `python scripts/tier_a_fixture_lock.py --write` so the
hash-level fixture contract moves with the audited evidence.

## Stata environment

- **Edition tested**: Stata 18 MP on the current parity machine.
  None of the 53 materialized modules trip the matrix limit.
- **`set type double`** is forced in `_common.do` so
  `import delimited` reads the CSV bytes at full IEEE-754 precision;
  without it, Stata's float default would cost 4-5 orders of
  magnitude in parity (1e-12 → 1e-8 on OLS).
- **JSON writer**: file-based (under `logs/<module>.rows.tmp`) rather
  than Mata-resident, because several Stata commands (`rdrobust`,
  `csdid`, `sdid`, others) call `mata mata clear` internally and
  would wipe a Mata accumulator mid-run.

## Required SSC / community packages

```stata
ssc install ivreg2 ranktest csdid drdid did_imputation eventstudyinteract \
    jwdid hdfe synth rdrobust rddensity honestdid bacondecomp \
    sfcross sfpanel sensemakr avar ppmlhdfe paramed evalue
```

`reghdfe`, `sdid`, `psmatch2`, and `oaxaca` were already on the test
machine; `mixed`, `melogit`, `xtfrontier`, `frontier`, `regress`,
`ivregress`, `teffects psmatch`, `xtreg`, `var`, `arima`, `qreg`,
`tobit`, `nbreg`, `heckman`, `mlogit`, `ologit`, `clogit`, `probit`,
`oprobit`, `xtabond`, and `newey` are Stata built-ins.

## How the JSS paper uses this

[`Paper-JSS/manuscript/sections/appendix.tex`](../../Paper-JSS/manuscript/sections/appendix.tex)
`\input`s [`manuscript/tables/appendix_b_parity.tex`](../../Paper-JSS/manuscript/tables/appendix_b_parity.tex),
which is a copy of `tests/r_parity/results/parity_table_3way.tex`
refreshed by `compare.py`. Re-running `compare.py` after any module
change is sufficient to keep the appendix in sync.
