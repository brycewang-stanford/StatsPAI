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

## Materialized Stata golden modules (61 of 64 Python modules)

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

### Modules **without** a materialized Stata JSON

These have no portable materialized Stata JSON yet. `compare.py::STATA_SKIP_REASON`
records the exact reason and the 3-way table prints it explicitly:

- **13 causal forest** — Stata 19's official `cate` is the candidate
  causal-forest/AIPW reference, but the verified runtime here is Stata 18 and
  `which cate` fails.
- **18 augsynth** — local `allsynth` is a candidate bias-corrected SCM
  reference, but its ridge de-biaser rejects the Basque outcome-only fixture
  with 16 controls and 15 pre-period predictors because it requires at least
  `K + 2` control units. A feasible California probe also follows a distinct
  `allsynth` bias-correction convention rather than the R `augsynth` estimand,
  so no like-for-like bridge is materialized.
- **19 gsynth** — Xu's `fect_stata` is the candidate generalized-SCM route
  and can be installed in a temporary Stata 18 ado path, but a two-way IFE
  probe selects `r=1` and reports ATT `0.679854` under `fect`'s convention,
  while the R/Python `gsynth` headline is `-0.324171`. An option grid over
  `force(two-way/unit/time/none)` does not recover the R `gsynth` convention,
  so no like-for-like Stata bridge is materialized.

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
