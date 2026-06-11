# `tests/r_parity/` — cross-language parity harness against R

This directory contains the **StatsPAI ↔ R numerical-parity harness**:
each module pair runs the same calibrated replica on both sides,
dumps a full-precision JSON result, and lets `compare.py` produce a
per-module headline table for the JSS paper's Appendix B.

It complements:

- [`tests/reference_parity/`](../reference_parity/) — pure-Python
  pytest tests that verify `sp.*` recovers the *true* parameter on
  deterministic DGPs (no R involved).
- [`tests/external_parity/`](../external_parity/) — pytest tests
  that pin replica outputs to constants documented in
  [`tests/external_parity/PUBLISHED_REFERENCE_VALUES.md`](../external_parity/PUBLISHED_REFERENCE_VALUES.md).

## Layout

```
tests/r_parity/
├── _common.py            # shared scaffolding for the Python side
├── _common.R             # shared scaffolding for the R side
├── compare.py            # joins JSONs, emits 2-way and 3-way parity tables
├── NN_<method>.py        # one Python script per module
├── NN_<method>.R         # the matching R script when a reference is materialized
├── data/                 # CSVs dumped from sp.datasets so R sees same bytes
└── results/
    ├── NN_<method>_{py,R}.json   # full-precision per-module results
    ├── parity_table.md           # human-readable R rollup
    ├── parity_table_3way.md      # human-readable R + Stata rollup
    └── parity_table_3way.tex     # LaTeX longtable for Appendix B
```

Historical verification worklog (not the current source-snapshot audit):
[`PARITY_TEST_WORKLOG_2026-05-29.md`](PARITY_TEST_WORKLOG_2026-05-29.md).

## Modules (64 materialized StatsPAI--R rows)

Module `50_xtabond` is now materialized on the R side through
`plm::pgmm`, so all modules 01--64 have committed StatsPAI--R rows.
The native Python rows used by the loose/reference-bridge audit are
modules 01--52; modules 53--56 are additional R-only
robust/cluster-SE parity rows, and modules 57--64 extend the GLM /
IV / system / limited-dependent-variable coverage (logit, Poisson,
LIML, SUR, beta regression, truncated regression, ZIP, ZINB).

| # | Module | StatsPAI | R / reference side |
| --- | --- | --- | --- |
| 01 | OLS + HC1 SE | `sp.regress` | `lm` + `sandwich::vcovHC` |
| 02 | 2SLS + HC1 SE | `sp.ivreg` | `AER::ivreg` |
| 03 | HDFE 2-way FE | `sp.fast.feols` | `fixest::feols` |
| 04 | CS-DiD simple ATT | `sp.callaway_santanna` | `did::att_gt` + `aggte` |
| 05 | Sun-Abraham event study | `sp.sun_abraham` | `fixest::sunab` |
| 06 | RD CCT bias-corrected | `sp.rdrobust` | `rdrobust::rdrobust` |
| 07 | Classical SCM | `sp.synth(method="classic", backend="native")` | `Synth::synth` |
| 08 | DML PLR | `sp.dml("plr")` | `DoubleML::DoubleMLPLR` |
| 09 | RD density (CJM) | `sp.rddensity(backend="native")` | `rddensity::rddensity` |
| 10 | Honest DiD smoothness | `sp.honest_did` | `HonestDiD::createSensitivityResults` |
| 11 | PSM 1:1 NN | `sp.psm` | `MatchIt::matchit` |
| 12 | Synthetic DID | `sp.sdid(backend="native")` | `synthdid::synthdid_estimate` |
| 13 | Causal forest (AIPW) | `sp.causal_forest` | `grf::causal_forest` |
| 14 | OLS + cluster SE | `sp.regress(cluster=)` | `lm` + `sandwich::vcovCL` |
| 15 | HDFE + cluster SE | `sp.fast.feols(cr1)` | `fixest::feols(cluster=)` |
| 16 | BJS imputation | `sp.did_imputation` | `didimputation::did_imputation` |
| 17 | Wooldridge ETWFE | `sp.etwfe` + `sp.etwfe_emfx` | `etwfe::etwfe` + `emfx` |
| 18 | Augmented SCM | `sp.augsynth(backend="native")` | `augsynth::augsynth` |
| 19 | Generalized SCM | `sp.gsynth(backend="native")` | `gsynth::gsynth` |
| 20 | Goodman--Bacon decomp | `sp.bacon_decomposition` | `bacondecomp::bacon` |
| 21 | Honest DiD relative-mags | `sp.honest_did(method="relative_magnitude", backend="honestdid")` | `HonestDiD::createSensitivityResults_relativeMagnitudes` |
| 22 | sensemakr | `sp.sensemakr` | `sensemakr::sensemakr` |
| 23 | E-value | `sp.evalue` | `EValue::evalues.RR` |
| 24 | Cox proportional hazards | `sp.survival.cox` | `survival::coxph` |
| 25 | LMM | `sp.mixed` | `lme4::lmer` |
| 26 | GLMM logit (Laplace) | `sp.melogit` | `lme4::glmer` |
| 27 | GLMM AGHQ (n=8) | `sp.melogit(nAGQ=8)` | `lme4::glmer(nAGQ=8)` |
| 28 | SFA cross-section | `sp.frontier` | `sfaR::sfacross` |
| 29 | Panel SFA Pitt-Lee | `sp.xtfrontier` | `frontier::sfa` |
| 30 | Blinder--Oaxaca | `sp.decompose("oaxaca")` | `oaxaca::oaxaca` |
| 31 | DFL reweighting | `sp.decompose("dfl")` | `ddecompose::dfl_decompose` |
| 32 | RIF / UQR (median) | `sp.decomposition.rif_decomposition` | `dineq::rif` + manual OLS |
| 33 | VAR | `sp.var` | `vars::VAR` |
| 34 | Local projections | `sp.local_projections(..., identification="lpirfs_cholesky")` | `lpirfs::lp_lin` |
| 35 | Panel FE/RE/Hausman | `sp.panel` | `plm::plm` + `plm::phtest` |
| 36 | Causal mediation | `sp.mediation` | `mediation::mediate` |
| 37 | PPML + HDFE | `sp.ppmlhdfe` | `fixest::fepois` |
| 38 | DR-DID (Sant'Anna-Zhao) | `sp.drdid` | `DRDID::drdid_imp_panel` |
| 39 | ARIMA(2,0,0) | `sp.arima` | `stats::arima` |
| 40 | Quantile regression | `sp.qreg` | `quantreg::rq` |
| 41 | Tobit | `sp.tobit` | `censReg::censReg` |
| 42 | Negative binomial | `sp.nbreg` | `MASS::glm.nb` |
| 43 | Heckman selection | `sp.heckman` | `sampleSelection::heckit` |
| 44 | Multinomial logit | `sp.mlogit` | `nnet::multinom` |
| 45 | Ordered logit | `sp.ologit` | `MASS::polr(method="logistic")` |
| 46 | Conditional logit | `sp.clogit` | `survival::clogit` |
| 47 | PPML + 3-way HDFE | `sp.ppmlhdfe` | `fixest::fepois` |
| 48 | Binary probit | `sp.probit` | `stats::glm(family=binomial("probit"))` |
| 49 | Ordered probit | `sp.oprobit` | `MASS::polr(method="probit")` |
| 50 | Arellano--Bond GMM | `sp.xtabond` | `plm::pgmm` |
| 51 | Newey-West HAC OLS | `sp.regress(robust="hac")` | `sandwich::NeweyWest` |
| 52 | Identified classical SCM DGP | `sp.synth(method="classic", backend="native")` | `Synth::synth` |
| 53 | Cluster-robust CR2 SE (+ CR3 jackknife) | `sp.cr2_se` | `clubSandwich::vcovCR(type="CR2"/"CR3")` |
| 54 | Two-way cluster-robust SE | `sp.twoway_cluster` | `sandwich::vcovCL(cluster=~g1+g2)` |
| 55 | HC2 / HC3 robust SE | `sp.regress` | `sandwich::vcovHC(type="HC2"/"HC3")` |
| 56 | Three-way cluster-robust SE | `sp.multiway_cluster_vcov` | `sandwich::vcovCL(cluster=~g1+g2+g3)` |
| 57 | Binary logit | `sp.logit` | `stats::glm(family=binomial("logit"))` |
| 58 | Poisson ML (no FE) | `sp.poisson` | `stats::glm(family=poisson())` |
| 59 | LIML k-class IV | `sp.liml` | `ivmodel::LIML` |
| 60 | SUR one-step FGLS | `sp.sureg` | `systemfit::systemfit(method="SUR", noDfCor)` |
| 61 | Beta regression | `sp.betareg` | `betareg::betareg(link.phi="log")` |
| 62 | Truncated regression | `sp.truncreg` | `truncreg::truncreg(method="NR")` |
| 63 | Zero-inflated Poisson | `sp.zip_model` | `pscl::zeroinfl(dist="poisson")` |
| 64 | Zero-inflated NB | `sp.zinb` | `pscl::zeroinfl(dist="negbin")` |

## Running

End-to-end run for a single module:

```bash
cd tests/r_parity
python3 11_psm.py     # writes data/11_psm.csv + results/11_psm_py.json
Rscript 11_psm.R      # reads same CSV + writes results/11_psm_R.json
python3 compare.py    # refresh parity tables
```

Run all materialized R modules:

```bash
cd tests/r_parity
for py in [0-9][0-9]_*.py; do
  n="${py%.py}"
  R="${n}.R"
  test -f "${R}" || continue
  python3 "${py}" && Rscript "${R}"
done
python3 compare.py
```

To execute the external runtime smoke tests from pytest on a machine
with R/Stata installed, run:

```bash
pytest tests/test_parity_runtime.py -m external_parity_runtime --no-cov
```

## Tier A fixture lock

The committed Tier A fixture set is hash-locked in
[`TIER_A_FIXTURE_LOCK.json`](TIER_A_FIXTURE_LOCK.json). The lock covers
the parity scripts, shared helpers, input CSVs, golden JSONs, rendered
Appendix B tables, `renv.lock`, and the R/Stata reference-environment
files. It is checked by the fast pytest contract suite, so a fixture can
no longer drift merely because `compare.py` was re-run.

Verify the lock without external R/Stata software:

```bash
python scripts/tier_a_fixture_lock.py
```

After intentionally changing a Tier A fixture, first regenerate the
materialized evidence (`NN_*.py` / `NN_*.R` / Stata `.do` as needed,
then `python tests/r_parity/compare.py`). Review the resulting diff,
then refresh the lock explicitly:

```bash
python scripts/tier_a_fixture_lock.py --write
pytest -o addopts='' tests/test_parity_harness_contract.py
```

## Tolerance budget (pre-registered)

Lives in [`compare.py::TOLERANCES`](compare.py); single source of
truth for the verdict column.

- closed-form estimators (OLS, 2SLS, HDFE): `rel_diff < 1e-6`
- iterative / cross-fit estimators: normally `rel_diff < 1e-3`
- stochastic or solver-sensitive rows: method-specific tolerances with
  the source of residual noise recorded in `extra`
- convention gaps are reported separately and are not ordinary parity
  passes
- Honest-DiD CI bounds: `abs_diff < 0.05`

## R dependencies

CRAN: `AER`, `fixest`, `did`, `HonestDiD`, `Synth`, `rdrobust`,
`rddensity`, `DoubleML`, `mlr3`, `mlr3learners`, `MatchIt`,
`sandwich`, `bacondecomp`, `didimputation`, `EValue`, `sensemakr`,
`lme4`, `oaxaca`, `sfaR`, `frontier`, `etwfe`, `gsynth`,
`ddecompose`, `dineq`, `vars`, `lpirfs`, `mediation`,
`survival`, `plm`, `Matching`, `DRDID`, `forecast`, `quantreg`,
`censReg`, `MASS`, `sampleSelection`, `nnet`, `lmtest`.

GitHub:

- `synthdid` (`remotes::install_github("synth-inference/synthdid")`)
- `augsynth` (`remotes::install_github("ebenmichael/augsynth")`)

## How the JSS paper uses this

[`Paper-JSS/manuscript/sections/appendix.tex`](../../Paper-JSS/manuscript/sections/appendix.tex)
`\input`s `manuscript/tables/appendix_b_parity.tex`, which is a
copy of `tests/r_parity/results/parity_table_3way.tex` refreshed by
`compare.py`. Re-running `compare.py` after any module change is
sufficient to keep the appendix in sync; the build step in
`Paper-JSS/replication/Makefile` should `cp` the table back into
`manuscript/tables/`.
