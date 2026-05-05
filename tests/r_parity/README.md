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
├── compare.py            # joins per-module JSONs, emits parity_table.{md,tex}
├── NN_<method>.py        # one Python script per module
├── NN_<method>.R         # the matching R script
├── data/                 # CSVs dumped from sp.datasets so R sees same bytes
└── results/
    ├── NN_<method>_{py,R}.json   # full-precision per-module results
    ├── parity_table.md           # human-readable rollup
    └── parity_table.tex          # LaTeX longtable for Appendix B
```

## Modules (36)

| # | Module | StatsPAI | R reference |
| --- | --- | --- | --- |
| 01 | OLS + HC1 SE | `sp.regress` | `lm` + `sandwich::vcovHC` |
| 02 | 2SLS + HC1 SE | `sp.ivreg` | `AER::ivreg` |
| 03 | HDFE 2-way FE | `sp.fast.feols` | `fixest::feols` |
| 04 | CS-DiD simple ATT | `sp.callaway_santanna` | `did::att_gt` + `aggte` |
| 05 | Sun-Abraham event study | `sp.sun_abraham` | `fixest::sunab` |
| 06 | RD CCT bias-corrected | `sp.rdrobust` | `rdrobust::rdrobust` |
| 07 | Classical SCM | `sp.synth("classic")` | `Synth::synth` |
| 08 | DML PLR | `sp.dml("plr")` | `DoubleML::DoubleMLPLR` |
| 09 | RD density (CJM) | `sp.rddensity` | `rddensity::rddensity` |
| 10 | Honest DiD smoothness | `sp.honest_did` | `HonestDiD::createSensitivityResults` |
| 11 | PSM 1:1 NN | `sp.psm` | `MatchIt::matchit` |
| 12 | Synthetic DID | `sp.synth("sdid")` | `synthdid::synthdid_estimate` |
| 13 | Causal forest (AIPW) | `sp.causal_forest` | `grf::causal_forest` |
| 14 | OLS + cluster SE | `sp.regress(cluster=)` | `lm` + `sandwich::vcovCL` |
| 15 | HDFE + cluster SE | `sp.fast.feols(cr1)` | `fixest::feols(cluster=)` |
| 16 | BJS imputation | `sp.did_imputation` | `didimputation::did_imputation` |
| 17 | Wooldridge ETWFE | `sp.etwfe` | `etwfe::etwfe` |
| 18 | Augmented SCM | `sp.synth("augmented")` | `augsynth::augsynth` |
| 19 | Generalized SCM | `sp.synth("gsynth")` | `gsynth::gsynth` |
| 20 | Goodman--Bacon decomp | `sp.bacon_decomposition` | `bacondecomp::bacon` |
| 21 | Honest DiD relative-mags | `sp.honest_did("relative")` | `HonestDiD::createSensitivityResults_relativeMagnitudes` |
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
| 34 | Local projections | `sp.local_projections` | `lpirfs::lp_lin` |
| 35 | Panel FE/RE/Hausman | `sp.panel` | `plm::plm` + `plm::phtest` |
| 36 | Causal mediation | `sp.mediation` | `mediation::mediate` |

## Running

End-to-end run for a single module:

```bash
cd tests/r_parity
python3 11_psm.py     # writes data/11_psm.csv + results/11_psm_py.json
Rscript 11_psm.R      # reads same CSV + writes results/11_psm_R.json
python3 compare.py    # refresh parity_table.{md,tex}
```

Run everything (36 modules):

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
`survival`, `plm`, `Matching`.

GitHub:

- `synthdid` (`remotes::install_github("synth-inference/synthdid")`)
- `augsynth` (`remotes::install_github("ebenmichael/augsynth")`)

## How the JSS paper uses this

[`Paper-JSS/manuscript/sections/appendix.tex`](../../Paper-JSS/manuscript/sections/appendix.tex)
`\input`s `manuscript/tables/appendix_b_parity.tex`, which is a
copy of `tests/r_parity/results/parity_table.tex` refreshed by
`compare.py`. Re-running `compare.py` after any module change is
sufficient to keep the appendix in sync; the build step in
`Paper-JSS/replication/Makefile` should `cp` the table back into
`manuscript/tables/`.
