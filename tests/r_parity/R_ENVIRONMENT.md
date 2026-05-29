# R / Stata reference environment for the Track A parity harness

> JSS reproducibility manifest (paper Section 8, "Reproducibility of
> this paper"). This file pins the exact reference-language software
> environment under which the committed golden artefacts in
> `tests/r_parity/results/*_R.json` and
> `tests/stata_parity/results/*_Stata.json` were produced and *verified
> to reproduce bit-for-bit*. A reviewer who observes a residual gap
> against their own run should first diff their `sessionInfo()` against
> the versions below: a package-version delta is the most common
> explanation, and `tests/r_parity/verify_reproduce.py` regenerates this
> evidence on demand.

The machine-readable form of the R package set is
[`renv.lock`](renv.lock); every `_R.json` additionally carries an inline
`provenance` block (R version, platform, OS, and the version of every
attached/loaded package) emitted by
[`_common.R`](./_common.R)`::.r_provenance`, so each golden value is
self-describing without reference to this file.

## How to regenerate and verify

```bash
# Re-run every R reference into a staging dir and diff each statistic
# against the committed golden JSON at a 1e-9 reproducibility tolerance:
python tests/r_parity/verify_reproduce.py            # -> results/REPRODUCIBILITY_REPORT.md

# A subset:
python tests/r_parity/verify_reproduce.py 01_ols 02_iv 03_hdfe
```

The verifier never overwrites the committed JSON: it writes to
`results/_repro_check/` (git-ignored) and reports the worst per-module
relative difference. A non-zero exit means a module drifted and the
maintainer must explain it (package upgrade, RNG, BLAS) before the
golden value is refreshed.

## R

| Field | Value |
|---|---|
| R version | R 4.5.2 (2025-10-31) |
| Platform | aarch64-apple-darwin20 |
| OS | macOS Tahoe 26.5 |
| BLAS | Accelerate `vecLib` (`.../vecLib.framework/.../libBLAS.dylib`) |
| LAPACK | R-bundled `libRlapack.dylib` (R 4.5-arm64) |

> **BLAS note.** The reference run uses Apple's Accelerate `vecLib`
> BLAS. For the closed-form estimators (OLS, 2SLS, HDFE, cluster SE)
> the parity harness verifies machine-precision agreement, so the BLAS
> choice does not affect the headline point estimates; it can perturb
> the last 1–2 ULPs of iterative MLE fits, which is far inside the
> `compare.py` tolerance budget. A reviewer on reference (OpenBLAS)
> BLAS who sees a >1e-9 difference on an iterative module should treat
> it as a BLAS artefact, not an algorithm gap.

### Reference packages (canonical R implementations)

| Package | Version | Module(s) |
|---|---|---|
| `AER` | 1.2.16 | 02 (2SLS `ivreg`) |
| `augsynth` | 0.2.0 | 18 (augmented SCM) |
| `bacondecomp` | 0.1.1 | 20 (Goodman–Bacon) |
| `did` | 2.3.0 | 04 (Callaway–Sant'Anna) |
| `didimputation` | 0.5.1 | 16 (BJS imputation) |
| `DoubleML` | 1.0.2 | 08 (DML PLR) |
| `etwfe` | 0.6.2 | 17 (Wooldridge ETWFE) |
| `fixest` | 0.14.0 | 03/15 (HDFE, cluster), 05 (`sunab`) |
| `frontier` | 1.1.8 | 28 (stochastic frontier) |
| `grf` | 2.6.1 | 13 (causal forest) |
| `gsynth` | 1.4.0 | 19 (generalized SCM) |
| `HonestDiD` | 0.2.8 | 10/21 (honest DiD) |
| `lme4` | 2.0.1 | 25 (linear mixed model), 26/27 (GLMM) |
| `lpirfs` | 0.2.5 | 34 (local projections) |
| `MatchIt` | 4.7.2 | 11 (1:1 NN PSM) |
| `mediation` | 4.5.1 | 36 (causal mediation) |
| `oaxaca` | 0.1.5 | 30 (Blinder–Oaxaca) |
| `plm` | 2.6.7 | 35 (panel FE/RE), 50 (`pgmm` secondary ref) |
| `quantreg` | 6.1 | 40 (quantile regression) |
| `rddensity` | 2.6 | 09 (CJM density) |
| `rdrobust` | 3.0.0 | 06 (RD bias-corrected) |
| `sandwich` | 3.1.1 | 01/14 (HC1, cluster vcov) |
| `survival` | 3.8.3 | 24 (Cox PH) |
| `Synth` | 1.1.10 | 07 (classical SCM) |
| `synthdid` | 0.0.9 | 12 (synthetic DiD) |
| `vars` | 1.6.1 | 33 (VAR) |
| `jsonlite` | 2.0.0 | (harness I/O — full-precision result serialisation) |

## Stata

| Field | Value |
|---|---|
| Version | Stata 18 |
| Edition | MP |
| Platform | Unix Mac (Apple Silicon) |
| Executable date | 07 Jun 2023 |

The selected `Stata` bridge ships `Stata` 18 do-files and frozen JSON
outputs for the modules listed in
[`../stata_parity/README.md`](../stata_parity/README.md). Executing the
do-files requires a separate `Stata` licence; the committed JSON
artefacts and the inline provenance let a reviewer audit the numbers
without one. User-contributed `ado` commands (`csdid`, `reghdfe`,
`rdrobust`, `rddensity`, `sdid`, `did_imputation`, `jwdid`,
`bacondecomp`, `eventstudyinteract`, `honestdid`, `frontier`,
`sensemakr`) are installed from SSC / the authors' repositories; their
install lines are recorded in `../stata_parity/_common.do`.

---

*Captured 2026-05-29 via `Rscript -e 'sessionInfo()'` and Stata
`c(stata_version)` on the maintainer's macOS arm64 workstation. Refresh
this file whenever the reference environment changes; the per-`_R.json`
`provenance` block is the authoritative per-result record.*
