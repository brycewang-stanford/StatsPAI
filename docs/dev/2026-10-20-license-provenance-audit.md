# License provenance audit of self-declared ports (2026-10-20)

StatsPAI is MIT-licensed. This audit lists every module under `src/statspai`
whose own docstrings or comments say it ports, transcribes or follows the
source of an external implementation, and records the license of that
implementation from its published metadata. It is a factual inventory, not a
legal opinion. The one working rule applied is the one in the task brief:
copyleft-derived code in an MIT package needs a clean-room rewrite or a
relicensing review.

Audited tree: `origin/main` at `6588902cf`.

## Summary

| Class | Meaning | Rows |
| --- | --- | --- |
| A | The text says the code was ported, transcribed or translated from the reference's source, or names the reference's internal (non-exported) functions, local variables or code branches. | 50 |
| B | The text says the module reproduces documented behaviour, conventions or outputs (black box), or states that the reference's source was not read. | 22 (listed below, not exhaustive) |
| C | The wording does not settle whether source was read (for example "mirrors R `pkg:::fn`" for a short formula that is also in the paper). | 13 |

| Risk | Rule | Rows |
| --- | --- | --- |
| HIGH | A + copyleft (GPL / LGPL / AGPL) | 27 |
| MEDIUM | C + copyleft, or A + no license stated / proprietary | 17 |
| LOW | A + permissive, all B, and C + no license stated / proprietary (a combination the brief does not list; placed one level below A + no license) | 41 |

`THIRD_PARTY_NOTICES.md` currently carries one notice (HonestDiD, MIT). The
permissive references under LOW below (MIT, BSD-3, Apache-2.0) also ask for
their copyright notice to travel with derived code, so those rows carry an
attribution action even though their risk is low.

## Method

1. **Search.** `grep -rniE` over `src/statspai/**/*.py` for: `port of`,
   `ported from`, `porting`, `translation of`, `translated from`,
   `line[- ]by[- ]line`, `line-for-line`, `verbatim`, `transcri(b|pt)`,
   `step for step`, `term by term`, `faithful`, `operation-for-operation`,
   `reverse-engineer`, `mirrors (R|Stata|...)`, `R source`, `package source`,
   `source code`, `re-implementation of`, `Mata`, `.ado`, R internal names
   (`pkg:::fn`), and docstrings that consist of an external function name
   (`"""R ``fn``."""`). About 400 hits; `agent/_translation/` (it translates
   user scripts, not reference code), wording such as "verbatim from
   paper.bib", and words such as "transport" / "report" were set aside. Each
   remaining hit was read in context (a few lines around it, plus the module
   docstring) and classified A / B / C.
2. **Licenses.** Read only from metadata, never from memory:
   - R: `packageDescription(pkg)$License` on the local R 4.5.2 library
     (written `R-local` below); CRAN `DESCRIPTION` via curl for packages not
     installed (`CRAN`); `DESCRIPTION` on GitHub for packages not on CRAN
     (`GH-DESCRIPTION`). Base R (`stats`, LINPACK `dqrdc2`): R's `license()`
     ("GNU General Public License, either Version 2 ... or Version 3").
   - Python: `https://pypi.org/pypi/<pkg>/json` (`PyPI`), fields `license`,
     `license_expression` and classifiers.
   - GitHub repositories: `api.github.com/repos/<owner>/<repo>` `license`
     field (`GH-API`).
   - Stata ado: `grep -iE 'licen[cs]e|GPL|copyright|\(c\)|all rights'` on the
     installed `.ado` / `.sthlp` only (`ado-header`, `sthlp`); no other part
     of any ado file was read for this audit. StataCorp's own ado files are
     proprietary Stata components.
3. **History.** `git log --follow --diff-filter=A` for the date a file
   entered, `git log -S"<evidence phrase>"` for the commit that introduced the
   port wording, and the commit message of that commit (does it say the source
   was read).
4. No GPL source was opened during this audit.

## HIGH: A + copyleft

Scope: *module* (the file is the port), *functions* (named functions inside a
larger file), *helper* (a short numeric routine).

| # | Module(s) | Reference | License (source) | Class / scope | Evidence (file:line) | Entered / port wording introduced |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | `synth/scpi.py`, `synth/_scpi_inference.py`, `synth/_scpi_solvers.py` | R `scpi` 4.0.1 (Cattaneo, Feng, Palomba, Titiunik) | GPL-3 (R-local). Python `scpi-pkg` 4.0.0 by the same group: GPL-3.0-only (PyPI) | A / module | `_scpi_inference.py:1` "port of the body of R ``scpi``"; `:29` `"""R ``regularize.w``."""`; `:57` `local.geom`; `:126` `df.EST`; `scpi.py:2`, `:421` "Port of the single-treated-unit path of R ``scpi``"; `:730` `w.constr.OBJ`; `_scpi_solvers.py:257` `shrinkage.EST`; `docs/dev/campaign_phase3/did_synth_scpi.md:76` "A full port of the single-treated-unit path" | `scpi.py` 2026-04-16 `b66f3122`; rewritten as a port 2026-09-22 `5eea7922` |
| H2 | `did/_rcs.py` | R `DRDID` (ported from 1.2.3; installed 1.3.0) | GPL-3 (R-local) | A / module | `:9` "ports **all eight** repeated-cross-section estimators of R ``DRDID`` 1.2.3"; `:189` "Port of ``DRDID::drdid_rc``"; `:305`, `:352`, `:404`, `:500`, `:569` | 2026-07-31 `8342af36`, message: "ports the three estimators R did::att_gt(panel=FALSE) dispatches to, confirmed from the source of ..." |
| H3 | `did/callaway_santanna.py` (DR / IPW influence corrections) | R `DRDID` | GPL-3 (R-local) | A / functions | `:1925`, `:1965` "Port of ``DRDID::drdid_panel``'s ``inf.treat.2``, ``inf.cont.2`` and ``inf.cont.3``" (internal variable names); `:2171` "Port of the ``asy.lin.rep.ps %*% M2`` term" | file 2026-04-03; wording 2026-08-11 `1395a086` |
| H4 | `did/_core.py` (`wif`), `did/aggte.py` | R `did` 2.5.1 | GPL-3 (R-local) | A / functions | `_core.py:850` "Port of R ``did:::wif``"; `aggte.py:921` "R ``did:::get_agg_inf_func``" | 2026-08-07 `6bcefece` |
| H5 | `nonparametric/lprobust.py` | R `nprobust` 1.0.0 (Calonico, Cattaneo, Farrell) | GPL-3 (R-local). Python `nprobust` 0.5.0, same authors: GPL-2.0 (PyPI) | A / module | `:371` "Port of ``nprobust``'s ``lprobust.bw``"; `:466` `lpbwselect.mse.dpi`; `:611` `lp.bw.fun`; `:682`, `:816`, `:913`, `:1021` | 2026-08-07 `8a20c3eb`/`31bedd17`; message of `31bedd17`: "lpbwselect.mse.dpi and lprobust.bw are ordinary readable R. The port is from those." |
| H6 | `rd/_cct_bandwidth.py` | R `rdrobust` 4.0.0 (`rdbwselect`) | GPL-3 (R-local). Python `rdrobust` 2.1.1: GPL-3.0-only (PyPI) | A / module | `:3` "A faithful port of ``rdrobust::rdbwselect`` (R package rdrobust 4.0.0)" | 2026-08-01 `82f65733` |
| H7 | `rd/_rdplot_core.py` | R `rdrobust::rdplot` 4.0.0; R `lpdensity` 2.5 | rdrobust GPL-3; lpdensity GPL-2 (R-local). Python `lpdensity` 3.0.0: GPL-3.0-only (PyPI) | A / module | `:1` "ported from R ``rdrobust::rdplot``"; `:3` "reproduces ... ``rdplot`` line for line"; `:477` `"""R ``lpdensity(data, grid, bw, ...)``` | 2026-09-22 `5eea7922` |
| H8 | `rd/_rdhonest.py`, `rd/bias_aware.py` (FRD path), `rd/rd_discrete.py` (`_bme`) | R `RDHonest` 1.0.1.9000 (Armstrong, Kolesar) | GPL-3 (R-local) | A / module + functions | `_rdhonest.py:1` "ported from R ``RDHonest``"; `:84` `RDHonest::CVb`; `:162` `sigmaNN`; `:355` `Moulton`; `:366` `IKBW`; `:454` `OptBW`; `:519` `MROT`; `bias_aware.py:404` "port of RDHonest's FRD path"; `rd_discrete.py:216` "Port of ``RDHonest::RDHonestBME``" | `_rdhonest.py` 2026-08-02 `93dafa6e`; others' wording 2026-09-22 `5eea7922` |
| H9 | `rd/_rd2d_location.py`, `rd/_rd2d_distance.py` | R `rd2d` 1.0.0 (Cattaneo, Titiunik, Yu) | GPL-3 (R-local). Python `rd2d` 1.0.0: GPL-3.0-only (PyPI) | A / module | `_rd2d_location.py:1` "port of R ``rd2d``", `:4-7` "Each function below mirrors one routine ... keeps its order of operations"; `_rd2d_distance.py:1`; `docs/dev/campaign_phase3/r2_rd_open.md:46` "Rebuilt as a line-by-line port of R rd2d 1.0.0" | 2026-09-22 `5eea7922` |
| H10 | `rd/hte.py` | R `rdhte` 0.2.0 | GPL-3 (R-local). Python `rdhte` 0.1.0: GPL-3.0-only (PyPI) | A / functions | `:65` `"""R ``rdhte:::.rdhte_normalize_vce``"""`; `:286` "The Python port of R ``rdhte``" | file 2026-04-16; wording 2026-09-22 `5eea7922` |
| H11 | `rd/rdpower.py` | R `rdpower` 3.0 | GPL-3 (R-local). Python `rdpower` 2.0.0: GPL-3.0-only (PyPI) | A / functions | `:230` "A transcription of ``rdpower:::rdpower.powerNR``", "The guard loops are R's"; `:384` "A transcription of R ``rdpower::rdsampsi(data = ...)``" | file 2026-04-15; wording 2026-09-11 `1a550974` |
| H12 | `diagnostics/rddensity.py` (native path) | R `rddensity` 2.6 | GPL-2 (R-local). Python `rddensity` 3.0: GPL-3.0-only (PyPI) | A / functions | `:170` "StatsPAI's Python port of the default ``rddensity`` ... path"; `:595` `rddensity:::rddensityUnique`; `:691` "Port of ``rdbwdensity(...)``"; `:922` "Port of ``rddensity:::rddensity_fV``" | file 2026-04-03; wording 2026-05-31 `d43f5dba` |
| H13 | `diagnostics/evalue.py` (`evalue_rd`, `_threshold`) | R `EValue` 4.1.4 | GPL-2 (R-local) | A / functions | `:353` "a faithful port of ``EValue::evalues.RD``"; `:625` "port of ``EValue:::threshold``" | file 2026-04-04; wording 2026-06-09 `78560c89` |
| H14 | `mendelian/frontier/raps.py` | R `mr.raps` 0.4.3 | GPL-3 (R-local) | A / module | `:10` "A port of the authors' R package ``mr.raps`` ..., variant for variant"; `:133` "exactly as mr.raps:::rho.huber / rho.tukey" | file 2026-04-21; rewritten 2026-09-12 `6d163cc5` ("mr_raps: rewritten as a port of mr.raps") |
| H15 | `mendelian/diagnostics.py` (`mr_presso`) | R `MRPRESSO` 1.0 | GPL-3 (R-local) | A / function | `:707` "A port of the authors' R package ``MRPRESSO::mr_presso``" | 2026-09-12 `6d163cc5` ("mr_presso: port of MRPRESSO") |
| H16 | `mendelian/frontier/cml.py` | R `MendelianRandomization` 0.10.0 | GPL-2 \| GPL-3 (R-local) | A / functions | `:119` "following ``MendelianRandomization:::cML_estimate``"; `:139` `cML_SdTheta`; `:221` "follows the reference's start and update order" | 2026-09-12 `6d163cc5` |
| H17 | `mendelian/frontier/grapple.py` | R `GRAPPLE` 0.2.2 (GitHub jingshuw/GRAPPLE) | GPL (>=2) (GH-DESCRIPTION; repo has no GH-API license) | A / function | `:355` "Transcribes ``grappleRobustEst``'s block for ``r = 1``" (internal names `temp1`, `temp`, `c1..c4`) | 2026-09-22 `5eea7922` |
| H18 | `matching/cbps.py` | R `CBPS` 0.24 | GPL (>= 2) (R-local) | A / module core | `:245` "This is a faithful port of ``CBPS:::CBPS.2Treat`` (CBPS 0.24). Every convention ... is reproduced"; `:336` | file 2026-04-20; wording 2026-07-31 `aceaf9af` |
| H19 | `fast/inference.py` (HTZ Wald) | R `clubSandwich` 0.6.2 | GPL-3 (R-local) | A / functions | `:1298` "1:1 translation of clubSandwich (R, v0.6.2) ``Wald_testing`` HTZ branch + ``get_P_array(all_terms=TRUE)`` + ``total_variance_mat``" | 2026-04-27 `fbdeb858` |
| H20 | `bartik/_akm.py` | R `ShiftShareSE` 1.1.0 (Kolesar); R base LINPACK `dqrdc2` (R's variant) | ShiftShareSE GPL-3 (R-local); R GPL-2 \| GPL-3 (`license()`) | A / module + helper | `:3` "``_akm_fit`` is a line-by-line port of the reference implementation ... R ``ShiftShareSE::reg_ss.fit`` / ``ivreg_ss.fit``"; `:44` "a port of LINPACK ``dqrdc2`` (R's variant)"; `:101` `ShiftShareSE:::drop_collinear` | 2026-09-22 `5eea7922` |
| H21 | `spatial/gwr/bandwidth.py` (`_gold`) | R `GWmodel` 2.4-1 | GPL (>= 2) (R-local) | A / helper | `:14` "``GWmodel``'s ``gold()`` routine, reproduced step for step"; `:56` "Golden-section minimiser of ``GWmodel:::gold``"; `:58` "Transcribed from the R source"; `docs/dev/campaign_phase3/spatial_survey.md:102` "`GWmodel:::gold` transcribed" | file 2026-04-15; wording 2026-09-22 `5eea7922` |
| H22 | `mediation/sensitivity.py` | R `mediation` 4.5.1 (`medsens`) | GPL (>= 2) (R-local) | A / function | `:192` "Transcribes the lm/lm branch of R ``mediation::medsens`` step by step" | file 2026-04-15; wording 2026-09-22 `5eea7922` |
| H23 | `regression/interflex.py` (`r_density`) | R base `stats::density.default` | R: GPL-2 \| GPL-3 (`license()`) | A / helper | `:24` "density from R's ``stats::density`` algorithm, ported here"; `:63` "Port of ``stats::density.default``" | 2026-09-05 `10dc8a87` ("exact port of R stats::density") |
| H24 | `decomposition/_common.py` (`_bw_nrd0`, density at a point) | R base `stats::bw.nrd0`, `stats::density.default`, `C_BinDist` | R: GPL-2 \| GPL-3 (`license()`) | A / helper | `:488` "Port of R ``stats::bw.nrd0``"; `:534` "Port of stats::density.default(...) plus C_BinDist from R's stats package" (`C_BinDist` is internal C code) | file 2026-04-16; wording 2026-06-10 `73c14764` |
| H25 | `gmm/_dynpanel/_diagnostics.py`, `gmm/_dynpanel/_fit.py` (AR tests) | Stata `xtabond2` (Roodman) | GPL-3 or later (ado-header: "it under the terms of the GNU General Public License ... version 3") | A / function | `_diagnostics.py:323` "Implements ``xtabond2``'s ``_ARTests`` term by term"; `_fit.py:475` "exactly as xtabond2's `_ARTests` does" | 2026-08-01 `164c5a64` |
| H26 | `synth/_sdid_staggered.py` | Stata `sdid` 2.0.2 (Pailanir, Clarke) | GPL-3.0 (GH-API, `Daniel-Pailanir/sdid` and `damiancclarke/sdid`); SSC copy states none (sthlp) | A / module | `:15` "what ``sdid.ado``'s ``projected()`` uses"; `:24`, `:288` "as ``sdid.ado`` does"; `:309` "(``sdid.ado`` jk branch)"; `:375` "sdid.ado redraws"; `:401` "``sdid.ado``'s refusals (all r(451))" | 2026-09-27 `46ec6369` |
| H27 | `inference/jackknife.py` (score wild bootstrap) | Stata `boottest` (Roodman), `boottest.mata` | GPL-3 or later (ado-header and `boottest.mata` header) | A (read to corroborate) / function | `:1515` "reverse-engineered from its enumerated bootstrap distribution and corroborated against ``boottest.mata``"; commit `689aa881` (2026-07-01): "against the shipped boottest.mata source (ClustShare centering)", "[boottest.mata: ..." | 2026-07-01 `689aa881` |

Notes on HIGH rows:

- For H1, H5–H7, H9–H12 the method authors also publish Python
  implementations, and every one of them is GPL (PyPI rows above). There is
  no permissively licensed implementation by the same authors to port from.
- For H2 / H3 the only Python implementations on PyPI (`drdid` 1.1.6, `csdid`
  0.4.2, both from `d2cml-ai`) are third-party ports labelled MIT, not by
  Sant'Anna or Callaway; they are not a clean alternative source.
- H27 is borderline between A and C: the commit says the algorithm was
  recovered from Stata's outputs and then checked against the GPL source. It is
  listed as A because the source was read and is cited by line in the commit.

## MEDIUM

| # | Module(s) | Reference | License (source) | Class / scope | Evidence (file:line) | Entered / wording |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | `diagnostics/_oster.py` | Stata `psacalc` 2.1 (Oster), Mata functions `bound`, `d1quadsol`, `dnot1cubsol` | No license stated (ado-header, sthlp). R `robomit` 1.0.7, which ports the same Mata code, is MIT (R-local, CRAN) | A / module | `:3-5` "A port of the Mata code of Oster's Stata ``psacalc`` 2.1 (functions ``bound``, ``d1quadsol`` and ``dnot1cubsol``)" | 2026-09-22 `5eea7922` |
| M2 | `bounds/lee_manski.py` (`_leebounds_trimmed`, SE routine) | Stata `leebounds` 1.5 (Tauchmann) | No license stated (ado-header, sthlp) | A / functions | `:630` "Trimmed mean exactly as ``leebounds.ado`` (v1.5) computes it"; `:665` "Transcribes Stata ``leebounds``' ``leesbound`` routine term by term" | file 2026-04-04; wording 2026-09-22 `5eea7922` |
| M3 | `matching/_pstest.py` | Stata `pstest` 4.2.2 (Leuven, Sianesi; ships with psmatch2) | No license stated (ado-header, sthlp) | A / module | `:8` "following ``pstest.ado`` v4.2.2"; commit `b20caa6e` (2026-08-06) cites "pstest.ado 282/286" (ado line numbers) | 2026-08-06 `b20caa6e` |
| M4 | `matching/_matched_frame.py`, `matching/match.py`, `matching/_stata_lpoly.py` | Stata `psmatch2` (Leuven, Sianesi) | No license grant; `psmatch2.sthlp:510-516` carries only a warranty disclaimer | A / functions | `_matched_frame.py:134` "Reproduces, digit for digit, the formula in ``psmatch2.ado``"; `match.py:88` "matching the definitions in Stata psmatch2.ado"; `match.py:2666`, `_stata_lpoly.py:6` quote ado code text ("do nearest neighbor if llr with tricube") | `_matched_frame.py` 2026-06-15 `0acecf1d` |
| M5 | `diagnostics/_weakiv_lm.py` | Stata `weakiv` (Finlay, Magnusson, Schaffer) | No license stated (ado-header, sthlp) | A / module | `:9` "Notation (the minimum-distance construction of ``weakiv.ado::compute_tests``)"; `:45` "``weakiv.ado::rk_kp`` with ``F = I``" | 2026-10-19 `d5c68413` |
| M6 | `timeseries/_break_tables.py` (Hansen table), `timeseries/structural_break.py` (`pv_sup`) | Hansen's `pv_sup.R` (users.ssc.wisc.edu/~bhansen/progs/jbes_97.html) | No license stated on the distribution page (page text checked). The same table is in R `strucchange` 1.5-4, GPL-2 \| GPL-3 (R-local) | A / data table + helper | `_break_tables.py:8` "Transcribed from Hansen's own ``pv_sup.R``"; `structural_break.py:172` "Transcribes ``pv_sup`` of Hansen's own code" | 2026-09-22 `5eea7922` |
| M7 | `diagnostics/sensemakr.py` (`robustness_value`) | R `sensemakr` 0.1.6 | GPL-3 (R-local). `PySensemakr` 0.0.8 (Python implementation from the sensemakr project): MIT classifier (PyPI) | C / formula | `:445` "Port of ``sensemakr::robustness_value.numeric`` for the OLS case" (the robustness value is a closed form in Cinelli and Hazlett 2020) | file 2026-04-03; wording 2026-06-10 `73c14764` |
| M8 | `imputation/mice.py` (Barnard-Rubin df) | R `mice` 3.19.0 | GPL (>= 2) (R-local) | C / formula | `:108` "written as R ``mice:::barnard.rubin`` writes it" | file 2026-04-05; wording 2026-09-22 `5eea7922` |
| M9 | `spatial/models/gmm.py` | R `spatialreg` 1.4-3 | GPL-2 (R-local) | C / functions | `:299` "(``spatialreg:::tsls``)"; `:327` "the ``G [lambda, lambda^2, sigma^2]' - g`` form of ``spatialreg:::.kpwuwu`` / ``.kpgm``" | file 2026-04-15; wording 2026-09-22 `5eea7922` |
| M10 | `survey/estimators.py`, `survey/design.py` | R `survey` 4.5 | GPL-2 \| GPL-3 (R-local) | C / functions | `estimators.py:103` "Mirrors R ``survey:::onestage`` / ``onestrat``"; `:531`, `:632` `survey:::svy.varcoef`; `design.py:209` `survey:::as.fpc` semantics | file 2026-04-04; wording 2026-09-22 `5eea7922` |
| M11 | `forest/_grf_fit.py` (`observation_weights`) | R `grf` 2.6.1 | GPL-3 (R-local; GH-API `grf-labs/grf` GPL-3.0) | C / helper | `:82` `"""``grf:::observation_weights`` ..."""`. Against this, `forest/_grf_engine.py:1-11` states "No third-party forest code is included ... that comparison is made on outputs only", and `docs/dev/modules/forest.md` forbids porting grf | 2026-09-18 `ba25df08` |
| M12 | `regression/ridge.py` | R `MASS::lm.ridge` (MASS 7.3-65) | GPL-2 \| GPL-3 (R-local) | C / module conventions | `:8` "The scaling, the GCV score and the two plug-in penalties (HKB, LW) are those of R ``MASS::lm.ridge``" | 2026-10-05 `8d9ba5a3` |
| M13 | `inference/jackknife.py` (`_matrix_power_sym`) | R `clubSandwich` 0.6.2 | GPL-3 (R-local) | C / helper | `:1361` "matching R ``clubSandwich:::matrix_power``" | 2026-04-05 file |
| M14 | `mendelian/extras.py` (`_mbe_bandwidth`) | R `MendelianRandomization` 0.10.0 | GPL-2 \| GPL-3 (R-local) | C / formula | `:104` "Hartwig et al.'s bandwidth, as ``MendelianRandomization:::mbe_est``" | 2026-09-12 `6d163cc5` |
| M15 | `synth/discos.py`, `synth/_core.py`; only the `stats::quantile(type = 7)` part | R `DiSCos` 0.1.4 (MIT) wrapping R base `stats::quantile` | DiSCos MIT (R-local); `stats` GPL-2 \| GPL-3 | C / helper | `discos.py:537` "operation-for-operation as R ``DiSCos:::quant7_sorted`` (``stats::quantile(type = 7)``)"; the DiSCos part is permissive; the type-7 definition is documented in `?quantile` | 2026-09-22 `5eea7922` |
| M16 | `decomposition/nonlinear.py` (Yun detail) | Stata `mvdcmp` (Powers, Yoshioka, Yun 2011) | **Not determined** (not installed locally; no SSC license file checked) | C / function | `:483` "Port of mvdcmp (Powers, Yoshioka & Yun 2011)"; commit `1793035f` "Now a port of mvdcmp" | 2026-09-13 `1793035f` |
| M17 | `decomposition/oaxaca.py` (Gelbach contributions) | Stata `b1x2` (Gelbach) | **Not determined** (not installed locally) | C / function | `:1011` "Contributions and their joint covariance (port of b1x2)" | 2026-09-13 `72a5c8de` |

M16 and M17 are placed at MEDIUM pending the license lookup; if both turn out
to state no license, they fall under "C + no license" and drop to LOW under
the rule used here.

## LOW: A + permissive (attribution needed)

These are ports by the module's own account, but the reference permits reuse
under notice. None of them is listed in `THIRD_PARTY_NOTICES.md` except
HonestDiD.

| Module(s) | Reference | License (source) | Evidence |
| --- | --- | --- | --- |
| `synth/fect.py`, `synth/_fect_cv.py`, `synth/_fect_tests.py`, `did/_equivalence.py` | R `fect` 2.4.1 | MIT + file LICENSE (R-local) | `fect.py:4` "Native port of the estimation core"; `_fect_cv.py:2`, `:67-329` `fect:::...`; `_equivalence.py:10` "all ported from ``fect:::diagtest``" |
| `regression/interflex.py` (all but `r_density`) | R `interflex` 1.4.0 | MIT + file LICENSE (R-local) | `:4` "Native port of the estimation core"; `:186` `interflex:::createFolds` |
| `rlasso/_core.py`, `rlasso/_logit.py`, `rlasso/effect.py`, `rlasso/logit_effect.py`, `rlasso/iv.py`, `iv/post_lasso.py` | R `hdm` 0.3.2 | MIT + file LICENSE (R-local) | `_core.py:1-3` "faithful port ... *line-for-line* re-implementation"; `iv.py:3`; `post_lasso.py:300` |
| `did/_twfe_weights.py` | R `TwoWayFEWeights` 2.1.0 | MIT + file LICENSE (R-local) | `:18-21` "line for line; every convention below is read from ``TwoWayFEWeights:::twowayfeweights_calculate``" |
| `did/gardner_2s.py` (covariance) | R `did2s` 1.2.1 | MIT + file LICENSE (R-local) | `:271` "the algebra below is the R ``did2s`` one verbatim" |
| `did/did_multiplegt_dyn.py` | R `DIDmultiplegtDYN` 2.3.4 | MIT + file LICENSE (R-local) | `:73` "``U_Gg_var`` in the package source"; `:148` "The source comments say ..." |
| `did/_bacon_controls.py` | R `bacondecomp` 0.1.1 | MIT + file LICENSE (R-local) | `:12` "follows R ``bacondecomp`` 0.1.1 (MIT) term by term" |
| `synth/discos.py`, `synth/_core.py` (`_eq_bounded_lsq`) | R `DiSCos` 0.1.4 | MIT + file LICENSE (R-local) | `discos.py:624` `DiSCos:::getGrid`; `_core.py:61` `DiSCos:::DiSCo_weights_reg` |
| `experimental/design.py` (`_complete_ra`) | R `randomizr` 2.0.1 | MIT + file LICENSE (R-local) | `:466` "Transcribes ``randomizr::complete_ra``"; `docs/dev/campaign_phase3/r2_misc_sens.md:59` "source transcribed" |
| `diagnostics/weak_iv.py` (tF table), `iv/iv_diag.py` | R `ivDiag` 1.0.6 | MIT + file LICENSE (R-local) | `weak_iv.py:532` "Transcribed from R ``ivDiag::tF``"; `iv_diag.py:4` "port of the post-2022 reporting standard ... mirroring R `ivDiag`" |
| `decomposition/yu_elwert.py` | R `cdgd` 1.0.1 | MIT + file LICENSE (R-local, CRAN) | `:323-326` "Efficient-influence-function estimator of ``cdgd::cdgd0_manual`` ... Transcribed from the package" |
| `timeseries/_break_tables.py` (Bai-Perron tables) | R `mbreaks` 1.0.1 | MIT + file LICENSE (R-local, CRAN) | `:16` "Transcribed from the MIT-licensed R package ``mbreaks``" |
| `diagnostics/sensitivity.py` (McCrary) | R `rdd` 0.57 (`DCdensity`) | Apache License (== 2.0) (R-local) | `:370` "A line-for-line port of R ``rdd::DCdensity``" |
| `synth/sdid.py` (weights, Frank-Wolfe, sparsify) | R `synthdid` 0.0.9 | GPL (>= 2) \| BSD_3_clause + file LICENSE, i.e. dual-licensed; the BSD-3 option is permissive (R-local) | `:1423`, `:1471`, `:1540`, `:1567`, `:1614`, `:1743`, `:1846` `synthdid:::...` |
| `spatial/gwr/mgwr.py` | PySAL `mgwr` 2.2.1 | 3-Clause BSD (PyPI) | `:3-5` "reproduces the authors' reference implementation, PySAL ``mgwr`` ... step for step" |
| `did/aggte.py` (`agg_weights='csdid'`) | Stata `csdid` (Rios-Avila) | MIT at repository level (GH-API `friosavila/stpackages`, which contains `csdid/`); the SSC copy states none (sthlp) | `:623` "read off ``csdid_estat.ado``"; `:632` "In Mata (``csdid_group``)" |
| `did/_arp.py`, `did/_arp_sets.py` | R `HonestDiD` 0.2.8 | MIT + file LICENSE (R-local) | `_arp.py:16-18`; notice already in `THIRD_PARTY_NOTICES.md` (compliant) |

## LOW: B (black box / documented conventions) and C + no license / proprietary

Representative B rows (not exhaustive; many more modules "match" R or Stata
output without any port wording):

- Explicit black-box statements: `did/pmatch.py:6` (PanelMatch, "used as a
  black box; it is GPL-3 and its source was not consulted"),
  `rd/_locrand_core.py:15` (rdlocrand, "its source is GPL and was not
  consulted"), `selection/elastic_net.py:12` (glmnet, "its source was not
  read"), `survival/competing_risks.py:36` (cmprsk, "No ``cmprsk`` source code
  is copied or translated"), `regression/robreg.py:19`, `mcmc/trees.py:13`,
  `synth/kernel.py:21`, `forest/_grf_engine.py:1-11`,
  `matching/_cem_extras.py:3` (MatchIt, "GPL; used as" a black box).
- Documented conventions or outputs only: `decomposition/rif.py:102`
  (`dineq`, GPL-3), `did/sun_abraham.py:236` (`eventstudyinteract`),
  `did/did_imputation.py:193` (`did_imputation`, GPL-3 per GH-API),
  `did/_etwfe_glm_design.py:3` (`jwdid`), `did/cic.py:25` (Kranker `cic`,
  output parity), `survival/causal_forest.py:32` (grf; `forest.md` records
  the comparison as black box), `rd/rdmulti.py:418` (pooling rule from the
  rdmulti documentation), `forest/_grf_family.py:699` (documented
  `grf::variable_importance`), `bartik/rotemberg_summary.py:3`,
  `did/spillover_rings.py:41`, `synth/experimental_design.py:17` (SCDesign:
  the module states it solves a different problem; repo has no license),
  `mendelian/mr.py:341` (weighted median from the paper's appendix code),
  `timeseries/_critvals.py` (tables from MacKinnon 2010 and Osterwald-Lenum).

C + no license stated / proprietary (LOW under the rule above):

- `inference/conley.py:62`, `:223`; Stata `acreg` (ado-header carries a
  copyright line, no license): two constants (`111` km per degree) and the
  documented behaviour of Mata `_makesymmetric`.
- `inference/rank_tests.py:779`; StataCorp `robvar.ado` (proprietary Stata
  component): the 10 % trimmed-mean definition, which the `[R] robvar` manual
  also states.

## Could not determine

- Licenses of Stata `mvdcmp` and `b1x2` (M16, M17): not installed locally.
- `did/functional_form.py:74` "The reference implementation bins an outcome
  ...": the reference is R `didFF` 0.1.0 (MIT + file LICENSE, R-local and
  GH-DESCRIPTION, identified from `tests/r_parity/79_didff.R`); whether the
  binning rule was read from its source is not stated. Permissive either way.
- `scpi` GitHub repository (`nppackages/scpi`) reports `NOASSERTION` through
  the GitHub API; the CRAN and PyPI metadata (GPL-3 / GPL-3.0-only) were used.
- Whether the HIGH rows copy *expression* (code structure) or only
  *algorithms* cannot be settled from docstrings. The rows record what the
  modules say about themselves.

## Recommended actions

For every HIGH row, and for M1–M6 (A + no license):

1. **Clean-room rewrite.** One contributor writes a specification from the
   paper, the package's public documentation (help pages, vignettes, the
   JSS / Stata Journal article) and black-box runs of the reference; a
   different contributor (or agent session) who has not read the reference
   source implements from that specification only. Remove internal-name
   docstrings (`R ``regularize.w``.`), local-variable names copied from the
   reference (`inf.cont.2`, `temp1`) and "port of" / "line by line" wording,
   and replace them with the paper equations they implement.
2. **Verify with the existing fixtures, not the source.** The parity fixtures
   were produced by running the reference, so they remain valid black-box
   targets: e.g. `tests/reference_parity/test_did_synth_scpi_parity.py` (H1),
   `test_drdid_family_parity.py`, `test_cs_rcs_parity.py`,
   `test_cs_rcs_options_parity.py` and Track A `38_drdid` (H2–H4),
   `test_lprobust_parity.py` (H5), Track A `88_rdbwselect` and
   `test_rdbwselect_*` (H6), `test_rdhonest_parity.py` /
   `test_rdhonest_cluster_R_parity.py` (H8), Track A `09_rddensity` and
   `test_rddensity_parity.py` (H12), Track A `23_evalue` (H13),
   `test_mr_raps_parity.py` / `test_mr_cml_parity.py` /
   `test_grapple_parity.py` (H14, H16, H17),
   `test_akm_collinear_shares_R_parity.py` (H20), Track A `87_interflex`
   (H23), Track A `50_xtabond` / `test_dynpanel_abdata_parity.py` (H25),
   Track A `12_sdid` (H26), `test_lee_bounds_*` (M2), `test_pstest_parity.py`
   (M3), `test_psmatch2_*` (M4), `test_weakiv_*` (M5). Rows without a listed
   fixture (H7 rdplot, H9 rd2d, H10 rdhte, H11 rdpower, H15 MRPRESSO, H18
   CBPS, H19 HTZ, H21 GWR, H22 medsens, H24 density, H27 boottest) need a
   fixture generated by running the reference on committed CSVs before the
   rewrite starts.
3. **Track A / JSS consequences.** Several rows sit on Track A estimation
   paths (H2–H4, H6, H12, H13, H23, H25, H26 and others). A rewrite changes
   source hashes, so the parity traces must be re-recorded and, while the JSS
   freeze is active, each changed frozen artifact logged in
   `docs/dev/jss_review_changes.md` (CLAUDE.md §5.1).
4. **Alternative to rewriting.** A relicensing review: ask the copyright
   holders for permission to distribute the derived code under MIT. Several
   references are maintained by the method authors (scpi / rdrobust / rd2d /
   rdhte / rdpower / nprobust / rddensity / lpdensity by the Cattaneo group;
   RDHonest and ShiftShareSE by Kolesar; mr.raps by Zhao; CBPS by Imai's
   group). Until resolved, the HIGH modules could be moved behind an optional
   GPL-licensed extra rather than shipped in the MIT wheel.

For M7–M15 (C + copyleft): confirm with the author of each change whether
the source was read. If not, reword the docstring to cite the paper formula
and the black-box check (the formula-level rows M7, M8, M13, M14, M15 are a
single equation each). If it was, treat the row as HIGH.

For M16–M17: look up the licenses (`ssc describe mvdcmp`, `ssc describe
b1x2`) and re-grade.

For LOW A + permissive rows: add each reference's copyright notice and
license text to `THIRD_PARTY_NOTICES.md` (MIT: fect, interflex, hdm,
TwoWayFEWeights, did2s, DIDmultiplegtDYN, bacondecomp, DiSCos, randomizr,
ivDiag, cdgd, mbreaks, csdid; BSD-3: synthdid under its BSD option, mgwr;
Apache-2.0: rdd, which also requires its NOTICE file, if any, to be kept).

## Out of scope, noted for completeness

- Optional runtime delegation to GPL software (`backend="r"` paths that call
  `Rscript`, the `rdrobust` Python package behind `bwselect="cct"`) does not
  put GPL code in the wheel; it was not assessed here.
- `agent/_translation/` translates user Stata / R scripts into `sp.*` calls;
  its "translation" wording does not refer to reference implementations.

## Addendum (2026-10-21)

| Row | Path | Reference | License | Action |
| --- | --- | --- | --- | --- |
| X1 | `tests/reference_parity/_fixtures/_verify_rdbwselect_cascade.R` | R `rdrobust` (internal `rdrobust:::rdrobust_bw`; the file's header says it hand-replicates `rdbwselect`'s three-stage cascade) | GPL-3 (row H6) | Deleted. Only its header comment was read for this decision. No test, script, workflow or Makefile referenced it (`grep` by file name over the tree); the remaining mentions are prose in `docs/rfc/rd_three_month_plan.md` (historical plan) and `docs/dev/cleanroom/lprobust.md` (listed as "not read"), left as historical records. `rd/_cct_bandwidth.py` was already rewritten clean-room. |
