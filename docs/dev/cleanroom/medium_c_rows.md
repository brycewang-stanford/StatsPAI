# MEDIUM C-class rows M7-M17: what each one is and what was done

Rows of `docs/dev/2026-10-20-license-provenance-audit.md` whose wording did not
settle whether reference source had been read. For each: is the code a
formula or convention published in a paper or in documentation? If yes, the
comment / docstring now cites that source (and the reference as a black-box
check); if the helper is small and its provenance could not be shown from
public sources, it was re-implemented from a specification without reading
its old body, and checked against recorded outputs of the old code.

| Row | Code | Published source | Action |
| --- | --- | --- | --- |
| M7 | `diagnostics/sensemakr.py`: `_robustness_value`, `_sensemakr_bound_scale` | Cinelli and Hazlett (2020) `[@cinelli2020making]` eqs. (18)-(20); Supplement B eqs. (73)-(84) for the benchmark bounds; eqs. (12)-(13) for the adjusted estimate / SE | Both helpers re-implemented (spec `sensemakr_mice_helpers.md`; the RV by a separate agent because the spec writer had seen two body lines). R `sensemakr` black-box fixture, 432 rows; comments in `sensemakr()` cite the equations |
| M8 | `imputation/mice.py`: `_barnard_rubin_df` | Barnard and Rubin (1999) `[@barnard1999small]`; van Buuren, FIMD 2nd ed., eqs. (2.24), (2.30)-(2.32) | Re-implemented by the separate agent; R `mice` black-box fixture, 72 rows |
| M9 | `spatial/models/gmm.py`: `_tsls`, `_kp_gm_lambda`, `sarar_gmm` | Kelejian and Prucha (1998) `[@kelejian1998generalized]` GS2SLS; Kelejian and Prucha (1999) GM moments; 2SLS | Docstrings cite the papers; R-internal names removed. The source-derived remark "`gstsls`'s `W2X` argument is not used by the code" replaced by a black-box fact (columbus data: identical estimates and SEs with `W2X = TRUE / FALSE`, spatialreg 1.4-3); `nlminb` start value now cited from the `gstsls` help page |
| M10 | `survey/estimators.py`, `survey/design.py` | Ultimate-cluster linearisation variance; the GLM linearisation sandwich; the two forms of the fpc argument are documented in `?survey::svydesign` | Docstrings / comments describe the formulas and cite the R documentation instead of `survey:::onestage`, `svy.varcoef`, `as.fpc` |
| M11 | `forest/_grf_fit.py` and `forest/_grf_family.py`: `observation_weights` | The rule is in the documentation of R `grf` (`?causal_forest`, arguments `sample.weights` and `equalize.cluster.weights`: "each observation is given weight 1/cluster size") | Resolved: the docstring named a grf internal (`grf:::observation_weights`) while `_grf_engine.py` and `docs/dev/modules/forest.md` say no grf code is used. Both helpers were re-implemented from the documented rule (one implementation, the other delegates); bit-identical to the old outputs on 16 recorded cases. String cluster labels, a `TypeError` before, now work. The engine statement stands |
| M12 | `regression/ridge.py` | Ridge (Hoerl-Kennard 1970), GCV (Golub-Heath-Wahba 1979); MASS documents none of its constants | `ridge()` re-implemented (spec `ridge.md`): constants pinned by black-box `MASS::lm.ridge` runs; 22 recorded calls of the old function agree to 1.4e-13 |
| M13 | `inference/jackknife.py`: `_matrix_power_sym` | Moore-Penrose power of a symmetric matrix as used by the CR2 adjustment (`[@bell2002bias]`, `[@pustejovsky2018small]`); threshold from clubSandwich outputs | Re-implemented (eigendecomposition of the symmetrised matrix, absolute cutoff `10**tol`); 300 recorded cases agree to 2e-16. A RuntimeWarning of the old code echoed its threshold line while probing; it showed only what the docstring already stated |
| M14 | `mendelian/extras.py`: `_mbe_bandwidth` | Hartwig, Davey Smith and Bowden (2017) `[@hartwig2017robust]`, eq. (7) (read from the full text, Europe PMC) | Re-implemented from eq. (7); the old outputs equal the formula with the n-1 sd to 3.7e-16 on 200 random cases |
| M15 | `synth/discos.py`: `_quant7_sorted` | Type-7 quantile defined in R's `?quantile`; the arithmetic order follows R `DiSCos` (MIT, now in `THIRD_PARTY_NOTICES.md`) | Docstring now states the definition and the MIT source; no GPL code involved |
| M16 | `decomposition/nonlinear.py` (Yun detail) | Yun (2004, 2005) `[@yun2004decomposing]`, `[@yun2005simple]` | Licence of Stata `mvdcmp` looked up: installed from SSC into a private ado directory; `grep` for licence / copyright / warranty lines finds none (only an unrelated "(C)" label in output text). "C + no licence": LOW. Comment cites Yun; Stata `mvdcmp` named as a black-box check |
| M17 | `decomposition/oaxaca.py` (Gelbach covariance) | Gelbach (2016) `[@gelbach2016covariates]`, whose abstract and appendix give the consistent covariance | Licence of Stata `b1x2` looked up the same way: a warranty disclaimer only, no licence grant. LOW. Comment cites Gelbach; `b1x2` named as a black-box check |

Only licence lines of the two ado packages were read (`grep -inE
'licen[cs]e|GPL|copyright|\(c\)|all rights|permission|redistribut|warrant|public domain'`);
the installed files were deleted afterwards.
