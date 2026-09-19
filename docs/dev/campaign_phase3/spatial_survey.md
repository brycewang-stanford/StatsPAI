# Phase 3 — spatial_survey family

Worktree `pc-spatial-survey`. All changes are uncommitted. The family was run
in three parts: spatial (lead), survey and structural/frontier (two sub-agents
working in the same worktree on disjoint files). Every number below was
measured on the committed fixture bytes.

Outcome classes: 1 T2 aligned or bit-exact · 2 defect fixed, then 1 ·
3 documented convention difference · 4 reference wrong or non-unique ·
5 stochastic (T3) · 6 no canonical reference, or the reference computes a
different estimand.

## 1. Per-function table

### Spatial — `tests/reference_parity/test_spatial_survey_R_parity.py`

Fixtures:
- `_generate_spatial_survey_R.R` → `spatial_survey_R.json`
- `_fixtures/_generate_spatial_survey_stata.do` → `spatial_survey_stata.json`

Both read the CSVs written once by `_prepare_spatial_survey_data.R`:
- Columbus polygons (spData 2.3.5)
- Georgia (GWmodel)
- `plm::Produc` with `splm::usaww`

| function | reference | class | max rel err, est / SE (observed) |
|---|---|---|---|
| `sp.queen_weights` | spdep 1.4.2 `poly2nb(queen=TRUE)`, `nb2listw` styles W / S / U | 1 | neighbour sets identical; weights 0 (tolerance 1e-12) |
| `sp.rook_weights` | spdep 1.4.2 `poly2nb(queen=FALSE)`, style W | 1 | identical / 0 |
| `sp.block_weights` | spdep 1.4.2 `nb2blocknb(NULL, ID)`, style W | 1 | identical / 0 |
| `sp.kernel_weights` | spdep `nb2listwdist(type="dpd", alpha=2)`, raw and W (fixed bisquare); GWmodel 2.4.1 `gw.weight` (fixed Gaussian and bisquare, adaptive bisquare) | 2 (the `W.transform` defect, S1), then 1; adaptive Gaussian is class 3 | dpd W 6.9e-16; gw.weight 3.3e-16 absolute |
| `sp.gwr` | GWmodel 2.4.1 `gwr.basic`; 8 kernel × adaptive/fixed configurations, including fractional bw and bw > n | 2 (S3), then 1; `local_R2` is class 3 / 4 | betas 1.3e-11, SE 3.5e-14; RSS / AIC / AICc / BIC / enp / edf ≤ 1e-11 |
| `sp.gwr_bandwidth` | GWmodel 2.4.1 `bw.gwr` (AICc and CV × bisquare and gaussian × adaptive and fixed); `gwr.aic` / `gwr.cv` criterion values | 2 (S4, S5), then 1 | 8/8 bandwidths exact (0.0); criteria 7.0e-15 |
| `sp.mgwr` | GWmodel 2.4.1 `gwr.multiscale` at fixed bandwidths (`bw.seled=TRUE`, `force.armadillo=TRUE`, `predictor.centered=FALSE`, `threshold=1e-12`) | 2 (S6, fixed-bandwidth fixed point), then 1; bandwidth search is class 6 / open | betas 2.9e-10 (convergence-limited; tolerance 1e-8) |
| `sp.sarar_gmm` | spatialreg 1.4.3 `gstsls` (default, `robust=TRUE`, `sig2n_k=TRUE`); PySAL spreg 1.9.0 `GM_Combo(w_lags=1)` as a Python check | 2 (S7), then 1 | coefs 3.8e-11, SE 3.0e-11; λ 6.5e-9 (gstsls's `nlminb` stop; tolerance 1e-7) |
| `sp.spatial_iv` | sphet 2.1.1 `spreg(model="lag", het=TRUE, lag.instr=FALSE)` | 2 (S8: docs and a silently ignored argument), then 1 | 9.5e-14 / 6.8e-14 |
| `sp.spatial_panel` | splm 1.6.5 `spml(model="within")`: SAR / SEM / SDM × individual / twoways. Stata 18 `xsmle` 1.4.5 (SSC), `fe type(ind)`, `vce(oim)` | 2 (S9, S10, S11), then 1 | R: est 7.2e-8, SE 1.3e-8 (splm `optimize` floor; tolerance 1e-7). Stata (ind): est 2.5e-13; β SE 8.5e-11; spatial SE 3.0e-9 (tolerance 1e-8) |

### Survey (sub-agent) — `test_survey_design_R_parity.py`, `test_survey_calib_R_parity.py`

| function | reference | class | max rel err, est / SE |
|---|---|---|---|
| `sp.svydesign` (+ `.mean`, `.total`, `.glm`, df, fpc, nest, lonely PSU, DEFF) | R survey 4.5 (`svydesign`, `svymean`, `svytotal`, `svyglm`, `degf`, `confint`); Stata 18 `svyset` + `svy: mean/total/regress/logit/poisson` + `estat effects` | 2 (V1–V5), then 1; df / DEFF / CI conventions are class 3 with options | R: mean 2.9e-16 / 1.1e-15, GLM 3.9e-14 / 2.3e-14, p 6.4e-13. Stata: GLM 4.0e-14 / 2.8e-14 |
| `sp.svymean` / `sp.svytotal` / `sp.svyglm` (complex designs; already graded weights-only) | same | 2, then 1 | as above |
| `sp.rake` | R `survey::rake` (to its fixed point) and `calibrate(calfun="raking")`; Stata 18 `svycal rake` | 2 (V6), then 1 | weights 1.5e-15 at `tol=1e-14`; 1.0e-10 at the default `tol=1e-10` |
| `sp.linear_calibration` | R `survey::calibrate(calfun="linear")`; Stata 18 `svycal regress` | 1 (docstring fixed; collinearity now warns) | weights 5.5e-15 |

### Structural / frontier (sub-agent) — `test_frontier_struct_R_parity.py`, `test_blp_pyblp_parity.py`

| function | reference | class | max rel err, est / SE |
|---|---|---|---|
| `sp.lcsf` | R sfaR 1.0.1 `sfalcmcross` (2 classes, half-normal; production with z, cost without z) | 2 (F1), then 1 | 1.1e-8 / 8.6e-8 (tolerance 1e-6) |
| `sp.markup` | Stata 18 + SSC `markupest` 1.0.1, `method(dlw) pmethod(lp) valueadded` | 1 (bit-exact) | 8.1e-13 η-corrected, 3.4e-15 uncorrected |
| `sp.blp` | pyblp 1.2.0 (a Python reference, not R or Stata), identical Halton nodes via `agent_data`, `center_moments=False` | 2 (F3, F4), then 1 | β 6.2e-9, σ 1.6e-8, SE 5.6e-9 (tolerance 1e-6) |
| `sp.metafrontier` | R metafrontier 0.3.1 (`method="sfa", engine="sfaR", objective="lp"`) | 2 (F2), plus class 3: matches only with the new non-default `envelope="own"` | 2.2e-8 meta coefficients, TGR 7.0e-8, TE 1.2e-7 |
| `sp.malmquist` | per-period frontiers vs sfaR `sfacross`; the index recomputed in R from sfaR's betas | 6 for the index (no reference computes this estimand); the frontiers align | betas 7.8e-8, M 1.3e-8 |
| `sp.zisf` | none | 6 (plus F1) | identity and known-truth checks only |

**Counts** (18 function rows, plus `svymean` / `svytotal` / `svyglm` re-covered):
- Class 1 after a fix (class 2): 15 — kernel_weights, gwr, gwr_bandwidth, mgwr (fixed-bandwidth), sarar_gmm, spatial_iv, spatial_panel, svydesign, rake, lcsf, blp, metafrontier (`envelope="own"` only). The svymean / svytotal / svyglm complex-design SEs are counted within svydesign.
- Class 1 with no fix needed: queen, rook, block, markup, linear_calibration.
- Class 6 or open: malmquist (index), zisf, mgwr bandwidth search, spatial_panel two-way on the Stata side.
- Class 3 conventions exposed as options: GWR local R², kernel_weights adaptive Gaussian, spatial_panel `vce`, svyglm `dof`, svymean `deff`, metafrontier `envelope`.

## 2. Defects

### Spatial (lead)

**S1 — `W.transform` threw away the constructed weights (⚠️ correctness).**
- *Where:* `spatial/weights/core.py`.
- *What was wrong:* every transform rebuilt the weights from `[1.0] * k`. `kernel_weights(...)` and `distance_band(binary=False)` followed by `w.transform = "R"` returned a binary row-standardised W. `"O"` after `"R"` returned binary, not the original.
- *How found:* first divergence against `nb2listwdist(type="dpd", style="W")`.
- *Before → after:* Georgia row 146 before was uniform 0.023256 (= 1/43); after it is the kernel / rowsum values 3.09e-5, 7.19e-5, ….
- *Fix:* keep `_original` and apply each style to it. Default output changes only for non-binary W with a transform.

**S2 — `transform="V"` was not spdep's / libpysal's variance-stabilising style (⚠️ correctness).**
- *What was wrong:* the global `n / Q` rescale was missing.
- *Before → after:* Columbus queen weights summed to 105.2985; after they sum to 49 (= n), equal to `nb2listw(style="S")` exactly.

**S3 — GWR kernels differed from GWmodel and mgwr (⚠️ correctness).**
- *Where:* `spatial/gwr/gwr.py`.
- *What was wrong, three parts:*
  1. The exponential kernel was truncated at u ≥ 1. Both references leave it untruncated.
  2. Adaptive Gaussian and exponential put weight only on the k nearest points.
  3. Non-integer adaptive bw was rounded up (both references floor), and bw > n was capped instead of extrapolated as `(bw/n)·max d`.
- *How found:* the gwr.basic diagnostics differed for exactly those kernels.
- *Before → after AICc (GWmodel value in parentheses):*

  | configuration | before | after (GWmodel) |
  |---|---:|---:|
  | Gaussian, adaptive, 40 | 896.3499 | 894.1283726 (894.1283726) |
  | exponential, adaptive, 40 | 896.7510 | 891.2529635 |
  | exponential, fixed, 60000 | 1105.0634 | 897.9314442 |

**S4 — `gwr_bandwidth(criterion="CV")` minimised in-sample RSS (⚠️ correctness).**
- *What was wrong:* the code comment said "fall back on plain RSS". RSS is monotone in the bandwidth, so CV always picked the smallest candidate.
- *Before → after:* adaptive bisquare 7 → 147 (GWmodel 147); fixed 48990 → 316468.4969 (GWmodel 316468.4968965).
- *Fix:* the exact LOO score Σ(e_i / (1 − S_ii))². The result now exposes `cv`, `influence` and `tr_StS`.

**S5 — the golden-section search did not reproduce `bw.gwr` (⚠️ default output).**
- *Why it matters:* the criterion is not unimodal on the neighbour lattice. For example, the Georgia CV at bw 140–159 is jagged, so the selected bandwidth is a property of the search path.
- *Fix:* `GWmodel:::gold` transcribed, with its bounds `[20, n]` / `[D/5000, D]`, floor / round probes and 1e-4 stop. `tol` (default 1e-3 → 1e-4) is now that stop.
- *Before → after (fixed AICc):* 211025.27 → 210996.3347 (GWmodel identical).

**S6 — `mgwr` had three problems.**
- A bare `except Exception: bw_j = bws[j]` silently reused the previous bandwidth.
- It recovered β as f/x, which gives 0 wherever x = 0.
- It gave no warning on non-convergence.

*Fix:* the exception now propagates, β is kept directly, and a `RuntimeWarning` is raised. `bws=` is added (fixed covariate bandwidths). Default numbers are unchanged except where x = 0 or the old exception path fired.

**S7 — `sarar_gmm` was not GS2SLS (⚠️ correctness).**
- *What was wrong:*
  - Stage 3 filtered the instruments.
  - Stage 1 used only [X, WX].
  - β and ρ SEs were the stage-1 SAR SEs, printed next to the stage-3 estimates (the code comment called them a "serviceable approximation").
  - λ came from Nelder–Mead started at 0.1 with xatol 1e-7.
- *Fix:* rewritten as `gstsls`, step for step. λ is now the exact minimiser of the profiled quartic GM objective over |λ| < 1. On Columbus the quartic's other minimum, λ = 4.07, has a lower objective but is inadmissible. An mpmath check agrees with the new λ to 2e-15. New arguments: `sig2n_k`, `w_lags` (default 2 = gstsls; 1 = spreg `GM_Combo`, now matched to spreg's optimiser level, about 2e-6 in λ).
- *Before → after (Columbus):*

  | quantity | before | after | gstsls |
  |---|---:|---:|---:|
  | const | 43.97302 | 43.54044357 | 43.54044357 |
  | ρ | 0.453550 | 0.46178656 | |
  | λ | −0.006961 | −0.01698126 | |
  | SE(const) | 11.2365 | 10.6284221 | |

**S8 — `spatial_iv`: the docstring and `alpha` did not match the code.**
- The module docstring said "Conley-style spatial HAC", but the code computes HC0. HC0 is exactly what sphet `het=TRUE` reports.
- `alpha` was accepted and ignored.

*Fix:* docstring corrected; `coefficients` gains `z`, `p` (from `norm.sf`) and the `ci_lower` / `ci_upper` bounds. Point estimates and SEs are unchanged.

**S9 — `spatial_panel` SAR / SDM β SEs ignored the uncertainty in ρ (⚠️ correctness).**
- *What was wrong:* β SE was σ²(X'X)⁻¹ conditional on ρ, and the ρ SE came from a finite-difference profile Hessian with h = 1e-4.
- *Fix:* the analytic information matrix of (σ², ρ, β), as splm's `splaglm` / `sperrorlm` use. New `vce="oim"` gives the analytic observed Hessian, matching Stata `xsmle` to 1e-10 (β) and 3e-9 (spatial parameter).
- *Before → after (SAR FE):* SE(lpcap) 0.02535161 → 0.02544250 (splm 0.02544250); SE(ρ) 0.02108514 → 0.02351640.

**S10 — `spatial_panel` maximiser precision.**
- *What was wrong:* bounded Brent with default xatol, so ρ was accurate only to about 1e-5 relative. The bounds were additionally clipped to ±0.99.
- *Fix:* the eigenvalue interval, then a `brentq` root of the analytic concentrated score. The new ρ equals the mpmath root to 2e-14.
- *Before → after:* ρ 0.27468821 → 0.2746887114326 (exact 0.27468871143265). The shift is 1.8e-6 relative, so it is below the old reporting precision.

**S11 — SDM with `effects="twoways"` lagged the demeaned X (⚠️ correctness).**
- *What was wrong:* the lagged regressors were W·(QX) instead of the within transform Q·(WX). The two differ for time effects because W is not column-stochastic.
- *Before → after:* ρ 0.367330749 → 0.368846212 (splm 0.3688462096).

### Survey (sub-agent)

- **V1 (⚠️):** fpc given as population counts was divided by #elements instead of #PSU. The SE was NaN (R 0.4951435315136789); after the fix it is 0.49514353151367907.
- **V2 (⚠️):** design df with PSU ids repeating across strata collapsed to 1. The CI was [6.7114, 20.2148]; after the fix it is [12.34201491229616, 14.58418673639966] with df 17, equal to R and Stata.
- **V3 (⚠️):** `svyglm` binomial / poisson used the Gaussian bread. Logit SE was [0.1179, 0.0385] against R [0.5098, 0.1731]; Poisson was [0.4900, 0.1605] against [0.2138, 0.0502]. Now 3.9e-14.
- **V4 (⚠️):** DEFF depended on the weight scale. Mean DEFF was 204.877 against R 3.9032; now 3.9032222795694005.
- **V5:** a single-PSU stratum was handled silently, and the docstring falsely claimed R's behaviour. New `lonely_psu=` argument (R's rules) and a warning; default numbers are unchanged. **Decision needed:** R's default is to fail.
- **V6 (⚠️):** `rake` convergence was an absolute change on O(1/n) weights, leaving a 4.3e-4 margin error at n = 100 000. Now a relative margin rule, `tol=1e-10`. Unknown or missing categories now raise.
- **V7:** the `linear_calibration` docstring stated the wrong objective. Collinearity now warns; numbers are unchanged.

### Structural / frontier (sub-agent)

- **F1 (⚠️):** `lcsf` / `zisf` L-BFGS-B stopped about 1e-3 short, and the Hessian used a fixed-step finite difference.
  - *Fix:* `newton_polish` plus `richardson_hessian` (`frontier/_core.py`).
  - *Before → after:* estimates 1.15e-3 → 1.1e-8, SEs 1.44e-3 → 8.6e-8 against sfaR.
- **F2 (⚠️):** `metafrontier(cost=True)` returned TGR ≡ 1 because the sign of the gap was wrong. TGR min / mean / max was 1 / 1 / 1; now 0.170 / 0.445 / 1.0. Also added `envelope="own"|"all"` (default `"all"` unchanged; `"own"` equals R).
- **F3 (⚠️):** `blp` SEs were too small by a factor of √N (linear) and N (σ); the Jacobian was not scaled and the joint (β, σ) sandwich was missing. Linear SEs (0.00434, 0.00070, 0.00187) → (0.1075, 0.0201, 0.0474); σ SE 4.99e-5 → 0.0304.
- **F4 (⚠️):** `blp` elasticities ignored the random price coefficient (`sigma_price` was unused). The error against pyblp was over 1e-2; now about 1e-8.

## 3. Proposed promotion records (`_FROZEN_PROMOTIONS`)

```python
"queen_weights": {
    "status": "bit-exact",
    "reference": "spdep::poly2nb(queen = TRUE) + nb2listw(style = 'W' / 'S' / 'U')",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "spdep": "1.4.2", "sf": "1.1.1", "spData": "2.3.5"},
    "tolerance": "neighbour sets exact; weights 1e-12 rel (observed 0)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Columbus polygons (spData) as WKT at 17 digits, read by both sides. transform 'R' / 'V' / 'D' = nb2listw styles W / S / U. The 'V' comparison exposed a missing n / Q rescale. Regenerate via _generate_spatial_survey_R.R.",
},
"rook_weights": {
    "status": "bit-exact",
    "reference": "spdep::poly2nb(queen = FALSE) + nb2listw(style = 'W')",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "spdep": "1.4.2", "sf": "1.1.1", "spData": "2.3.5"},
    "tolerance": "neighbour sets exact; weights 1e-12 rel (observed 0)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Columbus: 236 queen links vs 200 rook links, so the fixture discriminates the two criteria. Regenerate via _generate_spatial_survey_R.R.",
},
"block_weights": {
    "status": "bit-exact",
    "reference": "spdep::nb2blocknb(NULL, ID) + nb2listw(style = 'W')",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "spdep": "1.4.2"},
    "tolerance": "neighbour sets exact; weights 1e-12 rel (observed 0)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Four regimes (POLYID %% 4) on Columbus. Regenerate via _generate_spatial_survey_R.R.",
},
"kernel_weights": {
    "status": "bit-exact",
    "reference": "spdep::nb2listwdist(type = 'dpd', alpha = 2) on dnearneigh(0, h); GWmodel::gw.weight",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "spdep": "1.4.2", "GWmodel": "2.4.1"},
    "tolerance": "weights 1e-12 rel / 1e-15 abs (observed 6.9e-16 rel, 3.3e-16 abs)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Fixed bisquare = dpd alpha 2, raw and row-standardised. The W comparison found W.transform rebuilding the weights from 1.0, which silently turned kernel weights binary. Fixed Gaussian, fixed bisquare and adaptive bisquare (k = gw.weight bw k + 1) are compared against gw.weight. The adaptive Gaussian keeps only the k nearest neighbours (a documented convention; GWmodel weights all points), asserted as such. Regenerate via _generate_spatial_survey_R.R.",
},
"gwr": {
    "status": "bit-exact",
    "reference": "GWmodel::gwr.basic 2.4.1",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "GWmodel": "2.4.1"},
    "tolerance": "local betas and SEs 1e-9 rel (observed 1.3e-11 / 3.5e-14); RSS, AIC, AICc, BIC, enp, edf 1e-10 rel",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Georgia, bisquare / gaussian / exponential x adaptive / fixed, fractional bw and bw > n. Fixed on the way: the exponential kernel was truncated, the adaptive Gaussian / exponential were k-NN-truncated, and fractional bw was rounded up. Local R^2 is a documented difference: StatsPAI follows mgwr (local weighted mean). GWmodel uses the global mean and, under an adaptive kernel, the transposed weight matrix; both are reconstructed and asserted. Regenerate via _generate_spatial_survey_R.R.",
},
"gwr_bandwidth": {
    "status": "bit-exact",
    "reference": "GWmodel::bw.gwr 2.4.1 (golden section gold(); gwr.aic / gwr.cv)",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "GWmodel": "2.4.1"},
    "tolerance": "selected bandwidth 1e-10 rel (observed 0 on 8 configurations); criterion values 1e-11 rel (observed 7.0e-15)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "AICc and CV x bisquare and gaussian x adaptive and fixed. The CV criterion used to be in-sample RSS (always the smallest bandwidth). The search now transcribes GWmodel's gold() because the criterion is not unimodal on the neighbour lattice. Regenerate via _generate_spatial_survey_R.R.",
},
"mgwr": {
    "status": "aligned",
    "reference": "GWmodel::gwr.multiscale 2.4.1 at fixed bandwidths (bw.seled = TRUE, force.armadillo = TRUE, predictor.centered = FALSE)",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "GWmodel": "2.4.1"},
    "tolerance": "back-fitting fixed point: betas 1e-8 rel (observed 2.9e-10), fitted values 1e-10",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Only the fixed-bandwidth additive-model solution (sp.mgwr(bws=...)) is pinned; GWmodel converged to dCVR < 1e-12. The covariate-bandwidth search is NOT reproduced: GWmodel centres predictors, stops on dCVR, freezes bandwidths after bws.reOpts repeats and bounds bw.gwr2 by nlower. GWmodel's default C++ path ignores bw.seled, hence force.armadillo. Regenerate via _generate_spatial_survey_R.R.",
},
"sarar_gmm": {
    "status": "aligned",
    "reference": "spatialreg::gstsls 1.4.3 (Kelejian-Prucha GS2SLS)",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "spatialreg": "1.4.3", "spdep": "1.4.2"},
    "tolerance": "beta, rho and SEs 1e-9 rel (observed 3.8e-11 / 3.0e-11); lambda 1e-7 rel (observed 6.5e-9)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Default, robust = TRUE (HC0) and sig2n_k = TRUE. lambda is the exact admissible minimiser of the KP (1999) moment objective; gstsls's nlminb stops 6.5e-9 short on this flat objective (mpmath check). Rewritten in this sweep: the old path filtered the instruments and reported stage-1 SEs next to stage-3 estimates. sphet::spreg(model = 'sarar') is the KP (2010) weighted estimator, a different estimator. Regenerate via _generate_spatial_survey_R.R.",
},
"spatial_iv": {
    "status": "bit-exact",
    "reference": "sphet::spreg(model = 'lag', het = TRUE) 2.1.1",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "sphet": "2.1.1", "spdep": "1.4.2"},
    "tolerance": "coefficients and HC0 SEs 1e-10 rel (observed 9.5e-14 / 6.8e-14)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json"],
    "note": "Columbus: CRIME on INC with HOVAL endogenous and DISCBD excluded (not lagged), row-standardised queen W. The docstring used to claim Conley HAC SEs; the code, and sphet, report White / HC0. Regenerate via _generate_spatial_survey_R.R.",
},
"spatial_panel": {
    "status": "aligned",
    "reference": "splm::spml(model = 'within') 1.6.5; Stata xsmle 1.4.5 fe type(ind) vce(oim)",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "splm": "1.6.5", "plm": "2.6.7", "Stata": "18", "xsmle": "version 1.4.5 5jun2017"},
    "tolerance": "vs splm: estimates and SEs 1e-7 rel (observed 7.2e-8 / 1.3e-8, the splm optimize floor); vs xsmle (tightened ml tolerances): estimates and beta SEs 1e-9 (observed 2.5e-13 / 8.5e-11), spatial-parameter SE 1e-8 (observed 3.0e-9)",
    "sides": ["py", "R", "Stata"],
    "test": ["tests/reference_parity/test_spatial_survey_R_parity.py", "tests/reference_parity/_fixtures/spatial_survey_R.json", "tests/reference_parity/_fixtures/spatial_survey_stata.json"],
    "note": "Produc / usaww. SAR, SEM and SDM x individual and two-way effects against splm (SDM = SAR with per-period W-lagged raw X). Stata covers entity effects only: for two-way SAR / SDM, xsmle lags the within transform of W y and reports non-convergence, a documented reference disagreement. vce = 'information' (splm, default) or 'oim' (xsmle). Regenerate via _generate_spatial_survey_R.R and _fixtures/_generate_spatial_survey_stata.do.",
},
```

The `svydesign` / `rake` / `linear_calibration` / `lcsf` / `markup` / `blp` records are the sub-agents' records below, unchanged. `metafrontier` is only recommended with a scope-limited note.

```python
"svydesign": {
    "status": "bit-exact",
    "reference": "survey::svydesign + svymean/svytotal/svyglm/degf (strata, nested PSUs, fpc, survey.lonely.psu); Stata svyset + svy: mean/total/regress/logit/poisson + estat effects",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "survey": "4.5", "Stata": "18 MP"},
    "tolerance": "estimates, SEs, DEFF, CI bounds 1e-10 rel, p-values 1e-9 rel (observed <= 4e-14; p 6e-13)",
    "sides": ["py", "R", "Stata"],
    "test": ["tests/reference_parity/test_survey_design_R_parity.py", "tests/reference_parity/_fixtures/survey_design_R.json", "tests/reference_parity/_fixtures/survey_design_stata.json"],
    "note": "Stratified clustered design with PSU ids repeating across strata, unequal weights, fpc as PSU counts / fractions / element counts, cluster-only and element designs, four lonely-PSU rules. GLM df: dof='design' = Stata e(df_r), dof='residual' = R summary.svyglm. DEFF: deff='wor' = R deff=TRUE, 'replace' = Stata without fpc. R svyglm references refitted from the converged coefficients (one-pass glm.fit weights are one step stale, ~2e-7 in the logit SE). Fixed here: fpc counts divided by elements not PSUs (NaN SE), df on non-nested ids, logit/poisson bread, scale-dependent DEFF. Regenerate via _generate_survey_design_R.R and _fixtures/_generate_survey_design_stata.do.",
},
"rake": {
    "status": "bit-exact",
    "reference": "survey::rake (to its fixed point) and survey::calibrate(calfun='raking'); Stata svycal rake",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "survey": "4.5", "Stata": "18 MP"},
    "tolerance": "calibrated weight shares 1e-12 rel at tol=1e-14 (observed 1.5e-15); 1e-8 at the default tol=1e-10 (observed 1.0e-10)",
    "sides": ["py", "R", "Stata"],
    "test": ["tests/reference_parity/test_survey_calib_R_parity.py", "tests/reference_parity/_fixtures/survey_calib_R.json", "tests/reference_parity/_fixtures/survey_calib_stata.json"],
    "note": "sp.rake returns weights summing to 1; references divided by their sum (N = 10000). IPF and Newton raking share one fixed point (R's two routes agree to 2e-15). Fixed here: the convergence test was an absolute change on O(1/n) weights (4.3e-4 margin error at n = 100000). Weights only: fed to sp.svydesign they give the fixed-weights SE (matches R/Stata), not the calibration-adjusted SE. Regenerate via _generate_survey_calib_R.R and _fixtures/_generate_survey_calib_stata.do.",
},
"linear_calibration": {
    "status": "bit-exact",
    "reference": "survey::calibrate(calfun='linear', unbounded); Stata svycal regress",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "survey": "4.5", "Stata": "18 MP"},
    "tolerance": "calibrated weights 1e-12 rel (observed 5.5e-15)",
    "sides": ["py", "R", "Stata"],
    "test": ["tests/reference_parity/test_survey_calib_R_parity.py", "tests/reference_parity/_fixtures/survey_calib_R.json", "tests/reference_parity/_fixtures/survey_calib_stata.json"],
    "note": "No intercept added: ~0 + income + age, and ~sex + income via one / male columns. Closed-form chi-squared projection, also asserted reference-free. Regenerate via _generate_survey_calib_R.R and _fixtures/_generate_survey_calib_stata.do.",
},
"lcsf": {
    "status": "aligned",
    "reference": "sfaR::sfalcmcross 1.0.1 (2 classes, half-normal)",
    "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "sfaR": "1.0.1"},
    "tolerance": "estimates 1e-6 rel (observed 1.1e-8); OIM SEs 1e-6 rel (observed 8.6e-8)",
    "sides": ["py", "R"],
    "test": ["tests/reference_parity/test_frontier_struct_R_parity.py", "tests/reference_parity/_fixtures/frontier_struct_R.json"],
    "note": "sfaR reports log variances; compared as ln_sigma = Zu/2. Classes matched by ascending sigma_u. Class-1 sigma_u is weakly identified, so both sides are optimiser-limited; the fixture keeps the best of four sfaR optimisers by max |gradient|, and StatsPAI's log-likelihood is at least as high. Production-with-z and cost-without-z cases.",
},
"markup": {
    "status": "bit-exact",
    "reference": "Stata markupest 1.0.1 (Rovigatti), method(dlw) pmethod(lp)",
    "reference_versions": {"Stata": "18", "markupest": "1.0.1 10May2020"},
    "tolerance": "1e-10 rel (observed 8.1e-13 corrected, 3.4e-15 uncorrected)",
    "sides": ["py", "Stata"],
    "test": ["tests/reference_parity/test_frontier_struct_R_parity.py", "tests/reference_parity/_fixtures/frontier_struct_stata.json"],
    "note": "Markup of the free input l on the prodest parity panel; the eta-corrected and uncorrected shares are both pinned. Compared on the 2084 firm-years in StatsPAI's production sample; Stata also returns the 281 first-year rows.",
},
"blp": {
    "status": "aligned",
    "reference": "pyblp 1.2.0 (Conlon & Gortmaker), identical Halton nodes via agent_data",
    "reference_versions": {"pyblp": "1.2.0", "numpy": "2.2.6", "python": "3.10.20"},
    "tolerance": "beta, sigma, SEs, objective/N, own elasticities 1e-6 rel (observed <= 1.6e-8)",
    "sides": ["py"],
    "test": ["tests/reference_parity/test_blp_pyblp_parity.py", "tests/reference_parity/_fixtures/blp_pyblp.json"],
    "note": "Python cross-package reference, not R or Stata. Two-step GMM with center_moments=False. sp gmm_objective = N * pyblp objective. Random-price-coefficient elasticities are pinned at fixed parameters to 1e-8.",
},
```

Notes for the integrator:

- **`blp`, `sides: ["py"]`.** `_check_external_evidence` returns early when the sides contain neither R nor Stata. The record therefore survives, but nothing in the index marks it as cross-package. Decide whether a `"pyblp"`-style side token is wanted. The same applies if the spreg `GM_Combo` check of `sarar_gmm` should be surfaced.
- **Existing `svymean` / `svytotal` / `svyglm` records** could add `test_survey_design_R_parity.py`, sides `["py","R","Stata"]`, and the note "complex designs covered by the svydesign fixture".
- **`metafrontier`**, only with an `envelope="own"`-scoped note:
  - status: `"aligned"`
  - reference: `metafrontier::metafrontier 0.3.1 (engine sfaR, objective lp)`
  - tolerance: `1e-6 rel (observed 1.2e-7)`
  - sides: `["py","R"]`
  - note: "envelope='own' only; the default 'all' has no reference".

## 4. Proposed CHANGELOG / MIGRATION

**Added**
- `sp.mgwr(bws=...)`: fixed covariate bandwidths.
- `sp.sarar_gmm(sig2n_k=, w_lags=)`.
- `sp.spatial_panel(vce="information"|"oim")`.
- `sp.gwr` result gains `cv`, `influence` and `tr_StS`.
- `sp.spatial_iv` coefficient table gains `z`, `p`, `ci_lower` and `ci_upper` (the `alpha` argument now takes effect).
- `sp.svydesign(lonely_psu=...)`, `svymean` / `svytotal` `deff=`, `svyglm(dof=)`.
- `sp.metafrontier(envelope=)`.
- Parity fixtures:
  - spdep / GWmodel / spatialreg / sphet / splm, and Stata `xsmle`
  - survey R + Stata
  - sfaR, markupest, metafrontier R and pyblp

**⚠️ Correctness**
- `W.transform` now re-weights the constructed weights; kernel and inverse-distance W no longer turn binary under `"R"`. `"V"` is now spdep style S (rescaled to sum n).
- `sp.gwr`: the exponential kernel is untruncated; the adaptive Gaussian / exponential weight all points; fractional adaptive bw is floored; bw > n is extrapolated. Matches GWmodel::gwr.basic.
- `sp.gwr_bandwidth(criterion="CV")` was minimising in-sample RSS; it now uses the LOO CV score. The search reproduces GWmodel's `bw.gwr` (default bounds and stop changed; `tol` default 1e-3 → 1e-4 is the golden-section stop).
- `sp.sarar_gmm` is rewritten as Kelejian-Prucha GS2SLS (`spatialreg::gstsls`). Estimates change, and SEs now belong to the reported estimates.
- `sp.spatial_panel`: SEs from the joint information matrix (β SEs for SAR / SDM used to ignore ρ uncertainty); the ML maximiser is resolved to machine precision; SDM with two-way effects now demeans the lagged regressors.
- The survey defects V1–V4 and V6, and the frontier / structural defects F1–F4, as listed in §2.

**MIGRATION rows**
- `W.transform` on non-binary weights: to reproduce the old output, set `transform = "B"` first and then `"R"`.
- `gwr` with an exponential kernel or an adaptive Gaussian: no switch back; the old kernel was not a documented quantity.
- `gwr_bandwidth`: the old CV was not a CV. For the old search bounds, pass `bw_min` / `bw_max`.
- `sarar_gmm`: `w_lags=1` for PySAL `GM_Combo`; the old estimator is not reproducible (it was internally inconsistent).
- `spatial_panel` SEs: `vce="oim"` for Stata `xsmle`; the old conditional β SEs are not available.
- Survey and frontier rows as in the sub-agent sections.

**Schema / registry drift for the integrator**
- Signatures changed: `gwr_bandwidth` (`tol` default), `mgwr` (`bws`), `sarar_gmm` (`sig2n_k`, `w_lags`), `spatial_panel` (`vce`), plus the survey / frontier signatures. Run `python scripts/dump_schemas.py`.
- `test_registry_param_drift` fails only on `svydesign.lonely_psu`. It needs `ParamSpec("lonely_psu", "str", False, description="Single-PSU stratum rule: remove|certainty|adjust|average|fail (R survey.lonely.psu)")` in `registry.py`.

## 5. Not closed

- **`sp.mgwr` bandwidth search (class 6 / open).** Only the fixed-bandwidth fixed point is pinned.
  - GWmodel's selection (predictor centring, dCVR, `bws.reOpts` freezing, `nlower`) and PySAL mgwr's (its own `golden_section`, SOC criterion, `eps = 1.0000001` on the adaptive bandwidth) are different algorithms, and neither is reproduced.
  - GWmodel's default C++ path silently ignores `bw.seled`.
- **`sp.spatial_panel` two-way effects on the Stata side (open, mechanism partly located).**
  - xsmle `type(both)` SAR / SDM gives ρ 0.1969085 / 0.3701182 against splm 0.1966642 / 0.3688462. With tightened tolerances it reports `e(converged)=0`.
  - A within-transform-of-Wy lag reproduces ρ 0.1969145 / 0.3701682, i.e. xsmle to about 5e-5, which is non-convergence scale. SEM two-way agrees to about 3e-8.
  - The engine is compiled Mata (`lxsmle.mlib`), so the source could not be read.
  - Methodological question for the maintainer: splm, and hence StatsPAI, lag the demeaned y, W(Qy). That adds a ρ²‖c_t‖² term to the concentrated SSE (c_t is the per-period mean of W(Qy)), whereas Q(Wy) is the within transform of the lag regressor. Worth a decision before claiming Stata parity for two-way effects.
- **GWR local R² (class 3 / 4).** StatsPAI follows mgwr. GWmodel differs in two ways: it uses the global mean (a convention), and under adaptive kernels it applies the transposed weight matrix (a reference defect, reproduced exactly in the test).
- **`sp.kernel_weights`.** The triangular kernel has no reference: GWmodel lacks it, and mgwr's triangular is untruncated and can go negative.
- **`sp.sarar_gmm`, Stata side (class 6).** `spivregress` / `spregress, gs2sls` are the Drukker–Prucha–Raciborski / KP (2010) weighted estimators, and `sphet::spreg(model="sarar")` is that family too (λ 0.0767 vs gstsls −0.0170 on Columbus). They are a different estimator, not a check on GS2SLS.
- **Survey:**
  - There is no calibration-adjusted variance in StatsPAI (fixed-weight SE only; 7.5–21% above R's calibrated SE).
  - R and Stata calibrated SEs differ by 0.13–0.26%.
  - Not covered: multistage fpc, and missing values in GLM formulas (patsy row drop misaligns weights — an existing issue).
- **Frontier:**
  - `zisf`: no package exists (`ssc describe` rc 601 for zisf / sfzi / zisfa / sfmodel / sftfe / lcsf / sflcm / sfmixture / lcmsfa; no sfaR export).
  - `malmquist`: DEA references compute a different estimand. `metafrontier::malmquist_meta(method="sfa")` differs: its EC is a BC88 ratio, while StatsPAI's EC includes noise, and its TC is the reciprocal.
  - `metafrontier` default `envelope="all"`: no reference. The source paper (O'Donnell, Rao & Battese 2008) could not be retrieved, so the default is unverified.
  - Suggested follow-up: `frontier/sfa.py` and `panel.py` share the fixed-step Hessian that F1 fixed in `mixture.py`, which is a probable cause of the 1.3e-5 SE gap in Track A `28_frontier`. Not touched, because that would move Track A outputs.
- **Pervasive, out of scope:** `except Exception: pass` around `_attach_prov` in 129 files (including `spatial/iv.py`).

## 6. `.gitignore`

- `tests/reference_parity/_fixtures/_ado_spatial_survey/`: the private Stata ado tree, containing `xsmle`, `markupest`, `prodest`, `blp` and `sfcross`. Do not commit it.
- `_generate_blp_pyblp.py` needs pyblp. It was installed with `pip install --target` into a session scratch directory, not the venv; regenerating needs `pip install pyblp` somewhere on `PYTHONPATH`.
- R packages were installed into the user library: sphet 2.1.1, splm 1.6.5, GWmodel (version as recorded in the fixture), spgwr, Benchmarking, productivity, metafrontier, smfa, BLPestimatoR.

## Files

**Modified (spatial):**
- `src/statspai/spatial/weights/core.py`, `weights/distance.py`
- `gwr/gwr.py`, `gwr/bandwidth.py`, `gwr/mgwr.py`
- `models/gmm.py`, `iv.py`, `panel/estimator.py`
- `tests/spatial/test_models_gmm.py` (GM_Combo test now `w_lags=1`, rtol 5e-3 → 1e-5)

**New (spatial):**
- `tests/reference_parity/_prepare_spatial_survey_data.R`
- `tests/reference_parity/_generate_spatial_survey_R.R`
- `tests/reference_parity/test_spatial_survey_R_parity.py` (47 tests)
- `_fixtures/_generate_spatial_survey_stata.do`
- `_fixtures/spatial_survey_{R,stata}.json`
- `_fixtures/spatial_survey_{columbus,georgia,produc,usaww}.csv`

**Survey / frontier files:** listed in the sub-agent reports (`src/statspai/survey/*`, `src/statspai/frontier/{_core,mixture,metafrontier}.py`, `src/statspai/structural/blp.py`, plus their generators, fixtures and tests).

**Tests run, all green:**
- the new parity files (47 + 88)
- `tests/spatial` and the other spatial suites: `test_tierD_p2_spatial_*`, `test_estimator_provenance_round7`, `test_spatial_models_parity`, `test_spdep_parity` (180)
- the sub-agents' module suites

**One known failure elsewhere:** `test_registry_param_drift` (`svydesign.lonely_psu`, registry is integrator-owned).

## 4b. MIGRATION rows — survey and frontier

| Function | What changes | Old → new (example, exact numbers from your fixtures/§2) | Old number, if you need it |
|---|---|---|---|
| `sp.svydesign` (+ `svymean` / `svytotal` SE) with `fpc=` given as population counts and clusters | Sampling fraction per stratum is #PSU sampled / N_h (R `as.fpc`), not #elements / N_h | `full` design SE: NaN → 0.49514353151367907 (R 0.4951435315136789) | Not reachable; the element-based fraction is not a documented quantity (it gives 1 − f < 0 → NaN) |
| `sp.svydesign` design df (CIs, p-values) when PSU ids repeat across strata and `nest=False` | PSUs are keyed by (stratum, PSU), so df = #PSU − #strata, as in R and Stata | df 1, CI [6.7114, 20.2148] → df 17, CI [12.342014912296227, 14.584186736399586] (R/Stata [12.34201491229616, 14.58418673639966]) | Not reachable; the old df counted raw ids globally, inconsistent with the nested variance it was paired with |
| `sp.svyglm(family="binomial" \| "poisson")` SEs | Sandwich bread is (X′ diag(w·V(μ)) X)⁻¹, not the Gaussian (X′WX)⁻¹ | Logit SE [0.1179, 0.0385] → [0.5098, 0.1731] (R; match 3.9e-14); Poisson SE [0.4900, 0.1605] → [0.2138, 0.0502]; coefficients move ~1e-9 (tighter IRLS) | Not reachable; the Gaussian bread is wrong for non-Gaussian families |
| `sp.svymean` / `sp.svytotal` DEFF | Denominator is svyvar·(N − n)/(N·n), not svyvar/Σw, so it no longer depends on weight scale | Mean DEFF 204.877 → 3.9032222795694005; total DEFF 1511.21 → 28.790859422197673 (R `deff=TRUE`) | Not reachable; the scale-dependent value is not a documented quantity (`deff="replace"` gives Stata's no-fpc DEFF, not the old number) |
| `sp.rake` | Stops when every category share is within `tol` of its target (relative), default `tol=1e-10`; was an absolute change on O(1/n) weights at 1e-6 | n = 100 000: margin error 4.3e-4 → 8.1e-11; n = 200: gap to R's fixed point 2.4e-7 (design start) / 1.9e-6 (equal start) → 1.04e-10 | Not reachable; `tol` now means a relative margin gap, so old `tol=1e-6` runs but stops on a different rule |
| `sp.lcsf` / `sp.zisf` | Newton polish after L-BFGS-B, and a Richardson-extrapolated Hessian for OIM SEs | lcsf vs sfaR (two fixture cases): estimates max rel err 1.15e-3 / 1.43e-3 → 1.1e-8 / 6.2e-9; SEs 1.44e-3 / 2.13e-3 → 8.6e-8 / 6.7e-8; log-likelihood −324.1954416 → reaches sfaR's −324.1954399 | Not reachable; the old optimum was a premature stop, not a different estimand |
| `sp.metafrontier(cost=True)` | Sign of the technology gap flipped for cost frontiers; TE_meta changes with it | TGR min / mean / max 1.0 / 1.0 / 1.0 → 0.170 / 0.445 / 1.0 | Not reachable; TGR ≡ 1 came from clipping a wrong-signed gap |
| `sp.blp` standard errors | Joint (β, σ) robust GMM sandwich with 1/N-scaled Jacobian and analytic dδ/dσ | Linear SEs (0.00434, 0.00070, 0.00187) → (0.1075, 0.0201, 0.0474); σ SE 4.99e-5 → 0.0304 (fixture, N = 609; pyblp match 5.6e-9) | Not reachable; old SEs were too small by √N (linear) and N (σ) |
| `sp.blp` elasticities when price is in `x_random` | Each simulated consumer's price coefficient is α + σ_p·ν (`sigma_price` was unused) | Rel. error vs pyblp > 1e-2 → ~1e-8 | Not reachable; the old value ignored the estimated σ_p |
