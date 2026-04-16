# Changelog

All notable changes to StatsPAI will be documented in this file.

## [0.8.0] - 2026-04-16

### Spatial Econometrics Full-Stack + 10-Domain Breadth Upgrade

**Largest release in StatsPAI history. 60+ new functions across 10 domains.**

#### Spatial Econometrics (NEW — 38 API symbols)

From 3 functions / 419 LOC to **38 functions / 3,178 LOC / 69 tests**. Python's first unified spatial econometrics package.

- **Weights (L1)**: `W` (sparse CSR), `queen_weights`, `rook_weights`, `knn_weights`, `distance_band`, `kernel_weights`, `block_weights`
- **ESDA (L2)**: `moran` (global + local), `geary`, `getis_ord_g`, `getis_ord_local`, `join_counts`, `moran_plot`, `lisa_cluster_map`
- **ML Regression (L3)**: `sar`, `sem`, `sdm`, `slx`, `sac` — sparse-backed, dual log-det path (exact + Barry-Pace), scales to N=100K
- **GMM (L3)**: `sar_gmm`, `sem_gmm`, `sarar_gmm` — Kelejian-Prucha (1998/1999), heteroskedasticity-robust
- **Diagnostics**: `lm_tests` (Anselin 1988 full battery), `moran_residuals`
- **Effects**: `impacts` (LeSage-Pace 2009 direct/indirect/total + simulated SE)
- **GWR (L4)**: `gwr`, `mgwr` (Multiscale GWR), `gwr_bandwidth` (AICc/CV golden-section)
- **Spatial Panel (L5)**: `spatial_panel` (SAR-FE / SEM-FE / SDM-FE, entity + twoways)
- **Cross-validated**: Columbus rtol<1e-7 vs PySAL spreg 1.9.0; Georgia GWR bit-identical vs mgwr 2.2.1; GMM rtol<1e-4 vs spreg GM_*

#### Time Series

- `local_projections` — Jordà (2005) IRF with Newey-West HAC
- `garch` — GARCH(p,q) MLE with multi-step forecast
- `arima` — ARIMA/SARIMAX with auto (p,d,q) AICc grid search
- `bvar` — Bayesian VAR with Minnesota (Litterman) prior

#### Causal Discovery

- `lingam` — DirectLiNGAM (Shimizu 2011), bit-identical vs lingam package
- `ges` — Greedy Equivalence Search (Chickering 2002)

#### Matching

- `optimal_match` — Hungarian 1:1 matching (min total Mahalanobis distance)
- `cardinality_match` — Zubizarreta (2014) LP-based matching with balance constraints

#### Decomposition & Mediation

- `rifreg` — RIF regression (Firpo-Fortin-Lemieux 2009)
- `rif_decomposition` — RIF Oaxaca-Blinder for distributional statistics
- `mediate_sensitivity` — Imai-Keele-Yamamoto (2010) ρ-sensitivity

#### RD & Survey

- `rdpower`, `rdsampsi` — power/sample-size for RD designs
- `rake`, `linear_calibration` — survey calibration (Deville-Särndal 1992)

#### Survival

- `cox_frailty` — Cox with shared gamma frailty (Therneau-Grambsch)
- `aft` — Accelerated Failure Time (exponential/Weibull/lognormal/loglogistic)

#### ML-Causal (GRF)

- `CausalForest.variable_importance()`, `.best_linear_projection()`, `.ate()`, `.att()`
- **Bugfix**: honest leaf values now correctly vary per-leaf

#### Infrastructure

- OLS/IV `predict(data, what='confidence'|'prediction')` with intervals
- Pre-release code review: 3 critical + 2 high-priority bugs fixed

## [0.7.1] - 2026-04-15

DID-focused polish release. Brings the Wooldridge (2021) ETWFE
implementation to full feature parity with the R `etwfe` package,
adds a one-call method-robustness workflow, and closes 12 issues
uncovered by an internal code review round. All 27 new / updated
DID tests pass (`pytest tests/test_did_summary.py`).

### Added — ETWFE full parity with R `etwfe`

- **`sp.etwfe()` explicit API** aligned with R `etwfe` (McDermott 2023)
  naming. Thin alias over `wooldridge_did()` with a full argument-
  mapping table in the docstring.
- **`xvar=` covariate heterogeneity** (single string or list of names).
  Adds per-cohort × post × `(x_j − mean(x_j))` interactions; `detail`
  gains `slope_<x>` / `slope_<x>_se` / `slope_<x>_pvalue` columns.
  Baseline ATT is reported at the sample means of every covariate.
- **`panel=False` repeated cross-section mode** — replaces unit FE
  with cohort + time dummies (R `etwfe(ivar=NULL)` equivalent).
- **`cgroup='nevertreated'`** — per-cohort regressions restricted to
  (cohort g) ∪ (never-treated); cohort-size-weighted aggregation
  (R `etwfe(cgroup='never')` equivalent). Default `'notyet'` preserves
  prior ETWFE behaviour.
- **`sp.etwfe_emfx(result, type=…)`** — R `etwfe::emfx`-equivalent
  four aggregations: `'simple'`, `'group'`, `'event'`, `'calendar'`.
  `include_leads=True` returns full event-time output including pre-
  treatment leads for pre-trend inspection (`rel_time = -1` is the
  reference category).

### Added — one-call DID method-robustness workflow

- **`sp.did_summary()`** — fits five modern staggered-DID estimators
  (CS, SA, BJS, ETWFE, Stacked) to the same data and returns a tidy
  comparison table with per-method (estimate, SE, p, 95 % CI). Mean
  across methods + cross-method SD flag method-sensitivity of results.
- **`include_sensitivity=True`** — attaches the Rambachan-Roth (2023)
  breakdown `M*` to the CS row, giving a three-way robustness readout
  in a single call.
- **`sp.did_summary_plot()`** — forest plot of per-method estimates
  with cross-method mean line; `sort_by='estimate'` supported.
- **`sp.did_summary_to_markdown()` / `_to_latex()`** — publication-
  ready exports (GFM tables / booktabs LaTeX with auto-escaped
  ampersands).
- **`sp.did_report(save_to=dir)`** — one-call bundle that writes
  `did_summary.txt` / `.md` / `.tex` / `.png` / `.json` to a folder.

### Fixed — 12 issues from the internal code review

Blockers (C-severity):

- `etwfe(xvar=…)` now raises a clear `ValueError` when the covariate
  is all-NaN or (near-)constant. Previously returned `n_obs = 0,
  estimate = 0` silently.
- `etwfe(panel=False, cgroup='nevertreated')` now raises a crisp
  `NotImplementedError` instead of silently falling back to
  `'notyet'`.
- `did_summary` now validates column names up front (raises
  `KeyError` listing missing columns) and only catches narrow
  estimator-side exceptions inside the fit loop; user typos in
  `controls=` / `cluster=` surface as proper errors.
- `did_summary` results round-trip cleanly through stdlib
  serialisation (`DIDSummaryResult(CausalResult)` subclass with a
  real `.summary()` method, replacing the prior closure-bound
  instance attribute).

High-severity:

- `etwfe_emfx(type='event'/'calendar')` now computes SEs via the
  delta method on the stored event-study vcov instead of the
  independent-coefficient approximation. `model_info['se_method']`
  advertises which path was used.
- `etwfe_emfx(type='group')` headline `se` / `pvalue` / `ci` are now
  populated (match the underlying fit's overall ATT exactly).
- Validation for `did_summary_plot` / `_to_markdown` / `_to_latex`
  aligned on a single sentinel `model_info['_did_summary_marker']`.
- `_etwfe_never_only` no longer leaves a `_ft_cache` helper column
  on the caller's DataFrame.
- Slope indexing in `_etwfe_with_xvar` is now name-keyed
  (`coef_index` dict); regression test verifies swapping
  `xvar=['x1','x2']` vs `['x2','x1']` produces identical slopes per
  name.
- `etwfe(panel=False)` with rank-deficient designs emits a
  `RuntimeWarning` pointing at concrete remedies (previously fell
  through to `pinv` silently).

### Tests

- New test module `tests/test_did_summary.py` — 27 cases covering
  consistency with direct estimator calls, export formats, forest
  plot rendering, `etwfe_emfx` round-trips, xvar / panel / cgroup
  options, the 12 review fixes, and the `include_leads` mode.

## [0.7.0] - 2026-04-14

Focused release reaching feature parity with the R `did` / `HonestDiD`
packages and the Python `csdid` / `differences` packages for staggered
Difference-in-Differences.  All core algorithms are reimplemented from
the original papers — **no wrappers, no runtime dependencies on upstream
DID packages**.  Full DiD test suite: 47 → 170+ (including three rounds
of post-implementation audit that surfaced and fixed 9 bugs before
release).

### Added — Core estimation

- **`sp.aggte(result, type=...)`** — unified aggregation layer for
  `callaway_santanna()` results.  Four aggregation schemes (`simple`,
  `dynamic`, `group`, `calendar`) backed by a single weighted-
  influence-function engine.  Callaway & Sant'Anna (2021) Section 4.
- **Mammen (1993) multiplier bootstrap** — IQR-rescaled pointwise
  standard errors *and* simultaneous (uniform / sup-t) confidence
  bands over the aggregation dimension.  Matches the uniform-band
  behaviour of the R `did::aggte` function.
- **`balance_e` / `min_e` / `max_e`** — event-study cohort balancing
  and window truncation (CS2021 eq. 3.8).
- **`anticipation=δ`** parameter on `callaway_santanna()` — shifts
  the base period back by δ periods per CS2021 §3.2.
- **Repeated cross-sections** support via `callaway_santanna(panel=False)`
  — unconditional 2×2 cell-mean DID with observation-level influence
  functions (CS2021 eq. 2.4, RCS version).  Optional covariate
  residualisation with `x=[...]` for regression adjustment.  All
  downstream modules (`aggte`, `cs_report`, `ggdid`, `honest_did`)
  work on RCS results with no code changes.
- **dCDH joint inference** (`did_multiplegt`) — `joint_placebo_test`
  (Wald χ² across placebo lags with bootstrap covariance, dCDH 2024
  §3.3) and `avg_cumulative_effect` (mean of dynamic[0..L] with
  SE preserving cross-horizon covariance, dCDH 2024 §3.4).
- **`sp.bjs_pretrend_joint()`** — cluster-bootstrap joint Wald pre-
  trend test for BJS imputation results.  Upgrades the default
  sum-of-z² test (which assumes pre-period independence) to a full
  covariance-aware statistic.

### Added — Reporting & visualisation

- **`sp.cs_report(data, ...)`** — one-call report card.  Runs the
  full pipeline (ATT(g,t) → four aggregations with uniform bands →
  pre-trend Wald → Rambachan–Roth breakdown M\* for every post event
  time) under a single bootstrap seed and pretty-prints the result.
  Returns a structured `CSReport` dataclass.
- **`sp.ggdid(result)`** — plot routine for `aggte()` output,
  mirroring R `did::ggdid`.  Auto-dispatches on aggregation type;
  uniform band overlaid on pointwise CI.
- **`CSReport.plot()`** — one-call 2×2 summary figure: event study
  with uniform band (top-left), θ(g) per-cohort (top-right), θ(t)
  per-calendar-time (bottom-left), Rambachan–Roth breakdown M\*
  bars (bottom-right).
- **`CSReport.to_markdown()`** — GitHub-flavoured Markdown export
  with proper integer-column rendering and a configurable
  `float_format`.
- **`CSReport.to_latex()`** — publication-ready booktabs fragment
  wrapped in a `table` float.  Zero `jinja2` dependency (hand-rolled
  booktabs renderer); auto-escapes LaTeX special characters.
- **`CSReport.to_excel()`** — six-sheet workbook (`Summary`,
  `Dynamic`, `Group`, `Calendar`, `Breakdown`, `Meta`).  Engine
  autoselect (openpyxl → xlsxwriter) with a clear ImportError when
  neither is installed.
- **`cs_report(..., save_to='prefix')`** — one-call dump of the
  full export matrix: writes `<prefix>.{txt,md,tex,xlsx,png}` in
  a single invocation, auto-creating missing parent directories.
  Optional dependencies (openpyxl, matplotlib) are skipped silently
  so a minimal install still produces text + md + tex.
- **`sp.did(..., aggregation='dynamic', n_boot=..., random_state=...)`**
  — the top-level dispatcher now forwards CS-style arguments
  (`aggregation`, `panel`, `anticipation`) and can pipe a CS result
  straight through `aggte()` in a single call.

### Changed

- **`sun_abraham()` inference layer rewritten** — replaces the
  former ad-hoc `√(σ²/(total·T))` approximation with a Liang–Zeger
  cluster-robust sandwich `(X'X)⁻¹ Σ_c X_c' u_c u_c' X_c (X'X)⁻¹`
  (small-sample adjusted), delta-method IW aggregation SEs
  `w' V_β w`, iterative two-way within transformation (correct on
  unbalanced panels), and optional `control_group='lastcohort'` per
  SA 2021 §6.
- **`sp.honest_did()` / `sp.breakdown_m()` made polymorphic** — now
  accept the legacy `callaway_santanna()` / `sun_abraham()` format
  (event study in `model_info`) *and* the new `aggte(type='dynamic')`
  format (event study in `detail` with Mammen uniform bands).  The
  idiomatic pipeline `cs → aggte → honest_did → breakdown_m` now
  runs end-to-end with no manual plumbing.
- **README DiD parity matrix** added, comparing StatsPAI against
  `csdid`, `differences`, and R `did` + `HonestDiD` across 15
  capabilities.

### Fixed (from pre-release audit rounds)

- **Critical — `aggte(type='dynamic').estimate`** previously averaged
  pre- *and* post-treatment event times into the overall ATT,
  polluting the headline number with placebo signal.  Now averages
  only e ≥ 0, matching R `did::aggte`'s print convention.  On a
  typical DGP the bug shifted the reported overall by nearly a
  factor of 2.
- **LaTeX escape non-idempotence** in `CSReport.to_latex()`:
  `\` → `\textbackslash{}` followed by `{` → `\{` mangled the
  just-inserted braces.  Fixed with a single-pass `re.sub`.
- **`cs_report(save_to='~/study/…')`** did not expand `~`; fixed
  via `os.path.expanduser`.
- **`cs_report(sa_result)` / `aggte(sa_result)`** raised cryptic
  `KeyError: 'group'`; both entry points now detect non-CS input
  up-front and raise a clear `ValueError`.
- **`cs_report(pre_fitted_cs, estimator=…)`** silently ignored the
  override; now emits a `UserWarning` listing every shadowed arg.
- **`sp.did(method='2x2', aggregation='dynamic')`** silently ignored
  CS-only arguments; now raises an informative `ValueError`.
- **`bjs_pretrend_joint`** swallowed all exceptions as "bootstrap
  failed"; now narrows to expected failure modes and re-raises
  unexpected errors with context.
- **`matplotlib.use('Agg')`** in `_save_report_bundle` no longer
  switches the backend unconditionally (respects Jupyter sessions).

### References

- Callaway, B. and Sant'Anna, P.H.C. (2021). *J. of Econometrics* 225(2).
- Sun, L. and Abraham, S. (2021). *J. of Econometrics* 225(2).
- Mammen, E. (1993). *Ann. Statist.* 21(1).
- Liang, K.-Y. and Zeger, S.L. (1986). *Biometrika* 73(1).
- de Chaisemartin, C. and D'Haultfoeuille, X. (2020). *AER* 110(9).
- de Chaisemartin, C. and D'Haultfoeuille, X. (2024). *RESt*, forthcoming.
- Rambachan, A. and Roth, J. (2023). *Rev. Econ. Studies* 90(5).
- Borusyak, K., Jaravel, X. and Spiess, J. (2024). *ReStud* 91(6).

## [0.6.2] - 2026-04-12

### Added

- **OLS `predict()`**: `result.predict(newdata=)` for out-of-sample prediction on OLS results
- **`balance_panel()`**: Utility to keep only units observed in every time period (`sp.balance_panel()`)
- **Panel `balance=True`**: Convenience flag in `sp.panel()` to auto-balance before estimation
- **Analytical weights for DID**: `weights=` parameter added to `did()`, `ddd()`, and `event_study()` for population-weighted estimation (Stata `[aweight=...]` equivalent)
- **Matching `ps_poly=`**: Polynomial propensity score specification (`ps_poly=2` adds interactions/squares, following Cunningham 2021 Ch. 5)
- **Synth `rmspe` plot**: Post/pre RMSPE ratio histogram (`synthplot(result, type='rmspe')`) per Abadie et al. (2010)
- **Synth placebo gap plot**: Full spaghetti placebo gap paths with `rmspe_threshold` filter (Abadie et al. 2010, Figure 4)
- **Graddy (2006) replication**: Fulton Fish Market IV example added to `sp.replicate()` (Mixtape Ch. 7)
- **Numerical validation tests**: Cross-validated against Stata/R reference values with humanized error messages

### Fixed

- **`outreg2` format auto-detection**: Correctly infers `.xlsx`/`.csv`/`.tex` from filename extension
- **Synth placebo p-value**: Now uses RMSPE *ratio* (√post/√pre) instead of squared ratio, matching Abadie et al. (2010) convention

### Improved

- **DID/DDD/Event Study**: Weights propagation through WLS with proper normalization and validation
- **Synth placebos**: Store full placebo gap trajectories, per-unit RMSPE ratios, and unit labels for richer post-estimation analysis
- **Matching tests**: Added comprehensive test suite for PSM, Mahalanobis, CEM, and stratification methods

## [0.6.1] - 2026-04-07

### Fixed

- **Interactive Editor — Theme switching**: Themes now fully reset before applying, so switching between themes (e.g. ggplot → academic) correctly updates all visual properties instead of leaking stale settings
- **Interactive Editor — Apply button**: Fixed Apply button being clipped/hidden on the Layout tab due to panel overflow
- **Interactive Editor — Panel layout**: Fixed panel content disappearing when using flex layout for bottom-pinned Apply button
- **Interactive Editor — Style tab**: Fixed Style tab stuck on "Loading" after Theme tab was reordered to first position
- **Interactive Editor — Error visibility**: Widget callback errors now surface in the status bar instead of being silently swallowed

### Improved

- **Interactive Editor — Auto mode**: Clicking Auto now always refreshes the preview, giving immediate visual feedback
- **Interactive Editor — Auto/Manual toggle**: Compact toggle button moved to panel header with sticky positioning
- **Interactive Editor — Apply button**: Separated from Auto toggle and placed at panel bottom-right for better UX
- **Interactive Editor — Theme tab**: Moved to first position for better discoverability
- **Interactive Editor — Color pickers**: Added visual confirmation feedback on all color changes
- **Interactive Editor — Code generation**: Auto-generate reproducible code with text selection support in the editor
- **Smart recommendations**: Enhanced compare and recommend logic
- **Registry**: Expanded module support in the registry

## [0.1.0] - 2024-07-26

### Added
- **Core Regression Framework**
  - OLS (Ordinary Least Squares) regression with formula interface
  - Robust standard errors (HC0, HC1, HC2, HC3)
  - Clustered standard errors
  - Weighted Least Squares (WLS) support

- **Causal Inference Module**
  - Causal Forest implementation inspired by Wager & Athey (2018)
  - Honest estimation for unbiased treatment effect estimation
  - Bootstrap confidence intervals for treatment effects
  - Formula interface: `"outcome ~ treatment | features | controls"`

- **Output Management (outreg2)**
  - Excel export functionality similar to Stata's outreg2
  - Support for multiple regression models in single output
  - Customizable formatting options
  - Professional table layout

- **Unified API Design**
  - Consistent `reg()` function interface
  - Formula parsing: R/Stata-style syntax `"y ~ x1 + x2"`
  - Type hints throughout the codebase
  - Comprehensive documentation

### Technical Details
- Python 3.8+ support
- Dependencies: numpy, scipy, pandas, scikit-learn, openpyxl
- MIT License
- Comprehensive test suite
