# DiD Roadmap — Gap Audit (2026-04-23)

> Snapshot of what the `sp.did` family actually ships vs. the post-CS (Callaway & Sant'Anna 2021) frontier. Input for deciding the v1.7 DiD sprint. **Status claims below are based on source inspection, not a fresh parity run.** Any numeric claim needs a dedicated `tests/reference_parity/` run before it becomes a guide statement.

## 1. 存量盘点 — `src/statspai/did/`

| File | LOC | Public API (top-level `sp.*`) | Role | Current state |
|---|---:|---|---|---|
| `callaway_santanna.py` | 30k | `sp.callaway_santanna`, aggregators | CS (2021) ATT(g,t) + DR-DiD | **Production**. Registered in `registry.py:452`. Reference-parity guide in `docs/guides/callaway_santanna.md` + `cs_report.md`. |
| `sun_abraham.py` | 12k | `sp.sun_abraham` | IW event-study | Production. Registered (auto). |
| `did_imputation.py` | 22k | `sp.did_imputation` | Borusyak-Jaravel-Spiess imputation | Production + `bjs_inference.py` analytical variance. |
| `gardner_2s.py` | 12k | `sp.gardner_did`, `sp.did_2stage` | Gardner (2021) two-stage | Production. Registered in `registry.py:3662`. |
| `wooldridge_did.py` | 68k | `sp.wooldridge_did`, `sp.etwfe`, `sp.etwfe_emfx`, `sp.drdid`, `sp.twfe_decomposition` | Wooldridge ETWFE + DR-DiD + TWFE diagnostics | Production. |
| `honest_did.py` | 10k | `sp.honest_did`, `sp.breakdown_m` | Rambachan-Roth (2023) sensitivity | Production. Guide `honest_did.md`. |
| `bacon.py` | 9k | `sp.bacon_decomposition` | Goodman-Bacon (2021) decomposition | Production. |
| `ddd.py` | 9k | `sp.ddd` | Triple difference | Production. |
| `event_study.py` | 13k | `sp.event_study` | Event-study utility | Production. |
| `did_2x2.py` | 9k | `sp.did_2x2` | Canonical 2×2 DID | Production. |
| `did_multiplegt.py` | 24k | `sp.did_multiplegt` | dCDH (2020) DID_M **only** | **Partial** — see §3. |
| `continuous_did.py` | 14k | `sp.continuous_did` | Dose-quantile DID + TWFE + lpoly dose-response | **Partial** — see §2. |
| `cic.py` | 11k | `sp.cic` | Changes-in-changes | Production. |
| `stacked_did.py` | 14k | `sp.stacked_did` | Stacked DID | Production. |
| `cohort_anchored.py` | 9k | `sp.cohort_anchored` | Cohort-anchored DID | Production. |
| `aggte.py` | 18k | aggregation layer | ATT aggregation | Shared primitive. |
| `analysis.py` | 14k | `sp.did_analysis`, `DIDAnalysis` | Unified workflow | Production. |
| `harvest.py` | 15k | `sp.harvest_did` | — | Production. |
| `pretrends.py` | 25k | — | Pretrend toolkit | Shared primitive. |
| `report.py` | 34k | `sp.cs_report` | LaTeX / Word / Excel output | Shared. |
| `plots.py` | 49k | `sp.parallel_trends_plot`, `sp.bacon_plot`, `sp.group_time_plot`, `sp.did_plot` | Plotting | Shared. |
| `summary.py` | 25k | — | Summary helpers | Shared. |
| `bjs_inference.py` | 7k | — | BJS analytical variance | Shared. |
| `design_robust.py` | 6k | `sp.design_robust` | — | 2026-04-23 new. |
| `did_bcf.py` | 7k | `sp.did_bcf` | Bayesian causal forest DID | 2026-04-23 new. |
| `misclassified.py` | 10k | `sp.did_misclassified` | Measurement-error DID | 2026-04-23 new. |
| `overlap_did.py` | 7k | `sp.overlap_did` | Overlap diagnostic | 2026-04-23 new. |

### Registry coverage

Before today's edit, `continuous_did` and `did_multiplegt` were falling through to the auto-registration pass — they lacked hand-written `FunctionSpec` entries with agent-native metadata (`assumptions`, `failure_modes`, `alternatives`). Fixed in this commit: both now carry full specs at `registry.py:5374` (+190 LOC, no numerical path touched).

### Test coverage

| File | Tests | Scope |
|---|---|---|
| `tests/test_did.py` | core CS / TWFE / 2x2 | Core. |
| `tests/test_did_advanced.py` | aggregation, covariates | CS + DR-DiD. |
| `tests/test_did_frontiers.py` | 2026-04-23 additions | BCF / cohort-anchored / design-robust / misclassified. |
| `tests/test_did_multiplegt_joint.py` | dCDH 2020 + 2024 joint placebo + avg cumulative | Good — 5 tests. |
| `tests/test_did_numerical_fixtures.py` | numeric anchors | Anchor regression tests. |
| `tests/test_did_summary.py` | result object rendering | Output layer. |
| `tests/test_new_v06_modules.py:487` | `test_continuous_did_twfe` (ONE test) | **Gap — only TWFE mode covered; att_gt + dose_response modes untested.** |
| `tests/test_overlap_did.py` | overlap | New. |
| `tests/test_sp_did_aggregation.py` | aggregators | Shared. |
| `tests/test_harvest_did.py` | harvest | — |
| `tests/test_bayes_did.py`, `test_bayes_did_cohort.py` | bayes variant | — |
| `tests/test_honest_did_aggte.py`, `test_honest_did_sdid.py` | honest | — |
| `tests/test_sequential_sdid.py` | sdid variant | — |

**Concrete coverage gaps**:
- `continuous_did(method='att_gt')` — no test exercises the dose-quantile DID path.
- `continuous_did(method='dose_response')` — no test exercises the lpoly fallback path or the primary lpoly path.
- `did_multiplegt(controls=...)` — the residualisation branch in `_residualize` has no dedicated test.
- `did_multiplegt` reference parity — nothing compares output to the R `DIDmultiplegtDYN` package or Stata `did_multiplegt_dyn`. Unit structure tests exist; numerical parity does not.

## 2. `sp.continuous_did` — what it is vs. what CGS (2024) is

### What `continuous_did` currently computes

- **`method='twfe'`**: OLS on demeaned `y ~ dose × post + controls` with optional one-way cluster SE. Just a TWFE regression with a continuous treatment × post interaction. Reports one scalar `β`.
- **`method='att_gt'`**: Bins baseline dose into `n_quantiles` quantiles, drops dose == 0 as a control, takes the 2×2 DID mean-diff for each dose bin vs. the untreated arm, bootstraps SE, aggregates via a sample-size-weighted mean.
- **`method='dose_response'`**: Unit-level ΔY = Y_post − Y_pre regressed nonparametrically on baseline dose via `sp.lpoly`. Reports the average derivative over the grid.

### What Callaway-Goodman-Bacon-Sant'Anna (2024) actually identifies

[待核验 — CGS 2024 (NBER WP 32117) §3; will confirm against the PDF] the headline estimands are:

- **ATT(d|g,t)** = E[Y_t(d) − Y_t(0) | G=g, D=d] for treated with dose `d` in cohort `g` at time `t` — a level effect.
- **ACRT(d|g,t)** = d/dd ATT(d|g,t) — an average causal response on the treated, a slope effect.
- Aggregation over `g` and `t` yields overall ATT(d) and ACRT(d) curves.

Identification requires "**strong parallel trends**" (E[Y_t(d) − Y_{t-1}(d) | G, D=d] constant across cohorts — stronger than standard PT), and the paper gives a DR-style estimator with a specific influence function for asymptotic variance.

### Gap summary — continuous_did vs CGS 2024

| Dimension | Current | Needed for CGS 2024 |
|---|---|---|
| Estimand | Weighted mean of 2×2 DIDs by dose quintile | ATT(d\|g,t) level curve + ACRT(d\|g,t) slope curve |
| Identification | Implicit standard PT by dose bin | Strong PT explicit; diagnostic/test |
| Control group | dose==0 (auto-fallback to lowest quantile) | Never-treated OR not-yet-treated with explicit dose=0 baseline |
| Covariates | Additive in TWFE only | DR-DiD with outcome regression + propensity score in dose |
| Inference | Pairs bootstrap, no clustering | Analytical influence-function variance, multiplier bootstrap, clustering |
| Aggregation | Sample-weighted mean | Cohort- and time-weighted per CGS; overall vs. simple averages |
| Dose modelling | Empirical quantiles | Continuous — no binning required |
| Pre-trends | None | Pre-trend test on ATT(d\|g,t) with t < g |
| Diagnostics | None | Overlap of dose across cohorts; support of dose distribution |

**Conclusion**: the current function name is aspirational. The `att_gt` mode is closer to a "dose-bin DID" heuristic than to CGS 2024. To land the real thing without breaking users, the proposed move is:

1. Keep `sp.continuous_did(method='att_gt'|'twfe'|'dose_response')` as "heuristic" modes and rename internally with deprecation path (e.g., `method='att_gt'` → `method='dose_bin'` with a `DeprecationWarning`).
2. Add `method='cgs'` (default once tested) implementing ATT(d|g,t) + ACRT(d|g,t) per CGS (2024), with its own analytical inference.
3. Gate the rename behind a **reference-parity test suite** against the R `contdid` package (if it exists at release time).

## 3. `sp.did_multiplegt` — what it is vs. `did_multiplegt_dyn` (dCDH 2024)

### What `did_multiplegt` currently computes

- Consecutive-period switcher-vs-stayer cell DIDs, weighted by switcher count per cell.
- Sign flipped for "off-switchers" so the returned estimand is always the effect of treatment (not of switching).
- Optional first-difference residualisation on controls.
- `placebo=L` adds L placebo lags; `dynamic=H` adds H horizons, both via the same switcher structure.
- Cluster bootstrap for SE (relabels cluster IDs to avoid collision from resampling with replacement).
- Joint placebo Wald test + average cumulative dynamic effect summary (flagged as dCDH 2024).

This implements the **dCDH (2020) DID_M estimator**, plus the 2024 joint-placebo / avg-cumulative overlay. It is a *consecutive-pair* estimator.

### What `did_multiplegt_dyn` (dCDH 2024, Stata / R) actually does

[待核验 — dCDH 2024 *ReStat* §2-3 (DOI 10.1162/rest_a_01414 verified via `paper.bib`); arXiv working-paper version number TBD; package docs] the `_dyn` package is a different animal:

- It targets **event-study intertemporal effects** `δ_l` = E[Y_{F+l}(1) − Y_{F+l}(0)] where `F` is the first-switch time, for each horizon `l ≥ 0`, and **keeps "not-yet-treated" units as controls at each horizon** rather than pair-wise.
- It allows controls + weights entering the identification, with a specific **long-difference influence function** that is **not** a mechanical roll-up of consecutive-period DIDs.
- It handles **treatment reversal** (switch on → off → on) via a "stable treatment" restriction on control groups at each horizon.
- It carries its own **joint test** of dynamic + placebo effects with a covariance structure that is Wald-style over the horizons, not the pair-level averaging currently shipped.
- The **heteroskedasticity-robust weights** variant (2022 survey) adjusts cell weights away from the plain `n_switchers / total` rule.

### Gap summary — did_multiplegt vs did_multiplegt_dyn

| Dimension | Current (`did_multiplegt`) | Needed for `did_multiplegt_dyn` parity |
|---|---|---|
| Unit of identification | consecutive period pair | long difference from F−1 to F+l |
| Event-study horizons | yes (via `dynamic=`) but from pair rollup | direct long-difference per horizon |
| Control group per horizon | stayers at that pair | "not-yet-treated at horizon l" (stable in l) |
| Inference | cluster pairs bootstrap | influence-function variance per horizon + joint |
| Reversal handling | sign-flip on off-switchers in pair | explicit exclusion / separate estimand |
| Weights | `n_switchers / total` | heteroskedastic-weights variant available |
| Joint placebo test | Wald across placebo lags (shipped) | Wald across placebo + dynamic, joint covariance |
| Avg cumulative effect | mean of horizon estimates (shipped) | weighted per dCDH 2024 definition |

**Conclusion**: `sp.did_multiplegt` is the dCDH **2020** DID_M, which is a legitimate estimator and should stay. But the post-2021 literature's go-to estimator — and what users asking for "did_multiplegt_dyn" expect — is the 2024 event-study version. Proposed move:

1. Keep `sp.did_multiplegt` as-is (dCDH 2020 DID_M + joint placebo + avg cumulative).
2. Add `sp.did_multiplegt_dyn(...)` as a **separate** public function implementing the 2024 event-study estimator with its own influence-function variance.
3. Add `sp.did_multiplegt(..., weights='heteroskedastic')` as an extension once validated.

Both additions must land with reference-parity tests against the R `DIDmultiplegtDYN` package output on the canonical example data.

## 4. 其他 DiD 前沿（当前未覆盖）

These are on the radar but **not** part of the proposed v1.7 sprint:

- **Local-projections DiD (LP-DiD)**: Dube, Girardi, Jordà, Taylor (2023, NBER WP). One-file estimator, relatively cheap to add. Could slot in `did/local_projections_did.py`.
- **Triple-difference heterogeneity-robust** (Olden & Møen 2022, Strezhnev 2023): DDD without negative weights. Current `sp.ddd` is the textbook version.
- **Repeated cross-sections with covariate shifts**: Sant'Anna & Zhao (2020) DR-DiD already there; Caetano-Callaway-Payne-Rodrigues (2022) time-varying covariates is not.
- **Long-difference / spatial-temporal DiD**: outside scope.
- **Matrix completion / synthetic DiD** (Arkhangelsky et al. 2021, Athey et al. 2021): lives in `synth/`, not `did/`.

## 5. 建议的 v1.7 DiD sprint 抓手

In priority order, with honest scoping:

1. **`sp.did_multiplegt_dyn`** — *highest return*, matches external demand cleanly, has a public R reference package for parity. **Requires**: paper PDF in hand (dCDH 2024 ReStat), reference-parity fixtures from R `DIDmultiplegtDYN`, analytical variance. Estimated: 2-3k LOC + parity tests.
2. **`sp.continuous_did(method='cgs')`** — adds CGS (2024) ATT(d|g,t) + ACRT(d|g,t) alongside the existing heuristic modes, plus a deprecation-warning path. **Requires**: paper PDF, strong-PT diagnostic, influence-function variance, overlap diagnostic, reference-parity vs. R `contdid` if available. Estimated: 1.5-2k LOC.
3. **Reference-parity tests** for the existing `sp.did_multiplegt` — compares current output against the R 2020-version package on the canonical dCDH (2020) replication dataset. Low LOC, high trust return. This is a natural **pre-step** to (1).
4. **Fill test gaps** for `continuous_did(method='att_gt' | 'dose_response')` — at minimum one test per mode hitting the numerical path, plus one edge-case test for no-untreated / single-quantile data.

## 6. 今晚已落的（安全、非数值）改动

- `src/statspai/registry.py` — added hand-written `FunctionSpec` for `continuous_did` and `did_multiplegt` (lines 5374 onwards). Agent-card metadata (`assumptions`, `failure_modes`, `alternatives`, `typical_n_min`) included. **No numerical path modified. 41 existing DiD tests pass.**
- `docs/rfc/README.md`, `docs/rfc/did_roadmap_gap_audit.md` (this file), `docs/rfc/continuous_did_cgs.md`, `docs/rfc/multiplegt_dyn.md`.

## 7. Open questions for the next session

1. **Paper versions**: which version of CGS 2024 and dCDH 2024 do we lock to? (NBER WP revisions? ReStat accepted version?) The RFCs use `[待核验]` until you confirm.
2. **Reference-parity data**: do we want the canonical R-package example datasets in `tests/fixtures/`, or just generate synthetic data where we know the truth?
3. **Naming**: `sp.did_multiplegt_dyn` vs. `sp.did_multiplegt(..., version='dyn')`? Either works; the first keeps parity with user muscle memory from Stata/R.
4. **Deprecation policy**: if we make `method='cgs'` the default in `sp.continuous_did`, is a one-minor-version deprecation window enough, or do we keep the heuristic default longer?
