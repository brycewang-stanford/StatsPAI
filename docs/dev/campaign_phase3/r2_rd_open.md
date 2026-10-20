# Phase 3 round 2, family `rd_open` — rd2d rebuild, discrete RD, open RD extras

Worktree `.claude/worktrees/r2-rd-open`, all changes unstaged / untracked on
top of the staged round-1 baseline. `PYTHONPATH=src`, `sp.__file__` inside
the worktree (checked).

New evidence files:

| file | role |
| --- | --- |
| `tests/reference_parity/_fixtures/_generate_rd_open_data.py` | writes `rd_open_bd.csv` (2,500-row boundary design, boundary `x2 = 0.3 x1`, 120 clusters, fuzzy take-up), `rd_open_bd_mass.csv` (same on a 0.05 lattice), `rd_open_discrete.csv` (integer ages 8..28, cutoff 18), `rd_open_extrap.csv` (Angrist–Rokkanen design; not yet used by a test, see §5) |
| `tests/reference_parity/_generate_rd_open_R.R` | writes `_fixtures/rd_open_R.json` (R 4.5, rd2d 1.0.0, RDHonest 1.0.1.9000, sandwich 3.1.1) |
| `tests/reference_parity/test_rd_open_R_parity.py` | 47 tests (rd2d location/distance/bw/WBATE/radial, boundary_rd, rd_discrete) — all pass |
| `src/statspai/rd/_rd2d_location.py`, `_rd2d_distance.py`, `_rd2d_plot.py` | new: port of R rd2d (location, distance) and the plot, split out of `rd2d.py` |

## 1. Per-function outcome

| function | reference | class | max rel err est / SE (bw) | test |
| --- | --- | --- | --- | --- |
| `rd2d` (`approach="location"`, new default) | R `rd2d::rd2d` 1.0.0 | 2 (rebuilt) | 1.6e-11 / 3.8e-12 (bw 1.2e-12, cov 8.1e-12) | `test_rd_open_R_parity.py` |
| `rd2d` (`approach="distance"`) | R `rd2d::rd2d.distance` 1.0.0 | 2 (rebuilt) | 1.5e-12 / 1.1e-12 (bw 1.4e-13, cov 7.8e-11) | same |
| `rd2d` multi-point headline | R `summary(fit, WBATE = w)` | 2 | ≤1e-9 asserted (equal and 1:5 weights, both approaches) | same |
| `rd2d` with `kernel_type="rad"` | R `rd2d(kernel_type="rad")` | 4 (reference defect, see §2) | estimates / bw / N exact to 1e-9; SE pinned to lm()+`sandwich::vcovHC` at 1e-10, R's SE reconstructed from ours | same |
| `rd2d` (`approach="pooled"`) | none (Keele–Titiunik one-score design = `sp.rdrobust` on signed distance to the boundary curve) | 6 (identity test vs `sp.rdrobust`) | 1e-9 identity | same |
| `rd2d_bw` | R `rdbw2d` / `rdbw2d.distance` (their own defaults) | 2 (rebuilt) | 5.0e-13 | same |
| `boundary_rd` | alias of `rd2d` | 2 (inherits) | as `rd2d` | same |
| `rd_discrete` (`method="bsd"`) | R `RDHonest::RDHonest(y ~ x)` | 2 (rebuilt) | fixed h/M ≤1e-9; selected h,M ≤1e-6 asserted (RDHonest optimiser, same tier as `rd_honest`) | same |
| `rd_discrete` (`method="bme"`/`"bm"`) | R `RDHonest::RDHonestBME` | 2 (rebuilt); RDHonestBME p-value: 4 | ≤1e-9 (est, SE, max bias, CI, one-sided CIs, leverage) | same |
| `rd_extrapolate` | none exact; point estimate = Stata `teffects ra` ATE (checked in scratch: 1.8977845563 both sides) | **defect found, open** (§2) | – | – |
| `rd_external_validity` | none | 6 | – | – |
| `rd_multi_extrapolate` | none (Cattaneo–Keele–Titiunik–Vazquez-Bare replication code is not a package; `rdmulti` has no extrapolation) | 6 + suspected defect (§5) | – | – |
| `rdit`, `rd_distribution`, `rd_distributional_design`, `rd_flex`, `rd_interference`, `rd_bayes_hte`, `rd_forest`, `rd_boost`, `rd_lasso`, `rd_cate_summary` | not checked this round (time) | open | – | – |

Counts: class 2 = 7 (rd2d location / distance / WBATE, rd2d_bw, boundary_rd, rd_discrete bsd, rd_discrete bme); class 4 = 2 (rd2d radial SE, RDHonestBME p-value); class 6 = 3 (rd2d pooled, rd_external_validity, rd_multi_extrapolate); open = 11 (rd_extrapolate defect + 10 unchecked).

## 2. Defects

1. **`sp.rd2d` / `sp.rd2d_bw` / `sp.boundary_rd` computed a different estimator (⚠️ correctness, default output changed).**
   Through 1.28.0 the default `approach="distance"` ran a home-grown local
   linear fit on the signed distance to the boundary *line*, with a Silverman
   /curvature rule-of-thumb bandwidth, conventional (not bias-corrected)
   inference, one pooled number, and `model_info['bandwidth']` rounded to 6
   decimals; `approach="location"` pooled pointwise fits by inverse-variance
   weights. R rd2d estimates **pointwise** effects `tau(b)` with the rdbw2d
   MSE/CER bandwidths and robust bias-corrected inference (estimate.q /
   std.err.q). [2026-10-21: the port below was replaced by a clean-room
   rewrite, licence audit row H9; see `docs/dev/cleanroom/rd2d.md`.]
   std.err.q). Rebuilt as a line-by-line port of R rd2d 1.0.0 (every routine
   named in the docstrings): sharp and fuzzy, product/radial kernels, HC0–3,
   clusters, joint/separate fits, all 8 `bwselect`, dpi/rot, masspoints,
   bwcheck, derivatives, tangent vectors, distance kinks (known / unknown),
   user distance matrices, WBATE headline with the cross-point covariance.
   Not ported: `covs.eff` covariate adjustment, simulated uniform bands /
   LBATE (Monte-Carlo).
   API changes: default `approach` `"distance"` → `"location"` (R's
   `rd2d`); `rd2d_bw` returns a DataFrame of per-point bandwidths instead
   of a float; new keyword-only options mirror R; the old one-score
   estimator survives as `approach="pooled"`, now delegated to
   `sp.rdrobust` on the signed distance to the boundary curve (a documented
   quantity, Keele–Titiunik), not the old home-grown fit.
   Before → after on `rd_open_bd.csv`: no like-for-like number exists (the
   old default reported one pooled effect; R reports five pointwise effects
   0.446 / 0.929 / 1.020 / 0.704 / 1.357 bias-corrected, WBATE 0.891,
   SE 0.0903), which is the point of the fix.
2. **R rd2d radial kernel variance (class 4, reference defect, not copied).**
   With `kernel_type="rad"` R fits at radius `sqrt(hx^2+hy^2)`
   (`rd2d_h_normalize`) but rescales the variance with `hx*hy`
   (`rd2d_fit`, `get_invH`), so every radial SE is inflated by
   `sqrt((hx^2+hy^2)/(hx hy))` = √2 when hx = hy. Independent evidence in the
   fixture: weighted `lm()` at that radius + `sandwich::vcovHC(HC0)` equals
   our SE to 1e-10 and R's SE / √2 to 1e-10 (`rad_lm_check`). StatsPAI
   scales with the radius actually used; the test reconstructs R's number
   from ours. Upstream report suggested (rdpackages/rd2d).
3. **`sp.rd_discrete` was not the Kolesár–Rothe procedure (⚠️ correctness, default output changed).**
   First divergence: the local-linear-on-bin-means variance used
   `var(bin)/n_bin` inside a sandwich already weighted by `n_bin`, so SEs
   were understated by ~√(bin size) (0.0113 vs RDHonest 0.1121 at h=5,
   M=0.05); `M` came from max second differences (CI [-34.5, 35.8] on the
   fixture with default M); the `'bm'` bound `K·Σ|w|` is not the BME
   interval. Rebuilt: `'bsd'` delegates to the RDHonest engine behind
   `sp.rd_honest`; `'bme'` (alias `'bm'`) ports `RDHonestBME` (new
   `order=`; `K=` deprecated/ignored with DeprecationWarning). Before →
   after (fixture, est / SE / CI): bsd M=.05 h=5: 0.7208 / 0.0113 /
   [0.105, 1.337] → 0.7619 / 0.1121 / [0.445, 1.079] (= RDHonest); bsd
   default: 0.6636 / 0.0083 / [-34.5, 35.8] → 0.8401 / 0.1466 / [0.464,
   1.216]; bm h=4: 0.7271 / 0.0130 → 1.4953 / 0.0549 / [0.640, 2.355].
   Also: `RDHonestBME`'s own p-value adds `maximum.bias` (outcome units) to
   a z statistic (class 4); we report the honest p-value with bias/SE and
   keep R's as `p_value_rdhonest` (asserted equal to R).
4. **`sp.rd_extrapolate` "CATE(x)" is flat in x (found, NOT fixed — open).**
   `rd_extrapolate(method='ols')` evaluates every point at the unconditional
   mean of Z, so `detail.cate` is identical at all x (1.89778 at all 3
   points on `rd_open_extrap.csv`) and equals the global RA ATE (Stata
   `teffects ra` 1.897784556341 — matches). Angrist–Rokkanen's CATE(x) is
   `E[Y1−Y0 | X=x]`, i.e. averages over `Z | X=x`; the function never
   extrapolates. Fix needs a conditional-on-x averaging step (e.g. local
   average of the fitted CATE(Z) around x); not done this round.

## 3. Proposed promotion records

```python
    "rd2d": {
        "status": "bit-exact",
        "reference": "R rd2d::rd2d / rd2d.distance 1.0.0 (Cattaneo, Titiunik & Yu)",
        "reference_versions": {"R": "4.5", "rd2d": "1.0.0", "sandwich": "3.1.1"},
        "tolerance": (
            "estimate.p/q, std.err.p/q, t, CI, cross-point covariance rel 1e-9 "
            "(observed 1.6e-11); bandwidths rel 1e-8 (observed 1.2e-12); N exact"
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_open_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_open_R.json"],
        "note": (
            "Location and distance approaches, sharp/fuzzy, clusters, HC0-3, joint/"
            "separate, 8 bwselect rules, kinks, mass points, WBATE headline. Radial-"
            "kernel SEs deliberately differ: R scales by hx*hy while fitting at radius "
            "sqrt(hx^2+hy^2) (pinned to lm()+sandwich instead). Through 1.28.0 a "
            "different, pooled estimator."
        ),
    },
    "rd2d_bw": {
        "status": "bit-exact",
        "reference": "R rd2d::rdbw2d / rdbw2d.distance 1.0.0",
        "reference_versions": {"R": "4.5", "rd2d": "1.0.0"},
        "tolerance": "per-point bandwidths rel 1e-8 (observed 5.0e-13)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_open_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_open_R.json"],
        "note": "With the selectors' own defaults (bwcheck 20, scaleregul 1). Returns a DataFrame since 1.29.0.",
    },
    "rd_discrete": {
        "status": "aligned",
        "reference": "R RDHonest::RDHonest / RDHonestBME 1.0.1.9000 (Kolesar)",
        "reference_versions": {"R": "4.5", "RDHonest": "1.0.1.9000"},
        "tolerance": (
            "estimate, std.error, maximum.bias, conf.low/high rel 1e-9 at fixed h "
            "and for BME; 1e-6 when RDHonest selects h and M (optimiser)"
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_open_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_open_R.json"],
        "note": (
            "method='bsd' = RDHonest, 'bme' = RDHonestBME. RDHonestBME's p-value adds "
            "the bias in outcome units to a z statistic; reproduced as "
            "p_value_rdhonest, headline uses bias/SE. Through 1.28.0 SEs were "
            "understated by ~sqrt(bin size)."
        ),
    },
```

`boundary_rd` is an alias of `rd2d` (alias proof via
`test_boundary_rd_alias_matches_R`).

## 4. CHANGELOG / MIGRATION drafts

- ⚠️ Correctness: `sp.rd2d`, `sp.rd2d_bw`, `sp.boundary_rd` rebuilt as a port
  of R `rd2d` 1.0.0 (pointwise boundary effects, rdbw2d bandwidths, robust
  bias-corrected inference, WBATE headline); bit-exact to R on 40 cells.
  Default `approach` is now `"location"`; old one-score design is
  `approach="pooled"` (via `sp.rdrobust`); `rd2d_bw` returns a DataFrame.
- ⚠️ Correctness: `sp.rd_discrete` now reproduces `RDHonest` (`'bsd'`) and
  `RDHonestBME` (`'bme'`/`'bm'`); previous SEs understated by ~√(bin size).
  New `order=`, `kernel=`, `opt_criterion=`; `K=` deprecated (ignored).
- Added: `rd2d(..., weights=)` (WBATE), `q`, `deriv`, `tangvec`,
  `kernel_type`, `vce`, `cluster`, `fuzzy`, `fitmethod`, `bwparam`,
  `method`, `masspoints`, `bwcheck`, `scaleregul`, `scalebiascrct`,
  `stdvars`, `kink_unknown`, `kink_position`, `cqt`, `distance`, `side`.

MIGRATION rows:

| function | what changed | how to get old behaviour |
| --- | --- | --- |
| `rd2d` / `boundary_rd` | estimand (pointwise, bias-corrected), default approach, bandwidth, SE | none exactly; `approach="pooled"` for a single one-score effect (now via rdrobust) |
| `rd2d_bw` | returns DataFrame of per-point bandwidths; values = rdbw2d | none |
| `rd_discrete` | estimate/SE/CI for both methods; `K` ignored; `model_info['discrete']` keys (`n_left`, `bin_means`, `honest_ci` removed; `maximum_bias`, `bandwidth`, `eff_obs` added) | none |

## 5. Not closed

- `rd_extrapolate`: defect 4 open (flat CATE(x)); bootstrap SE (seed 42,
  200 draws) is T3 at best; no Angrist–Rokkanen package exists (their
  replication code only). `rd_open_extrap.csv` is written for the future
  test.
- `rd_multi_extrapolate`: no package implements CKTV (2021) extrapolation
  (`rdmulti` 2.0.0 exports rdmc/rdmcplot/rdms only). Implementation fits a
  line through rdrobust cutoff estimates with a residual-variance SE and an
  ATE SE `sqrt(mean(se^2)/n)` that ignores covariance — suspect, not the
  CKTV estimator (which uses the parallel-trends-across-cutoffs
  identification on Y(0)); unreviewed in detail.
- `rd_external_validity`: diagnostic bundle, class 6.
- Unchecked this round: `rdit`, `rd_distribution`,
  `rd_distributional_design`, `rd_flex`, `rd_interference`,
  `rd_bayes_hte`, `rd_forest`, `rd_boost`, `rd_lasso`, `rd_cate_summary`.
  Candidates to try next: `rd_flex` vs `rdrobust(covs=)` with linear
  learner; `rd_lasso` vs hdm + rdrobust(covs); `rdit` vs rdrobust on a time
  running variable; the ML ones are stochastic (T3 at best).
- rd2d `covs.eff` and uniform bands / LBATE not ported.

## 6. Integrator checklist

- No `.gitignore` line needed (no Stata ado dir created by a committed file;
  a scratch Stata run lived in the session scratchpad).
- `registry.py`: `rd_discrete` ParamSpec — method choices `["bsd","bme","bm"]`,
  add `order`, `kernel`, `opt_criterion`; K description "deprecated,
  ignored". `rd2d` / `rd2d_bw` are auto-registered; re-dump schemas
  (signatures changed: many new keyword-only args, `rd2d_bw` return type).
- Existing tests updated to the new API: `test_cov95_rd_rd2d.py`,
  `test_cov95_rd_gapfill_r2.py`, `test_rd_new_modules.py`,
  `test_rd_polish.py`. Pass: those files' rd2d / discrete tests plus the
  47 parity tests. `test_rd_cov_estimators.py`, `test_rd_validation.py`,
  `test_tierD_rd_multiscore_analytic.py`, `test_rd_aliases.py`,
  `test_rd_dispatcher.py`, `test_agent_native_contract.py`,
  `test_cov95_rd_misc.py`: pass.
- R packages used: rd2d 1.0.0, RDHonest 1.0.1.9000, sandwich 3.1.1 (all
  already installed).
- Paper sync: rd2d moves from "open" to Track-B bit-exact.
