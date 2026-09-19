# Phase 3 — `did_synth` family

Worktree `pc-did-synth`, nothing committed. The work ran as one lead line plus six sub-clusters. Each
sub-cluster wrote its own detailed report. This file contains the full results for the lead's
functions (aliases, synthdid, TWFE weights, `breakdown_m`) and a consolidated table for everything else.
For defect write-ups, before/after numbers, promotion dict literals and CHANGELOG / MIGRATION text of
the sub-clusters, see:

| cluster | report | functions |
| --- | --- | --- |
| shiftshare | [`did_synth_shiftshare.md`](did_synth_shiftshare.md) | `shift_share_se`, `bartik`, `ssaggregate` |
| scpi | [`did_synth_scpi.md`](did_synth_scpi.md) | `scdata`, `scest`, `scpi` |
| mc | [`did_synth_mc.md`](did_synth_mc.md) | `mc_panel`, `mc_synth` |
| synthvar | [`did_synth_synthvar.md`](did_synth_synthvar.md) | `staggered_synth`, `robust_synth`, `demeaned_synth`, `discos` |
| didvar | [`did_synth_didvar.md`](did_synth_didvar.md) | `continuous_did`, `distributional_did`, `did_timevarying_covariates` |
| misc | [`did_synth_misc.md`](did_synth_misc.md) | `spillover_did`, `harvest_did`, `causal_impact` |

## 1. Per-function table

Outcome classes: 1 aligned/bit-exact · 2 defect fixed then aligned · 3 convention · 4 reference wrong ·
5 stochastic (T3) · 6 no canonical reference / different estimand · open.

### Lead line (this file)

| function | reference | class | max rel err est / SE | test |
| --- | --- | --- | --- | --- |
| `bjs`, `borusyak_jaravel_spiess` | alias of `sp.did_imputation` (Track A `16_bjs`, `didimputation::did_imputation`) | 1 (alias proof) | 0 / 0; they are the same function object | `tests/reference_parity/test_track_a_alias_equivalence.py` |
| `did_2stage` | alias of `sp.gardner_did` (Track A `73_did2s`, `did2s::did2s`) | 1 (alias proof) | 0 / 0; same object | same |
| `synthdid_estimate` | alias of `sp.sdid(method='sdid')` (Track A `12_sdid`) + direct `synthdid::synthdid_estimate` 0.0.9 | 1 | est 1.5e-15, omega/lambda ≤3.3e-15 abs, jackknife SE 2.2e-15 | alias file + `test_did_synth_R_parity.py` |
| `sc_estimate` | `synthdid::sc_estimate` 0.0.9 | 2 (SE) → 1 | est 8.1e-15, omega 7e-15 abs; jackknife SE 4.8e-15; placebo draws 2.1e-11; bootstrap draws 1.4e-14 | `tests/reference_parity/test_did_synth_R_parity.py` |
| `did_estimate` | `synthdid::did_estimate` 0.0.9 | 2 (SE) → 1 | est 3.1e-15; jackknife SE 5.7e-16; placebo draws 1.9e-13; bootstrap draws 3.1e-15 | same |
| (`sdid` SE, all three methods) | `synthdid::synthdid_se` / `vcov` | 2 → 1 per draw; end-to-end placebo/bootstrap SE is 5 (T3) | see D1 | same |
| `staggered_cs`, `staggered_sa` | `staggered::staggered_cs` / `staggered_sa` 1.2.2 | 1 (already done; bookkeeping only) | 6.6e-14 / 5.7e-14 over 18 dataset×estimand×wrapper cells; Track A `82_staggered` rows `cs_simple`/`sa_simple` ≤1.1e-15 | `tests/reference_parity/test_staggered_extended_parity.py`, `tests/r_parity/82_staggered.*` |
| `breakdown_m` (smoothness, covariance available) | `HonestDiD::findOptimalFLCI` 0.2.8, breakdown by `uniroot` | 2 → 1 vs HonestDiD-with-exact-quantile; 3 (aligned) vs shipped HonestDiD | 1.3e-11 / 1.6e-12 / 6.1e-11 (e = 0, 1, 2) vs exact-quantile HonestDiD; 4.0e-4 / 4.0e-4 / 4.3e-4 vs shipped | `test_did_synth_R_parity.py` |
| `breakdown_m` (relative magnitudes / no covariance) | none: it inverts `honest_did`'s native approximate interval | 6 (closed form pinned; warns) | — | `tests/external_parity/test_honest_did_paper_parity.py` |
| `twfe_decomposition` | `bacondecomp::bacon` 0.1.1, `TwoWayFEWeights::twowayfeweights(type="feTR")` 2.1.0 | **open: defect found, not fixed** (the file is outside this line's ownership). `model_info['twfe_beta']` is right (1e-15). Headline, Bacon rows and "dCDH" weights are wrong. A reference-exact replacement primitive exists | primitive: cells 3.8e-14 / beta 7.6e-15 / summary measures ≤8.5e-14 on 3 datasets | `test_did_synth_R_parity.py` (strict xfail on the public function) |

### Sub-clusters (details in their reports)

| function | reference | class | headline agreement |
| --- | --- | --- | --- |
| `ssaggregate` | R `ShiftShareSE` 1.1.0, R `ssaggregate` (kylebutts@22df939), Stata `ivreg_ss`/`reg_ss`, Stata SSC `ssaggregate` 1.2.2 | 2 → 1 | ≤3e-13 vs R and Stata |
| `shift_share_se` | `ShiftShareSE::ivreg_ss`, Stata `ivreg_ss` | 2 → 1 | 3e-15 / 6e-14 |
| `bartik` | `AER::ivreg`+`sandwich`, Stata `ivregress`; `bartik.weight`, Stata `bartik_weight` | 1 (2SLS); 2 → 1 (Rotemberg weights added) | 5e-15; 2.5e-12 |
| `scdata` | `scpi::scdata` 4.0.1 | 1 | identical matrices |
| `scest` | `scpi::scest` 4.0.1 | 2 → aligned (ols/lasso 1e-12 to 1e-14; simplex/ridge/L1-L2 bound by R's solver gap, weights 2e-6 to 1.4e-5) | see report |
| `scpi` | `scpi::scpi` 4.0.1 | 2 → aligned with R draws and a tight solver (5e-12 / 4e-6); default end-to-end T3/T4 | see report |
| `mc_panel`, `mc_synth` | MCPanel@6b2706fd, `fect(method="mc")` 2.4.1, `gsynth` 1.4.0 | 2 → 1 at fixed lambda (2.6e-13 / 6.0e-13); SE and CV lambda: 6 / open | see report |
| `staggered_synth` | `augsynth::multisynth` 0.2.0 | 2 → 1 | ATT 5.6e-12, jackknife SE 1.8e-11 |
| `robust_synth` | `scpi::scest(ols)`, `lm`, `glmnet` 4.1.10 | 2 → 1 (regression paths); placebo SE 6 | ≤3.4e-11 |
| `demeaned_synth` | `augsynth(progfunc="None", fixedeff=TRUE)` | 2 → aligned (OSQP eps 1e-8 in augsynth) | 1.2e-9; `variant="detrended"` 6 |
| `discos` | `DiSCos::DiSCo` 0.1.4 | 2 → 1 (quantile); aligned vs DiSCo's SCS (mixture); permutation test 4 (DiSCos bug) | 1.9e-12 |
| `continuous_did` | `fixest::feols` 0.14.0 (`method="twfe"`) | 2 → 1 | 1.1e-15 / 2.2e-15; `att_gt` / `dose_response` modes 6 |
| `distributional_did` | `didFF::distDD` 0.1.0 | 1 (evidence already existed; 10 new configurations) | ≤1e-9 |
| `did_timevarying_covariates` | `ptetools::pte_default` 1.0.1, `did::att_gt` 2.3.0 | 2 → 1 (point estimates); bootstrap SE 5 | 3.8e-15 |
| `spillover_did` | `did::att_gt`+`aggte`, `fixest::feols` (Butts' regression) | 2 → 1 | ≤1e-14 |
| `harvest_did` | `did::att_gt(notyettreated, universal)` + `aggte(dynamic)` | 2 → 1 (cells, `n_treated` weighting); default `precision` aggregate 6 | 1.7e-14 |
| `causal_impact` | — (R `CausalImpact` is a different model: Bayesian local level + spike-and-slab via MCMC, with seed hard-coded) | 6; its actual computation is pinned to `KFAS::KFS` at 4.8e-15 | — |

**Counts over the 28 assigned functions.** Each function is counted once, at its headline grade;
parts that remain T3, T4 or 6 are listed in §5.

- **Class 1, correct as found (8):** `bjs`, `borusyak_jaravel_spiess`, `did_2stage`,
  `synthdid_estimate`, `staggered_cs`, `staggered_sa`, `scdata`, `distributional_did`.
- **Class 2, defect fixed and then aligned or bit-exact (18):** `sc_estimate`, `did_estimate`,
  `breakdown_m`, `ssaggregate`, `shift_share_se`, `bartik`, `scest`, `scpi`, `mc_panel`,
  `mc_synth`, `staggered_synth`, `robust_synth`, `demeaned_synth`, `discos`, `continuous_did`,
  `did_timevarying_covariates`, `spillover_did`, `harvest_did`. Also the three `sdid`-family SEs,
  counted under `sc_estimate` and `did_estimate`.
  - `bartik`: the 2SLS was already right; it gained Rotemberg weights.
  - `scest` and `scpi`: aligned, not bit-exact.
  - `demeaned_synth`: aligned.
- **Class 6 (1):** `causal_impact`.
- **Open (1):** `twfe_decomposition`. The defect is located and a reference-exact primitive is
  ready; the fix sits outside this line's file ownership.

## 2. Defects (lead line)

### D1 — `sp.sdid` / `sc_estimate` / `did_estimate` / `synthdid_estimate`: none of the three SEs was synthdid's (⚠️ correctness, default output)

- **What.** The docstring said the SEs match `synthdid::vcov`. Reading `synthdid:::placebo_se`,
  `bootstrap_sample`, `jackknife_se` and `vcov.synthdid_estimate` (0.0.9) showed they did not:
  - **placebo (default):** we re-estimated each control as placebo-treated from scratch. That meant a
    uniform start, a noise level recomputed on the placebo panel, and always exactly one placebo unit
    whatever `N1` was. The SE used `ddof=1`, and `n_reps` and `seed` were ignored. R instead draws
    `replications` random permutations. The last `N1` controls play treated. Frank-Wolfe is warm-started
    from the renormalised full-sample omega and lambda, with the full-sample `zeta.omega`,
    `zeta.lambda` and `min.decrease` frozen (`attr(estimate, 'opts')`). R reports
    `sqrt((r-1)/r)*sd`.
  - **bootstrap:** we resampled controls only, holding the treated units fixed. R resamples all
    units, redraws degenerate draws, uses the same warm start and frozen opts, and returns NA with a
    single treated unit.
  - **jackknife:** we dropped controls only and re-estimated the weights. R's jackknife (Algorithm 3)
    is leave-one-unit-out over *all* units with the weights **held fixed**, and returns NA when
    `N1 = 1` or only one omega is non-zero. For `N1 = 1` we returned a number where R returns NA.
- **How found.** Source of the four R functions, then per-draw replay. The first divergence was in
  which rows are resampled and in the weight initialisation. With R's own permutation / bootstrap index
  vectors (40 draws each, both datasets, all three methods), per-draw estimates now agree to
  ≤2.1e-11. The jackknife agrees to ≤4.8e-15.
- **Fix.** `src/statspai/synth/sdid.py`:
  - New `_synthdid_opts`, `_synthdid_refit`, `_synthdid_placebo_theta`, `_synthdid_bootstrap_theta`
    and `_synthdid_se` mirror `vcov` exactly.
  - An undefined SE now returns NaN with a `UserWarning`, including `N0 <= N1` for placebo, where R
    stops. It no longer returns a fabricated number.
  - `_sc_weight_fw` now carries `A x` incrementally. This is the same iterates, and makes the
    200-replication default about 4× faster than a direct port.
  - `covariates=` was silently ignored and now raises `MethodIncompatibility`.
  - `model_info['n_reps']` is now recorded for placebo.
- **Before → after:**
  - Prop. 99 replica, placebo, seed 42: sdid 2.6041 → 2.5630 (R's own 200-draw run: 2.6266),
    sc 3.4510 → 3.4350, did 4.9324 → 5.0013.
  - Prop. 99 sdid jackknife: 0.7476 → NaN plus a warning. R returns NA.
  - Five-treated panel, jackknife: sdid 0.1248 → 0.4382, sc 0.2378 → 1.5116, did 0.3189 → 0.7810,
    each equal to R to 1e-14.
  - Five-treated panel, placebo, seed 11: sdid 0.4722 → 0.2049, sc 0.6836 → 0.2455,
    did 1.6608 → 0.7621.
  - Point estimates are unchanged (1e-15).
- **Default output changed:** yes, the SE / p-value / CI of every `sp.sdid` family call.
- **Integrator note:** Track A `12_sdid` records `se_native_placebo = 2.6040746521236526` in
  `tests/r_parity/results/12_sdid_py.json`. The row is diagnostic, not headline, but the Python-side
  reproduction (`verify_reproduce_py.py`) will now report 2.5630226212963496 for it. Regenerate it with
  the harness. I did not touch `tests/r_parity/*`.

### D2 — `sp.breakdown_m` ignored `method` and did not invert the confidence set it names (⚠️ correctness, default output)

- **What.** It returned `(|θ̂| − z·SE)/(e+1)` for every input: `method` was validated and then
  unused. That closed form inverts only `honest_did`'s covariance-free *fallback* interval, not the
  Rambachan-Roth FLCI that `honest_did` reports whenever the covariance is available. The
  external-parity test called it "Definition 2 of the paper", and its own docstring said
  M* = sup{M : 0 ∉ CI(M)}.
- **How found.** Reading the function. Then HonestDiD: `findOptimalFLCI` + `uniroot` on the zero-facing
  bound.
- **Fix** (`did/honest_did.py`, `did/_flci.py`):
  - The new `breakdown_m_sd` root-finds on the FLCI bound, using a cached `_FLCIProgram` so that
    b(h) is solved once per h.
  - `breakdown_m` routes smoothness with covariance to it. Without covariance it keeps the closed
    form and warns, exactly mirroring `honest_did`.
  - `relative_magnitude` inverts `honest_did`'s native RM interval `(|θ̂| − z·SE)/max|pre|` and
    warns.
  - The docstring example is unchanged.
- **Before → after:**
  - Hand-crafted event study: e=0 0.1512 → 0.15345, e=1 0.0967 → 0.06460, e=2 0.0772 → 0.03829.
    HonestDiD with the exact quantile gives 0.153453 / 0.064599 / 0.038287.
  - mpdta Callaway-Sant'Anna e=0: 0.019253 → 0.007990.
  - Same fit, `method='relative_magnitude'`: 0.019253 → 1.948.

### D3 — native FLCI: SLSQP false convergence froze the worst-case bias b(h) (⚠️ correctness, `honest_did` smoothness)

- **What.** `worst_case_bias(h)` started SLSQP at (u = 1, w uniform). Whenever the variance
  constraint was slack there, SLSQP stopped after one iteration and reported `success` at a
  non-optimal point. That froze b(h) at the start's objective (4.0 here) for all larger h, where
  HonestDiD's ECOS gives 3.60 … 3.00. The h search was cut short and the FLCI was too wide.
- **How found.** Comparing against HonestDiD at e = 1. The CI bound was 12% off at M = 0.02 and
  123% off at M = 0.05. Bisected to b(h) on HonestDiD's own h grid.
- **Fix:**
  - A second start at the minimum-bias estimator, keeping the better of the two converged solves.
  - Analytic Jacobians.
  - `special.ndtr` in the folded-normal quantile: the same function without the `stats.norm`
    overhead, about 15× faster overall.
  - A bounded Brent refinement of h after the grid scan. This removes a grid discretisation error of
    up to 7e-4.
- **Before → after:**
  - Fixture e=1, M=0.02 lower bound: 0.10497 → 0.11913 (HonestDiD exact quantile: 0.11913).
  - mpdta CS `honest_did(e=1, m_grid=[0.01, 0.02])` lower bounds: −0.081896 / −0.121360 →
    −0.078258 / −0.107401.
- **Also in `honest_did`:** the native path rounded `ci_lower`, `ci_upper` and `M` to 6 decimals
  (the "rounding returned values" defect class). Rounding is removed. This does not change the
  backend='honestdid' path.

### D4 — `sp.twfe_decomposition` is not the decomposition it names (**open**, fix outside ownership)

- **What** (on the Track A mpdta bytes, against `bacondecomp::bacon` and `TwoWayFEWeights`):
  - **Headline.** The headline `estimate` is −0.02718. The TWFE β is −0.03751, and the Bacon weighted
    sum is exactly β.
  - **"Earlier vs later" rows.** These use a `t ≥ g_early` window on the whole panel, not Bacon's
    sub-windows, so the estimates are wrong. For example 2004 vs 2006 gives −0.0148 where the
    reference is −0.0327.
  - **Weights.** All weights are 1/9, labelled "Simplified: proportional to n_units". Bacon's range
    from 0.033 to 0.2.
  - **SE.** The SE is an ad hoc dispersion measure.
  - **"dCDH" weights.** There are 7 cohort×period rows, all positive, from a "Simplified formula".
    TwoWayFEWeights gives 875 unit×period cell weights, 125 of them negative, summing to 1.133 and
    −0.133.
  - **Treated vs never.** These rows and `model_info['twfe_beta']` are correct.
- **What I did.** The body lives in `did/wooldridge_did.py`, which is excluded from this line, so I
  did not change it.
  - I added the exact primitive `src/statspai/did/_twfe_weights.py::dcdh_fe_weights`, a line-for-line
    port of `twowayfeweights_calculate` and `twowayfeweights_result`, `feTR`. It is bit-exact on 3
    datasets: balanced unit-level mpdta, an unbalanced grouped panel with within-cell partial
    rollout, and the same panel with weights.
  - I pinned the public function's gaps in a **strict xfail**
    (`test_twfe_decomposition_matches_bacon_and_twowayfeweights`). It flips to XPASS, and so to a
    failure, once the function is fixed.
- **Proposed fix for the owner:**
  - Bacon rows ← `sp.bacon_decomposition(data, y, treat=D, time, id)["decomposition"]`, which is
    already bit-exact via Track A `20_bacon`.
  - Headline ← `weighted_sum`, which equals β.
  - dCDH ← `dcdh_fe_weights(data, y, group, time, D)`, exposing `cells`, `nr_minus`, `sum_minus`,
    `sensibility` and `sensibility2`.
  - Drop the ad hoc SE, or document it as a heterogeneity diagnostic.
  - This changes default output: ⚠️ correctness.
- **Convention noted.** With observation weights, TwoWayFEWeights replaces a within-cell-varying D by
  its *unweighted* cell mean, because `normalize_var` runs before the weights are attached. The
  primitive follows it and documents it. The R side is run with fixest `fixef.tol = 1e-11`. At the
  default 1e-6 the reference carries 2.8e-8 of demeaning error, which the fixture records
  (`twfew_grouped_default_tol`) and a test bounds.

### Other corrections

`tests/external_parity/test_honest_did_paper_parity.py` described the fallback closed form as
"Definition 2 of the paper". Its docstring now says what the formula is: the inversion of the
covariance-free fallback interval. The docstring's References section now holds only the bib key
`rambachan2023more`, no hand-written citation. The pinned values are unchanged. Also,
`test_honest_did_sdid.py`, `test_sdid_backend.py` and `test_cov95_synth_scpi_discos_sdid.py` pinned
the old SE values or the silently-ignored `covariates`. They are re-pinned, and each is labelled as a
characterisation pin, not a reference.

## 3. Proposed promotion records

### Alias table (already applied in the worktree; not a forbidden file)

`src/statspai/_parity_taxonomy.py::TRACK_A_ALIASES` gained `bjs`, `borusyak_jaravel_spiess`,
`did_2stage` and `synthdid_estimate`. Their proofs are in
`tests/reference_parity/test_track_a_alias_equivalence.py`, which is also added to that file's
`proven` set; 14 tests pass. I dry-ran `build_parity_index.py` on a copy, then restored it. All four
come out `bit-exact` with source `track_a_alias`, and `tests/test_parity_index.py` passes 24/24 on
the regenerated snapshot. **The integrator must regenerate `_parity_index.json` / `docs/parity.md`.**
Until then, three `test_parity_index.py` tests fail as "stale snapshot".

### `staggered_cs` / `staggered_sa`: bookkeeping only

These are already compared directly: Track A `82_staggered` rows `cs_simple` / `sa_simple`, and
`test_staggered_extended_parity.py`, 18 wrapper cells at abs 1e-9 (observed rel ≤6.6e-14). They
showed as `analytical-only` only because the Track A README row lists just `sp.staggered_rollout`.
The preferred fix is to change `tests/r_parity/README.md` row 82's py_api cell to
`` `sp.staggered_rollout` / `sp.staggered_cs` / `sp.staggered_sa` ``, which the builder's
`_leaf_functions` picks up. The alternative is `_FROZEN_PROMOTIONS`:

```python
    "staggered_cs": {
        "status": "bit-exact",
        "reference": "staggered::staggered_cs 1.2.2 (Roth & Sant'Anna)",
        "reference_versions": {"R": "4.5.2", "staggered": "1.2.2"},
        "tolerance": "estimate, Neyman SE and adjusted SE at abs 1e-9 on mpdta, a randomised rollout and a null panel x simple/cohort/calendar (observed rel 6.6e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_staggered_extended_parity.py",
            "tests/reference_parity/_fixtures/staggered_extended_reference.json",
        ],
        "note": "Also Track A module 82_staggered row cs_simple (R and Stata). Plug-in weights (beta = 1), every not-yet-treated cohort as control, units treated in the first period dropped -- as R staggered_cs. Regenerate via _generate_staggered_extended_R.R.",
    },
    "staggered_sa": {
        "status": "bit-exact",
        "reference": "staggered::staggered_sa 1.2.2 (Roth & Sant'Anna)",
        "reference_versions": {"R": "4.5.2", "staggered": "1.2.2"},
        "tolerance": "estimate, Neyman SE and adjusted SE at abs 1e-9 on mpdta, a randomised rollout and a null panel x simple/cohort/calendar (observed rel 6.6e-14)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_staggered_extended_parity.py",
            "tests/reference_parity/_fixtures/staggered_extended_reference.json",
        ],
        "note": "Also Track A module 82_staggered row sa_simple (R and Stata). Plug-in weights with only the last-treated cohort as control -- as R staggered_sa. Regenerate via _generate_staggered_extended_R.R.",
    },
```

### New records (lead line)

```python
    "sc_estimate": {
        "status": "bit-exact",
        "reference": "synthdid::sc_estimate 0.0.9 (Arkhangelsky, Athey, Hirshberg, Imbens & Wager)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "synthdid": "0.0.9"},
        "tolerance": "estimate, unit/time weights and jackknife SE at 1e-9 on the Prop. 99 replica and a five-treated panel (observed est 8.1e-15, jackknife SE 4.8e-15); each of R's 40 placebo and 40 bootstrap replications replayed at 1e-9 (observed 2.1e-11)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_R_parity.py",
            "tests/reference_parity/_fixtures/did_synth_R.json",
        ],
        "note": "Placebo and bootstrap SEs are Monte-Carlo draws: the replication map is pinned draw for draw against R's recorded index vectors, the end-to-end seeded SE only within pooled Monte-Carlo error (T3). Fixed in the did_synth sweep: all three SE methods now follow synthdid::vcov (warm start, frozen regularisation, fixed-weight jackknife over all units). Regenerate via _generate_did_synth_R.R.",
    },
    "did_estimate": {
        "status": "bit-exact",
        "reference": "synthdid::did_estimate 0.0.9",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "synthdid": "0.0.9"},
        "tolerance": "estimate and jackknife SE at 1e-9 on the Prop. 99 replica and a five-treated panel (observed est 3.1e-15, jackknife SE 5.7e-16); R's 40 placebo and 40 bootstrap replications replayed at 1e-9 (observed 1.9e-13)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_R_parity.py",
            "tests/reference_parity/_fixtures/did_synth_R.json",
        ],
        "note": "Uniform weights reduce to the 2x2 difference in means, asserted reference-free at 1e-12. Placebo/bootstrap end-to-end SE is T3 (see sc_estimate). Regenerate via _generate_did_synth_R.R.",
    },
    "breakdown_m": {
        "status": "aligned",
        "reference": "HonestDiD::findOptimalFLCI 0.2.8 (Rambachan & Roth), breakdown by uniroot on the bound facing zero",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "HonestDiD": "0.2.8", "CVXR": "1.8.2"},
        "tolerance": "vs HonestDiD with its Monte-Carlo folded-normal quantile replaced by the exact one: 1e-9 (observed 6.1e-11); vs HonestDiD as shipped: 1e-3 (observed 4.3e-4), the simulation error of .qfoldednormal (1e6 draws, seed 0; 1.96224 vs exact 1.95996 at mu = 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_did_synth_R_parity.py",
            "tests/reference_parity/_fixtures/did_synth_honest_R.json",
        ],
        "note": "Grade applies to method='smoothness' with a recoverable event-study covariance (Callaway-Sant'Anna fits). method='relative_magnitude' and the covariance-free fallback invert honest_did's native approximate intervals (closed forms, T1, warned). Fixed in the did_synth sweep: breakdown_m ignored `method` and returned (|theta| - z SE)/(e+1) everywhere; the native FLCI's SLSQP stopped at its start point and froze the worst-case bias. Regenerate via _generate_did_synth_honest_R.R.",
    },
```

`synthdid_estimate` needs no record: the alias gives it `bit-exact` via `12_sdid`. Optionally add
`did_synth_R_parity.py` to its test list, since the direct comparison is also there. For
`twfe_decomposition`, **do not promote**. When the owner wires it to `dcdh_fe_weights` and
`bacon_decomposition` and the strict xfail flips, a record can cite `did_synth_twfew_R.json`
(TwoWayFEWeights 2.1.0, bacondecomp 0.1.1, fixest 0.14.0; observed ≤8.5e-14).

The sub-cluster records are in each report's §3. Copy them verbatim. Note these points:
- misc proposes `harvest_did` as bit-exact, but its default `precision` aggregate is class 6. I
  recommend `aligned` scoped in the note to the cell/event-study quantities, or a status with the
  headline excluded, as that report asks.
- misc proposes no record for `causal_impact`.
- scpi's `scpi` record must stay `aligned`, scoped to "R's weights + R's draws + tight solver".

## 4. Proposed CHANGELOG / MIGRATION (lead line)

**⚠️ Correctness**
- `sp.sdid` / `sp.synthdid_estimate` / `sp.sc_estimate` / `sp.did_estimate`: the placebo, bootstrap
  and jackknife standard errors now follow `synthdid::vcov`. Placebo draws `n_reps` random
  permutations; each replication is warm-started from the full-sample weights with frozen
  regularisation; the SD uses divisor r. The bootstrap resamples all units. The jackknife is
  leave-one-unit-out over all units with fixed weights. Previously all three differed. For example
  the five-treated fixture's jackknife SE was 0.125 and is now 0.438 (R: 0.438). An SE that synthdid
  leaves undefined is NaN with a warning; the jackknife with one treated unit used to return a
  number. Point estimates are unchanged.
- `sp.breakdown_m`: now inverts the confidence set `sp.honest_did` reports. For smoothness with
  covariance this is the FLCI (R HonestDiD to 6e-11 with an exact quantile). `method=` is now
  honoured. Previously it returned `(|θ̂| − z·SE)/(e+1)` for every input, for example 0.0193 → 0.0080
  on mpdta.
- `sp.honest_did(method='smoothness')`, native: fixed an SLSQP false convergence in the worst-case
  bias program that widened the FLCI (up to 12% on a bound at M = 0.02). The best h is now refined
  beyond the grid. CI bounds and M are no longer rounded to 6 decimals.

**Fixed**
- `sp.sdid(covariates=...)` raised nothing and ignored the covariates; it now raises
  `MethodIncompatibility`.

**Added**
- Private `statspai.did._twfe_weights.dcdh_fe_weights`, which is bit-exact with
  `TwoWayFEWeights::twowayfeweights(type="feTR")`. `sp.twfe_decomposition` does not use it yet.
- Alias proofs for `sp.bjs`, `sp.borusyak_jaravel_spiess`, `sp.did_2stage` and
  `sp.synthdid_estimate`.

**MIGRATION rows**

| function | what changes | old → new (example) | keep old? |
| --- | --- | --- | --- |
| `sp.sdid` family | placebo / bootstrap / jackknife SE follow `synthdid::vcov`; undefined SEs are NaN | Prop. 99 placebo SE (seed 42) 2.6041 → 2.5630; five-treated jackknife 0.1248 → 0.4382 | no: the old SEs were not a documented quantity |
| `sp.breakdown_m` | inverts the FLCI; honours `method` | mpdta CS e=0: 0.01925 → 0.00799 | the old closed form is still returned when the covariance is unavailable (warned) |
| `sp.honest_did` native smoothness | FLCI bias program fixed; no rounding | mpdta CS e=1 M=0.02 lower −0.12136 → −0.10740 | no |

## 5. Not closed, and why

- **`twfe_decomposition`: open.** Fully located (D4), and a reference-exact primitive is ready. The
  wiring is blocked only by file ownership (`did/wooldridge_did.py`).
- **Stochastic parts (T3), by design:**
  - synthdid placebo / bootstrap SE end-to-end. For sdid and did the seeded 200-draw SE lies within 4
    pooled MC SEs of R's 200-draw SE. `sc` is skipped for run time; its per-draw map is exact.
  - `did_timevarying_covariates` bootstrap SE.
  - scpi simulated bounds with our RNG.
- **`breakdown_m` against the shipped HonestDiD** is `aligned` at 4.3e-4, not bit-exact. The
  mechanism is proven by the exact-quantile rerun. FLCI **bounds** agree with exact-quantile HonestDiD
  only to about 2e-5, while the half-length agrees to 1e-8. The residual is HonestDiD's derivative
  bisection over h, which stops at a step of (h0 − hMin)/100; the estimator's centre moves to first
  order in h. This is not a StatsPAI error, since our h is refined to 1e-12. Still, the bounds tolerance
  in the test is 5e-5, not 1e-6.
- **Native `honest_did(method='relative_magnitude')`** remains an approximation of the ARP
  conditional / hybrid set, and warns. There is no native reference-exact path.
  `breakdown_m(method='relative_magnitude')` inherits that.
- **Class 6:**
  - `causal_impact`: a different model from R CausalImpact.
  - `continuous_did` `att_gt` / `dose_response` modes.
  - `harvest_did` default `precision` aggregate.
  - `demeaned_synth(variant="detrended")`.
  - `mc_*` SEs and CV lambda.
  - `robust_synth` / `demeaned_synth` placebo SEs.
- **Stata sides not done:**
  - synthdid (Stata `sdid`, SSC). The R package is canonical and was not attempted.
  - `scpi` Stata: needs Python `scpi_pkg`, not installed; no return code recorded, so this stays open.
  - TwoWayFEWeights Stata (`twowayfeweights`, SSC): not attempted; R is the authors' package.
- **Reference defects to report upstream:**
  - DiSCos 0.1.4 `DiSCo_per_iter` leaves a zero quantile column (synthvar D5).
  - `ptetools` NSE bug with a cohort column named `g`.
  - `did` 2.3.0 drops never-treated units when `gname` is an integer column.
  - HonestDiD's `.qfoldednormal` Monte-Carlo quantile.
- **Integrator follow-ups in forbidden files:**
  - Regenerate `_parity_index.json`, `docs/parity.md` and the schemas. Signatures changed in
    `staggered_synth`, `discos`, `scest`, `mc_panel`, `mc_synth` and `did_timevarying_covariates`
    (`aggregation`).
  - `registry.py` text fixes listed in the didvar, misc and scpi reports.
  - The Track A `12_sdid` py-side diagnostic SE row (D1).
  - README row 82.
  - Also from the reports: `_article_aliases.py` `matrix_completion` doctest value;
    `shift_share_political_panel` shares the old AKM defect (`political.py`, out of scope).

## 6. `.gitignore`

```
tests/reference_parity/_fixtures/_ado_did_synth/
tests/reference_parity/_fixtures/_rlib_did_synth_scpi/
```

The first is the private Stata ado dir, shared by the did_synth clusters. The second is the private
R library for CVXR 1.9.2, which scpi 4.0.1 needs; the site library keeps 1.8.2 for HonestDiD, DiSCos
and synthdid. External installs that are **not** in the repo:
- `bartik.weight` (Makevars patched to build on R 4.5 / arm64).
- MCPanel@6b2706fd, with bundled Eigen removed to compile. It is GPL-2 and used as a test reference
  only, never a dependency.
- `ssaggregate` and `ptetools` from GitHub. SHAs are in the respective reports.

## Files (lead line)

- **Changed:** `src/statspai/_parity_taxonomy.py`, `src/statspai/synth/sdid.py`,
  `src/statspai/did/honest_did.py`, `src/statspai/did/_flci.py`,
  `tests/reference_parity/test_track_a_alias_equivalence.py`,
  `tests/external_parity/test_honest_did_paper_parity.py`, `tests/test_honest_did_sdid.py`,
  `tests/test_sdid_backend.py`, `tests/test_cov95_synth_scpi_discos_sdid.py`,
  `tests/test_cov95_synth_gapfill_r2.py` (a one-control jackknife used to pin SE = 0.0; it is now NaN plus a warning, as in synthdid).
- **New:** `src/statspai/did/_twfe_weights.py`,
  `tests/reference_parity/test_did_synth_R_parity.py` (48 passed, 1 strict xfail),
  `tests/reference_parity/_generate_did_synth_R.R`,
  `tests/reference_parity/_generate_did_synth_twfew_R.R`,
  `tests/reference_parity/_generate_did_synth_honest_R.R`,
  `tests/reference_parity/_fixtures/_generate_did_synth_data.py`,
  `_fixtures/did_synth_{sdid_panel,twfew_panel,honest_es}.csv`,
  `_fixtures/did_synth_{R,twfew_R,honest_R}.json`.
