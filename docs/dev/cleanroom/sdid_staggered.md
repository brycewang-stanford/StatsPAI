# Clean-room specification: staggered SDID inference (`sp.sdid(..., treat=...)`)

Audit row H26: `synth/_sdid_staggered.py` described itself as following
Stata `sdid.ado` (GPL-3; `projected()`, the jackknife branch, the bootstrap
redraw rule, the refusals). `_project`, `_refit`, `_jackknife`,
`_resampling_se` and `_check_se_method` were deleted by AST span without
being read and rewritten; the module docstring was replaced unread; the
first lines of the deleted docstrings (shown by the redaction helper) said
"beta from a two-way FE regression on the ...", "fixed-weight
leave-one-unit-out jackknife", "bootstrap or placebo SE: sqrt((B-1)/B) *
sd over replications" and "one bootstrap / placebo replication
(synthdid(data, inference=1, ...))". `_build_panel`, `_fit` and the entry
point were kept (they carry the point estimate, matched to Stata at 1e-10).

## Sources (and only these)

- Arkhangelsky, Athey, Hirshberg, Imbens and Wager (2021)
  `[@arkhangelsky2021synthetic]`, arXiv 1812.09970: Algorithms 2-4 and
  Appendix A.
- Clarke, Pailanir, Athey and Imbens (2024) `[@clarke2024synthetic]`,
  arXiv 2301.11859: Section 2.2 (projected covariates, after Kranz 2022),
  Section 2.3 (staggered estimator, eq. 7, and the staggered Algorithms
  2-4).
- The `sdid` help file (`sdid.sthlp`): requirements of each `vce()`.
- Black-box runs: the committed Stata fixture `sdid_stata.json`, and the
  previous StatsPAI functions run on crafted inputs.

## Specification

- **Projected covariates**: `beta` from OLS of the outcome on the
  covariates with unit and period fixed effects on the **never-treated
  units** (black box: 0.09995773487 against Stata's 0.09995773487; fitting
  on all untreated observations gives 0.243). Every unit's outcome becomes
  `Y - X beta`.
- **Jackknife** (Algorithm 3, staggered form): for each unit `i`, every
  cohort effect is recomputed without `i` with the full-sample weights
  (unit weights restricted to the remaining controls and rescaled to sum
  to one, time weights unchanged); projected covariates are re-estimated
  without `i`; the cohort effects are aggregated with eq. (7) weights of
  the remaining sample. `V = (N-1)/N sum_i (ATT^(-i) - mean ATT^(-.))^2`
  over all `N` units, i.e. centred at the mean of the leave-one-out
  estimates. Black box: centring at the full-sample estimate (as Algorithm
  3 is printed) misses Stata's per-cohort SEs by up to 8e-6; centring at
  the mean reproduces Stata and the previous function to 1e-15, including
  on crafted (non-optimal) weights.
- **Bootstrap** (Algorithm 2): `N` units drawn with replacement
  (`rng.integers(0, N, size=N)`); draws without treated or without
  never-treated units are discarded and drawn again; each replicate refits
  Algorithm 1 in full. **Placebo** (Algorithm 4): on the never-treated units,
  the first `N_tr` entries of `rng.permutation(N_co)` receive the treated
  units' adoption dates (panel order). SE = standard deviation of the `B`
  replicate ATTs with divisor `B`; per-cohort SEs over the replicates in
  which the cohort appears. These draw conventions reproduce the previous
  seeded streams (black box: identical replicates for `method='did'`).
- **Refusals** (help file; error 451 in Stata's fixture): bootstrap needs
  more than one treated unit; jackknife needs at least two treated units
  in every cohort; placebo needs more never-treated than treated units.

## Acceptance

`test_sdid_staggered_parity.py` (point estimates, projected covariates,
jackknife SEs at 1e-10 against Stata for sdid / did / sc and with
covariates, refusal codes, resampling screens) passes. Bootstrap and
placebo SEs are an S-level screen: with `method='did'` they equal the
previous function's to 1e-15; with `sdid` / `sc` replicate weights are now
solved from scratch (the previous replicate function took the
full-sample weights as arguments), which moves the seeded SEs by 0.01%-1.3% (per-cohort
up to 2.8%).
