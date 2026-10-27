# Option scan, round 4: eighteen modules never scanned before

*2026-10-27. Coordinator session statspai-c7, seven worktree lines
(`wt/c7-*`), all on main by `82061fd5f`.*

## Why these modules

Rounds 1 to 3 (2026-10-09 to 10-14) scanned did, rd, synth, dml, matching,
regression, forest, panel, iv and timeseries for option combinations no test
had tried; they found about sixty defects, twenty-odd of them silent wrong
numbers. A count of scan test files per source package showed eighteen
estimation packages with none. Seven lines took them:

| Line | Packages |
| --- | --- |
| spatial | `spatial/`, `bartik/` |
| epi | `survival/`, `epi/`, `mendelian/` |
| partialid | `qte/`, `bounds/`, `selection/` (and `censoring/ipcw.py`) |
| postest | `postestimation/`, `inference/`, `robustness/` |
| survey | `survey/`, `multilevel/`, `gmm/` |
| mlcausal | `metalearners/`, `tmle/`, `conformal_causal/`, `policy_learning/` |
| flips | the defaults announced for 1.41 (not a scan) |

## Method

The brief is the one of round 1, written into a file every agent read:
for each documented option, is there a test where changing it changes the
result as documented? Ground truth is a reference implementation run as a
black box (R; Stata 18 through the MCP tools when decisive), a known-truth
DGP, or an identity (relabelling, row order, an equivalent call). Every
suspect was written first as a strict xfail test of the correct behaviour,
then run against an unchanged copy of the source to confirm it fails for
the stated reason, then fixed, with a before/after snapshot of the calls the
fix must not touch (bit-identical).

Phase 1 was read-only because another session was timing Track C; agents
read code and wrote draft tests without executing anything. That phase
alone produced most of the suspects, and almost all of them survived
execution.

## Results

About 115 defects confirmed and fixed; the CHANGELOG `[Unreleased]` section
lists each with a before/after number and its reference. The largest:

- three-level `sp.mixed`: BLUPs keyed by the class label alone, so classes
  repeated across schools merged (predictions off by up to 3.84);
- `sp.bartik`: shares paired with rows by position (estimate -4.90 against
  1.55 after a reorder);
- `sp.ltmle_survival`: past treatment set to the observed rather than the
  regime value (treated survival 0.721 against a true 0.780);
- `sp.front_door(covariates=)`: arms weighted by P(D) instead of P(D | X)
  (0.773 against 1.134);
- `sp.test` / `lincom` / `nlcom` and `predict(data)` after `irr=True`:
  exp(b) used as the coefficient (Wald 560.9 against 64.2);
- `sp.twoway_cluster` prediction intervals: error variance RSS/(min(G)-1)
  (28.75 against 0.868);
- `sp.iv_bounds('monotone_iv')`: an interval that excluded the truth under
  its own assumption; now the Nevo-Rosen set (formulas read in two copies of
  the paper);
- `sp.cohen_kappa` SE, Mantel-Haenszel continuity correction, IRR exact
  p-value (now equal to R `vcd`, `mantelhaen.test`, `poisson.test`);
- `sp.mixed` LR test against the linear model (now chibar2(01), equal to
  Stata 18 `mixed`);
- `sp.policy_tree`: policy learned and evaluated on the same scores (null
  coverage of the gain interval 22%); now cross-fitted by default.

Found along the way, outside the scanned packages:

- `sp.mixed` returned the shim module `statspai.multilevel.mixed` when old
  code had imported that path first (lazy export now bound to the leaf);
- `sp.event_study` with one cluster divided by zero;
- `sp.hdfe_ols` without the Rust extension: the outcome column could stop
  short of the projection while X went to the direct solve (weighted SE 7e-5
  off); the weekly slow suite was red for two weeks on matplotlib 3.11 (a
  test bug) and on an FLCI tolerance tighter than its SLSQP solve.

The 1.41 line switched on the six announced defaults (`crit_dist='t'`,
`sun_abraham(aggregation='fixest_att')`, `hdfe_ols(cluster_df='min')`,
`sp.bcf` routed to `sp.bart_causal`, interference aliases to
`two_stage_interference`, removed zero-inflated `predicted_*` keys), each
pinned to a reference number already in the repo. The version number was
not bumped.

Track A: every module reproduces its committed result after the scan
(`verify_reproduce_py.py`: 89 reproduce, 0 drift); both trace ledgers were
re-recorded. Track C is stale on main because `core/results.py` changed; it
was not re-timed (the committed timings back the paper at v1.40.2).

## Left open

Decisions for the owner:

- `sp.absorb_ols` keeps `cluster_df='per_term'` while `sp.hdfe_ols` moved to
  `'min'` (the decision named only `hdfe_ols`).
- `sp.aipw(cluster=)` splits cross-fitting folds by row, not by cluster.
- `sp.distributional_te` SEs use ddof = 0.
- A valid variance for the `sp.policy_tree` gain at an exact null of no
  heterogeneity (the cross-fitted interval covers 81-85% there, 92-97% with
  heterogeneity).
- Overdue removals: `kan_dlate` (announced for 1.23), `surrogate/index.py`
  (1.33), `qte(method='quantile_regression')` (1.23), `sp.evalue(rare_outcome=)`
  ("next minor" since 1.18, lives in a Track C file).
- KM median at S = 0.5 exactly: Stata `stci` gives the time, R the midpoint;
  StatsPAI follows Stata (documented).

Defects found and not fixed:

- `sp.impacts`: the stability check uses a fixed +/-0.999, wrong for W that
  is not row-standardised; no SDEM / GNS / SLX impacts; `rho` lower bound
  clipped at -0.99.
- `sp.ipcw`: NaN weights when a covariate is missing.
- `sp.heckman`: no check that `select` is 0/1 (Track A module 43).
- `sp.aipw`: silent fallback to the arm mean when a fold has fewer than 3
  training rows; `sp.pate`: optimiser convergence unchecked;
  `sp.front_door` marginal integration uses 200 Monte Carlo draws.
- `sp.metalearner(weights=, learner='r')` with a continuous treatment: R-loss
  residuals come from unweighted fits although the docstring says weighted;
  `fold_indices=` rejected when weights or clusters have missing values.
- `sp.tmle(fold_indices=, cluster=)` does not check that clusters stay in one
  fold; `weighted_conformal_prediction` gives the test point weight 1.
- `odds_ratio(method="exact")` point estimate (Stata and R differ), PAF and
  NNT intervals, `mr_bma` limits, `ndcg` ties.
- A subgroup whose control is constant within the group is dropped silently
  by `subgroup_analysis`.
- `spatial/iv.py`, `did.py`, `shift_share.py` provenance helpers keep
  `except Exception: pass`.

Not scanned: `survival/causal_forest.py`, `survival/teffects.py`,
`mendelian/frontier/`, `epi/bradford_hill.py`; and the packages with no
scan yet: `agent`, `smart`, `core`, `diagnostics`, `mcmc`, `output`, `fast`,
`plots`, `dag`, `causal_discovery`, `structural`, `doe`, `bayes`,
`experimental`, `workflow`, `interference`, `network`, `neural_causal`,
`nonparametric`, `causal_llm`, `rlasso`, `crossval`, `power`.

## Process notes for the next round

- When another session needs the machine idle, a read-only phase costs
  little: suspects come from reading; execution only confirms. Unlock on
  the other session's explicit message, not on file timestamps (this round
  unlocked once too early).
- Merging seven lines: CHANGELOG / MIGRATION conflicts are append-append at
  the top of `[Unreleased]`; resolve by grafting each branch commit's added
  blocks onto the target (do not union conflict hunks: a shared `## `
  prefix interleaved two MIGRATION sections once).
- Stack the lines that touch Track A paths on one integration branch and
  run one full `verify_reproduce_py.py` and one trace re-record, rather than
  one per line. `postestimation/_covariance.py` and `core/results.py` sit on
  the paths of nearly every module.
- Run the full suite on the integrated branch with `STATSPAI_SKIP_RUST=1`
  as well: it found the `hdfe_ols` and fepois-test defects that the Rust
  path hides. Under `-n`, tests that rerun Track A scripts race on
  `tests/r_parity/data/*.csv`; rerun a collection error serially before
  believing it.

## Round 5 (same day, coordinator statspai-04)

The owner delegated the decisions above. Five lines (`wt/c4-*`) closed them
and scanned four more packages:

- Decided: `sp.aipw(cluster=)` folds by cluster (coverage 85.8% to 94.6% in
  a cluster-randomised DGP); `sp.distributional_te` bootstrap SEs use
  ddof = 1; `sp.absorb_ols` warns on multi-way clustering without
  `cluster_df` and moves to `'min'` in 1.42; the `sp.policy_tree` null
  coverage is documented and flagged in `diagnostics`, not re-derived;
  the four overdue removals are done (`surrogate/index.py` lost only the
  deprecated `SurrogateResult`; the file's live functions stay).
- Fixed from the open list: heckman 0/1 check, ipcw missing covariates
  (dropped with a warning, as in `sp.cox`), pate convergence, weighted
  R-learner nuisances, tmle cluster-in-one-fold check, weighted conformal
  test weights (`weights_test=`), `subgroup_analysis` keeps degenerate
  groups, `sp.impacts` and the spatial ML search on the eigenvalue
  interval, SLX / SDEM / GNS impacts, PAF with zero cells, the Altman NNT
  interval, `ndcg` ties, `mr_bma` limits, narrowed provenance excepts.
- Scanned: `power/`, `rlasso`, `crossval`, `interference/`, plus the
  leftover files of survival / mendelian / epi. Seventeen defects, among
  them `sp.regress("y ~ d | fe")` silently fitting pooled OLS and
  `sp.iv(absorb=)` t / p values not following the rescaled SEs.

Still open after round 5: `aipw(design=)` PSU folds; `pate` IPW separation
hidden by clipping; one-tailed normal-approximation power at a zero effect;
`power_two_proportions` Wald vs pooled chi2; `cluster_matched_pair`
variance; `dnc_gnn_did` network features; the `"parmigiani"` interference
alias (deprecation); `cross_validate` DML `n_folds` / `seed`, misspelt
`reference=`, fixed 95% CIs; `relative_risk` zero-cell Haldane point
estimate (Stata reports the plain ratio); the PAF interval treats
exposure prevalence as fixed; `ipcw` event column read with `astype(int)`.
Packages never scanned: `agent`, `smart`, `core`, `diagnostics`, `mcmc`,
`output`, `fast`, `plots`, `dag`, `causal_discovery`, `structural`, `doe`,
`bayes`, `experimental`, `workflow`, `network`, `neural_causal`,
`nonparametric`, `causal_llm`.
