# Clean-room specification: CS cell grid, repeated cross-sections, multiplier bootstrap

Follow-up to `did_drdid.md` (audit rows H2-H4). The DiD clean-room pass of
2026-10-20 found three more pieces whose comments quoted R `did` code (GPL-3)
or described themselves as mirroring it:

- `did/callaway_santanna.py::_get_gt_pairs` (the ATT(g, t) grid) and
  `_callaway_santanna_rcs` (the estimator on repeated cross-sections and
  unbalanced panels), with the helper `_rcs_residualise_on_controls`, which
  only that function called;
- `did/_core.py::multiplier_bootstrap` ("following R `did::mboot`");
- `did/wooldridge_did.py::_drdid_imp_panel_core` (and its helpers
  `_calibrated_pscore`, `_logistic_fit`, `_logistic_coefficients`,
  `_weighted_lstsq`), the unweighted improved panel DR-DiD of `sp.drdid`.

All of them were deleted without being read (by AST span; the comment blocks
next to them were removed unread too) and replaced from this specification.

## Sources (and only these)

- Callaway and Sant'Anna (2021) `[@callaway2021difference]`, arXiv
  1803.09015: Assumption 3 (limited anticipation), Section 4 (estimation and
  Algorithm 1, the multiplier bootstrap with Mammen weights and
  interquartile-range standard errors; Remark 10, clustered multipliers).
- Sant'Anna and Zhao (2020) `[@santanna2020doubly]`: the two-period
  estimators, already re-implemented in `did/_rcs.py` and `did/_dr_common.py`.
- R `did` 2.5.1 help pages rendered with `tools::Rd2txt` (`att_gt`, `mboot`,
  `aggte`): the meaning of `base_period`, `anticipation`, `panel`,
  `allow_unbalanced_panel`, `clustervars`, `est_method`.
- Black-box runs: R `did` 2.5.1 on our data, and the previous StatsPAI
  functions themselves, run as black boxes from an export of `origin/main`
  (inputs in, outputs compared; their source was not opened).
- The existing tests and fixtures in this repository.

Bib keys verified on 2026-10-20 (see `did_drdid.md`).

## 1. The ATT(g, t) grid (`_cs_rcs.gt_cells`, imported as `_get_gt_pairs`)

Input: cohorts, observed periods, `base_period`, `anticipation = delta`.

- The **base period of cohort g** is the latest observed period `b_g` with
  `b_g < g - delta` (Assumption 3: periods before `g - delta` are untreated;
  the help page writes the universal base as `g - anticipation - 1`, which is
  `b_g` on a gap-free calendar). A cohort with no such period has no
  identified cell: it is skipped with a `UserWarning` (its units remain
  comparisons where the control group allows).
- `'universal'`: one cell for every observed period `t != b_g`, all compared
  with `b_g`. The base cell itself is not reported (it is zero by
  construction).
- `'varying'`: post cells `t >= g` use `b_g`; a pre cell `t < g` uses the
  observed period immediately before `t` ("changes from period t-1 to period
  t", help page), so the first observed period has no cell. Cells inside the
  anticipation window (`g - delta <= t < g`) are pre cells.
- Output sorted by cohort, then period.

Black box: 120 grids (gap-free and gappy calendars, anticipation 0-2, cohorts
at the first and last periods) are identical to the previous function's;
`tests/reference_parity/test_cs_gaps_parity.py` checks the grid and values
against R `did` for anticipation 0-2 under both base periods.

## 2. Cohorts beyond the sample (`callaway_santanna`)

A cohort first treated after the last observed period is untreated in every
period of the sample, so it belongs to the never-treated group. R `did`
treats it so (black box: on a balanced panel and on repeated cross-sections
where half of the never-treated were recoded to a cohort after the sample,
R's ATT and SE equal those with the original coding, to 1e-15). StatsPAI
used to leave such units out of `control_group='nevertreated'` on both paths
(ATT 1.087 vs R 1.044 on the probe panel). ⚠️ correctness fix: they are now
recoded to 0 before estimation.

## 3. Repeated cross-sections and unbalanced panels (`_cs_rcs.cs_repeated_cross_sections`)

Rows are observations. `n_obs` = rows after dropping missing outcome,
period, cohort, covariates or weights.

- **Weights**: non-negative and finite (`MethodIncompatibility`), positive
  total (`DataInsufficient`), divided by their mean over rows. On an
  unbalanced panel they must be constant within unit (the unit's rows are
  folded into one contribution; `MethodIncompatibility`).
- **Cells**: the grid of section 1 over the estimated cohorts
  (`0 < g <= last period`, minus cohorts kept only as comparisons).
- **Cell estimator**: the rows of cohort `g` and of the comparison group
  (same rule as the panel path, `_rcs_control_mask`) in periods `t` and
  `base`. `estimator='reg'` without covariates is the 2x2 difference of
  (weighted) cell means; every other case is the matching two-period
  estimator of `did/_rcs.py` on that sub-sample (`dr` -> `drdid_rc`,
  `drimp` -> `drdid_imp_rc`, `ipw`/`stdipw` -> `std_ipw_did_rc`,
  `ipw_abadie` -> `ipw_did_rc`, `reg` -> `reg_did_rc`). The cell influence
  function is put on the `n_obs` scale (`n_obs / n_cell` times the
  sub-sample one), zero outside the cell.
- **Unbalanced panel** (`unit_col`): the row influence functions are summed
  within unit and multiplied by `n_units / n_obs`, so that
  `sum_rows(psi)/n_obs = sum_units(psi_u)/n_units`; cell SE
  `sqrt(mean(psi_u^2)/n_units)`. Inference treats the unit as the sampling
  unit; the unit's cohort and weight are those of its first row.
- `n_treated_obs` = rows of cohort `g` in the cell's two periods.
- **Unidentified cells**, **clustered analytic SEs**, **multiplier
  bootstrap**, **aggregation** (cohort-share weights with their estimation
  term), **event study** and **pre-test**: the same helpers the balanced
  panel path uses, with `n = n_units` (rows on repeated cross-sections).
- **Cohort sizes** for the aggregation weights: head counts (rows on
  repeated cross-sections, units on unbalanced panels), or the sum of the
  mean-one weights. (The previous code summed the row-normalised weight of
  each unit's first row on unbalanced panels; that differs from the sum of
  unit weights by a constant factor, so every aggregate is unchanged and
  only `model_info['cohort_sizes']` is rescaled.)
- **Row-level clusters** (repeated cross-sections): refused when the
  cluster is constant within every cohort x period cell, since each cell's
  influence contributions sum to zero within the cell (clustered variance
  exactly zero against never-treated comparisons).
- `variance_convention` is passed to `drdid_rc` for `estimator='dr'`
  (section 5).

Black box: 49 configurations (all six estimators with and without
covariates, both control groups and `notyet_cutoff` rules, both base
periods, anticipation, weights, clusters, bootstrap with and without bands
and clusters, Mammen weights, a design without never-treated units, and the
unbalanced route with each of these) reproduce the previous function's
estimate, SE, interval, detail table, influence matrix, event study, pre-test
and `model_info` to 1e-9 (worst observed difference 0, i.e. bit-identical,
except the documented cohort-size rescaling). The previous `model_info`
carried an `approach` entry ("residualisation on never-treated with period
FEs") that did not describe what was computed (the outputs equal the
`_rcs` estimators on the raw outcome); it is dropped, and the helper it
named is deleted.

## 4. Multiplier bootstrap (`_core.multiplier_bootstrap`)

Algorithm 1 of Callaway and Sant'Anna (2021). Input `psi` (`n_rows x K`),
the scale `n` (`theta_hat - theta ~ sum(psi)/n`), `alpha`, `B`, a seed.

1. Contributions: the rows of `psi`, or with `cluster_ids` the cluster sums
   (clusters ordered by sorted label; Remark 10: one multiplier per
   cluster). Centred at their mean over contributions.
2. Multipliers `V` (`B x m`): Rademacher (+-1 with probability 1/2) or
   Mammen (`(1 - sqrt5)/2` with probability `(sqrt5 + 1)/(2 sqrt5)`, else
   `(1 + sqrt5)/2`; paper Section 4.1), drawn in one call
   `numpy.random.default_rng(seed).choice(support, size=(B, m), p=probs)`.
3. `R* = sqrt(n) V C / n`.
4. `sigma_k = IQR_k(R*) / (z_.75 - z_.25)`, sample quantiles defined as the
   inverse of the empirical CDF (smallest draw with ECDF >= p); `se_k =
   sigma_k / sqrt(n)`, floored at `1e-12` when a column has no spread.
5. `crit` = the `1 - alpha` sample quantile (same definition) of
   `max_k |R*_k| / sigma_k` over columns with spread, never below
   `z_{1 - alpha/2}`.

Errors: unknown `weight_type` or a `cluster_ids` length different from the
rows of `psi` raise `ValueError` (existing contract).

Black box (the previous function on 102 inputs: three shapes, four seeds,
both weight types, 99 / 500 draws, with and without clusters, string
cluster labels, `n` different from the row count, a zero column): the same
seeded draws, standard errors and critical values to 4e-16 relative. One
deliberate difference: a column with a non-finite influence function now
reports a NaN standard error instead of `1e-12` (failing loudly instead of
reporting a spuriously tiny SE). The CS bootstrap remains an S-level
(stochastic) screen against R; the seeded stream of every committed StatsPAI
result is unchanged.

## 5. DR on repeated cross-sections: the pre-period regression term (owner's decision, 2026-10-20)

`did_drdid.md` documents that R `DRDID`'s `drdid_rc` / `drdid_rc1` carry the
estimation effect of the pre-period comparison regression with the opposite
sign to the delta method (paper Appendix B). The owner's delegate decided:

- `variance_convention='delta'` (the paper) is the default of
  `_rcs.drdid_rc`, `sp.drdid` (repeated cross-sections, `method='trad'`) and
  `sp.callaway_santanna` (`estimator='dr'` on the repeated-cross-section and
  unbalanced-panel route). ⚠️ correctness fix; `'reference'` reproduces R.
- The R gap is T4 (reference divergence), with
  `tests/reference_parity/_fixtures/_drdid_rc_pre_regression_mc.py` as the
  independent evidence (coverage 0.95 vs 0.81 under a misspecified
  propensity score, correct comparison outcome regressions). Point estimates
  are unaffected; the two conventions agree asymptotically when the
  propensity model is correct.

## 6. `sp.drdid(method='imp')` on a panel

The unweighted improved panel estimator is now the same engine as the
weighted one, `callaway_santanna._dr_imp_att` (inverse probability tilting,
odds-weighted comparison regression; `did_drdid.md`). Black box: on Track A
`38_drdid` the ATT and SE equal the previous ones to 1e-15 (the committed
result reproduces); on 12 further designs the largest difference is 5e-8
relative, on a near-separated design where the previous solver matched R's
stopping point and `ipt_fit` converges to the optimum (inside the 1e-7
tolerance of the DRDID black-box fixture). A panel with no treated or no
control units, or with fewer comparison units than regressors for an
estimator with a comparison outcome regression, raises `DataInsufficient`
instead of returning an infinite or minimum-norm result.
