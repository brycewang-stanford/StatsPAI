# Clean-room specification: GLM score wild cluster bootstrap

Audit row H27: `inference/jackknife.py::glm_score_wild_boot` (with
`score_wild_from_q` and `_draw_wild_weights`) was, by its commit message
(`689aa881`), recovered from Stata `boottest`'s outputs and then checked
against `boottest.mata` (GPL-3). The three functions were deleted by AST
span without being read and re-implemented in `inference/_score_wild.py`
(re-exported from `jackknife.py` for the existing import paths).

What the rewriter saw before starting, recorded here as required: a grep
printed lines 1513-1535 of the old `glm_score_wild_boot` docstring and two
single lines of the old bodies (a call `_draw_wild_weights(weight_type,
(n_boot, G), rng)` and a docstring line); the commit message of
`689aa881` (read for the audit context) states the convention, including a
one-line quote of `boottest.mata`'s cluster-share recentring. The same
convention is stated independently in Roodman et al. (2019, Appendix A.3,
eq. 69-70) and in the docstring of
`tests/reference_parity/test_feglm_wild_boottest_parity.py`; the
implementation follows the paper.

## Sources

- Roodman, Nielsen, MacKinnon and Webb (2019) `[@roodman2019fast]`, read in
  its CREATES Research Paper 2018-34 version: Section 6.2 (the score
  bootstrap of Kline and Santos 2012) and Appendix A.3 (recentring the
  bootstrapped scores by each cluster's share of the observations).
- Black-box: Stata 18 `boottest` 4.5.3 values frozen in
  `test_feglm_wild_boottest_parity.py`, and the previous StatsPAI functions
  run on 120 inputs.

## Specification

For `H0: beta_j = 0` in a GLM with clusters `g = 1..G`:

1. Restricted fit (column `j` dropped). Score `s_i = x_i (y_i - mu_i)
   mu'_i / V(mu_i)` and expected Hessian `H = X' diag(mu'^2/V) X` at the
   restricted fit, `X` the full design.
2. `q_g = (H^{-1} sum_{i in g} s_i)_j`; `c_g = n_g / n`.
3. For multipliers `w`: `N(w) = sum_g w_g q_g`, `J_g(w) = w_g q_g - c_g
   N(w)` (paper eq. 69: recentred bootstrapped scores), `t(w) = N(w) /
   sqrt(sum_g J_g(w)^2)`. The constant small-sample factor of the CRVE
   cancels in the p-value and is not applied (black box: `t(1)` equals
   boottest's `r(z)`, -1.0999434).
4. Observed statistic `t(1)`; p-value = share of draws with `|t(w)| >
   |t(1)|` beyond a relative tie margin of 1e-10 (draws proportional to the
   all-ones vector reproduce `t(1)` exactly in exact arithmetic and are not
   counted).
5. Draws: Rademacher grids are enumerated when `2^G <= n_boot` (all sign
   vectors); otherwise `numpy.random.default_rng(seed).choice` of shape
   `(n_boot, G)` over `[-1, 1]`, or over Webb's six points
   `+-sqrt(3/2), +-1, +-sqrt(1/2)` (never enumerated).
6. Fewer than two clusters, or all `q_g = 0`: `p = 1`, `t = 0`, no draws.
   Unknown `weight_type`: `MethodIncompatibility`.

## Acceptance

The four enumerated boottest p-values (Poisson and logit) are reproduced
exactly and `t(1)` to 1e-5 (`test_feglm_wild_boottest_parity.py`); the
previous function's outputs on 120 inputs (enumerated and random, both
weight types, the GLM path for Poisson and logit) are reproduced to 1e-9,
seeded draws included. The linear wild cluster bootstrap (including the
nine enumerated `bootcluster()` p-values of `bc2fabd3c`) does not use this
code and its tests pass unchanged.
