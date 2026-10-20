# Clean-room spec: `sp.grapple` (GRAPPLE robust profile likelihood, one exposure)

Scope: `src/statspai/mendelian/frontier/grapple.py` (audit row H17,
`docs/dev/2026-10-20-license-provenance-audit.md`). The whole module was
rewritten: `grapple`, `_grapple_vcov`, the loss helpers, `_t_values` and
`GrappleResult.summary`.

Reference used as a black box: GRAPPLE 0.2.2 (GitHub `jingshuw/GRAPPLE`
@ `317e837`, GPL (>= 2)), function `grappleRobustEst`. The package cannot be
installed (its `haploR` import is archived), so its estimator file
`R/mr_fun_general.R` is downloaded and `source()`d, exactly as the committed
generators do; the file was never opened, printed or searched.

## Sources read / not read

Read:

- Wang, Zhao, Bowden, Hemani, Davey Smith, Small & Zhang (2021),
  `wang2021causal`, PLOS Genetics 17(6) e1009575, full text (Europe PMC
  PMC8301661): Results ("Identify multiple pleiotropic pathways",
  Equations 2-4), Materials and methods ("Estimate the effects beta",
  Equations 10-12, the Tukey biweight with `c = 4.6851`).
- S2 Text of that paper ("Robustified profile likelihood methods for
  multivariate MR": estimating equations, the delta-method sandwich
  `A^-1 B A^-T` and the expressions for `A` and `B`).
- Zhao, Wang, Hemani, Bowden & Small (2020), `zhao2020statistical`,
  arXiv:1801.09652v3, for the robust adjusted profile score background.
- GRAPPLE's help page `man/grappleRobustEst.Rd` (rendered with
  `tools::Rd2txt`): arguments (defaults `k = 4.685` Tukey / `1.345` Huber,
  optional fixed `tau2`), return values.
- Outputs of `grappleRobustEst` on: the committed LDL fixture
  (`tests/reference_parity/_fixtures/misc_sens_R.json`, key `grapple`),
  seven further data sets (LDL plus six simulated ones) with the default fit,
  and the same seven with `tau2` fixed at 0, 1e-4 and 1e-3; the committed
  fixture `_fixtures/grapple_R.json` (written by
  `_fixtures/_generate_grapple_R.R`).
- The repository's tests (`tests/test_mr_frontier.py`,
  `tests/reference_parity/test_grapple_parity.py`,
  `tests/reference_parity/test_misc_sens_R_parity.py` and its generator),
  the registry entry, the agent card and the sibling spec
  `docs/dev/cleanroom/mr_raps.md`.
- The redacted view of the old `grapple.py` (every body removed) and the
  outputs of the previous implementation on 140 configurations (18 data sets
  x 7 option sets, plus 14 edge cases), run as a black box before deletion.
- Black-box calls of `statspai.mendelian.frontier.raps._rho` /
  `_gauss_moment` (values only) to see the loss scale the existing test uses.

Not read: any GRAPPLE or mr.raps source (no function bodies printed, no
GitHub/CRAN source browsing), the previous bodies in `grapple.py` or their
git history.

## Model and estimating equations

One exposure, independent GWAS: `gamma_hat_j ~ N(gamma_j, vx_j)`,
`Gamma_hat_j ~ N(beta gamma_j + alpha_j, vy_j)`, `alpha_j ~ N(0, tau2)`.

    t_j(beta, tau2) = (by_j - beta bx_j) / sqrt(vy_j + beta^2 vx_j + tau2)   (Eq. 10)

- `phi_1(beta, tau2) = sum_j rho'(t_j) dt_j/dbeta = 0` (stationary point of
  the robust profile likelihood `l = -sum_j rho(t_j)`, Eq. 11), with
  `dt_j/dbeta = -(v_j bx_j + (by_j - beta bx_j) beta vx_j) / v_j^1.5`;
- `phi_2(beta, tau2) = sum_j rho(t_j) - (p - 1) E[rho(Z)] = 0`, `Z ~ N(0,1)`
  (Eq. 12 with one degree of freedom for beta; see the conventions table).
  `sum_j rho(t_j)` decreases in `tau2`; if it is already below
  `(p - 1) E rho(Z)` at `tau2 = 0`, `tau2_hat = 0`.

Losses: Tukey `rho = c^2/6 [1 - (1 - (r/c)^2)^3]` for `|r| <= c`, `c^2/6`
beyond (default `c = 4.685`); Huber `r^2/2` / `k|r| - k^2/2` (`k = 1.345`);
L2 `r^2/2`. Moments `E rho(Z)`, `E rho'(Z)^2`, `Var rho(Z)`,
`E[Z rho'(Z)]` by quadrature split at the kinks `+-c`.

Algorithm (StatsPAI's own): start beta at `beta_init`, or at the L2 GRAPPLE
fit (itself started from the inverse-variance weighted estimate) for the
robust losses; one beta step at `tau2_init`; then alternate the `tau2` root
(Brent) and the beta score root nearest the current beta (downhill bracket +
Brent) until both change by less than `tol` relative (default `1e-13`) or
`max_iter` rounds (default 200; `converged = False` and a `RuntimeWarning`
otherwise). The fit reached the same mode as the reference on every data
set tried.

## Variance (S2 Text)

`V = A^-1 B A^-T` with `v_j = vy_j + beta^2 vx_j + tau2`, `d_j = dt_j/dbeta`:

- `B = diag(E[rho'(Z)^2] sum_j d_j^2, p Var[rho(Z)])`;
- `A_11 = E[rho''(Z)] sum_j d_j^2 + sum_j rho'(t_j) h_j`, `h_j = -vx_j by_j / v_j^1.5`;
- `A_12 = E[Z rho'(Z)] sum_j beta vx_j / v_j^2`;
- `A_21 = 0`;
- `A_22 = -E[Z rho'(Z)] / 2 sum_j 1 / v_j`.

`se = sqrt(V_11)`, `tau2_se = sqrt(V_22)` (NaN when `tau2_hat = 0`),
normal p-value and CI.

S2 Text writes the second term of `A_11` as the sample mean of
`rho'(t_j) d^2 t_j / dbeta^2`. At the true values
`E[rho'(t_j) d^2t_j/dbeta^2] = E[rho'(t_j) h_j] = -E[Z rho'(Z)] vx_j (vy_j + tau2) / v_j^2`
(both follow from `Cov(bx_j, by_j - beta bx_j) = -beta vx_j` and the
independence of `t_j` and `dt_j/dbeta`), so both are consistent plug-ins; the
reference reports the `h_j` version (identified below), and StatsPAI uses it.

## Black-box conventions established

| Convention | How established | Result |
| --- | --- | --- |
| `tau2` equation uses `(p - 1) E rho(Z)`, not `p E rho(Z)` | `sum_j rho(t_j) / E rho(Z)` at the reference's `(beta, tau2)` on 7 data sets x 3 losses | 27.000 for p = 28, 39.000 for p = 40, 29.00 for p = 30 (to the reference's root tolerance) |
| `B_22 = p Var rho(Z)` and `A_22` as above | reference `tau2_se` at its own estimates, L2 / Huber | 4e-14 (L2), 4e-7 (Huber, R `integrate`) |
| `A_12 = E[Z rho'(Z)] sum beta vx / v^2` (no factor 1/2), `A_21 = 0`, `A_11` with `h_j` | least squares of the implied `A_11` on candidate terms over 14 fixed-`tau2` and 7 estimated-`tau2` L2 fits | coefficients exactly (-1/2, 0, 1, 0, 0) on `(2 t bx beta vx v^-1.5, 3 t r beta^2 vx^2 v^-2.5, -t r vx v^-1.5, ...)`, residual 6e-13; i.e. `h_j = -vx_j (beta bx_j + r_j) / v_j^1.5 = -vx_j by_j / v_j^1.5` |
| the full 2 x 2 sandwich is used even when `tau2` is fixed or at 0 | fixed-`tau2` runs; `tau2_hat = 0` fits | same formula reproduces them (1e-13) |
| all of the above, out of sample | committed fixture `grapple_R.json` (5 new scenarios x 3 losses, estimated and fixed `tau2`), `test_grapple_R_parity.py` | sandwich at the reference's points: 1.6e-15 (L2), 6.8e-7 (Huber / Tukey, R `integrate`); fit: beta 3.8e-3, tau2 9e-4, SE 3.7e-4 (reference's optimiser stop) |
| default `k`: Tukey 4.685, Huber 1.345 | help page usage line | adopted (the paper says 4.6851) |
| `beta` from `optim` at its default tolerance, `tau2` root at `bound * eps^0.25` | reference beta score at its estimate is 1e-6 to 1.3 (not 0) | StatsPAI solves both equations to machine precision; fitted values agree to 1e-3 (beta), 1e-3 (tau2) |

## Documented divergence: Tukey loss scale (T4, reference inconsistency)

The reference evaluates `Var rho(Z)` (in `B_22`) and the `tau2` equation with
the biweight on its max-1 scale `1 - (1 - (r/c)^2)^3`, but `rho'` (in `A`)
on the paper's `c^2/6` scale. The `tau2` equation and the point estimates are
invariant to that factor; the sandwich is not. Its `tau2_se` is therefore
exactly `6 / c^2` (0.273 at `c = 4.685`) times the consistent value, and its
beta SE moves through the `A_12^2 B_22` term (1e-3 relative on the LDL data,
up to 15% on weak-instrument data with `tau2_hat = 0`). Mechanism checked:
the StatsPAI sandwich with `Var rho(Z)` multiplied by `(6/c^2)^2` reproduces
the reference's Tukey `se` and `tau2_se` to 1e-7 at its own estimates.

StatsPAI keeps one scale (the paper's). Independent evidence that this is the
calibrated one: the sandwich is invariant to rescaling `rho` only when the
same scale is used throughout (tested), and a Monte Carlo study (60 variants,
heterogeneous SEs, `tau = 0.02`, 400 replicates, seed 20261020) gives
sd(`tau2_hat`) = 1.59e-4 against a mean StatsPAI `tau2_se` of 1.55e-4 and a
mean reference-scale `tau2_se` of 4.2e-5; beta coverage of the StatsPAI SE
0.948. A weak-instrument design (50 variants, `gamma ~ N(0, 0.012^2)`,
SEs 0.01, `tau = 0.004`, 300 replicates) gives sd(`tau2_hat`) = 2.4e-5
against 3.2e-5 (StatsPAI) and 8.8e-6 (reference scale), so the reference's
`tau2_se` is again too small by about 3x; for beta both SEs are
conservative there (sd 0.162; mean SE 0.176 StatsPAI, 0.164 reference scale;
95% coverage 0.987 and 0.977), i.e. the reference-scale beta SE happens to be
closer in that design. The choice of the consistent scale rests on the
invariance argument and the `tau2_se` calibration, not on the beta SE.
Huber and L2 are unaffected (their value and derivative scales agree).
Scripts: `mc_tau.py`, `mc2.py` in the rewrite's scratch directory (not
committed; the numbers above are their full output).

## Defaults and edge cases (StatsPAI API)

- `loss` not in {"tukey", "huber", "l2"}: `MethodIncompatibility` (kept).
- New: non-finite inputs, negative `se_exposure`, non-positive `se_outcome`,
  `k <= 0` and `alpha` outside (0, 1) raise `ValueError` (the previous code
  returned NaN SEs for NaN input and accepted `k < 0`, `alpha = 0`).
- `se_exposure = 0` is allowed (no measurement error).
- `tau2_init < 0` is treated as 0; `max_iter` exhausted gives
  `converged = False` and a `RuntimeWarning` (kept).
- `tau2_se` is NaN at the boundary `tau2_hat = 0` (kept; the reference
  reports a number there).
- `loglik` is `-sum_j rho(t_j)` with the paper's loss scale (for Tukey the
  previous code used the max-1 scale: values are `c^2/6` times the old ones).
- Not implemented: multivariable fits, sample-overlap correlation
  (`cor.mat`), the mode diagnosis, a user-fixed `tau2`.

## Comparison with the previous implementation

On the 126 snapshot fits: estimates and `tau2` agree to 2e-13
relative (Huber with `k = 2`: moments by quadrature, 1e-12); Huber and L2
SEs, `tau2_se` and `loglik` agree to 1e-13. Tukey differs only as described
above: `tau2_se` x `c^2/6`, `loglik` x `c^2/6`, `se` by 2e-5 to 15% (largest
on the weak-instrument data sets with `tau2_hat = 0`).
