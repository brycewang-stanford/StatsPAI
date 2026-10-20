# Clean-room spec: MR-RAPS (`sp.mr_raps`)

Scope: `src/statspai/mendelian/frontier/raps.py`, the whole module (audit row
H14, `docs/dev/2026-10-20-license-provenance-audit.md`). Public API kept:
`mr_raps(...)` signature and defaults, `MRRapsResult` fields, and the private
helpers that tests and `frontier/grapple.py` import (`_rho`, `_gauss_moment`,
`_robust_vcov`) with the same call signatures.

Reference used as a black box: R `mr.raps` 0.4.3 (GPL-3), functions
`mr.raps.simple`, `mr.raps.overdispersed`, `mr.raps.overdispersed.robust`,
and the internal loss functions `rho.huber` / `rho.tukey`, which are
documented on their own help pages and were only called on a grid.

## Sources read / not read

Read:

- Zhao, Wang, Hemani, Bowden & Small (2020), `zhao2020statistical`, via
  arXiv:1801.09652v3: Sections 3 (profile likelihood, Theorems 3.2-3.3 and the
  plug-in estimates of `V1`, `V2`), 4 (APS, eqs. 4.2-4.3, Theorem 4.2 and the
  plug-in remark after it), 5 (losses, eqs. 5.1-5.3, Theorem 5.2 and its
  constants).
- Help pages of `mr.raps` 0.4.3 (`?mr.raps`, `?mr.raps.mle`, `?rho.huber`,
  `?rho.tukey`, `?rho.l2`) via `tools::Rd2txt`.
- Outputs of the reference on `tests/reference_parity/_fixtures/mr_ldl.csv`
  (the committed fixture `mr_R.json`) and on 11 scratch datasets (7 fits
  each: simple, L2 with and without pruning, Huber 1.345 / 2, Tukey 4.685 /
  3), plus `rho.huber` / `rho.tukey` (derivatives 0-2) on a grid.
- The repository's tests and fixtures (`tests/test_mr_frontier.py`,
  `tests/reference_parity/test_mr_raps_parity.py`,
  `tests/reference_parity/test_mr_R_parity.py`,
  `tests/reference_parity/test_misc_sens_R_parity.py`,
  `_generate_mr_R.R`), the registry entry.
- The redacted view of the old `raps.py` (all bodies removed; signatures,
  docstring first lines and module-level code kept) and the outputs of the
  old implementation on 104 configurations, run as a black box before it was
  deleted.

Not read: any `mr.raps` R source (no function bodies printed, no
CRAN/GitHub source); the old `raps.py` bodies or their git history.

## Model and estimators

Variant `j`: `bx_j ~ N(gamma_j, sx_j^2)`, `by_j ~ N(beta gamma_j, sy_j^2 +
tau^2)`. Write `r_j = by_j - beta bx_j`, `v_j = sx_j^2 beta^2 + sy_j^2 + tau^2`,
`t_j = r_j / sqrt(v_j)`.

1. **Profile score, no over-dispersion** (`over_dispersion=False`, L2 only).
   `beta_hat = argmax l(beta)`, `l = -1/2 sum r_j^2 / (sx_j^2 beta^2 + sy_j^2)`
   (eq. 3.2). `l` tends to a finite constant as `|beta| -> inf` and can have
   several stationary points, so it is scanned on `beta = tan(theta)` (4,001
   points) and the best cell refined by `brentq` on `l'`. Standard error
   `sqrt(V1_hat) / V2_hat` (Theorem 3.3) with
   `V2_hat = sum ((bx^2 - sx^2) sy^2 + (by^2 - sy^2) sx^2) / v^2`,
   `V1_hat = V2_hat + sum sx^2 sy^2 / v^2`.
2. **APS** (`over_dispersion=True, loss="l2"`): the finite root of
   `psi1 = sum r_j (by_j sx_j^2 beta + bx_j (sy_j^2 + tau^2)) / v_j^2` (4.2) and
   `psi2 = sum sx_j^2 (r_j^2 - v_j) / v_j^2` (4.3).
3. **RAPS** (`loss="huber"|"tukey"`): `psi1 = sum rho'(t_j) u_j` with
   `u_j = -dt_j/dbeta = (bx_j v_j + r_j sx_j^2 beta) / v_j^{3/2}` (5.2),
   `psi2 = sum sx_j^2 (t_j rho'(t_j) - delta) / v_j` (5.3),
   `delta = E[Z rho'(Z)]`. With `rho' = identity` these reduce to (4.2)-(4.3).

### Solver (our design, not the reference's)

Alternating steps from the profile-score estimate (APS) or from the APS
estimate (RAPS; the help page names this initialisation `"l2"`):

- `tau2` step: smallest positive root of `psi2(beta, .)` by geometric
  bracketing from `1e-6 * min(sy^2)` and `brentq`. `psi2 -> -delta sum
  sx^2/v < 0` as `tau2 -> inf`, so a root exists iff `psi2(beta, 0) > 0`;
  otherwise `tau2 = 0` and a warning says there is no detectable
  over-dispersion.
- `beta` step: walk uphill from the current `beta` (doubling steps) until
  `psi1` changes sign, then `brentq` (the root is the local maximum of
  `-sum rho(t_j)` reached from the start, which keeps the redescending Tukey
  fit on the branch the L2 start selects).
- stop when `|d beta| + |d tau2| < tol` (default `sqrt(eps)`), at most
  `niter` (20) rounds; `converged=False` plus a warning otherwise.
- after convergence with `tau2 > 0`, a joint `scipy.optimize.root` polish;
  accepted only if it succeeds, stays within `1e-4` (beta) / `1e-3` (tau2)
  relative of the alternating solution and lowers the residual.

### Variance (Theorems 4.2 and 5.2)

`V = B^{-1} A B^{-T}`, with sums weighted by `1/v_j^2` and the plug-ins
`gamma_j^2 -> bx_j^2 - sx_j^2`, `Gamma_j^2 -> by_j^2 - sy_j^2 - tau2_hat`
(the remark after Theorem 4.2):

- `A = diag(c1 sum((gamma^2 + sx^2)(sy^2 + tau2) + Gamma^2 sx^2),
  c2 sum 2 sx^4)`,
- `B = [[delta sum(gamma^2 (sy^2 + tau2) + Gamma^2 sx^2), delta sum sx^2 beta],
  [0, (delta + c3)/2 sum sx^2]]`,
- `c1 = E[rho'(Z)^2]`, `c2 = Var(Z rho'(Z))/2`, `c3 = E[Z^2 rho''(Z)]`;
  all equal 1 (and `delta = 1`) for L2.

`_robust_vcov(bx, by, sx, sy, beta, tau2, (delta, c1, var_zpsi, c3))` takes
`var_zpsi = Var(Z rho'(Z)) = 2 c2` (the existing tests pass that quantity).
Moments: `_gauss_moment(f) = E[f(Z)]` by `scipy.integrate.quad` split at
`0, +-5, +-10` (accuracy ~1e-12; R's `integrate` default is 1.2e-4 relative).

Reporting: two-sided normal p-value, normal CI at `1 - alpha`,
`loglik_robust = -sum rho(t_j)` (eq. 5.1; `-1/2 sum t_j^2` for L2).

## Conventions established from the black box

| Convention | How established |
| --- | --- |
| `rho.huber`: `r^2/2` / `k(|r| - k/2)`; derivs `clip(r, -k, k)`, `1{|r| <= k}` | grid call, max diff 2e-15 |
| `rho.tukey(deriv=0) = 1 - (1 - (r/k)^2)^3` (paper scale, max 1) but `deriv=1` is the biweight `r (1 - (r/k)^2)^2` (= `k^2/6` x the derivative of the paper's loss), `deriv=2` its derivative | grid call, max diff 6e-15; the factor cancels in the estimating equations and the sandwich |
| default `k`: Huber 1.345, Tukey 4.685 | help page usage line |
| pruning: drop `se_exposure > 10 * median` (strict) | datasets with one ratio 9.99 / 10.0 / 10.01: only 10.01 is pruned |
| `mr.raps.simple` has no pruning | help page; StatsPAI applies `pruning=` to every variant of `mr_raps` (as before the rewrite) |
| robust start = APS (L2) fit; APS start = simple fit | help page (`initialization = "l2"` / `"simple"`) |
| sandwich with the paper's plug-ins reproduces the reference SEs | at the reference's own `(beta, tau2)` and moments: 1e-15 (L2), 2e-15 (Huber, Tukey) |

## Differences from the reference (documented, not bugs)

- The reference stops its iterations at an absolute `sqrt(eps)` and its
  `tau2` root at `uniroot`'s default tolerance; StatsPAI solves the same
  equations to machine precision. On the LDL fixture: simple beta 8.7e-9,
  APS tau2 1.75e-8 relative; robust beta 2e-5 to 3e-5, tau2 5e-4 to 1e-3.
  In all 36 robust fits on the scratch datasets StatsPAI's residuals of both
  estimating equations are no larger than the reference's.
- When `psi2(beta, 0) <= 0` the reference's APS reports `tau2` of order
  1e-8 (a solver end point) and warns that it is "quite small"; StatsPAI
  reports exactly 0 and warns. beta moves by at most 5e-6 relative.
- Robust losses without over-dispersion raise `ValueError` (unchanged).

## Old vs new (StatsPAI before / after the rewrite, 104 configurations)

- `_rho` (all derivatives, 4 losses) identical; `_robust_vcov` identical to
  3e-16; `_gauss_moment` within 2e-9 (the new quadrature is the more
  accurate: Huber `delta` equals `2 Phi(k) - 1` to 1e-16).
- Estimates, SEs, tau2: within 1e-6 relative everywhere except the APS fits
  where the old code emulated the reference's stopping point (tau2 up to
  7e-6 relative, beta up to 5e-6) and the zero-tau2 cases (old 7e-9, new 0).
- `loglik_robust` is now `-sum rho(t)` for Huber / Tukey (eq. 5.1); the old
  field was half that. L2 values unchanged. Not used by any test.
- Warning texts reworded; the "tau2 set to 0" warning now also fires for the
  L2 fit. `converged` with `niter=3` became `True` on one dataset that
  converges in three rounds under the new solver.
- Error message for `over_dispersion=False` with a robust loss reworded.
