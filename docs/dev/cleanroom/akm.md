# Clean-room spec: AKM shift-share inference (`bartik/_akm.py`)

Scope: `src/statspai/bartik/_akm.py` in full (audit row H20,
`docs/dev/2026-10-20-license-provenance-audit.md`): `_akm_fit`, the
collinear-share selection (formerly a port of R's QR routine), and
`_bhj_aggregate`. The whole module was treated as contaminated and replaced
by a new file; the old one was never opened.

References used as black boxes: R `ShiftShareSE` 1.1.0 (GPL-3) and R's
`qr()` (base R, GPL-2 | GPL-3).

## Sources read / not read

Read:

- Adao, Kolesar & Morales (2019), QJE (`ado2019shift`), via the vignette's
  summary of Remark 6 and Proposition 4.
- `ShiftShareSE` documentation: `?ivreg_ss.fit` (arguments, the AKM0 "SE"
  definition, `beta0` affecting only p-values) and the vignette "Standard
  Errors in Shift-Share Regressions" (Kolesar, 2022; `kolesar2022shiftsharese`):
  section 3.1 (collinear shares dropped by a column-pivoting QR) and
  section 4 (the feasible AKM formula and the AKM0 set as a quadratic
  inequality).
- `?qr` (R base): the default decomposition moves columns with near-zero
  norm (relative to `tol = 1e-7`) to the right edge.
- The callers (`bartik/adao_correction.py`, `bartik/political.py`) and the
  tests, fixtures and generators that exercise the module, including the
  header comment of `tests/reference_parity/_generate_did_synth_shiftshare_R.R`,
  which states the reference's row conventions.
- The redacted view of the old module (signatures only; the helper strips
  docstrings). A grep for callers printed four lines of the old file: two
  sentences of docstring and the texts of the two warnings ("Share matrix is
  collinear; dropping N collinear share column(s) ..." and "The share matrix
  is numerically near-singular even after dropping exactly collinear columns
  (condition number ...)"). The warning texts are pinned by tests
  (`match="3 collinear share column"`, `"near-singular"`); the new texts were
  written to satisfy those patterns.
- Outputs of the previous implementation on 40 random designs (OLS and IV,
  weights, clusters, duplicated shares, weak first stages), recorded before
  the file was replaced.

Not read: any source of `ShiftShareSE`, `ssaggregate`, or R's QR routine.

## Specification

Inputs: outcome `y1`, shift-share variable `X = W g`, shares `W` (n x S),
controls `Z` (with intercept), endogenous `y2` (IV) or none (OLS: `y2 = X`),
weights `w`, region clusters, `beta0`, `alpha`.

1. `r(v)`: residual of the weighted least-squares regression of `v` on `Z`.
   `Xr = r(X)`, `y1r = r(y1)`, `y2r = r(y2)`.
2. `beta = <Xr, y1r>_w / <Xr, y2r>_w`, `e = y1r - beta y2r`.
3. Conventional rows (the reference's conventions, from the generator header
   and confirmed on the fixtures): IV homoscedastic `RSS/n`, EHW and region
   cluster without small-sample factors; OLS homoscedastic `RSS/(n - p)`,
   EHW times `n/(n - p)`, cluster times `G/(G-1) (n-1)/(n-p)`,
   `p = ncol(Z) + 1`. Scores `w_i Xr_i e_i`.
4. Collinear shares: a Householder QR that accepts columns left to right and
   sets aside a column whose trailing norm is below `1e-7` times its original
   norm. On 700 random black-box designs (duplicates, exact sums, zero
   columns, columns whose relative residual norm lies in `[0.3e-7, 3e-7]`)
   the accepted columns equal `qr(W)$pivot[1:rank]`. The norms are
   recomputed at every step (R's help page does not say how it tracks them).
   A warning names the number of dropped columns; a second one fires when
   the kept columns have a condition number above 1e8.
5. Control-adjusted shocks `h`: coefficients of the weighted regression of
   `Xr` on the kept share columns. Black-box check on the weighted fixture:
   unweighted `h` misses R by 8e-4 relative, weighted `h` matches to 1e-15.
6. `R_s(u) = sum_i W_is w_i u_i`. AKM SE
   `sqrt(sum_s (h_s R_s(e))^2) / |<Xr, y2r>_w|`, normal CI and p-value.
7. AKM0: `(a - b t)^2 <= z^2 sum_s (c_s - d_s t)^2`, `a = <Xr, y1r>_w`,
   `b = <Xr, y2r>_w`, `c_s = h_s R_s(y1r)`, `d_s = h_s R_s(y2r)`. Leading
   coefficient positive: bounded interval. Negative with positive
   discriminant: union of two half-lines, reported as `ci_l > ci_r` (both R
   and StatsPAI; the set is `(-inf, ci_r] U [ci_l, inf)`). Otherwise the
   whole line (`-inf, inf`). The AKM0 "SE" is the interval length over
   `2 z` (infinite unless bounded). The AKM0 p-value uses the null-imposed SE
   `sqrt(sum_s (c_s - d_s beta0)^2) / |b|`.
8. BHJ aggregation: residualise `y`, `x` on `Z` (weighted), exposure
   `e_n = sum_l w_l S_ln`, `s_n = e_n / sum e_n`,
   `ybar_n = sum_l w_l S_ln yr_l / e_n`; shock-level IV of `ybar` on `xbar`
   and an intercept with instrument `g`, weights `s_n`, HC0. Warn when the
   controls do not span the sum of shares (relative residual above 1e-8).

## Evidence

- `test_did_synth_shiftshare_parity.py` (R and Stata, all rows, BHJ),
  `test_shiftshare_weights_parity.py` (weighted, 1e-12),
  `test_akm_collinear_shares_R_parity.py` (kept columns equal R's; AKM SE),
  and the new `test_akm_ci_shapes_R_parity.py` (six designs x OLS / IV with
  every AKM0 set shape, weights, clusters, `beta0 = 0.5`, `alpha = 0.10`;
  fixture by `_generate_akm_ci_shapes_R.R`).
- The 40-design comparison with the previous implementation: every row
  equal to 1e-12 or better (AKM0 endpoints of near-degenerate sets 3e-11).
