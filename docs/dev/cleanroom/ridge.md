# Clean-room specification: `sp.ridge`

Audit row M12 of `docs/dev/2026-10-20-license-provenance-audit.md`:
`regression/ridge.py` said its scaling, GCV score and plug-in penalties "are
those of R `MASS::lm.ridge`" (MASS 7.3-65, GPL-2 | GPL-3) without saying how
they were obtained. `MASS`'s help page documents none of these formulas, so
the body of `ridge()` was deleted unread and rewritten from this
specification. `RidgeResult` (generic result plumbing: `predict`, `summary`,
`to_dict`) is unchanged.

## Sources (and only these)

- Hoerl and Kennard (1970) `[@hoerl1970ridge]`: the ridge estimator.
- Golub, Heath and Wahba (1979) `[@golub1979generalized]`: generalized
  cross-validation, `V(lambda) = (RSS/n) / (1 - tr(H)/n)^2`.
- The help page of `MASS::lm.ridge` (arguments, "the intercept is not
  penalized", returned components `scales`, `GCV`, `kHKB`, `kLW`).
- Black-box runs of `MASS::lm.ridge` on `longley` and on a synthetic file,
  with and without intercept (`_generate_ridge_blackbox_R.R` in
  `tests/reference_parity/_fixtures/`); each candidate formula below was
  fitted to those outputs and agrees to <= 4e-15 relative.
- Black-box runs of the previous `sp.ridge` (its outputs, warnings and error
  types on 22 calls), to keep StatsPAI's own API decisions.

## Estimator

Rows with a missing outcome or regressor are dropped. `X` excludes the
intercept column; `p` its number of columns, `n` rows, `c = 1` with an
intercept and `0` without.

1. Centre `y` and the columns of `X` at their means when there is an
   intercept (not at all without one). Scale each column by its root mean
   square, `s_j = sqrt(sum_i x_ij^2 / n)` of the centred column (black box:
   `scales` agree to 1.4e-15). A column with `s_j <= 1e-10 * max_i |x_ij|`
   (or all zero) does not vary: refuse it.
2. `Z = X_c / s`, SVD `Z = U D V'`. Singular values below
   `max(n, p) * eps * d_max` are treated as zero (rank `r`).
3. For each penalty `lambda >= 0`: `b(lambda) = V diag(d / (d^2 + lambda)) U' y_c`
   on the non-null subspace (at `lambda = 0` this is the minimum-norm
   least-squares solution), `df(lambda) = sum d^2 / (d^2 + lambda)` over
   the non-null singular values, residual sum of squares `RSS(lambda)` of
   the centred model, and `GCV(lambda) = RSS / (n - df)^2`. This is
   Golub-Heath-Wahba's criterion multiplied by the constant `1/n`, so it
   has the same minimiser; it is the scale `lm.ridge` reports (black box:
   agrees to 3e-15, while the `n RSS / (n - df)^2` form is off by the
   factor `n`).
4. Coefficients on the original scale: slopes `b / s`, intercept
   `ybar - xbar' (b / s)`.
5. Plug-in penalties from the `lambda = 0` fit `b0`, with
   `sigma2 = RSS(0) / (n - r - c)` (`r = p` for linearly independent
   regressors; for a rank-deficient design the previous `sp.ridge` used the
   rank, and so does this one):
   - Hoerl-Kennard-Baldwin: `(p - 2) sigma2 / (b0' b0)`;
   - Lawless-Wang: `(p - 2) sigma2 n / (b0' Z'Z b0)`.

   The rules are named after "Ridge regression: some simulations"
   (Hoerl, Kennard and Baldwin 1975, doi:10.1080/03610917508548342) and
   "A simulation study of ridge and other regression estimators" (1976,
   doi:10.1080/03610927608827353); both DOIs checked against Crossref and
   doi.org on 2026-10-20 (the papers themselves were not read, so no
   claim is made here about the constants they print). The constants
   above, including the factor `p - 2` and the residual degrees of
   freedom, are fixed by the black-box outputs: both agree to <= 2e-15
   with intercept (`n - p - 1`) and without (`n - p`); the alternatives
   with factor `p`, or with `n - p` / `n` degrees of freedom, are off by
   2.8% to 100%. Both are NaN when `p <= 2` or `n - r - c <= 0` (previous
   `sp.ridge`).

## API decisions kept from the previous `sp.ridge` (black box)

- Input: `formula` (via `core.utils.create_design_matrices`; it wins when
  `y` / `x` are also given) or `y` plus `x` (columns used as named, an
  intercept added, incomplete rows dropped). Neither:
  `MethodIncompatibility("ridge: provide a formula or (y, x).")`.
- `lambda_`: a number, a sequence of numbers, `None` or a rule name. `None`
  builds the default grid: 0 followed by 100 values log-spaced from
  `1e-4 * m` to `1e4 * m`, `m` the mean squared singular value of `Z`
  (101 rows). A rule name (`"gcv"`, `"hkb"`, `"lw"`) means the default grid
  with `select=<rule>`; any other string is refused ("rules"). Negative or
  non-finite penalties are refused (">= 0"); an empty sequence is refused.
- `select` in `{"gcv", "hkb", "lw"}`, else refused ("select").
- One penalty (scalar or one-element sequence): `selection = "given"`,
  `lambda_` and `lambda_gcv` are that value, whatever `select` says.
- Several penalties: the path keeps the given order; `lambda_gcv` is the
  first grid minimiser of GCV. `select="gcv"` uses it; `"hkb"` / `"lw"`
  refit at the plug-in value (refused when it is NaN, naming the
  regressor and degrees-of-freedom requirement); the path is unchanged.
- A rank-deficient design whose grid contains 0 warns (`UserWarning`,
  "minimum-norm") that the `lambda = 0` row is the minimum-norm solution.
- Result: `RidgeResult(params, lambda_, path, lambda_gcv, lambda_hkb,
  lambda_lw, n_obs, selection, formula, _design_info, _names, _fitted)`;
  `path` columns `lambda, gcv, df` then the coefficient names (intercept
  first, named `Intercept`); the reported `df` is the trace of the full hat
  matrix, i.e. the slopes' `df(lambda)` plus 1 for the intercept (GCV uses
  the slopes' `df` alone); `_design_info` is the formula layer's
  `design_info` when it has one (patsy designs), else `None`; `_names` the
  coefficient names; `_fitted` the in-sample fitted values at the chosen
  penalty.
