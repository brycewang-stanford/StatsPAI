# Clean-room spec: grid density estimator (`lpdensity_numbers`, audit row H7)

Clean-room record for `lpdensity_numbers` in
`src/statspai/rd/_lpdensity.py`, the numerical core of
`sp.rdplotdensity` (previously in `rd/_rdplot_core.py`). Written
2026-10-20. Reference: R `lpdensity` 2.5 (GPL-2), run as a black box
only; no reference source was read. **[bb]** marks facts learned only
from reference outputs.

## Inputs

- cattaneo2020simple: estimator (Section 2), Theorem 2 variance
  estimator, supplement Section 2.3.
- The `lpdensity` software article, Journal of Statistical Software
  101(2), 2022, doi:10.18637/jss.v101.i02 (not in `paper.bib`; DOI and
  issue read from the article's masthead and the R help page): `p`, `q`,
  `v`, kernels, robust bias-corrected `f_q`, `scale`, `massPoints`.
- R help page `lpdensity` (argument semantics, `Estimate` columns).
- Black-box runs of `lpdensity()` on fixed samples with and without ties.

## Signature (unchanged from the previous implementation)

`lpdensity_numbers(data, grid, bw, p=2, q=None, v=1,
kernel="triangular", scale=1.0, mass_points=True)` returns a dict with
`grid`, `bw` (one per grid point), `nh` (count of `|x - g| <= bw`),
`f_p`, `f_q`, `se_p`, `se_q`.

## Estimator

- Sorted sample `x` (non-finite dropped), `n` observations. Regressand
  `F(x_i) = #{j : x_j <= x_i} / n` (the ECDF, self included). With
  `mass_points=True` tied values share the group count; with `False`
  ties are broken by sort position, `F = i/n` **[bb]** (both matched R
  to 1e-12).
- At grid point `g` with bandwidth `h`: window `|x - g| <= h`, design
  `r(u)`, `u = (x - g)/h`, weights `K(u)/h` with triangular `1-|u|`,
  uniform `1/2` or Epanechnikov `0.75(1-u^2)`.
  `F^(v)(g) = v! beta_v / h^v`; `f_p` uses order `p`, `f_q` order `q`
  (default `p + 1`; `q <= p` leaves `f_q`, `se_q` NaN, as R's `NA`).
- `scale` multiplies estimates and standard errors (help page).
- Not identified (fewer distinct positive-weight values than `order + 1`)
  gives NaN, as R's `NA`.

## Standard error

Theorem 2 of cattaneo2020simple with the sums written out:
`psi_i = (1/n) sum_{j in window} r_j K_j (1[x_i <= x_j] - F(x_j))` for
every observation `i` of the full sample (`j = i` included; positions
instead of values without mass points),
`Var(beta) = A^-1 (sum_i psi_i psi_i') A^-1`, `A = X'WX`,
`se = v! sqrt(Var_vv) / h^v`. This is the plain plug-in form of the
theorem; it matched R to 1e-12 with and without ties **[bb]** (the
paper does not say which finite-sample variant the package uses).
Implementation: the inner sums are suffix sums over the sorted window,
so a grid point costs `O(n log n)`.

## Edge cases and validation

- `bw` must be positive and finite, scalar or one per grid point;
  `v <= p`; kernel one of the three names. Violations raise
  `MethodIncompatibility` (the previous implementation produced
  undefined numbers or silently used another kernel).
- A grid point with an empty window has `nh = 0` and NaN estimates.

## Evidence

`tests/reference_parity/test_lpdensity_parity.py` against
`_fixtures/lpdensity_R.json` (generator `_generate_lpdensity_R.R`): ten
cells (three kernels, orders 1-4, `v = 0, 1, 2`, per-point bandwidths,
`scale`, mass points on/off), all within 1.5e-11 of R; plus the exact
density of an equally spaced sample.
