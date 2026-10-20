# Clean-room spec: `sp.rdhte` / `sp.rdbwhte` / `sp.rdhte_lincom` (`statspai.rd.hte`)

Audit row H10. The previous `rd/hte.py` described its vce normaliser as R
`rdhte:::.rdhte_normalize_vce` and itself as "the Python port of R `rdhte`"
(rdhte 0.2.0, GPL-3). This file is the clean-room record for the replacement
(`rd/hte.py` for the API, `rd/_hte_core.py` for the least-squares core); every
function body was rewritten.

## Inputs used

- Papers: Calonico, Cattaneo, Farrell, Palomba and Titiunik, *Treatment Effect
  Heterogeneity in Regression Discontinuity Designs* (arXiv 2503.13696, bib key
  `calonico2025treatment`) and the software article *rdhte: Conditional Average
  Treatment Effects in RD Designs* (arXiv 2507.01128, `calonico2025rdhte`),
  sections 2-5 (estimator (8), the Stata `reg y t##c.x##c.w` description,
  subgroup analysis with per-group bandwidths, h = b, HC3 default, comparison
  with `rdrobust`).
- Help pages of R `rdhte` 0.2.0: `?rdhte`, `?rdbwhte`, `?rdhte_lincom`.
- Black-box runs of R `rdhte()` / `rdbwhte()` / `rdrobust::rdbwselect()` on
  `rd_iv_hte.csv` (scratch cells, then the committed fixture
  `tests/reference_parity/_fixtures/_generate_rdhte_R.R` -> `rdhte_R.json`) and
  the existing `rd_iv_rd_R.json` cells.
- Black-box calls of the old Python functions to keep StatsPAI's own API
  (evaluation grid, result fields, lincom over rows, `b=`, `p=0`).
- Seen anyway (contamination record): the redaction helper printed first
  docstring lines of the old helpers, naming `model.matrix(Y ~ T * Xp * W)`,
  "weighted `lm` on `|Xc| <= h`" and `sandwich::vcovCL(type=vce)`, and the
  docstring of `tests/reference_parity/test_rd_iv_rd_R_parity.py` states the
  `vcovCL` small-sample factors. These were treated as claims and each was
  settled by black-box runs (**BB**). No function body, R or Python, was read.

## Estimator (paper eq. (8), software article section 2.2)

Notation: cutoff `c`, `Xc = x - c`, `T = 1(Xc >= 0)`, moderators `W`
(`d` columns), polynomial order `p` (point estimate) and `q = p + 1` (bias
correction; user `q` allowed), kernel `K` on `[-1, 1]` (triangular `1-|u|`,
Epanechnikov `3/4 (1-u^2)`, uniform `1/2`).

1. Design: the full interaction `[1, T] x [1, Xc, ..., Xc^r] x [1, W]` (`r = p`
   for the estimate, `q` for the bias correction), i.e. Stata
   `reg y t##c.x##c.w`. **BB**: R's `coef.full` names are exactly these terms.
2. Window and weights: observation `i` enters when `|Xc_i| <= h_side(i)`, with
   weight `K(Xc_i / h_side(i)) / h_side(i)` (scale irrelevant). Left side uses
   `h_left`, right side `h_right`.
3. Weighted least squares; the reported conventional coefficients are those of
   `T` and `T:W_j` from the order-`p` fit, the bias-corrected ones (and all
   standard errors / covariances) from the order-`q` fit on the same window
   (h = b, article section 5.1). StatsPAI extension: `b=` gives the order-`q`
   fit its own window.
4. CATE at `w`: `beta_T + w' beta_{T:W}` (eq. (7)).

**BB**: with these rules `coef`, `coef.bc`, `se.rb` and `vcov` match R to
<= 3e-12 relative on 21 cells (four HC types, CR1, asymmetric `h.l / h.r`,
`p = 2`, `q = 3`, three kernels, a non-zero cutoff, a numeric 0/1/2 moderator,
binary and three-level factor moderators, per-group bandwidths, selected
bandwidths).

## Variance (`sandwich::vcovCL` conventions, settled by black box)

Bread `A = X'WX`, score rows `s_i = w_i x_i e_i`, `n` observations in the
window, `k` columns, leverage `l_i = w_i x_i' A^-1 x_i`:

| vce | meat |
| --- | --- |
| hc0 | `n/(n-1) sum s_i s_i'` |
| hc1 | `n/(n-k) sum s_i s_i'` |
| hc2 | `sum s_i s_i' / (1 - l_i)` |
| hc3 | `sum s_i s_i' / (1 - l_i)^2` |
| cr1 | `G/(G-1) (n-1)/(n-k) sum_g S_g S_g'` |

`V = A^-1 meat A^-1`. Defaults (help page): `hc3`; with `cluster=`, `cr1`.
Remapping (help page): with a cluster, `hc0` / `hc1` -> `cr1` with a warning;
without one, `cr1/cr2/cr3` -> `hc1/hc2/hc3` with a warning. Clustered HC2/HC3
(`cr2` / `cr3`) are not implemented and raise `MethodIncompatibility`.

## Moderator types

- Continuous (one or more numeric columns not of the next kind): one fit, one
  bandwidth pair; reported coefficients `T`, `T:<z>`... **BB**: a numeric
  0/1/2 column is treated as continuous.
- Subgroups: a single column whose values are exactly {0, 1}, or a
  categorical / object / bool column (new: R accepts factors). `W` = dummies of
  all but the first level (sorted; category order for `Categorical`), each
  group with its own window and bandwidth; reported effects are `beta_T` (first
  level) and `beta_T + beta_{T:W_g}` with the matching covariance (`L V L'`).
  **BB**: equal to R's `coef` / `vcov` for 0/1 and three-level factors.

## Bandwidths (`rdbwhte`)

`cct_bandwidth` (StatsPAI's clean CCT selector, which matches R `rdbwselect`)
on `(y, x)` with `p`, `q`, `kernel`, `bwselect`, `vce` (cluster: `hc1` with the
cluster ids, i.e. CR1); continuous moderators ignore `W`; subgroups run it on
each group's rows. **BB**: equal to R's `rdbwhte` / `rdhte` default `h` to
<= 1e-11 for mserd, msetwo, cerrd, HC1, clusters, `p = 2` and the uniform
kernel. `bw.joint` is not offered. Return value of `sp.rdbwhte` (old API): a
float when there is one common bandwidth, otherwise a frame
`level, h_left, h_right`.

User `h`: a scalar is used on both sides of every group; a pair is
`(h_left, h_right)` (R's `h.l`, `h.r`). **BB / reference quirk**: R's
`h = c(a, b)` with a continuous moderator recycles the two values across
observations (coefficients differ from `h.l = a, h.r = b` although the
reported `h` is the same); not reproduced.

## StatsPAI output (old API, black box)

- `detail`: one row per evaluation point (`z_value`, `cate`, `cate_bc`, `se`,
  `ci_lower`, `ci_upper`, `pvalue`), inference on `cate_bc` with `se` (normal).
  Subgroups: one row per level. Continuous, one moderator: `n_eval` quantiles
  of `z` at `linspace(0.1, 0.9, n_eval)`. Several moderators: per-column
  quantiles at `linspace(0.1, 0.9, k)`, `k = max(3, floor(n_eval^(1/d)))`, the
  Cartesian product (first column slowest), and when it has more than
  `n_eval` points the rows at `round(linspace(0, G-1, n_eval))`.
  `eval_points=` overrides (reshaped to `(-1, d)`).
- Headline: `estimate = mean(cate)`, `se = sqrt(w' L V L' w)` with equal
  weights, z / p-value / interval from the bias-corrected mean.
- `model_info`: `coef_names`, `coef`, `coef_bc`, `se_rb`, `vcov`,
  `bandwidth_h`, `bandwidth_by_level`, `bandwidth_b`, counts, `ate*`,
  `heterogeneity_test` (Wald on the `T:W` coefficients, or on the differences
  from the first group), `_L` (evaluation design).
- `p = 0` (local constant) is accepted (StatsPAI extension; R requires
  `p >= 1`).
- `rdhte_lincom`: `linfct` (rows over coefficients) or `weights` (over detail
  rows); estimate from the conventional coefficients, z / interval / p-value /
  joint Wald from the bias-corrected ones (**BB**: equal to R's
  `rdhte_lincom` individual and joint rows). One row returns scalars.

New input checks: `alpha` and `n_eval` validated, `q > p`, positive `h`.
