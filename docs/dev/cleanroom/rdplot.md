# Clean-room spec: `rdplot` numbers (`statspai.rd._rdplot_numbers`)

Audit row H7 (rdplot half). The previous `rd/_rdplot_core.py::rdplot_numbers`
declared itself a line-by-line port of R `rdrobust::rdplot` 4.0.0 (GPL-3). This
file is the clean-room record for its replacement, `rd/_rdplot_numbers.py`.

## Inputs used

- Papers: Calonico, Cattaneo and Titiunik, *Optimal Data-Driven Regression
  Discontinuity Plots* (bib key `calonico2015optimal`), main text sections 2-5 and
  supplement sections 1 and 3; the R Journal article on `rdrobust`
  (`calonico2015rdrobust`), section on `rdplot`; the 2017 Stata Journal update of
  `rdrobust` (section 2.4 and 5, RD plots), which has no bib key in `paper.bib`
  and is therefore not cited in code.
- Help pages: R `?rdplot` and Stata `rdplot.sthlp` (rdrobust 4.0.0).
- Black-box runs of R `rdplot()` (rdrobust 4.0.0, R 4.5.2) and of base R
  `quantile()`; the committed fixture is
  `tests/reference_parity/_fixtures/_generate_rdplot_R.R` -> `rdplot_R.json`.
- Black-box calls of the old Python function, to learn the returned keys and
  types and the validation messages that tests pin.
- Seen anyway (contamination record): the redaction helper printed the old
  module's docstring. It named, as R internals, a function `J.fun`, bin edges by
  `seq` / type-7 `quantile`, membership by `findInterval(..., rightmost.closed =
  TRUE)`, and a left-side `rev(-bin)` indexing quirk. Each of these was treated as
  a claim to test, and each fact below that depends on it was pinned by a
  black-box run (marked **BB**). No function body, R or Python, was read.

## Number of bins

Notation: side sample `(X_i, Y_i)`, `i = 1..N` (left: `X < c`; right: `X >= c`),
sorted by `X` with ties kept in input order (stable sort; `Y` are the
concomitants). `n` = total sample size after dropping incomplete rows. Range of
the side `L = c - min X` (left) or `max X - c` (right). `dX_i`, `dY_i` are the
consecutive differences, `Xbar_i` the midpoints.

Preliminary fits: least squares of `Y` and of `Y^2` on `1, (X-c), ..., (X-c)^4`
on the side sample, unweighted. The order 4 does not depend on `p`, the kernel,
`h` or `weights` (**BB**: the selected bins are identical for `p = 0..6`, all
kernels, any `h`, random weights). `mu'` is the derivative of the `Y` fit.

| constant | formula | source |
| --- | --- | --- |
| `B_ES` | `L^2 / (12 n) * sum_i mu'(X_i)^2` | eq. (8)/(10) |
| `B_QS` | `N^2 / (24 n) * sum_{i>=2} dX_i^2 mu'(Xbar_i)^2` | eq. (18)/(20) |
| `V_ES` spacings | `sum dX_i dY_i^2 / (2 L)` | eq. (7)/(9) |
| `V_QS` spacings | `sum dY_i^2 / (2 N)` | eq. (17)/(19) |
| `V_ES` polynomial | `sum dX_i s2(Xbar_i) / L` | Remark 1 |
| `V_QS` polynomial | `mean_i s2(X_i)` | Remark 2 |
| `v` | sample variance of `Y` on the side, divisor `N - 1` | "V-hat" of eq. (14) |

`s2(x) = mu2(x) - mu1(x)^2` from the two order-4 fits (Remark 1). **BB**: a
negative `s2` value is replaced by `v` before averaging. This was found by
probing a 12-point side with spike outcomes and exact polynomials: the implied
`V` was a homogeneous, shift- and sign-invariant but non-quadratic function of
`y`, and for exactly fitted polynomials `V / v` came out as `m/N` (7/12, 3/4,
1/4, 2/3, 1/8, 9/16 ...), i.e. the share of points where the rounding-level `s2`
is negative. With this rule 91 of 93 precise probes and 107 of 108 large-sample
cells match; the misses are noise-free exact fits where the sign of a
rounding-level `s2` decides (unmatchable in principle).

Selectors (`n` is the **total** sample size, **BB**; the paper's `n`):

- IMSE: `J_IMSE = ceil((2B/V)^(1/3) n^(1/3))`, eq. (1)/(4), (11)/(21), (14)/(24).
- Mimicking variance: `J_MV = ceil(v / V * n / log(n)^2)`, eq. (3)/(6), (14)/(16),
  (23)/(26).
- `es`, `espr`, `qs`, `qspr` use `J_IMSE`; the `mv` rules use `J_MV`; `pr` rules
  use the polynomial `V`, the others the spacings `V`. `B` is the same for both.
- `scale`: `J = ceil(scale * J)` (help pages; supplement section 1 writes the
  rescaled choice with a ceiling). **BB**: R 4.0.0 multiplies without rounding
  and stops with an error unless `scale * J` is an integer, and it refuses a
  two-element `scale` ("the condition has length > 1"); Stata documents
  `scale(l r)`. Here both forms are accepted; for integer factors the result is
  R's.
- `nbins` given: `J = nbins` (scalar applies to both sides), `scale` ignored,
  `J_IMSE` / `J_MV` still reported (**BB**). `binselect` still decides the
  spacing: R labels the result "manually evenly spaced" but builds quantile
  edges when `binselect` is a `qs*` rule (**BB**, edges match bit for bit).
- Side with constant outcome (`v = 0` or no positive `V`): **BB** R warns "Not
  enough variability in the outcome variable below/above the threshold", uses one
  bin (also when `nbins` is given) and reports `J_IMSE = Inf`, `J_MV = NaN`. Here
  the same warning and one bin; `J_IMSE = J_MV = 1` as the old function returned
  and `tests/test_core_degenerate_inputs.py` pins (the tuples are integers).

Mass points (`masspoints='adjust'`, default): a side is flagged when the number
of distinct `X` values is below 80% of its size (**BB**: 79 distinct of 100
switches, 80 does not). Then the spacings rules switch to their `pr` versions
and a warning "Mass points detected in the running variable." is issued.
`'check'` only warns; `'off'` does nothing. `mass_points_adjusted` is true when
mass points were found in `'adjust'` mode, whatever the rule (old behaviour).

## Partition

- ES (section 3): edges `e_j = lo + j * d`, `d = (hi - lo) / J`, `j = 0..J`,
  capped at `hi`; left `lo = min X`, `hi = c`; right `lo = c`, `hi = max X`.
  **BB**: R's interior edges equal these bit for bit in all 42 ES fixture cases.
- QS (section 4): edges are sample quantiles of the side sample at `p_j`.
  The paper uses `F^{-1}(j/J)`; **BB**: R uses the type-7 quantile,
  `r = 1 + (N-1) p`, `k = floor(r)`, `g = r - k`, `(1-g) X_(k) + g X_(k+1)`
  (`X_(k)` when equal), which matches base R `quantile()` bit for bit (300/300
  random cases), with `p_j = j * (1/J)` (not `j/J`; the two differ in the last
  bit, e.g. `6 * 0.1 = 0.6000000000000001`, and that decides which bin gets an
  observation lying on an edge; `rdplot_qs.csv` in the fixture pins this).
- Membership: bin `j` is `[e_j, e_{j+1})`, the last bin closed (paper, section
  2.1.2).
- **Deliberate difference 1 (BB)**: when `lo + J d` rounds one step below
  `max X`, R puts the largest observation in an extra right-hand bin whose
  `min/max/mean_bin` are missing. Here the last bin is closed at `max X` as in
  the paper (the old Python function did the same). The parity test merges R's
  overflow row into the last bin.
- **Deliberate difference 2 (BB)**: with empty left-side bins, R's
  `rdplot_min_bin` / `rdplot_max_bin` columns on the left are shifted against the
  other columns (e.g. a bin `[-0.5853, -0.55)` reported with mean `x = -0.5284`).
  Here every non-empty bin carries its own edges. Counts, means and intervals
  agree with R.
- Empty bins are dropped (R stops with an error when a bin is empty).
- The first/last edge R reports can differ from `min X` / `max X` by tens of
  ulps (**BB**); irrelevant at the 1e-9 tolerance and not reproduced.

## Per-bin numbers (help pages; 2017 Stata Journal section 2.4)

`rdplot_N`, `rdplot_mean_x`, `rdplot_mean_y` (unweighted, raw `y` even with
covariates or weights, **BB**), `rdplot_min_bin`, `rdplot_max_bin`,
`rdplot_mean_bin` = midpoint of the edges, `rdplot_se_y = sd(y)/sqrt(N_j)`
(divisor `N_j - 1`), interval `mean +- t_{N_j-1, 1-(1-ci/100)/2} * se`.
**BB**: a single-observation bin has `se = 0` and a degenerate interval.

`cluster` (StatsPAI addition, no R counterpart): Stata `mean y, vce(cluster g)`
within each bin, `se = sqrt(G/(G-1) sum_g (sum_{i in g} (y_i - ybar))^2) / N_j`,
`t(G-1)`; a bin with one cluster gets a missing `se` and interval (black box of
the old function; `tests/test_rd_cluster_diagnostics.py` checks the CR1 identity
against statsmodels).

## Global polynomial

- Weights `K((x - c)/h_s) / h_s` times `weights`, `h_s` the side's bandwidth;
  uniform `K = 1/2` on `|u| <= 1`, triangular `1 - |u|`, Epanechnikov
  `3/4 (1 - u^2)` (help pages; **BB** for the `1/h` scale and the closed
  support). Default `h = (c - min X, max X - c)`.
- Fit sample: observations with positive kernel weight; `N_h` counts them
  (**BB**: triangular / Epanechnikov exclude the extreme point at `|u| = 1`,
  uniform includes it). The window is decided from `u = (x - c)/h`, so the
  extreme point is at `|u| = 1` exactly.
- Without covariates: weighted least squares on `1, (x-c), ..., (x-c)^p` per
  side; `coef` is `(p+1) x 2` on powers of `x - c` (**BB**, non-zero cutoff
  cases). Rank deficiency: minimum-norm solution.
- With covariates (`covs_eval = "mean"`): one weighted least-squares fit of `y`
  on `[1(x<c) P(x-c), 1(x>=c) P(x-c), Z]` with a common `gamma` (**BB**; the
  joint fit of the 2017 Stata Journal article). Curve = side polynomial
  `+ Zbar' gamma`, `Zbar` the mean over the whole sample, not the window (**BB**).
  Collinear covariates: **BB** R keeps the first of a dependent set and reports
  only the kept coefficients (e.g. `(z, 2z, w)` -> `z, w`). Here columns
  collinear with the polynomial or with earlier covariates are dropped with a
  warning; for a constant covariate R keeps an unidentified coefficient
  (2.04 on one probe) while it is dropped here (the curve is identical).
- `vars_poly`: 500 points per side, `linspace(c - h_l, c)` and
  `linspace(c, c + h_r)` (**BB**, including `h` beyond the data).

## Validation

- `binselect` outside the eight rules, `masspoints` outside
  `adjust/check/off`, `kernel` outside uniform/triangular/Epanechnikov
  (`uni/tri/epa` accepted, case ignored): `MethodIncompatibility`.
- `c` not strictly inside the range of `x`: `MethodIncompatibility` "c should be
  set within the range of x" (R's condition and message, **BB**).
- Fewer than 20 complete observations: `DataInsufficient` "Not enough
  observations to perform bin calculations" (**BB**). A side with fewer than two
  observations: `DataInsufficient` (R stops with "missing value where
  TRUE/FALSE needed").
- `nbins` not positive integers, `scale` / `h` not positive, `p` not a
  non-negative integer, `ci` outside (0, 100), negative weights, misaligned
  lengths: `MethodIncompatibility` (the old function crashed or returned NaN).
- Rows with a missing `y`, `x`, weight, covariate or cluster are dropped.

## Evidence

- `tests/reference_parity/test_rdplot_R_parity.py`: 69 R cases (all eight rules
  x p in {1, 2, 4}; three kernels with and without `h`; `nbins` scalar and pair;
  `scale`; covariates; weights; cutoffs 0.2 and -0.3; `ci = 90`; a mass-point
  design under adjust / off / check; quantile edges on observations), rel 1e-9
  (coefficients 1e-8, see the test docstring), plus analytic checks.
- Old vs new on 170 random configurations: identical counts and `J`, largest
  scaled difference 7.9e-12.
