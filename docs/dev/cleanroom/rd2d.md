# Clean-room spec: `sp.rd2d` / `sp.rd2d_bw` (boundary discontinuity designs)

Audit row H9. `rd/_rd2d_location.py` and `rd/_rd2d_distance.py` described
themselves as a line-by-line port of R `rd2d` 1.0.0 (GPL-3), and
`rd/rd2d.py` was rebuilt in the same commit. All three are replaced:
`rd/rd2d.py` (API, results), `rd/_rd2d_inputs.py` (validation, evaluation
points, distances), `rd/_rd2d_common.py` (kernels, weighted fits,
variances), `rd/_rd2d_location.py`, `rd/_rd2d_distance.py`. Every function
body is new. `rd/_rd2d_plot.py` (plotting, not in the audit row) is
unchanged.

## Inputs used

- Papers: Cattaneo, Titiunik and Yu, *Estimation and Inference in Boundary
  Discontinuity Designs: Location-Based Methods* (arXiv 2505.05670, sections
  2-3.3 and SA-9) and *Distance-Based Methods* (arXiv 2510.26051, sections
  3-5.3); *rd2d: Causal Inference in Boundary Discontinuity Designs* (arXiv
  2505.07989, sections 3-4, including 3.7 "Regularization Strategies"); the
  CRAN package record is bib key `cattaneo2025boundary`.
- Help pages of R `rd2d` 1.0.0: `?rd2d`, `?rdbw2d`, `?rd2d.distance`,
  `?rdbw2d.distance`, `?summary.rd2d`.
- Black-box runs of R `rd2d()`, `rdbw2d()`, `rd2d.distance()`,
  `rdbw2d.distance()` (their `$bws`, `$mseconsts`, `$main`, `$params.cov`,
  `$main.0/1`) on the fixture data, on derived data (outcomes replaced by
  exact polynomials, rescaled scores, subsamples, single-observation
  perturbations of the outcome to read off the support of each constant,
  duplicated scores), and the committed fixtures `rd_open_R.json` and the new
  `rd2d_floor_R.json` (`_generate_rd2d_floor_R.R`).
- Black-box calls of the previous StatsPAI functions, to keep the result
  fields, validation messages, default evaluation points and warnings.
- Seen anyway (contamination record): the redaction helper printed the old
  modules' function names and signatures, whose first docstring lines named
  the R routines they mirrored (e.g. `rdbw2d_rot`, `get_coeff`,
  `rdbw2d_joint_effective_info`, `rdbw2d_bw_v2`, `rd2d_cer_factor`,
  `rd2d_distance_poly_lm`, `rdbw2d_distance_local_intercepts_multi`,
  `rd2d_bwcheck_bounds`, `get_cov_half_v2`) and two module-level constants: a
  kernel-name map and a table of kernel constants `(1/6, 4/(3 pi))`,
  `(3/20, 3/(2 pi))`, `(1/4, 1/pi)`, `(1, 1/(4 pi))`. The test files and
  `docs/dev/campaign_phase3/r2_rd_open.md` state R's radial-kernel variance
  quirk and name `rd2d_h_normalize` / `get_invH`. The names were treated as
  hints only; the kernel constants were re-derived (below) and every rule
  was settled by black-box runs. No function body, R or Python, was read.

## Location approach: estimation for given bandwidths (paper sections 3.1-3.3)

Point `b`, side `t` (0 control, 1 treated), bandwidths `(h_t1, h_t2)`:

- Weights: product `K(d1/h_t1) K(d2/h_t2) / (h_t1 h_t2)`, `d = x - b`;
  radial `K(|d| / r) / r^2` with `r = sqrt(h_t1^2 + h_t2^2)` (**BB**: the
  reference fits at that radius). Kernels on `[-1, 1]`: triangular
  `1-|u|`, Epanechnikov `3/4(1-u^2)`, uniform `1/2`; Gaussian unbounded.
- Fit: WLS of `y` on all monomials of degree `<= p` (and `<= q`) in `d`.
  Target: intercept; partial derivative `(n1, n2)`: `n1! n2!` times that
  coefficient; direction `v`: `v1 * coef(x1) + v2 * coef(x2)`.
- Estimate `tau = theta_1 - theta_0`; fuzzy: `tau_Y / tau_T` per order, its
  influence `infl_Y / tau_T - tau_Y / tau_T^2 * infl_T` (section 3.6).
- Variance: sandwich of each side's influence. HC0; HC1 with `N/(N-K)`
  per side ("separate") or with `N`, `K` pooled over both sides ("joint");
  HC2/HC3 leverage scaling. Clusters: CR1 `G/(G-1) (N-1)/(N-K)`, and with
  `vce='hc1'` also `N/(N-K)` (**BB**: both factors); "joint" sums each
  cluster's contributions from both sides before squaring (cross term
  `C01`), so `Var(tau) = V0 + V1 - 2 C01`. With clusters, hc2/hc3 are reset
  to hc1 with a warning.
- Covariance across points: the cross product of the points' influence
  vectors, each scaled by the square root of its own small-sample factor
  (**BB**: the geometric mean of the two points' factors), cluster sums as
  above.
- `bwcheck` in estimation (**BB**): product kernels take `lam`, the
  `bwcheck`-th smallest `max(|d1|/s1, |d2|/s2)` (`s` the score SDs when
  `stdvars`, else 1; distinct locations with `masspoints='adjust'`) and fit
  with `max(h_j, lam s_j)`; radial kernels take the `bwcheck`-th smallest
  raw Euclidean distance and use `max(h_j, lam)`. The reported bandwidth is
  the requested one. Observations on the window edge up to `1e-12` are on
  the edge (lattice data).

**BB**: estimates, SEs, covariances and counts match R to ~1e-13 for
HC0-HC3, joint/separate, clusters (HC0, HC1), `p = 2` Epanechnikov,
derivatives, directions, Gaussian, fuzzy, user bandwidths below the floor.
Radial SEs are the sandwich of the fit actually run; R's are
`(h1^2 + h2^2)/(h1 h2)` times larger (class-4 reference defect, unchanged
from 1.29).

## Location approach: bandwidth selection

Scores standardised by their SDs (divisor `n - 1`) when `stdvars`, the
bandwidth computed in those units and multiplied back per coordinate.

1. **Pilot** (normal reference; **BB** to 1e-15): for the radial kernel with
   second moment `mu2 = int u1^2 K` and roughness `R = int K^2` (unit disk,
   normalised: triangular `(3/20, 3/(2 pi))`, Epanechnikov
   `(1/6, 4/(3 pi))`, uniform `(1/4, 1/pi)`, Gaussian `(1, 1/(4 pi))`, from
   polar integration),
   `a = (2 R / (mu2 m I(S)))^(1/6)`, `S` the sample covariance of the
   (standardised) scores, `I(S) = [tr(S^-2)/2 + (tr S^-1)^2/4] /
   (4 pi |S|^(1/2))` the integrated squared Laplacian of the normal density,
   `m = n` (`masspoints='off'`) or the number of distinct score locations.
   One pilot for all points and both kernel types. With `bwcheck`, per side
   `a_t = max(a, k-th smallest Euclidean distance)` (distinct locations with
   'adjust').
2. **Constants** per point and side (all **BB**, single-observation
   perturbations located each constant's window):
   - `var_t = a_t^2 Var(theta_t)` from the order-`p` fit at `a_t` (the
     selected vce, joint/cluster rules as above; joint clusters give
     `var_01`).
   - `d_t` = median Euclidean distance from `b` to side `t`'s observations.
   - Bias factor `F_t(k) = sel' Gamma^-1 E[r_p(U) U^k K]` (scaled design
     `U = d / a_t`) for each monomial `k` of degree `p+1`; likewise `F2_t`
     for degree `p+2`.
   - Higher-order term `bias_hi_t = a_t * F2_t . c_(p+2)` with `c` the
     degree-`(p+2)` coefficients of an order-`(p+2)` fit at `d_t`.
   - `method='rot'`: `g_t = d_t`. `method='dpi'`: for
     `theta = F_t . c_(p+1)`: `Vth` its variance from an order-`(p+1)` fit at
     `a_t`; `H = F_t Gamma_(p+1)^-1 E[r_(p+1)(U) U^k K]` (degree `p+2`) at
     `a_t`; `B = H . c_(p+2)` and `Rb = Var(H . c_(p+2))` from the
     order-`(p+2)` fit at `d_t`;
     `g_t = ((p+2) Vth a_t^(2p+4) / (B^2 + s Rb))^(1/(2p+6))`.
   - `bias_t = F_t . c_(p+1)` from an order-`(p+1)` fit at `g_t`;
     `reg_t = Var(bias_t)`.
   - Fuzzy (`bwparam='main'`): all constants of the outcome
     `Y / tau_T - tau_Y / tau_T^2 * T`, both jumps from order-`p` fits at
     `a_t`; 'itt': `Y`.
3. **Bandwidth** (paper 3.7 with **BB** details):
   `h = ((2+2 nu) V / (2 (p+1-nu) (B^2 + s R)))^(1/(2p+4))`, `nu` the
   derivative order; common (`*rd`): `V = var_0 + var_1 - 2 var_01`,
   `B = sum_t (-1)^(t+1) (bias_t + scalebiascrct bias_hi_t)`,
   `R = reg_0 + reg_1 - 2 reg_01`; per side (`*two`): side terms; integrated
   (`i*`): sums of `V` and of `B^2 + s R` over points. `s = scaleregul`
   (3 in `rd2d`, 1 in `rd2d_bw`). Then, except for integrated selectors,
   `h >= k-th smallest Euclidean distance` (max over sides for a common
   bandwidth); then the CER rate `n^(1/(2p+4) - 1/(p+4))` (side `n` for
   `*two`). **BB**: the floor precedes the CER rate (Gaussian CER cell).
   Defaults: `bwcheck = 50+p+1` (`rd2d`), 20 (`rd2d_bw`); with mass points
   and no `bwcheck`, `50+p+1`.

**BB**: `$mseconsts` (all eight constants) match to ~1e-12 for every kernel
and type, HC0-HC3, joint/separate, clusters, p = 1, 2, derivatives,
directions (applied as given in standardised coordinates), fuzzy main/itt,
s = 0, 1, 2, scalebiascrct, rot/dpi; bandwidths for all eight selectors.

## Distance approach (distance paper 5.3, software article 4)

- Estimation: per side, an order-`p` (and `q`) polynomial in `|D|` with
  weights `K(|D|/h)/h`; inference fit at the RBC bandwidth; variance,
  clusters, covariances as in the location approach.
- Pilot (**BB**): `c = sigma (4 pi R / (mu2 n))^(1/6)`,
  `sigma^2 = mean(D^2)/2` (the normal reference of an isotropic score),
  per point; with `bwcheck`, `c_t = max(c, k-th smallest |D|_t)` (all
  observations, also with 'adjust').
- Bias polynomial: OLS of `y` on `1, |D|, ..., |D|^(p+1)` over the side's
  observations with `|D|` below the `cqt` quantile; `bias_t = F_t * coef_(p+1)`
  with `F_t` the bias factor of the order-`p` fit at `c_t`;
  `reg_t = F_t^2 Var_OLS(coef_(p+1))` (homoskedastic).
- `var_t = c_t^2 Var(intercept)` of the order-`p` fit at `c_t` whose
  residuals are those of the bias polynomial.
- Fuzzy main: jumps from order-`p` fits at `c_t` (floored).
- `h = (2V/((2p+2)(B^2 + s R)))^(1/(2p+4))` with the same combinations
  (`s` = 1 in both `rd2d` and `rd2d_bw`), floor (max over sides), CER rate,
  then kinks: unknown `h * n^(1/(2p+4) - 1/4)` (side `n` for `*two`), RBC
  bandwidth `h * n^(1/4 - 1/3)`; known kinks
  `min(h, max(h * rate, distance to the nearest kink))`; `q = p` by default
  with an unknown kink.

**BB**: `$mseconsts` and bandwidths to ~1e-13 on all selector, kernel,
`p`, `cqt`, `scaleregul`, vce, cluster, fuzzy, kink and floor variants tried.

## StatsPAI-only parts (old API, black box)

Default evaluation points: boundary `x1 = 0`, `x2` at its median (one point)
or evenly spaced between its 10th and 90th percentiles; a callable boundary
`x2 = f(x1)`: `x1` evenly spaced between the 10th and 90th percentiles (one
point: their midpoint). `approach='pooled'`: `sp.rdrobust` on the signed
distance to the boundary curve (nearest curve point by grid plus
golden-section search). Mass points: distinct locations below 80% of `n`
(**BB**: R warns at 0.7996, not at 0.8) warn "Mass points detected in the
running variables." (plus "Try using option masspoints=adjust." with
'check'). A first-stage jump below `1e-8` gives NaN with a warning (the
selector falls back to the reduced-form bandwidth).

Result fields, messages and defaults are those of 1.29+; old and new agree
to 1.2e-10 on 34 configurations.

## Open items

- `covs.eff` (efficiency covariates) and simulated uniform bands / LBATE are
  not implemented (as before).
- The bandwidth cap at the largest distance (`bw.max` in R's
  `$mseconsts`) is not applied; no design tried reached it.
