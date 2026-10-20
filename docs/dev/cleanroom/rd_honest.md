# Clean-room spec: honest RD (`rd/_rdhonest.py`, `rd/bias_aware.py`, `rd/rd_discrete.py`)

Audit row H8 (`docs/dev/2026-10-20-license-provenance-audit.md`). Reference: R `RDHonest`
1.0.1.9000 (GPL-3), used only as a black box. Written 2026-10-20.

## Inputs used

- Papers: Armstrong and Kolesár (2018, `armstrong2018optimal`; arXiv 1511.06028), Armstrong and
  Kolesár (2020, `armstrong2020simple`; arXiv 1606.01200), Kolesár and Rothe (2018,
  `kolesar2018inference`; arXiv 1606.04086, sections 5.1–5.2), Imbens and Kalyanaraman
  (NBER w14726, 2009, and CEMMAP CWP05/10, 2010; `imbens2012optimal`).
- Documentation: `RDHonest` help pages (`RDHonest`, `CVb`, `RDHonestBME`,
  `RDSmoothnessBound`, rendered with `tools::Rd2txt`), its vignette `vignettes/RDHonest.Rmd`
  (GitHub, documentation only), and `NEWS.md`.
- Black-box runs of `RDHonest()` / `RDHonestBME()` on the committed fixtures and on about 40
  simulated designs (outputs only, including the returned `data$sigma2` per-unit variances).
- Existing tests and fixtures (`test_rdhonest_parity.py`, `test_rdhonest_cluster_R_parity.py`,
  `test_rd_iv_rd_R_parity.py`, `test_rd_open_R_parity.py`, and their generators).
- Seen in the contaminated files (redacted view): the module docstrings of the three files (the
  `_rdhonest.py` docstring states the bias, standard-error and interval formulas, which are also in
  the vignette), function names, signatures, and one-line docstrings that name the reference's
  internal steps (a preliminary variance chain "EHW -> IK -> Silverman", an "order-0 uniform fit",
  a floor "enough distinct and enough raw points per side", a "side-wise mean", a "Moulton"
  correlation). These named steps guided which hypotheses were tested; every constant and rule below
  was then fixed by black-box output, not by those strings. No reference source was read.

## Sharp RD (`honest_rd`, `honest_fit`)

- Estimator: local linear on each side, kernel `K((x-c)/h)`, closed support `|u| <= 1`
  (uniform kernel includes the boundary point; checked on integer data). `tau = sum w_i Y_i`.
- Worst-case bias: Hölder `M/2 |sum w_i d_i^2 sign(d_i)|`; Taylor `M/2 sum |w_i| d_i^2` (vignette).
- SE: nearest-neighbour variance, computed **within the window** (units with positive kernel weight)
  separately on each side; neighbours tied at the J-th distance all enter and the factor is
  `m/(m+1)` with `m` the number used (black box: per-unit `data$sigma2`, 1e-12).
  With clusters: `sqrt(sum_g (sum_{i in g} w_i e_i)^2)`, `e` local linear residuals at `h`.
- Interval `tau ± cv_{1-alpha}(B/se) se`; `cv` = root of `Phi(c-t) - Phi(-c-t) = 1-alpha`.
  One-sided limits `tau ∓ (B + z_{1-alpha} se)`; p-value `P(|N(B/se,1)| > |tau|/se)`.
- Effective observations: window count × `sum k_unif^2 / sum w^2`; leverage `max w^2 / sum w^2`
  (vignette).
- Rule-of-thumb `M`: global quartic in `x - c` on each side, largest `|f''|` over that side's data
  range, max over sides (vignette; matches to 1e-11).

## Bandwidth choice

- Criterion under a preliminary variance homoskedastic on each side: MSE `B^2 + sd^2`, FLCI
  `cv(B/sd) sd`, OCI `2B + (z_{1-alpha} + z_beta) sd`.
- Preliminary variance: residuals of a **triangular** local linear fit (whatever the user's kernel)
  at the Imbens-Kalyanaraman pilot, floored at the smallest window with three support points per
  side; mean of squared residuals over in-window units on each side. Clusters add a common
  within-cluster covariance (average of `e_i e_j`, `i != j`, same cluster).
- IK pilot (identified on ~30 designs; all 279 R-selected bandwidths reproduced to <= 1.3e-7):
  `h1 = 1.84 sd(x) N^(-1/5)`; `f = (N_-(h1)+N_+(h1))/(2 N h1)`; side variances (divisor `n-1`)
  within `max(h1, three-support-point floor)` (the floor binds only with discrete data; found on the
  discrete fixture); `m3` from a global cubic with a jump on all data, no 0.01 floor;
  `h2 = 7200^(1/7) (s2_side/(f m3^2))^(1/7) N_side^(-1/7)` (unrounded constant);
  unweighted quadratics on `[0,h2]` give `m2`; `r = 2160 s2/(n2 h2^4)`; kernel constant `C_K`
  computed exactly for the edge kernel (3.4375439, not 3.4375). The two unrounded constants were
  found by backing out the implied regularisation constant (2168 = 2160 (3.56/7200^(1/7))^4).
- Search: Brent's bounded method (`fmin`) over (second distinct `|x-c|` (max over sides),
  `max |x-c|`), tight tolerance. A single bracketed search reproduces R even where the criterion has
  several near-equal local minima. Uniform kernel: global minimum over the grid of data distances.

## Fuzzy RD (`rd_bias_aware_fuzzy`)

- Jumps of `Y` and `D` with the same weights; 2x2 NN covariance (within window, per side) or the
  cluster sum; Wald ratio, delta-method SE, bias `(M_Y + |tau| M_D) B_1/|Delta_D|` (help page,
  vignette). Rule-of-thumb `M` on `Y` and `D` separately.
- Default `h`: minimise `(M_Y + |T0| M_D)^2 B_1(h)^2 + sum w^2 a_h' Sigma a_h` with
  `a_h = (1, -tau_h)`, `tau_h` the Wald ratio at `h`, `Sigma` the side-wise preliminary covariance of
  `(Y, D)` from triangular fits at the IK pilot of `Y`; `T0 = 0`. With clusters, plus
  `a' rho a (sum_g (sum w)^2 - sum w^2)`. Black box: h does not depend on `M_D` at `T0 = 0`; all
  checked cells match to <= 6e-8.
- Anderson-Rubin set (Noack-Rothe): roots of `|dY - t dD| - cv(b(t)/s(t)) s(t)`, solved exactly; may
  be a union of rays or unbounded. No reference implementation; pinned by its defining identity.

## Discrete running variable (`rd_discrete`)

- `bsd`: the sharp engine above.
- `bme` (vignette and Kolesár-Rothe section 5.2): window `|x-c| <= h`; polynomial design with the
  jump first; joint sandwich of `(theta, cell means)` scaled by `n/(n-1)` (black box: SE off by
  exactly `sqrt(n/(n-1))` without it); limits over all `W = (g-, g+, s-, s+)`, two-sided at
  `1-alpha/2` per side, one-sided at `1-alpha`. `eff.obs` = window count; leverage of the jump
  weights. R's p-value adds `maximum.bias` (outcome units) to the z statistic; reported as
  `p_value_rdhonest`.
- **Open item:** R's `maximum.bias` is always `|b(W)|` for one of the two binding `W` (upper or
  lower one-sided limit), but which end is reported follows no rule found from six designs. We
  report the bias at the farther end of the two-sided set; this matches the fixtures (symmetric
  cases) but can differ from R off-fixture. It affects only `maximum_bias` and the two p-values of
  `method='bme'`, not the interval.
