# Clean-room spec: `sp.rddensity` native path (audit row H12)

Clean-room record for `src/statspai/diagnostics/rddensity.py` (native
path) and the shared primitives in `src/statspai/rd/_lpdensity.py`.
Written 2026-10-20 from the inputs listed below. The reference
implementation (R `rddensity` 2.6, GPL-2; Stata `rddensity`) was run as a
black box only; no reference source was read. Facts marked **[bb]** were
learned only from reference outputs, not from a paper or help page.

## Inputs

- cattaneo2020simple (JASA 2020) and its supplemental appendix:
  estimator, Theorem 1 constants, variance estimator, manipulation
  statistic, Section 5.2 jackknife standard error.
- cattaneo2018manipulation (Stata Journal 2018): the unrestricted test,
  robust bias correction with `q = p + 1`, jackknife SE as default, the
  `comb` bandwidth rule, the options.
- R help pages `rddensity`, `rdbwdensity`, `rdplotdensity`; Stata help
  `rddensity.sthlp` (binomial-test conventions).
- Black-box runs: R `rddensity` / `rdbwdensity` on synthetic samples
  designed to isolate one quantity at a time (see "How the conventions
  were pinned").

## Estimator (default unrestricted model, triangular kernel)

Data `x_1..x_n` (non-finite values dropped), cutoff `c`, order `p`
(default 2), bandwidths `h_l`, `h_r`.

- Regressand: the leave-one-out empirical CDF of the **full** sample,
  `Y_i = #{j != i : x_j <= x_i} / (n - 1)` (cattaneo2018manipulation,
  Section 2.2: "(leave-one-out) c.d.f. estimator for the full sample").
  With tied values every member of a tie group gets the count of the
  whole group (mass-point handling, R's default `massPoints = TRUE`)
  **[bb]**.
- Side `s` uses the observations of that side with `|x - c| <= h_s`,
  design `r(u) = (1, u, ..., u^k)` in `u = (x - c)/h`, weights
  `K(u)/h`, `K(u) = 1 - |u|`. The density is
  `f_s = beta_1 / h`. Order `k = p` gives the conventional estimate,
  `k = p + 1` the bias-corrected one (cattaneo2018manipulation, 2.6).
- A side whose window holds fewer distinct values *with positive kernel
  weight* than `k + 1` is not identified: estimate and SE are NaN.

## Standard error (default `vce = jackknife`)

For window observations `i` let `psi_i = sum_{j in window, j != i}
r_j K_j 1[x_i <= x_j]` (the change in the normal-equation right-hand
side `X'W Y` when observation `i` is removed from the empirical CDF).
Then `Var(beta) = A^-1 (sum_{i in window} psi_i psi_i') A^-1 / (n-1)^2`,
`A = X'WX`, and `se_s = sqrt(Var_11) / h`. **[bb]** The paper gives the
U-statistic form of this estimator (supplement, Section 5.2); the exact
finite-sample form above (sum over the window only, uncentred, `j != i`,
values compared with ties for mass points) was identified by matching R
to 1e-11. The SE scales exactly as `1/(n-1)` when observations far from
the cutoff are added, which ruled out full-sample sums.

## Test

`T = (f_r - f_l) / sqrt(se_l^2 + se_r^2)` with the order-`p+1`
estimates and SEs, two-sided normal p-value. `model_info['conventional']`
is the same at order `p`. If the statistic is not computable (an
unidentified side, zero or NaN SE), StatsPAI reports `T = 0`, `se = 0`,
`p = 1` and NaN densities (kept from the previous implementation; R
returns NA; see "Open items").

`CausalResult`: `estimate`/`pvalue` = T and its p-value; `se`/`ci` on the
density-difference scale; `n_eff_*` = observations with `|x - c| <= h`
on the side.

## Default bandwidths (`bwselect = comb`)

For targets `l`, `r`, `diff`, `sum`:
`h = (V / (2p B^2 n))^(1/(2p+1))` (MSE-optimal for `V/(nh) + h^(2p) B^2`,
cattaneo2018manipulation 2.5; the formula and the `n` = full sample
were confirmed against `rdbwdensity`'s `variance`/`biassq` columns).

- `V_s = n h1 se_s(h1)^2`: jackknife SE of the order-`p` fit at a pilot
  `h1`; `V_diff = V_sum = V_l + V_r`.
- `B_s = F^(p+1)_s / (p+1)! * e_1' S_s^-1 c_s`: triangular-kernel
  boundary constant (Theorem 1 of cattaneo2020simple, integrals over
  `[0,1]` or `[-1,0]`) times the side's `F^(p+1)` estimate from an
  order-`p+2` fit (coefficient `p+1`) at a second pilot `h2`.
  `B_diff = B_r - B_l`, `B_sum = B_r + B_l`. If either side's derivative
  fit is not identified both `B` are NaN **[bb]** (rdbwdensity reports
  NA in all four rows).
- `comb`: `h_l = median(h_l, h_diff, h_sum)`, `h_r = median(h_r, h_diff,
  h_sum)` (help page).

### Pilots **[bb]**

Both pilots are normal-reference MSE-optimal bandwidths for a boundary
estimator of `F^(v)` with an order-`P` polynomial, **uniform-kernel**
constants, sample mean and `ddof = 1` SD, evaluated at `c`:

`h^(2P+1) = (2v-1) V_u f(c) / (2 (P+1-v) B_u^2 F^(P+1)(c)^2 n)`,
`V_u = (v!)^2 [S^-1 G S^-1]_vv`, `B_u = v! [S^-1 c]_v / (P+1)!`,
`S_ab = 1/(a+b+1)`, `c_a = 1/(a+P+2)`,
`G_ab = int int min(s,t) s^a t^b` over `[0,1]^2`.

- `h1`: `P = p`, `v = 1`. `h2`: `P = p + 2`, `v = p + 1`.
- Found by solving for the pilot that reproduces rdbwdensity's
  `variance` and `biassq` columns on normal samples (n = 100 to 1600,
  several means): `h1^(2p+1) n F^(p+1)(c)^2 / f(c)` is constant in `n`
  and in `(c - mean)/sd`, equal to 12/5, 960/7, 16800, 203212800/55 for
  `p = 1..4`, and the `h2` constant is 58240/3, 47174400/11 for
  `p = 1, 2`. Those rationals equal the uniform-kernel formula above
  exactly; the triangular-kernel version does not match.
  `tests/reference_parity/test_rddensity_bandwidth_regimes.py` pins the
  values.

### Sample-size rules **[bb]** (help: `nLocalMin = nUniqueMin = 20+p+1`)

Applied to both pilots and to the four final bandwidths:
`h <- max(h, d_k)` where `d_k` is the distance to the `k`-th closest
observation and to the `k`-th closest distinct value on each relevant
side (both sides for pilots, `diff`, `sum`; one side for `l`, `r`), a
side with fewer than `k` contributing its farthest observation; then
`h <- min(h, farthest observation)`. A NaN bandwidth (bias not
estimable) takes the lower bound. The cap and the NaN rule are not in
the help page; they were pinned on designs with a hole at the cutoff,
15 observations on one side, mass points and 9 support points
(`rdbwdensity_R.json`).

## Binomial tests (Stata convention)

Stata help: "the initial (smallest) window contains 20 observations, and
its length is also used as the increment". First half-width = distance
to the `bino_n`-th closest observation (both sides), or `bino_w` (one
value or one per side); increment `bino_wstep` (default: the first
half-width). Window `k` counts `c - w_l <= x < c` and `c <= x <= c + w_r`;
exact binomial test of `P(below) = bino_p` on the left count. No table
when `n < bino_n` (and no `bino_w`). R's default differs (20 per side);
StatsPAI follows Stata, which is what `test_rddensity_all_stata.py` pins.

## How the conventions were pinned

- Point estimates: `(i-1)/(n-1)` matched R to 1e-12; `i/n` did not.
- Jackknife SE: perturbation runs (move far observations, add
  observations, change the other side) showed the left SE depends only
  on left-window data and on `n` through `1/(n-1)`; the window-only
  uncentred sum then matched to 1e-11.
- Mass points: count-of-group regressand and value comparisons matched
  `massPoints = TRUE`; position-based versions matched `FALSE`.
- Bandwidths: `rdbwdensity` exposes `V` and `B^2`; the pilots were solved
  from them and checked against 15 regime cells and 77 design x order
  cells (T, densities, SEs and bandwidths within 3e-8 of R for
  `p <= 3`).

## Open items for the owner

- `T = 0, p = 1` when the statistic is not computable is a silent
  non-answer (R returns NA). Kept for API stability; candidate for NaN +
  warning.
- Orders `p >= 5` are numerically fragile on both sides (order-`p+2`
  fits); StatsPAI and R agree only to about 1e-4 at `p = 7`.
