# Clean-room spec: `nonparametric/lprobust.py` (audit row H5)

Rewrite of `sp.lprobust_at_point` and the five `sp.lpbwselect_*` selectors,
which until 2026-10-20 self-declared being a port of R `nprobust` (GPL-3).
Implementation: `src/statspai/nonparametric/lprobust.py` (public API,
validation) and `src/statspai/nonparametric/_lprobust_bw.py` (building
blocks).

## Inputs used

- Papers: Calonico, Cattaneo and Farrell (2019, JSS) [@calonico2019nprobust],
  Sections 2.1 to 2.6; Calonico, Cattaneo and Farrell (2018, JASA) and its
  supplement S.II.2 [@calonico2018effect]; the supplement of Calonico,
  Cattaneo and Titiunik (2014, ECMA) S.2.6 [@calonico2014robust] (pilot
  constants, regularization term).
- Help pages: R `?lprobust`, `?lpbwselect`; Stata `lprobust.sthlp`,
  `lpbwselect.sthlp`.
- Black box: R `nprobust` 1.0.0 (exported `lprobust()` / `lpbwselect()` only)
  on the committed CSVs and on constructed data; the replaced Python module
  through its public outputs only (to measure output changes).
- Not read: any source of `nprobust` / `rdrobust` / `lpdensity` (R, Stata,
  Python), the unredacted replaced module or its history,
  `tests/reference_parity/_fixtures/_verify_rdbwselect_cascade.R`.
  One line of the old module was shown once by a Python `RuntimeWarning`
  (a `bw = (...) ** r_exp if denom > 0 else inf` expression); it states the
  closed form below, which the paper gives anyway.

"Black-box" below means the statement was learned by running the reference,
not from the papers.

## Estimand and estimator (paper, Section 2.1 and 2.5)

At a point `x0`, with `u_h = (x - x0)/h`, `W_h = diag(K(u_h)/h)`, `R_p` the
basis `(1, u_h, ..., u_h^p)` and `q = p + 1`:

- conventional: `tau_us = nu! e_nu' (R_p'W_h R_p)^{-1} R_p'W_h Y / h^nu`;
- bias-corrected: replace `R_p'W_h` by
  `Xi = R_p'W_h - rho^(p+1) Lambda_p e_(p+1)' (R_q'W_b R_q)^{-1} R_q'W_b`,
  `rho = h/b`, `Lambda_p = R_p'W_h u_h^(p+1)`;
- both are linear in `Y`; `se = nu!/h^nu * sqrt(sum_i l_i^2 v_i)` with `l`
  the row of the corresponding operator.

Sample: rows with `K > 0` in the `h` window or the `b` window (union). The
compact kernels are supported on `|u| <= 1` (Epanechnikov `0.75(1-u^2)`,
triangular `1-|u|`, uniform `0.5`); the gaussian kernel uses every row.
Black-box: `N` counts the union window (Stata fixture: 317 at `h=0.8, b=1.2`).

Differences from the reference, kept from the replaced code: no `bwcheck` is
applied to supplied bandwidths (black-box: R `lprobust` enlarges a supplied
`h` and `b` to the distance to the 21st nearest observation by default;
`did_had` passes bandwidths it wants used as given). The function needs more
than `p + 2` rows in the window and raises if the `h` window holds fewer than
`p + 1` distinct `x` values or the `b` window fewer than `q + 1`.

## Variance: nearest-neighbour residuals (CCF 2018 supplement S.II.2.3)

`v_i = J_i/(J_i + 1) (y_i - mean of matched y)^2`; the match set is every
other row whose distance to `x_i` is at most the `nnmatch`-th smallest such
distance (ties at the last neighbour all enter), computed within the
estimation sample. Black-box: the reference's standard errors are reproduced
to 1e-15 with ties matched whole and residuals computed inside the window,
not on the full sample. Only `vce = "nn"` is implemented.

## MSE-optimal direct plug-in (`mse-dpi`, paper Section 2.3, `p - nu` odd)

Closed form `h = ((1+2nu) V / (2(o+1-nu) n D))^(1/(2o+3))` for a fit of order
`o` estimating derivative `nu`, with (all at a pilot bandwidth `v`):

- `V = n v nu!^2 sum_i l_i^2 v_i` (pre-asymptotic variance, NN residuals in
  the pilot window);
- bias multiplier `c = nu!/(o+1)! e_nu' Gamma^{-1} Lambda` at `v`;
- `D = c^2 (d^2 + 3 bwregul Var(d))`, `d` an estimate of `m^(o+1)(x0)` from
  an order-`o+1` local fit at a second bandwidth, `Var(d)` its NN sandwich
  variance. The `3 Var` regularization is the one of the CCT (2014)
  supplement. `D <= 0` gives an infinite bandwidth (then capped).

Black-box facts (each reproduced to 1e-12 or better against R):

1. Pilot `v = C_K min(sd, IQR/1.349) n^(-1/5)`, `sd` with `n-1`, quartiles
   type 7, `C_K` = 2.34 / 2.576 / 1.843 / 1.06 (epa / tri / uni / gau), the
   normal-reference constants of the CCT supplement rounded as shown.
2. Three stages ("second generation"):
   - stage 1: `o = p+2, nu = p+2`; `d` = `m^(p+3)` from an order-`p+3`
     kernel-weighted fit at bandwidth `max(x) - min(x)`; no regularization.
     Found on exactly quartic data, where only a weighted wide fit (not an
     unweighted global polynomial) reproduces the reference for
     non-uniform kernels;
   - stage 2: `o = p+1, nu = p+1` gives `b`, with `d` from stage 1's
     bandwidth; regularized;
   - stage 3: `o = p, nu = deriv` gives `h`, with `d` at `b`; regularized.
3. Every stage bandwidth (and the pilot) is floored at the distance to the
   `bwcheck`-th nearest observation; stage bandwidths are capped at
   `max|x - x0|`. R rejects `bwcheck = 0`; so does this module.
4. Returned extra keys (kept from the old API): `V_h = (1+2nu) V / nu!^2`,
   `B_h = 2(p+1-nu) (c d)^2 / nu!^2`, likewise `V_b`, `B_b` for stage 2,
   so `h^(2p+3) = V_h / (n (B_h + regularization))`.

`p - deriv` even raises `MethodIncompatibility` (the paper's Case 2 needs a
numerical minimization; not implemented, as before).

## Rule of thumb (`mse-rot`)

Black-box (old outputs and R): `h = ((1+2nu) V / (2(p+1-nu) n B^2))^(1/(2p+3))`
with `B = C1 m^(p+1)(x0)/(p+1)!` and `V = C2 sigma^2 / f(x0)`:

- `C1 = e_nu' S^{-1} c`, `C2 = e_nu' S^{-1} T S^{-1} e_nu` with the moments
  over the one-sided support `[0, 1]`, at every evaluation point (interior
  points included);
- `m^(p+1)` from an order-`p+3` kernel-weighted fit at bandwidth `range(x)`;
- `sigma^2 = SSR/(n - p - 4)` of an unweighted global polynomial of degree
  `p + 3`;
- `f` the uniform-kernel density estimate `#{|x - x0| <= v} / (2 n v)` at the
  pilot `v` of the chosen kernel;
- no cap, no `bwcheck` floor (black-box: R returns `h > max|x - x0|` and
  `h < bmin` unchanged). The gaussian kernel is rejected.

## Integrated selectors

- `imse-rot` (black-box): `V` and `B^2` of `mse-rot` averaged over
  `quantile(x, seq(0.05, 0.95, by = 0.025))` (37 points), then the same
  closed form. It ignores `eval` and `imsegrid`.
- `imse-dpi` (black-box, isolated on exactly cubic data where `d` is known):
  grid `imsegrid` equally spaced points on `[min x, max x]`; at each point the
  pointwise stages 1 and 2 above (with that point's floored pilot);
  `b = closed form(mean V_b, mean c^2 d^2)` (no regularization); `h = closed
  form(mean V_h, mean c^2 d^2)` where each point's `d` is estimated at that
  point's own regularized, capped and floored `mse-dpi` `b`. No cap or floor
  on the two averages. It ignores `eval`.

## Coverage-error rule of thumb (`ce-rot`, paper Section 2.6)

`h_ce = h_mse n^(-p/((2p+3)(p+3)))`, `b_ce = b_mse n^(-(q+2)/((2q+5)(q+3)))`.
Black-box: the exponent's parity is that of `o - nu` (for `b`, `q - (p+1)`
is even), and the `bwcheck` floor acts before the rescaling.

## Numerics

QR on `sqrt(W) R` for every weighted fit; the global polynomial on `x`
rescaled to `[-1, 1]`. Against a 50-digit evaluation of the same formulas:
`mse-rot` at the largest `x` of the Stata fixture (`p = 3`, uniform) gives
2.305973027596 (high precision), 2.305973027593 (this module),
2.305973106 (R), 2.305973787 (replaced code); `mse-dpi` (`p = 1`, epa, same
point) 2.813292466211719 / 2.8132924662117165 / – / 2.813292466380733.

## Verification

- `tests/reference_parity/test_lprobust_parity.py` (Stata fixture and R
  values pinned before the rewrite) passes unchanged.
- `tests/reference_parity/test_lprobust_cleanroom.py`: 600 R black-box cases
  from `_fixtures/_generate_lprobust_selectors_R.R` at rel 1e-9 (1e-8 for the
  rule of thumb at `p = 3`), plus identities.
- `did_had` tests unchanged.

## Output changes against the replaced code

Over ~9,300 calls (5 datasets, 4 kernels, orders 0-3, option sweeps):
`p = 1, deriv = 0` (the `did_had` path) agrees to 2e-9 relative; larger
differences (up to 1.5e-6, `mse-rot` `p = 3` at the sample maximum) are
rounding in ill-conditioned fits, and the high-precision checks above show
the new values are the accurate ones. Behaviour changes: `bwcheck = 0` now
raises (it returned the maximum distance for every bandwidth, an artefact of
indexing the `bwcheck`-th distance at position -1); singular pilot windows
raise `NumericalInstability` where the old code sometimes returned numbers
from a singular solve; supplied-bandwidth fits whose window has too few
distinct `x` values raise instead of returning unstable numbers.
