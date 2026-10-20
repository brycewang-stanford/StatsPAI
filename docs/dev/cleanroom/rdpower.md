# Clean-room spec: `sp.rdpower` / `sp.rdsampsi` (`statspai.rd.rdpower`)

Audit row H11. The previous `rd/rdpower.py` described two of its functions as
transcriptions of R `rdpower` 3.0 internals (GPL-3): the sample-size solver and
the data mode of `rdsampsi`. This file is the clean-room record for the
replacement; every function body in the module was rewritten.

## Inputs used

- Paper: Cattaneo, Titiunik and Vazquez-Bare (2019), *Power calculations for
  regression-discontinuity designs*, Stata Journal 19(1) (bib key
  `cattaneo2019power`), section 2 (power functions, the `rdpow` adjusted
  variance, the `rdsampsi` allocation and sample-size equation).
- Help pages of R `rdpower` 3.0: `?rdpower`, `?rdsampsi`.
- Black-box runs of R `rdpower()` / `rdsampsi()` on `rdsenate.csv`; the committed
  fixture is `tests/reference_parity/_fixtures/_generate_rdpower_R.R` ->
  `rdpower_R.json`, plus the older `rdlocrand_R.json` cells.
- Black-box calls of the old Python functions (design mode only), to keep the
  design-stage numbers, the result fields and the printed summaries.
- Seen anyway (contamination record): the redaction helper printed the first
  docstring line of the old helper `_power_newton_raphson` ("Smallest sample
  size reaching power ``beta``, as ``rdpower`` solves it") and of
  `_rdsampsi_from_data`. The docstring of
  `tests/reference_parity/test_rdlocrand_parity.py` states that R "ceilings the
  sample size inside the Newton-Raphson solve" and "splits the sides by
  sqrt(variance)". Both were treated as claims and settled by the black-box runs
  below (marked **BB**). No function body, R or Python, was read.

## Power function (paper, section 2.2)

With `z = Phi^-1(1 - alpha/2)` and the standard error `s` of the robust
bias-corrected estimator,

    power(tau) = 1 - Phi(tau / s + z) + Phi(tau / s - z).

It equals `alpha` at `tau = 0`, is symmetric in `tau`, and increases with
`|tau| / s`.

## Data mode of `rdpower`

`s` is `sp.rdrobust(data, y, x, c, **kwargs).se` (robust bias-corrected
standard error). **BB**: R `rdpower(data=)` reports `se.rbc` equal to
`rdrobust`'s robust standard error and `power.rbc = power(tau)` with that `s`
(fixture cells `power_*`, rel 1e-9). The result reports `n_left` / `n_right`
as the side sample sizes `N-` / `N+` (old behaviour), and the minimum detectable
effect `mde = (z + Phi^-1(target_power)) * s` (old behaviour; one-tail
approximation, `target_power` defaults to 0.8).

R's conventional power (`power.conv`) is not offered. **BB**: R adds the two
estimated side biases (`h+^(1+p) bias.r + h-^(1+p) bias.l`) where the paper's
`B` is their difference, so it is not reproduced.

## Data mode of `rdsampsi`

Notation from one `rdrobust` fit: side sizes `N-`, `N+`, window counts
`Nh-` (`c - h- <= X < c`), `Nh+` (`c <= X <= c + h+`), total `n = N- + N+`,
bandwidths `h-`, `h+`, derivative order `nu`, and the robust bias-corrected
variance of each side's contribution, `v-`, `v+` (`v- + v+ = s^2`). The scaled
variances of the paper are `V± = v± * n * h±^(1 + 2 nu)` (**BB**: equal to R's
`var.l` / `var.r` and `Vl.rb` / `Vr.rb`).

1. Variance at a hypothetical total sample size `m` (paper's `rdpow` formula
   with `m` replacing `n`): `V(m) = (V+ / h+^(1+2nu) + V- / h-^(1+2nu)) / m`.
2. `n* = ceil(m*)`, `m*` the root of `power(tau; sqrt(V(m))) = beta`, found by
   bracketing (Brent). **BB**: R's `sampsi.tot` equals this on every one of the
   41 cells that R returns (it is the paper's equation solved exactly, then
   rounded up).
3. Treated share of the window sample, paper section 2.2:
   `rho = sqrt(V+) / (sqrt(V-) + sqrt(V+))` (**BB**: R's default `nratio`).
4. Window-to-total factor `f = rho N+/Nh+ + (1 - rho) N-/Nh-`; window sample
   `M = n* / f`. Side sizes `M+ = ceil(rho M)`, `M- = ceil((1 - rho) M)`,
   reported total `M+ + M-`. **BB**: matches R's `sampsi.h.r`, `sampsi.h.l`,
   `sampsi.h.tot` exactly on all 41 cells (the paper says "rounding to the
   closest integer"; R rounds each side up, which is what is reproduced).

`ratio`, `var_*`, `h_*` (design-stage arguments) are not used in data mode, as
before.

**Reference divergence (BB)**: for some inputs R's iterative solver does not
return (on the Senate data `tau = 0.5`, or `tau = 1` with `beta = 0.9`; the
fixture records the timeout for `tau = 0.5`). Here the root is bracketed, so an
answer is always returned.

Per-side variances come from `rd._cct_inference.cct_bias_corrected(...,
components=)` at the bandwidths of the `rdrobust` fit. Options that this
primitive does not reproduce (`donut`, `bootstrap`, `scalepar != 1`, ...) are
refused; the sum `v- + v+` is checked against `rdrobust`'s robust variance
(rel 1e-6) and a mismatch raises instead of returning a number.

## Design mode (StatsPAI's own calculator; no R counterpart)

Arguments: outcome variances `var_left`, `var_right`, side sizes `n_left`,
`n_right`, bandwidths `h_left`, `h_right`. **BB** (old function):

    s^2 = 0.35 * (var_left / (n_left h_left) + var_right / (n_right h_right)).

The constant 0.35 is the previous StatsPAI convention, pinned by
`test_power_design_mode_is_unchanged` (power 0.809 at `tau = 0.15`,
`n = 500 + 500`) and `test_sampsi_design_mode_is_unchanged` (978 = 489 + 489).
It is a planning heuristic, not a property of any reference.

`rdsampsi` design mode: `n_right = ratio * n_left`; `n_left = ceil(r)` where `r`
solves `power(tau; s(r, ratio r)) = target_power` (continuous, both tails);
then `n_right = ceil(ratio * n_left)`, `n_total = n_left + n_right`, at least
one per side. **BB**: this reproduces the old function on every probe (ratio
0.33 / 0.7 / 1 / 1.5 / 2, unequal variances and bandwidths, alpha 0.1, target
0.9) except one: with `alpha = 0.2`, `target_power = 0.3`, `tau = 0.2` the old
function returned 21 + 21 where the smallest sample reaching power 0.3 is
17 + 17. The old numbers fit the one-tail approximation
`tau / s = z_(1-alpha/2) + z_target`, which overstates `n` when the opposite
tail carries non-negligible power (target close to `alpha`). The exact
equation is used here, as in data mode (⚠️ correctness note in CHANGELOG).

Input checks (new; the old function returned nonsense or crashed): `tau`
finite (non-zero for `rdsampsi`), variances, sizes and bandwidths positive,
`0 < alpha < 1`, `alpha < target_power < 1`, `ratio > 0`. Violations raise
`MethodIncompatibility` (a `ValueError`) with a recovery hint. `y=` / `x=`
without `data=` (or `data=` without both) raises as before.
