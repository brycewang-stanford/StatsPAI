# Clean-room specification: psmatch2-style kernel, radius and LLR matching

Audit row M4 of `docs/dev/2026-10-20-license-provenance-audit.md`. Stata
`psmatch2` (Leuven and Sianesi) carries no licence grant, only a warranty
disclaimer, and the code below said it followed `psmatch2.ado`:

- `matching/_matched_frame.py::psmatch2_se` (the analytic ATT standard error);
- `matching/_stata_lpoly.py` (the whole module: `stata_epanechnikov`,
  `lpoly_predict`, `psmatch2_llr_smoothed_outcome`);
- `matching/match.py`: `_kernel_weight`, `MatchEstimator._fit_kernel`
  (kernel, radius and local linear regression matching), the constants
  `_LLR_MIN_VARIANCE` / `_LLR_MIN_RELATIVE_VARIANCE` and the comment block
  above `_VALID_KERNELS`.

All of it was rewritten from the sources listed next. Nothing below comes
from `psmatch2.ado` or any other ado / Mata source.

## Sources

| Tag | Source |
| --- | --- |
| H | `psmatch2` help file (rendered `help psmatch2`, version 4.0.12): SE formula, kernel names, `bwidth` default, `_weight` semantics, `llr` + Epanechnikov uses the `lpoly` rule-of-thumb bandwidth |
| R-kd | Stata 18 *Base Reference Manual*, `[R] kdensity`, Methods and formulas (kernel table, bandwidth rule) |
| R-lp | Stata 18 *Base Reference Manual*, `[R] lpoly`, Options and Methods and formulas (local polynomial estimator, ROT bandwidth) |
| L | Lechner (2001), `lechner2001identification`: the variance approximation that treats matching weights as fixed |
| FG | Fan and Gijbels (1996), `fan1996local`: local polynomial regression |
| BB | Stata 18 MP + `psmatch2` 4.0.12 run as a black box; only displayed results, `r()` and the generated variables were read. Do-file and log: experiments E1-E8 below |
| FX | Committed fixtures (outputs of earlier black-box runs): `tests/reference_parity/_fixtures/psmatch2_stata.json`, `psmatch2_kernel_data.csv`, `psmatch2_radius_data.csv`, `psmatch2_llr_stata.json`, `psmatch2_llr_data.csv`, `psmatch2_llr_epan_data.csv` |
| OLD | The pre-rewrite StatsPAI functions, called as a black box (inputs in, outputs out) on a battery of 537 cases to pin StatsPAI's own conventions where Stata has none |

Experiments (data are small CSVs; the score is passed with `pscore()` so
no logit fit enters):

- E1 kernel matching with each `kerneltype()`, `bwidth(0.1)`, 40 units;
  `_weight`, `_y`, `r(att)`, `r(seatt)` exported; `r(seatt)` rebuilt from
  the generated variables.
- E2 a hand-made 6-unit design with a control exactly one bandwidth from a
  treated unit (`|u| = 1`), kernel, radius and llr.
- E4 llr when every control has the same score (no slope).
- E5 treated units with no control inside the bandwidth.
- E1b llr with tricube / normal / uniform / biweight, `bwidth(0.3)`.
- E7 `llr` with `kerneltype(epan)`: `r(seatt)` rebuilt from raw and smoothed
  outcomes.
- E8 `llr` without `bwidth()`.
- E6 `lpoly ..., degree(0|1|2) at()` on a 6-point design with tied `x`.

## 1. Kernels

### Matching kernels (`_kernel_weight`)

The help file (H) names five kernel types: `normal`, `biweight`, `epan`
(default), `uniform`, `tricube`. With `z = |u|`, `u = (p_j - p_i) / h`:

| kernel | K(u) used | window |
| --- | --- | --- |
| `epan` | `1 - z^2` | `z <= 1` |
| `biweight` | `(1 - z^2)^2` | `z <= 1` |
| `tricube` | `(1 - z^3)^3` | `z <= 1` |
| `uniform` | `1` | `z <= 1` |
| `normal` | `exp(-z^2/2) / sqrt(2 pi)` | unbounded |

- The shapes are the textbook ones (R-kd lists Epan2 `3/4 (1 - z^2)` and
  biweight `15/16 (1 - z^2)^2` on `|z| < 1`). The normalising constants are
  dropped because every treated unit's weights are divided by their sum, so
  a constant factor cancels exactly. *Documented* (shape) + algebra.
- That `psmatch2`'s `epan` is this compact form on `[-1, 1]`, not the
  unit-variance form of `[R] kdensity`: *black box* (E1 `_weight` and
  `r(att)` reproduce to 1e-15 with the compact form).
- The window is closed: at `|u| = 1` the uniform kernel weighs 1 (E2: a
  control exactly `h` away receives weight 1/4 in both `kernel
  kerneltype(uniform)` and `radius caliper(h)`), and `epan` gives that
  control no weight and leaves its `_weight` missing. *Black box* (E2).
- An unknown kernel name raises `MethodIncompatibility("unknown kernel
  '<name>'")`. Names are case sensitive (the caller lower-cases). *OLD*.

### Stata's Epanechnikov for `lpoly` (`stata_epanechnikov`)

R-kd: `K(z) = 3/4 (1 - z^2/5) / sqrt(5)` for `|z| < sqrt(5)`, 0 otherwise;
it has unit variance, so its half-width is `sqrt(5) h`. *Documented.*
StatsPAI evaluates the polynomial on the closed interval `|z| <= sqrt(5)`
(at the end point it is zero up to rounding, -7e-17). *OLD* (measure zero).

## 2. Kernel and radius matching (`_fit_kernel`, method `kernel` / `radius`)

For each treated unit `i` on common support, with `p` the score:

- `k_ij = K((p_j - p_i) / h)` over all controls `j`; controls with
  `k_ij > 0` are its matches; `w_ij = k_ij / sum_j k_ij`;
- `m_i = sum_j w_ij Y_j` (`_y`), effect `Y_i - m_i`;
- ATT = mean of `Y_i - m_i` over the treated that have at least one match.

*Documented*: H ("kernel matching", `bwidth()` default 0.06, `_weight` is
"the overall weight given to the matched observation", treated `_weight = 1`).
Normalisation per treated unit and the mean over treated: *black box* (E1:
ATT and `_y` to 1e-15 / CSV precision; FX `kernel_epan_bw0.5`).

Radius matching is the uniform kernel with `h = caliper` (H: "radius
matching within the specified radius given by caliper"): each treated unit
takes the plain mean of the controls with `|p_j - p_i| <= caliper`.
*Black box*: E2 radius equals uniform kernel; FX `radius_cal0.1`.
StatsPAI takes the caliper as given (no `caliper_scale` resolution on this
path). *OLD*.

Common support (`common_support='minmax'`, H `common`): treated units whose
score lies outside the controls' `[min, max]` are dropped before matching;
controls are always on support. Shared helper `common_support_mask`, not
part of this rewrite.

Failure handling:

- A treated unit with no control of positive weight is set off support
  (`_support = 0`, `_weight` and `_y` missing) and left out of the ATT.
  *Black box* (E5: 9 of 10 treated "Off support", ATT over the remaining
  one).
- If no treated unit is matched: `DataInsufficient("match: <method>
  matching found no treated unit with a control within bandwidth <h>.",
  recovery_hint="Increase bwidth/caliper or inspect common support.",
  diagnostics={"method", "bandwidth"})`. *OLD* (Stata has no estimate to
  report here).

## 3. Local linear regression matching (method `llr`)

For treated unit `i`, regress the control outcomes on the gap
`d_j = p_j - p_i` by kernel-weighted least squares with weights
`k_j = K(d_j / h)` and take the intercept, which estimates `E[Y | p = p_i]`
among controls (FG ch. 3; R-lp eq. 2 with degree 1 and `x_0 = p_i`). As a
linear smoother in `Y_j`, with `k_j` normalised to sum one,

    D = sum k_j d_j,  S2 = sum k_j d_j^2,  V = S2 - D^2,
    w_ij = k_j (S2 - D d_j) / V,

so `sum_j w_ij = 1` and `sum_j w_ij d_j = 0`. Derivation: the closed-form
2 x 2 WLS solution. `m_i = sum_j w_ij Y_j`; ATT as in section 2. The
weights can be negative.

- Which controls enter: those with `k_j > 0` (the kernel window). The
  intercept with `normal`, `uniform`, `biweight`, `tricube` reproduces
  `psmatch2 ..., llr kerneltype(<k>)` to 1e-15 (E1b, E2; FX
  `att_by_kernel`, including the per-control `_weight` of the tricube run).
  *Black box.*
- **No slope identified.** When the weighted variance of the gaps is not
  distinguishable from zero, `V < 1e-24` or `V < 1e-10 * S2`, the treated
  unit is set off support and dropped, like a unit without controls. The
  relative test is needed because `V` is computed as a difference of two
  mean squares, so when all gaps are equal it is rounding error of order
  `1e-16 * S2`, not zero. These two thresholds are StatsPAI's own
  convention (*OLD*). `psmatch2` does something else here (E4: every
  control at one score; it keeps every treated unit on support and reports
  `_y` = the control mean, i.e. the local-constant fit). StatsPAI keeps its
  convention: a local linear estimate without a slope is not what was
  asked for, and the unit is reported as unmatched.
- If no treated unit survives: the same `DataInsufficient` as section 2.
- `kernel='epan'` with `method='llr'` emits a `UserWarning` containing
  "does not perform local linear" (see section 5); StatsPAI still computes
  the estimator above.
- Floating point: in nearly collinear local designs
  (`1e-10 < V / S2 < 1e-5`, e.g. a normal kernel with `h` of 0.001-0.02 on
  the score) the weights are ill-conditioned; any floating-point evaluation
  of the formula, the pre-rewrite one included, is off from the exact
  rational value by up to ~2e-6 relative in such rows (checked with
  exact `fractions` arithmetic on 73 such rows).

## 4. Standard errors

### Analytic (`psmatch2_se`, `se_method='psmatch2'`)

H states the default SE:

    1/N1 * Var(Y | DM=1) + Sum(w_i^2; i in DM=0) / N1^2 * Var(Y | DM=0)

"where N1 is the number of matched treated, DM=1 denotes the matched
treated, DM=0 the matched controls and w_i is the weight given to control
i", assuming independent observations, fixed weights and homoskedasticity
within arms (L). *Documented.* The reading of each term, *black box* (E1,
all five kernels; E7): `r(seatt)` equals, to 1e-15,

- `Var(Y | DM=1)`: unweighted sample variance (`n - 1`) of `Y` over treated
  with `_support == 1`; `N1` their count;
- `Var(Y | DM=0)`: unweighted sample variance over controls with
  `_weight` not missing;
- `w_i`: the control's `_weight`.

StatsPAI conventions (*OLD*): a control weight of exactly 0 still counts as
matched (it is not missing); controls' support flags are ignored; treated
rows' weights are ignored; `nan` when fewer than two matched treated or
fewer than two matched controls, or when an outcome among them is missing.
`fit()` passes a support vector in which treated units without a match are
off support.

### The SE returned by `_fit_kernel`

`sd(Y_i - m_i) / sqrt(n)` over the `n` matched treated (`ddof = 1`). It is
what `se_method='ai'` reports for these methods; `fit()` overrides it for
`'psmatch2'` and `'bootstrap'`. With fewer than two matched treated it is
`nan` and `_se_undefined` is set to "fewer than two treated units with a
control inside the bandwidth". *OLD.*

### LLR

Stata reports `seatt = .` for `llr kerneltype(tricube)` (FX, E1b) but a
number for `normal`, `uniform` and `biweight` (E1b, E2). StatsPAI refuses
`se_method='psmatch2'` for `llr` (validation, not part of this rewrite) and
defaults to the bootstrap.

## 5. `llr` with the Epanechnikov kernel in `psmatch2`

H: with `llr` and the Epanechnikov kernel the default bandwidth is the
rule-of-thumb bandwidth of `lpoly`. Black box (FX `llr_epan_reroute`,
`psmatch2_llr_epan_data.csv`; E7): the result is not local linear regression
matching. `psmatch2` smooths the control outcome on the score with `lpoly`
(degree 1, Stata's Epanechnikov kernel, the given bandwidth) fitted on the
on-support controls and evaluated at every unit's score (`_s_y`; reproduced
to 1e-10), then does one-nearest-neighbour matching on the score, with
`_y` = the matched control's smoothed value. The reported `r(seatt)` is the
analytic formula of section 4 on the **raw** outcomes (E7: raw rebuild
1.16758520164544 = `r(seatt)`; the smoothed rebuild gives 1.0255).

StatsPAI reproduces this only on request (`llr_stata_compat=True`); the
nearest-neighbour part lives in `_fit_nearest` (outside this rewrite) and
calls `psmatch2_llr_smoothed_outcome`.

Rule-of-thumb bandwidth: R-lp eq. (3), `h = C_{0,p}(K) [ sigma^2 int w_0 /
(n int {m^{(p+1)}}^2 w_0 f) ]^{1/(2p+3)}` with `C_{0,p}(K)` from FG, `w_0`
the indicator of `[min + 0.05 range, max - 0.05 range]`, `sigma^2` and
`m^{(p+1)}` from a global polynomial of order `p + 3`. *Documented.* Not
implemented: StatsPAI always uses the explicit `bwidth` (default 0.06), so
`sp.match(method='llr', llr_stata_compat=True)` reproduces `psmatch2 ...,
llr` only when `bwidth()` is given on both sides. E8 (no `bwidth()`): the
smoothed values differ from `lpoly` with its own ROT bandwidth on the
control sample (max |diff| 5.98), so how `psmatch2` forms its default is
not pinned. Open item.

## 6. The `lpoly` smoother (`lpoly_predict`, `psmatch2_llr_smoothed_outcome`)

`lpoly_predict(x, y, at, bandwidth, degree=1)`: for each `x_0` in `at`,
solve `min_b sum_j K((x_j - x_0)/h) (y_j - sum_{q=0}^{p} b_q (x_j - x_0)^q)^2`
with `K` = Stata's Epanechnikov (section 1) and return `b_0` (R-lp eq. 2;
the `1/h` in `K_h` cancels). *Documented.* Implementation conventions,
*OLD*:

- pairs with a missing `x` or `y` are dropped;
- the local design uses the points with `K > 0`; the normal equations
  `X'WX b = X'Wy` are solved directly;
- `nan` where fewer than `degree + 1` points have positive weight, or where
  the normal equations are exactly singular (`LinAlgError`); otherwise the
  solve result is returned, even when it is ill-conditioned;
- `bandwidth <= 0` raises `ValueError("lpoly bandwidth must be positive,
  got <h>")`; `bandwidth=None` raises `TypeError` (no ROT bandwidth);
- `at` must be one-dimensional (a 0-d array raises `TypeError`).

Divergence from Stata in degenerate windows (*black box*, E6): where the
window holds a single distinct `x` (or too few for the degree), `lpoly`
still reports a value, the local mean or the lower-degree fit (e.g. degree
1, `h = 0.01`, three tied points at `x_0`: Stata 2, StatsPAI `nan`), and it
reports missing where only one observation is in the window (StatsPAI
returns that observation's `y` for degree 0). The parity fixture never
reaches these points (`_s_y` agrees to 1e-10). Kept as is (changing it
would change StatsPAI's numbers); open item.

`psmatch2_llr_smoothed_outcome(outcome, pscore, treated, support,
bandwidth, degree=1)`: `lpoly_predict` fitted on the controls
(`treated == 0`) with `support` true, evaluated at every unit's score.
*Black box* (FX `_s_y`, section 5). Accepts bool or 0/1 `support` /
`treated`; all-`nan` when there is no such control.

## 7. Outputs of `_fit_kernel` (the internal contract with `fit()`)

Returns `(att, se, balance, extra_info)`:

- `balance` = `self._balance_table(X, T, pscore)`;
- `extra_info = {"kernel": <name, 'uniform' for radius>, "bwidth": h,
  "n_on_support": number of units on support after dropping unmatched
  treated, "n_matched_treated": n}`;
- sets `self._assignment = {pscore, treated, idx_t, idx_c, matches,
  weights, support, outcome, neighbors=False}` where `matches[k]` are
  positions into `idx_c` (ascending) of the controls with positive kernel
  weight for treated `idx_t[k]` and `weights[k]` their `w_ij` (empty arrays
  for unmatched or off-support treated); `fit()` builds the `_weight` /
  `_y` frame from it;
- score from `self._score(X, T)` (given column or the fitted model).

## Verification record

- OLD battery (537 cases: every kernel x bandwidths {0.005, 0.06, 0.5, 3}
  x common support on three datasets, radius x calipers, every
  `se_method`, bootstrap with seeds, probit / quadratic score, frequency
  weights, given and tied scores, boundary `|u| = 1`, no control in
  bandwidth, collinear and nearly collinear LLR designs, the `sp.psmatch2`
  front door, the `llr_stata_compat` path, and the unit functions on edge
  inputs). Same exceptions (type, message), same warning categories, same
  `_assignment` structure in every case. Numbers: bitwise or <= 8.1e-12
  relative everywhere except nine LLR cases with a normal kernel and
  `h` in 0.001-0.02 (nearly collinear local designs, section 3), where the
  ATT moves by up to 4.5e-8, the SE by up to 1.0e-8 and single weights by up
  to 3.0e-6 relative. The kernel functions, `psmatch2_se` and the three
  `lpoly` functions are bitwise identical on their battery cases.
- Stata (E1-E7): ATT and SE agree with `psmatch2` to <= 3.3e-15 relative.
  Committed as `tests/reference_parity/_fixtures/psmatch2_cleanroom_stata.json`
  (generator `_generate_psmatch2_cleanroom.do`), tested by
  `tests/reference_parity/test_cleanroom_psmatch2.py`.
