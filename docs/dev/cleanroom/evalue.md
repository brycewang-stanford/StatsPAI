# Clean-room spec: E-value threshold and exact risk-difference E-value

Scope: `diagnostics/evalue.py::_threshold` and `::evalue_rd` (audit row H13,
`docs/dev/2026-10-20-license-provenance-audit.md`). The rest of the module
(`evalue`, `bias_factor`, `evalue_from_result`, the CI and conversion helpers)
was not flagged and is unchanged.

Reference used as a black box: R `EValue` 4.1.4 (GPL-2), function
`evalues.RD` (help page and outputs only).

## Sources read / not read

Read:

- Ding & VanderWeele (2016), *Sensitivity analysis without assumptions*,
  arXiv:1507.03984v1 (the preprint of `ding2016sensitivity`): main text,
  Appendix 2 and eAppendix 5 (Propositions A.8, A.11, Section 5.2 on finite-
  sample inference, equation (A.5) and the grid-search remark that follows it).
- VanderWeele & Ding (2017) (`vanderweele2017sensitivity`) for the E-value
  formula `E = RR + sqrt(RR (RR - 1))`.
- `?evalues.RD` (EValue 4.1.4 help page).
- Outputs of `EValue::evalues.RD` on the cases listed below.
- The repository's tests and fixtures for these functions
  (`tests/test_evalue.py`, `tests/reference_parity/test_inference_sens_R_parity.py`
  and its fixture, Track A `23_evalue`, original-data `11_nhefs_evalue`),
  the registry entry and agent cards.
- The redacted view of `diagnostics/evalue.py` (bodies of `_threshold` and
  `evalue_rd` removed) and the outputs of the previous implementation on 59
  input configurations, run as a black box before it was deleted.

Not read: any EValue R source (no function bodies printed, no CRAN/GitHub
source), and the previous bodies of `_threshold` / `evalue_rd` or their git
history.

## `_threshold(x, true)`: the E-value on the risk-ratio scale

Ding & VanderWeele (2016) show that confounding with risk ratios
`RR_EU`, `RR_UD` can move a risk ratio by at most the bounding factor
`B = RR_EU RR_UD / (RR_EU + RR_UD - 1)`, so the true risk ratio satisfies
`RR_obs / B <= RR_true` (and the reverse for a protective association). With
`RR_EU = RR_UD = e`, `B(e, e) = e^2 / (2e - 1)`, whose inverse on `[1, inf)` is

    e(B) = B + sqrt(B (B - 1)).

To move an observed ratio `x` to a reference value `true` the confounding must
supply a bias factor `r = x / true` when `x >= true` and `r = true / x` when
`x < true`; the E-value is `e(r)` (for `true = 1` this is the VanderWeele-Ding
formula).

Edge cases (kept from the previous behaviour, checked against it):

| input | output |
| --- | --- |
| `x` is `None` or NaN | `None` |
| `x < 0` | `MethodIncompatibility` |
| `true` NaN | NaN |
| `true = 0`, `x > 0` | `inf` (no finite confounding reaches a ratio of 0) |
| `true = 0`, `x = 0` | NaN |
| `x = 0`, `true > 0` | `inf` |
| `x = true` | 1 |

## `evalue_rd(n11, n10, n01, n00, true=0, alpha=0.05, grid=1e-4)`

Data: exposed `N1 = n11 + n10`, unexposed `N0 = n01 + n00`, `N = N1 + N0`;
`f = N1 / N`, `p1 = n11 / N1`, `p0 = n01 / N0`; observed `RD = p1 - p0`.
Binomial variances `s1^2 = p1 (1 - p1) / N1`, `s0^2 = p0 (1 - p0) / N0`,
`sf^2 = f (1 - f) / N`.

Target: the population-standardised causal risk difference. Proposition A.8:

    RD_true >= L(B) = (p1 - p0 B) (f + (1 - f) / B).

**Point E-value.** Setting `L(B) = true` gives the quadratic of
Proposition A.11, `(p0 f) B^2 + (p0 (1 - f) + true - p1 f) B - p1 (1 - f) = 0`,
whose positive root is

    B* = [ sqrt(q^2 + 4 p1 p0 f (1 - f)) - q ] / (2 p0 f),   q = true + p0 (1 - f) - p1 f.

The point E-value is `e(B*)`; `bias_factor` in the output is `B*`. For
`true = 0`, `B* = p1 / p0` (the observed risk ratio).

**Confidence-limit E-value.** eAppendix 5.2 gives the delta-method standard
error of `L(B)`:

    se(B)^2 = (s1^2 + s0^2 B^2) (f + (1 - f)/B)^2 + (p1 - p0 B)^2 (1 - 1/B)^2 sf^2,

and the lower confidence limit `LCL(B) = L(B) - z se(B)`, `z = Phi^{-1}(1 - alpha/2)`.
Equation (A.5) asks for the smallest `B` with `LCL(B) <= true`; the paper
suggests a grid search over `[1, B*]`. The grid is `B_k = 1 + k * grid`,
`k = 0, 1, ...` while `B_k <= B*` (with a 1e-10 relative fuzz on the count),
and the result is `e(B_k)` at the first `k` with `LCL(B_k) <= true`. When
`LCL(1) <= true` the interval already reaches `true` and the E-value is
`e(1) = 1`.

Black-box conventions established from `evalues.RD` outputs (grid 0.1, 0.05,
0.01 on two tables, `true` 0 and 0.001): the chosen point is the *first* grid
point whose lower limit is at or below `true` (not the point nearest to it),
and the grid starts at 1 with step `grid`. Building the grid as
`1 + k * grid` rather than by repeated addition keeps the grid points within a
few ulps of R's.

**Departure from the reference (documented).** When no grid point up to `B*`
satisfies the condition (a coarse grid with a very precise table), R returns
`NA` for the lower E-value. Here `B*` itself is appended as the last
candidate: `LCL(B*) = true - z se(B*) <= true` always holds, and the first
qualifying candidate still overshoots the exact root by less than one grid
step, the same guarantee as every other grid answer. The previous
implementation returned `1.0` in this case, which wrongly reports that the
interval already contains `true`.

**Errors.**

| condition | error |
| --- | --- |
| a cell is not a finite number (bools rejected), or negative | `MethodIncompatibility` |
| `alpha` not in (0, 1); `grid` not positive and finite; `true` not finite | `MethodIncompatibility` |
| `N1 = 0` or `N0 = 0` | `DataInsufficient` |
| `RD < 0` | `MethodIncompatibility` (relabel the exposure) |
| `true >= RD` | `MethodIncompatibility` |
| `p0 = 0` (no unexposed cases: `B*` is infinite) | `MethodIncompatibility` (the previous code raised a bare `ValueError` from NumPy) |

The grid is scanned in blocks so a large `B*` with a fine grid does not
allocate one huge array.

## Checks

- `tests/reference_parity/test_inference_sens_R_parity.py::test_evalue_rd_matches_EValue`
  (rtol 1e-12 against `evalues.RD`).
- `tests/test_evalue.py` (point, CI, crossing, non-null `true`, validation).
- Track A `23_evalue` and original-data `11_nhefs_evalue` reproduce their
  committed results.
- New: the R `NA` case and the `p0 = 0` case, and the previous-implementation
  snapshot (59 configurations) agree to 1e-12 except the two documented rows.
