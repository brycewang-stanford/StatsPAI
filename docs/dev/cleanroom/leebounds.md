# Clean-room specification: Lee (2009) bounds as Stata `leebounds` computes them

Audit row M2 of `docs/dev/2026-10-20-license-provenance-audit.md`:
`bounds/lee_manski.py` (`_stata_local_macro`, `_stata_pctile`,
`_leebounds_trimmed`, `_lee_analytic_se`), `bounds/_lee_tight.py` and
`bounds/_lee_weighted.py` described themselves as following the internals of
Stata `leebounds` (Tauchmann, SSC v1.5; no licence stated). Those functions
and both modules were replaced without their old bodies being read; the
helpers `_sample_quantile` and `_lee_trimmed`, which dispatch to them, were
rewritten too. Every rule below is either documented (source named) or
observed on `leebounds` run as a black box (experiment named, numbers given).

## Sources (and only these)

- Lee (2009) `[@lee2009training]`, NBER working-paper version pp. 17-20:
  estimator (6) (trimming share `p = (P(S|D=1) - P(S|D=0)) / P(S|D=1)`,
  trimmed means at the `p` / `1 - p` quantile of the more-retained arm),
  Proposition 3 variance (7) and the note that it is estimated by replacing
  each quantity by its sample analogue; footnote 28 (with discrete outcomes
  drop `[p n]` observations; with design weights drop until the cumulated
  weight reaches `p` times the total). Imbens and Manski (2004)
  `[@imbens2004confidence]` for the effect interval (unchanged code).
- Tauchmann's description of `leebounds` (SFB 823 Discussion Paper 35/2013,
  the PDF itself), sections 2.2-2.4: eqs. (1)-(7); with ties the command
  "excludes the q N_T (rounded down) smallest - respectively largest -
  values"; tightened bounds average cell bounds with weights
  `sum 1(J, S=1, T=0) / sum 1(S=1, T=0)` (`T = 0` the less-retained arm);
  `e(cellsel)` flags a selection direction that differs across cells.
- The `leebounds` help file (rendered with `translate`): syntax, `pweight`
  default, `vce(analytic)` default, saved results.
- Stata manuals: [D] pctile, Methods and formulas (weighted and unweighted
  `_pctile`: first `x(i)` with `W(i) > P`, average when `W(i-1) = P`);
  [P] scalar ("when converting to printable form, Stata produces
  representations containing up to 17 digits and, if necessary, uses
  scientific notation"); [U] 18.3 (`local m = exp` stores the evaluated
  result as text).
- Black-box runs (Stata 18 MP, `leebounds` 1.5 in a private ado
  directory), listed under each rule; my own small Stata programs for the
  macro text and `_pctile`; and the committed fixtures
  `leebounds_stata.json`, `leebounds_tight_Stata.json`,
  `leebounds_weights_stata.json`, `leebounds_missing_stata.json` (their
  tests and do-files were read).
- Outputs of the previous StatsPAI functions on a battery (below), recorded
  before replacement; used only to measure the change.

Not read: `leebounds.ado` or any ado / Mata source, the old function bodies,
their git history. The old `lee_bounds` docstring paragraphs that described
`leebounds` (Notes, `weights`, `trimming`) were read in order to replace
them; nothing from them is carried over unverified (each statement now rests
on an experiment below).

## Experiment data

Scratch generators (not committed) built: `u` 42 samples (n = 25, 160, 900;
outcomes rounded to 3 or 1 decimals, normal doubles, Poisson, binary,
float32-valued; either arm more retained; three designs with `(1 - p) n`
whole), each run without and with `[pw=]`, under the default
`set type float` and under `set type double`; `t` 19 samples with `tight(x)`
and `tight(x z)` (cells of opposite selection direction in 5 of them); `t2` 4
samples with no tied cell threshold; tiny designs (n = 19) varying one
outcome at a time; weight designs where only some groups carry general
weights; outcomes scaled by 1e-9 to 3e-6. The committed fixture
`leebounds_cleanroom_stata.json` (generator
`_generate_leebounds_cleanroom.do`, `set type double`) keeps the cases that
pin the rules below; `test_cleanroom_leebounds.py` checks them.

## Specification

### Numbers held in macros (black box, own Stata program)

`local m = v` stores at most 17 characters, sign not counted: fixed notation
with `k_f` significant digits, `k_f = 16` for `|v| >= 1` and
`min(16, 17 + e)` for `|v| < 1` (`e = floor(log10 |v|)`, leading `0` of
`0.x` dropped, trailing zeros removed), used when `e <= 15` and `k_f` is at
least the digits of scientific notation; otherwise scientific notation
`d.ddd...e+XX` filling 17 characters (11 decimals for a two-digit exponent,
trailing zeros kept). Rounding is half away from zero on the exact binary
value. Reading the text back is correctly rounded. Evidence: 2,067 doubles
(`|v|` from 1e-300 to 1e19, float32 values, exact binary half-way cases), all
reproduced; 1,054 read back, all equal to Python's `float(text)`.
`_stata_local_macro`.

### `_pctile` (documented; tolerance black box)

Definition as in [D] pctile. Stata treats `P = n p / 100` as whole when it is
within about `1e-15 P` of an integer: 16 `(n, k)` pairs with `p` swept over
+-60 ulps around `100 k / n`; the averaging window spans about 4-9 ulps of
`p` on each side, i.e. 0.8e-15 to 1.1e-15 relative in `P`. Implemented with
`1e-15` (`_stata_pctile`). Lee's rule (`trimming='quantile'` / `'exact'`)
uses `1e-12`, because `p` is a difference of retention rates whose rounding
can reach 1e-14 relative, and a whole `n p` must not move to the next order
statistic.

### Point estimates (black box)

- Trimmed arm: the one with the higher retention rate (`e(trimmed)`), share
  `q = (p_hi - p_lo) / p_hi` (documented). `e(trim)` equals `q` written
  through the macro text in 39 of 41 samples (the other two differ in the
  last digit). Equal retention: `leebounds` stops with "selection proportion
  exactly equal for both groups, no bounds computed" (r(498)).
- Thresholds: `_pctile` of the selected trimmed-arm outcomes at `100 q`
  (keep the upper tail) and `100 (1 - q)` (keep the lower tail), the
  percentage and the threshold both held as macro text. Fits all 41
  unweighted samples to <= 1.3e-15 (double type) and the eight committed
  samples to <= 1.6e-15; without the threshold rounding 8 of the 41 miss
  (at 1e-12), without the `_pctile` tolerance 2 miss.
  With outcomes of order 1e-7 the threshold is held with 12 significant
  digits (scientific notation): the rule reproduces Stata to <= 8e-11 on six
  such samples (Stata posts `e(b)` with 12 digits there), while 16-digit rounding misses a bound by up to 2.7%
  (fixture samples 1-2).
- Tie rule: when some outcome equals the held threshold, the kept mass is
  exactly `(1 - q) n`: outcomes strictly beyond the threshold in full, the
  tied ones sharing the rest. Tauchmann's "exclude floor(q N_T)" whole
  observations does not fit (at most 7 of the 41 samples; the shared mass
  fits all 41). Otherwise every outcome at or beyond the held threshold is
  kept, which drops the quantile observation when rounding moved the
  threshold past it.
- Bounds: lower = lower-tail mean minus the other arm's mean when the
  treated arm is trimmed; when the control arm is trimmed, lower = treated
  mean minus the control's upper-tail mean (and symmetrically).

### Analytic variance (documented formula; sample analogues black box)

`Var = [V_trim + Var(Y | S=1, other arm) / E(S (1 - T))] / n`, `n` all
units, the other arm's variance with divisor count - 1, and

    V_trim = s2 n / W + (y_q - mu)^2 q / (E(ST) (1 - q))
             + (y_q - mu)^2 ((1 - E(S|T=0)) - q (1 - E(T))) / (E(T) E(S|T=0) (1 - E(T)))

(Lee's (7); the last term is his third term rewritten with the trimmed arm
`T`). `W` is the kept mass, `mu` the tail mean, `y_q` the held threshold.
Black box: without a tie `s2` is the kept outcomes' variance with divisor
`W - 1` (so `s2 n / W` uses the kept count, not `(1 - q) n_T`); at a tie it
is the share-weighted variance with divisor `W`. 41 samples, double type,
both bounds: <= 3.1e-14. Tiny designs show that only kept outcomes enter.

Default `set type float`: the tie branch's second moment averages squares
of the strictly-inside outcomes held in single precision (fits all 41
default-type samples to <= 3.1e-14; tiny designs show the threshold and the
tied outcomes are exact). This moves variances by up to 1e-7 relative on
outcomes that are not integers. The unweighted implementation keeps double
precision, as the previous one did (the committed fixtures are within
3.3e-10 / 2.0e-9 / 9.0e-10 in SE); modelling it is a possible follow-up.
`e(b)`, `e(V)` and `e(trim)` pass through the macro text (12 significant
digits for 1e-8-sized numbers; fixture samples 1-2).

`trimming='exact'` and `'quantile'` use the same formula on their own kept
set: `'quantile'` with divisor `W - 1` and `y_q` the quantile (matches
`leebounds` where its rounded threshold stays below the quantile
observation, `test_teffects_R_parity.py`); `'exact'` with divisor `W` when
the tied observations carry a fractional share and `W - 1` otherwise (the
tie branch with unrounded thresholds, same file). With equal retention
(`q = 0`, no Stata reference) the formula is evaluated at `q = 0`.

### Sampling weights (`[pw=]`, black box)

- The weights are held in single precision; retention rates (hence `q`) use
  them as first held; everything else uses them rescaled to mean one over
  the estimation sample and held again in single precision. Bounds: 46
  samples (42 + the four committed) <= 7.2e-15; other storage choices miss
  by 1e-9 to 6e-8. Under `set type double` the weights are plain
  `w / mean(w)` (42 samples, <= 3.6e-15).
- Weighted `_pctile` ([D] pctile), the held percentage and threshold, the
  tie share proportional to the tied weights, `mu` and `E(...)` weighted.
- Variance: as above with weights; `n` the number of units; the other arm's
  variance with importance weights (divisor `sum(w) - 1`) over its weighted
  share; the tie branch's second moment as above; in the last term `q` is
  computed from the rescaled weights (a different rounding of the same
  `q`). 54 default-type samples <= 3.8e-14; double type <= 4.5e-14.
- When the tail keeping the upper outcomes has no tie, `leebounds` uses the
  unweighted variance of the other arm in that bound's variance (4 double-
  type samples differ by 0.1-7% and are reproduced to 4.5e-14 with it).
  StatsPAI weights it for both bounds (documented departure; the existing
  weights test rebuilds Stata's number).

### Tightened bounds (`tight()`, black box; weights from Tauchmann)

- Cells: combinations of the covariates in sorted order; the overall
  trimmed arm is trimmed in every cell, by the cell's own share
  `q_c = (p_T,c - p_O,c) / p_T,c`. A negative `q_c` is used as is: the
  keep mass `(1 - q_c) n_c` exceeds the cell, the extreme observation is
  the "tied" one and carries weight above one (nine cases with such cells,
  bounds <= 4.4e-15). `e(cellsel)` is `hetero` when some `q_c < 0`.
- Cell weights: the cell's share of the selected units of the other arm
  (documented); bounds = weighted average of cell tail means minus the
  other arm's overall mean. 38 cases <= 7.5e-14 relative (double type).
- Variance: `sum_c w_c^2 V_c + sum_c w_c (m_c - m)^2 / K + V_other`, with
  `V_c` the trimmed-arm part above computed within the cell (cell `n`,
  shares and `q_c`), `V_other` the other arm's term on the whole sample, and
  `K` the number of selected trimmed-arm units in the last cell (in cell
  order) whose threshold was tied for either tail, or overall when no cell
  had a tie. 38 double-type cases <= 2.1e-14; the four `t2` samples (no tie)
  show `K` = overall count, <= 4.8e-15 (fixture samples 3-4).
- Errors: a cell without both arms: "cells without variation in treatment";
  a cell without a selected outcome in one arm: "cells without information
  on y for one group" (both r(2000)). StatsPAI raises
  `MethodIncompatibility`; with `trimming='quantile'` / `'exact'` a cell
  with `q_c < 0` is left untrimmed (no Stata counterpart), and analytic
  variances are offered for `trimming='leebounds'` only.

## Result

- Battery (32 data sets: the committed fixtures, 12 synthetic designs,
  equal retention, heterogeneous cells; every trimming x analytic /
  bootstrap x covariates x weights; plus the private helpers): all 422
  `sp.lee_bounds` results equal the previous ones to <= 7.7e-14 relative
  (bounds, SEs, intervals, trimming share); same exceptions.
- Committed fixtures at their registered tolerances (bounds 1e-12 / 1e-14,
  SEs 1e-8); new fixture `leebounds_cleanroom_stata.json` at 1e-10.
- Changes from the previous code: for thresholds below 1e-5 in absolute
  value the macro text is now Stata's scientific notation (12 digits), so
  `trimming='leebounds'` on such outcomes now matches `leebounds` (it missed
  a bound by up to 2.7%). `tightened_lee_bounds` returns the divisor as
  `between_cell_divisor` (`model_info['between_cell_divisor']`, formerly
  `leebounds_ntall`) and refuses analytic variances for trimmings other than
  `'leebounds'` (unreachable from `sp.lee_bounds`, which already refused).
