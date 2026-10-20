# Clean-room specification: `pstest` balance table

Audit row M3 of `docs/dev/2026-10-20-license-provenance-audit.md`:
`matching/_pstest.py` said it followed `pstest.ado` v4.2.2 (Leuven and
Sianesi; no licence stated), and commit `b20caa6e` cited ado line numbers.
The module was replaced whole without its old bodies being read.

## Sources (and only these)

- The `pstest` help file (rendered with Stata's `translate`): the list of
  statistics, "t-tests ... based on a regression of the variable on a
  treatment indicator ... weighted using the matching weight variable ...
  and based on the on-support sample", the standardised % bias "formulae
  from Rosenbaum and Rubin (1985)", the pseudo-R2 and LR test of a probit
  "on raw samples, matched samples", Rubin's B and R on "the linear index of
  the propensity score".
- Rosenbaum and Rubin (1985) `[@rosenbaum1985constructing]`: the
  standardised difference with the pre-matching variances in the
  denominator. Rubin (2001) `[@rubin2001using]`: B, R and the rule of thumb.
- Stata's documented importance-weight moments (`summarize [iw=]`: the
  weight total is the sample size).
- Black-box outputs: the committed fixture (`pstest_stata.json`), three
  extra Stata 18 runs (a constant covariate; two reweightings that push the
  matched controls' index up), and the previous function on six matched
  frames (nearest neighbour k = 1 and 3, kernel, a binary and a constant
  covariate, a covariate subset).

## Specification

Matched sample: rows with `_support == 1` and a positive `_weight`.
Unmatched: every row. For each covariate: unmatched means / variances
(divisor n - 1); matched means / variances with the weights as importance
weights (divisor sum(w) - 1); `%bias = 100 (m1 - m0) / sqrt((v1u + v0u)/2)`
with the unmatched variances in both rows; `%reduct = -100 (|b_m| - |b_u|) /
|b_u|`; t and p of the treatment coefficient in `x ~ 1 + d` (unweighted;
importance-weighted on the matched sample, df = sum(w) - 2); variance ratios
treated / control. A constant covariate gives NaN for every derived field;
its means are kept.

Summary per sample: probit of `d` on `[1, covariates]` (importance-weighted
on the matched sample), constant / collinear columns dropped left to right;
pseudo-R2 `1 - ll / ll0`, LR `2 (ll - ll0)` with df = number of covariates
kept, p from chi-square; mean and median of |%bias| over defined rows;
`B = 100 |mean_t(xb) - mean_c(xb)| / sqrt((var_t + var_c)/2)`,
`R = var_t / var_c` (weighted on the matched sample); balanced when
`B < 25` and `0.5 <= R <= 2`.

## Result

Per-covariate table identical to the previous function (<= 1.1e-15) on all
six frames and to the Stata fixture at its tolerance. Rubin's B and R now
agree with Stata to ~1e-14 on the fixture (previously 1e-9 / 1e-10).
**Fixed**: with a constant or collinear covariate, the LR chi-square
p-value used one degree of freedom per covariate listed; Stata (and now
StatsPAI) uses the covariates the probit keeps (e.g. 1.497e-14 against the
previous 7.82e-14 on a 400-row sample).
