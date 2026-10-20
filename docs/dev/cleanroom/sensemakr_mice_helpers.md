# Clean-room specification: robustness value and Barnard-Rubin df

Audit rows M7 and M8 of `docs/dev/2026-10-20-license-provenance-audit.md`:
`statspai.diagnostics.sensemakr._robustness_value` (said to be a "port of
`sensemakr::robustness_value.numeric`", R sensemakr 0.1.6, GPL-3) and
`statspai.imputation.mice._barnard_rubin_df` (said to be "written as R
`mice:::barnard.rubin` writes it", R mice 3.19.0, GPL >= 2). Both are single
published formulas. The spec writer saw a few lines of both old bodies while
locating them, so the two functions were re-implemented by a separate agent
from this specification only. The benchmark-bound helper
`_sensemakr_bound_scale` (whose body nobody on this line read) is covered in
the last section.

## Sources (and only these)

- Cinelli and Hazlett (2020) `[@cinelli2020making]`, author version from
  carloscinelli.com: equations (12)-(14), (18)-(20), footnote 11 and
  Supplement B, equations (73)-(84).
- Barnard and Rubin (1999) `[@barnard1999small]`; van Buuren, *Flexible
  Imputation of Missing Data*, 2nd ed., Section 2.3.6, equations
  (2.24), (2.30)-(2.32) (online edition, read 2026-10-20).
- Help pages (`tools::Rd2txt`) of `sensemakr::robustness_value` and
  `mice::pool`.
- Black-box outputs of the references on grids chosen by us:
  `tests/reference_parity/_fixtures/_generate_sensemakr_rv_blackbox_R.R` and
  `_generate_mice_barnard_rubin_blackbox_R.R` (the CSVs beside them).

## Robustness value, `_robustness_value(t_statistic, dof, *, q=1, alpha=0.05, invert=False)`

Notation (paper eq. 14 and the text after it): the partial Cohen's f of the
treatment with the outcome is `f = |t| / sqrt(dof)`, with `t` the treatment
t-value and `dof` the residual degrees of freedom of the regression;
`f_q = q f` (eq. 18).

- `alpha = 1` (no sampling uncertainty): `RV_q = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2`
  (eq. 18).
- `alpha < 1`: `f*_{alpha} = |t*_{alpha, dof-1}| / sqrt(dof - 1)` where
  `t*_{alpha, dof-1}` is the two-sided critical value of a t distribution with
  `dof - 1` degrees of freedom (`scipy.stats.t.ppf(1 - alpha/2, dof - 1)`);
  `f_{q,alpha} = f_q - f*_{alpha}` (eq. 19). If `f_{q,alpha} < 0` the value is
  0. Otherwise `RV_{q,alpha} = (sqrt(f_{q,alpha}^4 + 4 f_{q,alpha}^2) - f_{q,alpha}^2) / 2`
  (eq. 20), except that when `RV_{q,alpha} <= 1 - 1/f_q^2` it is replaced by
  `(f_q^2 - f*_{alpha}^2) / (1 + f_q^2)` (text after eq. 20).
- `invert=True`: the help page describes it as computing "IRV instead of RV
  (i.e. is the estimate insignificant?)": the strength of confounding (equal
  partial R2 with treatment and outcome) that would be needed to make a
  non-significant estimate significant. Derive its closed form from the same
  bias algebra (eqs. 12-14) and check it against the `invert = TRUE` rows of
  the fixture.
- Return a Python float. Inputs are scalars.

Acceptance: every row of `sensemakr_rv_blackbox_R.csv` to `rtol = 1e-10`
(the reference prints 15 significant digits; the formula is closed form), and
the existing tests in `tests/test_sensemakr.py`, Track A `22_sensemakr`
(Stata `sensemakr` golden) and every other test that reaches the function.

## Barnard-Rubin degrees of freedom, `_barnard_rubin_df(m, b, t, dfcom)`

`m` imputations, `b` between-imputation variance, `t` total variance
(arrays, elementwise), `dfcom` complete-data degrees of freedom (scalar,
possibly `inf`).

- `lambda = (1 + 1/m) b / t` (FIMD 2.24).
- `nu_old = (m - 1) / lambda^2` (Rubin 1987; FIMD 2.30).
- `nu_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lambda)` (FIMD 2.31).
- `nu = nu_old nu_obs / (nu_old + nu_obs)` (FIMD 2.32; Barnard and Rubin
  1999). Write it so that `b = 0` (`lambda = 0`, `nu_old = inf`) returns
  `nu_obs` rather than `inf / inf`, e.g. as the reciprocal of the sum of
  reciprocals.
- `dfcom = inf`: `nu_obs = inf`, so `nu = nu_old` (Rubin's large-sample df,
  `inf` when `b = 0`).
- Only `lambda` in `[0, 1]` occurs (`t >= (1 + 1/m) b`); no behaviour outside
  that range is specified.

Acceptance: `mice_barnard_rubin_blackbox_R.csv` to `rtol = 1e-12`, and the
existing tests of `sp.mice` / `sp.mi_estimate` pooling (R `mice::pool` and
Stata `mi estimate` fixtures).

## Benchmark bounds, `_sensemakr_bound_scale(r2dxj_x, r2yxj_dx, *, kd=1, ky=1)`

Supplement B, "comparisons conditioning on D" (eqs. 73-84): with
`R2_D = r2dxj_x` (partial R2 of the benchmark covariate with the treatment)
and `R2_Y = r2yxj_dx` (with the outcome, given treatment and the other
covariates), and Cohen's `f^2 = R2 / (1 - R2)` (footnote 11):

- `r2dz_x = kd * R2_D / (1 - R2_D)` (eq. 22, first part).
- `f_KD = sqrt(kd) R_D / sqrt(1 - kd R2_D)` (eq. 78), `f_D = R_D / sqrt(1 - R2_D)`.
- `eta = (sqrt(ky) + f_KD f_D) / sqrt(1 - f_KD^2 f_D^2)` and
  `r2yz_dx = eta^2 R2_Y / (1 - R2_Y)` (eq. 84).

Written by the spec writer (who did not see the old body) and checked against
the existing NSW benchmark test and Track A `22_sensemakr`.
