# Clean-room spec: mediation sensitivity analysis (`sp.mediate_sensitivity`)

Scope: `src/statspai/mediation/sensitivity.py` (audit row H22,
`docs/dev/2026-10-20-license-provenance-audit.md`): `mediate_sensitivity`
and its fixed-correlation SUR helper were rewritten; the result class
`MediateSensitivityResult` (fields, `summary`, `plot`, `__repr__`) is
presentation code that was not flagged and is kept unchanged.

Reference used as a black box: R `mediation` 4.5.1 (GPL >= 2), function
`medsens` on `mediate` objects built from two `lm` fits (help page and
outputs only).

## Sources read / not read

Read:

- Imai, Keele & Yamamoto (2010), *Identification, inference and sensitivity
  analysis for causal mediation effects*, Statistical Science 25(1)
  (`imai2010identification`), arXiv:1011.1079: Sections 3.4 (the LSEM,
  equations 11-13), 5.1 (Theorem 4 and the remarks after it, including the
  recommendation to use the iterative feasible GLS of the seemingly
  unrelated regression), 5.2 (the R^2 parametrisation).
- Imai, Keele & Tingley (2010), *A general approach to causal mediation
  analysis*, Psychological Methods (`imai2010general`), the section on
  sensitivity analysis.
- `?medsens` (mediation 4.5.1 help page, via `tools::Rd2txt`).
- Outputs of `medsens` in `tests/reference_parity/_fixtures/misc_sens_R.json`
  (existing) and `medsens_R.json` (new, generator
  `_generate_medsens_R.R`).
- The repository's tests and generators for this function
  (`tests/test_mediation_sensitivity.py`,
  `tests/reference_parity/test_misc_sens_R_parity.py`,
  `_generate_misc_sens_R.R`, `_generate_misc_sens_data.py`). The header of
  `_generate_misc_sens_R.R` states the conventions it pins (n - 1 divisor,
  squared-change stopping rule, `solve(X'Omega^-1 X)`); each was re-derived
  from the fixture numbers below rather than taken from that text.
- The redacted view of the old module (bodies and docstrings of
  `_sur_fixed_rho` and `mediate_sensitivity` removed; the result class
  visible), and the old implementation's outputs on 18 configurations, run as
  a black box before it was deleted.

Not read: any `mediation` R source; the old function bodies or docstrings;
the old file's git history.

## Model, estimand, estimator

LSEM (IKY 2010, eqs. 12-13) with covariates `X`:

    M = a0 + a_T T + X'a_X + e2
    Y = b0 + b_T T + b_M M + X'b_X + e3,      rho = Corr(e2, e3)

ACME (no treatment-mediator interaction, so delta(0) = delta(1)) is
`a_T b_M` per unit of `T` (for a 0/1 treatment, the control-to-treatment
effect). For a fixed `rho in (-1, 1)` the system is a two-equation SUR
whose error covariance is
`Sigma(rho) = [[s2^2, rho s2 s3], [rho s2 s3, s3^2]] (x) I_n`.

Iterated FGLS for fixed `rho`:

1. start: equation-by-equation OLS;
2. `s2, s3`: standard deviations of the current residuals, divisor `n - 1`
   (the residuals have mean zero because both equations have an intercept
   and the GLS normal equations for the two intercepts force both residual
   sums to zero);
3. GLS: `coef = (X'Omega^{-1}X)^{-1} X'Omega^{-1}[M; Y]` with
   `Omega = Sigma(rho)` from step 2;
4. stop when the sum of squared changes of the stacked coefficient vector is
   below `eps`; with `eps=None` (StatsPAI's default) below
   `(1e-13)^2 (1 + |coef|^2)`, i.e. the fixed point; `max_iter` caps the steps
   (`ConvergenceWarning` naming the `rho` values that reach it).

Variance: `V = (X'Omega^{-1}X)^{-1}` from the last GLS step (the `Omega`
that produced the reported coefficients); first-order delta method
`var(a_T b_M) = b_M^2 V_aa + a_T^2 V_bb + 2 a_T b_M V_ab` (the cross-equation
covariance is not zero under SUR). Interval `ACME +/- z_{(1+conf)/2} se`.

Closed forms (IKY 2010, Theorem 4 and its third remark):

* `acme_at_zero`: OLS `a_T b_M` (at `rho = 0` the SUR is equation-by-equation
  OLS; the FGLS value at `rho = 0` equals it, tested).
* `rho_at_zero = Corr(e2_hat, e1_hat)`, `e1` the residual of the total-effect
  regression `Y ~ T + X`; ACME(rho) = 0 exactly there (tested by evaluating
  the FGLS at that `rho`). `None` only if the correlation is undefined.
* `rho_grid_zero`: the grid value with the smallest `|ACME|` (medsens's
  `err.cr.d` on the same grid).
* `r2_mediator`, `r2_outcome`: R^2 of the two OLS fits.
* Section 5.2: `rho^2 = R2*_M R2*_Y = R~2_M R~2_Y / ((1 - R2_M)(1 - R2_Y))`.
  `r2star_threshold = rho_at_zero^2`, `r2tilde_threshold =
  rho_at_zero^2 (1 - R2_M)(1 - R2_Y)` (exact crossing; medsens reports the
  same products at its grid crossing `err.cr.d`, which the tests reconstruct).
* `rho_grid = numpy.linspace(low, high, n_grid)`.

## Conventions established from the reference's outputs

On `misc_sens_med.csv` (existing fixture, `eps = sqrt(.Machine$double.eps)`)
an independent prototype was run with each candidate convention:

| choice | candidates | max rel. gap to medsens `d0` / CI |
| --- | --- | --- |
| stopping rule | sum of squared coefficient changes / max squared change / squared ACME change | 1e-13 / 1.6e-4 / 3.5e-4 |
| residual SD divisor | n / n - 1 | CI 0.19 / 6e-13 |
| `Omega` for the variance | last GLS step / recomputed from final residuals | 6e-13 / 7e-4 |

and the fixed point (`eps = 1e-26`) agreed to 2e-13. The R^2 values agree to
2e-16. `R2star.prod = rho^2` and `R2tilde.prod = rho^2 (1-R2_M)(1-R2_Y)` over
the whole grid, the crossing at the closed-form `rho`, the no-covariate case
and a continuous treatment (`control.value = 0, treat.value = 1`) are checked
against `medsens_R.json` (`test_medsens_R_parity.py`).

## Edge cases

| input | behaviour |
| --- | --- |
| `conf_level` outside (0, 1) | `MethodIncompatibility` |
| `rho_range` not `-1 < low <= high < 1` | `MethodIncompatibility` |
| `n_grid < 1` or not an integer | `MethodIncompatibility` (**new**; was a numpy `ValueError`) |
| `eps <= 0` | `MethodIncompatibility` (**new**) |
| missing column | `ColumnNotFound` (**new**; was a `KeyError`) |
| collinear / constant covariate | `MethodIncompatibility` "not full rank" (**new**; was `LinAlgError`) |
| too few rows | `MethodIncompatibility` |
| NA in a used column | row dropped |
| FGLS reaches `max_iter` | one `ConvergenceWarning` listing the `rho` values (was one `RuntimeWarning` per `rho`) |
| binary mediator / outcome | treated as linear (as before; medsens's probit branches are not implemented) |

## Comparison with the previous implementation

18 configurations (the fixture data with and without covariates, default and
medsens stopping rules, `eps = 1e-3`, `max_iter = 2`, `rho_range` near +/-1,
a single `rho`, NA rows, continuous treatment, binary mediator, six error
cases). Every field of every successful configuration agrees to <= 1.1e-11
relative (the largest gaps are interval limits close to zero); the iterates
are the same, including the non-converged `max_iter = 2` case. Error types
changed as listed above; numbers did not change.
