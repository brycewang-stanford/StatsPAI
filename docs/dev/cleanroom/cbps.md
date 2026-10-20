# Clean-room spec: covariate-balancing propensity score (`sp.cbps`)

Scope: `src/statspai/matching/cbps.py` in full (audit row H18,
`docs/dev/2026-10-20-license-provenance-audit.md`): the public `sp.cbps` and
every helper of the module. The whole module was treated as contaminated and
rewritten.

Reference used as a black box: R `CBPS` 0.24 (GPL >= 2), function `CBPS`
(help page and outputs only).

## Sources read / not read

Read:

- Imai & Ratkovic (2014), *Covariate balancing propensity score*, JRSS-B
  76(1) (`imai2014covariate`), Section 2 (equations 4-19), from the
  authors' copy of the published article.
- `?CBPS` (CBPS 0.24 help page, via `tools::Rd2txt`).
- Outputs of `CBPS::CBPS` on the committed data sets below and on
  throw-away simulated data sets used to pin conventions (all listed under
  "Conventions established from the reference's outputs").
- The repository's tests, fixtures and generators that call `sp.cbps`
  (`grep -rlE 'cbps' tests`), its callers in `src/statspai`
  (`sp.match(method='cbps')`, `did/_pmatch_refine.py`, the workflow, agent
  summaries), `docs/dev/modules/treatment_effects.md`.
- The redacted view of the old `matching/cbps.py` (all function bodies and
  their docstrings removed). That view still showed the module docstring and
  one comment block above the solver, which described the old code's
  structure: the design is standardised and orthonormalised, the GMM
  weighting matrix is the model-implied covariance "evaluated at the starting
  value", and the start is "logit MLE -> scalar rescaling on [0.8, 1.1] ->
  BFGS on the balance loss -> two BFGS runs on the GMM loss". This is
  recorded because it suggested which candidate weighting points to test;
  every convention below that the new code uses was then established from
  the reference's outputs (the tests that pin each are named). The
  coordinator reviewed this exposure on 2026-10-20 and accepted the rewrite
  without a fresh-agent redo: what was seen was a prose description of an
  algorithmic convention, not code expression, and the convention was
  confirmed independently from CBPS black-box outputs on ten data sets.
- Outputs of the previous StatsPAI implementation on 58 configurations, run
  as a black box before it was deleted (snapshot kept outside the repo).

Not read: any CBPS R source (no function bodies printed, no CRAN/GitHub
source); the bodies or docstrings of the old functions; the old file's git
history.

## Estimand and estimator

Binary treatment `T`, design `X` (constant + covariates), `n` units, `n1`
treated. Logistic model `pi(X; beta) = expit(X'beta)` (eq. 5). Sample moments
(all divided by `n`):

| block | ATE | ATT |
| --- | --- | --- |
| score (eq. 7) | `X'(T - pi)/n` | same |
| balance (eq. 10/11) | `X'w/n`, `w = T/pi - (1-T)/(1-pi)` | `X'w/n`, `w = (n/n1)(T - pi)/(1-pi)` |

Model-implied moment covariance (eq. 16 for the ATE, eq. 17 for the ATT),
`v = pi(1 - pi)`:

    ATE:  [[ X'diag(v)X,  X'X ],  [ X'X,  X'diag(1/v)X ]] / n
    ATT:  [[ X'diag(v)X,  (n/n1) X'diag(pi)X ],
           [ (n/n1) X'diag(pi)X,  (n/n1)^2 X'diag(pi/(1-pi))X ]] / n

* `variant='exact'`: the balance block alone, `b(beta) = 0` (just
  identified). The balance moments are the gradient of a strictly concave
  function of `beta`:
  ATE `sum_T (eta - e^{-eta}) - sum_C (eta + e^{eta})`, ATT
  `(n/n1)(sum_T eta - sum_C e^{eta})` (`eta = X'beta`, both over `n`), so the
  root is unique when it exists; it is found by Newton's method with an
  Armijo line search on that function, started at the logit MLE. Converged
  when `max |b| < 1e-8` (standardised basis). For the ATT this is the dual
  of entropy balancing, consistent with the `sp.weightit` note in
  `treatment_effects.md`.
* `variant='over'`: two-step GMM, `beta = argmin g(beta)' Sigma0^+ g(beta)`,
  `g = [score; balance]`, `Sigma0 = Sigma(a * beta_MLE)` with
  `a = argmin_{a in [0.8, 1.1]} g(a beta_MLE)' Sigma(a beta_MLE)^{-1} g(a beta_MLE)`
  (the continuously-updated criterion along the ray of the MLE; see below).
  Solved by Levenberg-Marquardt (`scipy.optimize.least_squares`) on the
  whitened residual `W g` (`W'W = Sigma0^+`, pseudo-inverse on the positive
  eigenspace so the intercept-only design with a singular `Sigma` works),
  from three starts (`a beta_MLE`, `beta_MLE`, the exact solution when it
  converged), keeping the lowest criterion.
* Both criteria are invariant to a non-singular linear reparametrisation of
  the covariates (the moments and `Sigma` transform together, and the MLE is
  equivariant), so the problem is solved on centred and scaled covariates
  and the coefficients mapped back. Standardising is a numerical device only.
* `gmm_objective`: the over-identified criterion at the solution (equals
  CBPS's reported `J`, which is `g'Sigma0^{-1}g` without the factor `n`); for
  `exact`, the balance-block criterion `b' Sigma_bb(beta)^{-1} b` at the
  solution (zero to rounding when the fit converged).

Effect: Hajek contrast with arm weights `T/pi`, `(1-T)/(1-pi)` (ATE) or
`T`, `(1-T) pi/(1-pi)` (ATT), each normalised to sum to one within its arm.
Scores are clipped to `[max(1e-6, trim), 1 - max(1e-6, trim)]` before
weighting (`trim` winsorises, it does not drop units).

Balance diagnostic `std_mean_diff_after[col]`: weighted treated mean minus
weighted control mean, divided by `sqrt((var_T + var_C)/2)` with unweighted
within-arm variances (divisor `n_arm`); 0 for the constant column. (The
denominator was identified from the previous implementation's outputs; it is
a StatsPAI reporting convention, not a CBPS output.)

Inference: nonparametric bootstrap. Resample rows with
`numpy.random.default_rng(seed).integers(0, n, n)`, refit the whole CBPS,
recompute the Hajek contrast; `se` is the sample SD (`ddof=1`) of the
replicate estimates; normal interval and two-sided p-value. A resample that
cannot be fitted (empty arm, lost constant, rank-deficient design) or whose
fit does not converge is redrawn, at most `n_bootstrap + 10` times in total,
and counted in `n_bootstrap_retries` (non-convergence also in
`n_bootstrap_nonconverged`), with a `ConvergenceWarning`. Fewer than two
replicates: `se`, `pvalue`, `ci` are NaN. A replicate equals a fresh
`sp.cbps` call on the resampled rows (tested).

## Conventions established from the reference's outputs

All on `CBPS(..., standardize = FALSE)` unless stated.

1. **Moment covariance = eq. 16 / 17.** With `twostep = FALSE` (the paper's
   continuously-updated estimator), `g(beta)'Sigma(beta)^{-1}g(beta)` at
   CBPS's coefficients reproduces its `J`, and at the glm MLE its `mle.J`, to
   1e-12 (ATE and ATT). Test: `test_cbps_R_parity.py::test_cue_J_is_the_eq16_17_criterion`.
2. **Two-step weighting point.** With the default `twostep = TRUE`, `mle.J`
   differs from the CUE `mle.J`, so `Sigma0` is not taken at the MLE. Solving
   `g(beta_MLE)'Sigma(beta_s)^{-1}g(beta_MLE) = mle.J` together with the
   first-order condition at CBPS's coefficients for `beta_s` gave
   `beta_s ~= 0.983 beta_MLE` (component ratios 0.9822-0.9834) on one data
   set; along the ray `a beta_MLE` the value of `a` that reproduces `mle.J`
   (0.98332 and 0.95838 on two data sets) is the minimiser of the CUE
   criterion along the ray (0.98331, 0.95838). On data sets where that
   minimiser is below 0.8 (0.66-0.74, eight cases) `mle.J` is reproduced at
   `a = 0.8`; where it is above 1.1 (1.19, 1.23) at `a = 1.1`. Hence
   `a in [0.8, 1.1]`. CBPS's scalar search stops about 1e-4 short of the
   bound, so `mle.J` agrees to ~2e-4 relative at the bounds; StatsPAI
   minimises to 1e-10. Tests: `test_two_step_weighting_point_reproduces_mle_J`
   (interior and bound cases).
3. **CBPS's reported `J`** equals our criterion at its coefficients with the
   same `Sigma0`.
4. **ATT, over-identified.** CBPS's ATT two-step fit frequently stops where
   the criterion is not stationary (on one data set its coefficients were
   exactly `a beta_MLE`, the start); StatsPAI reaches a lower value of the
   same criterion. Asserted as dominance, not parity
   (`test_att_over_attains_a_lower_criterion`,
   `test_matching_r_parity.py::test_cbps_att_over_dominates_r`).
5. **Weights.** CBPS's `standardize = TRUE` weights are the arm-normalised
   Hajek weights above (ATE: `T/pi`, `(1-T)/(1-pi)`; ATT: 1 for the treated,
   `pi/(1-pi)` for the controls), so the Hajek contrast is the same number
   computed from either. Test: `test_weights_are_the_hajek_weights`.

## Edge cases

| input | behaviour |
| --- | --- |
| `estimand` not ATE/ATT, `variant` not exact/over | `MethodIncompatibility` (a `ValueError`) |
| labelled treatment | `MethodIncompatibility` "holds labels" (`treatment_as_float`) |
| treatment not 0/1 | `MethodIncompatibility` "coded 0/1" |
| one arm empty after dropping NAs | `DataInsufficient` (**new**: the old code returned a number) |
| missing column | `ColumnNotFound` |
| NA in a used column | row dropped |
| no or several constant design columns | `MethodIncompatibility` "exactly one constant" |
| collinear covariates | `MethodIncompatibility` "not full rank" |
| fit not converged | `ConvergenceWarning`, `converged = False` (**new** warning) |
| scores at the 1e-6 bound (near separation) | `ConvergenceWarning` (**new**) |

## Comparison with the previous implementation

58 configurations (7 data sets including `matching_lalonde.csv`; ATE/ATT x
exact/over; bootstrap; trim; user constant; NA rows; categorical covariate;
errors). On every regular configuration the estimate agrees to <= 2e-7
relative, the coefficients to <= 7e-7, the bootstrap SE to <= 1.2e-7 (same
resamples), `gmm_objective` (over) to 4 significant digits; the differences
are the previous optimiser's stopping slack. Differences in behaviour: an
all-treated (or all-control) sample now raises `DataInsufficient` instead of
returning an estimate; a perfectly separated design now warns (the
over-identified estimate moved by 1.8%: there is no finite solution, both
codes return a point on a diverging path); `_intercept` in
`std_mean_diff_after` is exactly 0 (it was +/-1e-9 rounding noise);
`model_info` gains `weighting_point_scale` (the `a` above) for `over`.

## Parity evidence and tolerances

New fixture `tests/reference_parity/_fixtures/cbps_R.json` (generator
`_fixtures/_generate_cbps_R.R`, data `cbps_sim.csv` / `cbps_bound.csv` from
`_generate_cbps_medsens_data.py`, plus `matching_lalonde.csv`), test
`tests/reference_parity/test_cbps_R_parity.py` (37 cases). Measured gaps:

| quantity | sim ATE / ATT | bound ATE / ATT | lalonde ATE / ATT | test tol |
| --- | --- | --- | --- | --- |
| CUE `J`, `mle.J` at CBPS's coefficients (eqs. 16-17) | 2e-13 / 5e-13 | 3e-14 / 9e-14 | 8e-14 / 8e-12 | 1e-9 |
| two-step `mle.J` with our scale `a` | 6.6e-5 / 6.4e-5 | 2.2e-4 / 6.3e-6 | 1.7e-6 / 1.6e-6 | 1e-3 |
| two-step `J` at CBPS's coefficients, our `a` | 4.4e-5 / 6.9e-5 | 6.5e-5 / 3.2e-5 | 1.1e-6 / 4.4e-8 | 1e-3 |
| `a` | 0.9703 / 0.9680 | 0.8 (bound) / 0.8 (bound) | 0.9584 / 0.9664 | |
| coefficients, ATE over | 7.5e-5 | 8.9e-5 | 1.6e-3 | 2e-3 |
| coefficients, exact | 8.1e-5 / 1.5e-4 | 1.1e-4 / 3.5e-4 | 1.7e-3 / 3.2e-3 | 5e-3 |

The `mle.J` / `J` gaps are CBPS's scalar search stopping about 1e-4 short in
`a` (largest at the bound); the coefficient gaps are CBPS's `optim` slack:
on the exact fits it leaves a balance residual of 2e-5 to 3e-4 (standardised
units) where StatsPAI's is below 1e-14, asserted as `ours < 1e-12 and ours <
CBPS's`. ATT over-identified: dominance on the two-step criterion with the
same `Sigma0` (CBPS's coefficients differ by 22-476%). The existing
`matching_R.json` estimates (lalonde) agree to 7.3e-4 (ATE over), 1.1e-4
(ATE exact), 6.3e-4 (ATT exact), within the registered 5e-3 / 2e-3 of
`test_matching_r_parity.py`; ATT over is the dominance case there too.
