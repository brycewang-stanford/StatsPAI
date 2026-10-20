# Clean-room specification: Arellano-Bond serial-correlation tests (dynamic-panel GMM)

Audit row H25 of `docs/dev/2026-10-20-license-provenance-audit.md`:
`gmm/_dynpanel/_diagnostics.py` (`arellano_bond_ar_test`,
`_lagged_residual_vector`, `ar_test_cross_basis`) and the block of
`gmm/_dynpanel/_fit.py` that prepared their inputs described themselves as
implementing Stata `xtabond2`'s `_ARTests` (GPL-3) term by term. The three
functions were deleted by AST span without being read, the `_fit.py` block
(lines 450-557) was cut by line range after a structural map that printed
only statement types and assigned names, and the module docstring of
`_diagnostics.py` was replaced unread (its first five lines were printed
once by mistake; they named the AR(1)/AR(2) tests and the MA(1) induced by
differencing, nothing about the implementation).

## Sources (and only these)

- Arellano and Bond (1991) `[@arellano1991some]` (the test itself, as
  summarised by Roodman).
- Roodman (2009) `[@roodman2009xtabond]`, read in its CGD Working Paper 103
  version (December 2006), Section 3.5 "Testing for autocorrelation",
  including the variance formula after eq. (30) and the mapping paragraph
  for difference and system GMM.
- The `xtabond2` help file (`xtabond2.hlp`): the test is applied to
  differenced residuals, also after estimation in orthogonal deviations;
  `cluster()` changes the AR tests; `h()` options.
- Black-box runs of Stata 18 `xtabond` / `xtabond2` on `abdata`: the
  committed fixture `tests/reference_parity/_fixtures/dynpanel_stata.json`
  and nine new classical-variance specifications (system h(1)/h(2)/h(3),
  orthogonal deviations, h(1) with orthogonal deviations), plus the
  previous StatsPAI function run as a black box on 676 configurations.

## The statistic

`e` = first-differenced residuals of the test rows (transformed-equation
rows; level rows of system GMM are excluded); `q` = for each test row the
same unit's residual `l` periods earlier (period arithmetic, so a gap gives
zero), zero if that row does not exist. `z = q'e / sqrt(V)`,

`V = S_qq - 2 q'X M^{-1} X'Z A S_Zq + q'X Avar(b) X'q`

(Roodman 2009, after eq. 30), `X`, `Z`, `A` the estimator's regressors,
instruments and weight, `M = X'ZAZ'X`, `Avar(b)` the reported coefficient
covariance (classical, robust, or Windmeijer-corrected).

- **Robust / two-step**: `S_qq = sum_i (q_i'e_i)^2`,
  `S_Zq = sum_i Z_i' u_i (e_i' q_i)` with `u` the estimation residuals,
  summed over panel units even when the coefficient covariance is
  clustered more coarsely (black box: `cluster()` specs match to 1e-12 this
  way; clustering enters through `Avar`).
- **Classical one-step** (`Var(E) = s^2 H` in Roodman's notation):
  covariances implied by i.i.d. level errors of variance `s^2` on the
  transformed rows (`e_p - e_{p-1}` for differences; the balanced-grid
  deviation operator for orthogonal deviations); level rows contribute
  nothing. With `h=1` the identity is used instead (the assumption the
  one-step weight makes). Black box: xtabond2 classical system h(2)/h(3),
  difference, and orthogonal-deviation statistics match to 1e-13 after the
  `s^2` scale convention below; h(1) difference to the same scale factor.
- **Orthogonal deviations**: the test residuals are recomputed in first
  differences from the same specification (`build_design(...,
  transform='fd')`) at the estimated coefficients; `X` in `q'X` is the
  differenced design, the cross term pairs estimation-basis and test-basis
  rows of the same unit.

## sigma^2 (a correction found during the rewrite)

`level_sigma2` divided `e'e` by `2(N* - k)` for every transform. The factor
2 is the diagonal of `H` for first differences; after orthogonal deviations
(`M M' = I`) and with `h=1` (`H = I`) it is 1. ⚠️ correctness fix: classical
one-step standard errors after `transform='fod'` or `h=1` were too small by
about sqrt(2) (abdata FOD: SE of L.n 0.0642 vs xtabond2 0.0907; now 0.0908,
the remaining 0.2% being xtabond2's `N*` vs StatsPAI's `N* - k` divisor),
and the Sargan statistic doubled. Robust and two-step results are
unaffected. Remaining documented differences against xtabond2: the `s^2`
divisor (`N* - k`, xtabond's convention, vs xtabond2's `N*`; a constant
factor on classical z statistics), and for system GMM with both `h(1)` and
orthogonal deviations a larger constant scale factor in xtabond2's
classical AR statistic (its `s^2` there coincides with the level
residuals' mean square; not pursued).

## Acceptance

- Every Stata fixture AR value (`TestArellanoBondSerialCorrelation`, system,
  FOD, cluster, gapped-panel specs) at 1e-9; all robust and two-step
  configurations of the 676-configuration black-box grid within 2.4e-12 of
  the previous function. Classical system and classical FOD statistics
  change (the previous ones did not match xtabond2: abdata system -8.12 vs
  xtabond2 -8.09 after scaling, FOD -2.76 vs -6.44); they now match.
- Track A `50_xtabond` (one-step robust) reproduces.
