# Clean-room specification: `sp.scdata` / `sp.scest` / `sp.scpi`

Audit row H1 of `docs/dev/2026-10-20-license-provenance-audit.md`. The
previous `synth/scpi.py`, `synth/_scpi_inference.py` and
`synth/_scpi_solvers.py` described themselves as a port of the R package
`scpi` 4.0.1 (GPL-3; the Python `scpi-pkg` by the same authors is
GPL-3.0-only). They were deleted and rewritten from this specification.

## Sources

Used:

- Cattaneo, Feng and Titiunik (2021), JASA 116(536), 1865-1880,
  doi:10.1080/01621459.2021.1979561 (`cattaneo2021prediction`); read as
  arXiv:1912.07120v4.
- Cattaneo, Feng, Palomba and Titiunik (2025), JSS 113(1), 1-38,
  doi:10.18637/jss.v113.i01 (`cattaneo2025scpi`); the published PDF.
- Cattaneo, Feng, Palomba and Titiunik, "Uncertainty Quantification in
  Synthetic Controls with Staggered Treatment Adoption", Review of
  Economics and Statistics, doi:10.1162/rest_a_01588; read as
  arXiv:2210.05026v5 with its supplement (Sections 4.5, 6, S.2, S.5).
  Not cited in code (not in `paper.bib`).
- Restricted regression quantiles: He (1997), "Quantile Curves without
  Crossing", The American Statistician 51(2), 186-192,
  doi:10.1080/00031305.1997.10473959, cited by the `Qtools::rrq` help page
  and by Geraci (2016), "Qtools: A Collection of Models and Tools for
  Quantile Inference", The R Journal 8(2), doi:10.32614/RJ-2016-037 (both
  checked on Crossref and doi.org). He's paper itself was not available;
  the three-step location-scale construction (median regression, median
  regression of the absolute residuals, quantile regression of the
  residuals on the fitted scale through the origin) was confirmed against
  `Qtools::rrq(method = "br")` as a black box (agreement 7e-15 on random
  designs).
- R help pages of `scpi::scpi`, `scpi::scest`, `scpi::scdata`,
  `scpi::scdataMulti`, `Qtools::rrq`, `quantreg::rq` (rendered with
  `tools::Rd2txt`).
- R `scpi` 4.0.1 run as a black box (CVXR 1.9.2, ECOSolveR 0.6.1, Qtools
  1.6.0, quantreg 6.1) on the committed Germany / California fixtures and
  on about 60 synthetic panels; its documented return values
  (`inference.results$rho`, `Q.star`, `u.mean`, `u.var`, `Sigma`, `u.D`,
  `e.D`, `e.mean`, `e.var`, `bounds`, `failed.sims`, `est.results$w.constr`)
  and, through the hook the committed generator already uses, the per-draw
  in-sample values. The harness is in the session scratchpad, not in the
  repository.
- The repository's tests and fixtures, the pre-rewrite StatsPAI outputs
  (numbers only, as a regression target), `docs/dev/campaign_phase3/did_synth_scpi.md`,
  and the redacted signatures of the old modules.

Seen through the redaction helper before it was changed to strip module
docstrings: the module docstrings and section comments of the three old
files (`scpi.py`, `_scpi_inference.py`, `_scpi_solvers.py`) and the first
line of each function docstring. They name R internals (`shrinkage.EST`,
`w.constr.OBJ`, `b.est`, `local.geom`, `local.geom.2step`,
`insampleUncertaintyGetDiag`, `u.des.prep`, `u.sigma.est`, `df.EST`,
`e.des.prep`, `scpi.out`, `simultaneousPredGet`, `regularize.w`), list the
five weight problems and the in-sample problem `min c'x s.t.
(x - b)'Q(x - b) - 2G'(x - b) <= 0` with a locally relaxed constraint set,
and say the old code used an active set, closed forms and HiGHS LPs. No
formula or convention in this specification was taken from them: each is
pinned by the papers, the help pages or a black-box run cited here, and
none of those names appears in the new code.

Not read: any source of R `scpi`, Python `scpi-pkg`, Stata `scpi`,
`Qtools`, `quantreg`, `CVXR` or `ECOSolveR`; the bodies or full docstrings
of the old StatsPAI modules; their git history.

## Data (`scdata`)

`A` is the treated unit's pre-treatment outcome (T0), `B` the donors'
pre-treatment outcomes (T0 x J), `P` their post-treatment outcomes
(T1 x J); `C = None`, `KM = 0`. Pre-treatment periods are those before
`treatment_time`. Donors are sorted. Refused: duplicate (unit, time) rows,
a missing outcome of the treated unit, a donor missing a post-treatment
outcome, fewer than 2 pre-treatment or 1 post-treatment periods. Donors
missing a pre-treatment outcome are dropped with a warning.

## Weights (`scest`)

`V` = identity, so every problem is least squares on `(A, B)`:

| name | set | radius |
| --- | --- | --- |
| simplex | `w >= 0`, `sum(w) = Q` | `Q = 1` |
| lasso | `||w||_1 <= Q` | `Q = 1` |
| ridge | `||w||_2 <= Q` | data-driven, below |
| L1-L2 | `w >= 0`, `sum(w) = 1`, `||w||_2 <= Q2` | `Q2` = ridge `Q` |
| ols | none | |

Ridge radius (JSS Section 3.1, Hoerl-Kennard-Baldwin choice; black box
for the details): if `T0 - J > 10`, `b` = least squares on all donors and
`k = J`; otherwise the donors with a non-zero weight in the L1-ball fit
`||w||_1 <= 1` are kept, at most `max(T0 - 10, 2)` of them (the largest in
absolute value), `b` = least squares on them and `k` = their number. Then
`sigma^2 = RSS / (T0 - k)`, `lambda = k sigma^2 / ||b||^2`,
`Q = max(||b|| / (1 + lambda), 0.5)`. When `T0 < 5` R reports `lambda` as
missing (Q is still computed); `sp.scpi` then refuses the data-driven ridge
radius (R stops with an internal error there) and asks for `Q=`. For
L1-L2, `lambda` is the ridge one, or 0 when that is missing. A user `Q` /
`Q2` sets `lambda = None`.

Black-box evidence: Germany `Q = 0.66876...`, `lambda = 0.096076...` to
1e-14; the subset rule reproduces R's `lambda` to 1e-10 on 15 synthetic
panels covering `T0 - J` in {-4, 0, 1, 4, 6, 8, 9, 10, 11} and `T0` in
{3, 4, 5, 6}; the boundary is `T0 - J = 10` screened, `11` not.

All problems are solved exactly: an active-set QP for the simplex, the
L1 ball as a sum-constrained QP in `w = p - n` (or the smallest-L1
least-squares solution when the ball is slack), the secular equation for
the L2 ball, a monotone search on the L2 penalty for L1-L2. Minimum-norm
least squares when OLS is not identified (warned).

## Prediction intervals (`scpi`)

Notation: `res = A - B w`, `d0 = #{|w_j| >= 1e-6}`.

**Degrees of freedom** (JSS Section 4.3): ols `J`; lasso `d0`; simplex
`d0 - 1`; ridge and L1-L2 `sum_j s_j^2 / (s_j^2 + lambda)` with `s` the
singular values of `B` and `lambda` the radius-rule value (0 when the
radius is user-given). If `df >= T0` it is capped at `T0 - 1` with a
warning (R warns with the same diagnosis).

**rho** (help page, "Regularization"): `rho = sqrt(d0 log(J) log(T0)) C /
sqrt(T0)`, `C = sigma_u / min_j sigma_bj` (type-1) or
`max_j sigma_bj sigma_u / min_j sigma_bj^2` (type-2, default). Black box:
`sigma_u` is the population SD of `res`, `sigma_bj` the sample SD of
donor j (the ratio to R is exactly 1 on all five Germany constraints and
the type-1 panel); `rho = min(rho, rho_max)`; a data-driven value below
`1e-3` is replaced by `rho_max` (R: 0.000946 -> 0.2, 0.00103 kept). A user
number is used as given. A donor with zero pre-treatment variation is
refused (R refuses too).

**Active donors**: for simplex, lasso and L1-L2 the donors with
`|w_j| > rho` (if none, the largest; StatsPAI warns); for ridge and ols
all donors (Germany ridge: 17 design columns although 3 weights are below
rho).

**Simulation constraint set** (staggered paper eqs. 6.7-6.10; black box
for the radii): a constraint is binding when its slack is below rho.

- simplex and L1-L2: `w_j >= lb_j` with `lb_j = w_hat_j` if
  `w_hat_j < rho`, else 0; `sum(w) = Q`. L1-L2 adds `||w||_2 <= Q2*`,
  where `Q2*` follows the ridge rule below with `r = min(rho, rho_max)`:
  `Q2* = ||w_hat||^2 + r` if `||w_hat||^2 > Q2 - r`, else `Q2`. Black
  box (per-draw values; R does not report `Q2*`): on a T0 = 3 panel the
  enlarged radius reproduces R (data-driven and user Q2, max 3e-6) and Q2
  does not (0.15); with a user rho = 0.3 on three T0 = 12 panels Q2
  reproduces R (max 3e-5) and `||w_hat||^2 + 0.3` does not (0.07-0.29),
  which `r = min(rho, rho_max) = 0.2` explains (not binding). Whether R
  uses `rho_max` or a data-driven rho in that last case is not settled by
  these runs; for data-driven rho the two coincide. In the other twelve
  L1-L2 runs the ball does not bind in the simulation or the rule leaves
  Q2 unchanged.
- lasso: `||w||_1 <= Q*`, `Q* = ||w_hat||_1 + rho` if `||w_hat||_1 > Q -
  rho`, else `Q* = Q` (R `Q.star`: Q = 0.5, 1 binding -> `||w||_1 + 0.2`;
  Q = 1.3, 1.4 -> `1.2773 + 0.2`; Q = 1.5, 1.7, 1.9 -> Q).
- ridge: `||w||_2 <= Q*`, `Q* = ||w_hat||_2^2 + rho` if `||w_hat||_2^2 >
  Q - rho`, else `Q* = Q`. R compares the squared norm with Q, unlike the
  lasso rule; the black box leaves no other reading: 13 ridge fits (data
  -driven and user radii 0.3 to 10) are all classified correctly, the six
  enlarged ones have `Q.star - ||w_hat||^2 = 0.2 = rho` to 1e-9 (e.g. user
  Q = 0.8 gives 0.84 in five different panels), and the per-draw values of
  an enlarged case are reproduced with radius `Q*` (median 1e-10) but not
  with `Q` (0.14) or `sqrt(Q*)` (0.04). Germany (`||w_hat||^2 = 0.447 <
  0.669 - 0.065`) keeps `Q* = Q`. A user Q above the least-squares norm
  can still be enlarged (Q = 1.2 with `||w_hat||^2 = 1.039`: `Q* =
  1.239`, confirmed per draw).
- ols: none.

Checked against R's per-draw values: median gap 1e-8 to ECOS at 1e-12
for all five constraints.

**Pseudo-residual moments** (JSS Step 1): `u.missp = TRUE`: least squares
of `res` on `D_u = [B_active, 1]` (u.order 1) or `[1]` (0); the design
falls back to `[1]` when `T0 <= ncol(D_u) + 10` (R: T0 = 17 with 7
columns falls back, with 3 does not). `u.missp = FALSE`: mean 0,
`u.T = u.params = 0`. `u.var_t = v_t (res_t - u.mean_t)^2` with HC0 1,
HC1 `T0 / (T0 - df)`, HC2 `1 / (1 - h_t)`, HC3 `1 / (1 - h_t)^2`, HC4
`(1 - h_t)^-min(4, T0 h_t / df)`, `h` the leverage of `B` (all donors;
black box: matches to 1e-13, not the leverage of `D_u`).

**Sigma** = `B' diag(u.var) B / T0^2` (black box, exact).

**Simulation**: `zeta = Sigma^(1/2) z` (symmetric root via SVD),
`z ~ N(0, I_J)`, `M = B'B / T0`. For each draw and post period `t`,
lower `= -max p_t'y`, upper `= -min p_t'y` over
`{y : y'M y - 2 zeta'y <= 0, w_hat + y in Delta*}`. In-sample bounds:
type-7 quantiles `alpha_u / 2` of the lower and `1 - alpha_u / 2` of the
upper values over draws (NaN draws dropped; `failed.sims` = percent NaN).

**Out-of-sample** (CFT 2021 Section 5; black box for the estimators):
design `D_e` as `D_u` with `e.order`; post rows `[P_active, 1]`.

- `e.mean` = least-squares prediction of `res` at the post rows.
- `e.var = min(exp(log-variance prediction), ((Q_0.75 - Q_0.25) / 1.34)^2)`:
  the first is the least-squares fit of `log((res - fit)^2)` on `D_e`
  (CFT 2021 Section 5.1.1), the second uses restricted regression
  quantiles of `res` on `D_e`, also when `D_e` is a constant (an LP
  quantile, not the type-7 one: a T0 = 6 panel tells them apart). Black box: which entries of R's `e.var`
  depend on the quantile-regression method is exactly the set where the
  second term is smaller (Germany: entry 1 of simplex, entry 13 of ridge
  come from the first).
- gaussian: `e.mean -/+ sqrt(2 e.var log(2 / alpha_e))`.
- ls: `e.mean + sqrt(e.var) q`, `q` the type-7 quantiles of
  `(res - fit) / sqrt(exp(log-variance fit))` in sample.
- qreg: restricted regression quantiles at `alpha_e / 2`, `1 - alpha_e / 2`;
  with a constant `D_e` (e.order = 0 or the fallback), the type-7 sample
  quantiles of `res` (black box: exact on four panels).

The combined bounds add the in-sample bounds.

**Simultaneous bounds** (staggered paper Section 4.5; black box for the
form): with `L = T1`, `m_lo` / `m_hi` the type-7 quantiles of the per-draw
minimum of the lower values over periods and the maximum of the upper
values, `joint_t = [m_lo + sqrt(log(L + 1)) (e.mean_t - s_t),
m_hi + sqrt(log(L + 1)) (e.mean_t + s_t)]`, `s_t = sqrt(2 e.var_t log(2 /
alpha_e))`. Black box: the joint bounds ignore `e.bounds=`, and with
`w.bounds = 0` they equal the sub-Gaussian bounds times `sqrt(log 14)`
for T1 = 13 (and `sqrt(log 8)` for T1 = 7).

**Average effect** (`.ci`): R `scdataMulti(effect = "unit")` treats the
mean of the post-treatment rows as one period; the in-sample problem uses
`mean(P)` and the out-of-sample moments the mean of the post design rows.

## Agreement with R on all black-box runs

Fed R's weights and draws, the new code reproduces R's bounds within the
ECOS / Frisch-Newton tolerances (<= 2e-3, mostly 1e-6) on 110 of the 113
black-box runs. The three exceptions are reference-solver artefacts:

- one ridge panel (T0 = 18, J = 8, `T0 - J = 10`, screened radius): R's
  OSQP lasso leaves a donor at a small non-zero value that the exact L1-ball
  optimum has at 0 with a strict KKT margin (gradient 0.2746 against the
  multiplier 0.2814), so R screens 7 donors where the exact fit screens 6;
  R's radius is 1.403 and ours 1.101. Given R's radius, the per-draw values
  agree to 1e-5. Same class as the Germany simplex donor at 2e-6 (T4).
- two near-exact fits (residuals 1e-4): the out-of-sample bounds differ by
  up to 2e-3, the interior-point error of R's default `rrq` on near-zero
  residuals.

## Solving the simulation problems (StatsPAI, not R)

R hands every problem to ECOS. Here each is solved exactly
(`_scpi_insample.py`, `_scpi_simulate.py`, `_scpi_qp.py`):

1. primal active set over the lower bounds, with the working-face
   subproblem in closed form (one convex quadratic: the eigenbasis
   solution; with the L2 ball: the single-constraint solution at the
   multiplier ratio found by a monotone root search); bound multipliers
   from the face, or from a small LP when they are not unique;
2. if the active set does not settle (it can cycle when a released bound
   meets a curved constraint), SLSQP is used only to guess the active
   bounds, the face is solved in closed form and kept if it satisfies the
   KKT conditions;
3. otherwise the Lagrangian dual in the quadratic constraint, with an exact
   QP over the polyhedron for each multiplier (plus the ratio search for
   the ball);
4. SLSQP itself as the last resort, counted in `n_slsqp_fallback`.

On the fixtures step 4 never runs. On an ill-conditioned test panel
(`cond(B'B) = 3e3`, one dominant weight) the previous implementation fell
back to SLSQP in 294 of 4000 problems and left 11 unsolved; this one
solves all exactly and agrees with the previous one to 7e-13 where both
solved.

## Edge behaviour (StatsPAI decisions, not R)

- Warnings for rho too low (fallback to `rho_max`), rho too high (no
  weight exceeds it), more degrees of freedom than observations, all
  simulation problems unbounded (ols with `J > T0`).
- `sims >= 10`; `u_order` / `e_order` > 1, lags, covariates, several
  treated units in `sp.scpi` are not implemented.
- `se` / `pvalue` are NaN; `alpha` is the default for both `u_alpha` and
  `e_alpha`. The reported interval's `alpha` is `u_alpha + e_alpha` for
  `pi_type='both'` (nominal coverage `1 - u_alpha - e_alpha`), and
  `u_alpha` / `e_alpha` for the in-sample / out-of-sample interval alone.
- When a squared residual is exactly zero, its logarithm is taken at the
  smallest positive double (R would stop on `log(0)`).

## Acceptance

`tests/reference_parity/test_did_synth_scpi_parity.py` (48, Germany and
California), `tests/reference_parity/test_scpi_cleanroom_parity.py` (new:
17 configurations on three committed synthetic panels, generated by
`_fixtures/_generate_scpi_cleanroom_{data.py,R.R}`),
`tests/reference_parity/test_scest_multi_parity.py`,
`tests/test_cov95_synth_gaps_scpi.py` and every other test that calls
`scpi` / `scest` / `scdata` pass. On the Germany fixture with R's draws
the new code reproduces the pre-rewrite StatsPAI outputs to 1e-10, and
seeded runs on a synthetic panel to 1e-13 (same draw stream). On the 110
black-box panels the two implementations agree everywhere except L1-L2,
where the old in-sample bounds were off R by 1.6e-4 to 6.3e-3 on eight
of 26 panels (1.4e-2 on the fixture's short panel) and the new ones agree
with R to 5e-6 (a correctness fix, in
CHANGELOG); and on ill-conditioned problems the old code left draws
unsolved or fell back to SLSQP, where the new one solves them exactly. The scpi
files are not on a Track A, original-data or Track C traced path, so no
frozen artefact changes.
