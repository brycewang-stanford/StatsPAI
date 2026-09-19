# did_synth / `scpi` cluster: `sp.scdata`, `sp.scest`, `sp.scpi` vs R `scpi` 4.0.1

Reference: R `scpi` 4.0.1 (`scdata`, `scest`, `scpi`, `scdataMulti`), R 4.5.2.
Its solvers: CVXR 1.9.2 with CLARABEL 0.11.2 (OSQP 1.0.0 for lasso) for the weights,
ECOSolveR 0.6.1 for the in-sample simulation, and Qtools 1.6.0 `rrq` on quantreg 6.1 for the
out-of-sample moments.
Data: `scpi::scpi_germany` (the package's own vignette panel: West Germany, J = 16, T0 = 31,
T1 = 13) and `sp.california_prop99()` (J = 38 > T0 = 19, so Z'Z is singular).

Files:
- `tests/reference_parity/_generate_did_synth_scpi_R.R` writes `_fixtures/did_synth_scpi_R.json` and `_fixtures/did_synth_scpi_germany.csv`.
- `tests/reference_parity/_fixtures/_generate_did_synth_scpi_data.py` writes `_fixtures/did_synth_scpi_california.csv`.
- The test is `tests/reference_parity/test_did_synth_scpi_parity.py`: 48 tests, about 90 s.
- Source: `src/statspai/synth/scpi.py` was rewritten. It has two new private modules, `synth/_scpi_inference.py` and `synth/_scpi_solvers.py`, which only `scpi.py` uses.

## 1. Per-function table

| function | reference | class | max rel err est / SE-equivalent | test |
|---|---|---|---|---|
| `sp.scdata` | `scpi::scdata` 4.0.1 | 1 (bit-exact) | A, B, P identical (0); donor order = R `sort(B.names)` | test_did_synth_scpi_parity.py |
| `sp.scest` | `scpi::scest` 4.0.1 (CLARABEL / OSQP) | 2 → 1 (aligned; bound set by the solver) | **ols** 1.7e-14 and **lasso** 1.8e-12 (abs, weights); **simplex / L1-L2 / ridge**: R's objective exceeds ours by 2.3e-8 / 3.4e-9 / 1.6e-8 relative (CLARABEL gap 1e-8), so max weight gaps are 2.0e-6 / 2.9e-7 / 1.4e-5, inside the strong-convexity bound `sqrt(gap/λmin)`; ridge Q, lambda and L1-L2 Q2 agree to 1e-14 | same |
| `sp.scpi` | `scpi::scpi` 4.0.1 (ECOS + rrq) | 2 → 1 (aligned; bound set by the solver) for the deterministic path and the draw-fed simulation; 5 (T3) for simulated bounds under our own RNG; 4 for the default simplex end-to-end `rho` / `df` | On R's weights: rho, Q.star, u.mean, Omega, Sigma, e.mean agree to ≤ 1e-13. Out-of-sample bounds vs `rrq(method="br")` (exact LP): ≤ 5e-12. vs the default Frisch-Newton fit: ≤ 2.2e-4 abs. In-sample per-draw vs ECOS at 1e-12: median ~1e-8, max 8.6e-5. In-sample quantile bounds vs ECOS at 1e-12: ≤ 4e-6 abs. vs default ECOS: ≤ 1.7e-3 abs (ols) and ≤ 2.4e-4 (simplex). The average-effect CI matches `scdataMulti(effect="unit")` to 1e-4 | same |

## 2. Defects found

### D1. `sp.scest(w_constr="lasso"|"ridge")`: wrong estimator under the right name (⚠️ correctness)

- **What was wrong.** `lasso` ran penalised coordinate descent on *standardised, demeaned*
  data with `lasso_lambda=1`. `ridge` solved `(X'X + λI)^{-1}X'y` with `ridge_lambda=1`.
  R `scpi` does something else. For lasso it constrains `||w||_1 ≤ Q` with Q = 1. For ridge it
  constrains `||w||_2 ≤ Q`, with Q set by the `shrinkage.EST` rule:
  `max(||b_ols||/(1+λ), 0.5)`, where `λ = σ²J/||b_ols||²`.
- **How found.** The first divergence was the weight vector itself, compared against `scest`.
- **Before → after** (Germany). Max |w − w_R| was 0.227 (lasso) and 0.194 (ridge). It is now
  1.8e-12 (lasso) and 1.4e-5 (ridge; the CLARABEL gap). The lasso weights before summed to 1.038.
- **Fix.** R's constraint sets and radii, solved exactly:
  - lasso: `lars_path`, then a KKT re-solve;
  - ridge: the secular equation;
  - L1-L2 (new; R's fifth constraint): Brent search on the penalty of the simplex QP.

  `lasso_lambda` / `ridge_lambda` are now deprecated. They are ignored with a
  `DeprecationWarning`, and the new `Q=` / `Q2=` arguments replace them.
- **Default output changed?** Not for `w_constr="simplex"` (the default) or `"ols"`. Yes for
  lasso and ridge.

### D2. `sp.scpi`: not the Cattaneo-Feng-Titiunik procedure (⚠️ correctness)

- **What was wrong.** None of R's inference components existed:
  - the in-sample part subsampled floor(T0^{2/3}) pre-periods, re-estimated the weights and
    scaled their variance by b/T0;
  - the out-of-sample part was the residual variance (gaussian), a linear trend of |e_t|
    (ls), or an empirical IQR converted to a variance (qreg);
  - the PI was `effect ± z·sqrt(var_in + var_out)`;
  - it also reported an SE and a p-value that R does not define.

  R instead has: the rho regularisation, the local geometry (`local.geom`,
  `local.geom.2step`), the HC-type variance `Sigma` of the pseudo-residuals, and a
  simulation that solves a constrained QCQP per draw, horizon and side. It also has
  sub-Gaussian / location-scale / `rrq` quantile bounds and the joint bounds.
- **How found.** Reading `scpi.R`, `insampleUncertaintyGetDiag.R` and `scpi.out.R`: none of
  the reference quantities (rho, Sigma, e.mean, e.var) had a counterpart.
- **Before → after** (Germany, simplex, defaults, e_method gaussian; effect intervals):

  | | before | after | R (same weights, gaussian) |
  |---|---|---|---|
  | 1991 | [0.230, 0.774] | [-0.731, 1.316] | [-0.700, 1.302] |
  | 2003 | [-4.32, -2.61] | [-5.25, -1.11] | [-5.35, -1.28] |

  The "after" row is the new StatsPAI result with its own RNG, so it is Monte Carlo relative
  to R. The old intervals were 2–4× too narrow.

  The aggregate CI for the average effect went from (-2.29, -1.05) to (-3.29, -0.22).
  `se` / `pvalue` are now NaN: prediction intervals have no standard error.
  `alpha` now means u_alpha = e_alpha, so the combined nominal coverage is
  `1 - (u_alpha + e_alpha)` (0.90 by default), as in R.
- **Fix.** A full port of the single-treated-unit path (`effect="unit-time"`, V = I, outcome
  feature, `u.lags = e.lags = 0`, u/e order 0 or 1). Every conic sub-problem is solved exactly:
  - active-set primal QCQP with closed-form subproblems;
  - a 1-D ratio root when the L2 ball also binds;
  - null-space rays when Z'Z is singular;
  - exact HiGHS LPs for the quantile regressions.

  SLSQP remains as a fallback; it was never triggered on either fixture panel.

  New arguments: `sims`, `u_missp`, `u_sigma`, `u_order`, `u_alpha`, `e_order`, `e_alpha`,
  `rho`, `rho_max`, `Q`, `Q2` and `draws`. `draws` is a (J, sims) matrix of standard normals;
  passing R's reproduces R's simulation.

  `ci` is the R `scdataMulti(effect="unit")` construction, and the test verifies it.
  `model_info` carries `bounds`, `CI`, `rho`, `Sigma`, `u_var`, `e_mean`, `e_var`, `df`,
  `failed_sims` and `vsig`. `period_results` keeps `pi_lower` / `pi_upper` (the plots depend
  on them) and adds `joint_lower` / `joint_upper`. The old keys `in_sample_var` /
  `out_sample_var` are removed; no consumer existed.
- **Default output changed?** Yes: every `sp.scpi` interval, plus `se` and `pvalue`.

### D3. `sp.scdata`: silent handling of bad input (minor; fail loudly)

- **What was wrong.** `pivot_table` silently averaged duplicate (unit, time) rows. Donors with
  missing pre-period data were dropped without a message. Missing post-period donor values
  passed straight through.
- **Fix.** Duplicates now raise; dropped donors trigger a warning; missing post-period donor
  values raise. Donors are sorted as R sorts them.
- **Default output changed?** No, on valid balanced panels.

### Documented differences (not defects)

1. **Solver tolerances** (class 3/4; the mechanism is demonstrated in the fixture):
   - **Weights.** CLARABEL stops at a 1e-8 relative gap. The test proves R's point is feasible
     and its objective is ≥ ours, and bounds the weight distance by `sqrt(gap/λmin(B'B))`.
   - **In-sample draws.** R's ECOS defaults are 1e-8. The generator re-runs the same draws with
     ECOS at 1e-12 (`vsig_tight`). Against that run the median per-draw gap drops from ~1e-5
     to ~1e-8, and the bounds agree to 4e-6.
   - **Out-of-sample.** `Qtools::rrq` uses quantreg's Frisch-Newton interior point. The
     generator re-runs scpi with `rrq(method="br")` (exact simplex LP) and gets agreement at
     ≤ 5e-12.
2. **Simplex end-to-end rho / df** (T4, reference solver artefact).
   - R's CLARABEL leaves Norway at 2.0e-6, which is ≥ the 1e-6 threshold `df.EST` and
     `local.geom` use to count active donors. R therefore gets d0 = 7 and df = 6.
   - The exact optimum has Norway at 0, with a strictly positive KKT multiplier. StatsPAI
     gets d0 = 6 and df = 5.
   - As a result R's rho is 8% larger and its HC1 factor differs by 31/25 vs 31/26.
   - Independent evidence: R's own L1-L2 fit reaches the same optimum (its L2 bound is slack),
     leaves Norway at 2.9e-7, and gets StatsPAI's rho to 3e-9.
   - All other constraints agree end-to-end within the solver bounds above.
3. **Simulated bounds under StatsPAI's own RNG** are T3 relative to R's. Only the draw-fed
   comparison is a parity statement.

## 3. Proposed promotion records

```python
    "scdata": {
        "status": "bit-exact",
        "reference": "R scpi::scdata (features = outcome, no cov.adj, constant = FALSE)",
        "reference_versions": {"scpi": "4.0.1", "CVXR": "1.9.2", "ECOSolveR": "0.6.1", "Qtools": "1.6.0", "quantreg": "6.1"},
        "tolerance": "A, B, P matrices identical (0 difference) on scpi_germany; donor order = R sort(B.names).",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_scpi_parity.py"],
        "note": (
            "Added by the phase-3 did_synth scpi sweep, which found sp.scest's lasso / ridge were penalised estimators on standardised data rather than R scpi's norm-constrained weights, and sp.scpi was not the Cattaneo-Feng-Titiunik procedure (subsampling variance + Gaussian PI; intervals 2-4x too narrow on scpi_germany). Both were rewritten as a port of R scpi."
        ),
    },
    "scest": {
        "status": "aligned",
        "reference": "R scpi::scest (w.constr simplex / lasso / ridge / ols / L1-L2, V = 'separate')",
        "reference_versions": {"scpi": "4.0.1", "CVXR": "1.9.2", "clarabel": "0.11.2", "osqp": "1.0.0"},
        "tolerance": (
            "ols and lasso weights at 1e-9 abs; ridge Q / lambda and L1-L2 Q2 at 1e-10. simplex / ridge / L1-L2 weights are bounded by the conic solver: R's objective exceeds StatsPAI's exact optimum by <= 1e-7 relative (CLARABEL gap 1e-8), R's point is feasible, and ||w_R - w_py|| <= sqrt(gap / lambda_min(B'B)) (observed 2.0e-6 / 1.4e-5 / 2.9e-7)."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_scpi_parity.py"],
        "note": (
            "⚠️ correctness fix in the phase-3 sweep: lasso / ridge were penalised coordinate descent / ridge regression with lambda = 1 on standardised data (weights off by up to 0.23 from R); now ||w||_1 <= 1 and ||w||_2 <= Q with R's shrinkage.EST radius. lasso_lambda / ridge_lambda deprecated (ignored with DeprecationWarning); L1-L2 added."
        ),
    },
    "scpi": {
        "status": "aligned",
        "reference": "R scpi::scpi (effect = 'unit-time', u.missp, u.sigma = HC1, u.order = e.order = 1, rho = type-2, e.method = all)",
        "reference_versions": {"scpi": "4.0.1", "CVXR": "1.9.2", "ECOSolveR": "0.6.1", "Qtools": "1.6.0", "quantreg": "6.1"},
        "tolerance": (
            "On R's weights: rho, Q.star, u.mean, Omega, Sigma, e.mean at 1e-9; out-of-sample e.var and gaussian / ls / qreg bounds at 1e-9 against R with rrq(method = 'br') (exact LP) and 5e-4 abs against the default Frisch-Newton rrq. In-sample simulation fed R's draws: per-draw median <= 1e-6 and max <= 2e-4 vs ECOS at 1e-12, quantile bounds at 1e-5; vs default ECOS (1e-8) bounds within 2e-3 abs. Average-effect CI = scdataMulti(effect = 'unit') within 5e-4. J > T0 (California, 38 donors) covered."
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_did_synth_scpi_parity.py"],
        "note": (
            "⚠️ correctness fix in the phase-3 sweep: sp.scpi was a different procedure under the scpi name (subsampling in-sample variance, residual-variance out-of-sample term, Gaussian PI, invented SE / p-value); now a port of R scpi solved exactly (active-set QCQP per draw, exact LP quantile regressions). Simulated bounds with StatsPAI's own RNG are Monte Carlo relative to R (T3); pass draws= to reproduce R. Default simplex end-to-end rho / df differ from R because CLARABEL leaves one donor at 2.0e-6 >= scpi's 1e-6 active threshold (T4; R's own L1-L2 fit of the same optimum reproduces StatsPAI's rho)."
        ),
    },
```

## 4. CHANGELOG / MIGRATION

**⚠️ Correctness**
- `sp.scpi` now implements the Cattaneo-Feng-Titiunik prediction intervals as in R `scpi`
  4.0.1: rho-regularised local geometry, HC-type pseudo-residual variance, per-draw
  constrained simulation, and sub-Gaussian / location-scale / quantile-regression
  out-of-sample bounds plus joint bounds. The previous code was a different, ad hoc procedure,
  and its intervals were 2–4× too narrow on `scpi_germany`. `se` / `pvalue` are now NaN.
  `ci` is the average-effect interval of R's `scdataMulti(effect="unit")`. Nominal coverage
  is `1 - (u_alpha + e_alpha)`.
- `sp.scest(w_constr="lasso"|"ridge")` now bound `||w||_1 ≤ Q` / `||w||_2 ≤ Q` with R's radii.
  They were penalised fits on standardised data.

**Added**
- `sp.scest(..., Q=, Q2=)` and `w_constr="L1-L2"`.
- `sp.scpi(..., sims, u_missp, u_sigma, u_order, u_alpha, e_order, e_alpha, rho, rho_max, Q, Q2, draws)`.
- `model_info["bounds" | "CI" | "rho" | "Sigma" | "u_var" | "e_mean" | "e_var" | "df" | "vsig" | "failed_sims"]`.
- `scdata` now also returns R-named `A`, `B`, `C`, `P`, `J`, `KM`, `T0`, `T1`.

**Fixed**
- `sp.scdata` raises on duplicate (unit, time) rows (it averaged them before), warns when it
  drops donors, and raises on missing post-period donor values.

**Deprecated**
- `lasso_lambda` / `ridge_lambda` in `sp.scest` / `sp.scpi` are ignored, with a
  `DeprecationWarning`.

**MIGRATION rows**

| function | change | old → new |
|---|---|---|
| `sp.scpi` | intervals now from R scpi's procedure | Germany 1991 effect PI [0.230, 0.774] → [-0.731, 1.316] (R: [-0.700, 1.302]); `se`/`pvalue` numeric → NaN; `period_results` loses `in_sample_var`/`out_sample_var` |
| `sp.scest(w_constr="lasso")` | L1 ball ‖w‖₁ ≤ 1, not penalised CD | Germany max \|Δw\| 0.227 |
| `sp.scest(w_constr="ridge")` | L2 ball, Q from `shrinkage.EST` | Germany max \|Δw\| 0.194 |
| `lasso_lambda`, `ridge_lambda` | deprecated, ignored | use `Q=` / `Q2=` |

## 5. Not closed / caveats

- **Stata `scpi`.** Not attempted. Its engine calls Python `scpi_pkg` through Stata's Python
  integration, and `scpi_pkg` is not installed in the venv (checked with
  `importlib.import_module('scpi_pkg')` → ModuleNotFoundError). No return code was recorded,
  so this is an open item, not a verified skip reason.
- **Paths not ported.** They raise `NotImplementedError`, or are not reachable from the API:
  - covariates / `cov.adj` / `constant` / cointegrated data / multiple treated units /
    `V.mat`;
  - `u.lags`, `e.lags`, and u/e order > 1.
- **`cores`** is accepted but the simulation runs serially. The registry `cost_profile` for
  `scpi` says "cores= parallelises it"; that text needs updating (registry.py belongs to the
  lead).
- **Registry / schema.** The `scest` / `scpi` signatures gained arguments, so the lead's
  `dump_schemas` and registry param specs need regeneration.
- **Runtime.** `sp.scpi` with defaults (sims = 200) takes ~10 s on 16 donors and ~3 s on 38
  donors with sims = 50.
- **Fixture size.** `did_synth_scpi_R.json` is 835 KB (draws and per-draw ECOS values at
  17 digits).

## 6. .gitignore

- `tests/reference_parity/_fixtures/_rlib_did_synth_scpi/`: the private R library holding
  CVXR 1.9.2 + highs. scpi 4.0.1 needs CVXR > 1.9, but the site library's CVXR 1.8.2 serves
  the HonestDiD / DiSCos / synthdid fixtures, so it was not upgraded.
