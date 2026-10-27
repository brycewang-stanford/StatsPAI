# Line `smm_matching`: sp.smm and sp.choo_siow (AER 116(10), Oct 2026)

Motivation: Goldman, Gracie and Porter estimate a spatial Choo-Siow marriage
model by simulated method of moments and run a "no segregation" equilibrium
counterfactual. StatsPAI had `sp.gmm` and `structural/` (BLP, production
functions, path analysis) but no generic SMM and no matching-market model.

## What was added

| Function | File | Returns |
| --- | --- | --- |
| `sp.smm(simulate, data_moments, theta0, *, n_sim=10, W='optimal', bounds=None, seed=None, data_moment_cov=None, n_obs=None, sim_n_obs=None, vcov='iid', cluster=None, hac_bandwidth=None, jac_step=1e-5, optimizer='L-BFGS-B', maxiter=2000, tol=1e-10, param_names=None, moment_names=None, alpha=0.05)` | `src/statspai/gmm/smm.py`, helpers in `gmm/_smm_core.py` | `EconometricResults` |
| `sp.choo_siow(data=None, *, man=None, woman=None, mu=None, singles_men=None, singles_women=None, basis=None, alpha=0.05, tol=1e-12, maxiter=200)` | `src/statspai/structural/choo_siow.py`, kernels in `_choo_siow_core.py`, results in `_choo_siow_result.py` | `ChooSiowResult` |
| `ChooSiowResult.counterfactual(new_margins=None, *, phi=None, tol=1e-12, maxiter=100000)` | same | `MatchingEquilibrium` |

`sp.smm` is exported eagerly with `sp.gmm`; `choo_siow`, `ChooSiowResult`,
`MatchingEquilibrium` lazily with the rest of `structural`. Both are
registered with full `ParamSpec` lists (`registry.py`, before `markup`).

### Why `sp.smm` is not `sp.gmm(method='smm')`

`sp.gmm` takes a per-observation moment function and estimates the moment
covariance from its rows at the estimate; its `method` argument already
means the weight-updating scheme (one-step / two-step / iterated / CUE). In
SMM the data enter only through moments that do not depend on theta, the
model only through a simulator, and the variance has a separate simulation
part. Folding it in would give `moment_fn` two incompatible meanings. The
two share the moment-covariance estimator (`general_gmm._omega`, iid / HAC /
cluster, centred) and `_safe_inv`, imported rather than copied;
`general_gmm.py` itself is unchanged.

### SMM conventions

* Common random numbers: one child `SeedSequence` per replication, a fresh
  `Generator` built from it at every evaluation. The simulator is called
  twice at `theta0` with the same seed and must return identical output
  (catches use of the global RNG).
* Simulation correction `c = 1 + N / (S N_s)`; `1 + 1/S` when `N_s = N`.
  `N_s` is read from the simulator's row count when it returns
  per-observation contributions, else `sim_n_obs`, else assumed `= N`
  (recorded). Valid for draws independent of the data, reused across
  theta, and a correctly specified model.
* Variance `(D'WD)^-1 D'W (c Sigma) W D (D'WD)^-1`; J is
  `g'(c Sigma)^-1 g`, reported only at `W='optimal'`. The weighting matrix is `W=`, as in
  `sp.gmm` (`weight` is the house-style alias of observation `weights`).
* Covariance source precedence: per-observation data rows > supplied
  `data_moment_cov` > simulated at the step-one estimate (needs `n_obs`;
  the only route where two steps are needed, because a data-based `Sigma`
  does not depend on theta).
* Jacobian: central differences, `h_j = jac_step * max(1, |theta_j|)`,
  one-sided at a bound. A flat column (frequency simulator) raises
  `NumericalInstability`.
* Optimiser, then a projected Gauss-Newton finish certified by the
  Gauss-Newton decrement (`<= 1e-14 f`, or the objective's rounding floor);
  `converged`, `optimizer_message`, `gradient_norm`, `at_bound`,
  `n_simulator_calls` are in `diagnostics`, and failure warns
  (`ConvergenceWarning`), never silently.

### Choo-Siow conventions

* Nonparametric `Phi_xy = log(mu_xy^2 / (mu_x0 mu_0y))`; Choo and Siow's
  "total gains" are `Phi / 2`. Delta-method covariance under multinomial
  sampling of households is
  `4[xy = x'y']/mu_xy + [x = x']/mu_x0 + [y = y']/mu_0y` (the `-11'/H`
  term vanishes because each row of the log-count contrast sums to zero).
* Parametric: weighted Poisson GLM with two-way fixed effects (weights 2 on
  couples, 1 on singles), Newton on the convex dual, stopping on the Newton
  decrement. Sandwich SEs with a Poisson meat; the multinomial `-cc'/H`
  term moves only `(u, v) -> (u - 1, v - 1)`, so the beta block equals
  cupid's multinomial one (checked to 1e-11).
* Zero couple cell: nonparametric `-inf`, SE NaN, `AssumptionWarning`; the
  parametric fit uses it. Zero singles of a type: `DataInsufficient`.
* IPFP for the equilibrium; closed-form positive root written without
  cancellation; stops on the relative margin error (default 1e-12).

## Evidence

| Claim | Test | Tolerance | Observed |
| --- | --- | --- | --- |
| SMM with standardised draws = method of moments | `test_smm.py::test_standardised_draws_reproduce_gmm_times_sqrt_c` (S = 1, 5, 20) | 1e-10 abs | ~4e-15 |
| SMM SE = `sp.gmm` SE x sqrt(1 + 1/S) (identity) | same | 1e-6 rel | 1e-10 |
| Same ratio with iid draws (approximation O((S N)^-1/2)) | `test_iid_draws_se_ratio_approaches_sqrt_c` (S = 50, N = 4000) | 2% | 0.13% |
| Exactly identified: identity weight = optimal weight | `test_identity_weight_same_as_optimal_when_exactly_identified` | 1e-6 | |
| Supplied covariance = covariance from rows | `test_supplied_covariance_equals_data_rows` | 1e-10 | |
| AR(1): SMM (autocovariances, HAC) vs conditional MLE | `test_ar1_smm_agrees_with_mle` | < 1 SMM SE | 0.03 / 0.06 SE |
| J size, S = 1 (S-class screen, R = 200, N = 500) | `test_j_size_screen_with_and_without_simulation_correction` | corrected in [2%, 10%]; uncorrected > 11% | 6.0% / 17.0% (R = 300, N = 1000: 5.7% / 19.0%) |
| Choo-Siow beta and covariance vs cupid_matching 1.3 (Newton-Cholesky row) | `reference_parity/test_choo_siow_cupid_parity.py` | 1e-9 rel | 9e-12 / 1.2e-11 |
| cupid as shipped (L-BFGS) | same | 1e-4 abs | 3.3e-5; cupid's FOC residual 4.8e-6 vs ours 1.2e-15 |
| IPFP vs cupid `ipfp_homoskedastic_solver` | same | 1e-10 rel | 5e-13 |
| Saturated basis = nonparametric (estimate, covariance) | `test_choo_siow.py::test_saturated_basis_equals_nonparametric` | 1e-10 / 1e-9 | 1e-15 |
| Margins and moment matching at the parametric fit | `test_parametric_first_order_conditions` | 1e-12 rel | 1e-13 |
| Model-generated table returns its own Phi / beta | `test_model_generated_table_returns_its_surplus` | 1e-12 / 1e-10 | |
| IPFP fixed point | `test_ipfp_fixed_point_and_margins` | 1e-13 | |
| Gumbel large market (LP, 5000+ per type) recovers Phi | `test_gumbel_large_market_recovers_phi` | |z| < 3.5, |err| < 0.15 | max |z| 1.34, max |err| 0.083 |
| LP market = brute-force assignment | `test_market_lp_equals_brute_force_assignment` | 1e-12 rel | |

The cupid fixture is regenerated by
`tests/reference_parity/_generate_choo_siow_cupid.py` in a scratch venv
(cupid_matching 1.3 is MIT; it pins numpy < 2, so it is not a dependency).
Its `poisson_glm_newton` row runs cupid's own function with scikit-learn's
solver swapped to Newton-Cholesky; nothing else in cupid is changed.

Evidence tier: Choo-Siow parametric estimator and IPFP are T2-style strict
parity against the authors' reference package (Python, not R / Stata, so it
is not a Track A row). The SMM evidence is T1 (identities, known truth)
plus one S-class screen; there is no reference implementation in R or Stata
to align to.

## Citations added to paper.bib (verified 2026-10-28)

`mcfadden1989method`, `pakes1989simulation`, `lee1991simulation`,
`choo2006who`, `galichon2022cupid`, `galichon2024estimating`. Each checked
against the Crossref REST record, OpenAlex and RePEc IDEAS (pages from
RePEc where Crossref has only the first page); `galichon2024estimating`'s
method against the arXiv 2204.00362 abstract and the cupid source. Synced
to `src/statspai/paper.bib`.

## Proposed CHANGELOG entry (Unreleased, Added)

```
### Added
- `sp.smm`: simulated method of moments with common random numbers held
  fixed across theta (seeded, reproducible from `model_info['seed_entropy']`),
  efficient weighting, sandwich standard errors with the simulation
  correction `1 + N/(S N_s)` (`1 + 1/S` for equal sizes), a
  finite-difference Jacobian with a documented step, Hansen's J with the
  same correction, and convergence diagnostics in the result. Shares the
  moment-covariance estimators of `sp.gmm`. Evidence: standard errors equal
  `sp.gmm`'s times sqrt(1 + 1/S) to 1e-10 under standardised draws; AR(1)
  SMM agrees with the MLE; seeded J-size screen. Refs verified via Crossref,
  OpenAlex, RePEc.
- `sp.choo_siow`: Choo-Siow transferable-utility marriage-market model from
  household data or a couples table with singles. Nonparametric joint
  surplus with delta-method standard errors; parametric surplus by the
  weighted Poisson GLM of Galichon and Salanie (2024); `.counterfactual()`
  re-solves the equilibrium by IPFP for new type margins or a new surplus.
  Parity with cupid_matching 1.3 to 1e-11 (beta and covariance) and 5e-13
  (IPFP); a Gumbel large-market simulation recovers the surplus. Refs
  verified via Crossref, OpenAlex, RePEc, arXiv.
```

## Open items

* SMM: no reference implementation in R or Stata was run (none is
  canonical); a probit by SMM was not used as known truth because a
  frequency simulator has no usable derivative under common random numbers
  (caught by a test) and a smoothed simulator is biased by construction.
* SMM: `vcov='cluster'` is only available for data rows, not for a
  simulated covariance. No bootstrap option.
* SMM: the simulated-covariance route uses `Sigma(theta_step1)` for the
  variance too (standard two-step), not re-simulated at the final estimate.
* Choo-Siow: homoskedastic only (no gender- or type-specific scale, no
  nested logit, no "no singles" variant; cupid has all three). No survey
  weights. No standard errors for counterfactual equilibria (delta method
  through IPFP would need the implicit-function Jacobian). No spatial
  version (the AER paper's model is spatial; that is the SMM user's
  simulator).
* Choo-Siow minimum-distance estimator of Galichon and Salanie (2024) is not
  implemented (the GLM covers the semilinear Choo-Siow case).
* Generated artefacts not regenerated on this line (per instructions):
  schemas, parity index, `docs/stats.md`, README counts, evidence inventory,
  source failure modes, trace ledgers. `statspai/__init__.py` changed, so
  the ten Track A traces that hash it and one original-data trace go stale.
