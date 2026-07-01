# Reference Parity — sources and tolerances

This suite does not execute R or Stata in CI (dependency friction) but
validates every StatsPAI estimator against published identification
theory: every DGP is deterministic with a known population parameter,
and estimates must recover truth within a bounded number of standard
errors.

## Tolerance convention

- **4-sigma (`n_sigma=4.0`)**: default for recovery tests.  P(false
  failure under a valid estimator) = 6.3e-5, so flakes are negligible.
  True implementation bias typically exceeds 10 sigma, so the test
  retains full power.
- **Absolute tolerance (`tol=X`)**: used for synth where SEs are
  noisy / not well-defined at the unit-effect level.  Absolute
  tolerance is set by the DGP noise scale.
- **Cross-estimator parity (combined SE)**: for two estimators with
  independent sampling noise, we test `|A - B| <= 4 * sqrt(SE_A^2 + SE_B^2)`.

## DGPs and their population parameters

| DGP fixture | True parameter | Source / derivation |
| --- | --- | --- |
| `did_2x2_data` | ATT = 2.0 | `y = 1 + 0.3t + 0.5T + 2.0*T*t + u + e`; interaction coefficient = 2.0 (Angrist-Pischke MHE Ch. 5). |
| `did_staggered_homogeneous` | ATT = 1.5 | Homogeneous effect; TWFE, CS, SA, Wooldridge target post-period interaction = 1.5. |
| `did_staggered_heterogeneous` | Cohort-specific | Cohort effects 1.0 / 1.5 / 2.0; TWFE biased (de Chaisemartin-D'Haultfoeuille 2020); CS simple-ATT ≈ 1.5. |
| `rd_sharp_data` | ATE@c = 1.0 | Sharp RD with `m_1 - m_0 = 1.0` at `x=0` (Hahn-Todd-van der Klaauw 2001). |
| `rd_fuzzy_data` | LATE = 0.8 | Fuzzy RD, complier share 0.7, outcome jump 0.56, ratio 0.8 (Imbens-Lemieux 2008). |
| `iv_strong_data` | LATE = 1.5 | Binary-Z IV, strong first stage, homogeneous complier effect = 1.5 (Imbens-Angrist 1994). |
| `synth_factor_model_data` | ATT = -5.0 | 2-factor DGP, treated loadings in convex hull of donors; SCM exactly unbiased (Abadie-Diamond-Hainmueller 2010 §3). |
| `matching_cia_data` | ATT = 2.0 | CIA with homogeneous effect; matching / IPW / CBPS / ebalance target ATT = 2.0 (Imbens-Wooldridge 2009). |
| `saturated_data` (`test_ipw_parity.py`, module-scoped) | Closed-form stratified means (sample identity, not a population recovery) | One binary covariate ⇒ logit propensity is saturated, fitted ps = exact cell shares; Hajek/HT ATE/ATT/ATC reduce to stratified-mean identities, tol 1e-9 (Horvitz-Thompson 1952; Hirano-Imbens-Ridder 2003). |
| `randomized_data` (`test_ipw_parity.py`, module-scoped) | ATE = 1.2 | T ⫫ X by construction ⇒ constant population propensity; IPW must collapse to difference-in-means within 4 × combined SE. |
| `overlap_good_data` / `overlap_poor_data` (`test_ipw_parity.py`, module-scoped) | Invariance anchors (no recovery target) | `trim` winsorizes ps via clip: bitwise no-op when all ps ∈ (trim, 1−trim); under poor overlap it caps the max 1/ps weight at exactly 1/trim (trimming motivation: Crump-Hotz-Imbens-Mitnik 2009). |
| `gformula_data` (CSV fixture, `test_gformula_parity.py`) | ATE = 1.2 | Point-exposure linear-additive DGP (`y = 0.5 + 1.2d + 0.8x1 − 0.5x2 + 0.3x3 + e`, logistic confounded D); the default single additive OLS Q is correctly specified, so the g-formula contrast collapses exactly to the OLS treatment coefficient, tol 1e-8 (Robins 1986 g-formula; Hernán-Robins 2020 Ch. 13). |
| `saturated_cell_data` (`test_gformula_parity.py`, module-scoped) | Closed-form standardization (sample identity, not a population recovery) | One binary covariate, saturated Q (cell means): g-formula ATE = `Σ_x (ȳ₁ₓ − ȳ₀ₓ) P̂(X=x)`, ATT uses `P̂(X=x \| D=1)` — hand-computed via groupby, tol 1e-10; additive-OLS value differs by ~0.25, so the anchor is non-tautological (Robins 1986; Snowden-Rose-Mortimer 2011). |
| `saturated_binary_data` (`test_tmle_parity.py`, module-scoped) | TMLE = hand-stratified ATE (sample algebraic identity, not a population recovery) | One binary covariate ⇒ unpenalised-logistic propensity is saturated (fitted g = exact cell shares); the AIPW functional then collapses cell-by-cell to the stratified estimator for *any* within-cell-constant Q, and the targeting step makes the TMLE plug-in equal AIPW — so TMLE = AIPW = g-formula plug-in = stratified ATE (van der Laan-Rubin 2006, doi:10.2202/1557-4679.1043, bib `vanderlaan2006targeted`). Tol 1e-9; worst observed 7.1e-13 over 5 calibration seeds (no cross-fitting: SuperLearner predicts from full-data refits). |
| `tmle_data.csv` (`_fixtures/`, seed 7321, `test_tmle_parity.py`) | SATE (risk difference) ≈ 0.1643 | Binary-outcome confounded DGP (`_generate_tmle_data.py`); truth recomputed in-test from the DGP's potential-outcome probabilities using only the CSV columns. Naive risk difference is ≈ 6 sigma biased; TMLE with the correctly-specified logistic Q recovers within 4 sigma. The same CSV drives the EIF-zero anchor: mean of the efficient influence curve at the TMLE estimate ≤ 1e-8 (bound derived from the Newton stopping rule \|delta\| < 1e-8; observed ≈ 2e-17), with nuisances reconstructed independently via deterministic sklearn refits. |
| TMLE SE-sanity MC DGP (`test_tmle_parity.py`, seeds 9000–9029) | ATE = 1.0 | 30 replications × n=500, linear outcome + logistic propensity; the influence-curve SE must lie within 3x of the Monte-Carlo SD (observed ratio 0.95) and the MC mean must recover 1.0 within 4·SD/√30. |

> **Published certified-value fixtures** (NIST StRD linear regression) live in
> `tests/numerical_accuracy/`, not here — they certify numerical accuracy
> against multiple-precision certified values rather than recover a DGP or align
> with R/Stata, and are deliberately kept out of the JSS reference-parity count.

## Frozen R-value fixtures (true cross-package parity)

Some estimators are pinned to *exact* R numbers (not just DGP recovery).
Each ships a deterministic data CSV, a `_generate_*.R` script that
produces the reference JSON, and a frozen `*_R.json` with the R output.
The test asserts coefficient/SE equality to a tight tolerance.

| Fixture | sp entry point | R reference | Tolerance |
| --- | --- | --- | --- |
| `hdfe_*` | `sp.hdfe_ols` | `fixest::feols` two-way FE + cluster | coef 1e-4, cluster SE 5% |
| `panel_*` | `sp.panel(method='fe'/'re'/'between')` | `plm` within / Swamy-Arora RE / between (Croissant & Millo 2008, *JSS* 27(2), doi:10.18637/jss.v027.i02) | coef 1e-5, classical SE 1e-5, cluster SE 2e-4 |
| `count_quantile_*` | `sp.poisson` / `sp.nbreg` / `sp.qreg` / `sp.tobit` | `glm(poisson)` / `MASS::glm.nb` / `quantreg::rq` / `AER::tobit` | coef 1e-5, model SE 1e-4, qreg coef 1e-4 (SE not pinned) |
| `zeroinfl_*` | `sp.zip_model` / `sp.zinb` | `pscl::zeroinfl` (Zeileis-Kleiber-Jackman 2008, *JSS* 27(8), doi:10.18637/jss.v027.i08) | ZIP coef+SE 1e-4, ZINB coef+theta 1e-3 |
| `sdid_*` | `sp.sdid` | `synthdid` R package (Arkhangelsky et al. 2021, *AER* 111(12):4088-4118, doi:10.1257/aer.20190159) on `sp.california_prop99()` | estimate 1e-6 (SE not pinned — placebo randomisation) |
| `ipw_*` | `sp.ipw` | base R `stats::glm(t ~ x1 + x2, binomial)` + hand-rolled Hajek weighted means (no CRAN packages; valid because `sp.ipw`'s propensity is the unpenalized logit MLE with intercept) | Hajek ATE/ATT estimate 1e-9 (observed ≤ 2e-15; SE not pinned — bootstrap) |
| `gformula_*` | `sp.g_computation` | base R `stats::lm(y ~ d + x1 + x2 + x3)` standardization: `psi = mean(predict(fit, d=1)) − mean(predict(fit, d=0))` (Robins 1986 g-formula; no CRAN packages — model form mirrors the implementation's single additive OLS Q) | psi 1e-8 (observed ≤ 7e-16); bootstrap SE pinned only loosely, ±25% vs R classical OLS SE of `coef d` — bootstrap RNG streams differ across languages, and on this homoskedastic DGP the bootstrap SE targets the same sandwich≈classical quantity |
| `tmle_*` | `sp.tmle` | base R `stats::glm` TMLE (van der Laan & Rubin 2006, *Int. J. Biostat.* 2(1), doi:10.2202/1557-4679.1043): unpenalised logistic Q(A,W) and g(W), g truncated to (0.025, 0.975), fluctuation epsilon via `glm(y ~ -1 + H, offset=qlogis(QA), binomial)`, psi = mean(Q*₁−Q*₀), SE = sd(EIF)/√n — no CRAN packages; valid because `sp.tmle` with single-learner `LogisticRegression(penalty=None)` libraries fits the identical unpenalised MLEs (SuperLearner predicts from full-data refits, weight trivially 1.0) and solves the same 1-D fluctuation score by Newton | psi 1e-9 (observed 5.6e-12), EIF SE 1e-9 (observed 1.3e-12), epsilon 1e-8 (observed 5.8e-11) |

For `panel_*`, the cluster-robust convention is `plm::vcovHC(type="HC1",
cluster="group")`, which `sp.panel(method='fe', cluster=<entity>)`
reproduces. Regenerate with
`Rscript tests/reference_parity/_fixtures/_generate_panel.R`.

## What the tests verify

### Recovery tests

Each estimator is called on the DGP with the published-canonical call
signature.  The test asserts:

```text
|estimate - truth| <= n_sigma * estimated_SE
```

This simultaneously catches:

- **Bias**: a biased estimator will exceed `n_sigma * SE` on a large-N DGP.
- **SE miscalibration**: if SEs are too small, the true estimate
  drifts outside the 4-sigma band.

### Cross-estimator parity tests

On DGPs where two estimators are theoretically equivalent (e.g., CS
with never-treated vs not-yet-treated controls under homogeneity),
the tests assert pairwise agreement within combined SE.  This catches
implementation divergence that doesn't show up in any single recovery
test.

### Sign correctness

For each estimator, we include a test that a positive (or negative)
effect DGP produces a positive (or negative) point estimate.  This
is a smoke test for sign-flip bugs — the most common and
highest-impact regression in estimator code.

### Pinned fixtures (drift guards)

`tests/test_did_numerical_fixtures.py` already contains pinned ATT(g,t)
values for Callaway-Sant'Anna on a fixed-seed DGP.  That file is the
drift guard; this suite is the identification-validity guard.

## How to add a new parity test

1. Add a deterministic DGP to `conftest.py` with a `@pytest.fixture(scope='session')`.
2. Store the population parameter in `df.attrs['true_effect']`.
3. In the test file, call the estimator and assert recovery with
   `_within_n_se(estimate, truth, se, n_sigma=4.0)`.
4. Document the DGP source and derivation in this file.

## High-N analytical parity (tight tolerance)

`test_paper_parity.py` escalates the standard recovery tests from
4-sigma / n~2000 to **2-sigma / n=5000-8000**.  The tolerance is
derived from the closed-form population parameter implied by each
DGP, not from any external R or Stata output.

### DID (homogeneous, staggered)

- Fixture: `dgp_did(n_units=5000, effect=0.8, staggered=True, seed=2026)`.
- Population ATT: **0.8** (set by the DGP; all cohorts, all post-periods).
- Estimators tested: CS2021, Sun-Abraham, Wooldridge — all must
  recover 0.8 within 2 sigma, and agree with each other pairwise
  within 2 * combined SE.

### RD (sharp)

- Fixture: `n=8000, x~Unif(-1,1), y = 2 + 3x + x^2 + 1.0 * (x>=0) + N(0, 0.25)`.
- Population jump at cutoff: **1.0**.
- Estimators tested: `rdrobust` with MSE-RD and CER-RD bandwidths.
- Cross-bandwidth stability test: MSE vs CER estimates agree within
  2 * combined SE.

### IV (Wald = 2SLS algebraic identity)

- Fixture: `n=5000`, binary Z, binary D, no covariates.
- Claim: 2SLS estimate must equal the manual Wald ratio to 1e-8.
- This is an algebraic identity, not a statistical claim — any
  deviation larger than 1e-8 indicates a numerical bug.

### Matching / weighting (CIA)

- Fixture: `n=5000`, binary D, 3 covariates, homogeneous effect = **2.0**.
- Estimators tested: `ebalance`, `cbps(ATT)`, `aipw`.
- All must recover 2.0 within 2 sigma.

### Proximal causal inference family (`proximal`, `fortified_pci`, `bidirectional_pci`)

- **DGP** (`test_proximal_parity.py`, all seeded `np.random.default_rng`):
  unmeasured confounder `U ~ N(0,1)`; proxies `Z = U + N(0,1)`
  (treatment-side) and `W = U + N(0,1)` (outcome-side); measured
  covariate `X ~ N(0,1)`; outcome
  `Y = 1.5*D + 1.0*U + 0.3*X + N(0,1)`, so the **true ATE = `GAMMA_D = 1.5`**
  and naive OLS-on-`(D,X)` inherits `U`'s confounding. Continuous-D
  variant uses `D = 0.8*U + 0.5*X + N(0,1)`; binary-D variant draws
  `D ~ Bernoulli(expit(0.8*U + 0.5*X))`. A separate just-identified
  fixture drops `X` so the 2SLS system is square (`n=2000`).
- **Anchor A (closed-form, tol 1e-9):** with a single `Z`, single `W`
  and no covariates the proximal 2SLS reduces to the just-identified IV
  estimator `(Z'X)^{-1} Z'Y` (instruments `[1,D,Z]`, regressors
  `[1,D,W]`). `sp.proximal`'s coefficient on `D` must equal the
  hand-computed `np.linalg.solve` entry to machine precision (probed
  |diff| ~2e-16). Algebraic identity, not a finiteness check.
- **Anchor B (recovery, 4-sigma):** single-draw `proximal` within 4 of
  its own SE of 1.5 (probed z~1.1); 40-rep Monte-Carlo mean within
  `4*SD/sqrt(40)` of 1.5 (probed 1.5004).
- **Anchor C (naive-bias contrast):** hand-rolled OLS slope of `Y` on
  `(D,X)` is `> 6` sigma above truth (probed ~1.98, z~30); `proximal`
  recovers truth within 4 sigma AND lands strictly below naive by a
  0.10 margin (directional de-confounding).
- **Anchor D (family consistency):** (i) continuous `D` makes
  `bidirectional_pci`'s logistic Z-IPW fail, so it collapses to the
  outcome-bridge 2SLS and equals `sp.proximal` exactly (tol 1e-9,
  probed diff 0.0, `treatment_bridge_fallback` True); (ii) binary `D`
  engages the Z-IPW (`fallback` False) and both `bidirectional_pci` and
  `fortified_pci` land strictly inside the `(truth, naive)` band while
  `proximal` recovers truth — pinning the partial correctors between
  "does nothing" and "fully corrects".
- **Anchor E (orientation):** positive-effect DGPs (continuous and
  binary D) yield positive estimates from all three estimators.
- **Tolerance rationale:** machine-precision (1e-9) for the algebraic
  identity (A) and the fallback collapse (D-i); 4-sigma recovery bands
  (B, C) per the suite convention above; ordering / sign predicates
  (C-margin, D-ii band, E) are non-tautological — a 20% bias breaks them.
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@tchetgen2020introduction`, `@miao2018identifying`,
  `@cui2024semiparametric` (linear-bridge 2SLS implemented here),
  `@yu2025fortified` (fortified PCI), `@min2025regression`
  (bidirectional PCI).

### Principal stratification family (`principal_strat`, `survivor_average_causal_effect`)

- **DGPs** (`test_principal_strat_parity.py`, all seeded
  `np.random.default_rng`): (1) *encouragement* — AIR monotonicity
  (no-defiers) IV DGP with hand-set strata shares (always-takers 0.2,
  never-takers 0.3, compliers 0.5), random `Z`, uptake `D` (`= Z` for
  compliers), and outcome `Y = 1 + level(stratum) + TRUE_LATE*D + N(0,1)`
  with exclusion (the only `D`-driven term is `TRUE_LATE = 2.0`); (2)
  *monotone-strata* — no instrument, always-survivors 0.3 / compliers 0.4
  / never-survivors 0.3, `S = 1[AS] or D[CO]`; (3) *perfect-compliance* —
  `S == D` so everyone is a complier; (4) *survivor-bias* — truncation by
  death with always-survivors 0.5 (`Y(0)~N(5,1.5)`, `Y(1)~N(6,1.5)`, so
  `TRUE_SACE = 1.0`), a "protected" stratum 0.3 observed only under `D=1`
  with `Y~N(5.5,1.5)` contaminating the `(D=1,S=1)` cell, and doomed 0.2.
- **Anchor A (recovery, 4-sigma):** `principal_strat(instrument='z')`
  recovers the Wald LATE on `Y` within 4 bootstrap SE of `TRUE_LATE=2.0`
  (probed z~0.03), AND the first stage `P(D=1|Z=1)-P(D=1|Z=0)` recovers
  the 0.5 complier share within 4 two-proportion sigma (hand-rolled SE,
  probed z~0.7). A +20% bias on `tau_Y` lands ~8 sigma out.
- **Anchor B (closed-form, tol 1e-9):** (i) the monotonicity complier
  LATE equals the hand-computed Wald-mixture
  `(mu_11*p11 - mu_01*p10)/(p11-p10)` of the *sample* cell means (probed
  |diff| 0.0); (ii) perfect compliance forces `pi_complier=1`,
  `pi_always=pi_never=0` (exact) and collapses the LATE to
  `E[Y|D=1,S=1]=mean(Y|D=1)` (probed |diff| 0.0).
- **Anchor C (naive-bias contrast on the SACE):** the naive survivor
  comparison `E[Y|D=1,S=1]-E[Y|D=0,S=1]` is `>4` two-proportion sigma off
  `TRUE_SACE=1.0` (probed z~-8) because the `(D=1,S=1)` cell mixes
  always-survivors with the protected stratum; the Zhang-Rubin sharp
  bounds from `survivor_average_causal_effect` strictly bracket the truth
  (probed `[-0.15, 1.71] ∋ 1.0`). Both halves asserted.
- **Anchor D (internal consistency):** (i) the SACE `CausalResult.estimate`
  equals `(sace_lower+sace_upper)/2` exactly (tol 1e-12, probed 0.0) with
  `sace_lower <= sace_upper`; (ii) the three stratum proportions sum to 1
  exactly (telescoping identity) and the complier share recovers the
  hand-set 0.4 within 4 two-proportion sigma (probed z~1.0).
- **Anchor E (determinism):** the Zhang-Rubin point endpoints carry no
  bootstrap noise (sorted q-slice), so two same-seed
  `survivor_average_causal_effect` calls return bitwise-equal
  `sace_lower`/`sace_upper`/`estimate` — the partial-identification
  analogue of seed-stability.
- **Tolerance rationale:** machine-precision (1e-9 / 1e-12) for the
  closed-form cell-mean identities (B) and the midpoint / telescoping
  identities (D); 4-sigma recovery bands (A, C, D-ii) per the suite
  convention above; the bracket and determinism predicates (C, E) are
  non-tautological — a 20% estimate bias breaks A and B.
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@frangakis2002principal`, `@zhang2003estimation` (Zhang-Rubin sharp
  SACE bounds), `@angrist1996identification` (AIR / Wald LATE under the
  encouragement design), `@ding2017principal`.

### Distributional treatment effects (`distributional_te`, `stochastic_dominance`)

- **DGP** (`test_distributional_te_parity.py`, all seeded
  `np.random.default_rng`): a pure location shift. The DTE arm draws an
  equal-split sample with control `Y0 ~ N(0, 1)` and treated
  `Y1 ~ N(MU, 1)`, `MU = 1.2` hand-set; treatment is assigned
  independently of the outcome shocks, so `distributional_te`'s
  no-covariate IPW propensity is the constant `D.mean() = 0.5` and the
  counterfactual CDF is exactly the control ECDF. The dominance arm
  builds a 6-donor + 1-treated panel (donor levels 1..6, treated
  pre-level 3.5, treatment at `t=5`) fed to `sp.discos`; the
  post-treatment treated series is either a uniform `+3.0` shift (FOSD
  present) or a symmetric straddle (offsets `[-1,-0.5,0,0.5,1]`,
  mean-preserving, crossing).
- **Anchor 1 (quantile shift, 4-sigma):** a location shift moves every
  quantile by `MU`, so the median QTE from `distributional_te` recovers
  `MU = 1.2` within 4 of its bootstrap SE (probed ~1.24, z ~1.0); the
  0.25/0.5/0.75 QTEs all sit within 0.20 of `MU`.
- **Anchor 2 (CDF closed form, abs tol 0.03):** the treated CDF at the
  counterfactual median `y=0` estimates `F_{Y1}(0) = Phi(-MU) = 0.1151`
  (probed |diff| ~4e-4); the counterfactual CDF at `y=0` is ~0.5
  (control N(0,1) median), guarding against an arm swap.
- **Anchor 3 (mean shift = area between CDFs, abs tol 0.10):** the
  Hoeffding / layer-cake identity `E[Y1]-E[Y0] = ∫(F0 - F1) dy` recovers
  `MU` by trapezoidal integration of the two estimated CDFs (probed
  ~1.23; tol ~4 sigma of the mean-difference SE).
- **Anchor 4 (stochastic dominance, two-sided):** a uniform `+3.0` shift
  makes `Y1` first-order dominate `Y0`, so `stochastic_dominance(order=1)`
  returns `dominates=True`, `min_gap>0`, `fraction_positive=1.0` (probed
  min_gap ~2.99) and order-2 also dominates; the mean-preserving
  crossing spread returns `dominates=False` with `min_gap<0<max_gap`
  (probed -0.97, 0.94) — asserting BOTH directions is the
  non-tautological core.
- **Anchor 5 (DiSCo avg-QTE consistency, abs tol 0.10/0.20):** the
  average quantile treatment effect from `sp.discos` recovers the
  hand-set post shift (3.0 for FOSD, ~0 for the mean-preserving spread),
  tying the dominance fixtures' point estimates back to the same truth.
- **Tolerance rationale:** 4-sigma recovery band (anchor 1) per the
  suite convention; absolute tolerances on anchors 2/3/5 are ~2-4x the
  relevant ECDF / mean-difference sampling SD, so each pins the estimate
  to a closed-form normal quantity rather than checking finiteness; the
  dominance predicates (anchor 4) are pure sign / ordering facts. A 20%
  estimate bias breaks anchors 1, 3 and 5 (probed: median QTE 1.49 vs
  band, mean shift 0.91 vs 0.10 tol, DiSCo avg-QTE 3.61 vs 0.10 tol).
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@chernozhukov2013inference` (counterfactual-distribution inference,
  the framework `distributional_te` implements),
  `@gunsilius2023distributional` (Distributional Synthetic Controls, the
  DiSCo estimator feeding `stochastic_dominance`).

### Quantile treatment effects family (`qte`, `qdid`)

- **File:** `test_qte_parity.py`. First numerical anchor for the quantile
  treatment effect estimators `sp.qte` (Firpo 2007: quantile-regression
  and IPW-distribution variants) and `sp.qdid` (Athey & Imbens 2006
  quantile difference-in-differences), which previously had only smoke
  tests.
- **DGP:** the pure location-shift potential-outcome model `Y1 = Y0 +
  delta` with a HAND-SET constant shift. Under a constant shift every
  quantile of the treated distribution sits exactly `delta` above the
  matching control quantile, so the true `QTE(tau) = delta` is flat in
  `tau` (recovery uses `DELTA = 2.0` for the cross-section and a separate
  `DELTA_DID = 2.5` layered on a common `TREND = 1.0` for the four-cell
  `qdid` panel).
- **Anchor A (closed-form exact collapse, 1e-9):** when the treated
  empirical distribution is *exactly* the control ECDF shifted by `delta`
  (same baseline draw, duplicated and shifted), the no-covariate
  distribution-method QTE — uniform IPW weights, so a plain
  empirical-quantile difference — equals `delta` at every `tau` (probed
  ~4e-16). A matching four-cell `qdid` panel whose additive common trend
  cancels in the DID-quantile contrast recovers `delta` to ~7e-16.
- **Anchor B (known-DGP recovery, 4-sigma):** `qte` recovers `DELTA` at
  `tau in {.25,.5,.75}` within 4 bootstrap SE (probed z <2 at n=2000) and
  a 25-rep Monte-Carlo mean of the median QTE lands within `4*SD/sqrt(R)`
  of `DELTA` (probed mean 2.012, band 0.142); `qdid` recovers `DELTA_DID`
  within 4 SE (probed z ~0.2-0.75).
- **Anchor C (homogeneity):** a constant shift induces NO quantile
  heterogeneity, so the across-quantile spread of estimated effects is
  small (`<0.30`; probed <0.22) AND each effect sits within 0.30 of
  `DELTA` — "flat at the right level", not merely flat.
- **Anchor D (cross-method consistency, 0.10):** the two `qte` engines
  (quantile regression vs IPW distribution) agree on the per-quantile
  effects under a homogeneous shift (probed max gap ~0.01), since both
  target the same `QTE(tau)`.
- **Anchor E (orientation):** a strictly positive `delta` yields strictly
  positive estimates from both `qte` methods and `qdid`; a strictly
  negative `delta` flips every sign.
- **Tolerance rationale:** machine-precision (1e-9, ~6 orders over the
  probed ~4e-16 / ~7e-16 float slack) for the empirical-quantile
  identities (A); 4-sigma recovery bands (B) per the suite convention;
  the homogeneity spread / cross-method gap / sign predicates (C, D, E)
  are non-tautological ordering facts. A 20% multiplicative estimate bias
  breaks anchors A, B and C (probed: closed-form dev 0.60 vs 1e-9,
  recovery z up to 7.0, homogeneity dev 0.536 vs 0.30).
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@firpo2007efficient` (efficient semiparametric QTE, the
  quantile-regression / IPW estimators `qte` implements),
  `@athey2006identification` (nonlinear DiD / changes-in-changes, the
  quantile-DiD contrast `qdid` implements).

### Interference family (`spillover`, `network_exposure`, `interference`)

- **DGPs** (`test_interference_parity.py`, all seeded
  `np.random.default_rng`): *partial interference* — units in clusters of
  size 6, peer `exposure` = leave-one-out share of treated cluster-mates,
  outcome `Y = 1 + DIRECT*D + SPILL*exposure + N(0,1)` with hand-set
  **`DIRECT = 1.5`** and **`SPILL = 2.0`**; an i.i.d. variant draws own
  treatment Bernoulli(0.5), and a *correlated* variant lets a cluster-level
  propensity `p ~ U(0.1,0.9)` drive both own and peers' treatment (with
  `SPILL = 3.0`) so `D` and `exposure` are positively correlated. *Network*
  — a degree-2 ring under a Bernoulli(0.3) design,
  `Y = 1 + 2.0*Z + 1.0*1{>=1 treated neighbour} + N(0,1)`, so the AS4
  contrasts target **`DIRECT_NET = 2.0`** and **`SPILL_NET = 1.0`**.
- **Anchor A (closed-form additivity, tol 1e-12):** `spillover` reports
  `total == direct + spillover` and `estimate == total` exactly (the
  estimator literally sums the two scalars; probed |diff| = 0.0), and the
  `detail` table echoes the same three numbers; `network_exposure`'s AS4
  contrasts obey `composite(c11-c00) == direct(c10-c00) +
  spillover_on_treated(c11-c10)` and `composite == mu(c11)-mu(c00)`
  exactly (algebraic identity among the HT means; probed |diff| = 0.0).
- **Anchor B (recovery, 4-sigma / MC band):** (i) `spillover`'s
  exposure-stratified direct effect recovers `DIRECT` within 4 of its
  bootstrap SE on a single draw (probed z ~0.5) and a 40-rep MC mean
  within `4*SD/sqrt(40)` of 1.5 (probed 1.497); (ii) the
  `network_exposure` AS4 direct / spillover contrasts recover
  `DIRECT_NET` / `SPILL_NET` as 24-rep MC means within `4*SD/sqrt(24)`
  (probed 2.10 / 1.07) — the per-draw Aronow-Samii Theorem-1 variance
  bound is too conservative for a single-draw z, so recovery is on the MC
  mean; this leg also exercises the `sp.interference("network_exposure",
  ...)` dispatcher route.
- **Anchor C (naive-bias contrast):** on the correlated DGP the
  SUTVA-ignoring hand-rolled diff-in-means is `> 6` sigma above `DIRECT`
  (probed ~2.03, z ~8.7) because it absorbs the `SPILL`-driven exposure
  contamination; `spillover`'s direct effect recovers truth within 4 sigma
  (probed z ~0.8) AND lands strictly below naive by a 0.10 margin
  (directional de-confounding, both halves asserted).
- **Anchor D (null spillover):** with `SPILL = 0` the spillover effect is
  within 4 SEs of zero (probed z ~0.7), with `SPILL > 0` it is `> 4` SEs
  from zero (probed z ~8.5), and the direct effect recovers `DIRECT`
  under BOTH (invariant to spillover magnitude — rules out leakage into
  the direct slot).
- **Tolerance rationale:** machine-precision (1e-12) for the additivity /
  partition identities (A) — exact float sums of the components, not
  finiteness checks; 4-sigma single-draw and `4*SD/sqrt(R)` MC bands for
  recovery (B) per the suite convention; the naive margin (C) and
  null/non-null SE predicates (D) are non-tautological ordering facts. A
  20% multiplicative estimate bias breaks anchors B, C and D (probed:
  recovery z 4.7, naive-recovery z 4.4, network MC dev 0.52 vs band 0.34,
  null-DGP direct-invariance z 4.9).
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@hudgens2008toward` (partial-interference direct/spillover
  decomposition implemented by `spillover`), `@aronow2017estimating`
  (exposure-mapping Horvitz-Thompson estimator implemented by
  `network_exposure`).

### Causal-discovery family (`pc_algorithm`, `lingam`, `notears`)

- **File:** `test_causal_discovery_parity.py`. First **structure-recovery**
  anchor for the three discovery estimators (previously smoke-only).
  Because these return graphs, the ground truth is the *known DAG* of a
  hand-built linear SEM and the anchors are edge-set precision/recall,
  orientation correctness and seed-stability — not scalar tolerances.
- **DGPs** (all seeded `np.random.default_rng`): (i) **chain**
  `X1 -> X2 -> X3 -> X4`, coef `CHAIN_COEF = 1.5`, low noise — true
  skeleton `{X1-X2, X2-X3, X3-X4}`, true directed `{X1->X2, X2->X3,
  X3->X4}`; a Gaussian variant for PC/skeleton facts and a non-Gaussian
  (cubed-uniform disturbance) variant for LiNGAM, whose identifiability
  needs non-Gaussianity; (ii) **pure collider** `X0 -> X2 <- X1`
  (`X0 ⟂ X1`) — identifiable v-structure; (iii) **fork+collider**
  `X0->X1, X0->X2, X1->X3, X2->X3` for NOTEARS.
- **Anchor A (PC skeleton precision = recall = 1):** the recovered
  undirected skeleton on the Gaussian chain EQUALS the true skeleton as a
  set (probed exact on 8/8 seeds). An extra/missing edge drops a metric
  below 1.0.
- **Anchor B (PC naive-correlation contrast):** `|corr(X1,X4)|` ~0.92 —
  a marginal-correlation edge detector would link X1-X4 — yet PC, by
  conditioning on the mediator X3, leaves X1-X4 ABSENT. Both the strong
  marginal dependence AND the dropped edge are asserted (de-confounding a
  spurious link, not a finiteness check).
- **Anchor C (PC v-structure orientation):** on the pure collider PC
  orients exactly `{X0->X2, X1->X2}` (both into the collider) with no
  spurious X0-X1 edge.
- **Anchor D (LiNGAM directed precision = recall = 1 + coefficients):**
  on the non-Gaussian chain LiNGAM recovers the directed edge set
  `{X1->X2, X2->X3, X3->X4}` and the causal order `[X1,X2,X3,X4]`
  exactly; a 40-rep MC of each `B` coefficient (`adjacency[i,j]` = direct
  effect of j on i) sits within `4*SD/sqrt(40)` of `1.5` (probed means
  ~1.50, SD ~0.01-0.03). A +20% coefficient bias (-> 1.8) is many bands
  out.
- **Anchor E (NOTEARS skeleton recovery + valid DAG):** on the
  fork+collider NOTEARS recovers the undirected skeleton exactly
  (precision = recall = 1) and returns `h(W) = tr(e^{W∘W}) - d ~ 0`
  (probed 0.0, tol 1e-6) — a genuine acyclic graph. NOTEARS is
  deliberately NOT anchored on edge *orientation* (varsortability on
  standardised Gaussian data, Reisach et al. 2021 — out of scope here).
- **Anchor F (seed stability):** PC skeleton, LiNGAM order and NOTEARS
  skeleton are identical across 3 independent seeds — the recovered
  structure is a property of the DGP, not the RNG.
- **Tolerance rationale:** the structure-recovery anchors are exact
  set/orientation identities (precision = recall = 1, no tolerance); the
  coefficient anchor uses a 4-sigma MC band (suite convention); the DAG
  anchor uses 1e-6 on `h(W)` (the augmented Lagrangian targets `h_tol=1e-8`
  then zeroes tiny weights, so probed `h = 0`). A 20% structural/coefficient
  bias breaks A, C, D and the order/coefficient facts (probed: spurious
  edge -> precision 0.75; flipped order/reversed collider fail equality;
  coef 1.8 vs band ~0.013).
- **References:** `@spirtes2000causation` (PC) and `@zheng2018dags`
  (NOTEARS) are grep-confirmed in `paper.bib`. DirectLiNGAM (Shimizu
  et al., JMLR 12, 2011) has no bib key in `paper.bib`, so the method is
  named without a fabricated citation per CLAUDE.md §10.

### Matrix-completion causal panel family (`matrix_completion`, `mc_panel`)

- **File:** `test_matrix_completion_parity.py`. First numerical anchor
  for `sp.matrix_completion` (article alias, `d` -> `treat`) and
  `sp.mc_panel`, both routing to `MCPanel.fit` (soft-imputed
  nuclear-norm completion, Athey et al. 2021), which previously had only
  smoke tests.
- **DGPs:** (1) *pure completion* — a KNOWN rank-2 matrix
  `M = a⊗b + c⊗e` (exact rank 2; singular values ~`18.6, 15.6, 0, ...`)
  observed as `Y = M + noise`, with ~30% of cells HELD OUT by labelling
  them `treat == 1`. Held-out cells never enter the control mask
  `Omega` (mc_panel.py:234), so `model_info['completed_matrix']` is the
  estimator's reconstruction of `M` on those cells. (2) *causal panel* —
  a staggered treated/control panel whose control surface is a known
  rank-2 trend `M = level + loading*ramp`; the last 6 units get a
  hand-set additive `TAU = 3.0` from period `T0` on, so the true
  counterfactual is `M` and `ATT == TAU`.
- **Anchor A (known-rank-2 recovery, noise floor 0.05):** with
  `max_rank=2` and a small `lambda`, the relative Frobenius error
  `||L - M||_F / ||M||_F` on the held-out cells is below 0.05 (probed
  ~0.017 at noise sd 0.02) — pins the reconstruction to the hand-set
  `M`, not finiteness.
- **Anchor B (noiseless near-exact collapse, rtol 1e-2):** with `Y = M`
  exactly and `lambda=1e-4`, the masked relative Frobenius error
  collapses to ~1e-5, a near-closed-form recovery (~3 orders under
  1e-2).
- **Anchor C (singular-value gap / rank-2 recovery):** run with `lambda`
  ONLY (no `max_rank`), so rank must EMERGE from thresholding. At the
  converged fixed point the data matrix's 3rd singular value is a
  genuine nonzero (probed ~0.78; the raw zero-filled data has it at
  ~5.1) that `lambda=1.0` drives to exactly 0, leaving
  `effective_rank == 2` and `s[2]/s[1] == 0` (1e-12). NOT a tautology of
  `max_rank`.
- **Anchor D (causal ATT recovery + naive-bias contrast):** `sp.mc_panel`
  recovers `TAU = 3.0` within 4 bootstrap SE (probed z ~0.2) AND a
  trend-ignoring naive pre/post contrast on the treated units is biased
  high by > 1.5 (probed ~6.7 vs 3.0); both are asserted, and mc_panel
  lands strictly below the naive value by > 1.0 (directional
  de-confounding). `sp.matrix_completion(d=...)` returns the same
  estimate to 1e-12 (alias-mapping pin).
- **Anchor E (determinism / seed-stability):** `random_state` pins the
  bootstrap, so two identical calls return bitwise-equal `estimate` and
  `se` (probed diff 0.0).
- **Tolerance rationale:** noise-floor 0.05 (A, ~3x over probed 0.017)
  and rtol 1e-2 (B, ~3 orders over probed 1e-5) for Frobenius recovery;
  4-sigma recovery band (D) per the suite convention; the rank gap,
  alias identity and determinism are exact (1e-12). A 20% multiplicative
  estimate bias breaks anchors A and D (probed: rel-Fro 0.19 vs 0.05;
  ATT z ~15 vs 4).
- **References (bib key grep-confirmed in `paper.bib`):**
  `@athey2021matrix` (Matrix Completion Methods for Causal Panel Data
  Models, the estimator `mc_panel` implements).

### Bunching family (`bunching`, `general_bunching`)

`tests/reference_parity/test_bunching_parity.py` — first numerical
anchor for the bunching estimators (previously smoke-only). Bunching
bins the running variable around a policy threshold, fits a
*counterfactual* polynomial to the density EXCLUDING a bunching region,
and reads off the excess mass = observed - counterfactual there. Every
anchor pins the estimate to a hand-set integer excess or a hand-set
elasticity, never to finiteness.

- **Anchor A (closed-form excess-mass integral, tol 1e-9):** the
  histogram is built DETERMINISTICALLY (points placed at bin centres) so
  per-bin counts follow a polynomial I control — first flat, then linear
  `a + b*center`. With exactly `EXCESS` extra points in one in-region
  bin, np.polyfit reproduces the counterfactual exactly and
  `model_info['excess_mass_raw']` equals the integer I planted while
  `counterfactual_at_threshold` equals the polynomial intercept
  (observed |diff| = 0.0). The linear variant pins the in-region
  counterfactual INTEGRAL (sum over the four in-region bin centres),
  computed by hand, not merely a constant. For `general_bunching` a flat
  density pins the elasticity to
  `EXCESS / (n * f_at * bandwidth^2)` (observed ~5e-17).
- **Anchor B (known-DGP recovery, rtol 0.10):** smooth uniform base +
  `N_EXTRA=2000` planted bunchers inside the default region —
  `excess_mass_raw` recovers `N_EXTRA` within 10% (probed max rel error
  4.9% over 6 seeds). `general_bunching` recovers a strongly nonzero
  corrected elasticity at |z| ~25. A reported kink `elasticity` (when
  `dt` is given) is pinned finite and positive.
- **Anchor C (null: smooth density, NO notch):** with no planted
  bunchers the normalised excess / corrected elasticity is within 4 SE
  of zero (probed |z| < 1) — the required contrast to B; a method that
  fabricated mass would fail it.
- **Anchor D (internal-consistency identity, tol 1e-9):** the reported
  normalised B equals `excess_mass_raw / counterfactual_at_threshold`
  exactly (bunching.py:242), and `general_bunching`'s naive (order-2)
  and bias-corrected elasticities coincide on a flat counterfactual
  where higher-order terms vanish.
- **Tolerance rationale:** 1e-9 for the closed-form integral and the
  normalisation identity (exact in arithmetic; observed 0.0 / ~5e-17);
  rtol 0.10 for stochastic recovery (~2x over probed 4.9%); 4-sigma
  band for the null. A 20% multiplicative estimate bias breaks anchors A
  and B (probed: closed-form excess off by exactly 20%; recovery rel
  error 0.215 vs 0.10).
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@kleven2013using` (Using Notches to Uncover Optimization Frictions
  and Structural Elasticities) and `@chetty2011adjustment` (Adjustment
  Costs, Firm Responses, and Micro vs. Macro Labor Supply Elasticities)
  — the kink/notch bunching framework these estimators implement.

### Dynamic-panel GMM family (`xtabond`, `gmm`)

`tests/reference_parity/test_gmm_dynamic_panel_parity.py` — first
numerical anchor for the Arellano-Bond dynamic-panel GMM estimator
`sp.xtabond` and the generic moment-condition GMM `sp.gmm` (both
previously smoke-only). The DGP is the canonical Arellano-Bond panel
`y_it = RHO*y_{i,t-1} + BETA*x_it + alpha_i + e_it` with HAND-SET
`RHO = 0.5`, `BETA = 1.0`, a unit fixed effect `alpha_i` correlated with
the lagged dependent variable, a strictly-exogenous `x_it`, and a long
burn-in for near-stationarity (every draw seeded via `default_rng`).

- **Anchor A (known-DGP recovery, 4-sigma).** Over a 20-panel
  Monte-Carlo bank (N=250, T=7) the mean `xtabond` `rho_hat` recovers
  `RHO = 0.5` within `4·SD/√reps` (probed MC mean 0.504, band ~0.032;
  AB-GMM is essentially unbiased at this T); separately a single larger
  draw (N=400, T=7) recovers `BETA = 1.0` within 4 of its reported SE
  (probed `beta_hat` ~0.966, SE ~0.026).
- **Anchor B (Nickell-bias contrast).** The within-group (LSDV)
  estimator of `RHO` is biased DOWN at small T (Nickell 1981): across an
  8-seed bank its rho is asserted strictly below truth by a 0.10 margin
  (probed ~0.33, max 0.357). The test then asserts BOTH that AB-GMM
  recovers `RHO` within 4 sigma AND that it lands strictly above the
  within-group estimate on the same panel — directional de-biasing, not
  mere execution.
- **Anchor C (cross-method consistency: `sp.gmm` == `sp.xtabond`, rtol/atol
  1e-5).** The Arellano-Bond first-differenced moment system (regressors
  `[Δy_{t-1}, Δx]`, block-diagonal lagged-level instruments plus the `Δx`
  standard instrument, one-step weight `A = (Σ_i Z_i' H Z_i)^-1` with the
  MA(1) `_ab_H`) is rebuilt by hand and minimised through the *generic*
  `sp.gmm` one-step path with that `W = A`. Its `(rho, beta)` must equal
  `xtabond`'s coefficient table; probed max|diff| ~3e-7 (limited by BFGS
  `gtol=1e-8` vs xtabond's closed-form solve), so 1e-5 gives ~30x
  headroom while staying far tighter than any meaningful coefficient gap.
- **Anchor D (orientation).** `RHO > 0` and `BETA > 0` ⇒ both the
  lagged-Y and `x` coefficients come back positive.
- **Tolerance rationale.** 4-sigma recovery band (A, B) per the suite
  convention; `rtol/atol = 1e-5` for the GMM-vs-xtabond identity (C). A
  +20% multiplicative estimate bias breaks anchors A (MC mean →~0.605,
  ~3.3 band-widths out; BETA →1.2, ~8 sigma) and C (coefficients diverge
  by exactly 20% >> 1e-5) — confirmed by an adversarial mutation run
  (4 of 6 tests fail under injection).
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@arellano1991some` (Some Tests of Specification for Panel Data — the
  estimator `xtabond` implements), `@blundell1998initial`,
  `@roodman2009xtabond`, and `@hansen1982large` (Large Sample Properties
  of GMM Estimators — the framework `sp.gmm` implements). The Nickell
  small-T bias (Nickell 1981, *Econometrica* 49(6)) is named without a
  bib key — no entry in `paper.bib`.

### Continuous-treatment dose-response family (`dose_response`, `continuous_did`)

`tests/reference_parity/test_dose_response_parity.py` gives the GPS
dose-response curve (`sp.dose_response`, Hirano-Imbens generalized
propensity score) and continuous-treatment DiD (`sp.continuous_did`)
their first numerical anchor — both previously had only smoke tests.
Two DGP shapes drive everything: a linear-Gaussian cross-section
`Y = BETA*D + g*X + noise` (population dose-response is a line of slope
`BETA = 0.8`), and a two-period continuous-dose panel whose post-period
gain is `TAU*D` (`TAU = 0.5`).

- **Anchor A (recovery):** with the dose independent of `X`
  (unconfounded), `sp.dose_response`'s `avg_marginal_effect` recovers
  `BETA` within abs tol 0.10 (probed dev 0.019); the headline
  `effect_25_to_75` recovers `BETA*(d75-d25)` within 4 sigma.
- **Anchor B (closed-form / internal consistency):** with linear
  treatment/outcome models the curve is a straight line, so
  `effect_25_to_75` equals `curve_slope*(dose_75-dose_25)` read off that
  same curve, abs tol 5e-2 (probed |diff| ~8e-3 — `np.gradient` endpoint
  slack + the Gaussian-pdf GPS column).
- **Anchor C (naive-bias contrast):** on the X-confounded DGP the naive
  `Y~D` OLS slope is biased high (~1.77 vs truth 0.80) while the
  flexible-GBM GPS marginal effect lands STRICTLY between truth and the
  naive slope (~0.91) — directional de-confounding, both asserted.
- **Anchor D (recovery):** `continuous_did(method='twfe')`'s `dose*post`
  coefficient recovers `TAU` within 4 sigma (probed z ~-1.6).
- **Anchor E (naive-bias contrast):** with a unit fixed effect
  correlated with dose, a cross-section regression of post-period `Y` on
  dose is ~94 sigma biased high (~3.50 vs 0.50); the DiD differences the
  FE away and recovers `TAU` within 4 sigma (probed z ~-0.1), both
  asserted plus the DiD slope < naive - 1.0.
- **Anchor F (cross-method consistency):** `method='cgs'` ties the same
  hand-set `TAU` through two estimands — the level ATT over the treated
  support equals `TAU*E[D|D>0]` (2.72 vs 2.77, 4 sigma) and the
  `acrt_overall` derivative recovers the slope `TAU` (0.506, SE 0.015).
- **Anchor G (determinism / seed-stability):** both estimators pin the
  bootstrap on an explicit seed; two identical calls return
  bitwise-equal `estimate` and `se` (probed diff 0.0).
- **Tolerance rationale:** abs tol 0.10 on the GPS marginal (A, ~5x over
  probed 0.019) and 5e-2 on the curve identity (B, ~6x over probed
  8e-3); 4-sigma recovery bands (A-iqr, D, E, F) per the suite
  convention. A 20% multiplicative estimate bias breaks anchors A
  (marginal dev 0.14 > 0.10), C and E (estimate exceeds the naive
  ceiling / falls outside the band), D (twfe z ~20 vs 4) and F
  (cgs-level z ~4.9 vs 4).
- **References (bib keys grep-confirmed in `paper.bib`):**
  `@hirano2004propensity` (The Propensity Score with Continuous
  Treatments — the GPS curve `dose_response` implements),
  `@kennedy2017parametric` (doubly-robust continuous-treatment effects),
  `@callaway2024difference` (Difference-in-Differences with a Continuous
  Treatment — target of `method='cgs'`) and `@dechaisemartin2018fuzzy`
  (Fuzzy Differences-in-Differences).

## External parity (offline procedure)

For true cross-package numerical parity with R/Stata, run the
following offline and record results in a comment block inside
the relevant test file:

```r
library(did)
data(mpdta)
r <- att_gt(yname="lemp", tname="year", idname="countyreal",
            gname="first.treat", data=mpdta, control_group="nevertreated")
summary(r)
# Record r$att[1] etc. as EXPECTED_ATT_GT in the test file.
```

```stata
use https://users.nber.org/~rdehejia/data/nsw_dw.dta, clear
teffects ipw (re78) (treat age education), atet
matrix list r(b)
```

The offline values are pinned as constants with a comment citing the
command that produced them.  This gives the strong external-parity
guarantee without requiring R/Stata in CI.

## feols(vce="wild") vs Stata reghdfe + boottest

`tests/reference_parity/test_feols_wild_boottest_parity.py` pins
`sp.feols(..., vce="wild")` to Stata's `boottest` (David Roodman), the canonical
wild cluster bootstrap implementation. Produced on a fixed 600-obs / 15-cluster
panel with Stata 18 MP, `reghdfe` 6.13.1, `boottest` 4.5.3:

```stata
import delimited wild_parity.csv, clear
reghdfe y x z, absorb(firm) vce(cluster firm)   // _b[x]=.06576571  _se[x]=.02719398
boottest x, reps(99999) weighttype(rademacher) nograph
// with 15 clusters, boottest enumerates all 2^15=32768 Rademacher draws (exact)
// r(p)=.02630615   r(CI)=[.0089431, .12568656]
```

StatsPAI `feols(vce="wild", cluster="firm", wild_reps=99999, seed=12345)`:
coef=0.06576571, CRV1 SE=0.02719398 (both match reghdfe to ~1e-9), wild
p=0.026530 (vs boottest's exact 0.026306 — Monte-Carlo agreement). The data
generator is inlined in the test so the pinned Stata values apply without
R/Stata in CI.

## ivreg(vce="wild") vs Stata ivreg2 + boottest (WRE)

`tests/reference_parity/test_iv_wild_boottest_parity.py` pins
`sp.ivreg(vce="wild")` — the WRE (wild restricted efficient) bootstrap of
Davidson-MacKinnon (2010) — to Stata `boottest` after `ivreg2`. Produced with
Stata 18 MP, `ivreg2` 04.1.12, `boottest` 4.5.3:

```stata
* strong instruments (F~284), 600 obs / 20 clusters
ivreg2 y w (d = z1 z2), cluster(firm)        // _b[d]=.07075778
boottest d, reps(99999) weighttype(rademacher) nograph   // r(p)=.20155202

* weak instruments (F~4), 400 obs / 16 clusters  (discriminates efficient RF)
ivreg2 y (d = z1), cluster(firm)             // _b[d]=.19650046
boottest d, reps(99999) weighttype(rademacher) nograph   // r(p)=.34120178
```

StatsPAI `ivreg(vce="wild", cluster="firm", wild_reps=99999)`: coef matches
ivreg2 to ~1e-9; the WRE p-value matches boottest to Monte-Carlo error
(strong 0.2016 vs 0.20155; weak 0.3415 vs 0.3412). The weak-IV panel selects
the *efficient* reduced form (the naive variant gives 0.426), confirming the
implementation matches boottest's WRE rather than a simpler approximation.

## Multi-endogenous WRE + two-way IV cluster

`test_iv_wild_boottest_parity.py` also pins:

* **Two endogenous regressors** (700 obs / 22 clusters): `ivreg2 y w (d1 d2 =
  z1 z2 z3), cluster(firm)` gives `_b[d1]=.09126668`, `_b[d2]=-.16573432`;
  `boottest d1/d2` gives `r(p)=.21079211` / `.01408014`. StatsPAI WRE (99999
  reps): coefs match to ~1e-8, p = 0.2101 / 0.0151 (Monte-Carlo agreement).
* **Two-way IV cluster** (800 obs / 25×18 clusters): `ivreg2 y w (d = z1 z2),
  cluster(firm year) small` gives `_b[d]=.31606801`, `_se[d]=.0519819`.
  `sp.ivreg(..., cluster=["firm","year"])` matches both exactly (the
  `(G_min/(G_min-1))*((n-1)/(n-k))` finite-sample factor equals ivreg2 `small`).

## IV CR2 / CR3 vs R clubSandwich

`test_iv_wild_boottest_parity.py` pins `ivreg(vce="CR2"/"CR3")` to
`clubSandwich::vcovCR(ivreg(...), type=...)` (R 4.5, clubSandwich + AER):

```r
library(AER); library(clubSandwich)
m <- ivreg(y ~ w + d | w + z1 + z2, data=df)     # strong panel
sqrt(diag(vcovCR(m, cluster=df$firm, type="CR2")))["d"]  # 0.05302589
sqrt(diag(vcovCR(m, cluster=df$firm, type="CR3")))["d"]  # 0.05482302
```

`sp.ivreg(..., vce="CR2"/"CR3")` matches to machine precision on both the
strong (0.05302589 / 0.05482302) and weak (0.31924312 / 0.33575834) panels.
The adjustment is A_g = (I - H_g)^{-p} on the projected 2SLS regressors
(p = 1/2 for CR2, p = 1 for CR3), with H_g = Xhat_g (Xhat'X)^{-1} Xhat_g'.
