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
