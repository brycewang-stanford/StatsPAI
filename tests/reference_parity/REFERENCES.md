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
