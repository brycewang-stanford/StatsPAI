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

| DGP fixture                       | True parameter  | Source / derivation |
|-----------------------------------|-----------------|---------------------|
| `did_2x2_data`                    | ATT = 2.0       | `y = 1 + 0.3t + 0.5T + 2.0*T*t + u + e`; direct reading of the interaction coefficient gives ATT = 2.0 (Angrist-Pischke MHE Ch. 5). |
| `did_staggered_homogeneous`       | ATT = 1.5       | Homogeneous effect (cohort- and time-invariant) so TWFE, CS, SA, Wooldridge all target the same estimand = post-period interaction coefficient = 1.5. |
| `did_staggered_heterogeneous`     | Cohort-specific | Cohort effects 1.0 / 1.5 / 2.0; TWFE is biased (de Chaisemartin & D'Haultfoeuille 2020); CS-unit-uniform simple ATT is cohort-weighted mean ≈ 1.5. |
| `rd_sharp_data`                   | ATE@c = 1.0     | Sharp RD with `m_1 - m_0 = 1.0` at `x=0` (Hahn-Todd-van der Klaauw 2001). |
| `rd_fuzzy_data`                   | LATE = 0.8      | Fuzzy RD with complier share = 0.7; outcome jump = 0.56; ratio = 0.8 (Imbens-Lemieux 2008). |
| `iv_strong_data`                  | LATE = 1.5      | Binary-Z IV with strong first stage and homogeneous complier effect = 1.5 (Imbens-Angrist 1994). |
| `synth_factor_model_data`         | ATT = -5.0      | 2-factor DGP with treated loadings in the convex hull of donor loadings; SCM is exactly unbiased in the population (Abadie-Diamond-Hainmueller 2010 §3). |
| `matching_cia_data`               | ATT = 2.0       | Selection on observables with homogeneous effect; CIA matching / IPW / CBPS / ebalance all target ATT = 2.0 (Imbens-Wooldridge 2009). |

## What the tests verify

### Recovery tests
Each estimator is called on the DGP with the published-canonical call
signature.  The test asserts:
```
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

## External parity (offline procedure)

For true cross-package numerical parity with R/Stata, run the
following offline and record results in a comment block inside
the relevant test file:

```R
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
