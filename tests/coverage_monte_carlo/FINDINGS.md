# Monte Carlo CI coverage findings

The coverage suite validates that 95% CIs actually cover truth 95% of
the time.  This document logs findings — estimators whose inference is
or is not correctly calibrated.

## Calibrated (coverage within Wilson band around 95%)

As of the current JSS validation branch, the following estimators pass
the coverage test at B=300 unless noted otherwise:

| Estimator | DGP | Coverage (B=300) |
| --- | --- | --- |
| `sp.regress` (HC1) | RCT with covariates | ≈ 0.95 |
| `sp.did` (classic 2x2) | 2-period 2-group homogeneous | ≈ 0.95 |
| `sp.callaway_santanna` (REG, simple ATT) | Homogeneous staggered timing | 0.940 (B=200 cap) |
| `sp.rdrobust` (MSE-optimal, triangular) | Sharp RD, known jump | ≈ 0.95 |
| `sp.ivreg` (HC1) | Strong binary-Z IV | ≈ 0.95 |
| `sp.ebalance` | CIA with 2 covariates | 0.92-0.99 (B=200 cap; slightly conservative) |
| `sp.causal_question(design="dml")` | Binary-treatment IRM DGP | ≈ 0.95 (B=200 cap) |
| `sp.causal_question(design="causal_forest")` | AIPW-IF ATE DGP | ≈ 0.95 (B=200 cap) |

## Resolved finding

### `sp.callaway_santanna` simple-ATT aggregation

**Previous finding**: Empirical 95% CI coverage was about 50% on a
homogeneous staggered DGP (`test_cs_staggered_ci_coverage`).  Point
estimates were unbiased, but simple-ATT CIs were too tight.

**Root causes fixed**:

1. Group-time influence functions are estimated on the relevant
   treated/control subset, then embedded into the full unit universe for
   aggregation.  They must be multiplied by `n_total / n_relevant` during
   that embedding.
2. The outcome-regression (`estimator="reg"`) influence function must
   include uncertainty from the control outcome regression.  The previous
   implementation only carried the treated-side residual term.

**Current result**: The CS coverage test now passes without `xfail`.
On the B=200 cap used by the test, empirical coverage is `188/200 =
0.940`, inside the 99% Wilson acceptance band `[0.894, 0.997]`.
The `04_csdid` R/Stata parity row now reports simple-ATT point-estimate
parity at machine precision and analytic-SE parity within the registered
1% tolerance.

## How to run

Fast (always on, B=50):
```
pytest tests/coverage_monte_carlo/test_coverage.py::test_fast_ols_coverage_smoke
```

Full (opt-in, B=300; ~60-90s total):
```
pytest -m slow tests/coverage_monte_carlo/
```

Deep audit (B=1000, set via env):
```
STATSPAI_MC_DRAWS=1000 pytest -m slow tests/coverage_monte_carlo/
```
