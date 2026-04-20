# Monte Carlo CI coverage findings

The coverage suite validates that 95% CIs actually cover truth 95% of
the time.  This document logs findings — estimators whose inference is
or is not correctly calibrated.

## Calibrated (coverage ∈ Wilson band around 95%)

As of v0.9.4 / v0.9.5, the following estimators pass the coverage test
at B=300:

| Estimator | DGP | Coverage (B=300) |
| --- | --- | --- |
| `sp.regress` (HC1) | RCT with covariates | ≈ 0.95 |
| `sp.did` (classic 2x2) | 2-period 2-group homogeneous | ≈ 0.95 |
| `sp.rdrobust` (MSE-optimal, triangular) | Sharp RD, known jump | ≈ 0.95 |
| `sp.ivreg` (HC1) | Strong binary-Z IV | ≈ 0.95 |
| `sp.ebalance` | CIA with 2 covariates | 0.92-0.99 (slightly conservative) |

## Known under-coverage (xfail with remediation plan)

### `sp.callaway_santanna` simple-ATT aggregation

**Observed**: Empirical 95% CI coverage ≈ 50% on a homogeneous
staggered DGP (`test_cs_staggered_ci_coverage`).  The individual
ATT(g, t) point estimates are unbiased, and the ATT(g, t) SEs are
calibrated.  The miscalibration is introduced when aggregating to the
simple ATT.

**Root cause**: The current `aggte` implementation computes the
aggregated SE as if the ATT(g, t) estimates were independent:

```
SE_simple = sqrt(sum(w_gt^2 * SE_gt^2))
```

But ATT(g, t) for different (g, t) cells share the never-treated
control group, so their influence functions are strongly positively
correlated.  Ignoring that correlation underestimates the aggregated
variance by roughly a factor of (number of (g, t) cells), matching
the observed 3× SE compression.

**Fix (roadmap, v0.9.6)**: Use the full influence-function matrix to
compute

```
Var(simple_ATT) = w' * Cov(psi) * w
```

where `psi` is the (n × n_gt) matrix of per-unit influence-function
contributions to each ATT(g, t), and `Cov(psi) = psi' psi / n²`.  This
matches R's `did::aggte` multiplier bootstrap.

**Impact on v0.9.5 users**: point estimates are unaffected.  Anyone
quoting the SE or CI from `sp.callaway_santanna(...).summary()` under
the simple aggregation should replace them with (a) per-group ATTs'
own SEs, which are calibrated, or (b) cluster-bootstrapped SEs via
the ``multiplier_bootstrap`` parameter (to be added).

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
