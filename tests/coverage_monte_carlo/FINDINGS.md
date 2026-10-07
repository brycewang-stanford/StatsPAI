# Monte Carlo CI Coverage, Size, and Power Findings

This file records what the Monte Carlo suite is allowed to claim. The JSS
submission uses committed deep-audit artifacts in
`results_b1000/coverage_b1000.json`,
`results_b1000/coverage_robustness_b1000.json`, and
`results_b1000/size_power_b1000.json`; the regular pytest suite keeps lower
draw caps for wall-clock reasons.

The suite now validates all three faces of the inference machinery:

- **Coverage** — does the 95% CI cover the truth ~95% of the time?
  (`test_coverage.py`)
- **Size** — under the null, does a 5% test reject ~5% of the time, i.e. no
  false-positive inflation? (`test_size_power.py`)
- **Power** — under alternatives, does rejection rise monotonically with the
  effect size and approach 1? (`test_size_power.py`)

## Headline B=1000 Coverage Audit

The canonical Track B audit materializes twelve known-truth DGPs in
thirteen rows at `B=1000` (each row records B=1000 draws; the two-way FE
panel is run through `sp.panel` and, since the JSS v2 review, through the
suite's HDFE member `sp.fast.feols` itself), and since the 2026-09 JSS review
each row records every draw's estimate, SE and interval, so the table
reports why a rate is what it is: bias, Monte Carlo SD, mean reported SE
and their ratio (SE calibration). The 99% Wilson band around nominal 0.95
is approximately `[0.935, 0.967]`.

| Estimator | DGP | Coverage | Bias | MC SD | Mean SE | SE/SD |
| --- | --- | --- | --- | --- | --- | --- |
| `sp.regress` (HC1) | RCT with covariates | 0.952 | -0.0028 | 0.0707 | 0.0708 | 1.00 |
| `sp.regress` 2x2 DiD | 2-period homogeneous DiD | 0.955 | +0.0017 | 0.0974 | 0.0999 | 1.03 |
| `sp.ivreg` (HC1) | Strong binary-Z IV | 0.962 | -0.0041 | 0.3687 | 0.3627 | 0.98 |
| `sp.callaway_santanna` (REG, simple ATT) | Homogeneous staggered timing | 0.947 | -0.0007 | 0.1080 | 0.1090 | 1.01 |
| `sp.sun_abraham` (overall ATT) | Homogeneous staggered timing | 0.950 | -0.0007 | 0.1214 | 0.1234 | 1.02 |
| `sp.panel` two-way FE | Known-coefficient FE panel | 0.948 | -0.0024 | 0.1245 | 0.1233 | 0.99 |
| `sp.fast.feols` (CR1 by unit) | Same FE panel, same draws | 0.955 | -0.0057 | 0.1026 | 0.1013 | 0.99 |
| `sp.rdrobust` sharp (robust CI) | Known-jump curved RD | 0.934 | -0.0061 | 0.1222 | 0.1201 | 0.98 |
| `sp.sdid` (placebo SE) | Factor-model panel, 1 treated | 0.928 | +0.0003 | 0.3095 | 0.3071 | 0.99 |
| `sp.ebalance` (M-estimation SE) | CIA with 2 covariates | 0.945 | -0.0014 | 0.0799 | 0.0774 | 0.97 |
| `sp.causal_question(design="dml")` | Binary-treatment IRM ATE | 0.968 | -0.0074 | 0.1582 | 0.1653 | 1.04 |
| `sp.dml(model="plr")`, default learners | Continuous-treatment PLR | 0.883 | -0.0261 | 0.0533 | 0.0472 | 0.88 |
| `sp.causal_question(design="causal_forest")`, 2,000 trees | AIPW-IF ATE DGP | 0.959 | +0.0095 | 0.0934 | 0.0968 | 1.04 |

Interpretation:

- Closed-form OLS, DiD, IV, both staggered aggregations
  (Callaway-Sant'Anna and Sun-Abraham), the two-way FE panel (both entry
  points; `sp.fast.feols` with the pre-1.31 `ssc="statspai"` default
  over-covered at 0.979 with SE/SD 1.08, see `mechanisms/feols_ssc.py`), entropy
  balancing and the causal forest sit inside the Wilson band with SE/SD
  within a few percent of 1.
- Sharp RD (0.934) and SDID (0.928) have calibrated SEs (0.98, 0.99); the
  shortfall is in the shape of the sampling distribution. For RD it is the
  reference procedure's own finite-sample behaviour: R `rdrobust` 4.0.0
  returns the same intervals draw by draw to 6e-12 and covers in exactly
  the same draws (`mechanisms/rd_drawwise_reference.py`).
- DML PLR with the default gradient-boosting learners under-covers (0.883):
  bias is half a Monte Carlo SD. `sp.dml` equals DoubleML to 1e-16 on the
  same folds and learners; with the true nuisances coverage is 0.952, with a
  random forest 0.940, with a spline Lasso 0.938, and with the default
  learner 0.936 at n = 2,000 (`mechanisms/dml_plr_learners.py`). The DML
  IRM row (0.968) is a different configuration and does not stand in for PLR.
- Entropy balancing covered 1.000 before 2026-09 because its SE held the
  weights fixed (twice the Monte Carlo SD); the M-estimation SE is calibrated.
- The causal-forest row covered 0.977 before 2026-09 because
  `causal_question(design="causal_forest")` reported a separate AIPW
  estimator; it now reports the forest's own doubly-robust ATE at 2,000
  trees (`mechanisms/forest_trees.py` checks 300 trees too).

## Size and Power Audit (B=1000; RD at B=500, CS at B=300)

Coverage alone does not distinguish a valid test from a useless one: a CI
that is always [-inf, +inf] covers the truth 100% of the time but rejects
nothing. The size/power rows close that gap on the fast closed-form
estimators. The test statistic is the 95% CI itself (reject H0: effect = 0
iff 0 lies outside the interval), so size/power and coverage are guaranteed
consistent. Power deltas are the alternative effect sizes; `power[0]` is the
null point and equals the size.

| Estimator | Size (nominal 0.05) | Deltas | Power |
| --- | --- | --- | --- |
| `sp.regress` (HC1) RCT | 0.043 | [0, .100, .200, .300] | [.043, .208, .596, .903] |
| `sp.did` 2x2 | 0.024 | [0, .200, .400, .600] | [.024, .291, .871, .996] |
| `sp.ivreg` strong-Z | 0.046 | [0, .200, .400, .600] | [.046, .453, .935, .996] |
| `sp.rdrobust` sharp | 0.076 | [0, .200, .400, .600] | [.076, .404, .888, 1.0] |
| `sp.panel` two-way FE | 0.052 | [0, .150, .300, .450] | [.052, .231, .652, .954] |
| `sp.callaway_santanna` staggered | 0.050 | [0, .300, .600, .900] | [.050, .777, 1.0, 1.0] |
| `sp.ebalance` (M-estimation SE) | 0.055 | [0, .400, .700, 1.0] | [.055, 1.0, 1.0, 1.0] |

Interpretation:

- At the tested null no row over-rejects except sharp RD (0.076 at B=500),
  consistent with its 0.934 coverage and reproduced by the reference
  procedure; 2x2 DiD is conservative (0.024), mirroring its 0.955 coverage.
- `sp.ebalance` now sizes at 0.055. Under the pre-2026-09 SE it sized at
  0.000, the direct counterpart of its 1.000 over-coverage.
- `sp.callaway_santanna` sizes at 0.050, the size-side twin of its 0.947
  simple-ATT coverage.
- Every power curve is monotone in the effect size and reaches >=0.90 at the
  largest delta.
- Cross-fit DML, causal forest, and resampling-based SDID keep
  coverage-only rows: a multi-delta power sweep at B=1000 is too expensive
  for them, and their coverage rows already exercise the same SE machinery.

## Resolved Finding

### `sp.callaway_santanna` simple-ATT aggregation

Previous finding: empirical 95% CI coverage was about 50% on a
homogeneous staggered DGP. Point estimates were unbiased, but simple-ATT
CIs were too tight.

Root causes fixed:

1. Group-time influence functions are estimated on the relevant
   treated/control subset, then embedded into the full unit universe for
   aggregation. They must be multiplied by `n_total / n_relevant` during
   that embedding.
2. The outcome-regression (`estimator="reg"`) influence function must
   include uncertainty from the control outcome regression. The previous
   implementation only carried the treated-side residual term.

Current result: the B=1000 deep audit reports `947/1000 = 0.947`, inside
the 99% Wilson band `[0.935, 0.967]`. The `04_csdid` R/Stata parity row
reports simple-ATT point-estimate parity at machine precision and
analytic-SE parity within the registered 1% tolerance.

## Robustness DGPs

The sibling robustness suite runs selected estimators under DGPs that
stress or violate identification assumptions. These rows are descriptive
failure-mode checks, not pass/fail nominal-calibration claims. Each row
has a documented band; a movement outside the band means the estimator
changed and the finding must be reviewed.

| Estimator | Stressor | Deep-audit coverage | Documented band |
| --- | --- | --- | --- |
| `sp.ivreg` (HC1) | Weak instrument: pi=0.10, median F = 2.51 | 0.882 (B=1000) | [0.85, 0.95] |
| `sp.callaway_santanna` (REG) | Heterogeneous timing and magnitude | 0.954 (B=1000) | [0.92, 0.96] |
| `sp.causal_question(causal_forest)` | Severe propensity-overlap loss | 0.900 (B=300) | [0.85, 0.99] |

Findings interpretation:

- Weak-IV under-coverage is textbook. With a first-stage F far below
  Stock-Yogo critical values, HC1 2SLS intervals miss truth more often
  than nominal 0.95. User-facing recovery routes are LIML and
  Anderson-Rubin inference, surfaced by the preflight/design-detect path.
- CS-DiD remains calibrated under the heterogeneous timing/magnitude DGP,
  consistent with cell-level ATT(g, t) estimation and simple-ATT
  aggregation.
- Causal forest under severe overlap loss under-covers (0.900, B=300) now
  that the row reports the forest's own AIPW ATE: with propensities near 0
  and 1 the doubly-robust score is unstable and its interval too narrow.
  (Before 2026-09 the row measured a separate AIPW estimator and read
  0.983.) The audit flags overlap before interpretation; this is a
  failure-mode record, not a calibration claim.

## How to Run

Fast smoke (always run, not `slow`):

```bash
pytest tests/coverage_monte_carlo/test_coverage.py::test_fast_ols_coverage_smoke
pytest tests/coverage_monte_carlo/test_size_power.py::test_ols_size_smoke
```

Canonical slow pytest suite, using the default lower draw caps
(`STATSPAI_MC_DRAWS` overrides B; default 300):

```bash
pytest -m slow tests/coverage_monte_carlo/test_coverage.py
pytest -m slow tests/coverage_monte_carlo/test_size_power.py
```

Robustness DGPs, also lower-capped for routine pytest use:

```bash
pytest -m slow tests/coverage_monte_carlo/test_coverage_robustness.py
```

Deep JSS audit artifacts:

```bash
python tests/coverage_monte_carlo/run_b1000.py              # coverage, 11 rows
python tests/coverage_monte_carlo/run_robustness_b1000.py   # robustness rows
python tests/coverage_monte_carlo/run_size_power_b1000.py   # size + power
```
