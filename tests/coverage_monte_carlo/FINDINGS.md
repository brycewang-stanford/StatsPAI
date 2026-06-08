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

All six core estimators named in the project spec (`did`, `iv`, `rd`,
`synth`, `dml`, `panel`) now carry an explicit Monte Carlo coverage row.

## Headline B=1000 Coverage Audit

The canonical Track B audit materializes nine known-truth DGPs at
`B=1000`. The 99% Wilson band around nominal 0.95 is approximately
`[0.935, 0.967]`; rows above the band are treated as conservative
over-coverage, not as evidence of under-calibrated standard errors.

| Estimator | DGP | Coverage |
| --- | --- | --- |
| `sp.regress` (HC1) | RCT with covariates | 0.952 |
| `sp.regress` 2x2 DiD | 2-period homogeneous DiD | 0.955 |
| `sp.ivreg` (HC1) | Strong binary-Z IV | 0.962 |
| `sp.callaway_santanna` (REG, simple ATT) | Homogeneous staggered timing | 0.946 |
| `sp.panel` two-way FE | Unit+time FE, time-varying treatment | 0.948 |
| `sp.sdid` (placebo SE) | One treated unit, factor-model DGP | 0.939 |
| `sp.ebalance` | CIA with 2 covariates | 1.000 |
| `sp.causal_question(design="dml")` | Binary-treatment IRM ATE | 0.969 |
| `sp.causal_question(design="causal_forest")` | AIPW-IF ATE DGP | 0.977 |

Interpretation:

- Closed-form OLS, DiD, IV, the simple Callaway-Sant'Anna ATT, and the
  two-way FE panel rows sit inside the Wilson band.
- `sp.sdid` (0.939) sits at the lower-inside edge of the band — note that
  classic Abadie SCM has no analytic CI (placebo/permutation inference
  only), so the calibrated row uses synthetic difference-in-differences
  (Arkhangelsky et al. 2021, `arkhangelsky2021synthetic`); for a single
  treated unit the placebo variance estimator is the recommended one
  (jackknife is undefined with one treated unit and empirically
  under-covers at ~0.80 on this DGP).
- DML sits just above the upper edge; ebalance and causal forest are more
  visibly conservative. These are over-coverage findings, not hidden
  under-coverage.
- The expensive DML and causal-forest rows are no longer supported only by
  the lower-B pytest caps; the committed JSS artifacts record their
  explicit B=1000 rates.

## Size and Power Audit (B=1000, RD at B=500)

Coverage alone does not distinguish a valid test from a useless one: a CI
that is always [-inf, +inf] covers the truth 100% of the time but rejects
nothing. The size/power rows close that gap on the fast closed-form
estimators. The test statistic is the 95% CI itself (reject H0: effect = 0
iff 0 lies outside the interval), so size/power and coverage are guaranteed
consistent. Power deltas are the alternative effect sizes; `power[0]` is the
null point and equals the size.

| Estimator | Size (nominal 0.05) | Deltas | Power |
| --- | --- | --- | --- |
| `sp.regress` (HC1) RCT | 0.043 | [0, .10, .20, .30] | [.043, .208, .596, .903] |
| `sp.did` 2x2 | 0.024 | [0, .20, .40, .60] | [.024, .291, .871, .996] |
| `sp.ivreg` strong-Z | 0.046 | [0, .20, .40, .60] | [.046, .453, .935, .996] |
| `sp.rdrobust` sharp | 0.040 | [0, .20, .40, .60] | [.040, .148, .558, .824] |
| `sp.panel` two-way FE | 0.052 | [0, .15, .30, .45] | [.052, .231, .652, .954] |

Interpretation:

- Empirical size never exceeds the nominal 5% Wilson upper bound — there is
  no false-positive inflation. The size test is deliberately one-sided:
  over-rejection (anti-conservative SEs) fails; under-rejection is
  conservative-but-valid and is documented rather than failed.
- `sp.did` 2x2 sizes at 0.024 — conservative, exactly mirroring its 0.955
  over-coverage. The two findings are the same fact seen from two sides.
- Every power curve is monotone in the effect size and reaches >=0.82 at the
  largest delta, so each estimator is calibrated *and* discriminating.
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

Current result: the B=1000 deep audit reports `946/1000 = 0.946`, inside
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
| `sp.callaway_santanna` (REG) | Heterogeneous timing and magnitude | 0.946 (B=1000) | [0.92, 0.96] |
| `sp.causal_question(causal_forest)` | Severe propensity-overlap loss | 0.983 (B=300) | [0.85, 0.99] |

Findings interpretation:

- Weak-IV under-coverage is textbook. With a first-stage F far below
  Stock-Yogo critical values, HC1 2SLS intervals miss truth more often
  than nominal 0.95. User-facing recovery routes are LIML and
  Anderson-Rubin inference, surfaced by the preflight/design-detect path.
- CS-DiD remains calibrated under the heterogeneous timing/magnitude DGP,
  consistent with cell-level ATT(g, t) estimation and simple-ATT
  aggregation.
- Causal forest under severe overlap loss over-covers. Extreme
  propensities inflate the AIPW influence-function variance and produce
  wide intervals; the audit treats this as a diagnostic warning, not as an
  efficiency claim.

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
python tests/coverage_monte_carlo/run_b1000.py              # coverage, 9 rows
python tests/coverage_monte_carlo/run_robustness_b1000.py   # robustness rows
python tests/coverage_monte_carlo/run_size_power_b1000.py   # size + power
```
