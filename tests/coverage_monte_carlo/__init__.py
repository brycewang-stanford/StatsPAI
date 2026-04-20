"""Monte Carlo CI coverage validation.

Recovery tests (in ``reference_parity/``) ask: "does the point
estimate converge to truth on a single draw?".

These tests ask a different, stronger question:
    "If I rerun a seeded DGP 300+ times and build the 95% CI on
    each draw, does truth fall inside the CI 95% of the time?"

This is the only way to catch SE miscalibration — an estimator
can be unbiased but have SEs that are 30% too small, producing
CIs that cover only 75% of the truth.  Recovery tests wouldn't
flag it; coverage tests will.

## Tolerance

With B draws and nominal coverage 0.95, the Wilson 95% confidence
band around the empirical coverage is roughly ±0.024 (for B=300).
We accept [0.92, 0.98] as "correctly calibrated".  Estimators
falling outside that band are flagged as SE-miscalibrated.

## Execution time

Each test file takes 30-90 seconds.  To keep CI friendly:
- Marked with ``@pytest.mark.slow`` for opt-in execution.
- Default B=300 (adjustable via environment variable).
- Can be skipped in default `pytest tests/` run via pytest.ini
  markers; run explicitly with `pytest -m slow tests/coverage_monte_carlo/`.
"""
