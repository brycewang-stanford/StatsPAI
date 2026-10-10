# StatsPAI Performance Benchmarks

This directory measures StatsPAI's wall-clock speed on canonical
workloads, with and without comparison to alternative Python
econometrics libraries.

## Running

Quick run (n up to ~10k, seconds):

```bash
python3 benchmarks/run_all.py --quick
```

Full run (up to n=500k; takes 5-15 minutes):

```bash
python3 benchmarks/run_all.py --full
```

Results are written to ``benchmarks/RESULTS.md`` and
``benchmarks/results.json``.

## What's measured

| File | Benchmark | Comparison libs (when available) |
| --- | --- | --- |
| ``bench_regression.py`` | ``sp.regress`` scaling | statsmodels OLS |
| ``bench_hdfe.py`` | ``sp.absorb_ols`` HDFE | linearmodels PanelOLS, dummy-variable OLS |
| ``bench_did.py`` | ``sp.callaway_santanna`` | (StatsPAI only — no comparable Python package) |
| ``bench_rd.py`` | ``sp.rdrobust`` | (StatsPAI only) |
| ``bench_matching.py`` | ``sp.ebalance``, ``sp.cbps`` | (StatsPAI only) |

``bench_scaling.py`` is separate from the table above: it times about
120 estimators at three sample sizes and prints the growth exponent of
each (``python benchmarks/bench_scaling.py --only did``). Use it to look
for a method whose cost grows faster than it should; it is not part of
``run_all.py``.

Comparison libraries are imported lazily; if absent, the benchmark
falls back to showing only StatsPAI's numbers and notes the missing
comparator.

## Why benchmarks belong outside tests/

- They take minutes (``pytest tests/`` should stay fast).
- They measure speed, not correctness, so they make failure
  assertions on timings (which flake on shared CI runners).
- Results are consumed by CHANGELOG / README, not by CI.

## Current headline numbers

See ``RESULTS.md`` after running, or the ``## Benchmarks`` section
in the main README / CHANGELOG.
