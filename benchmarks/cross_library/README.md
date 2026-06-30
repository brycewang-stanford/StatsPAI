# Cross-library HDFE benchmark (additive, self-contained)

An **honest** head-to-head benchmark of StatsPAI's HDFE paths against `pyfixest`
and R `fixest` on one unified two-way fixed-effects DGP. The design goal is to
publish *where StatsPAI wins and where it loses*, with coefficient agreement
proving every backend computes the same estimator.

This directory is **deliberately self-contained**: it does not import from
`tests/perf/` and never writes outside itself — in particular it never touches
the Paper-JSS Track-C performance artifacts (the harness asserts its write path
stays under this directory).

## Run

```bash
# quick (10k, 100k; whatever backends are installed)
python benchmarks/cross_library/hdfe_benchmark.py

# full sweep with artifacts
python benchmarks/cross_library/hdfe_benchmark.py \
    --scales 10000,100000,1000000 --repeats 7 \
    --json benchmarks/cross_library/results.json \
    --markdown benchmarks/cross_library/RESULTS.md
```

Backends auto-skip (with a recorded reason) when a dependency is missing, so the
harness runs anywhere; install `pyfixest` and R `fixest` for the full picture.

## Files

| file | what |
| --- | --- |
| `hdfe_benchmark.py` | the harness (DGP, backends, timing, honest report) |
| `RESULTS.md` | a generated report (re-run on your own hardware before quoting) |
| `results_*.json` | raw results + provenance for one machine |

See [`docs/guides/performance.md`](../../docs/guides/performance.md) for the
public write-up and the honest win/lose discussion.
