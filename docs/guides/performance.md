# Performance: an honest cross-library benchmark

StatsPAI's position on performance is simple: **publish where we win and where
we lose, on one machine, with the numbers proving every backend computes the
same estimator.** Selectively quoting only the regimes where a library is fast
is the cheapest way to lose an econometrician's trust, so the benchmark that
backs this page is built to make that impossible — it always prints the rows
where StatsPAI is *slower*.

## The harness

[`benchmarks/cross_library/hdfe_benchmark.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/benchmarks/cross_library/hdfe_benchmark.py)
runs a single two-way (unit × time) fixed-effects OLS on one generated panel and
times four backends head-to-head:

| backend | what it is |
| --- | --- |
| `sp.absorb_ols` | StatsPAI's own HDFE kernel (alternating projections) |
| `sp.feols` | StatsPAI's `feols` API (pyfixest-backed) |
| `pyfixest` | pyfixest directly |
| `R fixest` | R `fixest::feols` via `Rscript` |

Design choices that keep it trustworthy:

- **Correctness gates speed.** Every row carries the estimated slope and SE, and
  the report states the maximum relative disagreement against a reference. A
  fast *wrong* answer is reported as `✗ DISAGREE`, never as a win.
- **Median + IQR over repeats, after a warm-up.** One untimed call absorbs
  JIT/import cost; the reported time is the median with the inter-quartile range,
  so warm-up and noise cannot masquerade as signal.
- **No silent scope.** A missing baseline (no `pyfixest`, no `Rscript`+`fixest`)
  is recorded as `skipped` *with the reason* — never dropped invisibly.
- **Provenance embedded.** Library versions, Python, platform, and CPU count are
  written into every result file.

Reproduce it yourself — and please do before quoting any number:

```bash
python benchmarks/cross_library/hdfe_benchmark.py \
    --scales 10000,100000,1000000 --repeats 7 \
    --json benchmarks/cross_library/results.json \
    --markdown benchmarks/cross_library/RESULTS.md
```

## What we see (and what we don't)

On a representative balanced two-way panel (Apple-silicon arm64, the run checked
into [`benchmarks/cross_library/RESULTS.md`](https://github.com/brycewang-stanford/StatsPAI/blob/main/benchmarks/cross_library/RESULTS.md)):

- **All four backends agree to ~1e-11** on the slope and SE. This is the
  precondition for any timing claim to be meaningful.
- At **n ≈ 10k–100k balanced panels**, StatsPAI's native `sp.absorb_ols` kernel
  is the **fastest** backend, with R `fixest` and `pyfixest` a small multiple
  behind. `sp.feols` carries a thin wrapper overhead on top of pyfixest, so it
  trails pyfixest slightly — that overhead is reported, not hidden.

What this run **does not** establish, and where we are explicitly *not* claiming
a win:

- **Very large n (≥ 1M) and high-cardinality fixed effects.** R `fixest` is a
  mature, heavily-optimised C++ engine; at large scale it is competitive with or
  faster than the StatsPAI/pyfixest paths in our broader profiling. Treat the
  small-to-medium-panel lead above as exactly that — a small-to-medium-panel
  lead — until you have re-run the harness at your own target scale.
- **Unbalanced panels, weights, IV, and clustered VCOV at scale.** The harness
  covers the canonical two-way FE OLS case; other regimes need their own runs.
- **Memory.** Peak RSS is recorded best-effort but is process-cumulative, so
  treat it as indicative, not authoritative.

The honest summary: StatsPAI is numerically in lockstep with the reference
implementations, and its native kernel is genuinely fast on small-to-medium
balanced panels — but R `fixest` remains the one to beat at the largest scales,
and we say so rather than quietly choosing scales where we look best.
