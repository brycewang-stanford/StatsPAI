# Workflow benchmark

`run.py` times the estimation tasks applied work runs -- two-way HDFE OLS on an
unbalanced panel, IV with HDFE, PPML with separated groups, staggered DiD
(Callaway-Sant'Anna), DML with repeated cross-fitting, and a wild cluster
bootstrap -- at 100k and 1M rows, against pyfixest, DoubleML and R (fixest,
did). Each backend runs in its own process; the report gives cold start
(import + first fit), warm median / IQR, peak RSS and failures, and compares
the estimates *before* the times.

```bash
python benchmarks/workflow_bench/run.py --sizes 100000 1000000 --reps 3
python benchmarks/workflow_bench/run.py --tasks hdfe --sizes 100000 --threads 1
```

## Snapshot (2026-09-26, Apple M3, 8 cores, default threads)

Every task's estimate agrees across all backends to <= 1.3e-13 (DML to
2e-6: different fold draws). Where StatsPAI is slower it says so:

| task (1M rows) | fastest | StatsPAI |
| --- | --- | --- |
| HDFE OLS (unbalanced) | `sp.hdfe_ols` 0.11 s | fastest; R fixest 1.07x; `sp.feols` (pyfixest) 1.55x; `sp.fast.feols` 1.69x |
| IV with HDFE | R fixest 0.28 s | `sp.feols` 1.37x (pyfixest backend), 3.3 GB peak |
| PPML with separation | R fixest 0.48 s | `sp.fepois` 1.84x; native `sp.fast.fepois` 1.93x (was 3.10x before the fused kernel; fastest at 100k rows; least memory) |
| Callaway-Sant'Anna (100k units x 10) | `sp.callaway_santanna` 0.25 s | fastest; R `did` 7.5x |
| DML PLR, 5 folds x 5 reps | `sp.dml` 3.40 s | equal to DoubleML (1.01x), half its memory |
| wild cluster bootstrap, 100k, B = 999 | -- | 0.016 s / 205 MB after the O(B x G) engine (was 0.64 s / 2.7 GB) |

Full tables: [RESULTS.md](RESULTS.md); raw numbers: `results.json`.

Notes:

* The R `did` call casts the cohort column to double first: `did` recodes
  never-treated `0` to `Inf`, which becomes `NA` on an integer column and
  silently drops every never-treated unit (the first run of this harness
  showed R at 1.12 vs 1.30 until the cast).
* R peak memory is not measured (no portable in-process peak RSS).
* `sp.fast.fepois` runs its NumPy/numba fallback here: the optional Rust
  HDFE extension (`rust/statspai_hdfe`, built with `maturin develop`) is not
  part of a plain `pip install`, so these are the numbers most users get.
  R fixest remains ~2x faster at 1M rows.
