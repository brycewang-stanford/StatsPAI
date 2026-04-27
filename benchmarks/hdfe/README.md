# HDFE Poisson Baseline (Phase 0)

Phase 0 of the HDFE roadmap: freeze the **starting line** that the
StatsPAI Rust HDFE backend (Phase 1+) is expected to beat.

## What this measures

Two-way crossed FE Poisson regression on three synthetic datasets:

| config  | n          | fe1 (high) | fe2 (low) |
| ------- | ---------- | ---------- | --------- |
| small   | 100,000    | 1,000      | 50        |
| medium  | 1,000,000  | 100,000    | 1,000     |
| large   | 10,000,000 | 1,000,000  | 10,000    |

Backends compared:

- `statspai`  — `sp.fepois` (currently a thin wrapper over pyfixest)
- `pyfixest`  — `pyfixest.fepois`
- `fixest`    — R `fixest::fepois` (auto-detected via `Rscript`)
- `ppmlhdfe`  — Stata, **manual only** (template in `run_stata.do`)

Reported metrics: wall-clock (warmup + repeated timing), peak RSS
(Python sub-processes only), coefficient cross-backend max-abs-diff,
and (when available) iteration counts.

## Quick start

```bash
# 1. From repo root, with dev install (`pip install -e ".[dev,fixest]"`)
cd benchmarks/hdfe

# 2. Generate just the small + medium CSVs (cached, idempotent).
python3 datasets.py --configs small medium

# 3. Run baselines for those datasets, write BASELINE.md + baseline.json.
python3 run_baseline.py --datasets small medium

# 4. Re-render the report only (no re-runs):
python3 run_baseline.py --datasets small medium --report-only
```

`large` is opt-in:

```bash
python3 datasets.py --configs large            # ~500MB on disk, ~5min
python3 run_baseline.py --datasets large       # ~30-60min wall time
```

## Adding Stata results

Stata is not auto-detected (the dev box has no Stata install). On a
machine with Stata 17+:

```bash
ssc install ppmlhdfe, replace
ssc install reghdfe,  replace
ssc install ftools,   replace

# Decompress (Stata cannot read .gz directly):
gunzip -k data/small.csv.gz

stata -b do run_stata.do small
# → writes results/small_ppmlhdfe.json (matches naming convention used
#   by every other backend — `<dataset>_<backend>.json`)
```

Then re-render the report:

```bash
python3 run_baseline.py --datasets small medium --report-only
```

## File layout

```
benchmarks/hdfe/
├── README.md            # this file
├── datasets.py          # DGP + CSV materialisation
├── run_python.py        # one-shot Python backend runner (subprocess)
├── run_r.R              # one-shot R backend runner (subprocess)
├── run_stata.do         # Stata template (manual)
├── run_baseline.py      # top-level driver + report renderer
├── data/                # cached datasets (.csv.gz + .meta.json)
├── results/             # per-(dataset, backend) JSON
├── baseline.json        # aggregated machine-readable report
└── BASELINE.md          # human-readable report (committed)
```

## Acceptance for Phase 0

- All three CSVs deterministic from seed (re-running `datasets.py`
  produces bit-identical output for the same git revision).
- `pyfixest` and `statspai` numbers match within 1e-12 (today they
  share the same code path).
- `fixest` vs `pyfixest` coefficient `max_abs_diff` ≤ 1e-6 on small
  and medium. (1e-6 is fixest's documented IRLS convergence floor.)
- `BASELINE.md` committed. Future PRs in Phase 1+ append a row /
  column with the new Rust backend numbers and never regress these
  baselines without an explicit ⚠️ correctness note in CHANGELOG.

## Why CSV (not parquet/feather)?

The local R install has no `arrow` package, and `data.table::fread`
on `.csv.gz` is fast enough at our sizes. Switching to Arrow/IPC is
a Phase 5 task (Polars/Arrow direct path) — adding it here would
couple Phase 0 to Phase 5 dependencies for no measurement benefit.
