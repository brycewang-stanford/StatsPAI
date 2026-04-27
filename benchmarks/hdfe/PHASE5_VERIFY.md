# Phase 5 — Verification Report

**Scope**: Polars / PyArrow input path for the Phase 1+ HDFE kernels.

## Deliverables

* New module ``src/statspai/fast/polars_io.py``.
* ``sp.fast.demean_polars(df, X_cols, fe_cols, ...)`` — within-transform
  on a polars DataFrame / LazyFrame.
* ``sp.fast.fepois_polars(df, formula, ...)`` — Poisson HDFE estimation
  on polars input. Lazily evaluates and projects only the referenced
  columns before pulling into pandas.
* Optional dep — module is silently skipped if polars is not installed.

## Honest scope statement

The original plan called for a **zero-copy Arrow C Data Interface path**
into the Rust kernel. That requires a chunk of Rust + arrow2 / arrow-rs
infrastructure that doesn't exist on the FFI boundary today. What
ships:

* Polars **input adapter** that extracts the needed columns and runs
  through the existing kernels.
* Lazy → eager collection happens once per call (we don't fuse the
  demean into the Polars query plan).

Numerical correctness vs the pandas path: bit-equivalent (the same
NumPy arrays end up in the kernel). Throughput vs pandas: roughly the
same on these sizes (we measured small/medium where the kernel itself
dominates). The genuine win — bypassing pandas entirely on 1e8+ row
inputs — requires the C Data Interface work; that's tracked for v1.8.x.

## Test suite

```
tests/test_fast_polars.py
  test_demean_polars_eager_matches_pandas .............. PASSED
  test_demean_polars_lazyframe_collected ............... PASSED
  test_demean_polars_missing_column_raises ............. PASSED
  test_fepois_polars_matches_pandas .................... PASSED
  test_fepois_polars_lazyframe ......................... PASSED
  test_fepois_polars_only_collects_needed_columns ...... PASSED
  test_fepois_polars_missing_column_raises ............. PASSED
7 passed in 1.66s
```

## Cumulative regression (Phases 1+2+3+4+5)

```
109 passed, 1 skipped
```

## Acceptance against original Phase 5 plan

| plan item                                              | status | note |
| ------------------------------------------------------ | ------ | ---- |
| Polars ``DataFrame`` input                             | ✅     | ``demean_polars``, ``fepois_polars`` |
| Polars ``LazyFrame`` input                             | ✅     | collected on entry |
| Pandas path remains the default                        | ✅     | unchanged |
| ``Arrow C Data Interface`` direct → Rust (zero-copy)   | ⏸     | Requires arrow-rs FFI in the Rust crate; tracked. Bypassing pandas is a Polars-input convenience today, not a true zero-copy boundary. |
| Bench 1e8 rows: Polars wall-time ≤ Pandas × 0.7        | ⏸     | Not measured. Closing this gap requires the zero-copy path; the adapter shipped here doesn't change kernel throughput. |
| Memory peak ≤ Pandas × 0.5                             | ⏸     | Same caveat. |

## Known cosmetic warning

The Rust extension surfaces a `DeprecationWarning` from numpy / PyO3
about `numpy.core.multiarray` being renamed to `numpy._core.multiarray`.
It comes from the upstream `numpy` Rust crate (v0.21) — a fix requires
upgrading to a newer numpy crate, which we'll do in a routine
dependency bump. No correctness impact.
