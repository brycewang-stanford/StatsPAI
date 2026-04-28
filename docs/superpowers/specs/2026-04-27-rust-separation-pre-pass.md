# Native Rust Poisson Separation Pre-Pass — Design Spec

> **Status**: design (pre-implementation). Authored 2026-04-27. Targets the
> v1.8.0 release alongside the Phase B1 native Rust IRLS and the prod_fn
> module. Path A from the post-B1 brainstorm: only move pre-passes that
> have a real Rust speedup; keep formula parsing / FePoisResult / vcov in
> Python where they already live efficiently.

## 1. Goal

Move the iterative Poisson-separation drop (currently `_drop_separation`
in `src/statspai/fast/fepois.py`) from Python to Rust. Reduce medium-
benchmark wall by ~50 ms (0.880 s → ~0.83 s = 1.30× of fixest::fepois).
No public API change.

## 2. Non-goals

Explicitly out of scope (per Path A vs B/C in the brainstorm):

- Formula parsing in Rust — `_parse_fepois_formula` stays Python.
- Column extraction / intercept handling — stay Python.
- `FePoisResult` construction / `tidy()` — stay Python.
- IID / HC1 / CR1 vcov computation — stay Python.
- The CR1 cluster-robust SE integration recovered in commit `39c94d0`
  is **byte-untouched**.

## 3. Background — what `_drop_separation` does

Poisson regression cannot identify FE coefficients for groups whose
outcomes are all zero (because `mu_g > 0` for all observations in the
group, but `Σ y_g = 0`, so the score equation has no interior solution).
The pre-pass at `fepois.py:_drop_separation` (~30 lines) iteratively:

1. For each FE dimension k, compute `sums = bincount(codes_k, y)`.
2. Find groups with `sums == 0`.
3. Drop rows in those groups; recompute on the survivor mask.
4. Repeat until no drops occur (one pass typically converges; pathological
   cases may need 2-3).

The cost is `O(n × n_iter × K)` where `n_iter ≤ 3` typically. On the
medium benchmark (n=1M, K=2, n_iter=2-3), the Python path is ~50 ms.

Profile breakdown of the current Python path (cProfile on a fepois call):

```
0.046 s   _drop_separation
0.019 s     np.unique (twice for the two FE dims)
0.019 s     np.isin
0.005 s     np.bincount
0.003 s     mask + indexing
```

The `np.unique` + `np.isin` are the load-bearing slow ops (each is `O(n
log n)`). A Rust port using `bincount` + direct iteration avoids both.

## 4. Surface

### 4.1 Rust crate (`statspai_hdfe`)

New module `rust/statspai_hdfe/src/separation.rs` with one `pub fn`:

```rust
/// Iterative Poisson separation detection. Returns a keep-mask of
/// length n: `keep[i] = false` iff observation i lives in an FE group
/// whose total y-sum is zero (after iterating).
///
/// Cost: O(n × n_iter × K) with n_iter ≤ 3 for typical inputs;
/// allocates K group-sum buffers (sized by g_per_fe) once and reuses
/// them across iterations. No `np.unique` / `np.isin`-like O(n log n)
/// passes.
pub fn separation_mask(
    y: &[f64],
    fe_codes: &[&[i64]],
    g_per_fe: &[usize],
) -> Vec<bool>;
```

Plus a PyO3 binding in `lib.rs`:

```rust
#[pyfunction]
fn separation_mask(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    g_per_fe: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<u8>>>;
```

Returns a `u8[n]` mask (1=keep, 0=drop) — same shape contract as
`singleton_mask` already exposes.

Crate `__version__` bumps to `"0.6.0"`; Cargo `version` bumps to
`"0.5.0-alpha.1"`.

### 4.2 Python wiring

In `src/statspai/fast/fepois.py`, the `_drop_separation` function stays
(canonical Python reference + Rust-unavailable fallback). The call site
(currently `keep_sep = _drop_separation(y[keep], sub_codes)`) routes
through a small dispatcher:

```python
def _drop_separation_dispatcher(y, fe_codes, g_per_fe):
    if _HAS_RUST_HDFE and hasattr(_rust_hdfe, "separation_mask"):
        mask_u8 = _rust_hdfe.separation_mask(y, list(fe_codes),
                                              np.asarray(g_per_fe, dtype=np.int64))
        return mask_u8.astype(bool, copy=False)
    return _drop_separation(y, fe_codes)  # NumPy fallback, unchanged
```

The single existing call site swaps to call the dispatcher.

## 5. Tests

### 5.1 Rust unit tests (in `separation.rs`)

1. `separation_no_drop`: y = [1,2,3,4], all groups have positive sums →
   mask is all-true.
2. `separation_one_pass`: y = [0,0,1,1] with two FEs structured so that
   group 0 of FE1 is all-zero → mask drops first 2 rows.
3. `separation_iterative`: 3-iter cascade where dropping one FE's
   zero-only group exposes a NEW zero-only group on the other FE.

### 5.2 Python parity test (in `tests/test_fast_fepois.py`)

`test_separation_rust_matches_python`: random synthetic with seeded
zero-cluster injection. Rust mask ↔ Python `_drop_separation` mask must
agree element-wise across 10 seeds.

### 5.3 Existing tests as regression cover

`test_fepois_separation_drop_count` and the cluster-SE suite already
exercise the separation pre-pass end-to-end via `fepois()`. They must
continue to pass after the dispatcher rewire.

## 6. Benchmark gate

Re-run `benchmarks/hdfe/run_fepois_phase_b.py` on the medium dataset
after the rewire. Target: median wall ≤ **0.85 s** (a 30 ms improvement
over Phase B1's 0.880 s). Acceptable miss range: ≤ 0.90 s with
documented thermal noise. Failure outside that range → AUDIT entry +
investigate.

## 7. Risk

- The Rust separation kernel must be byte-equivalent to Python's
  `_drop_separation` mask. A 5-seed × n=10k fuzz test at the kernel
  level catches drift.
- The CR1 cluster code is untouched. The dispatcher rewire only changes
  one Python line (the call to `_drop_separation` becomes a call to
  `_drop_separation_dispatcher`).
- The Rust-unavailable fallback path is the existing Python function,
  unchanged.

## 8. Acceptance summary

Ships when:

- Rust unit tests (3) pass.
- Python parity test (1) passes at 10/10 seeds.
- All 191 fast-fepois tests still pass.
- Medium benchmark median ≤ 0.85 s (or documented thermal-noise miss).
- Crate `__version__` 0.5.0 → 0.6.0; Cargo `0.4.0-alpha.1 → 0.5.0-alpha.1`.

## 9. References

- Correia (2015). "Singletons, Cluster-Robust Standard Errors and Fixed
  Effects." Working paper. Singleton + separation pre-pass theory.
- Correia, Guimarães, Zylkin (2020). PPML-HDFE Stata Journal. The
  underlying algorithm.
- Phase A `_drop_separation` implementation in `fepois.py` lines 294-324
  (Phase 0 ship `b4ba4a3`).
