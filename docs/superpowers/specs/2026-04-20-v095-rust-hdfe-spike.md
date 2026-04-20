# StatsPAI · Rust HDFE inner kernel spike (post-0.9.5 track)

**Author:** Bryce Wang · **Date:** 2026-04-20 · **Status:** design (not yet implemented)

> This spec converts Section 8's Rust commitment into a concrete,
> phased plan. v0.9.5 ships only the benchmark harness and the
> design; the actual PyO3 crate and wheel-build CI land on a
> dedicated branch and are scoped for 1.0.

## 1. Why Rust (again)

Numba-JIT v0.9.3 brought HDFE absorbing to ~3× the pure-NumPy baseline
on 200k-row × 3-FE × 2-regressor designs. That is fine for 90% of
applied-econ workloads (10k → 5M rows). It is **not** enough to catch
`fixest` on ≥10M-row panels, where Bergé's C++ + OpenMP engine runs
multi-threaded with vectorised inner loops.

The three bottlenecks left in the numba path:

1. **Python-level orchestration** of the alternating-projection outer
   loop (small GIL overhead per iteration).
2. **Single-threaded** demeaning inside each FE pass — numba `@njit`
   does not parallelise unless we add `prange`, which triggers GIL
   contention in the absorber.
3. **No SIMD guarantee** on Apple Silicon / Windows MSVC cross-compile
   — numba's SIMD story is LLVM-version-dependent.

Rust + PyO3 + Rayon fixes all three with one toolchain.

## 2. Goals for the port

| Goal | Target |
|---|---|
| Speed on 1M-row × 3-FE × 2-regressor panel | ≥ 5× numba baseline |
| Speed on 10M-row × 3-FE × 5-regressor panel | within 20 % of fixest |
| Correctness | bit-identical β̂ and residuals vs numba on 16+ DGPs |
| Cross-platform wheels | macOS arm64, macOS x86_64, Linux x86_64, Linux arm64, Windows x86_64 |
| Fallback | pure-Python path remains live — users without Rust wheels keep working |

## 3. Phasing

**Phase 0 — benchmark harness (this release, v0.9.5).**
Ship `statspai.fast.bench` so we can track wall-time + correctness
apples-to-apples across NumPy, Numba, and (later) Rust paths. No Rust
toolchain required.

**Phase 1 — scaffold the Rust crate (next branch).**
- `rust/statspai_hdfe/Cargo.toml` — crate scaffold with `pyo3` and
  `rayon`.
- `rust/statspai_hdfe/src/lib.rs` — Python-facing module with a
  single function: `group_demean(codes: PyReadonlyArray1<i64>, y:
  PyReadwriteArray1<f64>, sums: PyReadwriteArray1<f64>, counts:
  PyReadonlyArray1<i64>)`.
- `pyproject.toml` adds `maturin` to `[build-system].requires`.
- Fall back to numba when the compiled extension is missing
  (`ImportError` caught inside `statspai.panel.hdfe`).

**Phase 2 — wire into HDFE absorber.**
- Replace the single-FE inner loop with the Rust call.
- Keep the outer alternating-projection loop in Python so that we
  don't have to port the Irons-Tuck acceleration or the singleton
  pruning logic.

**Phase 3 — parallelise across regressors with Rayon.**
- `demean_block(codes, YX: &mut ndarray::Array2<f64>, sums, counts)`
  using `rayon::join` or a parallel iterator across columns.
- Benchmark: expect 2–3× on top of Phase 2 on machines with ≥4 cores.

**Phase 4 — CI, wheels, docs.**
- `cibuildwheel` matrix (see table below).
- `pipx run maturin build --release` on each matrix row.
- Upload sdist + wheels to PyPI.

**Phase 5 — absorb remaining hot paths.**
- Multiway-cluster Cameron-Gelbach-Miller sum (shipped in v0.9.4's
  NumPy path); Rust version using manual SIMD.
- Wild-cluster bootstrap inner loop.

## 4. Wheel build matrix

| Python | OS | Architecture | Builder | Status |
|---|---|---|---|---|
| 3.9-3.13 | macOS 12+ | arm64 | cibuildwheel + maturin | ⏳ Phase 4 |
| 3.9-3.13 | macOS 12+ | x86_64 | cibuildwheel + maturin | ⏳ Phase 4 |
| 3.9-3.13 | manylinux_2_17 | x86_64 | cibuildwheel + maturin | ⏳ Phase 4 |
| 3.9-3.13 | manylinux_2_17 | aarch64 | cibuildwheel + maturin | ⏳ Phase 4 |
| 3.9-3.13 | Windows | x86_64 | cibuildwheel + maturin | ⏳ Phase 4 |
| 3.9-3.13 | musllinux_1_2 | x86_64 | cibuildwheel + maturin | ⏳ Phase 4 |

If any row fails, we ship a pure-Python sdist for that row, and the
user gets the numba path automatically. **This graceful degradation
is the single most important API guarantee of the rollout.**

## 5. FFI surface — proposed

```rust
// rust/statspai_hdfe/src/lib.rs (sketch)
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use rayon::prelude::*;

#[pyfunction]
fn group_demean(
    py: Python<'_>,
    codes: PyReadonlyArray1<'_, i64>,
    y: PyReadwriteArray1<'_, f64>,
    sums: PyReadwriteArray1<'_, f64>,
    counts: PyReadonlyArray1<'_, i64>,
) -> PyResult<()> {
    // 1) accumulate sums[g] += y[i] into thread-local buckets, reduce.
    // 2) divide by counts[g] -> group means.
    // 3) subtract sums[codes[i]] from y[i].
    // ...
    Ok(())
}

#[pymodule]
fn statspai_hdfe(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(group_demean, m)?)?;
    Ok(())
}
```

Python side:

```python
# src/statspai/panel/hdfe_rust.py
try:
    from statspai_hdfe import group_demean as _rust_group_demean
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

def group_demean(codes, y, sums, counts):
    if HAS_RUST:
        _rust_group_demean(codes, y, sums, counts)
    else:
        _numba_group_demean(codes, y, sums, counts)
```

## 6. Performance contract

Every claim in this spec is backed by `statspai.fast.bench.hdfe_bench(...)`.
Before the Rust code lands, the contract is:

1. Numba baseline must beat pure NumPy by ≥ 2× on all benchmarks.
2. When Rust lands, Rust must beat Numba by ≥ 3× on 1M-row × 3-FE
   DGPs.
3. Neither path is allowed to land with β̂ drift > 1e-8 in absolute
   terms vs the reference Python implementation.

If (2) is ever broken at release time, we postpone the release and
publish the benchmark regression in the commit message.

## 7. What we SHIP in v0.9.5

- This spec (the file you are reading).
- `src/statspai/fast/bench.py` — a pure-Python benchmark harness.
- `src/statspai/fast/__init__.py` — wires the harness into the
  package so users / CI can call `sp.fast.hdfe_bench(...)`.
- Four tests in `tests/test_fast_bench.py` covering:
  1. `hdfe_bench(...).results` shape is `(n_paths × n_sizes)`.
  2. The numba path's β̂ matches the numpy path's β̂ to 1e-8.
  3. The harness skips paths that are not installed (e.g., numba
     missing) rather than crashing.
  4. A "dry run" mode (`n=100`) finishes in <1 s so CI can include it.

## 8. What we do NOT ship in v0.9.5

- The Rust crate itself.
- `maturin` in `pyproject.toml`.
- Any `cibuildwheel` config.
- Any build-system change that could break `pip install statspai` for
  a user with a default toolchain.

## 9. Risks

- **Silent ABI drift.** Rust-Python ABI is not stable across PyO3
  versions; pin PyO3 and bump deliberately.
- **Wheel matrix bitrot.** When Python 3.14 ships, manylinux
  tags migrate; we must re-test.
- **numpy 2.x vs ndarray interop.** Must use `numpy ^2.0` feature flag
  on `numpy` crate (phase 1 decision).
- **Parallel-FE determinism.** Rayon's work-stealing makes iteration
  order non-deterministic; accumulator order must be insensitive.
  We enforce this via float-reproducibility tests pegged at
  `atol=1e-10` for `1M` rows.

## 10. Owner

- **Bryce Wang** — directional reviewer + final-mile bencher.
- Open to external contributors: any PR that lands a working
  `group_demean` plus CI green on macOS + manylinux will be merged
  before Python-side integration.
