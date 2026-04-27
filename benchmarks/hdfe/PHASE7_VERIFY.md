# Phase 7 — Verification Report

**Scope**: GPU-ready backend for the K-way HDFE demean kernel via JAX.

## Deliverables

* New module ``src/statspai/fast/jax_backend.py``.
* New backend choice: ``sp.fast.demean(..., backend="jax")``.
* Diagnostic helper: ``sp.fast.jax_device_info()`` reports the active
  JAX device platform.
* Float64 enabled at jax-config level so numerical results match the
  Rust path (within JIT accumulator-order differences).

## Honest scope statement (the most important section)

The original Phase 7 plan called for:

- **Native cuSPARSE / GPU sparse-matrix HDFE** at 1e9 rows on a single
  A100, with a self-hosted CI runner.

The dev environment that produced this code has **no GPU**. There is no
realistic way for me to ship a tested CUDA path overnight. What I
shipped instead:

1. **Structural backend**: a JAX-on-CPU implementation that mirrors the
   Phase 1 algorithm (per-FE sweep + Aitken extrapolation + double
   threshold). On CPU it is **slower** than the Rust kernel — JAX's
   CPU XLA path is not tuned for the small bincount-style operations
   that dominate this code; the Rust path is. We're explicit about
   this in the docstring.
2. **GPU-readiness contract**: the same code path, when run on a host
   with `jaxlib` built for CUDA, automatically dispatches to the
   accelerator via JAX's standard device routing. A working CPU JAX
   implementation **is** a working GPU JAX implementation modulo the
   `jaxlib-cuda` build (well-tested upstream).
3. **No GPU benchmarks**: the original plan's "1e9 rows < 5 min on a
   single A100" cannot be measured here. It will be measured by
   whoever has GPU access; the target is unchanged.

Why ship this anyway? Because the API contract is the highest-cost
part: getting `backend="jax"` integrated, the dtype handshakes right,
the diagnostic helper, the test scaffolding. With those in place, the
"flip the GPU switch" PR is small.

## Test suite

```
tests/test_fast_jax.py
  test_jax_device_info_string ........ PASSED
  test_jax_demean_matches_rust ....... PASSED  (atol 1e-9 vs Rust)
  test_jax_demean_2d_input ........... PASSED
  test_jax_oneway_exact .............. PASSED
  test_jax_unknown_backend_rejected .. PASSED
  test_jax_when_unavailable_raises ... SKIPPED  (jax is installed locally)
5 passed, 1 skipped in 3.03s
```

## Cumulative regression (Phases 1+2+3+4+5+6+7)

```
121 passed, 2 skipped
```

(The two skips: the no-Rust path in Phase 1 tests, and the no-JAX path
here; both intentionally skip when the optional dep IS installed.)

## Acceptance against original Phase 7 plan

| plan item                                          | status | note |
| -------------------------------------------------- | ------ | ---- |
| ``backend="jax"`` accepted in the demean entry pt  | ✅     | Documented + tested |
| JAX CPU path numerically matches Rust to 1e-9      | ✅     | jit + Aitken + accel_period |
| Float64 enabled (no silent float32 truncation)     | ✅     | `jax.config.update("jax_enable_x64", True)` |
| Diagnostic helper reports active device            | ✅     | `sp.fast.jax_device_info()` |
| ``backend="jax"`` raises clearly when jax missing  | ✅     | `RuntimeError` with install hint |
| Native cuSPARSE / GPU sparse path                  | ⏸     | Requires CUDA hardware + jaxlib-cuda, neither available locally. The JAX backend will run on GPU when those are present, but the code path itself does not optimise for sparse cuSPARSE primitives — that's a follow-up PR. |
| Self-hosted GPU CI runner                          | ⏸     | Cost / infrastructure decision; tracked as v2.1 in the plan, deliberately unaddressed here. |
| 1e9 rows × medium FE on A100 < 5 min               | ⏸     | Cannot be measured without hardware. |
| GPU vs CPU coef agreement to 1e-5                  | ⏸     | Same. |

## Why the JAX path is slower than Rust on CPU (technical note)

For curious readers: the demean kernel does many small bincount-like
sums + scatter-subtractions. The Rust path issues these as tight loops
that the LLVM backend vectorises trivially. JAX's CPU XLA path traces
each `at[].add` into a separate launched kernel; for small N the
launch overhead dominates. On GPU, the launch overhead amortises and
the parallel-friendly nature of bincount is what wins. So the JAX
backend is structurally right for GPU, structurally wrong for CPU
versus a hand-tuned Rust kernel — which is exactly the trade we want
to expose to users.
