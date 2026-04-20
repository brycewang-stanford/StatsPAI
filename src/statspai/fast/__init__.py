"""
``statspai.fast`` — performance-instrumentation and future native-kernel home.

v0.9.5 ships only the benchmark harness (:func:`hdfe_bench`), used to
track wall-time + correctness regressions across kernel paths (pure
NumPy, Numba-JIT, and — once landed — Rust).

See ``docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md`` for
the phased plan.
"""
from .bench import hdfe_bench, HDFEBenchResult

__all__ = [
    'hdfe_bench',
    'HDFEBenchResult',
]
