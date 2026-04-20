"""Shared benchmark utilities."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable


@contextmanager
def timer():
    """Context manager that captures elapsed wall-clock time.

    Yields a list with one float (elapsed seconds), appended on exit.
    Usage:
        with timer() as t:
            f()
        print(t[0])
    """
    start = time.perf_counter()
    out = []
    yield out
    out.append(time.perf_counter() - start)


def bench(fn: Callable, n_warmup: int = 1, n_runs: int = 3) -> dict:
    """Run ``fn()`` with warmup + repeated timing.

    Returns ``{'mean_s': ..., 'min_s': ..., 'max_s': ..., 'runs': ...}``.
    """
    for _ in range(n_warmup):
        fn()
    runs = []
    for _ in range(n_runs):
        with timer() as t:
            fn()
        runs.append(t[0])
    return {
        'mean_s': sum(runs) / len(runs),
        'min_s': min(runs),
        'max_s': max(runs),
        'runs': runs,
    }


def fmt_ms(seconds: float) -> str:
    """Format elapsed seconds as an adaptive human-readable string."""
    if seconds < 1e-3:
        return f"{seconds*1e6:.0f}µs"
    if seconds < 1.0:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds:.2f}s"


def speedup_label(baseline: float, comparison: float) -> str:
    """Return '2.5x faster' / '1.3x slower' / 'parity'."""
    if baseline <= 0 or comparison <= 0:
        return '—'
    ratio = comparison / baseline
    if 0.9 <= ratio <= 1.1:
        return 'parity'
    if ratio > 1:
        return f"{ratio:.1f}x faster"
    return f"{1/ratio:.1f}x slower"
