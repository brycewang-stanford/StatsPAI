"""
HDFE kernel benchmark harness.

Compares wall-time and correctness of alternative group-demean
implementations (NumPy, Numba-JIT, and — once landed — Rust) on the
same synthetic DGPs, so we can track regressions release-over-release
without re-instrumenting anything.

Example
-------
>>> from statspai.fast import hdfe_bench
>>> res = hdfe_bench(n_list=(1_000, 10_000), n_groups=50, seed=0)
>>> print(res.to_dataframe())

The harness is **tool-agnostic**: a path is measured if its backend is
installed, and quietly marked "unavailable" otherwise. This means the
same command runs on CI where Numba may be missing and on a dev box
where a future Rust wheel is loaded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference implementation: pure NumPy
# ---------------------------------------------------------------------------

def _sweep_numpy(col: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> None:
    """In-place demean by group using numpy ``bincount``."""
    sums = np.bincount(codes, weights=col, minlength=counts.size)
    means = sums / np.maximum(counts, 1)
    col -= means[codes]


def _hdfe_numpy(y: np.ndarray, codes: np.ndarray,
                counts: np.ndarray) -> np.ndarray:
    out = y.astype(np.float64, copy=True)
    _sweep_numpy(out, codes, counts)
    return out


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

@dataclass
class _Backend:
    name: str
    fn: object           # callable (y, codes, counts) -> np.ndarray (copy)
    available: bool
    note: str = ''


def _detect_backends() -> Dict[str, _Backend]:
    """Probe which implementation paths are usable in the current env."""
    backends: Dict[str, _Backend] = {
        'numpy': _Backend(
            name='numpy',
            fn=_hdfe_numpy,
            available=True,
            note='pure NumPy bincount (reference)',
        )
    }

    # Numba path: delegate to the shipping statspai kernel so we measure
    # the exact code users run, not a re-implementation.
    try:
        from ..panel._hdfe_kernels import sweep as _numba_sweep, _HAS_NUMBA

        def _hdfe_numba(y, codes, counts):
            out = np.ascontiguousarray(y, dtype=np.float64).copy()
            _numba_sweep(out, np.asarray(codes, dtype=np.int64),
                         np.asarray(counts, dtype=np.int64))
            return out

        backends['numba'] = _Backend(
            name='numba',
            fn=_hdfe_numba,
            available=bool(_HAS_NUMBA),
            note=('Numba @njit path (statspai 0.9.3+)' if _HAS_NUMBA
                  else 'Numba not installed; path falls back to NumPy'),
        )
    except ImportError:  # pragma: no cover - defensive
        backends['numba'] = _Backend(
            name='numba', fn=_hdfe_numpy, available=False,
            note='statspai.panel._hdfe_kernels missing',
        )

    # Rust path: present for forward-compat but marked unavailable
    # until the PyO3 wheel lands (see spec 2026-04-20-v095-rust-hdfe-spike).
    try:
        from statspai_hdfe import group_demean as _rust_group_demean  # type: ignore

        def _hdfe_rust(y, codes, counts):
            out = np.ascontiguousarray(y, dtype=np.float64).copy()
            sums = np.zeros(counts.size, dtype=np.float64)
            _rust_group_demean(
                np.asarray(codes, dtype=np.int64),
                out, sums,
                np.asarray(counts, dtype=np.int64),
            )
            return out

        backends['rust'] = _Backend(
            name='rust', fn=_hdfe_rust, available=True,
            note='PyO3 + Rayon kernel (statspai_hdfe crate)',
        )
    except Exception:
        backends['rust'] = _Backend(
            name='rust', fn=_hdfe_numpy, available=False,
            note='Rust kernel not built (expected pre-1.0)',
        )

    return backends


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HDFEBenchResult:
    """Output of :func:`hdfe_bench`.

    Attributes
    ----------
    results : list of dict
        One row per (backend, n) combination with ``wall_time_s``,
        ``relative_to_numpy``, and correctness metadata.
    backends : dict of name -> _Backend
        Availability / notes for each backend probed.
    reference : str
        Name of the reference backend used for correctness comparison
        (always ``'numpy'``).
    """
    results: List[Dict[str, object]]
    backends: Dict[str, _Backend]
    reference: str = 'numpy'

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results)
        if not df.empty:
            df = df.sort_values(['n', 'backend']).reset_index(drop=True)
        return df

    def summary(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return 'hdfe_bench: no data.'
        lines = ['hdfe_bench']
        for name, b in self.backends.items():
            status = 'AVAILABLE' if b.available else 'unavailable'
            lines.append(f'  [{status:11s}] {name:6s} — {b.note}')
        lines.append('')
        lines.append(df.to_string(index=False))
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hdfe_bench(
    n_list: Sequence[int] = (1_000, 10_000, 100_000),
    n_groups: int = 50,
    repeat: int = 3,
    seed: int = 0,
    atol: float = 1e-10,
) -> HDFEBenchResult:
    """Benchmark group-demean wall time across available backends.

    Parameters
    ----------
    n_list : sequence of int
        Sample sizes to time at.
    n_groups : int, default 50
        Number of FE groups for the synthetic DGP. Code uniformly
        over [0, n_groups).
    repeat : int, default 3
        Number of timing repetitions per (backend, n). The reported
        ``wall_time_s`` is the minimum across repeats.
    seed : int, default 0
        RNG seed for the synthetic DGP.
    atol : float, default 1e-10
        Absolute tolerance used when checking that each non-reference
        backend agrees with the NumPy reference.

    Returns
    -------
    HDFEBenchResult
        Results + backend availability metadata.
    """
    backends = _detect_backends()
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    for n in n_list:
        codes = rng.integers(0, n_groups, size=n).astype(np.int64)
        counts = np.bincount(codes, minlength=n_groups).astype(np.int64)
        # Ensure no zero-count groups (avoid divide-by-zero in numpy path)
        counts = np.maximum(counts, 1)
        y = rng.normal(size=n).astype(np.float64)

        # Reference result for correctness comparison
        ref_out = backends[_Backend.__dataclass_fields__['name'].default
                           if False else 'numpy'].fn(y, codes, counts)  # type: ignore  # noqa: E501

        for name, b in backends.items():
            if not b.available:
                rows.append({
                    'backend': name,
                    'n': n,
                    'n_groups': n_groups,
                    'wall_time_s': np.nan,
                    'relative_to_numpy': np.nan,
                    'max_abs_err_vs_ref': np.nan,
                    'available': False,
                    'note': b.note,
                })
                continue

            best = float('inf')
            max_err = 0.0
            for _ in range(repeat):
                y_in = y.copy()
                t0 = time.perf_counter()
                out = b.fn(y_in, codes, counts)
                best = min(best, time.perf_counter() - t0)
                if name != 'numpy':
                    err = float(np.max(np.abs(out - ref_out)))
                    max_err = max(max_err, err)

            rows.append({
                'backend': name,
                'n': n,
                'n_groups': n_groups,
                'wall_time_s': best,
                'relative_to_numpy': np.nan,  # filled below
                'max_abs_err_vs_ref': max_err,
                'available': True,
                'note': b.note,
            })

    # Fill relative_to_numpy column
    df = pd.DataFrame(rows)
    if not df.empty:
        numpy_times = (df[df['backend'] == 'numpy']
                       .set_index('n')['wall_time_s'])
        rel = df.apply(
            lambda r: (r['wall_time_s'] / numpy_times.get(r['n'], np.nan))
            if np.isfinite(r['wall_time_s']) else np.nan,
            axis=1,
        )
        df['relative_to_numpy'] = rel
        rows = df.to_dict(orient='records')

    # Correctness guard — fail loud if a backend drifted outside atol
    for r in rows:
        if r['available'] and r['backend'] != 'numpy':
            err = r['max_abs_err_vs_ref']
            if np.isfinite(err) and err > atol:
                raise AssertionError(
                    f"Backend '{r['backend']}' drifted {err:.2e} vs "
                    f"numpy reference (atol={atol:.1e}) on n={r['n']}."
                )

    return HDFEBenchResult(results=rows, backends=backends, reference='numpy')
