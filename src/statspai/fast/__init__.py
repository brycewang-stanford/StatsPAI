"""
``statspai.fast`` — performance-instrumentation and native-kernel home.

Contents (v1.8 / Phase 1+):

- :func:`hdfe_bench` — wall-time + correctness regression harness.
- :func:`demean`     — multi-way HDFE within-transform with Aitken
  acceleration, backed by a Rust kernel (NumPy fallback).

The module exposes building blocks that Phase 2+ (PPML / GLM HDFE),
Phase 3 (`sp.within`), and Phase 5 (Polars/Arrow direct) sit on top of.
"""
from .bench import hdfe_bench, HDFEBenchResult
from .demean import demean, DemeanInfo
from .fepois import fepois, FePoisResult
from .within import within, WithinTransformer
from .dsl import i, fe_interact, sw, csw
from .inference import (
    crve,
    boottest,
    BootTestResult,
    boottest_wald,
    BootWaldResult,
)
from .event_study import event_study, EventStudyResult
from .etable import etable

# Optional JAX backend — exposes diagnostic helper at module level.
try:
    from .jax_backend import jax_device_info  # noqa: F401
except ImportError:  # pragma: no cover
    def jax_device_info() -> str:
        return "jax: not installed"

# Polars / Arrow direct path is optional; only loaded if polars is installed.
try:
    from .polars_io import demean_polars, fepois_polars  # noqa: F401
    _HAS_POLARS_IO = True
except ImportError:  # pragma: no cover
    demean_polars = None  # type: ignore
    fepois_polars = None  # type: ignore
    _HAS_POLARS_IO = False

__all__ = [
    'hdfe_bench',
    'HDFEBenchResult',
    'demean',
    'DemeanInfo',
    'fepois',
    'FePoisResult',
    'within',
    'WithinTransformer',
    'i',
    'fe_interact',
    'sw',
    'csw',
    'crve',
    'boottest',
    'BootTestResult',
    'boottest_wald',
    'BootWaldResult',
    'event_study',
    'EventStudyResult',
    'jax_device_info',
    'etable',
    'demean_polars',
    'fepois_polars',
]
