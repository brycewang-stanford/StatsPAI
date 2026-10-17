"""
``statspai.fast`` — performance-instrumentation and native-kernel home.

Contents (v1.8 / Phase 1+):

- :func:`hdfe_bench` — wall-time + correctness regression harness.
- :func:`demean`     — multi-way HDFE within-transform, backed by the
  optional Rust kernels (NumPy fallback).
- :func:`backend_info` / :func:`set_num_threads` / :func:`get_num_threads`
  — whether the Rust kernels (``pip install "statspai[rust]"``) are in
  use, and the size of their thread pool.

The module exposes building blocks that Phase 2+ (PPML / GLM HDFE),
Phase 3 (`sp.within`), and Phase 5 (Polars/Arrow direct) sit on top of.
"""

from ._backend import backend_info, get_num_threads, set_num_threads
from .bench import HDFEBenchResult, hdfe_bench
from .demean import DemeanInfo, demean
from .dsl import csw, fe_interact, i, sw
from .etable import etable
from .event_study import EventStudyResult, event_study
from .feols import FeolsResult, feols
from .fepois import FePoisResult, fepois
from .inference import (
    BootTestResult,
    BootWaldResult,
    WaldTestResult,
    boottest,
    boottest_wald,
    cluster_dof_bm,
    cluster_dof_wald_bm,
    cluster_dof_wald_htz,
    cluster_wald_htz,
    crve,
)
from .within import WithinTransformer, within

# Optional JAX backend — exposes diagnostic helper at module level.
try:
    from .jax_backend import jax_device_info  # noqa: F401
except ImportError:  # pragma: no cover

    def jax_device_info() -> str:
        return "jax: not installed"


# JAX-backed end-to-end feols (Phase 4). Drops in for sp.fast.feols on
# CUDA/TPU; CPU JAX path is correctness-grade but typically slower than
# the Rust/numpy default. Lazy-load — module import stays jax-free.
try:
    from .jax_feols import (  # noqa: F401
        FeolsBootstrapResult,
        feols_jax,
        feols_jax_bootstrap,
    )

    _HAS_JAX_FEOLS = True
except ImportError:  # pragma: no cover
    # Re-export the stubs from a dedicated module so ``feols_jax`` &c. stay
    # single-binding import-aliases (not an alias shadowed by a same-name
    # ``def``). The shadowing form trips griffe's alias resolver and breaks
    # ``mkdocs build --strict``; behaviour is byte-identical either way.
    from ._jax_fallback import (  # type: ignore[assignment]  # noqa: F401
        FeolsBootstrapResult,
        feols_jax,
        feols_jax_bootstrap,
    )

    _HAS_JAX_FEOLS = False

# Torch device diagnostic — mirrors jax_device_info for the optional
# neural backends (deepiv / neural_causal / cevae). See
# ``utils/_torch_device.py`` for the resolution policy.
from ..utils._torch_device import torch_device_info  # noqa: F401

# Polars / Arrow direct path is optional; only loaded if polars is installed.
try:
    from .polars_io import demean_polars, fepois_polars  # noqa: F401

    _HAS_POLARS_IO = True
except ImportError:  # pragma: no cover
    demean_polars = None  # type: ignore
    fepois_polars = None  # type: ignore
    _HAS_POLARS_IO = False

__all__ = [
    "backend_info",
    "set_num_threads",
    "get_num_threads",
    "hdfe_bench",
    "HDFEBenchResult",
    "demean",
    "DemeanInfo",
    "fepois",
    "FePoisResult",
    "feols",
    "FeolsResult",
    "within",
    "WithinTransformer",
    "i",
    "fe_interact",
    "sw",
    "csw",
    "crve",
    "boottest",
    "BootTestResult",
    "boottest_wald",
    "BootWaldResult",
    "cluster_dof_bm",
    "cluster_dof_wald_bm",
    "cluster_dof_wald_htz",
    "cluster_wald_htz",
    "WaldTestResult",
    "event_study",
    "EventStudyResult",
    "jax_device_info",
    "torch_device_info",
    "feols_jax",
    "feols_jax_bootstrap",
    "FeolsBootstrapResult",
    "etable",
    "demean_polars",
    "fepois_polars",
]
