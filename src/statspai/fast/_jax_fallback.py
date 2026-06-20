"""Import-only fallbacks for the optional JAX ``feols`` backend.

When ``jax`` is not installed, :mod:`statspai.fast` re-exports the
``feols_jax`` / ``feols_jax_bootstrap`` / ``FeolsBootstrapResult`` symbols
from this module instead of defining them inline in ``__init__``. Keeping
the stubs here means the package namespace binds those names to a single
import-alias in both branches (real backend vs. fallback), rather than an
import-alias shadowed by a same-name ``def`` — the latter trips griffe's
alias resolver and breaks ``mkdocs build --strict`` (the stub's own
parameter annotations become unresolvable). Behaviour is identical: each
symbol raises a helpful ``ImportError`` pointing at ``pip install jax``.
"""

from __future__ import annotations

from typing import Any, Optional

from .feols import FeolsResult

__all__ = ["feols_jax", "feols_jax_bootstrap", "FeolsBootstrapResult"]


def feols_jax(
    formula: str,
    data: Any,
    *,
    vcov: str = "iid",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
    dtype: str = "float64",
) -> FeolsResult:
    raise ImportError(
        "jax is not installed; pip install jax jaxlib to enable "
        "feols_jax. Plain sp.fast.feols runs without JAX."
    )


class FeolsBootstrapResult:  # type: ignore[no-redef]
    """Placeholder exported when the optional JAX backend is unavailable."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ImportError(
            "jax is not installed; pip install jax jaxlib to enable "
            "FeolsBootstrapResult."
        )


def feols_jax_bootstrap(
    formula: str,
    data: Any,
    *,
    n_boot: int = 1_000,
    seed: int = 0,
    bootstrap: str = "pairs",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    drop_singletons: bool = True,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
    ci_alpha: float = 0.05,
    vmap_chunk_size: int = 200,
    dtype: str = "float64",
) -> "FeolsBootstrapResult":
    raise ImportError(
        "jax is not installed; pip install jax jaxlib to enable " "feols_jax_bootstrap."
    )
