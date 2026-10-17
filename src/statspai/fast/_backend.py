"""Public face of the optional Rust kernels: status and thread pool."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import _rust_backend


def backend_info() -> Dict[str, Any]:
    """Report whether the Rust kernels are in use, and why not if they are not.

    The kernels live in the optional ``statspai-hdfe`` package
    (``pip install "statspai[rust]"``). Without it every ``sp.fast``
    estimator runs its NumPy / Numba path; the results agree with the
    Rust path to within the residualisation tolerance.

    Returns
    -------
    dict
        ``available`` (bool), ``version`` and ``api_version`` of the
        installed extension (``None`` if absent), ``min_api_version``
        StatsPAI needs, ``threads`` the kernels run on, ``reason`` they are
        not used (``None`` when they are), ``location`` of the extension
        and, when unavailable, the ``install`` command.

    Examples
    --------
    >>> import statspai as sp
    >>> info = sp.fast.backend_info()
    >>> isinstance(info["available"], bool)
    True
    """
    return _rust_backend.info()


def set_num_threads(n: int) -> None:
    """Set the number of threads the Rust kernels use.

    Parameters
    ----------
    n : int
        Thread count; ``0`` restores the default (one per logical CPU, or
        ``RAYON_NUM_THREADS``). The environment variable
        ``STATSPAI_NUM_THREADS`` sets the initial size.

    Raises
    ------
    MissingDependencyError
        The Rust kernels are not installed.

    Examples
    --------
    >>> import statspai as sp
    >>> if sp.fast.backend_info()["available"]:
    ...     sp.fast.set_num_threads(2)
    """
    _rust_backend.set_num_threads(n)


def get_num_threads() -> Optional[int]:
    """Number of threads the Rust kernels use (``None`` without them).

    Examples
    --------
    >>> import statspai as sp
    >>> n = sp.fast.get_num_threads()
    """
    return _rust_backend.get_num_threads()
