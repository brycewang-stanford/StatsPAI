"""Deterministic-RNG context manager for reproducible agent loops.

``sp.session(seed=42)`` snapshots the RNG state of Python's
``random`` and NumPy's **legacy global** MT19937 generator (the one
backing ``np.random.randn``, ``np.random.choice``, etc.), applies the
new seed for the duration of the ``with`` block, and restores the
prior state on exit. Optional extras (PyTorch, JAX) are seeded only
when those libraries are already importable — never auto-installed.

What it does NOT cover
----------------------
``np.random.default_rng()`` creates a fresh PCG64 generator seeded
from OS entropy each time it is called — those generators have no
process-global state for ``sp.session`` to manipulate. If your code
calls ``rng = np.random.default_rng()`` inside the block, the draws
will be different on every run regardless of the session seed. To
get deterministic ``default_rng`` draws, pass the seed explicitly:

>>> with sp.session(seed=42) as state:
...     rng = np.random.default_rng(state.seed)   # explicit seed
...     x = rng.normal(size=5)

Threading
---------
Not thread-safe. The snapshot lives in a context-manager local but
the *target* state (Python ``random`` and NumPy legacy globals) is
process-wide. Two threads that enter ``sp.session`` simultaneously
will trample each other's snapshots and produce non-deterministic
results with no error. For parallel workloads, instantiate
``np.random.default_rng(seed)`` per-thread and thread the generator
through your call stack instead of relying on ``sp.session``.

The point: agents iterate. A bootstrap CI that drifts between calls
because different RNG state was active is a debugging nightmare.
``with sp.session(seed=42): ...`` makes every call reproducible
without polluting the global RNG state outside the block.

Usage
-----

>>> import statspai as sp
>>> import numpy as np
>>>
>>> with sp.session(seed=42):
...     a = np.random.randn(3)
...     # any sp.xxx call inside is deterministic
>>>
>>> # State outside the block is untouched.
>>> b = np.random.randn(3)  # uses prior global state
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any, Iterator, Optional


@contextlib.contextmanager
def session(seed: Optional[int] = None,
            *,
            torch: bool = True,
            jax: bool = True,
            pythonhashseed: bool = False) -> Iterator[Any]:
    """Set every reachable RNG to a known seed for the duration of the
    ``with`` block, then restore the prior state on exit.

    Parameters
    ----------
    seed : int, optional
        Seed value. ``None`` (the default) means "snapshot current
        state but don't reseed" — useful for opportunistic save /
        restore around code that you don't want to leak RNG drift.
    torch : bool, default ``True``
        Seed PyTorch (CPU + CUDA) when the library is already
        imported. Never imports torch on its own.
    jax : bool, default ``True``
        Yield a fresh JAX ``PRNGKey(seed)`` to the caller via the
        ``jax_key`` attribute on the yielded session object, when
        JAX is already imported. Never imports jax on its own.
        (JAX has no global state so we can't seed it — agents must
        thread the key explicitly.)
    pythonhashseed : bool, default ``False``
        Set ``PYTHONHASHSEED`` for the duration of the block. Most
        causal-inference numerics don't depend on dict iteration
        order, but spec-curve enumerators and graph-based DAG search
        sometimes do. Off by default to avoid surprising downstream
        callers.

    Yields
    ------
    SessionState
        Object exposing the seed in use as ``.seed`` and (when JAX
        is imported) a ``.jax_key`` PRNGKey. Tests / agents can hold
        a reference to verify the seed inside the block.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> with sp.session(seed=42):
    ...     a = np.random.randn(3)
    >>> with sp.session(seed=42):
    ...     b = np.random.randn(3)
    >>> bool((a == b).all())
    True

    Notes
    -----
    Restoration is best-effort: if a library was lazily imported
    INSIDE the ``with`` block (and thus had no prior state), the
    exit handler skips restoring it. The intended use is small,
    deterministic blocks of estimator + bootstrap calls — not
    long-running session orchestration.
    """

    class _SessionState:
        __slots__ = ("seed", "jax_key")

    state = _SessionState()
    state.seed = seed
    state.jax_key = None

    # ---- snapshot prior state ---- #
    snapshots: dict = {}

    # Python ``random``
    snapshots["random"] = random.getstate()

    # NumPy global state. ``np.random`` carries a legacy global
    # MT19937 generator; modern code uses ``np.random.default_rng()``.
    # We snapshot the legacy state — it's what users actually see
    # leaking when they call ``np.random.randn()``.
    try:
        import numpy as np
        snapshots["numpy_legacy"] = np.random.get_state()
    except ImportError:  # pragma: no cover — numpy is a hard dep
        np = None  # type: ignore[assignment]

    # PYTHONHASHSEED — only restored if the caller asked us to set it.
    snapshots["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")

    # PyTorch — snapshot only if already imported.
    import sys
    if torch and "torch" in sys.modules:
        try:
            _torch = sys.modules["torch"]
            snapshots["torch_cpu"] = _torch.random.get_rng_state()
            if (hasattr(_torch, "cuda")
                    and _torch.cuda.is_available()):
                snapshots["torch_cuda"] = (
                    _torch.cuda.get_rng_state_all())
        except Exception:
            pass

    # ---- apply new seed ---- #
    if seed is not None:
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        if pythonhashseed:
            os.environ["PYTHONHASHSEED"] = str(seed)
        if torch and "torch" in sys.modules:
            try:
                _torch = sys.modules["torch"]
                _torch.manual_seed(seed)
                if (hasattr(_torch, "cuda")
                        and _torch.cuda.is_available()):
                    _torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
        if jax and "jax" in sys.modules:
            try:
                import jax  # type: ignore[import-not-found]
                state.jax_key = jax.random.PRNGKey(seed)
            except Exception:
                pass

    try:
        yield state
    finally:
        # ---- restore prior state ---- #
        random.setstate(snapshots["random"])
        if np is not None and "numpy_legacy" in snapshots:
            np.random.set_state(snapshots["numpy_legacy"])
        if pythonhashseed:
            prior = snapshots["python_hash_seed"]
            if prior is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = prior
        if "torch_cpu" in snapshots:
            try:
                _torch = sys.modules["torch"]
                _torch.random.set_rng_state(snapshots["torch_cpu"])
                if "torch_cuda" in snapshots:
                    _torch.cuda.set_rng_state_all(snapshots["torch_cuda"])
            except Exception:
                pass


__all__ = ["session"]
