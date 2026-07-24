"""Global-RNG isolation helpers.

Some estimators (neural / DML / structure-learning back ends) seed the
*global* legacy NumPy RNG (and, for torch back ends, the global torch RNG)
because the third-party libraries they wrap read from it. Doing so silently
resets the caller's own random stream — a reproducibility footgun that is very
hard to trace back to StatsPAI (CLAUDE.md §7: fail loudly / do not surprise the
caller).

``seeded_global_rng`` seeds the global RNG(s) for the duration of a ``with``
block and restores the caller's prior state on exit, so the estimator gets its
deterministic stream *without* leaking the reseed into the caller's session.
The internal random draws are byte-for-byte identical to a bare
``np.random.seed(seed)`` — only the after-effect on the global state changes —
so this carries no numerical migration.
"""

from __future__ import annotations

import contextlib
import functools
import sys
from typing import Any, Callable, Iterator, Optional, TypeVar

import numpy as np

__all__ = ["seeded_global_rng", "preserve_global_rng"]

_F = TypeVar("_F", bound=Callable[..., Any])


def preserve_global_rng(fn: _F) -> _F:
    """Decorator: restore the global NumPy/torch RNG state after ``fn`` runs.

    ``fn`` is free to call ``np.random.seed(...)`` / ``torch.manual_seed(...)``
    internally for deterministic behavior; this wrapper snapshots the caller's
    global RNG state on entry and restores it on exit, so the reseed does not
    leak into the caller's session. Internal random draws are unchanged, so
    this carries no numerical migration.

    torch state is snapshotted only when torch is already imported (checked via
    ``sys.modules``), so the decorator never forces a torch import.
    """

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        np_state = np.random.get_state()
        torch_mod = sys.modules.get("torch")
        torch_cpu_state = None
        torch_cuda_states = None
        if torch_mod is not None:
            try:
                torch_cpu_state = torch_mod.get_rng_state()
                if torch_mod.cuda.is_available():
                    torch_cuda_states = torch_mod.cuda.get_rng_state_all()
            except Exception:
                torch_cpu_state = None
                torch_cuda_states = None
        try:
            return fn(*args, **kwargs)
        finally:
            np.random.set_state(np_state)
            if torch_mod is not None and torch_cpu_state is not None:
                try:
                    torch_mod.set_rng_state(torch_cpu_state)
                    if torch_cuda_states is not None:
                        torch_mod.cuda.set_rng_state_all(torch_cuda_states)
                except Exception:
                    pass

    return _wrapper  # type: ignore[return-value]


@contextlib.contextmanager
def seeded_global_rng(
    seed: Optional[int], *, torch_seed: bool = False
) -> Iterator[None]:
    """Seed the global NumPy (and optionally torch) RNG, then restore it.

    Parameters
    ----------
    seed : int or None
        Seed for the global legacy NumPy RNG. ``None`` leaves NumPy's global
        state untouched (but the caller's state is still snapshotted/restored,
        which is a no-op).
    torch_seed : bool, default False
        Also snapshot/seed/restore torch's global CPU (and CUDA, if
        initialised) RNG. Only set this when torch is already imported by the
        caller; the helper imports torch lazily and skips it if unavailable.

    Yields
    ------
    None
        Runs the ``with`` body with the global RNG(s) seeded.
    """
    np_state = np.random.get_state()

    torch_mod = None
    torch_cpu_state = None
    torch_cuda_states = None
    if torch_seed:
        try:
            import torch as torch_mod  # type: ignore
        except Exception:
            torch_mod = None
        if torch_mod is not None:
            torch_cpu_state = torch_mod.get_rng_state()
            if torch_mod.cuda.is_available():
                torch_cuda_states = torch_mod.cuda.get_rng_state_all()

    try:
        if seed is not None:
            np.random.seed(seed)
            if torch_mod is not None:
                torch_mod.manual_seed(int(seed))
        yield
    finally:
        np.random.set_state(np_state)
        if torch_mod is not None and torch_cpu_state is not None:
            torch_mod.set_rng_state(torch_cpu_state)
            if torch_cuda_states is not None:
                torch_mod.cuda.set_rng_state_all(torch_cuda_states)
