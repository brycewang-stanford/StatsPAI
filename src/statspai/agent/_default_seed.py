"""Default seed for tool calls that leave a seed argument unset.

About half of the seed-taking estimators default to ``seed=None``, i.e. fresh
entropy on every call. For a person at a prompt that is the usual convention;
for an agent it means that repeating a tool call returns different numbers
and nothing in the first response lets a later reader reproduce it.

The library defaults are not changed here (that would move published seeded
numbers and belongs to a per-estimator migration). The tool dispatcher instead
fills the argument when the caller did not pass it and says so in the
response: ``seed_injected: {param, value, ...}``. The filled value is part of
the recorded arguments, so the ``replay`` snippet and the result card carry
it too.

``STATSPAI_MCP_DEFAULT_SEED`` sets the value (an integer, default 42);
``off`` / ``none`` disables the injection.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, Optional

from .._result_contract import _SEED_PARAMS

DEFAULT_SEED_ENV = "STATSPAI_MCP_DEFAULT_SEED"
DEFAULT_SEED = 42
_OFF = frozenset({"off", "none", "false", "no", ""})
#: ``rng`` takes a Generator on most functions, not an integer.
_INJECTABLE = tuple(p for p in _SEED_PARAMS if p != "rng")


def default_seed() -> Optional[int]:
    """The configured default seed, or ``None`` when injection is off."""
    raw = os.environ.get(DEFAULT_SEED_ENV)
    if raw is None:
        return DEFAULT_SEED
    text = raw.strip().lower()
    if text in _OFF:
        return None
    try:
        return int(text)
    except ValueError:
        # An unreadable setting must not silently change what a call does.
        return None


def inject_default_seed(
    fn: Callable[..., Any], arguments: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Fill the first unset seed argument of ``fn`` in ``arguments``.

    Touches only a parameter that is named like a seed, defaults to ``None``
    and was not passed (an explicit ``null`` from the caller is respected).
    Returns the note to attach to the response, or ``None`` when nothing was
    filled.
    """
    seed = default_seed()
    if seed is None:
        return None
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(name in arguments for name in _SEED_PARAMS):
        return None
    for name in _INJECTABLE:
        param = params.get(name)
        if param is None or param.default is not None:
            continue
        arguments[name] = seed
        return {
            "param": name,
            "value": seed,
            "note": (
                f"{name}= was not passed and defaults to None (fresh "
                "entropy for any stochastic step); the tool layer set "
                f"{name}={seed} so the call can be repeated. Pass {name}= "
                "yourself to choose another "
                "value; a direct Python call keeps the library default."
            ),
        }
    return None


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SEED_ENV",
    "default_seed",
    "inject_default_seed",
]
