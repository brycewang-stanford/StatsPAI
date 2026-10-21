"""Retired keyword spellings that keep working for a deprecation cycle.

Kept apart from :mod:`statspai._aliases` on purpose: that module's wrapper
runs on almost every estimation path, so every change to it re-stales the
Track A provenance traces of every module. This one runs only for the
functions that use it.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["deprecated_keywords"]


def deprecated_keywords(*, removed_in: str, **old_to_new: str) -> Callable[[F], F]:
    """Keep accepting a retired keyword, with a ``DeprecationWarning``.

    ``@deprecated_keywords(removed_in="1.42", group="treat")`` reads as
    "``group=`` still works, is forwarded to ``treat=``, and warns that it
    goes away no earlier than 1.42". Only the keyword spelling warns; a
    positional argument binds to the parameter by position and is silent.

    Apply it outermost (above :func:`statspai._aliases.accepts_aliases`),
    so the warning's ``stacklevel=2`` points at the caller's line. The old
    spellings stay in ``__statspai_aliases__`` (tool layers that check
    accepted names keep accepting them) and are also listed in
    ``__statspai_deprecated_aliases__``, which the schema layer reads to
    leave them out of the advertised spellings.
    """
    if not old_to_new:
        raise ValueError("deprecated_keywords requires at least one old=new pair")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in old_to_new.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise TypeError(
                        f"{func.__name__}() received both '{old}' and "
                        f"'{new}'; pass only '{new}'."
                    )
                kwargs[new] = kwargs.pop(old)
                warnings.warn(
                    f"{func.__name__}(): the keyword '{old}=' is deprecated "
                    f"and will be removed no earlier than StatsPAI "
                    f"{removed_in}; pass '{new}=' instead (same column, same "
                    "result).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        aliases = dict(getattr(func, "__statspai_aliases__", {}) or {})
        aliases.update(old_to_new)
        wrapper.__statspai_aliases__ = aliases  # type: ignore[attr-defined]
        deprecated = dict(getattr(func, "__statspai_deprecated_aliases__", {}) or {})
        deprecated.update(old_to_new)
        setattr(wrapper, "__statspai_deprecated_aliases__", deprecated)
        return wrapper  # type: ignore[return-value]

    return decorator
