"""Keyword-alias plumbing for grammar convergence.

The signature audit (2026-06-30) found the same idea spelled several ways
across the public API — most importantly the SE/variance keyword
(``robust`` / ``vcov`` / ``vce`` / ``se_type``).  The maintainer ratified
``vce`` as canonical, with ``y`` / ``treat`` for outcome / treatment.

This module lets an estimator *accept the canonical spelling* without renaming
its existing parameter — a strictly **additive, reversible** change suitable
during JSS review (CLAUDE.md §3): old call sites keep working unchanged, and a
new canonical call site is accepted and forwarded internally.

Direction of the map
--------------------
``@accepts_aliases(vce="robust")`` reads as *"callers may pass ``vce=`` and it
is forwarded to the existing ``robust=`` parameter."*  i.e. the keys are the
**new canonical spellings** the function will now also accept; the values are
the **current parameter names** in the function signature.

Warnings are **off by default** during JSS review — the goal this cycle is
acceptance, not nagging.  Post-review, after parameters are renamed to the
canonical spelling, the map is flipped (legacy → canonical) and
``warn=True`` can deprecate the old spelling on the normal schedule.

Examples
--------
>>> from statspai._aliases import accepts_aliases
>>> @accepts_aliases(vce="robust")
... def fit(formula, data, robust="nonrobust"):
...     return robust
>>> fit("y ~ x", None, vce="hc1")        # canonical spelling accepted
'hc1'
>>> fit("y ~ x", None, robust="hc1")     # existing spelling still works
'hc1'
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Any, Callable, Dict, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

#: Global switch for legacy-spelling deprecation warnings.  Held OFF during
#: JSS review; flip via ``STATSPAI_ALIAS_WARN=1`` or by setting this directly.
WARN_ON_LEGACY: bool = os.environ.get("STATSPAI_ALIAS_WARN", "") not in (
    "",
    "0",
    "false",
)


def accepts_aliases(_warn: bool | None = None, **alias_map: str) -> Callable[[F], F]:
    """Accept alternative keyword spellings, forwarding to existing params.

    Parameters
    ----------
    **alias_map
        ``new_spelling=existing_param_name`` pairs.  When a caller supplies
        ``new_spelling=...``, it is moved to ``existing_param_name`` before the
        wrapped function runs.
    _warn : bool, optional
        Override the module-level :data:`WARN_ON_LEGACY` for this function.
        Leading underscore keeps it from colliding with a real alias named
        ``warn``.

    Notes
    -----
    * Supplying both the alias and its target in the same call is a
      ``TypeError`` (ambiguous), mirroring Python's own duplicate-argument
      behaviour.
    * The original signature is preserved for introspection via
      :func:`functools.wraps`; the accepted aliases are recorded on
      ``__statspai_aliases__`` so ``sp.help`` / the registry can advertise
      them without re-parsing the decorator.
    """
    if not alias_map:
        raise ValueError("accepts_aliases requires at least one alias=target pair")

    def decorator(func: F) -> F:
        warn = WARN_ON_LEGACY if _warn is None else _warn

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for alias, target in alias_map.items():
                if alias in kwargs:
                    if target in kwargs:
                        raise TypeError(
                            f"{func.__name__}() received both '{alias}' and its "
                            f"canonical target '{target}'; pass only one."
                        )
                    value = kwargs.pop(alias)
                    kwargs[target] = value
                    if warn:
                        warnings.warn(
                            f"{func.__name__}(): '{alias}' is accepted as an "
                            f"alias for '{target}'.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
            return func(*args, **kwargs)

        # Advertise the accepted aliases (merge if stacked).
        existing: Dict[str, str] = dict(getattr(func, "__statspai_aliases__", {}))
        existing.update(alias_map)
        wrapper.__statspai_aliases__ = existing  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
