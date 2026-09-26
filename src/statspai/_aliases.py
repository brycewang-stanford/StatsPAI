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

import difflib
import functools
import inspect
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


def accepts_aliases(
    _warn: bool | None = None, _strict: bool = False, **alias_map: str
) -> Callable[[F], F]:
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
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins
            params = {}
        takes_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
        if params and _strict:
            # An alias that is also a real parameter would silently rebind it.
            # Opt-in: legacy call sites (e.g. regress's vce= -> robust=) fold
            # a declared parameter into another on purpose.
            clash = sorted(a for a in alias_map if a in params)
            missing = sorted(
                t for t in alias_map.values() if t not in params and not takes_var_kw
            )
            if clash or missing:
                raise ValueError(
                    f"accepts_aliases on {func.__name__}: aliases {clash} are "
                    f"existing parameters; targets {missing} are not parameters."
                )
        inherited: Dict[str, str] = dict(getattr(func, "__statspai_aliases__", {}))
        known = set(params) | set(alias_map) | set(inherited)

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
            if params and not takes_var_kw:
                unknown = [k for k in kwargs if k not in known]
                if unknown:
                    raise TypeError(_unexpected_message(func.__name__, unknown, known))
            return func(*args, **kwargs)

        # Advertise the accepted aliases (merge if stacked).
        existing = dict(inherited)
        existing.update(alias_map)
        wrapper.__statspai_aliases__ = existing  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def accepts_formula_first(formula_param: str = "formula") -> Callable[[F], F]:
    """Let a data-first estimator also take the formula first.

    Most formula estimators are ``f(formula, data)`` (``sp.regress``,
    ``sp.iv``, ``sp.feols``); a few are ``f(data, formula)``
    (``sp.qreg``, ``sp.panel``, the spatial family), so
    ``sp.qreg("y ~ x", data=df)`` died with "got multiple values for
    argument 'data'". A DataFrame is never a string, so a string in the
    first positional slot is unambiguously the formula: it is rebound to
    ``formula_param`` and any second positional becomes ``data``. Calls in
    the existing order are untouched, and the signature is unchanged.
    """

    def decorator(func: F) -> F:
        params = list(inspect.signature(func).parameters)
        if not params or params[0] != "data" or formula_param not in params:
            raise ValueError(
                f"accepts_formula_first on {func.__name__}: expected a "
                f"data-first signature with a {formula_param!r} parameter."
            )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if args and isinstance(args[0], str) and formula_param not in kwargs:
                kwargs[formula_param] = args[0]
                rest = args[1:]
                if rest and "data" not in kwargs:
                    kwargs["data"] = rest[0]
                    rest = rest[1:]
                if rest:
                    raise TypeError(
                        f"{func.__name__}(): with the formula first, pass the "
                        "remaining arguments by keyword."
                    )
                return func(**kwargs)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def _unexpected_message(name: str, unknown: list, known: set) -> str:
    """``TypeError`` text for unknown keywords, with did-you-mean suggestions."""
    parts = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(known), n=3, cutoff=0.6)
        hint = f" (did you mean {', '.join(repr(c) for c in close)}?)" if close else ""
        parts.append(f"{key!r}{hint}")
    return f"{name}() got unexpected keyword argument(s): {'; '.join(parts)}."
