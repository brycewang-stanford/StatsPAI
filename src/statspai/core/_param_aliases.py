"""Canonical parameter-name aliases for estimator entry points.

StatsPAI grew estimators whose treatment argument is spelled ``d``,
``T``, ``treat``, ``x_endog``, or ``treated_unit`` and whose outcome is
``y`` / ``Y`` / ``outcome`` depending on the source literature.  Humans
shrug; LLM agents generating calls mix them up constantly.  Entry
points therefore accept the canonical short names (``y`` outcome, ``d``
treatment, ``x`` covariates) as *aliases* alongside their native
spellings — additive only, never a rename, so every existing call keeps
working.

``resolve_alias`` is the one shared primitive: it returns the effective
value and refuses loudly when both spellings are supplied with
conflicting values (silently preferring one would hide a caller bug).
"""

from typing import Any, Optional

from ..exceptions import MethodIncompatibility

__all__ = ["resolve_alias"]


def resolve_alias(
    primary_name: str,
    primary: Optional[Any],
    alias_name: str,
    alias: Optional[Any],
) -> Optional[Any]:
    """Merge a parameter with its canonical alias.

    Returns ``primary`` when only it is set, ``alias`` when only it is
    set, and their common value when both agree.  Raises
    :class:`MethodIncompatibility` when both are set to different
    values — that is always a caller bug worth surfacing.
    """
    if primary is None:
        return alias
    if alias is None:
        return primary
    if primary is alias or primary == alias:
        return primary
    raise MethodIncompatibility(
        f"Both {primary_name}={primary!r} and its alias "
        f"{alias_name}={alias!r} were supplied with different values. "
        f"Pass only one of them.",
        recovery_hint=f"Drop either {primary_name}= or {alias_name}=.",
    )
