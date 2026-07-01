"""Canonical-name declarations for the BJS / Gardner DiD families.

Two public-API duplications that survived the v1 era:

* **BJS** (Borusyak-Jaravel-Spiess imputation estimator) is exposed under
  three names: ``bjs``, ``borusyak_jaravel_spiess``, ``did_imputation``.
* **Gardner two-stage** is exposed under two names: ``gardner_did``, ``did_2stage``.

This module declares the canonical name for each family (BJS -> ``bjs``;
Gardner -> ``gardner_did``) and provides a small lookup helper. It does
**not** remove, rename, or wrap any existing function — every legacy name
keeps working — but it gives the registry, ``sp.help``, and LLM-facing
discoverability a single canonical entry per family.

The hard consolidation (deleting the legacy names) is deferred to the
post-JSS-release window per the additive-only JSS-review constraint.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

#: ``{canonical: frozenset(aliases)}``.  The canonical name is the documented
#: entry point; every alias must keep resolving to the same callable until the
#: hard consolidation lands (post-JSS).
CANONICAL_ALIASES: Dict[str, FrozenSet[str]] = {
    "bjs": frozenset({"borusyak_jaravel_spiess", "did_imputation"}),
    "gardner_did": frozenset({"did_2stage"}),
}


def canonical_of(name: str) -> str:
    """Return the canonical name for ``name`` (or ``name`` itself if no alias)."""
    for canonical, aliases in CANONICAL_ALIASES.items():
        if name == canonical or name in aliases:
            return canonical
    return name


def is_legacy_alias(name: str) -> bool:
    """True if ``name`` is a non-canonical name that still resolves to a canonical."""
    return canonical_of(name) != name and name not in CANONICAL_ALIASES
