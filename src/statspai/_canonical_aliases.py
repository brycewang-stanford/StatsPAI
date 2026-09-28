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


#: ``{alias: canonical}`` for every registered name that is the same
#: callable as another (``sp.rosenbaum_gamma is sp.rosenbaum_bounds``) or a
#: thin wrapper that forwards to one unchanged apart from argument spelling
#: (``sp.rdd`` -> ``sp.rdrobust``). The registry copies it onto
#: ``FunctionSpec.alias_of`` so search ranks the canonical entry first and a
#: tool manifest can list each estimator once. Dispatchers that *choose*
#: between several targets (``sp.causal_discovery``,
#: ``sp.partial_identification``) are not aliases, and neither are core
#: verbs that also exist as a dispatcher option (``sp.ivreg``).
#: ``tests/test_discovery_aliases.py`` checks that every identical-callable
#: pair in the registry is listed here and that every target is registered.
FUNCTION_ALIAS_OF: Dict[str, str] = {
    # Same callable object under two names.
    "did_imputation": "bjs",
    "borusyak_jaravel_spiess": "bjs",
    "did_2stage": "gardner_did",
    "causal_survival": "causal_survival_forest",
    "model_averaging_dml": "dml_model_averaging",
    "opreg": "olley_pakes",
    "levpet": "levinsohn_petrin",
    "acf": "ackerberg_caves_frazer",
    "test_calibration": "calibration_test",
    "postestimation_report": "postestimation_contract",
    "verify": "verify_recommendation",
    "yun_nonlinear": "bauer_sinning",
    "rosenbaum_gamma": "rosenbaum_bounds",
    # Forwarding wrappers (article / Stata / R spellings).
    "rdd": "rdrobust",
    "psm": "match",
    "frontdoor": "front_door",
    "xlearner": "metalearner",
    "conformal_ite": "conformal_cate",
    "mediation": "mediate",
    "multi_cutoff_rd": "rdmc",
    "geographic_rd": "rdms",
    "boundary_rd": "rd2d",
    "multi_score_rd": "rd_multi_score",
    "diagnostic_test": "sensitivity_specificity",
    "kan_dlate": "dist_iv",
    "nonlinear_icp": "icp",
    "xtdpdsys": "xtabond",
    "synthdid_estimate": "sdid",
    "sc_estimate": "sdid",
    "did_estimate": "sdid",
}


def alias_target(name: str) -> "str | None":
    """Registered name ``name`` is an alias of, or ``None``."""
    return FUNCTION_ALIAS_OF.get(name)


def canonical_of(name: str) -> str:
    """Return the canonical name for ``name`` (or ``name`` itself if no alias)."""
    for canonical, aliases in CANONICAL_ALIASES.items():
        if name == canonical or name in aliases:
            return canonical
    return name


def is_legacy_alias(name: str) -> bool:
    """True if ``name`` is a non-canonical name that still resolves to a canonical."""
    return canonical_of(name) != name and name not in CANONICAL_ALIASES
