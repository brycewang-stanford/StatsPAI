"""API surface consistency between ``statspai.__all__`` and ``registry``.

Two contracts that an agent-native library cannot afford to drift on:

1. Every name in ``statspai.__all__`` must resolve on the top-level module.
   (Otherwise ``from statspai import *`` raises and IDEs lie.)

2. Every registered function must resolve as ``statspai.<name>`` *or* via the
   submodule path documented in its registry ``example`` field. (Otherwise
   ``sp.help(f)`` shows a function the user cannot then call.)

A frozen baseline records the *current* asymmetry between ``__all__`` and the
registry so the test passes today but fails on new drift. To accept a
deliberate change, update the constants below in the same commit.
"""

from __future__ import annotations

import importlib

import pytest

import statspai
import statspai as sp

# ---------------------------------------------------------------------------
# Frozen baseline. Update deliberately when intentionally adding / removing
# public surface; never silently. Both sets are alphabetised for easy diffs.
# ---------------------------------------------------------------------------

# Names in ``__all__`` that are *not* registry functions. These are legitimate
# module / constant re-exports — the registry only tracks callable estimators.
ALL_NOT_REGISTERED_BASELINE = frozenset(
    {
        "JOURNAL_PRESETS",
        "PAPER_TABLE_TEMPLATES",
        "STABILITY_TIERS",
        "VALIDATION_STATUSES",
        "epi",
        "exceptions",
        "longitudinal",
        "mendelian",
        "question",
        "tte",
    }
)

# Names registered in the registry but *not* in ``__all__``. Most are real
# estimators that ship today on ``sp.<name>`` but were never added to the
# star-import list. Reducing this set is the goal; growing it is a regression.
#
# 2026-06 drift repair: 19 estimators that already shipped on ``sp.<name>``
# but were invisible to ``sp.list_functions`` / ``sp.describe_function`` /
# ``sp.function_schema`` were added to ``__all__`` (BCF extensions, the
# proximal / negative-control family, DiD extras, shift-share, dose-response,
# ITS, longitudinal TMLE, etc.), shrinking this baseline from 44 → 25 and
# locking the gain in so they can never silently drop back out.
REGISTERED_NOT_IN_ALL_BASELINE = frozenset(
    {
        "anthropic_client",
        "assimilative_causal",
        "bayes_dml",
        "causal_bandit",
        "causal_kalman",
        "causal_mas",
        "causal_policy_forest",
        "conformal_continuous",
        "conformal_interference",
        "counterfactual_fairness",
        "counterfactual_policy_optimization",
        "demographic_parity",
        "echo_client",
        "equalized_odds",
        "evidence_without_injustice",
        "fairness_audit",
        "heterogeneity_of_effect",
        "long_term_from_short",
        "mr_bma",
        "mr_multivariable",
        "openai_client",
        "particle_filter",
        "proximal_surrogate_index",
        "rwd_rct_concordance",
        "sharp_ope_unobserved",
    }
)

# Registered functions that live on a submodule rather than the top-level
# module. The registry ``example`` already documents the correct path, so this
# is not a defect — just the contract the test must allow.
SUBMODULE_ONLY_BASELINE = frozenset(
    {
        "anthropic_client",
        "echo_client",
        "openai_client",
        "particle_filter",
    }
)


# ---------------------------------------------------------------------------
# Contract 1: ``__all__`` resolves
# ---------------------------------------------------------------------------


def test_all_names_in_dunder_all_resolve_on_module():
    """Every name listed in ``statspai.__all__`` must be accessible.

    A name in ``__all__`` that does not resolve means ``from statspai import *``
    raises ``ImportError`` at top-level and breaks every notebook that uses
    the star-import idiom.
    """
    missing = [n for n in statspai.__all__ if not hasattr(statspai, n)]
    assert (
        missing == []
    ), f"{len(missing)} name(s) in __all__ do not resolve: {missing!r}"


# ---------------------------------------------------------------------------
# Contract 2: registry → callable
# ---------------------------------------------------------------------------


def _registry_example_module(example: str | None) -> str | None:
    """Pull the documented submodule path out of an example like
    ``sp.causal_llm.anthropic_client(...)`` → ``causal_llm``.
    """
    if not example or not isinstance(example, str):
        return None
    head = example.strip().split("(", 1)[0]
    parts = head.split(".")
    # Expect form ``sp.<sub>.<name>(...)`` or ``statspai.<sub>.<name>(...)``.
    if len(parts) >= 3 and parts[0] in {"sp", "statspai"}:
        return parts[1]
    return None


def test_every_registered_function_is_callable_via_documented_path():
    """Each registered function resolves on ``statspai`` directly, or on the
    submodule that its registry ``example`` documents.
    """
    failures = []
    for name in sp.list_functions():
        if hasattr(statspai, name):
            continue  # accessible at top-level — fine
        spec = sp.describe_function(name) or {}
        submod = _registry_example_module(spec.get("example"))
        if submod is None:
            failures.append(
                f"{name}: no example documenting submodule, "
                "and not resolvable as statspai.<name>"
            )
            continue
        try:
            mod = importlib.import_module(f"statspai.{submod}")
        except ImportError as exc:
            failures.append(f"{name}: documented submodule fails to import ({exc})")
            continue
        if not hasattr(mod, name):
            failures.append(f"{name}: not on statspai and not on statspai.{submod}")
    assert (
        failures == []
    ), f"{len(failures)} registered function(s) are unreachable:\n  " + "\n  ".join(
        failures
    )


# ---------------------------------------------------------------------------
# Contract 3: drift against frozen baselines
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def asymmetry():
    all_set = frozenset(getattr(statspai, "__all__", []))
    reg_set = frozenset(sp.list_functions())
    return {
        "all_not_registered": all_set - reg_set,
        "registered_not_in_all": reg_set - all_set,
    }


def test_all_not_registered_matches_baseline(asymmetry):
    """``__all__`` entries that are not registry functions are tightly
    enumerated. Adding a new non-function name to ``__all__`` is allowed only
    by updating ``ALL_NOT_REGISTERED_BASELINE`` in the same commit.
    """
    current = asymmetry["all_not_registered"]
    extra = current - ALL_NOT_REGISTERED_BASELINE
    removed = ALL_NOT_REGISTERED_BASELINE - current
    assert not extra, (
        "New non-function entry in __all__ — register it or update the "
        f"baseline: {sorted(extra)}"
    )
    assert not removed, (
        "An expected non-function entry disappeared from __all__ — "
        f"update the baseline if intentional: {sorted(removed)}"
    )


def test_registered_not_in_all_does_not_grow(asymmetry):
    """Functions registered but missing from ``__all__`` should shrink, never
    grow. Net new entries here are silent agent-discoverability regressions.
    """
    current = asymmetry["registered_not_in_all"]
    new = current - REGISTERED_NOT_IN_ALL_BASELINE
    assert not new, (
        "New registered function(s) missing from __all__ "
        f"(add them or update baseline): {sorted(new)}"
    )


def test_submodule_only_baseline_still_holds():
    """Functions that live only on a submodule (documented via their example)
    are tightly enumerated; net new entries silently shrink the top-level
    surface and should be deliberate.
    """
    actual = set()
    for name in sp.list_functions():
        if not hasattr(statspai, name):
            actual.add(name)
    new = actual - SUBMODULE_ONLY_BASELINE
    assert not new, (
        "Function(s) silently dropped from top-level statspai (only "
        f"reachable via submodule): {sorted(new)}"
    )
