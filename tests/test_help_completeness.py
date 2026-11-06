"""Contract tests for the agent-native help system's documentation coverage.

These tests turn the coverage numbers reported by ``scripts/help_coverage.py``
into CI gates so help-system quality can only ratchet *up*:

  * **Hard gate** — every registered, callable public function MUST have a
    non-empty docstring (``sp.describe_function`` / ``sp.help`` are useless
    without one; this is the agent-native floor).
  * **Ratchet gates** — ``Parameters`` / ``Returns`` / ``Examples`` /
    ``References`` / agent-native-field coverage must never drop below the
    frozen floors in ``scripts.help_coverage.FLOORS``. When you improve
    coverage, *raise the floor* in that module so the gain is locked in.

The floors live in ``scripts/help_coverage.py`` (single source of truth, also
used by ``python scripts/help_coverage.py --check`` in CI), not here, so the
human report and the test gate can never disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import help_coverage as hc  # noqa: E402


@pytest.fixture(scope="module")
def coverage():
    return hc.compute_coverage()


def test_every_public_function_has_a_docstring(coverage):
    """Hard gate: no registered callable may ship without a docstring."""
    missing = coverage["missing_docstring"]
    assert not missing, (
        f"{len(missing)} registered function(s) have no docstring — "
        f"sp.describe_function / sp.help cannot describe them: {missing}"
    )
    assert coverage["docstring"]["pct"] == 100.0


@pytest.mark.parametrize("metric", sorted(hc.FLOORS))
def test_coverage_metric_not_below_frozen_floor(coverage, metric):
    """Ratchet gate: each metric stays at or above its frozen floor."""
    pct = coverage[metric]["pct"]
    floor = hc.FLOORS[metric]
    assert pct >= floor, (
        f"help-coverage regression: {metric} dropped to {pct:.1f}% "
        f"(floor {floor:.1f}%). Restore coverage or, if intentional, "
        f"justify lowering the floor in scripts/help_coverage.py."
    )


def test_validated_functions_without_reference_not_above_ceiling(coverage):
    """Ratchet gate: certified / validated functions lacking a method
    reference may only become fewer (rOpenSci G1.0)."""
    got = coverage["validated_without_reference"]
    ceiling = hc.CEILINGS["validated_without_reference"]
    assert got["count"] == len(got["names"])
    assert got["count"] <= ceiling, (
        f"{got['count']} certified/validated functions have no reference "
        f"(ceiling {ceiling}). Map a paper.bib key in "
        f"src/statspai/_reference_seeds.py or the spec."
    )


def test_check_floors_passes_overall(coverage):
    """The aggregate --check gate (used by CI) reports no violations."""
    violations = hc.check_floors(coverage)
    assert not violations, "help-coverage floor violations:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


# The causal / treatment-effect category is the project's differentiator: an
# agent planning a study leans on these fields most. Lock in the family-seed
# coverage (55.5% on 2026-05-29) so it cannot silently rot. Raise as it grows.
CAUSAL_AGENT_NATIVE_FLOOR = 55.0


def test_causal_family_agent_native_coverage(coverage):
    """The causal category keeps high agent-native (assumptions/etc.) coverage."""
    cat = coverage["by_category"].get("causal")
    assert cat is not None, "no 'causal' category found in coverage report"
    pct = cat["agent_native_pct"]
    assert pct >= CAUSAL_AGENT_NATIVE_FLOOR, (
        f"causal-category agent-native coverage fell to {pct:.1f}% "
        f"(floor {CAUSAL_AGENT_NATIVE_FLOOR:.1f}%). These assumptions / "
        f"failure-mode cards are the agent-native differentiator — restore them."
    )


def test_coverage_total_matches_registry():
    """The coverage scan must see the same callables the registry lists."""
    import statspai as sp

    callable_registered = sum(
        1 for n in sp.list_functions() if callable(getattr(sp, n, None))
    )
    assert hc.compute_coverage()["total"] == callable_registered
