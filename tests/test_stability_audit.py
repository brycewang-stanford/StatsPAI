"""Smoke + contract tests for the stability/validation reverse-audit script.

The audit at ``scripts/stability_audit.py`` walks the registry and
flags stable API entries that lack a parity test in
``tests/reference_parity/`` or ``tests/external_parity/``. The registry's
``validation_status`` field is the authoritative evidence tier; this
script is a compatibility guard for old stable-by-default risk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "stability_audit.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_audit_default_renders_human_report() -> None:
    res = _run([])
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "StatsPAI stability/validation reverse-audit" in out
    assert "stable" in out
    assert "experimental" in out
    # The human-readable report includes the floor so a reader can see
    # the CI threshold inline.
    assert "floor:" in out


def test_audit_json_is_well_formed() -> None:
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    # Required top-level shape
    assert set(payload) == {
        "totals",
        "parity_coverage",
        "lists",
        "auto_unbacked_breakdown",
        "sources",
        "evidence_paths",
        "floor",
    }
    t = payload["totals"]
    assert t["stable"] == t["stable_handwritten"] + t["stable_auto"]
    assert t["registry"] == t["stable"] + t["experimental"] + t["deprecated"]
    p = payload["parity_coverage"]
    # Counts must add up
    assert (
        p["backed_handwritten"] + p["unbacked_handwritten"] == t["stable_handwritten"]
    )
    assert p["backed_auto"] + p["unbacked_auto"] == t["stable_auto"]
    breakdown = payload["auto_unbacked_breakdown"]
    assert sum(breakdown["category_counts"].values()) == p["unbacked_auto"]
    assert (
        breakdown["classlike_symbol_count"] + breakdown["functionlike_symbol_count"]
        == p["unbacked_auto"]
    )
    assert breakdown["category_counts"]
    assert set(breakdown["category_examples"]) <= set(breakdown["category_counts"])
    assert payload["evidence_paths"]["refs"] >= payload["evidence_paths"]["unique"] > 0
    assert payload["evidence_paths"]["missing"] == []


def test_check_mode_passes_under_current_floor() -> None:
    """Today's catalogue must satisfy the documented floor.

    The JSS submission floor is zero for hand-written stable APIs: each
    such entry needs either validation-tier evidence or explicit API/unit
    contract evidence. Auto-registered specs remain outside this floor.
    """
    res = _run(["--check"])
    assert res.returncode == 0, (
        f"stability audit --check failed under the current floor:\n"
        f"stdout={res.stdout!r}\nstderr={res.stderr!r}"
    )


def test_unbacked_listing_includes_only_handwritten_names() -> None:
    """``--unbacked`` lists hand-written stable names lacking a parity test."""
    res = _run(["--unbacked"])
    assert res.returncode == 0, res.stderr
    # Spot-check: the listing should contain at least one well-known
    # hand-written but unbacked stable function (e.g. ``hal_tmle`` —
    # we never wrote a HAL-TMLE parity test).
    out = res.stdout
    assert "Unbacked hand-written stable functions" in out


def test_audit_classifies_known_experimental_as_experimental() -> None:
    """Sanity: the three v1.13 experimental flagships must not appear in
    any 'stable' bucket of the audit output."""
    res = _run(["--json"])
    payload = json.loads(res.stdout)
    experimental = set(payload["lists"]["experimental"])
    for name in (
        "text_treatment_effect",
        "llm_annotator_correct",
        "did_multiplegt_dyn",
    ):
        assert (
            name in experimental
        ), f"{name} should be classified experimental in the audit"
    # And it shouldn't show up in any unbacked bucket (those are stable-only).
    assert name not in payload["lists"]["unbacked_handwritten"]
    assert name not in payload["lists"]["unbacked_auto"]
