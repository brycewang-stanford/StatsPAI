"""CI ratchet for raw curated agent-card coverage.

Sister to :mod:`tests.test_stability_audit`.  Where that file gates
parity-evidence regressions, this file gates *curated metadata*
regressions: a PR that empties an existing function's ``assumptions``
or ``failure_modes`` (e.g. via a careless refactor that resets a
:class:`statspai.registry.FunctionSpec`) must fail CI.

The floor lives in ``scripts/agent_card_coverage_floor.json`` and is
maintained by hand — bump it after intentionally raising the bar.
Lowering the floor to make CI pass is explicitly disallowed in
``docs/agent_cards_spec.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "agent_card_coverage.py"
FLOOR_PATH = REPO_ROOT / "scripts" / "agent_card_coverage_floor.json"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_summary_renders_human_report() -> None:
    res = _run([])
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "Registered functions" in out
    assert "Tier-B" in out and "Tier-A" in out and "Tier-S" in out
    assert "Per-field coverage" in out


def test_by_category_renders_table() -> None:
    res = _run(["--by-category"])
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "category" in out and "tier-A" in out
    # Sanity: causal is the biggest bucket and must show up
    assert "causal" in out


def test_json_is_well_formed() -> None:
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert {
        "total",
        "field_counts",
        "validation_counts",
        "per_category",
        "tier_totals",
    } <= set(payload)
    # Tier-B is a superset of Tier-A is a superset of Tier-S
    tt = payload["tier_totals"]
    assert tt["tier_b"] >= tt["tier_a"] >= tt["tier_s"]


def test_floor_file_exists_and_well_formed() -> None:
    """The floor must be checked in so CI runs against a known baseline."""
    assert FLOOR_PATH.exists(), (
        f"{FLOOR_PATH} is missing — seed it with "
        f"`python scripts/agent_card_coverage.py --write-floor`."
    )
    floor = json.loads(FLOOR_PATH.read_text(encoding="utf-8"))
    # Required keys
    required = {
        "tier_b",
        "tier_a",
        "tier_s",
        "field_assumptions",
        "field_failure_modes",
        "field_alternatives",
        "field_pre_conditions",
        "field_typical_n_min",
        "field_tags",
        "field_example",
        "field_reference",
        "field_description_30",
        "field_any_param_description",
        "validation_certified",
        "validation_validated",
    }
    missing = required - set(floor)
    assert not missing, f"Floor file missing keys: {sorted(missing)}"
    # Sanity: every counter is a non-negative int
    for k, v in floor.items():
        assert isinstance(v, int) and v >= 0, f"floor[{k!r}]={v!r}"


def test_check_mode_passes_under_current_floor() -> None:
    """Curated coverage must not regress against the committed floor.

    If this fails, do NOT lower the floor.  Investigate which spec
    lost its curated content and restore it.  See
    ``docs/agent_cards_spec.md``.
    """
    res = _run(["--check"])
    assert res.returncode == 0, (
        f"agent_card_coverage --check failed:\n"
        f"stdout={res.stdout!r}\nstderr={res.stderr!r}"
    )


def test_inherits_from_merges_parent_assumptions() -> None:
    """A variant with empty own assumptions must surface parent's via agent_card.

    Spot-checked on ``borusyak_jaravel_spiess`` (DiD variant with empty
    own ``assumptions`` / ``failure_modes`` that nonetheless declares
    ``inherits_from='did'`` through ``_INHERITANCE_SEEDS``).  This
    locks the contract documented in ``docs/agent_cards_spec.md``.
    """
    import statspai  # noqa: F401
    from statspai.registry import _REGISTRY, _ensure_full_registry

    _ensure_full_registry()
    variant = _REGISTRY["borusyak_jaravel_spiess"]
    parent = _REGISTRY["did"]
    assert (
        variant.inherits_from == "did"
    ), "borusyak_jaravel_spiess must inherit from did per _INHERITANCE_SEEDS"

    raw = variant.agent_card(merge_inherited=False)
    merged = variant.agent_card(merge_inherited=True)
    # The contract is a UNION: the merged view must surface the parent's
    # family-shared assumptions / failure modes *on top of* whatever the
    # variant declares for itself.  (The fixture used to assume the
    # variant carried none of its own; that is no longer true now that
    # curated cards give borusyak its own parallel-trends / no-anticipation
    # assumptions — and over-specifying "own == []" made this test brittle
    # against exactly the metadata enrichment it is meant to encourage.)
    merged_assumptions = set(merged["assumptions"])
    assert (
        set(parent.assumptions) <= merged_assumptions
    ), "merged view must include the parent's assumptions"
    assert (
        set(raw["assumptions"]) <= merged_assumptions
    ), "merged view must not drop the variant's own assumptions"
    parent_symptoms = {fm.symptom for fm in parent.failure_modes}
    merged_symptoms = {fm["symptom"] for fm in merged["failure_modes"]}
    assert (
        parent_symptoms <= merged_symptoms
    ), "merged view must include the parent's failure modes"
    assert len(merged["assumptions"]) >= len(parent.assumptions)
    # typical_n_min falls back to parent when child is None
    if variant.typical_n_min is None:
        assert merged["typical_n_min"] == parent.typical_n_min


def test_inherits_from_never_inherits_method_specific_fields() -> None:
    """Description / example / params must stay variant-specific."""
    from statspai.registry import _REGISTRY, _ensure_full_registry

    _ensure_full_registry()
    variant = _REGISTRY["callaway_santanna"]
    parent = _REGISTRY["did"]
    merged = variant.agent_card(merge_inherited=True)
    # Even with inheritance, the variant's own description / example
    # are returned, never the parent's.
    assert merged["description"] != parent.description
    assert merged["example"] != parent.example


def test_inherits_from_cycle_does_not_hang() -> None:
    """A self-cycle must terminate and behave like no inheritance."""
    from statspai.registry import FunctionSpec, _REGISTRY, _ensure_full_registry

    _ensure_full_registry()
    # Construct a fake spec with self-cycle; never registered, just
    # exercise the merge helper.
    fake = FunctionSpec(
        name="__test_cycle__",
        category="utils",
        description="Cycle smoke test.",
        inherits_from="__test_cycle__",
    )
    _REGISTRY["__test_cycle__"] = fake
    try:
        card = fake.agent_card()
        # Helper bails out -> raw lists.
        assert card["assumptions"] == []
        assert card["failure_modes"] == []
    finally:
        _REGISTRY.pop("__test_cycle__", None)


def test_known_flagship_specs_are_tier_a() -> None:
    """Sanity: a handful of flagship estimators must stay Tier-A.

    Tier-A is *metadata completeness* (description / tags / example /
    reference / params / assumptions / pre_conditions / failure_modes /
    alternatives / typical_n_min).  Per ``docs/agent_cards_spec.md``
    §Tier-S it deliberately does **not** gate on the numerical-validation
    tier: that axis is evidence-driven (known-truth / parity / coverage)
    and an honest demotion to ``api_stable`` when the evidence does not
    qualify is the documented, correct behaviour — not a metadata
    regression.  If a flagship slips out of Tier-A, that's a release
    blocker; the separate Tier-S parity target is tracked by the JSS
    validation audit (``test_jss_validation_api.py``) and
    ``scripts/stability_audit.py``.
    """
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    # The --json view drops per-function detail to keep stdout small,
    # so reach into the registry directly for the spot-check.
    import statspai  # noqa: F401
    from statspai.registry import _REGISTRY, _ensure_full_registry

    _ensure_full_registry()
    flagships = ("regress", "did", "iv", "rdrobust", "synth")
    missing_from_registry = [f for f in flagships if f not in _REGISTRY]
    assert (
        not missing_from_registry
    ), f"Flagship functions absent from registry: {missing_from_registry}"
    weak = []
    for name in flagships:
        spec = _REGISTRY[name]
        missing = []
        if not (spec.description and len(spec.description.strip()) > 30):
            missing.append("description_30")
        if not spec.tags:
            missing.append("tags")
        if not spec.example:
            missing.append("example")
        if not spec.reference:
            missing.append("reference")
        if not any(p.description and p.description.strip() for p in spec.params):
            missing.append("any_param_description")
        if not spec.assumptions:
            missing.append("assumptions")
        if not spec.pre_conditions:
            missing.append("pre_conditions")
        if not spec.failure_modes:
            missing.append("failure_modes")
        if not spec.alternatives:
            missing.append("alternatives")
        if spec.typical_n_min is None:
            missing.append("typical_n_min")
        # Tier-A does not require {certified, validated}: commit 166a416
        # tightened `validated` to demand known-truth / reference-parity /
        # coverage evidence, so a flagship *dispatcher* like `did` may
        # honestly sit at `api_stable` while its parity work is tracked
        # separately.  The release-gate regression here is a flagship
        # silently degrading *below* the stable line.
        if spec.validation_status in {"experimental", "deprecated"}:
            missing.append(f"validation_status_degraded={spec.validation_status}")
        if missing:
            weak.append(f"{name}: {', '.join(missing)}")
    assert not weak, f"Flagship estimators must be Tier-A/S but these are not: {weak}"
