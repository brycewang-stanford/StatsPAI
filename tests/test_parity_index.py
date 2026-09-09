"""Tests for the queryable parity index (sp.parity_status / matrix / summary).

These lock four guarantees that make the parity claim auditable:

1. **API contract** — parity_status / parity_matrix / parity_summary behave.
2. **Taxonomy + artifact existence** — every record uses a valid grade and
   names test artifacts that actually exist on disk (no phantom evidence).
3. **No drift** — the committed snapshot equals a fresh regeneration from the
   parity artifacts (the CI drift gate, mirrored as a unit test).
4. **Reconciliation** — the index never grants a verified grade that the
   registry's own ``validation_status`` contradicts in the dangerous
   direction, and known benign divergences stay bounded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import statspai as sp
from statspai import parity as parity_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
#  1. API contract
# --------------------------------------------------------------------------- #
def test_parity_status_bit_exact_function():
    rec = sp.parity_status("regress")
    assert rec["status"] == "bit-exact"
    assert rec["reference"]  # a named reference, not empty
    assert rec["test"]  # at least one backing artifact
    assert "rel_vs_R" in rec["headline"]


def test_parity_status_unverified_is_honest():
    # A registered function with no parity evidence returns an explicit,
    # non-raising unverified record (the honest gap, not a crash).
    rec = sp.parity_status("causal_dqn")
    assert rec["status"] == "unverified"
    assert rec["notes"]


def test_parity_status_unknown_raises():
    with pytest.raises(KeyError):
        sp.parity_status("definitely_not_a_statspai_function_xyz")


def test_parity_matrix_covers_every_registered_function():
    rows = sp.parity_matrix(fmt="records")
    funcs = {r["function"] for r in rows}
    registered = set(sp.list_functions())
    assert funcs == registered
    assert len(rows) == len(registered)


def test_parity_matrix_status_filter():
    rows = sp.parity_matrix(status="bit-exact")
    assert rows  # non-empty
    assert all(r["status"] == "bit-exact" for r in rows)


def test_parity_matrix_rejects_bad_status():
    with pytest.raises(ValueError):
        sp.parity_matrix(status="not-a-grade")


def test_parity_summary_accounting_is_consistent():
    s = sp.parity_summary()
    assert s["total_functions"] == len(sp.list_functions())
    assert sum(s["by_status"].values()) == s["total_functions"]
    assert s["verified"] + s["unverified"] == s["total_functions"]
    assert s["by_status"]["bit-exact"] >= 50


# --------------------------------------------------------------------------- #
#  2. Taxonomy + artifact existence
# --------------------------------------------------------------------------- #
def test_every_record_uses_a_valid_grade():
    for rec in parity_mod._load_index()["records"]:
        assert rec["status"] in parity_mod.TAXONOMY


def test_verified_records_name_existing_artifacts():
    # No phantom evidence: every test path on a verified record must exist.
    missing = []
    for rec in parity_mod._load_index()["records"]:
        if rec["status"] == "unverified":
            continue
        for rel in rec.get("test", []):
            if not (REPO_ROOT / rel).exists():
                missing.append((rec["function"], rel))
    assert not missing, f"parity records cite non-existent artifacts: {missing}"


def test_bit_exact_and_aligned_have_a_named_reference():
    for rec in parity_mod._load_index()["records"]:
        if rec["status"] in {"bit-exact", "aligned"}:
            assert rec["reference"], f"{rec['function']} has no named reference"


# --------------------------------------------------------------------------- #
#  3. Drift gate (snapshot == fresh regeneration)
# --------------------------------------------------------------------------- #
def _load_generator():
    path = REPO_ROOT / "scripts" / "build_parity_index.py"
    spec = importlib.util.spec_from_file_location("build_parity_index", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_snapshot_matches_fresh_regeneration():
    gen = _load_generator()
    fresh, _warnings = gen.build_index()
    committed = parity_mod._load_index()
    fresh_by_fn = {r["function"]: r for r in fresh["records"]}
    committed_by_fn = {r["function"]: r for r in committed["records"]}
    assert set(fresh_by_fn) == set(
        committed_by_fn
    ), "parity snapshot is stale — run: python scripts/build_parity_index.py"
    for fn, rec in fresh_by_fn.items():
        assert (
            rec["status"] == committed_by_fn[fn]["status"]
        ), f"parity status drift for {fn} — regenerate the snapshot"


def test_no_committed_golden_underperforms_its_budget():
    # The generator emits a warning for any module whose committed golden
    # exceeds its own registered tolerance. There must be none.
    gen = _load_generator()
    _index, warnings = gen.build_index()
    budget_violations = [w for w in warnings if "exceeds registered" in w]
    assert not budget_violations, budget_violations


def test_public_parity_doc_is_in_sync():
    gen = _load_generator()
    fresh, _warnings = gen.build_index()
    total = len(sp.list_functions())
    expected = gen.render_parity_doc(fresh, total)
    committed = (REPO_ROOT / "docs" / "parity.md").read_text(encoding="utf-8")
    assert (
        committed == expected
    ), "docs/parity.md is stale — run: python scripts/build_parity_index.py"


# --------------------------------------------------------------------------- #
#  4. Reconciliation with registry validation_status
# --------------------------------------------------------------------------- #
# The registry derives ``validation_status`` from this index via
# ``statspai._parity_taxonomy.validation_tier_for`` (see
# ``registry._apply_validation_evidence``), so the two systems agree by
# construction for every *stable* symbol. The tests below hold that
# construction in place: they fail if someone reintroduces an independent
# tier assignment on either side.
#
# One carve-out is enumerated rather than allowlisted away. The registry tier
# and the evidence grade are orthogonal vocabularies (see the JSS terminology
# table): a function whose *API* is still experimental keeps the experimental
# tier even when its *numbers* are parity-backed, because promoting it would
# promise signature stability the package has not committed to.
_EXPERIMENTAL_WITH_PARITY_EVIDENCE = {"did_multiplegt_dyn"}


def _registry_and_index():
    from statspai import registry as R

    sp.list_functions()
    idx = {r["function"]: r["status"] for r in sp.parity_matrix(fmt="records")}
    return R._REGISTRY, idx


def test_validation_status_certified_is_backed_by_index():
    """No tier may claim evidence the index cannot produce an artifact for."""
    reg, idx = _registry_and_index()
    unbacked = sorted(
        fn
        for fn, spec in reg.items()
        if getattr(spec, "validation_status", None) in {"certified", "validated"}
        and idx.get(fn, "unverified") == "unverified"
    )
    assert not unbacked, (
        "validation_status claims certified/validated for functions the parity "
        f"index cannot back with an artifact: {unbacked}"
    )


def test_index_evidence_is_not_understated_by_the_registry():
    """The reverse direction: evidence must not be silently discarded.

    The registry used to under-state 85 functions that hold cross-language
    parity evidence, because its README scan reads one API name per module
    row while a module exercises several. Guarding only the over-claim
    direction let that gap grow unnoticed.
    """
    from statspai._parity_taxonomy import NON_ESTIMATOR_LEAVES, validation_tier_for

    reg, idx = _registry_and_index()
    understated = []
    for fn, status in idx.items():
        spec = reg.get(fn)
        if spec is None or spec.stability != "stable":
            continue
        if fn in NON_ESTIMATOR_LEAVES:
            continue
        expected = validation_tier_for(status)
        if expected == "api_stable":
            continue
        if spec.validation_status != expected:
            understated.append((fn, status, spec.validation_status, expected))
    assert not understated, (
        "the parity index holds evidence the registry tier does not reflect "
        f"(function, grade, tier, expected): {understated}"
    )


def test_non_estimator_leaves_never_carry_a_validation_tier():
    """Dataset loaders and DGP helpers must not read as validated."""
    from statspai._parity_taxonomy import NON_ESTIMATOR_LEAVES

    reg, _ = _registry_and_index()
    marked = sorted(
        fn
        for fn in NON_ESTIMATOR_LEAVES
        if fn in reg
        and getattr(reg[fn], "validation_status", None) in {"certified", "validated"}
    )
    assert not marked, (
        "non-estimator symbols carry a validation tier; they appear in parity "
        f"tests because they build the fixture, not because they were "
        f"compared against anything: {marked}"
    )


def test_registry_index_divergence_is_zero_for_stable_symbols():
    """The headline invariant: one function, one story."""
    from statspai._parity_taxonomy import NON_ESTIMATOR_LEAVES, validation_tier_for

    reg, idx = _registry_and_index()
    diverged = []
    for fn, spec in reg.items():
        if spec.stability != "stable" or fn in NON_ESTIMATOR_LEAVES:
            continue
        expected = validation_tier_for(idx.get(fn, "unverified"))
        if spec.validation_status != expected:
            diverged.append((fn, spec.validation_status, expected))
    assert not diverged, (
        "registry validation_status and the parity index disagree for stable "
        f"symbols (function, registry, expected-from-index): {diverged}"
    )


def test_experimental_parity_carve_out_is_enumerated():
    """Experimental APIs with parity evidence stay listed, not forgotten."""
    reg, idx = _registry_and_index()
    found = {
        fn
        for fn, spec in reg.items()
        if spec.stability == "experimental"
        and idx.get(fn, "unverified") != "unverified"
    }
    assert found == _EXPERIMENTAL_WITH_PARITY_EVIDENCE, (
        "the set of experimental-API-but-parity-backed functions changed; "
        "update _EXPERIMENTAL_WITH_PARITY_EVIDENCE deliberately (promoting "
        f"stability is an API promise, not a bookkeeping fix). Found: {found}"
    )


def test_track_a_aliases_are_all_proven():
    """Every alias credit traces to the pytest that measured it."""
    from statspai._parity_taxonomy import (
        ALIAS_PROOF_TEST,
        REFUTED_ALIASES,
        TRACK_A_ALIASES,
    )

    assert (REPO_ROOT / ALIAS_PROOF_TEST).exists(), (
        f"the alias proof suite {ALIAS_PROOF_TEST} is missing; without it "
        "every alias credit in the index is an unbacked assertion."
    )
    idx = {r["function"]: r for r in sp.parity_matrix(fmt="records")}
    for alias, proof in TRACK_A_ALIASES.items():
        rec = idx.get(alias)
        assert rec is not None, f"alias {alias} carries no index record"
        assert ALIAS_PROOF_TEST in (
            rec.get("test") or []
        ), f"alias {alias} is credited without citing {ALIAS_PROOF_TEST}"
        assert proof.legs, f"alias {alias} registers no measured leg"
    for refuted in REFUTED_ALIASES:
        assert refuted not in TRACK_A_ALIASES, (
            f"{refuted} is recorded as a refuted alias but is back in the "
            "alias table"
        )


def test_wheel_only_tiers_match_the_source_tree():
    """An installed wheel must grade functions exactly as a checkout does.

    Before the registry consumed the committed index, a wheel had no test
    tree to scan and fell back to a coarse hand-written seed, so
    ``sp.describe_function(...)["validation_status"]`` could differ between
    a checkout and the artifact users actually install. ``_parity_index.json``
    is packaged, so the two must now agree exactly -- including that no
    certified symbol arrives without an evidence note.
    """
    from statspai import registry as R

    R._ensure_full_registry()
    source_tree = {fn: spec.validation_status for fn, spec in R._REGISTRY.items()}
    saved_notes = {fn: list(spec.validation_notes) for fn, spec in R._REGISTRY.items()}
    saved_repo_root = R._repo_root
    try:
        R._repo_root = lambda: None
        R._VALIDATION_EVIDENCE_APPLIED = False
        for spec in R._REGISTRY.values():
            spec.validation_notes.clear()
            spec.validation_status = "api_stable"
        R._apply_validation_evidence()
        wheel_only = {fn: spec.validation_status for fn, spec in R._REGISTRY.items()}
        unnoted = sorted(
            fn
            for fn, spec in R._REGISTRY.items()
            if spec.validation_status == "certified" and not spec.validation_notes
        )
    finally:
        R._repo_root = saved_repo_root
        R._VALIDATION_EVIDENCE_APPLIED = False
        for fn, spec in R._REGISTRY.items():
            spec.validation_notes[:] = saved_notes[fn]
            spec.validation_status = source_tree[fn]
        R._VALIDATION_EVIDENCE_APPLIED = True

    drifted = sorted(fn for fn in source_tree if source_tree[fn] != wheel_only[fn])
    assert not drifted, (
        "validation_status differs between an installed wheel and a source "
        f"checkout for: {drifted[:20]}"
    )
    assert (
        not unnoted
    ), f"certified without an evidence note in the wheel-only path: {unnoted[:20]}"


def test_python_reference_rows_are_pinned():
    """Certified rows compared against a Python reference must stay enumerated.

    The manuscript states that every certified symbol links to a named R or
    Stata parity module. Two rows follow CLAUDE.md §5.1's "use the
    implementation the method's authors maintain" rule to a Python package
    instead, so the claim has to name them. If a third appears without the
    prose being updated, that sentence becomes false.
    """
    from statspai._parity_taxonomy import CROSS_LANGUAGE_STATUSES, PYTHON_REFERENCE_ROWS

    idx = {r["function"]: r for r in sp.parity_matrix(fmt="records")}
    python_only = {
        fn
        for fn, rec in idx.items()
        if rec.get("status") in CROSS_LANGUAGE_STATUSES
        and not (set(rec.get("sides") or []) & {"R", "Stata"})
    }
    assert python_only == set(PYTHON_REFERENCE_ROWS), (
        "the set of certified rows without an R/Stata side changed; update "
        "_parity_taxonomy.PYTHON_REFERENCE_ROWS and the manuscript sentence "
        f"that enumerates them. Found: {sorted(python_only)}"
    )


def test_every_certified_row_names_its_reference():
    """No certified grade may rest on an unnamed reference implementation."""
    from statspai import registry as R
    from statspai._parity_taxonomy import CROSS_LANGUAGE_STATUSES

    sp.list_functions()
    idx = {r["function"]: r for r in sp.parity_matrix(fmt="records")}
    unnamed = sorted(
        fn
        for fn, spec in R._REGISTRY.items()
        if spec.validation_status == "certified"
        and not str(idx.get(fn, {}).get("reference") or "").strip()
    )
    assert not unnamed, f"certified without a named reference implementation: {unnamed}"
    graded = sorted(
        fn
        for fn, spec in R._REGISTRY.items()
        if spec.validation_status == "certified"
        and idx.get(fn, {}).get("status") not in CROSS_LANGUAGE_STATUSES
    )
    assert not graded, f"certified without a cross-package parity grade: {graded}"
