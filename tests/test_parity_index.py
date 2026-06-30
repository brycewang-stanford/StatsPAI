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
    rec = sp.parity_status("conformal_ite")
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
# Benign, documented divergences (index is MORE accurate than the coarse seed,
# or the seed over-marks a non-estimator). Pinned so NEW divergences fail.
_KNOWN_VALIDATION_OVER_MARKS = {"california_prop99", "dgp_did"}


def test_validation_status_certified_is_backed_by_index():
    from statspai import registry as R

    sp.list_functions()
    idx = {r["function"]: r["status"] for r in sp.parity_matrix(fmt="records")}
    unbacked = sorted(
        fn
        for fn, spec in R._REGISTRY.items()
        if getattr(spec, "validation_status", None) in {"certified", "validated"}
        and idx.get(fn, "unverified") == "unverified"
        and fn not in _KNOWN_VALIDATION_OVER_MARKS
    )
    assert not unbacked, (
        "validation_status claims certified/validated for functions the parity "
        f"index cannot back with an artifact: {unbacked}"
    )
