"""Review-period freeze of the artifacts behind the JSS manuscript's numbers.

While the JSS paper is under review its tables must match the submitted
release, or the difference must be on the record. See
``scripts/jss_review_freeze.py`` for the scope and the workflow.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "jss_review_freeze", ROOT / "scripts" / "jss_review_freeze.py"
)
freeze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze)

pytestmark = pytest.mark.skipif(
    not freeze.MANIFEST.exists(), reason="no JSS review freeze has been written"
)


def test_every_change_to_a_frozen_artifact_is_on_the_record():
    missing = freeze.unrecorded()
    assert not missing, (
        "These artifacts behind the JSS manuscript changed after the submitted "
        f"release {freeze.load_manifest()['tag']} and are not named in "
        f"{freeze.LEDGER.relative_to(ROOT)}. Add an entry giving the reason and "
        "the effect on the paper (which table or number moves), with each path "
        "in backticks:\n  " + "\n  ".join(missing)
    )


def test_manifest_covers_the_manuscript_inputs():
    manifest = freeze.load_manifest()
    assert tuple(manifest["globs"]) == freeze.GLOBS
    assert manifest["tag"] == f"v{manifest['release']}"
    files = manifest["files"]
    for prefix in (
        "tests/r_parity/results/",
        "tests/orig_parity/results/",
        "tests/coverage_monte_carlo/results_b1000/",
        "tests/reference_parity/_fixtures/grf_seed_mc",
        "tests/perf/results/",
    ):
        assert any(p.startswith(prefix) for p in files), prefix


def test_unrecorded_change_is_caught(tmp_path, monkeypatch):
    manifest = json.loads(json.dumps(freeze.load_manifest()))
    manifest["active"] = True
    victim = next(iter(manifest["files"]))
    manifest["files"][victim] = "0" * 64
    ledger = tmp_path / "ledger.md"
    ledger.write_text("# empty\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "LEDGER", ledger)
    assert victim in freeze.unrecorded(manifest)

    ledger.write_text(f"## entry\n- `{victim}`\n", encoding="utf-8")
    assert victim not in freeze.unrecorded(manifest)


def test_inactive_freeze_passes():
    manifest = json.loads(json.dumps(freeze.load_manifest()))
    manifest["active"] = False
    manifest["files"] = {"tests/perf/results/nonexistent.json": "0" * 64}
    assert freeze.unrecorded(manifest) == []
