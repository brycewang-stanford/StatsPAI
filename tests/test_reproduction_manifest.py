"""``docs/reproduction_manifest.json``: when each reference was re-derived.

Review item R2 asked for a machine-readable record of the last
re-derivation of every reference: status, reference version, platform,
input and output hashes, and the reason where a side is missing. The
manifest is assembled from the three reproducibility reports, the
provenance block of every golden file and the CI workflow; nothing is
re-run to build it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_reproduction_manifest.py"
MANIFEST = ROOT / "docs" / "reproduction_manifest.json"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="source checkout only (scripts/ not installed)"
)


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_reproduction_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_committed_manifest_is_current(builder, manifest):
    fresh = builder.build(with_git=False)
    assert builder._strip_dates(manifest) == builder._strip_dates(fresh), (
        "docs/reproduction_manifest.json is stale: "
        "run python scripts/build_reproduction_manifest.py"
    )


def test_every_reference_module_has_a_status(manifest):
    sides = manifest["sides"]
    n_r = len(list((ROOT / "tests/r_parity/results").glob("*_R.json")))
    n_stata = len(list((ROOT / "tests/stata_parity/results").glob("*_Stata.json")))
    assert sides["R"]["n_modules"] == n_r >= 89
    assert sides["Stata"]["n_modules"] == n_stata >= 85
    for side in ("R", "Stata", "py"):
        block = sides[side]
        assert block["n_reproduce"] == block["n_modules"], (
            side,
            [
                m
                for m, e in block["modules"].items()
                if e["status"] not in ("reproduces", "no_reference")
            ],
        )
        assert len(block["report_sha256"]) == 64


def test_hashes_are_the_bytes_on_disk(manifest):
    entry = manifest["sides"]["R"]["modules"]["01_ols"]
    for key, path in (
        ("golden_sha256", "tests/r_parity/results/01_ols_R.json"),
        ("input_sha256", "tests/r_parity/data/01_ols.csv"),
    ):
        data = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert entry[key] == hashlib.sha256(data).hexdigest()
    assert entry["reference"]["r_version"].startswith("R version ")
    assert entry["worst_rel_estimate"] <= manifest["reproducibility_tolerance"]


def test_ci_coverage_is_stated_per_module(manifest):
    modules = manifest["sides"]["R"]["modules"]
    in_ci = sorted(m for m, e in modules.items() if e["rederived_in_ci"])
    assert len(in_ci) == 17 and "01_ols" in in_ci
    # A heavy reference is a frozen artefact, and says so.
    assert modules["13_causal_forest"]["rederived_in_ci"] is False


def test_missing_stata_side_carries_a_measured_reason(manifest):
    stata = manifest["sides"]["Stata"]["modules"]
    missing = {m: e for m, e in stata.items() if e["status"] == "no_reference"}
    assert (
        len(missing)
        == manifest["sides"]["R"]["n_modules"] - manifest["sides"]["Stata"]["n_modules"]
    )
    for module, entry in missing.items():
        assert entry["skip_reason"] and len(entry["skip_reason"]) > 30, module


def test_each_side_names_how_to_rederive_it(manifest):
    for side, block in manifest["sides"].items():
        assert block["rederive_with"].startswith("python tests/")
        assert (ROOT / block["report"]).exists()
