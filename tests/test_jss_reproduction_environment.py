"""Contracts for the JSS reviewer reproduction-environment audit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = (
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "reproduction_environment_audit.py"
)
RESULTS = REPO_ROOT / "Paper-JSS" / "replication" / "results"

# ``Paper-JSS/`` is a git-ignored, local-only submission working directory
# (see ``.gitignore``); it is absent from fresh clones and CI checkouts. The
# JSS source-snapshot audits only make sense against that local tree, so skip
# the whole module when it is absent rather than erroring on missing scripts.
pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "Paper-JSS").exists(),
    reason="Paper-JSS/ submission working dir is git-ignored / local-only",
)


def test_reproduction_environment_audit_guards_seeded_stochastic_outputs() -> None:
    """JSS requires seeded simulations; the reviewer audit must enforce that."""
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1780185600")
    res = subprocess.run(
        [sys.executable, str(AUDIT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert res.returncode == 0, res.stderr

    payload = json.loads((RESULTS / "reproduction_environment_audit.json").read_text())
    rng = payload["random_seeding"]
    requirements = payload["requirements"]
    makefile = payload["makefile"]
    reproduce = payload["reproduce"]

    assert payload["status"] == "PASS"
    assert requirements["package_count"] >= 20
    assert {
        "pyarrow",
        "pillow",
        "pytest",
        "pytest-cov",
        "rdrobust",
        "setuptools",
        "wheel",
    } <= set(requirements["packages"])
    assert makefile["pandoc_markdown_optional"] is True
    assert reproduce["tier1_live_external_call_count"] == 0
    assert reproduce["tier1_live_external_dependency_markers"] == []
    assert payload["tier1_transcript_no_r_stata"] is True
    assert rng["stochastic_file_count"] >= 10
    assert rng["seeded_stochastic_file_count"] == rng["stochastic_file_count"]
    assert rng["unseeded_stochastic_file_count"] == 0
    assert rng["unseeded_stochastic_files"] == []
    assert {
        "Paper-JSS/replication/scripts/ex06_causal_impact.py",
        "tests/coverage_monte_carlo/run_b1000.py",
        "tests/perf/01_hdfe_perf.py",
        "tests/perf/01_hdfe_perf.R",
    } <= set(rng["stochastic_files"])

    md = (RESULTS / "reproduction_environment_audit.md").read_text()
    assert "Python requirement packages: 23" in md
    assert "Optional Pandoc Markdown export: True" in md
    assert "Tier-1 transcript states no R/Stata headline path: True" in md
    assert "Tier-1 live R/Stata dependency markers: 0" in md
    assert "Seeded stochastic reproduction files:" in md
    assert "Unseeded stochastic reproduction files: 0" in md
