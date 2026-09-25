"""Every reviewer-facing harness script compiles on the running Python.

The package supports Python 3.9-3.13, and the JSS reproduction lock pins
3.10. ``tests/r_parity/verify_reproduce.py`` -- the Tier 2 entry point --
once used a backslash inside an f-string expression, which only Python 3.12
accepts, so the documented R reproduction path died with a ``SyntaxError``
in the lock environment while CI (which never imported the script) stayed
green. Compiling every harness script under each CI interpreter closes that
gap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_GLOBS = (
    "tests/r_parity/*.py",
    "tests/stata_parity/*.py",
    "tests/orig_parity/*.py",
    "tests/perf/*.py",
    "tests/coverage_monte_carlo/*.py",
    "tests/coverage_monte_carlo/mechanisms/*.py",
    "scripts/*.py",
    "Paper-JSS/replication/*.py",
    "Paper-JSS/replication/scripts/*.py",
)
SCRIPTS = sorted(
    {p for pattern in HARNESS_GLOBS for p in ROOT.glob(pattern) if p.is_file()}
)


@pytest.mark.skipif(not SCRIPTS, reason="harness scripts not present")
@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_harness_script_compiles(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
