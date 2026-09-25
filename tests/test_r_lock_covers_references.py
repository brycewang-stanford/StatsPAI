"""The R lock records every package the Track A R scripts call.

The manuscript says the R reference environment is pinned by
``tests/r_parity/renv.lock``. Until 1.32.0 the lock generator's hand-kept
reference list had fallen behind the parity scripts: fifteen reference
packages that modules 72-89 load (``did2s``, ``interflex``, ``staggered``,
``DIDmultiplegt`` and others) were not in the lock at all, and ``rdrobust``
was pinned at a version older than the one that produced the committed
goldens. This test scans every R parity script for ``library()``,
``require()``, ``requireNamespace()`` and ``pkg::`` calls and requires each
package to be in the lock, so the list cannot drift again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "r_parity"

BASE_R = {
    "base",
    "stats",
    "utils",
    "methods",
    "graphics",
    "grDevices",
    "tools",
    "parallel",
    "splines",
    "stats4",
    "grid",
    "compiler",
    "datasets",
    "tcltk",
}
#: Tokens the regex matches that are not packages (``R::`` in a comment
#: example, a data column accessed as ``year::``).
NOT_PACKAGES = {"R", "year"}

_CALL = re.compile(
    r"(?:library|require|requireNamespace)\(\s*[\"']?([A-Za-z][A-Za-z0-9.]*)"
)
_NS = re.compile(r"\b([A-Za-z][A-Za-z0-9.]*)::")


def _referenced_packages() -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for script in sorted(PARITY.glob("*.R")):
        code = re.sub(r"#.*", "", script.read_text(encoding="utf-8"))
        for pkg in set(_CALL.findall(code)) | set(_NS.findall(code)):
            used.setdefault(pkg, set()).add(script.name)
    return used


def test_every_referenced_r_package_is_in_the_lock():
    lock = json.loads((PARITY / "renv.lock").read_text(encoding="utf-8"))["Packages"]
    missing = {
        pkg: sorted(files)
        for pkg, files in _referenced_packages().items()
        if pkg not in lock and pkg not in BASE_R and pkg not in NOT_PACKAGES
    }
    assert not missing, (
        "R packages used by parity scripts but absent from renv.lock; add them "
        "to REFERENCE_PKGS in tests/r_parity/_gen_renv_lock.R and regenerate: "
        f"{missing}"
    )


def test_lock_versions_match_the_goldens_for_the_core_references():
    """The core references' locked versions produced the committed goldens."""
    lock = json.loads((PARITY / "renv.lock").read_text(encoding="utf-8"))["Packages"]
    golden = {
        "rdrobust": "06_rd",
        "did": "04_csdid",
        "fixest": "03_hdfe",
        "grf": "13_causal_forest",
        "HonestDiD": "10_honest_did",
    }
    for pkg, module in golden.items():
        text = (PARITY / "results" / f"{module}_R.json").read_text(encoding="utf-8")
        recorded = set(re.findall(rf'"{pkg}":\s*"([0-9][0-9.\-]*)"', text))
        assert recorded, (pkg, module)
        norm = {v.replace("-", ".") for v in recorded}
        assert lock[pkg]["Version"].replace("-", ".") in norm, (
            pkg,
            lock[pkg]["Version"],
            recorded,
        )
