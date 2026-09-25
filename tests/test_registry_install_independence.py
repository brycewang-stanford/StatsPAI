"""Guard: the registry reads the same from a pip install as from a checkout.

Until 1.31.0 ``registry._apply_validation_evidence`` scanned ``tests/`` at
import time whenever a source tree sat next to the package. The grades
agreed with a wheel install (both are set from the packaged
``_parity_index.json``), but 557 functions printed different
``validation_notes``: a checkout added scan-derived notes that an installed
release never shows. The JSS manuscript prints ``validation_notes`` in a
listing, so the listing described output no ``pip install`` could return.

This test imports a copy of the package from a directory with no
``pyproject.toml`` or ``tests/`` above it -- what an installed wheel sees --
and requires every function's ``describe_function`` record to equal the
in-tree one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import statspai as sp

_PACKAGE = Path(sp.__file__).resolve().parent

_DUMP = """
import json, sys
import statspai as sp
out = {name: sp.describe_function(name) for name in sp.list_functions()}
json.dump({"file": sp.__file__, "specs": out}, sys.stdout, sort_keys=True,
          default=str)
"""


def _dump(pythonpath: Path, cwd: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    env.pop("STATSPAI_REPO_ROOT", None)
    proc = subprocess.run(
        [sys.executable, "-c", _DUMP],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(proc.stdout)


def test_describe_function_is_identical_without_a_source_tree(tmp_path):
    detached = tmp_path / "site-packages"
    shutil.copytree(
        _PACKAGE,
        detached / "statspai",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    in_tree = _dump(_PACKAGE.parent, tmp_path)
    installed = _dump(detached, tmp_path)

    # The copy really was imported from outside any checkout.
    assert Path(installed["file"]).resolve().is_relative_to(detached.resolve())
    assert not Path(in_tree["file"]).resolve().is_relative_to(tmp_path.resolve())

    assert set(installed["specs"]) == set(in_tree["specs"])
    differing = sorted(
        name
        for name, spec in in_tree["specs"].items()
        if installed["specs"][name] != spec
    )
    assert differing == [], (
        f"{len(differing)} functions describe themselves differently when "
        f"installed, e.g. {differing[:5]}"
    )
