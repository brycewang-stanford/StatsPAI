"""Guards for the teaching notebooks in ``examples/notebooks``.

Two layers. The fast one reads the committed files: every tutorial ships
executed outputs with no error in them, names the standard kernel, leaks no
local path and is listed in ``examples/README.md``. The slow one executes
each tutorial end to end against this tree, so an API change that breaks a
tutorial fails here and not in a reader's hands:

    pytest -m slow tests/test_example_notebooks.py

The slow layer needs the ``notebooks`` extra and matplotlib, and runs the
kernel on the interpreter that runs pytest (a user-level ``python3`` kernel
may point at another environment).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "examples" / "notebooks"
TUTORIALS = sorted(NB_DIR.glob("tutorial_*.ipynb"))
IDS = [p.stem for p in TUTORIALS]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_tutorials_are_present():
    assert len(TUTORIALS) >= 10


@pytest.mark.parametrize("path", TUTORIALS, ids=IDS)
def test_committed_tutorial_is_executed_and_clean(path):
    nb = _load(path)
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["outputs"] for c in code if "".join(c["source"]).strip())
    for cell in code:
        for out in cell["outputs"]:
            assert out["output_type"] != "error", "".join(cell["source"])[:200]
            assert out.get("name") != "stderr", "".join(cell["source"])[:200]
    text = path.read_text(encoding="utf-8")
    for leak in ("/Users/", "/home/", "/private/", "C:\\\\Users"):
        assert leak not in text
    # One import, the documented alias.
    assert "import statspai as sp" in text


@pytest.mark.parametrize("path", TUTORIALS, ids=IDS)
def test_tutorial_is_listed_in_the_examples_readme(path):
    readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    assert f"notebooks/{path.name}" in readme


@pytest.mark.parametrize("path", TUTORIALS, ids=IDS)
def test_tutorial_references_come_from_the_master_bib(path):
    # References are printed by sp.bibtex(keys); a key that is not in
    # paper.bib would raise when the notebook runs. Checked here without
    # running it.
    import ast

    import statspai as sp

    nb = _load(path)
    last = "".join(nb["cells"][-1]["source"])
    assert "sp.bibtex(keys)" in last
    tree = ast.parse(last)
    keys = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.targets[0].id == "keys"
    )
    assert keys
    sp.bibtex(keys)


@pytest.mark.slow
@pytest.mark.parametrize("path", TUTORIALS, ids=IDS)
def test_tutorial_executes_against_this_tree(path, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")

    spec = tmp_path / "kernels" / "statspai-test"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "StatsPAI test",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "JUPYTER_PATH",
        os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("JUPYTER_PATH")])),
    )
    # The kernel must import this tree, also from a worktree.
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
        ),
    )
    monkeypatch.setenv("MPLBACKEND", "Agg")

    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=900,
        kernel_name="statspai-test",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not any(o.get("name") == "stderr" for o in cell.outputs)
