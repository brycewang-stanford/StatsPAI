"""``docs/guides/data_layer.md`` runs and prints what it shows.

Every ```python block of the guide is executed in order in one namespace;
when it is followed by a ```text block, the printed output must equal that
block exactly. The numbers themselves are checked against Stata 18 in
``test_data_layer.py``, ``test_stata_formula_syntax.py``,
``test_egen_public.py`` and ``test_stata_xtset_attrs.py``; this test keeps
the guide from drifting away from the code.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
import warnings

import pytest

GUIDE = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "guides" / "data_layer.md"
)
_FENCE = re.compile(r"```(python|text)\n(.*?)```", re.S)


def _blocks():
    text = GUIDE.read_text(encoding="utf-8")
    found = _FENCE.findall(text)
    pairs = []
    for i, (kind, body) in enumerate(found):
        if kind != "python":
            continue
        shown = (
            found[i + 1][1]
            if i + 1 < len(found) and found[i + 1][0] == "text"
            else None
        )
        pairs.append((body, shown))
    return pairs


def test_guide_has_examples():
    pairs = _blocks()
    assert len(pairs) >= 10
    assert all(shown is not None for _, shown in pairs)


def test_every_block_prints_what_the_guide_shows():
    namespace: dict = {}
    for k, (code, shown) in enumerate(_blocks()):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, f"data_layer.md block {k}", "exec"), namespace)
        if shown is not None:
            assert buf.getvalue() == shown, f"block {k} output differs"


def test_guide_is_in_the_nav():
    mkdocs = (GUIDE.parents[2] / "mkdocs.yml").read_text(encoding="utf-8")
    assert "guides/data_layer.md" in mkdocs


@pytest.mark.parametrize("ref", ["test_data_layer.py", "test_egen_public.py"])
def test_the_stata_references_named_exist(ref):
    assert (GUIDE.parents[2] / "tests" / ref).exists()
