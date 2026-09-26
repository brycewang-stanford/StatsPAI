"""Every ``sp.<name>`` the research-tasks guide mentions must exist."""

import re
from pathlib import Path

import statspai as sp

DOC = Path(__file__).resolve().parents[1] / "docs" / "guides" / "research_tasks.md"


def test_every_named_function_exists():
    text = DOC.read_text(encoding="utf-8")
    names = set(re.findall(r"`sp\.([A-Za-z_][\w.]*)", text))
    missing = []
    for dotted in sorted(names):
        obj = sp
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                missing.append(dotted)
                break
    assert not missing, missing


def test_linked_guides_exist():
    text = DOC.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+\.md)\)", text):
        assert (DOC.parent / target).resolve().exists(), target
