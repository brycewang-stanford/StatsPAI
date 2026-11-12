"""Run the task pages in docs/tasks/ exactly as a reader would.

These pages are the entry points search engines and AI assistants are most
likely to quote, so a stale number or a renamed argument there does the most
damage. Each page's ```python blocks run in order in one namespace, inside a
temporary directory; every non-``...`` line of each ```text block must then
appear, in order, in the printed output. Blocks fenced as ```py are shown
but not run (shell-dependent or agent-side snippets).
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings
from pathlib import Path

import pytest

import statspai as sp

TASKS = Path(__file__).resolve().parents[1] / "docs" / "tasks"
PAGES = sorted(p for p in TASKS.glob("*.md"))


def _blocks(text: str, lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)\n```", text, flags=re.DOTALL)


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def test_task_pages_exist():
    assert len(PAGES) >= 8, [p.name for p in PAGES]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_task_page_runs_and_prints_what_it_shows(page, tmp_path, monkeypatch):
    text = page.read_text(encoding="utf-8")
    code = _blocks(text, "python")
    if not code:
        pytest.skip("no runnable blocks")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    buf = io.StringIO()
    ns: dict = {}
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        for block in code:
            exec(compile(block, str(page), "exec"), ns)  # noqa: S102 - our own docs
    import matplotlib.pyplot as plt

    plt.close("all")
    out = [_norm(line) for line in buf.getvalue().splitlines() if line.strip()]
    for block in _blocks(text, "text"):
        pos = 0
        for want in (_norm(x) for x in block.splitlines()):
            if not want or want == "...":
                continue
            hits = [i for i in range(pos, len(out)) if want in out[i]]
            assert hits, f"{page.name}: {want!r} not printed (after line {pos})"
            pos = hits[0] + 1


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_task_page_names_and_links_resolve(page):
    text = page.read_text(encoding="utf-8")
    missing = []
    for dotted in sorted(set(re.findall(r"`sp\.([A-Za-z_][\w.]*)", text))):
        obj = sp
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                missing.append(dotted)
                break
    assert not missing, missing
    for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", text):
        assert (page.parent / target).resolve().exists(), target
