"""The docs-site hooks: per-page descriptions and the published llms.txt."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "seo", ROOT / "scripts" / "mkdocs_seo_hooks.py"
)
seo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seo)


def test_description_is_the_first_prose_paragraph_in_plain_text():
    md = (
        "# Title\n\n> a note\n\n```python\nx = 1\n```\n\n"
        "**StatsPAI** runs [DiD](x.md) with `sp.did` and *robust* errors.\n"
        "Second line.\n\nNext paragraph."
    )
    assert (
        seo.describe(md)
        == "StatsPAI runs DiD with sp.did and robust errors. Second line."
    )


def test_description_is_cut_at_a_word_and_short_text_is_ignored():
    long = "# T\n\n" + "word " * 60
    out = seo.describe(long)
    assert len(out) <= seo.MAX_LEN and out.endswith("…") and "  " not in out
    assert seo.describe("# T\n\nToo short.") == ""


def test_llms_links_become_absolute():
    text = (
        "- [A](AGENTS.md): x\n- [B](docs/guides/agent_api.md#mcp)\n"
        "- [C](docs/index.md)\n- [D](https://example.org/)\n"
    )
    out = seo.absolutize_llms(text, "https://h.io/S/")
    assert (
        "[A](https://github.com/brycewang-stanford/StatsPAI/blob/main/AGENTS.md)" in out
    )
    assert "[B](https://h.io/S/guides/agent_api/#mcp)" in out
    assert "[C](https://h.io/S/)" in out
    assert "[D](https://example.org/)" in out


def test_every_relative_link_in_llms_txt_resolves_in_the_repo():
    import re

    text = (ROOT / "llms.txt").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s#]+)", text):
        if "://" not in target:
            assert (ROOT / target).exists(), target
