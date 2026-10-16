"""CLAUDE.md §11 indexes docs/dev/modules/ one-to-one.

The per-area conventions moved out of CLAUDE.md on 2026-10-17. An agent
finds them only through the index, so a file missing from it is a set of
rules nobody reads, and an index entry without a file is a dead end.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_every_module_note_is_indexed_and_every_index_entry_exists():
    claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    section = claude[claude.index("## 11.") : claude.index("## 12.")]
    indexed = set(re.findall(r"\(docs/dev/modules/([\w-]+\.md)\)", section))
    on_disk = {p.name for p in (REPO / "docs" / "dev" / "modules").glob("*.md")}
    assert indexed == on_disk
