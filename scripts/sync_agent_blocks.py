#!/usr/bin/env python3
"""
Regenerate ``## For Agents`` blocks inside MkDocs guides.

For every guide that contains a marker pair like::

    <!-- AGENT-BLOCK-START: did -->
    ...old content...
    <!-- AGENT-BLOCK-END -->

this script replaces the content between the markers with the
auto-rendered output of :func:`statspai.render_agent_block` for
the named function. Guides without markers are left untouched.

Running the script is idempotent: re-running on a synced tree is
a no-op.  Intended to be invoked before committing registry
metadata changes, and in CI to fail if docs have drifted.

Usage
-----

    python scripts/sync_agent_blocks.py            # rewrite in-place
    python scripts/sync_agent_blocks.py --check    # CI mode: exit 1 if drift

Environment: requires the ``statspai`` package to be importable
(``pip install -e .`` from the repo root).
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

START_RE = re.compile(
    r"<!--\s*AGENT-BLOCK-START:\s*(?P<name>[A-Za-z0-9_]+)\s*-->"
)
END_RE = re.compile(r"<!--\s*AGENT-BLOCK-END\s*-->")


def _sync_text(text: str, path: Path) -> Tuple[str, List[str]]:
    """Return ``(new_text, [function_names_updated])``."""
    from statspai import render_agent_block  # lazy for import speed

    updated: List[str] = []
    out_lines: List[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = START_RE.search(line)
        if not m:
            out_lines.append(line)
            i += 1
            continue
        name = m.group("name")

        # Scan for matching END marker
        j = i + 1
        while j < len(lines) and not END_RE.search(lines[j]):
            j += 1
        if j >= len(lines):
            raise RuntimeError(
                f"{path}: unmatched AGENT-BLOCK-START:{name} (no END marker)"
            )

        # Emit: start marker, a blank line, rendered body, blank line, end marker.
        try:
            body = render_agent_block(name)
        except KeyError:
            raise RuntimeError(
                f"{path}: AGENT-BLOCK references unknown function '{name}'. "
                "Check the registry."
            )
        if not body:
            raise RuntimeError(
                f"{path}: function '{name}' has no agent-native metadata. "
                "Populate assumptions/failure_modes/alternatives in the registry "
                "before adding an AGENT-BLOCK marker."
            )

        updated.append(name)
        out_lines.append(line)  # start marker (preserved verbatim)
        out_lines.append("")
        out_lines.extend(body.rstrip().splitlines())
        out_lines.append("")
        out_lines.append(lines[j])  # end marker

        i = j + 1

    new_text = "\n".join(out_lines)
    if text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, updated


def _process(path: Path, check: bool) -> Tuple[int, int, int]:
    """Return (edited, drift, names_touched)."""
    text = path.read_text(encoding="utf-8")
    new_text, names = _sync_text(text, path)
    if not names:
        return 0, 0, 0
    if new_text == text:
        return 0, 0, len(names)
    if check:
        diff = "\n".join(difflib.unified_diff(
            text.splitlines(), new_text.splitlines(),
            fromfile=str(path), tofile=str(path) + " (expected)",
            lineterm="",
        ))
        print(diff, file=sys.stderr)
        return 0, 1, len(names)
    path.write_text(new_text, encoding="utf-8")
    return 1, 0, len(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if any doc is out of sync.",
    )
    args = parser.parse_args()

    guides = sorted(DOCS_DIR.rglob("*.md"))
    if not guides:
        print(f"No markdown files found under {DOCS_DIR}", file=sys.stderr)
        return 1

    total_edits = 0
    total_drift = 0
    total_blocks = 0
    for g in guides:
        edits, drift, touched = _process(g, args.check)
        total_edits += edits
        total_drift += drift
        total_blocks += touched
        if edits:
            print(f"  synced  {g.relative_to(REPO_ROOT)}")
        elif drift:
            print(f"  drift   {g.relative_to(REPO_ROOT)}", file=sys.stderr)

    print(
        f"sync_agent_blocks: blocks={total_blocks} edits={total_edits} "
        f"drift={total_drift}"
    )
    if args.check and total_drift:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
