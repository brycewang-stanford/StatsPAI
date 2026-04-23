#!/usr/bin/env python3
"""
Suggest ``[@bibkey]`` back-fills for paper.bib orphan entries.

An orphan is a paper.bib entry that no ``[@bibkey]`` citation in
src/ / docs/ / paper.md refers to. Many orphans are NOT dead weight —
they are papers whose **arXiv id or DOI is already quoted in a docstring
References section**, just without the canonical ``[@bibkey]`` link.

This tool finds those instant back-fill candidates by cross-indexing
identifiers: for each orphan,

  1. extract its arXiv id (from ``journal={arXiv:...}`` or ``eprint={...}``
     or bare ``arXiv:...`` text in the bib entry) and its DOI;
  2. grep src/ + docs/ + paper.md for each identifier;
  3. for every match, check whether a ``[@key]`` annotation is already
     present on the SAME line — if so, the line already has a (different
     or same) bibkey and we skip; otherwise this is a back-fill candidate.

By default the tool only prints the plan. Pass ``--apply`` to actually
append ``[@orphan_key]`` to each candidate line in place.

Usage
-----
    python tools/suggest_bibkey_backfills.py                # dry-run
    python tools/suggest_bibkey_backfills.py --apply        # mutate files
    python tools/suggest_bibkey_backfills.py --json out.json
    python tools/suggest_bibkey_backfills.py --bib custom.bib

Stdlib only — safe to run in any repo snapshot with the other two
auditors present.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(TOOLS_DIR))
from audit_bib_duplicates import parse_bib  # noqa: E402
from audit_bib_coverage import build_report  # noqa: E402

DEFAULT_BIB = REPO_ROOT / "paper.bib"
DEFAULT_ROOTS = ("src", "docs", "paper.md")

# Capture the arXiv id from a bib entry's own body. Accepts the
# common StatsPAI patterns: ``journal={arXiv:2408.12345}``,
# ``eprint={2408.12345}``, or bare ``arXiv:2408.12345`` prose.
ARXIV_IN_BIB = re.compile(
    r"(?:arXiv[:\s]*|eprint\s*=\s*[{\"]?)(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)

# A line already has a bibkey annotation if it contains ``[@xxx]``.
HAS_BIBKEY = re.compile(r"\[@[A-Za-z0-9][A-Za-z0-9_\-]*\]")

# Lines that look like Python string-literal fragments being
# concatenated into a bibtex entry (e.g. inside ``_CITATIONS`` dicts).
# Annotating these would PUSH ``[@bibkey]`` INTO the generated bibtex
# string, breaking biblatex downstream. We look for two clear signals:
#
#   1. the line ends with ``\n"`` or ``\n',`` — the escape-newline +
#      quote pattern is the canonical fragment-concatenation tail;
#   2. the line contains ``field={`` where ``field`` is a bibtex entry
#      key (``journal``, ``eprint``, ``doi``, ``author``, ``title``,
#      ``year``, ``volume``, ``number``, ``pages``) — that's a bibtex
#      synthesis line regardless of how it terminates.
#
# NOTE: triple-quote closings (``"""``) end with a quote too, but never
# with ``\n"``, so they pass through and can be annotated.
UNSAFE_BIBTEX_SYNTHESIS = re.compile(
    r"\\n[\"']\s*,?\s*$"
    r"|\b(?:journal|eprint|doi|author|title|year|volume|number|pages)\s*=\s*[{\"]",
    re.IGNORECASE,
)

# ``reference="..."`` kwargs — extremely common in ``registry.py``.
# Appending ``[@bibkey]`` after the closing quote either breaks Python
# syntax or puts the tag outside the value. Unsafe to auto-annotate.
UNSAFE_KWARG = re.compile(r"\breference\s*=\s*[\"']", re.IGNORECASE)

# Lines inside a dict value or tuple, ending with ``",`` or ``',``:
# appending a bibkey after the comma produces a bare expression.
UNSAFE_DICT_VALUE = re.compile(r"[\"']\s*,\s*$")


@dataclass
class BackfillCandidate:
    key: str                    # orphan bib key to add
    identifier: str             # the arxiv id or doi that triggered the match
    id_kind: str                # 'arxiv' | 'doi'
    file: str                   # repo-relative path
    line: int                   # 1-indexed
    original: str               # the current line content
    suggested: str              # the proposed new line content


def find_candidates(bib_path: Path, roots: list[Path],
                    *, cwd: Optional[Path] = None) -> list[BackfillCandidate]:
    """Scan orphan bib entries for back-fill candidates.

    ``cwd`` is the working directory for ``git grep`` — defaults to the
    parent of ``bib_path``, so the tool works on any repo snapshot, not
    just StatsPAI's hard-coded ``REPO_ROOT``.
    """
    if cwd is None:
        cwd = bib_path.resolve().parent
    report = build_report(bib_path, roots)
    entries = {e.key: e for e in parse_bib(bib_path.read_text(encoding="utf-8"))}

    candidates: list[BackfillCandidate] = []
    for key in sorted(report.orphan):
        entry = entries.get(key)
        if entry is None:
            continue
        # Collect identifiers: arXiv id (from bib body) + DOI.
        identifiers: list[tuple[str, str]] = []
        m = ARXIV_IN_BIB.search(entry.raw)
        if m:
            identifiers.append(("arxiv", m.group(1)))
        if entry.doi:
            identifiers.append(("doi", entry.doi))
        if not identifiers:
            continue

        for kind, ident in identifiers:
            # Build the list of scan-path args relative to the cwd so
            # ``git grep`` finds the requested roots under any repo.
            root_args: list[str] = []
            for root in roots:
                try:
                    root_args.append(str(root.resolve().relative_to(cwd)))
                except ValueError:
                    # Root lives outside cwd — fall back to absolute path.
                    root_args.append(str(root.resolve()))
            r = subprocess.run(
                ["git", "grep", "-n", "-F", ident, "--", *root_args],
                capture_output=True, text=True, check=False,
                cwd=cwd,
            )
            for raw in r.stdout.splitlines():
                if not raw.strip():
                    continue
                # git grep -n yields "path:line:content"
                try:
                    path_str, line_str, content = raw.split(":", 2)
                except ValueError:
                    continue
                # Skip paper.bib itself (the orphan entry lives there).
                if path_str == "paper.bib" or path_str.endswith("/paper.bib"):
                    continue
                try:
                    line_num = int(line_str)
                except ValueError:
                    continue
                # Skip lines that already carry a [@bibkey] annotation —
                # they are already the canonical form, adding another
                # bibkey would be wrong.
                if HAS_BIBKEY.search(content):
                    continue
                # Skip lines that are inside a Python string literal or
                # dict value — appending [@bibkey] would either inject
                # into a generated bibtex string, or cause a syntax error.
                if (UNSAFE_BIBTEX_SYNTHESIS.search(content)
                        or UNSAFE_KWARG.search(content)
                        or UNSAFE_DICT_VALUE.search(content)):
                    continue
                suggested = _append_bibkey(content, key)
                candidates.append(BackfillCandidate(
                    key=key, identifier=ident, id_kind=kind,
                    file=path_str, line=line_num,
                    original=content, suggested=suggested,
                ))
    return candidates


def _append_bibkey(line: str, key: str) -> str:
    """Append ``[@key]`` to the end of the line's content, preserving
    any trailing punctuation and any trailing quote-close markers common
    in Python docstrings written as string literals (e.g. the line
    ``"    arXiv:1234.5678\n"``).
    """
    stripped = line.rstrip("\n")
    # Detect trailing \n escape artifact from multi-line Python string
    # concatenation — common in _CITATIONS dicts where each line ends
    # with ``\n"``. We should insert BEFORE that escape/quote.
    #
    # Most common patterns (in order of detection):
    #   1. ``... arXiv:XXXX.YYYY``   (normal docstring line)
    #   2. ``... arXiv:XXXX.YYYY\n"``   (string literal fragment)
    #   3. ``... arXiv:XXXX.YYYY."``    (string literal with trailing period)
    m = re.match(r"^(.*?)(\\n[\"']|[\"'])\s*$", stripped)
    if m:
        prefix, tail = m.group(1), m.group(2)
        return f"{prefix} [@{key}]{tail}" + ("\n" if line.endswith("\n") else "")
    return f"{stripped} [@{key}]" + ("\n" if line.endswith("\n") else "")


def format_plan(candidates: list[BackfillCandidate], *, max_rows: int = 50) -> str:
    lines = [f"paper.bib orphan back-fill plan"]
    lines.append(f"  total candidate lines: {len(candidates)}")
    # Group by orphan key for readability.
    by_key: dict[str, list[BackfillCandidate]] = {}
    for c in candidates:
        by_key.setdefault(c.key, []).append(c)
    lines.append(f"  unique orphan keys that can be back-filled: {len(by_key)}")
    lines.append("")

    shown = 0
    for key, items in sorted(by_key.items()):
        if shown >= max_rows:
            remaining = len(by_key) - shown
            lines.append(f"  ... {remaining} more orphan keys (truncated)")
            break
        lines.append(f"[@{key}]  — {len(items)} line(s)")
        for c in items:
            lines.append(f"  {c.file}:{c.line}   (via {c.id_kind} {c.identifier})")
            lines.append(f"    - {c.original.rstrip()}")
            lines.append(f"    + {c.suggested.rstrip()}")
        lines.append("")
        shown += 1
    return "\n".join(lines)


def apply_candidates(candidates: list[BackfillCandidate]) -> dict[str, int]:
    """Mutate each target file in place. Returns {file: lines_changed}.

    Applies line replacements from highest line number to lowest so
    multiple patches on the same file don't shift offsets.
    """
    changes_by_file: dict[str, list[BackfillCandidate]] = {}
    for c in candidates:
        changes_by_file.setdefault(c.file, []).append(c)

    stats: dict[str, int] = {}
    for file, items in changes_by_file.items():
        path = REPO_ROOT / file
        if not path.exists():
            continue
        text_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        # Index from high to low so offsets remain valid.
        items.sort(key=lambda c: c.line, reverse=True)
        changed = 0
        for c in items:
            idx = c.line - 1
            if idx < 0 or idx >= len(text_lines):
                continue
            # Only modify if the current content still matches what we
            # recorded — defensive against concurrent edits.
            if text_lines[idx].rstrip("\n") != c.original.rstrip("\n"):
                continue
            text_lines[idx] = c.suggested if c.suggested.endswith("\n") \
                else c.suggested + "\n"
            # Normalise: if original didn't end with newline (last line
            # of file), don't insert a spurious one.
            if not c.original.endswith("\n") and text_lines[idx].endswith("\n"):
                text_lines[idx] = text_lines[idx][:-1]
            changed += 1
        if changed:
            path.write_text("".join(text_lines), encoding="utf-8")
            stats[file] = changed
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", default=str(DEFAULT_BIB),
                        help="Path to paper.bib (default: repo root)")
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS),
                        help="Paths to scan for [@bibkey] citations.")
    parser.add_argument("--apply", action="store_true",
                        help="Mutate files in place. Default is dry-run "
                             "(prints plan only).")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Dump structured candidate list to JSON.")
    parser.add_argument("--max-rows", type=int, default=50,
                        help="Truncate human report to N orphan keys "
                             "(0 = show all, default: 50).")
    args = parser.parse_args(argv)

    bib_path = Path(args.bib)
    if not bib_path.exists():
        print(f"error: bib file not found: {bib_path}", file=sys.stderr)
        return 2

    roots: list[Path] = []
    for r in args.roots:
        p = Path(r)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            roots.append(p)
    if not roots:
        print("error: no scan roots found", file=sys.stderr)
        return 2

    candidates = find_candidates(bib_path, roots)

    max_rows = args.max_rows if args.max_rows > 0 else 10**9
    print(format_plan(candidates, max_rows=max_rows))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([asdict(c) for c in candidates],
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}", file=sys.stderr)

    if args.apply and candidates:
        print("\napplying changes…", file=sys.stderr)
        stats = apply_candidates(candidates)
        for file, n in sorted(stats.items()):
            print(f"  {file}: {n} line(s) annotated", file=sys.stderr)
        total = sum(stats.values())
        print(f"\napplied: {total} annotations across {len(stats)} file(s)",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
