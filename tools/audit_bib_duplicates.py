#!/usr/bin/env python3
"""
Duplicate-entry auditor for ``paper.bib``.

Flags three classes of integrity bug that §10 "single-source bibliography"
needs to prevent:

1. **Duplicate bib keys** — the same key defined twice. biblatex treats the
   second definition as an error / warning and the rendered bibliography
   becomes order-dependent.

2. **Duplicate DOIs** — two different keys pointing at the same paper.
   Citations drift because half the codebase cites ``@a`` and the other
   half cites ``@b`` for the same work.

3. **Duplicate arXiv IDs** — same as above but on the arXiv side.

Usage
-----
    python tools/audit_bib_duplicates.py                     # human report
    python tools/audit_bib_duplicates.py --strict            # non-zero exit on any dup
    python tools/audit_bib_duplicates.py --bib paper.bib     # override path
    python tools/audit_bib_duplicates.py --json report.json  # machine-readable

Stdlib only — intended to run in CI as a §10 enforcement gate alongside
``audit_citations.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIB = REPO_ROOT / "paper.bib"

ENTRY_HEADER_RE = re.compile(r"@(?P<kind>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,")
DOI_FIELD_RE = re.compile(r"\bdoi\s*=\s*[{\"]([^}\"]+)[}\"]", re.IGNORECASE)
ARXIV_FIELD_RE = re.compile(
    r"(?:arXiv[:\s]*|eprint\s*=\s*[{\"]?)(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


@dataclass
class BibEntry:
    key: str
    kind: str
    doi: Optional[str]
    arxiv: Optional[str]
    line: int          # first line of the entry (1-indexed)
    raw: str


def parse_bib(text: str) -> list[BibEntry]:
    """Parse bib entries without depending on a full bibtex library.

    Splits on top-level ``@`` boundaries, which works for the way
    ``paper.bib`` is written (one entry per ``@... { ... }`` block separated
    by blank lines). Within an entry we only need three things: key, DOI,
    arXiv id.
    """
    entries: list[BibEntry] = []
    # Track starting line number for each @-delimited chunk.
    idx = 0
    line = 1
    # Walk character by character, tracking braces to find entry boundaries.
    n = len(text)
    while idx < n:
        at = text.find("@", idx)
        if at == -1:
            break
        # Advance line counter to the '@'.
        line += text.count("\n", idx, at)
        idx = at
        # Parse the header.
        header_match = ENTRY_HEADER_RE.match(text, idx)
        if not header_match:
            idx += 1
            continue
        # Walk to the matching closing brace of the entry.
        brace = text.find("{", idx)
        if brace == -1:
            break
        depth = 0
        j = brace
        while j < n:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        raw = text[idx:j]
        key = header_match.group("key")
        kind = header_match.group("kind")
        doi_match = DOI_FIELD_RE.search(raw)
        arxiv_match = ARXIV_FIELD_RE.search(raw)
        entries.append(
            BibEntry(
                key=key,
                kind=kind,
                doi=doi_match.group(1).strip().lower().rstrip(".") if doi_match else None,
                arxiv=arxiv_match.group(1) if arxiv_match else None,
                line=line,
                raw=raw,
            )
        )
        # Advance line counter past the entry.
        line += text.count("\n", idx, j)
        idx = j
    return entries


@dataclass
class DuplicateReport:
    duplicate_keys: dict[str, list[int]]         # key -> [line numbers]
    duplicate_dois: dict[str, list[tuple[str, int]]]   # doi -> [(key, line)]
    duplicate_arxiv: dict[str, list[tuple[str, int]]]  # arxiv id -> [(key, line)]

    @property
    def total_issues(self) -> int:
        return (
            len(self.duplicate_keys)
            + len(self.duplicate_dois)
            + len(self.duplicate_arxiv)
        )


def find_duplicates(entries: list[BibEntry]) -> DuplicateReport:
    by_key: dict[str, list[int]] = defaultdict(list)
    by_doi: dict[str, list[tuple[str, int]]] = defaultdict(list)
    by_arxiv: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for e in entries:
        by_key[e.key].append(e.line)
        if e.doi:
            by_doi[e.doi].append((e.key, e.line))
        if e.arxiv:
            by_arxiv[e.arxiv].append((e.key, e.line))
    return DuplicateReport(
        duplicate_keys={k: v for k, v in by_key.items() if len(v) > 1},
        duplicate_dois={k: v for k, v in by_doi.items()
                        if len({key for key, _ in v}) > 1},
        duplicate_arxiv={k: v for k, v in by_arxiv.items()
                         if len({key for key, _ in v}) > 1},
    )


def format_report(report: DuplicateReport, total_entries: int) -> str:
    lines: list[str] = []
    lines.append(f"paper.bib integrity report")
    lines.append(f"  total entries: {total_entries}")
    lines.append(f"  duplicate keys: {len(report.duplicate_keys)}")
    lines.append(f"  duplicate DOIs: {len(report.duplicate_dois)}")
    lines.append(f"  duplicate arXiv ids: {len(report.duplicate_arxiv)}")
    lines.append("")

    if report.duplicate_keys:
        lines.append("== DUPLICATE KEYS (biblatex error) ==")
        for key, ln in sorted(report.duplicate_keys.items()):
            lines.append(f"  {key}  at lines {ln}")
        lines.append("")

    if report.duplicate_dois:
        lines.append("== DUPLICATE DOIs ==")
        for doi, keys_lines in sorted(report.duplicate_dois.items()):
            lines.append(f"  {doi}")
            for key, ln in keys_lines:
                lines.append(f"    -> {key}  (line {ln})")
        lines.append("")

    if report.duplicate_arxiv:
        lines.append("== DUPLICATE arXiv IDs ==")
        for aid, keys_lines in sorted(report.duplicate_arxiv.items()):
            lines.append(f"  arXiv:{aid}")
            for key, ln in keys_lines:
                lines.append(f"    -> {key}  (line {ln})")
        lines.append("")

    if report.total_issues == 0:
        lines.append("OK: no duplicate keys / DOIs / arXiv ids.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", default=str(DEFAULT_BIB),
                        help="Path to paper.bib (default: repo root)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any duplicate is found")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Also emit a machine-readable JSON report")
    args = parser.parse_args(argv)

    bib_path = Path(args.bib)
    if not bib_path.exists():
        print(f"error: bib file not found: {bib_path}", file=sys.stderr)
        return 2

    text = bib_path.read_text(encoding="utf-8")
    entries = parse_bib(text)
    report = find_duplicates(entries)

    print(format_report(report, total_entries=len(entries)))

    if args.json_out:
        payload = {
            "total_entries": len(entries),
            "duplicate_keys": report.duplicate_keys,
            "duplicate_dois": report.duplicate_dois,
            "duplicate_arxiv": report.duplicate_arxiv,
        }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}", file=sys.stderr)

    if args.strict and report.total_issues > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
