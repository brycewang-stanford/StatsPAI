#!/usr/bin/env python3
"""
Bibliography coverage auditor for StatsPAI.

Closes the §10 red line from the opposite direction of the other two
auditors:

  * ``audit_citations.py``        — every cited paper must match primary
                                    source (authors / year).
  * ``audit_bib_duplicates.py``   — paper.bib must have no duplicates.
  * ``audit_bib_coverage.py`` ←   — the ``[@bibkey]`` graph must be
                                    consistent: every code citation
                                    resolves to a bib entry (no
                                    dangling refs) and optionally every
                                    bib entry is cited somewhere (no
                                    orphans).

Dangling refs are a hard build failure for ``paper.md`` (pandoc will
emit ``???`` instead of the citation), so ``--strict-dangling`` is the
default CI gate. Orphan entries are a softer code-smell and kept as an
advisory report by default.

Usage
-----
    python tools/audit_bib_coverage.py                 # report only
    python tools/audit_bib_coverage.py --strict-dangling
    python tools/audit_bib_coverage.py --strict        # dangling + orphan
    python tools/audit_bib_coverage.py --bib paper.bib --roots src docs
    python tools/audit_bib_coverage.py --json report.json

Stdlib only — safe to run in CI without extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent

# Re-use the bib parser from the duplicate auditor so we have a single
# source of truth for what a bib entry is.
sys.path.insert(0, str(TOOLS_DIR))
from audit_bib_duplicates import parse_bib  # noqa: E402

DEFAULT_ROOTS = ("src", "docs", "paper.md")
DEFAULT_BIB = REPO_ROOT / "paper.bib"
DEFAULT_EXTENSIONS = {".py", ".md", ".rst", ".txt"}

# Pandoc citation syntax: [@key] or [@key1; @key2]. The bracket form is
# how both docstrings and paper.md cite papers in this repo. We do NOT
# match bare ``@key`` — it collides with Python decorators and YAML.
CITE_RE = re.compile(r"\[@([A-Za-z0-9][A-Za-z0-9_\-]*)(?:[;,]\s*@[A-Za-z0-9][A-Za-z0-9_\-]*)*\]")
# For capturing all keys inside a multi-key bracket like [@a; @b; @c]:
MULTI_KEY_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9_\-]*)")


@dataclass
class CitationSite:
    """One ``[@bibkey]`` occurrence in a source file."""

    key: str
    file: str
    line: int


@dataclass
class CoverageReport:
    bib_keys: set[str]
    cited_keys: set[str]
    citations: list[CitationSite]
    # key -> list of (file, line) where it's cited
    citations_by_key: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    @property
    def dangling(self) -> set[str]:
        """Cited but not defined in paper.bib. HARD bug — breaks pandoc."""
        return self.cited_keys - self.bib_keys

    @property
    def orphan(self) -> set[str]:
        """Defined in paper.bib but never cited. Advisory: may be a
        deliberate reference placeholder, or stale."""
        return self.bib_keys - self.cited_keys

    @property
    def coverage_pct(self) -> float:
        if not self.bib_keys:
            return 0.0
        return 100.0 * len(self.bib_keys & self.cited_keys) / len(self.bib_keys)


def extract_citations(roots: list[Path]) -> list[CitationSite]:
    """Walk ``roots`` and collect ``[@bibkey]`` references line-by-line."""
    sites: list[CitationSite] = []
    for root in roots:
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = [p for p in sorted(root.rglob("*"))
                     if p.is_file() and p.suffix in DEFAULT_EXTENSIONS]
        else:
            continue
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                for match in CITE_RE.finditer(line):
                    # Each bracket may contain 1..N keys separated by ';' or ','.
                    for m2 in MULTI_KEY_RE.finditer(match.group(0)):
                        sites.append(CitationSite(
                            key=m2.group(1),
                            file=_rel(path),
                            line=i,
                        ))
    return sites


def _rel(path: Path) -> str:
    # Always emit POSIX-style separators so the report is stable across
    # Windows/POSIX (tests + downstream agents key on these strings).
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(bib_path: Path, roots: list[Path]) -> CoverageReport:
    entries = parse_bib(bib_path.read_text(encoding="utf-8"))
    bib_keys = {e.key for e in entries}

    sites = extract_citations(roots)
    cited_keys = {s.key for s in sites}

    citations_by_key: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for s in sites:
        citations_by_key[s.key].append((s.file, s.line))

    return CoverageReport(
        bib_keys=bib_keys,
        cited_keys=cited_keys,
        citations=sites,
        citations_by_key=dict(citations_by_key),
    )


def format_report(report: CoverageReport, *, show_orphans: bool = True,
                  max_rows: int = 50) -> str:
    lines: list[str] = []
    lines.append("paper.bib coverage report")
    lines.append(f"  bib entries:       {len(report.bib_keys)}")
    lines.append(f"  cited unique keys: {len(report.cited_keys)}")
    lines.append(f"  total citations:   {len(report.citations)}")
    lines.append(f"  coverage:          {report.coverage_pct:.1f}% "
                 f"({len(report.bib_keys & report.cited_keys)}/{len(report.bib_keys)})")
    lines.append(f"  dangling refs:     {len(report.dangling)}")
    lines.append(f"  orphan entries:    {len(report.orphan)}")
    lines.append("")

    if report.dangling:
        lines.append("== DANGLING REFERENCES (cited but not in paper.bib) ==")
        lines.append("These will render as `???` in pandoc and break paper.md build.")
        lines.append("")
        for key in sorted(report.dangling):
            lines.append(f"  [@{key}]")
            for file, line in report.citations_by_key.get(key, [])[:3]:
                lines.append(f"    -> {file}:{line}")
        lines.append("")

    if show_orphans and report.orphan:
        lines.append(f"== ORPHAN ENTRIES (in paper.bib but never cited) — first {max_rows} ==")
        lines.append("These are advisory. Real orphans can be dropped; intentional")
        lines.append("placeholders for future work should be annotated.")
        lines.append("")
        for key in sorted(report.orphan)[:max_rows]:
            lines.append(f"  {key}")
        if len(report.orphan) > max_rows:
            lines.append(f"  … {len(report.orphan) - max_rows} more")
        lines.append("")

    if not report.dangling and not report.orphan:
        lines.append("OK: every bib entry is cited and every citation resolves.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", default=str(DEFAULT_BIB),
                        help="Path to paper.bib (default: repo root)")
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS),
                        help="Paths to scan for [@bibkey] citations. "
                             "Files and directories both accepted.")
    parser.add_argument("--strict-dangling", action="store_true",
                        help="Exit non-zero if any citation has no matching "
                             "bib entry. Recommended for CI.")
    parser.add_argument("--strict-orphan", action="store_true",
                        help="Exit non-zero if any bib entry is uncited.")
    parser.add_argument("--strict", action="store_true",
                        help="Shorthand for --strict-dangling --strict-orphan.")
    parser.add_argument("--hide-orphans", action="store_true",
                        help="Omit the orphan list from the human report "
                             "(dangling refs always shown).")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Also write a machine-readable JSON report.")
    parser.add_argument("--max-orphans", type=int, default=50,
                        help="Truncate the orphan list to N rows in the "
                             "human report (default: 50). 0 = show all.")
    args = parser.parse_args(argv)

    bib_path = Path(args.bib)
    if not bib_path.exists():
        print(f"error: bib file not found: {bib_path}", file=sys.stderr)
        return 2

    roots = []
    for r in args.roots:
        p = Path(r)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            print(f"warning: root not found, skipping: {p}", file=sys.stderr)
            continue
        roots.append(p)

    if not roots:
        print("error: no valid scan roots", file=sys.stderr)
        return 2

    report = build_report(bib_path, roots)
    print(format_report(
        report,
        show_orphans=not args.hide_orphans,
        max_rows=args.max_orphans if args.max_orphans > 0 else 10**9,
    ))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "bib_entries": len(report.bib_keys),
            "cited_keys": sorted(report.cited_keys),
            "dangling": sorted(report.dangling),
            "orphan": sorted(report.orphan),
            "coverage_pct": report.coverage_pct,
            "citations_by_key": report.citations_by_key,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json_out}", file=sys.stderr)

    fail_dangling = (args.strict_dangling or args.strict) and report.dangling
    fail_orphan = (args.strict_orphan or args.strict) and report.orphan
    if fail_dangling or fail_orphan:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
