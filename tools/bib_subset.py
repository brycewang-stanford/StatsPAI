#!/usr/bin/env python3
"""
Derive per-paper bibliographies from the master ``paper.bib``.

CLAUDE.md §10 makes the root ``paper.bib`` the single source of truth
for every reference StatsPAI cites: docstrings, ``sp.bibtex()``, the
MCP ``bibtex`` tool, the JOSS paper and the JSS manuscript all point
into it. A paper's own ``.bib`` is therefore a *derived* artefact — a
subset of the master extracted from the keys the paper actually cites —
and must never be edited by hand, otherwise the same reference drifts
into two spellings (``athey2019grf`` vs ``athey2019generalized``) and the
zero-hallucination guarantee silently breaks.

Three sub-commands::

    # Write the subset of paper.bib cited by a manuscript (LaTeX or
    # pandoc-markdown roots; \\input / \\include are followed).
    python tools/bib_subset.py extract \\
        --roots Paper-JSS/manuscript/main.tex \\
        --out   Paper-JSS/manuscript/jss-bib.bib \\
        --keep  brown2020language rios2022csdid

    # Same arguments; exit 1 if --out is stale or cites unknown keys.
    python tools/bib_subset.py check --roots ... --out ...

    # A second derived file for keys cited only by non-compiled sources
    # (archival long-form sections): everything cited anywhere, minus
    # what the submission bib already carries.
    python tools/bib_subset.py extract \
        --roots Paper-JSS/manuscript/main.tex Paper-JSS/manuscript/sections/*.tex \
        --minus Paper-JSS/manuscript/jss-bib.bib \
        --out   Paper-JSS/manuscript/jss-bib-archival.bib

    # The wheel ships a copy of the master at src/statspai/paper.bib so
    # that pip users get the same verified entries. Keep it in sync.
    python tools/bib_subset.py packaged --check      # CI / pre-commit
    python tools/bib_subset.py packaged --sync       # copy root -> src

Determinism: the extracted file carries no timestamps or hashes, only
a fixed banner plus the entries sorted by key, so ``check`` can compare
bytes and a re-run on unchanged inputs is a no-op.

Stdlib only — safe in CI and pre-commit without extra dependencies.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_BIB = REPO_ROOT / "paper.bib"
PACKAGED_BIB = REPO_ROOT / "src" / "statspai" / "paper.bib"

# ---------------------------------------------------------------------------
# BibTeX parsing (brace-matched; mirrors the parser in
# statspai/agent/workflow_tools.py so both agree on what a "key" is).
# ---------------------------------------------------------------------------
_ENTRY_START = re.compile(r"@(\w+)\s*\{", re.IGNORECASE)
_NON_ENTRY_KINDS = {"comment", "string", "preamble"}


def parse_bib(text: str) -> Dict[str, str]:
    """Return ``{key: entry_text}`` in file order; raise on duplicate keys."""
    entries: Dict[str, str] = {}
    for m in _ENTRY_START.finditer(text):
        if m.group(1).lower() in _NON_ENTRY_KINDS:
            continue
        # A leading "%" on the same line means the "@" sits inside a
        # comment line; BibTeX has no comment syntax but the master file
        # never puts a real entry after "%" (see the note near line 1944).
        line_start = text.rfind("\n", 0, m.start()) + 1
        if text[line_start : m.start()].lstrip().startswith("%"):
            continue
        brace_open = text.index("{", m.end() - 1)
        depth = 0
        i = brace_open
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            raise ValueError(
                f"unbalanced braces in entry starting at offset {m.start()}"
            )
        entry = text[m.start() : i + 1]
        key = entry[brace_open - m.start() + 1 :].split(",", 1)[0].strip()
        if not key:
            raise ValueError(f"entry without key at offset {m.start()}")
        if key in entries:
            raise ValueError(f"duplicate key in bib: {key}")
        entries[key] = entry
    return entries


# ---------------------------------------------------------------------------
# Citation-key extraction from manuscript sources.
# ---------------------------------------------------------------------------
# LaTeX: \cite, \citep, \citet, \citealp, \citeauthor, \citeyear, \nocite,
# starred variants, up to two optional arguments, comma-separated keys.
_TEX_CITE_RE = re.compile(
    r"\\(?:cite[a-zA-Z]*|nocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}"
)
_TEX_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_TEX_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
# pandoc-markdown: [@key], [@a; @b], [-@key], and bare @key in prose. The
# bare form deliberately requires a preceding "[", ";", "(" or whitespace
# so that e-mail addresses (``name@host``) are not read as citations.
_MD_CITE_RE = re.compile(
    r"(?:(?<=[\[\s;(])|^)-?@([A-Za-z0-9][A-Za-z0-9_\-:.]*[A-Za-z0-9])"
)

_TEX_SUFFIXES = {".tex", ".ltx"}
_MD_SUFFIXES = {".md", ".qmd", ".rmd", ".markdown"}


def _tex_sources(root: Path, seen: Set[Path]) -> List[Path]:
    """``root`` plus every file it ``\\input``s / ``\\include``s, recursively."""
    root = root.resolve()
    if root in seen or not root.exists():
        return []
    seen.add(root)
    out = [root]
    text = _TEX_COMMENT_RE.sub("", root.read_text(encoding="utf-8"))
    for m in _TEX_INPUT_RE.finditer(text):
        rel = m.group(1).strip()
        if not rel.endswith(".tex"):
            rel += ".tex"
        child = (root.parent / rel).resolve()
        out.extend(_tex_sources(child, seen))
    return out


def cited_keys(roots: Sequence[Path]) -> Dict[str, List[str]]:
    """Return ``{key: [files citing it]}`` across all manuscript roots."""
    keys: Dict[str, List[str]] = {}
    seen: Set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.suffix.lower() in _TEX_SUFFIXES:
            files = _tex_sources(root, seen)
            regex = _TEX_CITE_RE
            strip_comments = True
        elif root.suffix.lower() in _MD_SUFFIXES:
            files = [root]
            regex = _MD_CITE_RE
            strip_comments = False
        else:
            raise SystemExit(
                f"unsupported manuscript root (want .tex/.md/.qmd): {root}"
            )
        for path in files:
            text = path.read_text(encoding="utf-8")
            if strip_comments:
                text = _TEX_COMMENT_RE.sub("", text)
            for m in regex.finditer(text):
                for raw in m.group(1).split(","):
                    k = raw.strip()
                    if k and k != "*":
                        keys.setdefault(k, []).append(str(path))
    return keys


# ---------------------------------------------------------------------------
# Subset rendering
# ---------------------------------------------------------------------------
def render_subset(
    master: Dict[str, str],
    keys: Iterable[str],
    *,
    master_label: str,
    roots_label: str,
) -> str:
    wanted = sorted(set(keys))
    banner = (
        "%% AUTO-GENERATED by tools/bib_subset.py -- DO NOT EDIT BY HAND.\n"
        f"%% Subset of the master bibliography ({master_label}) restricted to the\n"
        f"%% keys cited by: {roots_label}\n"
        "%% Add or fix references in the master file, then re-run\n"
        "%%   python tools/bib_subset.py extract ...\n"
        "%% (project citation policy: one verified source of truth, derived\n"
        f"%% per-paper bibliographies). {len(wanted)} entries, sorted by key.\n"
    )
    # The banner ships inside the JSS submission archive, whose verifier
    # rejects internal markers such as the CLAUDE.md file name; keep it
    # reader-facing.
    body = "\n\n".join(master[k] for k in wanted)
    return banner + "\n" + body + "\n"


def build_subset(
    master_path: Path,
    roots: Sequence[Path],
    keep: Sequence[str],
    minus: Sequence[Path] = (),
) -> Tuple[str, Dict[str, List[str]]]:
    """Return ``(rendered_text, missing)``.

    ``missing`` maps each cited-but-absent key to the files citing it.
    """
    master = parse_bib(master_path.read_text(encoding="utf-8"))
    cited = cited_keys(roots)
    for k in keep:
        cited.setdefault(k, ["--keep"])
    missing = {k: v for k, v in cited.items() if k not in master}
    excluded: Set[str] = set()
    for other in minus:
        excluded |= set(parse_bib(Path(other).read_text(encoding="utf-8")))
    present = [k for k in cited if k in master and k not in excluded]
    roots_label = ", ".join(_rel(Path(r)) for r in roots)
    if minus:
        roots_label += " minus " + ", ".join(_rel(Path(m)) for m in minus)
    text = render_subset(
        master,
        present,
        master_label=_rel(master_path),
        roots_label=roots_label,
    )
    return text, missing


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_extract(args: argparse.Namespace, *, check: bool) -> int:
    master_path = Path(args.master)
    out = Path(args.out)
    roots = [Path(r) for r in args.roots]
    absent_roots = [r for r in roots if not r.exists()]
    if absent_roots:
        if args.skip_missing:
            print(f"SKIP -- manuscript root not present: {absent_roots[0]}")
            return 0
        raise SystemExit(f"manuscript root not found: {absent_roots[0]}")

    minus = [Path(m) for m in args.minus]
    absent_minus = [m for m in minus if not m.exists()]
    if absent_minus:
        raise SystemExit(
            f"--minus file not found (extract it first): {absent_minus[0]}"
        )
    text, missing = build_subset(master_path, roots, args.keep, minus=minus)
    if missing:
        print(
            "FATAL: cited keys absent from the master bib "
            "(add them there, verified per CLAUDE.md section 10):"
        )
        for k, files in sorted(missing.items()):
            print(f"  {k}  <- {', '.join(sorted(set(files)))}")
        return 1

    n = text.count("\n@")
    if check:
        if not out.exists():
            print(f"DRIFT: {_rel(out)} does not exist -- run `extract`.")
            return 1
        current = out.read_text(encoding="utf-8")
        if current != text:
            have = set(parse_bib(current))
            want = set(parse_bib(text))
            print(
                f"DRIFT: {_rel(out)} is stale -- "
                "run `python tools/bib_subset.py extract ...`."
            )
            if want - have:
                print(f"  cited but missing from subset: {sorted(want - have)}")
            if have - want:
                print(f"  in subset but no longer cited: {sorted(have - want)}")
            if have == want:
                print(
                    "  same keys, but entry text differs from the master "
                    "(master was edited)."
                )
            return 1
        print(f"bib-subset OK: {_rel(out)} matches the master ({n} entries).")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {_rel(out)}: {n} entries from {_rel(master_path)}")
    return 0


def cmd_packaged(args: argparse.Namespace) -> int:
    root = Path(args.master)
    packaged = Path(args.packaged)
    if args.sync:
        packaged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root, packaged)
        print(f"synced {_rel(root)} -> {_rel(packaged)}")
        return 0
    if not packaged.exists():
        print(
            f"DRIFT: {_rel(packaged)} is missing -- "
            "run `python tools/bib_subset.py packaged --sync`."
        )
        return 1
    if packaged.read_bytes() != root.read_bytes():
        print(
            f"DRIFT: {_rel(packaged)} differs from {_rel(root)} -- "
            "the wheel would ship "
            "stale citations. Run `python tools/bib_subset.py packaged --sync`."
        )
        return 1
    print(f"packaged bib OK: {_rel(packaged)} == {_rel(root)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_subset_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--master",
            default=str(MASTER_BIB),
            help="master bib (default: repo paper.bib)",
        )
        sp.add_argument(
            "--roots",
            nargs="+",
            required=True,
            help="manuscript root(s): .tex (inputs followed) or .md/.qmd",
        )
        sp.add_argument("--out", required=True, help="derived .bib to write / check")
        sp.add_argument(
            "--keep", nargs="*", default=[], help="extra keys to keep even if not cited"
        )
        sp.add_argument(
            "--minus",
            nargs="*",
            default=[],
            help=(
                "derived .bib file(s) whose keys are excluded from this subset "
                "(e.g. an archival bib = everything cited anywhere minus the "
                "submission bib)"
            ),
        )
        sp.add_argument(
            "--skip-missing",
            action="store_true",
            help=(
                "exit 0 with a SKIP line when a manuscript root is absent "
                "(git-ignored manuscripts in CI)"
            ),
        )

    add_subset_args(sub.add_parser("extract", help="write the derived subset"))
    add_subset_args(sub.add_parser("check", help="fail if the derived subset is stale"))
    pk = sub.add_parser(
        "packaged", help="keep src/statspai/paper.bib identical to the root paper.bib"
    )
    pk.add_argument("--master", default=str(MASTER_BIB))
    pk.add_argument("--packaged", default=str(PACKAGED_BIB))
    g = pk.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--sync", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "extract":
        return cmd_extract(args, check=False)
    if args.cmd == "check":
        return cmd_extract(args, check=True)
    return cmd_packaged(args)


if __name__ == "__main__":
    sys.exit(main())
