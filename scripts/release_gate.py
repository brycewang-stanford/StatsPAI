#!/usr/bin/env python3
"""Checks a release has to pass before it is tagged.

Three problems reached PyPI in 1.33.0 / 1.34.0 and were only found when a
paper's replication archive was built from the tag, so each cost a patch
release (1.34.1, 1.34.2). They are properties of the package, so they are
checked here, where a release is made.

``census``
    ``docs/guides/stability.md`` and ``docs/jss_source_audit_dossier.md``
    quote the registry census (functions, certified / validated / api_stable
    / experimental, evidence files). The JSS claim linter holds those
    documents to the registry of the tagged release. ``--fix-census``
    rewrites the figures from the live registry; the check compares them.
    Other lines add functions all day, so the check is enforced only inside
    the release window (see :func:`in_release_window`), never on an ordinary
    push.

``ascii``
    The JSS archive transliterates non-ASCII source to ASCII
    (``scripts/ascii_source.py``). That is harmless in comments, docstrings
    and messages, and changes behaviour in a regular expression: a literal
    U+2026 became ``...``, which matches any three characters, and every
    schema enum was lost. A pattern passed to ``re`` must therefore be
    written with ASCII source (``\\u2026``, not the character).

``wording``
    Text that ships in the wheel must not carry promotional claims the JSS
    archive verifier rejects ("state of the art", "world-class", ...).

``server.json``
    The MCP Registry entry (``server.json``) names the package version
    twice, and the registry rejects a publish whose PyPI package lacks the
    ``mcp-name`` marker in its README. Both are checked with the census
    (inside the release window); ``--fix-census`` rewrites the versions.

Usage::

    # ascii + wording, and the census inside the release window
    python scripts/release_gate.py --check
    # all three, unconditionally
    python scripts/release_gate.py --check --release
    # rewrite the census figures from the live registry
    python scripts/release_gate.py --fix-census
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "statspai"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ascii_source import ascii_source_text  # noqa: E402

# --------------------------------------------------------------------- census

STABILITY = "docs/guides/stability.md"
DOSSIER = "docs/jss_source_audit_dossier.md"

_STATUS = (
    r"(?P<certified>\d+) `certified`, (?P<validated>\d+) `validated`, "
    r"(?P<api_stable>\d+) `api_stable`, and (?P<experimental>\d+) `experimental`"
)

#: (file, pattern). Every named group is a census key; whitespace in the
#: pattern matches any run of whitespace, so a reflowed paragraph still matches.
CENSUS_SENTENCES: Tuple[Tuple[str, str], ...] = (
    (STABILITY, _STATUS),
    (STABILITY, r"is that (?P<unbacked_auto>\d+) stable auto-registered symbols"),
    (STABILITY, r"includes (?P<evidence_files>\d+) such registry evidence files"),
    (DOSSIER, r"reports (?P<functions>[\d,]+) registered public functions"),
    (DOSSIER, _STATUS),
    (
        DOSSIER,
        r"therefore (?P<certified_validated>\d+) symbols, while "
        r"(?P<unbacked_auto>\d+) stable auto-registered symbols",
    ),
    (DOSSIER, r"all (?P<certified_validated>\d+) certified/validated symbols"),
    (
        DOSSIER,
        r"archive includes (?P<evidence_files>\d+) registry-evidence source files",
    ),
    (DOSSIER, r"tracks (?P<evidence_files>\d+) registry-evidence source files"),
    # README tier tables. They said "counts on the current main" and were
    # never checked, so by 1.42.1 they still showed 414 / 128 / 641 / 3 and
    # "roughly a third" verified against a live 507 / 495 / 674 / 4.
    *(
        (readme, rf"\| `{key}` \|[^|\n]+\| (?P<{key}>\d+) \|")
        for readme in ("README.md", "README_CN.md")
        for key in ("certified", "validated", "api_stable", "experimental")
    ),
    (
        "README.md",
        r"In other words, (?P<certified_validated>[\d,]+) of the "
        r"(?P<functions>[\d,]+) registered functions carry numerical",
    ),
    (
        "README_CN.md",
        r"(?P<functions>[\d,]+) 个注册函数中有 (?P<certified_validated>[\d,]+) 个带有数值证据",
    ),
)


def census() -> Dict[str, int]:
    """The registry census, from the same audit the JSS manuscript reads."""
    import stability_audit

    audit = stability_audit.collect()
    import statspai as sp

    status: Dict[str, int] = {}
    for name in sp.list_functions():
        key = sp.describe_function(name).get("validation_status")
        status[key] = status.get(key, 0) + 1
    return {
        "functions": audit["totals"]["registry"],
        "certified": status.get("certified", 0),
        "validated": status.get("validated", 0),
        "api_stable": status.get("api_stable", 0),
        "experimental": status.get("experimental", 0),
        "certified_validated": status.get("certified", 0) + status.get("validated", 0),
        "unbacked_auto": audit["parity_coverage"]["unbacked_auto"],
        "evidence_files": audit["evidence_paths"]["unique"],
    }


def _flex(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern.replace(" ", r"\s+"))


def _render(key: str, value: int, written: str) -> str:
    return f"{value:,}" if "," in written or key == "functions" else str(value)


def census_drift(live: Dict[str, int], *, fix: bool = False) -> List[str]:
    issues: List[str] = []
    texts: Dict[str, str] = {}
    for rel, pattern in CENSUS_SENTENCES:
        text = texts.setdefault(rel, (REPO_ROOT / rel).read_text(encoding="utf-8"))
        match = _flex(pattern).search(text)
        if match is None:
            issues.append(f"{rel}: census sentence not found: /{pattern}/")
            continue
        new = match.group(0)
        for key in sorted(match.groupdict(), key=lambda k: -match.start(k)):
            written = match.group(key)
            want = _render(key, live[key], written)
            if written.replace(",", "") != str(live[key]):
                issues.append(f"{rel}: {key} is written {written}, registry has {want}")
            a, b = match.start(key) - match.start(0), match.end(key) - match.start(0)
            new = new[:a] + want + new[b:]
        if fix and new != match.group(0):
            texts[rel] = text[: match.start(0)] + new + text[match.end(0) :]
    if fix:
        for rel, text in texts.items():
            (REPO_ROOT / rel).write_text(text, encoding="utf-8")
    return issues


def in_release_window() -> bool:
    """True between the version bump and the tag.

    The package version has no ``v<version>`` tag yet: a release is being
    prepared. Once the tag exists the census documents describe that release
    and later registry growth on ``main`` is not drift. A checkout that cannot
    see tags (a shallow CI clone) is treated as outside the window.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        return False
    tags = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "tag", "--list", "v*"],
        capture_output=True,
        text=True,
    )
    if tags.returncode != 0 or not tags.stdout.strip():
        return False
    return f"v{match.group(1)}" not in tags.stdout.split()


# ---------------------------------------------------------------------- ascii

_RE_FUNCS = {
    "compile",
    "search",
    "match",
    "fullmatch",
    "sub",
    "subn",
    "split",
    "findall",
    "finditer",
}


def _is_re_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _RE_FUNCS
        and isinstance(func.value, ast.Name)
        and func.value.id in {"re", "regex"}
    )


#: Raw string literals that are text, not patterns: transliterating them
#: changes a character a reader sees and nothing the code does.
RAW_TEXT_ALLOWLIST: Dict[str, str] = {
    "src/statspai/crossval/_external.py": "embedded R scripts; a dash in an R comment",
    "src/statspai/help.py": "the sp.help() banner (box-drawing characters)",
    "src/statspai/mediation/sensitivity.py": "a LaTeX axis label",
}


def _docstring_lines(tree: ast.AST) -> set:
    lines = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            lines.add(body[0].lineno)
    return lines


def _raw_string_issues(rel: str, source: str, tree: ast.AST) -> List[str]:
    """Raw string literals are where patterns kept in tables live."""
    import io
    import tokenize

    if rel in RAW_TEXT_ALLOWLIST:
        return []
    doc = _docstring_lines(tree)
    out: List[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.STRING or tok.start[0] in doc:
            continue
        prefix = tok.string[: len(tok.string) - len(tok.string.lstrip("rRbBfFuU"))]
        if "r" not in prefix.lower():
            continue
        if tok.string.isascii() or ascii_source_text(tok.string) == tok.string:
            continue
        bad = sorted({f"U+{ord(c):04X}" for c in tok.string if not c.isascii()})
        out.append(
            f"{rel}:{tok.start[0]}: raw string holds {', '.join(bad)}; write "
            "the escape (\\uXXXX) if it is a pattern, or list the file in "
            "RAW_TEXT_ALLOWLIST with the reason if it is text"
        )
    return out


def non_ascii_patterns() -> List[str]:
    """``re`` patterns whose *source* is changed by the ASCII transliteration."""
    issues: List[str] = []
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if source.isascii():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - the test suite would fail first
            issues.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: cannot parse ({exc})"
            )
            continue
        issues.extend(
            _raw_string_issues(
                str(path.relative_to(REPO_ROOT).as_posix()), source, tree
            )
        )
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_re_call(node) and node.args):
                continue
            segment = ast.get_source_segment(source, node.args[0]) or ""
            if segment.isascii() or ascii_source_text(segment) == segment:
                continue
            bad = sorted({f"U+{ord(c):04X}" for c in segment if not c.isascii()})
            issues.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}: re pattern holds "
                f"{', '.join(bad)}; write the escape (\\uXXXX) so the ASCII "
                "archive keeps the same pattern"
            )
    return issues


# -------------------------------------------------------------------- wording

#: Promotional wording the JSS archive verifier rejects in package-facing
#: text (Paper-JSS/replication/scripts/verify_submission_package.py). Matched
#: case-insensitively. Stale counts in that verifier's list are not repeated
#: here; they concern old manuscripts, not the package.
FORBIDDEN_WORDING: Tuple[str, ...] = (
    "state of the art",
    "state-of-the-art",
    "world-class",
    "best-in-class",
    "battle-tested",
    "gold standard",
    "most comprehensive",
    "most feature-complete",
    "no other package",
    "no competing package",
    "no other econometrics package",
    "unique to statspai",
    "drop-in replacement",
    "publication-grade",
    "journal-ready",
    "manuscript-ready",
    "one-click",
    "full-stack",
    "r has no equivalent",
    "neither stata nor r",
)

_WORDING_SUFFIXES = {".py", ".md", ".txt", ".json", ".cff"}


def forbidden_wording() -> List[str]:
    issues: List[str] = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _WORDING_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        folded = text.casefold()
        for snippet in FORBIDDEN_WORDING:
            if snippet not in folded:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if snippet in line.casefold():
                    issues.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}:{line_no}: {snippet!r}"
                    )
                    break
    return issues


# ----------------------------------------------------------------------- main


# ---------------------------------------------------------------- server.json

SERVER_JSON = "server.json"


def _package_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("no version in pyproject.toml")
    return match.group(1)


def server_json_drift(*, fix: bool = False) -> List[str]:
    """``server.json`` versions and the README ``mcp-name`` marker."""
    import json

    path = REPO_ROOT / SERVER_JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    version = _package_version()
    issues: List[str] = []
    slots = [("version", data)] + [
        (f"packages[{i}].version", pkg)
        for i, pkg in enumerate(data.get("packages", []))
    ]
    for label, holder in slots:
        if holder.get("version") != version:
            issues.append(
                f"{SERVER_JSON} {label} is {holder.get('version')!r}, "
                f"package is {version!r}"
            )
            holder["version"] = version
    if issues and fix:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    marker = f"mcp-name: {data.get('name')}"
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    if not re.search(re.escape(marker) + r"(\s|-->|<|$)", readme):
        issues.append(f"README.md lacks the registry ownership marker {marker!r}")
    return issues


# ------------------------------------------------------------ parity tables

PARITY_RESULTS = Path("tests") / "r_parity" / "results"


def parity_table_drift(*, fix: bool = False) -> List[str]:
    """Track A summary tables whose captions name the package version.

    ``tests/r_parity/compare.py`` stamps ``_release_version()`` into
    ``parity_table.tex`` (and the 3-way tables), so a version bump leaves
    them stale until the comparator is rerun. 1.42.1 shipped that way and
    every nightly failed ``test_generated_parity_tables_are_in_sync_with_
    comparator``. Only the StatsPAI-side tables are written here; the
    Paper-JSS snapshot that ``compare.py main()`` also refreshes belongs
    to the paper repo and is left alone.
    """
    import importlib.util

    script = REPO_ROOT / "tests" / "r_parity" / "compare.py"
    spec = importlib.util.spec_from_file_location("_release_gate_compare", script)
    if spec is None or spec.loader is None:
        return [f"cannot load {script.relative_to(REPO_ROOT).as_posix()}"]
    compare = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compare
    spec.loader.exec_module(compare)

    results = REPO_ROOT / PARITY_RESULTS
    modules = sorted(p.stem.replace("_py", "") for p in results.glob("*_py.json"))
    renderers = {
        "parity_table.md": compare.render_md,
        "parity_table.tex": compare.render_tex,
        "parity_table_3way.md": compare.render_md_3way,
        "parity_table_3way.tex": compare.render_tex_3way,
    }
    issues: List[str] = []
    for name, render in renderers.items():
        path = results / name
        if not path.exists():
            continue
        expected = render(modules)
        if path.read_text(encoding="utf-8") != expected:
            issues.append(
                f"{(PARITY_RESULTS / name).as_posix()} differs from compare.py output"
            )
            if fix:
                path.write_text(expected, encoding="utf-8")
    return issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="run the gates")
    parser.add_argument(
        "--release",
        action="store_true",
        help="enforce the census even outside the release window",
    )
    parser.add_argument(
        "--fix-census",
        action="store_true",
        help="rewrite the census figures from the live registry",
    )
    args = parser.parse_args(argv)
    if not (args.check or args.fix_census):
        parser.error("pass --check and/or --fix-census")

    failures: List[str] = []
    if args.fix_census:
        changed = census_drift(census(), fix=True)
        print(f"[release_gate] census: rewrote {len(changed)} figure(s)")
        fixed = server_json_drift(fix=True)
        print(
            f"[release_gate] server.json: {len(fixed)} issue(s) found, versions rewritten"
        )
        tables = parity_table_drift(fix=True)
        print(f"[release_gate] parity tables: rewrote {len(tables)} file(s)")
        if tables:
            # The Tier A fixture lock hashes these tables; a regenerated
            # caption would otherwise fail test_tier_a_parity_fixture_lock.
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "tier_a_fixture_lock.py"),
                    "--write",
                ],
                check=True,
            )
            print("[release_gate] tier A fixture lock: refreshed")
    if args.check:
        ascii_issues = non_ascii_patterns()
        wording_issues = forbidden_wording()
        failures += [f"ascii: {m}" for m in ascii_issues]
        failures += [f"wording: {m}" for m in wording_issues]
        if args.release or in_release_window():
            failures += [f"census: {m}" for m in census_drift(census())]
            failures += [f"census: {m}" for m in server_json_drift()]
            failures += [f"census: {m}" for m in parity_table_drift()]
            print("[release_gate] census: checked (incl. server.json, parity tables)")
        else:
            print(
                "[release_gate] census: skipped (the package version is already "
                "tagged; enforced from the version bump to the tag, or with --release)"
            )
    if failures:
        print("[release_gate] FAIL")
        for message in failures:
            print(f"  - {message}")
        if any(m.startswith("census:") for m in failures):
            print("  run: python scripts/release_gate.py --fix-census")
        return 1
    if args.check:
        print("[release_gate] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
