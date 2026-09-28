#!/usr/bin/env python3
"""Freeze the experiment artifacts the JSS manuscript reads, for the review period.

The submitted manuscript quotes numbers re-tabulated from frozen experiment
artifacts: the Track A and original-data parity results, the Track B coverage
runs and their mechanism studies, the forest seed study, and the Track C
timings. Between submission and decision those numbers must either stay put
or change on the record: a referee comparing the PDF with the package should
be able to find every later change, with its reason and its effect on the
paper, in one place.

``--write --release X.Y.Z`` hashes those artifacts into
``tests/jss_review_freeze.json`` at the submitted release. The test
``tests/test_jss_review_freeze.py`` then fails for any frozen artifact that
changed, disappeared or was added, unless ``docs/dev/jss_review_changes.md``
names its path in an entry. Set ``"active": false`` in the manifest once the
paper is decided; the check then passes vacuously.

Every commit after the tag that touched a frozen artifact must also be
credited: some ledger entry has to name both that commit's SHA and the path.
Naming the path once used to be enough, so a second change to the same file
passed on the first change's entry.

``--check`` prints the same comparison without pytest, and fails when the
checkout cannot see the tag (a shallow clone) rather than falling back.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "jss_review_freeze.json"
LEDGER = ROOT / "docs" / "dev" / "jss_review_changes.md"

#: The frozen inputs of the manuscript's re-tabulated tables and figures
#: (Paper-JSS/replication/results/reproduce_manifest.json, provenance
#: "re-tabulated from frozen artifacts").
GLOBS = (
    "tests/r_parity/results/*.json",
    "tests/stata_parity/results/*.json",
    "tests/orig_parity/results/*.json",
    "tests/coverage_monte_carlo/results_b1000/*.json",
    "tests/coverage_monte_carlo/mechanisms/results_*.json",
    "tests/reference_parity/_fixtures/grf_seed_mc*.json",
    "tests/perf/results/*.json",
)


def _sha256(path: Path) -> str:
    # Hash with LF line endings so a CRLF checkout (Windows CI) matches.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current_hashes() -> dict[str, str]:
    files = sorted({p for pattern in GLOBS for p in ROOT.glob(pattern) if p.is_file()})
    return {p.relative_to(ROOT).as_posix(): _sha256(p) for p in files}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def drift(manifest: dict | None = None) -> dict[str, list[str]]:
    """Paths whose state differs from the frozen manifest, by kind."""
    manifest = manifest if manifest is not None else load_manifest()
    frozen = manifest["files"]
    now = current_hashes()
    return {
        "changed": sorted(p for p in frozen if p in now and now[p] != frozen[p]),
        "removed": sorted(p for p in frozen if p not in now),
        "added": sorted(p for p in now if p not in frozen),
    }


def unrecorded(manifest: dict | None = None) -> list[str]:
    """Drifted paths that the review-period ledger does not name."""
    manifest = manifest if manifest is not None else load_manifest()
    if not manifest.get("active", False):
        return []
    ledger = LEDGER.read_text(encoding="utf-8") if LEDGER.exists() else ""
    paths = [p for kind in drift(manifest).values() for p in kind]
    return [p for p in paths if f"`{p}`" not in ledger]


class HistoryUnavailable(RuntimeError):
    """The freeze tag is not reachable in this checkout (shallow, no tags)."""


_ENTRY_SPLIT = re.compile(r"^### ", flags=re.MULTILINE)
_SHA_TOKEN = re.compile(r"`([0-9a-f]{7,40})`")


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise HistoryUnavailable(out.stderr.strip() or f"git {args[0]} failed")
    return out.stdout


def commits_touching(manifest: dict, paths: list[str]) -> dict[str, list[str]]:
    """Full SHAs of the commits after the freeze tag that changed each path.

    Merge commits count when they change the path relative to a parent, as
    ``git log -- <path>`` reports them: a merge that brings a frozen file in
    from another line is a change to the submitted state like any other.
    """
    tag = manifest["tag"]
    try:
        _git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
    except HistoryUnavailable:
        raise HistoryUnavailable(
            f"freeze tag {tag} is not in this checkout; fetch full history "
            "and tags (actions/checkout: fetch-depth: 0)"
        ) from None
    return {
        p: _git("log", "--format=%H", f"{tag}..HEAD", "--", p).split() for p in paths
    }


def uncredited_commits(manifest: dict | None = None) -> list[str]:
    """Commits that changed a frozen artifact with no ledger entry of their own.

    Naming a path once is not enough: a second change to the same file would
    otherwise ride on the first change's entry, and its effect on the paper
    would never be stated (2026-09-28: two re-traces went unrecorded this
    way). An entry credits a commit for a path when the same ``###`` entry
    gives both the path and the commit's SHA (a prefix of 7 or more hex
    characters), each in backticks. Returns ``"<short sha> <path>"`` items.
    """
    manifest = manifest if manifest is not None else load_manifest()
    if not manifest.get("active", False):
        return []
    ledger = LEDGER.read_text(encoding="utf-8") if LEDGER.exists() else ""
    entries = [(set(_SHA_TOKEN.findall(e)), e) for e in _ENTRY_SPLIT.split(ledger)[1:]]
    paths = sorted(set(manifest["files"]) | set(current_hashes()))
    missing = []
    for path, shas in commits_touching(manifest, paths).items():
        for sha in shas:
            credited = any(
                f"`{path}`" in text and any(sha.startswith(t) for t in tokens)
                for tokens, text in entries
            )
            if not credited:
                missing.append(f"{sha[:8]} {path}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write", action="store_true", help="freeze the current artifacts"
    )
    mode.add_argument(
        "--check", action="store_true", help="report drift against the freeze"
    )
    parser.add_argument(
        "--release", help="release the freeze is anchored to (with --write)"
    )
    args = parser.parse_args()

    if args.write:
        if not args.release:
            parser.error("--write needs --release")
        payload = {
            "description": (
                "SHA-256 (LF-normalised) of the experiment artifacts the JSS "
                "manuscript re-tabulates, frozen at the submitted release. "
                "Written by scripts/jss_review_freeze.py; checked by "
                "tests/test_jss_review_freeze.py against "
                "docs/dev/jss_review_changes.md."
            ),
            "release": args.release,
            "tag": f"v{args.release}",
            "frozen_on": _dt.date.today().isoformat(),
            "active": True,
            "globs": list(GLOBS),
            "files": current_hashes(),
        }
        MANIFEST.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(
            f"OK -- froze {len(payload['files'])} artifacts at {payload['tag']} "
            f"into {MANIFEST.relative_to(ROOT)}"
        )
        return 0

    manifest = load_manifest()
    report = drift(manifest)
    missing = unrecorded(manifest)
    for kind, paths in report.items():
        for path in paths:
            flag = "UNRECORDED" if path in missing else "recorded"
            print(f"{kind:8s} {flag:10s} {path}")
    try:
        commits = uncredited_commits(manifest)
    except HistoryUnavailable as exc:
        # Fail rather than pass on the weaker path-only check: a checkout
        # that cannot see the history cannot vouch for it.
        print(f"FAIL -- cannot check commits against the ledger: {exc}")
        return 1
    for item in commits:
        print(f"commit   UNCREDITED {item}")
    if missing or commits:
        print(
            f"FAIL -- {len(missing)} frozen artifact change(s) and "
            f"{len(commits)} commit(s) missing from {LEDGER.relative_to(ROOT)}; "
            "give each commit that touched a frozen artifact an entry naming "
            "its SHA and the path, both in backticks"
        )
        return 1
    print(f"OK -- frozen at {manifest['tag']}; every change is on the record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
