"""Freeze and verify the Tier A parity fixture contract.

This lock is intentionally hash-level: changing a parity script, input CSV,
golden JSON, rendered table, or reference-environment file must be paired with
an explicit refresh of ``tests/r_parity/TIER_A_FIXTURE_LOCK.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ascii_source import normalized_source_bytes  # noqa: E402

MANIFEST_PATH = ROOT / "tests" / "r_parity" / "TIER_A_FIXTURE_LOCK.json"
SCHEMA_VERSION = 1

#: The ``.gitignore`` files of the two harness directories are repository
#: hygiene, not fixtures: no number depends on them, and the JSS archive ships
#: no VCS metadata, so locking them made the lock unverifiable there.
LOCK_PATTERNS: tuple[str, ...] = (
    "scripts/ascii_source.py",
    "scripts/tier_a_fixture_lock.py",
    "tests/r_parity/[0-9][0-9]_*.R",
    "tests/r_parity/[0-9][0-9]_*.py",
    "tests/r_parity/README.md",
    "tests/r_parity/R_ENVIRONMENT.md",
    "tests/r_parity/_common.R",
    "tests/r_parity/_common.py",
    "tests/r_parity/_gen_renv_lock.R",
    "tests/r_parity/compare.py",
    "tests/r_parity/data/*.csv",
    "tests/r_parity/renv.lock",
    "tests/r_parity/results/*_R.json",
    "tests/r_parity/results/*_py.json",
    "tests/r_parity/results/REPRODUCIBILITY_REPORT.md",
    "tests/r_parity/results/parity_table.md",
    "tests/r_parity/results/parity_table.tex",
    "tests/r_parity/results/parity_table_3way.md",
    "tests/r_parity/results/parity_table_3way.tex",
    "tests/r_parity/verify_reproduce.py",
    "tests/stata_parity/[0-9][0-9]_*.do",
    "tests/stata_parity/README.md",
    "tests/stata_parity/STATA_ENVIRONMENT.md",
    "tests/stata_parity/_capture_stata_env.do",
    "tests/stata_parity/_common.do",
    "tests/stata_parity/_gen_stata_env.py",
    "tests/stata_parity/_quick_compare.py",
    "tests/stata_parity/results/*_Stata.json",
    "tests/stata_parity/results/REPRODUCIBILITY_REPORT_STATA.md",
    "tests/stata_parity/verify_reproduce_stata.py",
)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read_normalized(path: Path) -> bytes:
    """Read a locked file with line endings normalized to LF.

    Every file the lock covers is text (.R / .py / .csv / .json / .do /
    .gitignore). Hashing the raw bytes made the lock platform-dependent:
    a Windows checkout under ``core.autocrlf`` materializes CRLF, so each
    fixture hashed differently and the aggregate lock failed (surfaced by
    the 2026-06-23 full-matrix run on windows-latest). Normalizing CRLF/CR
    to LF makes the lock stable across OSes; on a LF checkout it is a
    no-op, so the committed manifest stays valid without a rewrite.
    """
    return normalized_source_bytes(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_normalized(path)).hexdigest()


def discover_files() -> list[Path]:
    """Return the deterministic set of files covered by the lock."""
    files: dict[str, Path] = {}
    for pattern in LOCK_PATTERNS:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Lock pattern matched no files: {pattern}")
        for path in matches:
            if path == MANIFEST_PATH or not path.is_file():
                continue
            files[_rel(path)] = path
    return [files[key] for key in sorted(files)]


def _count_suffix(files: list[dict[str, Any]], suffix: str) -> int:
    return sum(1 for item in files if item["path"].endswith(suffix))


def _count_prefix_suffix(
    files: list[dict[str, Any]],
    prefix: str,
    suffix: str,
) -> int:
    return sum(
        1
        for item in files
        if item["path"].startswith(prefix) and item["path"].endswith(suffix)
    )


def _aggregate_hash(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest() -> dict[str, Any]:
    files = [
        {
            "path": _rel(path),
            # Newline-normalized length (not st_size) so the lock matches
            # across LF/CRLF checkouts — see _read_normalized.
            "bytes": len(_read_normalized(path)),
            "sha256": _sha256(path),
        }
        for path in discover_files()
    ]
    counts = {
        "files": len(files),
        "r_scripts": _count_prefix_suffix(files, "tests/r_parity/", ".R"),
        "python_scripts": _count_prefix_suffix(files, "tests/r_parity/", ".py"),
        "stata_do_files": _count_prefix_suffix(
            files,
            "tests/stata_parity/",
            ".do",
        ),
        "input_csvs": _count_prefix_suffix(files, "tests/r_parity/data/", ".csv"),
        "python_golden_jsons": _count_suffix(files, "_py.json"),
        "r_golden_jsons": _count_suffix(files, "_R.json"),
        "stata_golden_jsons": _count_suffix(files, "_Stata.json"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": "tier_a_parity_fixture_lock",
        "description": (
            "Hash lock for the Track/Tier A parity scripts, fixture inputs, "
            "golden outputs, rendered tables, and R/Stata reference "
            "environment files."
        ),
        "generated_by": "python scripts/tier_a_fixture_lock.py --write",
        "counts": counts,
        "files": files,
    }
    manifest["aggregate_sha256"] = _aggregate_hash(files)
    return manifest


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def format_diffs(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    expected_files = {item["path"]: item for item in expected.get("files", [])}
    actual_files = {item["path"]: item for item in actual.get("files", [])}

    missing = sorted(set(expected_files) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_files))
    changed = sorted(
        path
        for path in set(expected_files) & set(actual_files)
        if expected_files[path] != actual_files[path]
    )

    lines: list[str] = []
    if actual.get("schema_version") != SCHEMA_VERSION:
        lines.append(
            f"schema_version: {actual.get('schema_version')} != {SCHEMA_VERSION}"
        )
    if actual.get("counts") != expected.get("counts"):
        lines.append(
            "counts differ: " f"{actual.get('counts')} != {expected.get('counts')}"
        )
    if actual.get("aggregate_sha256") != expected.get("aggregate_sha256"):
        lines.append(
            "aggregate_sha256 differs: "
            f"{actual.get('aggregate_sha256')} != "
            f"{expected.get('aggregate_sha256')}"
        )
    if missing:
        lines.append("missing from lock: " + ", ".join(missing[:12]))
    if extra:
        lines.append("extra in lock: " + ", ".join(extra[:12]))
    if changed:
        lines.append("changed files: " + ", ".join(changed[:20]))
    if len(changed) > 20:
        lines.append(f"... and {len(changed) - 20} more changed files")

    return "\n".join(lines)


def verify() -> tuple[bool, str]:
    expected = build_manifest()
    actual = load_manifest()
    diff = format_diffs(expected, actual)
    return not diff, diff


def write_manifest() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="refresh tests/r_parity/TIER_A_FIXTURE_LOCK.json",
    )
    args = parser.parse_args(argv)

    if args.write:
        write_manifest()
        print(f"wrote {_rel(MANIFEST_PATH)}")
        return 0

    ok, diff = verify()
    if ok:
        print(f"{_rel(MANIFEST_PATH)} is current")
        return 0
    print(
        "Tier A parity fixture lock is stale; run "
        "`python scripts/tier_a_fixture_lock.py --write` after reviewing "
        "the fixture change.\n" + diff,
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
