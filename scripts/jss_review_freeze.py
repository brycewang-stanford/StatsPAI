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

``--check`` prints the same comparison without pytest.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
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
    if missing:
        print(
            f"FAIL -- {len(missing)} frozen artifact change(s) missing from "
            f"{LEDGER.relative_to(ROOT)}"
        )
        return 1
    print(f"OK -- frozen at {manifest['tag']}; every change is on the record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
