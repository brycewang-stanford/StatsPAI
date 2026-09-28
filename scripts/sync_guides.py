#!/usr/bin/env python3
"""Keep the packaged estimator-choice guides identical to docs/guides.

``docs/guides/choosing_*_estimator.md`` are the source of truth (MkDocs
renders them). ``src/statspai/agent/_guides/`` is the copy that ships in the
wheel so ``statspai://guide/{family}`` and ``sp.decision_guide`` work on
an installed package. The two must be byte-identical.

    python scripts/sync_guides.py          # copy docs -> package
    python scripts/sync_guides.py --check  # exit 1 on drift (pre-push / CI)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "guides"
DST = ROOT / "src" / "statspai" / "agent" / "_guides"
PATTERN = "choosing_*_estimator.md"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    sources = sorted(SRC.glob(PATTERN))
    if not sources:
        print(f"[sync_guides] no {PATTERN} under {SRC}", file=sys.stderr)
        return 1
    drift: list[str] = []
    DST.mkdir(exist_ok=True)
    for src in sources:
        dst = DST / src.name
        same = dst.exists() and dst.read_bytes() == src.read_bytes()
        if same:
            continue
        if check:
            drift.append(src.name)
        else:
            shutil.copyfile(src, dst)
            print(f"[sync_guides] copied {src.name}")
    stale = sorted(p.name for p in DST.glob(PATTERN))
    extra = [n for n in stale if not (SRC / n).exists()]
    if extra:
        if check:
            drift.extend(f"{n} (no source)" for n in extra)
        else:
            for n in extra:
                (DST / n).unlink()
                print(f"[sync_guides] removed {n}")
    if check and drift:
        print(
            "[sync_guides] DRIFT: packaged guides differ from docs/guides: "
            + ", ".join(drift)
            + " — run `python scripts/sync_guides.py`",
            file=sys.stderr,
        )
        return 1
    print(f"[sync_guides] OK — {len(sources)} guides in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
