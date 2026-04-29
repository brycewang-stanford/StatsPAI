"""Tiny homegrown snapshot helper used by regtable / outreg2 / esttab tests.

Why not syrupy? One file, ~50 LOC, no extra dep. The whole point is to
guard renderer drift across the PR-B refactor — anything fancier is
yak-shaving.

Conventions:

- Snapshot files live next to the fixtures as
  ``<fixture_name>.<format>`` (e.g. ``f1_simple_ols.text``).
- Whitespace normalisation: trailing whitespace on each line is
  stripped, and a single trailing newline is enforced. This keeps
  diffs robust to editor newline handling without losing structural
  information.
- To regenerate after an intentional change run::

      STATSPAI_UPDATE_SNAPSHOTS=1 pytest tests/test_regtable_snapshots.py

  Inspect the diff before committing.
"""

from __future__ import annotations

import os
from pathlib import Path

SNAPSHOT_DIR = Path(__file__).parent

UPDATE_FLAG = "STATSPAI_UPDATE_SNAPSHOTS"


def normalize(text: str) -> str:
    """Strip trailing whitespace per line; enforce single trailing newline."""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).rstrip() + "\n"


def assert_snapshot(name: str, fmt: str, actual: str) -> None:
    """Compare *actual* against the saved snapshot ``<name>.<fmt>``.

    Set ``STATSPAI_UPDATE_SNAPSHOTS=1`` to regenerate.
    """
    path = SNAPSHOT_DIR / f"{name}.{fmt}"
    actual_norm = normalize(actual)
    if os.environ.get(UPDATE_FLAG):
        path.write_text(actual_norm, encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(
            f"Snapshot {path.name} does not exist. "
            f"Run `{UPDATE_FLAG}=1 pytest <this-test>` to create it."
        )
    expected = normalize(path.read_text(encoding="utf-8"))
    if expected != actual_norm:
        # Render a friendly unified-diff message
        import difflib
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                actual_norm.splitlines(),
                fromfile=f"snapshot:{path.name}",
                tofile="actual",
                lineterm="",
            )
        )
        raise AssertionError(
            f"Snapshot {path.name} drift detected:\n{diff}\n\n"
            f"If the change is intentional, regenerate with: "
            f"{UPDATE_FLAG}=1 pytest <this-test>"
        )
