"""Snapshot tests for ``sp.regtable`` text/html/latex/markdown rendering.

These tests are the safety net for the PR-B output-module consolidation
(see ``docs/rfc/output_pr_b_consolidation.md``). They lock down the
exact rendered output of five representative fixtures so that any
refactor of the renderer pipeline either reproduces the bytes
verbatim or surfaces a deliberate change for human review.

Excel / Word are intentionally **not** snapshotted — both produce
binary archives where byte equality is brittle (timestamps, ZIP
ordering). Coverage there is via the structural assertions in
``test_paper_tables_export.py`` plus existing integration tests.

Run with::

    pytest tests/test_regtable_snapshots.py

To regenerate after an intentional change::

    STATSPAI_UPDATE_SNAPSHOTS=1 pytest tests/test_regtable_snapshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow ``from output_snapshots._fixtures import ...`` regardless of
# pytest invocation directory.
sys.path.insert(0, str(Path(__file__).parent))

from output_snapshots._fixtures import FIXTURES  # noqa: E402
from output_snapshots._snapshot import assert_snapshot  # noqa: E402


_TEXT_FORMATS = ("text", "html", "latex", "markdown")


@pytest.mark.parametrize("fixture_name", list(FIXTURES))
@pytest.mark.parametrize("fmt", _TEXT_FORMATS)
def test_regtable_snapshot(fixture_name: str, fmt: str) -> None:
    """Each fixture × text-format combination is byte-stable (after
    whitespace normalisation)."""
    result = FIXTURES[fixture_name]()
    rendered = getattr(result, f"to_{fmt}")()
    assert_snapshot(fixture_name, fmt, rendered)
