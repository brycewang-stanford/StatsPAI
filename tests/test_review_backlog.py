"""The review status page is generated, and what it calls done exists.

``docs/dev/review_backlog.json`` is the machine-readable record of the
2026-10-02 repository review (review item G1). A status page written by
hand had already drifted once: the previous roadmap listed work as
pending after it had landed. Here the page is rendered from the backlog,
and a ``done`` item must point at files and tests that are really there.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs" / "dev" / "review_backlog.json"
SCRIPT = ROOT / "scripts" / "build_review_status.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="source checkout only (scripts/ not installed)"
)


@pytest.fixture(scope="module")
def backlog():
    return json.loads(BACKLOG.read_text(encoding="utf-8"))


def test_status_page_is_generated_from_the_backlog(backlog):
    spec = importlib.util.spec_from_file_location("build_review_status", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    page = (ROOT / "docs/dev/2026-10-02-review-status.md").read_text(encoding="utf-8")
    assert page == module.render(
        backlog
    ), "the status page is stale: run python scripts/build_review_status.py"


def test_every_item_has_a_known_status(backlog):
    statuses = {item["status"] for item in backlog["items"]}
    assert statuses <= {"done", "partial", "not_done"}
    assert len(backlog["items"]) >= 30


def test_done_items_point_at_things_that_exist(backlog):
    missing = []
    for item in backlog["items"]:
        if item["status"] == "not_done":
            continue
        for path in item["where"] + item["tests"]:
            if not (ROOT / path).exists():
                missing.append(f"{item['id']}: {path}")
    assert not missing, missing


def test_done_items_are_held_by_a_test_or_say_why_not(backlog):
    """A claim of done without a test is a claim nobody checks."""
    unguarded = [
        f"{item['id']}: {item['title'][:40]}"
        for item in backlog["items"]
        if item["status"] == "done" and not item["tests"]
    ]
    # CI configuration is exercised by CI itself, not by a pytest file.
    assert unguarded == ["G2: fast gate 覆盖整条 agent 链"], unguarded


def test_unfinished_items_give_a_reason(backlog):
    for item in backlog["items"]:
        if item["status"] in ("partial", "not_done"):
            assert len(item["note"]) > 15, item["id"]
