"""The frozen core API: no breaking change to ``sp.CORE_FUNCTIONS`` signatures.

See ``scripts/core_api_snapshot.py`` for the rules and for how an intended
break is acknowledged (deprecation first, then a MIGRATION.md entry).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import statspai as sp
from statspai.registry import CORE_FUNCTIONS

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "core_api_snapshot", REPO / "scripts" / "core_api_snapshot.py"
)
snap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snap)


def test_no_breaking_change_to_the_frozen_core_api():
    found = snap.breaks(snap.load()["functions"], snap.current())
    assert not found, (
        "Breaking change to the frozen core API (deprecate first, add a "
        "MIGRATION.md entry, then `python scripts/core_api_snapshot.py "
        "--write --ack FN.PARAM=TEXT`):\n"
        + "\n".join(f"  {b['function']}.{b['param']}: {b['change']}" for b in found)
    )


def test_snapshot_covers_exactly_the_core_functions():
    assert set(snap.load()["functions"]) == set(CORE_FUNCTIONS)


def test_every_acknowledged_break_points_at_migration_md():
    text = snap.migration_text()
    for entry in snap.load()["history"]:
        assert entry["migration"] in text, entry


def test_core_functions_are_stable_and_marked_frozen():
    for name in CORE_FUNCTIONS:
        card = sp.describe_function(name)
        assert card["stability"] == "stable", name
        assert card["api_contract"] == "frozen", name
    assert sp.describe_function("bacon_decomposition")["api_contract"] == (
        "deprecation-cycle"
    )


# --------------------------------------------------------------------------
# The rules, each on a synthetic pair of signatures
# --------------------------------------------------------------------------


def _rec(*params, aliases=None):
    return {
        "params": [
            {"name": n, "positional": pos, "default": d} for n, pos, d in params
        ],
        "aliases": aliases or {},
    }


BASE = _rec(
    ("data", True, "<required>"), ("y", True, "<required>"), ("alpha", False, "0.05")
)


@pytest.mark.parametrize(
    "new, expected",
    [
        (BASE, []),
        # an optional keyword added at the end is not a break
        (
            _rec(
                *[(p["name"], p["positional"], p["default"]) for p in BASE["params"]],
                ("seed", False, "None"),
            ),
            [],
        ),
        (
            _rec(("data", True, "<required>"), ("y", True, "<required>")),
            ["alpha: parameter removed or renamed"],
        ),
        # renamed, but the old spelling is still accepted
        (
            _rec(
                ("data", True, "<required>"),
                ("y", True, "<required>"),
                ("level", False, "0.05"),
                aliases={"level": "alpha"},
            ),
            [],
        ),
        (
            _rec(
                ("data", True, "<required>"),
                ("y", True, "<required>"),
                ("alpha", False, "0.1"),
            ),
            ["alpha: default changed from 0.05 to 0.1"],
        ),
        (
            _rec(
                ("data", True, "<required>"),
                ("y", True, "<required>"),
                ("alpha", False, "<required>"),
            ),
            ["alpha: optional parameter became required"],
        ),
        (
            _rec(
                ("y", True, "<required>"),
                ("data", True, "<required>"),
                ("alpha", False, "0.05"),
            ),
            ["*: positional order changed from ['data', 'y'] to ['y', 'data']"],
        ),
        (
            _rec(
                ("data", True, "<required>"),
                ("y", False, "<required>"),
                ("alpha", False, "0.05"),
            ),
            ["*: positional order changed from ['data', 'y'] to ['data']"],
        ),
        (
            _rec(
                *[(p["name"], p["positional"], p["default"]) for p in BASE["params"]],
                ("weights", False, "<required>"),
            ),
            ["weights: new required parameter"],
        ),
    ],
)
def test_break_rules(new, expected):
    got = [f"{b['param']}: {b['change']}" for b in snap.breaks({"f": BASE}, {"f": new})]
    assert got == expected


def test_removing_a_core_function_is_a_break():
    assert snap.breaks({"f": BASE}, {}) == [
        {"function": "f", "param": "*", "change": "core function removed"}
    ]
