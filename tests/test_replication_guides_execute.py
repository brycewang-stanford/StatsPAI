"""Every shipped replication guide must be runnable code.

The guides in ``statspai.smart.replicate`` are the first thing a user
copies out of ``sp.replicate(...)``.  Before this test they were never
executed, and four of them called ``sp.regtable(..., column_labels=...)``
— a keyword that does not exist, so the snippet raised ``TypeError`` on
paste.  This module executes each track's code block against the entry's
own dataset so a broken snippet fails CI instead of a user's session.

Entries whose data loader is a placeholder (``angrist_pischke_mhe``
returns an empty frame) are skipped for execution but still linted.
"""

from __future__ import annotations

import warnings

import pytest

import statspai as sp
from statspai.smart.replicate import _REPLICATIONS

# Entries with no runnable dataset behind them.
_NO_DATA = {"angrist_pischke_mhe"}

_TRACK_KEYS = ("classic", "modern")


def _code_blocks(entry):
    """Yield (track_name, source) for every code block in an entry."""
    for track in _TRACK_KEYS:
        block = entry.get(track)
        if block and block.get("code"):
            yield track, "\n".join(block["code"])
    if entry.get("classic") is None and entry.get("modern") is None:
        if entry.get("code"):
            yield "legacy", "\n".join(entry["code"])


_CASES = [
    (key, track, src)
    for key, entry in _REPLICATIONS.items()
    for track, src in _code_blocks(entry)
]


@pytest.mark.parametrize(
    "key,track,source", _CASES, ids=[f"{k}-{t}" for k, t, _ in _CASES]
)
def test_guide_code_block_executes(key, track, source):
    if key in _NO_DATA:
        pytest.skip(f"{key} has no runnable dataset")

    data, _guide = sp.replicate(key)
    namespace = {"sp": sp, "df": data, "data": data}

    with warnings.catch_warnings():
        # Guides legitimately trip assumption warnings; we care about
        # exceptions, not warnings.
        warnings.simplefilter("ignore")
        try:
            exec(compile(source, f"<guide:{key}:{track}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - we want the full context
            pytest.fail(
                f"replication guide {key!r} track {track!r} raised "
                f"{type(exc).__name__}: {exc}\n\n--- code ---\n{source}"
            )


def test_no_guide_uses_a_nonexistent_regtable_keyword():
    """Regression guard for the specific bug this module was written for."""
    import inspect

    valid = set(inspect.signature(sp.regtable).parameters)
    for key, entry in _REPLICATIONS.items():
        for track, source in _code_blocks(entry):
            if "regtable" not in source:
                continue
            assert "column_labels" not in source, (
                f"{key}/{track} uses sp.regtable(column_labels=...), which is "
                f"not a parameter of sp.regtable.  Use 'model_labels'."
            )
            assert "model_labels" in valid


def test_every_entry_declares_a_verifiable_reference():
    """Guides must cite something checkable — a bib key or a paper table."""
    for key, entry in _REPLICATIONS.items():
        classic = entry.get("classic")
        if classic is None:
            continue
        has_refs = bool(classic.get("references")) or bool(classic.get("paper_table"))
        assert has_refs, f"{key} classic track cites no reference"
