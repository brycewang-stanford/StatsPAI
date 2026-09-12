"""Data files must be ASCII at the source, not normalised on the way out.

The JSS submission packager transliterates *source* files (``.py``, ``.R``,
``.do``, ``.sh``, ``.toml``) to ASCII inside the archive, but deliberately
leaves *data* files (``.csv``, ``.json``, ``.lock``) byte-for-byte alone:
their bytes are what the Tier A fixture lock hashes and what the
data-provenance audit traces, so rewriting them in transit would make the
archive disagree with the repository it claims to snapshot. The package
checker therefore only *verifies* that data files are ASCII.

That check lives in ``Paper-JSS/``, which is git-ignored and absent from
every worktree and CI checkout, so it never ran where the offending bytes
were written. 1.26.0 put five em-dashes into ``src/statspai/_parity_index.json``
through the alias evidence note in ``_parity_taxonomy.AliasProof``, and the
first ``make submission-ready`` after the 1.27.0 tag was the first thing to
notice. This test runs the same check everywhere the suite runs.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirrors ASCII_DATA_SUFFIXES in Paper-JSS/replication/scripts/jss_submission_package.py.
ASCII_DATA_SUFFIXES = {".csv", ".json", ".lock"}
SCANNED_TREES = ("src", "schemas", "tests")


def _data_files():
    for top in SCANNED_TREES:
        for path in (REPO_ROOT / top).rglob("*"):
            if path.suffix not in ASCII_DATA_SUFFIXES or not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            yield path


def test_shipped_and_fixture_data_files_are_ascii() -> None:
    offenders = []
    for path in _data_files():
        data = path.read_bytes()
        try:
            data.decode("ascii")
        except UnicodeDecodeError as exc:
            text = data.decode("utf-8", errors="replace")
            ch = text[len(data[: exc.start].decode("utf-8", errors="replace"))]
            offenders.append(
                f"{path.relative_to(REPO_ROOT)} (first: {ch!r} U+{ord(ch):04X})"
            )
    assert not offenders, (
        "data files must be ASCII at the source -- the JSS archive verifies "
        "them rather than transliterating them. Fix the generator that wrote "
        "them (e.g. use '--' for a dash):\n  " + "\n  ".join(offenders)
    )


def test_alias_evidence_note_is_ascii() -> None:
    """The generator that produced the 1.26.0 regression, checked directly."""
    from statspai._parity_taxonomy import TRACK_A_ALIASES

    for proof in TRACK_A_ALIASES.values():
        note = proof.evidence_note()
        assert note.isascii(), f"non-ASCII alias evidence note: {note!r}"
