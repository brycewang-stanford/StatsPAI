"""Tests for ``tools/audit_bib_duplicates.py``.

These tests are stdlib-only — they never touch paper.bib itself, never
hit the network, and never shell out (except for the CLI smoke-test).
They guard the two integrity invariants the tool enforces:
  * ``parse_bib`` correctly locates entry keys + DOIs + arXiv ids even
    when titles carry nested braces.
  * ``find_duplicates`` counts different-key collisions but not the
    trivial "key defined N times" case as a DOI duplicate.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import audit_bib_duplicates as abd  # noqa: E402


# ---------------------------------------------------------------------------
# parse_bib
# ---------------------------------------------------------------------------


def test_parse_bib_single_entry_extracts_key_doi_arxiv():
    text = textwrap.dedent("""
        @article{smith2020example,
          title={A Study of X},
          author={Smith, Alice},
          year={2020},
          doi={10.1234/abcd.5678}
        }
    """).strip()
    entries = abd.parse_bib(text)
    assert len(entries) == 1
    e = entries[0]
    assert e.key == "smith2020example"
    assert e.kind == "article"
    assert e.doi == "10.1234/abcd.5678"
    assert e.arxiv is None


def test_parse_bib_handles_nested_title_braces():
    """Bibtex titles often protect capitalisation with nested braces.

    The parser must balance braces to find the entry's closing ``}``,
    not the first one after the opening.
    """
    text = textwrap.dedent("""
        @article{jones2021nested,
          title={Deep {RL} for {HTE}: Methods and Evidence},
          author={Jones, Bob},
          year={2021},
          doi={10.5555/xxxx.yyyy}
        }

        @article{doe2022next,
          title={Simple},
          author={Doe, Jane},
          year={2022},
          doi={10.6666/zzzz}
        }
    """).strip()
    entries = abd.parse_bib(text)
    assert [e.key for e in entries] == ["jones2021nested", "doe2022next"]
    assert entries[0].doi == "10.5555/xxxx.yyyy"
    assert entries[1].doi == "10.6666/zzzz"


def test_parse_bib_doi_lowercased_and_trailing_period_stripped():
    text = textwrap.dedent("""
        @article{upper2023case,
          title={Test},
          doi={10.1177/1536867X20909691.}
        }
    """).strip()
    entry = abd.parse_bib(text)[0]
    # Lowercased so `10.X/Y` and `10.x/y` dedupe together.
    assert entry.doi == "10.1177/1536867x20909691"


def test_parse_bib_arxiv_from_journal_field():
    text = textwrap.dedent("""
        @misc{foo2024preprint,
          title={Whatever},
          author={Foo, Bar},
          year={2024},
          journal={arXiv:2408.12345}
        }
    """).strip()
    entry = abd.parse_bib(text)[0]
    assert entry.arxiv == "2408.12345"


def test_parse_bib_arxiv_from_eprint_field():
    text = textwrap.dedent("""
        @misc{bar2024preprint,
          title={Whatever},
          eprint={2501.09876},
          archivePrefix={arXiv}
        }
    """).strip()
    entry = abd.parse_bib(text)[0]
    assert entry.arxiv == "2501.09876"


def test_parse_bib_records_starting_line_number():
    text = "\n\n\n" + textwrap.dedent("""
        @article{line4entry,
          title={X},
          doi={10.1/2}
        }
    """).strip()
    entry = abd.parse_bib(text)[0]
    # The '@' sits on line 4 (after three leading blank lines).
    assert entry.line == 4


def test_parse_bib_skips_non_entry_text():
    text = textwrap.dedent("""
        % This is a comment line describing the file.
        Some free prose about the bibliography.

        @article{real2020entry,
          title={Real},
          doi={10.1/real}
        }
    """).strip()
    entries = abd.parse_bib(text)
    assert len(entries) == 1
    assert entries[0].key == "real2020entry"


# ---------------------------------------------------------------------------
# find_duplicates
# ---------------------------------------------------------------------------


def _entries(*specs):
    """Build BibEntry list from (key, doi, arxiv) tuples for terse tests."""
    return [
        abd.BibEntry(key=k, kind="article", doi=d, arxiv=a, line=i + 1, raw="")
        for i, (k, d, a) in enumerate(specs)
    ]


def test_find_duplicates_flags_duplicate_keys():
    entries = _entries(
        ("same_key", "10.1/a", None),
        ("other",    "10.1/b", None),
        ("same_key", "10.1/c", None),  # same key, different DOI
    )
    report = abd.find_duplicates(entries)
    assert "same_key" in report.duplicate_keys
    assert report.duplicate_keys["same_key"] == [1, 3]
    assert "other" not in report.duplicate_keys


def test_find_duplicates_flags_same_doi_different_keys():
    entries = _entries(
        ("key_a", "10.1/shared", None),
        ("key_b", "10.1/shared", None),
        ("key_c", "10.1/unique", None),
    )
    report = abd.find_duplicates(entries)
    assert "10.1/shared" in report.duplicate_dois
    assert len(report.duplicate_dois["10.1/shared"]) == 2
    assert "10.1/unique" not in report.duplicate_dois


def test_find_duplicates_ignores_same_doi_same_key():
    """Two entries with the SAME key + same DOI is a duplicate-KEY bug,
    not a duplicate-DOI bug. Flagging it as both would double-count."""
    entries = _entries(
        ("same_key", "10.1/same_doi", None),
        ("same_key", "10.1/same_doi", None),
    )
    report = abd.find_duplicates(entries)
    assert "same_key" in report.duplicate_keys
    assert report.duplicate_dois == {}


def test_find_duplicates_flags_same_arxiv_different_keys():
    entries = _entries(
        ("key_a", None, "2408.12345"),
        ("key_b", None, "2408.12345"),
    )
    report = abd.find_duplicates(entries)
    assert "2408.12345" in report.duplicate_arxiv


def test_find_duplicates_clean_bib_reports_nothing():
    entries = _entries(
        ("a2020x", "10.1/a", None),
        ("b2021y", "10.1/b", "2401.00001"),
        ("c2022z", None,     "2402.00002"),
    )
    report = abd.find_duplicates(entries)
    assert report.duplicate_keys == {}
    assert report.duplicate_dois == {}
    assert report.duplicate_arxiv == {}
    assert report.total_issues == 0


# ---------------------------------------------------------------------------
# CLI --strict exit code
# ---------------------------------------------------------------------------


def test_cli_strict_exits_nonzero_on_duplicates(tmp_path):
    bib = tmp_path / "dirty.bib"
    bib.write_text(textwrap.dedent("""
        @article{dup_key,
          title={A},
          doi={10.1/a}
        }

        @article{dup_key,
          title={B},
          doi={10.1/b}
        }
    """).strip(), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "audit_bib_duplicates.py"),
         "--bib", str(bib), "--strict"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "dup_key" in result.stdout


def test_cli_strict_exits_zero_on_clean_bib(tmp_path):
    bib = tmp_path / "clean.bib"
    bib.write_text(textwrap.dedent("""
        @article{unique_one,
          title={A},
          doi={10.1/a}
        }

        @article{unique_two,
          title={B},
          doi={10.1/b}
        }
    """).strip(), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "audit_bib_duplicates.py"),
         "--bib", str(bib), "--strict"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_non_strict_reports_but_exits_zero(tmp_path):
    bib = tmp_path / "dirty.bib"
    bib.write_text(textwrap.dedent("""
        @article{dup_key,
          title={A},
          doi={10.1/a}
        }

        @article{dup_key,
          title={B},
          doi={10.1/b}
        }
    """).strip(), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "audit_bib_duplicates.py"),
         "--bib", str(bib)],
        capture_output=True, text=True, check=False,
    )
    # Without --strict: still report, but exit 0 so humans running it
    # locally don't get confused by non-zero exits.
    assert result.returncode == 0
    assert "dup_key" in result.stdout


def test_cli_missing_bib_file_returns_2(tmp_path):
    missing = tmp_path / "nope.bib"
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "audit_bib_duplicates.py"),
         "--bib", str(missing)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# Real paper.bib regression: current HEAD must be dup-free
# ---------------------------------------------------------------------------


def test_repository_paper_bib_has_no_duplicates():
    """Pin the §10 invariant: paper.bib on main must have 0 dups.

    If this fails, either (a) a duplicate was introduced, or (b) a
    legitimate new duplicate arose (same paper, multiple venues) — in
    which case the tool's semantics need an allowlist mechanism, not a
    silent regression.
    """
    bib_path = REPO_ROOT / "paper.bib"
    if not bib_path.exists():
        pytest.skip("paper.bib not present in this checkout")
    entries = abd.parse_bib(bib_path.read_text(encoding="utf-8"))
    report = abd.find_duplicates(entries)
    assert report.duplicate_keys == {}, (
        f"paper.bib has duplicate keys: {list(report.duplicate_keys)}"
    )
    assert report.duplicate_dois == {}, (
        f"paper.bib has duplicate DOIs: {list(report.duplicate_dois)}"
    )
    assert report.duplicate_arxiv == {}, (
        f"paper.bib has duplicate arXiv ids: {list(report.duplicate_arxiv)}"
    )
