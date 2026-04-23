"""Tests for ``tools/suggest_bibkey_backfills.py``.

Guards the three invariants the tool needs to be safe:
  * ``find_candidates`` surfaces orphan bib entries whose arXiv/DOI is
    in prose without a ``[@bibkey]`` annotation.
  * Lines that already have ``[@someotherkey]`` are skipped (idempotent).
  * Unsafe contexts — Python string literals, bibtex-synthesis code,
    ``reference=...`` kwargs — are FILTERED OUT so ``--apply`` cannot
    corrupt the codebase by injecting ``[@bibkey]`` into a generated
    bibtex string or causing Python syntax errors.
  * ``apply_candidates`` actually mutates files when called, but is
    defensive: it checks the current line still matches the recorded
    ``original`` before overwriting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import suggest_bibkey_backfills as sbb  # noqa: E402


# ---------------------------------------------------------------------------
# _append_bibkey — placement logic
# ---------------------------------------------------------------------------


def test_append_bibkey_to_plain_line():
    out = sbb._append_bibkey("    See arXiv:2408.12345 for details.\n", "smith2024")
    assert out == "    See arXiv:2408.12345 for details. [@smith2024]\n"


def test_append_bibkey_to_line_without_trailing_newline():
    out = sbb._append_bibkey("Citation: arXiv:2408.12345", "smith2024")
    assert out == "Citation: arXiv:2408.12345 [@smith2024]"


def test_append_bibkey_before_docstring_close():
    # Python _CITATIONS dicts often build bibtex strings line-by-line
    # with ``"..."`` fragments. We don't annotate these (UNSAFE filter),
    # but if called directly _append_bibkey must insert BEFORE the
    # closing quote to at least avoid trivial breakage.
    src = '    "  arXiv:2408.12345\\n"\n'
    out = sbb._append_bibkey(src, "smith2024")
    # The bibkey goes before the \n" tail, preserving the trailing newline.
    assert "[@smith2024]" in out
    assert out.endswith('\n')


# ---------------------------------------------------------------------------
# find_candidates — core scan
# ---------------------------------------------------------------------------


def _setup_tmp_repo(tmp_path: Path, monkeypatch) -> Path:
    """Create an isolated repo-like tree and point the tool at it."""
    monkeypatch.setattr(sbb, "REPO_ROOT", tmp_path)
    # build_report + audit_bib_* modules also dereference REPO_ROOT for
    # relative-path reporting; patch those too.
    import audit_bib_coverage
    import audit_bib_duplicates  # noqa
    monkeypatch.setattr(audit_bib_coverage, "REPO_ROOT", tmp_path)
    # Initialise a minimal git repo so ``git grep`` works.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    return tmp_path


def _write_bib(path: Path, **entries) -> None:
    """entries: bibkey -> (arxiv_id, doi). Either can be None."""
    chunks = []
    for key, (arxiv, doi) in entries.items():
        body = [f"@article{{{key},", f"  title={{T}},", f"  year={{2024}},"]
        if arxiv:
            body.append(f"  journal={{arXiv:{arxiv}}},")
        if doi:
            body.append(f"  doi={{{doi}}}")
        body.append("}")
        chunks.append("\n".join(body))
    path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")


def _commit(path: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_find_candidates_surfaces_orphan_with_arxiv_in_prose(tmp_path, monkeypatch):
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        'r"""See Smith (2024) arXiv:2408.12345 for proof."""\n',
        encoding="utf-8",
    )
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    assert len(cands) == 1
    c = cands[0]
    assert c.key == "orphan_key"
    assert c.identifier == "2408.12345"
    assert c.id_kind == "arxiv"
    assert "[@orphan_key]" in c.suggested


def test_find_candidates_skips_already_annotated_line(tmp_path, monkeypatch):
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        'r"""Smith (2024) arXiv:2408.12345. [@someotherkey]"""\n',
        encoding="utf-8",
    )
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    # Already annotated — adding a second bibkey would be wrong.
    assert cands == []


def test_find_candidates_skips_unsafe_bibtex_synthesis_line(tmp_path, monkeypatch):
    """Lines like ``"  journal={arXiv preprint arXiv:2408.12345},\\n"``
    are Python string fragments being concatenated into a bibtex entry.
    Appending [@bibkey] would PUSH the tag INTO the generated bibtex
    string, breaking biblatex."""
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        'CITATION = (\n'
        '    "@article{foo,\\n"\n'
        '    "  journal={arXiv preprint arXiv:2408.12345},\\n"\n'
        '    "  year={2024}\\n"\n'
        '    "}"\n'
        ')\n',
        encoding="utf-8",
    )
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    assert cands == [], (
        "must skip string-literal-inside-bibtex-synthesis lines "
        "(unsafe for --apply)"
    )


def test_find_candidates_skips_reference_kwarg(tmp_path, monkeypatch):
    """Lines like ``reference="arXiv:2408.12345 (2024)."`` are Python
    function kwargs. Appending [@bibkey] after the closing string quote
    would either break Python syntax or put the tag outside the value."""
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        'def _register():\n'
        '    add_func(\n'
        '        reference="arXiv:2408.12345 (2024).",\n'
        '    )\n',
        encoding="utf-8",
    )
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    assert cands == []


def test_find_candidates_uses_doi_when_arxiv_absent(tmp_path, monkeypatch):
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               doi_only_orphan=(None, "10.1234/abcd.5678"))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        'r"""Paper at doi 10.1234/abcd.5678 is relevant."""\n',
        encoding="utf-8",
    )
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    assert len(cands) == 1
    assert cands[0].identifier == "10.1234/abcd.5678"
    assert cands[0].id_kind == "doi"


def test_find_candidates_ignores_paper_bib_itself(tmp_path, monkeypatch):
    """The orphan's identifier appears in paper.bib trivially. The tool
    must not offer to annotate its own paper.bib line."""
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("# unrelated\n", encoding="utf-8")
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    # paper.bib mentions the id, but is skipped. No prose mention anywhere
    # else, so zero candidates.
    assert cands == []


# ---------------------------------------------------------------------------
# apply_candidates — actual mutation
# ---------------------------------------------------------------------------


def test_apply_candidates_mutates_file(tmp_path, monkeypatch):
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "mod.py"
    original = 'r"""Smith (2024) arXiv:2408.12345."""\n'
    target.write_text(original, encoding="utf-8")
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    assert cands
    stats = sbb.apply_candidates(cands)
    assert stats == {"src/mod.py": 1}
    new = target.read_text(encoding="utf-8")
    assert "[@orphan_key]" in new


def test_apply_candidates_skips_when_original_line_drifted(tmp_path, monkeypatch):
    """Defensive: if the file was edited after find_candidates ran, the
    recorded ``original`` won't match the current line — we must NOT
    overwrite in that case."""
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "mod.py"
    target.write_text(
        'r"""Smith (2024) arXiv:2408.12345."""\n',
        encoding="utf-8",
    )
    _commit(tmp_path)

    cands = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    assert cands
    # Simulate concurrent edit.
    target.write_text('r"""Completely different content."""\n',
                      encoding="utf-8")
    stats = sbb.apply_candidates(cands)
    # No lines changed — we protected against drift.
    assert stats == {}


def test_apply_candidates_is_idempotent(tmp_path, monkeypatch):
    """Re-running find_candidates + apply_candidates against the same
    tree must not double-annotate: the HAS_BIBKEY filter in
    find_candidates should detect the already-applied tag."""
    _setup_tmp_repo(tmp_path, monkeypatch)
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "mod.py"
    target.write_text(
        'r"""Smith (2024) arXiv:2408.12345."""\n',
        encoding="utf-8",
    )
    _commit(tmp_path)

    # Apply once.
    cands1 = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    sbb.apply_candidates(cands1)

    # Second round: find_candidates should return nothing because the
    # line now has [@orphan_key] on it (HAS_BIBKEY filter).
    cands2 = sbb.find_candidates(tmp_path / "paper.bib", [tmp_path / "src"])
    assert cands2 == []
    # And the file content must have the bibkey exactly once.
    text = target.read_text(encoding="utf-8")
    assert text.count("[@orphan_key]") == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_does_not_mutate(tmp_path):
    _write_bib(tmp_path / "paper.bib",
               orphan_key=("2408.12345", None))
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "mod.py"
    original = 'r"""Smith (2024) arXiv:2408.12345."""\n'
    target.write_text(original, encoding="utf-8")
    # Need git repo for `git grep` inside the tool.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    # Point tool at the tmp repo. Merge into os.environ so PATH stays
    # available for the ``git grep`` subprocess inside the tool — Windows
    # CreateProcess has no fallback path like POSIX execvpe (_CS_PATH),
    # so an empty-env child cannot resolve ``git.exe``.
    env = {**os.environ, "PYTHONPATH": str(TOOLS_DIR)}
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "suggest_bibkey_backfills.py"),
         "--bib", str(tmp_path / "paper.bib"),
         "--roots", str(tmp_path / "src")],
        capture_output=True, text=True, check=False,
        cwd=tmp_path, env=env,
    )
    # Default is dry-run; file must be untouched.
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == original
    assert "orphan_key" in result.stdout


def test_cli_missing_bib_returns_2(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "suggest_bibkey_backfills.py"),
         "--bib", str(tmp_path / "nope.bib")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr
