"""Tests for ``tools/audit_bib_coverage.py``.

Guards the bibkey-graph invariants:
  * ``[@bibkey]`` pandoc citations are collected from ``.py`` / ``.md`` /
    ``.rst`` / ``.txt`` files, including the multi-key bracket form
    ``[@a; @b, @c]``.
  * ``dangling`` = cited keys missing from paper.bib.
  * ``orphan``   = paper.bib keys never cited.
  * CLI exit code semantics: dangling always hard-fails under
    ``--strict-dangling``; orphan is advisory by default.
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

import audit_bib_coverage as abc  # noqa: E402


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


def test_extract_citations_simple_bracket(tmp_path, monkeypatch):
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    (tmp_path / "mod.py").write_text(
        '"""See [@smith2020example] for proof."""\n',
        encoding="utf-8",
    )
    sites = abc.extract_citations([tmp_path])
    assert [s.key for s in sites] == ["smith2020example"]
    assert sites[0].line == 1


def test_extract_citations_multi_key_bracket(tmp_path, monkeypatch):
    """Pandoc allows ``[@a; @b]`` and ``[@a, @b]`` for stacked refs.
    Both forms must yield one CitationSite per key."""
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    (tmp_path / "mod.md").write_text(
        "Method [@abadie2010synthetic; @abadie2015comparative, @doudchenko2016balancing].\n",
        encoding="utf-8",
    )
    sites = abc.extract_citations([tmp_path])
    keys = sorted(s.key for s in sites)
    assert keys == [
        "abadie2010synthetic",
        "abadie2015comparative",
        "doudchenko2016balancing",
    ]


def test_extract_citations_ignores_email(tmp_path, monkeypatch):
    """``user@domain.com`` must not be mistaken for a citation."""
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    (tmp_path / "mod.py").write_text(
        "# Contact user@domain.com for questions.\n",
        encoding="utf-8",
    )
    sites = abc.extract_citations([tmp_path])
    assert sites == []


def test_extract_citations_ignores_python_decorator(tmp_path, monkeypatch):
    """``@staticmethod`` and similar bare decorators must not match.
    Bracket form is the only one we accept; bare ``@key`` is skipped."""
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    (tmp_path / "mod.py").write_text(
        "class X:\n    @staticmethod\n    def foo(): pass\n",
        encoding="utf-8",
    )
    sites = abc.extract_citations([tmp_path])
    assert sites == []


def test_extract_citations_honours_extension_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    (tmp_path / "data.json").write_text('{"ref": "[@smith2020]"}', encoding="utf-8")
    sites = abc.extract_citations([tmp_path])
    assert sites == []  # .json is not in DEFAULT_EXTENSIONS


def test_extract_citations_accepts_single_file_root(tmp_path, monkeypatch):
    """A file path (not just a directory) is a valid scan root —
    paper.md is passed as a single file."""
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    f = tmp_path / "paper.md"
    f.write_text("See [@smith2020] for background.\n", encoding="utf-8")
    sites = abc.extract_citations([f])
    assert [s.key for s in sites] == ["smith2020"]


# ---------------------------------------------------------------------------
# build_report: dangling / orphan semantics
# ---------------------------------------------------------------------------


def _write_bib(path: Path, *keys: str) -> None:
    body = "\n\n".join(
        f"@article{{{k},\n  title={{X}},\n  doi={{10.1/{k}}}\n}}"
        for k in keys
    )
    path.write_text(body, encoding="utf-8")


def test_build_report_dangling_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    _write_bib(tmp_path / "paper.bib", "defined_key")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# Cite [@missing_key] and [@defined_key].\n",
        encoding="utf-8",
    )
    report = abc.build_report(tmp_path / "paper.bib", [tmp_path / "src"])
    assert report.dangling == {"missing_key"}
    assert report.orphan == set()
    assert report.cited_keys == {"missing_key", "defined_key"}


def test_build_report_orphan_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    _write_bib(tmp_path / "paper.bib", "cited_key", "orphan_key")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@cited_key] only\n",
        encoding="utf-8",
    )
    report = abc.build_report(tmp_path / "paper.bib", [tmp_path / "src"])
    assert report.orphan == {"orphan_key"}
    assert report.dangling == set()


def test_build_report_clean_reports_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    _write_bib(tmp_path / "paper.bib", "key_a", "key_b")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@key_a] and [@key_b]\n", encoding="utf-8",
    )
    report = abc.build_report(tmp_path / "paper.bib", [tmp_path / "src"])
    assert report.dangling == set()
    assert report.orphan == set()
    assert report.coverage_pct == 100.0


def test_build_report_coverage_percent_is_intersection_over_bib(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    _write_bib(tmp_path / "paper.bib", "a", "b", "c", "d")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@a] [@b] [@ghost]\n",  # 2/4 bib keys cited, plus 1 dangling
        encoding="utf-8",
    )
    report = abc.build_report(tmp_path / "paper.bib", [tmp_path / "src"])
    assert report.coverage_pct == 50.0
    assert report.dangling == {"ghost"}
    assert report.orphan == {"c", "d"}


def test_build_report_records_citation_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(abc, "REPO_ROOT", tmp_path)
    _write_bib(tmp_path / "paper.bib", "repeated")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        "# [@repeated]\n# [@repeated]\n", encoding="utf-8",
    )
    (tmp_path / "src" / "b.py").write_text(
        "# [@repeated]\n", encoding="utf-8",
    )
    report = abc.build_report(tmp_path / "paper.bib", [tmp_path / "src"])
    sites = report.citations_by_key["repeated"]
    assert len(sites) == 3
    # Both files represented
    files = {f for f, _ in sites}
    assert files == {"src/a.py", "src/b.py"}


# ---------------------------------------------------------------------------
# CLI exit code semantics
# ---------------------------------------------------------------------------


def _run_cli(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOLS_DIR / "audit_bib_coverage.py"),
         "--bib", str(tmp_path / "paper.bib"),
         "--roots", str(tmp_path / "src"),
         *extra_args],
        capture_output=True, text=True, check=False,
    )


def test_cli_strict_dangling_fails_on_dangling(tmp_path):
    _write_bib(tmp_path / "paper.bib", "defined")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@missing]\n", encoding="utf-8",
    )
    result = _run_cli(tmp_path, "--strict-dangling")
    assert result.returncode == 1
    assert "missing" in result.stdout


def test_cli_strict_dangling_passes_when_only_orphans_exist(tmp_path):
    """--strict-dangling must NOT fail just because of orphans."""
    _write_bib(tmp_path / "paper.bib", "cited", "uncited_orphan")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@cited]\n", encoding="utf-8",
    )
    result = _run_cli(tmp_path, "--strict-dangling")
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_strict_all_fails_on_orphan(tmp_path):
    """--strict (alias for both gates) fails on orphans too."""
    _write_bib(tmp_path / "paper.bib", "cited", "uncited")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@cited]\n", encoding="utf-8",
    )
    result = _run_cli(tmp_path, "--strict")
    assert result.returncode == 1
    assert "uncited" in result.stdout


def test_cli_non_strict_always_exits_zero(tmp_path):
    """Default (no flags) reports but never fails, even with dangling."""
    _write_bib(tmp_path / "paper.bib", "defined")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@missing]\n", encoding="utf-8",
    )
    result = _run_cli(tmp_path)
    assert result.returncode == 0
    assert "missing" in result.stdout  # still reported


def test_cli_hide_orphans_flag(tmp_path):
    _write_bib(tmp_path / "paper.bib", "cited", "uncited_should_not_show")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "# [@cited]\n", encoding="utf-8",
    )
    result = _run_cli(tmp_path, "--hide-orphans")
    assert result.returncode == 0
    assert "uncited_should_not_show" not in result.stdout
    # Summary header still reports the count.
    assert "orphan entries:" in result.stdout


def test_cli_missing_bib_returns_2(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "audit_bib_coverage.py"),
         "--bib", str(tmp_path / "nope.bib"),
         "--roots", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# Repository invariant: current HEAD must have 0 dangling refs
# ---------------------------------------------------------------------------


def test_repository_has_no_dangling_references():
    """§10 red line: paper.md + docstring citations must always
    resolve to a paper.bib entry. Any dangling ref is a pandoc-break
    regression.
    """
    bib = REPO_ROOT / "paper.bib"
    roots = [REPO_ROOT / "src", REPO_ROOT / "docs", REPO_ROOT / "paper.md"]
    roots = [r for r in roots if r.exists()]
    if not bib.exists() or not roots:
        pytest.skip("paper.bib or scan roots missing in this checkout")
    report = abc.build_report(bib, roots)
    assert report.dangling == set(), (
        f"dangling [@bibkey] references (cited but not defined in "
        f"paper.bib): {sorted(report.dangling)}"
    )
