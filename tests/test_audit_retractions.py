"""Offline tests for ``tools/audit_retractions.py`` — the §10 retraction
sweep.

These never touch the network or OpenAlex. They exercise the pure-Python
layer only: bib DOI extraction, the ``is_retracted`` classifier against
hand-built payloads, the summary/report formatting, and the CLI exit-code
contract (via a monkeypatched ``retraction_status`` so no HTTP happens).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_retractions as ar  # noqa: E402


# ---------------------------------------------------------------------------
# bib DOI extraction (delegates to audit_bib_duplicates.parse_bib)
# ---------------------------------------------------------------------------


def test_extract_doi_entries_keeps_only_entries_with_a_doi():
    bib = """
@article{smith2020,
  title = {A paper},
  doi = {10.1/abc},
}
@book{jones2019,
  title = {No DOI here},
}
@article{lee2021,
  doi = {10.2/XYZ.},
}
"""
    entries = ar.extract_doi_entries(bib)
    assert {k for k, _ in entries} == {"smith2020", "lee2021"}


def test_extract_doi_entries_normalises_doi_lowercase_no_trailing_dot():
    # parse_bib lower-cases and strips a trailing dot, so the OpenAlex
    # lookup key is stable regardless of how the bib author capitalised it.
    bib = "@article{lee2021,\n  doi = {10.2/XYZ.},\n}\n"
    assert dict(ar.extract_doi_entries(bib))["lee2021"] == "10.2/xyz"


# ---------------------------------------------------------------------------
# classifier — pure, payload-driven
# ---------------------------------------------------------------------------


def test_classify_openalex_true_and_false():
    assert ar.classify_openalex({"is_retracted": True}) is True
    assert ar.classify_openalex({"is_retracted": False}) is False


def test_classify_openalex_missing_field_is_none():
    assert ar.classify_openalex({"title": "x"}) is None


def test_classify_openalex_non_dict_is_none():
    assert ar.classify_openalex(None) is None
    assert ar.classify_openalex([1, 2, 3]) is None


def test_classify_openalex_non_bool_is_ignored():
    # A stray string must not be mistaken for a retraction verdict.
    assert ar.classify_openalex({"is_retracted": "true"}) is None


# ---------------------------------------------------------------------------
# summary + report formatting
# ---------------------------------------------------------------------------


def test_summarize_counts_each_status():
    rows = [
        ar.RetractionRow("a", "10.1/a", ar.OK),
        ar.RetractionRow("b", "10.1/b", ar.RETRACTED),
        ar.RetractionRow("c", "10.1/c", ar.UNRESOLVED),
        ar.RetractionRow("d", "10.1/d", ar.OK),
    ]
    counts = ar.summarize(rows)
    assert counts[ar.OK] == 2
    assert counts[ar.RETRACTED] == 1
    assert counts[ar.UNRESOLVED] == 1


def test_format_report_flags_retracted_row():
    rows = [ar.RetractionRow("bad", "10.x/retracted", ar.RETRACTED, title="Withdrawn")]
    out = ar.format_report(rows)
    assert "Retracted citations" in out
    assert "@bad" in out
    assert "10.x/retracted" in out


def test_format_report_clean_when_all_ok():
    out = ar.format_report([ar.RetractionRow("g", "10.x/ok", ar.OK)])
    assert "none are retracted" in out
    assert "Retracted citations" not in out


# ---------------------------------------------------------------------------
# CLI exit-code contract (monkeypatched — no network)
# ---------------------------------------------------------------------------


def _stub_status(monkeypatch, mapping):
    def fake(doi, *, refresh=False, sleep=0.0):
        return mapping[doi]

    monkeypatch.setattr(ar, "retraction_status", fake)


def _bib(tmp_path, *dois):
    lines = []
    for i, doi in enumerate(dois):
        lines.append(f"@article{{k{i},\n  doi = {{{doi}}},\n}}\n")
    p = tmp_path / "p.bib"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_main_clean_returns_0(tmp_path, monkeypatch, capsys):
    bib = _bib(tmp_path, "10.1/x")
    _stub_status(monkeypatch, {"10.1/x": (ar.OK, "Title", "")})
    assert ar.main(["--bib", str(bib)]) == 0


def test_main_retracted_returns_1(tmp_path, monkeypatch, capsys):
    bib = _bib(tmp_path, "10.1/x", "10.1/bad")
    _stub_status(
        monkeypatch,
        {"10.1/x": (ar.OK, "", ""), "10.1/bad": (ar.RETRACTED, "Gone", "")},
    )
    assert ar.main(["--bib", str(bib)]) == 1


def test_main_unresolved_is_soft_2_by_default(tmp_path, monkeypatch, capsys):
    bib = _bib(tmp_path, "10.1/x", "10.1/missing")
    _stub_status(
        monkeypatch,
        {"10.1/x": (ar.OK, "", ""), "10.1/missing": (ar.UNRESOLVED, "", "HTTP 404")},
    )
    assert ar.main(["--bib", str(bib)]) == 2


def test_main_unresolved_is_hard_1_under_strict(tmp_path, monkeypatch, capsys):
    bib = _bib(tmp_path, "10.1/missing")
    _stub_status(monkeypatch, {"10.1/missing": (ar.UNRESOLVED, "", "HTTP 404")})
    assert ar.main(["--bib", str(bib), "--strict"]) == 1


def test_main_retracted_beats_unresolved(tmp_path, monkeypatch, capsys):
    # A retraction is a hard failure even alongside unresolved DOIs.
    bib = _bib(tmp_path, "10.1/bad", "10.1/missing")
    _stub_status(
        monkeypatch,
        {"10.1/bad": (ar.RETRACTED, "", ""), "10.1/missing": (ar.UNRESOLVED, "", "")},
    )
    assert ar.main(["--bib", str(bib)]) == 1
