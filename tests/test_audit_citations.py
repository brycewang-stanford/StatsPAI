"""Tests for ``tools/audit_citations.py`` — the §10 citation auditor.

These tests never touch the network or the real arXiv / Crossref / NBER
endpoints. They exercise the pure-Python layer only:
  * regex extraction (``ARXIV_RE``, ``NBER_RE``, ``DOI_RE``)
  * name normalisation (``_normalise``, ``_last_name``)
  * the ``diff_citation`` classifier against a hand-built ``PaperMeta``
  * ``extract_citations`` over a tmp_path tree
  * CLI exit code semantics (via ``--kinds`` + a tmp source tree so the
    test never depends on the real paper.bib / src tree)

The two regressions fixed in commits ``fa35662`` and earlier are pinned
explicitly so a future regex / stopword tweak doesn't silently break:
  * DOI regex must NOT swallow a trailing ``}`` from a bibtex
    ``doi={10.x/y}`` literal.
  * ``Q-network`` must NOT be flagged as a phantom author surname.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import audit_citations as ac  # noqa: E402

# ---------------------------------------------------------------------------
# Regex: arXiv
# ---------------------------------------------------------------------------


def test_arxiv_regex_matches_standard_form():
    m = ac.ARXIV_RE.search("See arXiv:2408.12345 for details.")
    assert m is not None
    assert m.group("id") == "2408.12345"


def test_arxiv_regex_matches_5digit_suffix():
    m = ac.ARXIV_RE.search("arXiv:2408.12345 is fine but arXiv:2408.123456 too")
    assert m is not None
    assert m.group("id") == "2408.12345"


def test_arxiv_regex_strips_version_suffix():
    m = ac.ARXIV_RE.search("arXiv:2008.12345v3")
    assert m is not None
    assert m.group("id") == "2008.12345"  # v3 dropped


def test_arxiv_regex_ignores_bare_year_number():
    # "2008.12345" without the arXiv prefix must NOT match: the tool
    # only audits explicitly-flagged arXiv references.
    assert ac.ARXIV_RE.search("version 2008.12345 of the library") is None


# ---------------------------------------------------------------------------
# Regex: NBER
# ---------------------------------------------------------------------------


def test_nber_regex_requires_wp_marker():
    m = ac.NBER_RE.search("NBER Working Paper 26463")
    assert m is not None
    assert m.group("id") == "26463"


def test_nber_regex_accepts_w_prefix():
    m = ac.NBER_RE.search("NBER WP w34550 provides …")
    assert m is not None
    assert m.group("id") == "34550"


def test_nber_regex_ignores_bare_number():
    # A standalone 5-digit number is NOT an NBER reference.
    assert ac.NBER_RE.search("room 26463 at the conference") is None


# ---------------------------------------------------------------------------
# Regex: DOI — REGRESSION for the bibtex closing-brace bug
# ---------------------------------------------------------------------------


def test_doi_regex_excludes_closing_brace():
    """Regression for commit fa35662: the DOI regex used to greedy-eat
    the ``}`` that closes a bibtex ``doi={10.x/y}`` field. Real paper
    citations inside ``_CITATIONS`` dicts therefore queried Crossref
    with a literal ``}`` in the id and came back 404 UNRESOLVED.
    """
    line = '    "  doi={10.1080/01621459.2015.1012259}\\n"'
    m = ac.DOI_RE.search(line)
    assert m is not None
    assert m.group("id") == "10.1080/01621459.2015.1012259"
    assert "}" not in m.group("id")


def test_doi_regex_excludes_closing_paren():
    m = ac.DOI_RE.search("see (10.1234/abcd.5678) for proof")
    assert m is not None
    assert m.group("id") == "10.1234/abcd.5678"


def test_doi_regex_excludes_trailing_semicolon():
    m = ac.DOI_RE.search("DOI 10.1234/abcd.5678; published 2024")
    assert m is not None
    assert m.group("id") == "10.1234/abcd.5678"


def test_doi_regex_keeps_internal_digit_periods():
    """Real DOIs have multiple dot-separated numeric fragments (e.g.
    ``10.1080/01621459.2015.1012259``). The regex must NOT stop at
    internal ``.``-then-digit boundaries — only at ``.``-then-non-digit
    (which is a sentence-terminating period)."""
    m = ac.DOI_RE.search("see 10.1080/01621459.2015.1012259 for proof")
    assert m is not None
    assert m.group("id") == "10.1080/01621459.2015.1012259"


def test_doi_regex_stops_at_sentence_period():
    """A trailing ``.`` followed by a space marks end-of-sentence, not
    part of the DOI."""
    m = ac.DOI_RE.search("see 10.1234/xyz. Then move on.")
    assert m is not None
    assert m.group("id") == "10.1234/xyz"


# ---------------------------------------------------------------------------
# Network failures — must degrade to unresolved, not traceback
# ---------------------------------------------------------------------------


def test_verify_crossref_timeout_is_soft_failure(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise TimeoutError("read operation timed out")

    monkeypatch.setattr(ac, "_http_get", boom)

    assert ac.verify_crossref(["10.1234/example"]) == {}
    assert "TimeoutError" in capsys.readouterr().err


def test_verify_arxiv_timeout_is_soft_failure(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise TimeoutError("read operation timed out")

    monkeypatch.setattr(ac, "_http_get", boom)

    assert ac.verify_arxiv(["2408.12345"]) == {}
    assert "TimeoutError" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


def test_last_name_lastfirst_format():
    assert ac._last_name("Imbens, Guido") == "Imbens"


def test_last_name_firstlast_format():
    assert ac._last_name("Guido Imbens") == "Imbens"


def test_last_name_hyphenated_surname_preserved():
    assert ac._last_name("Andrew Goodman-Bacon") == "Goodman-Bacon"


def test_last_name_double_surname():
    # "Tchetgen Tchetgen" is one person, two-word surname. Without a
    # comma, we only get the last token — documented limitation.
    assert ac._last_name("Eric Tchetgen Tchetgen") == "Tchetgen"
    # Comma form keeps the full double surname:
    assert ac._last_name("Tchetgen Tchetgen, Eric") == "Tchetgen Tchetgen"


def test_normalise_strips_diacritics_and_lowercases():
    assert ac._normalise("Sant'Anna") == "sant'anna"
    assert ac._normalise("Guimarães") == "guimaraes"
    assert ac._normalise("O'Neill") == "o'neill"


def test_strip_diacritics_preserves_case():
    assert ac._strip_diacritics("Café") == "Cafe"


def test_paper_meta_last_names_normalises_each():
    meta = ac.PaperMeta(
        authors=["Guido Imbens", "Sant'Anna, Pedro", "Goodman-Bacon, Andrew"],
        title="x",
        year=2020,
        source="arxiv",
    )
    lasts = meta.last_names()
    assert "imbens" in lasts
    assert "sant'anna" in lasts
    assert "goodman-bacon" in lasts


# ---------------------------------------------------------------------------
# Stopwords — REGRESSION for the Q-network false positive
# ---------------------------------------------------------------------------


def test_q_network_in_stopwords():
    """Regression for commit fa35662: 'Q-network' was being parsed as
    a phantom author surname because it survived the stopword filter."""
    assert "q-network" in ac.SURNAME_STOPWORDS
    assert "q-learning" in ac.SURNAME_STOPWORDS


def test_common_ml_acronyms_in_stopwords():
    for tok in ("dqn", "ppo", "a3c", "trpo", "actor-critic"):
        assert tok in ac.SURNAME_STOPWORDS, f"{tok!r} should be a stopword"


def test_real_surnames_not_in_stopwords():
    """Sanity check: we shouldn't have accidentally nuked common
    surnames. Note that a few real surnames (``cattaneo``, ``jansson``,
    ``ma``) ARE stopwords by design — they appear in the codebase only
    as method-name shorthand ("Cattaneo-Jansson bootstrap"), never as
    the true author of the papers we cite."""
    for tok in (
        "imbens",
        "angrist",
        "bareinboim",
        "athey",
        "sant'anna",
        "pearl",
        "wooldridge",
    ):
        assert tok not in ac.SURNAME_STOPWORDS, (
            f"{tok!r} is a real author surname; removing from "
            "SURNAME_STOPWORDS blocks phantom detection for them"
        )


# ---------------------------------------------------------------------------
# diff_citation — classifier behaviour
# ---------------------------------------------------------------------------


def _make_citation(
    claim: str, *, kind: str = "arxiv", id: str = "2510.21110", line: int = 1
) -> ac.Citation:
    return ac.Citation(
        kind=kind,
        id=id,
        file="synthetic.py",
        line=line,
        claim_block=claim,
        same_line=claim.splitlines()[0] if claim else "",
        claimed_year=2025,
    )


def test_diff_citation_matching_single_author_is_clean():
    truth = ac.PaperMeta(
        authors=["Guido Imbens"],
        title="T",
        year=2020,
        source="arxiv",
    )
    c = _make_citation("(Imbens 2020) arXiv:2510.21110")
    assert ac.diff_citation(c, truth) == []


def test_diff_citation_wrong_author_is_flagged():
    truth = ac.PaperMeta(
        authors=["Guido Imbens", "Joshua Angrist"],
        title="T",
        year=2020,
        source="arxiv",
    )
    # Claim credits a surname that ISN'T on the paper, in author
    # position. The phantom detector requires an author-list punctuation
    # marker ('&', 'and', ',', 'et al.') after the surname, so we use
    # the classic "Morgan & Winship" phrasing to guarantee a trigger.
    c = _make_citation("(Morgan & Winship 2020) arXiv:2510.21110")
    issues = ac.diff_citation(c, truth)
    assert issues
    joined = " ".join(issues).lower()
    assert "morgan" in joined or "winship" in joined


def test_diff_citation_q_network_not_flagged_as_author():
    """Regression: 'Causal deep Q-network (Li, Zhang, Bareinboim 2025)'
    should be clean — Q-network is a method name, not an author."""
    truth = ac.PaperMeta(
        authors=["Mingxuan Li", "Junzhe Zhang", "Elias Bareinboim"],
        title="Causal Deep Q-Network …",
        year=2025,
        source="arxiv",
    )
    c = _make_citation(
        "Causal deep Q-network (Li, Zhang, Bareinboim 2025, arXiv:2510.21110) "
        "for offline policy learning"
    )
    issues = ac.diff_citation(c, truth)
    # Zero issues: Q-network is a stopword, and all three truth authors
    # are present.
    assert issues == [], issues


def test_diff_citation_quoted_title_words_not_flagged_as_author():
    """Regression: a correctly-attributed paper whose *quoted title*
    contains an ``X and Y`` phrase must not flag a title word as a phantom
    author. Real case: Scheidegger/Guo/Bühlmann (arXiv:2503.03530), title
    "... Efficient Instruments and Machine Learning" — "Instruments and
    Machine" used to read as a faux "Surname and Surname" author list.
    Authors always sit OUTSIDE the title quotes, so stripping the quoted
    span is precision-only.
    """
    truth = ac.PaperMeta(
        authors=["Cyrill Scheidegger", "Zijian Guo", "Peter Bühlmann"],
        title="Inference for Heterogeneous Treatment Effects with "
        "Efficient Instruments and Machine Learning",
        year=2025,
        source="arxiv",
    )
    c = _make_citation(
        'Scheidegger, C., Guo, Z. and Bühlmann, P. (2025). "Inference for '
        "Heterogeneous Treatment Effects with Efficient Instruments and "
        'Machine Learning." arXiv:2503.03530',
        id="2503.03530",
    )
    issues = ac.diff_citation(c, truth)
    assert issues == [], issues


def test_diff_citation_invented_author_outside_title_quote_still_flagged():
    """Guard the precision fix above does NOT blind the §10 gate: an
    invented co-author in author position (outside the title quotes) must
    still be caught."""
    truth = ac.PaperMeta(
        authors=["Cyrill Scheidegger", "Zijian Guo", "Peter Bühlmann"],
        title="Inference for Heterogeneous Treatment Effects",
        year=2025,
        source="arxiv",
    )
    c = _make_citation(
        "Scheidegger, C., Guo, Z., Bühlmann, P. and Fakeson, Q. (2025). "
        '"Inference for Heterogeneous Treatment Effects." arXiv:2503.03530',
        id="2503.03530",
    )
    issues = ac.diff_citation(c, truth)
    assert any("fakeson" in i.lower() for i in issues), issues


def test_diff_citation_bare_reference_does_not_spuriously_flag_missing():
    """Lines that are pure arXiv URL references (no author text) should
    not be flagged for 'missing author': there's no attribution text to
    check against.
    """
    truth = ac.PaperMeta(
        authors=["Alice Smith", "Bob Jones"],
        title="T",
        year=2024,
        source="arxiv",
    )
    c = _make_citation("See https://arxiv.org/abs/2510.21110 for details.")
    issues = ac.diff_citation(c, truth)
    assert issues == []


# ---------------------------------------------------------------------------
# extract_citations — filesystem walk
#
# ``extract_citations`` computes ``path.relative_to(REPO_ROOT)`` to record
# a repo-relative filename. To test on a tmp tree we monkeypatch REPO_ROOT
# to point at the tmp dir.
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "REPO_ROOT", tmp_path)
    return tmp_path


def test_extract_citations_finds_arxiv_and_doi_in_py_file(tmp_repo):
    src = tmp_repo / "mod.py"
    src.write_text(
        '"""Method of Foo et al.\n'
        "References\n"
        "----------\n"
        "Foo, Bar (2024). arXiv:2408.12345\n"
        "Baz, Qux (2023). doi:10.1234/abcd.5678\n"
        '"""\n',
        encoding="utf-8",
    )
    citations = ac.extract_citations([tmp_repo])
    kinds = {c.kind for c in citations}
    assert "arxiv" in kinds
    assert "doi" in kinds
    by_id = {c.id: c for c in citations}
    assert "2408.12345" in by_id
    assert "10.1234/abcd.5678" in by_id


def test_extract_citations_skips_unknown_extensions(tmp_repo):
    src = tmp_repo / "data.json"
    src.write_text('{"note": "arXiv:2408.12345"}', encoding="utf-8")
    citations = ac.extract_citations([tmp_repo])
    # .json is not in the extract list — must be skipped.
    assert citations == []


def test_extract_citations_accepts_markdown(tmp_repo):
    # The arXiv regex requires the literal "arXiv" marker followed by
    # ':' or whitespace — bare URLs (``https://arxiv.org/abs/XXXX``)
    # are intentionally skipped to avoid matching unrelated URL
    # patterns. Documentation conventions in this repo prefer
    # ``arXiv:XXXX`` citations, which this test pins.
    src = tmp_repo / "guide.md"
    src.write_text(
        "See Foo et al. (2024), arXiv:2408.12345, for the full proof.",
        encoding="utf-8",
    )
    citations = ac.extract_citations([tmp_repo])
    assert len(citations) == 1
    assert citations[0].id == "2408.12345"
    assert citations[0].file.endswith("guide.md")


def test_extract_citations_records_line_number(tmp_repo):
    src = tmp_repo / "mod.py"
    src.write_text(
        "# header\n" "# another\n" "# Foo (2024) arXiv:2408.12345\n",
        encoding="utf-8",
    )
    citations = ac.extract_citations([tmp_repo])
    assert len(citations) == 1
    assert citations[0].line == 3  # 1-indexed


# ---------------------------------------------------------------------------
# CLI — exit code semantics (no network: use --kinds to an empty bucket)
# ---------------------------------------------------------------------------


def test_cli_runs_without_crash_on_empty_tree(tmp_path):
    """Smoke test: the CLI must not traceback or segfault on a trivial
    invocation. We do NOT pass ``--strict`` here because the auditor
    resolves roots relative to its own ``REPO_ROOT`` constant (not
    ``cwd``), so in CI it actually scans the real src/ and makes live
    arXiv calls — which can hit HTTP 429 rate limits on a cold runner
    and would flip ``--strict`` into exit 1.

    Accepted exits:

    * ``0`` — clean run, no findings.
    * ``1`` — ran successfully and emitted a report listing
      mismatches / unresolved DOIs (regex-level surname false
      positives such as treating ``"Form"`` / ``"Behavior"`` /
      ``"SEs"`` as author surnames are a known auditor limitation,
      not a crash).
    * ``2`` — soft failure (rate limit, network) — acceptable.

    Any other exit, and any traceback in ``stderr``, is a real failure.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "mod.py").write_text("# no citations here\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "audit_citations.py"),
            "--roots",
            "src",
            "docs",
            "--out",
            str(tmp_path / "report.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert (
        "Traceback" not in result.stderr
    ), f"auditor crashed with traceback:\n{result.stderr}"
    assert result.returncode in (
        0,
        1,
        2,
    ), f"unexpected exit {result.returncode}: {result.stderr}"


def test_cli_strict_flag_is_recognised():
    """--strict must be a documented CLI flag (not swallowed as positional)."""
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "audit_citations.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--strict" in result.stdout


# ---------------------------------------------------------------------------
# _http_get — retry / back-off on 429 & 5xx
#
# arXiv's export API throttles GitHub's shared runner IP pool with HTTP
# 429s that clear after a short back-off. A single un-retried 429 drops
# a whole batch of ids and (under --strict) used to fail the §10 gate
# with 0 mismatch / N unresolved. _http_get now retries 429/5xx.
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_http_io(monkeypatch):
    """Bypass the disk cache and never actually sleep."""
    monkeypatch.setattr(ac, "_cache_get", lambda *a, **k: None)
    monkeypatch.setattr(ac, "_cache_put", lambda *a, **k: None)
    monkeypatch.setattr(ac.time, "sleep", lambda *_a, **_k: None)


def test_http_get_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ac.urllib.error.HTTPError(
                "http://x", 429, "Too Many Requests", {}, None
            )
        return _FakeResp(b"PAYLOAD")

    _stub_http_io(monkeypatch)
    monkeypatch.setattr(ac.urllib.request, "urlopen", fake_urlopen)

    assert ac._http_get("http://x", refresh=True) == b"PAYLOAD"
    assert calls["n"] == 3  # two 429s + one success


def test_http_get_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    def always_429(req, timeout=None, context=None):
        calls["n"] += 1
        raise ac.urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, None)

    _stub_http_io(monkeypatch)
    monkeypatch.setattr(ac.urllib.request, "urlopen", always_429)

    with pytest.raises(ac.urllib.error.HTTPError):
        ac._http_get("http://x", refresh=True)
    assert calls["n"] == 3


def test_http_get_does_not_retry_404(monkeypatch):
    """404 is a definitive miss, not a throttle — must not be retried."""
    calls = {"n": 0}

    def raise_404(req, timeout=None, context=None):
        calls["n"] += 1
        raise ac.urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)

    _stub_http_io(monkeypatch)
    monkeypatch.setattr(ac.urllib.request, "urlopen", raise_404)

    with pytest.raises(ac.urllib.error.HTTPError):
        ac._http_get("http://x", refresh=True)
    assert calls["n"] == 1  # no retry


def test_parse_retry_after_seconds_and_garbage():
    assert ac._parse_retry_after("5") == 5.0
    assert ac._parse_retry_after("  12  ") == 12.0
    assert ac._parse_retry_after(None) is None
    assert ac._parse_retry_after("Wed, 21 Oct 2025 07:28:00 GMT") is None


# ---------------------------------------------------------------------------
# verify_* — transient-failure tracking (soft failure vs genuine miss)
# ---------------------------------------------------------------------------


def test_verify_arxiv_records_transient_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("read operation timed out")

    monkeypatch.setattr(ac, "_http_get", boom)
    transient: set[str] = set()
    assert ac.verify_arxiv(["2408.12345", "2409.00001"], transient=transient) == {}
    # Whole chunk failed to reach arXiv → every id is transient.
    assert transient == {"2408.12345", "2409.00001"}


def test_verify_crossref_404_is_genuine_not_transient(monkeypatch):
    def raise_404(*a, **k):
        raise ac.urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)

    monkeypatch.setattr(ac, "_http_get", raise_404)
    monkeypatch.setattr(ac, "_verify_datacite_one", lambda doi, refresh=False: None)
    transient: set[str] = set()
    assert ac.verify_crossref(["10.1234/x"], transient=transient) == {}
    # A 404 means "Crossref definitively has no such DOI" — a genuine
    # §10 miss, NOT an infrastructure hiccup.
    assert transient == set()


def test_verify_crossref_records_transient_on_5xx(monkeypatch):
    def raise_503(*a, **k):
        raise ac.urllib.error.HTTPError(
            "http://x", 503, "Service Unavailable", {}, None
        )

    monkeypatch.setattr(ac, "_http_get", raise_503)
    transient: set[str] = set()
    assert ac.verify_crossref(["10.1234/x"], transient=transient) == {}
    assert transient == {"10.1234/x"}


# ---------------------------------------------------------------------------
# main() — exit-code contract: 1 = genuine §10 failure, 2 = soft failure
# ---------------------------------------------------------------------------


def _seed_one_arxiv_citation(tmp_repo):
    src = tmp_repo / "src"
    src.mkdir()
    (src / "mod.py").write_text("# Foo (2024) arXiv:2408.12345\n", encoding="utf-8")
    (tmp_repo / "docs").mkdir()


def test_main_strict_transient_unresolved_returns_2(tmp_repo, monkeypatch):
    _seed_one_arxiv_citation(tmp_repo)

    def throttled(ids, refresh=False, transient=None):
        if transient is not None:
            transient.update(ids)  # arXiv unreachable: all ids transient
        return {}

    monkeypatch.setattr(ac, "verify_arxiv", throttled)
    rc = ac.main(
        [
            "--roots",
            "src",
            "docs",
            "--kinds",
            "arxiv",
            "--strict",
            "--out",
            str(tmp_repo / "r.md"),
        ]
    )
    assert rc == 2  # soft failure — must not block a merge


def test_main_strict_genuine_unresolved_returns_1(tmp_repo, monkeypatch):
    _seed_one_arxiv_citation(tmp_repo)

    # Source reachable, id genuinely absent → transient stays empty.
    monkeypatch.setattr(
        ac, "verify_arxiv", lambda ids, refresh=False, transient=None: {}
    )
    rc = ac.main(
        [
            "--roots",
            "src",
            "docs",
            "--kinds",
            "arxiv",
            "--strict",
            "--out",
            str(tmp_repo / "r.md"),
        ]
    )
    assert rc == 1  # genuine §10 failure — blocks the merge


def test_main_nonstrict_unresolved_returns_0(tmp_repo, monkeypatch):
    _seed_one_arxiv_citation(tmp_repo)
    monkeypatch.setattr(
        ac, "verify_arxiv", lambda ids, refresh=False, transient=None: {}
    )
    rc = ac.main(
        ["--roots", "src", "docs", "--kinds", "arxiv", "--out", str(tmp_repo / "r.md")]
    )
    assert rc == 0  # non-strict: unresolved alone never fails
