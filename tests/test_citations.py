"""Tests for ``result.cite(format=...)`` and ``sp.bib_for(result)``.

Pins: format coverage (bibtex/apa/json), zero-hallucination policy
(every fact derived from the canonical BibTeX entry — never invented),
backward compat (zero-arg ``cite()`` still returns BibTeX),
and JSON-safety of the structured payload.
"""

from __future__ import annotations

import json

import pytest

import statspai as sp
from statspai.core.results import CausalResult
from statspai.smart.citations import (
    _format_apa,
    _format_authors_apa,
    _initials,
    _parse_bibtex,
    _split_authors,
    render_citation,
)


CALLAWAY_BIB = (
    "@article{callaway2021difference,\n"
    "  title={Difference-in-differences with multiple time periods},\n"
    "  author={Callaway, Brantly and Sant'Anna, Pedro H.C.},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={225},\n"
    "  number={2},\n"
    "  pages={200--230},\n"
    "  year={2021},\n"
    "  publisher={Elsevier}\n"
    "}"
)


# ---------------------------------------------------------------------------
#  BibTeX parser
# ---------------------------------------------------------------------------


class TestBibtexParser:

    def test_extracts_type_and_key(self):
        p = _parse_bibtex(CALLAWAY_BIB)
        assert p["type"] == "article"
        assert p["key"] == "callaway2021difference"

    def test_extracts_all_fields(self):
        p = _parse_bibtex(CALLAWAY_BIB)
        f = p["fields"]
        assert "Difference-in-differences" in f["title"]
        assert "Callaway" in f["author"]
        assert f["journal"] == "Journal of Econometrics"
        assert f["volume"] == "225"
        assert f["number"] == "2"
        assert f["pages"] == "200--230"
        assert f["year"] == "2021"
        assert f["publisher"] == "Elsevier"

    def test_normalises_latex_diacritics(self):
        bib = (
            "@article{x2020,\n"
            "  author={de Chaisemartin, Cl{\\'e}ment and "
            "D'Haultf{\\oe}uille, Xavier},\n"
            "  year={2020}\n"
            "}"
        )
        p = _parse_bibtex(bib)
        assert "Clément" in p["fields"]["author"]
        assert "œ" in p["fields"]["author"]

    def test_malformed_returns_none(self):
        assert _parse_bibtex("not a bibtex entry") is None
        assert _parse_bibtex("") is None
        assert _parse_bibtex(None) is None


# ---------------------------------------------------------------------------
#  Author splitting + APA formatting
# ---------------------------------------------------------------------------


class TestAuthors:

    def test_split_simple_two_authors(self):
        a = _split_authors("Callaway, Brantly and Sant'Anna, Pedro H.C.")
        assert len(a) == 2
        assert a[0] == {"last": "Callaway", "first": "Brantly"}
        assert a[1] == {"last": "Sant'Anna", "first": "Pedro H.C."}

    def test_split_first_last_form(self):
        a = _split_authors("Brantly Callaway and Pedro Sant'Anna")
        assert a[0]["last"] == "Callaway"
        assert a[1]["last"] == "Sant'Anna"

    def test_initials(self):
        assert _initials("Brantly") == "B."
        assert _initials("Pedro H.C.") == "P. H. C."
        assert _initials("Jean-Marc") == "J. M."

    def test_apa_two_authors_uses_ampersand(self):
        a = _split_authors("Callaway, Brantly and Sant'Anna, Pedro H.C.")
        s = _format_authors_apa(a)
        assert s == "Callaway, B., & Sant'Anna, P. H. C."

    def test_apa_one_author(self):
        a = _split_authors("Goodman-Bacon, Andrew")
        assert _format_authors_apa(a) == "Goodman-Bacon, A."

    def test_apa_three_plus_authors(self):
        a = _split_authors("Foo, A and Bar, B and Baz, C and Qux, D")
        s = _format_authors_apa(a)
        assert s == "Foo, A., Bar, B., Baz, C., & Qux, D."


# ---------------------------------------------------------------------------
#  APA prose
# ---------------------------------------------------------------------------


class TestApaProse:

    def test_full_apa_for_callaway(self):
        s = render_citation(CALLAWAY_BIB, fmt="apa")
        assert "Callaway, B., & Sant'Anna, P. H. C." in s
        assert "(2021)" in s
        assert "Difference-in-differences with multiple time periods" in s
        assert "Journal of Econometrics" in s
        assert "225(2)" in s
        # Pages should normalise -- → en-dash
        assert "200–230" in s

    def test_book_apa_uses_publisher(self):
        bib = (
            "@book{angrist2009mostly,\n"
            "  title={Mostly Harmless Econometrics},\n"
            "  author={Angrist, Joshua D and Pischke, J{\\\"o}rn-Steffen},\n"
            "  year={2009},\n"
            "  publisher={Princeton University Press}\n"
            "}"
        )
        s = render_citation(bib, fmt="apa")
        assert "Angrist, J. D., & Pischke, J." in s
        assert "(2009)" in s
        assert "Princeton University Press" in s

    def test_apa_omits_missing_fields_no_invention(self):
        # No journal / publisher → APA prose just stops; doesn't make
        # anything up.
        bib = (
            "@article{x2020,\n"
            "  author={Foo, B},\n"
            "  year={2020},\n"
            "  title={Untitled}\n"
            "}"
        )
        s = render_citation(bib, fmt="apa")
        assert "Foo, B." in s
        assert "2020" in s
        assert "Untitled" in s
        # Must not invent a journal name.
        for fake in ("Unknown Journal", "Manuscript", "preprint"):
            assert fake not in s


# ---------------------------------------------------------------------------
#  JSON payload
# ---------------------------------------------------------------------------


class TestJsonPayload:

    def test_json_has_canonical_keys(self):
        j = render_citation(CALLAWAY_BIB, fmt="json")
        for k in ("type", "key", "authors", "year", "title", "journal",
                  "volume", "number", "pages", "publisher", "fields"):
            assert k in j

    def test_json_authors_are_structured(self):
        j = render_citation(CALLAWAY_BIB, fmt="json")
        assert isinstance(j["authors"], list)
        assert all(set(a.keys()) >= {"last", "first"}
                   for a in j["authors"])

    def test_json_round_trips(self):
        j = render_citation(CALLAWAY_BIB, fmt="json")
        json.dumps(j)


# ---------------------------------------------------------------------------
#  CausalResult.cite(format=...)
# ---------------------------------------------------------------------------


@pytest.fixture
def did_result():
    return CausalResult(
        method="callaway_santanna", estimand="ATT",
        estimate=1.5, se=0.5, pvalue=0.003, ci=(0.5, 2.5),
        alpha=0.05, n_obs=1000, model_info={},
        _citation_key="callaway_santanna",
    )


class TestCausalResultCite:

    def test_default_is_bibtex(self, did_result):
        # Backward compat: zero-arg cite() must still return BibTeX
        # string, byte-identical to fmt="bibtex".
        assert did_result.cite() == did_result.cite(format="bibtex")
        assert did_result.cite().startswith("@article{")

    def test_apa_returns_string(self, did_result):
        s = did_result.cite(format="apa")
        assert isinstance(s, str)
        assert "Callaway" in s and "(2021)" in s

    def test_json_returns_dict(self, did_result):
        j = did_result.cite(format="json")
        assert isinstance(j, dict)
        assert j["key"] == "callaway2021difference"

    def test_invalid_format_raises(self, did_result):
        with pytest.raises(ValueError):
            did_result.cite(format="latex")

    def test_unregistered_method_returns_placeholder(self):
        r = CausalResult(method="totally_made_up", estimand="ATE",
                          estimate=0.0, se=0.0, pvalue=0.5, ci=(0, 0),
                          alpha=0.05, n_obs=100, model_info={})
        bib = r.cite()
        assert "No citation registered" in bib
        # APA / JSON also surface the placeholder rather than guessing.
        assert "No citation registered" in r.cite(format="apa")
        j = r.cite(format="json")
        assert j["key"] is None


# ---------------------------------------------------------------------------
#  sp.bib_for top-level
# ---------------------------------------------------------------------------


class TestBibForTopLevel:

    def test_returns_dict(self, did_result):
        b = sp.bib_for(did_result)
        assert isinstance(b, dict)
        assert b["key"] == "callaway2021difference"

    def test_in_all(self):
        assert "bib_for" in sp.__all__

    def test_typeerror_on_non_result(self):
        with pytest.raises(TypeError):
            sp.bib_for("not a result")

    def test_legacy_result_with_no_format_kwarg_falls_back(self):
        # A minimal stub that only accepts cite() (no kwargs) — must
        # still produce a structured payload via BibTeX parsing.
        class _LegacyStub:
            def cite(self):
                return CALLAWAY_BIB

        b = sp.bib_for(_LegacyStub())
        assert b["key"] == "callaway2021difference"
        assert b["title"].startswith("Difference-in-differences")


# ---------------------------------------------------------------------------
#  Zero-hallucination guard: APA fields must come from the BibTeX,
#  never from the formatter
# ---------------------------------------------------------------------------


class TestMultiEntryBibtex:
    """Some _CITATIONS slots store TWO @article{} entries concatenated
    (e.g. ``twfe_decomposition`` cites Goodman-Bacon 2021 AND
    de Chaisemartin & D'Haultfœuille 2020). The renderer must surface
    BOTH — never silently drop the second author."""

    TWO_ENTRY = (
        "@article{goodmanbacon2021difference,\n"
        "  title={Difference-in-differences with variation in treatment timing},\n"
        "  author={Goodman-Bacon, Andrew},\n"
        "  journal={Journal of Econometrics},\n"
        "  volume={225},\n"
        "  year={2021}\n"
        "}\n"
        "@article{dechaisemartin2020two,\n"
        "  title={Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects},\n"
        "  author={de Chaisemartin, Cl{\\'e}ment and D'Haultf{\\oe}uille, Xavier},\n"
        "  journal={American Economic Review},\n"
        "  volume={110},\n"
        "  year={2020}\n"
        "}"
    )

    def test_apa_concatenates_both_entries(self):
        s = render_citation(self.TWO_ENTRY, fmt="apa")
        assert "Goodman-Bacon" in s
        # The second entry must NOT be silently dropped.
        assert "Chaisemartin" in s
        assert "(2021)" in s
        assert "(2020)" in s

    def test_json_returns_list_for_multi_entry(self):
        j = render_citation(self.TWO_ENTRY, fmt="json")
        assert isinstance(j, list)
        assert len(j) == 2
        keys = {e["key"] for e in j}
        assert "goodmanbacon2021difference" in keys
        assert "dechaisemartin2020two" in keys

    def test_json_returns_dict_for_single_entry(self):
        # Backward compat: single-entry inputs still get a dict.
        single = (
            "@article{x2020,\n"
            "  author={Foo, B}, year={2020}, title={A}\n"
            "}"
        )
        j = render_citation(single, fmt="json")
        assert isinstance(j, dict)

    def test_bibtex_passes_through_unchanged(self):
        # bibtex format on multi-entry returns the raw string as-is.
        assert render_citation(self.TWO_ENTRY, fmt="bibtex") == self.TWO_ENTRY


class TestNoHallucination:
    """If the source BibTeX doesn't contain a field, the APA prose
    must NOT invent one. Regression guard for CLAUDE.md §10."""

    def test_no_phantom_publisher_for_article(self):
        bib = (
            "@article{x2020,\n"
            "  author={Foo, B}, year={2020}, title={A Title},\n"
            "  journal={Some J}\n"
            "}"
        )
        s = render_citation(bib, fmt="apa")
        # Common phantoms a model might insert.
        for fake in ("Springer", "Elsevier", "Wiley", "Oxford",
                      "MIT Press"):
            assert fake not in s, (
                f"hallucinated publisher {fake!r} in APA prose: {s!r}")

    def test_no_phantom_year_when_missing(self):
        bib = (
            "@article{x,\n"
            "  author={Foo, B}, title={Untitled}, journal={Some J}\n"
            "}"
        )
        s = render_citation(bib, fmt="apa")
        # The current decade should NOT appear if not in the source.
        for fake_year in ("2024", "2023", "2022", "2021", "2020"):
            assert fake_year not in s, (
                f"hallucinated year {fake_year!r} in APA prose: {s!r}")
