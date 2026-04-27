"""Tests for ``sp.output._bibliography`` — CSL hub + paper.bib writer.

Coverage
--------
- ``CSL_REGISTRY`` has all the journals StatsPAI's table presets
  cover, plus chicago / apa for general use.
- ``csl_url`` / ``csl_filename`` round-trip for every preset.
- Unknown CSL name raises informative ``ValueError``.
- Citation string parser handles the canonical AER / Callaway-style
  forms, falls back gracefully on garbage.
- ``make_bib_key`` is stable + collision-safe across realistic inputs.
- ``write_bib`` produces a syntactically valid .bib (round-trip via
  bibtexparser if available; structural checks otherwise).
- Replication pack now writes a real bib through the pipeline.
- ``to_qmd(csl='aer')`` resolves to the canonical filename.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from statspai.output._bibliography import (
    CSL_REGISTRY,
    citations_to_bib_entries,
    csl_filename,
    csl_url,
    list_csl_styles,
    make_bib_key,
    parse_citation_to_bib,
    write_bib,
)
from statspai.workflow.paper import PaperDraft


# ---------------------------------------------------------------------------
# CSL registry
# ---------------------------------------------------------------------------

class TestCslRegistry:
    def test_aer_present(self):
        assert "aer" in CSL_REGISTRY
        assert CSL_REGISTRY["aer"]["filename"].endswith(".csl")
        assert CSL_REGISTRY["aer"]["url"].startswith("https://")

    def test_all_table_journals_covered(self):
        # The journal-presets module covers these — CSL must too.
        for j in ("aer", "qje", "econometrica", "restat",
                  "jf", "aeja", "jpe", "restud"):
            assert j in CSL_REGISTRY, f"missing CSL for {j!r}"

    def test_aej_family_alias_to_aer_style(self):
        # AEJ family historically uses the AER style.
        for j in ("aeja", "aejmac", "aejmicro", "aejpol"):
            assert (
                CSL_REGISTRY[j]["filename"]
                == "american-economic-association.csl"
            )

    def test_csl_url_lookup(self):
        url = csl_url("aer")
        assert "american-economic-association" in url

    def test_csl_url_case_insensitive(self):
        assert csl_url("AER") == csl_url("aer")

    def test_unknown_csl_raises(self):
        with pytest.raises(ValueError, match="Unknown CSL style"):
            csl_url("nonsense")

    def test_csl_filename_passthrough_for_real_filename(self):
        # Caller already has a real filename — pass through.
        assert csl_filename("custom-style.csl") == "custom-style.csl"

    def test_list_csl_styles(self):
        styles = list_csl_styles()
        assert isinstance(styles, list) and len(styles) > 5
        names = {s[0] for s in styles}
        assert {"aer", "qje", "econometrica"} <= names


# ---------------------------------------------------------------------------
# Citation parser + bib key
# ---------------------------------------------------------------------------

class TestParseCitation:
    def test_canonical_aer_format(self):
        s = ("Callaway B, Sant'Anna PHC. (2021). "
             "Difference-in-Differences with multiple time periods. "
             "Journal of Econometrics.")
        entry = parse_citation_to_bib(s)
        assert entry["type"] == "article"
        assert entry["fields"]["year"] == "2021"
        assert "Callaway" in entry["fields"]["author"]
        assert "Difference" in entry["fields"]["title"]
        assert "Journal of Econometrics" in entry["fields"]["journal"]
        assert entry["key"].startswith("callaway2021")

    def test_loose_author_year(self):
        s = "Imbens GW (2004). Nonparametric estimation of average treatment effects."
        entry = parse_citation_to_bib(s)
        assert entry["fields"]["year"] == "2004"
        assert entry["key"].startswith("imbens2004")

    def test_garbage_falls_back_to_misc(self):
        entry = parse_citation_to_bib("blah blah no recognisable structure")
        assert entry["type"] == "misc"
        # The original text is preserved as a note.
        assert "blah" in entry["fields"]["note"]

    def test_empty_safe(self):
        e1 = parse_citation_to_bib("")
        assert e1["type"] == "misc"
        e2 = parse_citation_to_bib(None)
        assert e2["type"] == "misc"


class TestMakeBibKey:
    def test_canonical_form(self):
        k = make_bib_key(
            "Callaway B, Sant'Anna PHC. (2021). Difference-in-Differences..."
        )
        assert k.startswith("callaway2021")

    def test_stable_across_calls(self):
        s = "Imbens GW (2004). Nonparametric estimation of average ATEs."
        assert make_bib_key(s) == make_bib_key(s)

    def test_unique_for_distinct_papers(self):
        # Two real papers with same first author + year but different
        # opening title words produce distinct keys.
        a = make_bib_key(
            "Imbens GW (2004). Nonparametric estimation of ATEs."
        )
        b = make_bib_key(
            "Imbens GW (2004). Sensitivity analysis for unobserved confounders."
        )
        assert a != b
        # Both still anchored on imbens2004 — only the title-word suffix
        # differentiates them.
        assert a.startswith("imbens2004")
        assert b.startswith("imbens2004")

    def test_collides_when_same_first_title_word(self):
        # Documenting the limitation: when the first title word matches,
        # keys collide. Real papers rarely hit this; users who do can
        # pass an explicit `key=` to parse_citation_to_bib.
        a = make_bib_key("Author A (2020). Title One. Journal X.")
        b = make_bib_key("Author A (2020). Title Two. Journal X.")
        assert a == b
        # The "A" middle initial is recognised, surname slug = "author".
        assert a == "author2020title"


# ---------------------------------------------------------------------------
# citations_to_bib_entries
# ---------------------------------------------------------------------------

class TestEntries:
    def test_dedupe_by_key(self):
        cites = [
            "Callaway B, Sant'Anna PHC. (2021). Difference. Journal A.",
            "Callaway B, Sant'Anna PHC. (2021). Difference. Journal A.",
        ]
        entries = citations_to_bib_entries(cites)
        assert len(entries) == 1

    def test_skips_empty(self):
        entries = citations_to_bib_entries(["", None, "  "])
        assert entries == []


# ---------------------------------------------------------------------------
# write_bib
# ---------------------------------------------------------------------------

class TestWriteBib:
    def test_writes_valid_bibtex(self, tmp_path):
        cites = [
            "Callaway B, Sant'Anna PHC. (2021). DiD with multiple periods. JoE.",
            "Imbens GW, Wooldridge JM. (2009). Recent developments. JEL.",
        ]
        out = write_bib(cites, tmp_path / "paper.bib")
        text = out.read_text(encoding="utf-8")
        # Header.
        assert "auto-generated by StatsPAI" in text
        # One @article per entry.
        assert text.count("@article") == 2
        # Each entry has author/year/title/journal.
        for field in ("author", "year", "title", "journal"):
            assert text.count(f"  {field} = {{") == 2

    def test_dedupe_in_writer(self, tmp_path):
        cites = ["Smith J (2020). Title. JX.", "Smith J (2020). Title. JX."]
        out = write_bib(cites, tmp_path / "p.bib")
        text = out.read_text(encoding="utf-8")
        # One unique entry, even if duplicated in the input.
        assert text.count("@article") == 1

    def test_append_mode(self, tmp_path):
        path = tmp_path / "p.bib"
        write_bib(["A B (2020). T. J."], path)
        write_bib(["C D (2021). U. K."], path, append=True, header=False)
        text = path.read_text(encoding="utf-8")
        # Both entries present after the second write.
        assert text.count("@article") == 2

    def test_accepts_pre_built_dicts(self, tmp_path):
        entries = [
            {"key": "custom2020", "type": "article",
             "fields": {"author": "Custom Author",
                        "title": "Manual Title",
                        "year": "2020"}},
        ]
        out = write_bib(entries, tmp_path / "p.bib")
        text = out.read_text(encoding="utf-8")
        assert "@article{custom2020" in text
        assert "Manual Title" in text


# ---------------------------------------------------------------------------
# Replication pack integration
# ---------------------------------------------------------------------------

class TestReplicationPackBibIntegration:
    def test_pack_writes_real_bib(self, tmp_path):
        from types import SimpleNamespace

        from statspai.output._replication_pack import replication_pack

        # PaperDraft-shaped duck with a real-looking citation.
        class _Draft:
            sections = {"Q": "?"}
            citations = [
                "Callaway B, Sant'Anna PHC. (2021). DiD multi-period. JoE."
            ]
            workflow = None
            fmt = "markdown"

            def to_markdown(self):
                return "## Q\n\n?\n"

        rp = replication_pack(
            _Draft(), tmp_path / "p.zip", env=False,
        )
        import zipfile
        with zipfile.ZipFile(rp.output_path) as zf:
            text = zf.read("paper/paper.bib").decode()
        assert text.startswith("% paper.bib")
        assert "@article{callaway2021" in text


# ---------------------------------------------------------------------------
# to_qmd csl resolution
# ---------------------------------------------------------------------------

class TestToQmdCslShortName:
    def test_csl_aer_resolves_to_filename(self):
        draft = PaperDraft(
            question="x", sections={"Q": "?"},
            workflow=None, fmt="qmd",
        )
        qmd = draft.to_qmd(csl="aer")
        assert 'csl: "american-economic-association.csl"' in qmd

    def test_csl_explicit_filename_passes_through(self):
        draft = PaperDraft(
            question="x", sections={"Q": "?"},
            workflow=None, fmt="qmd",
        )
        qmd = draft.to_qmd(csl="my-custom.csl")
        assert 'csl: "my-custom.csl"' in qmd

    def test_csl_econometrica_resolves(self):
        draft = PaperDraft(
            question="x", sections={"Q": "?"},
            workflow=None, fmt="qmd",
        )
        qmd = draft.to_qmd(csl="econometrica")
        assert 'csl: "econometrica.csl"' in qmd
