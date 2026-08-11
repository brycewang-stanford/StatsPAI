"""Guard: every bracketed ``@bibkey`` citation resolves, and no work sits
under two bib keys.

Two failure modes, both of which had shipped.

**Broken links.** Six bracketed citations in ``tests/reference_parity/``
named keys that do not exist in ``paper.bib`` -- ``@therneau2000survival``
(twice), ``@dechaisemartin2020twoway``, ``@wooldridge2021twoway``,
``@correia2017hdfe``, ``@berge2018fixest``. Every one of those works was
*already* in ``paper.bib`` under a different key, so the citations were
plausible-looking guesses rather than missing literature -- exactly the
failure §10 of CLAUDE.md calls the cheapest possible credibility kill.

**Duplicate entries.** Six works were split across two or three keys
apiece, in every case a preprint/working-paper record beside the
published one: Belloni et al. (SSRN vs Econometrica), Gardner (three
keys), Darolles et al. (SSRN vs Econometrica), Lundberg (SocArXiv vs
*SMR*), Goldsmith-Pinkham et al. (NBER WP vs *AER*), Egami et al.,
Yadlowsky et al. That let code cite the preprint while its own
``References`` text printed the published reference -- the drift §10
warns about when the same work is written out in more than one place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BIB = _ROOT / "paper.bib"

_CITE = re.compile(r"\[@([a-zA-Z][a-zA-Z0-9_]*)\]")
# Entry *starts* only. A body-matching regex silently drops entries whose
# fields wrap onto a line beginning with "}" -- 31 of them here -- which
# would make every check below quietly under-report.
_ENTRY_START = re.compile(r"^@(\w+)\{([^,\s]+),", re.M)

# Keys that appear inside tests *about* the citation tooling, as synthetic
# fixtures. They are meant not to resolve.
_SYNTHETIC = {
    "bibkey",
    "key",
    "defined_key",
    "missing_key",
    "cited_key",
    "key_a",
    "key_b",
    "a",
    "b",
    "ghost",
    "repeated",
    "missing",
    "cited",
    "smith2020example",
    "smith2020",
    "smith2024",
    "orphan_key",
    "someotherkey",
}
_FIXTURE_FILES = {
    "test_audit_bib_coverage.py",
    "test_enrichment_citations.py",
    "test_suggest_bibkey_backfills.py",
    "test_citation_integrity.py",
}


def _entries() -> dict:
    """Map ``key -> body``, slicing each entry from its own start to the next."""
    text = _BIB.read_text(encoding="utf-8")
    starts = list(_ENTRY_START.finditer(text))
    out = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        out[m.group(2)] = text[m.end() : end]
    return out


def _field(body: str, name: str) -> str:
    # The trailing field of an entry has no newline after it -- _ENTRY has
    # already consumed it -- so accept end-of-string as a terminator too.
    m = re.search(rf"\b{name}\s*=\s*\{{(.*?)\}},?\s*(?:\n|$)", body, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _source_files():
    for root in ("src", "tests", "docs"):
        for p in (_ROOT / root).rglob("*"):
            if p.suffix in {".py", ".md", ".R"} and p.is_file():
                yield p
    yield _ROOT / "paper.md"


def test_bib_is_populated():
    """Guard the guard -- an unreadable bib would pass everything below."""
    assert len(_entries()) > 500


def test_every_citation_resolves_to_a_bib_entry():
    keys = set(_entries())
    broken = []
    for p in _source_files():
        if p.name in _FIXTURE_FILES:
            continue
        for i, line in enumerate(
            p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for k in _CITE.findall(line):
                if k not in keys and k not in _SYNTHETIC:
                    broken.append(f"{p.relative_to(_ROOT)}:{i} -> @{k}")
    assert not broken, (
        "citation names a key that is not in paper.bib. Check whether the work "
        "is already there under another key before adding a new entry:\n  "
        + "\n  ".join(sorted(broken))
    )


def test_no_two_entries_share_a_doi():
    by_doi: dict = {}
    for k, body in _entries().items():
        doi = _field(body, "doi").lower()
        if doi:
            by_doi.setdefault(doi, []).append(k)
    dupes = {d: ks for d, ks in by_doi.items() if len(ks) > 1}
    assert not dupes, "same DOI under multiple keys:\n  " + "\n  ".join(
        f"{d}: {sorted(ks)}" for d, ks in sorted(dupes.items())
    )


def test_no_two_entries_share_a_title():
    """A preprint and its published version are the same work: one entry."""
    by_title: dict = {}
    for k, body in _entries().items():
        title = re.sub(r"[^a-z0-9]", "", _field(body, "title").lower())
        if title:
            by_title.setdefault(title, []).append(k)
    dupes = {t: ks for t, ks in by_title.items() if len(ks) > 1}
    assert not dupes, (
        "the same title appears under multiple keys -- keep the published "
        "record and point every citation at it:\n  "
        + "\n  ".join(f"{sorted(ks)}" for ks in dupes.values())
    )


@pytest.mark.parametrize(
    "key",
    [
        "belloni2012sparse",
        "gardner2022twostage",
        "butts2022stage",
        "darolles2011nonparametric",
        "lundberg2024gap",
        "goldsmithpinkham2020bartik",
        "egami2023imperfect",
        "yadlowsky2025evaluating",
    ],
)
def test_consolidated_entries_kept_their_verified_identifiers(key):
    """These were merged from duplicates after two-source verification; a
    silent revert to the preprint record would undo that work."""
    body = _entries()[key]
    assert _field(body, "doi"), f"{key} lost its DOI"
    assert _field(body, "year"), f"{key} lost its year"
