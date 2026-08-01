"""Offline guard: an arXiv ID must not be written next to the wrong authors.

``tools/audit_citations.py`` already checks §10's four elements against the
live arXiv / Crossref / DataCite APIs, and it does compare surnames. It is
not what failed here. What failed is that its June 2026 findings were fixed
on a branch that never reached ``main``, so three fabricated attributions on
*real* arXiv IDs sat in shipped docstrings for two months while the auditor
that had already found them needed a network and a human to run it.

This test needs neither. ``paper.bib`` is §10's single source of truth, and
every ID involved was already in it, correctly. The bug was purely drift
between the bib entry and hand-written copies elsewhere — exactly what §10
forbids ("禁止在多处手写同一条引用"). Comparing the two is offline,
deterministic, and would have caught all three:

* ``causal_llm``: "(Kiciman-Sharma 2025, arXiv 2402.11068)" — 2402.11068 is
  Wan, Lu, Wu, Hu & Li (2024); the Kıcıman paper is 2305.00050.
* ``docs/guides/causal_mas.md``: same string, plus a wrong *title* on the
  Wan et al. entry that survived the June pass.
* ``docs/guides/qte_family.md``: "(Sharma-Xue 2025)" for ``sp.dist_iv``,
  whose arXiv 2502.07641 is Holovchak, Saengkyongam, Meinshausen & Shen.

Scope is deliberately narrow: it fires only on an explicit parenthesised
attribution sitting next to an arXiv ID that ``paper.bib`` already knows.
That is the shape that broke, and keeping it narrow keeps it from becoming
a test people learn to silence.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper.bib"
SCAN_ROOTS = (ROOT / "src", ROOT / "docs")
SCAN_SUFFIXES = {".py", ".md"}

# "arXiv 2402.11068" / "arXiv:2402.11068v3" / "arxiv 2502.07641"
ARXIV_RE = re.compile(r"ar[Xx]iv:?\s*(\d{4}\.\d{4,5})(?:v\d+)?")

# An attribution immediately before the ID, inside the same parenthesis:
#   (Kiciman-Sharma 2025, arXiv 2402.11068)
#   (Wuthrich-Zhu 2025, arXiv 2505.09706)
#   (Shaw et al. 2025, arXiv 2506.12765)
# Name tokens allow diacritics (Kıcıman) and internal hyphens.
NAME = r"[A-ZÀ-ɏ][\wÀ-ɏ'’]+"
ATTRIB_RE = re.compile(
    rf"\((?P<names>{NAME}(?:[-–—&]{NAME}|,\s*{NAME}|\s+(?:and|&)\s+{NAME})*"
    rf"(?:\s+et\s+al\.?)?)\s+(?P<year>(?:19|20|21)\d{{2}})\s*,\s*"
    rf"ar[Xx]iv:?\s*(?P<id>\d{{4}}\.\d{{4,5}})"
)

# Words that can start a parenthetical and are not surnames.
NOT_A_SURNAME = {
    "See",
    "Cf",
    "The",
    "A",
    "An",
    "Via",
    "From",
    "After",
    "Following",
    "Default",
    "Note",
}


def _fold(s: str) -> str:
    """Casefold and strip diacritics so Kıcıman == Kiciman == KICIMAN."""
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Turkish dotless i and similar do not decompose; map the ones we hit.
    return stripped.replace("ı", "i").replace("İ", "I").casefold()


def _bib_index() -> dict[str, tuple[str, set[str], str | None]]:
    """arXiv id -> (bib key, folded surnames, year), from ``paper.bib``."""
    text = BIB.read_text(encoding="utf-8")
    entries = re.split(r"\n(?=@)", text)
    index: dict[str, tuple[str, set[str], str | None]] = {}
    for entry in entries:
        ids = set(ARXIV_RE.findall(entry))
        if not ids:
            continue
        key_m = re.search(r"@\w+\{([^,]+),", entry)
        author_m = re.search(r"author\s*=\s*\{(.+?)\}\s*,?\s*\n", entry, re.S)
        if not author_m:
            author_m = re.search(r"author\s*=\s*\{(.+?)\}", entry, re.S)
        year_m = re.search(r"year\s*=\s*\{?(\d{4})", entry)
        if not (key_m and author_m):
            continue
        surnames = set()
        for person in re.split(r"\s+and\s+", author_m.group(1)):
            person = person.strip().strip("{}")
            if not person:
                continue
            # "Lastname, First" or "First Lastname"
            last = person.split(",")[0] if "," in person else person.split()[-1]
            last = re.sub(r"[{}\\\"'~^]", "", last)
            if last:
                surnames.add(_fold(last))
        for aid in ids:
            index[aid] = (key_m.group(1), surnames, year_m.group(1) if year_m else None)
    return index


def _scan_files() -> list[Path]:
    out = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix in SCAN_SUFFIXES and p.is_file():
                out.append(p)
    return out


def _surname_to_ids(index) -> dict[str, set[str]]:
    """Folded surname -> every arXiv ID paper.bib credits that surname with.

    This backs the second rule. The bib cross-check alone cannot catch the
    ``causal_llm`` regression, because there the *ID itself* was wrong
    (2402.11068 instead of 2305.00050) and 2402.11068 is in no bib entry —
    so there is nothing to compare against. But "Kiciman" is a surname
    paper.bib knows, and paper.bib never associates it with 2402.11068.
    A surname the bibliography knows, sitting next to an ID it never credits
    that surname with, is the fingerprint of a mis-remembered citation.
    """
    out: dict[str, set[str]] = {}
    for aid, (_key, surnames, _year) in index.items():
        for s in surnames:
            out.setdefault(s, set()).add(aid)
    return out


def _findings() -> list[str]:
    index = _bib_index()
    known_ids = _surname_to_ids(index)
    problems: list[str] = []
    for path in _scan_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        for m in ATTRIB_RE.finditer(text):
            aid = m.group("id")
            written = m.group("names")
            tokens = [
                t.strip() for t in re.split(r"[-–—&,]|\s+(?:and|et|al\.?)\s*", written)
            ]
            folded = {_fold(t) for t in tokens if t and t not in NOT_A_SURNAME}
            if not folded:
                continue
            rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            line = text[: m.start()].count("\n") + 1

            if aid in index:
                key, surnames, bib_year = index[aid]
                if not (folded & surnames):
                    problems.append(
                        f"{rel}:{line}: arXiv {aid} is written as "
                        f"'{written} {m.group('year')}' but paper.bib entry "
                        f"'{key}' has authors {sorted(surnames)}"
                        + (f" ({bib_year})" if bib_year else "")
                    )
                continue

            # ID unknown to paper.bib. A single shared surname proves nothing
            # — Zhang, Li and Chen each author dozens of unrelated papers, and
            # the bibliography simply not having the newer one is the normal
            # case, not a defect. Fire only when the *whole written team* sits
            # inside one bib entry that carries a different ID: the same
            # people credited with an ID the bibliography never gives them is
            # the fingerprint of an ID recalled from memory.
            for key, surnames, _yr in index.values():
                if folded <= surnames:
                    problems.append(
                        f"{rel}:{line}: arXiv {aid} is written as "
                        f"'{written} {m.group('year')}', but paper.bib lists "
                        f"exactly these authors under '{key}' with "
                        f"{sorted(known_ids[next(iter(folded))] & _ids_of(index, key))}"
                        f" — not {aid}"
                    )
                    break
    return problems


def _ids_of(index, key: str) -> set[str]:
    return {aid for aid, (k, _s, _y) in index.items() if k == key}


def test_no_arxiv_id_is_attributed_to_authors_paper_bib_disagrees_with():
    problems = _findings()
    assert not problems, (
        "citation drift — an arXiv ID is written next to authors that "
        "paper.bib does not list for it. paper.bib is the verified source of "
        "truth (§10); fix the prose, not the bib, unless you have re-verified "
        "the bib entry against two independent sources.\n\n" + "\n".join(problems)
    )


@pytest.mark.parametrize(
    "text, should_fire",
    [
        # The three real regressions this test exists for.
        ("LLM-assisted DAG proposal (Kiciman-Sharma 2025, arXiv 2402.11068).", True),
        ("names + domain description (Kiciman-Sharma 2025, arXiv 2402.11068).", True),
        # The corrected forms must pass.
        ("LLM-assisted DAG proposal (Kıcıman et al. 2023, arXiv 2305.00050).", False),
        ("Distributional IV (Holovchak et al. 2025, arXiv 2502.07641).", False),
        # Diacritic- and order-insensitive: folded surnames must match.
        ("(Kiciman 2023, arXiv 2305.00050)", False),
        ("(Sharma and Tan 2023, arXiv 2305.00050)", False),
        # Wrong surname on a real ID is the whole point.
        ("(Pearl 2023, arXiv 2305.00050)", True),
        # IDs paper.bib does not carry are left to the network auditor.
        ("(Nobody 2099, arXiv 9999.99999)", False),
    ],
)
def test_detector_fires_on_the_regressions_and_not_on_the_fixes(
    tmp_path, monkeypatch, text, should_fire
):
    """Prove the guard discriminates, rather than merely passing today.

    A drift detector that has never been shown to fire is indistinguishable
    from one whose regex silently stopped matching.
    """
    scratch = tmp_path / "src"
    scratch.mkdir()
    (scratch / "probe.py").write_text(f'"""{text}"""\n', encoding="utf-8")
    monkeypatch.setattr("tests.test_citation_attribution_drift.SCAN_ROOTS", (scratch,))
    assert bool(_findings()) is should_fire


def test_paper_bib_kiciman_entry_lists_all_four_authors():
    """The June 2026 fix said "Kıcıman, Ness & Sharma" — arXiv 2305.00050 has
    four authors. ``paper.bib`` is right; this pins it so a future "tidy-up"
    of the bib cannot quietly drop Chenhao Tan and make the prose right by
    corrupting the reference.

    Verified 2026-08-01 against arxiv.org/abs/2305.00050 and OpenAlex
    (doi:10.48550/arXiv.2305.00050): Kıcıman, Ness, Sharma, Tan (2023).
    """
    _, surnames, year = _bib_index()["2305.00050"]
    assert year == "2023"
    assert {_fold(s) for s in ("Kıcıman", "Ness", "Sharma", "Tan")} <= surnames
