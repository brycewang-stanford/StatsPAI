"""``sp.bibliography`` — CSL style hub + ``paper.bib`` writer.

Quarto's citation system needs two things on disk: a ``.bib`` file
listing the cited works, and a ``.csl`` file specifying the journal's
formatting style. StatsPAI captures citations as free-form strings on
each estimator's ``cite()`` method (``"Callaway B, Sant'Anna PHC.
(2021). Difference-in-Differences..."``), so this module bridges:

1. **CSL URL registry** — short journal names → canonical Zotero/styles
   URLs. We deliberately *do not* bundle ``.csl`` files: zotero/styles
   is CC-BY-SA-3.0, which is incompatible with StatsPAI's MIT license.
   :func:`csl_url` returns the URL so users can ``curl`` it once into
   their project.

2. **Citation string → BibTeX entry** — best-effort regex parse of
   common citation styles (Author Year. Title. Journal.). Captures
   author / year / title and emits a syntactically valid BibTeX entry
   with a stable key. Nothing fancy — for real bibliographies users
   should manage their own ``.bib`` and treat StatsPAI's parse as a
   convenience starter.

3. **``paper.bib`` writer** — ``write_bib(citations, path)`` dedupes
   by computed bib key and writes a clean BibTeX file Quarto can
   resolve.

These three sit between ``sp.replication_pack`` (which now writes a
real ``paper.bib`` instead of a free-text dump) and
``PaperDraft.to_qmd`` (which now accepts ``csl='aer'`` short names).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "CSL_REGISTRY",
    "csl_url",
    "csl_filename",
    "list_csl_styles",
    "parse_citation_to_bib",
    "make_bib_key",
    "citations_to_bib_entries",
    "write_bib",
]


# ---------------------------------------------------------------------------
# CSL URL registry
# ---------------------------------------------------------------------------

#: Short journal name → canonical Zotero/styles URL. Each URL points to
#: the *canonical* CSL file maintained by the Zotero/styles project; users
#: download once with ``curl <URL>.csl > paper-style.csl``. The mapping
#: collapses related journals onto the same style (e.g. all the AEJ
#: family uses the AER style).
CSL_REGISTRY: Dict[str, Dict[str, str]] = {
    "aer": {
        "label": "American Economic Review",
        "filename": "american-economic-association.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/american-economic-association.csl"
        ),
    },
    "aeja": {  # AEJ: Applied — same style as AER
        "label": "AEJ: Applied Economics",
        "filename": "american-economic-association.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/american-economic-association.csl"
        ),
    },
    "aejmac": {
        "label": "AEJ: Macroeconomics",
        "filename": "american-economic-association.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/american-economic-association.csl"
        ),
    },
    "aejmicro": {
        "label": "AEJ: Microeconomics",
        "filename": "american-economic-association.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/american-economic-association.csl"
        ),
    },
    "aejpol": {
        "label": "AEJ: Economic Policy",
        "filename": "american-economic-association.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/american-economic-association.csl"
        ),
    },
    "qje": {
        "label": "Quarterly Journal of Economics",
        "filename": "the-quarterly-journal-of-economics.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/the-quarterly-journal-of-economics.csl"
        ),
    },
    "econometrica": {
        "label": "Econometrica",
        "filename": "econometrica.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/econometrica.csl"
        ),
    },
    "restat": {
        "label": "Review of Economics and Statistics",
        "filename": "the-review-of-economics-and-statistics.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/the-review-of-economics-and-statistics.csl"
        ),
    },
    "restud": {
        "label": "Review of Economic Studies",
        "filename": "the-review-of-economic-studies.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/the-review-of-economic-studies.csl"
        ),
    },
    "jpe": {
        "label": "Journal of Political Economy",
        "filename": "the-journal-of-political-economy.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/the-journal-of-political-economy.csl"
        ),
    },
    "jf": {
        "label": "Journal of Finance",
        "filename": "the-journal-of-finance.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/the-journal-of-finance.csl"
        ),
    },
    "chicago-author-date": {
        "label": "Chicago Manual of Style (Author-Date)",
        "filename": "chicago-author-date.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/chicago-author-date.csl"
        ),
    },
    "apa": {
        "label": "American Psychological Association 7th Edition",
        "filename": "apa.csl",
        "url": (
            "https://raw.githubusercontent.com/citation-style-language/"
            "styles/master/apa.csl"
        ),
    },
}


def csl_url(name: str) -> str:
    """Return the canonical Zotero/styles URL for a CSL preset.

    Use the URL once at project setup:

    .. code-block:: bash

        curl -O $(python -c "import statspai as sp; print(sp.csl_url('aer'))")
        # → american-economic-association.csl in the current directory

    Then point Quarto at the local copy via ``csl: paper-style.csl`` in
    the YAML header. ``PaperDraft.to_qmd(csl='aer')`` does the local
    filename pass-through automatically.

    Raises
    ------
    ValueError
        Unknown short name. Use :func:`list_csl_styles` to enumerate.
    """
    key = (name or "").strip().lower()
    if key not in CSL_REGISTRY:
        raise ValueError(
            f"Unknown CSL style {name!r}. "
            f"Available: {', '.join(sorted(CSL_REGISTRY))}."
        )
    return CSL_REGISTRY[key]["url"]


def csl_filename(name: str) -> str:
    """Return the canonical ``.csl`` *filename* (no path) for a preset.

    Useful when emitting a ``csl: ...`` line into a Quarto YAML header
    where the user has already downloaded the style file alongside
    ``paper.qmd``.
    """
    key = (name or "").strip().lower()
    if key in CSL_REGISTRY:
        return CSL_REGISTRY[key]["filename"]
    # If the caller already passed a real .csl filename, hand it back.
    if name and name.endswith(".csl"):
        return name
    raise ValueError(
        f"Unknown CSL style {name!r}. "
        f"Available: {', '.join(sorted(CSL_REGISTRY))}."
    )


def list_csl_styles() -> List[Tuple[str, str]]:
    """List ``(short_name, full_label)`` pairs for every registered style."""
    return [(k, v["label"]) for k, v in CSL_REGISTRY.items()]


# ---------------------------------------------------------------------------
# Citation string -> BibTeX entry
# ---------------------------------------------------------------------------

# Best-effort regexes. We don't ship a CSL parser — citations come in
# many shapes; the goal is to handle the canonical one StatsPAI's
# estimator ``cite()`` returns ("Author A, Author B (YEAR). Title. Journal."),
# and fall back to a generic @misc entry for everything else.
_RE_AUTHOR_YEAR_TITLE_JOURNAL = re.compile(
    # Authors → (optional ()-wrapped) year → punctuation-tolerant gap →
    # title (must start with a letter, ≥1 char) → period → optional
    # journal (must start with a letter when present). Requiring a
    # real-letter title start avoids backtracking traps where stray
    # ")" or "." gets captured as the title.
    r"^\s*(?P<authors>[^()]+?)\s*\(?(?P<year>(19|20)\d{2})\)?[\s.,;:)]*"
    r"(?P<title>[A-Za-z][^.]*)\.\s*"
    r"(?:(?P<journal>[A-Za-z][^.]*)\.?)?",
)
_RE_AUTHOR_YEAR_LOOSE = re.compile(
    r"(?P<authors>.+?)\s*[\(]?(?P<year>(19|20)\d{2})[\)]?",
)


def _slug(s: str, max_len: int = 24) -> str:
    """Lowercase ASCII-ish slug for bib keys."""
    out = re.sub(r"[^A-Za-z0-9]+", "", s.lower())
    return out[:max_len] or "anon"


def _first_author_lastname(authors: str) -> str:
    """Best-effort: extract the first author's last name from a string.

    Handles "Smith J, Doe K" / "John Smith" / "Smith, John".
    """
    if not authors:
        return "anon"
    chunk = authors.split(",")[0].strip()
    chunk = re.split(r"\s+and\s+|;", chunk, maxsplit=1)[0].strip()
    if not chunk:
        return "anon"
    parts = chunk.split()
    if len(parts) == 1:
        return _slug(parts[0])
    # "Smith J" — last name first; "John Smith" — last name last.
    # Heuristic: if the last token is single-letter-ish (initial), the
    # first token is the surname; otherwise the last token is.
    if len(parts[-1]) <= 2:
        return _slug(parts[0])
    return _slug(parts[-1])


def make_bib_key(citation: str) -> str:
    """Compute a stable BibTeX key from a free-form citation string.

    Format: ``firstauthor + year + first-title-word``, e.g.
    ``"callaway2021difference"``. Falls back to a hash-derived key when
    we can't parse author+year.
    """
    if not citation:
        return "anon"
    m = _RE_AUTHOR_YEAR_TITLE_JOURNAL.match(citation)
    if m:
        author = _first_author_lastname(m.group("authors"))
        year = m.group("year")
        title_first = _slug(m.group("title").split()[0]) if m.group("title") else ""
        return f"{author}{year}{title_first}".rstrip()
    m = _RE_AUTHOR_YEAR_LOOSE.search(citation)
    if m:
        author = _first_author_lastname(m.group("authors"))
        year = m.group("year")
        return f"{author}{year}"
    # Last resort: short slug of the whole thing.
    return _slug(citation, max_len=20) or "anon"


def parse_citation_to_bib(
    citation: str,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a citation string into a BibTeX-shaped dict.

    Returns a dict with at least ``key``, ``type`` (``article`` /
    ``misc``), and as many of (``author``, ``year``, ``title``,
    ``journal``) as the regex can extract.

    For full-fidelity bibliographies, write your ``paper.bib`` by
    hand or via Zotero — this is a "quick-start" parser, deliberately
    conservative.
    """
    if not citation or not citation.strip():
        return {"key": key or "anon", "type": "misc",
                "fields": {"note": "(empty citation)"}}

    cleaned = citation.strip().rstrip(",;")
    bib_key = key or make_bib_key(cleaned)
    fields: Dict[str, str] = {}
    bib_type = "misc"

    m = _RE_AUTHOR_YEAR_TITLE_JOURNAL.match(cleaned)
    if m:
        fields["author"] = m.group("authors").strip()
        fields["year"] = m.group("year")
        fields["title"] = m.group("title").strip()
        journal = m.group("journal")
        if journal:
            fields["journal"] = journal.strip()
            bib_type = "article"
        else:
            # Title without journal — book / working paper / report.
            bib_type = "misc"
    else:
        m = _RE_AUTHOR_YEAR_LOOSE.search(cleaned)
        if m:
            fields["author"] = m.group("authors").strip()
            fields["year"] = m.group("year")
        # Fallback — keep the original string as a note so it doesn't
        # vanish from the bib file.
        fields.setdefault("note", cleaned)

    return {"key": bib_key, "type": bib_type, "fields": fields}


def _format_bib_entry(entry: Dict[str, Any]) -> str:
    key = entry.get("key") or "anon"
    bib_type = entry.get("type") or "misc"
    fields = entry.get("fields") or {}
    if not fields:
        return f"@{bib_type}{{{key}}}\n"
    field_lines = []
    for k, v in fields.items():
        if v is None:
            continue
        # BibTeX needs braces for value content; escape inner braces.
        s = str(v).replace("{", r"\{").replace("}", r"\}")
        field_lines.append(f"  {k} = {{{s}}},")
    body = "\n".join(field_lines).rstrip(",")
    return f"@{bib_type}{{{key},\n{body}\n}}\n"


def citations_to_bib_entries(
    citations: Iterable[str],
) -> List[Dict[str, Any]]:
    """Parse a sequence of citation strings into BibTeX-entry dicts.

    Deduplicates by key — the *first* occurrence wins (matches the
    ``replication_pack`` semantics where inner estimators register
    citations before outer wrappers).
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for c in citations:
        if c is None:
            continue
        s = str(c).strip()
        if not s:
            continue
        entry = parse_citation_to_bib(s)
        seen.setdefault(entry["key"], entry)
    return list(seen.values())


def write_bib(
    citations: Iterable[Union[str, Dict[str, Any]]],
    path: Union[str, Path],
    *,
    append: bool = False,
    header: bool = True,
) -> Path:
    """Write a clean BibTeX file from citation strings or entry dicts.

    Parameters
    ----------
    citations : iterable
        Either free-form citation strings (parsed via
        :func:`parse_citation_to_bib`) or pre-built entry dicts
        ``{"key": ..., "type": ..., "fields": {...}}``.
    path : str or Path
        Destination ``.bib`` file. Parent dirs are created.
    append : bool, default False
        Append to an existing file rather than overwriting.
    header : bool, default True
        Prepend a one-line ``% Auto-generated by StatsPAI ...`` comment
        at the top of fresh files (skipped on append).

    Returns
    -------
    Path
        Resolved path of the written file.

    Notes
    -----
    Deduplicates by computed bib key. Pre-built entry dicts are taken
    as-is; only string citations go through the regex parser.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Normalise input to entry dicts.
    entries: List[Dict[str, Any]] = []
    seen_keys = set()
    for c in citations:
        if isinstance(c, dict) and "key" in c:
            if c["key"] in seen_keys:
                continue
            entries.append(c)
            seen_keys.add(c["key"])
        elif c is not None:
            for entry in citations_to_bib_entries([c]):
                if entry["key"] in seen_keys:
                    continue
                entries.append(entry)
                seen_keys.add(entry["key"])

    body = "\n".join(_format_bib_entry(e) for e in entries)

    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as fh:
        if header and not append:
            try:
                from .. import __version__ as _v
            except Exception:
                _v = "unknown"
            fh.write(
                f"% paper.bib — auto-generated by StatsPAI v{_v}\n"
                f"% {len(entries)} entries\n\n"
            )
        fh.write(body)

    return out
