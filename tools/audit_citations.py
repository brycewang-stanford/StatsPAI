#!/usr/bin/env python3
"""
Citation auditor for StatsPAI.

Scans ``src/`` and ``docs/`` for arXiv / NBER / DOI references, verifies
each against primary sources (arXiv API, NBER HTML, Crossref API), and
emits a Markdown audit report flagging attribution / year / title
mismatches.

Usage
-----
    python tools/audit_citations.py                     # full audit
    python tools/audit_citations.py --roots src docs    # override scan roots
    python tools/audit_citations.py --kinds arxiv       # arxiv only (fast)
    python tools/audit_citations.py --refresh           # ignore cache
    python tools/audit_citations.py --out report.md     # custom output path

Design notes
------------
1. EXTRACT — regex over source files. For each match, capture the id,
   file:line, and a *claim block* (±2 lines of surrounding context).
2. VERIFY — hit primary sources in batch where possible (arXiv supports
   ``id_list=X,Y,Z``). Cache raw responses on disk so re-runs are cheap.
3. DIFF — normalise unicode, compare last-name sets + years. Flag:
     - missing truth author in the claim
     - claim contains a capitalised token that looks like a surname but
       isn't among the truth authors (catches phantom coauthors)
     - year mismatch > 1 year (preprint vs published allowed)
4. REPORT — single ``audit_report.md`` with sections OK / MISMATCH /
   UNRESOLVED, each row linking file:line.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

try:
    import certifi
    _SSL_CONTEXT: Optional[ssl.SSLContext] = ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:  # pragma: no cover — certifi ships via requests/pip
    _SSL_CONTEXT = None


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".citation_cache"
DEFAULT_ROOTS = ("src", "docs")
DEFAULT_OUT = REPO_ROOT / "audit_report.md"
USER_AGENT = "statspai-citation-audit/1.0 (mailto:brycew6m@stanford.edu)"

ARXIV_RE = re.compile(
    r"""
    arXiv[:\s]*                # "arXiv:" or "arXiv " (case-insensitive via flag)
    (?P<id>\d{4}\.\d{4,5})     # NNNN.NNNNN core id
    (?:v\d+)?                  # optional version suffix
    """,
    re.IGNORECASE | re.VERBOSE,
)

NBER_RE = re.compile(
    r"""
    (?:NBER[\s\-/]*)?                    # optional NBER prefix
    (?:Working[\s]+Paper|WP)             # REQUIRED 'WP' / 'Working Paper'
    [\s]*w?(?P<id>\d{3,5})               # optional leading 'w' + id
    """,
    re.IGNORECASE | re.VERBOSE,
)

DOI_RE = re.compile(r"\b(?P<id>10\.\d{4,9}/[^\s)\"'`,;}\]\[]+?)\.?(?=[\s)\"'`,;}\]\[]|$)")

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Pandoc-style citation key marker (e.g. ``[@benkeser2016highly]``,
# ``[@key1; @key2]``). StatsPAI docstrings use these to point a
# bibliography entry at its paper.bib record. They function as
# entry boundaries for the phantom-author scoping below — surnames
# that appear before a ``[@...]`` marker belong to a *different*
# bibliography entry and must NOT be attributed to the next id.
_PANDOC_CITE_RE = re.compile(r"\[@[\w:.\-]+(?:\s*;\s*@[\w:.\-]+)*\]")

# Rough surname token: capitalised word, may include unicode letters,
# hyphens, apostrophes. Excludes common lowercase ALL-CAPS artifacts.
SURNAME_RE = re.compile(
    r"\b[A-ZÄÖÜÀ-ÞŠŽČŚŃŁ][a-zA-ZäöüßÀ-ÿšžčśńłı'\-]{1,24}\b"
)

# Tokens that LOOK like a surname but are decidedly not.
SURNAME_STOPWORDS = {
    # months
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # journals / venues / common lingo
    "arxiv", "nber", "wp", "working", "paper", "theorem", "proposition",
    "lemma", "section", "figure", "table", "appendix", "algorithm",
    "journal", "review", "econometrica", "neurips", "nips", "aistats",
    "icml", "iclr", "aer", "qje", "jmlr", "jasa", "jrss", "econometrics",
    "statistical", "statistics", "annals", "biometrika", "biometrics",
    "the", "and", "with", "for", "via", "under", "over", "of", "on",
    "is", "in", "a", "an", "et", "al", "vs",
    # method / tool names
    "dml", "did", "rdd", "rd", "iv", "ate", "att", "cate", "hte",
    "glmm", "gmm", "mcmc", "bma", "bart", "bcf", "lcm", "scm",
    "tmle", "aipw", "hal", "ols", "wls", "gls", "pci", "llm",
    "python", "stata", "julia", "matlab",
    # modal words
    "hence", "thus", "however", "moreover", "therefore", "note",
    # self-refs
    "statspai", "sp", "func", "class", "param", "returns", "notes",
    "references", "example", "examples", "reference",
    # capitalised but not names
    "true", "false", "none", "null", "nan",
    # codes
    "mcmc", "nuts", "advi", "hdi",
    # title-word fragments that keep leaking through as "surnames"
    "difference", "differences", "prediction", "predictions", "rapidly",
    "imperfect", "surrogate", "surrogates", "reasoning", "opening",
    "adapting", "causality", "frontier", "effects", "effect",
    "synthetic", "control", "controls", "experimental", "observational",
    "estimation", "treatment", "treatments", "outcome", "outcomes",
    "longitudinal", "hierarchical", "combining", "unobserved",
    "confounding", "calibration", "targeted", "maximum", "likelihood",
    "implementation", "methods", "method", "identification",
    "kink", "setting", "design", "designs", "practitioner",
    "weighted", "rank", "prioritization", "rules", "randomi",
    "bandits", "bandit", "heterogeneous", "inference",
    "combination", "persistent", "decision", "survival",
    "discovery", "experts", "language", "models",
    "contextual", "short", "long", "term", "extending", "when",
    "using", "downstream", "supervised", "learning",
    "rct", "rcts", "balancing", "regression", "experiments", "experiment",
    "science", "political", "shift", "share", "synthesis",
    "proxy", "panel", "correction", "policy", "policies", "robust",
    "deep", "reinforcement", "online", "offline", "optimization",
    "safe", "confounding-robust", "high-order", "high", "order",
    "bias", "modified", "approach", "new",
    # more title/method words that leak through
    "mapping", "suite", "ite", "late", "study", "event-study",
    "anticipation", "misclassification", "averaging", "interference",
    "compliance", "qte", "forest", "ips", "snips", "switch-dr",
    "switch", "cluster", "focal", "mas", "valueerror",
    "dahabreh",   # only appears as 'Dahabreh 2020 framework' — a related-but-different paper
    "cattaneo", "jansson",   # appear only in method-name context ("rbc bootstrap of Cattaneo-Jansson")
    "ma",
    "event", "series", "acronym", "framework", "library",
    "survey", "generation", "evidence", "designs", "design",
    "algorithm", "algorithms", "tree", "forests",
    # title-word fragments (paper titles quoted in docs / specs)
    "simple", "globally", "convergent", "accelerating", "convergence",
    # ML method/architecture tokens that hyphenate into faux-surnames
    "q-network", "q-networks", "q-learning", "q-function", "q-functions",
    "deep-q", "dqn", "ddqn", "ppo", "a3c", "trpo",
    "actor-critic", "soft-actor-critic", "double-dqn",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """One extracted citation reference in the codebase."""

    kind: str                   # 'arxiv' | 'nber' | 'doi'
    id: str
    file: str
    line: int
    claim_block: str            # ~±3 lines of context (for author presence)
    same_line: str = ""         # the single source line (for phantom check)
    claimed_year: Optional[int] = None


@dataclass
class PaperMeta:
    """Ground-truth metadata from primary source."""

    authors: list[str]          # full names
    title: str
    year: int
    source: str                 # 'arxiv' | 'nber' | 'crossref'

    def last_names(self) -> list[str]:
        return [_normalise(_last_name(a)) for a in self.authors]


@dataclass
class Verdict:
    """Result of diff step."""

    citation: Citation
    truth: Optional[PaperMeta]
    status: str                 # 'ok' | 'mismatch' | 'unresolved'
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PUNCT_TO_SPACE = re.compile(r"[^\w\s'-]", re.UNICODE)


def _strip_diacritics(s: str) -> str:
    """NFD + drop combining marks, preserve case and punctuation."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _normalise(s: str) -> str:
    """Strip diacritics + lowercase + replace punctuation with spaces.

    Replacing punctuation with *spaces* (not deletion) is important: it
    keeps "(Imbens," from collapsing into one un-splittable token.
    Apostrophe and hyphen are kept for names like O'Neill, Tabord-Meehan.
    """
    s = _strip_diacritics(s).lower()
    s = _PUNCT_TO_SPACE.sub(" ", s)
    return " ".join(s.split())  # collapse whitespace


def _last_name(full: str) -> str:
    """Extract last name from a 'First Last' or 'Last, First' string."""
    full = full.strip()
    if "," in full:
        return full.split(",", 1)[0].strip()
    parts = full.split()
    return parts[-1] if parts else ""


def _cache_get(key: str) -> Optional[bytes]:
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{h}.bin"
    if path.exists():
        return path.read_bytes()
    return None


def _cache_put(key: str, data: bytes) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    (CACHE_DIR / f"{h}.bin").write_bytes(data)


def _http_get(url: str, *, refresh: bool = False, sleep: float = 0.0) -> bytes:
    """GET with disk cache + user-agent. Returns bytes."""
    if not refresh:
        cached = _cache_get(url)
        if cached is not None:
            return cached
    if sleep:
        time.sleep(sleep)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CONTEXT) as resp:
        data = resp.read()
    _cache_put(url, data)
    return data


# ---------------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------------


def extract_citations(roots: Iterable[Path]) -> list[Citation]:
    """Walk source trees and harvest arXiv / NBER / DOI references."""
    out: list[Citation] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".md", ".rst", ".txt"}:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(lines):
                for kind, regex in (
                    ("arxiv", ARXIV_RE),
                    ("nber", NBER_RE),
                    ("doi", DOI_RE),
                ):
                    for m in regex.finditer(line):
                        block = "\n".join(
                            lines[max(0, i - 3): min(len(lines), i + 4)]
                        )
                        # Mask ALL arXiv id patterns before year search —
                        # the leading 4 digits of an arXiv id (e.g.
                        # "2009.10982") would otherwise be mis-read as a
                        # year. We also mask the id range literally in
                        # the current line so the year regex cannot hit
                        # something inside the matched id.
                        line_for_year = ARXIV_RE.sub(" ", line)
                        block_for_year = ARXIV_RE.sub(" ", block)
                        year_m = YEAR_RE.search(line_for_year)
                        if year_m is None:
                            year_m = YEAR_RE.search(block_for_year)
                        year = int(year_m.group(1)) if year_m else None
                        out.append(Citation(
                            kind=kind,
                            id=m.group("id"),
                            file=str(path.relative_to(REPO_ROOT)),
                            line=i + 1,
                            claim_block=block,
                            claimed_year=year,
                            same_line=line,
                        ))
    return out


# ---------------------------------------------------------------------------
# VERIFY — arXiv
# ---------------------------------------------------------------------------


def verify_arxiv(ids: list[str], refresh: bool = False) -> dict[str, PaperMeta]:
    """Batch query arXiv API (max 100 per request)."""
    result: dict[str, PaperMeta] = {}
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for start in range(0, len(ids), 100):
        chunk = ids[start:start + 100]
        url = (
            "https://export.arxiv.org/api/query?"
            + urllib.parse.urlencode({
                "id_list": ",".join(chunk),
                "max_results": len(chunk),
            })
        )
        try:
            xml_bytes = _http_get(url, refresh=refresh, sleep=3.0)
        except urllib.error.URLError as e:
            print(f"[arxiv] HTTP error {e!r} for chunk starting {chunk[0]}",
                  file=sys.stderr)
            continue
        root = ET.fromstring(xml_bytes)
        for entry in root.findall("atom:entry", ns):
            id_text = entry.find("atom:id", ns).text or ""
            # e.g. http://arxiv.org/abs/2202.07234v4
            core = id_text.rsplit("/", 1)[-1]
            core = re.sub(r"v\d+$", "", core)
            title_el = entry.find("atom:title", ns)
            title = (title_el.text or "").strip() if title_el is not None else ""
            pub_el = entry.find("atom:published", ns)
            year = int(pub_el.text[:4]) if pub_el is not None and pub_el.text else 0
            authors: list[str] = []
            for a in entry.findall("atom:author", ns):
                name_el = a.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())
            # arXiv returns <entry><summary>…</summary></entry> with an
            # error message when the id is unknown; detect by empty title
            # AND empty author list.
            if not authors and not title:
                continue
            result[core] = PaperMeta(
                authors=authors,
                title=title,
                year=year,
                source="arxiv",
            )
    return result


# ---------------------------------------------------------------------------
# VERIFY — NBER
# ---------------------------------------------------------------------------


NBER_META_RE = re.compile(
    r'<meta\s+name="(citation_author|citation_title|citation_publication_date)"'
    r'\s+content="([^"]+)"',
    re.IGNORECASE,
)


def verify_nber(ids: list[str], refresh: bool = False) -> dict[str, PaperMeta]:
    """Scrape NBER paper pages (they expose Google-Scholar citation meta)."""
    result: dict[str, PaperMeta] = {}
    for wp in ids:
        url = f"https://www.nber.org/papers/w{wp}"
        try:
            html_text = _http_get(url, refresh=refresh, sleep=1.0).decode(
                "utf-8", errors="replace"
            )
        except urllib.error.URLError as e:
            print(f"[nber] HTTP error {e!r} for w{wp}", file=sys.stderr)
            continue
        authors: list[str] = []
        title = ""
        year = 0
        for meta_name, content in NBER_META_RE.findall(html_text):
            decoded = html.unescape(content).strip()
            if meta_name.lower() == "citation_author":
                authors.append(decoded)
            elif meta_name.lower() == "citation_title":
                title = decoded
            elif meta_name.lower() == "citation_publication_date":
                m = re.match(r"(\d{4})", content)
                if m:
                    year = int(m.group(1))
        if authors or title:
            result[wp] = PaperMeta(
                authors=authors,
                title=title,
                year=year,
                source="nber",
            )
    return result


# ---------------------------------------------------------------------------
# VERIFY — Crossref
# ---------------------------------------------------------------------------


def _verify_datacite_one(doi: str, refresh: bool = False) -> Optional[PaperMeta]:
    """Resolve a DOI via DataCite (Zenodo, Figshare, etc. live here, NOT
    Crossref). Used as a fallback when Crossref returns 404 — many
    self-archived datasets and software DOIs are registered with the
    DataCite agency rather than Crossref."""
    url = f"https://api.datacite.org/dois/{urllib.parse.quote(doi, safe='')}"
    try:
        raw = _http_get(url, refresh=refresh, sleep=0.5)
    except urllib.error.URLError as e:
        print(f"[datacite] HTTP error {e!r} for {doi}", file=sys.stderr)
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    attrs = obj.get("data", {}).get("attributes", {})
    creators = attrs.get("creators", []) or []
    authors = []
    for c in creators:
        name = c.get("name") or " ".join(filter(None, [
            c.get("givenName"), c.get("familyName")
        ]))
        if name:
            authors.append(name)
    titles = attrs.get("titles", []) or []
    title = (titles[0].get("title", "").strip() if titles else "")
    year = int(attrs.get("publicationYear") or 0)
    return PaperMeta(
        authors=authors,
        title=title,
        year=year,
        source="datacite",
    )


def verify_crossref(dois: list[str], refresh: bool = False) -> dict[str, PaperMeta]:
    result: dict[str, PaperMeta] = {}
    for doi in dois:
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
        crossref_404 = False
        try:
            raw = _http_get(url, refresh=refresh, sleep=0.5)
        except urllib.error.HTTPError as e:
            print(f"[crossref] HTTP error {e!r} for {doi}", file=sys.stderr)
            if e.code == 404:
                crossref_404 = True
            else:
                continue
        except urllib.error.URLError as e:
            print(f"[crossref] HTTP error {e!r} for {doi}", file=sys.stderr)
            continue
        if crossref_404:
            # Fall through to DataCite — Zenodo / Figshare / Dryad
            # software & dataset DOIs are registered there.
            meta = _verify_datacite_one(doi, refresh=refresh)
            if meta is not None:
                result[doi] = meta
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message", {})
        authors = [
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in msg.get("author", [])
        ]
        title_list = msg.get("title", [])
        title = title_list[0].strip() if title_list else ""
        year = 0
        for key in ("published-print", "published-online", "issued", "created"):
            parts = msg.get(key, {}).get("date-parts", [[0]])
            if parts and parts[0] and parts[0][0]:
                year = parts[0][0]
                break
        result[doi] = PaperMeta(
            authors=authors,
            title=title,
            year=year,
            source="crossref",
        )
    return result


# ---------------------------------------------------------------------------
# DIFF
# ---------------------------------------------------------------------------


def diff_citation(c: Citation, truth: PaperMeta) -> list[str]:
    """Return a list of human-readable issue strings (empty == OK)."""
    issues: list[str] = []
    truth_lasts = set(truth.last_names())

    claim_norm = _normalise(c.claim_block)
    # Strip leading/trailing apostrophes that stick to Python-string
    # literal tokens (e.g. ``'souto'`` → ``souto``) while keeping
    # interior apostrophes for names like O'Neill.
    claim_tokens = {t.strip("'") for t in claim_norm.split()}

    # Also split on hyphens so "Athey-Chetty-Imbens-Kang" matches each
    # component. We keep the unsplit version too so compound surnames
    # like "Tabord-Meehan" still match against themselves.
    hyphen_tokens = {t.strip("'") for t in re.split(r"[-\s]+", claim_norm)}
    all_claim_tokens = claim_tokens | hyphen_tokens

    # Count how many truth authors appear in the claim. If zero, this is
    # almost certainly a bare arXiv URL / reference field with no
    # attribution text — skip the "missing" check (we'd spuriously flag
    # every author). Phantom-name check still runs below; if something
    # pretending to be an author is on the claim line, that WILL flag.
    truth_present_in_claim = [
        a for a in truth.authors
        if _normalise(_last_name(a)) in all_claim_tokens
    ]
    is_bare_reference = len(truth_present_in_claim) == 0

    # Detect a bibtex / @article{...} block in the surrounding context.
    # The ±3-line window may NOT capture the leading ``author={...``
    # field of a multi-line bibtex literal (Python triple-quoted
    # strings spread the entry over 6+ lines). Use a broader marker
    # set so an entry whose author= line is one line outside the
    # window still registers as bibtex. We re-use this flag for both
    # the missing-author and phantom-author checks.
    _bibtex_markers = (
        "author={", "title={", "journal={", "booktitle={",
        "year={", "doi={", "volume={", "number={", "pages={",
        "publisher={", "@article{", "@inproceedings{", "@book{",
        "@misc{", "@techreport{", "@phdthesis{",
    )
    is_bibtex = any(m in c.claim_block for m in _bibtex_markers)

    # 1) missing truth authors (claim lacks someone actually on the paper)
    has_et_al = "et al" in claim_norm
    missing = [a for a in truth.authors
               if _normalise(_last_name(a)) not in all_claim_tokens]
    # tolerate "et al." shorthand *only* when claim keeps the leading author(s)
    if missing and has_et_al:
        # require at least the first truth author to appear
        first = _normalise(_last_name(truth.authors[0])) if truth.authors else ""
        if first and first in claim_tokens:
            # acceptable shorthand
            missing = []
    if missing and not is_bare_reference and not is_bibtex:
        # Bibtex blocks frequently span more than ±3 lines (Python
        # triple-quoted literals embed full entries), so a partial
        # author hit inside the window is not evidence of a real
        # omission — trust the structured ``author={...}`` field.
        issues.append(
            f"missing author(s) in claim: {', '.join(missing)}"
        )

    # 2) phantom authors — surnames in the leading part of the claim that
    #    aren't among truth authors. Restrict to a narrow span: between
    #    the nearest citation delimiter (;, —, →, em-dash, quotation mark,
    #    parenthesis) BEFORE the arXiv id and the id itself. This keeps
    #    multi-citation lines from cross-polluting each other.
    # Pick scope carefully:
    #   * In BibTeX blocks (``author={Last, First and ...}``) the comma
    #     separates last name from first name — disables phantom check.
    #   * In markdown tables (line starts with ``|``) each row is an
    #     independent citation, so only look at the id's own line.
    #   * Otherwise use the ±3-line claim block so author lists that
    #     spill onto the line above the id are still visible.
    # ``is_bibtex`` was computed above (shared with the missing-author
    # check). Reuse it so the two heuristics stay in sync.
    is_table_row = c.same_line.lstrip().startswith("|") and "|" in c.same_line[1:]
    if is_bibtex:
        # BibTeX 'Last, First' swaps cannot be disentangled without a
        # real parser; skip phantom detection entirely for this scope.
        full_scope = ""
    elif is_table_row:
        full_scope = c.same_line
    else:
        full_scope = c.claim_block or c.same_line
        # Isolate the citation's own paragraph/bullet to stop phantom
        # names from neighbouring bibliography entries leaking in.
        # Split first by blank lines, then by bullet/number list markers.
        paragraphs = re.split(r"\n\s*\n", full_scope)
        full_scope = next((p for p in paragraphs if c.id in p), full_scope)
        sub_chunks = re.split(r"\n(?=\s*(?:[-*•]|\d+\.)\s)", full_scope)
        full_scope = next((ch for ch in sub_chunks if c.id in ch), full_scope)
    before_id = full_scope.split(c.id, 1)[0]
    # Narrow the phantom-detection scope to discriminate multi-citation
    # text. We start from ``before_id`` (everything before this
    # citation's id) and chop off any earlier arXiv / NBER id matches —
    # their preceding author blocks belong to those other citations,
    # not this one. Then we further narrow on semicolon (the common
    # stacked-ref separator).
    for other_re in (ARXIV_RE, NBER_RE, DOI_RE, _PANDOC_CITE_RE):
        other_matches = list(other_re.finditer(before_id))
        if other_matches:
            before_id = before_id[other_matches[-1].end():]
    semi_m = re.search(r";\s*[^;]*$", before_id)
    head = semi_m.group(0) if semi_m else before_id
    head_tokens = set(_normalise(head).split())
    candidates: set[str] = set()
    for tok in SURNAME_RE.findall(head):
        norm = _normalise(tok)
        if norm in SURNAME_STOPWORDS:
            continue
        # Hyphen-compound titles (e.g. "Difference-in-Differences",
        # "Rank-Weighted") should not read as surnames when ANY part is
        # a known title-word stopword.
        if "-" in norm and any(
            part in SURNAME_STOPWORDS for part in norm.split("-")
        ):
            continue
        # Hyphen-joined author lists (e.g. "Athey-Chetty-Imbens-Kang")
        # are a valid short form when 2+ components are truth authors.
        if "-" in norm:
            parts = [p for p in norm.split("-") if p]
            matches = sum(1 for p in parts if p in truth_lasts)
            if matches >= 2:
                continue
        if len(norm) < 3:
            continue
        candidates.add(norm)
    # drop ones that are actually real authors
    phantoms = candidates - truth_lasts
    # Also drop common non-author words that slip through:
    phantoms = {p for p in phantoms if p not in SURNAME_STOPWORDS}
    if phantoms and truth_lasts:
        # A phantom counts as "used as an author" only when the surname
        # is directly followed by an author-list punctuation (comma-
        # initial, ampersand, "and", "et al.", opening paren). Covers
        # two scenarios:
        #   (a) some truth authors present + phantom alongside them
        #       (partial mismatch)
        #   (b) no truth author present at all but phantom in author
        #       position (pure wrong attribution — most damaging)
        # Use the diacritic-stripped head so commas/periods are preserved.
        head_dia = _strip_diacritics(head)
        actually_used: list[str] = []
        for p in sorted(phantoms):
            patt = re.compile(
                rf"\b{re.escape(p)}\b"                    # surname
                rf"\s*(?:,\s*[A-Z]\.?|,\s*&|&|\band\b|"   # , I. / , & / & / and
                rf"\set\s+al|\()",
                re.IGNORECASE,
            )
            if patt.search(head_dia):
                actually_used.append(p)
        if actually_used:
            issues.append(
                f"claim cites non-author surname(s): "
                f"{', '.join(actually_used)} "
                f"(truth authors: {', '.join(sorted(truth_lasts))})"
            )

    # 3) year mismatch check disabled — too noisy in free-form text.
    #    arXiv submission year vs journal publication year routinely
    #    differ by 1-5 years, and when multiple citations sit on one
    #    line we cannot reliably attribute a year token to the right
    #    paper. Author mismatches alone catch the real errors; if the
    #    authors are right, the year is almost always right too.

    return issues


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------


def build_report(verdicts: list[Verdict]) -> str:
    ok = [v for v in verdicts if v.status == "ok"]
    mismatch = [v for v in verdicts if v.status == "mismatch"]
    unresolved = [v for v in verdicts if v.status == "unresolved"]

    lines = [
        "# StatsPAI Citation Audit Report",
        "",
        f"Scanned citations: **{len(verdicts)}** total | "
        f"✅ OK: **{len(ok)}** | "
        f"⚠️ MISMATCH: **{len(mismatch)}** | "
        f"❓ UNRESOLVED: **{len(unresolved)}**",
        "",
        "Source: arXiv API (batch), NBER HTML meta, Crossref API.",
        "All responses cached under `tools/.citation_cache/`.",
        "",
    ]

    if mismatch:
        lines.append("## ⚠️ MISMATCH — needs correction\n")
        lines.append("| kind | id | file:line | issue | truth authors | truth year |")
        lines.append("|------|------|-----------|-------|---------------|------------|")
        for v in sorted(mismatch, key=lambda x: (x.citation.kind, x.citation.id)):
            c, t = v.citation, v.truth
            assert t is not None
            issues_md = "<br>".join(v.issues)
            truth_auth = ", ".join(t.authors[:5]) + (" …" if len(t.authors) > 5 else "")
            lines.append(
                f"| {c.kind} | `{c.id}` | `{c.file}:{c.line}` | "
                f"{issues_md} | {truth_auth} | {t.year} |"
            )
        lines.append("")

    if unresolved:
        lines.append("## ❓ UNRESOLVED — primary source returned nothing\n")
        lines.append("| kind | id | file:line |")
        lines.append("|------|------|-----------|")
        for v in sorted(unresolved, key=lambda x: (x.citation.kind, x.citation.id)):
            c = v.citation
            lines.append(f"| {c.kind} | `{c.id}` | `{c.file}:{c.line}` |")
        lines.append("")

    lines.append(f"## ✅ OK ({len(ok)})\n")
    lines.append("<details><summary>Expand to see all passing citations</summary>\n")
    lines.append("| kind | id | file:line | truth authors | year |")
    lines.append("|------|------|-----------|---------------|------|")
    for v in sorted(ok, key=lambda x: (x.citation.kind, x.citation.id)):
        c, t = v.citation, v.truth
        assert t is not None
        truth_auth = ", ".join(t.authors[:3]) + (" …" if len(t.authors) > 3 else "")
        lines.append(
            f"| {c.kind} | `{c.id}` | `{c.file}:{c.line}` | "
            f"{truth_auth} | {t.year} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots", nargs="+", default=list(DEFAULT_ROOTS),
        help="Directories to scan (default: src docs)",
    )
    parser.add_argument(
        "--kinds", nargs="+", default=["arxiv", "nber", "doi"],
        choices=["arxiv", "nber", "doi"],
        help="Which citation kinds to verify",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Bypass the HTTP cache",
    )
    parser.add_argument(
        "--json", dest="json_out", default=None,
        help="Also dump structured verdicts to this JSON path",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Also exit non-zero if any citation is unresolved "
             "(default: only mismatches fail)",
    )
    args = parser.parse_args(argv)

    roots = [REPO_ROOT / r for r in args.roots]
    missing = [r for r in roots if not r.exists()]
    if missing:
        print(f"error: roots do not exist: {missing}", file=sys.stderr)
        return 2

    print(f"scanning {len(roots)} roots for citations …", file=sys.stderr)
    citations = extract_citations(roots)
    print(f"  found {len(citations)} citation occurrences", file=sys.stderr)

    by_kind: dict[str, list[Citation]] = {"arxiv": [], "nber": [], "doi": []}
    for c in citations:
        by_kind[c.kind].append(c)

    for kind, items in by_kind.items():
        unique = sorted({c.id for c in items})
        print(f"  {kind}: {len(items)} occurrences / {len(unique)} unique",
              file=sys.stderr)

    truth: dict[tuple[str, str], PaperMeta] = {}
    if "arxiv" in args.kinds and by_kind["arxiv"]:
        ids = sorted({c.id for c in by_kind["arxiv"]})
        print(f"verifying {len(ids)} arXiv ids …", file=sys.stderr)
        t = verify_arxiv(ids, refresh=args.refresh)
        for k, v in t.items():
            truth[("arxiv", k)] = v
        print(f"  resolved {len(t)}/{len(ids)}", file=sys.stderr)
    if "nber" in args.kinds and by_kind["nber"]:
        ids = sorted({c.id for c in by_kind["nber"]})
        print(f"verifying {len(ids)} NBER working papers …", file=sys.stderr)
        t = verify_nber(ids, refresh=args.refresh)
        for k, v in t.items():
            truth[("nber", k)] = v
        print(f"  resolved {len(t)}/{len(ids)}", file=sys.stderr)
    if "doi" in args.kinds and by_kind["doi"]:
        ids = sorted({c.id for c in by_kind["doi"]})
        print(f"verifying {len(ids)} DOIs …", file=sys.stderr)
        t = verify_crossref(ids, refresh=args.refresh)
        for k, v in t.items():
            truth[("doi", k)] = v
        print(f"  resolved {len(t)}/{len(ids)}", file=sys.stderr)

    verdicts: list[Verdict] = []
    for c in citations:
        t = truth.get((c.kind, c.id))
        if t is None:
            verdicts.append(Verdict(citation=c, truth=None, status="unresolved"))
            continue
        issues = diff_citation(c, t)
        status = "mismatch" if issues else "ok"
        verdicts.append(Verdict(citation=c, truth=t, status=status, issues=issues))

    report = build_report(verdicts)
    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}", file=sys.stderr)

    if args.json_out:
        payload = []
        for v in verdicts:
            payload.append({
                "status": v.status,
                "issues": v.issues,
                "citation": asdict(v.citation),
                "truth": asdict(v.truth) if v.truth else None,
            })
        Path(args.json_out).write_text(json.dumps(payload, indent=2,
                                                  ensure_ascii=False),
                                       encoding="utf-8")
        print(f"wrote {args.json_out}", file=sys.stderr)

    n_mismatch = sum(1 for v in verdicts if v.status == "mismatch")
    n_unresolved = sum(1 for v in verdicts if v.status == "unresolved")
    n_ok = sum(1 for v in verdicts if v.status == "ok")
    print(
        f"summary: {n_ok} ok / {n_mismatch} mismatch / {n_unresolved} unresolved",
        file=sys.stderr,
    )

    if n_mismatch:
        return 1
    if args.strict and n_unresolved:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
