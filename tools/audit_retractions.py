#!/usr/bin/env python3
"""Retraction sweep for ``paper.bib`` — the §10 "zero-hallucination" red
line extended to *retracted* citations.

A fabricated citation and a citation to a **retracted** paper are the same
class of failure: a reader who clicks through loses trust in every number
StatsPAI reports. CrossRef-verified authors/years/titles (handled by
``tools/audit_citations.py``) do not catch a paper that was real but has
since been withdrawn, so this auditor closes that gap.

Mechanism
---------
Every ``doi = {...}`` in ``paper.bib`` is resolved against OpenAlex, whose
per-work ``is_retracted`` boolean is sourced from the Retraction Watch
database (Crossref Labs / OpenAlex integration). A ``True`` is a hard §10
failure; an unreachable / unknown DOI is a *soft* failure (network or
coverage gap), never silently treated as clean.

This module deliberately reuses the network + cache + back-off layer of
``audit_citations`` and the bibtex parser of ``audit_bib_duplicates`` so
there is one HTTP policy and one bib parser in the tree, not three.

Exit-code contract (mirrors ``tools/audit_citations.py``):
  0 — clean: every resolvable DOI is not retracted.
  1 — at least one cited DOI is retracted (a real §10 failure) — or, under
      ``--strict``, at least one DOI was genuinely unresolvable.
  2 — soft failure: the only problems were transient network / rate-limit
      errors (OpenAlex unreachable). No retraction detected, so it must not
      block a merge; surfaced as a warning.

Usage::

    python tools/audit_retractions.py                 # full sweep
    python tools/audit_retractions.py --refresh        # ignore disk cache
    python tools/audit_retractions.py --out report.md  # write markdown
    python tools/audit_retractions.py --strict         # unresolved -> exit 1
    python tools/audit_retractions.py --limit 20       # first N (debugging)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# tools/ is not a package; when run as ``python tools/audit_retractions.py``
# the script directory is sys.path[0], so these resolve as top-level modules
# (the same pattern tests/test_audit_*.py use).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_bib_duplicates as abd  # noqa: E402
import audit_citations as ac  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_BIB = REPO_ROOT / "paper.bib"
OPENALEX_MAILTO = "brycew6m@stanford.edu"

# Status tokens for one DOI.
OK = "ok"
RETRACTED = "retracted"
UNRESOLVED = "unresolved"


@dataclass
class RetractionRow:
    """One paper.bib DOI and its retraction verdict."""

    key: str
    doi: str
    status: str
    title: str = ""
    detail: str = ""


def extract_doi_entries(bib_text: str) -> list[tuple[str, str]]:
    """Return ``[(bibkey, doi), ...]`` for every entry that carries a DOI.

    Reuses ``audit_bib_duplicates.parse_bib`` so the DOI-field regex and
    normalisation (lower-cased, trailing-dot-stripped) live in exactly one
    place.
    """
    out: list[tuple[str, str]] = []
    for entry in abd.parse_bib(bib_text):
        if entry.doi:
            out.append((entry.key, entry.doi))
    return out


def classify_openalex(payload: Any) -> Optional[bool]:
    """Pure classifier: given a parsed OpenAlex work object, return its
    ``is_retracted`` boolean, or ``None`` when the field is absent / the
    payload is not a usable work record.

    Kept side-effect free so the retraction verdict can be unit-tested
    against fixed payloads without any network access.
    """
    if not isinstance(payload, dict):
        return None
    value = payload.get("is_retracted")
    if isinstance(value, bool):
        return value
    return None


def _openalex_url(doi: str) -> str:
    # OpenAlex resolves a DOI directly via the ``doi:`` selector; the polite
    # pool wants a mailto. quote the DOI so slashes / odd chars are safe.
    safe = urllib.parse.quote(doi, safe="")
    return (
        f"https://api.openalex.org/works/doi:{safe}"
        f"?mailto={urllib.parse.quote(OPENALEX_MAILTO)}"
        f"&select=id,doi,title,is_retracted"
    )


def retraction_status(
    doi: str, *, refresh: bool = False, sleep: float = 0.0
) -> tuple[str, str, str]:
    """Resolve one DOI to ``(status, title, detail)``.

    ``status`` is one of ``OK`` / ``RETRACTED`` / ``UNRESOLVED``. Network or
    rate-limit failures and DOIs absent from OpenAlex map to ``UNRESOLVED``
    (a soft signal), never to ``OK`` — an unknown citation is not a verified
    one.
    """
    url = _openalex_url(doi)
    try:
        raw = ac._http_get(url, refresh=refresh, sleep=sleep)
    except urllib.error.HTTPError as exc:  # 404 = not in OpenAlex
        return UNRESOLVED, "", f"HTTP {exc.code}"
    except ac._TRANSIENT_NETWORK_ERRORS as exc:
        return UNRESOLVED, "", f"network: {type(exc).__name__}"
    except Exception as exc:  # pragma: no cover - defensive
        return UNRESOLVED, "", f"error: {type(exc).__name__}"

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return UNRESOLVED, "", "bad JSON"

    is_retracted = classify_openalex(payload)
    title = (payload.get("title") or "") if isinstance(payload, dict) else ""
    if is_retracted is None:
        return UNRESOLVED, title, "no is_retracted field"
    return (RETRACTED if is_retracted else OK), title, ""


def audit(
    entries: Iterable[tuple[str, str]],
    *,
    refresh: bool = False,
    sleep: float = 0.0,
) -> list[RetractionRow]:
    rows: list[RetractionRow] = []
    for key, doi in entries:
        status, title, detail = retraction_status(doi, refresh=refresh, sleep=sleep)
        rows.append(RetractionRow(key=key, doi=doi, status=status, title=title, detail=detail))
    return rows


def summarize(rows: list[RetractionRow]) -> dict[str, int]:
    counts = {OK: 0, RETRACTED: 0, UNRESOLVED: 0}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def format_report(rows: list[RetractionRow]) -> str:
    counts = summarize(rows)
    lines = ["# paper.bib retraction sweep", ""]
    lines.append(
        f"Checked **{len(rows)}** DOIs via OpenAlex `is_retracted`: "
        f"{counts[OK]} clean, {counts[RETRACTED]} retracted, "
        f"{counts[UNRESOLVED]} unresolved."
    )
    lines.append("")
    retracted = [r for r in rows if r.status == RETRACTED]
    if retracted:
        lines.append("## ⚠️ Retracted citations (hard §10 failure)")
        lines.append("")
        for r in retracted:
            lines.append(f"- `@{r.key}` — {r.doi} — {r.title}")
        lines.append("")
    unresolved = [r for r in rows if r.status == UNRESOLVED]
    if unresolved:
        lines.append("## Unresolved (network / not in OpenAlex — soft)")
        lines.append("")
        for r in unresolved:
            lines.append(f"- `@{r.key}` — {r.doi} — {r.detail}")
        lines.append("")
    if not retracted and not unresolved:
        lines.append("All cited DOIs resolved and none are retracted. ✅")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", default=str(PAPER_BIB), help="path to paper.bib")
    parser.add_argument("--out", default=None, help="write the markdown report here")
    parser.add_argument("--refresh", action="store_true", help="ignore disk cache")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also exit non-zero when any DOI is unresolved",
    )
    parser.add_argument("--limit", type=int, default=None, help="check only the first N DOIs")
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="seconds to sleep between requests"
    )
    args = parser.parse_args(argv)

    bib_text = Path(args.bib).read_text(encoding="utf-8")
    entries = extract_doi_entries(bib_text)
    if args.limit is not None:
        entries = entries[: args.limit]

    rows = audit(entries, refresh=args.refresh, sleep=args.sleep)
    report = format_report(rows)
    counts = summarize(rows)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    print(report)

    if counts[RETRACTED] > 0:
        return 1
    if counts[UNRESOLVED] > 0:
        if args.strict:
            return 1
        # Soft failure: nothing retracted, only unreachable / unknown DOIs.
        print(
            f"::warning::retraction sweep could not resolve "
            f"{counts[UNRESOLVED]} DOI(s) (network / OpenAlex coverage); "
            f"no retraction detected — soft pass.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
