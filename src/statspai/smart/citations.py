"""Multi-format citation utilities for StatsPAI results.

Backs ``result.cite(format=...)`` and the top-level
``sp.bib_for(result)``. The single source of truth for the BibTeX
entries themselves is ``CausalResult._CITATIONS`` — this module
parses and reformats; it never invents bibliographic facts (per
CLAUDE.md §10 zero-hallucination rule).

Supported formats
-----------------

* ``"bibtex"`` — the canonical BibTeX entry as stored on the result
  class (single-citation form; multi-author method families return
  one canonical entry).
* ``"apa"`` — APA-style prose: ``"Author, A., & Author, B. (Year).
  Title. Journal, vol(num), pages."``. Built from the parsed BibTeX
  fields. Fields the source entry does not provide are simply
  omitted — never guessed.
* ``"json"`` — structured ``{key, type, fields: {author, year,
  title, journal, ...}}`` payload for agent consumption.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Union, cast

# ---------------------------------------------------------------------------
#  BibTeX parser (intentionally minimal — handles only the shape
#  used in CausalResult._CITATIONS)
# ---------------------------------------------------------------------------


_ENTRY_HEAD = re.compile(r"@(\w+)\{([^,\s]+)\s*,", re.MULTILINE)


def _parse_bibtex_entries(s: str) -> List[Dict[str, Any]]:
    """Parse one or more BibTeX entries from a single string.

    Some ``CausalResult._CITATIONS`` slots store *two* entries
    concatenated (e.g. ``twfe_decomposition`` cites both Goodman-Bacon
    2021 and de Chaisemartin & D'Haultfœuille 2020). A naive
    single-entry parser would silently drop the second author —
    instead we walk every ``@type{key, ...}`` head in the string and
    parse each independently.
    """
    if not isinstance(s, str) or not s:
        return []
    out: List[Dict[str, Any]] = []
    for match in _ENTRY_HEAD.finditer(s):
        # Find the matching closing brace for this entry's body.
        body_start = match.end()
        depth = 1
        i = body_start
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = s[body_start:i]
        parsed = _parse_entry_body(match.group(1), match.group(2), body)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_bibtex(entry: str) -> Optional[Dict[str, Any]]:
    """Parse a single BibTeX ``@type{key, field=..., ...}`` entry.

    Returns the FIRST entry's parsed payload as a dict, or ``None``
    on a malformed entry. Kept for backward compatibility with
    callers that expect a single-dict return; multi-entry strings
    should use :func:`_parse_bibtex_entries`.

    Field values lose their surrounding ``{...}`` or ``"..."``
    quoting; LaTeX-escape backslashes (``\\&``, ``{\\"o}``,
    ``{\\'e}``) are normalised to plain ASCII so APA prose isn't
    littered with TeX commands.
    """
    entries = _parse_bibtex_entries(entry) if isinstance(entry, str) else []
    return entries[0] if entries else None


def _parse_entry_body(
    entry_type: str, entry_key: str, body: str
) -> Optional[Dict[str, Any]]:
    """Walk one entry's ``key = {value}, ...`` body into a dict.

    Returns ``{"type", "key", "fields": {...}}`` or ``None`` if the
    body is unparseable.
    """
    entry_type = entry_type.lower()
    fields: Dict[str, str] = {}
    # Walk ``key = {value}`` pairs. Value can be ``{...}`` (possibly
    # nested) or ``"..."`` or a bare token.
    i = 0
    n = len(body)
    while i < n:
        # Skip whitespace + commas.
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        # Read the key.
        j = i
        while j < n and (body[j].isalnum() or body[j] == "_"):
            j += 1
        if j == i:
            break  # malformed — bail
        key = body[i:j].lower()
        i = j
        # Skip whitespace and the ``=``.
        while i < n and body[i] in " \t\r\n":
            i += 1
        if i >= n or body[i] != "=":
            break
        i += 1
        while i < n and body[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        # Read the value.
        if body[i] == "{":
            depth = 1
            i += 1
            start = i
            while i < n and depth > 0:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            value = body[start:i]
            i += 1  # consume closing ``}``
        elif body[i] == '"':
            i += 1
            start = i
            while i < n and body[i] != '"':
                i += 1
            value = body[start:i]
            i += 1
        else:
            start = i
            while i < n and body[i] not in " ,\t\r\n":
                i += 1
            value = body[start:i]
        fields[key] = _normalise_latex(value)

    return {"type": entry_type, "key": entry_key, "fields": fields}


def _normalise_latex(s: str) -> str:
    """Turn common LaTeX escapes into plain Unicode for APA prose.

    Best effort — anything we don't recognise is left intact so the
    user sees the source rather than a silent corruption.
    """
    if not s:
        return s
    # Common diacritics: {\\"o} → ö, {\\'e} → é, {\\`a} → à, etc.
    s = re.sub(
        r"\{\s*\\\"\s*([A-Za-z])\s*\}",
        lambda m: {
            "a": "ä",
            "A": "Ä",
            "o": "ö",
            "O": "Ö",
            "u": "ü",
            "U": "Ü",
            "e": "ë",
            "E": "Ë",
            "i": "ï",
            "I": "Ï",
        }.get(m.group(1), m.group(1)),
        s,
    )
    s = re.sub(
        r"\{\s*\\\'\s*([A-Za-z])\s*\}",
        lambda m: {
            "a": "á",
            "e": "é",
            "i": "í",
            "o": "ó",
            "u": "ú",
            "y": "ý",
            "A": "Á",
            "E": "É",
            "I": "Í",
            "O": "Ó",
            "U": "Ú",
            "Y": "Ý",
        }.get(m.group(1), m.group(1)),
        s,
    )
    s = re.sub(
        r"\{\s*\\\`\s*([A-Za-z])\s*\}",
        lambda m: {
            "a": "à",
            "e": "è",
            "i": "ì",
            "o": "ò",
            "u": "ù",
            "A": "À",
            "E": "È",
            "I": "Ì",
            "O": "Ò",
            "U": "Ù",
        }.get(m.group(1), m.group(1)),
        s,
    )
    s = s.replace(r"{\oe}", "œ").replace(r"{\OE}", "Œ")
    s = s.replace(r"{\ae}", "æ").replace(r"{\AE}", "Æ")
    # Slashed/ring/stroke letters: {\o}→ø (e.g. Møen), {\l}→ł, {\aa}→å, {\ss}→ß.
    s = s.replace(r"{\o}", "ø").replace(r"{\O}", "Ø")
    s = s.replace(r"{\l}", "ł").replace(r"{\L}", "Ł")
    s = s.replace(r"{\aa}", "å").replace(r"{\AA}", "Å")
    s = s.replace(r"{\ss}", "ß")
    s = s.replace(r"\&", "&").replace(r"\$", "$").replace(r"\%", "%")
    # Drop any remaining single-pair braces around plain text.
    s = re.sub(r"\{([^{}]*)\}", r"\1", s)
    s = s.strip()
    return s


# ---------------------------------------------------------------------------
#  Author parsing for APA "Last, F. M., & Last, F." formatting
# ---------------------------------------------------------------------------


def _split_authors(field: str) -> List[Dict[str, str]]:
    """Split a BibTeX ``author`` field into ``[{last, first, ...}]``."""
    if not field:
        return []
    # BibTeX separates authors with " and "; case-sensitive and surrounded
    # by spaces is the canonical form.
    parts = re.split(r"\s+and\s+", field)
    out: List[Dict[str, str]] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "," in p:
            # "Last, First Middle"
            last, first = [x.strip() for x in p.split(",", 1)]
        else:
            # "First Middle Last" — last token is the surname.
            tokens = p.split()
            last = tokens[-1]
            first = " ".join(tokens[:-1])
        out.append({"last": last, "first": first.strip()})
    return out


def _initials(name: str) -> str:
    """Turn a given name into APA-style initials: ``"Pedro H.C."`` →
    ``"P. H. C."``."""
    if not name:
        return ""
    # Tokenise on whitespace and dots; keep each token's leading char.
    tokens = re.split(r"[\s.\-]+", name)
    parts = [t[0].upper() + "." for t in tokens if t]
    return " ".join(parts)


def _format_authors_apa(authors: List[Dict[str, str]]) -> str:
    """APA author list: ``Last, F. M., Last, F. M., & Last, F. M.``"""
    if not authors:
        return ""
    formatted = [
        (
            f"{a['last']}, {_initials(a['first'])}".rstrip(", ")
            if a["first"]
            else a["last"]
        )
        for a in authors
    ]
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


# ---------------------------------------------------------------------------
#  APA + JSON formatters
# ---------------------------------------------------------------------------


def _format_apa(parsed: Dict[str, Any]) -> str:
    fields = parsed["fields"]
    parts: List[str] = []

    authors = _format_authors_apa(_split_authors(fields.get("author", "")))
    if authors:
        parts.append(authors)

    year = fields.get("year", "").strip()
    if year:
        parts.append(f"({year}).")
    elif parts:
        parts[-1] += "."

    title = fields.get("title", "").strip()
    if title:
        # APA: title in sentence case but the source already in title
        # case is acceptable — don't auto-lower (might break proper
        # nouns).
        if not title.endswith("."):
            title += "."
        parts.append(title)

    if parsed["type"] == "article":
        journal = fields.get("journal", "").strip()
        volume = fields.get("volume", "").strip()
        number = fields.get("number", "").strip()
        pages = fields.get("pages", "").strip().replace("--", "–")
        tail_bits: List[str] = []
        if journal:
            tail = journal
            if volume:
                tail += f", {volume}"
                if number:
                    tail += f"({number})"
            if pages:
                tail += f", {pages}"
            tail_bits.append(tail + ".")
        parts.extend(tail_bits)
    elif parsed["type"] == "book":
        publisher = fields.get("publisher", "").strip()
        if publisher:
            parts.append(publisher + ".")
    elif parsed["type"] in ("inproceedings", "incollection"):
        booktitle = fields.get("booktitle", "").strip()
        if booktitle:
            parts.append(f"In {booktitle}.")
        publisher = fields.get("publisher", "").strip()
        if publisher:
            parts.append(publisher + ".")

    return " ".join(p.strip() for p in parts if p).strip()


def _format_json(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Structured citation payload: type / key / fields / authors."""
    fields = dict(parsed["fields"])
    authors = _split_authors(fields.get("author", ""))
    return {
        "type": parsed["type"],
        "key": parsed["key"],
        "authors": authors,
        "year": fields.get("year"),
        "title": fields.get("title"),
        "journal": fields.get("journal"),
        "volume": fields.get("volume"),
        "number": fields.get("number"),
        "pages": fields.get("pages"),
        "publisher": fields.get("publisher"),
        "fields": fields,
    }


# ---------------------------------------------------------------------------
#  Public entry points
# ---------------------------------------------------------------------------


_VALID_FORMATS = ("bibtex", "apa", "json")


def render_citation(bibtex: str, fmt: str = "bibtex") -> Any:
    """Render a stored BibTeX string in the requested format.

    Parameters
    ----------
    bibtex : str
        Raw BibTeX entry as stored on the result class. May contain
        multiple ``@type{...}`` entries concatenated (some methods
        cite more than one paper); the renderer walks every entry.
    fmt : {"bibtex", "apa", "json"}
        Output format.

    Returns
    -------
    str | dict | list
        - ``"bibtex"`` → ``str`` (the raw string, multi-entry
          preserved as-is).
        - ``"apa"`` → ``str`` (single entry) or ``str`` with entries
          joined by a blank line (multi-entry).
        - ``"json"`` → ``dict`` for single-entry input (the original
          shape, preserved for backward compat) OR ``list[dict]``
          when the source contains multiple BibTeX entries.
    """
    if fmt not in _VALID_FORMATS:
        raise ValueError(f"format must be one of {_VALID_FORMATS}; got {fmt!r}")
    if fmt == "bibtex":
        return bibtex
    entries = _parse_bibtex_entries(bibtex) if isinstance(bibtex, str) else []
    if not entries:
        # Malformed / placeholder entry — surface the raw string in
        # APA mode and an empty payload in JSON mode rather than
        # silently corrupting agent output.
        if fmt == "apa":
            return bibtex.strip() if isinstance(bibtex, str) else ""
        return {"type": None, "key": None, "authors": [], "fields": {}, "raw": bibtex}
    if fmt == "apa":
        return "\n\n".join(_format_apa(p) for p in entries)
    if len(entries) == 1:
        return _format_json(entries[0])
    return [_format_json(p) for p in entries]


def bib_for(result: Any) -> Dict[str, Any]:
    """Top-level structured citation for a fitted result.

    Convenience entry that pairs with ``result.cite(format="json")``
    so agents that don't have direct access to the result method can
    pull the structured payload via ``sp.bib_for(...)`` instead.

    Parameters
    ----------
    result : CausalResult or EconometricResults
        Any fitted result object exposing a ``.cite()`` method.

    Returns
    -------
    dict
        Same shape as ``result.cite(format="json")``: ``{type, key,
        authors, year, title, journal, volume, number, pages,
        publisher, fields}``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(5)
    >>> rows = []
    >>> for i in range(200):
    ...     tr = 1 if i < 100 else 0
    ...     for t in (0, 1):
    ...         y = (1.0 + 0.3 * t + 0.5 * tr + 2.0 * tr * t
    ...              + rng.normal(scale=0.5))
    ...         rows.append({'i': i, 't': t, 'treated': tr, 'post': t, 'y': y})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.did(df, y='y', treat='treated', time='t', post='post')
    >>> sp.bib_for(r)['key']
    'angrist2009mostly'
    """
    cite_fn = getattr(result, "cite", None)
    if not callable(cite_fn):
        raise TypeError(f"result {type(result).__name__} has no .cite() method.")
    # First try the new format= path (CausalResult after this PR);
    # fall back to bibtex-string parsing for legacy result types.
    import inspect

    try:
        params: Any = inspect.signature(cite_fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "format" in params or "fmt" in params:
        try:
            kw = "format" if "format" in params else "fmt"
            payload = cite_fn(**{kw: "json"})
            if isinstance(payload, dict):
                return payload
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    bibtex = cite_fn()
    return cast(Dict[str, Any], render_citation(bibtex, fmt="json"))


def bibtex(keys: Union[str, Iterable[str]]) -> str:
    """Resolve verified BibTeX entries from ``paper.bib`` by citation key.

    The Python twin of the ``bibtex`` MCP tool, and the resolver that
    :func:`bib_for` and ``result.cite(format="json")`` advertise in their
    ``resolve_with`` hint: feed the ``citation_keys`` they return straight
    into ``sp.bibtex`` to obtain the full, verified ``@article{...}``
    entries from the project's single bibliographic source of truth
    (``paper.bib``). Entries are returned **verbatim** — never reformatted
    or invented (CLAUDE.md §10 zero-hallucination rule).

    Parameters
    ----------
    keys : str or iterable of str
        One or more BibTeX keys, e.g. ``"chernozhukov2016hdm"`` or
        ``["callaway2021difference", "rambachan2023more"]``.

    Returns
    -------
    str
        The matching entries in the order requested, separated by a blank
        line, ready to paste into a ``.bib`` file.

    Raises
    ------
    ValueError
        If ``keys`` is empty.
    KeyError
        If any requested key is absent from ``paper.bib``; the message
        lists the missing keys together with any close matches, so a typo
        is corrected rather than papered over with a fabricated entry.

    See Also
    --------
    bib_for : structured citation payload (and ``citation_keys``) for a
        fitted result; pipe its keys into this function.

    Examples
    --------
    >>> import statspai as sp
    >>> entry = sp.bibtex("chernozhukov2016hdm")
    >>> all(
    ...     x in entry
    ...     for x in ("Chernozhukov", "Hansen", "Spindler", "10.32614/RJ-2016-040")
    ... )
    True
    """
    if isinstance(keys, str):
        keys = [keys]
    key_list = [str(k) for k in keys]
    if not key_list:
        raise ValueError("`keys` must be a non-empty bib key or list of keys.")

    # Reuse the single paper.bib resolver that backs the MCP `bibtex`
    # tool, so the Python and MCP surfaces return identical entries.
    from statspai.agent.workflow_tools import _load_bibtex_index

    index = _load_bibtex_index()
    resolved: List[str] = []
    missing: List[str] = []
    for key in key_list:
        entry = index.get(key)
        if entry:
            resolved.append(entry)
        else:
            missing.append(key)

    if missing:
        from difflib import get_close_matches

        hints: Dict[str, List[str]] = {}
        for key in missing:
            close = get_close_matches(key, list(index), n=3, cutoff=0.55)
            if close:
                hints[key] = close
        message = f"bib key(s) not found in paper.bib: {missing}"
        if hints:
            message += f"; closest matches: {hints}"
        raise KeyError(message)

    return "\n\n".join(resolved)


__all__ = ["render_citation", "bib_for", "bibtex"]
