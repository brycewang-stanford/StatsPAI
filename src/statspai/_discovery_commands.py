"""Stata / R command names -> StatsPAI entry points, for discovery.

An economist who knows ``csdid`` or ``att_gt`` types that name into
:func:`statspai.search_functions` or tries ``sp.csdid``. Before this index
the search ranked ``cs_jackknife`` first for ``csdid`` and returned nothing
for ``ivreg2`` / ``bacondecomp`` / ``boottest``, and ``sp.csdid`` suggested
``sp.sdid``.

**Single source.** The index is read from the translators
(:mod:`statspai.agent._translation`): every Stata command in
``STATA_COMMAND_MAP``, every R function in ``R_FUNCTION_MAP`` and every
command the translators answer with guidance instead of a translation
(Stata ``honestdid`` / ``ddml`` / ``weakivtest``, R ``CausalImpact`` ...).
The ``sp.*`` target of a command is what
:func:`statspai.translation_coverage` parses from its
handler; for handlers that pick the tool at run time (``logit`` /
``probit`` / ``areg`` ...) the translator is run on a minimal command line
and the tool it emits is taken. Nothing here restates a mapping the
translators already hold.

:data:`SUPPLEMENT` lists the few names users type that the translators do
not recognise at all (``csdid2``, fixest's ``sunab``).
``tests/test_discovery_commands.py`` fails as soon as a translator learns
one of them, so the entry moves to the translator and is
deleted here.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

#: Names the translators do not recognise: ``{name: (language, targets)}``.
#: Each target must be a registered function; a name the translators gain
#: must be removed here (the test above enforces both).
SUPPLEMENT: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    # Stata (ssc) packages without a translator handler. eventstudyinteract
    # and xtabond2 moved to the translator on 2026-10-19 (a handler and a
    # guidance entry).
    "csdid2": ("stata", ("callaway_santanna",)),
    # R functions without a translator handler.
    "sunab": ("r", ("sun_abraham",)),
    "bacon": ("r", ("bacon_decomposition",)),
    "lm_robust": ("r", ("regress",)),
    # The R / Python package name; its estimator is synthdid_estimate().
    "synthdid": ("r", ("sdid",)),
}

#: Command names that are also ordinary English words. Inside a multi-word
#: query they are read as words, not commands ("median of the outcome"
#: is not Stata's ``median``); a query that is only the name still resolves.
AMBIGUOUS_WORDS = frozenset(
    {
        "areg",
        "bunching",
        "contrast",
        "correlate",
        "estat",
        "factor",
        "margins",
        "median",
        "mediate",
        "oneway",
        "summarize",
        "table",
        "threshold",
    }
)

_INDEX: Optional[Dict[str, List[Tuple[str, Tuple[str, ...]]]]] = None

#: What a translator bug looks like from outside (a malformed probe line
#: reaching an indexing / parsing step). Anything else propagates.
_TRANSLATOR_ERRORS = (ValueError, TypeError, KeyError, IndexError, AttributeError)


def _probe_tool(translate: Callable[[str], dict], line: str) -> Optional[str]:
    """Tool a translator emits for a minimal command line, or ``None``.

    Translators are documented never to raise (they return an ``error``
    payload), so a failure here is a translator bug; the command is then
    simply not indexed rather than taking search down with it.
    """
    try:
        out = translate(line)
    except _TRANSLATOR_ERRORS:
        return None
    tool = out.get("tool") if isinstance(out, dict) else None
    return tool if isinstance(tool, str) and tool else None


def _build() -> Dict[str, List[Tuple[str, Tuple[str, ...]]]]:
    from .agent._translation import _stata
    from .agent._translation._coverage import translation_coverage
    from .agent._translation._r import from_r
    from .agent._translation._stata import from_stata

    index: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}

    def add(name: str, lang: str, targets: List[str]) -> None:
        clean = tuple(
            dict.fromkeys(t[3:] if t.startswith("sp.") else t for t in targets)
        )
        if clean:
            index.setdefault(name.lower(), []).append((lang, clean))

    cov = translation_coverage()
    # Minimal lines; the first one a handler accepts names its main tool.
    probes = {
        "stata": ("{c} y x", "{c} y x (d = z), absorb(id)"),
        "r": ("{c}(y ~ x, data = df)",),
    }
    translators = {"stata": from_stata, "r": from_r}
    for lang in ("stata", "r"):
        for row in cov[lang]:
            for command in [row["command"], *row["aliases"]]:
                targets = [t[3:] if t.startswith("sp.") else t for t in row["targets"]]
                for line in probes[lang]:
                    tool = _probe_tool(translators[lang], line.format(c=command))
                    if tool and (tool in targets or not targets):
                        # The emitted tool leads; source order is only the
                        # order the handler's branches happen to be written.
                        targets = [tool] + [t for t in targets if t != tool]
                        break
                add(command, lang, targets)
    from .agent._translation import _r

    guidance_tables = (
        ("stata", getattr(_stata, "_UNTRANSLATED_GUIDANCE", {}) or {}),
        ("r", getattr(_r, "R_GUIDANCE", {}) or {}),
    )
    for lang, guidance in guidance_tables:
        for command, entry in guidance.items():
            funcs = entry[1] if isinstance(entry, tuple) and len(entry) > 1 else []
            if not any(row[0] == lang for row in index.get(command.lower(), [])):
                add(command, lang, list(funcs))
    for name, (lang, targets) in SUPPLEMENT.items():
        if name not in index:
            add(name, lang, list(targets))
    return index


def command_index() -> Dict[str, List[Tuple[str, Tuple[str, ...]]]]:
    """``{command: [(language, (sp targets, most canonical first)), ...]}``."""
    global _INDEX
    if _INDEX is None:
        _INDEX = _build()
    return _INDEX


def lookup(name: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """Index rows for ``name`` (case-insensitive; ``sp.`` / ``::`` stripped)."""
    key = name.strip().lower()
    if "::" in key:  # fixest::sunab, did::att_gt
        key = key.split("::", 1)[1]
    key = key.rstrip("()")
    return command_index().get(key, [])


def describe(name: str, registered: Optional[set] = None) -> Optional[str]:
    """One-line pointer for ``sp.<name>`` misses, or ``None``.

    Called while ``sp.__getattr__`` composes an AttributeError, so it must
    not raise anything else (``hasattr(sp, x)`` would propagate it): an
    index that cannot be built costs the hint, nothing more.
    """
    try:
        rows = lookup(name)
    except (ImportError, *_TRANSLATOR_ERRORS):
        return None
    parts: List[str] = []
    for lang, targets in rows:
        live = [t for t in targets if registered is None or t in registered]
        if not live:
            continue
        label = "Stata command" if lang == "stata" else "R function"
        parts.append(f"{label} `{name}` is " + " / ".join(f"sp.{t}" for t in live))
    if not parts:
        return None
    return "; ".join(parts)


__all__ = ["SUPPLEMENT", "AMBIGUOUS_WORDS", "command_index", "lookup", "describe"]
