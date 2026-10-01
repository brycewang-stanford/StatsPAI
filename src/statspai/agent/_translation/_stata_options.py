"""Grammar shared by every Stata command, applied before and after a handler.

The handlers in :mod:`._stata` each read their options by full name. Real
do-files use Stata's own abbreviations (``reg y x, r``, ``cl(id)``,
``reghdfe ..., a(id year)``), wrap commands in prefixes (``qui``,
``eststo m1:``) and refer to macros (``$controls``). None of that is
command-specific, so it is handled here once:

* prefixes are peeled; the ones that change what is estimated (``by``,
  ``bootstrap``, ``svy`` ...) are refused instead of dropped;
* macros are refused: their contents live in another line of the do-file;
* abbreviated options are expanded to the full names the handlers read;
* options a handler never looked at are reported, never dropped silently.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Prefixes
# ---------------------------------------------------------------------------

#: Prefixes that only change what Stata prints or stores. ``(full, minimum
#: abbreviation)``; ``quietly`` / ``noisily`` / ``capture`` may be written
#: with or without a colon.
_DISPLAY_PREFIXES = (("quietly", 3), ("noisily", 3), ("capture", 3))

#: Colon prefixes that store or relabel results without changing them.
_STORE_PREFIXES = {"eststo", "estpost", "xi", "version"}

#: Colon prefixes that change the estimator, the sample or the inference:
#: dropping them would translate a different model.
_SEMANTIC_PREFIXES = {
    "by": "runs the command separately for each group",
    "bysort": "runs the command separately for each group",
    "bootstrap": "replaces the standard errors with bootstrap ones",
    "jackknife": "replaces the standard errors with jackknife ones",
    "permute": "computes permutation p-values",
    "svy": "applies the survey design to point estimates and SEs",
    "rolling": "re-estimates over rolling windows",
    "statsby": "collects estimates per group",
    "simulate": "runs a simulation",
    "nestreg": "fits a sequence of nested models",
    "stepwise": "selects regressors stepwise",
}
_SEMANTIC_ALIASES = {
    "bys": "bysort",
    "byso": "bysort",
    "bysor": "bysort",
    "bs": "bootstrap",
}

_WORD = re.compile(r"[A-Za-z_]\w*")


_Table = Tuple[Tuple[str, int], ...]


def _expand_word(word: str, table: _Table) -> Optional[str]:
    low = word.lower()
    for full, minlen in table:
        if len(low) >= minlen and full.startswith(low):
            return full
    return None


def _top_level_colon(s: str) -> int:
    depth, quoted = 0, False
    for i, ch in enumerate(s):
        if ch == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
    return -1


def peel_prefixes(line: str) -> Tuple[List[str], Optional[str], str]:
    """Split ``line`` into ``(harmless prefixes, refused prefix, command)``.

    ``mi estimate:`` is left in place: it has a handler of its own.
    """
    peeled: List[str] = []
    s = line.strip()
    while True:
        m = _WORD.match(s)
        if not m:
            return peeled, None, s
        word = m.group(0)
        display = _expand_word(word, _DISPLAY_PREFIXES)
        if display:
            rest = s[m.end() :].lstrip()
            if rest.startswith(":"):
                rest = rest[1:].lstrip()
            if not rest:
                return peeled, None, s
            peeled.append(display)
            s = rest
            continue
        low = word.lower()
        low = _SEMANTIC_ALIASES.get(low, low)
        if low in _SEMANTIC_PREFIXES or low in _STORE_PREFIXES:
            colon = _top_level_colon(s)
            if colon < 0:  # ``xi i.g`` / ``bootstrap`` without a command
                return peeled, None, s
            if low in _SEMANTIC_PREFIXES:
                return peeled, low, s
            peeled.append(low)
            s = s[colon + 1 :].lstrip()
            continue
        return peeled, None, s


def semantic_prefix_error(prefix: str) -> str:
    return (
        f"the `{prefix}` prefix {_SEMANTIC_PREFIXES[prefix]}; translating the "
        "command without it would estimate something else. Call the sp "
        "function for the inner command and apply the prefix in Python."
    )


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

#: ``$name`` / ``${name}`` (global) and `` `name' `` (local). A compound
#: quote `` `"..."' `` is a string, not a macro.
_MACRO_RE = re.compile(r"\$\{?[A-Za-z_]\w*|`(?!\")[^']*'")


#: Options that only label output; a macro inside them does not touch the
#: estimate, so it does not block the translation.
_LABEL_OPTIONS = frozenset(
    "title subtitle note caption xtitle ytitle xlabel ylabel legend "
    "graph_options saving name".split()
)


def find_macro(text: str) -> Optional[str]:
    m = _MACRO_RE.search(text or "")
    return m.group(0) if m else None


def find_macro_in_command(cmd: Any) -> Optional[str]:
    """First macro in the varlist, ``if`` / ``in`` or an estimation option."""
    for text in [" ".join(cmd.varlist), cmd.if_cond, cmd.in_range]:
        hit = find_macro(text)
        if hit:
            return hit
    for name, value in cmd.options.items():
        if name in _LABEL_OPTIONS or name in DISPLAY_OPTIONS:
            continue
        hit = find_macro(name) or find_macro(value)
        if hit:
            return hit
    return None


def macro_error(token: str) -> str:
    return (
        f"Stata macro {token!r} is defined elsewhere in the do-file; expand "
        "it (write out the variable list) before translating"
    )


# ---------------------------------------------------------------------------
# Option abbreviations
# ---------------------------------------------------------------------------

#: Options whose abbreviations mean the same thing in every translated
#: command: ``(full name, minimum abbreviation)`` as in Stata's syntax
#: diagrams (the underlined part).
_COMMON_OPTIONS = (
    ("robust", 1),
    ("cluster", 2),
    ("noconstant", 3),
)

#: Options of the HDFE family (reghdfe / ivreghdfe / ppmlhdfe).
_HDFE_COMMANDS = {"reghdfe", "ivreghdfe", "ppmlhdfe"}
_HDFE_OPTIONS = (
    ("absorb", 1),
    ("keepsingletons", 7),
)

#: ``vce()`` types: ``vce(r)`` is ``vce(robust)``, ``vce(cl id)`` is
#: ``vce(cluster id)``.
_VCE_TYPES = (
    ("robust", 1),
    ("cluster", 2),
    ("unadjusted", 2),
    ("bootstrap", 4),
    ("jackknife", 4),
)


def canonicalise_options(
    command: str, options: Dict[str, Optional[str]]
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Expand abbreviated option names (and ``vce()`` types) to full names.

    Returns the new options and one line per expansion. An abbreviation is
    only expanded when the full name is not also given.
    """
    table = _COMMON_OPTIONS + (_HDFE_OPTIONS if command in _HDFE_COMMANDS else ())
    out: Dict[str, Optional[str]] = {}
    expanded: List[str] = []
    for name, value in options.items():
        full = _expand_word(name, table)
        if full and full != name and full not in options:
            expanded.append(f"{name} -> {full}")
            name = full
        if name == "vce" and value:
            head, _, rest = value.strip().partition(" ")
            vfull = _expand_word(head, _VCE_TYPES)
            if vfull and vfull != head.lower():
                expanded.append(f"vce({head}) -> vce({vfull})")
                value = (vfull + " " + rest).strip()
        out[name] = value
    return out, expanded


# ---------------------------------------------------------------------------
# Options a handler never read
# ---------------------------------------------------------------------------

#: Options that only change what Stata prints. Reported, but flagged as
#: having no effect on estimates.
DISPLAY_OPTIONS = frozenset(
    "noheader notable nolog noomitted baselevels allbaselevels noemptycells "
    "vsquish cformat pformat sformat nofvlabel fvwrap fvwrapon coeflegend "
    "nocnsreport nofootnote nodots dots verbose noisily".split()
)


#: Reporting options of specific commands: they change how coefficients
#: are printed (odds ratios, incidence-rate ratios, standardised betas),
#: not the fit.
_DISPLAY_BY_COMMAND = {
    "regress": {"beta"},
    "reg": {"beta"},
    "logit": {"or"},
    "mlogit": {"rrr"},
    "poisson": {"irr"},
    "nbreg": {"irr"},
    "xtnbreg": {"irr"},
    "ppmlhdfe": {"irr", "eform"},
    "rdrobust": {"all"},
}


def is_display_option(command: str, name: str) -> bool:
    return name in DISPLAY_OPTIONS or name in _DISPLAY_BY_COMMAND.get(command, ())


class TrackedOptions(Dict[str, Optional[str]]):
    """A dict that records which keys were looked at.

    Iterating over it (``set(opts)``, ``sorted(opts)``) counts as looking at
    every key: the handler is enumerating the options to report on them.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: Set[str] = set()

    def get(self, key: Any, default: Any = None) -> Any:
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Optional[str]:
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.read.add(str(key))
        return super().__contains__(key)

    def pop(self, key: Any, *default: Any) -> Any:
        self.read.add(key)
        return super().pop(key, *default)

    def __iter__(self) -> Iterator[str]:
        self.read.update(super().keys())
        return super().__iter__()

    def keys(self) -> Any:
        self.read.update(super().keys())
        return super().keys()

    def items(self) -> Any:
        self.read.update(super().keys())
        return super().items()

    def unread(self) -> List[str]:
        return [k for k in super().keys() if k not in self.read]


def untranslated_notes(
    lossy: List[str], display: List[str], options: Dict[str, Optional[str]]
) -> List[str]:
    """Notes for options the translation does not carry over."""

    def show(names: List[str]) -> str:
        return ", ".join(
            n + (f"({options[n]})" if options.get(n) else "") for n in names
        )

    notes: List[str] = []
    if lossy:
        notes.append(
            "Not translated, so the sp call does not apply: "
            + show(lossy)
            + ". Check whether they change the estimate, the sample or the SEs."
        )
    if display:
        notes.append("Display-only options ignored: " + show(display) + ".")
    return notes


_SE_ARGUMENTS = ("robust", "cluster", "vce", "vcov", "se_type", "cov_type")


def se_note(
    options: Dict[str, Optional[str]],
    arguments: Dict[str, Any],
    reported: List[str],
) -> Tuple[Optional[str], List[str]]:
    """Non-default SEs Stata asked for that the call does not request.

    Returns ``(note, option names)``. Options in ``reported`` were already
    flagged as untranslated and are skipped.
    """
    asked: List[str] = []
    names: List[str] = []
    if "robust" in options and "robust" not in reported:
        asked.append("robust")
        names.append("robust")
    if options.get("cluster") and "cluster" not in reported:
        asked.append(f"cluster({options['cluster']})")
        names.append("cluster")
    vce = options.get("vce") if "vce" not in reported else None
    if vce:
        asked.append(f"vce({vce})")
        names.append("vce")
    if not asked:
        return None, []
    vce_head = vce.split()[0].lower() if vce else ""
    known_vce = vce_head in ("", "robust", "cluster", "hc0", "hc1", "hc2", "hc3", "nn")
    if known_vce and any(arguments.get(k) for k in _SE_ARGUMENTS):
        return None, []
    if not known_vce and vce_head in str(arguments).lower():
        return None, []
    note = (
        "Standard errors: " + ", ".join(asked) + " is not carried over; the sp "
        "call uses its default SEs. Set them on the sp call."
    )
    return note, names


__all__ = [
    "peel_prefixes",
    "semantic_prefix_error",
    "find_macro",
    "macro_error",
    "canonicalise_options",
    "TrackedOptions",
    "untranslated_notes",
    "se_note",
    "is_display_option",
    "find_macro_in_command",
    "DISPLAY_OPTIONS",
]
