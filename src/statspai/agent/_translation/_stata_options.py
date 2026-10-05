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


#: Command-specific abbreviations, as underlined in each command's syntax
#: diagram. ``atet`` precedes ``ate`` so that the longer word wins.
_COMMAND_OPTIONS: Dict[str, _Table] = {
    # [R] qreg: quantile(#), minimum abbreviation q(#).
    "qreg": (("quantile", 1),),
    # [XT] xtgee: family(), link(), corr() and scale() by their first letter.
    "xtgee": (("family", 1), ("link", 1), ("corr", 1), ("scale", 1)),
    "teffects": (
        ("nneighbor", 2),
        ("ematch", 2),
        ("biasadj", 4),
        ("caliper", 3),
        ("osample", 2),
        ("generate", 3),
        ("metric", 3),
        ("dtolerance", 4),
        ("atet", 4),
        ("ate", 3),
    ),
    "did_imputation": (
        ("horizons", 1),
        ("pretrends", 3),
    ),
    "synth": (("figure", 3),),
    "drdid": (("ivar", 1), ("time", 1), ("treatment", 2)),
    "did2s": (("treatment", 5),),
    "csdid": (("ivar", 1), ("time", 1), ("gvar", 1)),
    "ttest": (("unpaired", 3), ("unequal", 3), ("welch", 1)),
    "ttesti": (("unequal", 3), ("welch", 1)),
    "sktest": (("noadjust", 3),),
    "ci": (("agresti", 2), ("jeffreys", 1), ("wilson", 2)),
    "tabstat": (("statistics", 1), ("columns", 1), ("format", 1)),
    "dfuller": (("regress", 3), ("trend", 2), ("drift", 2), ("lags", 1)),
    "estat": (("nomiss0", 4), ("lags", 1), ("cutoff", 3)),
    "prais": (("rhotype", 3), ("twostep", 3)),
    # [R] cnsreg: constraints(), minimum abbreviation c().
    "cnsreg": (("constraints", 1),),
    "nl": (("initial", 2),),
    # [XT] xtdpd: dgmmiv(), lgmmiv(), iv(), div(), liv(), twostep, hascons,
    # fodeviation, artests(); none is documented with an abbreviation.
    "xtdpd": (
        ("dgmmiv", 6),
        ("lgmmiv", 6),
        ("twostep", 7),
        ("hascons", 7),
        ("fodeviation", 11),
        ("artests", 7),
    ),
    # [XT] xthtaylor: endog(), constant(), varying(), amacurdy.
    "xthtaylor": (("endog", 4), ("constant", 4), ("varying", 4), ("amacurdy", 3)),
    # [MV] pca: components(), com(); mineigen(), mine(); covariance, cov.
    "pca": (
        ("components", 3),
        ("mineigen", 4),
        ("covariance", 3),
        ("correlation", 3),
    ),
    "factor": (("factors", 2), ("mineigen", 4)),
    # [TS] var: lags(numlist); the parser takes lag() as well.
    "var": (("lags", 3), ("exog", 2)),
    "svar": (("lags", 3), ("exog", 2)),
    # [R] heckman: select(), minimum abbreviation sel(); twostep, two.
    "heckman": (("select", 3), ("twostep", 3)),
    # [R] glm: family(), f(); link(), l(); scale(), sca().
    "glm": (("family", 1), ("link", 1), ("scale", 3), ("exposure", 1)),
    "corrgram": (("lags", 1),),
    "varsoc": (("maxlag", 1), ("exog", 2)),
    "varlmar": (("mlag", 2),),
    "veclmar": (("mlag", 2),),
    "vec": (("rank", 1), ("lags", 1), ("trend", 1)),
    "vecrank": (("lags", 1), ("trend", 1), ("max", 1)),
    "wntestq": (("lags", 1),),
    # rdlocrand / rdmulti: the authors' replication files write c() for
    # cutoff() and for cvar().
    "rdrandinf": (("cutoff", 1),),
    "rdwinselect": (("cutoff", 1),),
    "rdmc": (("cvar", 1),),
    "rdms": (("cvar", 1),),
    "rdmcplot": (("cvar", 1),),
    # [ST] streg: distribution(), frailty(), time and tratio.
    "streg": (("distribution", 1), ("frailty", 2), ("tratio", 2)),
    # [CAUSAL] etpoisson: treat(), intpoints().
    "etpoisson": (("treat", 2), ("intpoints", 4)),
}


def canonicalise_options(
    command: str, options: Dict[str, Optional[str]]
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Expand abbreviated option names (and ``vce()`` types) to full names.

    Returns the new options and one line per expansion. An abbreviation is
    only expanded when the full name is not also given.
    """
    table = _COMMON_OPTIONS + (_HDFE_OPTIONS if command in _HDFE_COMMANDS else ())
    if command == "areg":
        table = table + (("absorb", 1),)
    table = table + _COMMAND_OPTIONS.get(command, ())
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
    "ologit": {"or"},
    "mlogit": {"rrr"},
    "cloglog": {"eform"},
    "poisson": {"irr"},
    "nbreg": {"irr"},
    "xtnbreg": {"irr"},
    "ppmlhdfe": {"irr", "eform"},
    "rdrobust": {"all"},
    "rdplot": {"graph_options"},
    # `all` also prints the conventional statistic; the robust one, which
    # is the test, is unchanged.
    "rddensity": {
        "plot",
        "plot_range",
        "hist_range",
        "graph_opt",
        "all",
        # the binomial table is always in model_info['binomial_tests']
        "nobinomial",
    },
    "bitest": {"detail"},
    "bitesti": {"detail"},
    "rdmcplot": {"nodraw", "noscatter", "nopoly"},
    "rdwinselect": {"plot", "graph_options", "quietly"},
    "rdrandinf": {"quietly"},
    # sigf() / margin() / maxiter() tune Stata's optimiser, not the estimand
    "synth": {"figure", "keep", "sigf", "margin", "maxiter", "replace"},
    "sdid": {"graph", "g1on", "g1_opt", "g2_opt", "graph_export", "msize"},
    "bacondecomp": {"ddetail", "nograph", "stub", "gropt"},
    "mediate": {
        "all",
        "nie",
        "nde",
        "pnie",
        "tnde",
        "te",
        "pomeans",
        "aequations",
    },
    "boottest": {"nograph"},
    "dfuller": {"regress"},
    "xtreg": {"theta"},
    "xtserial": {"output"},
    "rcm": {"nofigure", "savegraph", "frame", "seed"},
    "synth2": {
        "nofigure",
        "savegraph",
        "frame",
        "sigf",
        "margin",
        "maxiter",
        "bound",
        "figure",
        "keep",
        "replace",
    },
    "varbasic": {"irf", "oirf", "fevd", "nograph", "step"},
    "varstable": {"graph"},
    "vecstable": {"graph"},
    "vecrank": {"max", "ic", "notrace"},
    "tabstat": {"columns", "format", "longstub", "labelwidth", "varwidth"},
    "oneway": {
        "tabulate",
        "noanova",
        "nolabel",
        "wrap",
        "missing",
        "means",
        "standard",
        "freq",
        "obs",
        "nomeans",
        "nostandard",
        "nofreq",
        "noobs",
    },
    "ranksum": {"porder"},
    "spearman": {"stats", "print", "star", "pw", "matrix"},
    "ktau": {"stats", "print", "star", "pw", "matrix"},
    "summarize": {"meanonly", "separator", "format"},
    "sum": {"meanonly", "separator", "format"},
    "su": {"meanonly", "separator", "format"},
    # reghdfe's way of saying "no fixed effects": what the call does without absorb()
    # noconstant: the constant is absorbed with the fixed effects, so the
    # option only removes the _cons row Stata prints
    "reghdfe": {"noabsorb", "noconstant"},
    # the plots and their axes; r2yz() only sets the scenarios of extremeplot
    "sensemakr": {
        "contourplot",
        "extremeplot",
        "tplot",
        "clim",
        "clines",
        "r2yz",
        "latex",
        "suppress",
    },
    "did_multiplegt_dyn": {"graph_off", "graphoptions", "_no_updates"},
    "did_had": {"graph_off", "graph_opts", "_no_updates"},
    "did_multiplegt_old": {"graphoptions"},
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


_SE_ARGUMENTS = (
    "robust",
    "cluster",
    "vce",
    "vcov",
    "se_type",
    "cov_type",
    "se_method",
)


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
    vce_words = vce.replace(",", " ").split() if vce else []
    vce_head = vce_words[0].lower() if vce_words else ""
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
