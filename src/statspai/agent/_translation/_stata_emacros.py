"""The text results an estimation command leaves in ``e()``.

``e(cmd)``, ``e(cmdline)``, ``e(depvar)``, ``e(model)``, ``e(vce)``,
``e(vcetype)`` and ``e(clustvar)``, and the model degrees of freedom
``e(df_m)`` where the generic count does not give Stata's number
(``xtreg``, ``areg``, ``reghdfe``).

Every string below was read off Stata 18 on a small panel (``xtset id
t``), one command and one ``vce()`` choice at a time; a combination that
was not checked leaves the macro unset, so that a later use of it is
refused rather than answered with a guess.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Tuple

from ._stata_lexer import _parse_options, _split_options
from ._stata_options import peel_prefixes

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["estimation_macros"]

#: the abbreviations of the commands below (Stata 18: `reg` and `regr` are
#: `regress`, `logi` is `logit`; the others are in the manual entries)
_ABBREVIATIONS = {"regress": 3, "logit": 4, "probit": 4, "poisson": 3, "tobit": 3}

_ML = frozenset(
    {
        "robust",
        "robust_opt",
        "cluster",
        "cluster_opt",
        "bootstrap",
        "jackknife",
        "oim",
        "opg",
    }
)

#: command -> the vce() choices whose e() strings were checked in Stata 18
#: (``robust_opt`` / ``cluster_opt``: the bare options ``robust`` /
#: ``cluster(v)``, which some commands refuse)
_ALLOWED: Dict[str, FrozenSet[str]] = {
    "regress": frozenset(
        {
            "robust",
            "robust_opt",
            "cluster",
            "cluster_opt",
            "bootstrap",
            "jackknife",
            "hc2",
            "hc3",
            "ols",
        }
    ),
    "areg": frozenset(
        {
            "robust",
            "robust_opt",
            "cluster",
            "cluster_opt",
            "bootstrap",
            "jackknife",
            "hc2",
        }
    ),
    "reghdfe": frozenset({"robust", "cluster", "cluster_opt", "unadjusted"}),
    "xtreg:re": frozenset(
        {"robust", "robust_opt", "cluster", "cluster_opt", "bootstrap", "jackknife"}
    ),
    "xtreg:fe": frozenset(
        {
            "robust",
            "robust_opt",
            "cluster",
            "cluster_opt",
            "bootstrap",
            "jackknife",
            "hc2",
        }
    ),
    "xtreg:be": frozenset({"bootstrap", "jackknife"}),
    "xtreg:ml": frozenset({"robust", "cluster", "bootstrap", "jackknife", "oim"}),
    "ivregress": frozenset(
        {"robust", "robust_opt", "cluster", "cluster_opt", "bootstrap", "jackknife"}
    ),
    "logit": _ML,
    "logistic": _ML,
    "probit": _ML,
    "cloglog": _ML,
    "poisson": _ML,
    "nbreg": _ML,
    "glm": _ML,
    "tobit": frozenset({"robust", "cluster", "bootstrap", "jackknife", "oim", "opg"}),
    "qreg": frozenset({"robust"}),
    "ologit": _ML - {"bootstrap"},
    "oprobit": _ML - {"bootstrap"},
    "mlogit": _ML - {"bootstrap", "jackknife"},
}

#: e(vce) / e(vcetype) without a vce() option
_DEFAULT: Dict[str, Tuple[str, Optional[str]]] = {
    "regress": ("ols", None),
    "areg": ("ols", None),
    "reghdfe": ("ols", None),
    "xtreg:re": ("conventional", None),
    "xtreg:fe": ("conventional", None),
    "xtreg:be": ("conventional", None),
    "xtreg:ml": ("oim", None),
    "ivregress:2sls": ("unadjusted", None),
    "ivregress:liml": ("unadjusted", None),
    "ivregress:gmm": ("robust", "Robust"),
    "qreg": ("iid", None),
    "glm": ("oim", "OIM"),
}

_STRINGS: Dict[str, Tuple[str, Optional[str]]] = {
    "robust": ("robust", "Robust"),
    "cluster": ("cluster", "Robust"),
    "bootstrap": ("bootstrap", "Bootstrap"),
    "jackknife": ("jackknife", "Jackknife"),
    "oim": ("oim", None),
    "opg": ("opg", "OPG"),
    "hc2": ("hc2", "Robust HC2"),
    "hc3": ("hc3", "Robust HC3"),
    "ols": ("ols", None),
    "unadjusted": ("ols", None),  # reghdfe, vce(unadjusted)
}

#: e(model)
_MODEL = {
    "regress": "ols",
    "areg": "ols",
    "reghdfe": "ols",
    "xtreg:re": "re",
    "xtreg:fe": "fe",
    "xtreg:be": "be",
    "xtreg:ml": "ml",
}

#: options that may change what the variance is called; not checked
_UNCHECKED = {"glm": ("irls", "scale"), "ivregress": ("wmatrix",)}

_INTERCEPTS = frozenset({"_cons", "Intercept", "const"})


def _prefix_of(word: str, full: str, shortest: int) -> bool:
    return shortest <= len(word) <= len(full) and full.startswith(word)


def _full_name(word: str) -> Optional[str]:
    word = word.lower()
    if word in _ALLOWED or word in ("xtreg", "ivregress"):
        return word
    for full, shortest in _ABBREVIATIONS.items():
        if _prefix_of(word, full, shortest):
            return full
    return None


def _vce_choice(opts: Dict[str, Optional[str]]) -> Tuple[Optional[str], Optional[str]]:
    """``(choice, cluster variable)`` from the option list; ``("default",
    None)`` without one and ``(None, None)`` for anything not listed."""
    choice, clustvar = "default", None
    for name, arg in opts.items():
        if name == "vce":
            words = (arg or "").replace(",", " ").split()
            if not words:
                return None, None
            head = words[0].lower()
            if _prefix_of(head, "robust", 1):
                choice = "robust"
            elif _prefix_of(head, "cluster", 2):
                if len(words) != 2:
                    return None, None  # no variable, or several
                choice, clustvar = "cluster", words[1]
            elif _prefix_of(head, "bootstrap", 4):
                choice = "bootstrap"
            elif _prefix_of(head, "jackknife", 4):
                choice = "jackknife"
            elif head in ("oim", "opg", "hc2", "hc3", "ols", "unadjusted"):
                choice = head
            else:
                return None, None
        elif _prefix_of(name, "robust", 1) and arg is None:
            if choice == "default":
                choice = "robust_opt"
        elif _prefix_of(name, "cluster", 2) and arg is not None:
            if len(arg.split()) != 1:
                return None, None
            choice, clustvar = "cluster_opt", arg.strip()
    return choice, clustvar


def _slopes(result: Any) -> Optional[int]:
    params = getattr(result, "params", None)
    if params is None:
        return None
    return sum(1 for name in params.index if str(name) not in _INTERCEPTS)


def estimation_macros(
    session: "StataSession", typed: Optional[str], logistic: bool = False
) -> None:
    """Set ``e()``'s text results (and correct ``e(df_m)``) after the
    estimation command ``typed``, the line as written with its macros
    expanded. A command or a ``vce()`` choice not checked against Stata
    leaves them unset."""
    macros: Dict[str, str] = {}
    session.stored["e_macros"] = macros
    if not typed:
        return
    _, refused, core = peel_prefixes(typed)
    m = re.match(r"\s*([A-Za-z_]\w*)", core)
    if refused is not None or m is None:
        return
    full = "logistic" if logistic else _full_name(m.group(1))
    if full is None:
        return
    rest = core[m.end() :].strip()
    head, options = _split_options(rest)
    opts = _parse_options(options)
    words = head.split()
    key = full
    if full == "xtreg":
        kinds = [k for k in ("re", "fe", "be", "mle") if k in opts]
        if len(kinds) > 1 or "pa" in opts:
            return
        key = "xtreg:" + ({"mle": "ml"}.get(kinds[0], kinds[0]) if kinds else "re")
    estimator = None
    if full == "ivregress":
        if not words or words[0].lower() not in ("2sls", "liml", "gmm"):
            return
        estimator = words.pop(0).lower()
    allowed = _ALLOWED[full if full == "ivregress" else key]
    macros["cmd"] = full
    macros["cmdline"] = f"{full} {rest}".rstrip()
    if words and re.fullmatch(r"[A-Za-z_][\w.]*", words[0]):
        macros["depvar"] = words[0]
    if key in _MODEL:
        macros["model"] = _MODEL[key]

    choice, clustvar = _vce_choice(opts)
    if any(o in opts for o in _UNCHECKED.get(full, ())):
        choice = None  # an option that may change the default, not checked
    weighted = re.search(r"\[\s*(?:pw|pweights?|iw|iweights?)\s*=", head, re.I)
    panel = opts.get("i") or session.panel[0]
    strings: Optional[Tuple[str, Optional[str]]] = None
    if choice == "default" and weighted and full == "regress":
        # regress with [pw=]: robust (Stata 18); other weighted fits unchecked
        if weighted.group(0).lower().lstrip("[ ").startswith("p"):
            strings = _STRINGS["robust"]
    elif choice == "default" and not weighted:
        default_key = f"ivregress:{estimator}" if estimator else key
        strings = _DEFAULT.get(default_key, ("oim", None))
    elif choice is not None and choice in allowed and not weighted:
        base = choice.replace("_opt", "")
        strings = _STRINGS[base]
        if base == "oim" and full == "glm":
            strings = ("oim", "OIM")
        if base == "cluster":
            macros["clustvar"] = str(clustvar)
        elif full == "xtreg" and panel and base != "oim":
            # xtreg: robust, bootstrap and jackknife resample or cluster
            # on the panel variable, and `fe` calls a robust fit `cluster`
            macros["clustvar"] = str(panel)
            if key == "xtreg:fe" and base == "robust":
                strings = _STRINGS["cluster"]
    if strings is not None:
        macros["vce"] = strings[0]
        if strings[1] is not None:
            macros["vcetype"] = strings[1]
    _model_df(session, key, strings)


def _model_df(
    session: "StataSession", key: str, strings: Optional[Tuple[str, Optional[str]]]
) -> None:
    """``e(df_m)`` of ``xtreg`` / ``areg`` / ``reghdfe`` (Stata 18): the
    slopes, except ``xtreg, fe``, which counts the panel intercepts (k + G -
    1) with its conventional standard errors and k - 1 with clustered or
    robust ones."""
    if not (key.startswith("xtreg:") or key in ("areg", "reghdfe")):
        return
    e = session.stored.setdefault("e", {})
    k = _slopes(session.last)
    e.pop("df_m", None)
    if k is None:
        return
    if key != "xtreg:fe":
        e["df_m"] = float(k)
        return
    vce = None if strings is None else strings[0]
    if vce == "conventional" and "N_g" in e:
        e["df_m"] = float(k + e["N_g"] - 1)
    elif vce in ("cluster", "hc2"):
        e["df_m"] = float(k - 1)
    elif vce in ("bootstrap", "jackknife"):
        e["df_m"] = float(k)
