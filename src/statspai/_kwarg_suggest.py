"""Did-you-mean suggestions for a misspelled or wrong-dialect keyword.

String similarity alone points the wrong way for the commonest slips: it
offers ``se_method`` for ``seed`` (the parameter is ``random_state``) and
nothing for ``cluster_var`` (``cluster``). Names that mean the same thing
across Stata / R / other Python libraries are listed here and offered
first; ``difflib`` fills in after them. Only names the function actually
takes are ever suggested.
"""

from __future__ import annotations

import difflib
import inspect
from typing import Any, Iterable, List, Optional

#: Spelling an agent writes -> parameter names it may mean, in order.
_SYNONYMS = {
    "seed": ("random_state", "rng_seed", "boot_seed", "random_seed"),
    "random_state": ("seed", "rng_seed", "random_seed"),
    "random_seed": ("seed", "random_state"),
    "rng": ("seed", "random_state"),
    "treatment": ("treat", "d", "D", "treatment_col"),
    "treat": ("treatment", "d", "D"),
    "d": ("treat", "treatment"),
    "outcome": ("y", "depvar", "dep_var"),
    "y": ("outcome",),
    "depvar": ("y", "outcome"),
    "cluster_var": ("cluster", "clustervars", "cluster_col"),
    "clustervar": ("cluster", "clustervars"),
    "clustervars": ("cluster",),
    "cluster": ("clustervars", "cluster_var"),
    "clustr": ("cluster",),
    "controls": ("covariates", "x", "X"),
    "covariates": ("controls", "covs", "x", "X"),
    "covs": ("covariates", "controls"),
    "unit": ("id", "i", "group", "unit_col"),
    "id": ("i", "unit", "group", "unit_id"),
    "i": ("id", "unit"),
    "entity": ("id", "i", "unit"),
    "time": ("t", "period", "tvar"),
    "t": ("time", "period"),
    "year": ("time", "t"),
    "period": ("time", "t"),
    "gvar": ("g", "cohort", "first_treat"),
    "cohort": ("g", "first_treat", "gvar"),
    "first_treat": ("g", "cohort", "gvar"),
    "running": ("x", "running_var"),
    "running_var": ("x", "running"),
    "cutoff": ("c",),
    "threshold": ("c", "cutoff"),
    "instrument": ("instruments", "z", "iv"),
    "instruments": ("instrument", "z"),
    "z": ("instrument", "instruments"),
    "weight": ("weights", "w", "sample_weight"),
    "weights": ("weight", "w", "sample_weight"),
    "w": ("weights",),
    "n_boot": ("n_bootstrap", "biters", "reps", "B"),
    "n_bootstrap": ("n_boot", "biters", "reps"),
    "reps": ("n_boot", "n_bootstrap"),
    "bootstrap": ("n_boot", "n_bootstrap", "se_method"),
    "robust": ("vce", "se_type"),
    "vce": ("robust", "se_type"),
    "se_type": ("vce", "robust"),
    "level": ("alpha",),
    "conf_level": ("alpha",),
    "absorb": ("fe",),
    "fe": ("absorb",),
    "formula": ("fml",),
}


def suggest_keywords(key: str, known: Iterable[str], n: int = 3) -> List[str]:
    """Parameters of ``known`` that ``key`` most likely meant, best first."""
    pool = [k for k in known if isinstance(k, str)]
    out: List[str] = [s for s in _SYNONYMS.get(key, ()) if s in pool and s != key]
    for cand in difflib.get_close_matches(key, sorted(pool), n=n, cutoff=0.6):
        if cand not in out:
            out.append(cand)
    return out[:n]


def signature_keywords(function: str) -> Optional[List[str]]:
    """Keyword parameters of ``sp.<function>``, or ``None`` if unresolvable.

    ``function`` may carry a call-shaped suffix, as error messages do
    (``"synth(method='scm')"``); only the name before ``(`` is looked up.
    Called on the error path only.
    """
    base = str(function).split("(", 1)[0].strip()
    try:
        import statspai

        fn: Any = getattr(statspai, base, None)
        if fn is None:
            return None
        params = inspect.signature(fn).parameters.values()
    except (ImportError, TypeError, ValueError):
        return None
    return [
        p.name
        for p in params
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name != "self"
    ]


def did_you_mean(key: str, known: Iterable[str]) -> str:
    """`` (did you mean 'a', 'b'?)`` or ``""``."""
    close = suggest_keywords(key, known)
    return f" (did you mean {', '.join(repr(c) for c in close)}?)" if close else ""
