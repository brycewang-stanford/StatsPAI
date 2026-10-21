"""Conventions of Callaway-Sant'Anna that differ between programs.

``sp.callaway_santanna(parity=...)`` and ``sp.aggte(parity=...)`` set, in one
argument, every convention on which StatsPAI's defaults differ from R ``did``
(``parity='r'``) or Stata ``csdid`` (``parity='stata'``). Each convention
below was established by running the reference program as a black box on
the same CSV bytes (``tests/reference_parity/test_cs_parity_switch.py``,
fixtures ``_generate_cs_parity_R.R`` / ``_generate_cs_parity_Stata.do``);
neither program's source was read.

========================  ===============  ===============  ===============
convention                StatsPAI default R did 2.5.1      Stata csdid 1.81
========================  ===============  ===============  ===============
base period               universal        varying          varying (short)
not-yet-treated cutoff    period           period           cohort
pre-test p-value          Hotelling F      chi2             chi2
unbalanced panel          pair             full             pair
DR pre-period regression  delta            reference        reference
aggregation weights       cohort shares    cohort shares    cell counts
aggregation SEs           follow the fit   follow the fit   analytic always
group-overall share term  included         included         not included
bootstrap multipliers     rademacher       rademacher       mammen
bootstrap draws           1000             1000             999
uniform band when boot    no               yes              yes
critical value            normal           normal           normal
always-treated units      dropped          dropped          dropped
========================  ===============  ===============  ===============

The first eight are *conventions*: they change a reported number and the
program offers no way to get StatsPAI's choice, or offers it under another
name. ``parity=`` fixes them, and an explicit argument that contradicts one
raises. The bootstrap rows are *options* every program exposes with the same
meaning, so ``parity=`` only supplies their defaults when ``bstrap=True``.
R's ``allow_unbalanced_panel = TRUE`` is an option of the same kind:
``parity='r'`` accepts ``allow_unbalanced_panel=True`` (or
``balance='none'``) and then reproduces it.

Two rows only bite off a balanced panel. "DR pre-period regression" is the
sign of the comparison-group pre-period regression's estimation effect in
the doubly robust estimator on repeated cross-sections (and on an
unbalanced panel read as one); see ``variance_convention`` of
``callaway_santanna``. "Aggregation weights": csdid weights each ATT(g, t)
by its own count of treated observations (repeated cross-sections) or of
treated units seen in both periods of the cell (unbalanced panel), in every
aggregate, and carries the estimation of those counts into the standard
errors; R weights by cohort shares. On a balanced panel the two coincide.
Established on the same CSV bytes in
``tests/reference_parity/test_cs_parity_rc_unbal.py``.

This module also holds the always-treated rule (units whose cohort is at or
before the first period plus ``anticipation``; R ``did`` and Stata
``csdid`` drop them, and so does StatsPAI) and the per-call record of which
arguments the caller passed explicitly.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from contextvars import ContextVar
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, IdentificationFailure, MethodIncompatibility

F = TypeVar("F", bound=Callable[..., Any])

#: Name of each reference program, for messages and notes.
PROGRAMS = {"r": "R did 2.5", "stata": "Stata csdid 1.8"}

#: Conventions of the fit that ``parity=`` fixes.
_FIT_CONVENTIONS: Dict[str, Dict[str, Any]] = {
    "r": {
        "base_period": "varying",
        "notyet_cutoff": "period",
        "variance_convention": "reference",
    },
    "stata": {
        "base_period": "varying",
        "notyet_cutoff": "cohort",
        "variance_convention": "reference",
    },
}

#: How each program handles a panel with missing unit-periods.
_BALANCE = {"r": "full", "stata": "pair"}

#: Bootstrap defaults, applied only to arguments the caller left unset.
_BOOT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "r": {"boot_weight_type": "rademacher", "biters": 1000, "cband": True},
    "stata": {"boot_weight_type": "mammen", "biters": 999, "cband": True},
}

#: One line printed by ``summary()`` when the base period was left at the
#: StatsPAI default.
BASE_PERIOD_NOTE = (
    "base_period='universal' (R did and Stata csdid default to 'varying'; "
    "pre-period ATT(g,t) differ, post-period ones do not)"
)

# ---------------------------------------------------------------------------
# Which arguments did the caller pass?
# ---------------------------------------------------------------------------

_EXPLICIT: ContextVar[Optional[FrozenSet[str]]] = ContextVar(
    "statspai_cs_explicit_args", default=None
)


def track_explicit(fn: F) -> F:
    """Record the names of the arguments a call passes, for :func:`explicit_args`.

    Applied innermost (below the alias decorators), so it sees canonical
    parameter names: ``n_boot=`` arrives here as ``biters``.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            names = frozenset(sig.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            return fn(*args, **kwargs)
        token = _EXPLICIT.set(names)
        try:
            return fn(*args, **kwargs)
        finally:
            _EXPLICIT.reset(token)

    return wrapper  # type: ignore[return-value]


def explicit_args() -> FrozenSet[str]:
    """Names of the arguments passed to the innermost tracked call."""
    names = _EXPLICIT.get()
    return names if names is not None else frozenset()


# ---------------------------------------------------------------------------
# parity= on callaway_santanna
# ---------------------------------------------------------------------------


def require_parity(parity: Any, *, function: str) -> Optional[str]:
    if parity is None or parity in PROGRAMS:
        return parity
    raise MethodIncompatibility(
        f"{function}: parity must be None, 'r' or 'stata'; got {parity!r}.",
        recovery_hint="Pass parity='r' (R did) or parity='stata' (Stata csdid).",
        diagnostics={"parity": parity},
    )


def _conflict(
    function: str, parity: str, name: str, given: Any, required: Any
) -> MethodIncompatibility:
    program = PROGRAMS[parity]
    return MethodIncompatibility(
        f"{function}: parity={parity!r} fixes {name}={required!r} ({program}'s "
        f"convention), but {name}={given!r} was passed.",
        recovery_hint=(
            f"Drop {name}= to keep parity={parity!r}, or drop parity= and set "
            "each convention yourself (the conventions a fit used are in "
            "model_info['conventions'])."
        ),
        diagnostics={"parity": parity, "argument": name, "given": given},
    )


def resolve_fit_parity(
    parity: Optional[str],
    *,
    explicit: FrozenSet[str],
    base_period: Any,
    notyet_cutoff: Any,
    balance: Any,
    allow_unbalanced_panel: Any,
    panel: Any,
    estimator: Any,
    variance_convention: Any = "delta",
) -> Dict[str, Any]:
    """Arguments of ``callaway_santanna`` after ``parity=`` is applied."""
    out: Dict[str, Any] = {
        "base_period": base_period,
        "notyet_cutoff": notyet_cutoff,
        "balance": balance,
        "variance_convention": variance_convention,
    }
    if parity is None:
        return out
    function = "callaway_santanna"
    given = {
        "base_period": base_period,
        "notyet_cutoff": notyet_cutoff,
        "variance_convention": variance_convention,
    }
    for name, required in _FIT_CONVENTIONS[parity].items():
        if name in explicit and given[name] != required:
            raise _conflict(function, parity, name, given[name], required)
        out[name] = required
    if panel is not False:
        # Units missing some periods: R did keeps only the complete units
        # (balance='full') unless allow_unbalanced_panel = TRUE, an option R
        # exposes (balance='none' here); csdid 1.8 uses each cell's
        # complete pairs and has no such option.
        required_balance = _BALANCE[parity]
        allowed = (None, required_balance) + (("none",) if parity == "r" else ())
        if "balance" in explicit and balance not in allowed:
            raise _conflict(function, parity, "balance", balance, required_balance)
        r_unbalanced = parity == "r" and (
            balance == "none"
            or ("allow_unbalanced_panel" in explicit and allow_unbalanced_panel is True)
        )
        if (
            parity == "stata"
            and "allow_unbalanced_panel" in explicit
            and allow_unbalanced_panel is True
        ):
            raise _conflict(
                function,
                parity,
                "allow_unbalanced_panel",
                True,
                f"False (balance={required_balance!r})",
            )
        if parity == "r":
            out["balance"] = "none" if r_unbalanced else required_balance
    if parity == "stata" and estimator in ("ipw", "stdipw") and "estimator" in explicit:
        warnings.warn(
            "callaway_santanna: estimator='ipw' is the stabilised IPW, which "
            "Stata csdid calls method(stdipw). csdid's method(ipw) is "
            "estimator='ipw_abadie'.",
            UserWarning,
            stacklevel=4,
        )
    return out


def bootstrap_defaults(
    parity: Optional[str], *, explicit: FrozenSet[str], bstrap: bool
) -> Dict[str, Any]:
    """Bootstrap options that ``parity=`` supplies (only the unset ones)."""
    if parity is None or not bstrap:
        return {}
    return {k: v for k, v in _BOOT_DEFAULTS[parity].items() if k not in explicit}


def fit_conventions(
    parity: Optional[str],
    *,
    base_period: str,
    base_period_explicit: bool,
    se_method: str,
    pretest: Optional[Dict[str, Any]],
    notyet_cutoff: str,
    balance: str,
    variance_convention: Optional[str] = None,
) -> Dict[str, Any]:
    """``model_info['conventions']`` of a Callaway-Sant'Anna fit."""
    return {
        "parity": parity,
        "base_period": base_period,
        "base_period_explicit": bool(base_period_explicit),
        "se_method": se_method,
        "crit_dist": "normal",
        "crit_df": None,
        # The fit's own simple aggregate and event study carry the
        # estimated-cohort-share term (R did and csdid both do).
        "share_variance": True,
        "pretest_pvalue": (pretest or {}).get(
            "pvalue_method", pretest_pvalue_method(parity)
        ),
        "notyet_cutoff": notyet_cutoff,
        "balance": balance,
        # Only the doubly robust estimator on the repeated-cross-section
        # path reads it; None elsewhere.
        "variance_convention": variance_convention,
    }


def pretest_pvalue_method(parity: Optional[str]) -> str:
    return "hotelling_f" if parity is None else "chi2"


def fit_notes(parity: Optional[str], *, base_period: str, explicit: bool) -> List[str]:
    """Lines ``summary()`` prints about the conventions of a fit."""
    notes: List[str] = []
    if parity is None and not explicit and base_period == "universal":
        notes.append(BASE_PERIOD_NOTE)
    if parity is not None:
        notes.append(
            f"parity={parity!r}: conventions of {PROGRAMS[parity]} (base "
            "period, not-yet-treated cutoff, chi2 pre-test p-value; see "
            "model_info['conventions'])"
        )
    return notes


# ---------------------------------------------------------------------------
# parity= on aggte
# ---------------------------------------------------------------------------


def resolve_aggte_parity(
    parity: Optional[str],
    *,
    fit_info: Dict[str, Any],
    explicit: FrozenSet[str],
    type: str,
    bstrap: Optional[bool],
    cband: Optional[bool],
    n_boot: Optional[int],
    share_variance: bool,
    agg_weights: str,
    has_cell_counts: bool,
) -> Dict[str, Any]:
    """Settings of ``aggte`` after ``parity=`` (or the fit's parity) is applied.

    Returns ``parity``, ``bstrap``, ``cband``, ``share_cells`` (the
    cohort-share term in each reported cell), ``share_overall`` (the term in
    the group aggregation's overall average) and ``agg_weights``.
    """
    function = "aggte"
    fit_conv = fit_info.get("conventions") or {}
    fit_parity = fit_conv.get("parity")
    if parity is not None and parity != fit_parity:
        raise MethodIncompatibility(
            f"aggte: parity={parity!r} was passed, but the fit was estimated "
            f"with parity={fit_parity!r}. The base period and comparison-group "
            "conventions are set when ATT(g, t) is estimated, so aggregating "
            "alone cannot reproduce the other program.",
            recovery_hint=(
                f"Refit with sp.callaway_santanna(..., parity={parity!r}) and "
                "aggregate that; aggte then inherits the parity."
            ),
            diagnostics={"parity": parity, "fit_parity": fit_parity},
        )
    parity = fit_parity if parity is None else parity
    out: Dict[str, Any] = {
        "parity": parity,
        "bstrap": bstrap,
        "cband": cband,
        "share_cells": bool(share_variance),
        "share_overall": bool(share_variance),
        "agg_weights": agg_weights,
    }
    if parity is None:
        return out
    if parity == "r":
        if "share_variance" in explicit and share_variance is not True:
            raise _conflict(function, parity, "share_variance", share_variance, True)
        if "agg_weights" in explicit and agg_weights != "did":
            raise _conflict(function, parity, "agg_weights", agg_weights, "did")
        out.update(share_cells=True, share_overall=True, agg_weights="did")
        return out
    # Stata csdid: estat reports analytic SEs even after wboot, and its group
    # average holds the cohort shares fixed (every other aggregate, and every
    # cell, carries the share term).
    if "bstrap" in explicit and bstrap:
        raise _conflict(function, parity, "bstrap", bstrap, False)
    if "cband" in explicit and cband:
        raise _conflict(function, parity, "cband", cband, False)
    if "n_boot" in explicit and n_boot is not None:
        raise _conflict(function, parity, "n_boot", n_boot, None)
    required_share = type != "group"
    if "share_variance" in explicit and bool(share_variance) != required_share:
        raise _conflict(
            function, parity, "share_variance", share_variance, required_share
        )
    # csdid weights every cell by its own treated count in every aggregate.
    # On a balanced panel that count is the cohort size, so the two rules
    # coincide and the cohort-share path (bit-identical to before) is kept.
    weights = "csdid" if has_cell_counts else "did"
    if "agg_weights" in explicit and agg_weights != weights:
        raise _conflict(function, parity, "agg_weights", agg_weights, weights)
    out.update(
        bstrap=False,
        cband=False,
        share_cells=True,
        share_overall=type != "group",
        agg_weights=weights,
    )
    return out


def aggte_conventions(
    fit_info: Dict[str, Any],
    *,
    parity: Optional[str],
    type: str,
    bstrap: bool,
    share_cells: bool,
    share_overall: bool,
    agg_weights: str,
) -> Dict[str, Any]:
    """``model_info['conventions']`` of an aggregation."""
    fit_conv = fit_info.get("conventions") or {}
    return {
        "parity": parity,
        "base_period": fit_conv.get("base_period", fit_info.get("base_period")),
        "base_period_explicit": fit_conv.get("base_period_explicit"),
        "se_method": "bootstrap" if bstrap else "analytic",
        "crit_dist": "normal",
        "crit_df": None,
        "share_variance": bool(share_overall if type == "group" else share_cells),
        "pretest_pvalue": fit_conv.get("pretest_pvalue"),
        "agg_weights": agg_weights,
    }


# ---------------------------------------------------------------------------
# Always-treated units
# ---------------------------------------------------------------------------


def drop_always_treated(
    data: pd.DataFrame,
    *,
    g: str,
    t: str,
    i: Optional[str],
    anticipation: int,
    rule: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove units first treated at or before the first period + anticipation.

    Such a unit has no untreated period, so none of its ATT(g, t) is
    identified, and it is treated in every period, so it never serves as a
    comparison either. R ``did`` (black box: ``g <= min(t) + anticipation``,
    the first period itself included) and Stata ``csdid`` (the help: groups
    treated before the first period "are excluded from the sample") drop
    them before estimating; so does this. On a panel the unit is ``i``; on
    repeated cross-sections (``i=None``) each row is its own unit.
    """
    if rule not in ("drop", "error"):
        raise MethodIncompatibility(
            f"callaway_santanna: always_treated must be 'drop' or 'error'; "
            f"got {rule!r}.",
            recovery_hint="Pass always_treated='drop' (R did) or 'error'.",
            diagnostics={"always_treated": rule},
        )
    info: Dict[str, Any] = {"n_units": 0, "n_obs": 0, "cohorts": []}
    if data.empty:
        return data, info
    g_num = pd.to_numeric(data[g], errors="coerce").replace([np.inf, -np.inf], np.nan)
    first = data[t].min()
    cutoff = first + anticipation
    mask = (g_num > 0) & (g_num <= cutoff)
    mask = mask.fillna(False).to_numpy(dtype=bool)
    if not mask.any():
        return data, info
    cohorts = sorted(float(c) for c in pd.unique(g_num[mask]))
    if i is not None:
        ids = pd.unique(data.loc[mask, i])
        rows = data[i].isin(ids).to_numpy()
        n_units = int(len(ids))
    else:
        ids = None
        rows = mask
        n_units = int(mask.sum())
    shown = ", ".join(f"{c:g}" for c in cohorts)
    where = (
        f"at or before the first period ({first}"
        + (f" + anticipation={anticipation}" if anticipation else "")
        + ")"
    )
    what = "units" if i is not None else "observations"
    if rule == "error":
        raise IdentificationFailure(
            f"callaway_santanna: {n_units} {what} (cohort g = {shown}) are "
            f"first treated {where}. They have no untreated period, so none "
            "of their ATT(g, t) is identified and they never serve as "
            "comparisons.",
            recovery_hint=(
                "Pass always_treated='drop' (what R did and Stata csdid do), or "
                "remove those units, or recode their cohort if it is wrong."
            ),
            diagnostics={
                "n_always_treated": n_units,
                "cohorts": cohorts,
                "first_period": first,
                "anticipation": anticipation,
            },
        )
    kept = data.loc[~rows]
    g_kept = pd.to_numeric(kept[g], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not bool((g_kept > 0).any()):
        raise DataInsufficient(
            "callaway_santanna: No valid (group, time) pairs to estimate. "
            f"Every treated cohort (g = {shown}) is first treated {where}, so "
            "no treated unit has an untreated period.",
            recovery_hint=(
                "Check the cohort coding of `g` (0 = never treated), or add "
                "periods before the first treatment."
            ),
            diagnostics={
                "n_always_treated": n_units,
                "cohorts": cohorts,
                "first_period": first,
                "anticipation": anticipation,
            },
        )
    info = {
        "n_units": n_units,
        "n_obs": int(rows.sum()),
        "cohorts": cohorts,
        "first_period": first,
        "anticipation": anticipation,
        "rule": "g <= first period + anticipation",
        "_ids": ids,
    }
    warnings.warn(
        f"callaway_santanna: dropped {n_units} {what} ({info['n_obs']} rows) "
        f"in cohort(s) g = {shown}, first treated {where}. They have no "
        "untreated period, so no ATT(g, t) of theirs is identified and they "
        "cannot serve as comparisons. R did and Stata csdid drop them too. "
        "Pass always_treated='error' to stop instead.",
        UserWarning,
        stacklevel=4,
    )
    return kept, info


def always_treated_record(info: Dict[str, Any]) -> Dict[str, Any]:
    """Public part of the always-treated record for ``model_info``."""
    return {k: v for k, v in info.items() if not k.startswith("_")}
