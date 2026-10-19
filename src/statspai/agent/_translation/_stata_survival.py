"""Stata's Cox and competing-risks commands: ``stcox`` and ``stcrreg``.

Both read the duration and the failure indicator from ``stset``, which
lives on another line: like ``streg`` they emit ``<stset ...>``
placeholders that ``sp.stata`` fills from an earlier ``stset timevar,
failure(eventvar)``.

Conventions pinned against Stata 18 (``tests/test_stata_translation_
survival.py``): ``stcox`` breaks ties with Breslow's method unless
``efron`` is given (``sp.cox`` defaults to Efron, so the method is always
written out); ``vce(cluster c)`` is ``sp.cox(cluster=c)``; ``vce(robust)``
is the one-observation-per-cluster variance with Stata's N/(N-1) factor,
which ``sp.cox(robust=)`` does not apply, so it is reported as not
translated. ``stcrreg`` is ``sp.finegray(small_sample=True)``: its default
robust variance carries the same N/(N-1).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ._stata import _emit, _emit_error, _vce_cluster
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS"]

_NEEDS_STSET = (
    "{cmd} reads the duration and the failure indicator from `stset`. Run "
    "it through sp.stata after `stset timevar, failure(eventvar)`, or "
    "replace the <stset ...> placeholders."
)


def _stset(cmd: StataCommand) -> tuple:
    duration = cmd.options.get("stset_time") or "<stset time>"
    event = cmd.options.get("stset_failure") or "<stset failure>"
    return str(duration).strip(), str(event).strip()


def _covariates(cmd: StataCommand, name: str) -> Any:
    xs = list(cmd.varlist)
    if not xs:
        return _emit_error(f"{name} needs at least one covariate", command=name)
    if any(not re.fullmatch(r"[A-Za-z_]\w*", x) for x in xs):
        return _emit_error(
            f"{name}: only plain covariate names are translated (got {xs!r}).",
            command=name,
            suggestions=[],
        )
    return xs


def _h_stcox(cmd: StataCommand) -> Dict[str, Any]:
    """``stcox x1 x2 [, efron strata(s) vce(cluster c)]`` -> ``sp.cox``."""
    xs = _covariates(cmd, "stcox")
    if isinstance(xs, dict):
        return xs
    opts = cmd.options
    for exact in ("exactm", "exactp"):
        if exact in opts:
            return _emit_error(
                f"stcox, {exact} (exact tie handling) has no sp.cox "
                "counterpart; breslow and efron do.",
                command="stcox",
                suggestions=[],
            )
    for unsupported in ("shared", "tvc", "texp", "offset"):
        if unsupported in opts:
            return _emit_error(
                f"stcox, {unsupported}() is not translated to sp.cox.",
                command="stcox",
                suggestions=[],
            )
    ties = "efron" if "efron" in opts else "breslow"
    opts.get("breslow")  # the default, written out below either way
    duration, event = _stset(cmd)
    args: Dict[str, Any] = {
        "duration": duration,
        "event": event,
        "x": xs,
        "ties": ties,
    }
    notes: List[str] = []
    untranslated: List[str] = []
    strata = (opts.get("strata") or "").split()
    if len(strata) > 1:
        return _emit_error(
            "stcox, strata() with several variables is not translated: "
            "sp.cox takes one strata column (group them into one first).",
            command="stcox",
            suggestions=[],
        )
    if strata:
        args["strata"] = strata[0]
    cluster = _vce_cluster(cmd)
    vce = (opts.get("vce") or "").split()
    if cluster:
        args["cluster"] = cluster
    elif "robust" in opts or (vce and vce[0].lower() == "robust"):
        untranslated.append("vce" if vce else "robust")
        notes.append(
            "stcox vce(robust) is the sandwich with Stata's N/(N-1) factor; "
            "sp.cox(robust='hc0') leaves the factor out (SEs smaller by "
            "sqrt((N-1)/N)). Not carried over."
        )
    if duration.startswith("<"):
        notes.append(_NEEDS_STSET.format(cmd="stcox"))
    code = ", ".join(["data=df"] + [f"{k}={v!r}" for k, v in args.items()])
    out = _emit(
        "cox",
        args,
        f"sp.cox({code})",
        notes,
        semantics=[
            f"stcox breaks ties with Breslow's method unless `efron` is "
            f"given; sp.cox defaults to Efron, so ties={ties!r} is written out."
        ],
    )
    if untranslated:
        out["untranslated_options"] = untranslated
    return out


def _h_stcrreg(cmd: StataCommand) -> Dict[str, Any]:
    """``stcrreg x1 x2, compete(cause == 2)`` -> ``sp.finegray``.

    ``stset time, failure(cause == 1)`` names the event of interest; the
    codes in ``compete()`` are the competing events and every other code is
    censoring. ``sp.finegray(event=cause, cause=1)`` reads every nonzero
    code other than ``cause`` as competing, so ``sp.stata`` checks that the
    event variable takes no code outside 0, 1 and the ``compete()`` codes.
    """
    xs = _covariates(cmd, "stcrreg")
    if isinstance(xs, dict):
        return xs
    opts = cmd.options
    raw = (opts.get("compete") or "").strip()
    m = re.fullmatch(r"([A-Za-z_]\w*)\s*==\s*(\d+(?:\s+\d+)*)", raw)
    if m is None:
        return _emit_error(
            "stcrreg needs compete(var == code [code ...]) on the failure "
            f"variable of stset; got compete({raw}).",
            command="stcrreg",
            suggestions=[],
        )
    var, codes = m.group(1), [int(c) for c in m.group(2).split()]
    if 1 in codes or 0 in codes:
        return _emit_error(
            "stcrreg compete(): codes 0 (censored) and 1 (the event of "
            "interest of `stset ..., failure(var == 1)`) cannot compete.",
            command="stcrreg",
            suggestions=[],
        )
    for unsupported in ("tvc", "texp", "offset", "noconstant"):
        if unsupported in opts:
            return _emit_error(
                f"stcrreg, {unsupported} is not translated to sp.finegray.",
                command="stcrreg",
                suggestions=[],
            )
    duration, event = _stset(cmd)
    if not event.startswith("<") and event != var:
        return _emit_error(
            f"stcrreg compete({raw}) is on {var!r}, but stset declared "
            f"failure({event} == 1); sp.finegray needs one variable coding "
            "the event of interest (1) and the competing events.",
            command="stcrreg",
            suggestions=[],
        )
    args: Dict[str, Any] = {
        "duration": duration,
        "event": var,
        "x": xs,
        "cause": 1,
        "small_sample": True,
    }
    notes: List[str] = []
    untranslated: List[str] = []
    vce = (opts.get("vce") or "").split()
    if _vce_cluster(cmd) or (vce and vce[0].lower() not in ("robust",)):
        untranslated.append("vce")
        notes.append(
            "stcrreg vce() other than robust is not carried over: "
            "sp.finegray reports the robust variance."
        )
    if duration.startswith("<"):
        notes.append(_NEEDS_STSET.format(cmd="stcrreg"))
    code = ", ".join(["data=df"] + [f"{k}={v!r}" for k, v in args.items()])
    out = _emit(
        "finegray",
        args,
        f"sp.finegray({code})",
        notes,
        semantics=[
            f"compete({raw}): codes {codes} of {var!r} compete with code 1; "
            "sp.finegray reads every other nonzero code as competing too, so "
            f"{var!r} must take no code besides 0, 1 and {codes}.",
            "small_sample=True: Stata's robust variance carries N/(N-1).",
        ],
    )
    out["requires_codes"] = {var: sorted({0, 1, *codes})}
    if untranslated:
        out["untranslated_options"] = untranslated
    return out


HANDLERS = {"stcox": _h_stcox, "stcrreg": _h_stcrreg}
