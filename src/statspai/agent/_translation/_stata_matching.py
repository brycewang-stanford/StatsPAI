"""Becker and Ichino's ``pscore`` / ``attnd``, McCrary's ``DCdensity`` and
``loneway``.

``pscore`` adds variables (the score, the block, ``comsup``); the call
translated here returns them in its result and ``sp.stata`` writes them
into the data in memory under the names the command gave.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ._stata import _emit, _emit_error
from ._stata_basics import _bad, _kw, _level, _number
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "pscore_variables"]


def _names(cmd: StataCommand) -> List[str]:
    return [v for v in cmd.varlist if v not in ("(", ")") and not v.startswith("[")]


def pscore_variables(cmd: StataCommand) -> Dict[str, Any]:
    """The variables a ``pscore`` line creates: name -> what it holds."""
    out: Dict[str, Any] = {}
    score = (cmd.options.get("pscore") or "").strip()
    if score:
        out[score] = "pscore"
    block = (cmd.options.get("blockid") or "").strip()
    if block:
        out[block] = "block"
    if "comsup" in cmd.options:
        out["comsup"] = "support"
    return out


def _h_pscore(cmd: StataCommand) -> Dict[str, Any]:
    """``pscore d x1 x2, pscore(ps) [blockid(b) logit comsup level(#)
    numblo(#)]`` -> ``sp.pscore``."""
    names = _names(cmd)
    if len(names) < 2:
        return _bad(cmd, "expected `pscore treatment covariates, pscore(newvar)`")
    if not (cmd.options.get("pscore") or "").strip():
        return _bad(cmd, "pscore(newvar) names the variable for the score")
    cmd.options.get("blockid")
    args: Dict[str, Any] = {"treat": names[0], "covariates": names[1:]}
    notes: List[str] = []
    if "logit" in cmd.options:
        args["ps_model"] = "logit"
    else:
        args["ps_model"] = "probit"
        notes.append(
            "Stata pscore fits a probit unless `logit` is given: ps_model='probit'."
        )
    if "comsup" in cmd.options:
        args["common_support"] = True
    lost: List[str] = []
    if cmd.options.get("level") is not None:
        value = _number(cmd.options.get("level"))
        if value is None or not 0 < value < 1:
            lost.append("level")
        else:
            args["level"] = value
    if cmd.options.get("numblo") is not None:
        value = _number(cmd.options.get("numblo"))
        if value is None or value < 1 or value != int(value):
            lost.append("numblo")
        else:
            args["n_blocks"] = int(value)
    created = pscore_variables(cmd)
    if created:
        notes.append(
            "The command stores "
            + ", ".join(f"{name} ({what})" for name, what in created.items())
            + ": result.assign(df, ...) adds them; sp.stata does it."
        )
    out = _emit("pscore", args, f"sp.pscore(df, {_kw(args)})", notes)
    out["untranslated_options"] = lost
    return out


def _h_attnd(cmd: StataCommand) -> Dict[str, Any]:
    """``attnd y d [x1 x2], [pscore(ps) logit comsup bootstrap reps(#)]``
    -> ``sp.psmatch2(..., ties=True)``.

    Nearest-neighbour matching on the score with every equally close
    control kept, and the analytic standard error both commands share.
    """
    names = _names(cmd)
    if len(names) < 2:
        return _bad(cmd, "expected `attnd outcome treatment [covariates]`")
    outcome, treat, covariates = names[0], names[1], names[2:]
    given = (cmd.options.get("pscore") or "").strip()
    args: Dict[str, Any] = {"treat": treat, "outcome": outcome}
    notes: List[str] = [
        "attnd keeps every control tied at the smallest distance (ties=True). "
        "A treated unit exactly half way between a control below and one "
        "above gets one of them at random in attnd and both here."
    ]
    lost: List[str] = []
    if given:
        args["pscore"] = given
        if covariates:
            args["covariates"] = covariates
        cmd.options.get("logit")
    elif covariates:
        args["covariates"] = covariates
        if "logit" in cmd.options:
            args["ps_model"] = "logit"
        else:
            args["ps_model"] = "probit"
            notes.append(
                "attnd fits a probit unless `logit` is given: ps_model='probit'."
            )
    else:
        return _bad(cmd, "give the covariates of the score or pscore(varname)")
    if "index" in cmd.options:
        # the matching would run on the linear index, not the probability
        lost.append("index")
    args["ties"] = True
    if "comsup" in cmd.options:
        args["common_support"] = "treated"
    if "bootstrap" in cmd.options:
        args["se"] = "bootstrap"
        reps = cmd.options.get("reps")
        if reps is not None:
            value = _number(reps)
            if value is None or value < 2 or value != int(value):
                lost.append("reps")
            else:
                args["bootstrap_reps"] = int(value)
        else:
            args["bootstrap_reps"] = 50  # the default of Stata's bootstrap
        notes.append(
            "The bootstrap draws come from numpy; the standard error agrees "
            "with Stata's up to simulation error."
        )
    out = _emit("psmatch2", args, f"sp.psmatch2(data=df, {_kw(args)})", notes)
    out["untranslated_options"] = lost
    return out


def _h_dcdensity(cmd: StataCommand) -> Dict[str, Any]:
    """``DCdensity x, breakpoint(c) [b(#) h(#)]`` -> ``sp.mccrary_test``."""
    names = _names(cmd)
    if len(names) != 1:
        return _bad(cmd, "expected `DCdensity runningvar, breakpoint(#)`")
    cut = _number(cmd.options.get("breakpoint"))
    if cut is None:
        return _bad(cmd, "breakpoint(#) is required")
    args: Dict[str, Any] = {"x": names[0], "c": cut}
    lost: List[str] = []
    for option, name in (("b", "bin_width"), ("h", "bw")):
        if cmd.options.get(option) is not None:
            value = _number(cmd.options.get(option))
            if value is None or value <= 0:
                lost.append(option)
            else:
                args[name] = value
    notes: List[str] = []
    if cmd.options.get("generate") is not None:
        notes.append(
            "generate() stores the histogram and the fitted density for a "
            "graph; those variables are not created."
        )
    out = _emit("mccrary_test", args, f"sp.mccrary_test(df, {_kw(args)})", notes)
    out["untranslated_options"] = lost
    return out


def _h_loneway(cmd: StataCommand) -> Dict[str, Any]:
    """``loneway y g [, level(#) exact]`` -> ``sp.loneway``."""
    names = _names(cmd)
    if len(names) != 2:
        return _bad(cmd, "expected `loneway response group`")
    args: Dict[str, Any] = {"y": names[0], "by": names[1]}
    bad = _level(cmd, args)
    if bad is not None:
        return bad
    if "exact" in cmd.options:
        args["exact"] = True
    return _emit("loneway", args, f"sp.loneway(df, {_kw(args)})")


def _h_pstest(cmd: StataCommand) -> Dict[str, Any]:
    return _emit_error(
        "pstest reads the variables the last psmatch2 left behind: call "
        ".pstest([covariates]) on the result of sp.psmatch2. sp.stata runs "
        "the line after a psmatch2.",
        command="pstest",
        suggestions=[],
    )


HANDLERS = {
    "pscore": _h_pscore,
    "attnd": _h_attnd,
    "dcdensity": _h_dcdensity,
    "loneway": _h_loneway,
    "pstest": _h_pstest,
}
