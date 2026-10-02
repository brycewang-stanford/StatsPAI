"""Translations of Stata's ``xt`` commands beyond ``xtreg, fe``.

``xtreg`` without ``fe`` (random effects, the between estimator, Gaussian
maximum likelihood), ``xtsum`` and ``xtserial``. All of them need the panel
declaration of ``xtset``, which lives on another line: the handlers accept
Stata's own ``i()`` / ``t()`` options, and ``sp.stata`` fills them in.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ._stata import (
    _build_formula,
    _emit,
    _emit_error,
    _robust_kind,
    _split_varlist_y_x,
    _vce_cluster,
)
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "xtreg_random"]

_NEEDS_PANEL = (
    "Couldn't recover the panel declaration from this command alone "
    "(Stata's `xtset id time` lives in another line). Replace <panel_id> "
    "and <panel_time> with the actual columns."
)


def _panel_of(cmd: StataCommand) -> tuple:
    unit = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    time = cmd.options.get("t") or "<panel_time>"
    return str(unit).strip(), str(time).strip()


def xtreg_random(cmd: StataCommand, y: str, xs: List[str]) -> Dict[str, Any]:
    """``xtreg y x [, re | be | mle]`` -> ``sp.panel`` / ``sp.mixed``."""
    opts = cmd.options
    unit, time = _panel_of(cmd)
    notes: List[str] = []
    if "<" in unit or "<" in time:
        notes.append(_NEEDS_PANEL)
    formula = _build_formula(y, xs)
    if "mle" in opts:
        if not xs:
            return _emit_error(
                "xtreg, mle needs at least one regressor",
                command="xtreg",
                suggestions=[],
            )
        args: Dict[str, Any] = {
            "formula": formula,
            "entity": unit,
            "time": time,
            "method": "mle",
        }
        kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "formula")
        return _emit(
            "panel",
            args,
            f"sp.panel(df, {formula!r}, {kw})",
            notes,
            semantics=[
                "xtreg, mle is the Gaussian random-effects model by maximum "
                "likelihood, with standard errors from the observed "
                "information; sigma_u, sigma_e and rho are in "
                "result.model_info."
            ],
        )
    method = "be" if "be" in opts else "re"
    opts.get("re")
    args = {
        "formula": formula,
        "entity": unit,
        "time": time,
        "method": method,
        "ssc": "stata",
    }
    cluster = _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster
    elif _robust_kind(cmd) == "hc1":
        if method == "be":
            return _emit_error(
                "xtreg, be does not allow vce(robust)",
                command="xtreg",
                suggestions=[],
            )
        args["robust"] = "robust"
    opts.get("theta")  # prints theta; it is in result.model_info
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "formula")
    semantics = [
        "ssc='stata' asks sp.panel for Stata's small-sample conventions "
        "(z statistics; with vce(robust) the variance is clustered on the "
        "panel with the factor G/(G-1) * (N-1)/(N-K))."
    ]
    return _emit(
        "panel", args, f"sp.panel(df, {formula!r}, {kw})", notes, semantics=semantics
    )


def _h_xtsum(cmd: StataCommand) -> Dict[str, Any]:
    """``xtsum varlist`` -> ``sp.xtsum``."""
    unit, _ = _panel_of(cmd)
    cmd.options.get("t")
    args: Dict[str, Any] = {"id": unit}
    if cmd.varlist:
        args["variables"] = list(cmd.varlist)
    notes = [_NEEDS_PANEL] if "<" in unit else []
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("xtsum", args, f"sp.xtsum(df, {kw})", notes)


def _h_xtserial(cmd: StataCommand) -> Dict[str, Any]:
    """``xtserial y x1 x2 [, output]`` -> ``sp.xtserial``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None or not xs:
        return _emit_error(
            "xtserial needs an outcome and at least one regressor",
            command="xtserial",
            suggestions=[],
        )
    unit, time = _panel_of(cmd)
    notes = [_NEEDS_PANEL] if "<" in unit or "<" in time else []
    args: Dict[str, Any] = {"y": y, "x": list(xs), "id": unit, "time": time}
    code = f"sp.xtserial(df, {y!r}, {list(xs)!r}, id={unit!r}, time={time!r})"
    return _emit(
        "xtserial",
        args,
        code,
        notes,
        semantics=[
            "`output` prints the first-difference regression: it is in the "
            "result's 'params' and 'std_errors'."
        ],
    )


HANDLERS = {"xtsum": _h_xtsum, "xtserial": _h_xtserial}
