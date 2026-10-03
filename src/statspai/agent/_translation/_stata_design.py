"""Translations of comparative-case-study commands: ``rcm`` and ``synth2``.

``rcm`` (Yan and Chen's regression control method command) maps to
``sp.synth(method='rcm')``; ``synth2`` (their wrapper of ``synth`` with
placebo and leave-one-out reports) to ``sp.synth(method='classic')`` with
the same reports. The panel declaration comes from ``xtset``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ._stata import (
    _PANEL_NOTE,
    _coerce_scalar,
    _emit,
    _emit_error,
    _h_synth,
    _numlist,
    _panel_options,
)
from ._stata_lexer import StataCommand, _parse_options

__all__ = ["HANDLERS"]


def _h_rcm(cmd: StataCommand) -> Dict[str, Any]:
    """``rcm y [covariates], trunit() trperiod() [counit() postperiod()
    method() criterion() placebo()]`` -> ``sp.synth(method='rcm')``."""
    if not cmd.varlist:
        return _emit_error("rcm requires an outcome variable", command="rcm")
    opts = cmd.options
    trunit, trperiod = opts.get("trunit"), opts.get("trperiod")
    if not (trunit and trperiod):
        return _emit_error(
            "rcm needs `trunit(<id>)` and `trperiod(<period>)`.", command="rcm"
        )
    unit, time = _panel_options(cmd)
    notes: List[str] = []
    lost: List[str] = []
    args: Dict[str, Any] = {
        "outcome": cmd.varlist[0],
        "unit": unit or "<panel_id>",
        "time": time or "<panel_time>",
        "treated_unit": _coerce_scalar(trunit),
        "treatment_time": _coerce_scalar(trperiod),
        "method": "rcm",
    }
    if len(cmd.varlist) > 1:
        # every unit's covariates join the candidate predictors
        args["covariates"] = list(cmd.varlist[1:])
    for stata_name, key in (
        ("counit", "donors"),
        ("ctrlunit", "donors"),
        ("preperiod", "pre_periods"),
        ("postperiod", "post_periods"),
    ):
        raw = opts.get(stata_name)
        if raw is None:
            continue
        values = _numlist(raw)
        if values is None:
            lost.append(stata_name)
        else:
            args[key] = values
    selection = str(opts.get("method") or "best").strip().lower()
    if selection not in ("best", "forward", "backward"):
        return _emit_error(
            f"rcm, method({selection}) is not translated; sp.synth(method="
            "'rcm') selects by best subset, forward or backward stepwise.",
            command="rcm",
            suggestions=[],
        )
    if selection != "best":
        args["selection"] = selection
    criterion = str(opts.get("criterion") or "aicc").strip().lower()
    if criterion not in ("aicc", "aic", "bic", "mbic"):
        return _emit_error(
            f"rcm, criterion({criterion}) is not translated (aicc, aic, bic "
            "and mbic are).",
            command="rcm",
            suggestions=[],
        )
    if criterion != "aicc":
        args["criterion"] = criterion
    estimate = opts.get("estimate")
    if estimate is not None and str(estimate).strip().lower() != "ols":
        lost.append("estimate")

    placebo = opts.get("placebo")
    args["placebo"] = False
    if placebo is not None:
        sub = _parse_options(placebo)
        if "unit" in sub:
            if sub["unit"]:
                chosen = _numlist(sub["unit"])
                if chosen is None:
                    lost.append("placebo")
                else:
                    args["placebo_units"] = chosen
            args["placebo"] = True
        if sub.get("cut") is not None:
            try:
                args["placebo_cutoff"] = float(sub["cut"])  # type: ignore[arg-type]
            except ValueError:
                lost.append("placebo")
        if sub.get("period") is not None:
            periods = _numlist(sub["period"] or "")
            if periods is None or len(periods) != 1:
                lost.append("placebo")
                notes.append("placebo(period()) takes one pretend treatment date.")
            else:
                args["placebo_time"] = periods[0]
    for name in ("scope", "fill", "grid", "fold", "seed", "frame"):
        if opts.get(name) is not None and name not in ("frame", "seed"):
            lost.append(name)
    if unit is None or time is None:
        notes.append(_PANEL_NOTE)
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit(
        "synth",
        args,
        f"sp.synth(data=df, {kw})",
        notes,
        semantics=[
            "The model-selection table is result.model_info['selection_table'], "
            "the pre-treatment regression model_info['coefficients'], the "
            "period-by-period effects result.detail."
        ],
    )
    out["untranslated_options"] = lost
    return out


def _h_synth2(cmd: StataCommand) -> Dict[str, Any]:
    """``synth2 y predictors, trunit() trperiod() [nested placebo() loo
    postperiod()]`` -> ``sp.synth(method='classic', ...)``.

    The fit is that of ``synth``; ``placebo(unit cut(c))``,
    ``placebo(period(t))``, ``loo`` and ``postperiod()`` become
    ``placebo=True, placebo_cutoff=c``, ``placebo_time=t``, ``loo=True`` and
    ``post_periods=[...]``.
    """
    out = _h_synth(cmd)
    if not out.get("ok"):
        if "error" in out:
            out["error"] = str(out["error"]).replace("synth ", "synth2 ", 1)
        return out
    opts = cmd.options
    args: Dict[str, Any] = dict(out["arguments"])
    notes: List[str] = list(out.get("notes") or [])
    lost: List[str] = list(out.get("untranslated_options") or [])

    placebo = opts.get("placebo")
    if placebo is not None:
        sub = _parse_options(placebo)
        known = {"unit", "cut", "cutoff", "period", "show"}
        if set(sub) - known:
            lost.append("placebo")
        if "unit" in sub:
            if sub["unit"]:
                chosen = _numlist(sub["unit"])
                if chosen is None:
                    lost.append("placebo")
                else:
                    args["placebo_units"] = chosen
            args["placebo"] = True
        cut = sub.get("cut", sub.get("cutoff"))
        if cut is not None:
            try:
                args["placebo_cutoff"] = float(cut)  # type: ignore[arg-type]
            except ValueError:
                lost.append("placebo")
        if sub.get("period") is not None:
            periods = _numlist(sub["period"] or "")
            if periods is None or len(periods) != 1:
                lost.append("placebo")
                notes.append(
                    "placebo(period()) with several dates: pass one "
                    "placebo_time per call."
                )
            else:
                args["placebo_time"] = periods[0]
    if "loo" in opts:
        args["loo"] = True
    raw = opts.get("postperiod")
    if raw is not None:
        values = _numlist(raw)
        if values is None:
            lost.append("postperiod")
        else:
            args["post_periods"] = values
    raw = opts.get("preperiod")
    if raw is not None:
        values = _numlist(raw)
        if values is None:
            lost.append("preperiod")
        else:
            args["pre_periods"] = values
    if opts.get("ctrlunit") is not None:
        lost.append("ctrlunit")
    if args.get("v_method") == "nested":
        notes.append(
            "The nested search is a non-convex problem: its solutions, here "
            "and in every placebo and leave-one-out run, need not be Stata's."
        )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    result = _emit(
        "synth",
        args,
        f"sp.synth(data=df, {kw})",
        notes,
        semantics=[
            "The unit table is result.model_info['placebo_table'], the "
            "period-by-period p-values model_info['placebo_effects'], the "
            "pretend treatment date model_info['placebo_time'] and the "
            "leave-one-out range model_info['loo'].",
            "synth2's R-squared divides by the variation of the synthetic "
            "path; model_info['pre_r2'] divides by the variation of the "
            "treated unit's outcome.",
        ],
    )
    result["untranslated_options"] = sorted(set(lost))
    return result


HANDLERS = {"rcm": _h_rcm, "synth2": _h_synth2}
