"""Translations of comparative-case-study commands: ``rcm``.

``rcm`` (Yan and Chen's regression control method command) maps to
``sp.synth(method='rcm')``. Its panel declaration comes from ``xtset``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ._stata import (
    _PANEL_NOTE,
    _coerce_scalar,
    _emit,
    _emit_error,
    _numlist,
    _panel_options,
)
from ._stata_lexer import StataCommand, _parse_options

__all__ = ["HANDLERS"]


def _h_rcm(cmd: StataCommand) -> Dict[str, Any]:
    """``rcm y, trunit() trperiod() [counit() postperiod() method()
    criterion() placebo()]`` -> ``sp.synth(method='rcm')``."""
    if not cmd.varlist:
        return _emit_error("rcm requires an outcome variable", command="rcm")
    if len(cmd.varlist) > 1:
        return _emit_error(
            "rcm with covariates (the Hsiao-Zhou extension) is not translated; "
            "sp.synth(method='rcm') predicts from the control units' outcomes "
            "only.",
            command="rcm",
            suggestions=[],
        )
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
                lost.append("placebo")
                notes.append(
                    "placebo(unit(numlist)) restricts the pretend-treated "
                    "units; sp.synth(method='rcm') uses every control unit."
                )
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


HANDLERS = {"rcm": _h_rcm}
