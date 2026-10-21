"""Translations of Stata's time-series commands.

``prais``, ``corrgram`` / ``wntestq``, the ``var`` family and ``lpirf``'s
dynamic multipliers. All of them
read the series in the row order of the data; ``sp.stata`` sorts by the
``tsset`` time variable, a single translated line cannot.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Optional

from ._stata import _build_formula, _emit, _emit_error, _split_varlist_y_x
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "POSTEST"]

_ROW_ORDER = (
    "Rows are taken in the DataFrame's order; sort by the time variable of "
    "`tsset` first (sp.stata does)."
)


def _int_option(cmd: StataCommand, name: str) -> Optional[Any]:
    """An integer option: the value, ``None`` when absent, or an error."""
    raw = cmd.options.get(name)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return _emit_error(
            f"{cmd.command}: {name}({raw}) is not an integer",
            command=cmd.command,
            suggestions=[],
        )


def _h_prais(cmd: StataCommand) -> Dict[str, Any]:
    """``prais y x [, corc twostep rhotype() vce(robust)]`` -> ``sp.prais``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("prais requires an outcome variable", command="prais")
    formula = _build_formula(y, xs)
    opts = cmd.options
    if "noconstant" in opts:
        formula += " - 1"
    args: Dict[str, Any] = {"formula": formula}
    if "corc" in opts:
        args["method"] = "corc"
    if "twostep" in opts:
        args["twostep"] = True
    rhotype = opts.get("rhotype")
    if rhotype:
        args["rhotype"] = str(rhotype).strip().lower()
    vce = str(opts.get("vce") or "").strip().lower()
    if "robust" in opts or vce == "robust":
        args["vce"] = "robust"
    elif vce in ("hc2", "hc3"):
        args["vce"] = vce
    elif vce not in ("", "ols"):
        return _emit_error(
            f"prais: vce({vce}) is not translated", command="prais", suggestions=[]
        )
    shown = ", ".join(
        [repr(formula), "data=df"]
        + [f"{k}={v!r}" for k, v in args.items() if k != "formula"]
    )
    return _emit("prais", args, f"sp.prais({shown})", semantics=[_ROW_ORDER])


def _h_corrgram(cmd: StataCommand) -> Dict[str, Any]:
    """``corrgram y [, lags(#) yw]`` and ``wntestq y [, lags(#)]`` ->
    ``sp.corrgram``."""
    if len(cmd.varlist) != 1:
        return _emit_error(
            f"{cmd.command} takes one variable", command=cmd.command, suggestions=[]
        )
    args: Dict[str, Any] = {"y": cmd.varlist[0]}
    lags = _int_option(cmd, "lags")
    if isinstance(lags, dict):
        return lags
    if lags is not None:
        args["lags"] = lags
    semantics: List[str] = [_ROW_ORDER]
    if cmd.command == "wntestq":
        semantics.append(
            "wntestq is the Q statistic of the last row of the correlogram."
        )
    elif "yw" in cmd.options:
        args["pac"] = "yw"
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "y")
    code = f"sp.corrgram(df, {args['y']!r}" + (f", {kw})" if kw else ")")
    return _emit("corrgram", args, code, semantics=semantics)


def _contiguous_lags(raw: Optional[str], command: str) -> Any:
    """``lags(1/3)`` / ``lags(1 2 3)`` -> 3. A list that skips a lag
    (``lags(2)`` is lag 2 alone in Stata) has no counterpart."""
    import re

    if raw is None:
        return 2  # Stata's default is lags(1 2)
    text = str(raw).strip()
    m = re.fullmatch(r"1\s*/\s*(\d+)", text)
    if m:
        return int(m.group(1))
    parts = text.split()
    if all(p.isdigit() for p in parts) and [int(p) for p in parts] == list(
        range(1, len(parts) + 1)
    ):
        return len(parts)
    return _emit_error(
        f"{command}: lags({text}) leaves out a lag; only lags(1/p) is "
        "translated (sp.var includes every lag from 1 to p)",
        command=command,
        suggestions=[],
    )


def _var_names(cmd: StataCommand) -> Any:
    names = list(cmd.varlist)
    if not names:
        return _emit_error(
            f"{cmd.command} needs at least one variable",
            command=cmd.command,
            suggestions=[],
        )
    return names


def _h_var(cmd: StataCommand) -> Dict[str, Any]:
    """``var y1 y2, lags(1/p)`` and ``varbasic`` -> ``sp.var``."""
    names = _var_names(cmd)
    if isinstance(names, dict):
        return names
    lags = _contiguous_lags(cmd.options.get("lags"), cmd.command)
    if isinstance(lags, dict):
        return lags
    args: Dict[str, Any] = {"variables": names, "lags": lags}
    if "noconstant" in cmd.options:
        args["trend"] = "n"
    if cmd.options.get("exog"):
        args["exog"] = str(cmd.options["exog"]).split()
    semantics = [
        _ROW_ORDER,
        "Standard errors use the maximum-likelihood residual covariance "
        "(divisor T), Stata's default; sp.var(se_df='r') gives the "
        "equation-by-equation OLS ones.",
    ]
    if cmd.command == "varbasic":
        for name in ("irf", "oirf", "fevd", "nograph", "step"):
            cmd.options.get(name)
        semantics.append(
            "varbasic also draws impulse responses: call result.irf() / "
            "result.plot_irf()."
        )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("var", args, f"sp.var(df, {kw})", semantics=semantics)


def _h_varsoc(cmd: StataCommand) -> Dict[str, Any]:
    names = _var_names(cmd)
    if isinstance(names, dict):
        return names
    args: Dict[str, Any] = {"variables": names}
    maxlag = _int_option(cmd, "maxlag")
    if isinstance(maxlag, dict):
        return maxlag
    if maxlag is not None:
        args["maxlag"] = maxlag
    if "noconstant" in cmd.options:
        args["trend"] = "n"
    if cmd.options.get("exog"):
        args["exog"] = str(cmd.options["exog"]).split()
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("varsoc", args, f"sp.varsoc(df, {kw})", semantics=[_ROW_ORDER])


def _h_var_postest(cmd: StataCommand) -> Dict[str, Any]:
    """``varwle`` / ``varlmar`` / ``varstable`` / ``vargranger`` ->
    ``sp.estat(result, ...)`` on the VAR in memory."""
    if cmd.varlist or cmd.if_cond or cmd.in_range:
        return _emit_error(
            f"{cmd.command} takes no variables", command=cmd.command, suggestions=[]
        )
    args: Dict[str, Any] = {"test": cmd.command}
    if cmd.command in ("varlmar", "veclmar"):
        mlag = _int_option(cmd, "mlag")
        if isinstance(mlag, dict):
            return mlag
        if mlag is not None:
            args["lags"] = mlag
    if cmd.command in ("varstable", "vecstable"):
        cmd.options.get("graph")
    args["print_results"] = False
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("estat", args, f"sp.estat(result, {shown})")


_VEC_TRENDS = {
    "none": "n",
    "rconstant": "rc",
    "constant": "c",
    "rtrend": "rt",
    "trend": "ct",
}


def _vec_common(cmd: StataCommand) -> Any:
    """Variables, lagged differences and trend of ``vecrank`` / ``vec``."""
    names = _var_names(cmd)
    if isinstance(names, dict):
        return names
    lags = _int_option(cmd, "lags")
    if isinstance(lags, dict):
        return lags
    lags = 2 if lags is None else lags
    if lags < 1:
        return _emit_error(
            f"{cmd.command}: lags({lags}) must be at least 1",
            command=cmd.command,
            suggestions=[],
        )
    raw = str(cmd.options.get("trend") or "constant").strip().lower()
    hits = [full for full in _VEC_TRENDS if full.startswith(raw)]
    if raw not in _VEC_TRENDS and len(hits) != 1:
        return _emit_error(
            f"{cmd.command}: trend({raw}) is not one of " + ", ".join(_VEC_TRENDS),
            command=cmd.command,
            suggestions=[],
        )
    trend = _VEC_TRENDS[raw if raw in _VEC_TRENDS else hits[0]]
    # Stata counts lags of the levels; sp counts lagged differences
    args: Dict[str, Any] = {"variables": names, "lags": lags - 1}
    if trend != "c":
        args["trend"] = trend
    return args


_LAGS_NOTE = (
    "Stata's lags(p) counts lags of the levels; sp counts lagged "
    "differences, so lags = p - 1."
)


def _h_vecrank(cmd: StataCommand) -> Dict[str, Any]:
    """``vecrank y1 y2, lags(p) trend()`` -> ``sp.johansen``."""
    args = _vec_common(cmd)
    if "ok" in args:
        return args
    semantics = [_ROW_ORDER, _LAGS_NOTE]
    if "max" in cmd.options:
        semantics.append(
            "The trace statistics are returned; `max` also prints the "
            "maximum-eigenvalue ones: sp.johansen(..., test='maxeig')."
        )
    level99 = "level99" in cmd.options
    if level99:
        args["alpha"] = 0.01
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("johansen", args, f"sp.johansen(df, {kw})", semantics=semantics)


def _h_vec(cmd: StataCommand) -> Dict[str, Any]:
    """``vec y1 y2, lags(p) rank(r) trend()`` -> ``sp.vec``."""
    args = _vec_common(cmd)
    if "ok" in args:
        return args
    rank = _int_option(cmd, "rank")
    if isinstance(rank, dict):
        return rank
    if rank is not None:
        args["rank"] = rank
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("vec", args, f"sp.vec(df, {kw})", semantics=[_ROW_ORDER, _LAGS_NOTE])


_LPIRF_FN = ["local_projections"]


def _lpirf_lags(raw: Optional[str]) -> Any:
    """``lags(numlist)`` -> int ``p`` for ``1/p``, else the sorted list."""
    import re

    if raw is None:
        return 2  # Stata's default is lags(1 2)
    out: List[int] = []
    for part in str(raw).replace(",", " ").split():
        m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", part)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            out.append(int(part))
        else:
            return None
    lags = sorted(set(out))
    if not lags or lags[0] != 1:
        return None
    return len(lags) if lags == list(range(1, len(lags) + 1)) else lags


def _h_lpirf(cmd: StataCommand) -> Dict[str, Any]:
    """``lpirf y, exog(x) [lags() step() vce() dfk small level()]`` ->
    ``sp.local_projections`` (the dynamic multiplier of ``x`` on ``y``).

    Checked against Stata 18 (``tests/reference_parity/
    test_ts_causal_parity.py``): every vce type, ``dfk``, skipped lags.
    """
    deps = list(cmd.varlist)
    exog = str(cmd.options.get("exog") or "").split()
    template = (
        "sp.local_projections(df, outcome=<response>, shock=<exog variable>, "
        "controls=<other exog variables>, endog_order=<depvarlist>, "
        "lags=<lags()>, horizons=<step() - 1>, common_sample=True, "
        "vce='ols' | 'robust' | 'hac', dfk=<dfk given>)"
    )
    if not exog:
        return _emit_error(
            "'lpirf' without exog() is not translated line by line: its "
            "responses to the dependent variables are their own first-lag "
            "coefficients, a reduced-form projection; for an observed or "
            f"instrumented shock call {template}",
            command="lpirf",
            suggestions=[],
            statspai_functions=_LPIRF_FN,
        )
    if len(deps) != 1 or len(exog) != 1:
        return _emit_error(
            "'lpirf' with several responses or exog() variables is one "
            f"projection per pair; write one call each: {template}",
            command="lpirf",
            suggestions=[],
            statspai_functions=_LPIRF_FN,
        )
    if any("." in v for v in deps + exog):
        return _emit_error(
            "lpirf: time-series operators in the variable lists are not "
            "translated; create the lagged column first",
            command="lpirf",
            suggestions=[],
            statspai_functions=_LPIRF_FN,
        )
    lags = _lpirf_lags(cmd.options.get("lags"))
    if lags is None:
        return _emit_error(
            f"lpirf: lags({cmd.options.get('lags')}) must be a numlist that "
            "includes lag 1",
            command="lpirf",
            suggestions=[],
        )
    step = _int_option(cmd, "step")
    if isinstance(step, dict):
        return step
    step = 8 if step is None else step
    if step < 1:
        return _emit_error("lpirf: step() must be positive", command="lpirf")
    args: Dict[str, Any] = {
        "outcome": deps[0],
        "shock": exog[0],
        "horizons": step - 1,
        "lags": lags,
        "endog_order": [deps[0]],
        "common_sample": True,
    }
    vce = " ".join(str(cmd.options.get("vce") or "").lower().split())
    if vce in ("", "ols"):
        args["vce"] = "ols"
    elif vce in ("r", "ro", "rob", "robu", "robus", "robust"):
        args["vce"] = "robust"
    else:
        parts = vce.split()
        if (
            parts[0] != "hac"
            or len(parts) not in (2, 3)
            or parts[1]
            not in (
                "nwest",
                "bartlett",
            )
        ):
            return _emit_error(
                f"lpirf: vce({vce}) is not translated (Newey-West HAC, robust "
                "and ols are)",
                command="lpirf",
                suggestions=[],
            )
        args["vce"] = "hac"
        if len(parts) == 3 and not parts[2].isdigit():
            return _emit_error(
                f"lpirf: vce({vce}) lag is not an integer", command="lpirf"
            )
        args["nw_lags"] = int(parts[2]) if len(parts) == 3 else step
    args["dfk"] = "dfk" in cmd.options
    level = cmd.options.get("level")
    if level is not None:
        try:
            args["alpha"] = round(1 - float(str(level).strip()) / 100, 10)
        except ValueError:
            return _emit_error(
                f"lpirf: level({level}) is not a number", command="lpirf"
            )
    semantics = [
        _ROW_ORDER,
        "irf[h] is lpirf's step h of the exog() block of e(b) (step 0 is the "
        "impact); the responses to the dependent variables' own lags, the "
        "other rows of lpirf's table, are not part of this call.",
    ]
    if "small" in cmd.options:
        semantics.append(
            "small: Stata reports t statistics; the coefficients and standard "
            "errors are the same, the bands here use the normal."
        )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit(
        "local_projections",
        args,
        f"sp.local_projections(df, {kw})",
        semantics=semantics,
    )


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "prais": _h_prais,
    "corrgram": _h_corrgram,
    "wntestq": _h_corrgram,
    "var": _h_var,
    "varbasic": _h_var,
    "varsoc": _h_varsoc,
    "varwle": _h_var_postest,
    "varlmar": _h_var_postest,
    "varstable": _h_var_postest,
    "vargranger": _h_var_postest,
    "veclmar": _h_var_postest,
    "vecstable": _h_var_postest,
    "vecrank": _h_vecrank,
    "vec": _h_vec,
    "lpirf": _h_lpirf,
}
POSTEST: FrozenSet[Callable[..., Dict[str, Any]]] = frozenset({_h_var_postest})
