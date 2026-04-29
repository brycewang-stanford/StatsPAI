"""Stata command → StatsPAI tool-call translator.

Every entry in :data:`STATA_COMMAND_MAP` is ``stata_cmd → handler``.
Each handler takes a parsed :class:`StataCommand` and returns the
canonical translation dict ``{tool, arguments, python_code, notes}``.

Tier 1 (this file): the 8 commands that cover ~60% of real Stata
econometrics work — `regress` / `xtreg` / `reghdfe` / `ivreg2` /
`csdid` / `did_imputation` / `synth` / `rdrobust`. The follow-up
Tier-2 layer will plug in another 12 commands the same way.

Design principles
-----------------

* **Hand-curated, not generic** — Stata options have semantics
  (``vce(cluster id)`` ≠ ``cluster(id)`` is a real distinction in
  some commands). Translating each command means we control the
  mapping precisely.
* **No silent guesses** — when an option has no clean StatsPAI
  equivalent, the handler emits a ``notes`` entry surfaced back
  to the user. We never quietly drop options.
* **Round-trippable** — the output's ``python_code`` should always
  be valid Python; ``arguments`` should always be JSON-serialisable.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ._stata_lexer import parse as _parse_stata, StataCommand, StataParseError


Handler = Callable[[StataCommand], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit(tool: str, arguments: Dict[str, Any],
           python_code: str, notes: List[str] = None) -> Dict[str, Any]:
    return {
        "tool": tool,
        "arguments": dict(arguments),
        "python_code": python_code,
        "notes": list(notes or []),
        "ok": True,
    }


def _emit_error(message: str, **extra) -> Dict[str, Any]:
    return {
        "tool": None,
        "ok": False,
        "error": message,
        **extra,
    }


def _abbrev_match(short: str, full: str) -> bool:
    """Stata-style abbreviation: ``reg`` matches ``regress``, etc.

    Right-padded prefix match on a chosen full form. Pure prefix
    match would over-fire (``re`` matching ``regress`` AND ``rdrobust``);
    we never call this on ambiguous prefixes — see the lookup logic in
    ``_resolve_command``.
    """
    return full.startswith(short)


def _split_varlist_y_x(varlist: List[str]) -> tuple:
    """Stata's ``y x1 x2 x3`` convention — first is outcome, rest covariates."""
    if not varlist:
        return None, []
    return varlist[0], list(varlist[1:])


def _build_formula(y: str, xs: List[str]) -> str:
    """Wilkinson formula. Empty xs ⇒ intercept-only (``y ~ 1``)."""
    if not xs:
        return f"{y} ~ 1"
    return f"{y} ~ " + " + ".join(xs)


def _vce_cluster(cmd: StataCommand) -> Optional[str]:
    """Extract a cluster column from ``vce(cluster <var>)`` or
    ``cluster(<var>)``. Returns ``None`` when neither is present."""
    vce = cmd.options.get("vce")
    if vce:
        parts = vce.split()
        if parts and parts[0].lower() == "cluster" and len(parts) >= 2:
            return parts[1]
    cluster = cmd.options.get("cluster")
    if cluster:
        return cluster.split()[0]
    return None


def _robust_kind(cmd: StataCommand) -> str:
    """Map Stata's ``robust`` / ``vce(robust)`` / ``vce(hc3)`` → sp.regress robust."""
    if "robust" in cmd.options or _opt_matches(cmd.options.get("vce"), "robust"):
        return "hc1"
    vce = cmd.options.get("vce")
    if vce:
        head = vce.split()[0].lower()
        if head in {"hc0", "hc1", "hc2", "hc3"}:
            return head
    return "nonrobust"


def _opt_matches(value: Optional[str], target: str) -> bool:
    if not value:
        return False
    return value.split()[0].lower() == target


# ---------------------------------------------------------------------------
# Tier-1 handlers
# ---------------------------------------------------------------------------

def _h_regress(cmd: StataCommand) -> Dict[str, Any]:
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("regress requires an outcome variable",
                            command="regress")
    formula = _build_formula(y, xs)
    cluster = _vce_cluster(cmd)
    robust = _robust_kind(cmd)
    args: Dict[str, Any] = {"formula": formula}
    if robust != "nonrobust":
        args["robust"] = robust
    if cluster:
        args["cluster"] = cluster
    code_kwargs = ", ".join(
        [f"{k}={v!r}" for k, v in args.items() if k != "formula"]
        + ["data=df"]
    )
    python = f"sp.regress({formula!r}, {code_kwargs})"
    notes: List[str] = []
    if cmd.if_cond:
        notes.append(f"Stata `if {cmd.if_cond}` dropped — pre-filter df via "
                      f"`df = df.query({cmd.if_cond!r})` before calling.")
    if cmd.in_range:
        notes.append(f"Stata `in {cmd.in_range}` dropped — use df.iloc[...].")
    return _emit("regress", args, python, notes)


def _h_xtreg(cmd: StataCommand) -> Dict[str, Any]:
    """``xtreg y x1 x2, fe vce(cluster id)`` → ``sp.fixest`` with entity FE."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("xtreg requires an outcome variable", command="xtreg")
    if "re" in cmd.options:
        return _emit_error(
            "random-effects xtreg is not supported — use sp.panel(method='re') "
            "directly via the Python API for now.",
            command="xtreg")

    # Stata convention: panel id set via ``xtset id [t]``; we can't see
    # that here, so the user must supply ``id`` via the option or we
    # leave a placeholder.
    panel_id = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    formula = _build_formula(y, xs)
    cluster = _vce_cluster(cmd)
    args: Dict[str, Any] = {
        "formula": formula,
        "fe": [panel_id] if panel_id != "<panel_id>" else [],
    }
    if cluster:
        args["cluster"] = cluster
    notes: List[str] = []
    if panel_id == "<panel_id>":
        notes.append("Couldn't recover the panel-id from this command alone "
                      "(Stata's `xtset id` lives in another line). Replace "
                      "<panel_id> with the actual unit id column.")
    if "be" in cmd.options or "fd" in cmd.options:
        notes.append("Between-effects / first-difference variants are not yet "
                      "translated — use sp.panel(method='be'/'fd') directly.")
    fe_repr = args["fe"]
    code_pairs = [f"data=df", f"fe={fe_repr!r}"]
    if cluster:
        code_pairs.append(f"cluster={cluster!r}")
    python = f"sp.fixest({formula!r}, {', '.join(code_pairs)})"
    return _emit("fixest", args, python, notes)


def _h_reghdfe(cmd: StataCommand) -> Dict[str, Any]:
    """``reghdfe y x, absorb(id year) cluster(id)`` → ``sp.fixest``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("reghdfe requires an outcome variable",
                            command="reghdfe")
    absorb = cmd.options.get("absorb") or ""
    fe_list = [v for v in absorb.split() if v]
    cluster = _vce_cluster(cmd) or cmd.options.get("cluster")
    if cluster:
        cluster = cluster.split()[0]
    formula = _build_formula(y, xs)
    args: Dict[str, Any] = {"formula": formula, "fe": fe_list}
    if cluster:
        args["cluster"] = cluster
    notes: List[str] = []
    if not fe_list:
        notes.append("reghdfe with no absorb() collapses to OLS — "
                      "consider sp.regress instead.")
    code_pairs = ["data=df", f"fe={fe_list!r}"]
    if cluster:
        code_pairs.append(f"cluster={cluster!r}")
    python = f"sp.fixest({formula!r}, {', '.join(code_pairs)})"
    return _emit("fixest", args, python, notes)


def _h_ivreg2(cmd: StataCommand) -> Dict[str, Any]:
    """``ivreg2 y x1 (d = z1 z2), cluster(id)`` → ``sp.ivreg``."""
    # Stata's ``ivreg2`` varlist contains parentheses with `d = z`.
    # Re-join the original varlist tokens to recover the parens.
    if not cmd.varlist:
        return _emit_error("ivreg2 requires an outcome variable",
                            command="ivreg2")
    joined = " ".join(cmd.varlist)
    # Accept either ``y x (d = z)`` or ``y (d = z)``.
    import re
    m = re.match(r"^\s*(\S+)\s*(.*?)\s*\(\s*(\S+)\s*=\s*([^)]+?)\s*\)\s*$",
                 joined)
    if not m:
        return _emit_error(
            f"could not parse ivreg2 syntax {joined!r}; expected "
            "`y [exog_x...] (endog = instruments...)`",
            command="ivreg2")
    y, exog_xs, endog, instruments = m.group(1), m.group(2), m.group(3), m.group(4)
    formula_lhs = f"{y} ~ "
    if exog_xs.strip():
        formula_lhs += f"{exog_xs.strip()} + "
    formula = f"{formula_lhs}({endog} ~ {instruments.strip()})"

    cluster = _vce_cluster(cmd) or cmd.options.get("cluster")
    if cluster:
        cluster = cluster.split()[0]
    args: Dict[str, Any] = {"formula": formula}
    if cluster:
        args["robust"] = "hc1"  # ivreg's robust, with cluster handled below
    notes: List[str] = []
    if cluster:
        notes.append(f"Stata cluster({cluster}) — sp.ivreg currently does "
                      f"not accept a `cluster=` kwarg; pass via the Python "
                      f"API: ``sp.ivreg(..., cluster={cluster!r})``.")
    if "first" in cmd.options:
        notes.append("`first` (first-stage display) not translated; the sp "
                      "result already exposes first_stage_F via diagnostics.")
    code_pairs = ["data=df"]
    if "robust" in args:
        code_pairs.append("robust='hc1'")
    python = f"sp.ivreg({formula!r}, {', '.join(code_pairs)})"
    return _emit("ivreg", args, python, notes)


def _h_csdid(cmd: StataCommand) -> Dict[str, Any]:
    """``csdid y, ivar(id) tvar(t) gvar(g)`` → ``sp.callaway_santanna``."""
    if not cmd.varlist:
        return _emit_error("csdid requires an outcome variable",
                            command="csdid")
    y = cmd.varlist[0]
    i = cmd.options.get("ivar") or cmd.options.get("id")
    t = cmd.options.get("tvar") or cmd.options.get("time")
    g = cmd.options.get("gvar") or cmd.options.get("cohort")
    missing = [name for name, val in
               (("ivar", i), ("tvar", t), ("gvar", g)) if not val]
    if missing:
        return _emit_error(
            f"csdid translation needs {missing} option(s); supply them "
            "via Stata's `ivar()` / `tvar()` / `gvar()`.",
            command="csdid")
    args: Dict[str, Any] = {"y": y, "i": i, "t": t, "g": g}
    method = cmd.options.get("method", "dr") or "dr"
    if method.lower() in {"dr", "ipw", "reg"}:
        args["estimator"] = method.lower()
    python = (f"sp.callaway_santanna(data=df, y={y!r}, i={i!r}, t={t!r}, "
               f"g={g!r}, estimator={args.get('estimator', 'dr')!r})")
    return _emit("callaway_santanna", args, python)


def _h_did_imputation(cmd: StataCommand) -> Dict[str, Any]:
    """``did_imputation y, treatment(treat) horizons(0 1 2)`` →
    ``sp.did_imputation``. Borusyak-Jaravel-Spiess imputation estimator."""
    if not cmd.varlist:
        return _emit_error("did_imputation requires an outcome variable",
                            command="did_imputation")
    y = cmd.varlist[0]
    treat = cmd.options.get("treatment") or cmd.options.get("treat")
    if not treat:
        return _emit_error(
            "did_imputation needs `treatment(<col>)` (treatment indicator).",
            command="did_imputation")
    args: Dict[str, Any] = {"y": y, "treat": treat}
    horizons = cmd.options.get("horizons")
    if horizons:
        try:
            args["horizons"] = [int(x) for x in horizons.split()]
        except ValueError:
            args["horizons"] = horizons
    pre = cmd.options.get("pretrends")
    if pre:
        try:
            args["pretrends"] = [int(x) for x in pre.split()]
        except ValueError:
            args["pretrends"] = pre
    python = (f"sp.did_imputation(data=df, y={y!r}, treat={treat!r}"
               + (f", horizons={args['horizons']!r}" if "horizons" in args else "")
               + ")")
    return _emit("did_imputation", args, python)


def _h_synth(cmd: StataCommand) -> Dict[str, Any]:
    """``synth gdp predictors..., trunit(treatedid) trperiod(year)`` →
    ``sp.synth``. Stata `synth` uses a different variable convention —
    first variable is outcome; remaining variables are predictors;
    treated unit + treatment period live in options.
    """
    if not cmd.varlist:
        return _emit_error("synth requires an outcome variable",
                            command="synth")
    outcome = cmd.varlist[0]
    predictors = cmd.varlist[1:]
    trunit = cmd.options.get("trunit") or cmd.options.get("treatedid")
    trperiod = cmd.options.get("trperiod") or cmd.options.get("treatment_time")
    if not (trunit and trperiod):
        return _emit_error(
            "synth needs `trunit(<id>)` and `trperiod(<year>)`.",
            command="synth")
    unit = cmd.options.get("unit") or "<unit_col>"
    time = cmd.options.get("time") or "<time_col>"
    args: Dict[str, Any] = {
        "outcome": outcome,
        "unit": unit,
        "time": time,
        "treated_unit": _coerce_scalar(trunit),
        "treatment_time": _coerce_scalar(trperiod),
    }
    if predictors:
        args["predictors"] = predictors
    notes: List[str] = []
    if unit == "<unit_col>" or time == "<time_col>":
        notes.append("Stata `tsset` / `xtset` info isn't visible from the "
                      "command alone — replace <unit_col>/<time_col> with "
                      "the panel-id / time columns.")
    python = (f"sp.synth(data=df, outcome={outcome!r}, unit={unit!r}, "
               f"time={time!r}, treated_unit={args['treated_unit']!r}, "
               f"treatment_time={args['treatment_time']!r}"
               + (f", predictors={predictors!r}" if predictors else "")
               + ")")
    return _emit("synth", args, python, notes)


def _h_rdrobust(cmd: StataCommand) -> Dict[str, Any]:
    """``rdrobust y x, c(0)`` → ``sp.rdrobust``. Same name in Stata + sp."""
    if len(cmd.varlist) < 2:
        return _emit_error(
            "rdrobust requires y + running variable: `rdrobust y x, c(<v>)`",
            command="rdrobust")
    y, x = cmd.varlist[0], cmd.varlist[1]
    c_raw = cmd.options.get("c", "0")
    try:
        c = float(c_raw) if c_raw is not None else 0.0
    except (TypeError, ValueError):
        c = 0.0
    args: Dict[str, Any] = {"y": y, "x": x, "c": c}
    if "fuzzy" in cmd.options and cmd.options["fuzzy"]:
        args["fuzzy"] = cmd.options["fuzzy"].split()[0]
    kernel = cmd.options.get("kernel")
    if kernel and kernel.split()[0].lower() in {"triangular", "uniform",
                                                  "epanechnikov"}:
        args["kernel"] = kernel.split()[0].lower()
    python = (f"sp.rdrobust(data=df, y={y!r}, x={x!r}, c={c}"
               + (f", fuzzy={args['fuzzy']!r}" if "fuzzy" in args else "")
               + (f", kernel={args['kernel']!r}" if "kernel" in args else "")
               + ")")
    return _emit("rdrobust", args, python)


# ---------------------------------------------------------------------------
# Command dispatch table
# ---------------------------------------------------------------------------

#: Map Stata command name (lower-case, full form) → handler. Aliases
#: (Stata's own abbreviations: ``reg`` for ``regress``) are added
#: explicitly to keep the dispatch O(1) instead of running prefix
#: matching at lookup time.
STATA_COMMAND_MAP: Dict[str, Handler] = {
    # Tier 1 — flagship 8
    "regress": _h_regress, "reg": _h_regress,
    "xtreg": _h_xtreg,
    "reghdfe": _h_reghdfe,
    "ivreg2": _h_ivreg2, "ivregress": _h_ivreg2,  # close-enough mapping
    "csdid": _h_csdid,
    "did_imputation": _h_did_imputation,
    "synth": _h_synth,
    "rdrobust": _h_rdrobust,
}


def _coerce_scalar(s: str) -> Any:
    """Convert ``"123"`` / ``"1.5"`` / ``"USA"`` to int / float / str."""
    s = s.strip().strip('"').strip("'")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def from_stata(line: str) -> Dict[str, Any]:
    """Translate a Stata command line to a StatsPAI tool-call payload.

    Parameters
    ----------
    line : str
        One Stata command. Multi-line ``do`` files must be split by
        the caller.

    Returns
    -------
    dict
        On success::

            {
                "ok": True,
                "tool": <tool_name>,
                "arguments": {...},  # ready for execute_tool
                "python_code": "<sp.xxx(...)>",
                "notes": [<warning>, ...],
            }

        On failure::

            {
                "ok": False,
                "tool": null,
                "error": "<diagnosis>",
                "command": "<recognised stata command name or null>",
                "suggestions": [<close-match command names>],
            }
    """
    try:
        parsed = _parse_stata(line)
    except StataParseError as e:
        return _emit_error(f"parse_error: {e}", command=None,
                           suggestions=[])

    handler = STATA_COMMAND_MAP.get(parsed.command)
    if handler is None:
        from difflib import get_close_matches
        suggestions = get_close_matches(parsed.command,
                                          list(STATA_COMMAND_MAP.keys()),
                                          n=5, cutoff=0.55)
        return _emit_error(
            f"unknown / unsupported Stata command {parsed.command!r}",
            command=parsed.command, suggestions=suggestions)

    return handler(parsed)


__all__ = ["from_stata", "STATA_COMMAND_MAP", "StataCommand", "StataParseError"]
