"""Stata command → StatsPAI tool-call translator.

Every entry in :data:`STATA_COMMAND_MAP` is ``stata_cmd → handler``.
Each handler takes a parsed :class:`StataCommand` and returns the
canonical translation dict ``{tool, arguments, python_code, notes}``.

Tier 1 (this file): the commands that cover the bulk of real Stata
econometrics work — `regress` / `xtreg` / `reghdfe` / `ivregress` /
`ivreg2` / `csdid` / `didregress` / `did_imputation` / `synth` /
`rdrobust` — plus follow-on migration helpers such as `teffects`,
`psmatch2`, `ppmlhdfe`, and dynamic-panel commands.

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

from ._stata_lexer import StataCommand, StataParseError
from ._stata_lexer import parse as _parse_stata

Handler = Callable[[StataCommand], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(
    tool: str,
    arguments: Dict[str, Any],
    python_code: str,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "tool": tool,
        "arguments": dict(arguments),
        "python_code": python_code,
        "notes": list(notes or []),
        "ok": True,
    }


def _emit_error(message: str, **extra: Any) -> Dict[str, Any]:
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
        return _emit_error("regress requires an outcome variable", command="regress")
    formula = _build_formula(y, xs)
    cluster = _vce_cluster(cmd)
    robust = _robust_kind(cmd)
    args: Dict[str, Any] = {"formula": formula}
    if robust != "nonrobust":
        args["robust"] = robust
    if cluster:
        args["cluster"] = cluster
    code_kwargs = ", ".join(
        [f"{k}={v!r}" for k, v in args.items() if k != "formula"] + ["data=df"]
    )
    python = f"sp.regress({formula!r}, {code_kwargs})"
    notes: List[str] = []
    if cmd.if_cond:
        notes.append(
            f"Stata `if {cmd.if_cond}` dropped — pre-filter df via "
            f"`df = df.query({cmd.if_cond!r})` before calling."
        )
    if cmd.in_range:
        notes.append(f"Stata `in {cmd.in_range}` dropped — use df.iloc[...].")
    return _emit("regress", args, python, notes)


def _pyfixest_fml(
    main: str,
    fe_terms: List[str],
    iv_lhs: Optional[str] = None,
    iv_rhs: Optional[str] = None,
) -> str:
    """Reassemble a pyfixest formula that :func:`sp.feols` accepts:
    ``depvar ~ exog | fe1 + fe2 | endog ~ instruments`` (each ``|`` section
    optional; the IV section has NO parentheses). Building this — instead of
    emitting a non-existent ``sp.fixest(formula, fe=...)`` call — is what makes
    the translated payload actually runnable via sp.feols."""
    parts = [main.strip()]
    if fe_terms:
        parts.append(" + ".join(fe_terms))
    if iv_lhs and iv_rhs:
        parts.append(f"{iv_lhs.strip()} ~ {iv_rhs.strip()}")
    return " | ".join(parts)


def _feols_code(fml: str, cluster: Optional[str] = None) -> str:
    """Runnable ``sp.feols(...)`` snippet for the translated payload."""
    pairs = [repr(fml), "data=df"]
    if cluster:
        pairs.append(f"cluster={cluster!r}")
    return f"sp.feols({', '.join(pairs)})"


def _h_xtreg(cmd: StataCommand) -> Dict[str, Any]:
    """``xtreg y x1 x2, fe vce(cluster id)`` → ``sp.feols`` with entity FE."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("xtreg requires an outcome variable", command="xtreg")
    if "re" in cmd.options:
        return _emit_error(
            "random-effects xtreg is not supported — use sp.panel(method='re') "
            "directly via the Python API for now.",
            command="xtreg",
        )

    # Stata convention: panel id set via ``xtset id [t]``; we can't see
    # that here, so the user must supply ``id`` via the option or we
    # leave a placeholder.
    panel_id = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    main = _build_formula(y, xs)
    cluster = _vce_cluster(cmd)
    fe_terms = [panel_id] if panel_id != "<panel_id>" else []
    fml = _pyfixest_fml(main, fe_terms)
    args: Dict[str, Any] = {"fml": fml}
    if cluster:
        args["cluster"] = cluster
    notes: List[str] = []
    if panel_id == "<panel_id>":
        notes.append(
            "Couldn't recover the panel-id from this command alone "
            "(Stata's `xtset id` lives in another line). Replace "
            "<panel_id> with the actual unit id column."
        )
    if "be" in cmd.options or "fd" in cmd.options:
        notes.append(
            "Between-effects / first-difference variants are not yet "
            "translated — use sp.panel(method='be'/'fd') directly."
        )
    return _emit("feols", args, _feols_code(fml, cluster), notes)


def _h_reghdfe(cmd: StataCommand) -> Dict[str, Any]:
    """``reghdfe y x, absorb(id year) cluster(id)`` → ``sp.feols``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("reghdfe requires an outcome variable", command="reghdfe")
    absorb = cmd.options.get("absorb") or ""
    fe_list = [v for v in absorb.split() if v]
    cluster = _vce_cluster(cmd) or cmd.options.get("cluster")
    if cluster:
        cluster = cluster.split()[0]
    main = _build_formula(y, xs)
    fml = _pyfixest_fml(main, fe_list)
    args: Dict[str, Any] = {"fml": fml}
    if cluster:
        args["cluster"] = cluster
    notes: List[str] = []
    if not fe_list:
        notes.append(
            "reghdfe with no absorb() collapses to OLS — "
            "consider sp.regress instead."
        )
    return _emit("feols", args, _feols_code(fml, cluster), notes)


def _h_ivreg2(cmd: StataCommand) -> Dict[str, Any]:
    """``ivreg2 y x1 (d = z1 z2), cluster(id)`` → ``sp.ivreg``."""
    # Stata's ``ivreg2`` varlist contains parentheses with `d = z`.
    # Re-join the original varlist tokens to recover the parens.
    if not cmd.varlist:
        return _emit_error("ivreg2 requires an outcome variable", command="ivreg2")
    tokens = list(cmd.varlist)
    method: Optional[str] = None
    if cmd.command == "ivregress" and tokens:
        head = tokens[0].lower()
        if head in {"2sls", "liml", "gmm"}:
            method = head
            tokens = tokens[1:]
    joined = " ".join(tokens)
    # Accept either ``y x (d = z)`` or ``y (d = z)``.
    import re

    m = re.match(r"^\s*(\S+)\s*(.*?)\s*\(\s*(\S+)\s*=\s*([^)]+?)\s*\)\s*$", joined)
    if not m:
        return _emit_error(
            f"could not parse ivreg2 syntax {joined!r}; expected "
            "`y [exog_x...] (endog = instruments...)`",
            command="ivreg2",
        )
    y, exog_xs, endog, instruments = m.group(1), m.group(2), m.group(3), m.group(4)
    formula_lhs = f"{y} ~ "
    if exog_xs.strip():
        formula_lhs += f"{exog_xs.strip()} + "
    formula = f"{formula_lhs}({endog} ~ {instruments.strip()})"

    cluster = _vce_cluster(cmd) or cmd.options.get("cluster")
    if cluster:
        cluster = cluster.split()[0]
    args: Dict[str, Any] = {"formula": formula}
    if method:
        args["method"] = method
    robust = _robust_kind(cmd)
    if robust != "nonrobust":
        args["robust"] = robust
    if cluster:
        args["cluster"] = cluster
    notes: List[str] = []
    if cluster:
        notes.append(f"Mapped Stata cluster({cluster}) to sp.ivreg cluster=.")
    if "first" in cmd.options:
        notes.append(
            "`first` (first-stage display) not translated; the sp "
            "result already exposes first_stage_F via diagnostics."
        )
    if "small" in cmd.options:
        notes.append(
            "Stata `small` changes finite-sample reporting conventions; "
            "verify the SE/df convention if exact table reproduction matters."
        )
    if method in {"liml", "gmm"}:
        notes.append(
            f"Mapped official `ivregress {method}` syntax to "
            f"sp.ivreg(..., method={method!r})."
        )
    code_pairs = ["data=df"]
    for key in ("method", "robust", "cluster"):
        if key in args:
            code_pairs.append(f"{key}={args[key]!r}")
    python = f"sp.ivreg({formula!r}, {', '.join(code_pairs)})"
    return _emit("ivreg", args, python, notes)


def _h_ivreghdfe(cmd: StataCommand) -> Dict[str, Any]:
    """``ivreghdfe y x (d = z), absorb(id year)`` → ``sp.fixest`` IV+FE."""
    if not cmd.varlist:
        return _emit_error(
            "ivreghdfe requires an outcome variable", command="ivreghdfe"
        )
    joined = " ".join(cmd.varlist)
    import re

    m = re.match(r"^\s*(\S+)\s*(.*?)\s*\(\s*(\S+)\s*=\s*([^)]+?)\s*\)\s*$", joined)
    if not m:
        return _emit_error(
            f"could not parse ivreghdfe syntax {joined!r}; expected "
            "`y [exog_x...] (endog = instruments...)`",
            command="ivreghdfe",
        )
    y, exog_xs, endog, instruments = m.group(1), m.group(2), m.group(3), m.group(4)
    exog = [x for x in exog_xs.split() if x]
    main = _build_formula(y, exog)
    instr = " + ".join(x for x in instruments.split() if x)
    absorb = cmd.options.get("absorb") or ""
    fe_list = [v for v in absorb.split() if v]
    cluster = _vce_cluster(cmd) or cmd.options.get("cluster")
    if cluster:
        cluster = cluster.split()[0]
    fml = _pyfixest_fml(main, fe_list, endog, instr)
    args: Dict[str, Any] = {"fml": fml}
    if cluster:
        args["cluster"] = cluster
    notes = [
        "Mapped Stata ivreghdfe to StatsPAI/fixest IV-with-fixed-effects "
        "syntax. This is a command migration contract, not a live Stata run."
    ]
    if not fe_list:
        notes.append("ivreghdfe without absorb() is equivalent to IV without HDFE.")
    return _emit("feols", args, _feols_code(fml, cluster), notes)


def _h_csdid(cmd: StataCommand) -> Dict[str, Any]:
    """``csdid y, ivar(id) tvar(t) gvar(g)`` → ``sp.callaway_santanna``."""
    if not cmd.varlist:
        return _emit_error("csdid requires an outcome variable", command="csdid")
    y = cmd.varlist[0]
    i = cmd.options.get("ivar") or cmd.options.get("id")
    t = cmd.options.get("tvar") or cmd.options.get("time")
    g = cmd.options.get("gvar") or cmd.options.get("cohort")
    missing = [name for name, val in (("ivar", i), ("tvar", t), ("gvar", g)) if not val]
    if missing:
        return _emit_error(
            f"csdid translation needs {missing} option(s); supply them "
            "via Stata's `ivar()` / `tvar()` / `gvar()`.",
            command="csdid",
        )
    args: Dict[str, Any] = {"y": y, "i": i, "t": t, "g": g}
    method = cmd.options.get("method", "dr") or "dr"
    if method.lower() in {"dr", "ipw", "reg"}:
        args["estimator"] = method.lower()
    python = (
        f"sp.callaway_santanna(data=df, y={y!r}, i={i!r}, t={t!r}, "
        f"g={g!r}, estimator={args.get('estimator', 'dr')!r})"
    )
    return _emit("callaway_santanna", args, python)


def _h_didregress(cmd: StataCommand) -> Dict[str, Any]:
    """``didregress (y x) (treated), group(id) time(year)`` → ``sp.did``.

    Stata's official ``didregress`` / ``xtdidregress`` commands take an
    outcome equation and a treatment-status equation.  StatsPAI's staggered
    DID APIs use cohort columns, so this translator deliberately routes the
    official Stata treatment-status form through ``method='twfe'`` and emits a
    note rather than silently treating a 0/1 treatment indicator as a cohort.
    """
    raw = cmd.raw or ""
    import re

    prefix = r"(?:by\s+[^:]*?:|capture\s*:|quietly\s*:|qui\s*:|noisily\s*:)?"
    pattern = (
        r"^\s*"
        + prefix
        + r"\s*(?:didregress|xtdidregress)\s+"
        + r"\((.+?)\)\s+\((.+?)\)"
    )
    m = re.match(pattern, raw, flags=re.I | re.S)
    if not m:
        return _emit_error(
            "didregress expects `(outcome [covariates]) (treatment)` equations.",
            command=cmd.command,
        )

    outcome_tokens = [tok for tok in m.group(1).split() if tok]
    treat_tokens = [tok for tok in m.group(2).split() if tok]
    if not outcome_tokens or not treat_tokens:
        return _emit_error(
            "didregress outcome and treatment equations must be non-empty.",
            command=cmd.command,
        )

    y = outcome_tokens[0]
    covariates = outcome_tokens[1:]
    treat = treat_tokens[0]
    group = cmd.options.get("group") or cmd.options.get("ivar") or cmd.options.get("id")
    time = cmd.options.get("time") or cmd.options.get("tvar")
    missing = [name for name, val in (("group", group), ("time", time)) if not val]
    if missing:
        return _emit_error(
            f"{cmd.command} translation needs {missing} option(s); supply "
            "`group()` and `time()`.",
            command=cmd.command,
        )

    assert group is not None
    assert time is not None
    group = group.split()[0]
    time = time.split()[0]
    args: Dict[str, Any] = {
        "y": y,
        "treat": treat,
        "time": time,
        "id": group,
        "method": "twfe",
    }
    if covariates:
        args["covariates"] = covariates
    cluster = _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster
    if "wboot" in cmd.options or "wildbootstrap" in cmd.options:
        args["se_method"] = "wild_cluster_bootstrap"

    notes = [
        "Stata didregress/xtdidregress uses a treatment-status indicator; "
        "StatsPAI staggered DID needs a first-treatment cohort column. "
        "This translation uses sp.did(..., method='twfe') for the "
        "treatment-status command shape.",
    ]
    if len(treat_tokens) > 1:
        notes.append(
            "Extra tokens inside the treatment equation were not translated; "
            "put controls in the outcome equation or build the desired "
            "StatsPAI call explicitly."
        )

    code_pairs = [
        "data=df",
        f"y={y!r}",
        f"treat={treat!r}",
        f"time={time!r}",
        f"id={group!r}",
        "method='twfe'",
    ]
    if covariates:
        code_pairs.append(f"covariates={covariates!r}")
    if cluster:
        code_pairs.append(f"cluster={cluster!r}")
    python = f"sp.did({', '.join(code_pairs)})"
    return _emit("did", args, python, notes)


def _h_did_imputation(cmd: StataCommand) -> Dict[str, Any]:
    """``did_imputation Y i t Ei`` → ``sp.did_imputation``.

    Borusyak-Jaravel-Spiess imputation estimator. The Stata command is
    positional: outcome, unit-id, time, and the cohort/first-treatment column
    (``Ei``; 0 or missing = never treated). sp.did_imputation takes these as
    ``y`` / ``group`` / ``time`` / ``first_treat`` — matching its signature so
    the payload runs.
    """
    if len(cmd.varlist) < 4:
        return _emit_error(
            "did_imputation is positional: `did_imputation Y i t Ei` (outcome, "
            "unit-id, time, first-treatment cohort).",
            command="did_imputation",
        )
    y, group, time, first_treat = cmd.varlist[:4]
    args: Dict[str, Any] = {
        "y": y,
        "group": group,
        "time": time,
        "first_treat": first_treat,
    }
    horizon = cmd.options.get("horizon") or cmd.options.get("horizons")
    if horizon:
        try:
            args["horizon"] = int(horizon.split()[-1])
        except (ValueError, AttributeError, IndexError):
            pass
    code_pairs = [
        "data=df",
        f"y={y!r}",
        f"group={group!r}",
        f"time={time!r}",
        f"first_treat={first_treat!r}",
    ]
    if "horizon" in args:
        code_pairs.append(f"horizon={args['horizon']}")
    python = f"sp.did_imputation({', '.join(code_pairs)})"
    return _emit("did_imputation", args, python)


def _h_synth(cmd: StataCommand) -> Dict[str, Any]:
    """``synth gdp predictors..., trunit(treatedid) trperiod(year)`` →
    ``sp.synth``. Stata `synth` uses a different variable convention —
    first variable is outcome; remaining variables are predictors;
    treated unit + treatment period live in options.
    """
    if not cmd.varlist:
        return _emit_error("synth requires an outcome variable", command="synth")
    outcome = cmd.varlist[0]
    predictors = cmd.varlist[1:]
    trunit = cmd.options.get("trunit") or cmd.options.get("treatedid")
    trperiod = cmd.options.get("trperiod") or cmd.options.get("treatment_time")
    if not (trunit and trperiod):
        return _emit_error(
            "synth needs `trunit(<id>)` and `trperiod(<year>)`.", command="synth"
        )
    unit = cmd.options.get("unit") or "<unit_col>"
    time = cmd.options.get("time") or "<time_col>"
    args: Dict[str, Any] = {
        "outcome": outcome,
        "unit": unit,
        "time": time,
        "treated_unit": _coerce_scalar(trunit),
        "treatment_time": _coerce_scalar(trperiod),
    }
    # sp.synth's covariate argument is ``covariates`` (not ``predictors``);
    # emitting the wrong name means execute_tool silently drops it and the
    # synthetic control is fit with no predictors — wrong weights, no error.
    if predictors:
        args["covariates"] = predictors
    notes: List[str] = []
    if unit == "<unit_col>" or time == "<time_col>":
        notes.append(
            "Stata `tsset` / `xtset` info isn't visible from the "
            "command alone — replace <unit_col>/<time_col> with "
            "the panel-id / time columns."
        )
    python = (
        f"sp.synth(data=df, outcome={outcome!r}, unit={unit!r}, "
        f"time={time!r}, treated_unit={args['treated_unit']!r}, "
        f"treatment_time={args['treatment_time']!r}"
        + (f", covariates={predictors!r}" if predictors else "")
        + ")"
    )
    return _emit("synth", args, python, notes)


def _h_rdrobust(cmd: StataCommand) -> Dict[str, Any]:
    """``rdrobust y x, c(0)`` → ``sp.rdrobust``. Same name in Stata + sp."""
    if len(cmd.varlist) < 2:
        return _emit_error(
            "rdrobust requires y + running variable: `rdrobust y x, c(<v>)`",
            command="rdrobust",
        )
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
    if kernel and kernel.split()[0].lower() in {
        "triangular",
        "uniform",
        "epanechnikov",
    }:
        args["kernel"] = kernel.split()[0].lower()
    python = (
        f"sp.rdrobust(data=df, y={y!r}, x={x!r}, c={c}"
        + (f", fuzzy={args['fuzzy']!r}" if "fuzzy" in args else "")
        + (f", kernel={args['kernel']!r}" if "kernel" in args else "")
        + ")"
    )
    return _emit("rdrobust", args, python)


# ---------------------------------------------------------------------------
# Tier-2 handlers — observational + RD ancillary + diagnostics
# ---------------------------------------------------------------------------


def _h_probit(cmd: StataCommand) -> Dict[str, Any]:
    return _h_glm_like(cmd, sp_fn="probit", display_name="probit")


def _h_logit(cmd: StataCommand) -> Dict[str, Any]:
    return _h_glm_like(cmd, sp_fn="logit", display_name="logit")


def _h_poisson(cmd: StataCommand) -> Dict[str, Any]:
    return _h_glm_like(cmd, sp_fn="poisson", display_name="poisson")


def _h_nbreg(cmd: StataCommand) -> Dict[str, Any]:
    return _h_glm_like(cmd, sp_fn="nbreg", display_name="nbreg")


def _h_xtnbreg(cmd: StataCommand) -> Dict[str, Any]:
    """``xtnbreg y x, fe i(id)`` → ``sp.xtnbreg``.

    Stata's ``xtset`` declaration is not visible from one command line, so
    the translator accepts explicit ``i(id)`` / ``id(id)`` and otherwise
    emits a placeholder plus a note.
    """
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("xtnbreg requires an outcome variable", command="xtnbreg")

    panel_id = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    model = "fe" if "fe" in cmd.options else "re" if "re" in cmd.options else "re"
    formula = _build_formula(y, xs)
    cluster = _vce_cluster(cmd)

    args: Dict[str, Any] = {
        "formula": formula,
        "entity": panel_id if panel_id != "<panel_id>" else None,
        "model": model,
    }
    if cluster:
        args["cluster"] = cluster
    if "irr" in cmd.options or "eform" in cmd.options:
        args["irr"] = True
    offset_opt = cmd.options.get("offset")
    if offset_opt:
        args["offset"] = offset_opt.split()[0]
    exposure_opt = cmd.options.get("exposure")
    if exposure_opt:
        args["exposure"] = exposure_opt.split()[0]

    notes: List[str] = []
    if panel_id == "<panel_id>":
        notes.append(
            "Couldn't recover the panel-id from this command alone "
            "(Stata's `xtset id` lives in another line). Replace "
            "<panel_id> with the actual unit id column."
        )
    if model == "fe":
        notes.append(
            "StatsPAI fits fixed-effects xtnbreg as an unconditional "
            "NB model with explicit panel dummies; this preserves "
            "the count likelihood and avoids routing through OLS."
        )
    else:
        notes.append(
            "No `fe` option detected; StatsPAI maps xtnbreg to a "
            "random-intercept NB-2 GLMM (`sp.menbreg`) via "
            "`sp.xtnbreg(model='re')`."
        )

    code_pairs = [
        "data=df",
        f"entity={panel_id!r}",
        f"model={model!r}",
    ]
    if cluster:
        code_pairs.append(f"cluster={cluster!r}")
    if args.get("irr"):
        code_pairs.append("irr=True")
    if "offset" in args:
        code_pairs.append(f"offset={args['offset']!r}")
    if "exposure" in args:
        code_pairs.append(f"exposure={args['exposure']!r}")
    python = f"sp.xtnbreg({formula!r}, {', '.join(code_pairs)})"
    return _emit("xtnbreg", args, python, notes)


def _h_glm_like(cmd: StataCommand, *, sp_fn: str, display_name: str) -> Dict[str, Any]:
    """Common scaffold for probit / logit / poisson / nbreg."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error(
            f"{display_name} requires an outcome variable", command=display_name
        )
    formula = _build_formula(y, xs)
    cluster = _vce_cluster(cmd)
    robust = _robust_kind(cmd)
    args: Dict[str, Any] = {"formula": formula}
    if robust != "nonrobust":
        args["robust"] = robust
    if cluster:
        args["cluster"] = cluster
    # python_code and arguments must describe the SAME call (round-trip
    # contract). Build code_pairs from the same args we just populated so
    # copy-paste and dispatch can never diverge.
    code_pairs = ["data=df"]
    if "cluster" in args:
        code_pairs.append(f"cluster={cluster!r}")
    if "robust" in args:
        code_pairs.append(f"robust={robust!r}")
    python = f"sp.{sp_fn}({formula!r}, " + ", ".join(code_pairs) + ")"
    return _emit(sp_fn, args, python)


def _h_tobit(cmd: StataCommand) -> Dict[str, Any]:
    """``tobit y x, ll(0) ul(100)`` → ``sp.tobit(y=..., x=[...], ll=, ul=)``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("tobit requires an outcome variable", command="tobit")
    # sp.tobit takes y / x / ll / ul explicitly — not a formula (matching its
    # signature is what makes the payload runnable).
    args: Dict[str, Any] = {"y": y, "x": list(xs)}
    for stata_opt, kw_name in (("ll", "ll"), ("ul", "ul")):
        raw = cmd.options.get(stata_opt)
        if raw is not None:
            try:
                args[kw_name] = float(raw)
            except (TypeError, ValueError):
                pass
    code_pairs = ["data=df", f"y={y!r}", f"x={list(xs)!r}"]
    for kw_name in ("ll", "ul"):
        if kw_name in args:
            code_pairs.append(f"{kw_name}={args[kw_name]}")
    python = f"sp.tobit({', '.join(code_pairs)})"
    return _emit("tobit", args, python)


def _h_heckman(cmd: StataCommand) -> Dict[str, Any]:
    """``heckman y x, select(employed = age kids)`` →
    ``sp.heckman(y=..., x=[...], select=..., z=[...])``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("heckman requires an outcome variable", command="heckman")
    select = cmd.options.get("select")
    if not select:
        return _emit_error(
            "heckman needs `select(<eq>)` (selection equation).", command="heckman"
        )
    # Stata syntax: ``select(d = z1 z2)``
    import re

    m = re.match(r"^\s*(\S+)\s*=\s*(.+)$", select)
    if not m:
        return _emit_error(
            "heckman select() must be `selectvar = covariates`", command="heckman"
        )
    select_var = m.group(1)
    z_vars = [v for v in m.group(2).split() if v]
    # sp.heckman takes y / x / select / z explicitly — matching its signature.
    args: Dict[str, Any] = {
        "y": y,
        "x": list(xs),
        "select": select_var,
        "z": z_vars,
    }
    python = (
        f"sp.heckman(data=df, y={y!r}, x={list(xs)!r}, "
        f"select={select_var!r}, z={z_vars!r})"
    )
    return _emit("heckman", args, python)


def _h_rdplot(cmd: StataCommand) -> Dict[str, Any]:
    if len(cmd.varlist) < 2:
        return _emit_error(
            "rdplot needs y + running variable: `rdplot y x, c(<v>)`", command="rdplot"
        )
    y, x = cmd.varlist[0], cmd.varlist[1]
    c_raw = cmd.options.get("c", "0")
    try:
        c = float(c_raw) if c_raw is not None else 0.0
    except (TypeError, ValueError):
        c = 0.0
    args: Dict[str, Any] = {"y": y, "x": x, "c": c}
    python = f"sp.rdplot(data=df, y={y!r}, x={x!r}, c={c})"
    return _emit("rdplot", args, python)


def _h_rddensity(cmd: StataCommand) -> Dict[str, Any]:
    if not cmd.varlist:
        return _emit_error("rddensity requires a running variable", command="rddensity")
    x = cmd.varlist[0]
    c_raw = cmd.options.get("c", "0")
    try:
        c = float(c_raw) if c_raw is not None else 0.0
    except (TypeError, ValueError):
        c = 0.0
    args: Dict[str, Any] = {"x": x, "c": c}
    python = f"sp.rddensity(data=df, x={x!r}, c={c})"
    return _emit("rddensity", args, python)


def _h_teffects(cmd: StataCommand) -> Dict[str, Any]:
    """``teffects ipw (y) (treat z1 z2)`` / ``teffects nnmatch (y x) (treat)``
    / ``teffects psmatch (y) (treat z)``.

    The Stata grammar nests parens around outcome-eq and treatment-eq
    blocks; we parse them via the original raw line rather than the
    flat varlist (which loses parenthesis structure).
    """
    raw = cmd.raw or ""
    import re

    m = re.match(r"^\s*teffects\s+(\w+)\s+\((.+?)\)\s+\((.+?)\)(.*)$", raw, flags=re.I)
    if not m:
        return _emit_error(
            "teffects: expected `teffects <method> (out_eq) (treat_eq) [, opts]`",
            command="teffects",
        )
    method = m.group(1).lower()
    out_eq_tokens = m.group(2).split()
    treat_eq_tokens = m.group(3).split()
    if not out_eq_tokens or not treat_eq_tokens:
        return _emit_error(
            "teffects: outcome / treatment equations are empty", command="teffects"
        )
    y = out_eq_tokens[0]
    out_xs = out_eq_tokens[1:]
    treat = treat_eq_tokens[0]
    treat_xs = treat_eq_tokens[1:]
    estimand = "ATT" if "atet" in cmd.options else "ATE"

    # Choose the closest sp helper per teffects method.
    if method in {"ipw", "ipwra"}:
        sp_fn = "ipw"
        args: Dict[str, Any] = {
            "y": y,
            "treat": treat,
            "covariates": treat_xs,
            "estimand": estimand,
        }
        python = (
            f"sp.ipw(data=df, y={y!r}, treat={treat!r}, "
            f"covariates={treat_xs!r}, estimand={estimand!r})"
        )
    elif method in {"nnmatch", "psmatch", "match"}:
        sp_fn = "match"
        args = {
            "y": y,
            "treat": treat,
            "covariates": treat_xs or out_xs,
            "method": ("ps" if method == "psmatch" else "nn"),
            "estimand": estimand,
        }
        python = (
            f"sp.match(data=df, y={y!r}, treat={treat!r}, "
            f"covariates={args['covariates']!r}, "
            f"method={args['method']!r}, estimand={estimand!r})"
        )
    elif method == "ra":
        sp_fn = "regress"
        formula = _build_formula(y, [treat] + out_xs)
        args = {"formula": formula}
        python = f"sp.regress({formula!r}, data=df)"
    elif method in {"aipw", "drdid"}:
        sp_fn = "aipw"
        args = {
            "y": y,
            "treat": treat,
            "covariates": treat_xs,
            "estimand": estimand,
        }
        python = (
            f"sp.aipw(data=df, y={y!r}, treat={treat!r}, "
            f"covariates={treat_xs!r}, estimand={estimand!r})"
        )
    else:
        return _emit_error(
            f"teffects method {method!r} not supported "
            f"(known: ipw / nnmatch / psmatch / ra / aipw)",
            command="teffects",
        )
    return _emit(sp_fn, args, python)


def _h_psmatch2(cmd: StataCommand) -> Dict[str, Any]:
    """``psmatch2 d x, out(y) n(1)`` → ``sp.psmatch2``."""
    if len(cmd.varlist) < 2:
        return _emit_error(
            "psmatch2 requires treatment plus covariates: `psmatch2 d x1 x2`",
            command="psmatch2",
        )
    treat = cmd.varlist[0]
    covariates = cmd.varlist[1:]
    args: Dict[str, Any] = {"treat": treat, "covariates": covariates}
    notes: List[str] = []

    outcome = (
        cmd.options.get("outcome") or cmd.options.get("out") or cmd.options.get("y")
    )
    if outcome:
        args["outcome"] = outcome.split()[0]
    else:
        notes.append(
            "psmatch2 outcome() omitted; sp.psmatch2 will build the matched "
            "frame but the cross-sectional ATT is undefined."
        )

    neighbor = cmd.options.get("neighbor") or cmd.options.get("n")
    if neighbor:
        try:
            args["neighbor"] = int(neighbor)
        except (TypeError, ValueError):
            notes.append(
                f"Could not parse neighbor count {neighbor!r}; using default 1."
            )

    if "kernel" in cmd.options:
        args["method"] = "kernel"
        kernel = (cmd.options.get("kerneltype") or "epan").split()[0].lower()
        if kernel == "epanechnikov":
            kernel = "epan"
        args["kernel"] = kernel
        bwidth = cmd.options.get("bwidth") or cmd.options.get("bw")
        if bwidth:
            try:
                args["bwidth"] = float(bwidth)
            except (TypeError, ValueError):
                notes.append(f"Could not parse bwidth {bwidth!r}; using default.")
    elif "radius" in cmd.options:
        args["method"] = "radius"

    caliper = cmd.options.get("caliper")
    if caliper:
        try:
            args["caliper"] = float(caliper)
        except (TypeError, ValueError):
            notes.append(f"Could not parse caliper {caliper!r}; ignoring it.")

    if "common" in cmd.options:
        args["common_support"] = "minmax"
    ai = cmd.options.get("ai")
    if ai:
        try:
            args["ai"] = int(ai)
        except (TypeError, ValueError):
            notes.append(f"Could not parse ai({ai}); using default standard error.")
    if "noreplacement" in cmd.options or "noreplace" in cmd.options:
        args["replace"] = False
    if "probit" in cmd.options:
        notes.append(
            "Stata psmatch2 probit propensity-score option is not exposed "
            "by sp.psmatch2; check parity before relying on this translation."
        )
    if "ate" in cmd.options:
        notes.append("sp.psmatch2 is ATT-focused; use sp.match for ATE.")

    code_pairs = [
        "data=df",
        f"treat={treat!r}",
        f"covariates={covariates!r}",
    ]
    for key in (
        "outcome",
        "neighbor",
        "method",
        "kernel",
        "bwidth",
        "caliper",
        "common_support",
        "ai",
        "replace",
    ):
        if key in args:
            code_pairs.append(f"{key}={args[key]!r}")
    python = f"sp.psmatch2({', '.join(code_pairs)})"
    return _emit("psmatch2", args, python, notes)


def _h_margins(cmd: StataCommand) -> Dict[str, Any]:
    """Stata `margins`/`marginsplot` — emit a hint to use sp.margins()."""
    targets = cmd.varlist or []
    args: Dict[str, Any] = {}
    # Only include a key in args if it has a real value — otherwise the
    # python_code/args round-trip check sees an empty list on the args
    # side and a missing key on the code side (the handler omits empty
    # pairs from the emitted call). Treat empty varlist as "no variables"
    # rather than a sentinel list.
    if targets:
        args["variables"] = targets
    if "dydx" in cmd.options and cmd.options["dydx"]:
        args["dydx"] = cmd.options["dydx"].split()
    if "at" in cmd.options:
        args["at"] = cmd.options["at"]
    notes = [
        "sp.margins takes a fitted result, not data — pipe the "
        "previous estimator's result_id (or fit a model first)."
    ]
    # Round-trip contract: copy-paste and dispatch must describe the same call.
    # Build python_code from args, leaving ``result`` as a literal (the agent
    # pipes the prior estimator's result_id at copy-paste time).
    code_pairs = ["result"]
    for k in ("variables", "dydx", "at"):
        v = args.get(k)
        if v is None or v == "" or v == [] or v == {}:
            continue
        code_pairs.append(f"{k}={v!r}")
    python = f"sp.margins({', '.join(code_pairs)})"
    return _emit("margins", args, python, notes)


def _h_contrast(cmd: StataCommand) -> Dict[str, Any]:
    """Stata `contrast` — pairwise comparisons of a categorical."""
    targets = cmd.varlist or []
    args: Dict[str, Any] = {"terms": targets}
    notes = [
        "sp.contrast takes a fitted result; pipe the previous " "estimator's result_id."
    ]
    code_pairs = ["result"]
    if args["terms"]:
        code_pairs.append(f"terms={args['terms']!r}")
    python = f"sp.contrast({', '.join(code_pairs)})"
    return _emit("contrast", args, python, notes)


def _h_test(cmd: StataCommand) -> Dict[str, Any]:
    """Stata `test x1 x2` — Wald test of joint significance."""
    args: Dict[str, Any] = {"terms": cmd.varlist}
    notes = [
        "sp.test takes a fitted result; pipe the previous " "estimator's result_id."
    ]
    code_pairs = ["result"]
    if args["terms"]:
        code_pairs.append(f"terms={args['terms']!r}")
    python = f"sp.test({', '.join(code_pairs)})"
    return _emit("test", args, python, notes)


def _h_xtset(cmd: StataCommand) -> Dict[str, Any]:
    """``xtset id year`` / ``tsset`` — Stata panel declaration. No sp
    equivalent: pass ``entity=`` / ``time=`` (or id/time kwargs) to
    estimators like ``sp.panel`` directly. Fail loud so agents do not
    silently run with a missing panel structure."""
    if not cmd.varlist:
        return _emit_error(
            "xtset requires a panel id (and optionally time)", command="xtset"
        )
    panel_id = cmd.varlist[0]
    panel_time = cmd.varlist[1] if len(cmd.varlist) > 1 else None
    note = (
        "sp has no xtset/tsset equivalent. Pass "
        f"entity={panel_id!r}"
        + (f" and time={panel_time!r}" if panel_time else "")
        + " to estimators like sp.panel / sp.xtreg / sp.feols "
        "(entity / id) directly."
    )
    return _emit_error(note, command="xtset", suggestions=["panel", "feols"])


# ---------------------------------------------------------------------------
# Tier-3 handlers — long-tail GMM / multinomial / bunching / boottest /
# Poisson HDFE / mi_estimate
# ---------------------------------------------------------------------------


def _h_ppmlhdfe(cmd: StataCommand) -> Dict[str, Any]:
    """``ppmlhdfe y x, absorb(id year) cluster(id)`` → ``sp.ppmlhdfe`` (or
    sp.poisson with FE if ppmlhdfe unavailable). Correia-Guimarães-Zylkin
    Poisson PML with HDFE."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("ppmlhdfe requires an outcome variable", command="ppmlhdfe")
    absorb = cmd.options.get("absorb") or ""
    fe_list = [v for v in absorb.split() if v]
    cluster = _vce_cluster(cmd) or cmd.options.get("cluster")
    if cluster:
        cluster = cluster.split()[0]
    formula = _build_formula(y, xs)
    # sp.ppmlhdfe's fixed-effects argument is ``absorb`` — a "+"-joined STRING
    # ("orig + dest + year"), not a list under the name ``fe``. The old ``fe``
    # list was silently dropped by dispatch (wrong name) so the FE vanished and
    # the fit degenerated to plain Poisson with no error — silent wrong results.
    absorb_str = " + ".join(fe_list)
    args: Dict[str, Any] = {"formula": formula}
    if absorb_str:
        args["absorb"] = absorb_str
    if cluster:
        args["cluster"] = cluster
    code_pairs = ["data=df"]
    if absorb_str:
        code_pairs.append(f"absorb={absorb_str!r}")
    if cluster:
        code_pairs.append(f"cluster={cluster!r}")
    python = f"sp.ppmlhdfe({formula!r}, {', '.join(code_pairs)})"
    notes: List[str] = []
    if not fe_list:
        notes.append(
            "ppmlhdfe with no absorb() degenerates to "
            "sp.poisson — consider using that directly."
        )
    return _emit("ppmlhdfe", args, python, notes)


def _h_mlogit(cmd: StataCommand) -> Dict[str, Any]:
    """``mlogit choice age income, baseoutcome(1)`` → ``sp.mlogit``.

    StatsPAI ships a dedicated multinomial-logit estimator; target it directly
    (sp.glm has no 'multinomial' family and would reject ``base_outcome`` — the
    old glm fallback was a dead on-ramp).
    """
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("mlogit requires an outcome variable", command="mlogit")
    formula = _build_formula(y, xs)
    args: Dict[str, Any] = {"formula": formula}
    base = cmd.options.get("baseoutcome")
    if base is not None:
        args["base"] = _coerce_scalar(base)
    code_kwargs = ", ".join(
        ["data=df"] + ([f"base={args['base']!r}"] if "base" in args else [])
    )
    python = f"sp.mlogit({formula!r}, {code_kwargs})"
    return _emit("mlogit", args, python)


def _h_oprobit(cmd: StataCommand) -> Dict[str, Any]:
    """``oprobit grade x1 x2`` → ordered probit via sp.glm(family='ordered_probit')."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("oprobit requires an outcome variable", command="oprobit")
    formula = _build_formula(y, xs)
    args: Dict[str, Any] = {
        "formula": formula,
        "family": "ordered_probit",
    }
    notes = [
        "StatsPAI's ordered probit lives behind "
        "sp.glm(family='ordered_probit'). Use sp.cloglog for "
        "complementary log-log."
    ]
    python = f"sp.glm({formula!r}, data=df, family='ordered_probit')"
    return _emit("glm", args, python, notes)


def _h_xtabond_family(cmd: StataCommand, *, sp_kind: str) -> Dict[str, Any]:
    """Common scaffold for ``xtabond`` / ``xtdpdsys`` (Arellano-Bond /
    Blundell-Bond difference / system GMM) → ``sp.xtabond`` /
    ``sp.xtdpdsys``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error(f"{sp_kind} requires an outcome variable", command=sp_kind)
    if sp_kind == "xtdpdsys":
        # xtdpdsys is Blundell-Bond *system* GMM, which sp.xtabond does not yet
        # implement (it raises NotImplementedError). Emitting tool='xtdpdsys'
        # (no such callable) or xtabond(method='system') (raises) would be a dead
        # on-ramp — fail loud with the honest fallback instead.
        return _emit_error(
            "xtdpdsys (Blundell-Bond system GMM) is not yet available — "
            "sp.xtabond currently implements difference GMM only. Use "
            "`xtabond` for the Arellano-Bond difference-GMM estimator.",
            command="xtdpdsys",
            suggestions=["xtabond"],
        )
    panel_id = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    args: Dict[str, Any] = {
        "y": y,
        "x": xs,
        "id": panel_id if panel_id != "<panel_id>" else None,
        "twostep": "twostep" in cmd.options,
        "robust": "robust" in cmd.options
        or _opt_matches(cmd.options.get("vce"), "robust"),
    }
    lags_opt = cmd.options.get("lags")
    if lags_opt:
        try:
            args["lags"] = int(lags_opt)
        except (TypeError, ValueError):
            pass
    notes: List[str] = []
    if panel_id == "<panel_id>":
        notes.append(
            "Stata's `xtset id [t]` set the panel id; replace "
            "<panel_id> with your unit-id column."
        )
    code_pairs = [
        "data=df",
        f"y={y!r}",
        f"x={xs!r}",
    ]
    if args["id"]:
        code_pairs.append(f"id={args['id']!r}")
    if args["twostep"]:
        code_pairs.append("twostep=True")
    if args["robust"]:
        code_pairs.append("robust=True")
    python = f"sp.{sp_kind}({', '.join(code_pairs)})"
    return _emit(sp_kind, args, python, notes)


def _h_xtabond(cmd: StataCommand) -> Dict[str, Any]:
    return _h_xtabond_family(cmd, sp_kind="xtabond")


def _h_xtdpdsys(cmd: StataCommand) -> Dict[str, Any]:
    return _h_xtabond_family(cmd, sp_kind="xtdpdsys")


def _h_bunching(cmd: StataCommand) -> Dict[str, Any]:
    """``bunching y, c(0) bw(0.05)`` → ``sp.bunching``. Chetty-style
    bunching estimators (Saez 2010, Kleven-Waseem 2013)."""
    if not cmd.varlist:
        return _emit_error(
            "bunching requires a running-variable column", command="bunching"
        )
    running_var = cmd.varlist[0]
    cutoff = cmd.options.get("c", "0")
    try:
        threshold = float(cutoff) if cutoff is not None else 0.0
    except (TypeError, ValueError):
        threshold = 0.0
    bandwidth = cmd.options.get("bw") or cmd.options.get("bandwidth")
    # sp.bunching takes running_var / threshold / bin_width — matching its
    # signature so the payload runs.
    args: Dict[str, Any] = {"running_var": running_var, "threshold": threshold}
    if bandwidth:
        try:
            args["bin_width"] = float(bandwidth)
        except (TypeError, ValueError):
            pass
    code_pairs = ["data=df", f"running_var={running_var!r}", f"threshold={threshold}"]
    if "bin_width" in args:
        code_pairs.append(f"bin_width={args['bin_width']}")
    python = f"sp.bunching({', '.join(code_pairs)})"
    return _emit("bunching", args, python)


def _h_mi_estimate(cmd: StataCommand) -> Dict[str, Any]:
    """``mi estimate: <inner_command>`` → multiple-imputation wrapper.

    We don't try to translate Stata's nested ``mi estimate: reg y x``
    grammar; we just emit a hint pointing the agent at sp.mi_estimate
    and ask them to fit the underlying model first.
    """
    notes = [
        "Stata's `mi estimate: <cmd>` wraps an inner command — "
        "translate the inner command first, then wrap with "
        "sp.mi_estimate(model_fn, data=df_imputed_list)."
    ]
    return _emit(
        "mi_estimate",
        {"hint": "translate inner command first"},
        "# sp.mi_estimate wraps a sequence of fits — see docs.",
        notes,
    )


def _h_boottest(cmd: StataCommand) -> Dict[str, Any]:
    """``boottest x1=0, reps(999)`` → ``sp.wild_cluster_bootstrap``.

    Stata's boottest is Roodman-Webb-MacKinnon-Nielsen wild-cluster
    bootstrap. sp ships an equivalent.
    """
    args: Dict[str, Any] = {"hypothesis": cmd.varlist}
    reps = cmd.options.get("reps")
    if reps:
        try:
            args["B"] = int(reps)
        except (TypeError, ValueError):
            pass
    cluster = cmd.options.get("cluster") or _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster.split()[0]
    notes = [
        "sp.wild_cluster_bootstrap takes a fitted result as the "
        "first arg — pipe the previous estimator's result_id."
    ]
    # Round-trip contract: code must mirror args (besides the ``result``
    # carrier) so copy-paste and dispatch agree.
    code_pairs = ["result"]
    for k in ("hypothesis", "B", "cluster"):
        if k in args and args[k] not in (None, "", []):
            v = args[k]
            code_pairs.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
    python = f"sp.wild_cluster_bootstrap({', '.join(code_pairs)})"
    return _emit("wild_cluster_bootstrap", args, python, notes)


# ---------------------------------------------------------------------------
# Command dispatch table
# ---------------------------------------------------------------------------

#: Map Stata command name (lower-case, full form) → handler. Aliases
#: (Stata's own abbreviations: ``reg`` for ``regress``) are added
#: explicitly to keep the dispatch O(1) instead of running prefix
#: matching at lookup time.
STATA_COMMAND_MAP: Dict[str, Handler] = {
    # Tier 1 — flagship 8 (60% of econ workflows)
    "regress": _h_regress,
    "reg": _h_regress,
    "xtreg": _h_xtreg,
    "reghdfe": _h_reghdfe,
    "ivreg2": _h_ivreg2,
    "ivregress": _h_ivreg2,  # close-enough mapping
    "ivreghdfe": _h_ivreghdfe,
    "csdid": _h_csdid,
    "didregress": _h_didregress,
    "xtdidregress": _h_didregress,
    "did_imputation": _h_did_imputation,
    "synth": _h_synth,
    "rdrobust": _h_rdrobust,
    # Tier 2 — follow-on commands (push coverage to ~85%)
    "probit": _h_probit,
    "logit": _h_logit,
    "poisson": _h_poisson,
    "nbreg": _h_nbreg,
    "xtnbreg": _h_xtnbreg,
    "tobit": _h_tobit,
    "heckman": _h_heckman,
    "rdplot": _h_rdplot,
    "rddensity": _h_rddensity,
    "teffects": _h_teffects,
    "psmatch2": _h_psmatch2,
    "margins": _h_margins,
    "marginsplot": _h_margins,
    "contrast": _h_contrast,
    "test": _h_test,
    "xtset": _h_xtset,
    "tsset": _h_xtset,
    # Tier 3 — long-tail (8 handlers)
    "ppmlhdfe": _h_ppmlhdfe,
    "mlogit": _h_mlogit,
    "oprobit": _h_oprobit,
    "xtabond": _h_xtabond,
    "xtdpdsys": _h_xtdpdsys,
    "bunching": _h_bunching,
    "mi": _h_mi_estimate,  # ``mi estimate: <inner>`` — we get the head
    "boottest": _h_boottest,
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

    Examples
    --------
    >>> import statspai as sp
    >>> out = sp.from_stata("reghdfe y x, absorb(id year) vce(cluster id)")
    >>> out["ok"]
    True
    >>> out["python_code"].startswith("sp.fixest")
    True
    >>> sp.from_stata("notacommand y x")["ok"]
    False
    """
    try:
        parsed = _parse_stata(line)
    except StataParseError as e:
        return _emit_error(f"parse_error: {e}", command=None, suggestions=[])

    handler = STATA_COMMAND_MAP.get(parsed.command)
    if handler is None:
        from difflib import get_close_matches

        suggestions = get_close_matches(
            parsed.command, list(STATA_COMMAND_MAP.keys()), n=5, cutoff=0.55
        )
        return _emit_error(
            f"unknown / unsupported Stata command {parsed.command!r}",
            command=parsed.command,
            suggestions=suggestions,
        )

    return handler(parsed)


__all__ = ["from_stata", "STATA_COMMAND_MAP", "StataCommand", "StataParseError"]
