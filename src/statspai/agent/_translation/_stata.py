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
  equivalent, it is surfaced back to the user. Handlers read options by
  full name; :mod:`._stata_options` expands Stata's abbreviations before
  they run and reports every option no handler looked at, so nothing is
  dropped quietly.
* **Round-trippable** — the output's ``python_code`` should always
  be valid Python; ``arguments`` should always be JSON-serialisable.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import _stata_options as _opts
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
    *,
    semantics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tool": tool,
        "arguments": dict(arguments),
        "python_code": python_code,
        "notes": list(notes or []),
        "ok": True,
    }
    if semantics:
        payload["semantics"] = list(semantics)
    return payload


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
    semantics: List[str] = []
    if "noconstant" in cmd.options:
        if not xs:
            return _emit_error(
                "regress, noconstant needs at least one regressor",
                command="regress",
                suggestions=[],
            )
        formula += " - 1"
        semantics.append(
            "noconstant -> `- 1` in the formula. R-squared is then the "
            "uncentred one (1 - RSS / sum of y squared), as Stata reports."
        )
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
    return _emit("regress", args, python, notes, semantics=semantics)


def _h_qreg(cmd: StataCommand) -> Dict[str, Any]:
    """``qreg y x [, quantile(#)]`` -> ``sp.qreg``.

    Only the default variance is translated: Stata's ``qreg`` default
    (iid, fitted density, Hall-Sheather bandwidth) is ``sp.qreg``'s. A
    ``vce()`` option is left unread, so it is reported as untranslated
    and ``sp.stata`` refuses to run the command.
    """
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None or not xs:
        return _emit_error("qreg requires an outcome and a regressor", command="qreg")
    formula = _build_formula(y, xs)
    args: Dict[str, Any] = {"formula": formula}
    raw = cmd.options.get("quantile")
    if raw is not None:
        try:
            q = float(str(raw).strip())
        except ValueError:
            return _emit_error(f"qreg: quantile({raw}) is not a number", command="qreg")
        # [R] qreg: a value of 1 or more is read as a percentage.
        if q >= 1:
            q = q / 100.0
        if not 0 < q < 1:
            return _emit_error(
                f"qreg: quantile({raw}) is outside (0, 100)", command="qreg"
            )
        args["quantile"] = q
    pairs = [repr(formula)] + [f"{k}={v!r}" for k, v in args.items() if k != "formula"]
    python = f"sp.qreg(df, {', '.join(pairs)})"
    notes: List[str] = []
    if cmd.if_cond:
        notes.append(
            f"Stata `if {cmd.if_cond}` dropped — pre-filter df via "
            f"`df = df.query({cmd.if_cond!r})` before calling."
        )
    if cmd.in_range:
        notes.append(f"Stata `in {cmd.in_range}` dropped — use df.iloc[...].")
    semantics = ["The constant is reported as `const`, Stata's `_cons`."]
    return _emit("qreg", args, python, notes, semantics=semantics)


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
    for model in ("fd", "pa"):
        if model in cmd.options:
            return _emit_error(
                f"xtreg, {model} is not translated"
                + (" — call sp.panel(method='fd') directly." if model == "fd" else "."),
                command="xtreg",
            )
    if "fe" not in cmd.options:
        # xtreg without `fe` is the random-effects estimator
        from ._stata_panel import xtreg_random

        return xtreg_random(cmd, y, xs)

    # Stata convention: panel id set via ``xtset id [t]``; we can't see
    # that here, so the user must supply ``id`` via the option or we
    # leave a placeholder.
    panel_id = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    main = _build_formula(y, xs)
    cluster = _vce_cluster(cmd)
    notes: List[str] = []
    if not cluster and _robust_kind(cmd) == "hc1":
        # [XT] xtreg: with fe, vce(robust) is vce(cluster panelvar)
        cluster = panel_id
        notes.append(
            "xtreg, fe vce(robust) clusters on the panel id in Stata; "
            f"translated to cluster={panel_id!r}."
        )
    # Keep the placeholder IN the formula: dropping it would print a pooled
    # OLS call (``y ~ x``) that runs and silently is not the fixed-effects model.
    fml = _pyfixest_fml(main, [panel_id])
    args: Dict[str, Any] = {"fml": fml}
    if cluster:
        args["cluster"] = cluster
    if panel_id == "<panel_id>":
        notes.append(
            "Couldn't recover the panel-id from this command alone "
            "(Stata's `xtset id` lives in another line). Replace "
            "<panel_id> with the actual unit id column."
        )
    semantics = [
        "xtreg, fe also prints _cons, the average of the fixed effects; "
        "sp.feols absorbs it, so the result has slope coefficients only."
    ]
    return _emit("feols", args, _feols_code(fml, cluster), notes, semantics=semantics)


_ABSORB_NAME = r"[^\W\d]\w*"


def _absorb_terms(absorb: str) -> Tuple[List[str], Optional[str]]:
    """Stata ``absorb()`` terms -> ``sp.hdfe_ols`` absorb syntax.

    ``a#b`` / ``i.a#i.b`` -> ``a^b`` (one FE per combination);
    ``i.g#c.x`` / ``c.x#i.g`` -> ``i.g#c.x`` (slope only) and ``##`` ->
    ``i.g##c.x`` (FE and slope); ``name=term`` (saved FE) and the
    ``, savefe`` suboptions are dropped. Returns ``(terms, error)``.
    """
    spec = absorb.split(",", 1)[0]
    out: List[str] = []
    for raw in spec.split():
        tok = raw.split("=", 1)[1] if "=" in raw else raw
        m = re.fullmatch(
            rf"(?:i\.)?({_ABSORB_NAME})(##|#)c\.({_ABSORB_NAME})", tok
        ) or re.fullmatch(rf"c\.({_ABSORB_NAME})(##|#)(?:i\.)?({_ABSORB_NAME})", tok)
        if m:
            if tok.startswith("c."):
                x, op, g = m.group(1), m.group(2), m.group(3)
            else:
                g, op, x = m.group(1), m.group(2), m.group(3)
            out.append(f"i.{g}{op}c.{x}")
            continue
        parts = tok.split("#")
        names = [re.fullmatch(rf"(?:i\.)?({_ABSORB_NAME})", q) for q in parts]
        if all(names):
            out.append("^".join(n.group(1) for n in names))  # type: ignore[union-attr]
            continue
        return [], f"absorb() term {raw!r} is not translated"
    return out, None


def _cluster_vars(cmd: StataCommand) -> List[str]:
    """All cluster variables of ``vce(cluster a b)`` / ``cluster(a b)``."""
    vce = cmd.options.get("vce")
    if vce:
        parts = vce.split()
        if parts and parts[0].lower().startswith("cl") and len(parts) >= 2:
            return parts[1:]
    cl = cmd.options.get("cluster")
    return cl.split() if cl else []


def _h_reghdfe(cmd: StataCommand) -> Dict[str, Any]:
    """``reghdfe y x, absorb(id year#q) cluster(id)`` -> ``sp.hdfe_ols``.

    ``sp.hdfe_ols`` is StatsPAI's reghdfe: same singleton pruning, absorbed
    degrees of freedom, cluster small-sample factor and ``t(G - 1)``
    reference; ``sp.feols`` (pyfixest) keeps singletons by default.
    """
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("reghdfe requires an outcome variable", command="reghdfe")
    absorb = cmd.options.get("absorb") or ""
    fe_list, err = _absorb_terms(absorb)
    if err is not None:
        return _emit_error(err, command="reghdfe", suggestions=[])
    clusters = _cluster_vars(cmd)
    main = _build_formula(y, xs)
    formula = main + (" | " + " + ".join(fe_list) if fe_list else "")
    args: Dict[str, Any] = {"formula": formula}
    if clusters:
        args["cluster"] = clusters[0] if len(clusters) == 1 else clusters
    elif "robust" in cmd.options or _opt_matches(cmd.options.get("vce"), "robust"):
        args["vce"] = "robust"
    if "keepsingletons" in cmd.options:
        args["drop_singletons"] = False
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "formula")
    python = f"sp.hdfe_ols({formula!r}, data=df" + (f", {kw})" if kw else ")")
    notes: List[str] = []
    if not fe_list:
        notes.append(
            "reghdfe with no absorb() collapses to OLS — "
            "consider sp.regress instead."
        )
    if cmd.if_cond:
        notes.append(
            f"Stata `if {cmd.if_cond}` dropped — pre-filter df via "
            f"`df = df.query({cmd.if_cond!r})` before calling (Stata treats "
            "missing as +infinity in comparisons; pandas does not)."
        )
    return _emit("hdfe_ols", args, python, notes)


def _h_areg(cmd: StataCommand) -> Dict[str, Any]:
    """``areg y x, absorb(g) vce(cluster c)`` -> ``sp.regress`` with ``C(g)``.

    ``areg`` counts the absorbed groups in the degrees of freedom of every
    variance estimator, clustered ones included, and keeps singleton groups.
    ``sp.hdfe_ols`` follows ``reghdfe`` on both points (a fixed effect
    nested in the cluster variable costs no degrees of freedom there), so
    the counterpart that reproduces ``areg`` is the dummy-variable
    regression. Checked against Stata 18 for the default, ``vce(robust)``,
    ``vce(cluster)`` on the absorbed variable and on another one, each
    with and without ``[aw=]``, on data with singleton groups.
    """
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("areg requires an outcome variable", command="areg")
    absorb = (cmd.options.get("absorb") or "").strip()
    m = re.fullmatch(rf"(?:i\.)?({_ABSORB_NAME})", absorb)
    if m is None:
        return _emit_error(
            f"areg absorbs exactly one categorical variable; got absorb({absorb}). "
            "Several fixed effects or interactions are reghdfe syntax.",
            command="areg",
            suggestions=[],
        )
    group = m.group(1)
    formula = _build_formula(y, xs + [f"C({group})"])
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
    notes = [
        f"areg absorb({group}) -> C({group}) dummies in sp.regress: areg's "
        "standard errors count the absorbed groups in the degrees of "
        "freedom. sp.hdfe_ols gives the same coefficients faster but "
        "follows reghdfe's degrees of freedom and drops singletons.",
        "The slope coefficients and their SEs equal areg's. The constant "
        "does not: areg's _cons is the intercept at the average absorbed "
        "effect, while `Intercept` here is the first group's level and the "
        f"C({group})[...] rows are contrasts with it. Do not report "
        "`Intercept` as areg's _cons.",
    ]
    if cmd.if_cond:
        notes.append(
            f"Stata `if {cmd.if_cond}` dropped — pre-filter df via "
            f"`df = df.query({cmd.if_cond!r})` before calling."
        )
    return _emit("regress", args, python, notes)


_SUM_STATS = {
    "n": "n",
    "count": "n",
    "mean": "mean",
    "sd": "sd",
    "min": "min",
    "max": "max",
    "sum": "sum",
    "range": "range",
    "variance": "variance",
    "var": "variance",
    "semean": "semean",
    "cv": "cv",
    "skewness": "skewness",
    "kurtosis": "kurtosis",
    "iqr": "iqr",
    "p1": "p1",
    "p5": "p5",
    "p10": "p10",
    "p25": "p25",
    "p50": "median",
    "median": "median",
    "p75": "p75",
    "p90": "p90",
    "p95": "p95",
    "p99": "p99",
}

#: What ``summarize, detail`` prints, in sp.sumstats names.
_DETAIL_STATS = [
    "n", "mean", "sd", "variance", "skewness", "kurtosis", "min", "max",
    "p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99",
]  # fmt: skip


def _h_summarize(cmd: StataCommand) -> Dict[str, Any]:
    """``summarize`` / ``sum2docx ... using f.docx, stats(...)`` -> ``sp.sumstats``."""
    toks = list(cmd.varlist)
    path = None
    if "using" in toks:
        i = toks.index("using")
        if i + 1 < len(toks):
            path = toks[i + 1].strip('"')
        toks = toks[:i]
    stats = ["n", "mean", "sd", "min", "max"]  # summarize's columns
    detail = "detail" in cmd.options or "d" in cmd.options
    if detail:
        stats = list(_DETAIL_STATS)
    spec = cmd.options.get("stats") or cmd.options.get("statistics")
    if spec:
        stats = []
        for tok in re.sub(r"\([^)]*\)", "", spec).split():
            key = _SUM_STATS.get(tok.lower())
            if key is None:
                return _emit_error(
                    f"summary statistic {tok!r} is not translated",
                    command=cmd.command,
                    suggestions=[],
                )
            stats.append(key)
    output = path or "numeric"
    args: Dict[str, Any] = {"stats": stats, "output": output}
    if any(st.startswith("p") or st in ("median", "iqr") for st in stats):
        # Stata's own rule for a percentile between two order statistics
        args["percentile_method"] = "stata"
    if toks:
        args["vars"] = toks
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    notes: List[str] = []
    if cmd.if_cond:
        notes.append(
            f"Stata `if {cmd.if_cond}` dropped — filter df first "
            f"(`df = df.query({cmd.if_cond!r})`)."
        )
    if spec and "(" in spec:
        notes.append(
            "Display formats such as mean(%9.3f) are not carried over; pass "
            "digits= for a fixed number of decimals."
        )
    semantics: List[str] = []
    if detail and not spec:
        semantics.append(
            "summarize, detail also lists the four smallest and largest "
            "values; those are not in the sp.sumstats call."
        )
    return _emit("sumstats", args, f"sp.sumstats(df, {kw})", notes, semantics=semantics)


def _h_tabstat(cmd: StataCommand) -> Dict[str, Any]:
    """``tabstat x y, statistics(mean sd p50) by(g)`` -> ``sp.sumstats``."""
    if not cmd.varlist:
        return _emit_error("tabstat requires a variable list", command="tabstat")
    spec = cmd.options.get("statistics") or "mean"
    stats: List[str] = []
    for tok in spec.split():
        low = tok.lower()
        if low == "q":  # the three quartiles
            stats += ["p25", "median", "p75"]
            continue
        key = _SUM_STATS.get(low)
        if key is None:
            return _emit_error(
                f"tabstat statistic {tok!r} is not translated",
                command="tabstat",
                suggestions=[],
            )
        stats.append(key)
    args: Dict[str, Any] = {"stats": stats, "output": "numeric"}
    if any(st.startswith("p") or st in ("median", "iqr") for st in stats):
        args["percentile_method"] = "stata"
    args["vars"] = list(cmd.varlist)
    semantics: List[str] = []
    by = cmd.options.get("by")
    if by:
        args["by"] = by.split()[0]
        if "nototal" not in cmd.options:
            semantics.append(
                "tabstat, by() also prints a Total row; sp.sumstats(by=) "
                "returns the groups only."
            )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("sumstats", args, f"sp.sumstats(df, {kw})", semantics=semantics)


def _h_correlate(cmd: StataCommand) -> Dict[str, Any]:
    """``correlate x y z`` / ``pwcorr x y z, obs`` -> ``sp.pwcorr``.

    ``correlate`` uses one common sample (casewise deletion); ``pwcorr``
    uses every pair's own sample unless ``listwise`` is given.
    """
    command = cmd.command or "correlate"
    listwise = command == "correlate" or "listwise" in cmd.options
    args: Dict[str, Any] = {"output": "dataframe", "listwise": listwise}
    if cmd.varlist:
        args["vars"] = list(cmd.varlist)
    if command == "pwcorr":
        if "obs" in cmd.options:
            args["obs"] = True
        # sig / star() / print() only choose what is printed next to each
        # coefficient; the p-values are on the returned frame either way.
        for shown in ("sig", "star", "print"):
            cmd.options.get(shown)
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    semantics = [
        "The p-values and pairwise observation counts are on the returned "
        "DataFrame as .attrs['pvalues'] and .attrs['nobs']."
    ]
    return _emit("pwcorr", args, f"sp.pwcorr(df, {kw})", semantics=semantics)


#: Stata commands StatsPAI covers but ``from_stata`` cannot translate line by
#: line (they wrap another command or read stored estimates).
_UNTRANSLATED_GUIDANCE: Dict[str, Tuple[str, List[str]]] = {
    "permute": (
        "`permute` wraps another estimation command; use sp.ri_test(data, y=, "
        "treat=, cluster=, n_perms=, stat=callable) with a statistic that "
        "refits the model on the permuted treatment.",
        ["ri_test"],
    ),
    "synth2": (
        "`synth2` is `synth` followed by placebo and leave-one-out runs. "
        "Fit with sp.synth(..., placebo=True) (the in-space placebo p-value "
        "is result.pvalue), then sp.synth_loo(...) for the leave-one-out "
        "range and sp.synth_time_placebo(...) for a pretend treatment date. "
        "With `nested` each placebo run repeats the predictor-weight search, "
        "which takes minutes per unit.",
        ["synth", "synth_loo", "synth_time_placebo"],
    ),
    "esttab": (
        "`esttab` tabulates stored estimates; pass the fitted results to "
        "sp.etable([r1, r2, ...]) (or sp.esttab).",
        ["etable", "esttab"],
    ),
    "estout": (
        "`estout` tabulates stored estimates; use sp.etable([...]).",
        ["etable"],
    ),
    "outreg2": (
        "`outreg2` tabulates stored estimates; use sp.outreg2 / sp.etable "
        "with the fitted results.",
        ["outreg2", "etable"],
    ),
    "coefplot": (
        "`coefplot` plots stored estimates; use sp.coefplot(result).",
        ["coefplot"],
    ),
    "dfgls": (
        "`dfgls` prints the statistic for every lag up to maxlag() on one "
        "common sample; call sp.unitroot(df, y, test='dfgls', trend='ct', "
        "lags=k) for a given lag order (trend is dfgls's default; "
        "`notrend` is trend='c'). sp.unitroot uses every observation the "
        "lag order allows, so its statistic differs from dfgls's unless "
        "k equals maxlag.",
        ["unitroot"],
    ),
    "ritest": (
        "`ritest` wraps another estimation command; use sp.ri_test(data, y=, "
        "treat=, strata=, cluster=, n_perms=, stat='ols', covariates=, "
        "absorb=) for the coefficient of a regression re-estimated on each "
        "re-randomization.",
        ["ri_test", "fisher_exact"],
    ),
    "honestdid": (
        "`honestdid` reads e(b) / e(V) of the last event study; pass the "
        "fitted result (or the coefficients and their covariance) to "
        "sp.honest_did_from_moments(betahat, sigma, num_pre_periods=, "
        "l_vec=, m_grid=, method='smoothness' | 'relative_magnitude').",
        ["honest_did", "honest_did_from_moments"],
    ),
    "pretrends": (
        "`pretrends` reads e(b) / e(V) of the last event study; use "
        "sp.pretrends_power / sp.pretrends_slope_for_power, which take a "
        "fitted event study or its coefficients and covariance.",
        ["pretrends_power", "pretrends_slope_for_power"],
    ),
    "fect": (
        "`fect` maps to sp.fect(data, y=, treat=, unit=, time=, "
        "method='fe' | 'ife' | 'mc', r=, vce=); its options are spelled "
        "differently enough that the call is best written directly.",
        ["fect"],
    ),
    "xthdidregress": (
        "`xthdidregress` fits one ATT per cohort and period. `ra` / `ipw` / "
        "`aipw` are sp.callaway_santanna(estimator='reg' | 'ipw' | 'dr', "
        "control_group=) followed by sp.aggte(type='simple' | 'group' | "
        "'calendar' | 'dynamic') for `estat aggregation`; `twfe` is "
        "sp.etwfe with sp.etwfe_emfx.",
        ["callaway_santanna", "aggte", "etwfe", "etwfe_emfx"],
    ),
    "hdidregress": (
        "`hdidregress` is the repeated-cross-section form of xthdidregress; "
        "use sp.callaway_santanna(panel=False) or sp.etwfe(panel=False).",
        ["callaway_santanna", "etwfe"],
    ),
    "synth_runner": (
        "`synth_runner` runs synth on the treated unit and on every donor; "
        "sp.synth(..., placebo=True) does both and reports the rank p-value "
        "of the post/pre RMSPE ratio, with the placebo gaps in model_info.",
        ["synth", "synth_time_placebo"],
    ),
    "ddml": (
        "`ddml` is a multi-line workflow (init, learners, crossfit, "
        "estimate); the whole of it is one call, sp.dml(data, y=, treat=, "
        "covariates=, model='plr' | 'irm' | 'pliv' | 'iivm', ml_g=, ml_m=, "
        "n_folds=).",
        ["dml"],
    ),
    "qddml": (
        "`qddml` is sp.dml(data, y=, treat=, covariates=, model=, ml_g=, "
        "ml_m=, n_folds=) in one call.",
        ["dml"],
    ),
    "poregress": (
        "`poregress` is partialling-out lasso with a plug-in penalty; the "
        "counterpart is sp.rlasso_effect (rigorous lasso, post-double "
        "selection or partialling out). Penalty rules differ in detail, so "
        "selected controls and the estimate can differ.",
        ["rlasso_effect", "dml"],
    ),
    "dsregress": (
        "`dsregress` is double-selection lasso; use sp.rlasso_effect("
        "method='double selection').",
        ["rlasso_effect"],
    ),
    "xporegress": (
        "`xporegress` is cross-fit partialling-out lasso; use sp.dml("
        "model='plr') with lasso learners.",
        ["dml"],
    ),
    "telasso": (
        "`telasso` is augmented IPW with lasso-selected models; use "
        "sp.dml(model='irm') or sp.aipw with the chosen learners.",
        ["dml", "aipw"],
    ),
    "tebalance": (
        "`tebalance summarize` after teffects nnmatch is the `detail` table "
        "of sp.match(method='nnmatch'); for other estimators use "
        "sp.balance_diagnostics.",
        ["match", "balance_diagnostics"],
    ),
    "teoverlap": (
        "`teoverlap` plots the propensity-score densities by arm; use "
        "sp.overlap_plot(data, treatment=, covariates=).",
        ["overlap_plot"],
    ),
    "weakivtest": (
        "`weakivtest` is sp.effective_f_test(data, endog=, instruments=, "
        "exog=, y=), which reports the effective F and the Montiel Olea-Pflueger "
        "critical values.",
        ["effective_f_test"],
    ),
    "panelview": (
        "`panelview` is sp.panel_view(data, unit=, time=, treat=, y=).",
        ["panel_view"],
    ),
    "event_plot": (
        "`event_plot` plots stored event-study estimates; call .plot() on "
        "the fitted result or sp.enhanced_event_study_plot.",
        ["enhanced_event_study_plot"],
    ),
}


_IV_BLOCK = re.compile(r"\(\s*([^()=]*?)\s*=\s*([^()]*?)\s*\)")


def _parse_iv_varlist(
    tokens: List[str], command: str
) -> Union[Tuple[str, List[str], List[str], List[str]], Dict[str, Any]]:
    """Split a Stata IV varlist into ``(y, exog, endog, instruments)``.

    Stata accepts exogenous regressors on either side of the single
    ``(endog = instruments)`` block -- ``ivregress 2sls y x1 (d = z) x2`` is
    as valid as ``y x1 x2 (d = z)`` -- and every list may hold several
    variables. Returns an ``_emit_error`` payload when the shape is wrong.
    """
    expected = "expected `y [exog...] (endog... = instruments...) [exog...]`"
    shown = " ".join(tokens)

    def bad() -> Dict[str, Any]:
        return _emit_error(
            f"could not parse {command} syntax {shown!r}; {expected}",
            command=command,
        )

    # The head arrives tokenised: the equation's "(", "=" and ")" are tokens
    # of their own, and a translated factor term (``C(g)``, ``x + z + x:z``)
    # is one token whatever it contains.
    if tokens.count("(") != 1 or tokens.count(")") != 1 or tokens.count("=") != 1:
        return bad()
    lo, eq, hi = tokens.index("("), tokens.index("="), tokens.index(")")
    if not lo < eq < hi:
        return bad()
    before, after = tokens[:lo], tokens[hi + 1 :]
    endog, instruments = tokens[lo + 1 : eq], tokens[eq + 1 : hi]
    if not before or not endog or not instruments or "," in tokens:
        return bad()
    return before[0], before[1:] + after, endog, instruments


def _h_ivreg2(cmd: StataCommand) -> Dict[str, Any]:
    """``ivreg2 y x1 (d = z1 z2), cluster(id)`` → ``sp.ivreg``.

    Also serves ``ivregress`` and the legacy ``ivreg``. ``sp.ivreg`` reports
    small-sample standard errors, which is what ``ivreg`` always does and
    what ``ivreg2`` / ``ivregress`` do under ``small``.
    """
    command = cmd.command or "ivreg2"
    if not cmd.varlist:
        return _emit_error(f"{command} requires an outcome variable", command=command)
    tokens = list(cmd.varlist)
    method: Optional[str] = None
    if cmd.command == "ivregress" and tokens:
        head = tokens[0].lower()
        if head in {"2sls", "liml", "gmm"}:
            method = head
            tokens = tokens[1:]
    parsed = _parse_iv_varlist(tokens, command)
    if isinstance(parsed, dict):
        return parsed
    y, exog, endog, instruments = parsed
    formula = f"{y} ~ "
    if exog:
        formula += " + ".join(exog) + " + "
    formula += f"({' + '.join(endog)} ~ {' + '.join(instruments)})"

    cluster = _vce_cluster(cmd) or cmd.options.get("cluster")
    if cluster:
        cluster = cluster.split()[0]
    args: Dict[str, Any] = {"formula": formula}
    if method:
        args["method"] = method
    robust = _robust_kind(cmd)
    # ``ivreg`` is small-sample by construction; the other two only with
    # ``small``. Read the option either way so it is accounted for.
    small = "small" in cmd.options or command == "ivreg"
    # Without `small`, ivregress and ivreg2 report large-sample statistics:
    # RSS / N, HC0 for `robust`, no finite-sample factor on clustered SEs,
    # z rather than t. sp.iv(small=False) reproduces all of that for 2SLS
    # and LIML; for the other estimators only the robust case has an exact
    # counterpart (HC0).
    large_sample_lost = False
    if not small:
        if method in (None, "2sls", "liml"):
            args["small"] = False
        elif robust == "hc1" and not cluster:
            robust = "hc0"
        else:
            large_sample_lost = True
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
    if args.get("small") is False:
        notes.append(
            "small=False: large-sample standard errors, as Stata reports "
            "without the `small` option."
        )
    if large_sample_lost:
        notes.append(
            f"sp.ivreg standard errors follow `{command} ..., small` (N-K "
            "divisor; cluster SEs add (N-1)/(N-K)). Without `small`, Stata "
            "reports large-sample SEs, which sp.iv(small=False) reproduces "
            f"for 2SLS and LIML only; for {method} they differ by a "
            "degrees-of-freedom factor of about sqrt(N/(N-K))."
        )
    if method in {"liml", "gmm"}:
        notes.append(
            f"Mapped official `ivregress {method}` syntax to "
            f"sp.ivreg(..., method={method!r})."
        )
    code_pairs = ["data=df"]
    for key in ("method", "robust", "cluster", "small"):
        if key in args:
            code_pairs.append(f"{key}={args[key]!r}")
    # small= is an argument of sp.iv; sp.ivreg always reports the `small`
    # convention.
    tool = "iv" if args.get("small") is False else "ivreg"
    python = f"sp.{tool}({formula!r}, {', '.join(code_pairs)})"
    return _emit(tool, args, python, notes)


def _h_ivreghdfe(cmd: StataCommand) -> Dict[str, Any]:
    """``ivreghdfe y x (d = z), absorb(id year)`` -> ``sp.hdfe_ols`` IV part.

    ``sp.hdfe_ols("y ~ x | fe | d ~ z")`` absorbs with the reghdfe engine and
    reports ivreg2's statistics (KP rk LM / Wald F, Cragg-Donald F,
    Anderson-Rubin, Hansen J) in ``result.iv_diagnostics``.
    """
    if not cmd.varlist:
        return _emit_error(
            "ivreghdfe requires an outcome variable", command="ivreghdfe"
        )
    if not any("=" in tok for tok in cmd.varlist):
        # no (endog = instruments) block: ivreghdfe then runs reghdfe
        out = _h_reghdfe(cmd)
        if out.get("ok"):
            out["notes"].append(
                "ivreghdfe without an (endog = instruments) block is reghdfe."
            )
        return out
    parsed = _parse_iv_varlist(list(cmd.varlist), "ivreghdfe")
    if isinstance(parsed, dict):
        return parsed
    y, exog, endog, instruments = parsed
    absorb = cmd.options.get("absorb") or ""
    fe_list, err = _absorb_terms(absorb)
    if err is not None:
        return _emit_error(err, command="ivreghdfe", suggestions=[])
    notes = [
        "Mapped Stata ivreghdfe to sp.hdfe_ols's IV part; first-stage and "
        "weak-instrument statistics are in result.iv_diagnostics."
    ]
    clusters = _cluster_vars(cmd)
    if not fe_list:
        fml = _pyfixest_fml(
            _build_formula(y, exog), [], " + ".join(endog), " + ".join(instruments)
        )
        args: Dict[str, Any] = {"fml": fml}
        if clusters:
            args["cluster"] = clusters[0]
        notes.append("ivreghdfe without absorb() is IV without HDFE (sp.feols).")
        return _emit("feols", args, _feols_code(fml, args.get("cluster")), notes)
    if len(clusters) > 1:
        return _emit_error(
            "ivreghdfe with multi-way clustering is not translated; "
            "sp.hdfe_ols IV supports one-way clusters.",
            command="ivreghdfe",
            suggestions=[],
        )
    formula = (
        f"{_build_formula(y, exog)} | {' + '.join(fe_list)} | "
        f"{' + '.join(endog)} ~ {' + '.join(instruments)}"
    )
    args = {"formula": formula}
    if clusters:
        args["cluster"] = clusters[0]
    elif "robust" in cmd.options or _opt_matches(cmd.options.get("vce"), "robust"):
        args["vce"] = "robust"
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "formula")
    python = f"sp.hdfe_ols({formula!r}, data=df" + (f", {kw})" if kw else ")")
    return _emit("hdfe_ols", args, python, notes)


#: csdid ``method()`` -> ``sp.callaway_santanna(estimator=)``; the table in
#: docs/guides/callaway_santanna.md. ``method(ipw)`` is Abadie's IPW, which
#: StatsPAI (following R ``did``) calls ``'ipw_abadie'``.
_CSDID_METHODS = {"dripw": "dr", "reg": "reg", "stdipw": "stdipw", "ipw": "ipw_abadie"}


def _h_csdid(cmd: StataCommand) -> Dict[str, Any]:
    """``csdid y [x], ivar(id) time(t) gvar(g)`` → ``sp.callaway_santanna``.

    The two commands differ in their defaults, so csdid's are written out:
    ``base_period='varying'`` unless ``long2``, and with ``notyet`` the
    cohort cutoff ``notyet_cutoff='cohort'`` unless ``asinr``.
    """
    if not cmd.varlist:
        return _emit_error("csdid requires an outcome variable", command="csdid")
    opts = cmd.options
    y, xs = cmd.varlist[0], list(cmd.varlist[1:])
    i = opts.get("ivar") or opts.get("id")
    t = opts.get("tvar") or opts.get("time")
    g = opts.get("gvar") or opts.get("cohort")
    missing = [name for name, val in (("ivar", i), ("tvar", t), ("gvar", g)) if not val]
    if missing:
        return _emit_error(
            f"csdid translation needs {missing} option(s); supply them "
            "via Stata's `ivar()` / `tvar()` / `gvar()`.",
            command="csdid",
        )
    notes: List[str] = []
    lost: List[str] = []
    args: Dict[str, Any] = {"y": y, "i": i, "t": t, "g": g}
    if xs:
        args["x"] = xs
    method = (opts.get("method") or "dripw").split()[0].lower()
    if method in _CSDID_METHODS:
        args["estimator"] = _CSDID_METHODS[method]
    else:
        return _emit_error(
            f"csdid method({method}) has no sp.callaway_santanna estimator; "
            f"translated methods: {sorted(_CSDID_METHODS)}.",
            command="csdid",
        )
    if "long" in opts:
        lost.append("long")
        notes.append(
            "csdid `long` flips the sign of the pre-treatment cells, which "
            "sp.callaway_santanna does not mirror; translated as `long2`."
        )
    args["base_period"] = (
        "universal" if "long2" in opts or "long" in lost else "varying"
    )
    if "notyet" in opts:
        args["control_group"] = "notyettreated"
        args["notyet_cutoff"] = "asinr" if "asinr" in opts else "cohort"
    if opts.get("pscoretrim") is not None:
        try:
            args["pscore_trim"] = float(opts.get("pscoretrim") or "")
        except ValueError:
            lost.append("pscoretrim")
            notes.append(f"pscoretrim({opts.get('pscoretrim')}) is not a number.")
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit(
        "callaway_santanna", args, f"sp.callaway_santanna(data=df, {kw})", notes
    )
    out["untranslated_options"] = lost
    out["semantics"] = [
        "csdid defaults written out: base_period='varying' (short gaps) "
        "unless long2; notyet_cutoff='cohort' unless asinr. csdid's "
        "method(ipw) is estimator='ipw_abadie'."
    ]
    return out


def _h_didregress(cmd: StataCommand) -> Dict[str, Any]:
    """``didregress (y x) (D), group(g) time(t)`` -> ``sp.didregress``.

    ``xtdidregress`` is the same call with ``id=`` set to the panel unit of
    ``xtset``, which switches the variance to the ``xtreg, fe`` convention.
    Options that change the estimator (``wildbootstrap()``, ``aggregate()``,
    ``nogteffects``) are not read, so they are reported as untranslated.
    """
    eqs = _equations(list(cmd.varlist))
    if not eqs or len(eqs) != 2 or eqs[0][1] or eqs[1][1]:
        return _emit_error(
            "didregress expects `(outcome [covariates]) (treatment)` equations.",
            command=cmd.command,
        )
    outcome_tokens, treat_tokens = eqs[0][0], eqs[1][0]
    if not outcome_tokens or len(treat_tokens) != 1:
        return _emit_error(
            "didregress needs an outcome and exactly one treatment variable "
            "(the `continuous` treatment form is not translated).",
            command=cmd.command,
        )
    group = cmd.options.get("group")
    time = cmd.options.get("time")
    missing = [name for name, val in (("group", group), ("time", time)) if not val]
    if missing:
        return _emit_error(
            f"{cmd.command} translation needs {missing} option(s); supply "
            "`group()` and `time()`.",
            command=cmd.command,
        )
    assert group is not None and time is not None
    if len(group.split()) != 1:
        return _emit_error(
            f"{cmd.command} with several group() variables (a triple "
            "difference) is not translated; see sp.ddd.",
            command=cmd.command,
        )
    args: Dict[str, Any] = {
        "y": outcome_tokens[0],
        "treat": treat_tokens[0],
        "group": group.split()[0],
        "time": time.split()[0],
    }
    covariates = outcome_tokens[1:]
    if covariates:
        args["covariates"] = covariates
    notes: List[str] = []
    if cmd.command == "xtdidregress":
        unit = cmd.options.get("i") or cmd.options.get("unit")
        args["id"] = unit.split()[0] if unit else "<panel_id>"
        cmd.options.get("t")
        if unit is None:
            notes.append(_PANEL_NOTE)
    cluster = _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster
    notes.append(
        "One coefficient for all treated periods. `estat ptrends` and "
        "`estat granger` are sp.estat(result, 'ptrends' | 'granger')."
    )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("didregress", args, f"sp.didregress(data=df, {kw})", notes)


def _numlist(text: str) -> Optional[List[int]]:
    """A Stata numlist of integers: ``0 1 2``, ``0/3``, ``-4(2)4``."""
    out: List[int] = []
    for tok in text.replace(",", " ").split():
        m = re.fullmatch(r"(-?\d+)(?:/(-?\d+)|\((\d+)\)(-?\d+))?", tok)
        if not m:
            return None
        lo = int(m.group(1))
        if m.group(2) is not None:
            hi, step = int(m.group(2)), 1
        elif m.group(4) is not None:
            hi, step = int(m.group(4)), int(m.group(3))
        else:
            hi, step = lo, 1
        if step < 1 or hi < lo:
            return None
        out.extend(range(lo, hi + 1, step))
    return out or None


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
    opts = cmd.options
    notes: List[str] = []
    lost: List[str] = []
    for name in ("horizons", "horizon"):
        if opts.get(name):
            values = _numlist(opts.get(name) or "")
            if values is None:
                lost.append(name)
                notes.append(f"{name}({opts.get(name)}) is not a numlist of integers.")
            else:
                args["horizon"] = values
    if opts.get("pretrends") is not None:
        try:
            args["pretrends"] = int(opts.get("pretrends") or "")
        except ValueError:
            lost.append("pretrends")
            notes.append(f"pretrends({opts.get('pretrends')}) is not an integer.")
    if "autosample" in opts:
        args["autosample"] = True
    if opts.get("controls"):
        args["controls"] = (opts.get("controls") or "").split()
    cluster = _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("did_imputation", args, f"sp.did_imputation(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    return out


_SYNTH_PREDICTOR = re.compile(r"^([^\W\d]\w*)\((.+)\)$")


def _synth_periods(text: str) -> Optional[List[int]]:
    """Periods of a synth predictor: ``1988``, ``1980(1)1988``, ``1980&1985``."""
    out: List[int] = []
    for piece in text.replace("&", " ").split():
        values = _numlist(piece)
        if values is None:
            return None
        out.extend(values)
    return out or None


def _panel_options(cmd: StataCommand) -> Tuple[Optional[str], Optional[str]]:
    """Unit and time columns from ``i()`` / ``t()``, which ``sp.stata``
    appends from an earlier ``xtset`` / ``tsset``."""
    opts = cmd.options
    unit = opts.get("i") or opts.get("unit")
    time = opts.get("t") or opts.get("time")
    return (unit.split()[0] if unit else None, time.split()[0] if time else None)


_PANEL_NOTE = (
    "The panel variables come from `tsset` / `xtset` on another line. "
    "Replace <panel_id> and <panel_time> with the unit and time columns "
    "(sp.stata does when the snippet declares them)."
)


def _h_synth(cmd: StataCommand) -> Dict[str, Any]:
    """``synth y x1 x2 y(1988) y(1975(1)1980), trunit() trperiod()`` ->
    ``sp.synth(method='classic')``.

    A plain predictor is averaged over ``xperiod()`` (the whole
    pre-treatment period when the option is absent); ``var(periods)``
    is averaged over those periods. Without ``nested`` Stata takes the
    predictor weights from a regression, which is ``v_method='regression'``.
    """
    if not cmd.varlist:
        return _emit_error("synth requires an outcome variable", command="synth")
    opts = cmd.options
    outcome = cmd.varlist[0]
    trunit = opts.get("trunit")
    trperiod = opts.get("trperiod")
    if not (trunit and trperiod):
        return _emit_error(
            "synth needs `trunit(<id>)` and `trperiod(<year>)`.", command="synth"
        )
    unit, time = _panel_options(cmd)
    notes: List[str] = []
    lost: List[str] = []

    xperiod: Optional[List[int]] = None
    if opts.get("xperiod") is not None:
        xperiod = _synth_periods(opts.get("xperiod") or "")
        if xperiod is None:
            lost.append("xperiod")
    covariates: List[str] = []
    special: List[Tuple[str, Any, str]] = []
    for tok in cmd.varlist[1:]:
        m = _SYNTH_PREDICTOR.match(tok)
        if m:
            periods = _synth_periods(m.group(2))
            if periods is None:
                return _emit_error(
                    f"synth: cannot read the periods of predictor {tok!r}.",
                    command="synth",
                    suggestions=[],
                )
            special.append(
                (m.group(1), periods[0] if len(periods) == 1 else periods, "mean")
            )
        elif xperiod is not None:
            special.append((tok, xperiod, "mean"))
        else:
            covariates.append(tok)

    args: Dict[str, Any] = {
        "outcome": outcome,
        "unit": unit or "<panel_id>",
        "time": time or "<panel_time>",
        "treated_unit": _coerce_scalar(trunit),
        "treatment_time": _coerce_scalar(trperiod),
        "method": "classic",
        # Stata's synth fits the treated unit only; the in-space placebo
        # runs are a separate step there (synth_runner, synth2)
        "placebo": False,
    }
    if covariates:
        args["covariates"] = covariates
    if special:
        args["special_predictors"] = special
    if covariates or special:
        if "nested" in opts:
            args["v_method"] = "nested"
            if "allopt" in opts:
                notes.append(
                    "allopt: sp.synth restarts the nested search from several "
                    "starting values by default."
                )
        else:
            args["v_method"] = "regression"
    else:
        "nested" in opts  # nothing to weight without predictors
    for name in ("mspeperiod", "resultsperiod", "counit", "customV"):
        if opts.get(name) is not None:
            lost.append(name)
    if unit is None or time is None:
        notes.append(_PANEL_NOTE)
    notes.append(
        "Stata stores the donor weights rounded to three decimals and builds "
        "e(Y_synthetic) from the rounded weights; sp.synth uses the exact ones."
    )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("synth", args, f"sp.synth(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    return out


_SDID_VCE = {"placebo", "bootstrap", "jackknife", "noinference"}


def _h_sdid(cmd: StataCommand) -> Dict[str, Any]:
    """``sdid y unit time treat, vce(placebo) reps() seed() method()`` ->
    ``sp.sdid``."""
    if len(cmd.varlist) < 4:
        return _emit_error(
            "sdid is positional: `sdid Y unit time treatment, vce(...)`.",
            command="sdid",
        )
    y, unit, time, treat = cmd.varlist[:4]
    opts = cmd.options
    args: Dict[str, Any] = {"outcome": y, "unit": unit, "time": time, "treat": treat}
    lost: List[str] = []
    notes: List[str] = []
    vce = (opts.get("vce") or "").split()
    if vce:
        if vce[0].lower() in _SDID_VCE:
            args["se_method"] = vce[0].lower()
        else:
            lost.append("vce")
    method = (opts.get("method") or "").split()
    if method:
        if method[0].lower() in {"sdid", "did", "sc"}:
            args["method"] = method[0].lower()
        else:
            lost.append("method")
    for name, key in (("reps", "n_reps"), ("seed", "seed")):
        if opts.get(name) is not None:
            try:
                args[key] = int(opts.get(name) or "")
            except ValueError:
                lost.append(name)
    if opts.get("covariates"):
        spec = (opts.get("covariates") or "").replace(",", " , ").split()
        names = spec[: spec.index(",")] if "," in spec else spec
        how = spec[spec.index(",") + 1 :] if "," in spec else []
        if [h.lower() for h in how] == ["projected"]:
            args["covariates"] = names
            args["covariate_method"] = "projected"
        else:
            lost.append("covariates")
            notes.append(
                "covariates() without `projected` is Stata's `optimized` "
                "method, an iteration stopped at its cap. "
                "sp.sdid(covariate_method='optimized', se_method="
                "'noinference') runs the R synthdid version of it, which "
                "differs from Stata's by about 3e-4, so it is not "
                "substituted here; `covariates(x, projected)` translates."
            )
    if "seed" in args:
        notes.append(
            "The seed fixes the placebo / bootstrap draws within StatsPAI; the "
            "draws are not Stata's, so the standard error agrees up to "
            "resampling noise."
        )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("sdid", args, f"sp.sdid(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    return out


def _h_mediate(cmd: StataCommand) -> Dict[str, Any]:
    """``mediate (y x) (m x [, logit]) (treat)`` -> ``sp.mediate`` with the
    potential-outcome estimator (``inference='robust'``)."""
    eqs = _equations(list(cmd.varlist))
    if not eqs or len(eqs) != 3 or not all(eq[0] for eq in eqs):
        return _emit_error(
            "mediate expects `(outcome x) (mediator x) (treatment)` equations.",
            command="mediate",
        )
    (out_vars, out_opts), (med_vars, med_opts), (treat_vars, treat_opts) = eqs
    if out_opts and [o.lower() for o in out_opts] != ["linear"]:
        return _emit_error(
            f"mediate: the outcome model {' '.join(out_opts)!r} is not "
            "translated; sp.mediate(inference='robust') has a linear outcome "
            "model.",
            command="mediate",
            suggestions=[],
        )
    link = [o.lower() for o in med_opts] or ["linear"]
    if len(link) != 1 or link[0] not in {"linear", "logit", "probit"}:
        return _emit_error(
            f"mediate: the mediator model {' '.join(med_opts)!r} is not "
            "translated (linear, logit and probit are).",
            command="mediate",
            suggestions=[],
        )
    if len(treat_vars) != 1 or treat_opts:
        return _emit_error(
            "mediate: only a binary treatment without options is translated.",
            command="mediate",
            suggestions=[],
        )
    if out_vars[1:] != med_vars[1:]:
        return _emit_error(
            "mediate with different covariates in the outcome and mediator "
            "equations is not translated: sp.mediate takes one covariate list.",
            command="mediate",
            suggestions=[],
        )
    covariates = out_vars[1:]
    bad = [c for c in covariates if not re.fullmatch(r"(?:C\()?[^\W\d]\w*\)?", c)]
    if bad:
        return _emit_error(
            f"mediate: covariate term(s) {bad} are not translated; create "
            "them as columns first.",
            command="mediate",
            suggestions=[],
        )
    opts = cmd.options
    args: Dict[str, Any] = {
        "y": out_vars[0],
        "treat": treat_vars[0],
        "mediator": med_vars[0],
        "covariates": covariates,
        "inference": "robust",
        "interaction": "nointeraction" not in opts,
        "mediator_model": link[0],
    }
    lost: List[str] = []
    vce = (opts.get("vce") or "").split()
    if vce and vce[0].lower() != "robust":
        lost.append("vce")
    for shown in ("all", "nie", "nde", "pnie", "tnde", "te", "pomeans", "aequations"):
        shown in opts  # which effects are printed; sp.mediate reports all
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    notes = [
        "Estimates agree with Stata to 1e-9. Effect standard errors agree to "
        "about 1e-4: Stata's variance uses numerical derivatives, sp.mediate "
        "the exact sandwich."
    ]
    out = _emit("mediate", args, f"sp.mediate(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    return out


def _h_bacondecomp(cmd: StataCommand) -> Dict[str, Any]:
    """``bacondecomp y treat, ddetail`` -> ``sp.bacon_decomposition``."""
    if len(cmd.varlist) < 2:
        return _emit_error(
            "bacondecomp needs an outcome and a treatment indicator.",
            command="bacondecomp",
        )
    if len(cmd.varlist) > 2:
        return _emit_error(
            "bacondecomp with control variables is not translated: "
            "sp.bacon_decomposition decomposes the two-way fixed effects "
            "coefficient without controls.",
            command="bacondecomp",
            suggestions=[],
        )
    y, treat = cmd.varlist[:2]
    unit, time = _panel_options(cmd)
    args: Dict[str, Any] = {
        "y": y,
        "treat": treat,
        "time": time or "<panel_time>",
        "id": unit or "<panel_id>",
    }
    notes: List[str] = []
    if unit is None or time is None:
        notes.append(_PANEL_NOTE)
    notes.append(
        "With no never-treated group, bacondecomp's table uses the last "
        "cohort as if it were never treated and need not add up to the "
        "two-way fixed effects coefficient; sp.bacon_decomposition lists "
        "every 2x2 comparison and its weighted sum is that coefficient."
    )
    for shown in ("ddetail", "nograph", "stub", "robust", "gropt"):
        cmd.options.get(shown)
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit(
        "bacon_decomposition", args, f"sp.bacon_decomposition(data=df, {kw})", notes
    )


_RD_KERNELS = {"tri": "triangular", "uni": "uniform", "epa": "epanechnikov"}


def _rd_numbers(raw: Optional[str]) -> Any:
    """``h(5)`` -> 5.0; ``h(5 8)`` -> (5.0, 8.0); anything else -> None."""
    try:
        vals = [float(v) for v in (raw or "").split()]
    except ValueError:
        return None
    if len(vals) == 1:
        return vals[0]
    return tuple(vals) if len(vals) == 2 else None


def _h_rdrobust(cmd: StataCommand) -> Dict[str, Any]:
    """``rdrobust y x, c(0) h(5) vce(cluster id)`` → ``sp.rdrobust``.

    The options keep their Stata names in ``sp.rdrobust``; ``vce(cluster
    v)`` becomes ``cluster=``, ``level(#)`` becomes ``alpha=``. An option
    value that cannot be carried over is reported in ``notes``.
    """
    if len(cmd.varlist) < 2:
        return _emit_error(
            "rdrobust requires y + running variable: `rdrobust y x, c(<v>)`",
            command="rdrobust",
        )
    y, x = cmd.varlist[0], cmd.varlist[1]
    opts = cmd.options
    notes: List[str] = []

    def text(name: str) -> str:
        return opts.get(name) or ""

    lost: List[str] = []

    def skipped(name: str) -> None:
        lost.append(name)
        notes.append(
            f"{name}({opts.get(name)}) is not translated; sp.rdrobust keeps "
            "its default."
        )

    c_raw = opts.get("c", "0")
    try:
        c = float(c_raw) if c_raw is not None else 0.0
    except (TypeError, ValueError):
        return _emit_error(f"rdrobust c({c_raw}) is not a number", command="rdrobust")
    args: Dict[str, Any] = {"y": y, "x": x, "c": c}
    if opts.get("fuzzy"):
        fz = text("fuzzy").split()
        args["fuzzy"] = fz[0]
        if len(fz) > 1:
            notes.append(f"fuzzy(... {' '.join(fz[1:])}) suboption is not translated.")
    for name in ("deriv", "p", "q"):
        if opts.get(name) is not None:
            try:
                args[name] = int(text(name))
            except ValueError:
                skipped(name)
    if opts.get("kernel"):
        kernel = _RD_KERNELS.get(text("kernel").split()[0].lower()[:3])
        if kernel:
            args["kernel"] = kernel
        else:
            skipped("kernel")
    if opts.get("bwselect"):
        args["bwselect"] = text("bwselect").split()[0].lower()
    for name in ("h", "b"):
        if opts.get(name) is not None:
            val = _rd_numbers(opts[name])
            if val is None:
                skipped(name)
            else:
                args[name] = val
    if opts.get("rho") is not None:
        val = _rd_numbers(opts["rho"])
        if isinstance(val, float):
            args["rho"] = val
        else:
            skipped("rho")
    if "h" in args and (
        args.get("b", args["h"]) != args["h"] or args.get("rho", 1.0) != 1.0
    ):
        notes.append(
            "b differs from h: sp.rdrobust's bias-corrected row is then not "
            "exactly Stata's (see the rho note in sp.rdrobust)."
        )
    if opts.get("covs"):
        args["covs"] = text("covs").split()
    if opts.get("vce"):
        vce = text("vce").split()
        kind = vce[0].lower()
        if kind == "cluster" and len(vce) == 2:
            args["cluster"] = vce[1]
        elif kind in {"hc0", "hc1", "hc2", "hc3"} and len(vce) == 1:
            args["vce"] = kind
        elif kind == "nn" and vce[1:] in ([], ["3"]):
            args["vce"] = "nn"
        else:
            skipped("vce")
    if opts.get("masspoints"):
        mp = text("masspoints").split()[0].lower()
        if mp in {"adjust", "check", "off"}:
            args["masspoints"] = mp
        else:
            skipped("masspoints")
    if opts.get("weights"):
        args["weights"] = text("weights").split()[0]
    if opts.get("scalepar") is not None:
        try:
            args["scalepar"] = float(text("scalepar"))
        except ValueError:
            skipped("scalepar")
    if opts.get("level") is not None:
        try:
            args["alpha"] = round(1 - float(text("level")) / 100, 10)
        except ValueError:
            skipped("level")
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k not in ("y", "x", "c"))
    python = f"sp.rdrobust(data=df, y={y!r}, x={x!r}, c={c}" + (
        f", {kw})" if kw else ")"
    )
    out = _emit("rdrobust", args, python, notes)
    out["untranslated_options"] = lost
    return out


def _h_rdbwselect(cmd: StataCommand) -> Dict[str, Any]:
    """``rdbwselect y x, c(0) bwselect(cerrd) vce(cluster id)`` -> ``sp.rdbwselect``.

    Options keep their Stata names; ``vce(cluster v)`` becomes ``cluster=``
    and ``all`` becomes ``all=True``. ``sp.rdbwselect`` uses the
    nearest-neighbour variance of ``vce(nn 3)``, Stata's default, so any
    other ``vce()`` is reported as not carried over, as are ``weights()``
    and ``scaleregul()``.
    """
    if len(cmd.varlist) < 2:
        return _emit_error(
            "rdbwselect requires y + running variable: `rdbwselect y x, c(<v>)`",
            command="rdbwselect",
        )
    y, x = cmd.varlist[0], cmd.varlist[1]
    opts = cmd.options
    notes: List[str] = []
    lost: List[str] = []

    def text(name: str) -> str:
        return opts.get(name) or ""

    def skipped(name: str) -> None:
        lost.append(name)
        notes.append(
            f"{name}({opts.get(name)}) is not translated; sp.rdbwselect keeps "
            "its default."
        )

    c_raw = opts.get("c", "0")
    try:
        c = float(c_raw) if c_raw is not None else 0.0
    except (TypeError, ValueError):
        return _emit_error(
            f"rdbwselect c({c_raw}) is not a number", command="rdbwselect"
        )
    args: Dict[str, Any] = {"y": y, "x": x, "c": c}
    if opts.get("fuzzy"):
        fz = text("fuzzy").split()
        args["fuzzy"] = fz[0]
        if len(fz) > 1:
            notes.append(f"fuzzy(... {' '.join(fz[1:])}) suboption is not translated.")
    for name in ("deriv", "p", "q"):
        if opts.get(name) is not None:
            try:
                args[name] = int(text(name))
            except ValueError:
                skipped(name)
    if opts.get("kernel"):
        kernel = _RD_KERNELS.get(text("kernel").split()[0].lower()[:3])
        if kernel:
            args["kernel"] = kernel
        else:
            skipped("kernel")
    if opts.get("bwselect"):
        args["bwselect"] = text("bwselect").split()[0].lower()
    if opts.get("covs"):
        args["covs"] = text("covs").split()
    vce_is_default = False
    if opts.get("vce"):
        vce = text("vce").split()
        kind = vce[0].lower()
        if kind == "cluster" and len(vce) == 2:
            args["cluster"] = vce[1]
        elif kind == "nn" and vce[1:] in ([], ["3"]):
            vce_is_default = True  # sp.rdbwselect's variance estimator
        else:
            skipped("vce")
    if "all" in opts:
        args["all"] = True
    if opts.get("masspoints"):
        mp = text("masspoints").split()[0].lower()
        if mp in {"adjust", "check", "off"}:
            args["masspoints"] = mp
        else:
            skipped("masspoints")
    for name in ("weights", "scaleregul"):
        if opts.get(name) is not None:
            skipped(name)
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items() if k not in ("y", "x", "c"))
    python = f"sp.rdbwselect(data=df, y={y!r}, x={x!r}, c={c}" + (
        f", {kw})" if kw else ")"
    )
    out = _emit("rdbwselect", args, python, notes)
    out["untranslated_options"] = lost
    if vce_is_default:
        out["_vce_is_sp_default"] = True
    return out


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
    if robust == "hc1":
        # Stata's vce(robust) for a maximum-likelihood command carries
        # N/(N-1), which is sp's robust='robust'; HC1's N/(N-K) is regress's.
        robust = "robust"
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
    for opt in ("exposure", "offset"):
        val = cmd.options.get(opt)
        if val and sp_fn in ("poisson", "nbreg"):
            args[opt] = val.split()[0]
            code_pairs.append(f"{opt}={args[opt]!r}")
        elif val:
            return _emit_error(
                f"{display_name}: option {opt}() is not carried over by "
                f"sp.{sp_fn}.",
                command=display_name,
                suggestions=[],
            )
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
    # sp.tobit's vce='robust' / cluster= carry Stata's ML factors
    cluster = _vce_cluster(cmd)
    if cluster:
        args["cluster"] = cluster
        code_pairs.append(f"cluster={cluster!r}")
    elif _robust_kind(cmd) == "hc1":
        args["vce"] = "robust"
        code_pairs.append("vce='robust'")
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
    """``rdplot y x, c() p() nbins() binselect() kernel() h() ci() shade``."""
    if len(cmd.varlist) < 2:
        return _emit_error(
            "rdplot needs y + running variable: `rdplot y x, c(<v>)`", command="rdplot"
        )
    y, x = cmd.varlist[0], cmd.varlist[1]
    opts = cmd.options
    c_raw = opts.get("c", "0")
    try:
        c = float(c_raw) if c_raw is not None else 0.0
    except (TypeError, ValueError):
        c = 0.0
    args: Dict[str, Any] = {"y": y, "x": x, "c": c}
    lost: List[str] = []
    notes: List[str] = []
    if opts.get("p") is not None:
        try:
            args["p"] = int(opts.get("p") or "")
        except ValueError:
            lost.append("p")
    if opts.get("nbins") is not None:
        bins = (opts.get("nbins") or "").split()
        try:
            counts = [int(b) for b in bins]
        except ValueError:
            counts = []
        if len(counts) in (1, 2) and len(set(counts)) == 1:
            args["nbins"] = counts[0]
        else:
            lost.append("nbins")
            notes.append(
                f"nbins({opts.get('nbins')}): sp.rdplot takes one bin count "
                "for both sides."
            )
    if opts.get("binselect"):
        args["binselect"] = (opts.get("binselect") or "").split()[0].lower()
    if opts.get("kernel"):
        kernel = _RD_KERNELS.get((opts.get("kernel") or "").split()[0].lower()[:3])
        if kernel:
            args["kernel"] = kernel
        else:
            lost.append("kernel")
    if opts.get("h") is not None:
        val = _rd_numbers(opts.get("h"))
        if isinstance(val, float):
            args["h"] = val
        else:
            lost.append("h")
    if opts.get("ci") is not None:
        try:
            args["ci_level"] = round(float(opts.get("ci") or "") / 100, 10)
        except ValueError:
            lost.append("ci")
    else:
        # rdplot draws no confidence intervals unless ci() is given.
        args["hide_ci"] = True
    if "shade" in opts:
        args["shade_ci"] = True
    if opts.get("covs"):
        args["covs"] = (opts.get("covs") or "").split()
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("rdplot", args, f"sp.rdplot(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    return out


def _h_rddensity(cmd: StataCommand) -> Dict[str, Any]:
    if not cmd.varlist:
        return _emit_error("rddensity requires a running variable", command="rddensity")
    x = cmd.varlist[0]
    opts = cmd.options
    c_raw = opts.get("c", "0")
    try:
        c = float(c_raw) if c_raw is not None else 0.0
    except (TypeError, ValueError):
        c = 0.0
    args: Dict[str, Any] = {"x": x, "c": c}
    lost: List[str] = []
    if opts.get("p") is not None:
        try:
            args["p"] = int(opts.get("p") or "")
        except ValueError:
            lost.append("p")
    if opts.get("h") is not None:
        val = _rd_numbers(opts.get("h"))
        if val is None:
            lost.append("h")
        else:
            args["h"] = val
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("rddensity", args, f"sp.rddensity(data=df, {kw})")
    out["untranslated_options"] = lost
    return out


def _equations(tokens: List[str]) -> Optional[List[Tuple[List[str], List[str]]]]:
    """Parenthesised equations of a tokenised head.

    ``( y x1 x2 ) ( treat z , logit )`` -> ``[(['y', 'x1', 'x2'], []),
    (['treat', 'z'], ['logit'])]``: the variables and the words after the
    equation's own comma. ``None`` when the tokens are not a sequence of
    equations.
    """
    eqs: List[Tuple[List[str], List[str]]] = []
    i = 0
    while i < len(tokens):
        if tokens[i] != "(":
            return None
        try:
            j = tokens.index(")", i)
        except ValueError:
            return None
        body = tokens[i + 1 : j]
        if "(" in body:
            return None
        if "," in body:
            k = body.index(",")
            eqs.append((body[:k], [t for t in body[k + 1 :] if t != ","]))
        else:
            eqs.append((body, []))
        i = j + 1
    return eqs


def _vce_nn(vce: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """``vce(robust, nn(3))`` -> ``('robust', 3)``; ``vce(iid)`` -> ``('iid', None)``."""
    if not vce:
        return None, None
    kind = vce.replace(",", " ").split()[0].lower()
    m = re.search(r"nn\(\s*(\d+)\s*\)", vce, flags=re.I)
    return kind, (int(m.group(1)) if m else None)


def _h_teffects(cmd: StataCommand) -> Dict[str, Any]:
    """``teffects nnmatch (y x) (treat)`` / ``teffects psmatch (y) (treat z)``
    / ``teffects ipw`` / ``teffects aipw``.

    ``nnmatch`` goes to ``sp.match(method='nnmatch')``, the same estimator
    (ties kept, Abadie-Imbens variance, ``biasadj()`` / ``ematch()``).
    ``psmatch`` goes to propensity-score matching with ties kept and, for
    the ATET, the Abadie-Imbens (2016) standard error Stata reports.
    ``ra`` and ``ipwra`` fit a separate outcome model in each treatment arm,
    which no single sp call reproduces, so they are refused.
    """
    if not cmd.varlist:
        return _emit_error(
            "teffects: expected `teffects <method> (out_eq) (treat_eq) [, opts]`",
            command="teffects",
        )
    method = cmd.varlist[0].lower()
    eqs = _equations(list(cmd.varlist[1:]))
    if not eqs or len(eqs) != 2 or not eqs[0][0] or not eqs[1][0]:
        return _emit_error(
            "teffects: expected `teffects <method> (out_eq) (treat_eq) [, opts]`",
            command="teffects",
        )
    (out_vars, out_opts), (treat_vars, treat_opts) = eqs
    y, out_xs = out_vars[0], out_vars[1:]
    treat, treat_xs = treat_vars[0], treat_vars[1:]
    opts = cmd.options
    estimand = "ATT" if "atet" in opts else "ATE"
    "ate" in opts  # the default; reading it marks it as handled
    notes: List[str] = []
    lost: List[str] = []

    if method in {"ra", "ipwra"}:
        return _emit_error(
            f"teffects {method} fits a separate outcome regression in each "
            "treatment arm and averages its predictions"
            + (" with inverse-probability weights" if method == "ipwra" else "")
            + "; it is not translated line by line. sp.aipw is the doubly "
            "robust estimator on the same two models.",
            command="teffects",
            suggestions=[],
        )
    if out_opts:
        return _emit_error(
            f"teffects: the outcome-model option {' '.join(out_opts)!r} is "
            "not translated.",
            command="teffects",
            suggestions=[],
        )

    if method == "nnmatch":
        if treat_xs or treat_opts:
            return _emit_error(
                "teffects nnmatch takes the matching covariates in the outcome "
                "equation: `teffects nnmatch (y x1 x2) (treat)`.",
                command="teffects",
            )
        if not out_xs:
            return _emit_error(
                "teffects nnmatch needs matching covariates: "
                "`teffects nnmatch (y x1 x2) (treat)`.",
                command="teffects",
            )
        args: Dict[str, Any] = {
            "y": y,
            "treat": treat,
            "covariates": out_xs,
            "method": "nnmatch",
            "estimand": estimand,
        }
        raw_nn = opts.get("nneighbor")
        if raw_nn is not None:
            try:
                args["n_matches"] = int(raw_nn)
            except (TypeError, ValueError):
                lost.append("nneighbor")
        metric = opts.get("metric")
        if metric is not None:
            head = metric.split()[0].lower() if metric.split() else ""
            full = next(
                (
                    m
                    for m in ("mahalanobis", "ivariance", "euclidean")
                    if head and m.startswith(head) and len(head) >= 3
                ),
                None,
            )
            if full is None:
                lost.append("metric")
                notes.append(
                    f"metric({metric}) is not one sp.match(method='nnmatch') has."
                )
            else:
                args["metric"] = full
        if opts.get("ematch"):
            args["exact"] = (opts.get("ematch") or "").split()
        if opts.get("biasadj"):
            args["bias_adjust"] = (opts.get("biasadj") or "").split()
        if opts.get("caliper") is not None:
            try:
                args["caliper"] = float(opts.get("caliper") or "")
            except ValueError:
                lost.append("caliper")
        if opts.get("dtolerance") is not None:
            try:
                args["dtol"] = float(opts.get("dtolerance") or "")
            except ValueError:
                lost.append("dtolerance")
        kind, h = _vce_nn(opts.get("vce"))
        if kind == "iid":
            args["vce"] = "iid"
        elif kind == "robust":
            args["vce"] = "robust"
            if h is not None:
                args["vce_nn"] = h
        elif kind is not None:
            lost.append("vce")
        if opts.get("osample") is not None:
            notes.append(
                "osample(): sp.match raises when a unit has no admissible "
                "match; the exception's `unmatched` attribute is the flag "
                "Stata would store. For the ATET only treated units are "
                "flagged, whereas Stata also flags controls no treated unit "
                "can use."
            )
        if opts.get("generate") is not None:
            notes.append(
                "generate(): the matching weights are in "
                "result.model_info['match_weights']."
            )
        kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
        out = _emit("match", args, f"sp.match(data=df, {kw})", notes)
        out["untranslated_options"] = lost
        return out

    if method == "psmatch":
        if out_xs:
            return _emit_error(
                "teffects psmatch takes only the outcome in its first "
                "equation: `teffects psmatch (y) (treat x1 x2)`.",
                command="teffects",
            )
        if not treat_xs:
            return _emit_error(
                "teffects psmatch needs propensity-score covariates: "
                "`teffects psmatch (y) (treat x1 x2)`.",
                command="teffects",
            )
        model = treat_opts[0].lower() if treat_opts else "logit"
        if model != "logit" or len(treat_opts) > 1:
            return _emit_error(
                f"teffects psmatch with the treatment model "
                f"{' '.join(treat_opts)!r} is not translated; sp.match "
                "estimates the propensity score by logit.",
                command="teffects",
                suggestions=[],
            )
        args = {
            "y": y,
            "treat": treat,
            "covariates": treat_xs,
            "distance": "propensity",
            "estimand": estimand,
            "ties": "all",
        }
        raw_nn = opts.get("nneighbor")
        if raw_nn is not None:
            try:
                args["n_matches"] = int(raw_nn)
            except (TypeError, ValueError):
                lost.append("nneighbor")
        if opts.get("caliper") is not None:
            try:
                args["caliper"] = float(opts.get("caliper") or "")
            except ValueError:
                lost.append("caliper")
        kind, h = _vce_nn(opts.get("vce"))
        if kind not in (None, "robust"):
            lost.append("vce")
        if estimand == "ATT":
            args["se_method"] = "abadie_imbens_2016"
            if h is not None:
                args["ai_matches"] = max(h - 1, 1)
            notes.append(
                "Standard error: Abadie-Imbens (2016), which charges for the "
                "estimated propensity score, as teffects psmatch reports."
            )
        else:
            lost.append("ate")
            notes.append(
                "The ATE point estimate is the same matching estimator, but "
                "sp.match has the Abadie-Imbens (2016) standard error for the "
                "ATET only; its ATE standard error is not Stata's."
            )
        if opts.get("generate") is not None:
            notes.append(
                "generate(): sp.stata creates the match variables (the row "
                "numbers of each treated unit's nearest controls); from the "
                "call itself they are in "
                "result.model_info['matched_data'] (_n1, _n2, ...)."
            )
        if opts.get("osample") is not None:
            notes.append(
                "osample(): sp.stata creates the variable and stops, as "
                "Stata does, when an observation has too few matches within "
                "the caliper; the call itself keeps the treated units it "
                "can match."
            )
        kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
        out = _emit("match", args, f"sp.match(data=df, {kw})", notes)
        out["untranslated_options"] = lost
        return out

    if treat_opts and [t.lower() for t in treat_opts] != ["logit"]:
        return _emit_error(
            f"teffects {method} with the treatment model "
            f"{' '.join(treat_opts)!r} is not translated.",
            command="teffects",
            suggestions=[],
        )
    if method == "ipw":
        args = {"y": y, "treat": treat, "covariates": treat_xs, "estimand": estimand}
        python = (
            f"sp.ipw(data=df, y={y!r}, treat={treat!r}, "
            f"covariates={treat_xs!r}, estimand={estimand!r})"
        )
        return _emit("ipw", args, python)
    if method == "aipw":
        if out_xs and treat_xs and out_xs != treat_xs:
            return _emit_error(
                "teffects aipw with different covariates in the outcome and "
                "treatment models is not translated: sp.aipw takes one "
                "covariate list for both.",
                command="teffects",
                suggestions=[],
            )
        covs = treat_xs or out_xs
        args = {"y": y, "treat": treat, "covariates": covs, "estimand": estimand}
        python = (
            f"sp.aipw(data=df, y={y!r}, treat={treat!r}, "
            f"covariates={covs!r}, estimand={estimand!r})"
        )
        return _emit("aipw", args, python)
    return _emit_error(
        f"teffects method {method!r} not supported "
        f"(known: nnmatch / psmatch / ipw / aipw)",
        command="teffects",
    )


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
    lost: List[str] = []
    cmd.options.get("probit")  # psmatch2's default: same message either way
    if "logit" not in cmd.options:
        lost.append("probit")
        notes.append(
            "Stata psmatch2 estimates the propensity score by probit unless "
            "`logit` is given; sp.psmatch2 uses a logit, so the scores and "
            "possibly the matches differ. Add `logit` in Stata for an exact "
            "counterpart."
        )
    if "ties" in cmd.options:
        args["ties"] = True
    if "ate" in cmd.options:
        args["ate"] = True

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
        "ties",
        "ate",
    ):
        if key in args:
            code_pairs.append(f"{key}={args[key]!r}")
    python = f"sp.psmatch2({', '.join(code_pairs)})"
    out = _emit("psmatch2", args, python, notes)
    out["untranslated_options"] = lost
    return out


#: ``margins`` options with a faithful ``sp.margins`` counterpart. Anything
#: else (``eyex``, ``over()``, ``predict()``, ``vce(unconditional)`` ...) asks
#: for a different quantity, so it is refused rather than dropped.
_MARGINS_OPTIONS = frozenset({"dydx", "atmeans", "at", "post", "noatlegend"})


def _h_margins(cmd: StataCommand) -> Dict[str, Any]:
    """Stata ``margins, dydx(...) [atmeans] [at(v=#)]`` → ``sp.margins``.

    Only marginal effects translate: bare ``margins`` / ``margins <factor>``
    are predictive margins, a different quantity from ``sp.margins``'s
    average marginal effects.
    """
    dydx = (cmd.options.get("dydx") or "").split()
    if not dydx:
        return _emit_error(
            "only `margins, dydx(...)` translates; predictive margins "
            "(`margins` / `margins <factor>`) map to sp.margins_at(result, "
            "data=df, at={...}) or sp.contrast.",
            command="margins",
        )
    unsupported = sorted(set(cmd.options) - _MARGINS_OPTIONS)
    if unsupported or cmd.varlist:
        return _emit_error(
            f"margins {'options ' + str(unsupported) if unsupported else 'varlist'}"
            " not translated; call sp.margins(result, ...) directly.",
            command="margins",
        )
    args: Dict[str, Any] = {}
    if not {"*", "_all"} & set(dydx):
        args["variables"] = dydx
    if "atmeans" in cmd.options:
        args["method"] = "mem"
    if cmd.options.get("at"):
        at: Dict[str, Any] = {}
        for part in cmd.options["at"].split():
            name, sep, value = part.partition("=")
            if not (sep and name and value) or "(" in value:
                return _emit_error(
                    f"at({cmd.options['at']}) not translated: only "
                    "`at(var=# var=# ...)` with one value per variable.",
                    command="margins",
                )
            at[name] = _coerce_scalar(value)
        args["at"] = at
    notes = [
        "sp.margins takes a fitted result, not data — pipe the "
        "previous estimator's result_id (or fit a model first)."
    ]
    pairs = ["result"] + [f"{k}={v!r}" for k, v in args.items()]
    return _emit("margins", args, f"sp.margins({', '.join(pairs)})", notes)


def _h_marginsplot(cmd: StataCommand) -> Dict[str, Any]:
    """Stata ``marginsplot`` → ``sp.marginsplot(margins_table)``."""
    notes = ["sp.marginsplot plots the table sp.margins returned; pipe that in."]
    return _emit("marginsplot", {}, "sp.marginsplot(result)", notes)


def _h_contrast(cmd: StataCommand) -> Dict[str, Any]:
    """Stata ``contrast x`` → ``sp.contrast(result, data, variable)``."""
    if len(cmd.varlist) != 1 or cmd.options:
        return _emit_error(
            "only `contrast <one variable>` translates; call "
            "sp.contrast(result, data=df, variable=...) directly.",
            command="contrast",
        )
    variable = cmd.varlist[0].split(".")[-1]  # Stata factor prefix ``i.x``
    args: Dict[str, Any] = {"variable": variable}
    notes = [
        "sp.contrast takes a fitted result and the estimation data; pipe the "
        "previous estimator's result_id."
    ]
    python = f"sp.contrast(result, data=df, variable={variable!r})"
    return _emit("contrast", args, python, notes)


def _h_test(cmd: StataCommand) -> Dict[str, Any]:
    """Stata ``test x1 x2`` / ``test x1 = x2`` → ``sp.test(result, hypothesis)``.

    ``sp.test`` parses Stata's own restriction syntax (joint ``x1 x2``,
    chained ``x1 = x2 = 0``, grouped ``(x1 = 0) (x2 = 1)``, ``_b[x]``), so the
    command text is passed through verbatim.
    """
    if not cmd.varlist:
        return _emit_error("test requires a restriction, e.g. `test x1 = x2`")
    if cmd.options:
        return _emit_error(
            f"test options {sorted(cmd.options)} are not translated; call "
            "sp.test(result, hypothesis) directly.",
            command="test",
        )
    hypothesis = _FV_COEFFICIENT.sub(_fv_coefficient, " ".join(cmd.varlist))
    args: Dict[str, Any] = {"hypothesis": hypothesis}
    notes = ["sp.test takes a fitted result; pipe the previous estimator's result_id."]
    python = f"sp.test(result, hypothesis={args['hypothesis']!r})"
    return _emit("test", args, python, notes)


#: An interaction of continuous variables named the way Stata prints it,
#: ``c.x#c.z``: the fitted result calls that coefficient ``x:z``.
_FV_COEFFICIENT = re.compile(r"(?<![\w.])c\.[^\W\d]\w*(?:#c\.[^\W\d]\w*)+(?![\w.(])")


def _fv_coefficient(m: "re.Match[str]") -> str:
    parts = [p[2:] for p in m.group(0).split("#")]
    if len(set(parts)) == 1:
        return f"I({parts[0]}**{len(parts)})"
    return ":".join(parts)


def _h_lincom(cmd: StataCommand) -> Dict[str, Any]:
    """Stata ``lincom x1 - 2*x2`` → ``sp.lincom(result, expression)``."""
    if not cmd.varlist:
        return _emit_error("lincom requires an expression, e.g. `lincom x1 + x2`")
    args: Dict[str, Any] = {"expression": " ".join(cmd.varlist)}
    level = cmd.options.pop("level", None) if cmd.options else None
    if cmd.options:
        return _emit_error(
            f"lincom options {sorted(cmd.options)} are not translated; call "
            "sp.lincom(result, expression) directly.",
            command="lincom",
        )
    if level is not None:
        args["alpha"] = round(1 - float(level) / 100, 10)
    notes = [
        "sp.lincom takes a fitted result; pipe the previous estimator's result_id."
    ]
    pairs = ["result"] + [f"{k}={v!r}" for k, v in args.items()]
    return _emit("lincom", args, f"sp.lincom({', '.join(pairs)})", notes)


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
    # sp.xtdpdsys reproduces Stata's xtdpdsys by default (exogenous regressors
    # in the differenced equation, h(2)); the translation maps straight onto it.
    panel_id = cmd.options.get("i") or cmd.options.get("id") or "<panel_id>"
    args: Dict[str, Any] = {
        "y": y,
        "x": xs,
        "id": panel_id if panel_id != "<panel_id>" else None,
        "twostep": "twostep" in cmd.options,
        "robust": "robust" in cmd.options
        or _opt_matches(cmd.options.get("vce"), "robust"),
    }
    lost: List[str] = []
    notes: List[str] = []
    panel_time = cmd.options.get("t") or cmd.options.get("time")
    if panel_time:
        args["time"] = panel_time
    lags_opt = cmd.options.get("lags")
    if lags_opt:
        try:
            args["lags"] = int(lags_opt)
        except (TypeError, ValueError):
            lost.append("lags")
            notes.append(f"lags({lags_opt}) is not an integer.")
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
    if panel_time:
        code_pairs.append(f"time={panel_time!r}")
    if "lags" in args:
        code_pairs.append(f"lags={args['lags']}")
    if args["twostep"]:
        code_pairs.append("twostep=True")
    # written out: the sp default is robust=True, Stata's is not
    code_pairs.append(f"robust={args['robust']}")
    python = f"sp.{sp_kind}({', '.join(code_pairs)})"
    out = _emit(sp_kind, args, python, notes)
    out["untranslated_options"] = lost
    return out


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


def _h_newey(cmd: StataCommand) -> Dict[str, Any]:
    """``newey y x, lag(4)`` -> ``sp.regress(..., robust='hac', hac_lags=4,
    hac_small=True)``: OLS with Newey-West standard errors, which Stata
    scales by N/(N-K)."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("newey requires an outcome variable", command="newey")
    raw = cmd.options.get("lag")
    try:
        lag = int(str(raw).strip())
    except (TypeError, ValueError):
        return _emit_error(
            "newey requires lag(#), the number of autocovariances",
            command="newey",
            suggestions=[],
        )
    formula = _build_formula(y, xs)
    args: Dict[str, Any] = {
        "formula": formula,
        "robust": "hac",
        "hac_lags": lag,
        "hac_small": True,
    }
    python = (
        f"sp.regress({formula!r}, robust='hac', hac_lags={lag}, "
        "hac_small=True, data=df)"
    )
    semantics = [
        "Rows are taken in the DataFrame's order; sort by the time variable "
        "of `tsset` first. newey refuses a series with gaps unless `force` "
        "is given; the sp call does not check."
    ]
    return _emit("regress", args, python, semantics=semantics)


def _h_dfuller(cmd: StataCommand) -> Dict[str, Any]:
    """``dfuller y, lags(4) trend`` -> ``sp.unitroot(df, 'y', test='adf')``."""
    if len(cmd.varlist) != 1:
        return _emit_error(
            "dfuller takes one variable", command="dfuller", suggestions=[]
        )
    args: Dict[str, Any] = {"y": cmd.varlist[0], "test": "adf", "lags": 0}
    raw = cmd.options.get("lags")
    if raw is not None:
        try:
            args["lags"] = int(str(raw).strip())
        except ValueError:
            return _emit_error(
                f"dfuller: lags({raw}) is not an integer", command="dfuller"
            )
    if "trend" in cmd.options:
        args["trend"] = "ct"
    elif "noconstant" in cmd.options:
        args["trend"] = "n"
    # `drift` changes the null distribution (a t distribution): left unread,
    # so it is reported as untranslated.
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    semantics = [
        "The series is read in the DataFrame's row order; sort by the time "
        "variable of `tsset` first.",
        "The statistic is Stata's. Critical values are MacKinnon's response "
        "surface, not the interpolated Fuller table dfuller prints, so they "
        "differ in the second decimal; the p-value is MacKinnon's, as "
        "Stata's.",
    ]
    return _emit("unitroot", args, f"sp.unitroot(df, {kw})", semantics=semantics)


_TTEST_EQ = re.compile(r"([^\W\d]\w*)\s*={1,2}\s*(\S+)")


def _h_ttest(cmd: StataCommand) -> Dict[str, Any]:
    """``ttest y == 5`` / ``ttest y, by(g) unequal`` / ``ttest y == x`` ->
    ``sp.ttest``."""
    text = " ".join(cmd.varlist).strip()
    args: Dict[str, Any] = {}
    eq = _TTEST_EQ.fullmatch(text)
    two_samples = False
    if eq:
        args["y"] = eq.group(1)
        try:
            args["mu"] = float(eq.group(2))
        except ValueError:
            args["other"] = eq.group(2)
            if "unpaired" in cmd.options:
                args["paired"] = False
                two_samples = True
    elif len(cmd.varlist) == 1 and cmd.options.get("by"):
        args["y"] = cmd.varlist[0]
        args["by"] = str(cmd.options["by"]).split()[0]
        two_samples = True
        cmd.options.get("unpaired")  # by() groups are unpaired already
    else:
        return _emit_error(
            "ttest: expected `ttest y == #`, `ttest y, by(group)` or "
            "`ttest y == x [, unpaired]`",
            command="ttest",
            suggestions=[],
        )
    if two_samples:
        # only meaningful for two independent samples; elsewhere they stay
        # unread and are reported as untranslated
        if "welch" in cmd.options:
            args["welch"] = True
        elif "unequal" in cmd.options:
            args["unequal"] = True
        cmd.options.get("unequal")
    level = cmd.options.get("level")
    if level is not None:
        try:
            args["alpha"] = round(1 - float(level) / 100, 10)
        except (TypeError, ValueError):
            return _emit_error(
                f"ttest: level({level}) is not a number", command="ttest"
            )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("ttest", args, f"sp.ttest(df, {kw})")


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
    "qreg": _h_qreg,
    "reghdfe": _h_reghdfe,
    "areg": _h_areg,
    "ivreg2": _h_ivreg2,
    "ivregress": _h_ivreg2,  # close-enough mapping
    "ivreg": _h_ivreg2,  # pre-Stata-10 syntax, still common in teaching files
    "ivreghdfe": _h_ivreghdfe,
    "csdid": _h_csdid,
    "didregress": _h_didregress,
    "xtdidregress": _h_didregress,
    "did_imputation": _h_did_imputation,
    "synth": _h_synth,
    "sdid": _h_sdid,
    "mediate": _h_mediate,
    "bacondecomp": _h_bacondecomp,
    "rdrobust": _h_rdrobust,
    "rdbwselect": _h_rdbwselect,
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
    "marginsplot": _h_marginsplot,
    "contrast": _h_contrast,
    "test": _h_test,
    "lincom": _h_lincom,
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
    "summarize": _h_summarize,
    "sum": _h_summarize,
    "su": _h_summarize,
    "sum2docx": _h_summarize,
    "tabstat": _h_tabstat,
    "correlate": _h_correlate,
    "pwcorr": _h_correlate,
    "ttest": _h_ttest,
    "dfuller": _h_dfuller,
    "newey": _h_newey,
}


#: Stata lets a command be cut down to its documented minimal abbreviation
#: (``regress`` to ``reg``, ``summarize`` to ``su``). ``full name -> shortest
#: accepted length``; only commands whose abbreviation rule is in the Stata
#: manual entry are listed.
_COMMAND_ABBREVIATIONS: Dict[str, int] = {
    "regress": 3,
    "summarize": 2,
    "correlate": 3,
    "probit": 4,
    "logit": 4,
    "poisson": 3,
    "tobit": 3,
    "test": 2,
}


def _resolve_command(name: str) -> str:
    """Expand a Stata command abbreviation; unknown names pass through."""
    if name in STATA_COMMAND_MAP:
        return name
    for full, shortest in _COMMAND_ABBREVIATIONS.items():
        if shortest <= len(name) < len(full) and full.startswith(name):
            return full
    return name


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


# ---------------------------------------------------------------------------
# Semantic normalisation shared by every handler
# ---------------------------------------------------------------------------
#
# The handlers were written against plain varlists. Two pieces of ordinary
# Stata grammar used to pass through verbatim -- factor-variable notation
# (``i.g``, ``c.x#c.x``) and the weight clause (``[aw=w]``) -- and ended up
# inside the formula with ``ok=True``, i.e. a confident but broken
# translation. They are now translated or refused before any handler runs.

_WEIGHT_RE = re.compile(
    r"\[\s*(aw|aweights?|pw|pweights?|fw|fweights?|iw|iweights?)\s*=\s*"
    r"([A-Za-z_]\w*)\s*\]",
    re.I,
)
_TS_OP_RE = re.compile(r"^(?:[LDFS]\d*|[LDF]\([^)]*\))\.[A-Za-z_]", re.I)
#: Tokens that use factor-variable syntax (anything else -- plain names,
#: lincom / test expressions -- passes through untouched).
_FV_RE = re.compile(r"^(?:i|c|ibn|ib\d+|\d+)\.[A-Za-z_(]|#", re.I)


def _fv_part(tok: str) -> Optional[str]:
    """One factor-variable component -> formula syntax (None: unsupported)."""
    m = re.fullmatch(r"i\.([A-Za-z_]\w*)", tok)
    if m:
        return f"C({m.group(1)})"
    m = re.fullmatch(r"ib(\d+)\.([A-Za-z_]\w*)", tok)
    if m:
        return f"C({m.group(2)}, Treatment({m.group(1)}))"
    m = re.fullmatch(r"ibn\.([A-Za-z_]\w*)", tok)
    if m:
        return f"C({m.group(1)})"
    m = re.fullmatch(r"c\.([A-Za-z_]\w*)", tok)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(\d+)\.([A-Za-z_]\w*)", tok)
    if m:
        # 1.post: the indicator of that level, as a number so that it stays
        # one column inside an interaction.
        return f"I(1 * ({m.group(2)} == {m.group(1)}))"
    if re.fullmatch(r"[A-Za-z_]\w*", tok):
        return tok
    return None


def _fv_token(tok: str) -> Optional[str]:
    """A varlist token in factor-variable notation -> formula syntax."""
    if "##" in tok:
        # a##b is a + b + a#b; going through the '#' rule keeps c.x##c.x
        # as x + I(x**2) (x*x would collapse to x in the formula language).
        raw = tok.split("##")
        mains = [_fv_part(t) for t in raw]
        inter = _fv_token("#".join(raw))
        if None in mains or inter is None:
            return None
        terms: List[str] = []
        for t in mains + [inter]:  # type: ignore[operator]
            if t not in terms:
                terms.append(t)  # type: ignore[arg-type]
        return " + ".join(terms)
    if "#" in tok:
        raw = tok.split("#")
        # In `a#b` a variable without a prefix is a factor (i.a). Next to a
        # single level (`treat#1.pre`) Stata then builds one column per level
        # of `a`, which a product of two columns does not reproduce.
        if any(re.fullmatch(r"\d+\.[A-Za-z_]\w*", t) for t in raw) and any(
            re.fullmatch(r"(?:i\.)?[A-Za-z_]\w*", t) for t in raw
        ):
            return None
        parts = [_fv_part(t) for t in raw]
        if None in parts:
            return None
        # c.x#c.x is the square of x (x:x would collapse to x)
        if len(set(parts)) == 1 and all(r.startswith("c.") for r in raw):
            return f"I({parts[0]}**{len(parts)})"
        return ":".join(parts)  # type: ignore[arg-type]
    return _fv_part(tok)


_VARLIST_RANGE = re.compile(r"^([^\W\d]\w*)-([^\W\d]\w*)$")
_NAME = re.compile(r"[^\W\d]\w*$")


def _tokenise_head(text: str) -> List[str]:
    """Split a command's varlist into tokens, equation punctuation apart.

    A parenthesis that opens an *equation* -- ``(d = z1 z2)``,
    ``(y x1 x2) (treat)`` -- becomes a token of its own, as do the ``=`` and
    ``,`` inside it and the closing parenthesis, so ``(y x)(treat)`` and
    ``(d=z)`` read the same as their spaced-out spellings. A parenthesis
    attached to what precedes it belongs to that token and is kept whole,
    blanks included: ``cigsale(1988)``, ``beer(1984(1)1988)``,
    ``c.(x x2)``, ``d##c.(x x2)#i.g`` -- unless it holds an ``=`` or is
    followed directly by another parenthesis, which only an equation is.
    """
    tokens: List[str] = []
    cur: List[str] = []
    eq_depth = 0
    i, n = 0, len(text)

    def flush() -> None:
        if cur:
            tokens.append("".join(cur))
            cur.clear()

    while i < n:
        ch = text[i]
        if ch.isspace():
            flush()
        elif ch == "(":
            depth, j = 0, i
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            # A parenthesis glued to a name is still an equation when it
            # holds an `=` (``x(d=z)``) or another one follows it directly
            # (``psmatch(y)(treat x)``).
            glued_equation = "=" in text[i:j] or text[j + 1 : j + 2] == "("
            if cur and not glued_equation:
                # attached: copy through the matching parenthesis
                cur.append(text[i : j + 1])
                i = j
            else:
                flush()
                tokens.append("(")
                eq_depth += 1
        elif ch == ")" and eq_depth > 0:
            flush()
            tokens.append(")")
            eq_depth -= 1
        elif ch in "=," and eq_depth > 0:
            flush()
            tokens.append(ch)
        else:
            cur.append(ch)
        i += 1
    flush()
    return _join_spaced_ranges(tokens)


def _join_spaced_ranges(tokens: List[str]) -> List[str]:
    """``a - b`` (and ``a -b``, ``a- b``) is the varlist range ``a-b``."""
    out: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if (
            tok == "-"
            and out
            and nxt is not None
            and _NAME.match(out[-1])
            and _NAME.match(nxt)
        ):
            out[-1] = f"{out[-1]}-{nxt}"
            i += 2
            continue
        if (
            tok.endswith("-")
            and _NAME.match(tok[:-1] or "-")
            and nxt
            and _NAME.match(nxt)
        ):
            out.append(tok + nxt)
            i += 2
            continue
        if (
            tok.startswith("-")
            and _NAME.match(tok[1:] or "-")
            and out
            and _NAME.match(out[-1])
        ):
            out[-1] = out[-1] + tok
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


_FV_GROUP = re.compile(r"^((?:i|c|ibn|ib\d+)\.)\((.*)\)$", re.I | re.S)


def _split_fv_operators(tok: str) -> Tuple[List[str], List[str]]:
    """Split a factor-variable term on ``#`` / ``##`` outside parentheses."""
    parts: List[str] = []
    ops: List[str] = []
    depth, start, i = 0, 0, 0
    while i < len(tok):
        ch = tok[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "#" and depth == 0:
            op = "##" if tok[i : i + 2] == "##" else "#"
            parts.append(tok[start:i])
            ops.append(op)
            i += len(op)
            start = i
            continue
        i += 1
    parts.append(tok[start:])
    return parts, ops


def _expand_prefixed_list(
    text: str, columns: Optional[Sequence[str]]
) -> Tuple[Optional[str], List[str]]:
    """``c.x1-x9 i.g`` -> ``['c.x1', ..., 'c.x9', 'i.g']``: in a
    parenthesised list each term keeps its own operator, and a range takes
    the operator of its first variable."""
    out: List[str] = []
    for item in _join_spaced_ranges(text.split()):
        m = re.fullmatch(r"((?:i|c|ibn|ib\d+)\.)?(.+)", item, re.I)
        prefix, rest = (m.group(1) or "", m.group(2)) if m else ("", item)
        err, names = _expand_abbreviations([rest], columns)
        if err is not None:
            return err, []
        if not names or not all(_NAME.match(nm) for nm in names):
            return f"factor-variable term {item!r} is not translated", []
        out.extend(prefix + nm for nm in names)
    return None, out


def _expand_fv_groups(
    tok: str, columns: Optional[Sequence[str]]
) -> Tuple[Optional[str], List[str]]:
    """Distribute grouped factor notation over its varlist.

    ``c.(x x2)`` is ``c.x c.x2``; ``d##c.(x x2)`` is
    ``d c.x c.x2 d#c.x d#c.x2`` and ``c.(a b)#i.g`` is ``c.a#i.g c.b#i.g``,
    in Stata's order (main effects first). The varlist inside a group may
    use ranges and wildcards, expanded against ``columns``.
    """
    if "(" not in tok:
        return None, [tok]
    parts, ops = _split_fv_operators(tok)
    choices: List[List[str]] = []
    for part in parts:
        m = _FV_GROUP.match(part)
        if not m:
            bare = re.fullmatch(r"\((.*)\)", part, re.S)
            if bare:
                # a list of terms that carry their own prefix:
                # (c.x1-x9) is c.x1 c.x2 ... c.x9
                err, terms = _expand_prefixed_list(bare.group(1), columns)
                if err is not None:
                    return err, [tok]
                choices.append(terms)
                continue
            if "(" in part:
                return f"factor-variable term {tok!r} is not translated", [tok]
            choices.append([part])
            continue
        inner = _join_spaced_ranges(m.group(2).split())
        err, names = _expand_abbreviations(inner, columns)
        if err is not None:
            return err, [tok]
        if not names or not all(_NAME.match(nm) for nm in names):
            return f"factor-variable term {tok!r} is not translated", [tok]
        choices.append([m.group(1) + nm for nm in names])
    if len(parts) == 1:
        return None, choices[0]
    if len(parts) > 2 and "##" in ops:
        return f"factor-variable term {tok!r} is not translated", [tok]
    out: List[str] = []
    if ops == ["##"]:
        for side in choices:
            out.extend(side)
    import itertools

    for combo in itertools.product(*choices):
        out.append("#".join(combo))
    # a##b with one variable a side keeps its own compact translation
    if ops == ["##"] and all(len(c) == 1 for c in choices):
        return None, ["##".join(c[0] for c in choices)]
    seen: List[str] = []
    for term in out:
        if term not in seen:
            seen.append(term)
    return None, seen


def _expand_abbreviations(
    toks: List[str], columns: Optional[Sequence[str]]
) -> Tuple[Optional[str], List[str]]:
    """Expand Stata varlist wildcards (``x*``, ``x?``) and ranges (``a-b``).

    They name columns of the dataset, so they can only be expanded against
    it: ``*`` / ``~`` match any run of characters and ``?`` one character,
    in dataset order; ``a-b`` is every column from ``a`` to ``b``. Without
    ``columns`` they are refused -- left in place, ``t2pre*`` reads as an
    interaction in a formula, a confident but wrong translation.
    """
    import fnmatch

    out: List[str] = []
    for tok in toks:
        rng = _VARLIST_RANGE.match(tok)
        wild = any(ch in tok for ch in "*?~") and not tok.startswith("(")
        if not (rng or wild):
            out.append(tok)
            continue
        if columns is None:
            return (
                f"varlist abbreviation {tok!r} names dataset columns; pass "
                "columns=list(df.columns) to expand it (sp.stata does), or "
                "list the variables",
                toks,
            )
        cols = [str(c) for c in columns]
        if rng:
            a, b = rng.group(1), rng.group(2)
            if a not in cols or b not in cols or cols.index(a) > cols.index(b):
                return f"varlist range {tok!r} does not match the dataset order", toks
            out.extend(cols[cols.index(a) : cols.index(b) + 1])
            continue
        pattern = tok.replace("~", "*")
        hits = [c for c in cols if fnmatch.fnmatchcase(c, pattern)]
        if not hits:
            return f"varlist wildcard {tok!r} matches no column", toks
        out.extend(hits)
    return None, out


def _normalise_command(
    cmd: StataCommand, columns: Optional[Sequence[str]] = None
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Strip the weight clause and translate factor notation in place.

    Returns ``(error, info)``; ``info`` carries ``weight`` = (kind, var) and
    ``semantics`` notes.
    """
    info: Dict[str, Any] = {"weight": None, "semantics": []}
    joined = " ".join(cmd.varlist)
    m = _WEIGHT_RE.search(joined)
    if m:
        info["weight"] = (m.group(1).lower()[:2], m.group(2))
        joined = (joined[: m.start()] + " " + joined[m.end() :]).strip()
    elif "[" in joined:
        return (
            "the weight clause in "
            + repr(joined)
            + " is an expression or an unknown weight type; sp.stata evaluates "
            "a weight expression, a single translated line cannot -- create "
            "the weight as a column first",
            info,
        )
    toks = _tokenise_head(joined)
    err, toks = _expand_abbreviations(toks, columns)
    if err is not None:
        return err, info
    grouped: List[str] = []
    for tok in toks:
        err, pieces = (
            _expand_fv_groups(tok, columns) if _FV_RE.search(tok) else (None, [tok])
        )
        if err is not None:
            return err, info
        grouped.extend(pieces)
    toks = grouped
    out: List[str] = []
    factor_used = False
    for tok in toks:
        if _TS_OP_RE.match(tok):
            return (
                f"time-series operator {tok!r} is not resolved here. "
                "sp.stata resolves L. / F. / D. on existing variables once "
                "`tsset time` or `xtset id time` has been given; otherwise "
                "create the lag / difference as a column first",
                info,
            )
        if not _FV_RE.search(tok):
            out.append(tok)
            continue
        new = _fv_token(tok)
        if new is None:
            return f"factor-variable term {tok!r} is not translated", info
        factor_used = factor_used or "C(" in new
        out.append(new)
    cmd.varlist = out
    if factor_used:
        info["semantics"].append(
            "i.var -> C(var): the base (omitted) level is the lowest one, as "
            "Stata's default; ib#. is mapped to Treatment(#)."
        )
    return None, info


#: Tools that take ``weights=`` through ``**kwargs``, so the signature check
#: below cannot see it. Each entry is pinned by a test that the weights
#: change the estimate (a ``**kwargs`` that swallowed them would not).
_WEIGHTS_VIA_KWARGS = frozenset({"ivreg", "iv"})


def _apply_weight(payload: Dict[str, Any], weight: Tuple[str, str]) -> Dict[str, Any]:
    """Attach a Stata weight to a translated call, or refuse it."""
    import inspect

    import statspai as sp

    kind, var = weight
    fn = getattr(sp, str(payload.get("tool") or ""), None)
    try:
        params = inspect.signature(inspect.unwrap(fn)).parameters if fn else {}
    except (TypeError, ValueError):
        params = {}
    if "weights" not in params and payload.get("tool") not in _WEIGHTS_VIA_KWARGS:
        return _emit_error(
            f"[{kind}={var}] cannot be carried over: sp.{payload.get('tool')} "
            "takes no weights= argument.",
            command=payload.get("tool"),
            suggestions=[],
        )
    code = payload["python_code"]
    sem = payload.setdefault("semantics", [])
    if kind == "iw":
        return _emit_error(
            f"[iw={var}] (importance weights) has no StatsPAI equivalent; "
            "their scaling is estimator-specific in Stata.",
            command=payload.get("tool"),
            suggestions=[],
        )
    if kind == "fw":
        # Frequency weights: each row stands for w identical rows. weights=
        # would treat them as analytic weights (other SEs / df), and a tool
        # call's arguments cannot carry a row expansion, so refuse with the
        # exact rewrite.
        expanded = code.replace(
            "data=df", f"data=df.loc[df.index.repeat(df[{var!r}])]", 1
        )
        return _emit_error(
            f"[fw={var}] (frequency weights) is exact only by expanding rows, "
            f"which a tool call cannot express; run: {expanded}",
            command=payload.get("tool"),
            suggestions=[expanded],
        )
    else:
        payload["arguments"]["weights"] = var
        code = code.replace("data=df", f"data=df, weights={var!r}", 1)
        if kind == "aw":
            sem.append(f"[aw={var}] -> weights={var!r} (analytic weights).")
        else:
            sem.append(
                f"[pw={var}] -> weights={var!r}; Stata's pweights imply "
                "robust standard errors."
            )
            args = payload["arguments"]
            if (
                "robust" in params
                and not args.get("robust")
                and not args.get("cluster")
            ):
                args["robust"] = "hc1"
                code = code.replace(
                    f"weights={var!r}", f"weights={var!r}, robust='hc1'", 1
                )
    payload["python_code"] = code
    return payload


# Handlers kept in their own modules register here. They import the helpers
# above, so the import has to come after them.
from . import _stata_design as _design  # noqa: E402
from . import _stata_panel as _panel  # noqa: E402
from . import _stata_postest as _postest  # noqa: E402
from . import _stata_ts as _ts  # noqa: E402

STATA_COMMAND_MAP.update(_postest.HANDLERS)
STATA_COMMAND_MAP.update(_ts.HANDLERS)
STATA_COMMAND_MAP.update(_panel.HANDLERS)
STATA_COMMAND_MAP.update(_design.HANDLERS)

_POSTEST_HANDLERS = frozenset(
    {
        _h_margins,
        _h_marginsplot,
        _h_contrast,
        _h_test,
        _h_lincom,
        _h_xtset,
        _h_boottest,
        _h_ttest,
    }
    | _postest.POSTEST
    | _ts.POSTEST
)


def from_stata(line: str, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Translate a Stata command line to a StatsPAI tool-call payload.

    Parameters
    ----------
    line : str
        One Stata command. Multi-line ``do`` files must be split by
        the caller. Stata's option abbreviations (``r``, ``cl()``,
        ``a()``, ``vce(cl id)``) are expanded and output-only prefixes
        (``quietly``, ``capture``, ``eststo:``, ``xi:``) are peeled;
        macros and prefixes that change the estimate (``by``,
        ``bootstrap``, ``svy`` ...) are refused.
    columns : sequence of str, optional
        The dataset's columns, in order. Varlist wildcards (``t2pre*``,
        ``x?``) and ranges (``a-b``) name columns and are expanded against
        them; without ``columns`` such a command is refused rather than
        translated into a formula where ``*`` would mean an interaction.

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
                "semantics": [<convention relied on>, ...],
                "untranslated_options": [<option name>, ...],
                "ignored_display_options": [<option name>, ...],
                "unapplied_sample": "<if / in qualifier>" or None,
            }

        ``untranslated_options`` lists the options the call does not carry
        over and ``unapplied_sample`` the ``if`` / ``in`` qualifier: when
        both are empty the call fits the same model. Options that only
        change what Stata prints go to ``ignored_display_options``.

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
    >>> out["python_code"]
    "sp.hdfe_ols('y ~ x | id + year', data=df, cluster='id')"
    >>> sp.from_stata("qui reghdfe y x, a(id year) cl(id)")["python_code"]
    "sp.hdfe_ols('y ~ x | id + year', data=df, cluster='id')"
    >>> sp.from_stata("reg y x, nocons")["python_code"]
    "sp.regress('y ~ x - 1', data=df)"
    >>> sp.from_stata("reg y x, hascons")["untranslated_options"]
    ['hascons']
    >>> sp.from_stata("notacommand y x")["ok"]
    False
    """
    prefixes, refused_prefix, core = _opts.peel_prefixes(line or "")
    if refused_prefix is not None:
        return _emit_error(
            _opts.semantic_prefix_error(refused_prefix),
            command=refused_prefix,
            suggestions=[],
        )
    try:
        parsed = _parse_stata(core)
    except StataParseError as e:
        return _emit_error(f"parse_error: {e}", command=None, suggestions=[])

    parsed.command = _resolve_command(parsed.command)
    handler = STATA_COMMAND_MAP.get(parsed.command)
    if handler is None and parsed.command in _UNTRANSLATED_GUIDANCE:
        msg, funcs = _UNTRANSLATED_GUIDANCE[parsed.command]
        return _emit_error(
            f"{parsed.command!r} is not translated line by line: {msg}",
            command=parsed.command,
            suggestions=[],
            statspai_functions=funcs,
        )
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

    canonical, expanded = _opts.canonicalise_options(parsed.command, parsed.options)
    macro = _opts.find_macro_in_command(
        StataCommand(
            command=parsed.command,
            varlist=parsed.varlist,
            if_cond=parsed.if_cond,
            in_range=parsed.in_range,
            options=canonical,
        )
    )
    if macro is not None:
        return _emit_error(
            _opts.macro_error(macro), command=parsed.command, suggestions=[]
        )
    tracked = _opts.TrackedOptions(canonical)
    parsed.options = tracked

    info: Dict[str, Any]
    if handler in _POSTEST_HANDLERS:
        # postestimation commands take variable names / expressions, which
        # their handlers interpret themselves (``contrast i.g`` -> ``g``)
        err, info = None, {"weight": None, "semantics": []}
    else:
        err, info = _normalise_command(parsed, columns)
    if err is not None:
        return _emit_error(err, command=parsed.command, suggestions=[])
    payload = handler(parsed)
    if not payload.get("ok"):
        return payload
    payload.setdefault("semantics", [])
    payload["semantics"] = list(info["semantics"]) + payload["semantics"]
    if info["weight"] is not None:
        payload = _apply_weight(payload, info["weight"])
    if payload.get("ok") and parsed.if_cond:
        payload["semantics"].append(
            "The `if` sample is not applied by this call. sp.stata(line, "
            "data=df) applies it with Stata's rules for missing values (a "
            "missing value is larger than any number, so `if x > 0` keeps "
            "the rows where x is missing); a pandas filter does not."
        )
    if payload.get("ok"):
        payload = _report_options(
            payload, tracked, dict(canonical), expanded, prefixes, parsed
        )
    return payload


def _carry_level(payload: Dict[str, Any], level: Optional[str]) -> str:
    """``level(#)`` -> ``alpha=`` when the sp function takes it.

    The confidence level does not change the fit; where the function has no
    ``alpha`` it is chosen when the intervals are reported.
    """
    import inspect

    import statspai as sp

    try:
        alpha = round(1 - float(level or "") / 100, 10)
    except ValueError:
        return f"level({level}) is not a number; ignored."
    fn = getattr(sp, str(payload.get("tool") or ""), None)
    try:
        params = inspect.signature(inspect.unwrap(fn)).parameters if fn else {}
    except (TypeError, ValueError):
        params = {}
    code = str(payload.get("python_code") or "")
    if "alpha" in params and code.endswith(")"):
        payload["arguments"]["alpha"] = alpha
        payload["python_code"] = f"{code[:-1]}, alpha={alpha})"
        return f"level({level}) -> alpha={alpha}."
    return (
        f"level({level}) only sets the confidence level of the printed "
        f"intervals; request alpha={alpha} when reporting them."
    )


def _report_options(
    payload: Dict[str, Any],
    tracked: "_opts.TrackedOptions",
    options: Dict[str, Optional[str]],
    expanded: List[str],
    prefixes: List[str],
    parsed: StataCommand,
) -> Dict[str, Any]:
    """Surface what the translation did with options, prefixes and sample.

    ``untranslated_options`` lists every option the call does not carry
    over and ``unapplied_sample`` the ``if`` / ``in`` qualifier, so a caller
    can tell a faithful translation from a partial one without reading the
    notes.
    """
    reported = list(payload.get("untranslated_options") or [])
    unread = [n for n in tracked.unread() if n not in reported]
    sem = payload.setdefault("semantics", [])
    if "level" in unread:
        unread.remove("level")
        sem.append(_carry_level(payload, options["level"]))
    display = [n for n in unread if _opts.is_display_option(parsed.command, n)]
    lossy = [n for n in unread if n not in display]
    notes = list(payload.get("notes") or [])
    notes += _opts.untranslated_notes(lossy, display, options)
    se, se_names = _opts.se_note(
        options, payload.get("arguments") or {}, reported + unread
    )
    if payload.pop("_vce_is_sp_default", False):
        # The handler checked that the vce() Stata was given is what the sp
        # function computes with no argument.
        se, se_names = None, []
    if se:
        notes.append(se)
    payload["notes"] = notes
    payload["untranslated_options"] = reported + lossy + se_names
    payload["ignored_display_options"] = display
    sample = " ".join(
        part
        for part in (
            f"if {parsed.if_cond}" if parsed.if_cond else "",
            f"in {parsed.in_range}" if parsed.in_range else "",
        )
        if part
    )
    payload["unapplied_sample"] = sample or None
    if parsed.in_range:
        sem.append("The `in` range is not applied by the call; slice df first.")
    if expanded:
        sem.append("Stata abbreviations expanded: " + ", ".join(expanded) + ".")
    if prefixes:
        sem.append(
            "Prefix " + ", ".join(f"`{p}`" for p in prefixes) + " only affects "
            "what Stata prints or stores; dropped."
        )
    return payload


__all__ = ["from_stata", "STATA_COMMAND_MAP", "StataCommand", "StataParseError"]
