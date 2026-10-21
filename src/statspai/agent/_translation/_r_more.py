"""More R functions for ``sp.from_r`` (2026-10-19).

Each mapping was run against the R package on the same data
(``tests/test_translation_from_r_more.py``): ``AER::ivreg`` / ``ivreg``,
``survival::coxph``, ``pwr::pwr.t.test``, ``survey::svydesign`` +
``svyglm``, ``lmtest::coeftest`` with ``sandwich`` covariances,
``etwfe::etwfe`` + ``emfx``, ``WeightIt::weightit`` (against the
WeightIt outputs of the ``sp.weightit`` parity fixture). ``did::aggte``,
``DIDmultiplegtDYN::did_multiplegt_dyn`` and ``modelsummary`` map to the
StatsPAI functions written to mirror them (their own parity tests pin the
numbers). Functions whose R inputs are objects built elsewhere (a weights
matrix, a design matrix of summary statistics, a fitted mediator model)
are refused with the sp call to make instead (:data:`R_GUIDANCE`).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ._r import (
    _column,
    _emit,
    _emit_error,
    _parse_c_vector,
    _split_top_level,
    _strip_quotes,
    _unread,
)

__all__ = ["HANDLERS", "R_GUIDANCE"]

_NAME = r"[A-Za-z.][A-Za-z0-9._]*"


def _terms(rhs: str) -> Optional[List[str]]:
    out = [t.strip() for t in rhs.split("+") if t.strip()]
    return out if all(re.fullmatch(_NAME, t) for t in out) else None


def _formula(pos: List[str], kw: Dict[str, str]) -> Optional[str]:
    raw = pos[0] if pos else kw.get("formula")
    return _strip_quotes(raw).strip() if raw else None


def _true(value: Optional[str]) -> bool:
    return _strip_quotes(value or "").upper() in ("TRUE", "T")


def _number(value: Optional[str]) -> Optional[float]:
    try:
        return float(_strip_quotes(value or ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# AER::ivreg / ivreg::ivreg
# ---------------------------------------------------------------------------


def _h_ivreg(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``ivreg(y ~ x1 + d | x1 + z)`` (or ``y ~ x1 | d | z``) -> ``sp.iv``.

    The two-part formula lists every regressor, then every instrument:
    regressors not among the instruments are endogenous, instruments not
    among the regressors are excluded. ``sp.iv``'s default variance is
    ``ivreg``'s (homoskedastic, N - K).
    """
    fml = _formula(pos, kw)
    if not fml or "~" not in fml:
        return _emit_error("ivreg requires a formula", command="ivreg")
    lhs, _, rhs = fml.partition("~")
    parts = [p.strip() for p in rhs.split("|")]
    y = lhs.strip()
    if len(parts) == 2:
        regs, insts = _terms(parts[0]), _terms(parts[1])
        if regs is not None and insts is not None:
            exog = [t for t in regs if t in insts]
            endog = [t for t in regs if t not in insts]
            excl = [t for t in insts if t not in regs]
        else:
            exog = endog = excl = None  # type: ignore[assignment]
    elif len(parts) == 3:
        exog, endog, excl = (_terms(p) for p in parts)  # type: ignore[assignment]
    else:
        return _emit_error(
            "ivreg needs `y ~ regressors | instruments` (or the three-part "
            "`y ~ exog | endog | instruments`).",
            command="ivreg",
        )
    if exog is None or endog is None or excl is None or not re.fullmatch(_NAME, y):
        return _emit_error(
            "ivreg: only formulas of column names are translated (build "
            "transformed terms as columns first).",
            command="ivreg",
            suggestions=["iv"],
        )
    if not endog or len(excl) < len(endog):
        return _emit_error(
            f"ivreg: {len(endog)} endogenous regressor(s) and {len(excl)} "
            "excluded instrument(s); the model is not identified as written.",
            command="ivreg",
        )
    formula = f"{y} ~ " + " + ".join(
        exog + [f"({' + '.join(endog)} ~ {' + '.join(excl)})"]
    )
    args: Dict[str, Any] = {"formula": formula, "method": "2sls"}
    read = ["formula"]
    code = f"sp.iv({formula!r}, data=df, method='2sls'"
    w = _column(kw.get("weights"))
    if w:
        read.append("weights")
        args["weights"] = w
        code += f", weights={w!r}"
    out = _emit("iv", args, code + ")")
    return _unread(out, kw, tuple(read))


# ---------------------------------------------------------------------------
# survival::coxph
# ---------------------------------------------------------------------------


def _h_coxph(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``coxph(Surv([start,] time, event) ~ x1 + strata(s), data, ties,
    weights, robust, cluster, id)`` -> ``sp.cox``.

    Efron ties (both defaults; written out). ``robust = TRUE`` is
    ``robust='hc0'`` (no small-sample factor in either). ``cluster = c`` /
    a ``cluster(c)`` term / ``id = c`` with ``robust = TRUE`` is
    ``cluster=c, ssc='r'``: coxph's clustered variance has no G/(G-1)
    factor. ``Surv(start, stop, event)`` is ``start=``. ``weights = w`` is
    ``weights=w``; coxph turns the robust variance on by itself when the
    weights are not all integers, which the translation cannot see, so
    that case is noted (``sp.cox`` warns and takes
    ``weights_as='sampling'``).
    """
    fml = _formula(pos, kw)
    m = re.fullmatch(
        rf"(?:survival::)?Surv\(\s*({_NAME})\s*,\s*({_NAME})\s*"
        rf"(?:,\s*({_NAME})\s*)?\)\s*~(.*)",
        fml or "",
        re.S,
    )
    if m is None:
        return _emit_error(
            "coxph: only `Surv(time, event) ~ x1 + ...` or `Surv(start, stop, "
            "event) ~ ...` with column names is translated.",
            command="coxph",
            suggestions=["cox"],
        )
    start: Optional[str] = None
    if m.group(3) is not None:
        start, duration, event = m.group(1), m.group(2), m.group(3)
    else:
        duration, event = m.group(1), m.group(2)
    rhs = m.group(4)
    xs: List[str] = []
    strata: List[str] = []
    cluster_terms: List[str] = []
    for term in (t.strip() for t in rhs.split("+")):
        s = re.fullmatch(rf"strata\(\s*({_NAME})\s*\)", term)
        c = re.fullmatch(rf"cluster\(\s*({_NAME})\s*\)", term)
        if s:
            strata.append(s.group(1))
        elif c:
            cluster_terms.append(c.group(1))
        elif re.fullmatch(_NAME, term):
            xs.append(term)
        else:
            return _emit_error(
                f"coxph: the term {term!r} is not translated (only column "
                "names and one strata(var)).",
                command="coxph",
                suggestions=["cox"],
            )
    if len(strata) > 1 or not xs:
        return _emit_error(
            "coxph: sp.cox takes covariates and at most one strata column.",
            command="coxph",
            suggestions=["cox"],
        )
    ties = _strip_quotes(kw.get("ties", "efron")).lower()
    if ties not in ("efron", "breslow"):
        return _emit_error(
            f"coxph ties = {ties!r} has no sp.cox counterpart (efron, breslow).",
            command="coxph",
        )
    args: Dict[str, Any] = {"duration": duration, "event": event, "x": xs, "ties": ties}
    if start is not None:
        args["start"] = start
    if strata:
        args["strata"] = strata[0]
    read = ["formula", "ties"]
    notes: List[str] = []
    robust = None
    if "robust" in kw:
        read.append("robust")
        robust = _true(kw["robust"])
    cluster = _column(kw.get("cluster")) if "cluster" in kw else None
    if "cluster" in kw:
        read.append("cluster")
    if cluster is None and cluster_terms:
        cluster = cluster_terms[0]
    if "id" in kw:
        read.append("id")
        if cluster is None and robust:
            # coxph clusters the robust variance on id
            cluster = _column(kw.get("id"))
    w = _column(kw.get("weights")) if "weights" in kw else None
    if w:
        read.append("weights")
        args["weights"] = w
    if cluster:
        args["cluster"] = cluster
        args["ssc"] = "r"
    elif robust:
        args["robust"] = "hc0"
    elif w and robust is False:
        args["weights_as"] = "frequency"
    elif w:
        notes.append(
            "coxph reports the robust variance by default when the weights "
            "are not all integers (IPT / survey weights); sp.cox does so with "
            "weights_as='sampling' and warns when it is missing."
        )
    code = ", ".join(["data=df"] + [f"{k}={v!r}" for k, v in args.items()])
    out = _emit("cox", args, f"sp.cox({code})", notes)
    return _unread(out, kw, tuple(read))


# ---------------------------------------------------------------------------
# pwr::pwr.t.test
# ---------------------------------------------------------------------------

_PWR_TYPES = {
    "two.sample": "two-sample",
    "one.sample": "one-sample",
    "paired": "paired",
}
_PWR_ALT = {"two.sided": "two-sided", "greater": "greater", "less": "less"}


def _h_pwr_t_test(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``pwr.t.test(n =, d =, sig.level =, power =, type =)`` -> ``sp.power_ttest``.

    ``d`` is the standardised effect, so ``sd=1``. pwr's ``n`` is per group
    for a two-sample test; ``sp.power_ttest``'s is the total.
    """
    vals = dict(zip(("n", "d", "sig.level", "power"), pos))
    for k in ("n", "d", "sig.level", "power", "type", "alternative"):
        if k in kw:
            vals[k] = kw[k]
    kind = _PWR_TYPES.get(_strip_quotes(vals.get("type", "two.sample")))
    alt = _PWR_ALT.get(_strip_quotes(vals.get("alternative", "two.sided")))
    d = _number(vals.get("d"))
    n = _number(vals.get("n")) if vals.get("n") not in (None, "NULL") else None
    power = (
        _number(vals.get("power")) if vals.get("power") not in (None, "NULL") else None
    )
    alpha = _number(vals.get("sig.level", "0.05"))
    if kind is None or alt is None or d is None or alpha is None:
        return _emit_error(
            "pwr.t.test: d, type and alternative must be given as literals "
            "(solving for d or sig.level is not translated).",
            command="pwr.t.test",
            suggestions=["power_ttest"],
        )
    if (n is None) == (power is None):
        return _emit_error(
            "pwr.t.test: give exactly one of n and power.",
            command="pwr.t.test",
        )
    args: Dict[str, Any] = {"delta": d, "sd": 1.0, "alpha": alpha, "type": kind}
    notes: List[str] = []
    if n is not None:
        args["n"] = 2 * n if kind == "two-sample" else n
        if kind == "two-sample":
            notes.append(f"pwr's n = {n:g} per group is n = {2 * n:g} in total here.")
    else:
        args["power"] = power
        notes.append(
            "pwr.t.test reports the n that solves the equation (fractional, "
            "per group for two samples); sp.power_ttest reports the total, "
            "rounded up to whole observations per group."
        )
    if alt != "two-sided":
        args["alternative"] = alt
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("power_ttest", args, f"sp.power_ttest({pairs})", notes)
    return _unread(out, kw, ("n", "d", "sig.level", "power", "type", "alternative"))


# ---------------------------------------------------------------------------
# survey::svydesign / svyglm
# ---------------------------------------------------------------------------


def _one_sided(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    text = _strip_quotes(value).strip().lstrip("~").strip()
    return [t.strip() for t in text.split("+") if t.strip()]


def _h_svydesign(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``svydesign(ids = ~psu, strata = ~st, weights = ~w, data, nest)`` ->
    ``sp.svydesign``. One sampling stage; ``probs`` is not translated."""
    ids = _one_sided(kw.get("ids", kw.get("id")))
    if ids is None:
        return _emit_error("svydesign needs ids = ~psu (or ~1)", command="svydesign")
    if len(ids) != 1 or not (ids[0] in ("0", "1") or re.fullmatch(_NAME, ids[0])):
        return _emit_error(
            f"svydesign(ids = ~{' + '.join(ids)}): multistage designs are not "
            "translated; sp.svydesign takes one PSU column.",
            command="svydesign",
            suggestions=["svydesign"],
        )
    weights = _one_sided(kw.get("weights"))
    if not weights or len(weights) != 1:
        return _emit_error(
            "svydesign: weights = ~w (one column) is required by sp.svydesign "
            "(probs = is not translated).",
            command="svydesign",
        )
    args: Dict[str, Any] = {"weights": weights[0]}
    strata = _one_sided(kw.get("strata"))
    if strata:
        if len(strata) != 1:
            return _emit_error("svydesign: one strata column only", command="svydesign")
        args["strata"] = strata[0]
    if ids[0] not in ("0", "1"):
        args["cluster"] = ids[0]
    fpc = _one_sided(kw.get("fpc"))
    if fpc:
        args["fpc"] = fpc[0]
    if "nest" in kw and _true(kw["nest"]):
        args["nest"] = True
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("svydesign", args, f"sp.svydesign(df, {pairs})")
    return _unread(out, kw, ("ids", "id", "weights", "strata", "fpc", "nest"))


_SVY_FAMILY = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "quasibinomial": "binomial",
    "poisson": "poisson",
    "quasipoisson": "poisson",
}


def _h_svyglm(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``svyglm(y ~ x, design = des, family = quasibinomial())`` ->
    ``sp.svyglm(..., dof='residual')``: R's summary uses the residual
    degrees of freedom; the design-based standard errors are the same
    for a family and its quasi- version."""
    fml = _formula(pos, kw)
    design = kw.get("design") or (pos[1] if len(pos) > 1 else None)
    if not fml or not design or not re.fullmatch(_NAME, design.strip()):
        return _emit_error(
            "svyglm needs a formula and design = <svydesign object>",
            command="svyglm",
        )
    fam_raw = _strip_quotes(kw.get("family", "gaussian()")).strip()
    fam = re.sub(r"\(\s*\)$", "", fam_raw).replace("stats::", "")
    family = _SVY_FAMILY.get(fam)
    if family is None:
        return _emit_error(
            f"svyglm family {fam_raw!r} has no sp.svyglm counterpart "
            "(gaussian, binomial, poisson and their quasi- forms).",
            command="svyglm",
        )
    args = {"formula": fml, "family": family, "dof": "residual"}
    code = (
        f"sp.svyglm({fml!r}, design={design.strip()}, family={family!r}, "
        "dof='residual')"
    )
    out = _emit(
        "svyglm",
        args,
        code,
        [
            f"`{design.strip()}` is the sp.svydesign(...) object "
            "(sp.from_r translates svydesign too)."
        ],
    )
    return _unread(out, kw, ("formula", "design", "family"))


# ---------------------------------------------------------------------------
# lmtest::coeftest(m, vcov = sandwich::vcovHC / vcovCL)
# ---------------------------------------------------------------------------


def _h_coeftest(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``coeftest(m, vcov = vcovHC(m, type = "HC1"))`` -> ``sp.vcov(m, 'hc1')``.

    ``vcovHC``'s default type is HC3; ``vcovCL(m, cluster = ~g)`` (HC1 with
    G/(G-1)) is the ``'cluster g'`` of ``sp.vcov``, both checked against
    sandwich 3.x on an ``lm`` fit.
    """
    if not pos:
        return _emit_error("coeftest needs the fitted model", command="coeftest")
    model = pos[0].strip()
    raw = kw.get("vcov.") or kw.get("vcov") or (pos[1] if len(pos) > 1 else None)
    if raw is None:
        return _emit_error(
            "coeftest(m) without vcov is the model's own coefficient table: "
            "use result.summary().",
            command="coeftest",
        )
    m = re.fullmatch(r"(?:sandwich::)?(vcovHC|vcovCL)\s*\((.*)\)\s*", raw.strip(), re.S)
    if m is None:
        return _emit_error(
            f"coeftest(vcov = {raw}) is not translated: vcovHC(m, type=) and "
            "vcovCL(m, cluster = ~g) are.",
            command="coeftest",
            suggestions=["vcov"],
        )
    fn, inner = m.group(1), m.group(2)
    sub: Dict[str, str] = {}
    for part in _split_top_level(inner, ",")[1:]:
        k, eq, v = part.partition("=")
        if eq:
            sub[k.strip()] = v.strip()
    if fn == "vcovHC":
        kind = _strip_quotes(sub.pop("type", '"HC3"')).lower()
        if kind not in ("hc0", "hc1", "hc2", "hc3"):
            return _emit_error(
                f"vcovHC(type = {kind!r}) is not translated (HC0 - HC3 are).",
                command="coeftest",
            )
        vce = kind
    else:
        cl = _one_sided(sub.pop("cluster", None))
        kind = _strip_quotes(sub.pop("type", '"HC1"')).lower()
        if not cl or kind != "hc1" or not all(re.fullmatch(_NAME, c) for c in cl):
            return _emit_error(
                "vcovCL is translated with cluster = ~g [+ h] and the default "
                "type HC1.",
                command="coeftest",
            )
        vce = "cluster " + " ".join(cl)
    if sub:
        return _emit_error(
            f"{fn}(): arguments {sorted(sub)} are not translated.",
            command="coeftest",
        )
    out = _emit(
        "vcov",
        {"vce": vce},
        f"sp.vcov({model}, {vce!r})",
        [
            f"`{model}` is the fitted StatsPAI result; sp.vcov refits it "
            f"with vce={vce!r}."
        ],
    )
    return _unread(out, kw, ("vcov", "vcov."))


# ---------------------------------------------------------------------------
# etwfe::etwfe / emfx
# ---------------------------------------------------------------------------


def _h_etwfe(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``etwfe(y ~ 0, tvar = t, gvar = g, data, [ivar = id], vcov = ~id)`` ->
    ``sp.etwfe`` (fe='cohort', or 'unit' with ``ivar``). ``emfx`` of the
    result is ``sp.etwfe_emfx``. Controls on the right-hand side, ``xvar``
    and nonlinear families are not translated."""
    fml = _formula(pos, kw) or _strip_quotes(kw.get("fml", ""))
    lhs, tilde, rhs = (fml or "").partition("~")
    if (
        not tilde
        or not re.fullmatch(_NAME, lhs.strip())
        or rhs.strip() not in ("0", "1")
    ):
        return _emit_error(
            "etwfe is translated for `y ~ 0` (no controls): controls in etwfe "
            "are demeaned and interacted with the cohorts, which sp.etwfe's "
            "controls= does not do the same way.",
            command="etwfe",
            suggestions=["etwfe"],
        )
    tvar = _column(kw.get("tvar"))
    gvar = _column(kw.get("gvar"))
    ivar = _column(kw.get("ivar")) if kw.get("ivar") not in (None, "NULL") else None
    if not tvar or not gvar:
        return _emit_error("etwfe needs tvar = and gvar = columns", command="etwfe")
    for unsupported in ("xvar", "family", "gref", "tref", "fe"):
        if unsupported in kw:
            return _emit_error(
                f"etwfe({unsupported} = ) is not translated.",
                command="etwfe",
                suggestions=["etwfe"],
            )
    cgroup = _strip_quotes(kw.get("cgroup", "notyet"))
    if cgroup not in ("notyet", "never"):
        return _emit_error(f"etwfe cgroup = {cgroup!r}", command="etwfe")
    vcov = kw.get("vcov")
    cluster = _one_sided(vcov) if vcov and vcov.strip().startswith("~") else None
    unit = ivar or (cluster[0] if cluster and len(cluster) == 1 else None)
    if unit is None:
        return _emit_error(
            "etwfe: sp.etwfe needs the unit column; give ivar = or " "vcov = ~unit.",
            command="etwfe",
            suggestions=["etwfe"],
        )
    args: Dict[str, Any] = {
        "y": lhs.strip(),
        "group": unit,
        "time": tvar,
        "first_treat": gvar,
        "fe": "unit" if ivar else "cohort",
        "cgroup": "nevertreated" if cgroup == "never" else "notyet",
    }
    notes = [
        "result.estimate is emfx(m)'s simple ATT; sp.etwfe_emfx(result, "
        "type=...) gives the other aggregations.",
    ]
    untranslated: List[str] = []
    if cluster is not None:
        if len(cluster) != 1:
            return _emit_error("etwfe: one cluster variable only", command="etwfe")
        args["cluster"] = cluster[0]
    else:
        untranslated.append("vcov")
        notes.append(
            "No vcov = ~cluster: etwfe uses fixest's default variance (IID in "
            "fixest 0.14), which sp.etwfe does not offer; the call clusters on "
            f"{unit!r}."
        )
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("etwfe", args, f"sp.etwfe(data=df, {pairs})", notes)
    if untranslated:
        out["untranslated_arguments"] = untranslated
    return _unread(
        out, kw, ("fml", "formula", "tvar", "gvar", "ivar", "cgroup", "vcov")
    )


_EMFX_TYPES = ("simple", "group", "calendar", "event")


def _h_emfx(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    if not pos:
        return _emit_error("emfx needs the etwfe model", command="emfx")
    kind = _strip_quotes(kw.get("type", pos[1] if len(pos) > 1 else "simple"))
    if kind not in _EMFX_TYPES:
        return _emit_error(f"emfx type = {kind!r}", command="emfx")
    args: Dict[str, Any] = {"type": kind}
    code = f"sp.etwfe_emfx({pos[0].strip()}, type={kind!r}"
    if "post_only" in kw and not _true(kw["post_only"]):
        args["include_leads"] = True
        code += ", include_leads=True"
    out = _emit(
        "etwfe_emfx", args, code + ")", ["The first argument is the sp.etwfe result."]
    )
    return _unread(out, kw, ("type", "post_only"))


# ---------------------------------------------------------------------------
# did::aggte, DIDmultiplegtDYN::did_multiplegt_dyn, modelsummary
# ---------------------------------------------------------------------------


def _h_aggte(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``aggte(out, type = "dynamic", ...)`` -> ``sp.aggte(result, ...)``."""
    if not pos:
        return _emit_error("aggte needs the att_gt result", command="aggte")
    kind = _strip_quotes(kw.get("type", "group"))
    if kind not in ("simple", "dynamic", "group", "calendar"):
        return _emit_error(f"aggte type = {kind!r}", command="aggte")
    args: Dict[str, Any] = {"type": kind}
    notes = []
    if "type" not in dict.keys(kw):
        notes.append("did::aggte defaults to type = 'group'; written out.")
    for r_name, sp_name in (
        ("balance_e", "balance_e"),
        ("min_e", "min_e"),
        ("max_e", "max_e"),
    ):
        if r_name in kw:
            v = _number(kw[r_name])
            if v is None:
                return _emit_error(f"aggte {r_name} must be a number", command="aggte")
            args[sp_name] = int(v) if r_name == "balance_e" else v
    for r_name, sp_name in (
        ("na.rm", "na_rm"),
        ("bstrap", "bstrap"),
        ("cband", "cband"),
    ):
        if r_name in kw and _strip_quotes(kw[r_name]).upper() != "NULL":
            args[sp_name] = _true(kw[r_name])
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit(
        "aggte",
        args,
        f"sp.aggte({pos[0].strip()}, {pairs})",
        notes + ["The first argument is the sp.callaway_santanna result."],
    )
    return _unread(
        out, kw, ("type", "balance_e", "min_e", "max_e", "na.rm", "bstrap", "cband")
    )


def _h_did_multiplegt_dyn(
    pos: List[str], kw: Dict[str, str], _: List[str]
) -> Dict[str, Any]:
    """``did_multiplegt_dyn(df, "y", "g", "t", "d", effects = 3, placebo = 2)``
    -> ``sp.did_multiplegt_dyn(..., dynamic=2, aggregation='switchers',
    se_method='analytic')``: R's ``effects = k`` estimates horizons
    ``0..k-1``, which is ``dynamic=k-1``; the headline is the package's
    ``Av_tot_eff`` and its analytic variance, pinned to 1e-6. ``effects``
    defaults to 1 in R and sp's ``dynamic`` to 3, so it is written out."""
    names = ("df", "outcome", "group", "time", "treatment")
    vals = dict(zip(names, pos))
    for k in names[1:]:
        if k in kw:
            vals[k] = kw[k]
    cols = {k: _column(vals.get(k)) for k in names[1:]}
    if not all(cols.values()):
        return _emit_error(
            "did_multiplegt_dyn needs outcome, group, time and treatment as "
            "column names",
            command="did_multiplegt_dyn",
        )
    effects = _number(kw.get("effects", "1"))
    placebo = _number(kw.get("placebo", "0"))
    if effects is None or placebo is None:
        return _emit_error(
            "did_multiplegt_dyn: effects / placebo must be numbers",
            command="did_multiplegt_dyn",
        )
    if int(effects) < 1:
        return _emit_error(
            "did_multiplegt_dyn: effects must be at least 1",
            command="did_multiplegt_dyn",
        )
    args: Dict[str, Any] = {
        "y": cols["outcome"],
        "group": cols["group"],
        "time": cols["time"],
        "treatment": cols["treatment"],
        "dynamic": int(effects) - 1,
        "placebo": int(placebo),
        "aggregation": "switchers",
        "se_method": "analytic",
    }
    read = ["outcome", "group", "time", "treatment", "effects", "placebo"]
    for r_name, sp_name in (("cluster", "cluster"), ("weight", "weights")):
        if kw.get(r_name) not in (None, "NULL"):
            c = _column(kw[r_name])
            if c:
                args[sp_name] = c
                read.append(r_name)
    if kw.get("controls") not in (None, "NULL"):
        ctrl = [c for c in _parse_c_vector(kw["controls"]) if _column(c)]
        if ctrl:
            args["controls"] = ctrl
            read.append("controls")
    if "normalized" in kw:
        args["normalized"] = _true(kw["normalized"])
        read.append("normalized")
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("did_multiplegt_dyn", args, f"sp.did_multiplegt_dyn(data=df, {pairs})")
    return _unread(out, kw, tuple(read))


def _h_modelsummary(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``modelsummary(list(m1, m2))`` -> ``sp.modelsummary(m1, m2)``."""
    if not pos:
        return _emit_error("modelsummary needs the models", command="modelsummary")
    first = pos[0].strip()
    m = re.fullmatch(r"list\((.*)\)", first, re.S)
    raw = _split_top_level(m.group(1), ",") if m else [first]
    models: List[str] = []
    names: List[str] = []
    for item in raw:
        k, eq, v = item.partition("=")
        if eq and not v.startswith("="):
            names.append(_strip_quotes(k.strip()))
            models.append(v.strip())
        else:
            models.append(item.strip())
    if not all(re.fullmatch(_NAME, x) for x in models):
        return _emit_error(
            "modelsummary: pass fitted models by name", command="modelsummary"
        )
    code = f"sp.modelsummary({', '.join(models)}"
    args: Dict[str, Any] = {}
    if names and len(names) == len(models):
        args["model_names"] = names
        code += f", model_names={names!r}"
    out = _emit(
        "modelsummary",
        args,
        code + ")",
        ["The models are the fitted StatsPAI results."],
    )
    return _unread(out, kw, ())


# ---------------------------------------------------------------------------
# WeightIt::weightit
# ---------------------------------------------------------------------------

#: WeightIt methods sp.weightit implements ("ps" is WeightIt's old name for
#: "glm").
_WEIGHTIT_METHODS = {"glm": "glm", "ps": "glm", "cbps": "cbps", "ebal": "ebal",
                     "energy": "energy"}  # fmt: skip


def _h_weightit(pos: List[str], kw: Dict[str, str], _: List[str]) -> Dict[str, Any]:
    """``weightit(treat ~ x1 + x2, data, method, estimand, focal, moments,
    int, d.moments)`` -> ``sp.weightit``.

    Methods ``glm`` (alias ``ps``), ``cbps``, ``ebal`` and ``energy``;
    WeightIt's default method is ``glm`` and its default estimand ``ATE``,
    as are sp.weightit's. ``int = TRUE`` is ``interactions=True``. Other
    arguments (``stabilize``, ``s.weights``, ``by``, ``ps``, the energy
    tuning arguments ...) change the weights and are listed as
    untranslated.
    """
    fml = _formula(pos, kw)
    lhs, sep, rhs = (fml or "").partition("~")
    covs = _terms(rhs) if sep else None
    treat = lhs.strip()
    if covs is None or not re.fullmatch(_NAME, treat):
        return _emit_error(
            "weightit: only `treat ~ x1 + x2 + ...` with column names is "
            "translated (build transformed terms as columns first).",
            command="weightit",
            suggestions=["weightit"],
        )
    read = ["formula", "method", "estimand", "focal", "moments", "int", "d.moments"]
    method_r = _strip_quotes(kw.get("method", '"glm"')).strip()
    method = _WEIGHTIT_METHODS.get(method_r)
    if method is None:
        return _emit_error(
            f"weightit: method = {method_r!r} is not implemented in "
            "sp.weightit (it has 'glm', 'cbps', 'ebal' and 'energy').",
            command="weightit",
            suggestions=["weightit"],
            statspai_functions=["weightit"],
        )
    args: Dict[str, Any] = {"treat": treat, "covariates": covs, "method": method}
    estimand = _strip_quotes(kw.get("estimand", '"ATE"')).strip().upper()
    args["estimand"] = estimand
    if "focal" in kw:
        args["focal"] = _strip_quotes(kw["focal"]).strip()
    for r_name, sp_name in (("moments", "moments"), ("d.moments", "d_moments")):
        if r_name in kw:
            value = _number(kw[r_name])
            if value is None or value != int(value):
                return _emit_error(
                    f"weightit: {r_name} must be a whole number.",
                    command="weightit",
                )
            args[sp_name] = int(value)
    if _true(kw.get("int")):
        args["interactions"] = True
    code = "sp.weightit(df, " + ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit(
        "weightit",
        args,
        code + ")",
        [
            "The weights are W.weights; W.balance() is cobalt::bal.tab. "
            "Estimate the effect with a weighted regression (e.g. "
            "sp.regress(..., weights=W.weights)) as after WeightIt."
        ],
    )
    return _unread(out, kw, tuple(read))


HANDLERS = {
    "ivreg": _h_ivreg,
    "coxph": _h_coxph,
    "pwr.t.test": _h_pwr_t_test,
    "svydesign": _h_svydesign,
    "svyglm": _h_svyglm,
    "coeftest": _h_coeftest,
    "etwfe": _h_etwfe,
    "emfx": _h_emfx,
    "aggte": _h_aggte,
    "did_multiplegt_dyn": _h_did_multiplegt_dyn,
    "modelsummary": _h_modelsummary,
    "weightit": _h_weightit,
}


#: R functions whose inputs are objects built in other lines; refused with
#: the sp call to make instead. ``(message, [sp functions])``.
R_GUIDANCE: Dict[str, Tuple[str, List[str]]] = {
    "synthdid_estimate": (
        "synthdid_estimate(Y, N0, T0) takes the matrices of panel.matrices(); "
        "call sp.synthdid_estimate(data, y=, unit=, time=, treat_unit=, "
        "treat_time=) (an alias of sp.sdid) on the long panel instead.",
        ["synthdid_estimate", "sdid"],
    ),
    "mediate": (
        "mediation::mediate(model.m, model.y, treat=, mediator=) takes two "
        "fitted models; sp.mediate(data, y=, treat=, mediator=, covariates=) "
        "fits both (one covariate set for the two equations).",
        ["mediate"],
    ),
    "CausalImpact": (
        "CausalImpact(data, pre.period, post.period) takes a matrix with the "
        "response first; sp.causal_impact(data, y=, time=, "
        "intervention_time=, covariates=) takes the long data frame. The "
        "models differ (bsts vs StatsPAI's structural model), so the "
        "intervals are not expected to match.",
        ["causal_impact"],
    ),
    "lagsarlm": (
        "spatialreg::lagsarlm(formula, data, listw) needs the listw object; "
        "build the weights matrix W and call sp.sar(W, data, formula).",
        ["sar"],
    ),
    "pc": (
        "pcalg::pc(suffStat, indepTest, alpha, labels) takes sufficient "
        "statistics; sp.pc_algorithm(data, alpha=, ci_test='fisherz') runs "
        "the PC algorithm with the Gaussian test (gaussCItest) on the data.",
        ["pc_algorithm"],
    ),
    "mr_ivw": (
        "MendelianRandomization::mr_ivw(mr_input(bx, bxse, by, byse)) takes "
        "an MRInput object; call sp.mr_ivw(beta_exposure=, beta_outcome=, "
        "se_exposure=, se_outcome=) with the four arrays.",
        ["mr_ivw"],
    ),
}
