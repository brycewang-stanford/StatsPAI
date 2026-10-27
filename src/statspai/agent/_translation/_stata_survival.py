"""Stata's Cox, competing-risks and survival treatment-effects commands:
``stcox``, ``stcrreg`` and ``stteffects``.

Both read the duration and the failure indicator from ``stset``, which
lives on another line: like ``streg`` they emit ``<stset ...>``
placeholders that ``sp.stata`` fills from an earlier ``stset timevar,
failure(eventvar)``.

Conventions pinned against Stata 18 (``tests/test_stata_translation_
survival.py``): ``stcox`` breaks ties with Breslow's method unless
``efron`` is given (``sp.cox`` defaults to Efron, so the method is always
written out); ``vce(cluster c)`` is ``sp.cox(cluster=c, ssc='stata')``
(G/(G-1)); ``vce(robust)`` is ``sp.cox(robust='hc0', ssc='stata')``
(N/(N-1)). A declaration ``stset t [pw=w], failure(d) time0(t0) id(i)``
(or ``enter(time t0)``) adds ``weights=w, weights_as='sampling'`` (robust
variance, as ``[pw=]`` forces), ``start=t0``, and clusters the robust
variance on the ``stset`` id, as ``stcox`` does. Stata refuses
``[pw=]`` with ``efron``, and so does the translation. ``tvc(varlist)
texp(exp)`` is ``sp.cox(tvc=[...], texp='exp')`` (``exp`` passes through
as a Stata expression in ``_t``; pinned against Stata 18 in
``tests/reference_parity/test_cox_tvc_parity.py``). ``stcrreg`` is
``sp.finegray(small_sample=True)``: its default robust variance carries
the same N/(N-1).

``stteffects ra | wra | ipw | ipwra`` is ``sp.stteffects`` (checked against
Stata 18 in ``tests/test_stata_translation_stteffects.py``). Two things the
call cannot carry are flagged rather than translated: ``noconstant`` in any
equation (``sp.stteffects`` always fits a constant) and ``ipwra, atet``,
which Stata averages over the whole sample and ``sp.stteffects`` over the
treated (``docs/dev/modules/survival.md``).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

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


def _stcox_tvc(opts: Any) -> Dict[str, Any]:
    """``tvc(varlist) texp(exp)`` -> ``{'tvc': [...], 'texp': 'exp'}``.

    ``texp`` is a Stata expression in ``_t``; ``stset`` declarations with
    ``origin()`` / ``scale()`` are refused upstream, so ``_t`` is the
    declared time variable and the expression passes through unchanged.
    It is read here with the same parser ``sp.cox`` uses, so an
    expression the fit cannot evaluate fails at translation. Returns an
    ``_emit_error`` payload (which carries ``ok``) when the options cannot
    be translated, and ``{}`` when neither is given.
    """
    raw_tvc = opts.get("tvc")
    raw_texp = opts.get("texp")
    if raw_tvc is None and raw_texp is None:
        return {}
    if raw_tvc is None:
        return _emit_error(
            "stcox, texp() without tvc(): Stata refuses it too.",
            command="stcox",
            suggestions=[],
        )
    tvc = str(raw_tvc).split()
    if not tvc or any(not re.fullmatch(r"[A-Za-z_]\w*", v) for v in tvc):
        return _emit_error(
            f"stcox, tvc(): only plain variable names are translated (got "
            f"tvc({raw_tvc})).",
            command="stcox",
            suggestions=[],
        )
    out: Dict[str, Any] = {"tvc": tvc}
    if raw_texp is not None:
        texp = " ".join(str(raw_texp).split())
        from ...exceptions import MethodIncompatibility
        from ...survival._cox_tvc import resolve_texp

        try:
            resolve_texp(texp)
        except MethodIncompatibility as exc:
            return _emit_error(
                f"stcox, texp({texp}) is not translated: {exc}",
                command="stcox",
                suggestions=[],
            )
        if texp != "_t":
            out["texp"] = texp
    return out


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
    for unsupported in ("shared", "offset"):
        if unsupported in opts:
            return _emit_error(
                f"stcox, {unsupported}() is not translated to sp.cox.",
                command="stcox",
                suggestions=[],
            )
    tvc_out = _stcox_tvc(opts)
    if isinstance(tvc_out, dict) and "ok" in tvc_out:
        return tvc_out
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
    if tvc_out:
        args.update(tvc_out)
    weight = (opts.get("stset_weight") or "").strip()
    time0 = (opts.get("stset_time0") or "").strip()
    st_id = (opts.get("stset_id") or "").strip()
    if weight and ties == "efron":
        return _emit_error(
            "stcox: Stata does not allow pweights with efron (r(101) "
            '"weights not allowed with efron option"); use breslow.',
            command="stcox",
            suggestions=[],
        )
    if time0:
        args["start"] = time0
    if weight:
        args["weights"] = weight
        args["weights_as"] = "sampling"
    cluster = _vce_cluster(cmd)
    vce = (opts.get("vce") or "").split()
    robust = "robust" in opts or (bool(vce) and vce[0].lower() == "robust")
    if cluster:
        args["cluster"] = cluster
        args["ssc"] = "stata"
    elif robust or weight:
        if st_id:
            # stcox clusters the robust variance on the stset id
            args["cluster"] = st_id
        elif not weight:
            args["robust"] = "hc0"
        args["ssc"] = "stata"
    if duration.startswith("<"):
        notes.append(_NEEDS_STSET.format(cmd="stcox"))
    code = ", ".join(["data=df"] + [f"{k}={v!r}" for k, v in args.items()])
    semantics = [
        f"stcox breaks ties with Breslow's method unless `efron` is "
        f"given; sp.cox defaults to Efron, so ties={ties!r} is written out."
    ]
    if "tvc" in args:
        semantics.append(
            "tvc(): the coefficients of z*g(_t) are named 'tvc:z' (Stata's tvc "
            "equation); texp() defaults to _t in both."
        )
    out = _emit("cox", args, f"sp.cox({code})", notes, semantics=semantics)
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


#: stteffects' omodel / cmodel words -> sp.stteffects dist names
_ST_DISTS = {"weibull": "weibull", "exponential": "exponential", "lnormal": "lognormal"}
#: the equations each stteffects estimator takes, in order
_ST_EQUATIONS = {
    "ra": ("outcome", "treat"),
    "wra": ("outcome", "treat", "censor"),
    "ipw": ("treat", "censor"),
    "ipwra": ("outcome", "treat", "censor"),
}
_ST_SYNTAX = {
    "ra": "stteffects ra (omvarlist [, omodel]) (tvar)",
    "wra": "stteffects wra (omvarlist [, omodel]) (tvar) (cmvarlist [, cmodel])",
    "ipw": "stteffects ipw (tvar [tmvarlist] [, tmodel]) (cmvarlist [, cmodel])",
    "ipwra": (
        "stteffects ipwra (omvarlist [, omodel]) (tvar [tmvarlist] [, tmodel]) "
        "[(cmvarlist [, cmodel])]"
    ),
}


def _st_refuse(message: str) -> Dict[str, Any]:
    return _emit_error(
        f"stteffects: {message}",
        command="stteffects",
        suggestions=[],
        statspai_functions=["stteffects"],
    )


def _h_stteffects(cmd: StataCommand) -> Dict[str, Any]:
    """``stteffects ra | wra | ipw | ipwra (...) (...) [(...)]`` ->
    ``sp.stteffects``.

    The equations are read by position, as Stata reads them: ``ra`` takes
    the outcome model and the treatment; ``wra`` adds the censoring model;
    ``ipw`` takes the treatment model and the censoring model; ``ipwra``
    the outcome and treatment models and, optionally, the censoring model
    (without it the censoring is in the likelihood). ``lnormal`` is
    ``dist='lognormal'``; the ``gamma`` model, ``hetprobit`` and an
    ``ancillary()`` equation are refused.
    """
    from ._stata import _equations

    if not cmd.varlist:
        return _st_refuse("expected `stteffects ra | wra | ipw | ipwra (...) (...)`.")
    method = str(cmd.varlist[0]).lower()
    if method not in _ST_EQUATIONS:
        return _st_refuse(f"estimator {method!r} is not one of ra, wra, ipw, ipwra.")
    eqs = _equations(list(cmd.varlist[1:]))
    wanted = _ST_EQUATIONS[method]
    lengths = (len(wanted) - 1, len(wanted)) if method == "ipwra" else (len(wanted),)
    # Stata reads an empty `()` as a syntax error; a constant-only model
    # is written `(, weibull)`
    if eqs is None or len(eqs) not in lengths or any(not v and not w for v, w in eqs):
        return _st_refuse(f"expected `{_ST_SYNTAX[method]}`.")
    parts: Dict[str, Any] = dict(zip(wanted, eqs))
    args: Dict[str, Any] = {}
    names = [v for vs, _ in eqs for v in vs]
    if any(not re.fullmatch(r"[A-Za-z_]\w*", v) for v in names):
        return _st_refuse(
            f"only plain variable names are translated (got {names!r}); "
            "build factor-variable indicators as columns first."
        )
    for role, (vars_, words) in parts.items():
        words = [w.lower() for w in words]
        if "noconstant" in words:
            return _st_refuse(
                f"noconstant in the {role} model is not translated: "
                "sp.stteffects always fits a constant."
            )
        if role == "treat":
            if not vars_:
                return _st_refuse(
                    "the treatment equation needs the treatment variable."
                )
            if method in ("ra", "wra") and len(vars_) > 1:
                return _st_refuse(
                    f"stteffects {method} takes no treatment-model covariates."
                )
            if method in ("ra", "wra") and words:
                return _st_refuse(f"stteffects {method} has no treatment model.")
            if words not in ([], ["logit"], ["probit"]):
                return _st_refuse(
                    f"treatment model {' '.join(words)!r} is not translated; "
                    "logit and probit are."
                )
            args["treat"] = vars_[0]
            if method in ("ipw", "ipwra"):
                args["treat_covariates"] = list(vars_[1:])
                if words == ["probit"]:
                    args["treat_model"] = "probit"
            continue
        if len(words) > 1 or (words and words[0] not in _ST_DISTS):
            return _st_refuse(
                f"{role} model {' '.join(words)!r} is not translated; "
                "weibull, exponential and lnormal are (gamma is not "
                "implemented in sp.stteffects)."
            )
        dist = _ST_DISTS[words[0]] if words else "weibull"
        if role == "outcome":
            args["covariates"] = list(vars_)
            if dist != "weibull":
                args["dist"] = dist
        else:
            args["censor_covariates"] = list(vars_)
            if dist != "weibull":
                args["censor_dist"] = dist
    duration, event = _stset(cmd)
    opts = cmd.options
    notes: List[str] = []
    untranslated: List[str] = []
    semantics: List[str] = []
    if "atet" in opts:
        args["estimand"] = "ATET"
        if method == "ipwra":
            untranslated.append("atet")
            notes.append(
                "stteffects ipwra, atet averages the outcome models' "
                "predictions over the whole sample, which estimates the ATE "
                "when the outcome model is right; sp.stteffects averages "
                "over the treated (docs/dev/modules/survival.md), so the "
                "numbers differ by design."
            )
    "ate" in opts  # the default; reading it marks it as handled
    if "pomeans" in opts:
        semantics.append(
            "pomeans: the same fit; Stata's two potential-outcome means are "
            "the rows POmean0 and POmean1 of result.detail."
        )
    vce = (opts.get("vce") or "").replace(",", " ").split()
    head = vce[0].lower() if vce else ""
    vce_default = not vce or (len(vce) == 1 and "robust".startswith(head))
    if duration.startswith("<"):
        notes.append(_NEEDS_STSET.format(cmd="stteffects"))
    treat = args.pop("treat")
    full = {"duration": duration, "event": event, "treat": treat, **args}
    if method != "ra":
        full["method"] = method
    code = ", ".join(["data=df"] + [f"{k}={v!r}" for k, v in full.items()])
    out = _emit(
        "stteffects",
        full,
        f"sp.stteffects({code})",
        notes,
        semantics=semantics
        + [
            "Standard errors: the robust sandwich of the stacked estimating "
            "equations, stteffects' default vce(robust)."
        ],
    )
    if vce_default:
        out["_vce_is_sp_default"] = True
    if untranslated:
        out["untranslated_options"] = untranslated
    return out


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "stcox": _h_stcox,
    "stcrreg": _h_stcrreg,
    "stteffects": _h_stteffects,
}
