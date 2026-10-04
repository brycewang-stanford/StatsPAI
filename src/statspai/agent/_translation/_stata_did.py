"""Translations of the user-written difference-in-differences commands.

``drdid`` (Rios-Avila, Sant'Anna and Naqvi's port of ``DRDID``) maps to
``sp.drdid``; ``did2s`` (Butts) to the general form of ``sp.gardner_did``; ``jwdid`` (Rios-Avila's extended TWFE) to ``sp.jwdid``;
``csdid_estat`` and the ``estat simple | group | calendar | event`` that
both ``csdid`` and ``jwdid`` define map to ``sp.estat``, which aggregates
with the conventions of the command the result came from.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from ._stata import _emit, _emit_error
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "POSTEST", "did_aggregation"]

#: drdid's estimator flags -> ``sp.drdid`` arguments. The command's default
#: is ``drimp``; ``sp.drdid``'s defaults are the same estimator, and are
#: written out so the call does not depend on them.
_DRDID_METHODS: Dict[str, Dict[str, Any]] = {
    "drimp": {"est_method": "dr", "method": "imp"},
    "dripw": {"est_method": "dr", "method": "trad"},
    "reg": {"est_method": "reg"},
    "stdipw": {"est_method": "ipw", "normalized": True},
    "ipw": {"est_method": "ipw", "normalized": False},
}

_AGGREGATIONS = ("simple", "group", "calendar", "event")


def _plain_names(names: List[str], command: str) -> Optional[Dict[str, Any]]:
    for name in names:
        if any(ch in name for ch in ".#():*"):
            return _emit_error(
                f"{command}: the factor-variable term {name!r} is not "
                "translated; create the indicator columns first and list them.",
                command=command,
                suggestions=[],
            )
    return None


def _h_drdid(cmd: StataCommand) -> Dict[str, Any]:
    """``drdid y [x], [ivar(id)] time(t) treatment(d) [estimator]`` ->
    ``sp.drdid``.

    With ``ivar()`` the data are a two-period panel; without it, repeated
    cross-sections. ``all`` prints the five estimators side by side, which
    is five calls, and ``ipwra`` has no counterpart: both are refused.
    """
    if not cmd.varlist:
        return _emit_error("drdid requires an outcome variable", command="drdid")
    opts = cmd.options
    y, xs = cmd.varlist[0], list(cmd.varlist[1:])
    bad = _plain_names(xs, "drdid")
    if bad:
        return bad
    time, treat = opts.get("time"), opts.get("treatment")
    if not time or not treat:
        return _emit_error(
            "drdid needs `time(<period variable>)` and "
            "`treatment(<treated-group indicator>)`.",
            command="drdid",
        )
    if "all" in opts:
        if any(name in opts for name in (*_DRDID_METHODS, "rc1")):
            return _emit_error(
                "drdid: all is every estimator; it does not combine with "
                "another estimator option.",
                command="drdid",
            )
        all_args: Dict[str, Any] = {
            "y": y,
            "group": treat.split()[0],
            "time": time.split()[0],
        }
        if xs:
            all_args["covariates"] = xs
        if opts.get("ivar"):
            all_args["id"] = (opts.get("ivar") or "").split()[0]
        all_args["est_method"] = "all"
        lost_all: List[str] = []
        notes_all: List[str] = []
        raw = opts.get("pscoretrim")
        if raw is not None:
            try:
                all_args["trim_level"] = float(raw)
            except ValueError:
                lost_all.append("pscoretrim")
                notes_all.append(f"pscoretrim({raw}) is not a number.")
        kw = ", ".join(f"{k}={v!r}" for k, v in all_args.items())
        out = _emit("drdid", all_args, f"sp.drdid(data=df, {kw})", notes_all)
        out["untranslated_options"] = lost_all
        out["semantics"] = [
            "The estimators are the rows of result.detail; the result "
            "itself is drimp. On a panel Stata also prints sipwra, which "
            "sp.drdid does not have."
        ]
        return out
    if "ipwra" in opts:
        return _emit_error(
            "drdid, ipwra (inverse-probability-weighted regression "
            "adjustment) has no sp.drdid estimator; the doubly robust ones "
            "are drimp and dripw.",
            command="drdid",
            suggestions=[],
        )
    chosen = [name for name in _DRDID_METHODS if name in opts]
    if len(chosen) > 1:
        return _emit_error(
            f"drdid: only one estimator may be selected, got {chosen}.",
            command="drdid",
        )
    method = chosen[0] if chosen else "drimp"
    args: Dict[str, Any] = {
        "y": y,
        "group": treat.split()[0],
        "time": time.split()[0],
    }
    if xs:
        args["covariates"] = xs
    ivar = opts.get("ivar")
    if ivar:
        args["id"] = ivar.split()[0]
    args.update(_DRDID_METHODS[method])
    notes: List[str] = []
    lost: List[str] = []
    if "rc1" in opts:
        if ivar or args["est_method"] != "dr":
            lost.append("rc1")
            notes.append(
                "rc1 selects the not-locally-efficient doubly robust "
                "estimator for repeated cross-sections; it does not apply "
                "to this call."
            )
        else:
            args["locally_efficient"] = False
    raw_trim = opts.get("pscoretrim")
    if raw_trim is not None:
        try:
            args["trim_level"] = float(raw_trim)
        except ValueError:
            lost.append("pscoretrim")
            notes.append(f"pscoretrim({raw_trim}) is not a number.")
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("drdid", args, f"sp.drdid(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    out["semantics"] = [
        "drdid needs exactly two periods in the estimation sample; the "
        "later one is the post-treatment period."
        + (
            ""
            if chosen
            else " No estimator was named, so this is drdid's default, drimp."
        )
    ]
    return out


def _h_jwdid(cmd: StataCommand) -> Dict[str, Any]:
    """``jwdid y [x], ivar(id) tvar(t) gvar(g) [never method() hettype()]``
    -> ``sp.jwdid``, whose result is the fit followed by ``estat simple``."""
    if not cmd.varlist:
        return _emit_error("jwdid requires an outcome variable", command="jwdid")
    opts = cmd.options
    y, xs = cmd.varlist[0], list(cmd.varlist[1:])
    ivar = opts.get("ivar")
    tvar = opts.get("tvar") or opts.get("time")
    gvar = opts.get("gvar")
    trtvar = opts.get("trtvar")
    if trtvar is not None:
        return _emit_error(
            "jwdid, trtvar() builds the cohort from a treatment indicator; "
            "sp.jwdid takes the first-treatment period. Create it "
            "(the first period with the indicator on, 0 if never) and pass "
            "it as gvar().",
            command="jwdid",
            suggestions=[],
        )
    if not (ivar and tvar and gvar):
        return _emit_error(
            "jwdid translation needs `ivar()`, `tvar()` and `gvar()`. "
            "Without ivar() jwdid uses cohort instead of unit fixed "
            "effects, which is sp.etwfe(fe='cohort').",
            command="jwdid",
        )
    args: Dict[str, Any] = {
        "y": y,
        "ivar": ivar.split()[0],
        "tvar": tvar.split()[0],
        "gvar": gvar.split()[0],
    }
    if xs:
        args["x"] = xs
    notes: List[str] = []
    lost: List[str] = []
    method = opts.get("method")
    if method is not None:
        head = method.replace(",", " ").split()
        if len(head) == 1 and head[0].lower() in {
            "regress",
            "reghdfe",
            "ppmlhdfe",
            "poisson",
            "logit",
        }:
            name = head[0].lower()
            if name != "reghdfe":
                args["method"] = name
        else:
            lost.append("method")
            notes.append(
                f"method({method}): only regress, ppmlhdfe, poisson and "
                "logit, without options of their own, are translated."
            )
    if "never" in opts:
        args["never"] = True
    hettype = opts.get("hettype")
    if hettype is not None:
        args["hettype"] = hettype.strip().lower()
    exovar = opts.get("exovar")
    if exovar:
        args["exovar"] = exovar.split()
    cluster = opts.get("cluster")
    if cluster:
        args["cluster"] = cluster.split()[0]
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("jwdid", args, f"sp.jwdid(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    out["semantics"] = [
        "The result's estimate is jwdid followed by `estat simple`; "
        "`estat event | group | calendar` aggregate the same fit."
    ]
    return out


_FE_TERM = re.compile(r"^i\.([A-Za-z_]\w*)(?:#i\.([A-Za-z_]\w*))?$")
_NAME = re.compile(r"^[A-Za-z_]\w*$")
_STAGE2_FACTOR = re.compile(r"^i(?:b\d+)?\.[A-Za-z_]\w*$")


def _h_did2s(cmd: StataCommand) -> Dict[str, Any]:
    """``did2s y, first_stage() second_stage() treatment(D) cluster(c)
    [unit(u)]`` -> ``sp.gardner_did`` in its general form.

    ``unit(u)`` demeans within ``u`` on the untreated rows, which is a unit
    fixed effect in the first stage. A second stage that is the treatment
    dummy itself is the static ATT and is left to the default.
    """
    from ._stata import _expand_abbreviations

    if len(cmd.varlist) != 1:
        return _emit_error("did2s takes one outcome variable", command="did2s")
    opts = cmd.options
    first, second = opts.get("first_stage"), opts.get("second_stage")
    treat, cluster = opts.get("treatment"), opts.get("cluster")
    if not (first and second and treat and cluster):
        return _emit_error(
            "did2s needs first_stage(), second_stage(), treatment() and " "cluster().",
            command="did2s",
        )
    treat = treat.split()[0]
    fe: List[str] = []
    unit = opts.get("unit")
    if unit:
        fe.append(unit.split()[0])
    err, first_tokens = _expand_abbreviations(first.split(), cmd.columns)
    if err:
        return _emit_error(f"did2s first_stage(): {err}", command="did2s")
    controls: List[str] = []
    for tok in first_tokens:
        m = _FE_TERM.match(tok)
        if m:
            spec = m.group(1) if m.group(2) is None else f"{m.group(1)}#{m.group(2)}"
            if spec not in fe:
                fe.append(spec)
        elif _NAME.match(tok):
            controls.append(tok)
        else:
            return _emit_error(
                f"did2s first_stage(): the term {tok!r} is not translated; "
                "fixed effects `i.x`, cells `i.a#i.b` and plain covariates "
                "are.",
                command="did2s",
                suggestions=[],
            )
    err, second_tokens = _expand_abbreviations(second.split(), cmd.columns)
    if err:
        return _emit_error(f"did2s second_stage(): {err}", command="did2s")
    for tok in second_tokens:
        if not (_NAME.match(tok) or _STAGE2_FACTOR.match(tok)):
            return _emit_error(
                f"did2s second_stage(): the term {tok!r} is not translated; "
                "columns and `i.x` / `ib<k>.x` are.",
                command="did2s",
                suggestions=[],
            )
    args: Dict[str, Any] = {"y": cmd.varlist[0], "treat": treat, "fe": fe}
    if controls:
        args["controls"] = controls
    if second_tokens not in ([treat], [f"i.{treat}"]):
        args["second_stage"] = second_tokens
    args["cluster"] = cluster.split()[0]
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("gardner_did", args, f"sp.gardner_did(data=df, {kw})")
    out["semantics"] = [
        "The second stage has no intercept. With several second-stage "
        "terms the coefficients are in result.detail and result.estimate "
        "is NaN."
    ]
    return out


def did_aggregation(
    command: str, kind: str, opts: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    """``estat <kind>`` after ``csdid`` / ``jwdid`` -> ``sp.estat(result, kind)``.

    ``post``, ``estore()``, ``esave()`` and ``replace`` decide where Stata
    keeps the table and are read so they are not reported as lost; the
    aggregated result is what the call returns.
    """
    for name in ("post", "estore", "esave", "replace", "plot"):
        opts.get(name)
    args: Dict[str, Any] = {"test": kind}
    raw = opts.get("window")
    if raw is not None:
        try:
            lo, hi = sorted(int(v) for v in raw.replace(",", " ").split())
        except ValueError:
            return _emit_error(
                f"{command} {kind}: window({raw}) is not two integers.",
                command=command,
                suggestions=[],
            )
        if kind != "event":
            return _emit_error(
                f"{command} {kind}: window() applies to the event aggregation.",
                command=command,
                suggestions=[],
            )
        args["window"] = (lo, hi)
    args["print_results"] = False
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit(
        "estat",
        args,
        f"sp.estat(result, {shown})",
        semantics=[
            "After sp.callaway_santanna this is sp.aggte with csdid's "
            "conventions (cohort shares held fixed in the group average); "
            "after sp.jwdid it is sp.etwfe_emfx."
        ],
    )


def _h_csdid_estat(cmd: StataCommand) -> Dict[str, Any]:
    """``csdid_estat simple | group | calendar | event [, window(a b)]``."""
    if not cmd.varlist:
        return _emit_error(
            "csdid_estat needs an aggregation: simple, group, calendar or event.",
            command="csdid_estat",
            suggestions=[],
        )
    kind = cmd.varlist[0].lower()
    if kind not in _AGGREGATIONS or len(cmd.varlist) > 1:
        return _emit_error(
            f"csdid_estat {' '.join(cmd.varlist)} is not translated; the "
            "aggregations are simple, group, calendar and event.",
            command="csdid_estat",
            suggestions=[],
        )
    return did_aggregation("csdid_estat", kind, cmd.options)


HANDLERS: Dict[str, Callable[[StataCommand], Dict[str, Any]]] = {
    "did2s": _h_did2s,
    "drdid": _h_drdid,
    "jwdid": _h_jwdid,
    "csdid_estat": _h_csdid_estat,
}
POSTEST: FrozenSet[Callable[[StataCommand], Dict[str, Any]]] = frozenset(
    {_h_csdid_estat}
)
