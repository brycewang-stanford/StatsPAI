"""Translations of the user-written difference-in-differences commands.

``drdid`` maps to ``sp.drdid``; ``did2s`` to the general form of
``sp.gardner_did``; ``jwdid`` to ``sp.jwdid``; ``csdid_estat`` and the
``estat simple | group | calendar | event`` that both ``csdid`` and ``jwdid``
define map to ``sp.estat``, which aggregates with the conventions of the
command the result came from.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from ._stata import _emit, _emit_error, _expand_abbreviations, _numlist
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
    # Stata reads `ib0. x` as `ib0.x`
    second = re.sub(r"\b(i(?:b\d+)?\.)\s+(?=[A-Za-z_])", r"\1", second)
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


def _option_varlist(
    cmd: StataCommand, name: str, command: str
) -> "tuple[Optional[Dict[str, Any]], List[str]]":
    """The variables named in a varlist option, wildcards and ranges expanded."""
    raw = cmd.options.get(name)
    if not raw:
        return None, []
    err, names = _expand_abbreviations(raw.split(), cmd.columns)
    if err:
        return _emit_error(f"{command} {name}(): {err}", command=command), []
    bad = _plain_names(names, command)
    return bad, names


def _positional(
    cmd: StataCommand, command: str, labels: str, at_least: int, at_most: int
) -> Optional[Dict[str, Any]]:
    if not at_least <= len(cmd.varlist) <= at_most:
        return _emit_error(
            f"{command} is positional: `{command} {labels}`.", command=command
        )
    return _plain_names(list(cmd.varlist), command)


def _h_twowayfeweights(cmd: StataCommand) -> Dict[str, Any]:
    """``twowayfeweights Y G T D [D0], type(feTR | fdTR) [controls()
    other_treatments() test_random_weights() weight()]`` ->
    ``sp.twowayfeweights``.

    ``type(feS)`` and ``type(fdS)`` have no counterpart and are refused.
    """
    name = "twowayfeweights"
    bad = _positional(cmd, name, "Y G T D [D0]", 4, 5)
    if bad:
        return bad
    opts = cmd.options
    kind = (opts.get("type") or "").strip()
    if kind not in ("feTR", "fdTR"):
        return _emit_error(
            f"twowayfeweights, type({kind}) is not translated: sp.twowayfeweights "
            "has type='feTR' (fixed effects regression) and 'fdTR' (first-"
            "difference regression).",
            command=name,
            suggestions=[],
        )
    y, group, time, treat = cmd.varlist[:4]
    args: Dict[str, Any] = {"y": y, "group": group, "time": time, "treat": treat}
    if kind == "fdTR":
        if len(cmd.varlist) != 5:
            return _emit_error(
                "twowayfeweights, type(fdTR) takes five variables: the first "
                "differences of the outcome and of the treatment, and the "
                "treatment in levels last.",
                command=name,
            )
        args["type"] = "fdTR"
        args["treat_level"] = cmd.varlist[4]
    elif len(cmd.varlist) == 5:
        return _emit_error(
            "twowayfeweights, type(feTR) takes four variables (Y G T D).",
            command=name,
        )
    for option, argument in (
        ("controls", "covariates"),
        ("other_treatments", "other_treatments"),
        ("test_random_weights", "test_random_weights"),
    ):
        bad, names = _option_varlist(cmd, option, name)
        if bad:
            return bad
        if names:
            args[argument] = names
    if opts.get("weight"):
        args["weights"] = (opts.get("weight") or "").split()[0]
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("twowayfeweights", args, f"sp.twowayfeweights(data=df, {kw})", [])


def _h_did_multiplegt_dyn(cmd: StataCommand) -> Dict[str, Any]:
    """``did_multiplegt_dyn Y G T D, effects() placebo() ...`` ->
    ``sp.did_multiplegt_dyn`` with the analytic variance and the
    switcher-weighted average, which is what the command reports.

    ``effects(k)`` is ``dynamic=k-1``: the command's ``Effect_1`` is the
    effect at the switch period, horizon 0.
    """
    name = "did_multiplegt_dyn"
    bad = _positional(cmd, name, "Y G T D", 4, 4)
    if bad:
        return bad
    opts = cmd.options
    y, group, time, treat = cmd.varlist
    args: Dict[str, Any] = {
        "y": y,
        "group": group,
        "time": time,
        "treatment": treat,
    }
    notes: List[str] = []
    lost: List[str] = []
    for option, default in (("effects", 1), ("placebo", 0)):
        raw = opts.get(option)
        try:
            value = default if raw is None else int(raw)
        except ValueError:
            return _emit_error(
                f"did_multiplegt_dyn, {option}({raw}) is not an integer.",
                command=name,
            )
        if option == "effects":
            if value < 1:
                return _emit_error(
                    "did_multiplegt_dyn, effects() must be at least 1.", command=name
                )
            args["dynamic"] = value - 1
        else:
            args["placebo"] = value
    args["se_method"] = "analytic"
    args["aggregation"] = "switchers"
    switchers = opts.get("switchers")
    if switchers is not None:
        if switchers.strip() in ("in", "out"):
            args["switchers"] = switchers.strip()
        else:
            lost.append("switchers")
            notes.append(f"switchers({switchers}) is neither in nor out.")
    if "only_never_switchers" in opts:
        args["control"] = "never_treated"
    for flag in ("normalized", "same_switchers", "normalized_weights"):
        if flag in opts:
            args[flag] = True
    design = opts.get("design")
    if design is not None:
        share, _, where = design.partition(",")
        try:
            covered = float(share) if share.strip() else 1.0
        except ValueError:
            covered = -1.0
        if 0 < covered <= 1 and where.strip() in ("", "console"):
            args["design"] = covered
        else:
            lost.append("design")
            notes.append(
                f"design({design}): only `design(p, console)` with a share p "
                "is translated; the table is model_info['design']."
            )
    paths = opts.get("by_path")
    if paths is not None:
        if paths.strip().isdigit() and int(paths) > 0:
            args["by_path"] = int(paths)
        else:
            lost.append("by_path")
            notes.append(f"by_path({paths}) is not a number of paths.")
    equal = opts.get("effects_equal")
    if equal is not None:
        text = equal.replace('"', "").strip()
        if text == "all":
            args["effects_equal"] = True
        else:
            bounds = _numlist(text)
            if bounds is None or len(bounds) != 2:
                lost.append("effects_equal")
                notes.append(
                    f"effects_equal({equal}) is neither all nor a pair of bounds."
                )
            else:
                # the command numbers effects from 1, sp horizons from 0
                args["effects_equal"] = (bounds[0] - 1, bounds[1] - 1)
    for option in ("controls", "trends_nonparam"):
        bad, names = _option_varlist(cmd, option, name)
        if bad:
            return bad
        if names:
            args[option] = names
    if opts.get("continuous") is not None:
        try:
            degree = int(opts.get("continuous") or "")
        except ValueError:
            degree = -1
        if degree > 0:
            args["continuous"] = degree
        elif degree != 0:
            lost.append("continuous")
            notes.append(f"continuous({opts.get('continuous')}) is not a degree.")
    if opts.get("weight"):
        args["weights"] = (opts.get("weight") or "").split()[0]
    if opts.get("cluster"):
        args["cluster"] = (opts.get("cluster") or "").split()[0]
    level = opts.get("ci_level")
    if level is not None:
        try:
            args["alpha"] = round(1 - float(level) / 100.0, 10)
        except ValueError:
            lost.append("ci_level")
            notes.append(f"ci_level({level}) is not a number.")
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit(
        "did_multiplegt_dyn", args, f"sp.did_multiplegt_dyn(data=df, {kw})", notes
    )
    out["untranslated_options"] = lost
    out["semantics"] = [
        "Effect_k of the command is relative_time k-1 in "
        "model_info['event_study']; Placebo_k is relative_time -k. "
        "Av_tot_eff is the result's estimate.",
        "On an unbalanced panel with a non-binary treatment the command can "
        "discard a not-yet-switched group as a control because of what its "
        "treatment does after it switches (it drops the periods of a "
        "baseline treatment that have no control, then the groups whose "
        "remaining post-switch treatment averages to their baseline). "
        "sp.did_multiplegt_dyn keeps such a group, so a few switchers more "
        "can have an estimable effect; otherwise the numbers are the same.",
    ]
    return out


def _h_did_had(cmd: StataCommand) -> Dict[str, Any]:
    """``did_had Y G T D, effects() placebo() [kernel() dynamic trends_lin
    yatchew level()]`` -> ``sp.did_had``."""
    name = "did_had"
    bad = _positional(cmd, name, "Y G T D", 4, 4)
    if bad:
        return bad
    opts = cmd.options
    y, group, time, treat = cmd.varlist
    args: Dict[str, Any] = {"y": y, "group": group, "time": time, "treat": treat}
    notes: List[str] = []
    lost: List[str] = []
    for option in ("effects", "placebo"):
        raw = opts.get(option)
        if raw is None:
            continue
        try:
            args[option] = int(raw)
        except ValueError:
            return _emit_error(
                f"did_had, {option}({raw}) is not an integer.", command=name
            )
    kernel = opts.get("kernel")
    if kernel is not None:
        short = {"epa": "epanechnikov", "tri": "triangular", "uni": "uniform"}
        key = kernel.strip().lower()
        if key in short or key in short.values() or key == "gau":
            args["kernel"] = short.get(key, key)
        else:
            lost.append("kernel")
            notes.append(f"kernel({kernel}) is not one of epa, tri, uni, gau.")
    for flag in ("dynamic", "trends_lin", "yatchew"):
        if flag in opts:
            args[flag] = True
    level = opts.get("level")
    if level is not None:
        try:
            args["alpha"] = float(level)
        except ValueError:
            lost.append("level")
            notes.append(f"level({level}) is not a number.")
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("did_had", args, f"sp.did_had(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    return out


def _h_did_multiplegt_old(cmd: StataCommand) -> Dict[str, Any]:
    """``did_multiplegt_old Y G T D [, placebo() breps() cluster() controls()
    seed()]`` -> ``sp.did_multiplegt``, the DID_M estimator.

    ``robust_dynamic`` and ``dynamic()`` switch the command to the
    intertemporal estimators, which are ``sp.did_multiplegt_dyn``'s: refused
    here so that the pair-by-pair estimator is not returned in their place.
    """
    name = cmd.command
    bad = _positional(cmd, name, "Y G T D", 4, 4)
    if bad:
        return bad
    opts = cmd.options
    if "robust_dynamic" in opts or opts.get("dynamic") not in (None, "0"):
        return _emit_error(
            f"{name}, robust_dynamic / dynamic() are the intertemporal "
            "estimators; use `did_multiplegt_dyn Y G T D, effects() "
            "placebo()`, which translates to sp.did_multiplegt_dyn.",
            command=name,
            suggestions=["did_multiplegt_dyn"],
        )
    y, group, time, treat = cmd.varlist
    args: Dict[str, Any] = {
        "y": y,
        "group": group,
        "time": time,
        "treatment": treat,
        # the command's placebos are forward differences
        "placebo_sign": "r",
    }
    notes: List[str] = []
    lost: List[str] = []
    for option, argument, default in (
        ("placebo", "placebo", None),
        ("breps", "n_boot", 50),
        ("seed", "seed", None),
    ):
        raw = opts.get(option)
        if raw is None:
            if default is not None:
                args[argument] = default
            continue
        try:
            args[argument] = int(raw)
        except ValueError:
            lost.append(option)
            notes.append(f"{option}({raw}) is not an integer.")
    bad, names = _option_varlist(cmd, "controls", name)
    if bad:
        return bad
    if names:
        args["controls"] = names
    if opts.get("cluster"):
        args["cluster"] = (opts.get("cluster") or "").split()[0]
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("did_multiplegt", args, f"sp.did_multiplegt(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    out["semantics"] = [
        "Standard errors are from a cluster bootstrap, as the command's: "
        "they agree with Stata's up to the draws, not digit by digit."
    ]
    return out


HANDLERS: Dict[str, Callable[[StataCommand], Dict[str, Any]]] = {
    "did2s": _h_did2s,
    "twowayfeweights": _h_twowayfeweights,
    "did_multiplegt_dyn": _h_did_multiplegt_dyn,
    "did_had": _h_did_had,
    "did_multiplegt_old": _h_did_multiplegt_old,
    "drdid": _h_drdid,
    "jwdid": _h_jwdid,
    "csdid_estat": _h_csdid_estat,
}
POSTEST: FrozenSet[Callable[[StataCommand], Dict[str, Any]]] = frozenset(
    {_h_csdid_estat}
)
