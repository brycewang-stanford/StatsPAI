"""Translation of ``fuzzydid`` (de Chaisemartin, D'Haultfoeuille and
Guyonvarch) to ``sp.fuzzy_did``.

``fuzzydid Y G T D`` (two periods, G the 0/1 switching group) and
``fuzzydid Y G G_F T D`` (several groups / periods: G the change into the
period, G_F the change out of it). Registered through
``_stata_did.HANDLERS``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ._stata import _emit, _emit_error, _expand_abbreviations
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS"]

#: Options with no counterpart in sp.fuzzy_did; refused, never dropped.
_REFUSED = {
    "lqte": "the local quantile treatment effects",
    "partial": "the bounds without a stable control group",
    "sieves": "the nonparametric (sieve) conditional means",
    "sieveorder": "the nonparametric (sieve) conditional means",
}


def _names(cmd: StataCommand, option: str) -> "tuple[str, List[str]]":
    raw = cmd.options.get(option)
    if not raw:
        return "", []
    err, names = _expand_abbreviations(raw.split(), cmd.columns)
    if err:
        return f"fuzzydid {option}(): {err}", []
    for n in names:
        if any(ch in n for ch in ".#():*"):
            return (
                f"fuzzydid {option}(): the factor-variable term {n!r} is not "
                "translated; list plain variables",
                [],
            )
    return "", names


def _h_fuzzydid(cmd: StataCommand) -> Dict[str, Any]:
    """``fuzzydid Y G [G_F] T D, did tc cic [newcateg() numerator nose
    cluster() breps() eqtest continuous() qualitative() modelx()]`` ->
    ``sp.fuzzy_did``.

    The point estimates equal the command's (Wald-CIC with
    ``cic_quantile='stata'``, the command's floating-point quantile rule).
    Standard errors are a bootstrap like the command's, so they agree up to
    the draws. ``lqte``, ``partial`` and ``sieves`` are refused.
    """
    name = "fuzzydid"
    if len(cmd.varlist) not in (4, 5):
        return _emit_error(
            "fuzzydid is positional: `fuzzydid Y G T D` or `fuzzydid Y G G_F T D`.",
            command=name,
        )
    for v in cmd.varlist:
        if any(ch in v for ch in ".#():*"):
            return _emit_error(
                f"fuzzydid: {v!r} is not a plain variable name.", command=name
            )
    opts = cmd.options
    for option, what in _REFUSED.items():
        if option in opts:
            return _emit_error(
                f"fuzzydid, {option} is not translated: sp.fuzzy_did does not "
                f"compute {what}.",
                command=name,
                suggestions=[],
            )
    if len(cmd.varlist) == 4:
        y, g, t, d = cmd.varlist
        args: Dict[str, Any] = {"y": y, "group_change": g, "time": t, "treat": d}
    else:
        y, g, gf, t, d = cmd.varlist
        args = {"y": y, "group_change": g, "time": t, "treat": d}
        args["group_change_next"] = gf
    est = [k for k in ("did", "tc", "cic") if k in opts]
    if not est:
        return _emit_error(
            "fuzzydid needs at least one of did, tc, cic (the command refuses "
            "it otherwise).",
            command=name,
        )
    args["estimators"] = est
    notes: List[str] = []
    lost: List[str] = []
    err, cont = _names(cmd, "continuous")
    if err:
        return _emit_error(err, command=name)
    err, qual = _names(cmd, "qualitative")
    if err:
        return _emit_error(err, command=name)
    covs = cont + [f"C({q})" for q in qual]
    if covs:
        args["covariates"] = covs
    raw_model = opts.get("modelx")
    if raw_model:
        models = raw_model.lower().split()
        if len(models) in (2, 3) and all(
            m in ("ols", "logit", "probit") for m in models
        ):
            args["modelx"] = models
        else:
            lost.append("modelx")
            notes.append(
                f"modelx({raw_model}) is not two or three of ols / logit / probit."
            )
    raw_cat = opts.get("newcateg")
    if raw_cat:
        try:
            args["newcateg"] = [float(v) for v in raw_cat.replace(",", " ").split()]
        except ValueError:
            lost.append("newcateg")
            notes.append(f"newcateg({raw_cat}) is not a list of numbers.")
    if "numerator" in opts:
        args["numerator"] = True
    if "nose" in opts:
        args["se_method"] = "none"
    else:
        args["se_method"] = "bootstrap"
        raw_b = opts.get("breps")
        try:
            args["n_boot"] = int(raw_b) if raw_b else 50
        except ValueError:
            lost.append("breps")
            notes.append(f"breps({raw_b}) is not an integer.")
        if opts.get("cluster"):
            args["cluster"] = (opts.get("cluster") or "").split()[0]
    if "eqtest" in opts:
        args["eqtest"] = True
    if "cic" in est:
        args["cic_quantile"] = "stata"
    if "tagobs" in opts:
        lost.append("tagobs")
        notes.append(
            "tagobs (a variable marking the estimation sample) is not created; "
            "the rows used are those with Y, D, T and a group code."
        )
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    out = _emit("fuzzy_did", args, f"sp.fuzzy_did(data=df, {kw})", notes)
    out["untranslated_options"] = lost
    out["semantics"] = [
        "Point estimates equal fuzzydid's. Standard errors are bootstrap "
        "standard deviations as in the command; they agree with Stata's up to "
        "the draws, not digit by digit. p-values and percentile intervals are "
        "computed the same way."
    ]
    if "cic" in est:
        out["semantics"].append(
            "cic_quantile='stata' reproduces the command's Wald-CIC, which "
            "computes ceil(u * n) in floating point; drop it for the exact "
            "quantile (they differ only when u * n is an integer)."
        )
    return out


HANDLERS = {"fuzzydid": _h_fuzzydid}
