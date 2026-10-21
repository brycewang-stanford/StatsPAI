"""``wyoung ..., permute()``: Westfall-Young stepdown -> ``sp.westfall_young``.

``wyoung y1 y2, cmd(regress OUTCOMEVAR d x [i.s], robust | cluster(c))
familyp(d) permute(d) [strata(s)] [cluster(c)] reps() seed()`` is the
permutation form of Reif's ``wyoung``: one regression per outcome,
re-randomizing ``d`` (within strata, by cluster), free step-down on the
p-values of ``d``. ``sp.westfall_young`` computes the same statistics
(coefficients, HC1 / CR1 t and their analytic p-values agree to 1e-13 on
``tests/reference_parity/_fixtures/wyoung_*.csv``) and the same adjusted
p-values, exactly when the re-randomizations can be enumerated and in
distribution otherwise.

Translated only when the two compute the same thing: classical, robust or
clustered SEs (``cmd(regress OUTCOMEVAR d x)`` without an SE option is
``robust=False``, the classical OLS t), stratum effects
``i.s`` in the model exactly when ``strata(s)`` is given (``sp.westfall_young``
adds them), and ``permute()`` of the familyp variable (without it ``wyoung``
bootstraps).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from ._stata import _emit, _emit_error
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "LITERAL"]

_NAME = r"[A-Za-z_]\w*"
_FUNCS = ["westfall_young", "romano_wolf"]

#: wyoung options that change what is tested or how it is resampled.
_UNSUPPORTED = (
    "subgroup",
    "controls",
    "controlsinteract",
    "familypexp",
    "permuteprogram",
    "noresampling",
    "force",
)


def _refuse(msg: str) -> Dict[str, Any]:
    return _emit_error(
        f"wyoung: {msg}",
        command="wyoung",
        suggestions=[],
        statspai_functions=list(_FUNCS),
    )


def _strip_quotes(text: str) -> str:
    t = text.strip()
    for open_, close in (('`"', "\"'"), ('"', '"')):
        if (
            t.startswith(open_)
            and t.endswith(close)
            and len(t) >= len(open_) + len(close)
        ):
            return t[len(open_) : -len(close)].strip()
    return t


def _se_spec(opts: str) -> Dict[str, Optional[str]]:
    """``{'kind': 'classical' | 'robust' | 'cluster', 'var': ...}`` from
    regress options (none, or ``vce(ols)``, is the classical variance)."""
    o = opts.strip().lower()
    if not o or re.fullmatch(r"vce\(\s*ols\s*\)", o):
        return {"kind": "classical", "var": None}
    m = re.fullmatch(
        r"(?:vce\(\s*)?(cl|clu|clus|clust|cluste|cluster)\s*\(?\s*("
        + _NAME
        + r")\s*\)?\s*\)?",
        o,
    )
    if m:
        return {"kind": "cluster", "var": m.group(2)}
    if re.fullmatch(
        r"r|ro|rob|robu|robus|robust|vce\(\s*(r|ro|rob|robu|robus|robust|hc1)\s*\)", o
    ):
        return {"kind": "robust", "var": None}
    return {"kind": "other", "var": o}


def _h_wyoung(cmd: StataCommand) -> Dict[str, Any]:
    outcomes = list(cmd.varlist)
    opts = cmd.options
    if not outcomes:
        return _refuse(
            "only syntax 1 (outcomes before the comma, OUTCOMEVAR in cmd()) "
            "is translated."
        )
    bad = [o for o in _UNSUPPORTED if o in opts]
    if bad:
        return _refuse(f"{', '.join(bad)}() is not translated.")
    raw_cmd = opts.get("cmd")
    family = (opts.get("familyp") or "").strip()
    permute = (opts.get("permute") or "").strip()
    if not raw_cmd or not re.fullmatch(_NAME, family):
        return _refuse("needs cmd(regress OUTCOMEVAR ...) and familyp(one variable).")
    if permute != family:
        return _refuse(
            "without permute(<familyp variable>) wyoung bootstraps; "
            "sp.westfall_young re-randomizes the treatment (permute form). For "
            "a bootstrap stepdown use sp.romano_wolf."
        )
    model, _, reg_opts = _strip_quotes(raw_cmd).partition(",")
    words = model.split()
    if (
        len(words) < 3
        or not "regress".startswith(words[0].lower())
        or len(words[0]) < 3
    ):
        return _refuse("only cmd(regress OUTCOMEVAR ...) is translated.")
    if words[1] != "OUTCOMEVAR":
        return _refuse("cmd() must name the outcome as OUTCOMEVAR.")
    rhs = words[2:]
    if any(w.lower() in ("if", "in") or "[" in w for w in rhs):
        return _refuse("if / in / weights inside cmd() are not translated.")
    strata = (opts.get("strata") or "").strip() or None
    cluster = (opts.get("cluster") or "").strip() or None
    if strata and not re.fullmatch(_NAME, strata):
        return _refuse("strata() must be one variable.")
    if cluster and not re.fullmatch(_NAME, cluster):
        return _refuse("cluster() must be one variable.")
    factors = [w for w in rhs if "." in w]
    expected = [f"i.{strata}"] if strata else []
    if factors != expected:
        return _refuse(
            "sp.westfall_young includes stratum effects exactly when strata() "
            f"is given; the model must contain {expected or 'no factor terms'}, "
            f"found {factors or 'none'}."
        )
    covariates = [w for w in rhs if "." not in w and w != family]
    if family not in rhs or not all(re.fullmatch(_NAME, w) for w in covariates):
        return _refuse(
            "familyp() must be a regressor of cmd(); covariates must be variables."
        )
    se = _se_spec(reg_opts)
    if se["kind"] == "other":
        return _refuse(
            "sp.westfall_young uses classical, HC1 (robust) or CR1 (cluster) "
            f"t-statistics; cmd() options {se['var']!r} are not translated."
        )
    if se["kind"] == "classical" and cluster:
        return _refuse("with cluster() the regression must cluster on it too.")
    if se["kind"] == "cluster" and se["var"] != cluster:
        return _refuse("cluster() must name the variable the regression clusters on.")
    if se["kind"] == "robust" and cluster:
        return _refuse("with cluster() the regression must cluster on it too.")
    args: Dict[str, Any] = {"y": outcomes, "treat": family}
    if covariates:
        args["covariates"] = covariates
    if strata:
        args["strata"] = strata
    if cluster:
        args["cluster"] = cluster
    if se["kind"] == "classical":
        args["robust"] = False
    reps = opts.get("reps")
    try:
        args["n_perms"] = int(str(reps).strip()) if reps is not None else 100
    except ValueError:
        return _refuse("reps() must be an integer.")
    if opts.get("seed") is not None:
        try:
            args["seed"] = int(str(opts.get("seed")).strip())
        except ValueError:
            return _refuse("seed() must be an integer.")
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    notes: List[str] = []
    if "step" in opts:
        notes.append(
            "step (single-step p-values) has no counterpart; the step-down "
            "p-values are column p_wy."
        )
    return _emit(
        "westfall_young",
        args,
        f"sp.westfall_young(df, {kw})",
        notes,
        semantics=[
            "Column p_wy is wyoung's pwyoung and p_analytic its p. When the "
            "design admits no more than n_perms re-randomizations, "
            "sp.westfall_young enumerates them and its p-values are exact; "
            "wyoung samples reps() of them, so the two agree up to Monte Carlo "
            "error. wyoung's default reps(100) is carried over; use 10,000+."
        ],
    )


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {"wyoung": _h_wyoung}

#: cmd() holds a whole Stata command; the handler reads it itself.
LITERAL = frozenset({_h_wyoung})
