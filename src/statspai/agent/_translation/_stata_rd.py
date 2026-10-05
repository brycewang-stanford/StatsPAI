"""Translations of the local-randomization and multi-cutoff RD commands.

``rdrandinf`` and ``rdwinselect`` (the ``rdlocrand`` package) and ``rdmc``
(``rdmulti``) keep their option names in StatsPAI, with three renames:
``cutoff()`` is ``c=``, ``reps()`` is ``n_perms=`` and ``level()`` of
``rdwinselect`` is ``alpha=``. Options that are not listed here are left
unread and so reported as untranslated.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._stata import _emit, _emit_error
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS"]

_SEED_NOTE = (
    "seed() fixes StatsPAI's random draws, not Stata's: randomization "
    "p-values agree up to simulation error, not digit for digit."
)
_STATISTICS = {"diffmeans", "ttest", "ksmirnov", "ranksum", "all"}
_KERNELS = {"uniform", "triangular", "epan"}
_VCE = {"hc1", "hc2", "hc3"}
_VCE_NOTE = (
    "p() > 0 without vce(): HC3, the default of rdlocrand 3.0. Releases up "
    "to 2.0 used HC2; add vce(hc2) to reproduce a log from one of those."
)


def _float_numlist(text: str) -> Optional[List[float]]:
    """A Stata numlist of reals: ``-1 0 1.5`` or ``-20(0.10)20``."""
    out: List[float] = []
    for piece in text.split():
        m = re.fullmatch(r"([-+.\d]+)\(([-+.\d]+)\)([-+.\d]+)", piece)
        try:
            if m is None:
                out.append(float(piece))
                continue
            lo, step, hi = (float(g) for g in m.groups())
        except ValueError:
            return None
        if step <= 0 or hi < lo:
            return None
        n = int(round((hi - lo) / step))
        out.extend(round(lo + k * step, 10) for k in range(n + 1))
    return out or None


def _call(fn: str, args: Dict[str, Any]) -> str:
    kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"sp.{fn}(data=df, {kw})"


def _number(
    opts: Dict[str, Optional[str]],
    name: str,
    args: Dict[str, Any],
    lost: List[str],
    *,
    key: Optional[str] = None,
    integer: bool = False,
) -> None:
    """Carry a numeric option over, or record that its value did not parse."""
    raw = opts.get(name)
    if raw is None:
        return
    try:
        args[key or name] = int(raw) if integer else float(raw)
    except ValueError:
        lost.append(name)


def _choice(
    opts: Dict[str, Optional[str]],
    name: str,
    allowed: set,
    args: Dict[str, Any],
    lost: List[str],
) -> None:
    raw = opts.get(name)
    if raw is None:
        return
    value = raw.strip().lower()
    if value in allowed:
        args[name] = value
    else:
        lost.append(name)


def _h_rdrandinf(cmd: StataCommand) -> Dict[str, Any]:
    """``rdrandinf y x, wl() wr() [cutoff() statistic() p() kernel() vce() fuzzy()
    nulltau() d() dscale() ci() bernoulli() reps() seed() evall() evalr()]``
    -> ``sp.rdrandinf``."""
    if len(cmd.varlist) != 2:
        return _emit_error(
            "rdrandinf requires an outcome and a running variable: "
            "`rdrandinf y x, wl(<left>) wr(<right>)`",
            command="rdrandinf",
        )
    opts = cmd.options
    if opts.get("wl") is None or opts.get("wr") is None:
        return _emit_error(
            "rdrandinf without wl() and wr() selects the window from "
            "covariates(); that step is sp.rdwinselect. Translate the "
            "window selection first and pass its window as wl() / wr().",
            command="rdrandinf",
        )
    y, x = cmd.varlist
    args: Dict[str, Any] = {"y": y, "x": x}
    lost: List[str] = []
    notes: List[str] = []
    _number(opts, "cutoff", args, lost, key="c")
    _number(opts, "wl", args, lost)
    _number(opts, "wr", args, lost)
    _choice(opts, "statistic", _STATISTICS, args, lost)
    _number(opts, "p", args, lost, integer=True)
    _choice(opts, "kernel", _KERNELS, args, lost)
    _choice(opts, "vce", _VCE, args, lost)
    if args.get("p", 0) > 0 and "vce" not in args and "vce" not in lost:
        notes.append(_VCE_NOTE)
    for name in ("nulltau", "d", "dscale", "evall", "evalr", "interfci"):
        _number(opts, name, args, lost)
    _number(opts, "reps", args, lost, key="n_perms", integer=True)
    if opts.get("seed") is not None:
        _number(opts, "seed", args, lost, integer=True)
        notes.append(_SEED_NOTE)
    if opts.get("bernoulli") is not None:
        args["bernoulli"] = (opts.get("bernoulli") or "").strip()
    if opts.get("fuzzy") is not None:
        parts = (opts.get("fuzzy") or "").split()
        stat = parts[1].lower() if len(parts) == 2 else "ar"
        if len(parts) not in (1, 2) or stat not in ("ar", "itt", "tsls"):
            lost.append("fuzzy")
        else:
            args["fuzzy"] = parts[0]
            args["fuzzy_stat"] = "tsls" if stat == "tsls" else "itt"
    if opts.get("ci") is not None:
        values = _float_numlist(opts.get("ci") or "")
        if values is None or not 0 < values[0] < 1:
            lost.append("ci")
        else:
            args["alpha"] = values[0]
            if len(values) > 2:
                args["ci"] = values[1:]
            elif len(values) == 2:
                lost.append("ci")
            else:
                notes.append(
                    "ci() without a grid: sp.rdrandinf tests 201 effects "
                    "over the estimate +/- 5 standard errors; Stata's "
                    "default grid differs."
                )
    else:
        # Stata computes no interval unless asked.
        args["ci"] = False
    out = _emit("rdrandinf", args, _call("rdrandinf", args), notes)
    out["untranslated_options"] = lost
    return out


def _h_rdwinselect(cmd: StataCommand) -> Dict[str, Any]:
    """``rdwinselect x [covariates], [cutoff() obsmin() wmin() wobs() wstep()
    nwindows() statistic() p() kernel() approx level() reps() seed()
    wasymmetric dropmissing]`` -> ``sp.rdwinselect``."""
    if not cmd.varlist:
        return _emit_error(
            "rdwinselect requires a running variable", command="rdwinselect"
        )
    opts = cmd.options
    args: Dict[str, Any] = {"x": cmd.varlist[0]}
    if len(cmd.varlist) > 1:
        args["covs"] = list(cmd.varlist[1:])
    lost: List[str] = []
    notes: List[str] = []
    _number(opts, "cutoff", args, lost, key="c")
    _number(opts, "obsmin", args, lost, integer=True)
    _number(opts, "wmin", args, lost)
    _number(opts, "wobs", args, lost, integer=True)
    _number(opts, "wstep", args, lost)
    _number(opts, "nwindows", args, lost, integer=True)
    _choice(opts, "statistic", (_STATISTICS - {"all"}) | {"hotelling"}, args, lost)
    _number(opts, "p", args, lost, integer=True)
    _choice(opts, "kernel", _KERNELS, args, lost)
    _choice(opts, "vce", _VCE, args, lost)
    if args.get("p", 0) > 0 and "vce" not in args and "vce" not in lost:
        notes.append(_VCE_NOTE)
    _number(opts, "level", args, lost, key="alpha")
    _number(opts, "reps", args, lost, key="n_perms", integer=True)
    for flag in ("approx", "wasymmetric", "dropmissing", "wmasspoints"):
        if flag in opts:
            args[flag] = True
    if opts.get("seed") is not None:
        _number(opts, "seed", args, lost, integer=True)
        if "approx" not in opts:
            notes.append(_SEED_NOTE)
    if "wmin" not in args:
        notes.append(
            "Without wmin(), the first window is the smallest with obsmin "
            "observations on each side, as in rdlocrand 1.0 and 3.0; releases "
            "1.1 and 2.0 start one observation short on the left, so their "
            "default sequence can differ."
        )
    out = _emit("rdwinselect", args, _call("rdwinselect", args), notes)
    out["untranslated_options"] = lost
    return out


def _h_rdmc(cmd: StataCommand) -> Dict[str, Any]:
    """``rdmc y x, cvar(cutoff_var)`` -> ``sp.rdmc(cutoff_var=)``."""
    if len(cmd.varlist) != 2:
        return _emit_error(
            "rdmc requires an outcome and a running variable: "
            "`rdmc y x, cvar(<cutoff variable>)`",
            command="rdmc",
        )
    cvar = (cmd.options.get("cvar") or "").strip()
    if not cvar:
        return _emit_error(
            "rdmc needs `cvar(<variable holding each unit's cutoff>)`.",
            command="rdmc",
        )
    y, x = cmd.varlist
    args: Dict[str, Any] = {"y": y, "x": x, "cutoff_var": cvar}
    return _emit("rdmc", args, _call("rdmc", args))


#: Marks a translation that needs values read from the data; ``sp.stata``
#: fills them in (see ``_stata_run._boundary_points``).
BOUNDARY_POINTS_NOTE = (
    "cvar() (and range()) name variables whose first rows hold the cutoffs. "
    "sp.stata reads them from the data; in a hand translation pass the "
    "values as cutoff1=[...] (and cutoff2=[...], ranges=[...])."
)


def _h_rdms(cmd: StataCommand) -> Dict[str, Any]:
    """``rdms y x1 x2 zvar, cvar(c1 c2) [xnorm(v)]`` and ``rdms y x,
    cvar(c) [range(lo hi)]`` -> ``sp.rdms``.

    In Stata the cutoffs are the leading values of the ``cvar()``
    variables (and the ranges those of the ``range()`` variables). A
    one-line translation has no data, so the values arrive through
    ``cutoff1()`` / ``cutoff2()`` / ``range1()`` / ``range2()``, which
    ``sp.stata`` appends after reading them.
    """
    opts = cmd.options
    cvar = (opts.get("cvar") or "").split()
    two_scores = len(cmd.varlist) == 4 and len(cvar) == 2
    one_score = len(cmd.varlist) == 2 and len(cvar) == 1
    if not (two_scores or one_score):
        return _emit_error(
            "rdms: expected `rdms y x1 x2 treat, cvar(c1 c2)` (two scores) or "
            "`rdms y x, cvar(c)` (one score, cumulative cutoffs).",
            command="rdms",
        )
    c1 = _float_numlist(opts.get("cutoff1") or "")
    if c1 is None:
        return _emit_error(BOUNDARY_POINTS_NOTE, command="rdms")
    args: Dict[str, Any] = {"y": cmd.varlist[0], "x1": cmd.varlist[1]}
    if two_scores:
        c2 = _float_numlist(opts.get("cutoff2") or "")
        if c2 is None or len(c1) != len(c2):
            return _emit_error(BOUNDARY_POINTS_NOTE, command="rdms")
        args.update(x2=cmd.varlist[2], treat=cmd.varlist[3], cutoff1=c1, cutoff2=c2)
    else:
        args["cutoff1"] = c1
        if opts.get("range") is not None:
            lo = _float_numlist(opts.get("range1") or "")
            hi = _float_numlist(opts.get("range2") or "")
            if lo is None or hi is None or not len(lo) == len(hi) == len(c1):
                return _emit_error(BOUNDARY_POINTS_NOTE, command="rdms")
            args["ranges"] = [(a, b) for a, b in zip(lo, hi)]
    if opts.get("xnorm") is not None:
        args["xnorm"] = (opts.get("xnorm") or "").strip()
    return _emit("rdms", args, _call("rdms", args))


def _h_rdmcplot(cmd: StataCommand) -> Dict[str, Any]:
    """``rdmcplot y x, cvar(c) [pvar() nbinsvar() nbinsrightvar()
    binselectvar() hvar()]`` -> ``sp.rdmcplot``.

    The ``*var()`` options name variables whose leading values are the
    per-cutoff settings, in the order of the sorted cutoffs. ``sp.stata``
    reads them and appends ``pvec()`` / ``nbinsvec()`` / ``nbinsrightvec()``
    / ``binselectvec()`` / ``hvec()``; without them a ``*var()`` option is
    reported as untranslated.
    """
    if len(cmd.varlist) != 2:
        return _emit_error(
            "rdmcplot requires an outcome and a running variable: "
            "`rdmcplot y x, cvar(<cutoff variable>)`",
            command="rdmcplot",
        )
    opts = cmd.options
    cvar = (opts.get("cvar") or "").strip()
    if not cvar:
        return _emit_error(
            "rdmcplot needs `cvar(<variable holding each unit's cutoff>)`.",
            command="rdmcplot",
        )
    args: Dict[str, Any] = {
        "y": cmd.varlist[0],
        "x": cmd.varlist[1],
        "cutoff_var": cvar,
    }
    lost: List[str] = []

    def _vec(var_opt: str, vec_opt: str) -> Optional[List[float]]:
        if opts.get(var_opt) is None:
            return None
        values = _float_numlist(opts.get(vec_opt) or "")
        if values is None:
            lost.append(var_opt)
        return values

    pvec = _vec("pvar", "pvec")
    if pvec is not None:
        args["p"] = [int(v) for v in pvec]
    left, right = _vec("nbinsvar", "nbinsvec"), _vec("nbinsrightvar", "nbinsrightvec")
    if left is not None and right is not None and len(left) == len(right):
        args["nbins"] = [(int(a), int(b)) for a, b in zip(left, right)]
    elif left is not None and opts.get("nbinsrightvar") is None:
        args["nbins"] = [int(a) for a in left]
    hvec = _vec("hvar", "hvec")
    if hvec is not None:
        args["h"] = hvec
    if opts.get("binselectvar") is not None:
        names = (opts.get("binselectvec") or "").split()
        if names:
            args["binselect"] = names
        else:
            lost.append("binselectvar")
    if opts.get("ci") is not None:
        try:
            args["ci_level"] = float(opts.get("ci") or "") / 100.0
            args["hide_ci"] = False
        except ValueError:
            lost.append("ci")
    out = _emit("rdmcplot", args, _call("rdmcplot", args))
    out["untranslated_options"] = lost
    return out


HANDLERS = {
    "rdrandinf": _h_rdrandinf,
    "rdwinselect": _h_rdwinselect,
    "rdmc": _h_rdmc,
    "rdms": _h_rdms,
    "rdmcplot": _h_rdmcplot,
}
