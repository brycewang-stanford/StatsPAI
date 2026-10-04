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
    """``rdrandinf y x, wl() wr() [cutoff() statistic() p() kernel() fuzzy()
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
    for name in ("nulltau", "d", "dscale", "evall", "evalr"):
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
    _choice(opts, "statistic", _STATISTICS - {"all"}, args, lost)
    _number(opts, "p", args, lost, integer=True)
    _choice(opts, "kernel", _KERNELS, args, lost)
    _number(opts, "level", args, lost, key="alpha")
    _number(opts, "reps", args, lost, key="n_perms", integer=True)
    for flag in ("approx", "wasymmetric", "dropmissing"):
        if flag in opts:
            args[flag] = True
    if opts.get("seed") is not None:
        _number(opts, "seed", args, lost, integer=True)
        if "approx" not in opts:
            notes.append(_SEED_NOTE)
    if "wmin" not in args:
        notes.append(
            "Without wmin(), the first window is the smallest with obsmin "
            "observations on each side; rdlocrand 2.0 starts one observation "
            "short on the left, so its default sequence can differ."
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


HANDLERS = {
    "rdrandinf": _h_rdrandinf,
    "rdwinselect": _h_rdwinselect,
    "rdmc": _h_rdmc,
}
