"""Translation of ``sensemakr`` (Cinelli, Ferwerda and Hazlett's Stata
command for omitted-variable-bias sensitivity) to ``sp.sensemakr``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._stata import _emit, _emit_error, _expand_abbreviations
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS"]


def _numbers(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    return [float(tok) for tok in str(raw).split()]


def _h_sensemakr(cmd: StataCommand) -> Dict[str, Any]:
    """``sensemakr y d x1 x2 ..., treat(d) [benchmark(x1) gbenchmark(x2 x3)
    gname(label) kd(#...) ky(#...) alpha(#)]`` -> ``sp.sensemakr``.

    The regressors are listed with the treatment among them; ``controls``
    is the rest. ``benchmark()`` gives one row per variable,
    ``gbenchmark()`` one row for the group as a whole, under ``gname()``.
    """
    opts = cmd.options
    treat = (opts.get("treat") or "").split()
    if len(cmd.varlist) < 2 or len(treat) != 1:
        return _emit_error(
            "sensemakr needs `sensemakr depvar covariates, treat(var)` "
            "with one treatment variable.",
            command="sensemakr",
        )
    y, treat_name = cmd.varlist[0], treat[0]
    controls = [v for v in cmd.varlist[1:] if v != treat_name]
    args: Dict[str, Any] = {"y": y, "treat": treat_name, "controls": controls}

    columns = getattr(cmd, "columns", None)
    groups: Dict[str, List[str]] = {}
    for option in ("benchmark", "gbenchmark"):
        raw = opts.get(option)
        if not raw:
            continue
        err, names = _expand_abbreviations(raw.split(), columns)
        if err is not None:
            return _emit_error(f"sensemakr {option}(): {err}", command="sensemakr")
        # the treatment is not a benchmark for itself
        names = [n for n in names if n != treat_name]
        if option == "benchmark":
            groups.update({n: [n] for n in names})
        else:
            groups[(opts.get("gname") or "group").strip()] = names
    opts.get("gname")
    if groups:
        single = all(members == [label] for label, members in groups.items())
        args["benchmark"] = list(groups) if single else groups
    try:
        kd, ky = _numbers(opts.get("kd")), _numbers(opts.get("ky"))
        alpha = opts.get("alpha")
        if alpha is not None:
            args["alpha"] = float(alpha)
    except ValueError:
        return _emit_error(
            "sensemakr: kd(), ky() and alpha() take plain numbers.",
            command="sensemakr",
        )
    if kd is not None:
        args["kd"] = kd[0] if len(kd) == 1 else kd
    if ky is not None:
        args["ky"] = ky[0] if len(ky) == 1 else ky
    pairs = ["data=df"] + [f"{k}={v!r}" for k, v in args.items()]
    out = _emit("sensemakr", args, f"sp.sensemakr({', '.join(pairs)})", [])
    out["untranslated_options"] = []
    return out


HANDLERS = {"sensemakr": _h_sensemakr}
