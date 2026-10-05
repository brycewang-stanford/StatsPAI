"""Translations of ``cnsreg``, ``nl``, ``pca`` and ``factor``.

``cnsreg`` refers to constraints by number; the numbers are defined by
earlier ``constraint define`` lines, which ``sp.stata`` keeps and writes
into the option before the line reaches the handler (``constraints(x1 + x2
= 1 | x3 = 0)``). A single translated line that still holds a number is
reported as needing them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._stata import _build_formula, _emit, _emit_error, _split_varlist_y_x
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "EXPRESSION", "constraint_line", "constraints_written_out"]

_DEFINE = re.compile(r"\s*constraint\s+(?:define\s+)?(\d+)\s+(.+?)\s*$", re.I)
_DROP = re.compile(r"\s*constraint\s+drop\s+(.+?)\s*$", re.I)
_OPTION = re.compile(
    r"\b(c(?:o(?:n(?:s(?:t(?:r(?:a(?:i(?:n(?:t(?:s)?)?)?)?)?)?)?)?)?)?)\(([\d\s/-]+)\)"
)


def constraint_line(store: Dict[int, str], line: str) -> bool:
    """Run ``constraint define # ...`` / ``constraint drop ...`` against
    ``store``. False when ``line`` is neither."""
    m = _DEFINE.match(line)
    if m and not m.group(2).lower().startswith("drop"):
        store[int(m.group(1))] = m.group(2)
        return True
    m = _DROP.match(line)
    if m:
        if m.group(1).strip() == "_all":
            store.clear()
        else:
            for word in m.group(1).split():
                if word.isdigit():
                    store.pop(int(word), None)
        return True
    return False


def _numbers(text: str) -> List[int]:
    out: List[int] = []
    for word in text.split():
        lo, sep, hi = word.partition("-" if "-" in word else "/")
        if sep:
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(word))
    return out


def constraints_written_out(store: Dict[int, str], line: str) -> str:
    """``constraints(1 2)`` with the text of constraints 1 and 2. Raises
    ``KeyError`` for a number that was never defined."""
    if not re.match(r"\s*cnsreg\b", line):
        return line

    def write(m: "re.Match[str]") -> str:
        return "constraints(" + " | ".join(store[n] for n in _numbers(m.group(2))) + ")"

    head, comma, tail = line.partition(",")
    return head + comma + _OPTION.sub(write, tail)


def _vce(cmd: StataCommand, args: Dict[str, Any], allowed: tuple) -> Optional[str]:
    """Read ``robust`` / ``vce()`` / ``cluster()`` into ``args``; the text
    of a covariance type that has no counterpart otherwise."""
    opts = cmd.options
    vce = str(opts.get("vce") or "").strip()
    cluster = opts.get("cluster")
    words = vce.split()
    if words and words[0].lower() == "cluster" and len(words) == 2:
        cluster = words[1]
        vce = ""
    if cluster:
        args["cluster"] = str(cluster).strip()
    elif "robust" in opts or vce.lower() == "robust":
        args["vce"] = "robust"
    elif vce.lower() in allowed:
        args["vce"] = vce.lower()
    elif vce.lower() not in ("", "ols", "gnr"):
        return vce
    return None


def _call(name: str, lead: List[str], args: Dict[str, Any], skip: tuple) -> str:
    shown = lead + [f"{k}={v!r}" for k, v in args.items() if k not in skip]
    return f"sp.{name}({', '.join(shown)})"


def _h_cnsreg(cmd: StataCommand) -> Dict[str, Any]:
    """``cnsreg y x, constraints(...)`` -> ``sp.cnsreg``."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return _emit_error("cnsreg requires an outcome variable", command="cnsreg")
    formula = _build_formula(y, xs)
    if "noconstant" in cmd.options:
        formula += " - 1"
    raw = cmd.options.get("constraints")
    if not raw:
        return _emit_error(
            "cnsreg requires constraints()", command="cnsreg", suggestions=[]
        )
    if re.fullmatch(r"[\d\s/-]+", str(raw)):
        return _emit_error(
            f"cnsreg: constraints({raw}) refers to `constraint define` lines "
            "that are not part of this command; sp.stata reads them when "
            "they come first",
            command="cnsreg",
            suggestions=[],
        )
    constraints = [part.strip() for part in str(raw).split("|") if part.strip()]
    args: Dict[str, Any] = {"formula": formula, "constraints": constraints}
    bad = _vce(cmd, args, ())
    if bad is not None:
        return _emit_error(
            f"cnsreg: vce({bad}) is not translated", command="cnsreg", suggestions=[]
        )
    code = _call("cnsreg", [repr(formula), "data=df"], args, ("formula",))
    return _emit("cnsreg", args, code)


def _h_nl(cmd: StataCommand) -> Dict[str, Any]:
    """``nl (y = <expression with {parameters}>), initial(a 1 b 2)`` ->
    ``sp.nls``."""
    text = " ".join(cmd.varlist).strip()
    if not (text.startswith("(") and text.endswith(")")) or "=" not in text:
        return _emit_error(
            "nl: only the substitutable-expression form `nl (y = ...)` is "
            "translated, not function-evaluator programs",
            command="nl",
            suggestions=[],
        )
    left, _, right = text[1:-1].partition("=")
    y = left.strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", y):
        return _emit_error(
            f"nl: the outcome {y!r} is not a variable name", command="nl"
        )
    if re.search(r"\{[^{}]*:", right):
        return _emit_error(
            "nl: linear-combination parameters {name: varlist} are not "
            "translated; write the terms out",
            command="nl",
            suggestions=[],
        )
    formula = f"{y} ~ {right.strip()}"
    args: Dict[str, Any] = {"formula": formula}
    raw = cmd.options.get("initial")
    if raw:
        words = str(raw).split()
        try:
            start = {words[i]: float(words[i + 1]) for i in range(0, len(words), 2)}
        except (IndexError, ValueError):
            return _emit_error(
                f"nl: initial({raw}) is not a list of name-value pairs",
                command="nl",
                suggestions=[],
            )
        args["start"] = start
    else:
        # Stata starts a parameter without an initial value at zero
        names = re.findall(r"\{\s*([A-Za-z_]\w*)\s*\}", right)
        args["start"] = {n: 0.0 for n in dict.fromkeys(names)}
    bad = _vce(cmd, args, ("hc2", "hc3"))
    if bad is not None:
        return _emit_error(
            f"nl: vce({bad}) is not translated", command="nl", suggestions=[]
        )
    code = _call("nls", [repr(formula), "data=df"], args, ("formula",))
    return _emit(
        "nls",
        args,
        code,
        semantics=[
            "A sum of squares with several local minima can send the two "
            "programs to different ones; compare diagnostics['Residual SS'] "
            "with Stata's residual SS."
        ],
    )


def _count(cmd: StataCommand, name: str) -> Any:
    raw = cmd.options.get(name)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return _emit_error(
            f"{cmd.command}: {name}({raw}) is not an integer",
            command=cmd.command,
            suggestions=[],
        )


def _threshold(cmd: StataCommand) -> Any:
    raw = cmd.options.get("mineigen")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return _emit_error(
            f"{cmd.command}: mineigen({raw}) is not a number",
            command=cmd.command,
            suggestions=[],
        )


def _h_pca(cmd: StataCommand) -> Dict[str, Any]:
    """``pca varlist [, components(#) mineigen(#) covariance]`` -> ``sp.pca``."""
    if len(cmd.varlist) < 2:
        return _emit_error("pca needs at least two variables", command="pca")
    args: Dict[str, Any] = {"variables": list(cmd.varlist)}
    keep = _count(cmd, "components")
    floor = _threshold(cmd)
    for value in (keep, floor):
        if isinstance(value, dict):
            return value
    if keep is not None:
        args["n_components"] = keep
    if floor is not None:
        args["min_eigenvalue"] = floor
    if "covariance" in cmd.options:
        args["covariance"] = True
    cmd.options.pop("correlation", None)
    code = _call("pca", ["df", repr(args["variables"])], args, ("variables",))
    return _emit("pca", args, code)


def _h_factor(cmd: StataCommand) -> Dict[str, Any]:
    """``factor varlist [, pf | pcf | ipf | ml factors(#) mineigen(#)]`` ->
    ``sp.factor``."""
    if len(cmd.varlist) < 2:
        return _emit_error("factor needs at least two variables", command="factor")
    args: Dict[str, Any] = {"variables": list(cmd.varlist)}
    chosen = [m for m in ("pf", "pcf", "ipf", "ml") if m in cmd.options]
    if len(chosen) > 1:
        return _emit_error(
            f"factor: {' and '.join(chosen)} cannot be combined", command="factor"
        )
    if chosen:
        args["method"] = chosen[0]
    keep = _count(cmd, "factors")
    floor = _threshold(cmd)
    for value in (keep, floor):
        if isinstance(value, dict):
            return value
    if keep is not None:
        args["n_factors"] = keep
    if floor is not None:
        args["min_eigenvalue"] = floor
    notes: List[str] = []
    if args.get("method") == "ml" and keep is None:
        return _emit_error(
            "factor, ml without factors(): Stata fits the largest number of "
            "factors the data identify; pass factors(#)",
            command="factor",
            suggestions=[],
        )
    if args.get("method") in ("ml", "ipf"):
        notes.append(
            "Stata stops iterating at a looser tolerance; loadings agree to "
            "about four decimals unless Stata is run with tight tolerances."
        )
    code = _call("factor", ["df", repr(args["variables"])], args, ("variables",))
    return _emit("factor", args, code, notes or None)


HANDLERS = {
    "cnsreg": _h_cnsreg,
    "nl": _h_nl,
    "pca": _h_pca,
    "factor": _h_factor,
}
#: handlers whose argument is an expression, not a varlist
EXPRESSION = frozenset({_h_nl})
