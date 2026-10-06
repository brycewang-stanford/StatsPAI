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
from ._stata_expr import StataExprError
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


def _h_xthtaylor(cmd: StataCommand) -> Dict[str, Any]:
    """``xthtaylor y x..., endog(varlist) [vce(robust)]`` -> ``sp.xthtaylor``.

    Stata takes the panel variable from ``xtset``; a single line carries it
    as ``i()``, which ``sp.stata`` appends from the declaration."""
    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None or not xs:
        return _emit_error(
            "xthtaylor requires an outcome and regressors", command="xthtaylor"
        )
    raw = cmd.options.get("endog")
    if not raw:
        return _emit_error(
            "xthtaylor requires endog()", command="xthtaylor", suggestions=[]
        )
    amacurdy = "amacurdy" in cmd.options
    period = cmd.options.get("t")
    if amacurdy and not period and cmd.options.get("i"):
        return _emit_error(
            "xthtaylor, amacurdy needs the time variable of `xtset id time`",
            command="xthtaylor",
            suggestions=[],
        )
    # constant(), varying: Stata's hints about which regressors vary; the
    # data decide that here
    cmd.options.pop("constant", None)
    cmd.options.pop("varying", None)
    endog = [w[2:] if w.startswith("i.") else w for w in str(raw).split()]
    endog = [re.sub(r"^C\((\w+)\)$", r"\1", w) for w in endog]
    unit = cmd.options.get("i") or "<panel_id>"
    args: Dict[str, Any] = {
        "formula": _build_formula(y, xs),
        "id": None if unit == "<panel_id>" else unit,
        "endog": endog,
    }
    if amacurdy:
        args["method"] = "amacurdy"
        args["time"] = period or "<panel_time>"
    vce = str(cmd.options.get("vce") or "").strip()
    words = vce.split()
    if vce.lower() == "robust" or "robust" in cmd.options:
        args["vce"] = "robust"
    elif len(words) == 2 and words[0].lower() == "cluster":
        args["cluster"] = words[1]
    elif vce.lower() not in ("", "conventional"):
        return _emit_error(
            f"xthtaylor: vce({vce}) is not translated", command="xthtaylor",
            suggestions=[],
        )  # fmt: skip
    notes = []
    if unit == "<panel_id>":
        notes.append(
            "Stata's `xtset id [t]` set the panel id; replace <panel_id> "
            "with your unit-id column."
        )
    shown = [repr(args["formula"]), "data=df", f"id={unit!r}"] + [
        f"{k}={v!r}" for k, v in args.items() if k not in ("formula", "id")
    ]
    return _emit("xthtaylor", args, f"sp.xthtaylor({', '.join(shown)})", notes)


_GMM_WINDOW = re.compile(r"^l[a-z]*\(\s*(\d+)(?:\s+(\d+|\.))?\s*\)$", re.I)


def _h_xtdpd(cmd: StataCommand) -> Dict[str, Any]:
    """``xtdpd L(0/p).y x..., dgmmiv() lgmmiv() [iv()] [twostep]
    [vce(robust)]`` -> ``sp.xtdpdsys``.

    ``sp.stata`` has already written ``L(0/2).y`` as ``y y_L1 y_L2``. The
    variables of ``dgmmiv()`` are instrumented GMM-style, the ones of
    ``iv()`` are their own instruments, and ``i.<time>`` in both places is
    ``time_dummies=True``. Stata's weight for the one-step fit (``h=2``)
    and its use of each ``iv()`` variable in both equations are written
    out, because the defaults of ``sp.xtdpdsys`` are those of ``xtdpdsys``.
    """

    def refuse(what: str) -> Dict[str, Any]:
        return _emit_error(f"xtdpd: {what}", command="xtdpd", suggestions=[])

    if not cmd.varlist:
        return refuse("an outcome variable is required")
    y, rest = cmd.varlist[0], list(cmd.varlist[1:])
    for name in ("div", "liv", "hascons", "fodeviation", "noconstant"):
        if name in cmd.options:
            return refuse(f"{name} is not translated")

    def split(raw: Any) -> "tuple[List[str], str]":
        head, _, tail = str(raw or "").partition(",")
        return head.split(), tail.strip()

    gmm_vars, window = split(cmd.options.get("dgmmiv"))
    if y not in gmm_vars:
        return refuse(f"dgmmiv() has to name the outcome {y!r}")
    gmm_lags: List[Optional[int]] = [2, None]
    if window:
        m = _GMM_WINDOW.match(window)
        if m is None:
            return refuse(f"dgmmiv(, {window}) is not translated")
        gmm_lags = [int(m.group(1)), None]
        if m.group(2) not in (None, "."):
            gmm_lags[1] = int(m.group(2))
    if "lgmmiv" not in cmd.options:
        return refuse(
            "without lgmmiv() this is difference GMM with a constant from "
            "the level equation, which is not translated; xtabond is"
        )
    level_vars, level_lag = split(cmd.options.get("lgmmiv"))
    if sorted(level_vars) != sorted(gmm_vars):
        return refuse("lgmmiv() and dgmmiv() have to name the same variables")
    if level_lag and re.sub(r"\s", "", level_lag.lower()) not in ("lag(1)", "l(1)"):
        return refuse(f"lgmmiv(, {level_lag}) is not translated")
    standard, standard_opts = split(cmd.options.get("iv"))
    if standard_opts:
        return refuse(f"iv(, {standard_opts}) is not translated")

    def factor(word: str) -> Optional[str]:
        m = re.match(r"^(?:i\.(\w+)|C\((\w+)\))$", word)
        return (m.group(1) or m.group(2)) if m else None

    time = cmd.options.get("t")
    own_lags: List[int] = []
    exog: List[str] = []
    endog: List[str] = []
    dummies = False
    for word in rest:
        m = re.match(r"^(\w+?)_L(\d+)$", word)
        base, lag = (m.group(1), int(m.group(2))) if m else (word, 0)
        if factor(word) is not None:
            if time is not None and factor(word) != time:
                return refuse(f"{word} is not the time variable of xtset")
            dummies = True
        elif base == y and lag:
            own_lags.append(lag)
        elif base in gmm_vars:
            endog.append(f"L{lag}.{base}" if lag else base)
        else:
            exog.append(word)
    if not own_lags or sorted(own_lags) != list(range(1, len(own_lags) + 1)):
        return refuse(
            "the lags of the outcome have to be written L(0/p).y after "
            "`xtset id time`"
        )
    iv_dummies = [w for w in standard if factor(w) is not None]
    iv_plain = [w for w in standard if factor(w) is None]
    if sorted(iv_plain) != sorted(exog) or bool(iv_dummies) != dummies:
        return refuse(
            "a regressor outside dgmmiv() has to be in iv(), and iv() may "
            "hold nothing else"
        )
    vce = str(cmd.options.get("vce") or "gmm").strip().lower()
    if vce not in ("gmm", "robust"):
        return refuse(f"vce({vce}) is not translated")
    unit = cmd.options.get("i") or "<panel_id>"
    args: Dict[str, Any] = {
        "y": y,
        "id": None if unit == "<panel_id>" else unit,
        "lags": len(own_lags),
        "gmm_lags": tuple(gmm_lags),
    }
    if time is not None:
        args["time"] = time
    if exog:
        args["x"] = exog
    if endog:
        args["endogenous"] = endog
        args["endogenous_lags"] = tuple(gmm_lags)
    if dummies:
        args["time_dummies"] = True
    args["twostep"] = "twostep" in cmd.options
    args["robust"] = vce == "robust"
    args["h"] = 2
    args["iv_equation"] = "both"
    notes = []
    if unit == "<panel_id>":
        notes.append(
            "Stata's `xtset id [t]` set the panel id; replace <panel_id> "
            "with your unit-id column."
        )
    shown = ["data=df", f"y={y!r}", f"id={unit!r}"] + [
        f"{k}={v!r}" for k, v in args.items() if k not in ("y", "id")
    ]
    return _emit("xtdpdsys", args, f"sp.xtdpdsys({', '.join(shown)})", notes)


def _h_threshold(cmd: StataCommand) -> Dict[str, Any]:
    """``threshold y [x], threshvar(q) [regionvars(z)] [trim(#)]
    [vce(robust)]`` -> ``sp.threshold``.

    Stata's default covariance is the classical one and its ``vce(robust)``
    has no small-sample factor; both are written out. Stata prints the
    coefficients of each region, ``sp.threshold`` the lower region and the
    change above the threshold (``model_info['regimes']`` has both)."""

    def refuse(what: str) -> Dict[str, Any]:
        return _emit_error(f"threshold: {what}", command="threshold", suggestions=[])

    y, xs = _split_varlist_y_x(cmd.varlist)
    if y is None:
        return refuse("an outcome variable is required")
    q = str(cmd.options.get("threshvar") or "").strip()
    if not q or len(q.split()) != 1:
        return refuse("threshvar() has to name one variable")
    number = str(cmd.options.get("nthresholds") or "1").strip()
    if number != "1":
        return refuse(f"nthresholds({number}): only one threshold is translated")
    for name in ("optthresh", "consinvariant", "noconstant", "ssrs"):
        if name in cmd.options:
            return refuse(f"{name} is not translated")
    regime = str(cmd.options.get("regionvars") or "").split()
    args: Dict[str, Any] = {
        "formula": _build_formula(y, list(xs) + [r for r in regime if r not in xs]),
        "threshold": q,
        "regime": regime,
    }
    trim = cmd.options.get("trim")
    if trim is not None:
        try:
            args["trim"] = float(trim) / 100.0
        except (TypeError, ValueError):
            return refuse(f"trim({trim}) is not a number")
    vce = str(cmd.options.get("vce") or "oim").strip().lower()
    if vce not in ("oim", "robust"):
        return refuse(f"vce({vce}) is not translated")
    args["vce"] = "hc0" if vce == "robust" else "ols"
    shown = [repr(args["formula"]), "data=df"] + [
        f"{k}={v!r}" for k, v in args.items() if k != "formula"
    ]
    return _emit("threshold", args, f"sp.threshold({', '.join(shown)})", [])


_ROTATE = re.compile(r"\s*rotate\b\s*(?:,\s*(.*))?$", re.I)


def rotate_line(session: Any, line: str) -> Optional[bool]:
    """``rotate [, varimax | promax[(#)]] [normalize]`` after ``factor``:
    the last result is replaced by its rotated copy. ``None`` for any other
    line; ``StataExprError`` when the rotation cannot be done."""
    m = _ROTATE.match(line)
    if m is None:
        return None
    last = getattr(session, "last", None)
    if not hasattr(last, "rotate") or not hasattr(last, "uniqueness"):
        raise StataExprError("rotate has to follow factor")
    method, power, normalize = "varimax", 3.0, False
    for word in re.findall(r"[a-z]+(?:\([^)]*\))?", (m.group(1) or "").lower()):
        name, _, argument = word.partition("(")
        if name in ("varimax", "orthogonal"):
            method = "varimax"
        elif name == "promax":
            method = "promax"
            if argument.rstrip(")").strip():
                power = float(argument.rstrip(")"))
        elif name in ("normalize", "kaiser"):
            normalize = True
        elif name not in ("blanks", "noblanks", "format"):
            raise StataExprError(f"rotate, {name} is not implemented")
    rotated = last.rotate(method, normalize=normalize, power=power)
    session.output = rotated
    session.last = rotated
    return True


HANDLERS = {
    "threshold": _h_threshold,
    "xtdpd": _h_xtdpd,
    "xthtaylor": _h_xthtaylor,
    "cnsreg": _h_cnsreg,
    "nl": _h_nl,
    "pca": _h_pca,
    "factor": _h_factor,
}
#: handlers whose argument is an expression, not a varlist
EXPRESSION = frozenset({_h_nl})
