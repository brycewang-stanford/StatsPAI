"""``gsbounds``: group sequential stopping boundaries -> ``sp.group_sequential``.

``sp.group_sequential`` implements the boundary families of Stata 18's
``gsbounds`` with its conventions (two-sided alpha split by tail, beta
spending for futility, Pampallona-Tsiatis classical futility bounds) and
agrees with it to 3e-7 on 117 designs when Stata's own tolerances are
tightened (``tests/reference_parity/test_group_sequential_stata_parity.py``).
At Stata's default ``infotol(1e-6)`` its futility bounds can sit up to 3e-5
from the exact solution, which is what ``sp.group_sequential`` returns.

``gsdesign <method>`` (sample sizes) is not translated: it combines a
``power`` calculation with the boundaries, so the call is two lines (see
:data:`GSDESIGN_GUIDANCE`).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

from ._stata import _emit, _emit_error
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "LITERAL", "GSDESIGN_GUIDANCE"]

#: (Stata boundary name, minimum abbreviation, sp name, takes a parameter)
_BOUNDS = (
    ("obfleming", 3, "obfleming", False),
    ("pocock", 3, "pocock", False),
    ("wtsiatis", 2, "wtsiatis", True),
    ("errpocock", 6, "errpocock", False),
    ("errobfleming", 6, "errobfleming", False),
    ("kdemets", 2, "kdemets", True),
    ("hsdecani", 3, "hsdecani", True),
)

#: Numerical tuning options of gsbounds; sp.group_sequential solves the
#: boundary equations to ~1e-10 and needs none of them.
_TUNING = (
    "intpointsscale",
    "infotol",
    "initscale",
    "initinfo",
    "technique",
    "iterate",
    "tolerance",
    "ltolerance",
    "nrtolerance",
)

GSDESIGN_GUIDANCE = (
    "`gsdesign` adds sample sizes to gsbounds: compute the fixed-design size "
    "first and pass it on, e.g. for `gsdesign twomeans 50 60, sd(35) "
    "nlooks(3)`: n_fix = sp.power_ttest(delta=10, sd=35, power=0.8)"
    ".params['n_exact']; sp.group_sequential(n_looks=3, n_fixed=n_fix) "
    "(N max is info_ratio * n_fix; Stata rounds it up per group)."
)


def _boundary(text: Optional[str]) -> Tuple[Optional[str], Optional[float], bool, str]:
    """``(name, parameter, binding, error)`` from ``efficacy()`` / ``futility()``."""
    if text is None:
        return None, None, False, ""
    body, _, sub = text.partition(",")
    sub = sub.strip().lower()
    binding = False
    if sub:
        if "binding".startswith(sub) and len(sub) >= 4:
            binding = True
        else:
            return None, None, False, f"unknown suboption {sub!r}"
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*(?:\(\s*([-+0-9.eE]+)\s*\))?\s*", body)
    if not m:
        return None, None, False, f"cannot read the boundary {body.strip()!r}"
    word, par = m.group(1).lower(), m.group(2)
    for full, minlen, name, takes in _BOUNDS:
        if len(word) >= minlen and full.startswith(word):
            if takes != (par is not None):
                need = "needs a parameter" if takes else "takes no parameter"
                return None, None, False, f"{full} {need}"
            return name, (float(par) if par is not None else None), binding, ""
    return None, None, False, f"unknown boundary {word!r}"


def _num(opts: Any, key: str) -> Tuple[Optional[float], bool]:
    raw = opts.get(key)
    if raw is None:
        return None, True
    try:
        return float(str(raw).strip()), True
    except ValueError:
        return None, False


def _h_gsbounds(cmd: StataCommand) -> Dict[str, Any]:
    if cmd.varlist:
        return _emit_error(
            "gsbounds takes no arguments before the comma.",
            command="gsbounds",
            suggestions=["group_sequential"],
        )
    opts = cmd.options
    args: Dict[str, Any] = {}
    eff, eff_p, _, err = _boundary(opts.get("efficacy"))
    if err:
        return _emit_error(
            f"gsbounds efficacy(): {err}.",
            command="gsbounds",
            suggestions=["group_sequential"],
        )
    fut, fut_p, binding, err = _boundary(opts.get("futility"))
    if err:
        return _emit_error(
            f"gsbounds futility(): {err}.",
            command="gsbounds",
            suggestions=["group_sequential"],
        )
    if eff:
        args["efficacy"] = eff
        if eff_p is not None:
            args["efficacy_param"] = eff_p
    if fut:
        args["futility"] = fut
        if fut_p is not None:
            args["futility_param"] = fut_p
        if binding:
            args["binding"] = True
    if opts.get("information") is not None:
        try:
            info = [float(v) for v in str(opts.get("information")).split()]
        except ValueError:
            return _emit_error(
                "gsbounds information(): only a list of numbers is translated "
                "(numlist ranges such as 1(1)4 are not).",
                command="gsbounds",
                suggestions=["group_sequential"],
            )
        args["information"] = info
    if opts.get("nlooks") is not None:
        raw = str(opts.get("nlooks")).split(",")[0].strip()
        if not raw.isdigit():
            return _emit_error(
                "gsbounds nlooks() must be an integer.",
                command="gsbounds",
                suggestions=[],
            )
        args["n_looks"] = int(raw)
    alpha, ok_a = _num(opts, "alpha")
    power, ok_p = _num(opts, "power")
    beta, ok_b = _num(opts, "beta")
    if not (ok_a and ok_p and ok_b):
        return _emit_error(
            "gsbounds alpha() / power() / beta() must be numbers.",
            command="gsbounds",
            suggestions=[],
        )
    if power is not None and beta is not None:
        return _emit_error(
            "gsbounds: give power() or beta(), not both.",
            command="gsbounds",
            suggestions=[],
        )
    if alpha is not None:
        args["alpha"] = alpha
    if power is not None:
        args["power"] = power
    elif beta is not None:
        args["power"] = 1.0 - beta
    sides = [s for s in ("upper", "lower", "onesided") if s in opts]
    if len(sides) > 1 and set(sides) != {"upper", "onesided"}:
        return _emit_error(
            "gsbounds: upper and lower cannot both be given.",
            command="gsbounds",
            suggestions=[],
        )
    if sides:
        args["side"] = "lower" if "lower" in sides else "upper"
    tuning = [t for t in _TUNING if t in opts]
    notes = []
    if tuning:
        notes.append(
            "Stata's numerical options ("
            + ", ".join(tuning)
            + ") are not needed: sp.group_sequential solves the boundary "
            "equations to about 1e-10."
        )
    pairs = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit(
        "group_sequential",
        args,
        f"sp.group_sequential({pairs})",
        notes,
        semantics=[
            "Critical values are on the z scale, as gsbounds reports them; "
            "info_ratio is Stata's r(info_ratio). At Stata's default "
            "infotol(1e-6) its futility bounds can differ from the exact "
            "solution by up to 3e-5."
        ],
    )


def _h_gsdesign(cmd: StataCommand) -> Dict[str, Any]:
    return _emit_error(
        GSDESIGN_GUIDANCE,
        command="gsdesign",
        suggestions=["group_sequential", "power_ttest"],
    )


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "gsbounds": _h_gsbounds,
    "gsdesign": _h_gsdesign,
}

#: Arguments are numbers and keywords, not variables.
LITERAL = frozenset({_h_gsbounds, _h_gsdesign})
