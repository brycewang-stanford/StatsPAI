"""``cmset`` and ``cmclogit`` in an ``sp.stata`` session.

Stata's choice-model commands read long data, one row per case and
alternative, declared by ``cmset caseid altvar``. ``cmclogit`` is
McFadden's conditional logit with alternative-specific regressors (one
coefficient each) and case-specific ones (one coefficient per alternative
but the base, plus a constant per alternative). That is a conditional logit
on a design with the case-specific regressors interacted with alternative
indicators, which is what is built and passed to ``sp.clogit`` here.

The coefficients are named ``cost`` for an alternative-specific regressor
and ``air:income`` / ``air:_cons`` for the equation of an alternative,
using the value label of the alternative when it has one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from ._stata_expr import StataExprError
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["choice_line"]

_CMSET = re.compile(r"\s*cmset\s+(\w+)(?:\s+(\w+))?\s*(?:,.*)?$", re.I)
_CMCLOGIT = re.compile(r"\s*cmclogit\b", re.I)
_OPTIONS = {
    "basealternative": 4, "casevars": 5, "noconstant": 6, "vce": 3,
    "robust": 1, "altwise": 4, "nolog": 3,
}  # fmt: skip


def _labels(session: "StataSession", var: str) -> Dict[Any, str]:
    steps = getattr(session, "_steps", None)
    name = getattr(steps, "_set_of", {}).get(var)
    table = getattr(steps, "_label_sets", {}).get(name, {}) if name else {}
    return {code: str(text) for code, text in table.items()}


def _option(options: Dict[str, Optional[str]], name: str) -> Optional[str]:
    """The value of ``name`` under any abbreviation Stata allows."""
    for key in list(options):
        if len(key) >= _OPTIONS[name] and name.startswith(key):
            value = options.pop(key)
            return "" if value is None else str(value)
    return None


def _cmclogit(session: "StataSession", line: str) -> bool:
    import statspai as sp

    from ._stata_run import _qualified

    declared = session.stored.get("cm")
    if not declared or declared[1] is None:
        raise StataExprError(
            "cmclogit needs the case and alternative variables: put `cmset "
            "caseid altvar` before it"
        )
    case, alt = declared
    try:
        cmd = _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None
    data = session.data
    if data is None or not cmd.varlist:
        raise StataExprError("cmclogit needs data and an outcome variable")
    y, alt_vars = cmd.varlist[0], list(cmd.varlist[1:])
    options = dict(cmd.options)
    base_text = _option(options, "basealternative")
    case_vars = (_option(options, "casevars") or "").split()
    constant = _option(options, "noconstant") is None
    vce = (_option(options, "vce") or "").strip().lower()
    robust = _option(options, "robust") is not None or vce == "robust"
    _option(options, "nolog")
    if options or vce not in ("", "robust", "oim"):
        raise StataExprError(
            f"cmclogit: option(s) {sorted(options) or ['vce(' + vce + ')']} "
            "are not implemented"
        )
    if cmd.if_cond or cmd.in_range:
        data = _qualified(line, data, session.stored)
    needed = [y, case, alt] + alt_vars + case_vars
    missing = [v for v in needed if v not in data.columns]
    if missing:
        raise StataExprError(f"variable(s) {missing} are not in the data")
    frame = data[list(dict.fromkeys(needed))].dropna().copy()
    # Stata drops a case in which any alternative has a missing value
    whole = frame.groupby(case)[alt].transform("size") == data.groupby(case)[
        alt
    ].transform("size").reindex(frame.index)
    frame = frame[whole]

    labels = _labels(session, alt)
    levels = sorted(frame[alt].unique())
    name_of = {lv: labels.get(lv, labels.get(int(lv), f"{lv:g}")) for lv in levels}
    if base_text is None or base_text == "":
        # Stata's default: the alternative chosen most often
        chosen = frame.loc[frame[y] == 1, alt].value_counts()
        base = chosen.index[0]
    else:
        match = [lv for lv in levels if name_of[lv] == base_text]
        if not match and re.fullmatch(r"-?\d+(\.\d+)?", base_text):
            match = [lv for lv in levels if float(lv) == float(base_text)]
        if not match:
            raise StataExprError(
                f"cmclogit: basealternative({base_text}) is not a level of {alt}"
            )
        base = match[0]

    columns: List[str] = list(alt_vars)
    for level in levels:
        if level == base:
            continue
        at = (frame[alt] == level).to_numpy(dtype=float)
        for var in case_vars:
            name = f"{name_of[level]}:{var}"
            frame[name] = frame[var].to_numpy(dtype=float) * at
            columns.append(name)
        if constant:
            name = f"{name_of[level]}:_cons"
            frame[name] = at
            columns.append(name)
    if not columns:
        raise StataExprError("cmclogit: the model has no regressor")
    result = sp.clogit(
        data=frame, y=y, x=columns, group=case,
        robust="robust" if robust else "nonrobust",
    )  # fmt: skip
    info = getattr(result, "model_info", None)
    if isinstance(info, dict):
        info["base_alternative"] = name_of[base]
        info["alternatives"] = [name_of[lv] for lv in levels]
    session.stored["cm_fit"] = {
        "result": result, "frame": frame, "columns": columns, "case": case,
        "alt": alt, "levels": levels, "names": name_of, "alt_vars": alt_vars,
    }  # fmt: skip
    session.output = result
    session.last, session.last_data = result, frame
    session._last_call = {"tool": "clogit", "arguments": {"y": y, "x": columns}}
    session._store_estimates(result)
    session.stored["e"]["N_case"] = float(np.unique(frame[case]).size)
    return True


def _level(fit: Dict[str, Any], text: str, what: str) -> Any:
    names = fit["names"]
    match = [lv for lv in fit["levels"] if names[lv] == text]
    if not match and re.fullmatch(r"-?\d+(\.\d+)?", text):
        match = [lv for lv in fit["levels"] if float(lv) == float(text)]
    if not match:
        raise StataExprError(f"margins: {what}({text}) is not an alternative")
    return match[0]


def _cm_margins(session: "StataSession", line: str) -> Optional[bool]:
    """``margins, dydx(x) outcome(a) alternative(b)`` after ``cmclogit``:
    the average effect on the probability of choosing ``a`` of a unit
    change in ``x`` for alternative ``b``, with a delta-method standard
    error. For the conditional logit the effect in one case is
    ``beta * P_a * (1 - P_a)`` when ``a`` is ``b`` and ``-beta * P_a * P_b``
    otherwise."""
    fit = session.stored.get("cm_fit")
    if fit is None or session.last is not fit["result"]:
        return None
    try:
        cmd = _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None
    options = dict(cmd.options)
    if "outcome" not in options or "alternative" not in options:
        return None
    var = str(options.pop("dydx", "") or "").strip()
    outcome = _level(fit, str(options.pop("outcome")).strip(), "outcome")
    target = _level(fit, str(options.pop("alternative")).strip(), "alternative")
    if options or cmd.varlist:
        raise StataExprError(
            f"margins after cmclogit: {sorted(options) or cmd.varlist} "
            "are not implemented"
        )
    if var not in fit["alt_vars"]:
        raise StataExprError(
            f"margins after cmclogit: dydx({var}) must be one "
            f"alternative-specific regressor ({fit['alt_vars']})"
        )
    frame, columns = fit["frame"], fit["columns"]
    X = frame[columns].to_numpy(dtype=float)
    codes = np.unique(frame[fit["case"]].to_numpy(), return_inverse=True)[1]
    n_cases = int(codes.max()) + 1
    at_outcome = (frame[fit["alt"]] == outcome).to_numpy()
    at_target = (frame[fit["alt"]] == target).to_numpy()
    position = columns.index(var)

    def effect(beta: np.ndarray) -> float:
        xb = X @ beta
        centre = np.bincount(codes, weights=xb) / np.bincount(codes)
        e = np.exp(xb - centre[codes])
        p = e / np.bincount(codes, weights=e)[codes]
        p_out = np.bincount(codes, weights=p * at_outcome, minlength=n_cases)
        p_tar = np.bincount(codes, weights=p * at_target, minlength=n_cases)
        own = 1.0 if outcome == target else 0.0
        return float(np.mean(beta[position] * p_out * (own - p_tar)))

    result = fit["result"]
    beta = np.asarray(result.params, dtype=float)
    cov = (
        np.asarray(result.data_info.get("var_cov"), dtype=float)
        if (result.data_info.get("var_cov") is not None)
        else np.asarray(result.vcov(), dtype=float)
    )
    gradient = np.zeros(beta.size)
    for j in range(beta.size):
        h = 1e-6 * max(abs(beta[j]), 1.0)
        up, down = beta.copy(), beta.copy()
        up[j] += h
        down[j] -= h
        gradient[j] = (effect(up) - effect(down)) / (2 * h)
    estimate = effect(beta)
    se = float(np.sqrt(gradient @ cov @ gradient))
    session.output = {
        "variable": var,
        "outcome": fit["names"][outcome],
        "alternative": fit["names"][target],
        "dydx": estimate,
        "se": se,
        "z": estimate / se if se > 0 else float("nan"),
    }
    return True


def choice_line(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``cmset`` / ``cmclogit`` and ``margins`` after it. ``None`` for
    any other line."""
    if re.match(r"\s*margins\b", line):
        return _cm_margins(session, line)
    m = _CMSET.match(line)
    if m:
        session.stored["cm"] = (m.group(1), m.group(2))
        return False
    if _CMCLOGIT.match(line):
        return _cmclogit(session, line)
    if re.match(r"\s*(?:cm[a-z]+|nlogit|asclogit|asmprobit)\b", line):
        # another choice model, which is not fitted here: `margins` after
        # it must not fall back on an earlier cmclogit
        session.stored.pop("cm_fit", None)
    return None
