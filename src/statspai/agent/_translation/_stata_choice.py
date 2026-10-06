"""``cmset``, ``cmclogit`` and ``nlogit`` in an ``sp.stata`` session.

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
    "robust": 1, "altwise": 4, "nolog": 3, "scalealternative": 5,
    "correlation": 3, "stddev": 3, "intpoints": 4, "intmethod": 4,
    "random": 4,
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


_NLOGITGEN = re.compile(
    r"\s*nlogitgen\s+(\w+)\s*=\s*(\w+)\s*\((.*)\)\s*(?:,.*)?$", re.I
)
_TAU = re.compile(r"^\s*\[/?\s*(\w+)\s*\]\s*(\w+?)_tau\s*=\s*([-+.\deE]+)\s*$")


def _levels_of(session: "StataSession", var: str, words: List[str]) -> List[Any]:
    """Alternatives written as value labels or as numbers."""
    data = session.data
    assert data is not None
    present = sorted(data[var].dropna().unique())
    by_label = {text: code for code, text in _labels(session, var).items()}
    out: List[Any] = []
    for word in words:
        if word in by_label:
            code = by_label[word]
        elif re.fullmatch(r"-?\d+(\.\d+)?", word):
            code = float(word)
        else:
            raise StataExprError(f"{word!r} is not a value or a label of {var}")
        match = [lv for lv in present if float(lv) == float(code)]
        if not match:
            raise StataExprError(f"{word!r} is not an alternative in {var}")
        out.append(match[0])
    return out


def _nlogitgen(session: "StataSession", m: "re.Match[str]") -> bool:
    """``nlogitgen type = alt(Nest1: a | b, Nest2: c | d)``: the nests of a
    two-level tree, kept for the ``nlogit`` that follows, and the variable
    that numbers them."""
    new, alt, body = m.group(1), m.group(2), m.group(3)
    data = session.data
    if data is None or alt not in data.columns:
        raise StataExprError(f"nlogitgen: {alt} is not a variable")
    nests: Dict[str, List[Any]] = {}
    for k, branch in enumerate(body.split(","), start=1):
        name, sep, members = branch.partition(":")
        if not sep:
            name, members = f"{new}{k}", branch
        words = [w.strip() for w in members.split("|") if w.strip()]
        if not words:
            raise StataExprError(f"nlogitgen: branch {k} is empty")
        nests[name.strip()] = _levels_of(session, alt, words)
    listed = [lv for items in nests.values() for lv in items]
    if len(set(listed)) != len(listed):
        raise StataExprError("nlogitgen: an alternative is in two branches")
    code = {lv: k for k, items in enumerate(nests.values(), start=1) for lv in items}
    steps = getattr(session, "_steps", None)
    assert steps is not None
    steps.data = data.assign(**{new: data[alt].map(code)})
    session.stored.setdefault("nlogit_trees", {})[new] = (alt, nests)
    return False


def _nlogit(session: "StataSession", line: str) -> bool:
    """``nlogit y x || type: || alt:, case(id) [base()] [constraints()]``
    -> ``sp.nlogit``. Two levels; regressors enter at the bottom level; a
    constraint may fix a dissimilarity parameter."""
    import statspai as sp

    from ._stata_run import _qualified

    parts = [part.strip() for part in line.split("||")]
    if len(parts) != 3:
        raise StataExprError(
            "nlogit: only two-level trees `y x || nestvar: || altvar:, case()` "
            "are implemented"
        )
    nest_var, _, nest_rhs = parts[1].partition(":")
    bottom, _, options_text = parts[2].partition(",")
    alt, _, alt_rhs = bottom.partition(":")
    nest_var, alt = nest_var.strip(), alt.strip()
    if nest_rhs.strip() or alt_rhs.strip():
        raise StataExprError(
            "nlogit: level-specific regressors after the colon are not implemented"
        )
    trees = session.stored.get("nlogit_trees") or {}
    if nest_var not in trees or trees[nest_var][0] != alt:
        raise StataExprError(
            f"nlogit: `nlogitgen {nest_var} = {alt}(...)` has to come first"
        )
    nests = trees[nest_var][1]
    try:
        cmd = _parse(parts[0] + (", " + options_text if options_text.strip() else ""))
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None
    data = session.data
    if data is None or not cmd.varlist:
        raise StataExprError("nlogit needs data and an outcome variable")
    y, xs = cmd.varlist[0], list(cmd.varlist[1:])
    options = dict(cmd.options)
    case = options.pop("case", None)
    base_text = options.pop("base", None)
    numbers = options.pop("constraints", None)
    constant = "noconstant" not in options
    options.pop("noconstant", None)
    options.pop("nolog", None)
    vce = str(options.pop("vce", "") or "").strip()
    robust = options.pop("robust", "absent") != "absent"
    if options or case is None:
        raise StataExprError(
            "nlogit: case() is required"
            if case is None
            else f"nlogit: option(s) {sorted(options)} are not implemented"
        )
    kind: Optional[str] = None
    cluster: Optional[str] = None
    words = vce.split()
    if robust or vce.lower() == "robust":
        kind = "robust"
    elif len(words) == 2 and words[0].lower().startswith("cl"):
        kind, cluster = "cluster", words[1]
    elif vce.lower() not in ("", "oim"):
        raise StataExprError(f"nlogit: vce({vce}) is not implemented")
    fixed: Dict[str, float] = {}
    for word in str(numbers or "").split():
        text = session.constraints.get(int(word)) if word.isdigit() else None
        tau = _TAU.match(text or "")
        if tau is None or tau.group(1) != nest_var or tau.group(2) not in nests:
            raise StataExprError(
                f"nlogit: constraint {word} ({text!r}) is not of the form "
                f"[/{nest_var}]<nest>_tau = #"
            )
        fixed[tau.group(2)] = float(tau.group(3))
    if cmd.if_cond or cmd.in_range:
        data = _qualified(line.split("||")[0], data, session.stored)
    base = None
    if base_text:
        base = _levels_of(session, alt, [str(base_text).strip()])[0]
    result = sp.nlogit(
        data, y=y, x=xs, chid=str(case).strip(), alt=alt, nests=nests,
        constants=constant, fixed_lambda=fixed or None, base=base,
        vce=kind, cluster=cluster,
    )  # fmt: skip
    labels = _labels(session, alt)
    info = getattr(result, "model_info", None)
    if isinstance(info, dict):
        info["alternative_labels"] = {
            lv: labels.get(lv, labels.get(int(lv), f"{lv:g}"))
            for lv in info.get("alternatives", [])
        }
    session.stored.pop("cm_fit", None)
    session.output = result
    session.last, session.last_data = result, data
    session._last_call = {"tool": "nlogit", "arguments": {"y": y, "x": xs}}
    session._store_estimates(result)
    return True


def _cmmprobit(session: "StataSession", line: str) -> bool:
    """``cmmprobit y x, casevars() [basealternative() scalealternative()
    correlation() stddev()]`` -> ``sp.mprobit``. Stata simulates the choice
    probabilities with ``intpoints()`` points; ``sp.mprobit`` integrates
    them, so the option has nothing to set and the estimates agree up to
    Stata's simulation error."""
    import statspai as sp

    from ._stata_run import _qualified

    declared = session.stored.get("cm")
    if not declared or declared[1] is None:
        raise StataExprError(
            "cmmprobit needs the case and alternative variables: put `cmset "
            "caseid altvar` before it"
        )
    case, alt = declared
    try:
        cmd = _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None
    data = session.data
    if data is None or not cmd.varlist:
        raise StataExprError("cmmprobit needs data and an outcome variable")
    y, alt_vars = cmd.varlist[0], list(cmd.varlist[1:])
    options = dict(cmd.options)
    base_text = _option(options, "basealternative")
    scale_text = _option(options, "scalealternative")
    case_vars = (_option(options, "casevars") or "").split()
    constant = _option(options, "noconstant") is None
    correlation = (_option(options, "correlation") or "unstructured").strip().lower()
    stddev = (_option(options, "stddev") or "heteroskedastic").strip().lower()
    for name in ("intpoints", "intmethod", "nolog"):
        _option(options, name)
    structures = {"unstructured": "unstructured", "independent": "independent"}
    spreads = {"heteroskedastic": "heteroskedastic", "homoskedastic": "homoskedastic"}
    kind = [v for k, v in structures.items() if k.startswith(correlation)]
    spread = [v for k, v in spreads.items() if k.startswith(stddev)]
    if options or len(kind) != 1 or len(spread) != 1:
        raise StataExprError(
            f"cmmprobit: option(s) {sorted(options)} are not implemented"
            if options
            else f"cmmprobit: correlation({correlation}) stddev({stddev}) is "
            "not implemented"
        )
    if cmd.if_cond or cmd.in_range:
        data = _qualified(line, data, session.stored)
    labels = _labels(session, alt)
    levels = sorted(data[alt].dropna().unique())
    name_of = {lv: labels.get(lv, labels.get(int(lv), f"{lv:g}")) for lv in levels}
    if base_text:
        base = _levels_of(session, alt, [base_text.strip()])[0]
    else:
        # Stata's default: the alternative chosen most often
        base = data.loc[data[y] == 1, alt].value_counts().index[0]
    scale = _levels_of(session, alt, [scale_text.strip()])[0] if scale_text else None
    try:
        result = sp.mprobit(
            data, y=y, x=alt_vars, case_vars=case_vars, chid=case, alt=alt,
            base=base, scale=scale, correlation=kind[0], stddev=spread[0],
            constants=constant,
        )  # fmt: skip
    except ValueError as exc:
        raise StataExprError(str(exc).split("\n")[0]) from None

    def relabel(name: str) -> str:
        head, sep, tail = str(name).partition(":")
        for lv in levels:
            if sep and head == str(lv):
                return f"{name_of[lv]}:{tail}"
            if sep and tail == str(lv):
                return f"{head}:{name_of[lv]}"
        return str(name)

    names = [relabel(n) for n in result.params.index]
    result.params.index = names
    result.std_errors.index = names
    result.data_info["var_names"] = names
    info = result.model_info
    for key in ("covariance", "correlation"):
        table = info[key]
        shown = [
            next((name_of[lv] for lv in levels if str(lv) == str(c)), str(c))
            for c in table.columns
        ]
        info[key] = table.set_axis(shown, axis=0).set_axis(shown, axis=1)
    info["base_alternative"] = name_of[base]
    info["alternatives"] = [name_of[lv] for lv in levels]
    session.stored["cm_fit"] = {
        "result": result, "levels": levels, "names": name_of,
        "alt_vars": alt_vars, "effect": result._choice_effect,
    }  # fmt: skip
    session.output = result
    session.last, session.last_data = result, data
    session._last_call = {"tool": "mprobit", "arguments": {"y": y, "x": alt_vars}}
    session._store_estimates(result)
    session.stored["e"]["N_case"] = float(data[case].nunique())
    return True


def _cmmixlogit(session: "StataSession", line: str) -> bool:
    """``cmmixlogit y x, random(z) casevars() [basealternative()]`` ->
    ``sp.mixlogit`` with normal random coefficients on ``z``. Both programs
    simulate the likelihood, with different point sets, so the estimates
    agree to about three digits."""
    import statspai as sp

    from ._stata_run import _qualified

    declared = session.stored.get("cm")
    if not declared or declared[1] is None:
        raise StataExprError(
            "cmmixlogit needs the case and alternative variables: put `cmset "
            "caseid altvar` before it"
        )
    case, alt = declared
    try:
        cmd = _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None
    data = session.data
    if data is None or not cmd.varlist:
        raise StataExprError("cmmixlogit needs data and an outcome variable")
    y, alt_vars = cmd.varlist[0], list(cmd.varlist[1:])
    options = dict(cmd.options)
    base_text = _option(options, "basealternative")
    case_vars = (_option(options, "casevars") or "").split()
    random_text = _option(options, "random") or ""
    constant = _option(options, "noconstant") is None
    for name in ("intpoints", "intmethod", "nolog"):
        _option(options, name)
    names, _, shape = random_text.partition(",")
    random_vars = names.split()
    if options or not random_vars or shape.strip().lower() not in ("", "normal"):
        raise StataExprError(
            f"cmmixlogit: option(s) {sorted(options)} are not implemented"
            if options
            else "cmmixlogit: random() has to name regressors with normal "
            "coefficients"
        )
    if cmd.if_cond or cmd.in_range:
        data = _qualified(line, data, session.stored)
    needed = [y, case, alt] + alt_vars + random_vars + case_vars
    frame = data[list(dict.fromkeys(needed))].dropna().copy()
    frame = frame.sort_values([case, alt])
    labels = _labels(session, alt)
    levels = sorted(frame[alt].unique())
    name_of = {lv: labels.get(lv, labels.get(int(lv), f"{lv:g}")) for lv in levels}
    if base_text:
        base = _levels_of(session, alt, [base_text.strip()])[0]
    else:
        base = frame.loc[frame[y] == 1, alt].value_counts().index[0]
    columns: List[str] = list(alt_vars)
    for level in levels:
        if level == base:
            continue
        at = (frame[alt] == level).to_numpy(dtype=float)
        for var in case_vars:
            frame[f"{name_of[level]}:{var}"] = frame[var].to_numpy(dtype=float) * at
            columns.append(f"{name_of[level]}:{var}")
        if constant:
            frame[f"{name_of[level]}:_cons"] = at
            columns.append(f"{name_of[level]}:_cons")
    result = sp.mixlogit(
        frame, y=y, chid=case, x_fixed=columns, x_random=random_vars, alt=alt,
        robust=False,
    )  # fmt: skip
    info = getattr(result, "model_info", None)
    if isinstance(info, dict):
        info["base_alternative"] = name_of[base]
        info["alternatives"] = [name_of[lv] for lv in levels]

    codes = np.unique(frame[case].to_numpy(), return_inverse=True)[1]
    n_cases = int(codes.max()) + 1
    Xf = frame[columns].to_numpy(dtype=float)
    Xr = frame[random_vars].to_numpy(dtype=float)
    order = [str(n) for n in result.params.index]
    fixed_at = [order.index(c) for c in columns]
    mean_at = [order.index(f"mean_{v}") for v in random_vars]
    sd_at = [order.index(f"sd_{v}") for v in random_vars]
    if len(random_vars) > 2:
        nodes, weights = None, None
    else:
        z, w = np.polynomial.hermite_e.hermegauss(24)
        w = w / w.sum()
        if len(random_vars) == 1:
            nodes, weights = z[:, None], w
        else:
            nodes = np.stack(np.meshgrid(z, z, indexing="ij"), -1).reshape(-1, 2)
            weights = np.outer(w, w).ravel()

    def effect(variable: str, outcome: Any, target: Any) -> "tuple[float, float]":
        if nodes is None or weights is None:
            raise StataExprError(
                "margins after cmmixlogit with more than two random "
                "coefficients is not implemented"
            )
        at_out = (frame[alt] == outcome).to_numpy()
        at_tar = (frame[alt] == target).to_numpy()
        own = 1.0 if outcome == target else 0.0

        def average(theta: np.ndarray) -> float:
            base_xb = Xf @ theta[fixed_at]
            total = 0.0
            for node, weight in zip(nodes, weights):
                coef = theta[mean_at] + np.abs(theta[sd_at]) * node
                xb = base_xb + Xr @ coef
                centre = np.bincount(codes, weights=xb) / np.bincount(codes)
                e = np.exp(xb - centre[codes])
                p = e / np.bincount(codes, weights=e)[codes]
                p_out = np.bincount(codes, weights=p * at_out, minlength=n_cases)
                p_tar = np.bincount(codes, weights=p * at_tar, minlength=n_cases)
                slope = (
                    coef[random_vars.index(variable)]
                    if variable in random_vars
                    else theta[order.index(variable)]
                )
                total += weight * float(np.mean(slope * p_out * (own - p_tar)))
            return total

        theta = np.asarray(result.params, dtype=float)
        held = result.data_info.get("var_cov")
        cov = np.asarray(held if held is not None else result.vcov(), dtype=float)
        g = np.zeros(theta.size)
        for j in range(theta.size):
            h = 1e-6 * max(abs(theta[j]), 1.0)
            up, down = theta.copy(), theta.copy()
            up[j] += h
            down[j] -= h
            g[j] = (average(up) - average(down)) / (2.0 * h)
        return average(theta), float(np.sqrt(g @ cov @ g))

    session.stored["cm_fit"] = {
        "result": result, "levels": levels, "names": name_of,
        "alt_vars": alt_vars + random_vars, "effect": effect,
    }  # fmt: skip
    session.output = result
    session.last, session.last_data = result, frame
    session._last_call = {"tool": "mixlogit", "arguments": {"y": y, "x": columns}}
    session._store_estimates(result)
    session.stored["e"]["N_case"] = float(n_cases)
    return True


def _cm_estat(session: "StataSession", line: str) -> Optional[bool]:
    """``estat covariance`` / ``estat correlation`` after ``cmmprobit``."""
    m = re.match(r"\s*estat\s+(cov|cor)[a-z]*\s*(?:,.*)?$", line, re.I)
    fit = session.stored.get("cm_fit")
    if m is None or fit is None or session.last is not fit["result"]:
        return None
    info = getattr(fit["result"], "model_info", {})
    key = "covariance" if m.group(1).lower() == "cov" else "correlation"
    if key not in info:
        return None
    session.output = info[key]
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
    if "effect" in fit:
        estimate, se = fit["effect"](var, outcome, target)
        session.output = {
            "variable": var,
            "outcome": fit["names"][outcome],
            "alternative": fit["names"][target],
            "dydx": estimate,
            "se": se,
            "z": estimate / se if se > 0 else float("nan"),
        }
        return True
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
    if re.match(r"\s*cmmprobit\b", line):
        return _cmmprobit(session, line)
    if re.match(r"\s*cmmixlogit\b", line):
        return _cmmixlogit(session, line)
    estat = _cm_estat(session, line)
    if estat is not None:
        return estat
    m = _NLOGITGEN.match(line)
    if m:
        return _nlogitgen(session, m)
    if re.match(r"\s*nlogit\s", line) and "||" in line:
        return _nlogit(session, line)
    if re.match(r"\s*(?:cm[a-z]+|nlogit|asclogit|asmprobit)\b", line):
        # another choice model, which is not fitted here: `margins` after
        # it must not fall back on an earlier cmclogit
        session.stored.pop("cm_fit", None)
    return None
