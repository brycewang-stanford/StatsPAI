"""Commands that act on the session itself, not on one estimation call.

:func:`statspai.from_stata` maps a command to one ``sp.*`` call. Some lines
of a do-file have no such call because what they do lives *between*
commands: they change the data in memory (``set obs``, ``clear``,
``tabulate, generate()``), name a fitted model for later (``estimates
store``), tabulate the models named so far (``estimates table``,
``esttab``), or repeat a descriptive command over groups (``bysort g:
summarize``). :class:`~._stata_run.StataSession` hands those lines to
:func:`run_session_command`.

Weights are prepared here too. A weight may be an expression
(``[aw=1/e2f]``), which is evaluated into a column; a frequency weight
(``[fw=n]``) stands for ``n`` identical rows, which is what the session
builds before the command runs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._stata_datastep import DataSteps, _generate_option, _is_tabulate, row_mask
from ._stata_expr import StataExprError, evaluate
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = [
    "run_session_command",
    "prepare_weights",
    "expand_frequency",
    "xtreg_extras",
    "absorbed_constant",
    "teffects_before",
    "teffects_after",
    "stata_percentile",
    "psmatch2_after",
]

_SET = re.compile(r"\s*set\s+(seed|obs)\s+(\S+)\s*$", re.I)
_CLEAR = re.compile(r"\s*(?:clear|drop\s+_all)\s*$", re.I)
_BY = re.compile(
    r"\s*(?:bys(?:o(?:rt?)?)?|by)\s+(?P<by>[^:,]+?)\s*(?:,\s*(?P<opts>[^:]*))?:"
    r"\s*(?P<cmd>.+)\Z",
    re.I | re.S,
)
_ESTIMATES = re.compile(
    r"\s*(?:est(?:i(?:m(?:a(?:t(?:es?)?)?)?)?)?)\s+(\w+)\s*(.*)\Z", re.S
)
_BSAMPLE = re.compile(r"\s*bsample\b(.*)\Z", re.I | re.S)
_DUPLICATES = re.compile(r"\s*duplicates\s+drop\b(.*)\Z", re.I | re.S)
_COUNT = re.compile(r"\s*count\b(.*)\Z", re.I | re.S)
_PCTILE = re.compile(r"\s*_pctile\b(.*)\Z", re.I | re.S)
_ESTSTO = re.compile(r"\s*eststo\s+([A-Za-z_]\w*)\s*$")
_ESTTAB = re.compile(r"\s*(esttab|estout)\b\s*(.*)\Z", re.S)
_WEIGHT = re.compile(
    r"\[\s*(?P<kind>aw|aweights?|pw|pweights?|fw|fweights?|iw|iweights?)\s*=\s*"
    r"(?P<exp>[^\]]+)\]",
    re.I,
)
_PLAIN = re.compile(r"[A-Za-z_]\w*\Z")
_HAUSMAN = re.compile(r"\s*hausman\s+(.+)\Z", re.S)
_FCAST = re.compile(
    r"\s*fcast\s+c(?:o(?:m(?:p(?:u(?:te?)?)?)?)?)?\s+\S+\s*,?(.*)\Z", re.S
)


# ------------------------------------------------------------------ weights
def prepare_weights(session: "StataSession", line: str) -> Tuple[str, Optional[str]]:
    """Make the weight clause of ``line`` something a call can carry.

    Returns the line to translate and, for a frequency weight, the name of
    the column holding the counts (the clause is then removed: the caller
    expands the rows). A weight *expression* is evaluated into a column and
    the clause rewritten to name it.
    """
    m = _WEIGHT.search(line)
    if m is None or session._steps is None:
        return line, None
    kind = m.group("kind").lower()[:2]
    exp = m.group("exp").strip()
    data = session._steps.data
    if _PLAIN.match(exp) and exp in data.columns:
        name = exp
    else:
        values = evaluate(exp, data, session.stored)
        if values.dtype == object:
            raise StataExprError("a weight must be numeric")
        name = _free_name(data, "_weight")
        session._steps.add_column(name, values, double=True)
        session._scratch.append(name)
    if kind == "fw":
        return (line[: m.start()] + " " + line[m.end() :]).strip(), name
    return line[: m.start()] + f"[{kind}={name}]" + line[m.end() :], None


def _free_name(data: pd.DataFrame, stem: str) -> str:
    k = 1
    while f"{stem}{k}" in data.columns:
        k += 1
    return f"{stem}{k}"


def expand_frequency(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each row repeated as many times as its frequency weight says."""
    w = data[column].to_numpy(dtype=float, na_value=np.nan)
    held = ~np.isnan(w)
    if np.any(w[held] < 0) or np.any(w[held] != np.round(w[held])):
        raise StataExprError(
            "frequency weights must be non-negative integers (Stata: "
            "`may not use noninteger frequency weights`)"
        )
    counts = np.where(held, w, 0).astype(int)
    return data.loc[data.index.repeat(counts)].reset_index(drop=True)


# ------------------------------------------------------------- the commands
def run_session_command(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``line`` if it is a session command.

    Returns ``None`` when it is not one, otherwise whether it produced
    output (``session.output``).
    """
    m = _SET.match(line)
    if m:
        return _set(session, m.group(1).lower(), m.group(2))
    if _CLEAR.match(line):
        if session._steps is None:
            session._steps = DataSteps(pd.DataFrame())
            session._steps.stored = session.stored
        session._steps.reset(0)
        session.simulated = False
        session.panel = (None, None)
        session.stored.pop("time_var", None)
        session.stored.pop("panel_var", None)
        return False
    m = _ESTSTO.match(line)
    if m:
        _store(session, m.group(1))
        return False
    m = _ESTIMATES.match(line)
    if m:
        return _estimates(session, m.group(1).lower(), m.group(2), line)
    m = _ESTTAB.match(line)
    if m:
        return _table(session, m.group(2), line, command=m.group(1).lower())
    m = _BY.match(line)
    if m:
        return _by(session, m, line)
    m = _FCAST.match(line)
    if m:
        return _fcast(session, m.group(1), line)
    m = _HAUSMAN.match(line)
    if m:
        return _hausman(session, m.group(1), line)
    if re.match(r"\s*xttest0\s*$", line):
        return _xttest0(session)
    if re.match(r"\s*tebalance\s+su", line):
        return _tebalance(session)
    if re.match(r"\s*xtoverid\s*(?:,.*)?$", line):
        return _xtoverid(session, line)
    head = line.split(None, 1)[0].rstrip(",").lower() if line.strip() else ""
    if _is_tabulate(head) or head == "tab1":
        return _tabulate(session, line)
    m = _COUNT.match(line)
    if m and session._steps is not None:
        return _count(session, m.group(1))
    m = _PCTILE.match(line)
    if m and session._steps is not None:
        return _pctile(session, m.group(1), line)
    m = _BSAMPLE.match(line)
    if m and session._steps is not None:
        return _bsample(session, m.group(1))
    m = _DUPLICATES.match(line)
    if m and session._steps is not None:
        return _duplicates_drop(session, m.group(1))
    return None


def _duplicates_drop(session: "StataSession", rest: str) -> bool:
    """``duplicates drop [varlist] [, force]``: keep the first row of each
    group of rows that agree on the variables (on all of them when none is
    named)."""
    assert session._steps is not None
    cmd = _parse("duplicates_drop " + rest)
    options = dict(cmd.options)
    force = "force" in options
    options.pop("force", None)
    if cmd.if_cond or cmd.in_range or options:
        raise StataExprError(
            "`duplicates drop` is run as `duplicates drop [varlist] [, force]`"
        )
    data = session._steps.data
    names = list(cmd.varlist)
    unknown = [v for v in names if v not in data.columns]
    if unknown:
        raise StataExprError(f"variable(s) {unknown} are not in the data")
    if names and not force:
        raise StataExprError(
            "`duplicates drop varlist` needs the `force` option, as in Stata"
        )
    keep = ~data.duplicated(subset=names or None, keep="first").to_numpy()
    session._steps.replace_data(data.loc[keep])
    return False


def _bsample(session: "StataSession", rest: str) -> bool:
    """``bsample [, cluster(g)]``: the data replaced by a bootstrap sample.

    Rows (or whole clusters) are drawn with replacement by numpy, so the
    sample is not the one Stata draws from the same seed.
    """
    assert session._steps is not None
    cmd = _parse("bsample " + rest)
    options = dict(cmd.options)
    cluster = options.pop("cluster", None)
    if cmd.varlist or cmd.if_cond or cmd.in_range or options:
        raise StataExprError("`bsample` is run as `bsample [, cluster(varname)]`")
    data = session._steps.data
    rng = session.stored.get("rng")
    if rng is None:
        rng = session.stored["rng"] = np.random.default_rng()
    if cluster is None:
        rows = rng.integers(0, len(data), size=len(data))
    else:
        if cluster not in data.columns:
            raise StataExprError(f"variable {cluster!r} is not in the data")
        codes, levels = pd.factorize(data[cluster], sort=True)
        members = [np.flatnonzero(codes == g) for g in range(len(levels))]
        drawn = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[g] for g in drawn])
    session._steps.replace_data(data.iloc[rows].reset_index(drop=True))
    session.stored["random_draws"] = True
    return False


def stata_percentile(x: np.ndarray, p: float) -> float:
    """The ``p``-th percentile as ``summarize, detail`` and ``_pctile``
    define it: with ``P = n p / 100``, the mean of the ``P``-th and
    ``(P+1)``-th order statistics when ``P`` is an integer, otherwise the
    next order statistic above ``P``."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    pos = n * p / 100.0
    k = int(np.floor(pos + 1e-12))
    if abs(pos - k) < 1e-12:
        lo = x[max(k, 1) - 1]
        hi = x[min(k + 1, n) - 1]
        return float((lo + hi) / 2.0)
    return float(x[min(k + 1, n) - 1])


def _count(session: "StataSession", rest: str) -> bool:
    """``count [if exp] [in range]``: the number of rows, left in ``r(N)``."""
    assert session._steps is not None
    data = session._steps.data
    cmd = _parse("count " + rest)
    if cmd.varlist or cmd.options:
        raise StataExprError("`count` takes only an if / in qualifier")
    n = int(row_mask(data, cmd.if_cond, cmd.in_range, session.stored).sum())
    session.stored["r"] = {"N": float(n)}
    session.output = n
    return True


def _pctile(session: "StataSession", rest: str, line: str) -> bool:
    """``_pctile x [if], percentiles(# ...)``: ``r(r1)``, ``r(r2)`` ..."""
    assert session._steps is not None
    data = session._steps.data
    cmd = _parse("_pctile " + rest)
    options = dict(cmd.options)
    spec = options.pop("percentiles", None) or options.pop("p", None)
    if len(cmd.varlist) != 1 or options:
        raise StataExprError(
            "`_pctile` is run as `_pctile var [if], percentiles(# ...)`"
        )
    name = cmd.varlist[0]
    if name not in data.columns:
        raise StataExprError(f"variable {name!r} is not in the data")
    try:
        wanted = [float(t) for t in (spec or "50").replace(",", " ").split()]
    except ValueError:
        raise StataExprError(
            f"percentiles({spec}): a list of numbers is expected"
        ) from None
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    x = data.loc[mask, name].dropna().to_numpy(dtype=float)
    values = [stata_percentile(x, q) for q in wanted]
    session.stored["r"] = {f"r{i + 1}": v for i, v in enumerate(values)}
    session.output = pd.Series(values, index=wanted, name=name)
    return True


def _set(session: "StataSession", what: str, raw: str) -> bool:
    try:
        number = int(raw)
    except ValueError:
        raise StataExprError(f"set {what} {raw}: expected an integer") from None
    if what == "seed":
        session.stored["rng"] = np.random.default_rng(number)
        return False
    if session._steps is None:
        session._steps = DataSteps(pd.DataFrame())
        session._steps.stored = session.stored
    session._steps.set_obs(number)
    return False


# ------------------------------------------------------- stored estimates
def _store(session: "StataSession", name: str) -> None:
    if session.last is None:
        raise StataExprError(
            f"`estimates store {name}`: there is no estimation result to store"
        )
    session.estimates[name] = (session.last, session.last_data, session._last_call)


def _lookup(session: "StataSession", names: List[str], line: str) -> List[Any]:
    if not names or names == ["*"] or names == ["_all"]:
        names[:] = list(session.estimates)
    missing = [n for n in names if n not in session.estimates]
    if missing or not names:
        raise StataExprError(
            f"estimation result(s) {missing or '(none)'} were not stored; "
            f"stored: {', '.join(session.estimates) or 'none'}"
        )
    return [session.estimates[n][0] for n in names]


def _estimates(session: "StataSession", sub: str, rest: str, line: str) -> bool:
    if "store".startswith(sub) and len(sub) >= 3:
        names = rest.split(",")[0].split()
        if len(names) != 1:
            raise StataExprError("expected `estimates store name`")
        _store(session, names[0])
        return False
    if "restore".startswith(sub) and len(sub) >= 3:
        names = rest.split(",")[0].split()
        if len(names) != 1 or names[0] not in session.estimates:
            raise StataExprError(f"`{line.strip()}`: no such stored result")
        session.last, session.last_data, session._last_call = session.estimates[
            names[0]
        ]
        session._store_estimates(session.last)
        return False
    if "table".startswith(sub) and len(sub) >= 1:
        return _table(session, rest, line, command="estimates table")
    if sub in ("dir", "query", "describe", "replay", "notes"):
        return False
    if sub in ("clear", "drop"):
        if sub == "clear":
            session.estimates.clear()
        else:
            for name in rest.split(",")[0].split():
                session.estimates.pop(name, None)
        return False
    raise StataExprError(f"`estimates {sub}` is not implemented")


_STAR = re.compile(r"([\d.]+)")


def _table(session: "StataSession", rest: str, line: str, *, command: str) -> bool:
    """``estimates table`` / ``esttab``: the stored models side by side."""
    import statspai as sp

    from ._stata_lexer import _parse_options, _split_options

    head, tail = _split_options(rest)
    using = re.search(r"\busing\b\s+(\S+)", head)
    if using:
        head = head[: using.start()]
        session.warn(
            f"sp.stata: `{command}` did not write {using.group(1)}; the table "
            "is returned. Call .to_word() / .to_latex() / .to_excel() on it."
        )
    names = head.split()
    results = _lookup(session, names, line)
    options = _parse_options(tail) if tail else {}
    kwargs: Dict[str, Any] = {"names": names}
    # esttab prints t statistics unless told otherwise; estimates table
    # prints coefficients alone
    shown = [k for k in ("se", "t", "p", "ci") if k in options]
    if command == "estimates table":
        kwargs.update(se="se" in shown, t="t" in shown, p="p" in shown)
        kwargs["stars"] = "star" in options
    else:
        kwargs.update(
            se="se" in shown,
            t="t" in shown or not shown,
            p="p" in shown,
            ci="ci" in shown,
            stars="nostar" not in options,
        )
    levels = [float(x) for x in _STAR.findall(str(options.get("star") or ""))]
    if levels:
        kwargs["star_levels"] = tuple(sorted(levels, reverse=True))
    elif kwargs.get("stars") and command != "estimates table":
        kwargs["star_levels"] = (0.05, 0.01, 0.001)  # esttab's default
    session.output = sp.esttab(*results, **kwargs)
    return True


# ------------------------------------------------------------- by: prefix
def _by(session: "StataSession", m: "re.Match[str]", line: str) -> Optional[bool]:
    """``bysort g: summarize ...``: the descriptive command, group by group."""
    inner = m.group("cmd").strip()
    word = inner.split(None, 1)[0].rstrip(",").lower()
    descriptive = len(word) >= 2 and "summarize".startswith(word)
    if session._steps is None:
        return None
    if not descriptive:
        # `by g: generate` / `bysort g (t): replace`: a data step by group
        spec = m.group("by")
        paren = re.search(r"\(([^)]*)\)", spec)
        order = paren.group(1).split() if paren else []
        keys = re.sub(r"\([^)]*\)", " ", spec).split()
        sorts = line.lstrip().lower().startswith("bys") or "sort" in (
            m.group("opts") or ""
        )
        if keys and session._steps.by_assign(keys, order, sorts, inner):
            return False
        return None
    keys = [k for k in m.group("by").replace("(", " ").replace(")", " ").split()]
    data = session._steps.data
    unknown = [k for k in keys if k not in data.columns]
    if unknown:
        raise StataExprError(f"by: variable(s) {unknown} are not in the data")
    pieces: Dict[Any, Any] = {}
    full = session._steps.data
    try:
        for level, rows in data.groupby(keys if len(keys) > 1 else keys[0], sort=True):
            session._steps.data = rows.drop(columns=keys)
            session.run(inner)
            pieces[level] = session.output
    finally:
        session._steps.data = full
    session.output = pd.concat(pieces, names=keys if len(keys) > 1 else [keys[0]])
    return True


# --------------------------------------------------------------- tabulate
def _tabulate(session: "StataSession", line: str) -> bool:
    """``tabulate v [, generate(stub) missing]`` and ``tabulate v1 v2``."""
    if session._steps is None:
        raise StataExprError("`tabulate` needs data")
    try:
        cmd = _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None
    options = dict(cmd.options)
    stub = _generate_option(options)
    missing = any(options.pop(k, 0) is None for k in ("missing", "miss", "m"))
    nolabel = any(
        [options.pop(k, 0) is None for k in ("nolabel", "nolabe", "nolab", "nol")]
    )
    options.pop("sort", None)
    if options:
        raise StataExprError(
            f"tabulate: option(s) {sorted(options)} are not implemented"
        )
    data = session._steps.data
    varlist = [v for v in cmd.varlist if not v.startswith("[")]
    if len(varlist) != len(cmd.varlist):
        raise StataExprError("tabulate with weights is not implemented")
    unknown = [v for v in varlist if v not in data.columns]
    if unknown or len(varlist) not in (1, 2):
        raise StataExprError(
            f"tabulate: variable(s) {unknown} are not in the data"
            if unknown
            else "tabulate takes one or two variables"
        )
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    if stub is not None:
        if len(varlist) != 1:
            raise StataExprError("tabulate, generate() takes one variable")
        session._steps.tabulate_generate(varlist[0], stub, mask)
    rows = data.loc[mask]
    if not nolabel:
        # Stata prints the value labels, in the order of the codes
        from ...output.tab import _with_value_labels

        rows = _with_value_labels(rows, varlist)
    if len(varlist) == 1:
        counts = rows[varlist[0]].value_counts(dropna=not missing).sort_index()
        if isinstance(counts.index, pd.CategoricalIndex):
            counts.index = counts.index.astype(object)
        total = float(counts.sum())
        table = pd.DataFrame(
            {
                "Freq.": counts.astype(int),
                "Percent": 100.0 * counts / total if total else np.nan,
            }
        )
        table["Cum."] = table["Percent"].cumsum()
        table.index.name = varlist[0]
        session.output = table
        session.stored["r"] = {"N": total, "r": float(len(counts))}
    else:
        a, b = (rows[v] if missing else rows[v] for v in varlist)
        if not missing:
            keep = a.notna() & b.notna()
            a, b = a[keep], b[keep]
        session.output = pd.crosstab(
            a, b, margins=True, margins_name="Total", dropna=False
        )
        session.stored["r"] = {"N": float(len(a))}
    return True


# ------------------------------------------------------------ fcast compute
def _fcast(session: "StataSession", tail: str, line: str) -> bool:
    """``fcast compute prefix, step(#)``: dynamic forecasts of the VAR in
    memory. Stata appends them to the data as new variables and rows; here
    they are the command's output, a table indexed by horizon."""
    from ._stata_lexer import _parse_options

    last = session.last
    if last is None or not hasattr(last, "forecast"):
        raise StataExprError("`fcast compute` needs a VAR (`var` / `varbasic`) first")
    options = _parse_options(tail) if tail.strip() else {}
    raw = options.pop("step", options.pop("st", "1"))
    unknown = [k for k in options if k not in ("replace",)]
    if unknown:
        raise StataExprError(
            f"fcast compute: option(s) {sorted(unknown)} are not implemented"
        )
    try:
        steps = int(str(raw).strip())
    except ValueError:
        raise StataExprError(f"fcast compute: step({raw}) is not an integer") from None
    session.output = last.forecast(steps=steps)
    return True


# ------------------------------------------------------------------ hausman
def _hausman(session: "StataSession", rest: str, line: str) -> bool:
    """``hausman consistent efficient [, constant sigmamore sigmaless]`` on
    two models named by ``estimates store``. ``.`` is the last estimates."""
    import statspai as sp

    from ._stata_lexer import _parse_options, _split_options

    head, tail = _split_options(rest)
    names = head.split()
    if len(names) == 1:
        names.append(".")
    if len(names) != 2:
        raise StataExprError("expected `hausman name-consistent name-efficient`")
    results = []
    for name in names:
        if name == ".":
            if session.last is None:
                raise StataExprError("hausman: there is no last estimation result")
            results.append(session.last)
        elif name in session.estimates:
            results.append(session.estimates[name][0])
        else:
            raise StataExprError(
                f"hausman: estimation result {name!r} was not stored; stored: "
                f"{', '.join(session.estimates) or 'none'}"
            )
    options = _parse_options(tail) if tail else {}
    known = {
        "constant": "constant", "c": "constant", "cons": "constant",
        "sigmamore": "sigmamore", "sigmaless": "sigmaless",
    }  # fmt: skip
    kwargs = {}
    for key in list(options):
        if key in known:
            kwargs[known[key]] = True
        elif key not in ("alleqs", "skipeqs", "equations", "force", "df"):
            raise StataExprError(f"hausman: option {key} is not implemented")
    for key in ("alleqs", "skipeqs", "equations", "force", "df"):
        if key in options:
            raise StataExprError(f"hausman: option {key} is not implemented")
    session.output = sp.hausman(results[0], results[1], **kwargs)
    return True


# ------------------------------------------------------------- after xtreg
def _design(data: pd.DataFrame, names: List[str]) -> Optional[pd.DataFrame]:
    """The regressors a result names, as columns: a plain variable, or the
    indicator of a factor level written ``C(g)[T.2]``. ``None`` when a name
    is neither."""
    out: Dict[str, Any] = {}
    for name in names:
        if name in data.columns:
            out[name] = data[name].to_numpy(dtype=float, na_value=np.nan)
            continue
        m = re.fullmatch(r"C\((\w+)\)\[T\.(.+)\]", name)
        if m is None or m.group(1) not in data.columns:
            return None
        col = data[m.group(1)]
        try:
            level: Any = float(m.group(2))
        except ValueError:
            level = m.group(2)
        hit = (col == level).to_numpy().astype(float)
        out[name] = np.where(col.isna().to_numpy(), np.nan, hit)
    return pd.DataFrame(out, index=data.index)


def xtreg_extras(
    session: "StataSession", call: Dict[str, Any], data: pd.DataFrame
) -> None:
    """What ``xtreg`` prints besides the slopes: ``sigma_u``, ``sigma_e``,
    ``rho``, the three R-squared and, for ``fe``, the constant. They go
    into ``e()`` / ``_b[_cons]`` and onto the result (``model_info['xt']``)
    so that ``hausman, constant`` can use them."""
    from ...diagnostics.hausman import _coef_cov
    from ...panel.xt_tools import xt_statistics

    result = session.last
    arguments = call.get("arguments") or {}
    if call.get("tool") == "feols":
        left, _, absorbed = str(arguments.get("fml") or "").partition("|")
        method, unit = "fe", absorbed.strip()
    elif call.get("tool") == "panel" and arguments.get("method") == "mle":
        e = session.stored.setdefault("e", {})
        info = getattr(result, "model_info", None) or {}
        for key in ("sigma_u", "sigma_e", "rho", "ll"):
            if key in info:
                e[key] = float(info[key])
        return
    elif call.get("tool") == "panel" and arguments.get("method") in ("re", "be"):
        left = str(arguments.get("formula") or "")
        method, unit = str(arguments["method"]), str(arguments.get("entity") or "")
    else:
        return
    outcome = left.split("~", 1)[0].strip()
    if unit not in data.columns or outcome not in data.columns:
        return
    coef, cov = _coef_cov(result)
    slopes = [n for n in coef.index if n != "_cons"]
    design = _design(data, slopes)
    if design is None or not slopes:
        return
    frame = pd.concat([data[[unit, outcome]], design], axis=1)
    stats_ = xt_statistics(
        frame,
        outcome,
        slopes,
        id=unit,
        params=coef[slopes].to_dict(),
        method=method,
        cov=cov.loc[slopes, slopes].to_numpy(),
    )
    e = session.stored.setdefault("e", {})
    for key, name in (
        ("sigma_u", "sigma_u"), ("sigma_e", "sigma_e"), ("rho", "rho"),
        ("r2_w", "r2_within"), ("r2_b", "r2_between"), ("r2_o", "r2_overall"),
        ("theta", "theta"), ("corr", "corr_u_xb"), ("N_g", "n_groups"),
    ):  # fmt: skip
        if name in stats_ and np.isfinite(stats_[name]):
            e[key] = float(stats_[name])
    if method == "fe" and "cons_se" in stats_:
        session.stored.setdefault("_b", {})["_cons"] = stats_["cons"]
        session.stored.setdefault("_se", {})["_cons"] = stats_["cons_se"]
    info = getattr(result, "model_info", None)
    if isinstance(info, dict):
        info["xt"] = stats_
        if method == "fe" and "sigma_e" in stats_:
            # the root MSE of the within regression, which `hausman,
            # sigmamore` rescales by
            info.setdefault("rmse", stats_["sigma_e"])


def absorbed_constant(
    session: "StataSession", call: Dict[str, Any], data: pd.DataFrame
) -> None:
    """The ``_cons`` that ``reghdfe`` / ``areg`` print.

    With absorbed fixed effects no constant is estimated. Stata reports the
    one that makes the fixed effects average to zero over the estimation
    sample, ``ybar - xbar'b``; its variance follows from that of the slopes,
    ``xbar' V xbar``. It goes into ``_b[_cons]`` / ``_se[_cons]`` when the
    estimation sample can be reconstructed (no weights, plain regressors).
    """
    result = session.last
    arguments = call.get("arguments") or {}
    formula = str(arguments.get("formula") or arguments.get("fml") or "")
    absorber = getattr(result, "absorber", None)
    if "~" not in formula or absorber is None or arguments.get("weights"):
        return
    outcome = formula.split("~", 1)[0].strip()
    names = [str(n) for n in result.params.index]
    parts = {n: n.split(":") for n in names}  # a:b is the product of a and b
    if outcome not in data.columns or any(
        p not in data.columns for cols in parts.values() for p in cols
    ):
        return
    used = [
        c
        for c in data.columns
        if re.search(rf"(?<![\w.]){re.escape(str(c))}(?![\w(])", formula)
    ]
    cluster = arguments.get("cluster")
    for extra in [cluster] if isinstance(cluster, str) else list(cluster or []):
        if extra in data.columns and extra not in used:
            used.append(extra)
    rows = data[used].dropna()
    keep = np.asarray(getattr(absorber, "keep_mask", []), dtype=bool)
    if keep.size != len(rows):
        return
    sample = rows.loc[keep]
    if len(sample) != int(getattr(result, "nobs", -1)) or not names:
        return
    design = np.column_stack(
        [sample[cols].to_numpy(dtype=float).prod(axis=1) for cols in parts.values()]
    )
    xbar = design.mean(axis=0)
    b = result.params.to_numpy(dtype=float)
    cov = np.asarray(result.vcov, dtype=float)
    if cov.shape != (len(names), len(names)):
        return
    variance = float(xbar @ cov @ xbar)
    kind = str(getattr(result, "se_type", "") or "").lower()
    if kind in ("iid", "classical", "nonrobust", "unadjusted", "ols"):
        # with a classical covariance the mean residual adds sigma^2 / N;
        # the robust and clustered forms give it no weight
        variance += float(getattr(result, "rmse", 0.0)) ** 2 / len(sample)
    session.stored.setdefault("_b", {})["_cons"] = float(
        sample[outcome].to_numpy(dtype=float).mean() - xbar @ b
    )
    session.stored.setdefault("_se", {})["_cons"] = float(np.sqrt(variance))


def _xttest0(session: "StataSession") -> bool:
    """``xttest0``: Breusch-Pagan LM test for random effects, after
    ``xtreg, re``."""
    last = session.last
    if last is None or not hasattr(last, "bp_lm_test"):
        raise StataExprError("`xttest0` follows `xtreg, re`")
    session.output = last.bp_lm_test()
    return True


def _xtoverid(session: "StataSession", line: str) -> bool:
    """``xtoverid`` after ``xtreg, re``: the robust test of random against
    fixed effects, clustered as the model was (on the panel by default)."""
    import statspai as sp

    call = session._last_call or {}
    arguments = call.get("arguments") or {}
    if call.get("tool") != "panel" or arguments.get("method") != "re":
        raise StataExprError("`xtoverid` follows `xtreg, re`")
    if "," in line and line.split(",", 1)[1].strip():
        raise StataExprError("xtoverid: options are not implemented")
    left, _, right = str(arguments["formula"]).partition("~")
    xs = [t.strip() for t in right.split("+") if t.strip() not in ("", "1")]
    session.output = sp.xtoverid(
        session.last_data,
        left.strip(),
        xs,
        id=str(arguments["entity"]),
        cluster=arguments.get("cluster"),
    )
    return True


# ----------------------------------------------------------- after teffects
def _teffects_options(line: str) -> Dict[str, Optional[str]]:
    from ._stata_lexer import _parse_options, _split_options
    from ._stata_options import canonicalise_options

    tail = _split_options(line)[1]
    options, _ = canonicalise_options("teffects", _parse_options(tail) if tail else {})
    return options


def teffects_before(
    session: "StataSession", line: str, call: Dict[str, Any], data: pd.DataFrame
) -> None:
    """``teffects psmatch ..., caliper(#) osample(name)``.

    Stata looks for the requested number of matches within the caliper for
    every observation, treated and control. When some have fewer it flags
    them in ``name`` and stops, so that the next line can drop them. The
    same is done here, on the propensity score the call would estimate.
    """
    import statspai as sp

    options = _teffects_options(line)
    name = (options.get("osample") or "").strip()
    arguments = call.get("arguments") or {}
    if not name or arguments.get("distance") != "propensity":
        return
    if session._steps is None or arguments.get("caliper") is None:
        return
    if name in session._steps.data.columns:
        raise StataExprError(f"osample({name}): variable {name!r} already exists")
    treat, covariates = str(arguments["treat"]), list(arguments["covariates"])
    fit = sp.logit(f"{treat} ~ " + " + ".join(covariates), data=data)
    used = data[[treat] + covariates].notna().all(axis=1).to_numpy()
    rows = data.loc[used]
    ps = np.asarray(fit.data_info["fitted_values"], dtype=float)
    if ps.size != len(rows):
        return
    d = rows[treat].to_numpy(dtype=float) != 0
    k = int(arguments.get("n_matches") or 1)
    caliper = float(arguments["caliper"])
    few = np.zeros(ps.size, dtype=bool)
    for own, other in ((d, ~d), (~d, d)):
        pool = np.sort(ps[other])
        lo = np.searchsorted(pool, ps[own] - caliper, side="left")
        hi = np.searchsorted(pool, ps[own] + caliper, side="right")
        few[own] = (hi - lo) < k
    flag = pd.Series(np.nan, index=session._steps.data.index)
    flag.loc[rows.index] = few.astype(float)
    session._steps.add_column(name, flag.to_numpy(), double=False)
    if few.any():
        raise StataExprError(
            f"{int(few.sum())} observations have fewer than {k} "
            f"propensity-score matches within caliper {caliper:g}; they are "
            f"identified in the osample() variable {name!r} (Stata stops here "
            "with r(459))"
        )


def teffects_after(
    session: "StataSession", line: str, call: Dict[str, Any], data: pd.DataFrame
) -> None:
    """``generate(stub)``: for each treated unit, the observation numbers of
    its nearest controls as ``stub1``, ``stub2`` ... (ties beyond the
    requested number are not listed)."""
    options = _teffects_options(line)
    stub = (options.get("generate") or "").strip()
    if not stub or session._steps is None:
        return
    info = getattr(session.last, "model_info", None) or {}
    matched = info.get("matched_data")
    if matched is None or len(matched) != len(data):
        raise StataExprError(
            "generate(): the fit dropped rows, so the matches cannot be "
            "written back as observation numbers"
        )
    full = session._steps.data
    position = pd.Series(np.arange(1, len(full) + 1, dtype=float), index=full.index)
    obs = position.loc[data.index].to_numpy()  # row number of each fitted row
    k = 1
    while f"_n{k}" in matched.columns:
        ids = matched[f"_n{k}"].to_numpy(dtype=float, na_value=np.nan)
        ok = ~np.isnan(ids)
        target = np.full(len(data), np.nan)
        target[ok] = obs[ids[ok].astype(int) - 1]
        column = pd.Series(np.nan, index=full.index)
        column.loc[data.index] = target
        session._steps.add_column(f"{stub}{k}", column.to_numpy(), double=True)
        k += 1


_PSMATCH2_VARS = (
    "_pscore",
    "_treated",
    "_support",
    "_weight",
    "_id",
    "_n1",
    "_nn",
    "_pdif",
)


def psmatch2_after(session: "StataSession", data: pd.DataFrame) -> None:
    """The variables ``psmatch2`` leaves in the data: ``_pscore``,
    ``_treated``, ``_support``, ``_weight``, ``_id``, ``_n1``, ``_nn`` and
    ``_pdif``, replaced if they are already there, missing on the rows the
    fit did not use."""
    if session._steps is None:
        return
    matched = getattr(session.last, "matched_data", None)
    if not isinstance(matched, pd.DataFrame):
        return
    full = session._steps.data
    if not matched.index.isin(full.index).all():
        return  # fitted on a filtered copy whose rows cannot be placed
    for name in _PSMATCH2_VARS:
        if name not in matched.columns:
            continue
        column = pd.Series(np.nan, index=full.index)
        column.loc[matched.index] = matched[name].to_numpy(dtype=float, na_value=np.nan)
        if name in session._steps.data.columns:
            session._steps._own()
            session._steps.data = session._steps.data.drop(columns=[name])
        session._steps.add_column(name, column.to_numpy(), double=True)


def _weighted_moments(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    total = float(w.sum())
    mean = float(w @ x) / total
    return mean, float(w @ (x - mean) ** 2) / (total - 1.0)


def _tebalance(session: "StataSession") -> bool:
    """``tebalance summarize``: standardised differences and variance ratios
    of the covariates, in the raw data and in the matched sample."""
    info = getattr(session.last, "model_info", None) or {}
    matched = info.get("matched_data")
    call = session._last_call or {}
    covariates = list((call.get("arguments") or {}).get("covariates") or [])
    if matched is None or not covariates or "_weight" not in matched.columns:
        raise StataExprError("`tebalance summarize` follows `teffects psmatch`")
    treated = matched["_treated"].to_numpy(dtype=float) == 1
    w = np.nan_to_num(matched["_weight"].to_numpy(dtype=float))
    rows = {}
    for name in covariates:
        x = matched[name].to_numpy(dtype=float)
        out = {}
        for label, wt, wc in (
            ("raw", np.ones(treated.sum()), np.ones((~treated).sum())),
            ("matched", w[treated], w[~treated]),
        ):
            mt, vt = _weighted_moments(x[treated], wt)
            mc, vc = _weighted_moments(x[~treated], wc)
            out[f"std_diff_{label}"] = (mt - mc) / np.sqrt((vt + vc) / 2.0)
            out[f"var_ratio_{label}"] = vt / vc if vc > 0 else np.nan
        rows[name] = out
    table = pd.DataFrame.from_dict(rows, orient="index")
    table = table[
        ["std_diff_raw", "std_diff_matched", "var_ratio_raw", "var_ratio_matched"]
    ]
    table.attrs.update(
        n_raw=int(len(matched)),
        n_treated=int(treated.sum()),
        n_control_matched=float(w[~treated].sum()),
    )
    session.output = table
    return True
