"""Data-management commands of a Stata session.

The commands a do-file uses between ``use`` and the first estimation:
``recode``, ``xtile`` / ``pctile``, ``tostring`` / ``destring``, ``order``,
``expand``, ``contract``, ``separate``, ``split``, ``sample``,
``duplicates report`` / ``tag``, ``mark`` / ``markout``, ``levelsof``,
``ds`` / ``unab``, ``assert`` / ``confirm`` / ``isid``, ``joinby``.

Each follows the command's entry in the Stata manual ([D], [P]). A form
that is not implemented raises :class:`StataExprError`, which ``sp.stata``
turns into a refusal; nothing is approximated.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._stata_datastep import _numlist, _split_options, row_mask
from ._stata_expr import StataExprError, evaluate
from ._stata_functions import stata_format
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["run_manage", "MANAGE_COMMANDS", "macro_number_list"]


def _steps(session: "StataSession") -> Any:
    if session._steps is None:
        raise StataExprError("no data in memory")
    return session._steps


def _cmd(line: str) -> Any:
    try:
        return _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None


def _flag(options: Dict[str, Any], full: str, shortest: int) -> bool:
    """Pop a bare option given by any abbreviation down to ``shortest``."""
    for key in list(options):
        if shortest <= len(key) <= len(full) and full.startswith(key):
            if options[key] is None:
                options.pop(key)
                return True
    return False


def _valued(options: Dict[str, Any], full: str, shortest: int) -> Optional[str]:
    for key in list(options):
        if shortest <= len(key) <= len(full) and full.startswith(key):
            if options[key] is not None:
                return str(options.pop(key)).strip()
    return None


def _leftover(options: Dict[str, Any], command: str) -> None:
    if options:
        raise StataExprError(
            f"{command}: option(s) {sorted(options)} are not implemented"
        )


def _mask(session: "StataSession", cmd: Any) -> np.ndarray:
    data = _steps(session).data
    return row_mask(data, cmd.if_cond, cmd.in_range, session.stored)


def _numeric(data: pd.DataFrame, name: str, what: str) -> np.ndarray:
    col = data[name]
    if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
        raise StataExprError(f"{what}: {name!r} is a string variable")
    return np.asarray(col.to_numpy(dtype=float, na_value=np.nan), dtype=float)


def _new_name(data: pd.DataFrame, name: str, what: str) -> str:
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise StataExprError(f"{what}: {name!r} is not a variable name")
    if name in data.columns:
        raise StataExprError(f"{what}: variable {name!r} already exists")
    return name


def macro_number_list(values: Any) -> str:
    """Numbers as ``levelsof`` writes them into a macro: format
    ``%18.0g``, which keeps at most 16 significant digits of a value that
    is not an integer. A ``float`` variable holding 16.1 is listed as
    ``16.10000038146973``, which is not the stored value any more: the
    loop ``if x == `level'`` then finds no row, in Stata and here."""
    out = []
    for v in values:
        v = float(v)
        if v == int(v) and abs(v) < 1e15:
            out.append(str(int(v)))
            continue
        text = repr(v)
        for digits in range(17, 0, -1):
            text = format(abs(v), f".{digits}g")
            if text.startswith("0."):
                text = text[1:]
            if len(text) <= 17:
                break
        out.append(("-" if v < 0 else "") + text)
    return " ".join(out)


# ------------------------------------------------------------------ recode
_RULE_VALUE = r"(?:-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|\.[a-z]?|min|max)"
_RULE = re.compile(
    rf"\s*(?P<lhs>.+?)\s*=\s*(?P<to>{_RULE_VALUE})\s*(?P<label>\"[^\"]*\")?\s*\Z",
    re.S,
)


def _recode_parts(text: str) -> Tuple[str, List[str], str]:
    """``varlist (rule) (rule) ... rest`` -> the three pieces."""
    first = text.find("(")
    if first < 0:
        # one rule without parentheses: `recode x 1 2 = 3`
        m = re.match(r"\s*((?:[A-Za-z_][\w*?~-]*\s+)+?)(\S.*=.*)\Z", text, re.S)
        if m is None:
            raise StataExprError("recode: expected `recode varlist (rule) ...`")
        rule, _, rest = m.group(2).partition(" if ")
        return m.group(1), [rule], (" if " + rest) if rest else ""
    varlist, pos, rules = text[:first], first, []
    while pos < len(text) and text[pos] == "(":
        close = text.find(")", pos)
        if close < 0:
            raise StataExprError("recode: unbalanced parentheses")
        rules.append(text[pos + 1 : close])
        pos = close + 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return varlist, rules, text[pos:]


def _recode_rule(rule: str, lo: float, hi: float) -> Tuple[Any, float, bool, Any]:
    """One rule -> (test on a column, the new value, extended?, label)."""
    m = _RULE.match(rule)
    if m is None:
        raise StataExprError(f"recode: cannot read the rule ({rule})")
    to = m.group("to")
    extended = bool(re.fullmatch(r"\.[a-z]", to))
    value = (
        lo
        if to == "min"
        else (
            hi
            if to == "max"
            else (np.nan if to.startswith(".") and not to[1:].isdigit() else float(to))
        )
    )
    label = m.group("label")[1:-1] if m.group("label") else None
    lhs = m.group("lhs").strip()
    low = lhs.lower()
    if low in ("else", "*"):
        return (lambda x: np.ones(x.shape, bool)), value, extended, label
    if low in ("missing", "miss", "mis"):
        return (lambda x: np.isnan(x)), value, extended, label
    if low in ("nonmissing", "nonmiss", "nonm"):
        return (lambda x: ~np.isnan(x)), value, extended, label

    def bound(tok: str) -> float:
        return lo if tok == "min" else hi if tok == "max" else float(tok)

    tests: List[Any] = []
    number = r"-?(?:\d+\.?\d*|\.\d+)|min|max"
    for tok in re.findall(rf"(?:{number})\s*(?:/|thru)\s*(?:{number})|\S+", lhs):
        ranged = re.fullmatch(rf"({number})\s*(?:/|thru)\s*({number})", tok)
        if ranged:
            a, b = bound(ranged.group(1)), bound(ranged.group(2))
            tests.append(lambda x, a=a, b=b: (x >= a) & (x <= b))
        elif tok == ".":
            tests.append(lambda x: np.isnan(x))
        elif re.fullmatch(r"\.[a-z]", tok):
            raise StataExprError(
                f"recode: the rule ({rule}) names one kind of extended "
                "missing value, and the data keep all kinds as one"
            )
        else:
            try:
                v = bound(tok)
            except ValueError:
                raise StataExprError(
                    f"recode: cannot read {tok!r} in the rule ({rule})"
                ) from None
            tests.append(lambda x, v=v: x == v)

    def test(x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.asarray(np.logical_or.reduce([t(x) for t in tests]), dtype=bool)

    return test, value, extended, label


def _recode(session: "StataSession", rest: str) -> bool:
    """``recode varlist (rule) ... [if] [in] [, generate() prefix()
    copyrest]``; a value takes the first rule that fits it."""
    steps = _steps(session)
    body, _, option_text = _split_options(rest)
    varlist_text, rules, tail = _recode_parts(body)
    cmd = _cmd("recode _x " + tail + ("," + option_text if option_text else ""))
    options = dict(cmd.options)
    generate = _valued(options, "generate", 1)
    into = _valued(options, "into", 4)
    prefix = _valued(options, "prefix", 3)
    copyrest = _flag(options, "copyrest", 4)
    _flag(options, "test", 1)
    label_name = _valued(options, "label", 1)
    _leftover(options, "recode")
    varlist = steps.expand_varlist(varlist_text.split())
    data = steps.data
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    targets = (generate or into or "").split()
    if prefix is not None:
        targets = [prefix + v for v in varlist]
    if targets and len(targets) != len(varlist):
        raise StataExprError("recode: generate() needs one new name per variable")
    for name in targets:
        _new_name(data, name, "recode")
    steps._own()
    for position, name in enumerate(varlist):
        x = _numeric(steps.data, name, "recode")
        held = x[~np.isnan(x)]
        lo, hi = (float(held.min()), float(held.max())) if held.size else (0.0, 0.0)
        out = x.copy()
        done = np.zeros(x.shape, dtype=bool)
        coded = name in steps.coded
        labels: Dict[Any, str] = {}
        for rule in rules:
            test, value, extended, label = _recode_rule(rule, lo, hi)
            hit = test(x) & ~done
            out = np.where(hit, value, out)
            done |= hit
            if bool((hit & mask)[np.isnan(x)].all()) and bool(mask.all()):
                # every missing value was given one new value: none of the
                # old kinds is left
                coded = extended and bool(hit.any())
            coded = coded or (extended and bool(hit.any()))
            if label is not None and value == value:
                labels[int(value) if value == int(value) else value] = label
        if targets:
            new = targets[position]
            source = x if copyrest else np.where(mask, x, np.nan)
            steps.data[new] = np.where(mask, out, source)
            target = new
            if labels:
                set_name = label_name or new
                steps._label_sets[set_name] = labels
                steps._set_of[new] = set_name
                steps._sync_value_labels([new])
        else:
            steps.data[name] = np.where(mask, out, x)
            target = name
        if coded:
            steps.coded.add(target)
        else:
            steps.coded.discard(target)
    return False


# ------------------------------------------------------- xtile and pctile
_NEWVAR_EXP = re.compile(r"\s*(?:(\w+)\s+)?([A-Za-z_]\w*)\s*=\s*(.+)\Z", re.S)


def _quantile_cuts(x: np.ndarray, nq: int, altdef: bool) -> List[float]:
    from ._stata_session import stata_percentile

    x = np.sort(x[~np.isnan(x)])
    if not altdef:
        return [stata_percentile(x, 100.0 * k / nq) for k in range(1, nq)]
    n, out = x.size, []
    for k in range(1, nq):
        pos = (n + 1) * k / nq
        i = int(np.floor(pos))
        if i < 1:
            out.append(float(x[0]))
        elif i >= n:
            out.append(float(x[-1]))
        else:
            out.append(float(x[i - 1] + (pos - i) * (x[i] - x[i - 1])))
    return out


def _weighted_cuts(x: np.ndarray, w: np.ndarray, nq: int) -> List[float]:
    """Quantiles with weights, as ``_pctile`` defines them."""
    from ...output.sumstats import weighted_percentile

    keep = ~np.isnan(x) & ~np.isnan(w) & (w > 0)
    return [weighted_percentile(x[keep], w[keep], 100.0 * k / nq) for k in range(1, nq)]


def _weight_of(session: "StataSession", text: str) -> Tuple[str, Optional[np.ndarray]]:
    """Split a ``[aw=exp]`` clause off ``text``; the weights it names."""
    m = re.search(r"\[\s*([a-z]+)\s*=\s*([^\]]+)\]", text, re.I)
    if m is None:
        return text, None
    w = evaluate(m.group(2), _steps(session).data, session.stored)
    if w.dtype == object:
        raise StataExprError("a weight must be numeric")
    return (text[: m.start()] + " " + text[m.end() :]).strip(), w


def _xtile(session: "StataSession", rest: str, *, pctile: bool) -> bool:
    """``xtile new = exp [if] [in] [weight], nquantiles(#)`` and
    ``pctile new = exp ..., nquantiles(#) [genp(new)]``."""
    steps = _steps(session)
    what = "pctile" if pctile else "xtile"
    body, _, option_text = _split_options(rest)
    body, weights = _weight_of(session, body)
    m = _NEWVAR_EXP.match(body)
    if m is None:
        raise StataExprError(f"{what}: expected `{what} newvar = exp, nquantiles(#)`")
    cmd = _cmd(f"{what} _x {_qualifier_tail(m.group(3))[1]} ," + option_text)
    expr = _qualifier_tail(m.group(3))[0]
    options = dict(cmd.options)
    nq = _valued(options, "nquantiles", 1)
    cutvar = _valued(options, "cutpoints", 1)
    genp = _valued(options, "genp", 1)
    altdef = _flag(options, "altdef", 1)
    _leftover(options, what)
    data = steps.data
    new = _new_name(data, m.group(2), what)
    value = evaluate(expr, data, session.stored)
    if value.dtype == object:
        raise StataExprError(f"{what}: the expression is a string")
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    x = np.where(mask, value, np.nan)
    if weights is not None:
        x = np.where(np.isnan(weights) | (weights == 0), np.nan, x)
    if cutvar is not None:
        if pctile:
            raise StataExprError("pctile takes no cutpoints()")
        cuts = sorted(set(_numeric(data, cutvar, what)[~np.isnan(data[cutvar])]))
    else:
        k = int(float(nq)) if nq is not None else 2
        if k < 2:
            raise StataExprError(f"{what}: nquantiles() must be at least 2")
        if weights is not None:
            if altdef:
                raise StataExprError(f"{what}: altdef may not be used with weights")
            cuts = _weighted_cuts(x, weights, k)
        else:
            cuts = _quantile_cuts(x, k, altdef)
    steps._own()
    if pctile:
        if len(cuts) > len(data):
            raise StataExprError("pctile: more quantiles than observations")
        column = np.full(len(data), np.nan)
        column[: len(cuts)] = cuts
        steps.data[new] = column
        if genp is not None:
            share = np.full(len(data), np.nan)
            share[: len(cuts)] = [100.0 * (i + 1) / (len(cuts) + 1)
                                  for i in range(len(cuts))]  # fmt: skip
            steps.data[_new_name(steps.data, genp, what)] = share
        return False
    group = np.full(len(data), np.nan)
    held = ~np.isnan(x)
    # category j holds the values in (cut[j-1], cut[j]]
    group[held] = np.searchsorted(np.asarray(cuts, float), x[held], side="left") + 1
    steps.data[new] = group
    return False


def _qualifier_tail(text: str) -> Tuple[str, str]:
    """``exp [if ...] [in ...]`` -> the expression and the qualifier."""
    m = re.search(r"\s(if|in)\s", " " + text + " ")
    if m is None:
        return text.strip(), ""
    cut = m.start() - 1
    return text[: max(cut, 0)].strip(), text[max(cut, 0) :].strip()


# ---------------------------------------------------- tostring / destring
def _tostring(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    cmd = _cmd("tostring " + rest)
    options = dict(cmd.options)
    generate = _valued(options, "generate", 1)
    replace = _flag(options, "replace", 7)
    force = _flag(options, "force", 5)
    fmt = _valued(options, "format", 6)
    _flag(options, "usedisplayformat", 1)
    _leftover(options, "tostring")
    varlist = steps.expand_varlist(list(cmd.varlist))
    names = generate.split() if generate else varlist
    if (generate is None) == (not replace) or len(names) != len(varlist):
        raise StataExprError("tostring: give generate(newvars) or replace")
    steps._own()
    for old, new in zip(varlist, names):
        col = steps.data[old]
        if not pd.api.types.is_numeric_dtype(col):
            continue  # already a string: Stata leaves it with a note
        x = col.to_numpy(dtype=float, na_value=np.nan)
        text = [stata_format(v, fmt or "%12.0g") for v in x]
        if not force and fmt is None:
            back = np.array([np.nan if t == "." else float(t) for t in text])
            if not np.array_equal(back, x, equal_nan=True):
                raise StataExprError(
                    f"tostring: {old!r} cannot be converted reversibly; Stata "
                    "stops here unless format() or force is given"
                )
        if generate:
            _new_name(steps.data, new, "tostring")
        steps.data[new] = np.array(text, dtype=object)
        steps._float.discard(new)
    return False


def _destring(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    cmd = _cmd("destring " + rest)
    options = dict(cmd.options)
    generate = _valued(options, "generate", 1)
    replace = _flag(options, "replace", 7)
    force = _flag(options, "force", 5)
    ignore = _valued(options, "ignore", 1)
    percent = _flag(options, "percent", 7)
    dpcomma = _flag(options, "dpcomma", 7)
    _flag(options, "float", 5)
    _leftover(options, "destring")
    varlist = steps.expand_varlist(list(cmd.varlist) or ["_all"])
    names = generate.split() if generate else varlist
    if (generate is None) == (not replace) or len(names) != len(varlist):
        raise StataExprError("destring: give generate(newvars) or replace")
    strip = ""
    if ignore is not None:
        strip = re.sub(r",\s*(?:asbytes|aschars|illegal)\s*$", "", ignore).strip()
        strip = strip[1:-1] if strip.startswith('"') and strip.endswith('"') else strip
        strip = strip.replace('" "', "").replace('"', "")
    number = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\Z")
    steps._own()
    for old, new in zip(varlist, names):
        col = steps.data[old]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            continue  # Stata: "already numeric; no replace"
        out = np.full(len(col), np.nan)
        bad = False
        for i, raw in enumerate(col.astype(object).where(col.notna(), "")):
            text = str(raw).strip()
            for ch in strip:
                text = text.replace(ch, "")
            scale = 1.0
            if percent and text.endswith("%"):
                text, scale = text[:-1], 0.01
            if dpcomma:
                text = text.replace(".", "").replace(",", ".")
            if text in ("", "."):
                continue
            if number.match(text):
                out[i] = float(text) * scale
            else:
                bad = True
        if bad and not force:
            # Stata reports "contains nonnumeric characters" and moves on
            session.warn(
                f"sp.stata: destring left {old!r} a string: it contains "
                "nonnumeric characters (use ignore() or force)."
            )
            continue
        if generate:
            _new_name(steps.data, new, "destring")
        steps.data[new] = out
    return False


# -------------------------------------------------------- order and friends
def _order(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    cmd = _cmd("order " + rest)
    options = dict(cmd.options)
    last = _flag(options, "last", 4)
    _flag(options, "first", 5)
    before = _valued(options, "before", 1)
    after = _valued(options, "after", 1)
    alpha = _flag(options, "alphabetic", 5)
    sequential = _flag(options, "sequential", 3)
    _leftover(options, "order")
    cols = [str(c) for c in steps.data.columns]
    moved = steps.expand_varlist(list(cmd.varlist) or ["_all"])
    if alpha or sequential:

        def key(name: str) -> Any:
            if sequential:
                m = re.fullmatch(r"(.*?)(\d+)", name)
                return (m.group(1), int(m.group(2))) if m else (name, -1)
            return name

        moved = sorted(moved, key=key)
    others = [c for c in cols if c not in set(moved)]
    anchor = before or after
    if anchor is not None:
        anchor = steps.expand_varlist([anchor])[0]
        if anchor in moved:
            raise StataExprError("order: the anchor variable is in the list")
        at = others.index(anchor) + (0 if before else 1)
        new = others[:at] + moved + others[at:]
    else:
        new = others + moved if last else moved + others
    steps._own()
    steps.data = steps.data[new]
    return False


def _expand(session: "StataSession", rest: str) -> bool:
    """``expand exp [if] [in] [, generate(new)]``: n copies of each row;
    the copies are added below the data, as Stata adds them."""
    steps = _steps(session)
    body, _, option_text = _split_options(rest)
    expr, tail = _qualifier_tail(body)
    cmd = _cmd(f"expand _x {tail} ," + option_text)
    options = dict(cmd.options)
    generate = _valued(options, "generate", 1)
    _leftover(options, "expand")
    data = steps.data
    n = evaluate(expr, data, session.stored)
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    copies = np.where(mask & ~np.isnan(n) & (n >= 2), np.floor(n) - 1, 0).astype(int)
    extra = data.loc[data.index.repeat(copies)]
    out = pd.concat([data, extra], ignore_index=True)
    if generate is not None:
        out[_new_name(data, generate, "expand")] = np.r_[
            np.zeros(len(data)), np.ones(len(extra))
        ]
    attrs = dict(data.attrs)
    steps.replace_data(out)
    steps.data.attrs.update(attrs)
    return False


def _duplicates(session: "StataSession", sub: str, rest: str) -> Optional[bool]:
    """``duplicates report [varlist]`` and ``duplicates tag, generate()``."""
    steps = _steps(session)
    cmd = _cmd("duplicates " + rest)
    options = dict(cmd.options)
    data = steps.data
    keys = steps.expand_varlist(list(cmd.varlist) or ["_all"])
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    size = (
        data.loc[mask]
        .groupby(keys, dropna=False, sort=False)[keys[0]]
        .transform("size")
    )
    if sub.startswith("r"):
        _leftover(options, "duplicates report")
        counts = size.value_counts().sort_index()
        table = pd.DataFrame(
            {
                "observations": counts.astype(int),
                "surplus": (counts - counts / counts.index).astype(int),
            }
        )
        table.index.name = "copies"
        session.output = table
        unique = int((counts / counts.index).sum())
        session.stored["r"] = {"N": float(mask.sum()), "unique_value": float(unique)}
        return True
    if sub.startswith("t"):
        generate = _valued(options, "generate", 1)
        _leftover(options, "duplicates tag")
        if generate is None:
            raise StataExprError("duplicates tag needs generate(newvar)")
        column = np.full(len(data), np.nan)
        column[mask] = size.to_numpy(dtype=float) - 1
        steps._own()
        steps.data[_new_name(data, generate, "duplicates tag")] = column
        return False
    return None


def _contract(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    cmd = _cmd("contract " + rest)
    options = dict(cmd.options)
    freq = _valued(options, "freq", 1) or "_freq"
    cfreq = _valued(options, "cfreq", 2)
    percent = _valued(options, "percent", 1)
    cpercent = _valued(options, "cpercent", 2)
    nomiss = _flag(options, "nomiss", 6)
    _leftover(options, "contract")
    data = steps.data
    keys = steps.expand_varlist(list(cmd.varlist))
    rows = data.loc[row_mask(data, cmd.if_cond, cmd.in_range, session.stored)]
    if nomiss:
        rows = rows.dropna(subset=keys)
    out = (
        rows.groupby(keys, dropna=False, sort=True)
        .size()
        .rename(freq)
        .reset_index()
        .sort_values(keys, kind="stable", na_position="last")
        .reset_index(drop=True)
    )
    out[freq] = out[freq].astype(float)
    if cfreq:
        out[cfreq] = out[freq].cumsum()
    if percent:
        out[percent] = 100.0 * out[freq] / out[freq].sum()
    if cpercent:
        out[cpercent] = (100.0 * out[freq] / out[freq].sum()).cumsum()
    attrs = dict(data.attrs)
    steps.replace_data(out)
    steps.data.attrs.update(attrs)
    return False


def _separate(session: "StataSession", rest: str) -> bool:
    """``separate var [if], by(byvar | exp) [generate(stub)]``: one new
    variable per value of the by-variable, named stub + value."""
    steps = _steps(session)
    cmd = _cmd("separate " + rest)
    options = dict(cmd.options)
    by = _valued(options, "by", 2)
    stub = _valued(options, "generate", 1)
    missing = _flag(options, "missing", 1)
    _flag(options, "shortlabel", 5)
    _flag(options, "veryshortlabel", 5)
    _leftover(options, "separate")
    if by is None or len(cmd.varlist) != 1:
        raise StataExprError("separate: expected `separate var, by(byvar)`")
    data = steps.data
    name = steps.expand_varlist(list(cmd.varlist))[0]
    stub = stub or name
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    source = data[name]
    if by in data.columns:
        groups = _numeric(data, by, "separate")
        levels = sorted(set(groups[mask & ~np.isnan(groups)]))
        named = [(lv, str(int(lv)) if lv == int(lv) else None) for lv in levels]
        if any(tag is None for _, tag in named):
            named = [(lv, str(i + 1)) for i, lv in enumerate(levels)]
    else:
        groups = evaluate(by, data, session.stored)
        named = [(0.0, "0"), (1.0, "1")]
        groups = np.where(np.isnan(groups), np.nan, (groups != 0).astype(float))
    if missing and np.isnan(groups[mask]).any():
        raise StataExprError("separate, missing is not implemented")
    steps._own()
    created = []
    for level, tag in named:
        new = _new_name(steps.data, stub + str(tag), "separate")
        steps.data[new] = source.where(mask & (groups == level))
        created.append(new)
    session.stored["r"] = {}
    session.stored.setdefault("r_macros", {})["varlist"] = " ".join(created)
    return False


def _split(session: "StataSession", rest: str) -> bool:
    """``split strvar [, parse(str ...) generate(stub) limit(#)
    destring]``: the pieces of a string, in stub1, stub2, ..."""
    steps = _steps(session)
    cmd = _cmd("split " + rest)
    options = dict(cmd.options)
    parse = _valued(options, "parse", 1)
    stub = _valued(options, "generate", 1)
    limit = _valued(options, "limit", 1)
    destring = _flag(options, "destring", 8)
    notrim = _flag(options, "notrim", 6)
    _leftover(options, "split")
    if len(cmd.varlist) != 1:
        raise StataExprError("split: expected one string variable")
    data = steps.data
    name = steps.expand_varlist(list(cmd.varlist))[0]
    col = data[name]
    if pd.api.types.is_numeric_dtype(col):
        raise StataExprError(f"split: {name!r} is not a string variable")
    seps = re.findall(r'"([^"]*)"|(\S+)', parse) if parse else []
    separators = [a or b for a, b in seps] or [" "]
    pattern = "|".join(re.escape(sep) for sep in separators)
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    pieces: List[List[str]] = []
    for keep, raw in zip(mask, col.astype(object).where(col.notna(), "")):
        text = str(raw) if notrim else str(raw).strip()
        if not keep:
            pieces.append([])
        elif parse is None:
            pieces.append(text.split())
        else:
            pieces.append(re.split(pattern, text))
    width = max((len(p) for p in pieces), default=0)
    if limit is not None:
        width = min(width, int(float(limit)))
    steps._own()
    stub = stub or name
    for k in range(width):
        values = [p[k] if k < len(p) else "" for p in pieces]
        if not notrim:
            values = [v.strip() for v in values]
        new = _new_name(steps.data, f"{stub}{k + 1}", "split")
        steps.data[new] = np.array(values, dtype=object)
    session.stored["r"] = {"nvars": float(width)}
    if destring:
        for k in range(width):
            _destring(session, f"{stub}{k + 1}, replace")
    return False


# ------------------------------------------------------------- random rows
def _rng(session: "StataSession") -> np.random.Generator:
    rng = session.stored.get("rng")
    if rng is None:
        rng = session.stored["rng"] = np.random.default_rng()
    session.stored["random_draws"] = True
    return rng


def _sample(session: "StataSession", rest: str) -> bool:
    """``sample # [if] [in] [, count by(vars)]``: keep a random # percent
    (or # rows) of the selected rows; rows not selected are all kept.
    The draw is numpy's, not Stata's."""
    steps = _steps(session)
    cmd = _cmd("sample " + rest)
    options = dict(cmd.options)
    count = _flag(options, "count", 5)
    by = _valued(options, "by", 2)
    _leftover(options, "sample")
    if len(cmd.varlist) != 1:
        raise StataExprError("sample: expected `sample # [, count]`")
    try:
        size = float(cmd.varlist[0])
    except ValueError:
        raise StataExprError(f"sample: {cmd.varlist[0]!r} is not a number") from None
    data = steps.data
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    rng = _rng(session)
    keep = ~mask
    rows = np.flatnonzero(mask)
    if by:
        keys = steps.expand_varlist(by.split())
        codes = data.iloc[rows].groupby(keys, dropna=False, sort=False).ngroup()
        groups = [rows[codes.to_numpy() == g] for g in range(codes.max() + 1)]
    else:
        groups = [rows]
    for members in groups:
        if count:
            k = int(size)
        else:
            k = int(np.floor(len(members) * size / 100.0 + 0.5))
        if k >= len(members):
            keep[members] = True
        elif k > 0:
            keep[rng.choice(members, size=k, replace=False)] = True
    attrs = dict(data.attrs)
    steps.replace_data(data.loc[keep])
    steps.data.attrs.update(attrs)
    return False


def _splitsample(session: "StataSession", rest: str) -> bool:
    """``splitsample [if], generate(new) [nsplit(#) split(numlist)]``:
    a random partition of the rows. The draw is numpy's."""
    steps = _steps(session)
    cmd = _cmd("splitsample " + rest)
    options = dict(cmd.options)
    generate = _valued(options, "generate", 1)
    nsplit = _valued(options, "nsplit", 1)
    split = _valued(options, "split", 5)
    _valued(options, "rseed", 5)
    _leftover(options, "splitsample")
    if generate is None or cmd.varlist:
        raise StataExprError("splitsample: expected `splitsample, generate(new)`")
    data = steps.data
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    if split:
        shares = np.asarray(_numlist(split, "splitsample"), dtype=float)
    else:
        k = int(float(nsplit)) if nsplit else 2
        shares = np.ones(k)
    shares = shares / shares.sum()
    rows = _rng(session).permutation(np.flatnonzero(mask))
    bounds = np.floor(np.cumsum(shares) * len(rows) + 0.5).astype(int)
    column = np.full(len(data), np.nan)
    start = 0
    for group, stop in enumerate(bounds, 1):
        column[rows[start:stop]] = group
        start = stop
    steps._own()
    steps.data[_new_name(data, generate, "splitsample")] = column
    return False


# ------------------------------------------------------- marks and checks
def _mark(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    cmd = _cmd("mark " + rest)
    options = dict(cmd.options)
    zero = _flag(options, "zeroweight", 1)
    _leftover(options, "mark")
    words = [w for w in cmd.varlist if not w.startswith("[")]
    if len(words) != 1:
        raise StataExprError("mark: expected `mark newvar [if] [in]`")
    data = steps.data
    column = row_mask(data, cmd.if_cond, cmd.in_range, session.stored).astype(float)
    if cmd.weight is not None and not zero:
        w = _numeric(data, cmd.weight[1], "mark")
        column = np.where(np.isnan(w) | (w == 0), 0.0, column)
    steps._own()
    steps.data[_new_name(data, words[0], "mark")] = column
    return False


def _markout(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    cmd = _cmd("markout " + rest)
    options = dict(cmd.options)
    strok = _flag(options, "strok", 5)
    _flag(options, "sysmissok", 7)
    _leftover(options, "markout")
    if len(cmd.varlist) < 1:
        raise StataExprError("markout: expected `markout markvar varlist`")
    data = steps.data
    flag = cmd.varlist[0]
    if flag not in data.columns:
        raise StataExprError(f"markout: variable {flag!r} is not in the data")
    column = _numeric(data, flag, "markout").copy()
    for name in steps.expand_varlist(list(cmd.varlist[1:])) if cmd.varlist[1:] else []:
        col = data[name]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            column[col.isna().to_numpy()] = 0.0
        elif strok:
            column[(col.isna() | (col == "")).to_numpy()] = 0.0
        else:
            column[:] = 0.0  # a string variable without strok marks out every row
    steps._own()
    steps.data[flag] = column
    return False


class StataAssertion(StataExprError):
    """An ``assert`` / ``confirm`` / ``isid`` that is false: Stata stops."""


def _assert(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    body, _, option_text = _split_options(rest)
    expr, tail = _qualifier_tail(body)
    cmd = _cmd(f"assert _x {tail}")
    data = steps.data
    value = evaluate(expr, data, session.stored)
    if value.dtype == object:
        raise StataExprError("assert: the expression is a string")
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    false = int((mask & (value == 0)).sum())
    if false:
        raise StataAssertion(
            f"assertion is false in {false} observation(s); Stata stops here " "(r(9))"
        )
    return False


def _isid(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    cmd = _cmd("isid " + rest)
    keys = steps.expand_varlist(list(cmd.varlist))
    if steps.data.duplicated(subset=keys).any():
        raise StataAssertion(
            f"variable(s) {' '.join(keys)} do not uniquely identify the "
            "observations; Stata stops here (r(459))"
        )
    return False


def _confirm(session: "StataSession", rest: str) -> bool:
    """``confirm [new|numeric|string] variable varlist``, ``confirm
    number`` / ``integer number``, ``confirm existence``."""
    words = rest.split()
    if not words:
        raise StataExprError("confirm: nothing to confirm")
    low = [w.lower() for w in words]
    data = session._steps.data if session._steps is not None else pd.DataFrame()

    def var_at(k: int) -> bool:
        return len(low) > k and len(low[k]) >= 1 and "variable".startswith(low[k])

    if low[0] in ("e", "ex", "exi", "exis", "exist", "existe", "existen",
                  "existenc", "existence"):  # fmt: skip
        if len(words) < 2:
            raise StataAssertion("'' found where something expected; Stata stops")
        return False
    if low[0] in ("n", "nu", "num", "numb", "numbe", "number") or (
        low[0].startswith("int") and len(low) > 1 and low[1].startswith("n")
    ):
        integer = low[0].startswith("int")
        text = " ".join(words[2:] if integer else words[1:])
        try:
            number = float(text)
        except ValueError:
            raise StataAssertion(
                f"'{text}' found where number expected; Stata stops here"
            ) from None
        if integer and number != int(number):
            raise StataAssertion(f"'{text}' found where integer expected")
        return False
    kind = None
    if low[0] in ("new", "numeric", "string") or low[0].startswith("str"):
        kind, words, low = low[0], words[1:], low[1:]
    if not var_at(0):
        raise StataExprError(f"`confirm {rest}` is not implemented")
    names = words[1:]
    if not names:
        raise StataAssertion("'' found where varname expected; Stata stops here")
    if kind == "new":
        for name in names:
            if name in data.columns:
                raise StataAssertion(
                    f"variable {name} already defined; Stata stops here (r(110))"
                )
            if not re.fullmatch(r"[A-Za-z_]\w{0,31}", name):
                raise StataAssertion(f"{name} invalid name; Stata stops here")
        return False
    from ._stata_datastep import expand_varlist

    cols = [str(c) for c in data.columns]
    try:
        found = expand_varlist(cols, names)
    except StataExprError as exc:
        raise StataAssertion(f"{exc}; Stata stops here (r(111))") from None
    missing = [v for v in found if v not in cols]
    if missing:
        raise StataAssertion(
            f"variable {missing[0]} not found; Stata stops here (r(111))"
        )
    for name in found:
        numeric = pd.api.types.is_numeric_dtype(data[name])
        if kind == "numeric" and not numeric:
            raise StataAssertion(f"'{name}' found where numeric variable expected")
        if kind is not None and kind.startswith("str") and numeric:
            raise StataAssertion(f"'{name}' found where string variable expected")
    return False


# --------------------------------------------------------- lists of things
def _levelsof(session: "StataSession", rest: str) -> bool:
    """``levelsof var [if] [in] [, local(name) missing clean
    separate(str)]``: the distinct values, sorted, in r(levels)."""
    steps = _steps(session)
    cmd = _cmd("levelsof " + rest)
    options = dict(cmd.options)
    local = _valued(options, "local", 1)
    missing = _flag(options, "missing", 4)
    clean = _flag(options, "clean", 1)
    separate = _valued(options, "separate", 1)
    _valued(options, "matrow", 6)
    _valued(options, "matcell", 7)
    _valued(options, "hexadecimal", 3)
    _leftover(options, "levelsof")
    if len(cmd.varlist) != 1:
        raise StataExprError("levelsof: expected one variable")
    data = steps.data
    name = steps.expand_varlist(list(cmd.varlist))[0]
    rows = data.loc[row_mask(data, cmd.if_cond, cmd.in_range, session.stored), name]
    sep = " " if separate is None else separate.strip('"')
    if pd.api.types.is_numeric_dtype(rows) or pd.api.types.is_bool_dtype(rows):
        x = rows.to_numpy(dtype=float, na_value=np.nan)
        values = sorted(set(x[~np.isnan(x)]))
        words = macro_number_list(values).split()
        if missing and np.isnan(x).any():
            words.append(".")
        count = len(words)
        text = sep.join(words)
    else:
        held = sorted(
            {str(v) for v in rows.dropna() if str(v) != "" or missing},
            key=lambda s: s.encode("utf-8", "surrogateescape"),
        )
        count = len(held)
        text = sep.join(held if clean else [f'`"{v}"\'' for v in held])
    session.stored["r"] = {"r": float(count), "N": float(rows.notna().sum())}
    session.stored.setdefault("r_macros", {})["levels"] = text
    if local:
        session._macros.locals[local] = text
    session.output = text
    return True


def _ds(session: "StataSession", rest: str) -> bool:
    """``ds [varlist] [, has(type ...) not(type ...)]``: r(varlist)."""
    steps = _steps(session)
    cmd = _cmd("ds " + rest)
    options = dict(cmd.options)
    has = _valued(options, "has", 1)
    excl = _valued(options, "not", 1)
    not_flag = _flag(options, "not", 1)
    for display_only in ("alpha", "detail", "varwidth", "skip"):
        _flag(options, display_only, 1)
        _valued(options, display_only, 1)
    _leftover(options, "ds")
    data = steps.data
    names = steps.expand_varlist(list(cmd.varlist) or ["_all"])

    def fits(name: str, spec: str) -> bool:
        words = spec.split()
        if not words or words[0] != "type":
            raise StataExprError(f"ds: has({spec}) is not implemented (only type)")
        numeric = pd.api.types.is_numeric_dtype(data[name])
        answer = False
        for kind in words[1:]:
            if kind == "numeric":
                answer |= numeric
            elif kind.startswith("str"):
                answer |= not numeric
            else:
                raise StataExprError(f"ds: the type {kind!r} is not tracked here")
        return answer

    if has:
        names = [n for n in names if fits(n, has)]
    if excl:
        names = [n for n in names if not fits(n, excl)]
    if not_flag and cmd.varlist:
        listed = set(names)
        names = [str(c) for c in data.columns if str(c) not in listed]
    session.stored.setdefault("r_macros", {})["varlist"] = " ".join(names)
    session.stored["r"] = {}
    session.output = names
    return True


def _unab(session: "StataSession", rest: str) -> bool:
    m = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*(.*?)(?:,.*)?\Z", rest, re.S)
    if m is None:
        raise StataExprError("unab: expected `unab lname : varlist`")
    names = _steps(session).expand_varlist(m.group(2).split())
    session._macros.locals[m.group(1)] = " ".join(names)
    return False


def _joinby(session: "StataSession", rest: str) -> bool:
    """``joinby varlist using file [, unmatched(none|both|master|using)]``:
    every pairing of the rows that agree on the key variables."""
    from ._stata_multi import _dataset

    steps = _steps(session)
    m = re.match(r"\s*(.*?)\s+using\s+(.+?)\s*(?:,(.*))?\Z", rest, re.S | re.I)
    if m is None:
        raise StataExprError("joinby: expected `joinby varlist using file`")
    cmd = _cmd("joinby _x ," + (m.group(3) or ""))
    options = dict(cmd.options)
    unmatched = (_valued(options, "unmatched", 2) or "none").lower()
    _flag(options, "update", 6)
    _leftover(options, "joinby")
    data = steps.data
    other = _dataset(session, m.group(2))
    if other is None:
        raise StataExprError(f"joinby: the dataset {m.group(2)!r} is not known")
    keys = steps.expand_varlist(m.group(1).split())
    missing = [k for k in keys if k not in other.columns]
    if missing:
        raise StataExprError(f"joinby: {missing} are not in the using data")
    how = {"none": "inner", "both": "outer", "master": "left", "using": "right"}.get(
        unmatched[:1]
        and {"n": "none", "b": "both", "m": "master", "u": "using"}.get(
            unmatched[0], ""
        )
    )
    if how is None:
        raise StataExprError(f"joinby: unmatched({unmatched}) is not an option")
    shared = [c for c in other.columns if c in data.columns and c not in keys]
    out = data.merge(other.drop(columns=shared), on=keys, how=how, sort=True)
    attrs = dict(data.attrs)
    steps.replace_data(out)
    steps.data.attrs.update(attrs)
    steps.adopt_missing_codes(other)
    return False


def _recast(session: "StataSession", rest: str) -> bool:
    steps = _steps(session)
    words = rest.replace(",", " ").split()
    if len(words) < 2:
        raise StataExprError("recast: expected `recast type varlist`")
    kind = words[0]
    names = steps.expand_varlist([w for w in words[1:] if w != "force"])
    if kind == "double":
        steps._float -= set(names)
    elif kind == "float":
        steps._own()
        for name in names:
            x = _numeric(steps.data, name, "recast")
            with np.errstate(over="ignore"):
                steps.data[name] = x.astype(np.float32).astype(np.float64)
            steps._float.add(name)
    elif kind not in ("byte", "int", "long") and not kind.startswith("str"):
        raise StataExprError(f"recast: the type {kind!r} is not known")
    return False


MANAGE_COMMANDS: Dict[str, Any] = {
    "recode": _recode,
    "xtile": lambda s, r: _xtile(s, r, pctile=False),
    "pctile": lambda s, r: _xtile(s, r, pctile=True),
    "tostring": _tostring,
    "destring": _destring,
    "order": _order,
    "expand": _expand,
    "contract": _contract,
    "separate": _separate,
    "split": _split,
    "sample": _sample,
    "splitsample": _splitsample,
    "mark": _mark,
    "markout": _markout,
    "assert": _assert,
    "isid": _isid,
    "confirm": _confirm,
    "levelsof": _levelsof,
    "ds": _ds,
    "unab": _unab,
    "joinby": _joinby,
    "recast": _recast,
    "compress": lambda s, r: False,
}

_HEAD = re.compile(r"\s*([A-Za-z_]\w*)\b\s*(.*)\Z", re.S)


def run_manage(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``line`` if it is one of the commands of this module.

    Returns ``None`` when it is not, otherwise whether it produced output.
    """
    m = _HEAD.match(line)
    if m is None:
        return None
    word, rest = m.group(1), m.group(2)
    if word == "duplicates":
        sub = rest.split(None, 1)
        if sub and sub[0][:1] in ("r", "t") and not sub[0].startswith("dr"):
            return _duplicates(session, sub[0], sub[1] if len(sub) > 1 else "")
        return None
    handler = MANAGE_COMMANDS.get(word)
    if handler is None:
        return None
    done: Optional[bool] = handler(session, rest)
    return done
