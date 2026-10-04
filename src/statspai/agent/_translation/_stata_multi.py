"""More than one dataset in a ``sp.stata`` session.

Stata holds one dataset at a time, and a do-file works around that with
``tempfile`` / ``save`` / ``use``, ``append``, ``merge`` and, since Stata
16, frames. (``reshape`` is a data step: ``_stata_reshape.py``.)

Nothing is read from or written to disk. ``save name`` keeps a copy of the
data in the session under that name, and ``use`` / ``append using`` /
``merge ... using`` find it there, or among the frames handed to
``sp.stata(..., files={"name": DataFrame})``. A name is the file name
without its folder and without ``.dta``.

``merge``
    ``merge 1:1 | m:1 | 1:m varlist using name`` (and ``1:1 _n``) with
    ``keep()``, ``assert()``, ``keepusing()``, ``generate()`` and
    ``nogenerate``. The keys must identify rows where the type says so. A
    variable present on both sides keeps the master's values (a row found
    only in the using data has the using data's). ``_merge`` is
    1 (master only), 2 (using only) or 3 (both). As in Stata the master's
    rows come out sorted by the key variables, followed by the rows found
    only in the using data. ``m:m``, ``update`` and
    ``replace`` are refused.
frames
    ``frame create``, ``frame copy``, ``frame change`` / ``cwf``, ``frame
    drop``, ``frames reset``, ``frame put varlist, into()``, ``frame post``,
    ``frame name: command``, ``frlink 1:1 | m:1 varlist, frame()`` and
    ``frget varlist, from()``. A ``frame name { }`` block is refused.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd

from ._stata_datastep import DataSteps, row_mask
from ._stata_expr import StataExprError
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["multi_line", "file_key"]

_SAVE = re.compile(r"\s*save(?:old)?\s+(.+?)\s*$", re.I | re.S)
_USE = re.compile(r"\s*use\s+(.+?)\s*$", re.I | re.S)
_APPEND = re.compile(r"\s*append\s+using\s+(.+?)\s*$", re.I | re.S)
_MERGE = re.compile(
    r"\s*merge\s+(1:1|m:1|1:m|m:m|n:1|1:n|n:n)\s+(.+?)\s+using\s+(.+?)\s*$",
    re.I | re.S,
)
_FRAME = re.compile(r"\s*(?:frames?|cwf)\b(.*)$", re.I | re.S)
_FRLINK = re.compile(r"\s*frlink\s+(1:1|m:1)\s+(.+?)\s*$", re.I | re.S)
_FRGET = re.compile(r"\s*frget\s+(.+?)\s*$", re.I | re.S)
_RESULTS = {"master": 1, "1": 1, "using": 2, "2": 2, "match": 3, "matched": 3, "3": 3}


def file_key(name: str) -> str:
    """``"data/panel.dta"`` -> ``panel``: how a dataset is named here."""
    text = name.strip().strip('"').strip()
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    return text[:-4] if text.lower().endswith(".dta") else text


def _split(text: str) -> tuple:
    """``name, options`` -> (name part, {option: value})."""
    try:
        cmd = _parse("x " + text)
    except StataParseError as exc:
        raise StataExprError(f"cannot read {text!r} ({exc})") from exc
    return cmd, {str(k).lower(): v for k, v in dict(cmd.options).items()}


def _steps(session: "StataSession") -> DataSteps:
    if session._steps is None:
        session._steps = DataSteps(pd.DataFrame())
        session._steps.stored = session.stored
    return session._steps


def _dataset(session: "StataSession", name: str) -> Optional[pd.DataFrame]:
    key = file_key(name)
    held = session.files.get(key)
    if held is None and session.file_loader is not None:
        held = session.file_loader(key)
        if held is not None:
            session.files[key] = held
    return held


# ------------------------------------------------------------ save and use
def _save(session: "StataSession", rest: str) -> bool:
    cmd, options = _split(rest)
    if len(cmd.varlist) != 1 or set(options) - {"replace", "emptyok", "nolabel"}:
        raise StataExprError("`save` is run as `save name [, replace]`")
    key = file_key(cmd.varlist[0])
    if key in session.files and "replace" not in options:
        raise StataExprError(f"file {key} already exists; `save ..., replace`")
    session.files[key] = _steps(session).data.copy()
    if not key.startswith("__file"):
        warnings.warn(
            f"sp.stata: `save {key}` keeps the data in the session under that "
            "name; no file is written.",
            UserWarning,
            stacklevel=4,
        )
    return False


def _use(session: "StataSession", rest: str) -> Optional[bool]:
    cmd, options = _split(rest)
    words = list(cmd.varlist)
    keep: Optional[List[str]] = None
    if "using" in words:
        at = words.index("using")
        keep, words = words[:at], words[at + 1 :]
    if len(words) != 1 or set(options) - {"clear", "nolabel"}:
        return None
    held = _dataset(session, words[0])
    if held is None:
        return None  # not a dataset of this session: refused further on
    frame = held.copy()
    if cmd.if_cond or cmd.in_range:
        frame = frame.loc[row_mask(frame, cmd.if_cond, cmd.in_range, session.stored)]
    if keep:
        missing = [v for v in keep if v not in frame.columns]
        if missing:
            raise StataExprError(f"variable(s) {missing} are not in {words[0]}")
        frame = frame[keep]
    session.use(frame.reset_index(drop=True))
    return False


def _append(session: "StataSession", rest: str) -> bool:
    cmd, options = _split(rest)
    if set(options) - {"force", "nolabel", "nonotes"} or not cmd.varlist:
        raise StataExprError("`append` is run as `append using name [name ...]`")
    for name in cmd.varlist:
        held = _dataset(session, name)
        if held is None:
            raise StataExprError(f"dataset {file_key(name)!r} is not in the session")
        session.append(held.copy())
    return False


# ------------------------------------------------------------------ merge
def _results(spec: Any, what: str) -> List[int]:
    codes = []
    for word in str(spec or "").split():
        if word.lower() not in _RESULTS:
            raise StataExprError(f"merge {what}({spec}): {word!r} is not implemented")
        codes.append(_RESULTS[word.lower()])
    return codes


def _merge(session: "StataSession", kind: str, keys_text: str, rest: str) -> bool:
    kind = kind.lower().replace("n", "m")
    cmd, options = _split(rest)
    if len(cmd.varlist) != 1:
        raise StataExprError("`merge` takes one using dataset")
    allowed = {"keep", "assert", "keepusing", "generate", "gen", "nogenerate",
               "nogen", "nolabel", "nonotes", "noreport", "force"}  # fmt: skip
    extra = set(options) - allowed
    if extra or kind == "m:m":
        raise StataExprError(
            "merge m:m is not implemented"
            if kind == "m:m"
            else f"merge option(s) {sorted(extra)} are not implemented"
        )
    using = _dataset(session, cmd.varlist[0])
    if using is None:
        raise StataExprError(
            f"dataset {file_key(cmd.varlist[0])!r} is not in the session; "
            "`save` it first or pass it in files="
        )
    steps = _steps(session)
    master = steps.data
    keys = keys_text.split()
    positional = keys == ["_n"]
    using = using.copy()
    if options.get("keepusing"):
        wanted = str(options["keepusing"]).split()
        missing = [v for v in wanted if v not in using.columns]
        if missing:
            raise StataExprError(f"keepusing(): {missing} are not in the using data")
        using = using[(keys if not positional else []) + wanted]
    gen = options.get("generate") or options.get("gen") or "_merge"
    no_gen = "nogenerate" in options or "nogen" in options
    if not no_gen and gen in master.columns:
        raise StataExprError(f"variable {gen} already exists")

    if positional:
        if kind != "1:1":
            raise StataExprError("merge on _n is 1:1")
        master = master.reset_index(drop=True)
        using = using.reset_index(drop=True)
        left = master.assign(__key=np.arange(len(master)))
        right = using.assign(__key=np.arange(len(using)))
        keys = ["__key"]
    else:
        for side, frame in (("master", master), ("using", using)):
            missing = [k for k in keys if k not in frame.columns]
            if missing:
                raise StataExprError(f"merge: {missing} are not in the {side} data")
        if kind in ("1:1", "1:m") and master.duplicated(keys).any():
            raise StataExprError(
                "variable(s) " + " ".join(keys) + " do not uniquely identify "
                "observations in the master data"
            )
        if kind in ("1:1", "m:1") and using.duplicated(keys).any():
            raise StataExprError(
                "variable(s) " + " ".join(keys) + " do not uniquely identify "
                "observations in the using data"
            )
        left, right = master, using
    # a variable on both sides keeps the master's values on the master's
    # rows; a row found only in the using data has the using data's
    shared = [c for c in right.columns if c in left.columns and c not in keys]
    renamed = right.rename(columns={c: f"__using_{c}" for c in shared})
    merged = left.merge(renamed, on=keys, how="outer", indicator="__side", sort=False)
    code = merged["__side"].map({"left_only": 1.0, "right_only": 2.0, "both": 3.0})
    merged = merged.drop(columns=["__side"])
    merged["__code"] = code.astype(float)
    only_using = (merged["__code"] == 2.0).to_numpy()
    for c in shared:
        if only_using.any():
            merged.loc[only_using, c] = merged.loc[only_using, f"__using_{c}"]
        merged = merged.drop(columns=[f"__using_{c}"])
    asserted = _results(options.get("assert"), "assert")
    if asserted and not merged["__code"].isin(asserted).all():
        raise StataExprError(
            "merge: after merge, not all observations " + str(options["assert"])
        )
    kept = _results(options.get("keep"), "keep")
    if kept:
        merged = merged[merged["__code"].isin(kept)]
    # Stata leaves the master's rows sorted by the keys and puts the rows
    # found only in the using data after them
    merged["__last"] = (merged["__code"] == 2.0).astype(int)
    by = ["__last"] + keys
    merged = merged.sort_values(by, kind="stable", na_position="last")
    merged = merged.drop(columns=["__last"])
    if positional:
        merged = merged.drop(columns=["__key"])
    if no_gen:
        merged = merged.drop(columns=["__code"])
    else:
        merged = merged.rename(columns={"__code": gen})
    attrs = dict(master.attrs)
    merged = merged.reset_index(drop=True)
    merged.attrs.update(attrs)
    steps.replace_data(merged)
    return False


# ----------------------------------------------------------------- frames
def _current(session: "StataSession") -> None:
    """Put the frame in memory back into the table of frames."""
    session.frames[session.frame] = _steps(session)


def _switch(session: "StataSession", name: str) -> None:
    if name not in session.frames:
        raise StataExprError(f"frame {name} not found")
    _current(session)
    session.frame = name
    session._steps = session.frames[name]
    session._steps.stored = session.stored
    session.panel = (None, None)


def _new_steps(session: "StataSession", data: pd.DataFrame) -> DataSteps:
    steps = DataSteps(data)
    steps.stored = session.stored
    return steps


def _frame(session: "StataSession", line: str, rest: str) -> Optional[bool]:
    head = line.strip().split(None, 1)[0].lower()
    words = rest.strip()
    if head == "cwf":
        _switch(session, words)
        return False
    if head == "frames" and re.fullmatch(r"reset", words, re.I):
        session.frames = {}
        session.frame = "default"
        session.links = {}
        session._steps = _new_steps(session, pd.DataFrame())
        return False
    if head == "frames" and re.fullmatch(r"dir|describe", words, re.I):
        return False
    _current(session)
    prefixed = re.match(r"([A-Za-z_]\w*)\s*:\s*(.+)\Z", words, re.S)
    if prefixed and prefixed.group(1).lower() not in ("create", "copy", "change"):
        # frame name: command -- run it there, come back
        target, inner = prefixed.group(1), prefixed.group(2)
        home = session.frame
        _switch(session, target)
        try:
            return session.run(inner)
        finally:
            _switch(session, home)
    sub, _, tail = words.partition(" ")
    sub = sub.lower()
    tail = tail.strip()
    if sub == "create":
        names = tail.split()
        if not names or names[0] in session.frames:
            raise StataExprError(
                "frame create needs a new name"
                if not names
                else f"frame {names[0]} already exists"
            )
        types = ("byte", "int", "long", "float", "double")
        columns = [
            n for n in names[1:] if n.lower() not in types and not n.startswith("str")
        ]
        session.frames[names[0]] = _new_steps(
            session, pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
        )
        return False
    if sub == "copy":
        cmd, options = _split(tail)
        if len(cmd.varlist) != 2 or set(options) - {"replace"}:
            raise StataExprError(
                "`frame copy` is run as `frame copy from to [, replace]`"
            )
        source, target = cmd.varlist
        if source not in session.frames:
            raise StataExprError(f"frame {source} not found")
        if target in session.frames and "replace" not in options:
            raise StataExprError(f"frame {target} already exists")
        if target == session.frame:
            raise StataExprError("the frame in use cannot be replaced")
        session.frames[target] = _new_steps(session, session.frames[source].data.copy())
        return False
    if sub == "change":
        _switch(session, tail)
        return False
    if sub == "drop":
        for name in tail.split():
            if name == session.frame:
                raise StataExprError("the frame in use cannot be dropped")
            if name not in session.frames:
                raise StataExprError(f"frame {name} not found")
            del session.frames[name]
        return False
    if sub == "put":
        cmd, options = _split(tail)
        target = str(options.get("into") or "").strip()
        if not target or set(options) - {"into"}:
            raise StataExprError("`frame put` is run as `frame put varlist, into(new)`")
        if target in session.frames:
            raise StataExprError(f"frame {target} already exists")
        data = _steps(session).data
        chosen: List[str] = []
        for token in cmd.varlist:
            if token in ("*", "_all"):
                chosen.extend(str(c) for c in data.columns)
            elif token in data.columns:
                chosen.append(token)
            else:
                raise StataExprError(f"variable {token!r} is not in the data")
        mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
        picked = data.loc[mask, list(dict.fromkeys(chosen))].reset_index(drop=True)
        session.frames[target] = _new_steps(session, picked.copy())
        return False
    if sub == "post":
        name, _, values = tail.partition(" ")
        if name not in session.frames:
            raise StataExprError(f"frame {name} not found")
        target = session.frames[name]
        exprs = re.findall(r"\(((?:[^()]|\([^()]*\))*)\)", values)
        if len(exprs) != len(target.data.columns):
            raise StataExprError(
                f"frame post: {len(exprs)} values for "
                f"{len(target.data.columns)} variables"
            )
        row = [session.value(e) for e in exprs]
        target._own()
        target.data.loc[len(target.data)] = row
        return False
    if words.endswith("{"):
        raise StataExprError("a `frame name { }` block is not implemented")
    raise StataExprError(f"frame {sub} is not implemented")


def _frlink(session: "StataSession", kind: str, rest: str) -> bool:
    cmd, options = _split(rest)
    target_spec = str(options.get("frame") or "").split()
    if not target_spec or set(options) - {"frame", "generate", "gen"}:
        raise StataExprError("`frlink` is run as `frlink m:1 varlist, frame(name)`")
    target = target_spec[0]
    keys = list(cmd.varlist)
    other_keys = target_spec[1:] or keys
    _current(session)
    if target not in session.frames:
        raise StataExprError(f"frame {target} not found")
    here = _steps(session).data
    there = session.frames[target].data
    missing = [k for k in keys if k not in here.columns]
    missing += [k for k in other_keys if k not in there.columns]
    if missing or len(keys) != len(other_keys):
        raise StataExprError(f"frlink: cannot match on {missing or keys}")
    if there.duplicated(other_keys).any():
        raise StataExprError(
            "variable(s) " + " ".join(other_keys) + f" do not identify rows of {target}"
        )
    if kind.lower() == "1:1" and here.duplicated(keys).any():
        raise StataExprError("frlink 1:1: the keys repeat in the current frame")
    name = str(options.get("generate") or options.get("gen") or target)
    if name in here.columns:
        raise StataExprError(f"variable {name} already exists")
    lookup = there[other_keys].copy()
    lookup.columns = keys
    lookup["__row"] = np.arange(1.0, len(there) + 1.0)
    row = here[keys].merge(lookup, on=keys, how="left")["__row"].to_numpy()
    _steps(session).add_column(name, row, double=True)
    session.links[(session.frame, name)] = target
    return False


def _frget(session: "StataSession", rest: str) -> bool:
    cmd, options = _split(rest)
    link = str(options.get("from") or "").strip()
    if not link or set(options) - {"from"}:
        raise StataExprError("`frget` is run as `frget varlist, from(linkvar)`")
    here = _steps(session)
    if (session.frame, link) not in session.links:
        # Stata accepts an unambiguous abbreviation of a variable name
        full = [
            n for (f, n) in session.links if f == session.frame and n.startswith(link)
        ]
        if len(full) == 1:
            link = full[0]
    target = session.links.get((session.frame, link))
    if target is None or link not in here.data.columns:
        raise StataExprError(f"{link} is not a link variable created by frlink")
    _current(session)
    there = session.frames[target].data
    tokens = [t for t in cmd.varlist if t != "="]
    pairs = (
        [(tokens[0], tokens[1])]
        if "=" in cmd.varlist and len(tokens) == 2
        else [(t, t) for t in tokens]
    )
    row = here.data[link].to_numpy(dtype=float)
    ok = ~np.isnan(row)
    for new, old in pairs:
        if old not in there.columns:
            raise StataExprError(f"variable {old!r} is not in frame {target}")
        if new in here.data.columns:
            raise StataExprError(f"variable {new} already exists")
        source = there[old].to_numpy()
        if source.dtype == object:
            values = np.full(len(row), "", dtype=object)
            values[ok] = source[row[ok].astype(int) - 1]
            here._own()
            here.data[new] = values
        else:
            values = np.full(len(row), np.nan)
            values[ok] = source.astype(float)[row[ok].astype(int) - 1]
            here.add_column(new, values, double=True)
    return False


def multi_line(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``line`` if it is one of this module's commands, else ``None``."""
    m = _SAVE.match(line)
    if m:
        return _save(session, m.group(1))
    m = _USE.match(line)
    if m:
        return _use(session, m.group(1))
    m = _APPEND.match(line)
    if m:
        return _append(session, m.group(1))
    m = _MERGE.match(line)
    if m:
        return _merge(session, m.group(1), m.group(2), m.group(3))
    m = _FRLINK.match(line)
    if m:
        return _frlink(session, m.group(1), m.group(2))
    m = _FRGET.match(line)
    if m:
        return _frget(session, m.group(1))
    m = _FRAME.match(line)
    if m:
        return _frame(session, line, m.group(1))
    return None
