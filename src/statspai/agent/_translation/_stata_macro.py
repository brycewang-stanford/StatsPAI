"""The programming tools of a do-file: extended macro functions, ``syntax``
and the commands that take a command line apart.

* ``local n : word count ...`` and the other extended macro functions
  ([P] macro), also inline as `` `: word 2 of ...' ``;
* ``syntax`` ([P] syntax): the standard grammar of a user-written command
  (variable list, ``if`` / ``in``, weight, options with their minimal
  abbreviations and types) and ``marksample``;
* ``tokenize``, ``gettoken``, ``macro shift``, ``return local``;
* ``if exp command`` / ``else command`` on one line;
* ``scalar drop``, ``matrix list e(b)``.

What is not covered raises :class:`StataExprError`; nothing is guessed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._stata_datastep import expand_varlist, row_mask
from ._stata_expr import StataExprError
from ._stata_functions import stata_format
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse
from ._stata_script import ScriptError

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["extended_function", "tool_line", "ProgramExit"]


class ProgramExit(Exception):
    """``exit`` inside a program: the program ends, the session goes on."""


def _unquote(text: str) -> str:
    text = text.strip()
    if text.startswith('`"') and text.endswith("\"'"):
        return text[2:-2]
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def _words(text: str) -> List[str]:
    """Words of a macro: blanks separate, quotes bind."""
    out = re.findall(r'`"[^`]*?"\'|"[^"]*"|\S+', text)
    return [_unquote(w) for w in out]


def _data(session: "StataSession") -> pd.DataFrame:
    if session._steps is None:
        raise StataExprError("no data in memory")
    return session._steps.data


def _one_variable(session: "StataSession", name: str) -> str:
    data = _data(session)
    found = expand_varlist([str(c) for c in data.columns], [name.strip()])
    if len(found) != 1 or found[0] not in data.columns:
        raise StataExprError(f"variable {name.strip()!r} not found")
    return found[0]


def _storage_type(session: "StataSession", name: str) -> str:
    """The storage type as far as the frame shows it: ``str#`` for a
    string, otherwise the narrowest Stata type that holds the dtype."""
    col = _data(session)[name]
    if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
        width = int(col.dropna().astype(str).str.encode("utf-8").str.len().max() or 1)
        return f"str{max(width, 1)}"
    kind = str(col.dtype)
    if kind in ("int8", "bool"):
        return "byte"
    if kind == "int16":
        return "int"
    if kind in ("int32", "int64"):
        return "long"
    if kind == "float32" or (
        session._steps is not None and name in session._steps._float
    ):
        return "float"
    return "double"


def extended_function(session: "StataSession", text: str) -> str:
    """The value of an extended macro function (the text after the colon
    of ``local name : ...``)."""
    text = text.strip()
    low = text.lower()
    steps = session._steps

    m = re.match(r"word\s+count\s*(.*)\Z", text, re.S | re.I)
    if m:
        return str(len(_words(m.group(1))))
    m = re.match(r"word\s+(-?\d+)\s+of\s*(.*)\Z", text, re.S | re.I)
    if m:
        words, k = _words(m.group(2)), int(m.group(1))
        return words[k - 1] if 1 <= k <= len(words) else ""
    m = re.match(r"piece\s+(\d+)\s+(\d+)\s+of\s+(.*?)(?:,.*)?\Z", text, re.S | re.I)
    if m:
        k, width = int(m.group(1)), int(m.group(2))
        body = _unquote(m.group(3))
        return body[(k - 1) * width : k * width]
    m = re.match(r"var(?:iable)?\s+l(?:abel)?\s+(\S+)\s*\Z", text, re.I)
    if m:
        name = _one_variable(session, m.group(1))
        return str((_data(session).attrs.get("_labels") or {}).get(name, ""))
    m = re.match(r"val(?:ue)?\s+l(?:abel)?\s+(\S+)\s*\Z", text, re.I)
    if m:
        name = _one_variable(session, m.group(1))
        return str(steps._set_of.get(name, "")) if steps is not None else ""
    m = re.match(r"data\s+l(?:abel)?\s*\Z", text, re.I)
    if m:
        return str(_data(session).attrs.get("_data_label") or "")
    m = re.match(r"l(?:abel)?\s+(\(\s*\S+\s*\)|\S+)\s+(\S+)(\s*,\s*strict)?\s*\Z",
                 text, re.I)  # fmt: skip
    if m and not low.startswith("list"):
        target, raw = m.group(1), m.group(2)
        if steps is None:
            raise StataExprError("no data in memory")
        if target.startswith("("):
            name = _one_variable(session, target[1:-1])
            set_name = steps._set_of.get(name, "")
        else:
            set_name = target
        table = steps._label_sets.get(set_name, {})
        try:
            number = float(raw)
        except ValueError:
            raise StataExprError(f"label: {raw!r} is not a number") from None
        key: Any = int(number) if number == int(number) else number
        if key in table:
            return str(table[key])
        return "" if m.group(3) else raw
    m = re.match(r"type\s+(\S+)\s*\Z", text, re.I)
    if m:
        return _storage_type(session, _one_variable(session, m.group(1)))
    m = re.match(r"format\s+(\S+)\s*\Z", text, re.I)
    if m:
        name = _one_variable(session, m.group(1))
        held = (_data(session).attrs.get("_formats") or {}).get(name)
        if held:
            return str(held)
        kind = _storage_type(session, name)
        if kind.startswith("str"):
            return f"%{kind[3:]}s"
        return {"byte": "%8.0g", "int": "%8.0g", "long": "%12.0g",
                "float": "%9.0g", "double": "%10.0g"}[kind]  # fmt: skip
    m = re.match(r"sortedby\s*\Z", text, re.I)
    if m:
        raise StataExprError("`: sortedby` is not tracked here")
    m = re.match(r"(?:length|strlen|ustrlen)\s+(local|global)\s+(\S+)\s*\Z", text, re.I)
    if m:
        macros: Any = (
            session._macros.locals
            if m.group(1).lower() == "local"
            else session._macros.globals
        )
        return str(len(macros.get(m.group(2)) or ""))
    m = re.match(r"subinstr\s+(local|global)\s+(\S+)\s+(.*)\Z", text, re.S | re.I)
    if m:
        macros = (
            session._macros.locals
            if m.group(1).lower() == "local"
            else session._macros.globals
        )
        body = macros.get(m.group(2)) or ""
        rest, _, option_text = m.group(3).partition(",")
        pieces = re.findall(r'`"[^`]*?"\'|"[^"]*"', rest)
        if len(pieces) != 2:
            raise StataExprError('subinstr: expected "from" "to"')
        old, new = _unquote(pieces[0]), _unquote(pieces[1])
        options = option_text.split()
        everything = "all" in options
        if "word" in options:
            words = body.split()
            done = False
            for i, w in enumerate(words):
                if w == old and (everything or not done):
                    words[i], done = new, True
            return " ".join(w for w in words if w != "")
        return body.replace(old, new) if everything else body.replace(old, new, 1)
    m = re.match(r"di(?:splay)?\s+(%\S+)\s+(.+)\Z", text, re.S | re.I)
    if m:
        return stata_format(session.value(m.group(2)), m.group(1)).strip()
    m = re.match(r"di(?:splay)?\s+(.+)\Z", text, re.S | re.I)
    if m:
        body = m.group(1).strip()
        if body.startswith('"') or body.startswith('`"'):
            return _unquote(body)
        return stata_format(session.value(body), "%10.0g")
    m = re.match(r"list\s+(.*)\Z", text, re.S | re.I)
    if m:
        return _list_function(session, m.group(1).strip())
    m = re.match(r"(col|row)(full)?names\s+(\S+)\s*\Z", text, re.I)
    if m:
        held = (session.stored.get("matrices") or {}).get(m.group(3))
        if held is None:
            raise StataExprError(f"matrix {m.group(3)} not found")
        return " ".join(
            str(n) for n in held["cols" if m.group(1).lower() == "col" else "rows"]
        )
    raise StataExprError(f"the extended macro function `: {text}` is not implemented")


def _macro_list(session: "StataSession", name: str) -> List[str]:
    return _words(session._macros.locals.get(name) or "")


def _join(words: List[str]) -> str:
    return " ".join(f'"{w}"' if (" " in w or w == "") else w for w in words)


def _list_function(session: "StataSession", text: str) -> str:
    """``: list ...`` -- the macro-list functions of [P] macro lists."""
    m = re.match(r"(uniq|dups|sort|clean|retokenize|sizeof)\s+(\S+)\s*\Z", text)
    if m:
        words = _macro_list(session, m.group(2))
        what = m.group(1)
        if what == "sizeof":
            return str(len(words))
        if what == "uniq":
            return _join(list(dict.fromkeys(words)))
        if what == "dups":
            seen, dups = set(), []
            for w in words:
                if w in seen:
                    dups.append(w)
                seen.add(w)
            return _join(dups)
        if what == "sort":
            return _join(sorted(words))
        return _join(words)
    m = re.match(r'posof\s+(`"[^`]*?"\'|"[^"]*"|\S+)\s+in\s+(\S+)\s*\Z', text)
    if m:
        words = _macro_list(session, m.group(2))
        target = _unquote(m.group(1))
        return str(words.index(target) + 1 if target in words else 0)
    m = re.match(r"(\S+)\s*(\||&|-|===|==|in)\s*(\S+)\s*\Z", text)
    if m:
        a, op, b = (
            _macro_list(session, m.group(1)),
            m.group(2),
            _macro_list(session, m.group(3)),
        )
        if op == "|":
            return _join(a + [w for w in b if w not in a])
        if op == "&":
            return _join([w for w in a if w in b])
        if op == "-":
            return _join([w for w in a if w not in b])
        if op == "==":
            return str(int(sorted(a) == sorted(b)))
        if op == "===":
            return str(int(a == b))
        return str(int(all(w in b for w in a)))
    raise StataExprError(f"`: list {text}` is not implemented")


# ------------------------------------------------------------------ syntax
_ELEMENT = re.compile(
    r"(new)?var(?:list|name)\b(?:\s*\(([^)]*)\))?|namelist\b(?:\s*\(([^)]*)\))?"
    r"|anything\b(?:\s*\(([^)]*)\))?",
    re.I,
)
_OPTION_SPEC = re.compile(r"(no)?([A-Za-z_]\w*)(?:\(([^)]*)\))?|\*")


def _split_spec(spec: str) -> Tuple[str, str, bool]:
    """``syntax`` description -> (before the comma, options, comma is
    optional). The comma that starts the options is the first one outside
    parentheses."""
    depth = 0
    for i, ch in enumerate(spec):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            head = spec[:i]
            optional = head.count("[") > head.count("]")
            return head.rstrip(" ["), spec[i + 1 :], optional
    return spec, "", True


def _option_specs(text: str, all_optional: bool) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    optional = all_optional
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        ch = text[pos]
        if ch.isspace():
            pos += 1
        elif ch == "[":
            optional, pos = True, pos + 1
        elif ch == "]":
            optional, pos = all_optional, pos + 1
        else:
            m = _OPTION_SPEC.match(text, pos)
            if m is None or m.end() == pos:
                raise StataExprError(
                    f"syntax: cannot read the options at {text[pos:]!r}"
                )
            pos = m.end()
            if m.group(0) == "*":
                specs.append({"star": True})
                continue
            name = m.group(2)
            capitals = re.match(r"[A-Z_0-9]*", name)
            minimum = (capitals.group(0) if capitals else "") or name
            specs.append(
                {
                    "name": name.lower(),
                    "minimum": len(minimum),
                    "negatable": bool(m.group(1)),
                    "argument": m.group(3),
                    "optional": optional or bool(m.group(1)) or m.group(3) is None,
                }
            )
    return specs


def _given(
    options: Dict[str, Any], spec: Dict[str, Any], prefix: str = ""
) -> Optional[str]:
    """The key under which the user wrote the option, if at all."""
    full = prefix + spec["name"]
    shortest = len(prefix) + spec["minimum"]
    for key in options:
        if shortest <= len(key) <= len(full) and full.startswith(key):
            return key
    return None


def _syntax(session: "StataSession", spec: str, line: str) -> None:
    locals_ = session._macros.locals
    command_line = locals_.get("0") or ""
    head_spec, option_spec, comma_optional = _split_spec(spec)
    try:
        cmd = _parse("_cmd " + command_line)
    except StataParseError as exc:
        raise StataExprError(f"syntax: cannot read the command line ({exc})") from None
    words = [w for w in cmd.varlist if not w.startswith("[")]
    weight = next((w for w in cmd.varlist if w.startswith("[")), None)
    using = None
    if "using" in words:
        at = words.index("using")
        using, words = " ".join(words[at + 1 :]), words[:at]
    # --- the variable list
    element = _ELEMENT.search(head_spec)
    remainder = head_spec
    if element is not None:
        before = head_spec[: element.start()]
        optional = before.count("[") > before.count("]")
        detail = element.group(2) or element.group(3) or element.group(4) or ""
        kind = element.group(0).lower()
        target = "varlist"
        named = re.search(r"\bname\s*=\s*(\w+)", detail)
        if kind.startswith("namelist"):
            target = "namelist"
        if kind.startswith("anything"):
            target = "anything"
        if named:
            target = named.group(1)
        if kind.startswith(("namelist", "anything", "new")):
            value = " ".join(words)
            if kind.startswith("new"):
                data = _data(session)
                taken = [w for w in words if w in data.columns]
                if taken:
                    raise StataExprError(f"variable {taken[0]} already defined")
        else:
            data = _data(session)
            cols = [str(c) for c in data.columns]
            if words:
                found = expand_varlist(cols, words)
                missing = [v for v in found if v not in cols]
                if missing:
                    raise StataExprError(f"variable {missing[0]} not found")
            elif optional and "default=none" not in detail.replace(" ", ""):
                found = cols
            else:
                found = []
            if "numeric" in detail:
                bad = [v for v in found if not pd.api.types.is_numeric_dtype(data[v])]
                if bad:
                    raise StataExprError(
                        f"string variables not allowed in varlist; {bad[0]}"
                    )
            if "string" in detail.split():
                bad = [v for v in found if pd.api.types.is_numeric_dtype(data[v])]
                if bad:
                    raise StataExprError(
                        f"numeric variables not allowed in varlist; {bad[0]}"
                    )
            low, high = (1, 1) if "name" in kind else (1, None)
            got = re.search(r"\bmin\s*=\s*(\d+)", detail)
            if got:
                low = int(got.group(1))
            got = re.search(r"\bmax\s*=\s*(\d+)", detail)
            if got:
                high = int(got.group(1))
            if not found and not optional:
                raise StataExprError("varlist required")
            if found and (len(found) < low or (high is not None and len(found) > high)):
                raise StataExprError(
                    "too few variables specified"
                    if len(found) < low
                    else "too many variables specified"
                )
            value = " ".join(found)
        if not value and not optional:
            raise StataExprError(f"{target} required")
        locals_[target] = value
        remainder = head_spec[: element.start()] + head_spec[element.end() :]
    elif words:
        raise StataExprError("varlist not allowed")
    # --- if, in, using, weight
    allowed = remainder.lower()
    for key, given, shown in (
        ("if", cmd.if_cond, f"if {cmd.if_cond}"),
        ("in", cmd.in_range, f"in {cmd.in_range}"),
    ):
        if given and not re.search(rf"\b{key}\b", allowed):
            raise StataExprError(f"{key} not allowed")
        locals_[key] = shown if given else ""
    if using is not None and "using" not in allowed:
        raise StataExprError("using not allowed")
    if "using" in allowed:
        locals_["using"] = f"using {using}" if using else ""
    kinds = re.findall(r"\b([fapi]w(?:eights?)?)\b", allowed)
    if weight is not None:
        m = re.match(r"\[\s*(\w+)\s*=\s*(.*?)\s*\]\Z", weight, re.S)
        if not kinds or m is None:
            raise StataExprError("weights not allowed")
        full = {"fw": "fweight", "aw": "aweight", "pw": "pweight", "iw": "iweight"}
        kind = full.get(m.group(1).lower()[:2], m.group(1).lower())
        if kind[:2] not in {k[:2] for k in kinds}:
            raise StataExprError(f"{kind}s not allowed")
        locals_["weight"], locals_["exp"] = kind, f"= {m.group(2)}"
    elif kinds:
        locals_["weight"], locals_["exp"] = "", ""
    # --- options
    options = dict(cmd.options)
    specs = _option_specs(option_spec, comma_optional and "[" not in option_spec)
    star = any(s.get("star") for s in specs)
    for s in specs:
        if s.get("star"):
            continue
        name = s["name"]
        given = _given(options, s)
        negated = _given(options, s, "no") if s["negatable"] else None
        if s["negatable"]:
            locals_[name] = "no" + name if negated is not None else ""
            options.pop(negated, None) if negated else None
            options.pop(given, None) if given else None
            continue
        if s["argument"] is None:
            if given is not None and options[given] is not None:
                raise StataExprError(f"option {name}() not allowed")
            locals_[name] = name if given is not None else ""
            options.pop(given, None) if given else None
            continue
        kind, _, default = s["argument"].strip().partition(" ")
        kind, default = kind.lower(), default.strip()
        if given is None or options[given] is None:
            if given is not None:
                raise StataExprError(f"option {name}() requires an argument")
            if not s["optional"] and not default and kind not in ("integer", "real"):
                raise StataExprError(f"option {name}() required")
            if kind in ("integer", "real") and not default and not s["optional"]:
                raise StataExprError(f"option {name}() required")
            locals_[name] = default if kind in ("integer", "real", "cilevel") else ""
            continue
        raw = str(options.pop(given)).strip()
        if kind in ("integer", "real"):
            try:
                number = float(raw)
            except ValueError:
                raise StataExprError(f"option {name}() incorrectly specified") from None
            if kind == "integer" and number != int(number):
                raise StataExprError(f"option {name}() incorrectly specified")
            locals_[name] = raw
        elif kind in ("varname", "varlist"):
            data = _data(session)
            cols = [str(c) for c in data.columns]
            found = expand_varlist(cols, raw.split())
            missing = [v for v in found if v not in cols]
            if missing or (kind == "varname" and len(found) != 1):
                raise StataExprError(f"option {name}(): variable {raw} not found")
            locals_[name] = " ".join(found)
        elif kind == "passthru":
            locals_[name] = f"{name}({raw})"
        elif kind in ("string", "str", "name", "namelist", "numlist", "asis",
                      "newvarname", "newvarlist"):  # fmt: skip
            locals_[name] = _unquote(raw) if kind in ("string", "str") else raw
        else:
            raise StataExprError(f"syntax: the option type {kind!r} is not implemented")
    rest = " ".join(k if v is None else f"{k}({v})" for k, v in options.items())
    if options and not star:
        raise StataExprError(f"option {next(iter(options))} not allowed")
    if star:
        locals_["options"] = rest
    del line


def _marksample(session: "StataSession", rest: str) -> None:
    """``marksample touse [, novarlist strok zeroweight]``: 1 where the
    row is in the ``if`` / ``in`` sample of the command line and none of
    the variables of `varlist' is missing."""
    name, _, option_text = rest.partition(",")
    name = name.strip()
    options = option_text.split()
    locals_ = session._macros.locals
    steps = session._steps
    if steps is None or not name:
        raise StataExprError("marksample: expected `marksample name`")
    data = steps.data
    if_cond = (locals_.get("if") or "")[3:].strip() or None
    in_range = (locals_.get("in") or "")[3:].strip() or None
    mask = row_mask(data, if_cond, in_range, session.stored)
    if "novarlist" not in options:
        for v in (locals_.get("varlist") or "").split():
            col = data[v]
            if pd.api.types.is_numeric_dtype(col):
                mask &= col.notna().to_numpy()
            elif "strok" not in options:
                mask[:] = False
    if (locals_.get("weight") or "") and "zeroweight" not in options:
        from ._stata_expr import evaluate

        w = evaluate((locals_.get("exp") or "= 1")[1:], data, session.stored)
        mask &= ~np.isnan(w) & (w != 0)
    session._temp_count += 1
    column = f"__tempvar{session._temp_count:06d}"
    locals_[name] = column
    steps._own()
    steps.data[column] = mask.astype(float)


# ---------------------------------------------------------- command lines
_TOKENIZE = re.compile(r"\s*tokeni[sz]e\b\s*(.*)\Z", re.S | re.I)
_GETTOKEN = re.compile(
    r"\s*gettoken\s+(\(?(?:local|global)?\)?\s*\w+)(?:\s+(\w+))?\s*:\s*(\w+)"
    r"\s*(?:,(.*))?\Z",
    re.S | re.I,
)
_SHIFT = re.compile(r"\s*mac(?:ro)?\s+shift\s*(\d+)?\s*\Z", re.I)
_RETURN_LOCAL = re.compile(r"\s*(e)?return\s+loc(?:al)?\s+(\w+)\s*(.*)\Z", re.S | re.I)
_EXIT = re.compile(r"\s*exit\b\s*(\d+)?\s*(?:,.*)?\Z", re.I)
_SCALAR_DROP = re.compile(r"\s*sca(?:lar)?\s+drop\s+(.+)\Z", re.I)
_MATRIX_LIST = re.compile(r"\s*mat(?:rix)?\s+l(?:ist)?\s+e\((b|V)\)\s*(?:,.*)?\Z", re.I)
_IF_LINE = re.compile(r"\s*if\s+(.+)\Z", re.S | re.I)
_ELSE_LINE = re.compile(r"\s*else\s+(?!if\b)(.+)\Z", re.S | re.I)
_SYNTAX = re.compile(r"\s*syntax\b\s*(.*)\Z", re.S | re.I)
_MARKSAMPLE = re.compile(r"\s*marksample\s+(.+)\Z", re.I)
_EXTENDED = re.compile(
    r"\s*(gl(?:o(?:b(?:al?)?)?)?|loc(?:al?)?)\s+([A-Za-z_]\w*)\s*:\s*(.+)\Z",
    re.S | re.I,
)


def _parse_characters(option_text: Optional[str]) -> Optional[str]:
    if not option_text:
        return None
    m = re.search(r"parse\(\s*(`\"[^`]*?\"'|\"[^\"]*\")\s*\)", option_text)
    return _unquote(m.group(1)) if m else None


def _tokens(text: str, parse: Optional[str]) -> List[str]:
    if parse is None:
        return _words(text)
    out, current = [], ""
    separators = set(parse)
    for ch in text:
        if ch in separators:
            if current:
                out.append(current)
            if ch != " ":
                out.append(ch)
            current = ""
        else:
            current += ch
    if current:
        out.append(current)
    return out


def _one_line_if(session: "StataSession", body: str) -> Optional[Tuple[str, str]]:
    """Split ``exp command`` where the expression ends and a command
    begins: the shortest prefix that is an expression and is followed by
    a word."""
    if body.rstrip().endswith("{"):
        return None
    tokens = re.findall(r'`"[^`]*?"\'|"[^"]*"|\S+', body)
    for k in range(1, len(tokens)):
        nxt = tokens[k]
        if not re.fullmatch(r"[A-Za-z_#][\w.]*:?", nxt):
            continue
        if re.fullmatch(r"[-+*/^&|<>=!~(,]+|.*[-+*/^&|<>=!~(,]", tokens[k - 1]):
            continue
        expr = " ".join(tokens[:k])
        try:
            session.value(session._macros.expand(expr))
        except (StataExprError, ScriptError):
            continue  # not an expression yet: try a longer one
        return expr, " ".join(tokens[k:])
    return None


def tool_line(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``line`` if it is one of the tools of this module."""
    locals_ = session._macros.locals
    m = _EXTENDED.match(line)
    if m:
        try:
            body = session._macros.expand(m.group(3))
        except ScriptError as exc:
            raise StataExprError(str(exc)) from None
        table = (
            session._macros.globals if m.group(1).lower().startswith("g") else locals_
        )
        try:
            table[m.group(2)] = extended_function(session, body)
        except StataExprError as exc:
            if "is not implemented" not in str(exc):
                raise
            # a function only a running Stata can answer (`: dir`,
            # `: sysdir`): the macro table records the macro as unknown and
            # refuses the line that uses it
            return None
        return False
    if session._program_depth:
        m = _SYNTAX.match(line)
        if m:
            _syntax(session, m.group(1), line)
            return False
        m = _MARKSAMPLE.match(line)
        if m:
            _marksample(session, m.group(1))
            return False
        m = _SHIFT.match(line)
        if m:
            by = int(m.group(1) or 1)
            held = []
            k = 1
            while str(k) in locals_:
                held.append(locals_.pop(str(k)))
                k += 1
            for i, value in enumerate(held[by:], 1):
                locals_[str(i)] = value
            return False
        m = _RETURN_LOCAL.match(line)
        if m:
            value = _unquote(session._macros.expand(m.group(3)))
            session.stored.setdefault("returned_macros", {})[m.group(2)] = value
            return False
        m = _EXIT.match(line)
        if m:
            if m.group(1) and int(m.group(1)) != 0:
                raise StataExprError(f"the program exits with error r({m.group(1)})")
            raise ProgramExit()
    m = _TOKENIZE.match(line)
    if m:
        body, _, option_text = m.group(1).partition(",")
        if '"' in body and "," in m.group(1) and m.group(1).count('"') % 2 == 0:
            body, option_text = m.group(1), ""
            cut = re.search(r",\s*parse\(", m.group(1))
            if cut:
                body, option_text = (
                    m.group(1)[: cut.start()],
                    m.group(1)[cut.start() + 1 :],
                )
        body = session._macros.expand(body)
        k = 1
        while str(k) in locals_:
            del locals_[str(k)]
            k += 1
        for i, token in enumerate(
            _tokens(_unquote(body), _parse_characters(option_text)), 1
        ):
            locals_[str(i)] = token
        return False
    m = _GETTOKEN.match(line)
    if m:
        first = m.group(1).split()[-1].strip("()")
        source = locals_.get(m.group(3)) or ""
        if first == "0" or m.group(3) == "0":
            source = locals_.get("0") or ""
        parse = _parse_characters(m.group(4))
        stripped = source.lstrip()
        if parse is None:
            found = re.match(r'`"[^`]*?"\'|"[^"]*"|\S+', stripped)
            token = found.group(0) if found else ""
            rest = stripped[len(token) :]
            token = _unquote(token)
        else:
            tokens = _tokens(stripped, parse)
            token = tokens[0] if tokens else ""
            rest = stripped[len(token) :] if stripped.startswith(token) else ""
        locals_[first] = token
        if m.group(2):
            locals_[m.group(2)] = rest
        return False
    m = _SCALAR_DROP.match(line)
    if m:
        scalars = session.stored.get("scalars", {})
        names = m.group(1).split()
        if names == ["_all"]:
            scalars.clear()
        for name in names:
            scalars.pop(name, None)
        return False
    m = _MATRIX_LIST.match(line)
    if m and session.last is not None:
        params = getattr(session.last, "params")
        if m.group(1) == "b":
            session.output = pd.DataFrame(
                [np.asarray(params, dtype=float)],
                index=["y1"],
                columns=list(params.index),
            )
        else:
            cov = getattr(session.last, "vcov", None)
            cov = cov() if callable(cov) else cov
            session.output = pd.DataFrame(np.asarray(cov, dtype=float),
                                          index=list(params.index),
                                          columns=list(params.index))  # fmt: skip
        return True
    m = _ELSE_LINE.match(line)
    if m and not line.rstrip().endswith("{"):
        taken = session._if_taken
        if taken is None:
            raise StataExprError("`else` without an `if` before it")
        session._if_taken = None
        if taken:
            return False
        return bool(session.run(m.group(1)))
    m = _IF_LINE.match(line)
    if m:
        split = _one_line_if(session, m.group(1))
        if split is None:
            return None
        expr, command = split
        truth = session.value(session._macros.expand(expr))
        taken = bool(truth != 0)  # a missing value is true, as in Stata
        produced = bool(session.run(command)) if taken else False
        session._if_taken = taken
        return produced
    return None


# ------------------------------------------------------------------ display
_DIRECTIVE = re.compile(
    r"_n(?:ewline)?(?:\((\d+)\))?\b|_s(?:kip)?\((\d+)\)|_col(?:umn)?\((\d+)\)"
    r"|_c(?:ontinue)?\b|_dup\((\d+)\)|_request\(\w+\)"
    r"|as\s+(?:text|txt|result|res|error|err|input|inp)\b"
    r"|in\s+(?:smcl|red|green|yellow|white|blue)\b|,",
    re.I,
)
_FORMAT_ITEM = re.compile(r"%-?[\d.,]*(?:t[dcCwmqhy]\S*|[a-zA-Z]+)")
_STRING_ITEM = re.compile(r'`"(.*?)"\'|"([^"]*)"', re.S)


def _expression_end(body: str, start: int) -> int:
    """Where an expression item of ``display`` ends: at the next string,
    format or directive outside parentheses and brackets."""
    depth, pos = 0, start
    while pos < len(body):
        ch = body[pos]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and pos > start and body[pos - 1].isspace():
            if ch == '"' or body.startswith('`"', pos) or ch == "%":
                break
            m = _DIRECTIVE.match(body, pos)
            if m is not None and m.group(0) != ",":
                break
        elif ch == '"' and depth > 0:
            close = body.find('"', pos + 1)
            pos = close if close > 0 else pos
        pos += 1
    return pos


def display_items(session: "StataSession", body: str) -> List[Any]:
    """What ``display body`` shows, item by item: text as ``str``, the
    value of a numeric expression as ``(value, format)``."""
    from ._stata_expr import evaluate

    data = session.data
    if data is None or data.empty:
        frame = pd.DataFrame({"_": [0.0]})
    else:
        frame = data if re.search(r"\b_[nN]\b|\[", body) else data.iloc[:1]

    def value_of(text: str) -> Any:
        out = evaluate(text, frame, session.stored)
        return out[0]

    try:
        whole = value_of(body)
        return [whole if isinstance(whole, str) else (float(whole), None)]
    except StataExprError:
        pass
    items: List[Any] = []
    pos, fmt = 0, None
    while pos < len(body):
        if body[pos].isspace():
            pos += 1
            continue
        m = _STRING_ITEM.match(body, pos)
        if m is not None:
            items.append(m.group(1) if m.group(1) is not None else m.group(2))
            pos = m.end()
            continue
        m = _FORMAT_ITEM.match(body, pos)
        if m is not None:
            fmt, pos = m.group(0), m.end()
            continue
        m = _DIRECTIVE.match(body, pos)
        if m is not None:
            if m.group(0).lower().startswith("_n"):
                items.append("\n" * int(m.group(1) or 1))
            elif m.group(2):
                items.append(" " * int(m.group(2)))
            elif m.group(3):
                shown = "".join(i if isinstance(i, str) else "" for i in items)
                width = len(shown.rsplit("\n", 1)[-1])
                items.append(" " * max(int(m.group(3)) - 1 - width, 0))
            pos = m.end()
            continue
        end = _expression_end(body, pos)
        text = body[pos:end].strip()
        value = value_of(text)
        items.append(value if isinstance(value, str) else (float(value), fmt))
        fmt, pos = None, end
    return items


def display_text(items: List[Any]) -> str:
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        else:
            value, fmt = item
            out.append(stata_format(value, fmt or "%10.0g"))
    return "".join(out)
