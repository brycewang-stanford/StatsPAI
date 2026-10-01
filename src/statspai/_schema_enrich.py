"""Schema enrichment for the agent-facing registry export.

The registry (:mod:`statspai.registry`) turns every public callable into a
JSON-schema tool description. A 2026-09-28 audit of the committed
``schemas/*.json`` bundle found that the *shape* of that description was
thinner than the code it describes:

* option-like string parameters (``method`` / ``kernel`` / ``vce`` …) had no
  ``enum`` even when the signature says ``Literal[...]`` or the docstring
  lists the choices;
* column-name parameters (``y`` / ``treat`` / ``id`` / ``covariates`` …)
  were indistinguishable from free text, and ``data`` was typed ``string``;
* legacy spellings (``robust`` / ``unit`` / ``outcome``) were exported with
  no pointer to the house-style name the function also accepts;
* array parameters with numeric defaults (``propensity_bounds=[0.05,
  0.95]``) declared ``string`` items;
* nothing said what a call *returns*.

This module holds the pure helpers that close those gaps. Everything here is
derived from the code itself — annotations, docstrings, the ``@accepts_aliases``
record, the return annotation — and a derived value is only emitted when it is
grounded (an ``enum`` from a docstring must contain the default and every value
must appear as a string literal in the function's own module). Nothing here
changes a numerical output: it only describes the call surface.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, TypeVar

# --------------------------------------------------------------------------
# Option-like parameters (the set the enum-coverage ratchet measures)
# --------------------------------------------------------------------------

#: Parameter names that select among a closed set of string choices in most
#: of the public API. ``tests/test_discovery_schema.py`` ratchets the share of
#: string-typed parameters with these names that carry an ``enum``.
OPTION_PARAM_NAMES = frozenset(
    {
        "method",
        "kernel",
        "vce",
        "model",
        "estimator",
        "robust",
        "vcov",
        "se_type",
        "cov_type",
        "variant",
        "solver",
        "bwselect",
        "alternative",
        "se_method",
        "inference",
        "weights_type",
        "aggregation",
        "agg",
        "type",
        "how",
        "test",
        "criterion",
        "link",
        "family",
        "distribution",
        "loss",
        "penalty",
        "base_period",
        "control_group",
        "est_method",
        "approach",
        "backend",
        "engine",
        "strategy",
        "norm",
        "score",
        "algorithm",
        "mode",
        "design",
        "target",
        "metric",
        "ci_method",
        "variance",
        "weighting",
        "spec",
    }
)

# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

_LITERAL_TEXT_RE = re.compile(r"Literal\[([^\[\]]*)\]")
_QUOTED_RE = re.compile(r"""['"]([^'"]*)['"]""")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:+\-]+")


def literal_choices(annotation: Any) -> Optional[List[str]]:
    """String choices of a ``Literal[...]`` annotation (``Optional`` allowed).

    Accepts a live ``typing`` object or its string form (modules using
    ``from __future__ import annotations`` leave annotations as text).
    Returns ``None`` unless every non-``None`` member is a string.
    """
    if annotation is None or annotation is inspect.Parameter.empty:
        return None
    text = annotation if isinstance(annotation, str) else None
    if text is None:
        try:
            import typing

            origin = typing.get_origin(annotation)
            args = typing.get_args(annotation)
        except Exception:  # pragma: no cover - exotic annotation objects
            origin, args = None, ()
        if origin is typing.Literal:
            vals = [a for a in args if a is not None]
            if vals and all(isinstance(v, str) for v in vals):
                return list(dict.fromkeys(vals))
            return None
        if args:
            # Optional[Literal[...]] / Union[Literal[...], None]
            out: List[str] = []
            for a in args:
                if a is type(None):
                    continue
                sub = literal_choices(a)
                if sub is None:
                    return None
                out.extend(sub)
            return list(dict.fromkeys(out)) or None
        text = str(annotation)
    matches = _LITERAL_TEXT_RE.findall(text)
    if not matches:
        return None
    # Any non-Literal, non-None member of the union means the parameter also
    # accepts free values; only a pure Literal (optionally Optional) is closed.
    residual = _LITERAL_TEXT_RE.sub("", text)
    residual = re.sub(r"typing\.|Optional|Union|None|NoneType|[\[\],|\s]", "", residual)
    if residual:
        return None
    out = []
    for body in matches:
        items = [s.strip() for s in body.split(",") if s.strip()]
        for item in items:
            q = _QUOTED_RE.fullmatch(item)
            if not q:
                if item == "None":
                    continue
                return None
            out.append(q.group(1))
    out = list(dict.fromkeys(out))
    return out if len(out) >= 1 else None


# Quoted item in a docstring: 'a', "a", ``'a'``, ``"a"`` or ``a``.
_ITEM = (
    r"(?:``)?(?:'([A-Za-z0-9_.:+\-]+)'|\"([A-Za-z0-9_.:+\-]+)\""
    r"|``([A-Za-z0-9_.:+\-]+)``)(?:``)?"
)
_ITEM_RE = re.compile(_ITEM)
_SEP_RE = re.compile(r"\s*(?:,\s*or\s+|,\s*|\s+or\s+|\s*/\s*|\s*\|\s*)")
_PAREN_RE = re.compile(r"\s*\([^()]{0,160}\)")
_MARKER_RE = re.compile(
    r"\b(?:one of|options?(?: are)?|either|must be(?: one of)?|choices?(?: are)?|"
    r"supported(?: values)?(?: are)?|valid(?: values)?(?: are)?)\s*:?\s*",
    re.I,
)
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ,/()\-']{0,60}:\s+")
#: Phrases that signal an open (non-exhaustive) value set. The ellipsis is
#: written as the escape ``\u2026`` (``re`` resolves it): the JSS archive
#: transliterates non-ASCII source to ASCII, which turned a literal one into an
#: unescaped ``...`` that matched any three characters and dropped every enum.
_OPEN_SET_RE = re.compile(
    r"e\.g\.|\betc\b|such as|for example|\bany\b|\.\.\.|\u2026|\balias|also accept|"
    r"\bor an? \b|callable|\bdict\b|\bcustom\b|\blike\b|including",
    re.I,
)


def _item_value(m: "re.Match[str]") -> str:
    return next(g for g in m.groups() if g is not None)


def _consume_list(text: str, pos: int) -> Tuple[List[str], int, bool]:
    """Read ``'a', 'b' or 'c'`` starting at ``pos``.

    Returns ``(items, end, saw_or)``. Parenthetical glosses after an item
    (``'ols' (equation-by-equation)``) are skipped.
    """
    items: List[str] = []
    saw_or = False
    i = pos
    while True:
        m = _ITEM_RE.match(text, i)
        if not m:
            break
        items.append(_item_value(m))
        i = m.end()
        p = _PAREN_RE.match(text, i)
        if p:
            i = p.end()
        s = _SEP_RE.match(text, i)
        if not s or s.end() == i:
            break
        if "or" in s.group(0):
            saw_or = True
        if not _ITEM_RE.match(text, s.end()):
            break
        i = s.end()
    return items, i, saw_or


def doc_choices(description: str) -> Optional[Tuple[List[str], str]]:
    """Choice list stated by a parameter description, with how it was found.

    Recognised shapes (whitespace already collapsed):

    * a marker — ``One of 'a', 'b'`` / ``Options: "a", "b"`` / ``either
      'a' or 'b'``;
    * a leading list closed by ``or`` — ``Kernel function: 'triangular',
      'uniform', or 'epanechnikov'.``;
    * a description that *is* the list — ``'individual', 'cluster',
      'stratified'.``;
    * a bullet list — ``- ``'mserd'`` : ... - ``'msetwo'`` : ...``.

    Descriptions that signal an open set (``e.g.``, ``etc``, ``alias`` …)
    return ``None``.
    """
    text = re.sub(r"\s+", " ", description or "").strip()
    if not text or _OPEN_SET_RE.search(text):
        return None
    # Marker form.
    m = _MARKER_RE.search(text)
    if m:
        items, _end, _or = _consume_list(text, m.end())
        if len(items) >= 2:
            return items, "marker"
        bullets = _bullet_items(text[m.end() :])
        if len(bullets) >= 2:
            return bullets, "bullets"
    # Leading list (optionally after a short ``Label:``).
    start = 0
    lab = _LABEL_RE.match(text)
    if lab:
        start = lab.end()
    items, end, saw_or = _consume_list(text, start)
    rest = text[end:].strip()
    if len(items) >= 2 and (saw_or or rest in ("", ".")):
        if rest in ("", ".") or rest[:1] in (".", ";"):
            return items, "list"
    bullets = _bullet_items(text[start:])
    if len(bullets) >= 2 and text[start:].lstrip().startswith(("-", "*")):
        return bullets, "bullets"
    return None


_BULLET_MARK_RE = re.compile(r"(?:^|\s)[-*]\s+")


def _bullet_items(text: str) -> List[str]:
    """Choices introduced by ``- ``'a'``[, ``'b'`` or ``'c'``] : ...`` bullets."""
    out: List[str] = []
    for m in _BULLET_MARK_RE.finditer(text):
        items, _end, _or = _consume_list(text, m.end())
        out.extend(items)
    return list(dict.fromkeys(out))


@lru_cache(maxsize=512)
def _module_source(module_name: str) -> str:
    mod = sys.modules.get(module_name)
    if mod is None:
        return ""
    try:
        return inspect.getsource(mod)
    except (OSError, TypeError):
        return ""


def _function_source_blob(obj: Any) -> str:
    """Source text of ``obj``'s defining module (and of its unwrapped target)."""
    blobs: List[str] = []
    seen = set()
    for cand in (obj, inspect.unwrap(obj) if callable(obj) else obj):
        mod = getattr(cand, "__module__", None)
        if mod and mod not in seen:
            seen.add(mod)
            blobs.append(_module_source(mod))
    return "\n".join(blobs)


def source_confirms(obj: Any, values: List[str]) -> bool:
    """True when every value appears as a quoted literal in ``obj``'s module.

    A cheap grounding check for docstring-derived enums: a choice the code
    never spells is either a docstring typo or handled somewhere we cannot
    see, and in both cases advertising it as a closed set would be a guess.
    """
    blob = _function_source_blob(obj).lower()
    if not blob:
        return False
    for v in values:
        lv = v.lower()
        if f"'{lv}'" not in blob and f'"{lv}"' not in blob:
            return False
    return True


#: Standard-error keywords: never given a docstring-derived enum.
SE_PARAM_NAMES = frozenset(
    {"robust", "vce", "vcov", "se_type", "cov_type", "vcov_type"}
)


def grounded_doc_enum(
    description: str, default: Any, required: bool, obj: Any, param: str = ""
) -> Optional[List[str]]:
    """Docstring enum that passes every confidence gate, else ``None``.

    Gates: at least two choices; a string default must be one of them (and a
    ``None`` default is only accepted with an explicit ``one of`` / bullet
    marker); every choice is spelled as a string literal in the function's
    module source.
    """
    found = doc_choices(description)
    if not found:
        return None
    if param in SE_PARAM_NAMES:
        # SE keywords go through the shared variance parser, which accepts
        # more spellings (``hc2`` / ``cluster <var>`` …) than a docstring
        # typically lists; only a ``Literal`` annotation is trusted there.
        return None
    items, how = found
    if len(items) < 2:
        return None
    if isinstance(default, str):
        if default not in items:
            return None
    elif default is None and not required:
        if how not in ("marker", "bullets"):
            return None
    elif not required:
        return None
    if obj is None or not source_confirms(obj, items):
        return None
    return items


# Dispatchers whose accepted values live in a registry the package exposes.
# ``(function, parameter) -> zero-arg loader``; every loader is cheap (no
# heavy imports) and returns the full accepted set, aliases included.


def _synth_methods() -> List[str]:
    from .synth import scm as _scm

    src = inspect.getsource(_scm._dispatch_synth_impl)
    vals: List[str] = []
    for body in re.findall(
        r"\bmethod\s*(?:==|in)\s*(\([^)]*\)|\"[^\"]+\"|'[^']+')", src
    ):
        vals.extend(_QUOTED_RE.findall(body))
    return list(dict.fromkeys(vals))


def _decompose_methods() -> List[str]:
    import statspai as sp

    return list(sp.available_methods())


def _mr_methods() -> List[str]:
    from .mendelian import mr_available_methods

    return list(mr_available_methods())


def _interference_designs() -> List[str]:
    from .interference import interference_available_designs

    return list(interference_available_designs())


DISPATCHER_ENUMS: Dict[Tuple[str, str], Any] = {
    ("synth", "method"): _synth_methods,
    ("decompose", "method"): _decompose_methods,
    ("mr", "method"): _mr_methods,
    ("interference", "design"): _interference_designs,
}


def dispatcher_enum(func: str, param: str, default: Any) -> Optional[List[str]]:
    loader = DISPATCHER_ENUMS.get((func, param))
    if loader is None:
        return None
    try:
        vals = [str(v) for v in loader()]
    except (ImportError, AttributeError, OSError, TypeError):
        return None
    if len(vals) < 2:
        return None
    if isinstance(default, str) and default not in vals:
        return None
    return vals


# --------------------------------------------------------------------------
# Column roles
# --------------------------------------------------------------------------

_DATAFRAME_NAMES = frozenset({"data", "df", "frame", "dataset"})
_FORMULA_NAMES = frozenset({"formula", "fml"})
_COLUMN_NAMES = frozenset(
    {
        "y",
        "outcome",
        "depvar",
        "dependent",
        "yname",
        "treat",
        "treatment",
        "treatvar",
        "treat_var",
        "d_var",
        "id",
        "unit",
        "idname",
        "panel_id",
        "ivar",
        "entity",
        "time",
        "tname",
        "period",
        "cluster",
        "clustervar",
        "running",
        "running_var",
        "instrument",
        "mediator",
        "gname",
        "cohort",
        "group",
        "strata",
        "first_treat",
        "weight",
        "weights",
        "exposure",
        "offset",
        "event",
        "duration",
        "censor",
        "status",
        "text_col",
    }
)
_COLUMNS_NAMES = frozenset(
    {
        "covariates",
        "controls",
        "covs",
        "xvars",
        "covars",
        "instruments",
        "absorb",
        "exog",
        "endog",
        "confounders",
        "mediators",
        "outcomes",
        "x_cols",
        "feature_cols",
        "unit_covariates",
        "time_covariates",
    }
)
#: Short symbols that are column names only when annotated as strings.
_SHORT_COLUMN_NAMES = frozenset({"x", "d", "m", "g", "t", "i", "z", "w", "s"})


def column_role(name: str, type_text: str, json_type: Any) -> Optional[str]:
    """``"dataframe"`` / ``"formula"`` / ``"column"`` / ``"columns"`` or None.

    Decided from the parameter name (house-style canonical names and their
    registered legacy spellings) and confirmed by the declared type: a
    ``column`` must be string-typed, a ``columns`` must admit a list of
    strings. Single-letter symbols (``x`` / ``d`` / ``t`` …) are column names
    only when the annotation says ``str`` — in the array API they are data.
    """
    types = set(json_type) if isinstance(json_type, list) else {json_type}
    lower_type = (type_text or "").lower()
    if name in _DATAFRAME_NAMES and "dataframe" in lower_type:
        return "dataframe"
    stringish = "string" in types
    arrayish = "array" in types
    if name in _FORMULA_NAMES or name.endswith("_formula"):
        return "formula" if stringish else None
    arraylike_annot = any(
        tok in lower_type for tok in ("ndarray", "array", "series", "dataframe")
    )
    if name in _COLUMNS_NAMES or name.endswith(("_cols", "_vars", "_columns")):
        if arraylike_annot and "str" not in lower_type:
            return None
        if arrayish:
            return "columns"
        if stringish and ("list" in lower_type or "sequence" in lower_type):
            return "columns"
        return "columns" if stringish and "str" in lower_type else None
    if name in _COLUMN_NAMES or name.endswith(("_col", "_var", "_column")):
        if not stringish:
            return None
        if arraylike_annot and "str" not in lower_type:
            return None
        if (
            name in ("weights", "weight", "exposure", "offset")
            and "str" not in lower_type
        ):
            return None
        return "column"
    if name in _SHORT_COLUMN_NAMES:
        if stringish and re.search(r"\bstr\b", lower_type) and not arraylike_annot:
            return "column"
    return None


# --------------------------------------------------------------------------
# Parameter spelling: accepted aliases and house-style canonical names
# --------------------------------------------------------------------------


def param_spelling(func_name: str, obj: Any) -> Dict[str, Dict[str, Any]]:
    """Per-parameter spelling facts for ``obj``.

    Returns ``{param: {"aliases": [...], "canonical": str | None,
    "canonical_accepted": bool}}`` for every signature parameter that has an
    ``@accepts_aliases`` spelling or a house-style canonical name different
    from its own. ``aliases`` are the extra keywords the call accepts for
    that parameter; ``canonical`` is the house-style spelling of the concept
    (``vce`` for ``robust``); ``canonical_accepted`` says whether the call
    also accepts that canonical spelling.
    """
    from ._house_style import canonical_for, is_false_friend

    out: Dict[str, Dict[str, Any]] = {}
    if obj is None:
        return out
    alias_map: Dict[str, str] = dict(getattr(obj, "__statspai_aliases__", {}) or {})
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return out
    module = getattr(inspect.unwrap(obj), "__module__", "") or ""
    by_target: Dict[str, List[str]] = {}
    for alias, target in alias_map.items():
        if target in params and alias not in params:
            by_target.setdefault(target, []).append(alias)
    for pname, p in params.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        aliases = sorted(by_target.get(pname, []))
        canonical = canonical_for(pname)
        if (
            canonical == pname
            or canonical in params
            or len(pname) == 1
            and pname.isupper()
            or is_false_friend(pname, func_name, module)
        ):
            canonical = None
        if canonical and pname == "robust" and isinstance(p.default, bool):
            # ``robust=True`` on/off switch: not an SE-type string.
            canonical = None
        if not aliases and not canonical:
            continue
        out[pname] = {
            "aliases": aliases,
            "canonical": canonical,
            "canonical_accepted": bool(canonical and canonical in aliases),
        }
    return out


# --------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------


def item_type_from_default(default: Any) -> Any:
    """JSON item type implied by a list/tuple default (``[0.05, 0.95]``).

    ``None`` members (``figsize=(8, None)``) make the item type nullable:
    ``["integer", "null"]``.
    """
    if not isinstance(default, (list, tuple)) or not default:
        return None
    kinds = set()
    nullable = False
    for v in default:
        if v is None:
            nullable = True
        elif isinstance(v, bool):
            kinds.add("boolean")
        elif isinstance(v, int):
            kinds.add("integer")
        elif isinstance(v, float):
            kinds.add("number")
        elif isinstance(v, str):
            kinds.add("string")
        else:
            return None
    if kinds == {"integer", "number"}:
        kinds = {"number"}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    return [kind, "null"] if nullable else kind


# --------------------------------------------------------------------------
# Return types
# --------------------------------------------------------------------------

_BUILTIN_RETURN_NAMES = {
    "DataFrame": "DataFrame",
    "pd.DataFrame": "DataFrame",
    "pandas.DataFrame": "DataFrame",
    "Series": "Series",
    "pd.Series": "Series",
    "pandas.Series": "Series",
    "ndarray": "ndarray",
    "np.ndarray": "ndarray",
    "numpy.ndarray": "ndarray",
    "dict": "dict",
    "Dict": "dict",
    "Mapping": "dict",
    "list": "list",
    "List": "list",
    "Sequence": "list",
    "tuple": "tuple",
    "Tuple": "tuple",
    "float": "float",
    "int": "int",
    "bool": "bool",
    "str": "str",
    "Path": "Path",
    "Callable": "callable",
    "Iterator": "iterator",
    "Generator": "iterator",
    "Iterable": "iterable",
    "Figure": "Figure",
    "Axes": "Axes",
}

#: Explicit return class for dispatchers annotated ``Any`` / ``Union`` whose
#: every branch returns the same envelope.
EXPLICIT_RESULT_CLASS: Dict[str, str] = {}


def _resolve_name(text: str, obj: Any) -> Optional[Any]:
    """Resolve a (possibly dotted) annotation name in ``obj``'s globals."""
    target = inspect.unwrap(obj) if callable(obj) else obj
    g = dict(getattr(target, "__globals__", {}) or {})
    mod = sys.modules.get(getattr(target, "__module__", "") or "")
    parts = text.split(".")
    cur: Any = g.get(parts[0])
    if cur is None and mod is not None:
        cur = getattr(mod, parts[0], None)
    if cur is None:
        import statspai as sp

        cur = getattr(sp, parts[0], None)
    if cur is None and len(parts) == 1:
        cur = _statspai_class_named(parts[0])
    for part in parts[1:]:
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


@lru_cache(maxsize=256)
def _statspai_class_named(cname: str) -> Optional[Any]:
    """A class called ``cname`` defined in any loaded ``statspai`` module.

    Covers annotations transplanted onto a wrapper whose own module never
    imported the class (``sp.anderson_rubin_ci`` -> ``WeakIVConfidenceSet``).
    """
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("statspai") or mod is None:
            continue
        obj = getattr(mod, cname, None)
        if inspect.isclass(obj) and getattr(obj, "__module__", "") == modname:
            return obj
    return None


def _outer_name(text: str) -> str:
    return text.split("[", 1)[0].strip().strip("'\"")


def _candidate_names(text: str) -> List[str]:
    """Outer name first, then (for Optional/Union) each member."""
    t = (text or "").replace("typing.", "").strip().strip("'\"")
    outer = _outer_name(t)
    if outer in ("Optional", "Union") and "[" in t:
        inner = t[t.index("[") + 1 : t.rindex("]")]
        depth = 0
        cur = ""
        members = []
        for ch in inner:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if ch == "," and depth == 0:
                members.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            members.append(cur.strip())
        return [
            _outer_name(m) for m in members if m.strip() not in ("None", "NoneType")
        ]
    if "|" in t:
        return [_outer_name(m) for m in t.split("|") if m.strip() not in ("None",)]
    return [outer]


def result_class_for(name: str, obj: Any, returns_text: str = "") -> Optional[Any]:
    """``(class_name, class_object_or_None)`` a call to ``obj`` returns.

    Order: an explicit dispatcher override; the return annotation (resolved
    in the function's module so ``'CausalResult'`` strings work); a StatsPAI
    result class named at the start of the docstring ``Returns`` text.
    """
    if name in EXPLICIT_RESULT_CLASS:
        cname = EXPLICIT_RESULT_CLASS[name]
        import statspai as sp

        return cname, getattr(sp, cname, None)
    if obj is None or inspect.isclass(obj):
        return None
    try:
        ann = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        ann = inspect.Signature.empty
    texts: List[str] = []
    if ann is not inspect.Signature.empty and ann is not None:
        if isinstance(ann, str):
            texts.append(ann)
        elif ann is Any or isinstance(ann, TypeVar):
            texts = []
        elif inspect.isclass(ann):
            texts.append(ann.__module__ + "." + ann.__qualname__)
            if ann.__module__ == "builtins":
                return ann.__name__, None
            if ann.__module__.startswith("statspai"):
                return ann.__name__, ann
            short = ann.__name__
            return _BUILTIN_RETURN_NAMES.get(short, short), None
        else:
            texts.append(str(ann))
    elif ann is None:
        return "None", None
    for text in texts:
        for cand in _candidate_names(text):
            if cand in ("Any", "object", ""):
                continue
            if cand == "None":
                return "None", None
            if cand in _BUILTIN_RETURN_NAMES:
                return _BUILTIN_RETURN_NAMES[cand], None
            resolved = _resolve_name(cand, obj)
            if resolved is Any or isinstance(resolved, TypeVar):
                continue
            if inspect.isclass(resolved):
                modname = getattr(resolved, "__module__", "")
                if modname.startswith("statspai"):
                    return resolved.__name__, resolved
                short = resolved.__name__
                return _BUILTIN_RETURN_NAMES.get(short, short), None
    # Docstring ``Returns`` text: ``CausalResult: ...`` / ``EconometricResults``.
    m = re.match(r"\s*:?(?:class:)?`?~?([A-Za-z_][\w.]*)", returns_text or "")
    if m:
        cand = m.group(1).split(".")[-1]
        import statspai as sp

        cls = getattr(sp, cand, None)
        if inspect.isclass(cls) and cls.__module__.startswith("statspai"):
            return cls.__name__, cls
        if cand in _BUILTIN_RETURN_NAMES:
            return _BUILTIN_RETURN_NAMES[cand], None
    return None


def _type_text(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return "Any"
    if isinstance(ann, str):
        return ann.replace("typing.", "")
    if inspect.isclass(ann) and not getattr(ann, "__args__", None):
        return ann.__name__
    return str(ann).replace("typing.", "")


@lru_cache(maxsize=1024)
def result_fields(cls: Any) -> Tuple[Tuple[str, str], ...]:
    """Public data fields of a result class, as ``(name, type)`` pairs.

    Dataclass fields when the class is a dataclass; otherwise class-level
    annotations across the MRO; otherwise the constructor's parameters
    (StatsPAI result classes store their constructor arguments as
    attributes — ``CausalResult(method, estimand, estimate, se, …)``).
    Private names and ALL-CAPS constants are dropped.
    """
    out: Dict[str, str] = {}
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if not f.name.startswith("_"):
                out[f.name] = _type_text(f.type)
    else:
        for klass in reversed(getattr(cls, "__mro__", (cls,))):
            if klass is object:
                continue
            for k, v in (
                getattr(klass, "__dict__", {}).get("__annotations__") or {}
            ).items():
                if k.startswith("_") or k.isupper():
                    continue
                out[k] = _type_text(v)
        sig = None
        if getattr(cls, "__init__", object.__init__) is not object.__init__:
            try:
                sig = inspect.signature(cls.__init__)
            except (TypeError, ValueError):
                sig = None
        if sig is not None:
            for pname, p in sig.parameters.items():
                if pname == "self" or pname.startswith("_"):
                    continue
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                out.setdefault(pname, _type_text(p.annotation))
    return tuple(out.items())


def _has_agent_payload(cls: Any) -> bool:
    for klass in getattr(cls, "__mro__", ()):
        if klass.__name__ in ("CausalResult", "EconometricResults") and (
            klass.__module__ == "statspai.core.results"
        ):
            return True
    return False


def returns_block(
    name: str, obj: Any, returns_text: str = "", *, cls: Any = None, cname: str = ""
) -> Dict[str, Any]:
    """``{"class", "fields", "description"}`` for the agent schema / card."""
    if not cname:
        found = result_class_for(name, obj, returns_text)
        if found:
            cname, cls = found
    fields: Dict[str, str] = {}
    if cls is not None:
        try:
            fields = dict(result_fields(cls))
        except (TypeError, ValueError):
            fields = {}
        # pandas < 3 renders DataFrame as pandas.core.frame.DataFrame; pin
        # the public path so the bundle does not depend on the pandas version
        # (the same rule the registry applies to signatures).
        from .registry import _canonicalize_annotation_path

        fields = {
            k: _canonicalize_annotation_path(v) if isinstance(v, str) else v
            for k, v in fields.items()
        }
    block: Dict[str, Any] = {
        "class": cname or None,
        "fields": fields,
        "description": returns_text or "",
    }
    if cls is not None and _has_agent_payload(cls):
        # ``result.to_dict(detail='agent')`` is described by the bundled
        # ``result.schema.json`` for the two core envelopes and subclasses.
        block["payload_schema"] = "result.schema.json"
    return block


__all__ = [
    "OPTION_PARAM_NAMES",
    "literal_choices",
    "doc_choices",
    "grounded_doc_enum",
    "dispatcher_enum",
    "column_role",
    "param_spelling",
    "item_type_from_default",
    "result_class_for",
    "result_fields",
    "returns_block",
]
