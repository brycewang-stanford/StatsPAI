"""MCP ``completion/complete``: argument suggestions for prompts and
resource templates.

The protocol offers completion for two kinds of reference only: a prompt
argument (``ref/prompt``) and a variable of a resource URI template
(``ref/resource``). Tool arguments are not completable in MCP, so column
names are out of scope here; what the server can enumerate is

* function names, for ``statspai://function/{name}``;
* estimator families, for ``statspai://guide/{family}``;
* live handles, for ``statspai://data/{id}``, ``statspai://result/{id}``
  and its ``/bundle`` form, and for the ``result_id`` / ``data_id``
  arguments of the prompts that take one.

Any other argument of a known reference completes to an empty list,
which is a valid answer. An unknown prompt or template is ``-32602``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

#: The protocol caps one response at 100 values.
MAX_VALUES = 100


def _rank(candidates: List[str], typed: str) -> List[str]:
    """Prefix matches first, then substring matches; case-insensitive."""
    needle = typed.strip().lower()
    names = sorted(set(candidates))
    if not needle:
        return names
    head = [c for c in names if c.lower().startswith(needle)]
    tail = [c for c in names if needle in c.lower() and c not in set(head)]
    return head + tail


def _function_names() -> List[str]:
    from ._resources import functions_index

    return [str(f["name"]) for f in functions_index()]


def _families() -> List[str]:
    from .._routing import FAMILIES

    return list(FAMILIES)


def _result_ids() -> List[str]:
    from ._result_cache import RESULT_CACHE

    return list(RESULT_CACHE.keys())


def _data_ids() -> List[str]:
    from ._data_cache import DATA_CACHE

    return list(DATA_CACHE.keys())


def _template_sources() -> Dict[str, Dict[str, Callable[[], List[str]]]]:
    from ._resources import (
        BUNDLE_SUFFIX,
        DATA_URI_PREFIX,
        FUNCTION_URI_PREFIX,
        GUIDE_URI_PREFIX,
        RESULT_URI_PREFIX,
    )

    return {
        FUNCTION_URI_PREFIX + "{name}": {"name": _function_names},
        GUIDE_URI_PREFIX + "{family}": {"family": _families},
        DATA_URI_PREFIX + "{id}": {"id": _data_ids},
        RESULT_URI_PREFIX + "{id}": {"id": _result_ids},
        RESULT_URI_PREFIX + "{id}" + BUNDLE_SUFFIX: {"id": _result_ids},
    }


def handle_completion_complete(
    params: Dict[str, Any], *, InvalidParamsError: Type[Exception]
) -> Dict[str, Any]:
    ref = params.get("ref")
    argument = params.get("argument")
    if not isinstance(ref, dict) or not isinstance(argument, dict):
        raise InvalidParamsError("`ref` and `argument` must be objects")
    arg_name = argument.get("name")
    typed = argument.get("value", "")
    if not isinstance(arg_name, str) or not isinstance(typed, str):
        raise InvalidParamsError("`argument.name` and `argument.value` must be strings")

    kind = ref.get("type")
    candidates: List[str] = []
    if kind == "ref/prompt":
        from ._prompts import PROMPTS

        prompt = next((p for p in PROMPTS if p["name"] == ref.get("name")), None)
        if prompt is None:
            raise InvalidParamsError(f"Unknown prompt: {ref.get('name')!r}")
        declared = {a["name"] for a in prompt.get("arguments") or []}
        if arg_name == "result_id" and arg_name in declared:
            candidates = _result_ids()
        elif arg_name == "data_id" and arg_name in declared:
            candidates = _data_ids()
    elif kind == "ref/resource":
        sources = _template_sources().get(str(ref.get("uri")))
        if sources is None:
            raise InvalidParamsError(f"Unknown resource template: {ref.get('uri')!r}")
        source = sources.get(arg_name)
        if source is not None:
            candidates = source()
    else:
        raise InvalidParamsError(
            "`ref.type` must be 'ref/prompt' or 'ref/resource'; " f"got {kind!r}"
        )

    ranked = _rank(candidates, typed)
    return {
        "completion": {
            "values": ranked[:MAX_VALUES],
            "total": len(ranked),
            "hasMore": len(ranked) > MAX_VALUES,
        }
    }
