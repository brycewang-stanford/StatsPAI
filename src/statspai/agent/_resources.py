"""MCP ``resources/*`` plumbing — catalog text, function detail,
result-handle reads, URI templates.

Decoupled from ``mcp_server.py`` so the (large) resource bodies don't
bloat the JSON-RPC dispatch file. The handler functions accept the
JSON encoder + error classes as arguments rather than importing them
from ``mcp_server`` — avoids the circular import that would otherwise
form (mcp_server → _resources → mcp_server).

Public entry points
-------------------

* :data:`FUNCTION_URI_PREFIX` / :data:`RESULT_URI_PREFIX`
* :func:`catalog_text` — markdown catalog body
* :func:`functions_index` — JSON ``[{name, description}, ...]`` list
* :func:`function_detail` — full agent card for one tool name
* :func:`handle_resources_list` — the three top-level resources
* :func:`handle_resources_read` — URI dispatch (catalog / functions /
  function/<name> / result/<id>)
* :func:`handle_resources_templates_list` — the parameterised URI
  templates
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Type

FUNCTION_URI_PREFIX = "statspai://function/"
RESULT_URI_PREFIX = "statspai://result/"

#: Fallback URI for the result-schema resource. ``mcp_server`` owns the
#: canonical :data:`RESULT_SCHEMA_URI` (it must reference it early, before
#: this module is imported), so we read it from there lazily and only fall
#: back to this literal if that import is unavailable.
_RESULT_SCHEMA_URI_FALLBACK = "statspai://schema/result"


def _result_schema_uri() -> str:
    """Canonical result-schema URI (from ``mcp_server`` when importable)."""
    try:
        from .mcp_server import RESULT_SCHEMA_URI

        return RESULT_SCHEMA_URI
    except (ImportError, AttributeError):  # pragma: no cover
        return _RESULT_SCHEMA_URI_FALLBACK


def _result_output_schema() -> Dict[str, Any]:
    """The full documented result envelope served by the schema resource."""
    try:
        from .mcp_server import _RESULT_OUTPUT_SCHEMA

        return _RESULT_OUTPUT_SCHEMA
    except (ImportError, AttributeError):  # pragma: no cover
        return {"type": "object", "additionalProperties": True}


# ----------------------------------------------------------------------
# Catalog / index / detail helpers
# ----------------------------------------------------------------------


def _resource_manifest() -> List[Dict[str, Any]]:
    """Return the cached MCP manifest when available.

    ``mcp_server`` imports this module, so the import stays inside the
    helper to avoid a module-load cycle. Once the server is loaded this
    reuses its static tools/list cache instead of rebuilding the agent
    manifest for resources/read.
    """
    try:
        from .mcp_server import _build_mcp_tools
    except (ImportError, AttributeError):
        from .tools import tool_manifest

        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("input_schema") or {},
            }
            for t in tool_manifest()
        ]
    return _build_mcp_tools()


@lru_cache(maxsize=8)
def catalog_text(server_version: str) -> str:
    """Return a Markdown catalog of every StatsPAI tool."""
    manifest = _resource_manifest()
    lines = [
        "# StatsPAI tool catalog",
        "",
        f"Version: {server_version}. {len(manifest)} tools registered.",
        "",
        "**Per-function detail**: read "
        f"`{FUNCTION_URI_PREFIX}<name>` for the full agent card "
        "(assumptions, failure modes, alternatives, typical_n_min, "
        "example) of any tool listed below.",
        "",
        "**Machine-readable index**: read `statspai://functions` for a "
        "JSON array of `{name, description}` entries.",
        "",
    ]
    for t in manifest:
        lines.append(f"## {t['name']}")
        lines.append("")
        desc = t.get("description", "").strip()
        if desc:
            lines.append(desc)
            lines.append("")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def functions_index() -> List[Dict[str, str]]:
    """Return a JSON-ready ``[{name, description}, …]`` list."""
    return [
        {"name": t["name"], "description": (t.get("description") or "").strip()}
        for t in _resource_manifest()
    ]


@lru_cache(maxsize=256)
def function_detail(name: str) -> Optional[Dict[str, Any]]:
    """Return the rich agent card for one tool, or ``None`` if unknown.

    Prefers ``statspai.registry.agent_card`` (full card with
    assumptions / failure_modes / alternatives / typical_n_min) and
    falls back to the manifest entry for tools that exist in the
    auto-generated layer but lack a hand-curated registry spec.
    """
    # Try the registry first — it has the agent-native metadata.
    try:
        from ..registry import agent_card as _agent_card

        card = _agent_card(name)
        if card:
            return card
    except Exception:
        pass

    # Fallback: synthesise from the merged manifest so any registered
    # tool — even auto-generated ones without a curated spec — still
    # resolves to *something* readable.
    for t in _resource_manifest():
        if t["name"] == name:
            return {
                "name": t["name"],
                "description": (t.get("description") or "").strip(),
                "signature": {
                    "name": t["name"],
                    "description": (t.get("description") or "").strip(),
                    "parameters": (t.get("input_schema") or t.get("inputSchema") or {}),
                },
                "pre_conditions": [],
                "assumptions": [],
                "failure_modes": [],
                "alternatives": [],
                "typical_n_min": None,
                "reference": "",
                "example": "",
            }
    return None


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------


def handle_resources_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """Enumerate the top-level resources only.

    Per-function URIs (``statspai://function/<name>``) are intentionally
    *not* listed — there are 100+ of them and putting each in a client
    UI is noise. The catalog explicitly documents the pattern, and
    ``resources/read`` accepts any valid name on demand.
    """
    return {
        "resources": [
            {
                "uri": "statspai://catalog",
                "name": "StatsPAI estimator catalog",
                "mimeType": "text/markdown",
                "description": "Markdown list of every registered "
                "StatsPAI estimator with its description "
                "and a pointer to the per-function "
                "agent-card URI pattern.",
            },
            {
                "uri": "statspai://functions",
                "name": "StatsPAI tool index (machine-readable)",
                "mimeType": "application/json",
                "description": "JSON array of {name, description} "
                "entries. Read this once during session "
                "setup to enumerate available tools.",
            },
            {
                "uri": _result_schema_uri(),
                "name": "StatsPAI result output schema",
                "mimeType": "application/json",
                "description": "JSON Schema for the agent-facing result "
                "envelope returned by every tools/call "
                "(estimate / std_error / conf_int / method / "
                "diagnostics / violations / next_steps / "
                "citations / error …). Each tool's "
                "outputSchema points here for the full "
                "field-by-field reference.",
            },
        ],
    }


def handle_resources_read(
    params: Dict[str, Any],
    *,
    json_default: Callable[[Any], Any],
    server_version: str,
    InvalidParamsError: Type[Exception],
    ResourceNotFoundError: Type[Exception],
    clean_for_json: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """Dispatch a ``resources/read`` URI to its renderer.

    The encoder + error classes are passed in to avoid a circular
    import through ``mcp_server`` — this module is meant to be a leaf.

    ``clean_for_json`` is the recursive nan/inf scrubber from
    ``mcp_server`` — passed in (rather than imported) for the same
    leaf-module reason as ``json_default``. Falls back to identity when
    the caller doesn't supply one (older callers / legacy tests).
    """
    _clean = clean_for_json if clean_for_json is not None else (lambda x: x)
    uri = params.get("uri")
    if not isinstance(uri, str):
        raise InvalidParamsError(f"`uri` must be a string; got {uri!r}")

    if uri == "statspai://catalog":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": catalog_text(server_version),
                },
            ],
        }
    if uri == "statspai://functions":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(
                        _clean(functions_index()), default=json_default, allow_nan=False
                    ),
                },
            ],
        }
    if uri == _result_schema_uri():
        # The full, documented result envelope. Per-tool ``outputSchema``
        # entries are kept compact to avoid duplicating this ~2.7 KB schema
        # across every tool in tools/list; this resource serves the
        # complete field-by-field reference once, on demand.
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(
                        _clean(_result_output_schema()),
                        default=json_default,
                        allow_nan=False,
                    ),
                },
            ],
        }
    if uri.startswith(FUNCTION_URI_PREFIX):
        name = uri[len(FUNCTION_URI_PREFIX) :]
        if not name:
            raise InvalidParamsError(
                f"Function name is empty in URI {uri!r}; "
                f"expected {FUNCTION_URI_PREFIX}<name>"
            )
        # Embedded slashes are not part of the {name} template — surface
        # the malformed-URI condition as -32602 (invalid params), not
        # -32002 (resource not found), so clients don't auto-retry with
        # a "did you mean" prompt.
        if "/" in name:
            raise InvalidParamsError(
                f"Function name in URI {uri!r} must not contain '/'; "
                f"the URI template is {FUNCTION_URI_PREFIX}{{name}}."
            )
        card = function_detail(name)
        if card is None:
            raise ResourceNotFoundError(
                f"Unknown StatsPAI tool: {name!r}. "
                f"Read statspai://functions for the full index."
            )
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(
                        _clean(card), default=json_default, allow_nan=False
                    ),
                },
            ],
        }

    if uri.startswith(RESULT_URI_PREFIX):
        rid = uri[len(RESULT_URI_PREFIX) :]
        if not rid or "/" in rid:
            raise InvalidParamsError(
                f"Result handle in URI {uri!r} is empty or malformed; "
                f"expected {RESULT_URI_PREFIX}<id>."
            )
        from ._result_cache import RESULT_CACHE

        entry = RESULT_CACHE.get_entry(rid)
        if entry is None:
            raise ResourceNotFoundError(
                f"Result {rid!r} not in server cache. LRU cache evicts "
                f"oldest entries; re-fit with as_handle=true for a "
                f"fresh handle."
            )
        # Render the result the same way an agent would have seen it
        # at fit time: registry-style ``to_dict(detail='agent')`` if
        # available, else a structural summary.
        from .tools import _default_serializer

        try:
            payload = _default_serializer(entry.obj, detail="agent")
        except Exception:  # pragma: no cover — fallback for odd objects
            payload = {"result_class": type(entry.obj).__name__}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        payload["provenance"] = entry.to_metadata()
        payload["result_id"] = rid
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(
                        _clean(payload), default=json_default, allow_nan=False
                    ),
                },
            ],
        }

    raise ResourceNotFoundError(f"Unknown resource: {uri!r}")


def handle_resources_templates_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the parameterised ``statspai://function/{name}`` and
    ``statspai://result/{id}`` URIs.

    Per MCP 2024-11-05, ``resources/templates/list`` is the protocol-
    level mechanism for parameterised resources. Clients that do
    autocomplete on resource URIs use this; the static ``resources/list``
    entries don't enumerate per-function URIs (would be 100+ items in
    client UIs) so a template is the right vehicle.
    """
    return {
        "resourceTemplates": [
            {
                "uriTemplate": FUNCTION_URI_PREFIX + "{name}",
                "name": "StatsPAI function agent card",
                "mimeType": "application/json",
                "description": (
                    "Agent-native detail card for one tool: "
                    "description, JSON-schema signature, identifying "
                    "assumptions, common failure modes with recovery "
                    "hints, ranked alternatives, typical_n_min, and "
                    "an example call. Read "
                    "statspai://functions for the list of valid "
                    "{name} values."
                ),
            },
            {
                "uriTemplate": RESULT_URI_PREFIX + "{id}",
                "name": "StatsPAI fitted-result handle",
                "mimeType": "application/json",
                "description": (
                    "Read a server-cached fitted result by id. The id "
                    "is returned by any tools/call invoked with "
                    "as_handle=true. Body shape mirrors the original "
                    "tool output (estimate / SE / CI / diagnostics) "
                    "plus a provenance block tagging the tool + args "
                    "that produced it. Cache is LRU; missing handles "
                    "raise -32002 (resource not found) — re-fit with "
                    "as_handle=true to refresh."
                ),
            },
        ],
    }


__all__ = [
    "FUNCTION_URI_PREFIX",
    "RESULT_URI_PREFIX",
    "catalog_text",
    "functions_index",
    "function_detail",
    "handle_resources_list",
    "handle_resources_read",
    "handle_resources_templates_list",
]
