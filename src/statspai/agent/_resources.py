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
* :func:`handle_resources_list` — top-level resources
* :func:`handle_resources_read` — URI dispatch (catalog / functions /
  function/<name> / result/<id>)
* :func:`handle_resources_templates_list` — the parameterised URI
  templates
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

FUNCTION_URI_PREFIX = "statspai://function/"
RESULT_URI_PREFIX = "statspai://result/"
PARITY_TRACK_A_URI = "statspai://parity/track-a-summary"

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
    # Only unknown-name / missing-registry lookups may fall through to the
    # synthesized card; a genuine registry bug must stay visible.
    try:
        from ..registry import agent_card as _agent_card

        card = _agent_card(name)
        if card:
            return card
    except (ImportError, KeyError, LookupError):
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


def _repo_root() -> Path:
    """Best-effort source-checkout root for optional evidence artifacts."""
    return Path(__file__).resolve().parents[3]


def _field(block: str, key: str) -> Optional[str]:
    """Extract a markdown bullet field of the form ``- **key**: value``."""
    pattern = rf"^- \*\*{re.escape(key)}\*\*: (.+)$"
    match = re.search(pattern, block, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _unquote_markdown_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if value.startswith("`"):
        match = re.match(r"`([^`]+)`", value)
        if match:
            return match.group(1)
    return value


def _parity_tool_aliases(module: str) -> List[str]:
    """Best-effort StatsPAI tool aliases for a Track A parity module."""
    aliases = {
        "01_ols": ["regress"],
        "02_iv": ["ivreg"],
        "03_hdfe": ["fixest", "reghdfe"],
        "04_csdid": ["callaway_santanna", "csdid"],
        "05_sunab": ["sunab"],
        "06_rd": ["rdrobust"],
        "07_scm": ["synth"],
        "08_dml": ["dml"],
        "09_rddensity": ["rddensity"],
        "10_honest_did": ["honest_did"],
        "11_psm": ["match", "psm"],
        "12_sdid": ["sdid"],
        "13_causal_forest": ["causal_forest"],
        "14_ols_cluster": ["regress"],
        "15_hdfe_cluster": ["fixest", "reghdfe"],
        "16_bjs": ["did_imputation"],
        "17_etwfe": ["etwfe"],
        "18_augsynth": ["augsynth"],
        "19_gsynth": ["gsynth"],
        "20_bacon": ["bacon"],
        "21_honest_relmags": ["honest_did"],
        "22_sensemakr": ["sensemakr"],
        "23_evalue": ["evalue"],
        "24_coxph": ["coxph"],
        "25_lmm": ["multilevel", "lmm"],
        "26_glmm_logit": ["glmer", "glmm"],
        "27_glmm_aghq": ["glmer", "glmm"],
        "28_frontier": ["frontier"],
        "29_panel_sfa": ["xtfrontier"],
        "30_oaxaca": ["oaxaca"],
        "31_dfl": ["dfl"],
        "32_rif": ["rif"],
        "33_var": ["var"],
        "34_lp": ["local_projection", "lpirf"],
        "35_panel": ["panel", "xtreg"],
        "36_mediation": ["mediation"],
        "37_ppmlhdfe": ["ppmlhdfe"],
        "38_drdid": ["drdid"],
        "39_arima": ["arima"],
        "40_qreg": ["qreg"],
        "41_tobit": ["tobit"],
        "42_nbreg": ["nbreg"],
        "43_heckman": ["heckman"],
        "44_mlogit": ["mlogit"],
        "45_ologit": ["ologit"],
        "46_clogit": ["clogit"],
        "47_ppmlhdfe_3fe": ["ppmlhdfe"],
        "48_probit": ["probit"],
        "49_oprobit": ["oprobit"],
        "50_xtabond": ["xtabond"],
        "51_newey": ["newey"],
        "52_scm_unique": ["synth"],
        "53_cr2": ["cr2"],
        "54_twoway_cluster": ["regress"],
        "55_hc2_hc3": ["regress"],
        "56_multiway_cluster": ["fixest", "reghdfe"],
        "57_logit": ["logit"],
        "58_poisson": ["poisson"],
        "59_liml": ["liml", "ivreg"],
        "60_sureg": ["sureg"],
        "61_betareg": ["betareg"],
        "62_truncreg": ["truncreg"],
        "63_zip": ["zip"],
        "64_zinb": ["zinb"],
    }
    return aliases.get(module, [])


@lru_cache(maxsize=1)
def track_a_parity_summary() -> Dict[str, Any]:
    """Return a compact machine-readable summary of committed parity evidence.

    The full Track A report is a long markdown artifact for humans. MCP
    clients need headline counts and per-module convention labels without
    burning context on every table row. Wheels normally do not ship
    ``tests/``, so this resource reports ``available=false`` rather than
    pretending live R/Stata evidence exists.
    """
    rel = "tests/r_parity/results/parity_table_3way.md"
    report = _repo_root() / rel
    if not report.exists():
        return {
            "available": False,
            "artifact": rel,
            "summary": (
                "Track A parity markdown is not present in this installed "
                "environment. Use a source checkout or committed release "
                "artifact to inspect cross-language evidence."
            ),
            "caution": (
                "This resource summarizes committed artifacts only; it is "
                "not a live Stata/R execution."
            ),
        }

    text = report.read_text(encoding="utf-8")
    strict_line = ""
    for line in text.splitlines():
        if line.startswith("**Strictness-tier breakdown**"):
            strict_line = line
            break

    tiers: Dict[str, int] = {}
    for key, pattern in {
        "machine": r"(\d+)\s+machine-level",
        "iterative_cross_fit": r"(\d+)\s+iterative/cross-fit",
        "moderate": r"(\d+)\s+moderate",
        "methodological": r"(\d+)\s+methodological/T4",
    }.items():
        match = re.search(pattern, strict_line)
        if match:
            tiers[key] = int(match.group(1))

    modules: List[Dict[str, Any]] = []
    tool_evidence: Dict[str, List[Dict[str, Any]]] = {}
    for raw_block in text.split("\n## Module ")[1:]:
        header, _, rest = raw_block.partition("\n")
        module = header.strip()
        row: Dict[str, Any] = {"module": module}
        for key in (
            "strictness_tier",
            "method",
            "formula",
            "vcov",
            "stata_command",
            "validation_tier",
            "reference_backend",
            "parity_note",
            "se_note",
            "bandwidth_parity_note",
            "native_note",
            "stata_bridge_status",
        ):
            value = _unquote_markdown_value(_field(rest, key))
            if value:
                row[key] = value
        modules.append(row)
        evidence = {
            k: row[k]
            for k in (
                "module",
                "strictness_tier",
                "method",
                "stata_command",
                "reference_backend",
                "parity_note",
            )
            if k in row
        }
        for alias in _parity_tool_aliases(module):
            tool_evidence.setdefault(alias, []).append(evidence)

    return {
        "available": True,
        "artifact": rel,
        "strictness_tiers": tiers,
        "module_count": len(modules),
        "modules": modules,
        "tool_evidence": tool_evidence,
        "caution": (
            "Summary of committed Python/R/Stata artifacts. This is not a "
            "live Stata/R execution, and methodological convention gaps "
            "should be preserved in downstream claims."
        ),
    }


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
            {
                "uri": PARITY_TRACK_A_URI,
                "name": "StatsPAI Track A parity summary",
                "mimeType": "application/json",
                "description": "Machine-readable summary of committed "
                "Python/R/Stata parity evidence: strictness-tier counts, "
                "module ids, Stata commands, and convention notes where "
                "available. This summarizes artifacts; it is not a live "
                "external Stata/R run.",
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
    if uri == PARITY_TRACK_A_URI:
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(
                        _clean(track_a_parity_summary()),
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
    "PARITY_TRACK_A_URI",
    "catalog_text",
    "functions_index",
    "function_detail",
    "track_a_parity_summary",
    "handle_resources_list",
    "handle_resources_read",
    "handle_resources_templates_list",
]
