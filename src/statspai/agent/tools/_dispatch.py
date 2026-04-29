"""Dispatch + manifest assembly for the curated TOOL_REGISTRY.

Splits cleanly from the spec data so the spec files stay declarative
and this module owns the runtime concerns (kwargs filtering, error
envelopes, result-handle caching, output enrichment, manifest merge).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ._helpers import _default_serializer
from ._specs import TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------

def tool_manifest(*, curated_only: bool = False) -> List[Dict[str, Any]]:
    """Return the list of tool specifications for an LLM agent.

    Each spec conforms to the Anthropic / OpenAI tool-use JSON schema
    format.  Drop directly into ``client.messages.create(tools=...)``
    (Anthropic) or ``client.chat.completions.create(tools=[...])``
    (OpenAI).

    Parameters
    ----------
    curated_only : bool, default False
        If True, return only the hand-curated tools (the bespoke 13 +
        the workflow / handle / bibtex tools registered by
        :mod:`statspai.agent.workflow_tools` + the pipeline composites
        from :mod:`statspai.agent.pipeline_tools`). The default merges
        them with the auto-generated manifest covering every
        agent-safe registered function so the caller sees the full
        catalogue (~470 tools).

    Returns
    -------
    list of dict
        Each with keys ``'name'``, ``'description'``, ``'input_schema'``.
    """
    curated: List[Dict[str, Any]] = [
        {
            'name': t['name'],
            'description': t['description'],
            'input_schema': t['input_schema'],
        }
        for t in TOOL_REGISTRY
    ]
    # Workflow tools (audit_result / brief_result / sensitivity_from_result
    # / honest_did_from_result / audit / preflight / detect_design /
    # brief / bibtex / plot_from_result) are first-class hand-curated
    # entries — they're what the prompt templates reference and what
    # closes the chained "fit → audit → sensitivity" loop. Append
    # before the auto-merge so collisions on auto-generated stubs of
    # the same name resolve to the hand-curated version.
    from ..workflow_tools import workflow_tool_manifest
    from ..pipeline_tools import pipeline_tool_manifest
    seen = {t['name'] for t in curated}
    for wt in workflow_tool_manifest():
        if wt['name'] not in seen:
            curated.append(wt)
            seen.add(wt['name'])
    for pt in pipeline_tool_manifest():
        if pt['name'] not in seen:
            curated.append(pt)
            seen.add(pt['name'])

    if curated_only:
        return curated

    # Lazy import: auto_tools walks the registry, which can trigger
    # submodule imports — best deferred until someone actually asks.
    from ..auto_tools import merged_tool_manifest
    try:
        return merged_tool_manifest(curated)
    except Exception as e:
        # Loud degradation: silently dropping the auto-tools is exactly
        # the kind of failure CLAUDE.md §3 #7 prohibits ("失败要响亮").
        # Emit a warning the operator (or CI log scraper) can spot.
        import warnings
        warnings.warn(
            f"auto_tool_manifest failed; falling back to curated tools. "
            f"Reason: {type(e).__name__}: {e}",
            RuntimeWarning, stacklevel=2,
        )
        return curated


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _resolve_fn(fn_name: str) -> Callable:
    """Import and return the statspai callable for the given name."""
    import statspai as sp
    fn = getattr(sp, fn_name, None)
    if fn is None:
        raise ValueError(f"Tool {fn_name!r} not found on statspai.")
    return fn


def execute_tool(name: str,
                 arguments: Dict[str, Any],
                 data: Optional[pd.DataFrame] = None,
                 *,
                 detail: str = "agent",
                 result_id: Optional[str] = None,
                 as_handle: bool = False) -> Dict[str, Any]:
    """Dispatch a tool call to the right StatsPAI function.

    Parameters
    ----------
    name : str
        Tool name (must match a ``TOOL_REGISTRY`` entry, an
        auto-registered registry function, or a built-in ``*_result`` /
        ``bibtex`` workflow tool).
    arguments : dict
        Tool-call arguments as provided by the LLM (JSON object).
    data : pd.DataFrame, optional
        Dataset the estimator runs on. Required by most tools.
    detail : {"minimal", "standard", "agent"}, default ``"agent"``
        Payload depth requested by the caller. Forwarded to the
        default serializer's ``r.to_dict(detail=...)``.
    result_id : str, optional
        Handle to a previously-fitted result cached by the server. When
        supplied, ``*_from_result`` tools resolve it from the result
        cache; other tools merge selected fields from the cached result
        (e.g. ``betas`` / ``sigma`` for ``honest_did_from_result``).
    as_handle : bool, default False
        If True, cache the fitted result and inject ``result_id`` /
        ``result_uri`` into the returned dict so a subsequent
        ``execute_tool`` call can reference it without re-fitting.

    Returns
    -------
    dict
        JSON-serialisable result, suitable for returning to the LLM as
        tool output. On error, returns ``{'error': <str>,
        'remediation': <dict>}`` — the agent can use ``remediation``
        to repair its next call.
    """
    # Workflow / result-handle tools live outside the curated
    # TOOL_REGISTRY (they're synthesised) but must be dispatched here so
    # the MCP layer never needs to know about a separate registry.
    from ..workflow_tools import (
        WORKFLOW_TOOL_NAMES,
        execute_workflow_tool,
    )
    if name in WORKFLOW_TOOL_NAMES:
        return execute_workflow_tool(
            name, arguments,
            data=data, detail=detail,
            result_id=result_id, as_handle=as_handle,
        )

    # Composite pipeline tools (pipeline_did / pipeline_iv / pipeline_rd
    # — multi-stage end-to-end workflows). They embed result-cache
    # writes themselves, so we don't re-cache here.
    from ..pipeline_tools import (
        PIPELINE_TOOL_NAMES,
        execute_pipeline_tool,
    )
    if name in PIPELINE_TOOL_NAMES:
        return execute_pipeline_tool(
            name, arguments,
            data=data, detail=detail, as_handle=as_handle,
        )

    spec = next((t for t in TOOL_REGISTRY if t['name'] == name), None)
    if spec is None:
        # Fall back to a registry-driven dispatch so auto-generated
        # tools (the 100+ from auto_tool_manifest) are callable too.
        from ..auto_dispatch import dispatch_registry_tool
        try:
            return dispatch_registry_tool(
                name, arguments,
                data=data, detail=detail, as_handle=as_handle,
            )
        except KeyError:
            return {
                'error': f"Unknown tool: {name!r}",
                'available_tools': [t['name'] for t in TOOL_REGISTRY],
                'hint': ("Read statspai://functions for the full "
                         "machine-readable index of registered tools."),
            }

    # Look ``_resolve_fn`` up via the parent package so test fixtures
    # that ``monkeypatch.setattr(agent.tools, '_resolve_fn', …)`` still
    # take effect. Pre-1.11 ``tools`` was a single module; the v1.11
    # split would otherwise force every existing fixture to retarget
    # ``agent.tools._dispatch`` instead.
    from . import _resolve_fn as _public_resolve_fn  # re-export pointer
    fn = _public_resolve_fn(spec['statspai_fn'])
    serialize = spec.get('serializer', _default_serializer)

    # Most tools take `data=` as first positional (or kwarg).
    # Formula-based ones (regress, ivreg) also take data.
    kwargs = dict(arguments)
    if data is not None:
        kwargs['data'] = data

    def _serialize(result_obj):
        """Invoke ``serialize`` with ``detail=`` when supported.

        Custom serializers in TOOL_REGISTRY (e.g. for ``causal``,
        ``recommend``) emit a fixed shape and don't accept ``detail`` —
        forwarding the kwarg would crash them. We use
        ``inspect.signature`` to decide rather than catching
        ``TypeError``; the latter would silently swallow genuine
        ``TypeError`` bugs raised inside the serializer body.
        """
        if serialize is _default_serializer:
            return serialize(result_obj, detail=detail)
        import inspect
        try:
            params = inspect.signature(serialize).parameters
        except (TypeError, ValueError):
            # Built-in / C-extension callable without an introspectable
            # signature — assume it takes only the result.
            return serialize(result_obj)
        if "detail" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()):
            return serialize(result_obj, detail=detail)
        return serialize(result_obj)

    # Estimator call. Any failure here is attributed to the estimator
    # itself — that's where structured StatsPAIError instances come from.
    try:
        result = fn(**kwargs)
    except Exception as e:
        from ..remediation import remediate as _remediate
        envelope: Dict[str, Any] = {
            'error': f"{type(e).__name__}: {e}",
            'tool': name,
            'arguments': {k: v for k, v in arguments.items()
                          if not isinstance(v, pd.DataFrame)},
            'remediation': _remediate(e, context={'tool': name,
                                                   'arguments': arguments}),
        }
        # Surface the structured StatsPAIError payload alongside the
        # legacy fields so MCP-mediated agents can branch on
        # ``error_kind`` (e.g. ``"assumption_violation"``,
        # ``"identification_failure"``) without parsing free-text
        # messages, and read ``recovery_hint`` / ``diagnostics`` /
        # ``alternative_functions`` directly from ``error_payload``.
        from ...exceptions import StatsPAIError
        if isinstance(e, StatsPAIError):
            try:
                envelope['error_kind'] = e.code
                envelope['error_payload'] = e.to_dict()
            except Exception:
                # Defensive fallback: a malformed diagnostics dict (e.g.
                # a live DataFrame) shouldn't crash the error handler
                # and lose the original exception. ``e.code`` is a
                # class attribute with a string default on every
                # ``StatsPAIError`` subclass, so reading it cannot fail.
                envelope['error_kind'] = e.code
                envelope['error_payload'] = {
                    'kind': e.code,
                    'class': type(e).__name__,
                    'message': str(e),
                }
        return envelope

    # Result serialization. A failure here is a *serializer* bug, not
    # an estimator failure — attribute it accordingly so agents don't
    # see misleading ``remediation`` advice for working call args.
    try:
        out = _serialize(result)
    except Exception as e:
        return {
            'error': f"serializer_error: {type(e).__name__}: {e}",
            'tool': name,
            'arguments': {k: v for k, v in arguments.items()
                          if not isinstance(v, pd.DataFrame)},
            'stage': 'serializer',
        }

    if not isinstance(out, dict):
        out = {'value': out}

    # Result-handle caching. When ``as_handle=True`` we stash the live
    # fitted result in the process-local LRU cache and surface a handle
    # so the next tools/call can reach it without re-loading the CSV
    # and re-fitting. This is the foundational primitive for chained
    # workflows (did → audit → sensitivity → honest_did_from_result).
    rid: Optional[str] = None
    if as_handle:
        from .._result_cache import RESULT_CACHE
        rid = RESULT_CACHE.put(
            result, tool=name,
            arguments={k: v for k, v in arguments.items()
                       if not isinstance(v, pd.DataFrame)},
        )
        out['result_id'] = rid
        out['result_uri'] = f"statspai://result/{rid}"

    # Output enrichment: pre-built next_calls + verified citations +
    # short narrative. Agents on per-call billing get more value per
    # roundtrip; agents on per-token billing can request
    # detail='minimal' to skip these or strip them client-side.
    from .._enrichment import enrich_payload
    enrich_payload(out, tool_name=name, result_id=rid,
                   base_args={k: v for k, v in arguments.items()
                              if not isinstance(v, pd.DataFrame)})

    return out


__all__ = ["tool_manifest", "execute_tool", "_resolve_fn"]
