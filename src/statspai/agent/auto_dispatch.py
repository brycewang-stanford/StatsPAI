"""Registry-driven dispatch for auto-generated MCP tools.

The hand-curated :data:`statspai.agent.tools.TOOL_REGISTRY` covers ~13
flagship estimators with bespoke serializers. Hundreds more are
visible in the manifest (via :func:`auto_tool_manifest`) but lacked a
dispatch path before this module — calling them through
``execute_tool('foo', ...)`` would 404.

This module fills the gap: it looks the function up on the
``statspai`` package, filters arguments against the registered
``ParamSpec`` list (so the LLM can't pass random kwargs that crash the
estimator), runs it, and applies the standard serializer.

The output mirrors the curated path so downstream tooling (image
content extraction, result caching, JSON wrapping in the MCP layer)
sees a uniform shape regardless of whether a tool was hand-curated or
auto-dispatched.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def _allowed_kwargs(name: str) -> Optional[set]:
    """Return the names of kwargs the registry knows about for ``name``.

    ``None`` means the registry has no entry — caller falls back to
    forwarding all arguments verbatim.
    """
    try:
        from ..registry import _REGISTRY, _ensure_full_registry
        _ensure_full_registry()
        spec = _REGISTRY.get(name)
        if spec is None:
            return None
        return {p.name for p in (spec.params or [])}
    except Exception:
        return None


def dispatch_registry_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    data: Optional[pd.DataFrame] = None,
    detail: str = "agent",
    as_handle: bool = False,
) -> Dict[str, Any]:
    """Run any registered ``sp.<name>`` function as a tool call.

    Raises
    ------
    KeyError
        If ``name`` does not resolve to a public statspai callable —
        the caller (``execute_tool``) translates this into a friendly
        ``{'error': ...}`` envelope.
    """
    import statspai as sp
    fn = getattr(sp, name, None)
    if fn is None or not callable(fn):
        raise KeyError(name)

    allowed = _allowed_kwargs(name)
    kwargs = dict(arguments)
    if allowed is not None:
        # Drop unknown kwargs — the LLM occasionally invents arguments
        # that look plausible but the estimator rejects. Better to log
        # and proceed than to crash a chained workflow on a typo.
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    if data is not None and "data" not in kwargs:
        kwargs["data"] = data

    from .tools import _default_serializer
    from ._result_cache import RESULT_CACHE
    from .remediation import remediate as _remediate
    from ..exceptions import StatsPAIError

    try:
        result = fn(**kwargs)
    except Exception as e:
        envelope: Dict[str, Any] = {
            'error': f"{type(e).__name__}: {e}",
            'tool': name,
            'arguments': {k: v for k, v in arguments.items()
                          if not isinstance(v, pd.DataFrame)},
            'remediation': _remediate(e, context={'tool': name}),
        }
        if isinstance(e, StatsPAIError):
            try:
                envelope['error_kind'] = e.code
                envelope['error_payload'] = e.to_dict()
            except Exception:
                envelope['error_kind'] = e.code
                envelope['error_payload'] = {
                    'kind': e.code,
                    'class': type(e).__name__,
                    'message': str(e),
                }
        return envelope

    try:
        out = _default_serializer(result, detail=detail)
    except Exception as e:
        return {
            'error': f"serializer_error: {type(e).__name__}: {e}",
            'tool': name,
            'stage': 'serializer',
        }

    if not isinstance(out, dict):
        out = {'value': out}

    if as_handle:
        rid = RESULT_CACHE.put(result, tool=name,
                                arguments={k: v for k, v in arguments.items()
                                            if not isinstance(v, pd.DataFrame)})
        out['result_id'] = rid
        out['result_uri'] = f"statspai://result/{rid}"

    return out


__all__ = ["dispatch_registry_tool"]
