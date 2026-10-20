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


def _allowed_kwargs(name: str, fn: Any = None) -> Optional[set]:
    """Return the keyword names ``sp.<name>`` can bind.

    The union of the registry ``ParamSpec`` names, the live signature
    (hand-written registry entries can lag the signature — ``did`` lists
    8 of its ~28 parameters) and any ``@accepts_aliases`` spellings.
    ``None`` means "forward everything": the function takes ``**kwargs``
    or nothing about it could be introspected.
    """
    import inspect

    names: set = set()
    if fn is not None:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = None
        if params is not None:
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return None
            names.update(params)
        names.update(getattr(fn, "__statspai_aliases__", {}) or {})
    try:
        from ..registry import _REGISTRY, _ensure_full_registry

        _ensure_full_registry()
        spec = _REGISTRY.get(name)
        if spec is not None:
            names.update(p.name for p in (spec.params or []))
    except Exception:
        pass
    return names or None


def dispatch_registry_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    data: Optional[pd.DataFrame] = None,
    detail: str = "agent",
    as_handle: bool = False,
    result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run any registered ``sp.<name>`` function as a tool call.

    ``result_id`` resolves a cached fit into the function's ``result``
    argument when it has one. An unresolvable handle returns the
    structured ``missing_result_handle`` error; a handle given to a
    function without a ``result`` parameter is listed under
    ``_unsupported_args``.

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

    from ._default_seed import inject_default_seed

    seed_injected = inject_default_seed(fn, arguments)

    allowed = _allowed_kwargs(name, fn)
    kwargs = dict(arguments)
    unsupported: list = []
    if allowed is not None:
        # An argument the function cannot bind refuses the call with an
        # ``unknown_argument`` error and did_you_mean (mirroring the
        # curated path): running without a misspelt ``cluster=`` would
        # return standard errors the agent did not ask for (CLAUDE.md
        # §3.7). Server-handled names a Python caller left in
        # ``arguments`` are reported under ``_unsupported_args`` instead.
        from .tools._dispatch import _SERVER_HANDLED_ARGS, unknown_argument_error

        unsupported = sorted(k for k in kwargs if k not in allowed)
        unknown = [k for k in unsupported if k not in _SERVER_HANDLED_ARGS]
        if unknown:
            return unknown_argument_error(name, unknown, allowed, arguments=arguments)
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    if data is not None and "data" not in kwargs:
        if allowed is None or "data" in allowed:
            kwargs["data"] = data
        else:
            # A function without a data= parameter (slopes / hypotheses /
            # predictions after a result_id) used to get one anyway and
            # fail with a raw TypeError (2026-10-20 probe). The loaded data
            # is reported as not applied; the fit's own data is used.
            unsupported = sorted(set(unsupported) | {"data"})

    from ..exceptions import StatsPAIError
    from ._replay import build_replay
    from ._result_cache import RESULT_CACHE, missing_result_error
    from .remediation import remediate as _remediate
    from .tools import _default_serializer

    if result_id:
        if RESULT_CACHE.get_entry(result_id) is None:
            return dict(missing_result_error(result_id), tool=name)
        if "result" not in kwargs and allowed is not None and "result" in allowed:
            kwargs["result"] = RESULT_CACHE.get(result_id)
        elif "result" not in kwargs:
            unsupported = sorted(unsupported + ["result_id"])

    try:
        result = fn(**kwargs)
    except Exception as e:
        envelope: Dict[str, Any] = {
            "error": f"{type(e).__name__}: {e}",
            "tool": name,
            "arguments": {
                k: v for k, v in arguments.items() if not isinstance(v, pd.DataFrame)
            },
            "remediation": _remediate(e, context={"tool": name}),
        }
        if unsupported:
            envelope["_unsupported_args"] = unsupported
        if isinstance(e, StatsPAIError):
            try:
                envelope["error_kind"] = e.code
                envelope["error_payload"] = e.to_dict()
            except Exception:
                envelope["error_kind"] = e.code
                envelope["error_payload"] = {
                    "kind": e.code,
                    "class": type(e).__name__,
                    "message": str(e),
                }
        return envelope

    try:
        out = _default_serializer(result, detail=detail)
    except Exception as e:
        return {
            "error": f"serializer_error: {type(e).__name__}: {e}",
            "tool": name,
            "stage": "serializer",
        }

    if not isinstance(out, dict):
        out = {"value": out}
    if unsupported:
        out["_unsupported_args"] = unsupported
        out.setdefault(
            "_unsupported_args_note",
            "These arguments are not accepted by the estimator and were NOT "
            "applied; check describe_function for the accepted names.",
        )

    if seed_injected:
        out["seed_injected"] = seed_injected

    replay = build_replay(name, kwargs, result_id=result_id)
    out["replay"] = replay

    rid: Optional[str] = None
    if as_handle:
        rid = RESULT_CACHE.put(
            result,
            tool=name,
            arguments={
                k: v for k, v in arguments.items() if not isinstance(v, pd.DataFrame)
            },
            replay=replay,
        )
        out["result_id"] = rid
        out["result_uri"] = f"statspai://result/{rid}"

    # Result card, as on the curated path: without it a function reached
    # through the registry (the long tail, including the forests) answered
    # with no configuration-level evidence and no diagnostic status, and an
    # agent could only fall back on the function-level tier.
    if detail != "minimal" and (
        hasattr(result, "params")
        or hasattr(result, "estimate")
        or isinstance(getattr(result, "model_info", None), dict)
    ):
        from ..result_card import result_card as _result_card
        from ..workflow._degradation import record_degradation

        try:
            out["result_card"] = dict(_result_card(result))
        except Exception as e:  # noqa: BLE001 - the fit itself succeeded
            degr: list = []
            record_degradation(degr, section="result_card", exc=e, detail=name)
            out["result_card_error"] = degr[0] if degr else repr(e)

    from ._enrichment import enrich_payload

    enrich_payload(
        out,
        tool_name=name,
        result_id=rid,
        base_args={
            k: v for k, v in arguments.items() if not isinstance(v, pd.DataFrame)
        },
    )

    return out


__all__ = ["dispatch_registry_tool"]
