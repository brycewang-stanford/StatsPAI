"""End-to-end orchestrator + advisor tool specs (causal, recommend).

Both tools have bespoke serializers — they emit a *fixed* shape (top
recommendation / verdict / scalar estimate) rather than the standard
``CausalResult`` JSON. Defining the serializers as module-level
functions (rather than the original lambdas) keeps stack traces
readable and unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .._helpers import _scalar_or_none


def _causal_serializer(w: Any) -> Dict[str, Any]:
    """Serialise a ``CausalWorkflow`` result to a fixed-shape JSON dict."""
    return {
        "design": w.design,
        "verdict": w.diagnostics.verdict if w.diagnostics else None,
        "top_method": (
            w.recommendation.recommendations[0]["method"]
            if (w.recommendation and w.recommendation.recommendations)
            else None
        ),
        # Guard against non-scalar `.estimate` / `.se` (e.g.
        # EconometricResults exposes Series-valued tidy aliases).
        "estimate": _scalar_or_none(
            w.result.estimate
            if (w.result is not None and hasattr(w.result, "estimate"))
            else None
        ),
        "std_error": _scalar_or_none(
            w.result.se if (w.result is not None and hasattr(w.result, "se")) else None
        ),
        "conf_low": (
            _scalar_or_none(w.result.ci[0])
            if (
                w.result is not None
                and hasattr(w.result, "ci")
                and w.result.ci is not None
                and not isinstance(w.result.ci, (pd.DataFrame, pd.Series))
                and hasattr(w.result.ci, "__len__")
                and len(w.result.ci) == 2
            )
            else None
        ),
        "conf_high": (
            _scalar_or_none(w.result.ci[1])
            if (
                w.result is not None
                and hasattr(w.result, "ci")
                and w.result.ci is not None
                and not isinstance(w.result.ci, (pd.DataFrame, pd.Series))
                and hasattr(w.result.ci, "__len__")
                and len(w.result.ci) == 2
            )
            else None
        ),
        "robustness": {
            k: v
            for k, v in w.robustness_findings.items()
            if isinstance(v, (int, float, str))
        },
    }


def _json_param(value: Any) -> bool:
    """A recommendation parameter an agent can send as a tool argument."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(isinstance(v, (str, int, float, bool)) for v in value)
    return False


def _recommendation_entry(x: Dict[str, Any], source_columns: Any) -> Dict[str, Any]:
    """One recommendation as the agent sees it, with the derived columns named.

    ``sp.recommend`` emits ``method`` / ``function`` / ``reason`` /
    ``assumptions`` / ``code`` / ``params``; before 2026-10-16 this
    serializer read ``reasoning`` and ``score``, which it never emits, so
    every entry came back as ``reasoning: null, score: null``.
    """
    raw = dict(x.get("params") or {})
    params = {k: v for k, v in raw.items() if k != "data" and _json_param(v)}
    entry: Dict[str, Any] = {
        "method": x.get("method"),
        "function": x.get("function"),
        "reasoning": x.get("reason"),
        "assumptions": list(x.get("assumptions") or []),
        "params": params,
    }
    if x.get("robustness"):
        entry["robustness"] = x.get("robustness")
    if x.get("code"):
        entry["code"] = x.get("code")
    # A ``prep`` step adds a column the agent's dataset does not have (the
    # staggered cards derive ``_cohort_<treat>`` = first treated period).
    prepared = raw.get("data")
    if (
        x.get("prep") is not None
        and isinstance(prepared, pd.DataFrame)
        and source_columns is not None
    ):
        derived = sorted(set(prepared.columns) - set(source_columns))
        names = sorted(k for k, v in params.items() if v in derived)
        if names:
            entry["derived_columns"] = {k: params[k] for k in names}
    return entry


def _recommendation_next_call(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A ``tools/call`` payload for one recommendation, readiness annotated."""
    fn = entry.get("function")
    if not isinstance(fn, str) or not fn:
        return None
    from ..._enrichment import _accepted_args, _annotate_readiness

    accepted = _accepted_args(fn)
    derived = entry.get("derived_columns") or {}
    args = {
        k: v
        for k, v in entry["params"].items()
        if k not in derived and (accepted is None or k in accepted)
    }
    call: Dict[str, Any] = {
        "tool": fn,
        "arguments": args,
        "rationale": entry.get("reasoning") or entry.get("method"),
    }
    if accepted is None:
        # Not an advertised tool: reachable only through call_function,
        # which needs the dataset passed explicitly.
        call = {
            "tool": "call_function",
            "arguments": {"function": fn, "arguments": args},
            "rationale": call["rationale"],
            "ready": False,
            "missing_arguments": ["data_id"],
            "hint": "Add data_id (the dataset handle) before dispatching.",
        }
        return call
    _annotate_readiness(call)
    if derived:
        missing = sorted(set(call.get("missing_arguments") or []) | set(derived))
        call["ready"] = False
        call["missing_arguments"] = missing
        # ``note`` survives the server's readiness re-check, which rewrites
        # ``hint``.
        call["note"] = (
            "recommend derived "
            + ", ".join(f"{k}={v!r}" for k, v in derived.items())
            + " from the treatment indicator (each unit's first treated "
            "period, 0 = never treated); that column is not in your data. "
            "Pass an existing cohort column, or create one with "
            "transform_data, then dispatch."
        )
    return call


def _recommend_serializer(r: Any) -> Dict[str, Any]:
    """Serialise a ``Recommendation`` result to a fixed-shape JSON dict.

    ``next_calls`` carries a payload for the top recommendation; the MCP
    server adds the dataset handle, so it is ``ready`` when every required
    argument is known.
    """
    recs = list(getattr(r, "recommendations", []) or [])
    data = getattr(r, "_data", None)
    source_columns = list(data.columns) if isinstance(data, pd.DataFrame) else None
    top = [_recommendation_entry(x, source_columns) for x in recs[:5]]
    out: Dict[str, Any] = {
        "design": getattr(r, "design", None),
        "top_recommendations": top,
        "n_recommendations": len(recs),
    }
    if top:
        call = _recommendation_next_call(top[0])
        if call is not None:
            out["next_calls"] = [call]
    return out


SPECS: List[Dict[str, Any]] = [
    {
        "name": "causal",
        "description": (
            "End-to-end causal workflow: diagnose -> recommend estimator "
            "-> fit -> run robustness -> return result.  The one-shot "
            "entry point that lets an agent analyse a dataset in a "
            "single call without orchestrating stages itself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string"},
                "treatment": {"type": "string"},
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "id": {"type": "string"},
                "time": {"type": "string"},
                "cohort": {"type": "string"},
                "running_var": {"type": "string"},
                "instrument": {"type": "string"},
                "design": {
                    "type": "string",
                    "enum": [
                        "did",
                        "rd",
                        "iv",
                        "observational",
                        "panel",
                        "rct",
                    ],
                },
            },
            "required": ["y"],
        },
        "statspai_fn": "causal",
        "serializer": _causal_serializer,
    },
    {
        "name": "recommend",
        "description": (
            "Method advisor: given a dataset + research question, "
            "recommends a ranked list of estimators with reasoning, "
            "precondition checks, and a full suggested workflow.  "
            "This is the first call an agent should make if it "
            "doesn't know which estimator to run. Supports DAG input, "
            "mediator / proxy / principal-strata variables, and "
            "optional resampling-stability verification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string", "description": "Outcome column."},
                "treatment": {
                    "type": "string",
                    "description": "Treatment / exposure column.",
                },
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Covariate columns.",
                },
                "id": {"type": "string", "description": "Unit identifier (panel)."},
                "time": {"type": "string", "description": "Time column (panel / DID)."},
                "running_var": {
                    "type": "string",
                    "description": "Running variable (RD).",
                },
                "instrument": {
                    "type": "string",
                    "description": "Instrumental variable.",
                },
                "cutoff": {"type": "number", "description": "RD cutoff value."},
                "design": {
                    "type": "string",
                    "enum": [
                        "rct",
                        "did",
                        "rd",
                        "iv",
                        "observational",
                        "panel",
                        "cross-section",
                    ],
                    "description": "Override auto-detected design.",
                },
                "verify": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If True, run resampling-stability "
                        "checks on top recommendations."
                    ),
                },
            },
            "required": ["y"],
        },
        "statspai_fn": "recommend",
        "serializer": _recommend_serializer,
    },
]
