"""Internal helpers shared by every tool spec + the dispatch layer.

Lives in its own module so spec files can ``from .._helpers import
_default_serializer`` without dragging in the full dispatch path.

Public surface
--------------

* :func:`_scalar_or_none` — coerce a numeric-or-Series scalar candidate
  to ``float`` or ``None``; tolerant of pandas/numpy edge shapes.
* :func:`_default_serializer` — fall-back ``CausalResult`` /
  ``EconometricResults`` → JSON dict, honouring the ``detail`` enum.
* :func:`_identification_serializer` — ``IdentificationReport`` →
  JSON dict (used by ``check_identification``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _scalar_or_none(v: Any) -> Optional[float]:
    """Return ``float(v)`` when ``v`` represents a single scalar value.

    Handles 0-d arrays, length-1 Series/arrays, and Python numeric
    scalars.  Multi-element Series/DataFrames return ``None`` so callers
    can safely drop those fields from a JSON payload instead of crashing
    ``float(...)``.
    """
    if v is None:
        return None
    if isinstance(v, pd.DataFrame):
        return None
    if isinstance(v, pd.Series):
        # A genuinely single-coefficient Series (size 1) is still a
        # scalar; anything wider is not serialisable to a single float.
        if v.size != 1:
            return None
        try:
            return float(v.iloc[0])
        except (TypeError, ValueError):
            return None
    try:
        arr = np.asarray(v)
        if arr.ndim == 0:
            return float(arr.item())
        if arr.size == 1:
            return float(arr.ravel()[0])
    except Exception:
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


#: Per-observation arrays an estimator attaches to its payload. At the
#: default detail they are cut to :data:`OBS_ARRAY_KEEP` values (an ARIMA
#: fit on 2,000 periods otherwise answered with ~100 KB of residuals and
#: fitted values).
OBS_ARRAY_KEYS = frozenset(
    {
        "residuals",
        "resid",
        "fitted_values",
        "fittedvalues",
        "fitted",
        "predicted",
        "predictions",
        "y_hat",
        "influence_function",
        "scores",
        "policy",
        "propensity",
        "pscore",
    }
)
OBS_ARRAY_KEEP = 20

#: Rows kept from a table, items kept from a list or mapping, inside a
#: plain-dict / DataFrame result, per detail level. Above these the output
#: byte budget of the MCP layer still applies.
_TABLE_ROWS = {"minimal": 5, "standard": 200, "agent": 200}
_LIST_ITEMS = {"minimal": 10, "standard": 1000, "agent": 1000}

#: Containers whose ``_``-prefixed keys are internal bookkeeping
#: (``_cluster_ids``); elsewhere a leading underscore can be data (a
#: coefficient named ``_cons``), so only these and the top level are
#: scrubbed.
_SCRUBBED_CONTAINERS = frozenset({"diagnostics", "model_info"})

#: Top-level private keys the MCP layer itself consumes.
_KEEP_PRIVATE = frozenset({"_plot_png"})


def _json_key(k: Any) -> str:
    """A JSON object key: tuples (``('x1', 'x3')``) join with ``|``."""
    if isinstance(k, str):
        return k
    if isinstance(k, tuple):
        return "|".join(_json_key(x) for x in k)
    if isinstance(k, (np.integer, np.floating, np.bool_)):
        return str(k.item())
    return str(k)


def _cut(
    seq: List[Any], keep: int, path: str, capped: List[Dict[str, Any]]
) -> List[Any]:
    if len(seq) > keep:
        capped.append({"path": path, "total": len(seq), "shown": keep})
        return seq[:keep]
    return seq


def _sanitize(
    v: Any, detail: str, path: str, capped: List[Dict[str, Any]], depth: int = 0
) -> Any:
    """JSON-safe copy of an arbitrary returned value, honouring ``detail``.

    Tuple keys become ``a|b`` strings, DataFrames become row records (with
    the index when it is not a plain range), numpy values become Python
    values, sets become sorted lists, and keys starting with ``_`` are
    dropped from the top-level mapping. Long tables keep :data:`_TABLE_ROWS`
    rows and long lists / mappings :data:`_LIST_ITEMS` items; each cut is
    recorded in ``capped`` as ``{"path", "total", "shown"}``.
    """
    rows = _TABLE_ROWS.get(detail, 200)
    items_cap = _LIST_ITEMS.get(detail, 1000)
    if depth > 8:
        return str(v)[:200]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.bool_, np.integer, np.floating)):
        return v.item()
    if isinstance(v, pd.DataFrame):
        frame = v
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index()
        frame = frame.head(rows) if len(frame) > rows else frame
        if len(v) > rows:
            capped.append({"path": path, "total": int(len(v)), "shown": rows})
        records = frame.to_dict(orient="records")
        return [
            {
                _json_key(c): _sanitize(x, detail, path, capped, depth + 1)
                for c, x in r.items()
            }
            for r in records
        ]
    if isinstance(v, pd.Series):
        v = v.to_dict()
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        v = v.tolist()
    if isinstance(v, (set, frozenset)):
        v = sorted(v, key=str)
    if isinstance(v, dict):
        items = [
            (k, x) for k, x in v.items() if depth > 0 or not str(k).startswith("_")
        ]
        if depth > 0 and len(items) > items_cap:
            capped.append({"path": path, "total": len(items), "shown": items_cap})
            items = items[:items_cap]
        return {
            _json_key(k): _sanitize(
                x, detail, f"{path}/{_json_key(k)}", capped, depth + 1
            )
            for k, x in items
        }
    if isinstance(v, (list, tuple)):
        seq = _cut(list(v), items_cap, path, capped)
        return [
            _sanitize(x, detail, f"{path}/{i}", capped, depth + 1)
            for i, x in enumerate(seq)
        ]
    if v is pd.NaT or v is pd.NA:
        return None
    to_dict = getattr(v, "to_dict", None)
    if callable(to_dict):
        try:
            return _sanitize(to_dict(), detail, path, capped, depth + 1)
        except (TypeError, ValueError):
            pass
    # An object with no JSON form (a fitted tree, a model) is described,
    # not walked: its attributes can hold the training data.
    text = str(v)
    if text.startswith("<") and " at 0x" in text:
        return f"<{type(v).__name__} object>"
    return text[:200]


def _scrub_payload(out: Dict[str, Any], detail: str) -> Dict[str, Any]:
    """Drop private keys and cap per-observation arrays in a serialized payload.

    Applied to every payload the default serializer returns. Keys starting
    with ``_`` at the top level and inside ``diagnostics`` / ``model_info``
    are internal (``_cluster_ids``, ``_model``, ``_index``, ``_det``) and
    are removed, except the top-level ``_plot_png`` the MCP layer turns
    into an image block; deeper, a leading underscore can be data (a
    coefficient named ``_cons``) and is kept. At any detail
    other than the full Python object, the arrays named in
    :data:`OBS_ARRAY_KEYS` keep :data:`OBS_ARRAY_KEEP` values; the cut is
    listed under ``arrays_capped`` as ``{"path", "total", "shown"}``.
    """
    capped: List[Dict[str, Any]] = []

    def _walk(node: Any, path: str, scrub: bool, top: bool) -> Any:
        if isinstance(node, dict):
            kept: Dict[str, Any] = {}
            for k, x in node.items():
                key = str(k)
                if scrub and key.startswith("_") and not (top and key in _KEEP_PRIVATE):
                    continue
                sub = f"{path}/{key}"
                if (
                    key in OBS_ARRAY_KEYS
                    and isinstance(x, (list, tuple, np.ndarray))
                    and len(x) > OBS_ARRAY_KEEP
                ):
                    capped.append(
                        {"path": sub, "total": len(x), "shown": OBS_ARRAY_KEEP}
                    )
                    x = list(x[:OBS_ARRAY_KEEP])
                kept[k] = _walk(x, sub, top and key in _SCRUBBED_CONTAINERS, False)
            return kept
        if isinstance(node, list):
            return [_walk(x, f"{path}/{i}", False, False) for i, x in enumerate(node)]
        return node

    clean: Dict[str, Any] = _walk(out, "", True, True)
    if capped:
        clean.setdefault("arrays_capped", []).extend(capped)
    return clean


def full_array(obj: Any, path: str) -> List[Any]:
    """The whole per-observation array a payload capped at ``path``.

    ``path`` is the ``arrays_capped`` entry's path (``/residuals``,
    ``/diagnostics/residuals``). The result is serialized again without
    the cap and the array is read off it. Only arrays of the capped kinds
    are served, so this cannot be used to read anything else out of a
    cached result.
    """
    parts = [p for p in str(path).split("/") if p]
    if not parts or parts[-1] not in OBS_ARRAY_KEYS:
        raise KeyError(
            f"{path!r} is not a per-observation array; the paths are the "
            "ones listed under arrays_capped."
        )
    for detail in ("agent", "standard"):
        node: Any = _serialize_value(obj, detail=detail)
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                node = None
                break
        if isinstance(node, (list, tuple, np.ndarray)):
            return list(node)
    raise KeyError(f"{path!r} is not in this result.")


def _serialize_mapping(r: Any, detail: str) -> Dict[str, Any]:
    """A dict (or dict subclass) result, sanitised, plus its envelope.

    A dict subclass with ``to_dict(detail=...)`` (``PolicyTreeResult``)
    contributes the agent envelope (``method``, ``n_obs``, ``violations``,
    ``next_steps``, ...) on top of its own content; its ``to_dict`` alone
    returns no content for a dict-based result.
    """
    capped: List[Dict[str, Any]] = []
    content: Dict[str, Any] = _sanitize(dict(r), detail, "", capped)
    if detail == "minimal":
        # Headline fields only: a value over 512 bytes of JSON (a table,
        # per-observation arrays) is left out and named.
        import json

        omitted = [
            k for k, x in content.items() if len(json.dumps(x, default=str)) > 512
        ]
        content = {k: x for k, x in content.items() if k not in omitted}
        if omitted:
            content["omitted_at_minimal"] = omitted
    to_dict = getattr(r, "to_dict", None)
    if callable(to_dict) and not isinstance(r, pd.DataFrame):
        import inspect

        try:
            names = set(inspect.signature(to_dict).parameters)
        except (TypeError, ValueError):
            names = set()
        if "detail" in names:
            envelope = to_dict(detail=detail)
            if isinstance(envelope, dict):
                for k, x in envelope.items():
                    content.setdefault(k, x)
    if capped:
        content["arrays_capped"] = capped
    return content


def _default_serializer(r: Any, *, detail: str = "agent") -> Dict[str, Any]:
    """Serialise a returned value to a JSON dict, honouring ``detail``.

    Results with ``to_dict(detail=...)`` serialise themselves; a plain
    dict, a dict subclass and a DataFrame are sanitised here (tuple keys,
    numpy values, nested frames) with ``detail``-dependent caps. Every
    payload then loses its ``_``-prefixed keys and has its
    per-observation arrays capped (:func:`_scrub_payload`).
    """
    return _scrub_payload(_serialize_value(r, detail=detail), detail)


def _serialize_value(r: Any, *, detail: str = "agent") -> Dict[str, Any]:
    """``_default_serializer`` before the private-key / array scrub.

    LLM tool-use protocol requires JSON-serialisable return values.
    Prefers ``result.to_dict(detail=detail)`` so the LLM gets the
    payload size it asked for through MCP — ``"minimal"`` for cheap
    sub-step calls, ``"standard"`` for normal use, ``"agent"`` (the
    default) when violations + next-step hints + suggested-functions
    should ride along. Falls back to the legacy ``to_dict()`` signature
    for older result types that don't accept the ``detail`` kwarg, and
    finally to the field-by-field extraction below.
    """
    # Plain dicts (``sp.evalue_from_result``, ``sp.pc_algorithm``) and dict
    # subclasses (``PolicyTreeResult``) used to be passed through as they
    # were: ``pc_algorithm`` then failed JSON encoding on its tuple keys,
    # and ``policy_tree`` returned 207 KB (its tree object, training data
    # included) even at detail='minimal'.
    if isinstance(r, dict):
        return _serialize_mapping(r, detail)
    if isinstance(r, pd.DataFrame):
        # Column-oriented, as ``DataFrame.to_dict()`` (the shape this
        # serializer always produced for a frame), with rows capped by
        # detail and the labels made JSON keys.
        capped: List[Dict[str, Any]] = []
        rows = _TABLE_ROWS.get(detail, 200)
        if len(r) > rows:
            capped.append({"path": "", "total": int(len(r)), "shown": rows})
        out_df: Dict[str, Any] = _sanitize(
            r.head(rows).to_dict(), detail, "", capped, depth=1
        )
        if capped:
            out_df["arrays_capped"] = capped
        return out_df
    if isinstance(
        r, (np.ndarray, pd.Series, list, tuple, np.generic, int, float, str, bool)
    ):
        # An array / scalar return (``sp.adjust_pvalues``) has no fields to
        # read; it used to serialise to ``{}``, an answer with no answer.
        capped = []
        out_v: Dict[str, Any] = {"value": _sanitize(r, detail, "/value", capped)}
        if capped:
            out_v["arrays_capped"] = capped
        return out_v

    to_dict = getattr(r, "to_dict", None)
    if callable(to_dict):
        # Preferred: caller-chosen detail level. Use ``inspect.signature``
        # to decide whether the result class supports the ``detail``
        # kwarg — that's a precise discriminant, unlike a blanket
        # ``except TypeError`` which would also swallow internal
        # serialisation bugs that happen to raise TypeError.
        import inspect

        params: Any
        try:
            params = inspect.signature(to_dict).parameters
        except (TypeError, ValueError):
            params = {}
        if "detail" in params:
            serialized = to_dict(detail=detail)
        else:
            # Legacy result type that predates the unified ``detail``
            # parameter. Call the zero-arg form; any exception here is
            # a real bug — let it propagate so ``execute_tool``'s
            # outer error envelope reports it cleanly.
            serialized = to_dict()
        if isinstance(serialized, dict) and serialized:
            return serialized

    def _is_scalar(v: Any) -> bool:
        # A scalar estimate is neither a pandas Series nor a multi-element
        # numpy array; ``float(...)`` only makes sense in that case.
        if isinstance(v, (pd.Series, pd.DataFrame)):
            return False
        try:
            arr = np.asarray(v)
            return arr.ndim == 0 or arr.size == 1
        except Exception:
            return True

    out: Dict[str, Any] = {}
    if hasattr(r, "estimate") and _is_scalar(r.estimate):
        out["estimate"] = float(r.estimate)
    if hasattr(r, "se") and _is_scalar(r.se):
        out["std_error"] = float(r.se)
    if hasattr(r, "pvalue") and _is_scalar(r.pvalue):
        out["p_value"] = float(r.pvalue)
    if hasattr(r, "ci") and r.ci is not None:
        # CausalResult.ci is a (lower, upper) tuple; EconometricResults.ci
        # is a DataFrame — only the tuple form is meaningful here.
        ci = r.ci
        if (
            not isinstance(ci, (pd.DataFrame, pd.Series))
            and hasattr(ci, "__len__")
            and len(ci) == 2
        ):
            try:
                out["conf_low"] = float(ci[0])
                out["conf_high"] = float(ci[1])
            except (TypeError, ValueError):
                pass
    if hasattr(r, "estimand"):
        out["estimand"] = str(r.estimand)
    if hasattr(r, "method"):
        out["method"] = str(r.method)
    if hasattr(r, "n_obs"):
        out["n_obs"] = int(r.n_obs)
    # Regression-style: extract coefficient table if no 'estimate'
    if not out.get("estimate") and hasattr(r, "params"):
        try:
            names = list(r.params.index)

            # pvalues / std_errors can be either pandas Series (indexable
            # by name) or numpy arrays (indexable by position); handle both.
            def _get(obj: Any, name: Any, pos: int) -> Optional[float]:
                if obj is None:
                    return None
                if isinstance(obj, pd.Series):
                    return float(obj[name])
                try:
                    return float(obj[pos])
                except (TypeError, IndexError, KeyError):
                    return None

            coefs = {}
            for pos, k in enumerate(names):
                coefs[str(k)] = {
                    "estimate": float(r.params.iloc[pos]),
                    "std_error": _get(
                        getattr(r, "std_errors", None),
                        k,
                        pos,
                    ),
                    "p_value": _get(getattr(r, "pvalues", None), k, pos),
                }
            out["coefficients"] = coefs
        except Exception as exc:
            # §3.7: don't silently drop the coefficient block — a downstream
            # agent would then report "no coefficients" with no way to tell a
            # genuinely coefficient-free result from a serialization failure
            # (e.g. std_errors misaligned with params). Surface a marker.
            out["coefficients_error"] = f"{type(exc).__name__}: {exc}"
    if hasattr(r, "diagnostics"):
        diag = {}
        for k, v in r.diagnostics.items():
            if isinstance(v, (int, float, str, bool)):
                diag[str(k)] = v
        if diag:
            out["diagnostics"] = diag
    return out


def _identification_serializer(r: Any) -> Dict[str, Any]:
    """Serialise an ``IdentificationReport`` to a JSON dict."""
    return {
        "verdict": r.verdict,
        "design": r.design,
        "n_obs": r.n_obs,
        "n_units": r.n_units,
        "findings": [
            {
                "severity": f.severity,
                "category": f.category,
                "message": f.message,
                "suggestion": f.suggestion,
                "evidence": {
                    k: v
                    for k, v in f.evidence.items()
                    if isinstance(v, (int, float, str, bool))
                },
            }
            for f in r.findings
        ],
    }


__all__ = [
    "_scalar_or_none",
    "_default_serializer",
    "_identification_serializer",
]
