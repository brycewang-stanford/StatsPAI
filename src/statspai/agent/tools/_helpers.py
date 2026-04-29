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

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _scalar_or_none(v) -> Optional[float]:
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


def _default_serializer(r, *, detail: str = "agent") -> Dict[str, Any]:
    """Serialise a ``CausalResult`` / ``EconometricResults`` to a JSON dict.

    LLM tool-use protocol requires JSON-serialisable return values.
    Prefers ``result.to_dict(detail=detail)`` so the LLM gets the
    payload size it asked for through MCP — ``"minimal"`` for cheap
    sub-step calls, ``"standard"`` for normal use, ``"agent"`` (the
    default) when violations + next-step hints + suggested-functions
    should ride along. Falls back to the legacy ``to_dict()`` signature
    for older result types that don't accept the ``detail`` kwarg, and
    finally to the field-by-field extraction below.
    """
    to_dict = getattr(r, 'to_dict', None)
    if callable(to_dict):
        # Preferred: caller-chosen detail level. Use ``inspect.signature``
        # to decide whether the result class supports the ``detail``
        # kwarg — that's a precise discriminant, unlike a blanket
        # ``except TypeError`` which would also swallow internal
        # serialisation bugs that happen to raise TypeError.
        import inspect
        try:
            params = inspect.signature(to_dict).parameters
        except (TypeError, ValueError):
            params = {}
        if "detail" in params:
            out = to_dict(detail=detail)
        else:
            # Legacy result type that predates the unified ``detail``
            # parameter. Call the zero-arg form; any exception here is
            # a real bug — let it propagate so ``execute_tool``'s
            # outer error envelope reports it cleanly.
            out = to_dict()
        if isinstance(out, dict) and out:
            return out

    def _is_scalar(v) -> bool:
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
    if hasattr(r, 'estimate') and _is_scalar(r.estimate):
        out['estimate'] = float(r.estimate)
    if hasattr(r, 'se') and _is_scalar(r.se):
        out['std_error'] = float(r.se)
    if hasattr(r, 'pvalue') and _is_scalar(r.pvalue):
        out['p_value'] = float(r.pvalue)
    if hasattr(r, 'ci') and r.ci is not None:
        # CausalResult.ci is a (lower, upper) tuple; EconometricResults.ci
        # is a DataFrame — only the tuple form is meaningful here.
        ci = r.ci
        if (not isinstance(ci, (pd.DataFrame, pd.Series))
                and hasattr(ci, '__len__') and len(ci) == 2):
            try:
                out['conf_low'] = float(ci[0])
                out['conf_high'] = float(ci[1])
            except (TypeError, ValueError):
                pass
    if hasattr(r, 'estimand'):
        out['estimand'] = str(r.estimand)
    if hasattr(r, 'method'):
        out['method'] = str(r.method)
    if hasattr(r, 'n_obs'):
        out['n_obs'] = int(r.n_obs)
    # Regression-style: extract coefficient table if no 'estimate'
    if not out.get('estimate') and hasattr(r, 'params'):
        try:
            names = list(r.params.index)

            # pvalues / std_errors can be either pandas Series (indexable
            # by name) or numpy arrays (indexable by position); handle both.
            def _get(obj, name, pos):
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
                    'estimate': float(r.params.iloc[pos]),
                    'std_error': _get(getattr(r, 'std_errors', None),
                                       k, pos),
                    'p_value': _get(getattr(r, 'pvalues', None), k, pos),
                }
            out['coefficients'] = coefs
        except Exception:
            pass
    if hasattr(r, 'diagnostics'):
        diag = {}
        for k, v in r.diagnostics.items():
            if isinstance(v, (int, float, str, bool)):
                diag[str(k)] = v
        if diag:
            out['diagnostics'] = diag
    return out


def _identification_serializer(r) -> Dict[str, Any]:
    """Serialise an ``IdentificationReport`` to a JSON dict."""
    return {
        'verdict': r.verdict,
        'design': r.design,
        'n_obs': r.n_obs,
        'n_units': r.n_units,
        'findings': [
            {
                'severity': f.severity,
                'category': f.category,
                'message': f.message,
                'suggestion': f.suggestion,
                'evidence': {k: v for k, v in f.evidence.items()
                             if isinstance(v, (int, float, str, bool))},
            }
            for f in r.findings
        ],
    }


__all__ = [
    "_scalar_or_none",
    "_default_serializer",
    "_identification_serializer",
]
