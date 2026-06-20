"""Post-estimation capability contract for fitted StatsPAI results.

The rest of :mod:`statspai.postestimation` provides Stata/R-style
commands (``margins``, ``lincom``, ``test``). This module answers the
agent-facing question that comes before calling any of them: which
post-estimation actions are actually available for this result object?
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

_CORE_ACTIONS = {
    "tidy": "long coefficient/effect table",
    "glance": "one-row model summary",
    "summary": "human-readable summary",
    "to_dict": "structured result payload",
    "to_json": "JSON serialization",
}

_EXPORT_ACTIONS = {
    "to_latex": "LaTeX table/document export",
    "to_docx": "Word export",
    "to_html": "HTML/Jupyter rendering",
}


def postestimation_contract(
    result: Any,
    *,
    data: Optional[pd.DataFrame] = None,
    include_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Return the post-estimation actions supported by *result*.

    Parameters
    ----------
    result : object
        Fitted StatsPAI result, estimator, or compatible object.
    data : DataFrame, optional
        Analysis frame available for data-dependent actions. When
        supplied, the contract marks ``margins``/``predict`` paths as
        ready rather than merely method-available.
    include_diagnostics : bool, default True
        Include scalar diagnostics from ``model_info`` / ``diagnostics``.

    Returns
    -------
    dict
        Machine-readable contract with ``available``, ``missing``,
        ``recommended_next``, and optional ``diagnostics`` keys.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=200),
    ...     "x2": rng.normal(size=200),
    ... })
    >>> df["y"] = (1.0 + 0.5 * df["x1"] - 0.3 * df["x2"]
    ...            + rng.normal(size=200))
    >>> res = sp.regress("y ~ x1 + x2", data=df)
    >>> con = sp.postestimation_contract(res, data=df)
    >>> con["result_type"]
    'EconometricResults'
    >>> "margins" in con["available"]   # data supplied -> ready
    True
    >>> con["has_data"]
    True
    """
    if result is None:
        raise ValueError("postestimation_contract requires a fitted result object")

    available: Dict[str, str] = {}
    missing: Dict[str, str] = {}

    for name, description in _CORE_ACTIONS.items():
        _record_capability(result, name, description, available, missing)
    for name, description in _EXPORT_ACTIONS.items():
        _record_capability(result, name, description, available, missing)

    has_params = all(hasattr(result, attr) for attr in ("params", "std_errors"))
    has_effect = all(hasattr(result, attr) for attr in ("estimate", "se", "ci"))
    has_model_data = data is not None

    if callable(getattr(result, "predict", None)) or callable(
        getattr(result, "effect", None)
    ):
        available["predict"] = "prediction or treatment-effect prediction"
    else:
        missing["predict"] = "result has no predict/effect method"

    if has_params:
        available["lincom"] = "linear combinations of coefficients"
        available["test"] = "Wald tests of linear restrictions"
        if has_model_data:
            available["margins"] = "marginal effects using supplied data"
        else:
            missing["margins"] = "requires the analysis data frame"
    else:
        missing["lincom"] = "requires coefficient vector and covariance/SEs"
        missing["test"] = "requires coefficient vector and covariance/SEs"
        missing["margins"] = "requires coefficient model plus analysis data"

    if has_effect:
        available["effect_summary"] = "single-estimand effect summary"
    else:
        missing["effect_summary"] = "no scalar estimate/se/ci attributes"

    if callable(getattr(result, "violations", None)):
        available["violations"] = "assumption and diagnostic violations"
    else:
        missing["violations"] = "result does not expose violations()"

    diagnostics: Dict[str, Any] = {}
    if include_diagnostics:
        diagnostics.update(_scalar_items(getattr(result, "model_info", None)))
        diagnostics.update(_scalar_items(getattr(result, "diagnostics", None)))

    return {
        "result_type": type(result).__name__,
        "available": available,
        "missing": missing,
        "recommended_next": _recommended_next(available, missing, diagnostics),
        "diagnostics": diagnostics,
        "has_data": data is not None,
    }


def _record_capability(
    result: Any,
    name: str,
    description: str,
    available: Dict[str, str],
    missing: Dict[str, str],
) -> None:
    if callable(getattr(result, name, None)):
        available[name] = description
    else:
        missing[name] = f"result has no {name}() method"


def _scalar_items(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if callable(obj):
        try:
            obj = obj()
        except TypeError:
            return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
    return out


def _recommended_next(
    available: Dict[str, str],
    missing: Dict[str, str],
    diagnostics: Dict[str, Any],
) -> List[str]:
    steps: List[str] = []
    if "tidy" in available:
        steps.append("Inspect result.tidy() for coefficient/effect rows.")
    if "glance" in available:
        steps.append("Use result.glance() for model-level reporting.")
    if "violations" in available:
        steps.append("Run result.violations() before publication export.")
    if "margins" in missing and "lincom" in available:
        steps.append("Pass the analysis DataFrame to enable margins().")
    if "to_latex" in available or "to_docx" in available:
        steps.append("Export through sp.collect() when combining models.")
    if diagnostics:
        for key in ("pretrend_pvalue", "first_stage_F", "pscore_min"):
            if key in diagnostics:
                steps.append(f"Review diagnostic `{key}` before reporting.")
                break
    return _dedupe(steps)


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


# Friendly alias for prose contexts.
postestimation_report = postestimation_contract


__all__ = ["postestimation_contract", "postestimation_report"]
