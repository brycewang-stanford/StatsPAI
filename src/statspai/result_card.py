"""One auditable card per fitted result: what was estimated, on which sample,
how inference was done, where the numbers came from, and what evidence covers
*this* configuration.

The pieces already existed but lived in different places -- the estimand on
``CausalResult``, the call arguments and data fingerprint on the provenance
record, the covariance convention in ``model_info`` / ``data_info``, the
identifying assumptions and known limitations in the registry, and the
configuration-level evidence in :func:`sp.validation_scope`. A user, a
co-author, a referee and an agent should not each reassemble them.
``sp.result_card`` does, with the same content in Python and in MCP tool
responses.

The card never upgrades evidence. For the validation-suite estimators the
evidence section is the configuration x output map of
:func:`sp.validation_scope`; for every other function it states that only a
function-level tier exists, which is a statement about the function, not
about the configuration that was run.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

__all__ = ["result_card", "ResultCard"]

#: model_info keys that count rows removed before or during estimation.
_EXCLUSION_RE = re.compile(
    r"^n_.*(dropped|trimmed|removed|excluded|singleton|missing|separat)", re.I
)
#: model_info keys describing the sample structure (reported verbatim).
_STRUCTURE_KEYS = (
    "n_clusters",
    "n_units",
    "n_treated_units",
    "n_control_units",
    "n_periods",
    "n_cohorts",
    "n_groups",
    "n_treated",
    "n_control",
    "n_left",
    "n_right",
    "n_effective",
)
#: model_info keys that pin down the target population / aggregation.
_ESTIMAND_KEYS = (
    "control_group",
    "aggregation",
    "summary_aggregation",
    "target_sample",
    "target",
    "anticipation",
    "base_period",
    "cutoff",
    "rd_type",
    "dml_model",
    "score",
)
#: model_info keys that describe the covariance / inference convention.
_INFERENCE_KEYS = (
    "vcov_type",
    "robust",
    "cluster",
    "clustervars",
    "se_type",
    "se_method",
    "vce",
    "bstrap",
    "biters",
    "n_boot",
    "cband",
    "boot_weight_type",
    "ssc",
)
#: model_info keys naming the implementation that ran.
_BACKEND_KEYS = ("backend", "engine", "implementation", "solver", "cct_delegation")


class ResultCard(dict):
    """Structured, JSON-safe card for one fitted result (a ``dict``).

    Keys: ``function``, ``estimand``, ``sample``, ``specification``,
    ``inference``, ``provenance``, ``evidence``, ``assumptions``,
    ``limitations``. ``str(card)`` / ``card.to_markdown()`` render it.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.card_1995()
    >>> fit = sp.regress("lwage ~ educ + exper", data=df, robust="hc1")
    >>> card = sp.result_card(fit)
    >>> card["function"]
    'regress'
    >>> card["inference"]["reference_distribution"].startswith("t(")
    True
    """

    __slots__ = ()

    def to_markdown(self) -> str:
        lines = [f"### Result card: `sp.{self.get('function') or '?'}`", ""]
        for section in (
            "estimand",
            "sample",
            "specification",
            "inference",
            "evidence",
            "assumptions",
            "limitations",
            "provenance",
        ):
            body = self.get(section)
            if not body:
                continue
            lines.append(f"**{section}**")
            if isinstance(body, Mapping):
                for k, v in body.items():
                    if v in (None, [], {}):
                        continue
                    lines.append(f"- {k}: {_short(v)}")
            else:
                for item in body:
                    lines.append(f"- {_short(item)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    __str__ = to_markdown

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.to_markdown()


def _short(v: Any, limit: int = 240) -> str:
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _jsonable(v: Any) -> Any:
    """JSON-safe copy: numpy scalars -> Python, frames/arrays summarised."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, pd.DataFrame):
        return f"<DataFrame {v.shape[0]}x{v.shape[1]}>"
    if isinstance(v, (pd.Series, np.ndarray)):
        arr = np.asarray(v)
        if arr.size <= 12:
            return [_jsonable(x) for x in arr.ravel().tolist()]
        return f"<array shape={list(arr.shape)}>"
    return repr(v)


def _info(result: Any, name: str) -> Dict[str, Any]:
    d = getattr(result, name, None)
    return dict(d) if isinstance(d, Mapping) else {}


def _function_name(result: Any) -> Optional[str]:
    prov = getattr(result, "_provenance", None)
    fn = getattr(prov, "function", None)
    if isinstance(fn, str) and fn:
        return fn.rsplit(".", 1)[-1]
    from .validation_scope import _function_of

    return _function_of(result)


def _estimand(result: Any, mi: Dict[str, Any], scale: Optional[str]) -> Dict[str, Any]:
    label = getattr(result, "estimand", None)
    if not isinstance(label, str) or not label:
        model_type = mi.get("model_type") or type(result).__name__
        label = f"coefficients of {model_type}"
    out: Dict[str, Any] = {"label": label}
    if scale:
        out["scale"] = scale
    for k in _ESTIMAND_KEYS:
        if mi.get(k) is not None:
            out[k] = _jsonable(mi[k])
    return out


def _sample(result: Any, mi: Dict[str, Any], di: Dict[str, Any]) -> Dict[str, Any]:
    n_used = di.get("nobs")
    if n_used is None:
        n_used = getattr(result, "n_obs", None)
    prov = getattr(result, "_provenance", None)
    shape = getattr(prov, "data_shape", None)
    # markout decorators record the caller's row count; an estimator that
    # attaches provenance after the markout only saw the reduced frame.
    n_input = mi.get("n_input_rows")
    if n_input is None and shape:
        n_input = int(shape[0])
    out: Dict[str, Any] = {
        "n_used": _jsonable(n_used),
        "n_input_rows": _jsonable(n_input),
    }
    if n_input is not None and n_used is not None:
        try:
            out["n_not_used"] = int(n_input) - int(n_used)
        except (TypeError, ValueError):
            pass
    exclusions = {
        k: _jsonable(v)
        for k, v in mi.items()
        if _EXCLUSION_RE.match(str(k)) and isinstance(v, (int, np.integer))
    }
    if exclusions:
        out["exclusions"] = exclusions
    for k in _STRUCTURE_KEYS:
        if mi.get(k) is not None:
            out[k] = _jsonable(mi[k])
    return out


def _specification(result: Any, mi: Dict[str, Any]) -> Dict[str, Any]:
    prov = getattr(result, "_provenance", None)
    params = dict(getattr(prov, "params", None) or {})
    call = {k: _jsonable(v) for k, v in params.items() if v is not None}
    out: Dict[str, Any] = {}
    formula = mi.get("formula") or params.get("formula") or params.get("fml")
    if formula:
        out["formula"] = formula
    for key in ("weights", "offset", "exposure", "absorb", "fixed_effects"):
        v = mi.get(key, params.get(key))
        if v is not None and v is not False:
            out[key] = _jsonable(v)
    seed = params.get("random_state", params.get("seed", mi.get("random_state")))
    if seed is not None:
        out["seed"] = _jsonable(seed)
    if call:
        out["call_arguments"] = call
    return out


def _inference(result: Any, mi: Dict[str, Any], di: Dict[str, Any]) -> Dict[str, Any]:
    from .postestimation._covariance import coefficient_covariance, inference_df

    out: Dict[str, Any] = {}
    for k in _INFERENCE_KEYS:
        if mi.get(k) is not None:
            out[k] = _jsonable(mi[k])
    df = inference_df(result)
    out["reference_distribution"] = (
        f"t({int(df) if float(df).is_integer() else round(df, 3)})"
        if np.isfinite(df)
        else "normal"
    )
    alpha = getattr(result, "alpha", None)
    if isinstance(alpha, (int, float)):
        out["ci_level"] = round(1.0 - float(alpha), 6)
    if type(result).__name__ != "CausalResult" and isinstance(
        getattr(result, "params", None), pd.Series
    ):
        V, _ = coefficient_covariance(result)
        out["full_covariance"] = V is not None
    else:
        es_vcov = mi.get("event_study_vcov")
        out["joint_covariance"] = es_vcov is not None or (
            getattr(result, "_influence_funcs", None) is not None
        )
    return out


def _provenance_section(result: Any, mi: Dict[str, Any]) -> Dict[str, Any]:
    prov = getattr(result, "_provenance", None)
    out: Dict[str, Any] = {}
    for k in (
        "function",
        "statspai_version",
        "python_version",
        "data_hash",
        "data_shape",
        "run_id",
        "timestamp",
    ):
        v = getattr(prov, k, None)
        if v is not None:
            out[k] = _jsonable(v)
    backend = {k: _jsonable(mi[k]) for k in _BACKEND_KEYS if mi.get(k) is not None}
    for k in ("backend_version", "implementation"):
        if mi.get(k) is not None:
            backend[k] = _jsonable(mi[k])
    attr_backend = getattr(result, "backend", None)
    if isinstance(attr_backend, str) and "backend" not in backend:
        backend["backend"] = attr_backend
    if backend:
        out["backend"] = backend
    mod = type(result).__module__
    if mod and not mod.startswith("statspai"):
        out["result_class"] = f"{mod}.{type(result).__name__}"
    if not out.get("function"):
        out["note"] = (
            "no provenance record attached; call arguments and data fingerprint "
            "are unavailable"
        )
    return out


def _evidence(result: Any, function: Optional[str]) -> Dict[str, Any]:
    from .exceptions import MethodIncompatibility
    from .validation_scope import SCOPES, validation_scope

    name = function if function in SCOPES else None
    if name is not None:
        try:
            scope = validation_scope(result, function=name)
        except MethodIncompatibility as exc:  # e.g. out-of-domain config
            return {
                "level": "configuration",
                "status": "not_covered",
                "note": f"validation_scope could not map this fit: {exc}",
            }
        return {
            "level": "configuration",
            "status": scope["status"],
            "outputs": {o: v["status"] for o, v in scope["outputs"].items()},
            "unchecked_dimensions": list(scope["unchecked"]),
            "artifacts": sorted({m["artifact"] for m in scope["matched"]}),
            "detail": f"sp.validation_scope(result, function={name!r})",
        }
    tier = None
    if function:
        try:
            from .registry import describe_function

            tier = describe_function(function).get("validation_status")
        except (KeyError, ValueError, TypeError, AttributeError):
            tier = None
    return {
        "level": "function",
        "status": "no_configuration_map",
        "function_tier": tier,
        "note": (
            "Only a function-level evidence tier exists; it does not say "
            "whether the options used in this call were compared."
        ),
    }


def _registry_view(function: Optional[str]) -> Dict[str, Any]:
    if not function:
        return {}
    try:
        from .registry import describe_function

        return describe_function(function)
    except (KeyError, ValueError, TypeError, AttributeError):
        return {}


def _assumptions(
    result: Any, mi: Dict[str, Any], reg: Dict[str, Any]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if reg.get("assumptions"):
        out["identifying"] = list(reg["assumptions"])
        out["status"] = (
            "declared by the method; not established by this fit -- a "
            "passed diagnostic is a failure to reject, not a proof"
        )
    checks: Dict[str, Any] = {}
    pt = mi.get("pretrend_test")
    if isinstance(pt, Mapping):
        checks["pretrend_test"] = {
            k: _jsonable(v)
            for k, v in pt.items()
            if k in ("statistic", "pvalue", "p_value", "df", "test")
        }
    diag = _info(result, "diagnostics")
    for k, v in diag.items():
        if re.search(r"first[- ]stage|weak|overid|hansen|sargan", str(k), re.I):
            checks[str(k)] = _jsonable(v)
    if checks:
        out["diagnostics_run"] = checks
    return out


def _limitations(result: Any, mi: Dict[str, Any], reg: Dict[str, Any]) -> List[str]:
    out: List[str] = [str(x) for x in reg.get("limitations") or []]
    bad_se = mi.get("nonfinite_se_terms")
    if bad_se:
        out.append(f"non-finite standard errors for: {', '.join(map(str, bad_se))}")
    for d in getattr(result, "degradations", None) or []:
        if isinstance(d, Mapping):
            out.append(
                f"degraded step {d.get('section')}: {d.get('error_type')}: "
                f"{d.get('message')}"
            )
    stab = reg.get("stability")
    if stab in ("experimental", "deprecated"):
        out.append(f"registry stability: {stab}")
    return out


def _scale(result: Any, mi: Dict[str, Any]) -> Optional[str]:
    link = mi.get("link")
    if link is None:
        return None
    return {
        "identity": "linear index",
        "logit": "log-odds index (margins report Pr(y=1))",
        "probit": "probit index (margins report Pr(y=1))",
        "cloglog": "cloglog index (margins report Pr(y=1))",
        "log": "log index (margins report the expected count)",
    }.get(str(link).lower(), f"{link} index")


def result_card(result: Any) -> ResultCard:
    """Auditable summary of a fitted result, shared by Python and MCP.

    Parameters
    ----------
    result : fitted result
        Any StatsPAI result (``EconometricResults``, ``CausalResult`` or a
        domain result object). Sections the result does not record are
        left out or marked, never guessed.

    Returns
    -------
    ResultCard
        A JSON-safe ``dict`` with sections

        * ``estimand`` -- target quantity, scale, control group /
          aggregation / target sample when recorded;
        * ``sample`` -- rows used vs rows in the input data, recorded
          exclusions (missing-cluster markout, trimming, singletons, ...),
          clusters / units / periods;
        * ``specification`` -- formula, weights, offsets, fixed effects,
          seed, and the exact call arguments from the provenance record;
        * ``inference`` -- covariance convention, reference distribution
          (t(df) or normal), CI level, whether the full covariance is
          available for joint tests;
        * ``provenance`` -- function, versions, data hash and shape, run id,
          backend;
        * ``evidence`` -- configuration-level coverage from
          :func:`sp.validation_scope` where a map exists, otherwise the
          function-level tier flagged as such;
        * ``assumptions`` -- identifying assumptions (declared, not
          verified) and the diagnostics that were actually run;
        * ``limitations`` -- registry limitations, non-finite SEs,
          degraded workflow steps, experimental status.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.mpdta()
    >>> fit = sp.callaway_santanna(df, y="lemp", g="first_treat",
    ...                            t="year", i="countyreal")
    >>> card = sp.result_card(fit)
    >>> card["estimand"]["label"]
    'ATT'
    >>> card["evidence"]["level"]
    'configuration'
    >>> print(card.to_markdown().splitlines()[0])
    ### Result card: `sp.callaway_santanna`
    """
    mi = _info(result, "model_info")
    di = _info(result, "data_info")
    function = _function_name(result)
    reg = _registry_view(function)
    card = ResultCard(
        {
            "function": function,
            "estimand": _estimand(result, mi, _scale(result, mi)),
            "sample": _sample(result, mi, di),
            "specification": _specification(result, mi),
            "inference": _inference(result, mi, di),
            "provenance": _provenance_section(result, mi),
            "evidence": {
                **_evidence(result, function),
                **(
                    {"support_tier": reg["support_tier"]}
                    if reg.get("support_tier")
                    else {}
                ),
            },
            "assumptions": _assumptions(result, mi, reg),
            "limitations": _limitations(result, mi, reg),
        }
    )
    return card
