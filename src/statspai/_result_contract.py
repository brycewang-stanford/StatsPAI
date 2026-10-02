"""Generic agent result contract for result classes outside the core trees.

``AGENTS.md`` and ``schemas/result.schema.json`` promise that *every* fitted
result answers ``to_dict(detail=...)``, ``violations()``, ``next_steps()``,
``result_card()`` and ``cite()``. :class:`~statspai.core.results.CausalResult`
and :class:`~statspai.core.results.EconometricResults` implement these with
method-specific logic. The ~250 lighter domain result classes share
:class:`~statspai._result_serialize.ResultProtocolMixin`, which delegates to
the helpers here so the contract has one generic implementation (CLAUDE.md §4).

The helpers only *read* what a result already stores — they never refit,
never guess an estimate the class did not record, and fill a missing field
with ``None`` rather than inventing it.

Kept in its own module (imported lazily from the mixin methods) so the cold
import path does not pay for it.
"""

from __future__ import annotations

import inspect
import numbers
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: ``detail=`` levels accepted by every result's ``to_dict``.
DETAIL_LEVELS: Tuple[str, ...] = ("minimal", "standard", "agent")

#: Keys of the ``detail="minimal"`` envelope, in order. Same set as
#: ``CausalResult.to_dict(detail="minimal")``.
MINIMAL_KEYS: Tuple[str, ...] = (
    "method",
    "estimand",
    "estimate",
    "se",
    "pvalue",
    "ci",
    "alpha",
    "n_obs",
    "citation_key",
)

#: Keys the ``detail="agent"`` level adds on top of the minimal envelope.
AGENT_KEYS: Tuple[str, ...] = (
    "diagnostics",
    "violations",
    "warnings",
    "next_steps",
    "suggested_functions",
    "degradations",
)

# Attribute spellings probed for each envelope field, first hit wins. Only
# names whose meaning is unambiguous across the package are listed.
_METHOD_ATTRS = ("method", "method_name", "estimator", "model_type")
_ESTIMAND_ATTRS = ("estimand",)
_ESTIMATE_ATTRS = ("estimate", "point_estimate", "att", "ate", "late")
_SE_ATTRS = ("se", "std_error", "standard_error", "stderr")
_PVALUE_ATTRS = ("pvalue", "p_value", "pval")
_CI_ATTRS = ("ci", "conf_int", "confint")
_CI_PAIR_ATTRS = (("ci_lower", "ci_upper"), ("ci_low", "ci_high"))
_NOBS_ATTRS = ("n_obs", "nobs", "n")

#: Argument names that carry a seed. Until 2026-10 only the first two were
#: read, so a function spelling it otherwise (``boot_seed`` on the
#: imputation / two-stage DiD estimators, ``bootstrap_seed``, ...) got a
#: card that said nothing about the seed of its stochastic output.
#: ``scripts/seed_inventory.py`` lists every seed-like parameter in the
#: registry and ``tests/test_seed_contract.py`` fails on a new spelling.
_SEED_PARAMS = (
    "seed",
    "random_state",
    "boot_seed",
    "bootstrap_seed",
    "rng_seed",
    "wild_seed",
    "halton_seed",
    "rng",
)


def check_detail(detail: str) -> None:
    """Raise ``ValueError`` for an unknown ``detail`` level (core wording)."""
    if detail not in DETAIL_LEVELS:
        raise ValueError(
            "detail must be 'minimal', 'standard', or 'agent'; " f"got {detail!r}"
        )


# ---------------------------------------------------------------------- #
#  Attribute probing
# ---------------------------------------------------------------------- #


class _Reader:
    """Read attributes off a result, recording (not hiding) failures."""

    def __init__(self, obj: Any, degradations: List[Dict[str, Any]]):
        self.obj = obj
        self.degradations = degradations

    def get(self, name: str) -> Any:
        try:
            return getattr(self.obj, name, None)
        except Exception as exc:  # a raising property is a class defect
            from .workflow._degradation import record_degradation

            record_degradation(self.degradations, section=f"attribute:{name}", exc=exc)
            return None

    def first(self, names: Tuple[str, ...], accept: Any) -> Any:
        for name in names:
            val = self.get(name)
            if val is not None and accept(val):
                return val
        return None


def _is_number(v: Any) -> bool:
    return isinstance(v, numbers.Number) and not isinstance(v, bool)


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


def _is_count(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, numbers.Number):
        return False
    try:
        return float(v).is_integer() and float(v) >= 0
    except (TypeError, ValueError, OverflowError):
        return False


def _ci_pair(reader: _Reader) -> Optional[List[Any]]:
    from .core.results import _to_jsonable

    ci = reader.first(_CI_ATTRS, lambda v: not isinstance(v, (str, Mapping)))
    if ci is not None:
        try:
            if len(ci) == 2 and _is_number(ci[0]) and _is_number(ci[1]):
                return [_to_jsonable(ci[0]), _to_jsonable(ci[1])]
        except (TypeError, KeyError, IndexError):
            pass
    for lo_name, hi_name in _CI_PAIR_ATTRS:
        lo, hi = reader.get(lo_name), reader.get(hi_name)
        if _is_number(lo) and _is_number(hi):
            return [_to_jsonable(lo), _to_jsonable(hi)]
    return None


def citation_keys(obj: Any) -> List[str]:
    """Verified paper.bib keys a result declares (never generated)."""
    keys: List[str] = []
    single = getattr(obj, "_citation_key", None)
    if isinstance(single, str) and single:
        keys.append(single)
    for attr in ("_citation_keys", "bib_keys"):
        seq = getattr(obj, attr, None)
        if isinstance(seq, (list, tuple)):
            keys.extend(str(k) for k in seq if isinstance(k, str) and k)
    seen: List[str] = []
    for k in keys:
        if k not in seen:
            seen.append(k)
    return seen


def _diagnostics(reader: _Reader) -> Dict[str, Any]:
    from .core.results import _filter_jsonable_scalars

    out: Dict[str, Any] = {}
    for name in ("model_info", "diagnostics"):
        val = reader.get(name)
        if isinstance(val, Mapping):
            out.update(_filter_jsonable_scalars(dict(val)))
    return out


def minimal_envelope(
    obj: Any, degradations: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """The ``detail="minimal"`` envelope, read off whatever ``obj`` stores."""
    from .core.results import _to_jsonable

    reader = _Reader(obj, degradations if degradations is not None else [])
    method = reader.first(_METHOD_ATTRS, _is_str)
    if method is None:
        mi = reader.get("model_info")
        if isinstance(mi, Mapping):
            method = next(
                (mi[k] for k in ("method", "model_type") if _is_str(mi.get(k))),
                None,
            )
    keys = citation_keys(obj)
    alpha = reader.get("alpha")
    return {
        "method": str(method) if method is not None else type(obj).__name__,
        "estimand": _to_jsonable(reader.first(_ESTIMAND_ATTRS, _is_str)),
        "estimate": _to_jsonable(reader.first(_ESTIMATE_ATTRS, _is_number)),
        "se": _to_jsonable(reader.first(_SE_ATTRS, _is_number)),
        "pvalue": _to_jsonable(reader.first(_PVALUE_ATTRS, _is_number)),
        "ci": _ci_pair(reader),
        "alpha": (
            _to_jsonable(alpha)
            if _is_number(alpha) and 0.0 < float(alpha) < 1.0
            else None
        ),
        "n_obs": _to_jsonable(reader.first(_NOBS_ATTRS, _is_count)),
        "citation_key": keys[0] if keys else None,
    }


# ---------------------------------------------------------------------- #
#  violations / next_steps
# ---------------------------------------------------------------------- #


def generic_violations(obj: Any) -> List[Dict[str, Any]]:
    """Violations derivable from what a domain result already stored.

    Runs the same pattern-matching detector ``CausalResult.violations`` uses
    on the result's ``model_info`` and ``diagnostics`` mappings (pre-trend
    p-value, first-stage F, rhat / ESS, few clusters, non-finite SE, ...),
    and turns every recorded ``degradations`` entry into a ``warning``. It
    never re-runs a test; an empty list means nothing stored was flagged.
    """
    from types import SimpleNamespace

    from .core._agent_summary import causal_violations

    merged: Dict[str, Any] = {}
    for name in ("model_info", "diagnostics"):
        val = getattr(obj, name, None)
        if isinstance(val, Mapping):
            merged.update(val)
    env = minimal_envelope(obj)
    reader = _Reader(obj, [])
    # Raw values (NaN kept): a NaN estimate is a finding, an estimate the
    # class never records is not.
    est = reader.first(_ESTIMATE_ATTRS, _is_number)
    se = reader.first(_SE_ATTRS, _is_number)
    out: List[Dict[str, Any]] = []
    if merged or est is not None:
        view = SimpleNamespace(
            model_info=merged, method=env["method"], estimate=est, se=se
        )
        for v in causal_violations(view):
            if v.get("test") == "estimate_finite" and est is None:
                continue
            if v.get("test") == "se_positive" and se is None:
                continue
            out.append(v)
    for d in getattr(obj, "degradations", None) or []:
        if not isinstance(d, Mapping):
            continue
        section = str(d.get("section", "unknown"))
        out.append(
            {
                "kind": "degradation",
                "severity": "warning",
                "test": f"degraded:{section}",
                "value": None,
                "threshold": None,
                "message": (
                    f"Step '{section}' degraded: {d.get('error_type')}: "
                    f"{d.get('message')}"
                ),
                "recovery_hint": (
                    "Inspect result.degradations; the reported numbers "
                    "exclude the degraded step."
                ),
                "alternatives": [],
            }
        )
    return out


def generic_next_steps(obj: Any) -> List[Dict[str, str]]:
    """Registry-derived follow-ups for a domain result.

    Uses the producing function's registry card: each recorded failure mode
    becomes a ``diagnostics`` step carrying its remedy, each registered
    alternative a ``robustness`` step. Returns ``[]`` when the producing
    function cannot be resolved unambiguously (no guessing).
    """
    function = resolve_function(obj)
    if not function:
        return []
    card = _registry_card(function)
    steps: List[Dict[str, str]] = []
    for fm in card.get("failure_modes") or []:
        if not isinstance(fm, Mapping) or not fm.get("remedy"):
            continue
        steps.append(
            {
                "action": str(fm["remedy"]),
                "reason": f"Known failure mode of sp.{function}: {fm.get('symptom')}",
                "priority": "recommended",
                "category": "diagnostics",
            }
        )
    for alt in card.get("alternatives") or []:
        alt_name = str(alt).split("(")[0].strip()
        if alt_name.startswith("sp."):
            alt_name = alt_name[3:]
        if not alt_name:
            continue
        steps.append(
            {
                "action": f"sp.{alt_name}(...)",
                "reason": (
                    f"Registered alternative to sp.{function}; compare the "
                    "estimates as a robustness check."
                ),
                "priority": "optional",
                "category": "robustness",
                "suggest_function": f"sp.{alt_name}",
            }
        )
    return steps


def _registry_card(function: str) -> Dict[str, Any]:
    from .registry import describe_function

    try:
        return dict(describe_function(function))
    except (KeyError, ValueError):
        return {}


_RETURN_INDEX: Optional[Dict[str, List[str]]] = None
_RETURN_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)")


def _return_class_index() -> Dict[str, List[str]]:
    """``{result class name: [registered functions returning it]}``."""
    global _RETURN_INDEX
    if _RETURN_INDEX is None:
        from . import registry as _reg

        _reg._ensure_full_registry()
        index: Dict[str, List[str]] = {}
        for name, spec in _reg._REGISTRY.items():
            m = _RETURN_RE.match(getattr(spec, "returns", "") or "")
            if m:
                index.setdefault(m.group(1), []).append(name)
        _RETURN_INDEX = index
    return _RETURN_INDEX


def resolve_function(obj: Any) -> Optional[str]:
    """The registered ``sp.<name>`` that produced ``obj``, when knowable.

    The provenance record wins; otherwise the registry's ``returns`` field
    is matched against the result class name and used only when exactly one
    function returns that class.
    """
    prov = getattr(obj, "_provenance", None)
    fn = getattr(prov, "function", None)
    if isinstance(fn, str) and fn:
        return fn.rsplit(".", 1)[-1]
    candidates = _return_class_index().get(type(obj).__name__, [])
    return candidates[0] if len(candidates) == 1 else None


# ---------------------------------------------------------------------- #
#  Seeds (result_card provenance)
# ---------------------------------------------------------------------- #


def _resolve_callable(function: str) -> Any:
    import statspai

    obj: Any = statspai
    for part in function.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if callable(obj) else None


def seed_record(
    function: Optional[str],
    params: Mapping[str, Any],
    model_info: Mapping[str, Any],
) -> Dict[str, Any]:
    """``{seed, reproducible, seed_source[, seed_note]}`` for a result's card.

    * A seed recorded in the call arguments or ``model_info`` is reported
      with ``reproducible: True`` (an integer) or ``None`` (a Generator /
      RandomState, whose state the card cannot see).
    * A seed *recorded as* ``None`` is reported as ``seed: None,
      reproducible: False`` rather than omitted.
    * When the producing function takes ``seed`` / ``random_state`` but the
      record does not mention it, the value is unknown -- estimators that
      attach provenance by hand often record a curated subset of the call
      -- so ``reproducible`` is ``None`` with ``seed_source:
      "not_recorded"``; the card never claims a value it did not see.
    * Empty when nothing seed-like is known (non-stochastic function).

    Reads the record only -- never changes an estimator's default seed.
    """
    for source, record in (("call", params), ("model_info", model_info)):
        for key in _SEED_PARAMS:
            if record.get(key) is not None:
                return _seed_value(record[key], source)
    for source, record in (("call", params), ("model_info", model_info)):
        for key in _SEED_PARAMS:
            if key in record:  # recorded, and recorded as None
                return {
                    "seed": None,
                    "reproducible": False,
                    "seed_source": source,
                    "seed_note": (
                        f"{key}=None: any stochastic step this call ran "
                        "(bootstrap, sample splitting, random "
                        "initialisation) is not reproducible run-to-run; "
                        "deterministic paths are unaffected."
                    ),
                }
    fn = _resolve_callable(function) if function else None
    if fn is None:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    name = next((k for k in _SEED_PARAMS if k in sig.parameters), None)
    if name is None:
        return {}
    return {
        "seed": None,
        "reproducible": None,
        "seed_source": "not_recorded",
        "seed_note": (
            f"sp.{function} accepts {name}= but the call record does not "
            "include it, so the seed used is unknown."
        ),
    }


def _seed_value(value: Any, source: str) -> Dict[str, Any]:
    import numpy as np

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {"seed": int(value), "reproducible": True, "seed_source": source}
    # A Generator / RandomState / summarised repr: reproducibility depends
    # on state we cannot see.
    return {"seed": str(value), "reproducible": None, "seed_source": source}


# ---------------------------------------------------------------------- #
#  to_dict(detail=...) assembly
# ---------------------------------------------------------------------- #


def build_payload(obj: Any, detail: str, base: Any = None) -> Dict[str, Any]:
    """Assemble ``to_dict(detail=...)`` for ``"minimal"`` / ``"agent"``.

    ``base`` is the class's own (legacy, ``"standard"``) dict. At the agent
    level its fields are kept top-level — as ``CausalResult`` keeps its
    standard fields — and the envelope is laid over them: a class field
    that already carries an envelope name (``method``, ``estimate``,
    ``diagnostics`` ...) wins, while the contract lists (``violations``,
    ``warnings``, ``next_steps``, ``suggested_functions``,
    ``degradations``) always hold the contract value. A class field that
    collided with one of those lists is kept as ``<name>_field``.

    Never raises on a result's content: failures of the detectors are
    recorded under ``degradations`` (and warned about), as the core
    classes do (CLAUDE.md §3.7).
    """
    from .core.results import _to_jsonable
    from .workflow._degradation import record_degradation

    check_detail(detail)
    degradations: List[Dict[str, Any]] = []
    env = minimal_envelope(obj, degradations)
    if detail == "minimal":
        return env

    if base is None:
        base = {}
    elif not isinstance(base, Mapping):
        base = {"value": base}

    try:
        viols = list(obj.violations() or [])
    except Exception as exc:
        viols = []
        record_degradation(degradations, section="violations", exc=exc)
    try:
        steps = _call_next_steps(obj)
    except Exception as exc:
        steps = []
        record_degradation(degradations, section="next_steps", exc=exc)

    warns: List[str] = [
        str(v.get("message"))
        for v in viols
        if isinstance(v, Mapping) and v.get("message")
    ]
    own_warn = getattr(obj, "warnings", None)
    if isinstance(own_warn, (list, tuple)):
        for w in own_warn:
            if isinstance(w, str) and w not in warns:
                warns.append(w)

    suggested: List[str] = []
    for s in steps:
        fn = (
            s.get("suggest_function") or s.get("function")
            if isinstance(s, Mapping)
            else None
        )
        if fn and fn not in suggested:
            suggested.append(fn)
    for v in viols:
        if not isinstance(v, Mapping):
            continue
        for alt in v.get("alternatives", []) or []:
            if alt and alt not in suggested:
                suggested.append(alt)

    own_degr = [
        dict(d)
        for d in (getattr(obj, "degradations", None) or [])
        if isinstance(d, Mapping)
    ]

    out: Dict[str, Any] = dict(base)
    for key, val in env.items():
        if out.get(key) is None:
            out[key] = val
    if not isinstance(out.get("diagnostics"), Mapping):
        if "diagnostics" in out and out["diagnostics"] is not None:
            out["diagnostics_field"] = out["diagnostics"]
        out["diagnostics"] = _diagnostics(_Reader(obj, degradations))
    contract = {
        "violations": viols,
        "warnings": warns,
        "next_steps": steps[:8],
        "suggested_functions": suggested,
        "degradations": own_degr + degradations,
    }
    for key, val in contract.items():
        if key in out and out[key] not in (None, [], val) and key != "degradations":
            out[f"{key}_field"] = out[key]
        out[key] = val
    out.setdefault("result_class", type(obj).__name__)
    keys = citation_keys(obj)
    if len(keys) > 1:
        out.setdefault("citation_keys", keys)
    payload: Dict[str, Any] = _to_jsonable(out)
    return payload


def _call_next_steps(obj: Any) -> List[Dict[str, Any]]:
    """Call ``obj.next_steps`` without letting it print, whatever its shape."""
    fn = obj.next_steps
    try:
        takes_flag = "print_result" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes_flag = False
    raw = fn(print_result=False) if takes_flag else fn()
    steps: List[Dict[str, Any]] = []
    for s in raw or []:
        if isinstance(s, Mapping):
            steps.append(dict(s))
        elif isinstance(s, str):
            # Some classes return bare action strings; normalise to the
            # core Step shape so agents see one type.
            steps.append(
                {
                    "action": s,
                    "reason": "",
                    "priority": "recommended",
                    "category": "workflow",
                }
            )
        elif hasattr(s, "to_dict"):
            steps.append(dict(s.to_dict()))
    return steps
