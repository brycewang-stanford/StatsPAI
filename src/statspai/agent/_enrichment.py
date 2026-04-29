"""Output enrichment for agent / MCP tool returns.

The bare ``to_dict(detail='agent')`` payload from a fitted result is
correct but lean. With token budgets relaxed (per-call billing rather
than per-token), we can ship more value-per-roundtrip:

* ``next_calls`` — a list of ready-to-dispatch JSON-RPC ``tools/call``
  payloads the agent can copy-paste verbatim. Eliminates the "what do
  I call next?" reflection step.
* ``citations`` — bib keys for the methods used + verified BibTeX
  bodies pulled from ``paper.bib``. Closes the citation-hallucination
  loophole at the source.
* ``narrative`` — a short markdown summary the agent can quote in chat
  without re-paraphrasing the JSON.

The functions in this module are pure (no I/O, no caching) so they're
safe to call inside any serializer. They degrade gracefully when the
result object lacks the expected fields.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


# ----------------------------------------------------------------------
# next_calls: pre-built JSON-RPC payloads
# ----------------------------------------------------------------------

#: Map from tool family → list of follow-up tool calls. Each entry is
#: a templated tools/call payload; the caller substitutes ``result_id``
#: and any required arg values. The order encodes "what should you do
#: next" — most-important first, so an agent on a tight token budget
#: can stop reading after entry [0] and still make a defensible move.
_FOLLOWUP_BY_TOOL: Dict[str, List[Dict[str, Any]]] = {
    "did": [
        {"tool": "audit_result",
         "rationale": "Reviewer-grade checklist of robustness checks for DID."},
        {"tool": "honest_did_from_result",
         "arguments": {"method": "SD"},
         "rationale": ("Rambachan-Roth (2023) honest CIs in case the "
                        "pre-trend test is borderline.")},
        {"tool": "bacon_decomposition",
         "rationale": ("Goodman-Bacon weights — does TWFE give negative "
                        "weight to any 2x2 comparison?")},
    ],
    "callaway_santanna": [
        {"tool": "audit_result",
         "rationale": "Cohort-by-time event study + balance + placebo."},
        {"tool": "honest_did_from_result",
         "arguments": {"method": "SD"},
         "rationale": "Sensitivity to parallel-trends violation."},
    ],
    "did_imputation": [
        {"tool": "audit_result"},
        {"tool": "honest_did_from_result",
         "arguments": {"method": "SD"}},
    ],
    "sun_abraham": [
        {"tool": "audit_result"},
        {"tool": "honest_did_from_result",
         "arguments": {"method": "SD"}},
    ],
    "rdrobust": [
        {"tool": "rdplot",
         "rationale": "Visual sanity-check of the discontinuity."},
        {"tool": "rddensity",
         "rationale": "McCrary-style manipulation test on the running variable."},
        {"tool": "rdsensitivity",
         "rationale": "Bandwidth + kernel sensitivity."},
    ],
    "ivreg": [
        {"tool": "effective_f_test",
         "rationale": "Olea-Pflueger weak-IV diagnostic."},
        {"tool": "anderson_rubin_test",
         "rationale": "Weak-instrument-robust CI / test."},
        {"tool": "sensitivity_from_result",
         "arguments": {"method": "evalue"},
         "rationale": "E-value bound on omitted-confounder strength."},
    ],
    "iv": [
        {"tool": "effective_f_test"},
        {"tool": "anderson_rubin_test"},
    ],
    "synth": [
        {"tool": "synthdid_placebo",
         "rationale": "In-space placebo for inference."},
        {"tool": "robust_synth",
         "rationale": "Robust SC variant — Ben-Michael-Feller-Rothstein 2021."},
    ],
    "regress": [
        {"tool": "vif",
         "rationale": "Multicollinearity diagnostic."},
        {"tool": "het_test",
         "rationale": "Heteroskedasticity test (Breusch-Pagan / White)."},
        {"tool": "sensitivity_from_result",
         "arguments": {"method": "evalue"}},
    ],
    "dml": [
        {"tool": "audit_result"},
        {"tool": "spec_curve"},
    ],
    "causal_forest": [
        {"tool": "cate_summary",
         "rationale": "ATE + heterogeneity quantiles."},
        {"tool": "blp_test",
         "rationale": "Best linear projection — is heterogeneity real?"},
        {"tool": "calibration_test",
         "rationale": "Calibration of CATE predictions."},
    ],
    "metalearner": [
        {"tool": "cate_summary"},
        {"tool": "calibration_test"},
    ],
    "auto_cate": [
        {"tool": "cate_summary"},
        {"tool": "calibration_test"},
    ],
    "matching": [
        {"tool": "balance_panel",
         "rationale": "Post-match SMD balance audit."},
        {"tool": "love_plot",
         "rationale": "Visual covariate-balance sanity-check."},
    ],
    "ebalance": [
        {"tool": "balanceplot"},
    ],
    "causal": [
        {"tool": "audit_result"},
        {"tool": "spec_curve"},
    ],
}


def _instantiate_followup(
    template: Dict[str, Any],
    *,
    result_id: Optional[str],
    base_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Materialise a follow-up template into a callable tools/call payload."""
    out: Dict[str, Any] = {
        "tool": template["tool"],
        "arguments": dict(template.get("arguments") or {}),
    }
    if result_id and "result_id" not in out["arguments"]:
        out["arguments"]["result_id"] = result_id
    if "rationale" in template:
        out["rationale"] = template["rationale"]
    # Pass through select base args that any chained call may reuse
    # (e.g. agents commonly want the same outcome / treatment / time
    # column names without restating them). We never overwrite an
    # explicit value the template carries.
    for key in ("y", "treat", "time", "id", "data_path",
                 "running_var", "instrument"):
        if key in base_args and key not in out["arguments"]:
            out["arguments"][key] = base_args[key]
    return out


def build_next_calls(
    tool_name: str,
    *,
    result_id: Optional[str] = None,
    base_args: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return a list of pre-filled tools/call payloads for follow-ups."""
    base_args = base_args or {}
    templates = _FOLLOWUP_BY_TOOL.get(tool_name, [])
    return [_instantiate_followup(t, result_id=result_id,
                                    base_args=base_args)
            for t in templates]


# ----------------------------------------------------------------------
# citations: bib keys per estimator
# ----------------------------------------------------------------------

#: Map from tool name → ranked list of bib keys (most-canonical first).
#: Keys are pulled from the project's paper.bib — verified per CLAUDE.md
#: §10. Empty list means we explicitly do not have a verified key for
#: that estimator (don't synthesise one).
_CITATIONS_BY_TOOL: Dict[str, List[str]] = {
    "did": [],
    "callaway_santanna": ["callaway2021difference"],
    "did_imputation": ["borusyak2024revisiting"],
    "sun_abraham": ["sun2021estimating"],
    "did_multiplegt": ["dechaisemartin2020twoway"],
    "honest_did": ["rambachan2023more"],
    "bacon_decomposition": ["goodmanbacon2021difference"],
    "rdrobust": ["calonico2014robust"],
    "rdplot": ["calonico2014robust"],
    "rddensity": ["cattaneo2018manipulation"],
    "rdrandinf": ["cattaneo2015randomization"],
    "ivreg": ["andrews2019weak"],
    "iv": ["andrews2019weak"],
    "effective_f_test": ["olea2013robust"],
    "anderson_rubin_test": ["anderson1949estimation"],
    "synth": ["abadie2003economic", "abadie2010synthetic"],
    "synthdid_estimate": ["arkhangelsky2021synthetic"],
    "robust_synth": ["benmichael2021augmented"],
    "augsynth": ["benmichael2021augmented"],
    "ebalance": ["hainmueller2012entropy"],
    "sbw": ["zubizarreta2015stable"],
    "cbps": ["imai2014covariate"],
    "match": ["abadie2006large"],
    "genmatch": ["diamond2013genetic"],
    "regress": [],
    "dml": ["chernozhukov2018double"],
    "causal_forest": ["athey2019generalized", "wager2018estimation"],
    "metalearner": ["kunzel2019metalearners"],
    "tmle": ["vanderlaan2011targeted"],
    "drdid": ["santanna2020doubly"],
    "evalue": ["vanderweele2017sensitivity"],
    "sensemakr": ["cinelli2020making"],
    "oster_bounds": ["oster2019unobservable"],
    "spec_curve": ["simonsohn2020specification"],
    "bacon_plot": ["goodmanbacon2021difference"],
}


def build_citations(tool_name: str) -> List[str]:
    """Return the verified bib keys for ``tool_name`` (empty list = none)."""
    return list(_CITATIONS_BY_TOOL.get(tool_name, []))


def fetch_bibtex(keys: Iterable[str]) -> Dict[str, str]:
    """Look up bib bodies for ``keys``. Empty string ⇒ key absent."""
    if not keys:
        return {}
    from .workflow_tools import _load_bibtex_index
    index = _load_bibtex_index()
    return {k: index.get(k, "") for k in keys}


# ----------------------------------------------------------------------
# narrative: short markdown digest
# ----------------------------------------------------------------------

def build_narrative(
    tool_name: str,
    payload: Dict[str, Any],
) -> str:
    """Return a short markdown narrative for ``payload``.

    The narrative is deliberately minimal — 4–8 lines. Agents that want
    more depth should call the bespoke ``brief`` tool, which wraps
    ``CausalResult.brief()`` and produces a per-method richer summary.
    """
    if not isinstance(payload, dict):
        return ""

    parts: List[str] = []
    method = payload.get("method") or tool_name
    estimand = payload.get("estimand")
    n_obs = payload.get("n_obs")

    header_bits = [f"**{method}**"]
    if estimand:
        header_bits.append(f"({estimand})")
    parts.append(" ".join(header_bits))

    est = payload.get("estimate")
    se = payload.get("std_error") or payload.get("se")
    p = payload.get("p_value") or payload.get("pvalue")
    lo = payload.get("conf_low")
    hi = payload.get("conf_high")
    if est is not None:
        line = f"Point estimate: {est:.4g}"
        if se is not None:
            line += f" (SE {se:.3g})"
        if lo is not None and hi is not None:
            line += f"; 95% CI [{lo:.3g}, {hi:.3g}]"
        if p is not None:
            line += f"; p={p:.3g}"
        parts.append(line)

    if n_obs:
        parts.append(f"N = {n_obs:,}")

    violations = payload.get("violations") or []
    if violations:
        parts.append(f"Violations flagged: {len(violations)}")

    next_steps = payload.get("next_steps") or []
    if next_steps:
        head = "; ".join(str(s) for s in list(next_steps)[:2])
        parts.append(f"Next steps: {head}")

    return "  \n".join(parts)


# ----------------------------------------------------------------------
# enrich: glue
# ----------------------------------------------------------------------

def enrich_payload(
    payload: Dict[str, Any],
    *,
    tool_name: str,
    result_id: Optional[str] = None,
    base_args: Optional[Dict[str, Any]] = None,
    include_bibtex: bool = True,
) -> Dict[str, Any]:
    """Annotate ``payload`` with next_calls / citations / narrative.

    Mutates ``payload`` in place AND returns it (for fluent chaining).
    Existing keys win on collision so a serializer can pre-populate
    a richer narrative or override the canonical citation list.
    """
    if not isinstance(payload, dict):
        return payload

    if "next_calls" not in payload:
        nc = build_next_calls(tool_name,
                                result_id=result_id,
                                base_args=base_args or {})
        if nc:
            payload["next_calls"] = nc

    if "citations" not in payload:
        keys = build_citations(tool_name)
        if keys:
            entry: Dict[str, Any] = {"keys": keys}
            if include_bibtex:
                bib = fetch_bibtex(keys)
                # Emit BibTeX bodies only for keys we actually found —
                # missing keys signal "needs paper.bib update", and we
                # never want to invite hallucination by emitting an
                # empty stub the agent might fill in.
                bib_present = {k: v for k, v in bib.items() if v}
                if bib_present:
                    entry["bibtex"] = bib_present
            payload["citations"] = entry

    if "narrative" not in payload:
        text = build_narrative(tool_name, payload)
        if text:
            payload["narrative"] = text

    return payload


__all__ = [
    "build_next_calls",
    "build_citations",
    "fetch_bibtex",
    "build_narrative",
    "enrich_payload",
]
