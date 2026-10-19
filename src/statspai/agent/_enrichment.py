"""Output enrichment for agent / MCP tool returns.

The bare ``to_dict(detail='agent')`` payload from a fitted result is
correct but lean. With token budgets relaxed (per-call billing rather
than per-token), we can ship more value-per-roundtrip:

* ``next_calls`` — a list of JSON-RPC ``tools/call`` payloads annotated
  with ``ready`` / ``missing_arguments``. Complete calls can be
  copy-pasted verbatim; incomplete calls say exactly what the agent must
  ask for or infer before dispatch.
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

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ----------------------------------------------------------------------
# next_calls: pre-built JSON-RPC payloads
# ----------------------------------------------------------------------

#: Map from tool family → list of follow-up tool calls. Each entry is
#: a templated tools/call payload; the caller substitutes ``result_id``
#: and any known arg values. Readiness metadata is added after alias
#: propagation so incomplete calls are explicit, not implied runnable.
#: The order encodes "what should you do next" -- most-important first,
#: so an agent on a tight token budget can stop reading after entry [0]
#: and still make a defensible move.
_FOLLOWUP_BY_TOOL: Dict[str, List[Dict[str, Any]]] = {
    "did": [
        {
            "tool": "audit_result",
            "rationale": "Reviewer-grade checklist of robustness checks for DID.",
        },
        {
            "tool": "honest_did_from_result",
            "arguments": {"method": "SD"},
            "rationale": (
                "Rambachan-Roth (2023) honest CIs in case the "
                "pre-trend test is borderline."
            ),
        },
        {
            "tool": "bacon_decomposition",
            "rationale": (
                "Goodman-Bacon weights — does TWFE give negative "
                "weight to any 2x2 comparison?"
            ),
        },
    ],
    "callaway_santanna": [
        {
            "tool": "audit_result",
            "rationale": "Cohort-by-time event study + balance + placebo.",
        },
        {
            "tool": "honest_did_from_result",
            "arguments": {"method": "SD"},
            "rationale": "Sensitivity to parallel-trends violation.",
        },
    ],
    "did_imputation": [
        {"tool": "audit_result"},
        {"tool": "honest_did_from_result", "arguments": {"method": "SD"}},
    ],
    "sun_abraham": [
        {"tool": "audit_result"},
        {"tool": "honest_did_from_result", "arguments": {"method": "SD"}},
    ],
    "rdrobust": [
        {"tool": "rdplot", "rationale": "Visual sanity-check of the discontinuity."},
        {
            "tool": "rddensity",
            "rationale": "McCrary-style manipulation test on the running variable.",
        },
        {"tool": "rdsensitivity", "rationale": "Bandwidth + kernel sensitivity."},
    ],
    "ivreg": [
        {"tool": "effective_f_test", "rationale": "Olea-Pflueger weak-IV diagnostic."},
        {
            "tool": "anderson_rubin_test",
            "rationale": "Weak-instrument-robust CI / test.",
        },
    ],
    "iv": [
        {"tool": "effective_f_test"},
        {"tool": "anderson_rubin_test"},
    ],
    "synth": [
        {"tool": "synthdid_placebo", "rationale": "In-space placebo for inference."},
        {
            "tool": "robust_synth",
            "rationale": "Robust SC variant — Ben-Michael-Feller-Rothstein 2021.",
        },
    ],
    "regress": [
        {"tool": "vif", "rationale": "Multicollinearity diagnostic."},
        {
            "tool": "het_test",
            "rationale": "Heteroskedasticity test (Breusch-Pagan / White).",
        },
    ],
    "dml": [
        {"tool": "audit_result"},
        {"tool": "spec_curve"},
    ],
    "causal_forest": [
        {"tool": "cate_summary", "rationale": "ATE + heterogeneity quantiles."},
        {
            "tool": "blp_test",
            "rationale": "Best linear projection — is heterogeneity real?",
        },
        {"tool": "calibration_test", "rationale": "Calibration of CATE predictions."},
        {
            "tool": "calibrate_cate",
            "rationale": "Rescale out-of-bag CATEs by their calibration slope.",
        },
    ],
    "did_forest": [
        {
            "tool": "calibration_test",
            "rationale": "Per-cell heterogeneity tests are in result.att_gt.",
        },
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
        {"tool": "balance_panel", "rationale": "Post-match SMD balance audit."},
        {"tool": "love_plot", "rationale": "Visual covariate-balance sanity-check."},
    ],
    "ebalance": [
        {"tool": "balanceplot"},
    ],
    "causal": [
        {"tool": "audit_result"},
        {"tool": "spec_curve"},
    ],
}


@lru_cache(maxsize=1)
def _tool_schema_index() -> Dict[str, Tuple[Set[str], Set[str]]]:
    """Return ``{tool: (required args, accepted args)}`` for advertised tools.

    Prefer the generated ``schemas/tools.json`` snapshot because this is
    the offline contract agents inspect and it avoids importing the full
    live registry inside result enrichment. If the snapshot is absent
    (an installed wheel), fall back to the live manifest. Enrichment must
    never make the primary estimator fail merely because readiness
    metadata could not be built.
    """

    def _index(
        manifest: Iterable[Dict[str, Any]],
    ) -> Dict[str, Tuple[Set[str], Set[str]]]:
        out: Dict[str, Tuple[Set[str], Set[str]]] = {}
        for tool in manifest:
            schema = tool.get("input_schema") or tool.get("inputSchema") or {}
            out[str(tool.get("name"))] = (
                set(schema.get("required") or []),
                set((schema.get("properties") or {}).keys()),
            )
        return out

    try:
        import json
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        snapshot = repo_root / "schemas" / "tools.json"
        if snapshot.exists():
            return _index(json.loads(snapshot.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, AttributeError):
        pass

    try:
        from .tools import tool_manifest

        return _index(tool_manifest())
    except (ImportError, TypeError, ValueError, AttributeError):
        return {}


def _required_args_by_tool() -> Dict[str, Set[str]]:
    """Return required manifest args for advertised tools."""
    return {name: req for name, (req, _props) in _tool_schema_index().items()}


#: Arguments the MCP server adds to every tool's schema at ``tools/list``
#: time (``mcp_server._COMMON_ARGUMENT_ORDER``); the offline snapshot
#: does not carry them.
_SERVER_COMMON_ARGS = frozenset(
    {
        "data_path",
        "data_id",
        "data_records",
        "data_csv",
        "data_columns",
        "data_sample_n",
        "result_id",
        "max_output_bytes",
        "as_handle",
        "detail",
    }
)


def _followup_base_args(base_args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common column aliases for chained follow-up calls."""
    out = dict(base_args)
    aliases = (
        ("running_var", "x"),
        ("x", "running_var"),
        ("treat", "treatment"),
        ("treatment", "treat"),
        ("id", "unit"),
        ("id", "entity"),
        ("unit", "id"),
        ("unit", "entity"),
        ("cohort", "g"),
        ("g", "cohort"),
        ("time", "t"),
        ("t", "time"),
        ("instrument", "instruments"),
        ("instruments", "instrument"),
    )
    for source, target in aliases:
        if source in out and target not in out:
            out[target] = out[source]
    return out


def _own_args(tool: str) -> Optional[Set[str]]:
    """The tool's own arguments (no server-injected ones), or ``None``."""
    entry = _tool_schema_index().get(tool)
    return None if entry is None else set(entry[1])


def _accepted_args(tool: str) -> Optional[Set[str]]:
    """Arguments ``tool`` accepts, or ``None`` when its schema is unknown."""
    own = _own_args(tool)
    return None if own is None else own | _SERVER_COMMON_ARGS


def _reads_result(tool: str, own: Optional[Set[str]]) -> bool:
    """Whether ``tool`` works from a fitted result (``result_id`` / ``result``)."""
    return own is not None and bool({"result_id", "result"} & own)


@lru_cache(maxsize=1)
def _dataless_tools() -> "frozenset[str]":
    # ``_dataless_tool_names`` handles its own registry failures (it falls
    # back to the hand-written override set); only the import can fail here.
    try:
        from .mcp_server import _dataless_tool_names
    except ImportError:
        return frozenset()
    return frozenset(_dataless_tool_names())


def _needs_data(tool: str, own: Optional[Set[str]]) -> bool:
    """Whether a call to ``tool`` needs a dataset (``data_id`` / ``data_path``)."""
    if own is None or _reads_result(tool, own):
        return False
    return tool not in _dataless_tools()


#: Server arguments that name the dataset a follow-up should run on.
_DATA_SOURCE_ARGS = ("data_id", "data_path")


def _annotate_readiness(out: Dict[str, Any]) -> Dict[str, Any]:
    """Mark whether a follow-up payload can be dispatched as written.

    Ready means: every required argument of the tool is present (a
    ``result`` parameter counts as present when ``result_id`` is, since the
    server resolves the handle), and a tool that reads a dataset has one.
    """
    tool = str(out.get("tool") or "")
    required = set(_required_args_by_tool().get(tool, set()))
    args = out.get("arguments") or {}
    if "result_id" in args:
        required.discard("result")
    missing = sorted(required - set(args))
    if _needs_data(tool, _own_args(tool)) and not any(
        k in args for k in _DATA_SOURCE_ARGS
    ):
        missing.append("data_id")
    out.pop("missing_arguments", None)
    out.pop("hint", None)
    if missing:
        out["ready"] = False
        out["missing_arguments"] = missing
        out["hint"] = "Supply missing_arguments before dispatching this follow-up" + (
            " (data_id: the dataset handle from load_data)."
            if "data_id" in missing
            else "."
        )
    else:
        out["ready"] = True
    return out


def _instantiate_followup(
    template: Dict[str, Any],
    *,
    result_id: Optional[str],
    base_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Materialise a follow-up template and annotate dispatch readiness."""
    out: Dict[str, Any] = {
        "tool": template["tool"],
        "arguments": dict(template.get("arguments") or {}),
    }
    own = _own_args(out["tool"])
    # The handle goes only to tools that read a fitted result; ``rdplot``
    # given a result_id still needs the data and reports the handle as an
    # unsupported argument.
    if (
        result_id
        and "result_id" not in out["arguments"]
        and _reads_result(out["tool"], own)
    ):
        out["arguments"]["result_id"] = result_id
    if "rationale" in template:
        out["rationale"] = template["rationale"]
    # Pass through the fit's column arguments under the target tool's own
    # parameter names. We never overwrite an explicit value the template
    # carries, and a name the tool does not take is not copied: a client
    # that validates against the schema rejects the call, and the server
    # reports it under ``_unsupported_args``.
    normalized_base_args = _followup_base_args(base_args)
    for key in (
        "y",
        "x",
        "treat",
        "treatment",
        "time",
        "t",
        "id",
        "unit",
        "entity",
        "cohort",
        "g",
        "covariates",
        "running_var",
        "c",
        "instrument",
        "instruments",
        "endog",
        "formula",
    ):
        if key in normalized_base_args and key not in out["arguments"]:
            if own is not None and key not in own:
                continue
            out["arguments"][key] = normalized_base_args[key]
    if _needs_data(out["tool"], own):
        for key in _DATA_SOURCE_ARGS:
            if base_args.get(key):
                out["arguments"].setdefault(key, base_args[key])
                break
    return _annotate_readiness(out)


def attach_data_source(payload: Dict[str, Any], data_args: Dict[str, Any]) -> None:
    """Give the dataset handle to follow-ups that need data, then re-check them.

    The MCP server strips ``data_id`` / ``data_path`` from the arguments
    before the estimator (and the enrichment) sees them; it calls this
    afterwards with the source it resolved, so a follow-up such as
    ``rdplot`` arrives with the same dataset instead of a missing ``data``.
    """
    calls = payload.get("next_calls") if isinstance(payload, dict) else None
    if not calls:
        return
    source = {k: data_args[k] for k in _DATA_SOURCE_ARGS if data_args.get(k)}
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool") or "")
        args = call.setdefault("arguments", {})
        if (
            source
            and _needs_data(tool, _own_args(tool))
            and not any(k in args for k in _DATA_SOURCE_ARGS)
        ):
            key = "data_id" if "data_id" in source else "data_path"
            args[key] = source[key]
        if tool in _tool_schema_index():
            _annotate_readiness(call)


def violation_next_calls(
    payload: Dict[str, Any],
    *,
    tool_name: str,
    result_id: Optional[str] = None,
    base_args: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Follow-up calls carried by the payload's ``violations``.

    The design guardrails (``core/_design_guardrails.py``) attach
    ``next_calls`` templates to their violations. A template with
    ``same_call: true`` re-runs this tool with the caller's arguments, minus
    ``drop_arguments``, plus its own ``arguments`` (for example the same
    regression clustered at the treatment's assignment level).
    """
    base_args = base_args or {}
    out: List[Dict[str, Any]] = []
    for v in payload.get("violations") or []:
        if not isinstance(v, dict):
            continue
        for t in v.get("next_calls") or []:
            if not isinstance(t, dict) or not t.get("tool"):
                continue
            if t.get("same_call"):
                drop = set(t.get("drop_arguments") or [])
                args = {k: val for k, val in base_args.items() if k not in drop}
                args.update(t.get("arguments") or {})
                call: Dict[str, Any] = {"tool": tool_name, "arguments": args}
                if "rationale" in t:
                    call["rationale"] = t["rationale"]
                out.append(_annotate_readiness(call))
            else:
                out.append(
                    _instantiate_followup(t, result_id=result_id, base_args=base_args)
                )
    return out


def build_next_calls(
    tool_name: str,
    *,
    result_id: Optional[str] = None,
    base_args: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return pre-filled follow-up calls with readiness metadata."""
    base_args = base_args or {}
    templates = _FOLLOWUP_BY_TOOL.get(tool_name, [])
    return [
        _instantiate_followup(t, result_id=result_id, base_args=base_args)
        for t in templates
    ]


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
    "did_multiplegt": ["dechaisemartin2020two"],
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
    "did_forest": [
        "gavrilova2025difference",
        "callaway2021difference",
        "athey2019generalized",
    ],
    "calibrate_cate": ["chernozhukov2025generic"],
    "calibration_test": ["chernozhukov2025generic"],
    "metalearner": ["kunzel2019metalearners"],
    "tmle": ["vanderlaan2011targeted"],
    "drdid": ["santanna2020doubly"],
    "evalue": ["vanderweele2017sensitivity"],
    "sensemakr": ["cinelli2020making"],
    "oster_bounds": ["oster2019unobservable"],
    "spec_curve": ["simonsohn2020specification"],
    "bacon_plot": ["goodmanbacon2021difference"],
}


def _bib_keys_from_reference(tool_name: str) -> List[str]:
    """Derive bib keys from a registered function's ``reference`` field.

    The curated convention writes references as ``Author (Year)`` plus a
    Pandoc citation marker. We extract those citation tokens and keep only
    keys that actually resolve in ``paper.bib`` — so the citation-
    hallucination red line (CLAUDE.md §10) holds even though this path is
    auto-derived: an unverified key is silently dropped, never surfaced.
    """
    import re

    try:
        from ..registry import _REGISTRY, _ensure_full_registry

        _ensure_full_registry()
        spec = _REGISTRY.get(tool_name)
    except (AttributeError, ImportError, KeyError, RuntimeError):
        spec = None
    if spec is None or not getattr(spec, "reference", ""):
        return []
    candidates = re.findall(r"\[@([\w:./-]+)\]", spec.reference)
    if not candidates:
        return []
    present = fetch_bibtex(candidates)
    # Preserve order, drop unverified keys.
    return [k for k in candidates if present.get(k)]


def build_citations(tool_name: str) -> List[str]:
    """Return the verified bib keys for ``tool_name`` (empty list = none).

    Precedence: the hand-curated :data:`_CITATIONS_BY_TOOL` map wins (it
    is reviewed and may rank multiple canonical refs); otherwise fall back
    to bib keys parsed from the function's registry ``reference`` field.
    The fallback only emits keys verified to exist in ``paper.bib``, so it
    cannot introduce a hallucinated citation — it just lets the hundreds of
    carded estimators carry their reference automatically instead of
    requiring a second hand-maintained table.
    """
    keys = list(_CITATIONS_BY_TOOL.get(tool_name, []))
    if keys:
        return keys
    return _bib_keys_from_reference(tool_name)


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

    def _first(*keys: str) -> Any:
        # ``0.0`` is a legitimate p-value / estimate: test for presence,
        # not truthiness.
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        return None

    est = payload.get("estimate")
    se = _first("std_error", "se")
    p = _first("p_value", "pvalue")
    lo, hi = payload.get("conf_low"), payload.get("conf_high")
    ci = payload.get("ci")
    if (lo is None or hi is None) and isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo, hi = ci
    alpha = payload.get("alpha")
    level = 95.0
    if isinstance(alpha, (int, float)) and 0 < alpha < 1:
        level = 100.0 * (1.0 - float(alpha))
    if isinstance(est, (int, float)):
        line = f"Point estimate: {est:.4g}"
        if isinstance(se, (int, float)):
            line += f" (SE {se:.3g})"
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            line += f"; {level:g}% CI [{lo:.3g}, {hi:.3g}]"
        if isinstance(p, (int, float)):
            line += f"; p={p:.3g}"
        parts.append(line)

    if n_obs:
        parts.append(f"N = {n_obs:,}")

    violations = payload.get("violations") or []
    if violations:
        parts.append(f"Violations flagged: {len(violations)}")

    next_steps = payload.get("next_steps") or []
    if next_steps:
        actions = []
        for step in list(next_steps)[:2]:
            if isinstance(step, dict):
                actions.append(
                    f"`{step.get('action')}`"
                    if step.get("action")
                    else str(step.get("reason", ""))
                )
            else:
                actions.append(str(step))
        parts.append("Next steps: " + "; ".join(a for a in actions if a))

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

    try:
        flagged = violation_next_calls(
            payload, tool_name=tool_name, result_id=result_id, base_args=base_args
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        from ..workflow._degradation import record_degradation

        degradations = payload.setdefault("degradations", [])
        if isinstance(degradations, list):
            record_degradation(
                degradations, section="next_calls", exc=exc, detail=tool_name
            )
        flagged = []

    if "next_calls" not in payload:
        try:
            nc = build_next_calls(
                tool_name, result_id=result_id, base_args=base_args or {}
            )
        except (
            AttributeError,
            ImportError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:  # the fit itself succeeded; say the follow-ups are missing
            from ..workflow._degradation import record_degradation

            degradations = payload.setdefault("degradations", [])
            if isinstance(degradations, list):
                record_degradation(
                    degradations, section="next_calls", exc=exc, detail=tool_name
                )
            nc = []
        if nc:
            payload["next_calls"] = nc
    if flagged:
        # design problems first: they say the estimate answers another
        # question, which outranks the routine follow-ups
        rest = [
            c
            for c in payload.get("next_calls") or []
            if not any(
                isinstance(c, dict)
                and c.get("tool") == f["tool"]
                and c.get("arguments") == f["arguments"]
                for f in flagged
            )
        ]
        payload["next_calls"] = flagged + rest

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
    "violation_next_calls",
    "build_citations",
    "fetch_bibtex",
    "build_narrative",
    "enrich_payload",
]
