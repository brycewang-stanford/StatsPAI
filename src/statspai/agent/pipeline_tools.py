"""Composite "pipeline" tools that run an end-to-end design workflow.

Each pipeline tool is a single MCP call that orchestrates several
estimator + diagnostic stages, packages everything into one rich JSON
return (with a markdown narrative), and caches the primary result so
follow-up tool calls can chain.

The motivating observation: an LLM running a single ``did → audit →
honest_did`` chain is paying three round-trips for what is really
"do the canonical reviewer-grade DID workflow". With per-call billing,
shipping the whole chain in one call cuts cost AND latency AND the
agent's failure surface (no chance of dropping a step).

Pipelines available
-------------------

* ``pipeline_did`` — DID / staggered-DID + audit + honest CIs + bacon
  decomposition + brief.
* ``pipeline_iv`` — IV + first-stage F + Anderson-Rubin + e-value.
* ``pipeline_rd`` — RD + rdplot + density test + bandwidth sensitivity.

Each pipeline tool returns:

* ``primary_result`` — a serialised view of the canonical estimate,
  cached under ``result_id``.
* ``stages`` — a list of ``{name, status, summary}`` entries, one per
  sub-step.
* ``narrative`` — a markdown report (header → estimate → diagnostics →
  robustness → conclusion) the agent can paste verbatim.
* ``next_calls`` — anything the audit flagged as "missing high-importance
  check" plus a paper-render starter.

Failures in a sub-stage are surfaced (``status: 'failed'`` + the
exception message) but never abort the pipeline — partial results are
more useful than a hard crash midway.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ._result_cache import RESULT_CACHE

# ----------------------------------------------------------------------
# Schema definitions
# ----------------------------------------------------------------------

PIPELINE_TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "pipeline_did",
        "description": (
            "End-to-end DID workflow: preflight → did/CS estimator → "
            "audit → honest-DID sensitivity → bacon decomposition → "
            "brief. Returns one markdown report + the primary "
            "result_id. Use this when the user pastes a DID dataset "
            "and asks 'is the effect real?' — the pipeline runs every "
            "diagnostic the literature expects."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string", "description": "Outcome column."},
                "treat": {
                    "type": "string",
                    "description": "Binary treatment indicator.",
                },
                "time": {"type": "string", "description": "Time column."},
                "id": {
                    "type": "string",
                    "description": ("Unit id (panel) — required for staggered-DID."),
                },
                "cohort": {
                    "type": "string",
                    "description": (
                        "First-treatment cohort column. "
                        "When supplied, dispatches "
                        "callaway_santanna instead of "
                        "classic 2x2 did."
                    ),
                },
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["y", "treat", "time"],
        },
    },
    {
        "name": "pipeline_iv",
        "description": (
            "End-to-end IV workflow: ivreg → first-stage F (effective + "
            "Olea-Pflueger) → Anderson-Rubin CI → e-value. Returns one "
            "markdown report + result_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "'y ~ x_exog + (d_endog ~ z_instrument)' style.",
                },
            },
            "required": ["formula"],
        },
    },
    {
        "name": "pipeline_rd",
        "description": (
            "End-to-end RD workflow: rdrobust → rdplot (PNG image) → "
            "rddensity (McCrary) → rdsensitivity (bandwidth). Returns "
            "one markdown report + result_id + an image content block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string"},
                "x": {"type": "string", "description": "Running variable column."},
                "c": {"type": "number", "default": 0.0, "description": "Cutoff value."},
                "fuzzy": {
                    "type": "string",
                    "description": "Treatment column for fuzzy RD (optional).",
                },
            },
            "required": ["y", "x"],
        },
    },
]


PIPELINE_TOOL_NAMES = frozenset(t["name"] for t in PIPELINE_TOOL_SPECS)


def pipeline_tool_manifest() -> List[Dict[str, Any]]:
    return [dict(t) for t in PIPELINE_TOOL_SPECS]


# ----------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------


def _stage(
    name: str,
    status: str = "ok",
    summary: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    out = {"name": name, "status": status, "summary": summary}
    out.update(extra)
    return out


def _safe_call(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, Optional[str]]:
    """Invoke ``fn`` and return ``(result, error_msg_or_none)``."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _iv_endog_summary(result: Any, formula: str) -> str:
    """``beta (SE se)`` for the endogenous regressor of an IV fit."""
    from ..core.utils import parse_formula

    try:
        endog = list(parse_formula(formula)["endogenous"])[0]
        beta = float(result.params[endog])
    except (KeyError, IndexError, TypeError, ValueError):
        return ""
    bits = [f"{endog}={beta:.4g}"]
    try:
        bits.append(f"(SE {float(result.std_errors[endog]):.3g})")
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return " ".join(bits)


def _honest_did_summary(honest: Any) -> str:
    """Breakdown M for a Rambachan-Roth sensitivity curve.

    The stage used to report the string "computed", which tells a reader
    nothing about the only question honest_did exists to answer: how far
    parallel trends can be violated before the effect stops being
    distinguishable from zero.
    """
    payload = _light_serialize(honest)
    if not isinstance(payload, dict):
        return "computed"
    grid, rejects = payload.get("M"), payload.get("rejects_zero")
    if not isinstance(grid, dict) or not isinstance(rejects, dict):
        return "computed"
    keys = sorted(grid, key=lambda k: int(k))
    surviving = [float(grid[k]) for k in keys if rejects.get(k)]
    if not surviving:
        return "zero is inside the interval even at M=0"
    breakdown = max(surviving)
    if len(surviving) == len(keys):
        return f"rejects zero across the whole grid (M<={breakdown:.3g})"
    return f"breakdown M={breakdown:.3g} (rejects zero up to there)"


def _bacon_summary(bacon: Any) -> str:
    """Negative-weight share — the reason to run Goodman-Bacon at all."""
    payload = _light_serialize(bacon)
    if not isinstance(payload, dict):
        return "computed weight decomposition"
    bits = []
    share = payload.get("negative_weight_share")
    if isinstance(share, (int, float)):
        bits.append(f"negative weight share={float(share):.3g}")
    n_comp = payload.get("n_comparisons")
    if isinstance(n_comp, (int, float)):
        bits.append(f"{int(n_comp)} 2x2 comparisons")
    beta = payload.get("beta_twfe")
    if isinstance(beta, (int, float)):
        bits.append(f"TWFE beta={float(beta):.4g}")
    return "; ".join(bits) or "computed weight decomposition"


def _short_estimate(obj: Any) -> str:
    """Return a one-line ``estimate (SE) [CI]`` summary for ``obj``."""
    try:
        from .tools import _default_serializer

        d = _default_serializer(obj, detail="standard")
    except Exception:
        return ""
    if not isinstance(d, dict):
        return ""
    est = d.get("estimate")
    se = d.get("std_error") or d.get("se")
    lo = d.get("conf_low")
    hi = d.get("conf_high")
    bits = []
    if est is not None:
        bits.append(f"{est:.4g}")
        if se is not None:
            bits.append(f"(SE {se:.3g})")
        if lo is not None and hi is not None:
            bits.append(f"[CI {lo:.3g}, {hi:.3g}]")
    return " ".join(bits)


# ----------------------------------------------------------------------
# pipeline_did
# ----------------------------------------------------------------------


def _pipeline_did(
    arguments: Dict[str, Any],
    data: Optional[pd.DataFrame],
    *,
    detail: str,
    as_handle: bool,
) -> Dict[str, Any]:
    if data is None:
        return {"error": "pipeline_did requires data_path"}
    import statspai as sp

    y = arguments.get("y")
    treat = arguments.get("treat")
    time = arguments.get("time")
    cohort = arguments.get("cohort")
    unit_id = arguments.get("id")
    covariates = arguments.get("covariates") or []
    if not (y and treat and time):
        return {"error": "pipeline_did requires y / treat / time"}

    stages: List[Dict[str, Any]] = []

    # Stage 1: preflight
    preflight_fn = getattr(sp, "preflight", None)
    if preflight_fn is not None:
        verdict_args = {"y": y, "treatment": treat, "time": time}
        if unit_id:
            verdict_args["id"] = unit_id
        if covariates:
            verdict_args["covariates"] = covariates
        result, err = _safe_call(preflight_fn, data, "did", **verdict_args)
        if err:
            stages.append(_stage("preflight", "failed", err))
        else:
            verdict = getattr(result, "verdict", None) or (
                result.get("verdict") if isinstance(result, dict) else None
            )
            stages.append(
                _stage(
                    "preflight",
                    "ok" if verdict in {"PASS", "WARN", None} else "failed",
                    f"verdict={verdict}",
                )
            )
    else:
        stages.append(_stage("preflight", "skipped", "sp.preflight not available"))

    # Stage 2: estimator dispatch
    if cohort and unit_id:
        fit_fn = getattr(sp, "callaway_santanna", None)
        if fit_fn is not None:
            est_args = {"y": y, "g": cohort, "t": time, "i": unit_id}
            primary, err = _safe_call(fit_fn, data, **est_args)
            method = "callaway_santanna"
        else:
            primary, err = None, "sp.callaway_santanna not available"
            method = "callaway_santanna"
    else:
        fit_fn = getattr(sp, "did", None)
        if fit_fn is not None:
            est_args = {"y": y, "treat": treat, "time": time}
            primary, err = _safe_call(fit_fn, data, **est_args)
            method = "did"
        else:
            primary, err = None, "sp.did not available"
            method = "did"

    if err or primary is None:
        stages.append(_stage("estimate", "failed", err or "no result"))
        return {
            "pipeline": "pipeline_did",
            "stages": stages,
            "error": err or "estimator failed",
        }

    primary_summary = _short_estimate(primary)
    stages.append(
        _stage("estimate", "ok", f"{method}: {primary_summary}", method=method)
    )

    # Cache the primary result so follow-up tools chain via result_id
    primary_rid = RESULT_CACHE.put(
        primary,
        tool=method,
        arguments={
            k: v for k, v in arguments.items() if not isinstance(v, pd.DataFrame)
        },
    )

    # Stage 3: audit
    audit_fn = getattr(sp, "audit", None)
    audit_payload: Dict[str, Any] = {}
    if audit_fn is not None:
        report, err = _safe_call(audit_fn, primary)
        if err:
            stages.append(_stage("audit", "failed", err))
        else:
            audit_payload = _audit_to_dict(report)
            n_missing = _count_missing(audit_payload)
            stages.append(
                _stage(
                    "audit",
                    "ok",
                    f"{n_missing} missing high-importance checks",
                )
            )
    else:
        stages.append(_stage("audit", "skipped", "sp.audit not available"))

    # Stage 4: honest_did sensitivity (best-effort — needs event-study betas)
    honest_payload: Optional[Dict[str, Any]] = None
    honest_fn = getattr(sp, "honest_did", None)
    if honest_fn is not None:
        # NOTE: this used to call honest_did(betas=, sigma=, num_pre_periods=,
        # num_post_periods=, method="SD"). No such signature exists — the
        # current one is honest_did(result, e=, m_grid=, method=, alpha=,
        # backend=). Every invocation therefore raised TypeError and the stage
        # recorded a failure, so this branch of pipeline_did never once
        # produced honest CIs. It now passes the fitted result through, which
        # is what honest_did actually takes.
        from .workflow_tools import _coerce_event_study_result

        # The guard used to be `_extract_event_study(primary)` returning
        # non-None betas/sigma. That helper does not recognise the shape
        # sp.event_study actually produces, so it returns None even for a
        # genuine event study and the stage was skipped every time — the
        # second reason this branch never ran. honest_did only needs a result
        # it can read an event-study table off, so ask it directly and report
        # whatever comes back.
        coerced = _coerce_event_study_result(primary)
        if coerced is None:
            stages.append(
                _stage(
                    "honest_did", "skipped", "no event-study betas in primary result"
                )
            )
        else:
            honest, err = _safe_call(honest_fn, coerced, e=0, method="smoothness")
            if err:
                # "This estimator has no event-study table" (e.g. a plain 2x2
                # DiD, which has no pre-periods) is inapplicability, not
                # breakage — reporting it as a failure trains readers to
                # ignore the status field.
                inapplicable = "MethodIncompatibility" in str(err)
                stages.append(
                    _stage("honest_did", "skipped" if inapplicable else "failed", err)
                )
            else:
                honest_payload = _light_serialize(honest)
                stages.append(
                    _stage(
                        "honest_did",
                        "ok",
                        _short_estimate(honest) or _honest_did_summary(honest),
                    )
                )

    # Stage 5: Bacon decomposition (only meaningful for staggered TWFE)
    if cohort and unit_id:
        bacon_fn = getattr(sp, "bacon_decomposition", None)
        if bacon_fn is not None:
            bacon, err = _safe_call(
                bacon_fn,
                data,
                y=y,
                treat=treat,
                time=time,
                id=unit_id,
            )
            if err:
                stages.append(_stage("bacon_decomposition", "failed", err))
            else:
                stages.append(
                    _stage(
                        "bacon_decomposition",
                        "ok",
                        _bacon_summary(bacon),
                    )
                )
        else:
            stages.append(
                _stage(
                    "bacon_decomposition",
                    "skipped",
                    "sp.bacon_decomposition not available",
                )
            )

    # Stage 6: brief
    brief_fn = getattr(sp, "brief", None)
    brief_text = ""
    if brief_fn is not None:
        text, err = _safe_call(brief_fn, primary)
        if not err and text:
            brief_text = str(text)
            stages.append(_stage("brief", "ok", brief_text))
        else:
            stages.append(_stage("brief", "skipped", err or "no brief"))

    # Compose narrative
    narrative = _did_narrative(
        method=method,
        primary_summary=primary_summary,
        stages=stages,
        audit_payload=audit_payload,
        honest_payload=honest_payload,
        brief_text=brief_text,
    )

    out: Dict[str, Any] = {
        "pipeline": "pipeline_did",
        "method": method,
        "result_id": primary_rid,
        "result_uri": f"statspai://result/{primary_rid}",
        "primary_summary": primary_summary,
        "stages": stages,
        "audit": audit_payload,
        "narrative": narrative,
    }
    if honest_payload is not None:
        out["honest_did"] = honest_payload

    # Pre-built next_calls — chain into a paper-style report or further
    # sensitivity work.
    out["next_calls"] = [
        {
            "tool": "plot_from_result",
            "arguments": {"result_id": primary_rid, "kind": "event_study"},
            "rationale": "Event-study plot for the executive summary.",
        },
        {
            "tool": "sensitivity_from_result",
            "arguments": {"result_id": primary_rid, "method": "evalue"},
            "rationale": "E-value bound on omitted-confounder strength.",
        },
        {
            "tool": "spec_curve",
            "arguments": {
                "y": y,
                "treatment": treat,
                "covariates": covariates,
                "model_family": "did",
            },
            "rationale": "Specification curve over researcher degrees of freedom.",
        },
    ]

    # Citations from the enrichment layer
    from ._enrichment import build_citations, fetch_bibtex

    keys = list(
        dict.fromkeys(  # preserve order, dedupe
            build_citations(method)
            + build_citations("honest_did")
            + build_citations("bacon_decomposition")
        )
    )
    if keys:
        bib_present = {k: v for k, v in fetch_bibtex(keys).items() if v}
        out["citations"] = {"keys": keys}
        if bib_present:
            out["citations"]["bibtex"] = bib_present
    return out


def _audit_to_dict(report: Any) -> Dict[str, Any]:
    if isinstance(report, dict):
        return dict(report)
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, dict):
            return out
    if hasattr(report, "__dict__"):
        return {k: v for k, v in vars(report).items() if not k.startswith("_")}
    return {}


def _count_missing(audit_payload: Dict[str, Any]) -> int:
    items = audit_payload.get("items") or audit_payload.get("checks") or []
    if not isinstance(items, list):
        return 0
    return sum(
        1
        for it in items
        if isinstance(it, dict)
        and it.get("status") == "missing"
        and it.get("importance") in {"high", "critical"}
    )


def _light_serialize(obj: Any) -> Dict[str, Any]:
    try:
        from .tools import _default_serializer

        d = _default_serializer(obj, detail="standard")
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {"value": str(obj)[:200]}


def _did_narrative(
    *,
    method: str,
    primary_summary: str,
    stages: List[Dict[str, Any]],
    audit_payload: Dict[str, Any],
    honest_payload: Optional[Dict[str, Any]],
    brief_text: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# DID workflow ({method})")
    lines.append("")
    if brief_text:
        lines.append(brief_text)
        lines.append("")
    if primary_summary:
        lines.append(f"**Primary estimate**: {primary_summary}")
        lines.append("")
    lines.append("## Stages")
    for s in stages:
        bullet = (
            "✓" if s["status"] == "ok" else ("·" if s["status"] == "skipped" else "✗")
        )
        lines.append(f"- {bullet} **{s['name']}** — {s.get('summary', '')}")
    lines.append("")
    n_missing = _count_missing(audit_payload)
    if n_missing:
        lines.append("## Robustness gaps")
        lines.append(
            f"{n_missing} high-importance checks flagged as missing. "
            "See the `audit` field for details and the `next_calls` "
            "list for ready-to-dispatch follow-ups."
        )
        lines.append("")
    if honest_payload:
        lines.append("## Honest-DID sensitivity")
        est = _short_estimate_dict(honest_payload)
        if est:
            lines.append(f"Rambachan-Roth (2023) SD-bounded CI: {est}")
            lines.append("")
    return "\n".join(lines).strip()


def _short_estimate_dict(d: Dict[str, Any]) -> str:
    est = d.get("estimate")
    lo = d.get("conf_low")
    hi = d.get("conf_high")
    if est is None:
        return ""
    s = f"{est:.4g}"
    if lo is not None and hi is not None:
        s += f" [CI {lo:.3g}, {hi:.3g}]"
    return s


# ----------------------------------------------------------------------
# pipeline_iv
# ----------------------------------------------------------------------


def _pipeline_iv(
    arguments: Dict[str, Any],
    data: Optional[pd.DataFrame],
    *,
    detail: str,
    as_handle: bool,
) -> Dict[str, Any]:
    if data is None:
        return {"error": "pipeline_iv requires data_path"}
    formula = arguments.get("formula")
    if not formula:
        return {"error": "pipeline_iv requires `formula`"}

    import statspai as sp

    stages: List[Dict[str, Any]] = []

    fit_fn = getattr(sp, "ivreg", None) or getattr(sp, "iv", None)
    if fit_fn is None:
        return {"error": "sp.ivreg / sp.iv not available"}

    primary, err = _safe_call(fit_fn, formula, data=data)
    if err or primary is None:
        stages.append(_stage("estimate", "failed", err or "no result"))
        return {
            "pipeline": "pipeline_iv",
            "stages": stages,
            "error": err or "estimator failed",
        }
    # An ivreg fit exposes a params vector, not a scalar `.estimate`, so
    # the generic one-liner came back empty and the stage read
    # "ivreg: " — an ok status with no number in it. Report the
    # endogenous regressor's coefficient, which is the estimate the whole
    # pipeline exists to produce.
    summary = _short_estimate(primary)
    if not summary:
        summary = _iv_endog_summary(primary, formula)
    stages.append(_stage("estimate", "ok", f"ivreg: {summary}".rstrip()))

    rid = RESULT_CACHE.put(primary, tool="ivreg", arguments={"formula": formula})

    # Both diagnostics below take (data, column names), not a fitted
    # result. Passing `primary` positionally made every call raise
    # "missing N required positional arguments", so the pipeline reported
    # three failed stages on every IV run while still returning
    # pipeline="pipeline_iv" — which was all its test checked.
    from ..core.utils import parse_formula as _parse_formula

    try:
        spec = _parse_formula(formula)
        iv_y = spec["dependent"]
        iv_endog = list(spec["endogenous"])
        iv_instr = list(spec["instruments"])
        iv_exog = list(spec["exogenous"]) or None
    except (KeyError, TypeError, ValueError) as exc:
        spec = None
        stages.append(
            _stage("effective_f_test", "skipped", f"cannot parse formula: {exc}")
        )
        stages.append(
            _stage("anderson_rubin_test", "skipped", f"cannot parse formula: {exc}")
        )

    # Effective F
    f_fn = getattr(sp, "effective_f_test", None) if spec is not None else None
    fF: Optional[float] = None
    if f_fn is not None:
        ftest, err = _safe_call(
            f_fn, data, endog=iv_endog[0], instruments=iv_instr, exog=iv_exog
        )
        if err:
            stages.append(_stage("effective_f_test", "failed", err))
        else:
            # sp.effective_f_test returns a dict keyed F_eff (the
            # Olea-Pflueger effective F). The attribute lookups below it
            # never matched, so this stage reported "F=nan" as an ok
            # status on every run.
            if isinstance(ftest, dict):
                fF = float(ftest.get("F_eff", ftest.get("first_stage_F", float("nan"))))
                strength = str(ftest.get("strength") or "").split("(")[0].strip()
            else:
                fF = float(
                    getattr(
                        ftest,
                        "F_eff",
                        getattr(
                            ftest,
                            "F",
                            getattr(
                                ftest,
                                "statistic",
                                getattr(ftest, "value", float("nan")),
                            ),
                        ),
                    )
                )
                strength = ""
            label = f"F={fF:.2f}" + (f" ({strength})" if strength else "")
            stages.append(_stage("effective_f_test", "ok", label))
    else:
        stages.append(
            _stage(
                "effective_f_test",
                "skipped",
                "sp.effective_f_test unavailable",
            )
        )

    # Anderson-Rubin
    ar_fn = getattr(sp, "anderson_rubin_test", None) if spec is not None else None
    ar_payload: Optional[Dict[str, Any]] = None
    if ar_fn is not None:
        ar, err = _safe_call(
            ar_fn,
            data,
            y=iv_y,
            endog=iv_endog[0],
            instruments=iv_instr,
            exog=iv_exog,
        )
        if err:
            stages.append(_stage("anderson_rubin_test", "failed", err))
        else:
            ar_payload = _light_serialize(ar)
            stages.append(_stage("anderson_rubin_test", "ok", "computed"))
    else:
        stages.append(
            _stage(
                "anderson_rubin_test",
                "skipped",
                "sp.anderson_rubin_test unavailable",
            )
        )

    # E-value
    ev_fn = getattr(sp, "evalue_from_result", None) or getattr(sp, "evalue", None)
    ev_payload: Optional[Dict[str, Any]] = None
    if ev_fn is not None:
        ev, err = _safe_call(ev_fn, primary)
        if err:
            # An IV fit carries a coefficient vector, not the single
            # estimate evalue_from_result needs, so this is a design
            # boundary rather than a failure. Reporting it as "failed"
            # made every IV pipeline look broken.
            status = "skipped" if "expects a CausalResult" in str(err) else "failed"
            stages.append(_stage("evalue", status, err))
        else:
            ev_payload = _light_serialize(ev)
            stages.append(
                _stage(
                    "evalue",
                    "ok",
                    _short_estimate_dict(ev_payload) or "computed",
                )
            )

    narrative_lines = ["# IV workflow", "", f"**Primary estimate**: {summary}", ""]
    if fF is not None:
        narrative_lines.append(f"First-stage effective F = {fF:.2f}")
        if fF < 10:
            narrative_lines.append(
                "Below the Staiger-Stock 10 threshold — 2SLS is biased; "
                "lean on the Anderson-Rubin CI for inference."
            )
        narrative_lines.append("")
    narrative_lines.append("## Stages")
    for s in stages:
        bullet = (
            "✓" if s["status"] == "ok" else ("·" if s["status"] == "skipped" else "✗")
        )
        narrative_lines.append(f"- {bullet} **{s['name']}** — {s.get('summary', '')}")

    out: Dict[str, Any] = {
        "pipeline": "pipeline_iv",
        "method": "ivreg",
        "result_id": rid,
        "result_uri": f"statspai://result/{rid}",
        "primary_summary": summary,
        "effective_F": fF,
        "stages": stages,
        "narrative": "\n".join(narrative_lines).strip(),
        "next_calls": [
            {
                "tool": "sensitivity_from_result",
                "arguments": {"result_id": rid, "method": "evalue"},
            },
        ],
    }
    if ar_payload:
        out["anderson_rubin"] = ar_payload
    if ev_payload:
        out["evalue"] = ev_payload

    from ._enrichment import build_citations, fetch_bibtex

    keys = list(
        dict.fromkeys(
            build_citations("ivreg")
            + build_citations("effective_f_test")
            + build_citations("anderson_rubin_test")
            + build_citations("evalue")
        )
    )
    if keys:
        bib_present = {k: v for k, v in fetch_bibtex(keys).items() if v}
        out["citations"] = {"keys": keys}
        if bib_present:
            out["citations"]["bibtex"] = bib_present
    return out


# ----------------------------------------------------------------------
# pipeline_rd
# ----------------------------------------------------------------------


def _pipeline_rd(
    arguments: Dict[str, Any],
    data: Optional[pd.DataFrame],
    *,
    detail: str,
    as_handle: bool,
) -> Dict[str, Any]:
    if data is None:
        return {"error": "pipeline_rd requires data_path"}
    y = arguments.get("y")
    x = arguments.get("x")
    if not (y and x):
        return {"error": "pipeline_rd requires y + x (running variable)"}
    c = arguments.get("c", 0.0)
    fuzzy = arguments.get("fuzzy")

    import statspai as sp

    stages: List[Dict[str, Any]] = []

    fit_fn = getattr(sp, "rdrobust", None)
    if fit_fn is None:
        return {"error": "sp.rdrobust not available"}
    kwargs = {"y": y, "x": x, "c": c}
    if fuzzy:
        kwargs["fuzzy"] = fuzzy
    primary, err = _safe_call(fit_fn, data, **kwargs)
    if err or primary is None:
        stages.append(_stage("estimate", "failed", err or "no result"))
        return {
            "pipeline": "pipeline_rd",
            "stages": stages,
            "error": err or "estimator failed",
        }
    summary = _short_estimate(primary)
    stages.append(_stage("estimate", "ok", f"rdrobust: {summary}"))

    rid = RESULT_CACHE.put(primary, tool="rdrobust", arguments=arguments)

    # rddensity (McCrary)
    dens_fn = getattr(sp, "rddensity", None)
    if dens_fn is not None:
        dens, err = _safe_call(dens_fn, data, x=x, c=c)
        if err:
            stages.append(_stage("rddensity", "failed", err))
        else:
            p = getattr(dens, "p_value", getattr(dens, "pvalue", None))
            if p is None and isinstance(dens, dict):
                p = dens.get("p_value") or dens.get("pvalue")
            stages.append(
                _stage(
                    "rddensity",
                    "ok",
                    (
                        f"density-discontinuity p={p:.3g}"
                        if p is not None
                        else "computed"
                    ),
                )
            )
    else:
        stages.append(_stage("rddensity", "skipped", "sp.rddensity unavailable"))

    # rdsensitivity (bandwidth/kernel)
    sens_fn = getattr(sp, "rdbwsensitivity", None)
    if sens_fn is not None:
        sens, err = _safe_call(sens_fn, data, y=y, x=x, c=c)
        if err:
            stages.append(_stage("rdbwsensitivity", "failed", err))
        else:
            stages.append(_stage("rdbwsensitivity", "ok", "computed"))

    # rdplot — try to render PNG
    plot_png = None
    plot_fn = getattr(sp, "rdplot", None)
    if plot_fn is not None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            import io

            import matplotlib.pyplot as plt  # noqa: E401

            fig_or_obj, err = _safe_call(plot_fn, data, y=y, x=x, c=c)
            if not err:
                from .workflow_tools import _coerce_to_fig

                fig = _coerce_to_fig(fig_or_obj)
                if fig is not None:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    plot_png = buf.getvalue()
                    stages.append(
                        _stage("rdplot", "ok", f"PNG ({len(plot_png)} bytes)")
                    )
                else:
                    stages.append(
                        _stage(
                            "rdplot",
                            "skipped",
                            "plot helper returned no figure",
                        )
                    )
            else:
                stages.append(_stage("rdplot", "failed", err))
        except Exception as e:
            stages.append(_stage("rdplot", "skipped", f"matplotlib unavailable: {e}"))
    else:
        stages.append(_stage("rdplot", "skipped", "sp.rdplot unavailable"))

    narrative_lines = [
        "# RD workflow",
        "",
        f"**Primary estimate**: {summary}",
        "",
        "## Stages",
    ]
    for s in stages:
        bullet = (
            "✓" if s["status"] == "ok" else ("·" if s["status"] == "skipped" else "✗")
        )
        narrative_lines.append(f"- {bullet} **{s['name']}** — {s.get('summary', '')}")

    out: Dict[str, Any] = {
        "pipeline": "pipeline_rd",
        "method": "rdrobust",
        "result_id": rid,
        "result_uri": f"statspai://result/{rid}",
        "primary_summary": summary,
        "stages": stages,
        "narrative": "\n".join(narrative_lines).strip(),
        "next_calls": [
            {
                "tool": "sensitivity_from_result",
                "arguments": {"result_id": rid, "method": "evalue"},
            },
        ],
    }
    if plot_png is not None:
        out["_plot_png"] = plot_png

    from ._enrichment import build_citations, fetch_bibtex

    keys = list(
        dict.fromkeys(build_citations("rdrobust") + build_citations("rddensity"))
    )
    if keys:
        bib_present = {k: v for k, v in fetch_bibtex(keys).items() if v}
        out["citations"] = {"keys": keys}
        if bib_present:
            out["citations"]["bibtex"] = bib_present
    return out


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


def execute_pipeline_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    data: Optional[pd.DataFrame] = None,
    detail: str = "agent",
    as_handle: bool = False,
) -> Dict[str, Any]:
    if name == "pipeline_did":
        return _pipeline_did(
            arguments,
            data,
            detail=detail,
            as_handle=as_handle,
        )
    if name == "pipeline_iv":
        return _pipeline_iv(
            arguments,
            data,
            detail=detail,
            as_handle=as_handle,
        )
    if name == "pipeline_rd":
        return _pipeline_rd(
            arguments,
            data,
            detail=detail,
            as_handle=as_handle,
        )
    return {
        "error": f"unknown pipeline tool: {name!r}",
        "available_pipelines": sorted(PIPELINE_TOOL_NAMES),
    }


__all__ = [
    "PIPELINE_TOOL_SPECS",
    "PIPELINE_TOOL_NAMES",
    "pipeline_tool_manifest",
    "execute_pipeline_tool",
]
