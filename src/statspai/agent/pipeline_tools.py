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

from typing import Any, Dict, List, Optional

import pandas as pd

from ._result_cache import RESULT_CACHE


# ----------------------------------------------------------------------
# Schema definitions
# ----------------------------------------------------------------------

PIPELINE_TOOL_SPECS: List[Dict[str, Any]] = [
    {
        'name': 'pipeline_did',
        'description': (
            "End-to-end DID workflow: preflight → did/CS estimator → "
            "audit → honest-DID sensitivity → bacon decomposition → "
            "brief. Returns one markdown report + the primary "
            "result_id. Use this when the user pastes a DID dataset "
            "and asks 'is the effect real?' — the pipeline runs every "
            "diagnostic the literature expects."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string', 'description': 'Outcome column.'},
                'treat': {'type': 'string',
                          'description': 'Binary treatment indicator.'},
                'time': {'type': 'string', 'description': 'Time column.'},
                'id': {'type': 'string',
                       'description': 'Unit id (panel) — required for staggered-DID.'},
                'cohort': {'type': 'string',
                           'description': ('First-treatment cohort column. '
                                            'When supplied, dispatches '
                                            'callaway_santanna instead of '
                                            'classic 2x2 did.')},
                'covariates': {'type': 'array',
                                'items': {'type': 'string'}},
            },
            'required': ['y', 'treat', 'time'],
        },
    },
    {
        'name': 'pipeline_iv',
        'description': (
            "End-to-end IV workflow: ivreg → first-stage F (effective + "
            "Olea-Pflueger) → Anderson-Rubin CI → e-value. Returns one "
            "markdown report + result_id."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'formula': {
                    'type': 'string',
                    'description': "'y ~ x_exog + (d_endog ~ z_instrument)' style.",
                },
            },
            'required': ['formula'],
        },
    },
    {
        'name': 'pipeline_rd',
        'description': (
            "End-to-end RD workflow: rdrobust → rdplot (PNG image) → "
            "rddensity (McCrary) → rdsensitivity (bandwidth). Returns "
            "one markdown report + result_id + an image content block."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string'},
                'x': {'type': 'string',
                      'description': 'Running variable column.'},
                'c': {'type': 'number', 'default': 0.0,
                      'description': 'Cutoff value.'},
                'fuzzy': {'type': 'string',
                          'description': 'Treatment column for fuzzy RD (optional).'},
            },
            'required': ['y', 'x'],
        },
    },
]


PIPELINE_TOOL_NAMES = frozenset(t['name'] for t in PIPELINE_TOOL_SPECS)


def pipeline_tool_manifest() -> List[Dict[str, Any]]:
    return [dict(t) for t in PIPELINE_TOOL_SPECS]


# ----------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------


def _stage(name: str, status: str = "ok", summary: str = "",
           **extra) -> Dict[str, Any]:
    out = {"name": name, "status": status, "summary": summary}
    out.update(extra)
    return out


def _safe_call(fn, *args, **kwargs):
    """Invoke ``fn`` and return ``(result, error_msg_or_none)``."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


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

def _pipeline_did(arguments: Dict[str, Any],
                   data: Optional[pd.DataFrame],
                   *, detail: str,
                   as_handle: bool) -> Dict[str, Any]:
    if data is None:
        return {'error': 'pipeline_did requires data_path'}
    import statspai as sp

    y = arguments.get('y')
    treat = arguments.get('treat')
    time = arguments.get('time')
    cohort = arguments.get('cohort')
    unit_id = arguments.get('id')
    covariates = arguments.get('covariates') or []
    if not (y and treat and time):
        return {'error': "pipeline_did requires y / treat / time"}

    stages: List[Dict[str, Any]] = []

    # Stage 1: preflight
    preflight_fn = getattr(sp, 'preflight', None)
    if preflight_fn is not None:
        verdict_args = {'y': y, 'treatment': treat, 'time': time}
        if unit_id:
            verdict_args['id'] = unit_id
        if covariates:
            verdict_args['covariates'] = covariates
        result, err = _safe_call(preflight_fn, data, 'did', **verdict_args)
        if err:
            stages.append(_stage('preflight', 'failed', err))
        else:
            verdict = getattr(result, 'verdict', None) or (
                result.get('verdict') if isinstance(result, dict) else None)
            stages.append(_stage('preflight',
                                   'ok' if verdict in {'PASS', 'WARN', None}
                                   else 'failed',
                                   f"verdict={verdict}"))
    else:
        stages.append(_stage('preflight', 'skipped',
                              'sp.preflight not available'))

    # Stage 2: estimator dispatch
    if cohort and unit_id:
        fit_fn = getattr(sp, 'callaway_santanna', None)
        if fit_fn is not None:
            est_args = {'y': y, 'g': cohort, 't': time, 'i': unit_id}
            primary, err = _safe_call(fit_fn, data, **est_args)
            method = 'callaway_santanna'
        else:
            primary, err = None, 'sp.callaway_santanna not available'
            method = 'callaway_santanna'
    else:
        fit_fn = getattr(sp, 'did', None)
        if fit_fn is not None:
            est_args = {'y': y, 'treat': treat, 'time': time}
            primary, err = _safe_call(fit_fn, data, **est_args)
            method = 'did'
        else:
            primary, err = None, 'sp.did not available'
            method = 'did'

    if err or primary is None:
        stages.append(_stage('estimate', 'failed', err or 'no result'))
        return {
            'pipeline': 'pipeline_did',
            'stages': stages,
            'error': err or 'estimator failed',
        }

    primary_summary = _short_estimate(primary)
    stages.append(_stage('estimate', 'ok',
                          f"{method}: {primary_summary}",
                          method=method))

    # Cache the primary result so follow-up tools chain via result_id
    primary_rid = RESULT_CACHE.put(primary, tool=method,
                                     arguments={k: v for k, v in arguments.items()
                                                 if not isinstance(v, pd.DataFrame)})

    # Stage 3: audit
    audit_fn = getattr(sp, 'audit', None)
    audit_payload: Dict[str, Any] = {}
    if audit_fn is not None:
        report, err = _safe_call(audit_fn, primary)
        if err:
            stages.append(_stage('audit', 'failed', err))
        else:
            audit_payload = _audit_to_dict(report)
            n_missing = _count_missing(audit_payload)
            stages.append(_stage('audit', 'ok',
                                   f"{n_missing} missing high-importance checks"))
    else:
        stages.append(_stage('audit', 'skipped', 'sp.audit not available'))

    # Stage 4: honest_did sensitivity (best-effort — needs event-study betas)
    honest_payload: Optional[Dict[str, Any]] = None
    honest_fn = getattr(sp, 'honest_did', None)
    if honest_fn is not None:
        from .workflow_tools import _extract_event_study, _listify_sigma
        betas, sigma, n_pre, n_post = _extract_event_study(primary)
        if betas is not None and sigma is not None:
            honest, err = _safe_call(
                honest_fn,
                betas=list(betas), sigma=_listify_sigma(sigma),
                num_pre_periods=int(n_pre), num_post_periods=int(n_post),
                method='SD',
            )
            if err:
                stages.append(_stage('honest_did', 'failed', err))
            else:
                honest_payload = _light_serialize(honest)
                stages.append(_stage('honest_did', 'ok',
                                       _short_estimate(honest) or 'computed'))
        else:
            stages.append(_stage('honest_did', 'skipped',
                                   'no event-study betas in primary result'))

    # Stage 5: Bacon decomposition (only meaningful for staggered TWFE)
    if cohort and unit_id:
        bacon_fn = getattr(sp, 'bacon_decomposition', None)
        if bacon_fn is not None:
            bacon, err = _safe_call(bacon_fn, data,
                                      y=y, treat=treat,
                                      time=time, id=unit_id)
            if err:
                stages.append(_stage('bacon_decomposition', 'failed', err))
            else:
                stages.append(_stage('bacon_decomposition', 'ok',
                                       'computed weight decomposition'))
        else:
            stages.append(_stage('bacon_decomposition', 'skipped',
                                   'sp.bacon_decomposition not available'))

    # Stage 6: brief
    brief_fn = getattr(sp, 'brief', None)
    brief_text = ''
    if brief_fn is not None:
        text, err = _safe_call(brief_fn, primary)
        if not err and text:
            brief_text = str(text)
            stages.append(_stage('brief', 'ok', brief_text))
        else:
            stages.append(_stage('brief', 'skipped', err or 'no brief'))

    # Compose narrative
    narrative = _did_narrative(method=method,
                                primary_summary=primary_summary,
                                stages=stages,
                                audit_payload=audit_payload,
                                honest_payload=honest_payload,
                                brief_text=brief_text)

    out: Dict[str, Any] = {
        'pipeline': 'pipeline_did',
        'method': method,
        'result_id': primary_rid,
        'result_uri': f"statspai://result/{primary_rid}",
        'primary_summary': primary_summary,
        'stages': stages,
        'audit': audit_payload,
        'narrative': narrative,
    }
    if honest_payload is not None:
        out['honest_did'] = honest_payload

    # Pre-built next_calls — chain into a paper-style report or further
    # sensitivity work.
    out['next_calls'] = [
        {'tool': 'plot_from_result',
         'arguments': {'result_id': primary_rid, 'kind': 'event_study'},
         'rationale': 'Event-study plot for the executive summary.'},
        {'tool': 'sensitivity_from_result',
         'arguments': {'result_id': primary_rid, 'method': 'evalue'},
         'rationale': 'E-value bound on omitted-confounder strength.'},
        {'tool': 'spec_curve',
         'arguments': {'y': y, 'treatment': treat,
                        'covariates': covariates,
                        'model_family': 'did'},
         'rationale': 'Specification curve over researcher degrees of freedom.'},
    ]

    # Citations from the enrichment layer
    from ._enrichment import build_citations, fetch_bibtex
    keys = list(dict.fromkeys(  # preserve order, dedupe
        build_citations(method)
        + build_citations('honest_did')
        + build_citations('bacon_decomposition')
    ))
    if keys:
        bib_present = {k: v for k, v in fetch_bibtex(keys).items() if v}
        out['citations'] = {'keys': keys}
        if bib_present:
            out['citations']['bibtex'] = bib_present
    return out


def _audit_to_dict(report) -> Dict[str, Any]:
    if isinstance(report, dict):
        return dict(report)
    to_dict = getattr(report, 'to_dict', None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, dict):
            return out
    if hasattr(report, '__dict__'):
        return {k: v for k, v in vars(report).items()
                if not k.startswith('_')}
    return {}


def _count_missing(audit_payload: Dict[str, Any]) -> int:
    items = audit_payload.get('items') or audit_payload.get('checks') or []
    if not isinstance(items, list):
        return 0
    return sum(1 for it in items
               if isinstance(it, dict)
               and it.get('status') == 'missing'
               and it.get('importance') in {'high', 'critical'})


def _light_serialize(obj) -> Dict[str, Any]:
    try:
        from .tools import _default_serializer
        d = _default_serializer(obj, detail="standard")
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {'value': str(obj)[:200]}


def _did_narrative(*, method: str,
                    primary_summary: str,
                    stages: List[Dict[str, Any]],
                    audit_payload: Dict[str, Any],
                    honest_payload: Optional[Dict[str, Any]],
                    brief_text: str) -> str:
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
        bullet = "✓" if s['status'] == 'ok' else (
            "·" if s['status'] == 'skipped' else "✗")
        lines.append(f"- {bullet} **{s['name']}** — {s.get('summary', '')}")
    lines.append("")
    n_missing = _count_missing(audit_payload)
    if n_missing:
        lines.append(f"## Robustness gaps")
        lines.append(f"{n_missing} high-importance checks flagged as missing. "
                     f"See the `audit` field for details and the `next_calls` "
                     f"list for ready-to-dispatch follow-ups.")
        lines.append("")
    if honest_payload:
        lines.append(f"## Honest-DID sensitivity")
        est = _short_estimate_dict(honest_payload)
        if est:
            lines.append(f"Rambachan-Roth (2023) SD-bounded CI: {est}")
            lines.append("")
    return "\n".join(lines).strip()


def _short_estimate_dict(d: Dict[str, Any]) -> str:
    est = d.get('estimate')
    lo = d.get('conf_low')
    hi = d.get('conf_high')
    if est is None:
        return ''
    s = f"{est:.4g}"
    if lo is not None and hi is not None:
        s += f" [CI {lo:.3g}, {hi:.3g}]"
    return s


# ----------------------------------------------------------------------
# pipeline_iv
# ----------------------------------------------------------------------

def _pipeline_iv(arguments: Dict[str, Any],
                  data: Optional[pd.DataFrame],
                  *, detail: str,
                  as_handle: bool) -> Dict[str, Any]:
    if data is None:
        return {'error': 'pipeline_iv requires data_path'}
    formula = arguments.get('formula')
    if not formula:
        return {'error': "pipeline_iv requires `formula`"}

    import statspai as sp

    stages: List[Dict[str, Any]] = []

    fit_fn = getattr(sp, 'ivreg', None) or getattr(sp, 'iv', None)
    if fit_fn is None:
        return {'error': 'sp.ivreg / sp.iv not available'}

    primary, err = _safe_call(fit_fn, formula, data=data)
    if err or primary is None:
        stages.append(_stage('estimate', 'failed', err or 'no result'))
        return {'pipeline': 'pipeline_iv', 'stages': stages,
                'error': err or 'estimator failed'}
    summary = _short_estimate(primary)
    stages.append(_stage('estimate', 'ok', f"ivreg: {summary}"))

    rid = RESULT_CACHE.put(primary, tool='ivreg', arguments={'formula': formula})

    # Effective F
    f_fn = getattr(sp, 'effective_f_test', None)
    fF: Optional[float] = None
    if f_fn is not None:
        ftest, err = _safe_call(f_fn, primary)
        if err:
            stages.append(_stage('effective_f_test', 'failed', err))
        else:
            fF = float(getattr(ftest, 'F', getattr(ftest, 'statistic',
                                                      getattr(ftest, 'value', float('nan')))))
            stages.append(_stage('effective_f_test', 'ok', f"F={fF:.2f}"))
    else:
        stages.append(_stage('effective_f_test', 'skipped',
                              'sp.effective_f_test unavailable'))

    # Anderson-Rubin
    ar_fn = getattr(sp, 'anderson_rubin_test', None)
    ar_payload: Optional[Dict[str, Any]] = None
    if ar_fn is not None:
        ar, err = _safe_call(ar_fn, primary)
        if err:
            stages.append(_stage('anderson_rubin_test', 'failed', err))
        else:
            ar_payload = _light_serialize(ar)
            stages.append(_stage('anderson_rubin_test', 'ok', 'computed'))
    else:
        stages.append(_stage('anderson_rubin_test', 'skipped',
                              'sp.anderson_rubin_test unavailable'))

    # E-value
    ev_fn = getattr(sp, 'evalue_from_result', None) or getattr(sp, 'evalue', None)
    ev_payload: Optional[Dict[str, Any]] = None
    if ev_fn is not None:
        ev, err = _safe_call(ev_fn, primary)
        if err:
            stages.append(_stage('evalue', 'failed', err))
        else:
            ev_payload = _light_serialize(ev)
            stages.append(_stage('evalue', 'ok',
                                   _short_estimate_dict(ev_payload) or 'computed'))

    narrative_lines = [f"# IV workflow", "",
                        f"**Primary estimate**: {summary}", ""]
    if fF is not None:
        narrative_lines.append(f"First-stage effective F = {fF:.2f}")
        if fF < 10:
            narrative_lines.append(
                "Below the Staiger-Stock 10 threshold — 2SLS is biased; "
                "lean on the Anderson-Rubin CI for inference.")
        narrative_lines.append("")
    narrative_lines.append("## Stages")
    for s in stages:
        bullet = "✓" if s['status'] == 'ok' else (
            "·" if s['status'] == 'skipped' else "✗")
        narrative_lines.append(f"- {bullet} **{s['name']}** — {s.get('summary', '')}")

    out: Dict[str, Any] = {
        'pipeline': 'pipeline_iv',
        'method': 'ivreg',
        'result_id': rid,
        'result_uri': f"statspai://result/{rid}",
        'primary_summary': summary,
        'effective_F': fF,
        'stages': stages,
        'narrative': "\n".join(narrative_lines).strip(),
        'next_calls': [
            {'tool': 'sensitivity_from_result',
             'arguments': {'result_id': rid, 'method': 'evalue'}},
        ],
    }
    if ar_payload:
        out['anderson_rubin'] = ar_payload
    if ev_payload:
        out['evalue'] = ev_payload

    from ._enrichment import build_citations, fetch_bibtex
    keys = list(dict.fromkeys(
        build_citations('ivreg')
        + build_citations('effective_f_test')
        + build_citations('anderson_rubin_test')
        + build_citations('evalue')
    ))
    if keys:
        bib_present = {k: v for k, v in fetch_bibtex(keys).items() if v}
        out['citations'] = {'keys': keys}
        if bib_present:
            out['citations']['bibtex'] = bib_present
    return out


# ----------------------------------------------------------------------
# pipeline_rd
# ----------------------------------------------------------------------

def _pipeline_rd(arguments: Dict[str, Any],
                  data: Optional[pd.DataFrame],
                  *, detail: str,
                  as_handle: bool) -> Dict[str, Any]:
    if data is None:
        return {'error': 'pipeline_rd requires data_path'}
    y = arguments.get('y')
    x = arguments.get('x')
    if not (y and x):
        return {'error': "pipeline_rd requires y + x (running variable)"}
    c = arguments.get('c', 0.0)
    fuzzy = arguments.get('fuzzy')

    import statspai as sp

    stages: List[Dict[str, Any]] = []

    fit_fn = getattr(sp, 'rdrobust', None)
    if fit_fn is None:
        return {'error': 'sp.rdrobust not available'}
    kwargs = {'y': y, 'x': x, 'c': c}
    if fuzzy:
        kwargs['fuzzy'] = fuzzy
    primary, err = _safe_call(fit_fn, data, **kwargs)
    if err or primary is None:
        stages.append(_stage('estimate', 'failed', err or 'no result'))
        return {'pipeline': 'pipeline_rd', 'stages': stages,
                'error': err or 'estimator failed'}
    summary = _short_estimate(primary)
    stages.append(_stage('estimate', 'ok', f"rdrobust: {summary}"))

    rid = RESULT_CACHE.put(primary, tool='rdrobust', arguments=arguments)

    # rddensity (McCrary)
    dens_fn = getattr(sp, 'rddensity', None)
    if dens_fn is not None:
        dens, err = _safe_call(dens_fn, data, x=x, c=c)
        if err:
            stages.append(_stage('rddensity', 'failed', err))
        else:
            p = getattr(dens, 'p_value', getattr(dens, 'pvalue', None))
            if p is None and isinstance(dens, dict):
                p = dens.get('p_value') or dens.get('pvalue')
            stages.append(_stage('rddensity', 'ok',
                                   f"density-discontinuity p={p:.3g}"
                                   if p is not None else "computed"))
    else:
        stages.append(_stage('rddensity', 'skipped',
                              'sp.rddensity unavailable'))

    # rdsensitivity (bandwidth/kernel)
    sens_fn = getattr(sp, 'rdbwsensitivity', None)
    if sens_fn is not None:
        sens, err = _safe_call(sens_fn, data, y=y, x=x, c=c)
        if err:
            stages.append(_stage('rdbwsensitivity', 'failed', err))
        else:
            stages.append(_stage('rdbwsensitivity', 'ok', 'computed'))

    # rdplot — try to render PNG
    plot_png = None
    plot_fn = getattr(sp, 'rdplot', None)
    if plot_fn is not None:
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt, io  # noqa: E401
            fig_or_obj, err = _safe_call(plot_fn, data, y=y, x=x, c=c)
            if not err:
                from .workflow_tools import _coerce_to_fig
                fig = _coerce_to_fig(fig_or_obj)
                if fig is not None:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120,
                                bbox_inches="tight")
                    plt.close(fig)
                    plot_png = buf.getvalue()
                    stages.append(_stage('rdplot', 'ok',
                                           f"PNG ({len(plot_png)} bytes)"))
                else:
                    stages.append(_stage('rdplot', 'skipped',
                                           'plot helper returned no figure'))
            else:
                stages.append(_stage('rdplot', 'failed', err))
        except Exception as e:
            stages.append(_stage('rdplot', 'skipped',
                                   f'matplotlib unavailable: {e}'))
    else:
        stages.append(_stage('rdplot', 'skipped', 'sp.rdplot unavailable'))

    narrative_lines = [
        f"# RD workflow", "",
        f"**Primary estimate**: {summary}", "",
        "## Stages",
    ]
    for s in stages:
        bullet = "✓" if s['status'] == 'ok' else (
            "·" if s['status'] == 'skipped' else "✗")
        narrative_lines.append(f"- {bullet} **{s['name']}** — {s.get('summary', '')}")

    out: Dict[str, Any] = {
        'pipeline': 'pipeline_rd',
        'method': 'rdrobust',
        'result_id': rid,
        'result_uri': f"statspai://result/{rid}",
        'primary_summary': summary,
        'stages': stages,
        'narrative': "\n".join(narrative_lines).strip(),
        'next_calls': [
            {'tool': 'sensitivity_from_result',
             'arguments': {'result_id': rid, 'method': 'evalue'}},
        ],
    }
    if plot_png is not None:
        out['_plot_png'] = plot_png

    from ._enrichment import build_citations, fetch_bibtex
    keys = list(dict.fromkeys(
        build_citations('rdrobust')
        + build_citations('rddensity')
    ))
    if keys:
        bib_present = {k: v for k, v in fetch_bibtex(keys).items() if v}
        out['citations'] = {'keys': keys}
        if bib_present:
            out['citations']['bibtex'] = bib_present
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
    if name == 'pipeline_did':
        return _pipeline_did(arguments, data, detail=detail, as_handle=as_handle)
    if name == 'pipeline_iv':
        return _pipeline_iv(arguments, data, detail=detail, as_handle=as_handle)
    if name == 'pipeline_rd':
        return _pipeline_rd(arguments, data, detail=detail, as_handle=as_handle)
    return {
        'error': f"unknown pipeline tool: {name!r}",
        'available_pipelines': sorted(PIPELINE_TOOL_NAMES),
    }


__all__ = [
    "PIPELINE_TOOL_SPECS",
    "PIPELINE_TOOL_NAMES",
    "pipeline_tool_manifest",
    "execute_pipeline_tool",
]
