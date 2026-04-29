"""Hand-curated workflow / handle-based / citation tools.

These are the "Tier-0" tools that close the agent feedback loop:

* ``audit_result`` / ``brief_result`` / ``sensitivity_from_result`` /
  ``honest_did_from_result`` — operate on a cached result handle
  produced by an earlier tool call (``as_handle=True``). They eliminate
  the LLM having to ferry back arrays and CSV paths between turns.
* ``bibtex`` — return verified BibTeX entries from the project's
  ``paper.bib`` (the single source of truth per CLAUDE.md §10). Closes
  the citation-hallucination loophole.
* ``audit`` / ``preflight`` / ``detect_design`` / ``brief`` — explicit
  hand-curated wrappers for the smart-workflow primitives that the
  prompt templates reference. Auto-tools used to surface these with
  one-line descriptions; the bespoke schemas below give agents proper
  signposting.

Every workflow tool returns a dict shaped like the standard estimator
serializer output (``estimate`` / ``method`` / ``next_calls`` / …) so
the MCP layer doesn't need to special-case their content blocks.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ._result_cache import RESULT_CACHE


# ----------------------------------------------------------------------
# Schema definitions surfaced via tool_manifest()
# ----------------------------------------------------------------------

def _result_id_schema(description: str) -> Dict[str, Any]:
    return {
        'type': 'object',
        'properties': {
            'result_id': {
                'type': 'string',
                'description': description,
            },
        },
        'required': ['result_id'],
    }


WORKFLOW_TOOL_SPECS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # Handle-based extensions to the curated tools — break the
    # "LLM ferries arrays" anti-pattern.
    # ------------------------------------------------------------------
    {
        'name': 'audit_result',
        'description': (
            "Reviewer-grade audit on a previously-fitted result. Pass "
            "the result_id returned by an earlier tool call (with "
            "as_handle=true). Returns the same checklist sp.audit() "
            "produces — every robustness check the literature expects "
            "for the design, with status='present|missing|run' and "
            "concrete suggested_function names for the missing ones."
        ),
        'input_schema': _result_id_schema(
            "Handle returned by an earlier estimator call. Must be in "
            "the server result cache (LRU-evicted; refit if missing)."
        ),
    },
    {
        'name': 'brief_result',
        'description': (
            "Return the one-line agent-friendly brief for a fitted "
            "result. Uses sp.brief(). Useful when an agent wants to "
            "summarise a chained workflow without paying for the full "
            "JSON payload again."
        ),
        'input_schema': _result_id_schema(
            "Handle to a previously-fitted result."
        ),
    },
    {
        'name': 'sensitivity_from_result',
        'description': (
            "Run sp.sensitivity / sp.evalue / sp.oster_bounds / "
            "sp.sensemakr on a cached result. Pass method='evalue' "
            "(default) for the omitted-confounder-strength bound, "
            "'oster' for delta/R-max, 'cinelli_hazlett' for OVB bounds."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'result_id': {
                    'type': 'string',
                    'description': "Handle to a fitted causal result.",
                },
                'method': {
                    'type': 'string',
                    'enum': ['evalue', 'oster', 'cinelli_hazlett', 'auto'],
                    'default': 'evalue',
                },
                'benchmark_covariate': {
                    'type': 'string',
                    'description': "Cinelli-Hazlett benchmark column (optional).",
                },
            },
            'required': ['result_id'],
        },
    },
    {
        'name': 'honest_did_from_result',
        'description': (
            "Rambachan-Roth (2023) honest CIs on a fitted DID / "
            "event-study result. Auto-extracts betas + sigma + "
            "pre/post-period counts from the result; the LLM never "
            "ferries arrays."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'result_id': {
                    'type': 'string',
                    'description': "Handle to a DID / event-study result.",
                },
                'method': {
                    'type': 'string',
                    'enum': ['SD', 'RM'],
                    'default': 'SD',
                    'description': ('SD = smoothness deviation '
                                     '(Rambachan-Roth default); '
                                     'RM = relative magnitude.'),
                },
                'm_bar': {
                    'type': 'number',
                    'description': "Bound on deviation magnitude (optional).",
                },
            },
            'required': ['result_id'],
        },
    },
    # ------------------------------------------------------------------
    # Workflow primitives — explicit registrations so prompt templates
    # have first-class entries instead of auto-generated stubs.
    # ------------------------------------------------------------------
    {
        'name': 'audit',
        'description': (
            "Reviewer-grade audit on a result. Returns the literature "
            "checklist (parallel-trends test, honest-DID, Bacon "
            "decomposition, placebo, balance, …) with status per item "
            "and the concrete suggest_function to call to fill any "
            "missing high-importance check."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'result_id': {
                    'type': 'string',
                    'description': ("Result handle. Required unless "
                                     "you also pass a fitted result via "
                                     "the result kwarg (programmatic "
                                     "use)."),
                },
            },
            'required': [],
        },
    },
    {
        'name': 'preflight',
        'description': (
            "Run pre-fit identification checks for a chosen method on a "
            "DataFrame. Verdict in {PASS, WARN, FAIL}. ALWAYS call this "
            "before fitting on an unfamiliar dataset to surface design "
            "problems (overlap, cohort sizes, IV first-stage F, "
            "running-variable density at the cutoff)."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'method': {
                    'type': 'string',
                    'description': ("Estimator name: 'did', 'rd', 'iv', "
                                     "'synth', 'matching', 'dml', …"),
                },
                'y': {'type': 'string', 'description': "Outcome column."},
                'treatment': {'type': 'string'},
                'time': {'type': 'string'},
                'id': {'type': 'string', 'description': "Unit id column."},
                'cohort': {'type': 'string'},
                'running_var': {'type': 'string'},
                'instrument': {'type': 'string'},
                'covariates': {'type': 'array',
                                'items': {'type': 'string'}},
            },
            'required': ['method'],
        },
    },
    {
        'name': 'detect_design',
        'description': (
            "Auto-detect the study design (panel / cross-section / RD "
            "/ IV-style) from column shapes and types. Returns the "
            "guessed design plus the columns that drove the inference. "
            "Call this BEFORE recommend() when the user pastes a CSV "
            "with no context."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'time_col_hint': {'type': 'string'},
                'id_col_hint': {'type': 'string'},
            },
            'required': [],
        },
    },
    {
        'name': 'brief',
        'description': (
            "One-line agent-friendly brief for a fitted result. "
            "Cheaper than calling brief_result if you already have the "
            "result object in scope."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'result_id': {'type': 'string'},
            },
            'required': [],
        },
    },
    # ------------------------------------------------------------------
    # Citation tool — the kill-switch for citation hallucination.
    # ------------------------------------------------------------------
    {
        'name': 'bibtex',
        'description': (
            "Return verified BibTeX entries from paper.bib (StatsPAI's "
            "single source of truth for citations). Pass one or more "
            "bib keys (e.g. 'callaway2021difference'). NEVER invent "
            "citations — call this tool instead. Unknown keys return "
            "an empty entry plus a list of close matches."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'keys': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': ("Bib keys to look up. Most "
                                     "estimators advertise their key "
                                     "in agent_card.reference."),
                },
            },
            'required': ['keys'],
        },
    },
]


WORKFLOW_TOOL_NAMES = frozenset(t['name'] for t in WORKFLOW_TOOL_SPECS)


def workflow_tool_manifest() -> List[Dict[str, Any]]:
    """Return manifest entries for every workflow tool."""
    return [dict(t) for t in WORKFLOW_TOOL_SPECS]


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

def execute_workflow_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    data: Optional[pd.DataFrame] = None,
    detail: str = "agent",
    result_id: Optional[str] = None,
    as_handle: bool = False,
) -> Dict[str, Any]:
    """Dispatch a workflow tool call.

    Parameters
    ----------
    name : str
        One of :data:`WORKFLOW_TOOL_NAMES`.
    arguments : dict
        Tool-call arguments (already stripped of MCP-only kwargs).
    data : DataFrame, optional
        Loaded by the MCP layer for tools that need fresh data
        (``preflight``, ``detect_design``).
    detail : str
        Forwarded to result serializers.
    result_id : str, optional
        Used as the default for tools that take ``result_id`` if the
        caller didn't include it in ``arguments``.
    as_handle : bool
        Cache the new fitted result and return ``result_id`` /
        ``result_uri``.
    """
    rid_arg = arguments.get('result_id') or result_id

    if name == 'bibtex':
        return _tool_bibtex(arguments)

    if name == 'detect_design':
        return _tool_detect_design(arguments, data, detail=detail,
                                     as_handle=as_handle)

    if name == 'preflight':
        return _tool_preflight(arguments, data, detail=detail,
                                 as_handle=as_handle)

    if name in {'audit_result', 'audit'}:
        return _tool_audit(rid_arg, detail=detail)

    if name in {'brief_result', 'brief'}:
        return _tool_brief(rid_arg)

    if name == 'sensitivity_from_result':
        return _tool_sensitivity_from_result(
            rid_arg, arguments, detail=detail, as_handle=as_handle)

    if name == 'honest_did_from_result':
        return _tool_honest_did_from_result(
            rid_arg, arguments, detail=detail, as_handle=as_handle)

    return {
        'error': f"workflow_tool dispatch missed name {name!r}",
        'available_workflow_tools': sorted(WORKFLOW_TOOL_NAMES),
    }


# ----------------------------------------------------------------------
# Individual tool implementations
# ----------------------------------------------------------------------

def _need_result(rid: Optional[str]) -> Any:
    """Resolve a result_id to its cached object or raise a friendly dict."""
    if not rid:
        return {
            'error': "result_id is required",
            'hint': ("Re-run the upstream estimator with as_handle=true "
                     "to get a result_id, then pass it here."),
        }
    obj = RESULT_CACHE.get(rid)
    if obj is None:
        return {
            'error': f"result_id {rid!r} not found in cache",
            'hint': ("LRU cache evicts oldest entries; re-fit the "
                     "estimator with as_handle=true to obtain a fresh "
                     "handle."),
            'available_result_ids': RESULT_CACHE.keys(),
        }
    return obj


def _tool_audit(rid: Optional[str], *, detail: str) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and 'error' in obj:
        return obj
    import statspai as sp
    audit_fn = getattr(sp, 'audit', None)
    if audit_fn is None:
        return {'error': "sp.audit is not available in this build"}
    try:
        report = audit_fn(obj)
    except Exception as e:
        from .remediation import remediate
        return {
            'error': f"{type(e).__name__}: {e}",
            'remediation': remediate(e, context={'tool': 'audit'}),
        }
    out = _audit_to_dict(report)
    out['result_id'] = rid
    return out


def _audit_to_dict(report: Any) -> Dict[str, Any]:
    """Normalize whatever sp.audit returns into a JSON-friendly dict."""
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
    return {'value': report}


def _tool_brief(rid: Optional[str]) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and 'error' in obj:
        return obj
    import statspai as sp
    brief_fn = getattr(sp, 'brief', None)
    if brief_fn is None:
        return {'error': "sp.brief is not available in this build"}
    try:
        text = brief_fn(obj)
    except Exception as e:
        from .remediation import remediate
        return {
            'error': f"{type(e).__name__}: {e}",
            'remediation': remediate(e, context={'tool': 'brief'}),
        }
    return {'brief': str(text), 'result_id': rid}


def _tool_sensitivity_from_result(rid: Optional[str],
                                   arguments: Dict[str, Any],
                                   *, detail: str,
                                   as_handle: bool) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and 'error' in obj:
        return obj
    method = arguments.get('method', 'evalue')
    benchmark = arguments.get('benchmark_covariate')

    import statspai as sp
    try:
        if method == 'evalue':
            fn = getattr(sp, 'evalue_from_result', None) or getattr(sp, 'evalue', None)
            result = fn(obj) if fn else None
        elif method == 'oster':
            fn = getattr(sp, 'oster_bounds', None)
            result = fn(obj) if fn else None
        elif method == 'cinelli_hazlett':
            fn = getattr(sp, 'sensemakr', None)
            kwargs = {'benchmark_covariate': benchmark} if benchmark else {}
            result = fn(obj, **kwargs) if fn else None
        else:
            fn = getattr(sp, 'sensitivity', None)
            result = fn(obj) if fn else None
    except Exception as e:
        from .remediation import remediate
        return {
            'error': f"{type(e).__name__}: {e}",
            'remediation': remediate(e, context={'tool': 'sensitivity_from_result'}),
        }
    if result is None:
        return {'error': f"sensitivity method {method!r} not available "
                          "in this build"}

    from .tools import _default_serializer
    out = _default_serializer(result, detail=detail)
    if not isinstance(out, dict):
        out = {'value': out}
    out['source_result_id'] = rid
    if as_handle:
        new_rid = RESULT_CACHE.put(result, tool='sensitivity_from_result',
                                     arguments={'source': rid, 'method': method})
        out['result_id'] = new_rid
        out['result_uri'] = f"statspai://result/{new_rid}"
    return out


def _tool_honest_did_from_result(rid: Optional[str],
                                  arguments: Dict[str, Any],
                                  *, detail: str,
                                  as_handle: bool) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and 'error' in obj:
        return obj

    betas, sigma, n_pre, n_post = _extract_event_study(obj)
    if betas is None or sigma is None:
        return {
            'error': ("could not extract event-study coefficients + "
                      "covariance from the cached result"),
            'hint': ("honest_did_from_result expects a result fitted by "
                     "sp.event_study / sp.callaway_santanna / "
                     "sp.did_imputation / sp.sun_abraham. Run one of "
                     "those with as_handle=true first."),
        }

    method = arguments.get('method', 'SD')
    m_bar = arguments.get('m_bar')

    import statspai as sp
    fn = getattr(sp, 'honest_did', None)
    if fn is None:
        return {'error': "sp.honest_did is not available in this build"}
    kwargs = dict(betas=list(betas), sigma=_listify_sigma(sigma),
                   num_pre_periods=int(n_pre),
                   num_post_periods=int(n_post),
                   method=method)
    if m_bar is not None:
        kwargs['m_bar'] = float(m_bar)
    try:
        result = fn(**kwargs)
    except Exception as e:
        from .remediation import remediate
        return {
            'error': f"{type(e).__name__}: {e}",
            'remediation': remediate(e, context={'tool': 'honest_did_from_result'}),
        }

    from .tools import _default_serializer
    out = _default_serializer(result, detail=detail)
    if not isinstance(out, dict):
        out = {'value': out}
    out['source_result_id'] = rid
    out['extracted_n_pre'] = int(n_pre)
    out['extracted_n_post'] = int(n_post)
    if as_handle:
        new_rid = RESULT_CACHE.put(result, tool='honest_did_from_result',
                                     arguments={'source': rid, 'method': method})
        out['result_id'] = new_rid
        out['result_uri'] = f"statspai://result/{new_rid}"
    return out


def _extract_event_study(obj: Any):
    """Best-effort extraction of (betas, sigma, n_pre, n_post)."""
    import numpy as np
    # Direct attribute lookup
    betas = getattr(obj, 'event_study_betas', None) or \
            getattr(obj, 'betas', None) or \
            getattr(obj, 'coefficients', None)
    sigma = getattr(obj, 'event_study_sigma', None) or \
            getattr(obj, 'sigma', None) or \
            getattr(obj, 'vcov', None)
    n_pre = getattr(obj, 'num_pre_periods', None) or \
            getattr(obj, 'n_pre', None)
    n_post = getattr(obj, 'num_post_periods', None) or \
             getattr(obj, 'n_post', None)
    # Common nested shape: result.event_study has its own betas / sigma
    if betas is None or sigma is None:
        es = getattr(obj, 'event_study', None)
        if es is not None:
            betas = betas or getattr(es, 'betas', None)
            sigma = sigma or getattr(es, 'sigma', None)
            n_pre = n_pre or getattr(es, 'num_pre_periods', None)
            n_post = n_post or getattr(es, 'num_post_periods', None)
    if betas is None or sigma is None:
        return None, None, None, None
    try:
        betas_arr = np.asarray(betas, dtype=float).ravel()
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 1:
            sigma_arr = np.diag(sigma_arr)
    except Exception:
        return None, None, None, None
    if n_pre is None or n_post is None:
        # Heuristic: half-and-half when caller didn't tell us
        total = betas_arr.shape[0]
        n_pre_h = total // 2
        n_post_h = total - n_pre_h
        n_pre = n_pre or n_pre_h
        n_post = n_post or n_post_h
    return betas_arr, sigma_arr, n_pre, n_post


def _listify_sigma(sigma) -> List[List[float]]:
    return [[float(x) for x in row] for row in sigma]


# ----------------------------------------------------------------------
# Workflow primitives that take a DataFrame
# ----------------------------------------------------------------------

def _tool_detect_design(arguments: Dict[str, Any],
                          data: Optional[pd.DataFrame],
                          *, detail: str,
                          as_handle: bool) -> Dict[str, Any]:
    if data is None:
        return {'error': "detect_design requires data_path"}
    import statspai as sp
    fn = getattr(sp, 'detect_design', None)
    if fn is None:
        return {'error': "sp.detect_design is not available"}
    kwargs = {k: v for k, v in arguments.items() if v is not None}
    try:
        out = fn(data, **kwargs)
    except Exception as e:
        from .remediation import remediate
        return {
            'error': f"{type(e).__name__}: {e}",
            'remediation': remediate(e, context={'tool': 'detect_design'}),
        }
    if isinstance(out, dict):
        result_dict = dict(out)
    elif hasattr(out, 'to_dict'):
        result_dict = out.to_dict()
    else:
        result_dict = {'value': str(out)}
    if as_handle:
        rid = RESULT_CACHE.put(out, tool='detect_design', arguments=arguments)
        result_dict['result_id'] = rid
        result_dict['result_uri'] = f"statspai://result/{rid}"
    return result_dict


def _tool_preflight(arguments: Dict[str, Any],
                     data: Optional[pd.DataFrame],
                     *, detail: str,
                     as_handle: bool) -> Dict[str, Any]:
    if data is None:
        return {'error': "preflight requires data_path"}
    import statspai as sp
    fn = getattr(sp, 'preflight', None)
    if fn is None:
        return {'error': "sp.preflight is not available"}
    method = arguments.get('method')
    if not method:
        return {'error': "preflight requires `method`"}
    kwargs = {k: v for k, v in arguments.items()
              if k != 'method' and v is not None}
    try:
        out = fn(data, method, **kwargs)
    except Exception as e:
        from .remediation import remediate
        return {
            'error': f"{type(e).__name__}: {e}",
            'remediation': remediate(e, context={'tool': 'preflight'}),
        }
    if isinstance(out, dict):
        result_dict = dict(out)
    elif hasattr(out, 'to_dict'):
        result_dict = out.to_dict()
    else:
        result_dict = {'value': str(out), 'verdict': getattr(out, 'verdict', None)}
    if as_handle:
        rid = RESULT_CACHE.put(out, tool='preflight', arguments=arguments)
        result_dict['result_id'] = rid
        result_dict['result_uri'] = f"statspai://result/{rid}"
    return result_dict


# ----------------------------------------------------------------------
# bibtex tool — citation source-of-truth lookup
# ----------------------------------------------------------------------

_BIBTEX_CACHE: Optional[Dict[str, str]] = None


def _load_bibtex_index() -> Dict[str, str]:
    """Parse paper.bib once and cache key → entry text mapping.

    The parser is intentionally simple — paper.bib uses standard
    ``@article{key, ...}`` syntax with balanced braces. A heavyweight
    bibtex parser would add a dependency; this hand-rolled version
    handles every entry in the project's bib file.
    """
    global _BIBTEX_CACHE
    if _BIBTEX_CACHE is not None:
        return _BIBTEX_CACHE

    from pathlib import Path

    candidates = []
    try:
        import statspai as sp
        sp_dir = Path(sp.__file__).resolve().parent
        candidates.append(sp_dir.parent.parent / 'paper.bib')
    except Exception:
        pass
    candidates.append(Path.cwd() / 'paper.bib')

    bib_path: Optional[Path] = None
    for cand in candidates:
        if cand.exists():
            bib_path = cand
            break

    if bib_path is None:
        _BIBTEX_CACHE = {}
        return _BIBTEX_CACHE

    text = bib_path.read_text(encoding='utf-8')
    entries: Dict[str, str] = {}
    i = 0
    while i < len(text):
        at = text.find('@', i)
        if at < 0:
            break
        # Skip ``@string{...}`` / ``@comment{...}`` non-bib entries.
        brace = text.find('{', at)
        if brace < 0:
            break
        kind = text[at + 1:brace].strip().lower()
        if kind in {'string', 'comment', 'preamble'}:
            i = brace + 1
            continue
        # Find the matching closing brace via depth counting.
        depth = 1
        j = brace + 1
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            j += 1
        entry = text[at:j]
        # Key is the bit between '{' and the first ',' inside the entry.
        comma = entry.find(',', brace - at)
        if comma > 0:
            key = entry[(brace - at) + 1:comma].strip()
            if key:
                entries[key] = entry.strip()
        i = j

    _BIBTEX_CACHE = entries
    return _BIBTEX_CACHE


def _tool_bibtex(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from difflib import get_close_matches

    keys = arguments.get('keys') or []
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list) or not keys:
        return {
            'error': "`keys` is required (list of bib keys).",
            'example': {'keys': ['callaway2021difference', 'rambachan2023more']},
        }

    index = _load_bibtex_index()
    out_entries: Dict[str, Any] = {}
    suggestions: Dict[str, list] = {}
    for k in keys:
        k_str = str(k)
        if k_str in index:
            out_entries[k_str] = index[k_str]
        else:
            out_entries[k_str] = ""
            close = get_close_matches(k_str, list(index.keys()), n=5, cutoff=0.55)
            if close:
                suggestions[k_str] = close

    return {
        'keys': list(out_entries.keys()),
        'bibtex': out_entries,
        'unknown_keys': [k for k, v in out_entries.items() if not v],
        'suggestions': suggestions,
        'source': 'paper.bib',
        'note': ('Empty entries mean the bib key is not in paper.bib. '
                 'Do NOT fabricate — see CLAUDE.md §10.'),
    }


__all__ = [
    "WORKFLOW_TOOL_SPECS",
    "WORKFLOW_TOOL_NAMES",
    "workflow_tool_manifest",
    "execute_workflow_tool",
]
