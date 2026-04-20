"""JSON-schema tool definitions + dispatch for StatsPAI estimators.

Each tool specification follows the Anthropic / OpenAI tool-use
format: ``{'name': ..., 'description': ..., 'input_schema': {...}}``.
Agents built on top of Claude, GPT-4, etc. can use these directly
without wrapping.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
#
# Each entry:
#   'name'            : canonical tool name (what the LLM sees)
#   'description'     : short one-paragraph doc
#   'input_schema'    : JSON schema for arguments (strict types)
#   'statspai_fn'     : path resolved lazily
#   'serializer'      : optional callable (result -> JSON-friendly dict)

def _default_serializer(r) -> Dict[str, Any]:
    """Serialise a ``CausalResult`` / ``EconometricResults`` to a JSON dict.

    LLM tool-use protocol requires JSON-serialisable return values,
    so convert floats / arrays / DataFrames to primitives.
    """
    out: Dict[str, Any] = {}
    if hasattr(r, 'estimate'):
        out['estimate'] = float(r.estimate)
    if hasattr(r, 'se'):
        out['std_error'] = float(r.se)
    if hasattr(r, 'pvalue'):
        out['p_value'] = float(r.pvalue)
    if hasattr(r, 'ci') and r.ci is not None:
        out['conf_low'] = float(r.ci[0])
        out['conf_high'] = float(r.ci[1])
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
                import pandas as pd
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
    """Serialise IdentificationReport."""
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


TOOL_REGISTRY: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------
    # OLS
    # --------------------------------------------------------------------
    {
        'name': 'regress',
        'description': (
            "Fit an OLS regression with robust (HC1) or clustered SEs. "
            "Input is a Wilkinson-style formula like 'y ~ x1 + x2'. "
            "Use this for baseline specifications or covariate-adjusted "
            "RCT analyses."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'formula': {
                    'type': 'string',
                    'description': "R-style formula, e.g. 'y ~ x1 + x2'",
                },
                'robust': {
                    'type': 'string',
                    'enum': ['hc1', 'hc2', 'hc3', 'nonrobust'],
                    'default': 'hc1',
                },
                'cluster': {
                    'type': 'string',
                    'description': 'Column name for cluster-robust SEs.',
                },
            },
            'required': ['formula'],
        },
        'statspai_fn': 'regress',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # DID classic
    # --------------------------------------------------------------------
    {
        'name': 'did',
        'description': (
            "Fit a classic 2-period 2-group difference-in-differences. "
            "Pass treatment / time / post column names. "
            "For staggered adoption across many cohorts use "
            "callaway_santanna instead."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string', 'description': 'Outcome column'},
                'treat': {
                    'type': 'string',
                    'description': 'Binary treatment-group indicator',
                },
                'time': {'type': 'string', 'description': 'Time column'},
                'post': {
                    'type': 'string',
                    'description': 'Binary post-treatment period indicator',
                },
            },
            'required': ['y', 'treat', 'time'],
        },
        'statspai_fn': 'did',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # DID Callaway-Sant'Anna
    # --------------------------------------------------------------------
    {
        'name': 'callaway_santanna',
        'description': (
            "Staggered DID (Callaway-Sant'Anna 2021): group-time ATT with "
            "doubly-robust, IPW, or regression-adjusted estimators. "
            "Robust to heterogeneous treatment effects where TWFE fails. "
            "Requires a cohort column g (first-treatment period; 0 = "
            "never-treated)."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string'},
                'g': {
                    'type': 'string',
                    'description': 'First-treatment-period cohort column '
                                   '(0 for never-treated).',
                },
                't': {'type': 'string', 'description': 'Time column'},
                'i': {'type': 'string', 'description': 'Unit ID column'},
                'estimator': {
                    'type': 'string',
                    'enum': ['dr', 'ipw', 'reg'],
                    'default': 'dr',
                },
                'control_group': {
                    'type': 'string',
                    'enum': ['nevertreated', 'notyettreated'],
                    'default': 'nevertreated',
                },
            },
            'required': ['y', 'g', 't', 'i'],
        },
        'statspai_fn': 'callaway_santanna',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # RD
    # --------------------------------------------------------------------
    {
        'name': 'rdrobust',
        'description': (
            "Sharp or fuzzy regression-discontinuity with robust bias-"
            "corrected CIs (Calonico-Cattaneo-Titiunik 2014). "
            "Use fuzzy= for IV-style fuzzy RD."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string'},
                'x': {
                    'type': 'string',
                    'description': 'Running variable column.',
                },
                'c': {
                    'type': 'number',
                    'description': 'Cutoff value.',
                    'default': 0.0,
                },
                'fuzzy': {
                    'type': 'string',
                    'description': 'Treatment column for fuzzy RD (optional).',
                },
                'kernel': {
                    'type': 'string',
                    'enum': ['triangular', 'uniform', 'epanechnikov'],
                    'default': 'triangular',
                },
            },
            'required': ['y', 'x'],
        },
        'statspai_fn': 'rdrobust',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # IV
    # --------------------------------------------------------------------
    {
        'name': 'ivreg',
        'description': (
            "2SLS instrumental-variables regression with robust or "
            "clustered SEs and first-stage F diagnostics. "
            "Formula syntax: 'y ~ x_exog + (d_endog ~ z_instrument)'."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'formula': {
                    'type': 'string',
                    'description': "'y ~ x + (d ~ z)' style.",
                },
                'robust': {
                    'type': 'string',
                    'enum': ['hc1', 'hc2', 'hc3', 'nonrobust'],
                    'default': 'hc1',
                },
            },
            'required': ['formula'],
        },
        'statspai_fn': 'ivreg',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # Matching / weighting
    # --------------------------------------------------------------------
    {
        'name': 'ebalance',
        'description': (
            "Hainmueller (2012) entropy balancing.  Targets the ATT by "
            "exactly balancing covariate means across treatment groups. "
            "No propensity-score model specification needed."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string'},
                'treat': {'type': 'string'},
                'covariates': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'moments': {
                    'type': 'integer',
                    'description': 'Max moment balanced (1=means, 2=vars).',
                    'default': 1,
                },
            },
            'required': ['y', 'treat', 'covariates'],
        },
        'statspai_fn': 'ebalance',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------
    {
        'name': 'check_identification',
        'description': (
            "Design-level identification diagnostics: bad controls, "
            "overlap, cohort sizes, IV first-stage F, clustering.  Run "
            "BEFORE fitting any estimator to surface design problems."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string'},
                'treatment': {'type': 'string'},
                'covariates': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'id': {'type': 'string'},
                'time': {'type': 'string'},
                'cohort': {'type': 'string'},
                'running_var': {'type': 'string'},
                'instrument': {'type': 'string'},
                'design': {
                    'type': 'string',
                    'enum': ['did', 'rd', 'iv', 'observational',
                             'panel', 'rct', 'cross-section'],
                },
                'strict': {
                    'type': 'boolean',
                    'description': 'Raise IdentificationError on BLOCKERS.',
                    'default': False,
                },
            },
            'required': ['y'],
        },
        'statspai_fn': 'check_identification',
        'serializer': _identification_serializer,
    },

    # --------------------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------------------
    {
        'name': 'causal',
        'description': (
            "End-to-end causal workflow: diagnose -> recommend estimator "
            "-> fit -> run robustness -> return result.  The one-shot "
            "entry point that lets an agent analyse a dataset in a "
            "single call without orchestrating stages itself."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string'},
                'treatment': {'type': 'string'},
                'covariates': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'id': {'type': 'string'},
                'time': {'type': 'string'},
                'cohort': {'type': 'string'},
                'running_var': {'type': 'string'},
                'instrument': {'type': 'string'},
                'design': {'type': 'string',
                            'enum': ['did', 'rd', 'iv',
                                     'observational', 'panel', 'rct']},
            },
            'required': ['y'],
        },
        'statspai_fn': 'causal',
        'serializer': lambda w: {
            'design': w.design,
            'verdict': w.diagnostics.verdict if w.diagnostics else None,
            'top_method': (
                w.recommendation.recommendations[0]['method']
                if (w.recommendation and w.recommendation.recommendations)
                else None
            ),
            'estimate': (float(w.result.estimate)
                         if w.result is not None
                            and hasattr(w.result, 'estimate')
                         else None),
            'std_error': (float(w.result.se)
                          if w.result is not None
                             and hasattr(w.result, 'se')
                          else None),
            'conf_low': (float(w.result.ci[0])
                         if w.result is not None
                            and hasattr(w.result, 'ci')
                         else None),
            'conf_high': (float(w.result.ci[1])
                          if w.result is not None
                             and hasattr(w.result, 'ci')
                          else None),
            'robustness': {
                k: v for k, v in w.robustness_findings.items()
                if isinstance(v, (int, float, str))
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tool_manifest() -> List[Dict[str, Any]]:
    """Return the list of tool specifications for an LLM agent.

    Each spec conforms to the Anthropic / OpenAI tool-use JSON schema
    format.  Drop directly into ``client.messages.create(tools=...)``
    (Anthropic) or ``client.chat.completions.create(tools=[...])``
    (OpenAI).

    Returns
    -------
    list of dict
        Each with keys ``'name'``, ``'description'``, ``'input_schema'``.
    """
    return [
        {
            'name': t['name'],
            'description': t['description'],
            'input_schema': t['input_schema'],
        }
        for t in TOOL_REGISTRY
    ]


def _resolve_fn(fn_name: str) -> Callable:
    """Import and return the statspai callable for the given name."""
    import statspai as sp
    fn = getattr(sp, fn_name, None)
    if fn is None:
        raise ValueError(f"Tool {fn_name!r} not found on statspai.")
    return fn


def execute_tool(name: str,
                 arguments: Dict[str, Any],
                 data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Dispatch a tool call to the right StatsPAI function.

    Parameters
    ----------
    name : str
        Tool name (must match a ``TOOL_REGISTRY`` entry).
    arguments : dict
        Tool-call arguments as provided by the LLM (JSON object).
    data : pd.DataFrame, optional
        Dataset the estimator runs on.  Required by most tools.

    Returns
    -------
    dict
        JSON-serialisable result, suitable for returning to the LLM as
        tool output.  On error, returns ``{'error': <str>,
        'remediation': <dict>}`` — the agent can use ``remediation``
        to repair its next call.
    """
    spec = next((t for t in TOOL_REGISTRY if t['name'] == name), None)
    if spec is None:
        return {
            'error': f"Unknown tool: {name!r}",
            'available_tools': [t['name'] for t in TOOL_REGISTRY],
        }

    fn = _resolve_fn(spec['statspai_fn'])
    serialize = spec.get('serializer', _default_serializer)

    # Most tools take `data=` as first positional (or kwarg).
    # Formula-based ones (regress, ivreg) also take data.
    kwargs = dict(arguments)
    if data is not None:
        kwargs['data'] = data

    try:
        result = fn(**kwargs)
        return serialize(result)
    except Exception as e:
        # Lazy import to avoid cycles
        from .remediation import remediate as _remediate
        return {
            'error': f"{type(e).__name__}: {e}",
            'tool': name,
            'arguments': {k: v for k, v in arguments.items()
                          if not isinstance(v, pd.DataFrame)},
            'remediation': _remediate(e, context={'tool': name,
                                                   'arguments': arguments}),
        }
