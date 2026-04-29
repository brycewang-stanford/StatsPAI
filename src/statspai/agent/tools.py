"""JSON-schema tool definitions + dispatch for StatsPAI estimators.

Each tool specification follows the Anthropic / OpenAI tool-use
format: ``{'name': ..., 'description': ..., 'input_schema': {...}}``.
Agents built on top of Claude, GPT-4, etc. can use these directly
without wrapping.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

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

    Parameters
    ----------
    r : CausalResult or EconometricResults
        Any fitted StatsPAI result.
    detail : {"minimal", "standard", "agent"}, default ``"agent"``
        Forwarded to ``r.to_dict(detail=...)``. Agents pass this
        through ``tools/call`` arguments to control token cost per
        call; the MCP server strips it before estimator dispatch.
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

    import pandas as _pd

    def _is_scalar(v) -> bool:
        # A scalar estimate is neither a pandas Series nor a multi-element
        # numpy array; ``float(...)`` only makes sense in that case.
        if isinstance(v, (_pd.Series, _pd.DataFrame)):
            return False
        try:
            import numpy as _np
            arr = _np.asarray(v)
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
        if (not isinstance(ci, (_pd.DataFrame, _pd.Series))
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
            # Guard against non-scalar `.estimate` / `.se` (e.g.
            # EconometricResults exposes Series-valued tidy aliases).
            'estimate': _scalar_or_none(
                w.result.estimate if (w.result is not None and hasattr(w.result, 'estimate')) else None
            ),
            'std_error': _scalar_or_none(
                w.result.se if (w.result is not None and hasattr(w.result, 'se')) else None
            ),
            'conf_low': (
                _scalar_or_none(w.result.ci[0])
                if (w.result is not None and hasattr(w.result, 'ci')
                    and w.result.ci is not None
                    and not isinstance(w.result.ci, (pd.DataFrame, pd.Series))
                    and hasattr(w.result.ci, '__len__') and len(w.result.ci) == 2)
                else None
            ),
            'conf_high': (
                _scalar_or_none(w.result.ci[1])
                if (w.result is not None and hasattr(w.result, 'ci')
                    and w.result.ci is not None
                    and not isinstance(w.result.ci, (pd.DataFrame, pd.Series))
                    and hasattr(w.result.ci, '__len__') and len(w.result.ci) == 2)
                else None
            ),
            'robustness': {
                k: v for k, v in w.robustness_findings.items()
                if isinstance(v, (int, float, str))
            },
        },
    },

    # --------------------------------------------------------------------
    # Method advisor — one of the most valuable agent entry points
    # --------------------------------------------------------------------
    {
        'name': 'recommend',
        'description': (
            "Method advisor: given a dataset + research question, "
            "recommends a ranked list of estimators with reasoning, "
            "precondition checks, and a full suggested workflow.  "
            "This is the first call an agent should make if it "
            "doesn't know which estimator to run. Supports DAG input, "
            "mediator / proxy / principal-strata variables, and "
            "optional resampling-stability verification."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string', 'description': 'Outcome column.'},
                'treatment': {
                    'type': 'string',
                    'description': 'Treatment / exposure column.',
                },
                'covariates': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Covariate columns.',
                },
                'id': {'type': 'string',
                       'description': 'Unit identifier (panel).'},
                'time': {'type': 'string',
                         'description': 'Time column (panel / DID).'},
                'running_var': {
                    'type': 'string',
                    'description': 'Running variable (RD).',
                },
                'instrument': {'type': 'string',
                               'description': 'Instrumental variable.'},
                'cutoff': {'type': 'number',
                           'description': 'RD cutoff value.'},
                'design': {
                    'type': 'string',
                    'enum': ['rct', 'did', 'rd', 'iv',
                             'observational', 'panel', 'cross-section'],
                    'description': 'Override auto-detected design.',
                },
                'verify': {
                    'type': 'boolean',
                    'default': False,
                    'description': ('If True, run resampling-stability '
                                    'checks on top recommendations.'),
                },
            },
            'required': ['y'],
        },
        'statspai_fn': 'recommend',
        'serializer': lambda r: {
            'design': getattr(r, 'design', None),
            'top_recommendations': [
                {'method': x.get('method'),
                 'reasoning': x.get('reasoning'),
                 'score': x.get('score')}
                for x in (getattr(r, 'recommendations', []) or [])[:5]
            ],
            'n_recommendations': len(getattr(r, 'recommendations', []) or []),
        },
    },

    # --------------------------------------------------------------------
    # Pre-trend sensitivity (Rambachan-Roth 2023)
    # --------------------------------------------------------------------
    {
        'name': 'honest_did',
        'description': (
            "Rambachan-Roth (2023) 'honest' DID sensitivity analysis. "
            "Takes an existing DID / event-study estimate and returns "
            "honest confidence intervals under varying degrees of "
            "parallel-trends violation (smoothness or relative-magnitude "
            "restrictions).  Call this when a pre-trend test rejects "
            "at low power instead of abandoning the design."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'betas': {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'description': 'Event-study coefficients.',
                },
                'sigma': {
                    'type': 'array',
                    'description': ('Covariance matrix of betas '
                                     '(square 2-D array).'),
                },
                'num_pre_periods': {'type': 'integer'},
                'num_post_periods': {'type': 'integer'},
                'method': {
                    'type': 'string',
                    'enum': ['SD', 'RM'],
                    'default': 'SD',
                    'description': ('SD = smoothness deviation; '
                                     'RM = relative magnitude.'),
                },
                'm_bar': {
                    'type': 'number',
                    'description': 'Bound on deviation magnitude.',
                },
            },
            'required': ['betas', 'sigma',
                         'num_pre_periods', 'num_post_periods'],
        },
        'statspai_fn': 'honest_did',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # TWFE weight decomposition (Goodman-Bacon 2021)
    # --------------------------------------------------------------------
    {
        'name': 'bacon_decomposition',
        'description': (
            "Goodman-Bacon (2021) decomposition: breaks the two-way "
            "fixed-effects DID estimator into its 2x2 comparison "
            "weights.  Reveals whether treated-vs-treated comparisons "
            "(which can have negative weights) dominate the estimate. "
            "Run this before trusting a TWFE-DID point estimate."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'y': {'type': 'string'},
                'treat': {'type': 'string'},
                'time': {'type': 'string'},
                'id': {'type': 'string'},
            },
            'required': ['y', 'treat', 'time', 'id'],
        },
        'statspai_fn': 'bacon_decomposition',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # Unconditional / conditional sensitivity (Oster / Cinelli-Hazlett)
    # --------------------------------------------------------------------
    {
        'name': 'sensitivity',
        'description': (
            "Unified sensitivity analysis for observational causal "
            "estimates — supports Oster (2019) delta/R-max, Cinelli-"
            "Hazlett (2020) omitted-variable bias bounds, and E-values "
            "(VanderWeele-Ding 2017).  Tells the agent how strong an "
            "unobserved confounder would have to be to overturn the "
            "result."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'result': {
                    'type': 'string',
                    'description': ('Fitted regression / causal result '
                                     'handle (set by the caller).'),
                },
                'method': {
                    'type': 'string',
                    'enum': ['oster', 'cinelli_hazlett', 'evalue', 'auto'],
                    'default': 'auto',
                },
                'treatment': {'type': 'string'},
                'benchmark_covariate': {
                    'type': 'string',
                    'description': ('Covariate used as the benchmark '
                                     'for unobserved-confounder strength.'),
                },
            },
            'required': [],
        },
        'statspai_fn': 'sensitivity',
        'serializer': _default_serializer,
    },

    # --------------------------------------------------------------------
    # Specification-curve robustness
    # --------------------------------------------------------------------
    {
        'name': 'spec_curve',
        'description': (
            "Specification-curve analysis (Simonsohn et al. 2020): "
            "enumerates every combination of model choices the user "
            "declares defensible, runs them all, and returns the sign/"
            "magnitude distribution.  Use when an agent needs to report "
            "robustness across a researcher-degree-of-freedom multiverse."
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
                'model_family': {
                    'type': 'string',
                    'enum': ['ols', 'did', 'iv', 'panel'],
                    'default': 'ols',
                },
                'subsample_vars': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': ('Variables defining subsample splits '
                                     'to include in the curve.'),
                },
            },
            'required': ['y', 'treatment'],
        },
        'statspai_fn': 'spec_curve',
        'serializer': _default_serializer,
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tool_manifest(*, curated_only: bool = False) -> List[Dict[str, Any]]:
    """Return the list of tool specifications for an LLM agent.

    Each spec conforms to the Anthropic / OpenAI tool-use JSON schema
    format.  Drop directly into ``client.messages.create(tools=...)``
    (Anthropic) or ``client.chat.completions.create(tools=[...])``
    (OpenAI).

    Parameters
    ----------
    curated_only : bool, default False
        If True, return only the hand-curated tools (the bespoke 13 +
        the workflow / handle / bibtex tools registered by
        :mod:`statspai.agent.workflow_tools`). The default merges
        them with the auto-generated manifest covering every
        agent-safe registered function so the caller sees the full
        catalogue (~100+ tools).

    Returns
    -------
    list of dict
        Each with keys ``'name'``, ``'description'``, ``'input_schema'``.
    """
    curated: List[Dict[str, Any]] = [
        {
            'name': t['name'],
            'description': t['description'],
            'input_schema': t['input_schema'],
        }
        for t in TOOL_REGISTRY
    ]
    # Workflow tools (audit_result / brief_result / sensitivity_from_result
    # / honest_did_from_result / audit / preflight / detect_design /
    # brief / bibtex) are first-class hand-curated entries — they're
    # what the prompt templates reference and what closes the chained
    # "fit → audit → sensitivity" loop. Append before the auto-merge
    # so collisions on auto-generated stubs of the same name resolve to
    # the hand-curated version.
    from .workflow_tools import workflow_tool_manifest
    from .pipeline_tools import pipeline_tool_manifest
    seen = {t['name'] for t in curated}
    for wt in workflow_tool_manifest():
        if wt['name'] not in seen:
            curated.append(wt)
            seen.add(wt['name'])
    for pt in pipeline_tool_manifest():
        if pt['name'] not in seen:
            curated.append(pt)
            seen.add(pt['name'])

    if curated_only:
        return curated

    # Lazy import: auto_tools walks the registry, which can trigger
    # submodule imports — best deferred until someone actually asks.
    from .auto_tools import merged_tool_manifest
    try:
        return merged_tool_manifest(curated)
    except Exception as e:
        # Loud degradation: silently dropping the auto-tools is exactly
        # the kind of failure CLAUDE.md §3 #7 prohibits ("失败要响亮").
        # Emit a warning the operator (or CI log scraper) can spot.
        import warnings
        warnings.warn(
            f"auto_tool_manifest failed; falling back to curated tools. "
            f"Reason: {type(e).__name__}: {e}",
            RuntimeWarning, stacklevel=2,
        )
        return curated


def _resolve_fn(fn_name: str) -> Callable:
    """Import and return the statspai callable for the given name."""
    import statspai as sp
    fn = getattr(sp, fn_name, None)
    if fn is None:
        raise ValueError(f"Tool {fn_name!r} not found on statspai.")
    return fn


def execute_tool(name: str,
                 arguments: Dict[str, Any],
                 data: Optional[pd.DataFrame] = None,
                 *,
                 detail: str = "agent",
                 result_id: Optional[str] = None,
                 as_handle: bool = False) -> Dict[str, Any]:
    """Dispatch a tool call to the right StatsPAI function.

    Parameters
    ----------
    name : str
        Tool name (must match a ``TOOL_REGISTRY`` entry, an
        auto-registered registry function, or a built-in ``*_result`` /
        ``bibtex`` workflow tool).
    arguments : dict
        Tool-call arguments as provided by the LLM (JSON object).
    data : pd.DataFrame, optional
        Dataset the estimator runs on. Required by most tools.
    detail : {"minimal", "standard", "agent"}, default ``"agent"``
        Payload depth requested by the caller. Forwarded to the
        default serializer's ``r.to_dict(detail=...)``.
    result_id : str, optional
        Handle to a previously-fitted result cached by the server. When
        supplied, ``*_from_result`` tools resolve it from the result
        cache; other tools merge selected fields from the cached result
        (e.g. ``betas`` / ``sigma`` for ``honest_did_from_result``).
    as_handle : bool, default False
        If True, cache the fitted result and inject ``result_id`` /
        ``result_uri`` into the returned dict so a subsequent
        ``execute_tool`` call can reference it without re-fitting.

    Returns
    -------
    dict
        JSON-serialisable result, suitable for returning to the LLM as
        tool output. On error, returns ``{'error': <str>,
        'remediation': <dict>}`` — the agent can use ``remediation``
        to repair its next call.
    """
    # Workflow / result-handle tools live outside the curated
    # TOOL_REGISTRY (they're synthesised) but must be dispatched here so
    # the MCP layer never needs to know about a separate registry.
    from .workflow_tools import (
        WORKFLOW_TOOL_NAMES,
        execute_workflow_tool,
    )
    if name in WORKFLOW_TOOL_NAMES:
        return execute_workflow_tool(
            name, arguments,
            data=data, detail=detail,
            result_id=result_id, as_handle=as_handle,
        )

    # Composite pipeline tools (pipeline_did / pipeline_iv / pipeline_rd
    # — multi-stage end-to-end workflows). They embed result-cache
    # writes themselves, so we don't re-cache here.
    from .pipeline_tools import (
        PIPELINE_TOOL_NAMES,
        execute_pipeline_tool,
    )
    if name in PIPELINE_TOOL_NAMES:
        return execute_pipeline_tool(
            name, arguments,
            data=data, detail=detail, as_handle=as_handle,
        )

    spec = next((t for t in TOOL_REGISTRY if t['name'] == name), None)
    if spec is None:
        # Fall back to a registry-driven dispatch so auto-generated
        # tools (the 100+ from auto_tool_manifest) are callable too.
        from .auto_dispatch import dispatch_registry_tool
        try:
            return dispatch_registry_tool(
                name, arguments,
                data=data, detail=detail, as_handle=as_handle,
            )
        except KeyError:
            return {
                'error': f"Unknown tool: {name!r}",
                'available_tools': [t['name'] for t in TOOL_REGISTRY],
                'hint': ("Read statspai://functions for the full "
                         "machine-readable index of registered tools."),
            }

    fn = _resolve_fn(spec['statspai_fn'])
    serialize = spec.get('serializer', _default_serializer)

    # Most tools take `data=` as first positional (or kwarg).
    # Formula-based ones (regress, ivreg) also take data.
    kwargs = dict(arguments)
    if data is not None:
        kwargs['data'] = data

    def _serialize(result_obj):
        """Invoke ``serialize`` with ``detail=`` when supported.

        Custom serializers in TOOL_REGISTRY (e.g. for ``causal``,
        ``recommend``) emit a fixed shape and don't accept ``detail`` —
        forwarding the kwarg would crash them. We use
        ``inspect.signature`` to decide rather than catching
        ``TypeError``; the latter would silently swallow genuine
        ``TypeError`` bugs raised inside the serializer body.
        """
        if serialize is _default_serializer:
            return serialize(result_obj, detail=detail)
        import inspect
        try:
            params = inspect.signature(serialize).parameters
        except (TypeError, ValueError):
            # Built-in / C-extension callable without an introspectable
            # signature — assume it takes only the result.
            return serialize(result_obj)
        if "detail" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()):
            return serialize(result_obj, detail=detail)
        return serialize(result_obj)

    # Estimator call. Any failure here is attributed to the estimator
    # itself — that's where structured StatsPAIError instances come from.
    try:
        result = fn(**kwargs)
    except Exception as e:
        from .remediation import remediate as _remediate
        envelope: Dict[str, Any] = {
            'error': f"{type(e).__name__}: {e}",
            'tool': name,
            'arguments': {k: v for k, v in arguments.items()
                          if not isinstance(v, pd.DataFrame)},
            'remediation': _remediate(e, context={'tool': name,
                                                   'arguments': arguments}),
        }
        # Surface the structured StatsPAIError payload alongside the
        # legacy fields so MCP-mediated agents can branch on
        # ``error_kind`` (e.g. ``"assumption_violation"``,
        # ``"identification_failure"``) without parsing free-text
        # messages, and read ``recovery_hint`` / ``diagnostics`` /
        # ``alternative_functions`` directly from ``error_payload``.
        from ..exceptions import StatsPAIError
        if isinstance(e, StatsPAIError):
            try:
                envelope['error_kind'] = e.code
                envelope['error_payload'] = e.to_dict()
            except Exception:
                # Defensive fallback: a malformed diagnostics dict (e.g.
                # a live DataFrame) shouldn't crash the error handler
                # and lose the original exception. ``e.code`` is a
                # class attribute with a string default on every
                # ``StatsPAIError`` subclass, so reading it cannot fail.
                envelope['error_kind'] = e.code
                envelope['error_payload'] = {
                    'kind': e.code,
                    'class': type(e).__name__,
                    'message': str(e),
                }
        return envelope

    # Result serialization. A failure here is a *serializer* bug, not
    # an estimator failure — attribute it accordingly so agents don't
    # see misleading ``remediation`` advice for working call args.
    try:
        out = _serialize(result)
    except Exception as e:
        return {
            'error': f"serializer_error: {type(e).__name__}: {e}",
            'tool': name,
            'arguments': {k: v for k, v in arguments.items()
                          if not isinstance(v, pd.DataFrame)},
            'stage': 'serializer',
        }

    if not isinstance(out, dict):
        out = {'value': out}

    # Result-handle caching. When ``as_handle=True`` we stash the live
    # fitted result in the process-local LRU cache and surface a handle
    # so the next tools/call can reach it without re-loading the CSV
    # and re-fitting. This is the foundational primitive for chained
    # workflows (did → audit → sensitivity → honest_did_from_result).
    rid: Optional[str] = None
    if as_handle:
        from ._result_cache import RESULT_CACHE
        rid = RESULT_CACHE.put(
            result, tool=name,
            arguments={k: v for k, v in arguments.items()
                       if not isinstance(v, pd.DataFrame)},
        )
        out['result_id'] = rid
        out['result_uri'] = f"statspai://result/{rid}"

    # Output enrichment: pre-built next_calls + verified citations +
    # short narrative. Agents on per-call billing get more value per
    # roundtrip; agents on per-token billing can request
    # detail='minimal' to skip these or strip them client-side.
    from ._enrichment import enrich_payload
    enrich_payload(out, tool_name=name, result_id=rid,
                   base_args={k: v for k, v in arguments.items()
                              if not isinstance(v, pd.DataFrame)})

    return out
