"""End-to-end orchestrator + advisor tool specs (causal, recommend).

Both tools have bespoke serializers — they emit a *fixed* shape (top
recommendation / verdict / scalar estimate) rather than the standard
``CausalResult`` JSON. Defining the serializers as module-level
functions (rather than the original lambdas) keeps stack traces
readable and unit-testable.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .._helpers import _scalar_or_none


def _causal_serializer(w) -> Dict[str, Any]:
    """Serialise a ``CausalWorkflow`` result to a fixed-shape JSON dict."""
    return {
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
            w.result.estimate
            if (w.result is not None and hasattr(w.result, 'estimate'))
            else None
        ),
        'std_error': _scalar_or_none(
            w.result.se
            if (w.result is not None and hasattr(w.result, 'se'))
            else None
        ),
        'conf_low': (
            _scalar_or_none(w.result.ci[0])
            if (w.result is not None and hasattr(w.result, 'ci')
                and w.result.ci is not None
                and not isinstance(w.result.ci, (pd.DataFrame, pd.Series))
                and hasattr(w.result.ci, '__len__')
                and len(w.result.ci) == 2)
            else None
        ),
        'conf_high': (
            _scalar_or_none(w.result.ci[1])
            if (w.result is not None and hasattr(w.result, 'ci')
                and w.result.ci is not None
                and not isinstance(w.result.ci, (pd.DataFrame, pd.Series))
                and hasattr(w.result.ci, '__len__')
                and len(w.result.ci) == 2)
            else None
        ),
        'robustness': {
            k: v for k, v in w.robustness_findings.items()
            if isinstance(v, (int, float, str))
        },
    }


def _recommend_serializer(r) -> Dict[str, Any]:
    """Serialise a ``Recommendation`` result to a fixed-shape JSON dict."""
    return {
        'design': getattr(r, 'design', None),
        'top_recommendations': [
            {'method': x.get('method'),
             'reasoning': x.get('reasoning'),
             'score': x.get('score')}
            for x in (getattr(r, 'recommendations', []) or [])[:5]
        ],
        'n_recommendations': len(getattr(r, 'recommendations', []) or []),
    }


SPECS: List[Dict[str, Any]] = [
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
        'serializer': _causal_serializer,
    },
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
        'serializer': _recommend_serializer,
    },
]
