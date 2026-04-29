"""Diagnostics + sensitivity + spec-curve tool specs."""
from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer, _identification_serializer


SPECS: List[Dict[str, Any]] = [
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
