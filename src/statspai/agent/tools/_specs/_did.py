"""DID family tool specs: classic 2x2, Callaway-Sant'Anna, honest_did,
Bacon decomposition.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer


SPECS: List[Dict[str, Any]] = [
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
]
