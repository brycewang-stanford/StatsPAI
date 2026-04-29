"""Matching / weighting family tool specs."""
from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer


SPECS: List[Dict[str, Any]] = [
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
]
