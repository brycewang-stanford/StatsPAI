"""OLS / regression family tool specs."""
from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer


SPECS: List[Dict[str, Any]] = [
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
]
