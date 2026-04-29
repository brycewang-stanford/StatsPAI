"""IV family tool specs."""
from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer


SPECS: List[Dict[str, Any]] = [
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
]
