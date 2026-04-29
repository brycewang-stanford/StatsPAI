"""RD family tool specs."""
from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer


SPECS: List[Dict[str, Any]] = [
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
]
