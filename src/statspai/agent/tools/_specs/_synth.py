"""Synthetic-control family tool specs."""

from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer

#: ``sp.synth(method=...)`` values, as the registry lists them.
_SYNTH_METHODS = [
    "classic",
    "penalized",
    "demeaned",
    "detrended",
    "unconstrained",
    "elastic_net",
    "augmented",
    "sdid",
    "gsynth",
    "staggered",
    "mc",
    "discos",
    "multi_outcome",
    "scpi",
    "bayesian",
    "bsts",
    "penscm",
    "fdid",
    "rcm",
    "cluster",
    "sparse",
    "kernel",
]

SPECS: List[Dict[str, Any]] = [
    {
        "name": "synth",
        "description": (
            "Synthetic control (Abadie et al.) and its variants through one "
            "entry: classic SCM with placebo inference, synthetic DiD, "
            "augmented / penalized SCM, generalized SC (gsynth), staggered "
            "adoption and more via method=. Panel in long format, one "
            "treated unit (staggered: a treatment column instead)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "description": "Outcome column."},
                "unit": {"type": "string", "description": "Unit column."},
                "time": {"type": "string", "description": "Period column."},
                "treated_unit": {
                    "type": ["string", "integer"],
                    "description": "Value of unit that is treated.",
                },
                "treatment_time": {
                    "type": ["integer", "number", "string"],
                    "description": "First treated period.",
                },
                "method": {
                    "type": "string",
                    "enum": _SYNTH_METHODS,
                    "default": "classic",
                },
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Predictor columns matched on.",
                },
                "treatment": {
                    "type": "string",
                    "description": "0/1 column, method='staggered' only.",
                },
                "placebo": {
                    "type": "boolean",
                    "default": True,
                    "description": "In-space placebo inference.",
                },
            },
            "required": ["outcome", "unit", "time"],
        },
        "statspai_fn": "synth",
        "serializer": _default_serializer,
    },
]
