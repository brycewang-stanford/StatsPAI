"""Matching / weighting family tool specs."""

from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer

SPECS: List[Dict[str, Any]] = [
    {
        "name": "match",
        "description": (
            "Matching estimator dispatcher (ATT by default): nearest-"
            "neighbour on the propensity score or Mahalanobis distance, "
            "Abadie-Imbens nnmatch, CEM, optimal / full matching, kernel, "
            "radius, ... via method=. Returns balance diagnostics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string"},
                "treat": {"type": "string", "description": "0/1 treatment."},
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "method": {
                    "type": "string",
                    "enum": [
                        "nearest",
                        "psm",
                        "mahalanobis",
                        "nnmatch",
                        "cem",
                        "optimal",
                        "full",
                        "kernel",
                        "radius",
                    ],
                    "default": "nearest",
                },
                "estimand": {
                    "type": "string",
                    "enum": ["ATT", "ATE", "ATC"],
                    "default": "ATT",
                },
            },
            "required": ["y", "treat", "covariates"],
        },
        "statspai_fn": "match",
        "serializer": _default_serializer,
    },
    {
        "name": "ebalance",
        "description": (
            "Hainmueller (2012) entropy balancing.  Targets the ATT by "
            "exactly balancing covariate means across treatment groups. "
            "No propensity-score model specification needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string"},
                "treat": {"type": "string"},
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "moments": {
                    "type": "integer",
                    "description": "Max moment balanced (1=means, 2=vars).",
                    "default": 1,
                },
            },
            "required": ["y", "treat", "covariates"],
        },
        "statspai_fn": "ebalance",
        "serializer": _default_serializer,
    },
]
