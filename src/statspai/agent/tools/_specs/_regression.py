"""OLS / regression family tool specs."""

from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer

SPECS: List[Dict[str, Any]] = [
    {
        "name": "regress",
        "description": (
            "Fit an OLS regression with robust (HC1) or clustered SEs. "
            "Input is a Wilkinson-style formula like 'y ~ x1 + x2'. "
            "Use this for baseline specifications or covariate-adjusted "
            "RCT analyses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "R-style formula, e.g. 'y ~ x1 + x2'",
                },
                "robust": {
                    "type": "string",
                    "enum": ["hc1", "hc2", "hc3", "nonrobust"],
                    "default": "hc1",
                },
                "cluster": {
                    "type": "string",
                    "description": "Column name for cluster-robust SEs.",
                },
            },
            "required": ["formula"],
        },
        "statspai_fn": "regress",
        "serializer": _default_serializer,
    },
    {
        "name": "nbreg",
        "description": (
            "Fit a negative-binomial count model. Use this for overdispersed "
            "non-negative count outcomes; formulas may include explicit "
            "fixed effects with 'y ~ x | id' for moderate panels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "R-style formula, e.g. 'count ~ x1 + x2 | id'",
                },
                "robust": {
                    "type": "string",
                    "enum": ["nonrobust", "robust", "hc0", "hc1"],
                    "default": "nonrobust",
                },
                "cluster": {
                    "type": "string",
                    "description": "Column name for cluster-robust SEs.",
                },
                "offset": {
                    "type": "string",
                    "description": "Column containing a log offset.",
                },
                "exposure": {
                    "type": "string",
                    "description": "Positive exposure column; log(exposure) is used as offset.",
                },
                "irr": {
                    "type": "boolean",
                    "default": False,
                    "description": "Report incidence-rate ratios instead of log coefficients.",
                },
                "dispersion": {
                    "type": "string",
                    "enum": ["mean", "constant"],
                    "default": "mean",
                    "description": "NB2 mean dispersion or NB1 constant dispersion.",
                },
            },
            "required": ["formula"],
        },
        "statspai_fn": "nbreg",
        "serializer": _default_serializer,
    },
    {
        "name": "xtnbreg",
        "description": (
            "Fit panel negative-binomial regression. Use model='fe' for "
            "explicit entity fixed effects via nbreg, or model='re' for a "
            "random-intercept NB-2 GLMM via menbreg. Do not use feols for "
            "negative-binomial outcomes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "Formula such as 'count ~ x1 + x2'.",
                },
                "entity": {
                    "type": "string",
                    "description": "Panel/unit identifier column.",
                },
                "time": {
                    "type": "string",
                    "description": "Optional time column.",
                },
                "model": {
                    "type": "string",
                    "enum": ["fe", "re", "pooled"],
                    "default": "fe",
                },
                "time_effects": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include time dummies in the fixed-effects model.",
                },
                "cluster": {
                    "type": "string",
                    "description": 'Cluster variable; defaults to entity for model="fe".',
                },
                "offset": {"type": "string"},
                "exposure": {"type": "string"},
                "irr": {"type": "boolean", "default": False},
            },
            "required": ["formula", "entity"],
        },
        "statspai_fn": "xtnbreg",
        "serializer": _default_serializer,
    },
]
