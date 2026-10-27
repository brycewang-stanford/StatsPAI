"""OLS / regression family tool specs."""

from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer

SPECS: List[Dict[str, Any]] = [
    {
        "name": "regress",
        "description": (
            "Fit an OLS regression; classical SEs unless robust= or "
            "cluster=. "
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
                    "default": "nonrobust",
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
        "name": "feols",
        "description": (
            "OLS with high-dimensional fixed effects (R fixest / Stata "
            "reghdfe syntax): 'y ~ x1 + x2 | unit + year'. IV as "
            "'y ~ x | fe | d ~ z'. vcov: 'iid', 'hetero', or "
            "{'CRV1': 'cluster_col'} (two-way: 'a + b')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fml": {
                    "type": "string",
                    "description": "fixest formula, e.g. 'y ~ x | unit + year'",
                },
                "vcov": {
                    "type": ["string", "object"],
                    "description": "'iid', 'hetero' or {'CRV1': 'col'}",
                },
                "cluster": {
                    "type": "string",
                    "description": "Cluster column (same as {'CRV1': col}).",
                },
                "weights": {"type": "string", "description": "Weight column."},
            },
            "required": ["fml"],
        },
        "statspai_fn": "feols",
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
                    "description": (
                        "Positive exposure column; log(exposure) is used as offset."
                    ),
                },
                "irr": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Report incidence-rate ratios instead of log coefficients."
                    ),
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
        # Reachable by name (tools/call, call_function) under every profile
        # but not advertised in the curated list: count models are a niche
        # next to the fixed-effects / IV / matching entry points.
        "profile": "full",
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
                    "description": (
                        'Cluster variable; defaults to entity for model="fe".'
                    ),
                },
                "offset": {"type": "string"},
                "exposure": {"type": "string"},
                "irr": {"type": "boolean", "default": False},
            },
            "required": ["formula", "entity"],
        },
        "statspai_fn": "xtnbreg",
        "serializer": _default_serializer,
        "profile": "full",
    },
]
