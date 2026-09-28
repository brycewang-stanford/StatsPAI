"""DID family tool specs: classic 2x2, Callaway-Sant'Anna, honest_did,
Bacon decomposition.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer

SPECS: List[Dict[str, Any]] = [
    {
        "name": "did",
        "description": (
            "Difference-in-differences dispatcher. 2x2 (two periods, "
            "0/1 treat) when id is omitted; with id, a 0/1 indicator on a "
            "multi-period panel is converted to first-treatment cohorts "
            "and Callaway-Sant'Anna runs (method='auto'). Pass method= "
            "to force '2x2', 'cs', 'sa', 'bjs', 'sdid' or 'ddd'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string", "description": "Outcome column"},
                "treat": {
                    "type": "string",
                    "description": (
                        "Binary treatment indicator (0/1) or first-treatment "
                        "period column (0 = never treated)"
                    ),
                },
                "time": {"type": "string", "description": "Time period column"},
                "id": {
                    "type": "string",
                    "description": (
                        "Unit identifier; required for staggered designs "
                        "(method='cs'/'sa'/'bjs') and SDID"
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "2x2",
                        "ddd",
                        "cs",
                        "callaway_santanna",
                        "sa",
                        "sun_abraham",
                        "bjs",
                        "sdid",
                        "staggered_rollout",
                    ],
                    "default": "auto",
                    "description": "Estimator (see sp.describe_function('did')).",
                },
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional control columns",
                },
                "cluster": {
                    "type": "string",
                    "description": "Cluster column for cluster-robust SEs",
                },
                "subgroup": {
                    "type": "string",
                    "description": "Affected-subgroup column (method='ddd' only)",
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["simple", "dynamic", "group", "calendar"],
                    "description": "Callaway-Sant'Anna aggregation of ATT(g,t)",
                },
            },
            "required": ["y", "treat", "time"],
        },
        "statspai_fn": "did",
        "serializer": _default_serializer,
    },
    {
        "name": "callaway_santanna",
        "description": (
            "Staggered DID (Callaway-Sant'Anna 2021): group-time ATT with "
            "doubly-robust, IPW, or regression-adjusted estimators. "
            "Robust to heterogeneous treatment effects where TWFE fails. "
            "Requires a cohort column g (first-treatment period; 0 = "
            "never-treated)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string"},
                "g": {
                    "type": "string",
                    "description": "First-treatment-period cohort column "
                    "(0 for never-treated).",
                },
                "t": {"type": "string", "description": "Time column"},
                "i": {"type": "string", "description": "Unit ID column"},
                "estimator": {
                    "type": "string",
                    "enum": ["dr", "ipw", "reg"],
                    "default": "dr",
                },
                "control_group": {
                    "type": "string",
                    "enum": ["nevertreated", "notyettreated"],
                    "default": "nevertreated",
                },
            },
            "required": ["y", "g", "t", "i"],
        },
        "statspai_fn": "callaway_santanna",
        "serializer": _default_serializer,
    },
    {
        "name": "honest_did",
        "description": (
            "Rambachan-Roth (2023) 'honest' DID sensitivity analysis. "
            "Operates on a fitted event-study / DID result: pass "
            "``result_id`` from a prior ``sp.event_study`` / "
            "``sp.callaway_santanna`` / ``sp.did_imputation`` / "
            "``sp.sun_abraham`` fit run with ``as_handle=true`` (or use "
            "the ``honest_did_from_result`` tool, which auto-extracts the "
            "event-study coefficients). Returns honest confidence "
            "intervals under varying degrees of parallel-trends violation "
            "(smoothness or relative-magnitude restrictions). Call this "
            "when a pre-trend test rejects at low power instead of "
            "abandoning the design."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "e": {
                    "type": "integer",
                    "default": 0,
                    "description": (
                        "Event-time period (post-treatment horizon) to "
                        "build the honest CI for."
                    ),
                },
                "m_grid": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Grid of M bounds on the parallel-trends " "violation to sweep."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": ["smoothness", "relative_magnitude"],
                    "default": "smoothness",
                    "description": (
                        "Restriction family: 'smoothness' (second "
                        "differences) or 'relative_magnitude'."
                    ),
                },
                "alpha": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Significance level for the honest CI.",
                },
            },
            "required": [],
        },
        "statspai_fn": "honest_did",
        "serializer": _default_serializer,
    },
    {
        "name": "bacon_decomposition",
        "description": (
            "Goodman-Bacon (2021) decomposition: breaks the two-way "
            "fixed-effects DID estimator into its 2x2 comparison "
            "weights.  Reveals whether treated-vs-treated comparisons "
            "(which can have negative weights) dominate the estimate. "
            "Run this before trusting a TWFE-DID point estimate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string"},
                "treat": {"type": "string"},
                "time": {"type": "string"},
                "id": {"type": "string"},
            },
            "required": ["y", "treat", "time", "id"],
        },
        "statspai_fn": "bacon_decomposition",
        "serializer": _default_serializer,
    },
]
