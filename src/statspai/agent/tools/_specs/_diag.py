"""Diagnostics + sensitivity + spec-curve tool specs."""

from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer, _identification_serializer

SPECS: List[Dict[str, Any]] = [
    {
        "name": "check_identification",
        "description": (
            "Design-level identification diagnostics: bad controls, "
            "overlap, cohort sizes, IV first-stage F, clustering.  Run "
            "BEFORE fitting any estimator to surface design problems."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string"},
                "treatment": {"type": "string"},
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "id": {"type": "string"},
                "time": {"type": "string"},
                "cohort": {"type": "string"},
                "running_var": {"type": "string"},
                "instrument": {"type": "string"},
                "design": {
                    "type": "string",
                    "enum": [
                        "did",
                        "rd",
                        "iv",
                        "observational",
                        "panel",
                        "rct",
                        "cross-section",
                    ],
                },
                "strict": {
                    "type": "boolean",
                    "description": "Raise IdentificationError on BLOCKERS.",
                    "default": False,
                },
            },
            "required": ["y"],
        },
        "statspai_fn": "check_identification",
        "serializer": _identification_serializer,
    },
    {
        "name": "sensitivity",
        "description": (
            "Unified sensitivity analysis for observational causal "
            "estimates — supports Oster (2019) delta/R-max, Cinelli-"
            "Hazlett (2020) omitted-variable bias bounds, and E-values "
            "(VanderWeele-Ding 2017).  Tells the agent how strong an "
            "unobserved confounder would have to be to overturn the "
            "result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": (
                        "Fitted regression / causal result handle "
                        "(result_id from a prior fit run with "
                        "as_handle=true). Required — the bounds are "
                        "computed relative to this estimate."
                    ),
                },
                "y": {
                    "type": "string",
                    "description": (
                        "Outcome column (lets the bound recompute "
                        "covariate R^2 from data alongside the result)."
                    ),
                },
                "treat": {"type": "string", "description": "Treatment column."},
                "controls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Observed control columns.",
                },
                "rho_max": {
                    "type": "number",
                    "description": (
                        "Oster R_max for the Oster bound. Default: "
                        "min(1, 1.3 x R^2 of the controlled regression)."
                    ),
                },
                "include_oster": {"type": "boolean", "default": True},
                "include_rosenbaum": {"type": "boolean", "default": True},
                "include_sensemakr": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        # The description names the unified Oster / Cinelli-Hazlett /
        # E-value dashboard, which is ``sp.unified_sensitivity``. The
        # legacy ``"sensitivity"`` target did not exist on the package
        # (agents hit an unhandled error); the schema above is faithful to
        # ``sp.unified_sensitivity``'s signature.
        "statspai_fn": "unified_sensitivity",
        "serializer": _default_serializer,
    },
    {
        "name": "spec_curve",
        "description": (
            "Specification-curve analysis (Simonsohn et al. 2020): "
            "enumerates every combination of model choices the user "
            "declares defensible, runs them all, and returns the sign/"
            "magnitude distribution.  Use when an agent needs to report "
            "robustness across a researcher-degree-of-freedom multiverse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string", "description": "Outcome column."},
                "x": {
                    "type": "string",
                    "description": "Focal regressor (treatment) column.",
                },
                "controls": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": (
                        "The multiverse of control sets: a list of "
                        "control-column lists, one per specification."
                    ),
                },
                "se_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Standard-error flavours to sweep (e.g. "
                        "'classical', 'hc1', 'cluster')."
                    ),
                },
                "cluster_var": {
                    "type": "string",
                    "description": "Cluster column for clustered SEs.",
                },
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": ["y", "x"],
        },
        "statspai_fn": "spec_curve",
        "serializer": _default_serializer,
    },
]
