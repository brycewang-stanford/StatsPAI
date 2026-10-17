"""ML-causal family tool specs."""

from __future__ import annotations

from typing import Any, Dict, List

from .._helpers import _default_serializer

#: String learners ``sp.dml`` resolves (``dml/_learners.py``); an MCP
#: client cannot pass an estimator object.
_REGRESSORS = ["rf", "gbm", "lasso", "rlasso", "ridge", "linear", "xgb", "lgbm"]

SPECS: List[Dict[str, Any]] = [
    {
        "name": "dml",
        "description": (
            "Double/debiased machine learning (Chernozhukov et al. 2018) "
            "with cross-fitting: partially linear (plr), interactive "
            "(irm, binary treatment), and the IV versions (pliv, iivm). "
            "Treatment effect under selection on observables with "
            "flexible nuisance models."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y": {"type": "string", "description": "Outcome column."},
                "treat": {"type": "string", "description": "Treatment column."},
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Control columns.",
                },
                "model": {
                    "type": "string",
                    "enum": ["plr", "irm", "pliv", "iivm"],
                    "default": "plr",
                },
                "instrument": {
                    "type": "string",
                    "description": "Instrument, pliv / iivm only.",
                },
                "model_y": {
                    "type": "string",
                    "enum": _REGRESSORS,
                    "description": "Outcome learner (default gbm).",
                },
                "model_d": {
                    "type": "string",
                    "enum": _REGRESSORS,
                    "description": "Treatment learner; irm uses its classifier.",
                },
                "n_folds": {"type": "integer", "default": 5},
                "cluster": {
                    "type": "string",
                    "description": "Cluster column (folds and variance).",
                },
                "random_state": {"type": "integer", "default": 42},
            },
            "required": ["y", "treat", "covariates"],
        },
        "statspai_fn": "dml",
        "serializer": _default_serializer,
    },
]
