"""Result-protocol tests for JAX bootstrap containers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import NumericalInstability
from statspai.fast.jax_feols import FeolsBootstrapResult, _require_finite_outputs


def test_feols_jax_bootstrap_result_protocol_json_safe():
    result = FeolsBootstrapResult(
        coef=pd.Series({"x1": 0.25, "x2": -0.10}),
        se_boot=pd.Series({"x1": 0.05, "x2": 0.04}),
        ci_lower=pd.Series({"x1": 0.15, "x2": -0.18}),
        ci_upper=pd.Series({"x1": 0.35, "x2": -0.02}),
        boot_betas=pd.DataFrame(
            {
                "x1": [0.20, 0.25, 0.30],
                "x2": [-0.12, -0.10, -0.08],
            }
        ),
        n_boot=3,
        bootstrap_type="pairs",
    )

    full = result.to_dict()
    agent = result.to_agent_summary(max_terms=1)

    assert full["kind"] == "fast_feols_jax_bootstrap_result"
    assert full["n_boot"] == 3
    assert len(full["coefficients"]) == 2
    assert len(full["boot_betas"]) == 3
    assert agent["kind"] == "fast_feols_jax_bootstrap_agent_summary"
    assert len(agent["coefficients"]) == 1
    assert agent["truncated_terms"] == 1
    assert agent["bootstrap_distributions"]["x1"]["n"] == 3
    json.dumps(full)
    json.dumps(agent)


def test_jax_output_guard_reports_nonfinite_payloads():
    with pytest.raises(NumericalInstability) as exc:
        _require_finite_outputs(
            "feols_jax_test",
            beta=np.array([1.0, np.nan]),
            bread=np.eye(2),
        )

    assert exc.value.diagnostics["nonfinite_outputs"] == ["beta"]
    assert exc.value.alternative_functions == ["sp.fast.feols"]
