"""Tests for the post-estimation capability contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import statspai as sp


def test_postestimation_contract_for_regression_result():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x": rng.normal(size=100),
        }
    )
    res = sp.regress("y ~ x", data=df)

    contract = sp.postestimation_contract(res, data=df)

    assert contract["result_type"] == "EconometricResults"
    assert "tidy" in contract["available"]
    assert "glance" in contract["available"]
    assert "lincom" in contract["available"]
    assert "test" in contract["available"]
    assert "margins" in contract["available"]
    assert contract["has_data"] is True


def test_postestimation_contract_preserves_scalar_diagnostics():
    result = SimpleNamespace(
        params=pd.Series({"Intercept": 1.0, "x": 0.5}),
        std_errors=pd.Series({"Intercept": 0.2, "x": 0.1}),
        model_info={"first_stage_F": 12.5, "note": "kept"},
        tidy=lambda: pd.DataFrame(),
        glance=lambda: pd.DataFrame(),
    )
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.5]})

    contract = sp.postestimation_contract(result, data=df)

    np.testing.assert_allclose(contract["diagnostics"]["first_stage_F"], 12.5)
    assert contract["diagnostics"]["note"] == "kept"
    assert "lincom" in contract["available"]
    assert "margins" in contract["available"]


def test_postestimation_contract_for_causal_result():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(80):
        treated = int(i >= 40)
        for post in (0, 1):
            rows.append(
                {
                    "y": rng.normal() + treated * post,
                    "t": treated,
                    "post": post,
                }
            )
    df = pd.DataFrame(rows)
    res = sp.did_2x2(df, y="y", treat="t", time="post")

    contract = sp.postestimation_report(res)

    assert "effect_summary" in contract["available"]
    assert "tidy" in contract["available"]
    assert "recommended_next" in contract
