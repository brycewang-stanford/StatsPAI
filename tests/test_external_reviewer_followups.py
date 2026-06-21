"""Focused coverage for external reviewer follow-up items.

These tests target user-facing paths called out during review: generic
``EconometricResults.predict()``, DAG reasoning helpers, and advanced
post-estimation margins/comparison helpers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import EconometricResults
from statspai.exceptions import MethodIncompatibility


def _linear_result(
    params: pd.Series,
    *,
    se: float = 0.1,
    fitted_values: Optional[np.ndarray] = None,
) -> EconometricResults:
    data_info = {"df_resid": 50}
    if fitted_values is not None:
        data_info["fitted_values"] = fitted_values
    return EconometricResults(
        params=params,
        std_errors=pd.Series(se, index=params.index),
        model_info={"model_type": "test-linear", "method": "unit-test"},
        data_info=data_info,
    )


class TestEconometricResultsPredict:
    def test_predict_returns_stored_in_sample_fitted_values(self):
        fitted = np.array([1.0, 2.0, 3.0])
        result = _linear_result(
            pd.Series({"Intercept": 1.0, "x": 2.0}),
            fitted_values=fitted,
        )

        np.testing.assert_array_equal(result.predict(), fitted)

    def test_predict_out_of_sample_orders_intercept_and_columns(self):
        result = _linear_result(pd.Series({"Intercept": 1.0, "x": 2.0, "z": -1.0}))
        new = pd.DataFrame({"z": [3.0, 4.0], "x": [10.0, 20.0]})

        np.testing.assert_allclose(result.predict(new), [18.0, 37.0])

    def test_predict_missing_simple_column_names_the_missing_column(self):
        result = _linear_result(pd.Series({"Intercept": 1.0, "x": 2.0, "z": -1.0}))

        with pytest.raises(MethodIncompatibility, match="missing column") as excinfo:
            result.predict(pd.DataFrame({"x": [1.0]}))
        assert excinfo.value.diagnostics["missing_columns"] == ["z"]

    def test_predict_rejects_formula_derived_terms_out_of_sample(self):
        result = _linear_result(
            pd.Series({"Intercept": 1.0, "x": 2.0, "z": -1.0, "x:z": 0.5})
        )

        with pytest.raises(
            MethodIncompatibility, match="formula transforms"
        ) as excinfo:
            result.predict(pd.DataFrame({"x": [1.0], "z": [2.0]}))
        assert excinfo.value.diagnostics["derived_terms"] == ["x:z"]

    def test_predict_rejects_non_dataframe_out_of_sample_data(self):
        result = _linear_result(pd.Series({"Intercept": 1.0, "x": 2.0}))

        with pytest.raises(MethodIncompatibility, match="pandas DataFrame"):
            result.predict([[1.0]])

    def test_predict_rejects_nonnumeric_prediction_columns(self):
        result = _linear_result(pd.Series({"Intercept": 1.0, "x": 2.0}))

        with pytest.raises(MethodIncompatibility, match="must be numeric"):
            result.predict(pd.DataFrame({"x": ["bad"]}))


class TestDAGReasoningHelpers:
    def test_classify_variable_roles(self):
        confounded = sp.dag("Z -> X; Z -> Y; X -> M -> Y; X -> Y")
        assert {
            "confounder",
            "ancestor_of_treatment",
            "ancestor_of_outcome",
        }.issubset(confounded.classify_variable("Z", "X", "Y"))
        assert "mediator" in confounded.classify_variable("M", "X", "Y")

        valid_iv = sp.dag("Z -> X; X -> Y")
        assert "instrument" in valid_iv.classify_variable("Z", "X", "Y")

        exclusion_violation = sp.dag("Z -> X; Z -> Y; X -> Y")
        assert "instrument" not in exclusion_violation.classify_variable("Z", "X", "Y")

    def test_do_removes_incoming_edges_to_intervened_nodes_only(self):
        graph = sp.dag("Z -> X; W -> X; X -> M -> Y; Z -> Y")

        intervened = graph.do("X")

        assert ("Z", "X") not in intervened.edges
        assert ("W", "X") not in intervened.edges
        assert ("X", "M") in intervened.edges
        assert ("M", "Y") in intervened.edges
        assert ("Z", "Y") in intervened.edges

    def test_frontdoor_sets_match_textbook_frontdoor_graph(self):
        graph = sp.dag("X <-> Y; X -> M -> Y")

        assert {"M"} in graph.frontdoor_sets("X", "Y")


class TestAdvancedPostEstimationMargins:
    @pytest.fixture
    def interaction_result_and_data(self):
        params = pd.Series(
            {
                "Intercept": 1.0,
                "x": 2.0,
                "z": 3.0,
                "x:z": 4.0,
                "group": 0.5,
            }
        )
        result = _linear_result(params, se=0.2)
        data = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "z": [0.0, 1.0, 2.0, 3.0],
                "group": [0, 1, 2, 0],
            }
        )
        return result, data

    def test_numerical_margins_cover_interactions_and_at_values(
        self, interaction_result_and_data
    ):
        result, data = interaction_result_and_data

        out = sp.margins(result, data=data, variables=["x"], at={"group": 1})

        assert list(out["variable"]) == ["x"]
        assert out.loc[0, "dy/dx"] == pytest.approx(2.0 + 4.0 * data["z"].mean())
        assert out.loc[0, "se"] > 0

    def test_margins_at_uses_prediction_grid_and_delta_method(
        self, interaction_result_and_data
    ):
        result, data = interaction_result_and_data

        out = sp.margins_at(
            result,
            data=data,
            at={"x": [0.0, 2.0], "group": [1]},
        )

        assert out.shape[0] == 2
        margin_x0 = out.loc[out["x"] == 0.0, "margin"].iloc[0]
        margin_x2 = out.loc[out["x"] == 2.0, "margin"].iloc[0]
        assert margin_x2 - margin_x0 == pytest.approx(
            2.0 * (2.0 + 4.0 * data["z"].mean())
        )
        assert (out["se"] > 0).all()

    def test_contrast_supports_reference_adjacent_and_grand_mean(
        self, interaction_result_and_data
    ):
        result, data = interaction_result_and_data

        ref = sp.contrast(
            result,
            data=data,
            variable="group",
            method="r",
            reference=0,
        )
        adj = sp.contrast(result, data=data, variable="group", method="ar")
        grand = sp.contrast(result, data=data, variable="group", method="gw")

        assert set(ref["contrast_label"]) == {"1 vs 0", "2 vs 0"}
        contrast_1_vs_0 = ref.loc[ref["contrast_label"] == "1 vs 0", "contrast"].iloc[0]
        assert contrast_1_vs_0 == pytest.approx(0.5)
        assert set(adj["contrast_label"]) == {"1 vs 0", "2 vs 1"}
        assert len(grand) == 3

    def test_pwcompare_adjusts_pvalues_and_intervals(self, interaction_result_and_data):
        result, data = interaction_result_and_data

        out = sp.pwcompare(
            result,
            data=data,
            variable="group",
            adjust="bonferroni",
        )

        assert set(out["comparison"]) == {"1 vs 0", "2 vs 0", "2 vs 1"}
        expected = pd.Series(
            {"1 vs 0": 0.5, "2 vs 0": 1.0, "2 vs 1": 0.5},
            name="diff",
        )
        observed = out.set_index("comparison")["diff"].sort_index()
        np.testing.assert_allclose(observed, expected.sort_index())
        assert (out["pvalue_adj"] >= out["pvalue"]).all()
        assert (out["ci_lower"] < out["diff"]).all()
        assert (out["diff"] < out["ci_upper"]).all()

    def test_pwcompare_rejects_unknown_adjustment(self, interaction_result_and_data):
        result, data = interaction_result_and_data

        with pytest.raises(ValueError, match="Unknown adjustment"):
            sp.pwcompare(result, data=data, variable="group", adjust="magic")
