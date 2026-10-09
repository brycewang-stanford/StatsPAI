"""Coverage gaps (batch B) for statspai.did._core.

Edge branches of the shared DiD primitives: empty / single-cluster
influence-function standard errors, the joint Wald helper on scalar and
singular covariances, calendar-time indexing, the weight-estimation
influence when no cohort has mass, the ``se_method`` vocabulary and the
covariate-formula materialiser.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statspai.did._core import (
    calendar_time_aware,
    cohort_share_context,
    covariates_from_formula,
    index_calendar_time,
    influence_function_se,
    influence_se_did,
    joint_wald,
    normalize_se_method,
    parallel_trends_block,
    weight_influence,
)
from statspai.exceptions import MethodIncompatibility


# ----------------------------------------------------------------------
# influence-function standard errors
# ----------------------------------------------------------------------
def test_influence_function_se_empty_input_is_nan():
    assert np.isnan(influence_function_se(np.array([])))
    out = influence_function_se(np.empty((0, 3)))
    assert out.shape == (3,) and np.isnan(out).all()


def test_influence_function_se_single_cluster_is_nan():
    psi = np.random.default_rng(0).normal(size=(12, 2))
    out = influence_function_se(psi, cluster_ids=np.zeros(12))
    assert out.shape == (2,) and np.isnan(out).all()
    # two clusters are enough for a finite cluster-robust SE
    two = influence_function_se(psi, cluster_ids=np.repeat([0, 1], 6))
    assert np.isfinite(two).all() and (two > 0).all()


def test_influence_se_did_cluster_length_mismatch():
    with pytest.raises(MethodIncompatibility, match="cluster_ids has 3 entries for 5"):
        influence_se_did(np.ones(5), 5, cluster_ids=np.array([0, 1, 2]))


# ----------------------------------------------------------------------
# joint_wald
# ----------------------------------------------------------------------
def test_joint_wald_scalar_covariance():
    out = joint_wald(np.array([2.0]), np.array(4.0), ridge=0.0)
    assert out["statistic"] == pytest.approx(1.0)
    assert out["df"] == 1
    assert out["pvalue"] == pytest.approx(stats.chi2.sf(1.0, 1))


def test_joint_wald_shape_mismatch():
    with pytest.raises(ValueError, match=r"covariance shape \(2, 2\) inconsistent"):
        joint_wald(np.array([1.0, 2.0, 3.0]), np.eye(2))


def test_joint_wald_singular_covariance_uses_pseudo_inverse():
    est = np.array([1.0, 1.0])
    cov = np.ones((2, 2))  # exactly singular, no ridge
    out = joint_wald(est, cov, ridge=0.0)
    assert out["statistic"] == pytest.approx(float(est @ np.linalg.pinv(cov) @ est))
    assert out["statistic"] == pytest.approx(1.0)
    assert out["df"] == 2


# ----------------------------------------------------------------------
# index_calendar_time / calendar_time_aware
# ----------------------------------------------------------------------
def test_calendar_time_with_all_zero_numeric_cohort_is_never_treated():
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    df = pd.DataFrame(
        {"t": np.tile(dates, 2), "g": [0, 0, 0, np.nan, np.nan, np.nan], "y": 1.0}
    )
    out, info = index_calendar_time(df, "t", "g", function="f")
    assert out["t"].tolist() == [1, 2, 3, 1, 2, 3]
    assert (out["g"] == 0).all()
    assert info["regular"] is True and list(info["periods"]) == list(dates)


def test_calendar_time_period_dtype_steps_use_ordinals():
    regular = pd.DataFrame({"t": pd.period_range("2020-01", periods=4, freq="M")})
    _, info = index_calendar_time(regular, "t", function="f")
    assert info["regular"] is True

    gappy = pd.DataFrame(
        {"t": pd.PeriodIndex(["2020-01", "2020-02", "2020-03", "2020-09"], freq="M")}
    )
    with pytest.warns(UserWarning, match="not evenly spaced"):
        out, info = index_calendar_time(gappy, "t", function="f")
    assert info["regular"] is False
    assert out["t"].tolist() == [1, 2, 3, 4]


def test_calendar_time_aware_lets_signature_errors_through():
    @calendar_time_aware(time="time")
    def fit(data, time):
        return (data, time)

    # A call that does not bind is passed on so Python reports it as usual.
    with pytest.raises(TypeError, match="unexpected keyword argument 'tiem'"):
        fit(pd.DataFrame({"a": [1]}), tiem="a")
    frame = pd.DataFrame({"a": [1]})
    out = fit(frame, "a")
    assert out[0] is frame and out[1] == "a"


# ----------------------------------------------------------------------
# weight-estimation influence
# ----------------------------------------------------------------------
def test_cohort_share_context_weight_length_mismatch():
    with pytest.raises(ValueError, match="unit_weights has length 2, expected 4"):
        cohort_share_context(
            np.array([3.0, 4.0]), np.array([3.0, 3.0, 4.0, 0.0]), np.ones(2)
        )


def test_weight_influence_is_zero_without_cohort_mass():
    ind = np.zeros((5, 2))
    out = weight_influence(np.zeros(2), ind)
    assert out.shape == (5, 2) and not out.any()


# ----------------------------------------------------------------------
# vocabulary helpers
# ----------------------------------------------------------------------
def test_normalize_se_method_auto_without_analytic_takes_first_supported():
    got = normalize_se_method(
        "auto", supported=("jackknife", "placebo"), function="f", n_clusters=500
    )
    assert got == "jackknife"


def test_parallel_trends_block_unknown_label():
    with pytest.raises(ValueError, match="unknown parallel-trends label 'PT-XYZ'"):
        parallel_trends_block("PT-XYZ")


# ----------------------------------------------------------------------
# covariates_from_formula
# ----------------------------------------------------------------------
@pytest.fixture()
def frame():
    rng = np.random.default_rng(1)
    return pd.DataFrame({"x": rng.uniform(1, 2, 8), "z": rng.normal(size=8)})


def test_formula_with_unknown_name_is_reported(frame):
    with pytest.raises(MethodIncompatibility, match="could not evaluate") as ei:
        covariates_from_formula(frame, "~ x + missing_col")
    assert ei.value.diagnostics["formula"] == "~ x + missing_col"


def test_formula_with_only_an_intercept_term_adds_nothing(frame):
    out, cols = covariates_from_formula(frame, "~ 1 + 1")
    assert cols == [] and out is frame


def test_formula_passthrough_term_reuses_existing_column(frame):
    out, cols = covariates_from_formula(frame, "~ x + np.log(x)")
    assert cols == ["x", "np.log(x)"]
    np.testing.assert_allclose(out["np.log(x)"], np.log(frame["x"]))
    np.testing.assert_array_equal(out["x"], frame["x"])


def test_formula_clash_with_different_values_is_rejected(frame):
    df = frame.assign(**{"np.log(x)": 0.0})
    with pytest.raises(MethodIncompatibility, match="already exist in the data") as ei:
        covariates_from_formula(df, "~ np.log(x)")
    assert ei.value.diagnostics["clashes"] == ["np.log(x)"]


def test_formula_clash_with_non_numeric_column_is_rejected(frame):
    df = frame.assign(**{"np.log(x)": "text"})
    with pytest.raises(MethodIncompatibility, match="already exist in the data") as ei:
        covariates_from_formula(df, "~ np.log(x)")
    assert ei.value.diagnostics["clashes"] == ["np.log(x)"]
