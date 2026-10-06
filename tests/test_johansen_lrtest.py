"""``sp.johansen_lrtest``: behaviour and argument checks."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(11)
    T = 400
    trend = np.cumsum(rng.normal(size=T))
    return pd.DataFrame(
        {
            "c": trend + rng.normal(size=T),
            "y": trend + rng.normal(size=T),
            "x": np.cumsum(rng.normal(size=T)),
        }
    )


def test_true_vector_is_not_rejected_and_a_false_one_is(system):
    true = sp.johansen_lrtest(system, rank=1, beta_known=[1, -1, 0])
    assert true.df == 2
    assert true.pvalue > 0.05
    false = sp.johansen_lrtest(system, rank=1, beta_known=[1, 0, -1])
    assert false.pvalue < 0.001


def test_unrestricted_h_gives_a_zero_statistic(system):
    # H spanning everything but one direction that beta does not use
    full = sp.johansen_lrtest(system, rank=1, beta=np.eye(3)[:, :2])
    free = sp.johansen(system, lags=1)
    # eigenvalues without the restriction are those of sp.johansen
    np.testing.assert_allclose(full.eigenvalues, free.eigenvalues, rtol=1e-10)
    assert full.statistic >= 0
    assert full.eigenvalues_restricted[0] <= full.eigenvalues[0] + 1e-12


def test_dataframe_restriction_is_matched_by_name(system):
    H = pd.DataFrame({"gap": [-1.0, 1.0, 0.0], "x": [0.0, 0.0, 1.0]}, ["y", "c", "x"])
    by_name = sp.johansen_lrtest(system, rank=1, beta=H)
    by_pos = sp.johansen_lrtest(
        system, rank=1, beta=np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]).T
    )
    assert by_name.statistic == pytest.approx(by_pos.statistic, rel=1e-10)
    assert "LR statistic" in by_name.summary()


def test_weak_exogeneity_of_the_unrelated_series(system):
    test = sp.johansen_lrtest(system, rank=1, loading=np.eye(3)[:, :2])
    assert test.df == 1
    assert test.pvalue > 0.01
    assert test.loading.loc["x", "_ce1"] == pytest.approx(0.0, abs=1e-12)


def test_argument_errors(system):
    with pytest.raises(MethodIncompatibility, match="exactly one"):
        sp.johansen_lrtest(system, rank=1)
    with pytest.raises(MethodIncompatibility, match="exactly one"):
        sp.johansen_lrtest(system, rank=1, beta=np.eye(3)[:, :2], loading=np.eye(3))
    with pytest.raises(MethodIncompatibility, match="rank"):
        sp.johansen_lrtest(system, rank=3, beta=np.eye(3)[:, :2])
    with pytest.raises(MethodIncompatibility, match="rows"):
        sp.johansen_lrtest(system, rank=1, beta=np.eye(2))
    with pytest.raises(MethodIncompatibility, match="columns"):
        sp.johansen_lrtest(system, rank=2, beta=np.eye(3)[:, :1])
    with pytest.raises(MethodIncompatibility, match="exceed"):
        sp.johansen_lrtest(system, rank=1, beta_known=np.eye(3)[:, :2])
    with pytest.raises(MethodIncompatibility, match="dependent"):
        sp.johansen_lrtest(system, rank=1, beta=[[1, 2], [1, 2], [0, 0]])


def test_johansen_summary_names_the_level_and_the_sample(system):
    out = sp.johansen(system, lags=1, alpha=0.01).summary()
    assert "1% CV" in out
    assert "Used: 398" in out
