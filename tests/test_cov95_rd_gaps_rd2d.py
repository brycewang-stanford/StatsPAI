"""Branch tests for ``statspai.rd.rd2d`` / ``sp.rd2d_bw`` option handling.

Validation branches, one-sided intervals, the pooled approach with fuzzy
treatment and clusters, user-supplied distances, and the clustered-vce reset.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

# ``statspai.rd.rd2d`` the attribute is the function; fetch the module.
m = importlib.import_module("statspai.rd.rd2d")

B3 = [[0.0, -0.5], [0.0, 0.0], [0.0, 0.5]]


def _df(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    d = (x1 >= 0).astype(float)
    take = np.where(rng.uniform(size=n) < 0.1 + 0.8 * d, 1.0, 0.0)
    y = 0.3 * x1 + 0.2 * x2 + 2.0 * d + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "d": d,
            "take": take,
            "yf": 0.3 * x1 + 2.0 * take + rng.normal(0, 0.5, n),
            "cl": rng.integers(0, 60, n),
        }
    )


DF = _df()
KW = dict(y="y", x1="x1", x2="x2", treatment="d")


# -------------------------------------------------------------- helpers


def test_single_eval_point_given_as_a_flat_pair():
    b = m._eval_points([0.0, 0.25], None, None, None, 1)
    assert b.shape == (1, 2) and b.tolist() == [[0.0, 0.25]]


def test_bwcheck_value():
    assert m._bwcheck_value("auto", 52) == 52
    assert m._bwcheck_value(None, 52) is None
    assert m._bwcheck_value(np.int64(7), 52) == 7
    for bad in ("always", 0, True, 2.5):
        with pytest.raises(MethodIncompatibility, match="bwcheck must be"):
            m._bwcheck_value(bad, 52)


def test_kink_position_accepts_mask_or_indices():
    assert m._kink_position([False, True, False], 3).tolist() == [False, True, False]
    assert m._kink_position([1], 3).tolist() == [False, True, False]
    with pytest.raises(MethodIncompatibility, match="one True/False value"):
        m._kink_position([True, False], 3)
    with pytest.raises(MethodIncompatibility):
        m._kink_position([3], 3)


def test_kink_unknown_must_be_bool_or_ordered_pair():
    assert m._kink_unknown(True) == (True, True)
    assert m._kink_unknown((True, False)) == (True, False)
    with pytest.raises(MethodIncompatibility, match="pair of bools"):
        m._kink_unknown((True, False, True))
    with pytest.raises(MethodIncompatibility):
        m._kink_unknown((False, True))


def test_one_sided_intervals():
    est, se = np.array([1.0, -2.0]), np.array([0.5, 0.25])
    z = stats.norm.ppf(0.95)
    t, pv, lo, hi = m._inference(est, se, 0.05, "left")
    assert np.isneginf(lo).all()
    np.testing.assert_allclose(hi, est + z * se)
    t, pv, lo, hi = m._inference(est, se, 0.05, "right")
    assert np.isposinf(hi).all()
    np.testing.assert_allclose(lo, est - z * se)
    np.testing.assert_allclose(t, est / se)
    np.testing.assert_allclose(pv, 2 * stats.norm.sf(np.abs(est / se)))


def test_side_option_reaches_the_reported_interval():
    two = sp.rd2d(DF, eval_points=B3, h=0.5, **KW)
    left = sp.rd2d(DF, eval_points=B3, h=0.5, side="left", **KW)
    np.testing.assert_allclose(left.detail["estimate_q"], two.detail["estimate_q"])
    assert np.isneginf(left.detail["ci_lower"]).all()
    # a one-sided 95% bound is tighter than the two-sided upper limit
    assert (left.detail["ci_upper"] < two.detail["ci_upper"]).all()


# ------------------------------------------------------ rd2d validation


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"p": 2, "q": 1}, "q"),
        ({"weights": [1.0, 1.0]}, "weights"),
        ({"tangvec": [[1, 0]] * 3, "p": 0}, "tangvec requires p >= 1"),
        ({"deriv": (1, 1)}, "deriv must be two non-negative integers"),
        ({"h": np.ones((2, 4))}, "h must be"),
        ({"approach": "distance", "h": np.ones((2, 2))}, "shape (k, 2)"),
        (
            {"approach": "distance", "kink_unknown": True, "kink_position": [1]},
            "not both",
        ),
        ({"approach": "distance", "h": 0.5, "kink_unknown": True}, "kink"),
        ({"approach": "distance", "distance": np.ones((1500, 2))}, "distance"),
    ],
)
def test_rd2d_rejects_inconsistent_options(kwargs, fragment):
    with pytest.raises(MethodIncompatibility) as err:
        sp.rd2d(DF, eval_points=B3, **{**KW, **kwargs})
    assert fragment in str(err.value)


def test_treatment_must_be_binary():
    bad = DF.assign(d=DF["d"] * 2)
    with pytest.raises(MethodIncompatibility, match="0/1 indicator"):
        sp.rd2d(bad, **KW)
    with pytest.raises(MethodIncompatibility, match="0/1 indicator"):
        sp.rd2d_bw(bad, **KW)


def test_bwcheck_larger_than_the_sample_is_a_data_problem():
    small = DF.head(120)
    with pytest.raises(DataInsufficient, match="bwcheck") as err:
        sp.rd2d(small, eval_points=B3, bwcheck=500, **KW)
    assert err.value.diagnostics["bwcheck"] == 500
    with pytest.raises(DataInsufficient, match="bwcheck"):
        sp.rd2d_bw(small, eval_points=B3, bwcheck=500, **KW)


# ------------------------------------------------------------ clusters


def test_clustered_fit_resets_an_incompatible_vce():
    with pytest.warns(RuntimeWarning, match="Resetting vce"):
        a = sp.rd2d(DF, eval_points=B3, h=0.5, cluster="cl", vce="hc2", **KW)
    b = sp.rd2d(DF, eval_points=B3, h=0.5, cluster="cl", vce="hc1", **KW)
    np.testing.assert_allclose(a.detail["std_err_q"], b.detail["std_err_q"])
    with pytest.warns(RuntimeWarning, match="Resetting vce"):
        bw_a = sp.rd2d_bw(DF, eval_points=B3, cluster="cl", vce="hc3", **KW)
    bw_b = sp.rd2d_bw(DF, eval_points=B3, cluster="cl", vce="hc1", **KW)
    pd.testing.assert_frame_equal(bw_a, bw_b)


def test_rd2d_bw_tangvec_and_kink_conflict():
    tv = sp.rd2d_bw(DF, eval_points=B3, tangvec=[[0.0, 1.0]] * 3, **KW)
    assert list(tv.columns)[:2] == ["b1", "b2"] and (tv.iloc[:, 2:] > 0).all().all()
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.rd2d_bw(
            DF,
            eval_points=B3,
            approach="distance",
            kink_unknown=True,
            kink_position=[1],
            **KW,
        )


# -------------------------------------------------------------- pooled


def test_pooled_default_boundary_is_rdrobust_on_signed_x1():
    res = sp.rd2d(
        DF,
        y="yf",
        x1="x1",
        x2="x2",
        treatment="d",
        approach="pooled",
        fuzzy="take",
        cluster="cl",
        h=0.5,
    )
    ref = sp.rdrobust(
        DF.assign(dist=DF["x1"].abs() * np.where(DF["d"] == 1, 1, -1)),
        y="yf",
        x="dist",
        fuzzy="take",
        cluster="cl",
        h=0.5,
        manipulation_test=False,
    )
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert res.se == pytest.approx(ref.se, rel=1e-12)
    assert res.model_info["approach"] == "pooled"
    assert res.model_info["boundary"] == "x1=0"
    assert abs(res.estimate - 2.0) < 0.5


def test_pooled_rejects_a_boundary_that_is_not_finite():
    with pytest.raises(MethodIncompatibility):
        sp.rd2d(DF, approach="pooled", boundary=lambda v: np.nan, **KW)


# ------------------------------------------------------ user distances


def test_distance_columns_and_one_dimensional_array_agree():
    sign = np.where(DF["d"] == 1, 1.0, -1.0)
    dist = sign * np.hypot(DF["x1"], DF["x2"])  # distance to the point (0, 0)
    df = DF.assign(dist0=dist)
    common = dict(approach="distance", eval_points=[[0.0, 0.0]], h=0.5)
    by_col = sp.rd2d(df, distance=["dist0"], **common, **KW)
    by_arr = sp.rd2d(df, distance=dist.to_numpy(), **common, **KW)
    builtin = sp.rd2d(df, **common, **KW)
    assert by_col.estimate == pytest.approx(by_arr.estimate, rel=1e-12)
    assert by_col.estimate == pytest.approx(builtin.estimate, rel=1e-10)
    assert by_col.se == pytest.approx(builtin.se, rel=1e-10)
