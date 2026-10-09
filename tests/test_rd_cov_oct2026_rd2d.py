"""Boundary-RD (``sp.rd2d`` / ``sp.rd2d_bw``) branches not reached elsewhere.

Three kinds of assertion:

* the bandwidth rules of the distance approach (CER shrinkage, IMSE pooling,
  known / unknown kink rates) checked against the closed-form rate factors
  written out here, not imported from the module;
* small-sample variance multipliers (``hc0`` / ``hc1`` / ``hc2`` / ``hc3``,
  joint and separate fits) checked as exact ratios;
* option validation: the specific exception class and message.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.rd import _rd2d_distance as dist_mod

B = [[0.0, -0.4], [0.0, 0.4]]
KW = dict(y="y", x1="x1", x2="x2", treatment="t")
DIST = dict(KW, eval_points=B, approach="distance")


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    t = (x1 >= 0).astype(int)
    y = 1.5 * t + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    out = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "t": t})
    out["cl"] = rng.integers(0, 40, n)
    out["d"] = np.where(rng.uniform(size=n) < 0.9, t, 1 - t)
    return out


def _h(res):
    return res.detail[["h0", "h1"]].to_numpy()


# ------------------------------------------------------------ evaluation points


def test_a_single_pair_is_one_evaluation_point(df):
    flat = sp.rd2d(df, eval_points=[0.0, 0.1], **KW)
    nested = sp.rd2d(df, eval_points=[[0.0, 0.1]], **KW)
    assert len(flat.detail) == 1
    pd.testing.assert_frame_equal(flat.detail, nested.detail)


# ------------------------------------------------------------ distance: bandwidths


def test_cer_bandwidth_is_the_mse_bandwidth_times_the_cer_rate(df):
    mse = _h(sp.rd2d(df, bwselect="mserd", **DIST))
    cer = _h(sp.rd2d(df, bwselect="cerrd", **DIST))
    n, p = len(df), 1
    # n^{1/(2p+4) - 1/(p+4)}: the MSE rate swapped for the coverage-error rate.
    factor = n ** (1 / (2 * p + 4) - 1 / (p + 4))
    # same floating-point operations in a different order: 1e-12 relative
    np.testing.assert_allclose(cer, mse * factor, rtol=1e-12)
    assert factor < 1


def test_imse_bandwidth_is_common_to_every_evaluation_point(df):
    res = sp.rd2d(df, bwselect="imserd", **DIST)
    h = _h(res)
    assert np.ptp(h) == 0.0  # one number for both sides and both points
    pointwise = _h(sp.rd2d(df, bwselect="mserd", **DIST))
    # the pooled bandwidth lies between the pointwise ones
    assert pointwise.min() < h[0, 0] < pointwise.max()


def test_unknown_kink_shrinks_main_and_bias_bandwidths_at_the_stated_rates(df):
    base = sp.rd2d(df, bwselect="mserd", **DIST)
    kink = sp.rd2d(df, bwselect="mserd", kink_unknown=(True, True), **DIST)
    n, p = len(df), 1
    smooth = 1 / (2 * p + 4)
    main = n ** (smooth - 0.25)  # n^{-1/(2p+4)} rate -> n^{-1/4}
    rbc = n ** (0.25 - 1 / 3)  # and n^{-1/3} for the bias-correction fit
    np.testing.assert_allclose(_h(kink), _h(base) * main, rtol=1e-12)
    np.testing.assert_allclose(
        kink.detail[["h0_rbc", "h1_rbc"]].to_numpy(), _h(kink) * rbc, rtol=1e-12
    )
    # a bare True only changes the main bandwidth's rate ...
    only_main = sp.rd2d(df, bwselect="mserd", kink_unknown=(True, False), **DIST)
    np.testing.assert_array_equal(_h(only_main), _h(kink))
    np.testing.assert_array_equal(
        only_main.detail[["h0_rbc", "h1_rbc"]].to_numpy(), _h(only_main)
    )


def test_two_sided_kink_rates_use_each_sides_own_sample_size(df):
    base = sp.rd2d(df, bwselect="msetwo", **DIST)
    kink = sp.rd2d(df, bwselect="msetwo", kink_unknown=(True, True), **DIST)
    n0, n1 = int((df.t == 0).sum()), int((df.t == 1).sum())
    assert n0 != n1
    rate = np.array([n0 ** (1 / 6 - 0.25), n1 ** (1 / 6 - 0.25)])
    np.testing.assert_allclose(_h(kink), _h(base) * rate, rtol=1e-12)
    rbc = np.array([n0 ** (0.25 - 1 / 3), n1 ** (0.25 - 1 / 3)])
    np.testing.assert_allclose(
        kink.detail[["h0_rbc", "h1_rbc"]].to_numpy(), _h(kink) * rbc, rtol=1e-12
    )


@pytest.mark.parametrize("bwselect", ["mserd", "msetwo"])
def test_known_kink_only_shrinks_the_bandwidth_at_the_kink(df, bwselect):
    base = _h(sp.rd2d(df, bwselect=bwselect, **DIST))
    by_index = sp.rd2d(df, bwselect=bwselect, kink_position=[0], **DIST)
    by_mask = sp.rd2d(df, bwselect=bwselect, kink_position=[True, False], **DIST)
    pd.testing.assert_frame_equal(by_index.detail, by_mask.detail)
    h = _h(by_index)
    # The other point is 0.8 from the kink, further than its bandwidth, so the
    # bound max(h * rate, distance to kink) does not bind there.
    np.testing.assert_array_equal(h[1], base[1])
    if bwselect == "mserd":
        rate = np.full(2, len(df) ** (1 / 6 - 0.25))
    else:
        rate = np.array(
            [(df.t == 0).sum() ** (1 / 6 - 0.25), (df.t == 1).sum() ** (1 / 6 - 0.25)]
        )
    np.testing.assert_allclose(h[0], base[0] * rate, rtol=1e-12)


def test_cer_two_sided_bandwidth_uses_side_specific_counts(df):
    mse = _h(sp.rd2d(df, bwselect="msetwo", **DIST))
    cer = _h(sp.rd2d(df, bwselect="certwo", **DIST))
    n0, n1 = int((df.t == 0).sum()), int((df.t == 1).sum())
    f = np.array([n0 ** (1 / 6 - 1 / 5), n1 ** (1 / 6 - 1 / 5)])
    np.testing.assert_allclose(cer, mse * f, rtol=1e-12)


# ------------------------------------------------------------ distance: variance


def test_hc_multipliers_are_exact_ratios_of_the_hc0_standard_errors(df):
    fits = {
        (v, fm): sp.rd2d(df, vce=v, fitmethod=fm, h=0.5, **DIST)
        for v in ("hc0", "hc1", "hc2", "hc3")
        for fm in ("joint", "separate")
    }
    hc0 = fits[("hc0", "joint")]
    n0 = hc0.detail["n_co"].to_numpy()
    n1 = hc0.detail["n_tr"].to_numpy()
    se0, se1 = hc0.model_info["se0_p"], hc0.model_info["se1_p"]
    k = 2  # local-linear: intercept and slope on each side
    # The point estimate never depends on the variance estimator.
    for fit in fits.values():
        np.testing.assert_array_equal(
            fit.detail["estimate_p"], hc0.detail["estimate_p"]
        )
    np.testing.assert_array_equal(
        fits[("hc0", "separate")].detail["std_err_p"], hc0.detail["std_err_p"]
    )
    # hc1, joint fit: one n/(n - 2k) factor over both sides.
    np.testing.assert_allclose(
        fits[("hc1", "joint")].detail["std_err_p"],
        hc0.detail["std_err_p"] * np.sqrt((n0 + n1) / (n0 + n1 - 2 * k)),
        rtol=1e-12,
    )
    # hc1, separate fits: each side carries its own n_s/(n_s - k).
    sep = fits[("hc1", "separate")]
    np.testing.assert_allclose(
        sep.model_info["se0_p"], se0 * np.sqrt(n0 / (n0 - k)), rtol=1e-12
    )
    np.testing.assert_allclose(
        sep.model_info["se1_p"], se1 * np.sqrt(n1 / (n1 - k)), rtol=1e-12
    )
    np.testing.assert_allclose(
        sep.detail["std_err_p"],
        np.sqrt(sep.model_info["se0_p"] ** 2 + sep.model_info["se1_p"] ** 2),
        rtol=1e-12,
    )
    # Leverage corrections inflate every residual: hc3 >= hc2 >= hc0, strictly
    # here because no leverage is zero.
    s0 = hc0.detail["std_err_p"].to_numpy()
    s2 = fits[("hc2", "joint")].detail["std_err_p"].to_numpy()
    s3 = fits[("hc3", "joint")].detail["std_err_p"].to_numpy()
    assert np.all(s3 > s2) and np.all(s2 > s0)


def test_cluster_resets_vce_to_hc1_and_says_so(df):
    with pytest.warns(RuntimeWarning, match="Resetting vce to 'hc1'"):
        reset = sp.rd2d(df, eval_points=B, cluster="cl", vce="hc2", **KW)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asked = sp.rd2d(df, eval_points=B, cluster="cl", vce="hc1", **KW)
    pd.testing.assert_frame_equal(reset.detail, asked.detail)
    assert reset.model_info["vce"] == "hc1"

    with pytest.warns(RuntimeWarning, match="Resetting vce to 'hc1'"):
        bw_reset = sp.rd2d_bw(df, eval_points=B, cluster="cl", vce="hc3", **KW)
    bw_asked = sp.rd2d_bw(df, eval_points=B, cluster="cl", vce="hc1", **KW)
    pd.testing.assert_frame_equal(bw_reset, bw_asked)
    # rd2d's own bandwidths are these once the two defaults that differ
    # between the entry points (bwcheck, scaleregul) are set equal.
    same = sp.rd2d_bw(
        df,
        eval_points=B,
        cluster="cl",
        bwcheck=reset.model_info["bwcheck"],
        scaleregul=3.0,
        **KW,
    )
    np.testing.assert_allclose(
        same[["h01", "h02", "h11", "h12"]].to_numpy(),
        reset.detail[["h01", "h02", "h11", "h12"]].to_numpy(),
        rtol=1e-12,
    )


# ------------------------------------------------------------ one-sided intervals


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_sided_interval_uses_the_one_sided_normal_quantile(df, side):
    alpha = 0.1
    res = sp.rd2d(df, eval_points=B, side=side, alpha=alpha, **KW)
    two = sp.rd2d(df, eval_points=B, alpha=alpha, **KW)
    d = res.detail
    z = stats.norm.ppf(1 - alpha)
    if side == "left":
        assert np.all(np.isneginf(d["ci_lower"]))
        np.testing.assert_allclose(
            d["ci_upper"], d["estimate_q"] + z * d["std_err_q"], rtol=1e-12
        )
        # tighter than the two-sided bound at the same alpha
        assert np.all(d["ci_upper"] < two.detail["ci_upper"])
    else:
        assert np.all(np.isposinf(d["ci_upper"]))
        np.testing.assert_allclose(
            d["ci_lower"], d["estimate_q"] - z * d["std_err_q"], rtol=1e-12
        )
        assert np.all(d["ci_lower"] > two.detail["ci_lower"])
    np.testing.assert_array_equal(d["p_value"], two.detail["p_value"])


# ------------------------------------------------------------ user distances


def test_user_distances_reproduce_the_default_euclidean_ones(df):
    b = np.asarray(B)
    sign = 2.0 * df["t"].to_numpy() - 1.0
    D = np.column_stack(
        [np.hypot(df["x1"] - b[j, 0], df["x2"] - b[j, 1]) * sign for j in range(2)]
    )
    default = sp.rd2d(df, **DIST)
    as_matrix = sp.rd2d(df, distance=D, **DIST)
    frame = df.assign(d1=D[:, 0], d2=D[:, 1])
    as_columns = sp.rd2d(frame, distance=["d1", "d2"], **DIST)
    pd.testing.assert_frame_equal(default.detail, as_matrix.detail)
    pd.testing.assert_frame_equal(default.detail, as_columns.detail)

    # one evaluation point: a plain vector is one column
    one = dict(DIST, eval_points=[B[0]])
    pd.testing.assert_frame_equal(
        sp.rd2d(df, distance=D[:, 0], **one).detail, sp.rd2d(df, **one).detail
    )
    with pytest.raises(MethodIncompatibility, match="one column per evaluation"):
        sp.rd2d(df, distance=D[:, 0], **DIST)


# ------------------------------------------------------------ pooled approach


def test_pooled_with_the_default_boundary_is_rdrobust_on_signed_abs_x1(df):
    pooled = sp.rd2d(df, approach="pooled", fuzzy="d", cluster="cl", **KW)
    frame = pd.DataFrame(
        {
            "y": df["y"],
            "dist": np.abs(df["x1"]) * np.where(df["t"] == 1, 1.0, -1.0),
            "fz": df["d"],
            "cl": df["cl"],
        }
    )
    direct = sp.rdrobust(
        frame, y="y", x="dist", c=0.0, fuzzy="fz", cluster="cl", manipulation_test=False
    )
    assert pooled.estimate == direct.estimate
    assert pooled.se == direct.se
    assert pooled.model_info["boundary"] == "x1=0"
    assert pooled.model_info["approach"] == "pooled"


def test_pooled_rejects_a_boundary_undefined_on_the_data(df):
    with pytest.raises(MethodIncompatibility, match="non-finite values"):
        sp.rd2d(df, approach="pooled", boundary=lambda v: np.nan, **KW)


# ------------------------------------------------------------ validation


@pytest.mark.parametrize("bad", ["yes", 0, True, 2.5, -3])
def test_bwcheck_must_be_auto_none_or_a_positive_integer(df, bad):
    with pytest.raises(MethodIncompatibility, match="bwcheck must be 'auto'"):
        sp.rd2d(df, eval_points=B, bwcheck=bad, **KW)


def test_an_integer_bwcheck_is_recorded_and_too_large_a_one_is_refused(df):
    res = sp.rd2d(df, eval_points=B, bwcheck=np.int64(30), **KW)
    assert res.model_info["bwcheck"] == 30
    with pytest.raises(DataInsufficient, match="bwcheck=5000") as err:
        sp.rd2d(df, eval_points=B, bwcheck=5000, **KW)
    assert "bwcheck" in str(err.value)
    with pytest.raises(DataInsufficient, match="bwcheck=5000"):
        sp.rd2d_bw(df, eval_points=B, bwcheck=5000, **KW)


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(q=0, p=1), "q must be no smaller than p"),
        (dict(weights=[1.0]), "one per evaluation point"),
        (dict(weights=[1.0, -1.0]), "nonzero sum"),
        (dict(tangvec=[[1, 0], [1, 0]], p=0), "tangvec requires p >= 1"),
        (dict(deriv=(1, 1)), "summing to at most p"),
        (dict(deriv=(0, -1)), "non-negative integers"),
        (dict(h=np.ones((2, 3))), r"shape \(k, 4\)"),
        (dict(approach="distance", h=np.ones((2, 3))), r"shape \(k, 2\)"),
        (
            dict(approach="distance", kink_position=[0], kink_unknown=True),
            "either kink_position or kink_unknown",
        ),
        (
            dict(approach="distance", kink_unknown=True, h=0.5),
            "only to automatic bandwidth selection",
        ),
        (
            dict(approach="distance", kink_position=[True]),
            "one True/False value per boundary point",
        ),
        (dict(approach="distance", kink_position=[2]), "0-based"),
        (dict(approach="distance", kink_unknown=(True, False, True)), "pair of bools"),
        (
            dict(approach="distance", kink_unknown=(False, True)),
            r"kink_unknown\[1\] can be True only",
        ),
    ],
)
def test_rd2d_refuses_inconsistent_options(df, kwargs, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        sp.rd2d(df, **{**KW, "eval_points": B, **kwargs})


def test_treatment_must_be_binary(df):
    bad = df.assign(t=df["t"] * 2)
    with pytest.raises(MethodIncompatibility, match="0/1 indicator"):
        sp.rd2d(bad, eval_points=B, **KW)
    with pytest.raises(MethodIncompatibility, match="0/1 indicator"):
        sp.rd2d_bw(bad, eval_points=B, **KW)


def test_rd2d_bw_distance_refuses_both_kink_options(df):
    with pytest.raises(MethodIncompatibility, match="either kink_position"):
        sp.rd2d_bw(df, kink_position=[0], kink_unknown=True, **DIST)


def test_rd2d_bw_drops_rows_with_a_missing_cluster_and_takes_a_tangent(df):
    holed = df.copy()
    holed["cl"] = holed["cl"].astype(float)
    holed.loc[holed.index[:25], "cl"] = np.nan
    got = sp.rd2d_bw(holed, eval_points=B, cluster="cl", **KW)
    want = sp.rd2d_bw(holed.iloc[25:], eval_points=B, cluster="cl", **KW)
    pd.testing.assert_frame_equal(got, want)
    assert np.all(got[["h01", "h02", "h11", "h12"]].to_numpy() > 0)

    # A tangent vector along the boundary selects the directional derivative;
    # the bandwidth frame keeps its layout, with different values.
    tang = sp.rd2d_bw(df, eval_points=B, tangvec=[[0, 1], [0, 1]], **KW)
    plain = sp.rd2d_bw(df, eval_points=B, **KW)
    assert list(tang.columns) == list(plain.columns)
    assert not np.allclose(tang["h01"], plain["h01"])


# ------------------------------------------------------------ mass points, fuzzy


def test_mass_points_warn_and_switch_the_bandwidth_floor_on(df):
    lumpy = df.copy()
    lumpy["x1"] = np.round(lumpy["x1"] * 5) / 5 + 0.1
    lumpy["x2"] = np.round(lumpy["x2"] * 4) / 4
    lumpy["t"] = (lumpy["x1"] >= 0).astype(int)
    kw = dict(DIST, eval_points=[[0.0, 0.0]])
    with pytest.warns(RuntimeWarning) as rec:
        res = sp.rd2d(lumpy, bwcheck=None, **kw)
    msgs = [str(w.message) for w in rec]
    assert any("Mass points detected" in m for m in msgs)
    assert any("masspoints=adjust" in m for m in msgs)
    # bwcheck=None is overridden by the R default 50 + p + 1 once mass points
    # are found.
    assert res.model_info["bwcheck"] == 52

    with pytest.warns(RuntimeWarning) as rec:
        sp.rd2d(lumpy, masspoints="adjust", **kw)
    msgs = [str(w.message) for w in rec]
    assert any("Mass points detected" in m for m in msgs)
    assert not any("masspoints=adjust" in m for m in msgs)


def test_a_zero_first_stage_returns_nan_with_a_warning_not_a_number(df):
    flat = df.assign(always=1.0)
    with pytest.warns(RuntimeWarning) as rec:
        res = sp.rd2d(flat, fuzzy="always", **DIST)
    msgs = " | ".join(str(w.message) for w in rec)
    assert "first-stage fuzzy RD estimate detected in bandwidth" in msgs
    assert "returning NaN" in msgs
    assert res.detail["estimate_p"].isna().all()
    assert res.detail["estimate_q"].isna().all()


# ------------------------------------------------------------ small primitives


def test_distance_primitives_on_degenerate_input():
    lo, hi = dist_mod._bwcheck_limits(np.array([]), 5)
    assert np.isnan(lo) and np.isnan(hi)
    # no kink anywhere: every point is infinitely far from one
    np.testing.assert_array_equal(
        dist_mod._kink_distance(np.zeros((3, 2)), np.zeros(3, dtype=bool)),
        np.full(3, np.inf),
    )
    # a quadratic needs more than three points
    with pytest.raises(DataInsufficient, match="cqt quantile"):
        dist_mod._poly_lm(np.arange(3.0), np.arange(3.0), 2)
    # fewer kernel-weighted points than coefficients: no intercept
    y = np.arange(5.0)
    d = np.array([0.1, 0.2, 5.0, 6.0, 7.0])
    out = dist_mod._local_intercepts(y, None, d, 1.0, 1, "tri")
    assert np.isnan(out[0]) and np.isnan(out[1])
