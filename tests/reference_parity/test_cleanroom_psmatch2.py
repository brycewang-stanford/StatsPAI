"""Clean-room kernel / radius / LLR matching against psmatch2 as a black box.

The functions behind ``sp.match(method='kernel' | 'radius' | 'llr')`` and
the analytic ``psmatch2`` standard error were rewritten from the psmatch2
help file, the Stata manual entries ``[R] kdensity`` / ``[R] lpoly`` and
published formulas (spec: ``docs/dev/cleanroom/psmatch2.md``). These tests
pin them to numbers produced by running Stata 18 + psmatch2 4.0.12 on a
given propensity score (``_fixtures/_generate_psmatch2_cleanroom.do``), and
check the algebraic identities the estimators must satisfy.

Tolerance: 1e-12 relative. The score is passed in, so both sides do the
same arithmetic on the same doubles; observed agreement is ~1e-15.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient
from statspai.matching._matched_frame import psmatch2_se
from statspai.matching._stata_lpoly import lpoly_predict, stata_epanechnikov

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_RTOL = 1e-12


@pytest.fixture(scope="module")
def stata() -> dict:
    path = _FIX / "psmatch2_cleanroom_stata.json"
    if not path.exists():  # pragma: no cover - ships with the repo
        pytest.skip("missing psmatch2_cleanroom_stata.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kd() -> pd.DataFrame:
    df = pd.read_csv(_FIX / "psmatch2_kernel_data.csv", encoding="utf-8")
    df = df[["x1", "x2", "d", "y", "_pscore"]].rename(columns={"_pscore": "ps"})
    return df


def _boundary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 10.0, 20.0, 30.0, 40.0],
            "d": [1, 1, 0, 0, 0, 0],
            "ps": [0.5, 0.25, 0.25, 0.5, 0.75, 0.375],
            "x": 0.0,
        }
    )


def _fit(df, covariates, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(df, y="y", treat="d", covariates=covariates, pscore="ps", **kw)


@pytest.mark.parametrize("kernel", ["epan", "normal", "biweight", "uniform", "tricube"])
def test_kernel_matching_att_and_se_match_psmatch2(kd, stata, kernel):
    r = _fit(kd, ["x1", "x2"], method="kernel", kernel=kernel, bwidth=0.1)
    ref = stata[f"kernel_{kernel}"]
    assert r.estimate == pytest.approx(ref["att"], rel=_RTOL)
    assert r.model_info["se_method"] == "psmatch2"
    assert r.se == pytest.approx(ref["seatt"], rel=_RTOL)


@pytest.mark.parametrize("kernel", ["tricube", "normal", "uniform", "biweight"])
def test_llr_att_matches_psmatch2(kd, stata, kernel):
    r = _fit(kd, ["x1", "x2"], method="llr", kernel=kernel, bwidth=0.3, se_method="ai")
    assert r.estimate == pytest.approx(stata[f"llr_{kernel}"]["att"], rel=_RTOL)


def test_llr_epan_compat_matches_psmatch2(kd, stata):
    r = _fit(
        kd,
        ["x1", "x2"],
        method="llr",
        kernel="epan",
        bwidth=0.3,
        llr_stata_compat=True,
    )
    ref = stata["llr_epan"]
    assert r.estimate == pytest.approx(ref["att"], rel=_RTOL)
    # psmatch2's SE here is the analytic formula on the *raw* outcomes
    assert r.se == pytest.approx(ref["seatt"], rel=_RTOL)


@pytest.mark.parametrize("kernel", ["epan", "uniform", "tricube", "biweight", "normal"])
def test_boundary_design_matches_psmatch2(stata, kernel):
    r = _fit(_boundary(), ["x"], method="kernel", kernel=kernel, bwidth=0.25)
    ref = stata[f"boundary_kernel_{kernel}"]
    assert r.estimate == pytest.approx(ref["att"], rel=_RTOL)
    assert r.se == pytest.approx(ref["seatt"], rel=_RTOL)


def test_window_is_closed_at_one_bandwidth(stata):
    """|u| = 1 is inside: uniform weighs it, epan gives it no weight."""
    df = _boundary()
    radius = _fit(df, ["x"], method="radius", caliper=0.25)
    assert radius.estimate == pytest.approx(stata["boundary_radius"]["att"], rel=_RTOL)
    uni = _fit(df, ["x"], method="kernel", kernel="uniform", bwidth=0.25)
    w = uni.matched_data["_weight"].to_numpy(dtype=float)
    assert w[4] == pytest.approx(stata["boundary_uniform_weight_at_h"], rel=_RTOL)
    epan = _fit(df, ["x"], method="kernel", kernel="epan", bwidth=0.25)
    assert stata["boundary_epan_weight_at_h_missing"] is True
    assert np.isnan(epan.matched_data["_weight"].to_numpy(dtype=float)[4])


def test_llr_weights_are_a_local_linear_smoother(kd):
    """Per treated unit: weights sum to 1 and annihilate the score gap."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(
            kd,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            pscore="ps",
            method="llr",
            kernel="tricube",
            bwidth=0.3,
            se_method="ai",
        )
    md = r.matched_data
    t = md["_treated"].to_numpy() == 1
    # every matched treated unit contributes total control weight 1
    n_matched = int(r.model_info["n_matched_treated"])
    assert np.nansum(md["_weight"].to_numpy(dtype=float)[~t]) == pytest.approx(
        n_matched, rel=1e-12
    )
    # the matched outcome is a weighted mean: ATT = mean(y - _y)
    keep = t & md["_y"].notna().to_numpy()
    att = float(np.mean(md["y"].to_numpy()[keep] - md["_y"].to_numpy()[keep]))
    assert att == pytest.approx(r.estimate, abs=1e-12)


def test_llr_reproduces_a_line_exactly():
    """A local linear fit is exact when E[Y | p] is linear among controls."""
    rng = np.random.default_rng(3)
    n = 120
    ps = rng.uniform(0.2, 0.8, n)
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 2.0 + 5.0 * ps + 1.5 * d  # true effect 1.5, controls exactly linear
    df = pd.DataFrame({"y": y, "d": d, "ps": ps, "x": 0.0})
    r = _fit(df, ["x"], method="llr", kernel="tricube", bwidth=0.2, se_method="ai")
    assert r.estimate == pytest.approx(1.5, abs=1e-10)


def test_psmatch2_se_formula_by_hand():
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0])
    d = np.array([1, 1, 1, 0, 0, 0, 0])
    support = np.array([1, 1, 0, 1, 1, 1, 1])
    w = np.array([1.0, 1.0, np.nan, 0.5, 1.5, 0.0, np.nan])
    # treated on support: y = 1, 3; controls with a weight: y = 5, 4, 6
    v1 = np.var([1.0, 3.0], ddof=1)
    v0 = np.var([5.0, 4.0, 6.0], ddof=1)
    expect = np.sqrt(v1 / 2 + v0 * (0.25 + 2.25 + 0.0) / 4)
    assert psmatch2_se(y, d, support, w) == pytest.approx(expect, rel=1e-15)
    # one matched control: the control variance is undefined
    w1 = np.array([1.0, 1.0, np.nan, 2.0, np.nan, np.nan, np.nan])
    assert np.isnan(psmatch2_se(y, d, support, w1))


def test_stata_epanechnikov_is_the_kdensity_kernel():
    u = np.linspace(-2.2, 2.2, 9)
    expect = np.where(np.abs(u) < np.sqrt(5), 0.75 * (1 - u**2 / 5) / np.sqrt(5), 0)
    np.testing.assert_allclose(stata_epanechnikov(u), expect, rtol=0, atol=1e-16)
    assert stata_epanechnikov(np.array([np.sqrt(5) + 1e-9]))[0] == 0.0


def test_lpoly_predict_is_weighted_least_squares():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 80)
    y = np.cos(4 * x) + rng.normal(0, 0.1, 80)
    at = np.array([0.2, 0.5, 0.8])
    h = 0.15
    for degree in (0, 1, 2):
        got = lpoly_predict(x, y, at, h, degree=degree)
        for g, x0 in zip(got, at):
            k = stata_epanechnikov((x - x0) / h)
            keep = k > 0
            # np.polyfit minimises sum (w_i r_i)^2, so pass sqrt(K)
            coef = np.polyfit(x[keep] - x0, y[keep], degree, w=np.sqrt(k[keep]))
            assert g == pytest.approx(coef[-1], rel=1e-9)


def test_lpoly_predict_rejects_a_nonpositive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        lpoly_predict(np.arange(3.0), np.arange(3.0), np.array([1.0]), 0.0)


def test_no_control_in_bandwidth_raises():
    df = pd.DataFrame(
        {
            "y": np.arange(6.0),
            "d": [1, 1, 1, 0, 0, 0],
            "ps": [0.8, 0.85, 0.9, 0.1, 0.15, 0.2],
            "x": 0.0,
        }
    )
    with pytest.raises(DataInsufficient, match="no treated unit"):
        _fit(df, ["x"], method="kernel", bwidth=0.05)
