"""``vce=`` on ``sp.rdbwselect`` and the helpers that call ``sp.rdrobust``.

``rdrobust``'s variance kinds are pinned against R / Stata (Track A); the
helpers below used to call it with the default ``vce='nn'`` whatever the user
wanted. The evidence here is identity: each helper's number equals the one
``rdrobust`` gives with the same ``vce``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

KINDS = ["nn", "hc0", "hc1", "hc2", "hc3"]


@pytest.fixture(scope="module")
def rd():
    rng = np.random.default_rng(11)
    n = 1500
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + 1.0 * (x >= 0) + rng.normal(0, 0.4 + 0.3 * (x > 0), n)
    return pd.DataFrame(
        {"y": y, "x": x, "z": rng.normal(size=n), "g": rng.integers(0, 80, n)}
    )


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _hb(result):
    h = np.atleast_1d(result.model_info["bandwidth_h"]).astype(float)
    b = np.atleast_1d(result.model_info["bandwidth_b"]).astype(float)
    return np.r_[np.resize(h, 2), np.resize(b, 2)]


@pytest.mark.parametrize("cluster", [None, "g"])
@pytest.mark.parametrize("vce", KINDS)
def test_rdbwselect_equals_rdrobust_bandwidths(rd, vce, cluster):
    bw = _quiet(sp.rdbwselect, rd, y="y", x="x", vce=vce, cluster=cluster)
    fit = _quiet(sp.rdrobust, rd, y="y", x="x", vce=vce, cluster=cluster)
    got = bw[["h_left", "h_right", "b_left", "b_right"]].to_numpy(float).ravel()
    np.testing.assert_allclose(got, _hb(fit), rtol=1e-13, atol=0)


def test_rdbwselect_vce_changes_the_bandwidth(rd):
    nn = _quiet(sp.rdbwselect, rd, y="y", x="x")["h_left"].iloc[0]
    hc3 = _quiet(sp.rdbwselect, rd, y="y", x="x", vce="hc3")["h_left"].iloc[0]
    assert nn != hc3


def test_rdbwselect_rejects_unknown_vce(rd):
    with pytest.raises(ValueError, match="vce must be one of"):
        sp.rdbwselect(rd, y="y", x="x", vce="cr1")


@pytest.mark.parametrize("vce", ["hc2", "nn"])
def test_rdbalance_uses_the_requested_vce(rd, vce):
    tab = _quiet(sp.rdbalance, rd, x="x", covs=["z"], vce=vce)
    ref = _quiet(sp.rdrobust, rd, y="z", x="x", vce=vce)
    row = tab.iloc[0]
    est_col = next(c for c in tab.columns if c.lower() in ("estimate", "coef", "diff"))
    np.testing.assert_allclose(float(row[est_col]), float(ref.estimate), rtol=1e-12)
    se_col = next(c for c in tab.columns if c.lower() in ("se", "std_err", "std. err."))
    np.testing.assert_allclose(float(row[se_col]), float(ref.se), rtol=1e-12)


def test_rd_flex_without_covariates_is_rdrobust_with_vce(rd):
    a = _quiet(sp.rd_flex, rd, y="y", x="x", vce="hc1")
    b = _quiet(sp.rdrobust, rd, y="y", x="x", vce="hc1")
    np.testing.assert_allclose(float(a.estimate), float(b.estimate), rtol=1e-13)
    np.testing.assert_allclose(float(a.se), float(b.se), rtol=1e-13)
