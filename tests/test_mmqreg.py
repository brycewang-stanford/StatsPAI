"""sp.mmqreg (quantiles via moments with absorbed fixed effects) against
Stata 18 MP.

Reference ``tests/fixtures/mmqreg/stata_mmqreg.do`` on 60 unbalanced panels
plus a one-observation panel: ``mmqreg`` 2.6 (Rios-Avila; SSC) with one and
two absorbed sets and without any, each ``vce`` (default, ``robust``,
``cluster(id)``), and ``xtqreg`` 1.3 (Machado and Santos Silva; SSC). Both
are used as black boxes (help files and outputs only).

* ``mmqreg``: the location, scale and quantile coefficients and the whole
  12 x 12 covariance of the three equations agree to 1e-10 (observed about
  1e-13); the density at the quantile equals its ``e(fden)``.
* ``xtqreg``: coefficients, ``e(q)``, the location and scale coefficients
  agree to 1e-10. Its covariance matrices follow other conventions: the
  location and scale blocks are panel-clustered with the factor
  ``G/(G-1) (N-1)/(N-K)`` (reproduced below), and the quantile block is a
  GLS form that differs from ``mmqreg``'s (documented in the module notes,
  not reproduced).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "mmqreg"
_spec = importlib.util.spec_from_file_location("_mmqreg_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_mmqreg.json").read_text(encoding="utf-8"))
XS = ["x1", "x2", "ax1"]


@pytest.fixture(scope="module")
def df():
    # panel 61 is a single observation: the Stata runs drop it (see
    # stata_mmqreg.do); test_singletons_are_dropped covers it here
    full = _mod.make_panel()
    return full[full["id"] != 61]


def _mat(tag, name):
    m = STATA[tag][name]
    return np.array(m["data"]), m["rows"], m["cols"]


def _fit(df, absorb, q, vce):
    kw = {"vce": "cluster", "cluster": "id"} if vce == "cluster" else {"vce": vce}
    return sp.mmqreg(df, "y", XS, absorb=absorb, quantile=q / 100, **kw)


_CASES = [
    ("fe1", ["id"], q, v) for q in (25, 50, 90) for v in ("gls", "robust", "cluster")
]
_CASES += [("fe2", ["id", "t"], 50, v) for v in ("gls", "robust", "cluster")]
_CASES += [("nofe", None, 50, v) for v in ("gls", "robust", "cluster")]


@pytest.mark.parametrize("design,absorb,q,vce", _CASES)
def test_matches_stata_mmqreg(df, design, absorb, q, vce):
    r = _fit(df, absorb, q, vce)
    V, rows, cols = _mat(f"{design}_{vce}_{q}", "e(V)")
    ours = r.model_info["vcov_all"].loc[rows, cols].to_numpy()
    np.testing.assert_allclose(ours, V, rtol=1e-10, atol=1e-14 * np.abs(V).max())
    np.testing.assert_allclose(r.vcov().to_numpy(), V[8:, 8:], rtol=1e-10)
    gls = f"{design}_gls_{q}"
    b, brows, bcols = _mat(gls, "e(b)")
    coef = np.r_[
        r.model_info["location"]["coef"],
        r.model_info["scale"]["coef"],
        r.params.to_numpy(),
    ]
    np.testing.assert_allclose(coef, b.ravel(), rtol=1e-10, atol=1e-13)
    assert r.nobs == STATA[gls]["e(N)"]["data"][0][0] == 396
    if design == "fe1" and vce == "gls":
        fden = STATA[gls]["e(fden)"]["data"][0][0]
        qval = STATA[gls]["e(qval)"]["data"][0][0]
        assert r.model_info["density"] == pytest.approx(fden, rel=1e-10)
        assert r.model_info["q"] == pytest.approx(qval, rel=1e-12)


@pytest.mark.parametrize("q", [25, 50, 90])
def test_point_estimates_match_xtqreg(df, q):
    r = sp.mmqreg(
        df, "y", XS, absorb="id", quantile=q / 100, vce="cluster", cluster="id"
    )
    tag = f"xtqreg_{q}"
    b = _mat(tag, "e(b)")[0].ravel()
    np.testing.assert_allclose(r.params[XS], b, rtol=1e-10)
    assert r.model_info["q"] == pytest.approx(_mat(tag, "e(q)")[0][0, 0], rel=1e-12)
    loc = r.model_info["location"]["coef"][XS].to_numpy()
    sca = r.model_info["scale"]["coef"][XS].to_numpy()
    np.testing.assert_allclose(loc, _mat(tag, "e(b_location)")[0].ravel(), rtol=1e-10)
    np.testing.assert_allclose(sca, _mat(tag, "e(b_scale)")[0].ravel(), rtol=1e-10)
    # xtqreg's location / scale covariances are our cluster ones times
    # Stata's small-sample factor G/(G-1) (N-1)/(N-K)
    n, G, K = 396, 60, 4
    fac = G / (G - 1) * (n - 1) / (n - K)
    V = r.model_info["vcov_all"].to_numpy()
    np.testing.assert_allclose(
        fac * V[:3, :3], _mat(tag, "e(V_location)")[0], rtol=1e-10
    )
    np.testing.assert_allclose(
        fac * V[4:7, 4:7], _mat(tag, "e(V_scale)")[0], rtol=1e-10
    )


def test_known_truth_location_scale_dgp():
    # y = 1 + x + (1 + 0.5 x) e, e ~ N(0, 1) scaled so E|e| = 1 is the
    # model's normalization: beta(tau) = 1 + 0.5 * Q_e(tau) with
    # Q_e(tau) = sqrt(pi / 2) * Phi^{-1}(tau)
    from scipy.stats import norm

    rng = np.random.default_rng(3)
    n = 200_000
    x = rng.uniform(0, 2, n)
    e = rng.normal(size=n) * np.sqrt(np.pi / 2)
    d = pd.DataFrame({"x": x, "y": 1 + x + (1 + 0.5 * x) * e})
    for tau in (0.1, 0.5, 0.8):
        r = sp.mmqreg(d, "y", "x", quantile=tau, vce="robust")
        truth = 1 + 0.5 * np.sqrt(np.pi / 2) * norm.ppf(tau)
        # sampling SE about 0.006 at this n; 5 SE
        assert abs(r.params["x"] - truth) < 5 * r.std_errors["x"]
        assert r.model_info["q"] == pytest.approx(
            np.sqrt(np.pi / 2) * norm.ppf(tau), abs=0.02
        )


def test_row_order_and_fe_labels_do_not_matter(df):
    a = sp.mmqreg(df, "y", XS, absorb="id", quantile=0.3, vce="robust")
    sh = df.sample(frac=1, random_state=4).assign(id=lambda f: f["id"] * 3 + 1000)
    b = sp.mmqreg(sh, "y", XS, absorb="id", quantile=0.3, vce="robust")
    pd.testing.assert_series_equal(a.params, b.params, rtol=1e-11)
    pd.testing.assert_series_equal(a.std_errors, b.std_errors, rtol=1e-9)


def test_refusals_and_warnings(df):
    f = sp.mmqreg
    with pytest.raises(MethodIncompatibility, match="strictly between"):
        f(df, "y", XS, quantile=50)
    with pytest.raises(MethodIncompatibility, match="go together"):
        f(df, "y", XS, vce="cluster")
    with pytest.raises(MethodIncompatibility, match="vce"):
        f(df, "y", XS, vce="hc1")
    with pytest.raises(MethodIncompatibility, match="collinear"):
        f(df.assign(z=df["id"] * 2.0), "y", XS + ["z"], absorb="id")
    # E|nu| = 3 x^2 fitted linearly on [0, 1] is negative near 0
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 2000)
    d = pd.DataFrame({"x": x, "y": x + 3 * x**2 * rng.normal(size=2000)})
    with pytest.warns(AssumptionWarning, match="not positive"):
        f(d, "y", "x")


def test_singletons_are_dropped(df):
    full = _mod.make_panel()
    with pytest.warns(AssumptionWarning, match="dropped 1 observation"):
        r = sp.mmqreg(full, "y", XS, absorb="id", quantile=0.25)
    ref = sp.mmqreg(df, "y", XS, absorb="id", quantile=0.25)
    assert r.model_info["n_singletons_dropped"] == 1 and r.nobs == 396
    pd.testing.assert_series_equal(r.params, ref.params, rtol=1e-12)
    pd.testing.assert_series_equal(r.std_errors, ref.std_errors, rtol=1e-12)
