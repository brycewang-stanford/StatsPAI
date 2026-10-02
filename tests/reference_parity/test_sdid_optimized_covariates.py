"""``sp.sdid(covariate_method='optimized')`` against R ``synthdid``.

Data: ``sp.california_prop99()`` with two deterministic covariates (states
numbered alphabetically from 1, ``t = year - 1970``)::

    x  = 50 + 10 sin(0.7 s + 0.3 t) + 0.5 t
    x2 = cos(1.3 s) t / 10 + sin(0.9 t + s)

written to a CSV with 17 significant digits and, in R ``synthdid`` 0.0.9::

    synthdid_estimate(Y, N0, T0, X = X)

Evidence tier. The estimate, ``beta`` and the weights agree with R to
1e-12: T2 against the implementation of the method's authors.

Stata ``sdid`` (whose default this method is) gives -17.56573 and -17.54417
on the same data, about 3e-4 from R. Both run a first-order iteration to
its cap of 10,000 steps (R reports 10,000 objective values), so each
returns the state of its own iteration, not a common optimum: T4, a
divergence between the two references. The test pins that gap so that a
change on either side is noticed. The ``projected`` method, a regression,
has no such problem and agrees with Stata to its printed digits.
"""

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def df():
    d = sp.california_prop99().copy()
    s = d["state"].astype("category").cat.codes.to_numpy() + 1
    t = d["year"] - 1970
    d["x"] = 50 + 10 * np.sin(s * 0.7 + t * 0.3) + 0.5 * t
    d["x2"] = np.cos(s * 1.3) * t / 10 + np.sin(t * 0.9 + s)
    return d


def _fit(df, covariates, **kwargs):
    kwargs.setdefault("covariate_method", "optimized")
    kwargs.setdefault("se_method", "noinference")
    return sp.sdid(
        df,
        "packspercapita",
        "state",
        "year",
        treat="treated",
        covariates=covariates,
        **kwargs,
    )


def test_one_covariate_matches_r(df):
    res = _fit(df, ["x"])
    assert res.estimate == pytest.approx(-17.561000491802, rel=1e-11)
    assert res.model_info["covariate_beta"]["x"] == pytest.approx(
        0.035107939459, rel=1e-9
    )
    lam = res.model_info["time_weights"].iloc[:, 0]
    np.testing.assert_allclose(
        lam.loc[[1986, 1987, 1988]],
        [0.1413804118, 0.3302253806, 0.5282451663],
        atol=2e-10,
    )
    omega = res.model_info["unit_weights"].iloc[:, 0]
    assert omega.max() == pytest.approx(0.0665355396, abs=2e-10)
    assert omega.sum() == pytest.approx(1.0, abs=1e-12)
    # the stopping rule never fires: the cap does
    assert res.model_info["covariate_iterations"] == 10000
    assert np.isnan(res.se)


def test_two_covariates_match_r(df):
    res = _fit(df, ["x", "x2"])
    assert res.estimate == pytest.approx(-17.543091268449, rel=1e-11)
    beta = res.model_info["covariate_beta"]
    assert beta["x"] == pytest.approx(0.035369041746, rel=1e-9)
    assert beta["x2"] == pytest.approx(0.090130263594, rel=1e-9)


def test_stata_lands_elsewhere_on_the_same_data(df):
    # sdid packspercapita sid year treated, vce(noinference) covariates(x)
    # -> -17.56573 ; covariates(x x2) -> -17.54417 (Stata stores 5 decimals)
    one = _fit(df, ["x"]).estimate
    two = _fit(df, ["x", "x2"]).estimate
    assert one == pytest.approx(-17.56573, rel=5e-4)
    assert two == pytest.approx(-17.54417, rel=5e-4)
    assert abs(one - -17.56573) > 1e-3


def test_projected_matches_stata(df):
    # ..., covariates(x, projected) -> -18.00969 ; (x x2, projected) -> -18.00677
    one = _fit(df, ["x"], covariate_method="projected").estimate
    two = _fit(df, ["x", "x2"], covariate_method="projected").estimate
    assert one == pytest.approx(-18.00969, abs=6e-6)
    assert two == pytest.approx(-18.00677, abs=6e-6)


def test_the_result_depends_on_the_scale_of_the_covariate(df):
    # an unconverged gradient iteration is not invariant to units
    rescaled = df.assign(x=df["x"] / 100.0)
    assert _fit(rescaled, ["x"]).estimate != pytest.approx(
        _fit(df, ["x"]).estimate, rel=1e-6
    )
    # the projected method is
    a = _fit(df, ["x"], covariate_method="projected").estimate
    b = _fit(rescaled, ["x"], covariate_method="projected").estimate
    assert a == pytest.approx(b, rel=1e-9)


def test_limits_are_stated(df):
    with pytest.raises(MethodIncompatibility, match="point estimate only"):
        _fit(df, ["x"], se_method="placebo")
    with pytest.raises(MethodIncompatibility, match="method='sdid'"):
        _fit(df, ["x"], method="sc")
    with pytest.raises(MethodIncompatibility, match="covariate_method="):
        _fit(df, ["x"], covariate_method=None)
    with pytest.raises(MethodIncompatibility, match="not one of"):
        _fit(df, ["x"], covariate_method="kranz")
    staggered = df.copy()
    nevada = (staggered["state"] == "Nevada") & (staggered["year"] >= 1992)
    staggered.loc[nevada, "treated"] = 1
    with pytest.raises(MethodIncompatibility, match="single adoption date"):
        _fit(staggered, ["x"])


def test_stata_default_is_not_substituted():
    out = sp.from_stata(
        "sdid packspercapita sid year treated, vce(noinference) covariates(x)"
    )
    assert out["untranslated_options"] == ["covariates"]
    assert any("3e-4" in note for note in out["notes"])
