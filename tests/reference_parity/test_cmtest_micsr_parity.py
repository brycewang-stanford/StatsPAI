"""Conditional moment tests: ``sp.cmtest`` against R ``micsr::cmtest``.

Fixture: ``_fixtures/_generate_cmtest_micsr.R`` simulates the data, writes
them at 17 digits and stores what ``micsr::cmtest`` prints for a probit and
a tobit fit. micsr is the companion package of Croissant (2025),
*Microeconometrics with R*; it is GPL and is used here as a program whose
output is compared, nothing more.

Tolerances. The tobit statistics are held to 1e-9 (observed 4e-11 or
better). The probit ones are held to 1e-6 (observed 8e-7 and 4e-8):
``micsr::binomreg`` stops 1e-8 short of the probit optimum (its
coefficients are stored in the fixture and compared below), and the
statistic inherits that.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
X = ["x1", "x2", "x3"]
TOBIT_TESTS = ["normality", "heterosc", "skewness", "kurtosis"]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "cmtest_data.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((_FIX / "cmtest_micsr.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probit(data):
    return sp.probit(data=data, y="yb", x=X)


@pytest.fixture(scope="module")
def tobit(data):
    return sp.tobit(data, "yc", X)


def test_the_two_sides_test_the_same_fit(probit, tobit, ref):
    np.testing.assert_allclose(probit.params.values, ref["probit_coef"], atol=1e-7)
    np.testing.assert_allclose(tobit.params.values, ref["tobit_coef"], atol=1e-9)


@pytest.mark.parametrize("test", ["normality", "heterosc"])
def test_probit(test, probit, ref):
    out, want = sp.cmtest(probit, test), ref[f"probit_{test}"]
    assert out["df"] == want["df"]
    assert out["statistic"] == pytest.approx(want["statistic"], rel=1e-6)
    assert out["pvalue"] == pytest.approx(want["pvalue"], rel=1e-6)
    assert out["model"] == "probit"


@pytest.mark.parametrize("opg", [False, True])
@pytest.mark.parametrize("test", TOBIT_TESTS)
def test_tobit(test, opg, tobit, ref):
    out = sp.cmtest(tobit, test, opg=opg)
    want = ref[f"tobit_{test}" + ("_opg" if opg else "")]
    if test in ("skewness", "kurtosis"):
        # micsr prints the one-moment tests as a z statistic.
        assert out["df"] == 1
        assert abs(out["z"]) == pytest.approx(abs(want["statistic"]), rel=1e-9)
        assert out["statistic"] == pytest.approx(want["statistic"] ** 2, rel=1e-9)
    else:
        assert out["df"] == want["df"]
        assert out["statistic"] == pytest.approx(want["statistic"], rel=1e-9)
    assert out["pvalue"] == pytest.approx(want["pvalue"], rel=1e-7)


def test_heterosc_names_the_regressors(tobit, probit):
    assert sp.cmtest(tobit, "heterosc")["variables"] == X
    assert sp.cmtest(probit, "heterosc")["variables"] == X


def test_size_under_the_null():
    """With a normal homoskedastic error the tests reject at about 5%."""
    rng = np.random.default_rng(20261004)
    reject = {"normality": 0, "heterosc": 0}
    reps = 200
    for _ in range(reps):
        n = 400
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        ystar = 0.5 + x1 - 0.5 * x2 + rng.normal(size=n)
        df = pd.DataFrame({"y": np.maximum(ystar, 0.0), "x1": x1, "x2": x2})
        fit = sp.tobit(df, "y", ["x1", "x2"])
        for t in reject:
            reject[t] += sp.cmtest(fit, t)["pvalue"] < 0.05
    for t, k in reject.items():
        # 5% nominal; 200 replications put a correct test inside [1%, 10.5%].
        assert 0.01 <= k / reps <= 0.105, (t, k / reps)


def test_power_against_heteroskedasticity():
    rng = np.random.default_rng(7)
    n = 1500
    x1 = rng.normal(size=n)
    ystar = 0.5 + x1 + np.exp(0.6 * x1) * rng.normal(size=n)
    df = pd.DataFrame({"y": np.maximum(ystar, 0.0), "x1": x1})
    assert sp.cmtest(sp.tobit(df, "y", ["x1"]), "heterosc")["pvalue"] < 1e-6


def test_two_limit_and_upper_only_tobit_are_handled():
    """Censoring from above mirrors censoring from below."""
    rng = np.random.default_rng(3)
    n = 800
    x1 = rng.normal(size=n)
    ystar = 0.2 + x1 + rng.normal(size=n)
    low = pd.DataFrame({"y": np.maximum(ystar, 0.0), "x1": x1})
    # Reflect: -y* censored from above at 0 is the same problem.
    high = pd.DataFrame({"y": np.minimum(-ystar, 0.0), "x1": x1})
    a = sp.cmtest(sp.tobit(low, "y", ["x1"], ll=0), "normality")
    b = sp.cmtest(sp.tobit(high, "y", ["x1"], ll=-np.inf, ul=0), "normality")
    assert a["statistic"] == pytest.approx(b["statistic"], rel=1e-6)
    s = sp.cmtest(sp.tobit(low, "y", ["x1"], ll=0), "skewness")
    t = sp.cmtest(sp.tobit(high, "y", ["x1"], ll=-np.inf, ul=0), "skewness")
    assert s["z"] == pytest.approx(-t["z"], rel=1e-6)


def test_refusals(data, probit):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="sp.probit or sp.tobit"):
        sp.cmtest(sp.logit(data=data, y="yb", x=X))
    with pytest.raises(bad, match="not available"):
        sp.cmtest(probit, test="reset")
    w = data.assign(w=1.0 + (data["x2"] > 0))
    with pytest.raises(bad, match="sp.probit or sp.tobit|weights"):
        sp.cmtest(sp.tobit(w, "yc", X, weights="w"))
