"""sp.regress(robust='hac'): the lag length and the small-sample factor.

The Newey-West estimator has two conventions that software disagrees on:
how many autocovariances to use, and whether to scale by N/(N-K). Both are
arguments here. The references are the committed Track A goldens for module
51 (Stata ``newey y x, lag(4)`` and R ``sandwich::NeweyWest(lag = 4,
adjust = FALSE)``, same CSV bytes) and statsmodels.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

ROOT = Path(__file__).resolve().parent
ROWS = (("Intercept", "beta_intercept"), ("x", "beta_x"))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ROOT / "r_parity" / "data" / "51_newey.csv")


def _golden(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["statistic"]: row for row in payload["rows"]}


def test_hac_small_reproduces_stata_newey(df):
    stata = _golden(ROOT / "stata_parity" / "results" / "51_newey_Stata.json")
    res = sp.regress("y ~ x", df, robust="hac", hac_lags=4, hac_small=True)
    for name, key in ROWS:
        # same estimator, same factor: Stata's printed precision is the floor
        np.testing.assert_allclose(res.params[name], stata[key]["estimate"], rtol=1e-12)
        np.testing.assert_allclose(res.std_errors[name], stata[key]["se"], rtol=1e-12)
    assert res.model_info["hac_lags"] == 4 and res.model_info["hac_small"] is True


def test_default_factor_reproduces_r_sandwich(df):
    r = _golden(ROOT / "r_parity" / "results" / "51_newey_R.json")
    res = sp.regress("y ~ x", df, robust="hac", hac_lags=4)
    for name, key in ROWS:
        np.testing.assert_allclose(res.std_errors[name], r[key]["se"], rtol=1e-12)
    # the two conventions differ by exactly sqrt(N / (N - K))
    small = sp.regress("y ~ x", df, robust="hac", hac_lags=4, hac_small=True)
    np.testing.assert_allclose(
        small.std_errors / res.std_errors, np.sqrt(len(df) / (len(df) - 2)), rtol=1e-13
    )


def test_default_lag_rule_is_unchanged(df):
    default = sp.regress("y ~ x", df, robust="hac")
    rule = int(np.floor(4 * (len(df) / 100) ** (2 / 9)))
    assert default.model_info["hac_lags"] == rule == 4
    explicit = sp.regress("y ~ x", df, robust="hac", hac_lags=rule)
    np.testing.assert_array_equal(default.std_errors, explicit.std_errors)


@pytest.mark.parametrize("lags", [0, 1, 3, 9])
@pytest.mark.parametrize("small", [False, True])
def test_any_lag_matches_statsmodels(df, lags, small):
    smf = pytest.importorskip("statsmodels.formula.api")
    ref = smf.ols("y ~ x", df).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": small}
    )
    res = sp.regress("y ~ x", df, robust="hac", hac_lags=lags, hac_small=small)
    np.testing.assert_allclose(res.std_errors["x"], ref.bse["x"], rtol=1e-12)


def test_zero_lags_is_hc0(df):
    hac0 = sp.regress("y ~ x", df, robust="hac", hac_lags=0)
    hc0 = sp.regress("y ~ x", df, robust="hc0")
    np.testing.assert_allclose(hac0.std_errors, hc0.std_errors, rtol=1e-13)


def test_more_lags_widen_the_interval_under_positive_autocorrelation():
    # x and the error are both AR(1) with rho = 0.8: the long-run variance of
    # x*e is (1 + 0.64) / (1 - 0.64) = 4.6 times its variance, so the HAC
    # standard error should approach sqrt(4.6) = 2.1 times the HC0 one.
    rng = np.random.default_rng(4)
    n = 4000
    x, e = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal()
        e[t] = 0.8 * e[t - 1] + rng.normal()
    data = pd.DataFrame({"x": x, "y": 1 + 0.5 * x + e})
    se = [
        sp.regress("y ~ x", data, robust="hac", hac_lags=L).std_errors["x"]
        for L in (0, 5, 40)
    ]
    assert se[0] < se[1] < se[2]
    assert 1.7 < se[2] / se[0] < 2.4


def test_options_are_refused_without_hac(df):
    with pytest.raises(MethodIncompatibility, match="only apply to robust='hac'"):
        sp.regress("y ~ x", df, hac_lags=4)
    with pytest.raises(MethodIncompatibility, match="only apply to robust='hac'"):
        sp.regress("y ~ x", df, robust="hc1", hac_small=True)
    with pytest.raises(MethodIncompatibility, match="non-negative integer"):
        sp.regress("y ~ x", df, robust="hac", hac_lags=-1)
    with pytest.raises(MethodIncompatibility, match="non-negative integer"):
        sp.regress("y ~ x", df, robust="hac", hac_lags=2.5)
    with pytest.raises(DataInsufficient):
        sp.regress("y ~ x", df, robust="hac", hac_lags=len(df))


def test_stata_newey_translates_to_the_same_call(df):
    out = sp.from_stata("newey y x, lag(4)")
    assert out["ok"] and out["untranslated_options"] == [], out
    assert out["arguments"] == {
        "formula": "y ~ x",
        "robust": "hac",
        "hac_lags": 4,
        "hac_small": True,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = sp.stata("newey y x, lag(4)", data=df)
    stata = _golden(ROOT / "stata_parity" / "results" / "51_newey_Stata.json")
    np.testing.assert_allclose(got.std_errors["x"], stata["beta_x"]["se"], rtol=1e-12)
    assert sp.from_stata("newey y x")["ok"] is False
    assert sp.from_stata("newey y x, lag(4) force")["untranslated_options"] == ["force"]
