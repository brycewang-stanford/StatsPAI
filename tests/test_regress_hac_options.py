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


# ------------------------------------------------------------------ EWC
# The equal-weighted cosine estimator of Lazarus, Lewis, Stock and Watson
# (2018): eq. (10) for the estimator, eq. (4) for the number of terms,
# eq. (14) for the F statistic. No other package ships it, so the evidence
# is the definition computed by an independent route, an exact identity,
# and the rejection rate on a design where the null is true.
def _ar1_pair(rng, n, rho):
    x, e = np.zeros(n), np.zeros(n)
    ex, ee = rng.normal(size=n), rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + ex[t]
        e[t] = rho * e[t - 1] + ee[t]
    return x, e


def test_ewc_covariance_is_the_definition(df):
    nu = 7
    res = sp.regress("y ~ x", df, robust="ewc", ewc_df=nu)
    X = np.column_stack([np.ones(len(df)), df.x])
    u = df.y.to_numpy() - X @ np.linalg.solve(X.T @ X, X.T @ df.y.to_numpy())
    T = len(df)
    omega = np.zeros((2, 2))
    for j in range(1, nu + 1):
        lam = np.zeros(2)
        for t in range(1, T + 1):  # a plain double loop, on purpose
            lam += X[t - 1] * u[t - 1] * np.cos(np.pi * j * (t - 0.5) / T)
        lam *= np.sqrt(2 / T)
        omega += np.outer(lam, lam) / nu
    bread = np.linalg.inv(X.T @ X)
    want = np.sqrt(np.diag(bread @ (T * omega) @ bread))
    np.testing.assert_allclose(res.std_errors.to_numpy(), want, rtol=1e-10)
    assert res.model_info["ewc_df"] == nu


def test_ewc_with_every_cosine_is_hc0(df):
    # The T - 1 Type II cosines with j >= 1 span everything orthogonal to a
    # constant, and OLS scores sum to zero, so the average of all T - 1
    # outer products is sum(z z') / (T - 1): HC0 scaled by T / (T - 1).
    T = len(df)
    ewc = sp.regress("y ~ x", df, robust="ewc", ewc_df=T - 1)
    hc0 = sp.regress("y ~ x", df, robust="hc0")
    np.testing.assert_allclose(
        ewc.std_errors.to_numpy(),
        hc0.std_errors.to_numpy() * np.sqrt(T / (T - 1)),
        rtol=1e-10,
    )


def test_ewc_default_terms_and_t_reference(df):
    from scipy import stats

    res = sp.regress("y ~ x", df, robust="ewc")
    nu = int(np.floor(0.4 * len(df) ** (2 / 3)))  # 13 at T = 200
    assert res.model_info["ewc_df"] == nu == 13
    t = res.params["x"] / res.std_errors["x"]
    np.testing.assert_allclose(res.pvalues["x"], 2 * stats.t.sf(abs(t), nu), rtol=1e-12)
    # wider than the normal interval the same standard error would give
    lo, hi = res.conf_int().loc["x"]
    assert hi - lo > 2 * 1.96 * res.std_errors["x"]


def test_ewc_joint_test_uses_the_rescaled_f():
    from scipy import stats

    rng = np.random.default_rng(3)
    x, e = _ar1_pair(rng, 240, 0.5)
    data = pd.DataFrame({"x": x, "w": rng.normal(size=240), "y": 1 + 0.2 * x + e})
    res = sp.regress("y ~ x + w", data, robust="ewc", ewc_df=12)
    out = sp.test(res, "x w")
    b = res.params[["x", "w"]].to_numpy()
    V = res.cov_params().loc[["x", "w"], ["x", "w"]].to_numpy()
    wald = float(b @ np.linalg.solve(V, b))
    m, B = 2, 12
    np.testing.assert_allclose(out["chi2"], wald, rtol=1e-12)
    np.testing.assert_allclose(
        out["statistic"], wald * (B - m + 1) / (B * m), rtol=1e-12
    )
    assert out["df"] == (m, B - m + 1)
    np.testing.assert_allclose(
        out["pvalue"], stats.f.sf(out["statistic"], m, B - m + 1), rtol=1e-12
    )
    few = sp.regress("y ~ x + w", data, robust="ewc", ewc_df=1)
    with pytest.raises(MethodIncompatibility, match="cosine terms"):
        sp.test(few, "x w")


def test_ewc_holds_size_better_than_newey_west_under_persistence():
    # x and the error are AR(1) with rho = 0.7 and the true slope is zero.
    rng = np.random.default_rng(2018)
    reps, T = 400, 200
    m = int(np.ceil(0.75 * T ** (1 / 3)))  # the textbook truncation, 5
    reject = {"hc1": 0, "nw": 0, "ewc": 0}
    for _ in range(reps):
        x, e = _ar1_pair(rng, T, 0.7)
        data = pd.DataFrame({"x": x, "y": e})
        reject["hc1"] += sp.regress("y ~ x", data, robust="hc1").pvalues["x"] < 0.05
        reject["nw"] += (
            sp.regress("y ~ x", data, robust="hac", hac_lags=m - 1).pvalues["x"] < 0.05
        )
        reject["ewc"] += sp.regress("y ~ x", data, robust="ewc").pvalues["x"] < 0.05
    rate = {k: v / reps for k, v in reject.items()}
    # nominal 5%; binomial SE at 400 draws is about 0.011 to 0.022
    assert rate["hc1"] > 0.18
    assert rate["ewc"] < rate["nw"] < rate["hc1"]
    assert rate["ewc"] < 0.11


def test_ewc_options_are_checked(df):
    with pytest.raises(MethodIncompatibility, match="only applies to robust='ewc'"):
        sp.regress("y ~ x", df, robust="hac", ewc_df=8)
    with pytest.raises(MethodIncompatibility, match="positive integer"):
        sp.regress("y ~ x", df, robust="ewc", ewc_df=0)
    with pytest.raises(DataInsufficient):
        sp.regress("y ~ x", df, robust="ewc", ewc_df=len(df))
    clustered = df.assign(g=np.arange(len(df)) // 10)
    with pytest.raises(MethodIncompatibility, match="cannot be combined"):
        sp.regress("y ~ x", clustered, robust="ewc", cluster="g")
