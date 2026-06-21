"""Reference-alignment tests: StatsPAI's core regression estimators must match
the canonical Python references (statsmodels, linearmodels) to numerical
precision.

These are deliberately separate from ``tests/reference_parity/`` (the R/Stata
suite). statsmodels and linearmodels are already project dependencies, so
these run everywhere and pin OLS / robust-SE / logit / Poisson / 2SLS to their
established implementations — a fast guard against any silent numerical drift
in the regression core.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

sm_api = pytest.importorskip("statsmodels.formula.api")


@pytest.fixture
def reg_data():
    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    df["y"] = 1.0 + 2.0 * df["x1"] - 1.5 * df["x2"] + rng.randn(n)
    df["d"] = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(0.5 * df["x1"])))).astype(
        int
    )
    df["count"] = rng.poisson(np.exp(0.3 + 0.4 * df["x1"]))
    return df


def test_ols_matches_statsmodels(reg_data):
    sp_ols = sp.regress("y ~ x1 + x2", reg_data)
    sm_ols = sm_api.ols("y ~ x1 + x2", reg_data).fit()
    np.testing.assert_allclose(
        sp_ols.params.values, sm_ols.params.values, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        sp_ols.std_errors.values, sm_ols.bse.values, rtol=0, atol=1e-10
    )


def test_ols_hc1_robust_se_matches_statsmodels(reg_data):
    sp_r = sp.regress("y ~ x1 + x2", reg_data, robust="hc1")
    sm_r = sm_api.ols("y ~ x1 + x2", reg_data).fit(cov_type="HC1")
    np.testing.assert_allclose(
        sp_r.std_errors.values, sm_r.bse.values, rtol=0, atol=1e-10
    )


def test_logit_matches_statsmodels(reg_data):
    sp_lg = sp.logit("d ~ x1", reg_data)
    sm_lg = sm_api.logit("d ~ x1", reg_data).fit(disp=0)
    np.testing.assert_allclose(
        sp_lg.params.values, sm_lg.params.values, rtol=0, atol=1e-7
    )


def test_poisson_matches_statsmodels(reg_data):
    sp_po = sp.poisson("count ~ x1", reg_data)
    sm_po = sm_api.poisson("count ~ x1", reg_data).fit(disp=0)
    np.testing.assert_allclose(
        sp_po.params.values, sm_po.params.values, rtol=0, atol=1e-7
    )


def test_iv_2sls_matches_linearmodels():
    iv_mod = pytest.importorskip("linearmodels.iv")
    rng = np.random.RandomState(0)
    n = 800
    z = rng.randn(n)
    u = rng.randn(n)
    x = 0.8 * z + u + rng.randn(n) * 0.5
    y = 1.0 * x + 2.0 * u + rng.randn(n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})

    sp_iv = sp.ivreg("y ~ (x ~ z)", df)
    df2 = df.assign(const=1.0)
    lm = iv_mod.IV2SLS(df2["y"], df2[["const"]], df2[["x"]], df2[["z"]]).fit(
        cov_type="unadjusted"
    )
    assert float(sp_iv.params["x"]) == pytest.approx(float(lm.params["x"]), abs=1e-8)
