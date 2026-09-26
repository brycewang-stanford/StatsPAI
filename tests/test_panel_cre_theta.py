"""The corrected CRE fit (``statspai.panel._cre``) against linearmodels.

On a design without unit-mean columns the correction is a no-op, so the
wrapper must reproduce ``RandomEffects`` attribute for attribute; with the
means added, theta must equal the plain random-effects theta (the means add
columns but no rank). Stata parity of the CRE fits themselves lives in
``tests/reference_parity/test_panel_ssc_stata_parity.py``.
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from linearmodels.panel import RandomEffects
from statsmodels.tools import add_constant

import statspai as sp
from statspai.panel._cre import fit_cre

_DATA = (
    pathlib.Path(__file__).parent
    / "reference_parity"
    / "_fixtures"
    / "panel_ssc_data.csv"
)


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_DATA).set_index(["id", "t"])


@pytest.mark.parametrize(
    "cov",
    [
        {"cov_type": "unadjusted"},
        {"cov_type": "robust"},
        {"cov_type": "clustered", "cluster_entity": True},
    ],
)
def test_reproduces_random_effects_without_mean_columns(panel, cov):
    X = add_constant(panel[["x1", "x2"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = RandomEffects(panel.y, X).fit(**cov)
        got = fit_cre(panel.y, X, [], cov)
    for attr in ("params", "std_errors", "pvalues", "tstats"):
        np.testing.assert_allclose(getattr(got, attr), getattr(ref, attr), rtol=1e-12)
    np.testing.assert_allclose(got.cov, ref.cov, rtol=1e-12)
    np.testing.assert_allclose(got.resids, ref.resids, atol=1e-12)
    np.testing.assert_allclose(got.fitted_values, ref.fitted_values, atol=1e-12)
    np.testing.assert_allclose(got.theta, ref.theta, rtol=1e-14)
    for attr in ("rsquared", "rsquared_within", "rsquared_between", "rsquared_overall"):
        assert getattr(got, attr) == pytest.approx(getattr(ref, attr), rel=1e-12)
    assert got.f_statistic.stat == pytest.approx(ref.f_statistic.stat, rel=1e-12)
    assert got.nobs == ref.nobs and got.df_resid == ref.df_resid


def test_mundlak_theta_is_the_plain_random_effects_theta(panel):
    df = panel.reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.panel(df, "y ~ x1 + x2", entity="id", time="t", method="mundlak")
        plain = RandomEffects(panel.y, add_constant(panel[["x1", "x2"]])).fit()
        naive = RandomEffects(
            panel.y,
            add_constant(
                panel[["x1", "x2"]].assign(
                    _mean_x1=panel.groupby(level=0).x1.transform("mean"),
                    _mean_x2=panel.groupby(level=0).x2.transform("mean"),
                )
            ),
        ).fit()
    np.testing.assert_allclose(res._lm_result.theta, plain.theta, rtol=1e-14)
    # linearmodels on the augmented design counts the mean columns and gets
    # a different theta: the defect the wrapper exists to fix.
    assert np.max(np.abs(naive.theta.values - plain.theta.values)) > 1e-4
    # The regressors' own coefficients are the FE estimates either way.
    fe = sp.panel(df, "y ~ x1 + x2", entity="id", time="t", method="fe")
    np.testing.assert_allclose(res.params[["x1", "x2"]], fe.params, rtol=1e-10)
