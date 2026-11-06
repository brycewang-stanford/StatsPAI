"""Closed-form evidence for ``sp.ibragimov_mueller`` (analytical-only).

No package implements the Ibragimov-Mueller group t-test as a command, so
there is no external reference to align with. The test is fully determined
by its definition: estimate the coefficient separately in each of the q
groups and apply the one-sample t-test with q - 1 degrees of freedom to
the q estimates. Here the group estimates are computed with numpy (least
squares and two-stage least squares written out, sharing no code with
StatsPAI) and the test with ``scipy.stats.ttest_1samp`` /
``scipy.stats.t``, on the committed ``_fixtures/ivreport_fx.csv`` (600
rows, 12 regions). Agreement is to 1e-10 (rounding).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def fx():
    return pd.read_csv(FX / "ivreport_fx.csv", float_precision="round_trip")


def _ols_slope(g: pd.DataFrame) -> float:
    X = np.column_stack([np.ones(len(g)), g[["x", "w1", "w2", "d"]].to_numpy()])
    return float(np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)[0][1])


def _tsls_slope(g: pd.DataFrame) -> float:
    exog = np.column_stack([np.ones(len(g)), g[["w1", "w2", "d"]].to_numpy()])
    X = np.column_stack([g["x"].to_numpy(), exog])
    Z = np.column_stack([g[["z", "z2"]].to_numpy(), exog])
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return float(np.linalg.solve(Xhat.T @ X, Xhat.T @ g["y"].to_numpy())[0])


def _reference(fx, slope, h0, alpha):
    est = np.array([slope(g) for _, g in fx.groupby("region")])
    q = len(est)
    t = stats.ttest_1samp(est, h0)
    se = est.std(ddof=1) / np.sqrt(q)
    crit = stats.t.ppf(1 - alpha / 2, q - 1)
    return (
        est,
        est.mean(),
        se,
        t.statistic,
        t.pvalue,
        (est.mean() - crit * se, est.mean() + crit * se),
    )


@pytest.mark.parametrize(
    "estimator, formula, slope",
    [
        ("ols", "y ~ x + w1 + w2 + d", _ols_slope),
        ("iv", "y ~ (x ~ z + z2) + w1 + w2 + d", _tsls_slope),
    ],
)
@pytest.mark.parametrize("h0, alpha", [(0.0, 0.05), (0.3, 0.05), (0.0, 0.01)])
def test_group_t_test_is_the_one_sample_t_test_of_group_estimates(
    fx, estimator, formula, slope, h0, alpha
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.ibragimov_mueller(
            fx, formula, cluster="region", coef="x", estimator=estimator,
            h0=h0, alpha=alpha,
        )  # fmt: skip
    est, mean, se, t, p, ci = _reference(fx, slope, h0, alpha)
    ours = np.sort(res.group_estimates["estimate"].to_numpy())
    np.testing.assert_allclose(ours, np.sort(est), rtol=1e-10)
    assert res.df == len(est) - 1 == 11
    assert res.estimate == pytest.approx(mean, rel=1e-10)
    assert res.se == pytest.approx(se, rel=1e-10)
    assert res.tstat == pytest.approx(t, rel=1e-10)
    assert res.pvalue == pytest.approx(p, rel=1e-10)
    np.testing.assert_allclose(res.ci, ci, rtol=1e-10)


def test_standard_error_options_do_not_change_the_test(fx):
    """Only the group point estimates enter: a clustered or robust fit in
    each group gives the same statistic."""
    base = sp.ibragimov_mueller(fx, "y ~ x + w1 + w2 + d", cluster="region", coef="x")
    rob = sp.ibragimov_mueller(
        fx, "y ~ x + w1 + w2 + d", cluster="region", coef="x",
        fit_kwargs={"vce": "robust"},
    )  # fmt: skip
    assert rob.tstat == base.tstat and rob.pvalue == base.pvalue
