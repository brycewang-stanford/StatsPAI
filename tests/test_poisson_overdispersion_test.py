"""The Cameron-Trivedi overdispersion test reported by ``sp.poisson``.

The statistic is the t ratio of a no-intercept regression of
``((y - mu)^2 - y) / mu`` on ``mu``, so the reference is that regression
written out with statsmodels on the same fitted means. On the accident
counts of Das, *Causal Inference in R*, ch. 7 it is 3.9746864 with
alpha = 0.5461809, the values ``AER::dispersiontest(trafo = 2)`` prints
(3.974686, 0.5461809); see docs/dev/2026-10-06-das-causal-inference-in-r-review.md.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _data(seed: int, overdispersed: bool, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    mu = np.exp(0.8 + 0.5 * x)
    if overdispersed:
        mu = mu * rng.gamma(2.0, 0.5, n)
    return pd.DataFrame({"y": rng.poisson(mu), "x": x})


def _by_hand(d: pd.DataFrame):
    """The auxiliary regression, written out with statsmodels."""
    import statsmodels.api as sm

    fit = sm.GLM(d["y"], sm.add_constant(d[["x"]]), family=sm.families.Poisson()).fit()
    mu = fit.fittedvalues.to_numpy()
    aux = ((d["y"].to_numpy() - mu) ** 2 - d["y"].to_numpy()) / mu
    ols = sm.OLS(aux, mu).fit()
    pearson = float(np.sum((d["y"] - mu) ** 2 / mu)) / (len(d) - 2)
    return float(ols.tvalues[0]), float(ols.pvalues[0]), float(ols.params[0]), pearson


@pytest.mark.parametrize("overdispersed", [True, False])
def test_matches_the_auxiliary_regression(overdispersed):
    d = _data(11, overdispersed)
    diag = sp.poisson("y ~ x", data=d).diagnostics
    t, p, alpha, pearson = _by_hand(d)
    assert diag["Overdispersion test (C-T)"] == pytest.approx(t, rel=1e-8)
    assert diag["Overdispersion p-value"] == pytest.approx(p, rel=1e-7)
    assert diag["Overdispersion alpha (C-T)"] == pytest.approx(alpha, rel=1e-8)
    assert diag["Dispersion (Pearson chi2 / df)"] == pytest.approx(pearson, rel=1e-8)


def test_strong_overdispersion_is_rejected():
    """Gamma-mixed counts with Var = mu + 0.5 mu^2: alpha is recovered and
    the test rejects. The pre-1.39 statistic did neither reliably."""
    d = _data(5, overdispersed=True, n=2000)
    diag = sp.poisson("y ~ x", data=d).diagnostics
    assert diag["Overdispersion p-value"] < 1e-6
    assert diag["Overdispersion alpha (C-T)"] == pytest.approx(0.5, abs=0.12)
    assert diag["Dispersion (Pearson chi2 / df)"] > 1.5


def test_size_under_the_poisson():
    rejections = 0
    for seed in range(300):
        diag = sp.poisson("y ~ x", data=_data(seed, False, n=300)).diagnostics
        rejections += diag["Overdispersion p-value"] < 0.05
    # 300 draws of a 5% test: between 2% and 9% with probability > 0.99
    assert 0.02 <= rejections / 300 <= 0.09
