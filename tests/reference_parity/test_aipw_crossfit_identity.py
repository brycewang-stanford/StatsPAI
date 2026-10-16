"""``sp.aipw``'s default (cross-fitted) fit, rebuilt by hand.

The default ``cross_fit=True`` cell had no reference. This test rebuilds it
independently: the same fold draw (``default_rng(seed).choice(n_folds, n)``,
the documented assignment), a statsmodels logit and per-arm OLS on each
training fold, the AIPW scores on the held-out fold, and the influence-
function SE ``sd(psi) / sqrt(n)``. Estimate and SE must match to 1e-9 for
ATE and ATT (ATT's influence function is ``psi - tau * D / p``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import statspai as sp


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 800
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 0.6 * x1 - 0.4 * x2)))).astype(
        int
    )
    y = 1 + 1.5 * d + x1 + 0.5 * x2 + 0.8 * d * x1 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def _by_hand(df, seed, n_folds, estimand):
    X = df[["x1", "x2"]].to_numpy()
    D = df["d"].to_numpy()
    Y = df["y"].to_numpy()
    n = len(df)
    folds = np.random.default_rng(seed).choice(n_folds, size=n)
    e, mu1, mu0 = np.zeros(n), np.zeros(n), np.zeros(n)
    Xc = sm.add_constant(X)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        e[te] = sm.Logit(D[tr], Xc[tr]).fit(disp=0).predict(Xc[te])
        for arm, out in ((1, mu1), (0, mu0)):
            m = tr & (D == arm)
            beta = np.linalg.lstsq(Xc[m], Y[m], rcond=None)[0]
            out[te] = Xc[te] @ beta
    e = np.clip(e, 0.01, 0.99)
    if estimand == "ATE":
        psi = mu1 + D * (Y - mu1) / e - (mu0 + (1 - D) * (Y - mu0) / (1 - e))
        tau = psi.mean()
        return tau, np.std(psi, ddof=1) / np.sqrt(n)
    p = D.mean()
    psi = D * (Y - mu0) / p - (1 - D) * e * (Y - mu0) / ((1 - e) * p)
    tau = psi.mean()
    return tau, np.std(psi - tau * D / p, ddof=1) / np.sqrt(n)


@pytest.mark.parametrize("estimand", ["ATE", "ATT"])
@pytest.mark.parametrize("seed,n_folds", [(3, 5), (42, 2)])
def test_cross_fitted_aipw_equals_the_hand_computation(data, estimand, seed, n_folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.aipw(
            data, y="y", treat="d", covariates=["x1", "x2"],
            estimand=estimand, seed=seed, n_folds=n_folds,
        )  # fmt: skip
    assert fit.model_info["cross_fit"] is True
    assert fit.model_info["n_propensity_clipped"] == 0
    tau, se = _by_hand(data, seed, n_folds, estimand)
    assert fit.estimate == pytest.approx(tau, rel=1e-9)
    assert fit.se == pytest.approx(se, rel=1e-9)
