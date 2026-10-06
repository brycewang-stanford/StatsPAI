"""The local IV form behind ``sp.bayes_mte(mte_method='polynomial')``.

Known truth, no PyMC needed. In the Heckman-Vytlacil index model
``D = 1{p(Z) > U_D}`` the mean of the outcome given the propensity is

    E[Y | P = p] = E[Y_0] + int_0^p MTE(u) du,

so a polynomial MTE makes ``E[Y | p]`` linear in the integrated powers that
``statspai.bayes.mte.integrated_mte_powers`` returns. Least squares on
those regressors must then recover the MTE coefficients. The model
``Y = alpha + D * g(p)`` that ``sp.bayes_mte`` fitted before 1.39 must not,
once the untreated outcome is selected on.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from statspai.bayes.mte import integrated_mte_powers


@pytest.mark.parametrize("selection", ["uniform", "normal"])
def test_integrated_powers_are_the_integrals(selection):
    p = np.array([0.03, 0.2, 0.5, 0.77, 0.96])
    K = integrated_mte_powers(p, 4, selection)
    for i, pi in enumerate(p):
        for k in range(5):
            if selection == "uniform":
                exact = integrate.quad(lambda u: u**k, 0.0, pi)[0]
            else:
                exact = integrate.quad(
                    lambda t: t**k * norm.pdf(t), -np.inf, norm.ppf(pi)
                )[0]
            # quad on an infinite range is itself good to about 1e-8
            assert K[i, k] == pytest.approx(exact, abs=1e-7)


def _ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def test_local_iv_recovers_a_normal_selection_model():
    """MTE(v) = 1 - 0.8 v on the probit scale, E[Y_0] = 0.5."""
    rng = np.random.default_rng(0)
    n = 400_000
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    index = 0.2 + 0.8 * z
    d = (index > v).astype(float)
    p = norm.cdf(index)
    u0 = 0.5 * v + 0.8 * rng.normal(size=n)
    u1 = -0.3 * v + 0.8 * rng.normal(size=n)
    y = np.where(d == 1, 1.5 + u1, 0.5 + u0)
    X = np.column_stack([np.ones(n), integrated_mte_powers(p, 1, "normal")])
    alpha, b0, b1 = _ols(X, y)
    # standard errors here are about 0.006, 0.011 and 0.012
    assert alpha == pytest.approx(0.5, abs=0.03)
    assert b0 == pytest.approx(1.0, abs=0.05)
    assert b1 == pytest.approx(-0.8, abs=0.05)
    # the model fitted before 1.39: the slope is off by a factor of five
    old = _ols(np.column_stack([np.ones(n), d, d * norm.ppf(p)]), y)
    assert abs(old[2] - (-0.8)) > 0.5
    assert abs(old[1] - 1.0) > 0.08


def test_local_iv_recovers_a_uniform_scale_mte():
    """MTE(u) = 2 - 3 u, selection on levels, no normality anywhere."""
    rng = np.random.default_rng(1)
    n = 400_000
    z = rng.normal(size=n)
    u = rng.uniform(size=n)
    p = norm.cdf(0.2 + 0.8 * z)
    d = (u < p).astype(float)
    y0 = 1.0 * (u - 0.5) + 0.5 * rng.standard_t(5, size=n)
    y = np.where(d == 1, y0 + 2.0 - 3.0 * u, y0)
    X = np.column_stack([np.ones(n), integrated_mte_powers(p, 1, "uniform")])
    alpha, b0, b1 = _ols(X, y)
    assert alpha == pytest.approx(0.0, abs=0.03)
    assert b0 == pytest.approx(2.0, abs=0.08)
    assert b1 == pytest.approx(-3.0, abs=0.15)
    old = _ols(np.column_stack([np.ones(n), d, d * p]), y)
    assert abs(old[1] - 2.0) > 0.5 and abs(old[2] - (-3.0)) > 1.5
