"""``sp.smm``: known truths of the simulated method of moments.

No R or Stata command estimates SMM from a user simulator, so the evidence
is known truth (T1):

* Identity. With standardised simulation draws (mean exactly 0, variance
  exactly 1) the simulated mean / second-moment pair equals its population
  value at every theta, so SMM on (E y, E y^2) returns the sample mean and
  the (biased) sample standard deviation, and its standard errors are
  exactly ``sp.gmm``'s for the same moments times ``sqrt(1 + 1/S)``.
* Approximation. With iid draws the Jacobian of the simulated moments
  differs from the population one by O(1 / sqrt(S N)), so the same SE ratio
  holds to that order, and the estimate recovers the DGP.
* Known truth. An AR(1) estimated by SMM on autocovariances agrees with its
  (conditional) MLE within sampling error.
* Coverage. On a normal DGP with known (mu, sigma), the 95% intervals with
  the simulation correction cover at the nominal rate at S = 1, where the
  correction doubles the variance; the same intervals without it do not.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

import statspai as sp

N = 4000
_RNG = np.random.default_rng(1)
Y = _RNG.normal(1.0, 2.0, size=N)
CONTRIB = np.column_stack([Y, Y**2])
BOUNDS = [(None, None), (1e-6, None)]


def _sim_standardised(theta, rng):
    z = rng.standard_normal(N)
    z = (z - z.mean()) / z.std()
    ys = theta[0] + theta[1] * z
    return np.column_stack([ys, ys**2])


def _sim_iid(theta, rng):
    ys = theta[0] + theta[1] * rng.standard_normal(N)
    return np.column_stack([ys, ys**2])


def _gmm_mean_sd():
    return sp.gmm(
        lambda t, d: np.column_stack([Y - t[0], Y**2 - t[0] ** 2 - t[1] ** 2]),
        np.array([Y.mean(), Y.std()]),
    )


def _ar_contrib(v):
    return np.column_stack([v[2:] ** 2, v[2:] * v[1:-1], v[2:] * v[:-2]])


def _wilson(k: int, n: int, z: float = 2.5758293035489004):
    """Wilson score interval for a binomial proportion (default 99%)."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


@pytest.mark.parametrize("S", [1, 5, 20])
def test_standardised_draws_reproduce_gmm_times_sqrt_c(S):
    res = sp.smm(_sim_standardised, CONTRIB, [0.0, 1.0], n_sim=S, seed=3, bounds=BOUNDS)
    mu_hat = Y.mean()
    sd_hat = np.sqrt((Y**2).mean() - mu_hat**2)
    # exactly identified and the simulated moments equal their population
    # values: the estimate is the method-of-moments one (rounding only)
    np.testing.assert_allclose(res.params.to_numpy(), [mu_hat, sd_hat], atol=1e-10)
    assert res.model_info["correction_factor"] == pytest.approx(1 + 1 / S, abs=0)
    g = _gmm_mean_sd()
    ratio = res.std_errors.to_numpy() / g.std_errors.to_numpy()
    # identity up to the finite-difference Jacobians (central differences of
    # polynomials of degree <= 2 are exact; 1e-6 covers rounding)
    np.testing.assert_allclose(ratio, np.sqrt(1 + 1 / S), rtol=1e-6)
    assert res.diagnostics["converged"]
    assert (
        res.diagnostics["J_p"] is None
        and "Exactly identified" in res.diagnostics["J_note"]
    )


def test_iid_draws_se_ratio_approaches_sqrt_c():
    S = 50
    res = sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], n_sim=S, seed=3, bounds=BOUNDS)
    g = _gmm_mean_sd()
    ratio = res.std_errors.to_numpy() / g.std_errors.to_numpy()
    # D_sim - D_pop = O(1/sqrt(S N)) = O(2e-3) relative; 2% is ten times that
    np.testing.assert_allclose(ratio, np.sqrt(1 + 1 / S), rtol=0.02)
    # and the estimate recovers the truth (mu = 1, sigma = 2) within 3 SE
    z = (res.params.to_numpy() - [1.0, 2.0]) / res.std_errors.to_numpy()
    assert np.all(np.abs(z) < 3)


def test_ar1_smm_agrees_with_mle():
    T, rho, sig = 2000, 0.6, 1.0
    rng = np.random.default_rng(1)
    x = lfilter([sig], [1.0, -rho], rng.standard_normal(T + 200))[200:]

    def sim_ar(theta, r):
        return _ar_contrib(
            lfilter([theta[1]], [1.0, -theta[0]], r.standard_normal(T + 200))[200:]
        )

    res = sp.smm(
        sim_ar,
        _ar_contrib(x),
        [0.3, 0.5],
        n_sim=10,
        seed=7,
        vcov="hac",
        bounds=[(-0.99, 0.99), (1e-6, None)],
        param_names=["rho", "sigma"],
    )
    X, Yl = x[:-1], x[1:]
    rho_mle = X @ Yl / (X @ X)
    sig_mle = np.sqrt(np.mean((Yl - rho_mle * X) ** 2))
    se = res.std_errors.to_numpy()
    # both estimators are consistent; their difference is O_p(T^-1/2). One
    # SMM standard error is a seeded screen of that (observed 0.03 / 0.06 SE)
    assert abs(res.params["rho"] - rho_mle) < se[0]
    assert abs(res.params["sigma"] - sig_mle) < se[1]
    # and both are within 3 SE of the truth
    assert np.all(np.abs((res.params.to_numpy() - [rho, sig]) / se) < 3)
    assert res.model_info["hac_lags"] == int(np.floor(4 * (T / 100) ** (2 / 9)))
    assert res.diagnostics["J_df"] == 1 and 0 <= res.diagnostics["J_p"] <= 1


def test_simulation_corrected_intervals_cover_at_the_nominal_rate():
    """1,000 samples of n = 500 from N(0.5, 1.5^2), one simulated sample per
    evaluation (S = 1, so c = 2). The 95% Wald intervals must cover each
    true parameter at a rate whose 99% Wilson interval contains 0.95.
    Dropping the correction (SE / sqrt(2)) gives intervals of nominal level
    P(|Z| < 1.96 / sqrt 2) = 0.834, whose Wilson interval must exclude 0.95:
    the band has the power to see a wrong variance. Observed 2026-11-05:
    0.953 / 0.944 corrected, 0.844 / 0.831 uncorrected."""
    n, R = 500, 1000
    truth = np.array([0.5, 1.5])

    def contrib(v):
        return np.column_stack([v, v**2])

    def sim(theta, rng):
        return contrib(theta[0] + theta[1] * rng.standard_normal(n))

    z95 = 1.959963984540054
    hit = np.zeros((R, 2), dtype=bool)
    hit_raw = np.zeros((R, 2), dtype=bool)
    est = np.empty((R, 2))
    for r in range(R):
        y = np.random.default_rng(30_000 + r).normal(truth[0], truth[1], n)
        res = sp.smm(
            sim,
            contrib(y),
            [y.mean(), y.std()],
            n_sim=1,
            seed=40_000 + r,
            bounds=BOUNDS,
        )
        assert res.model_info["correction_factor"] == 2.0
        th, se = res.params.to_numpy(), res.std_errors.to_numpy()
        est[r] = th
        hit[r] = np.abs(th - truth) <= z95 * se
        hit_raw[r] = np.abs(th - truth) <= z95 * se / np.sqrt(2.0)
    for j in range(2):
        lo, hi = _wilson(int(hit[:, j].sum()), R)
        assert lo <= 0.95 <= hi, (j, hit[:, j].mean())
        lo, hi = _wilson(int(hit_raw[:, j].sum()), R)
        assert hi < 0.95, (j, hit_raw[:, j].mean())
    # Consistent: the mean estimate is within 4 Monte Carlo SEs of the truth.
    mc_se = est.std(axis=0, ddof=1) / np.sqrt(R)
    assert np.all(np.abs(est.mean(axis=0) - truth) < 4 * mc_se)
