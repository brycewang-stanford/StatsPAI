"""Tests for sp.smm (simulated method of moments).

Evidence, by kind:

* Identities. With standardised simulation draws (mean exactly 0, variance
  exactly 1) the simulated mean/second-moment pair equals its population
  value at every theta, so SMM on (E y, E y^2) returns the sample mean and
  the (biased) sample standard deviation, and its standard errors are
  exactly sp.gmm's for the same moments times sqrt(1 + 1/S).
* Approximation. With iid draws the Jacobian of the simulated moments
  differs from the population one by O(1 / sqrt(S N)), so the same SE ratio
  holds to that order.
* Known truth. An AR(1) estimated by SMM on autocovariances agrees with its
  (conditional) MLE within sampling error.
* Monte Carlo screen (seeded, S-class: reported, not graded). The size of
  the simulation-corrected J test at S = 1, and the over-rejection of the
  uncorrected one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

import statspai as sp
from statspai.exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
)

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


def test_identity_weight_same_as_optimal_when_exactly_identified():
    # exactly identified: estimator and sandwich do not depend on W
    a = sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], n_sim=5, seed=4, bounds=BOUNDS)
    b = sp.smm(
        _sim_iid, CONTRIB, [0.0, 1.0], n_sim=5, seed=4, bounds=BOUNDS, W="identity"
    )
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-6)
    assert b.diagnostics["J_stat"] is None and b.diagnostics["J_note"]


def test_supplied_covariance_equals_data_rows():
    a = sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], n_sim=5, seed=4, bounds=BOUNDS)
    cov = np.cov(CONTRIB, rowvar=False, ddof=0) / N  # centred Omega / N
    b = sp.smm(
        _sim_iid,
        CONTRIB.mean(axis=0),
        [0.0, 1.0],
        n_sim=5,
        seed=4,
        bounds=BOUNDS,
        data_moment_cov=cov,
        n_obs=N,
    )
    assert a.model_info["cov_source"] == "data"
    assert b.model_info["cov_source"] == "supplied"
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-10)


def test_simulated_covariance_route_is_two_step_and_close_to_data_route():
    contrib3 = np.column_stack([Y, Y**2, Y**3])

    def sim3(theta, rng):
        ys = theta[0] + theta[1] * rng.standard_normal(N)
        return np.column_stack([ys, ys**2, ys**3])

    a = sp.smm(sim3, contrib3, [0.0, 1.0], n_sim=10, seed=5, bounds=BOUNDS)
    b = sp.smm(
        sim3,
        contrib3.mean(axis=0),
        [0.0, 1.0],
        n_sim=10,
        seed=5,
        bounds=BOUNDS,
        n_obs=N,
    )
    assert b.model_info["cov_source"] == "simulated"
    assert b.model_info["weight_steps"] == 2
    assert b.model_info["step1_params"] is not None
    # Omega estimated from the data vs from the model at theta_hat: both
    # consistent; at N = 4000 the third-moment variance is estimated to a
    # few percent, so the SEs agree to 10%
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=0.10)
    np.testing.assert_allclose(a.params, b.params, atol=2 * a.std_errors.max())


def test_correction_uses_simulated_sample_size():
    def sim_double(theta, rng):
        ys = theta[0] + theta[1] * rng.standard_normal(2 * N)
        return np.column_stack([ys, ys**2])

    res = sp.smm(sim_double, CONTRIB, [0.0, 1.0], n_sim=4, seed=1, bounds=BOUNDS)
    # c = 1 + N / (S N_s) with N_s = 2N
    assert res.model_info["correction_factor"] == pytest.approx(1 + 1 / 8, abs=1e-15)
    assert res.model_info["sim_n_obs"] == 2 * N


def test_vector_simulator_assumes_equal_sample_size():
    def sim_vec(theta, rng):
        ys = theta[0] + theta[1] * rng.standard_normal(N)
        return np.array([ys.mean(), (ys**2).mean()])

    res = sp.smm(sim_vec, CONTRIB, [0.0, 1.0], n_sim=4, seed=1, bounds=BOUNDS)
    assert res.model_info["correction_factor"] == pytest.approx(1.25, abs=1e-15)
    assert res.model_info["sim_size_assumed_equal_to_data"] is True


def _ar_contrib(v):
    return np.column_stack([v[2:] ** 2, v[2:] * v[1:-1], v[2:] * v[:-2]])


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


def test_j_size_screen_with_and_without_simulation_correction():
    """Seeded screen (S-class), R = 200, S = 1, N = 500, one overidentifying
    restriction. Corrected J: nominal 5%, binomial sd 1.5 points, band
    [2%, 10%]. Uncorrected (c = 2 dropped, i.e. 2J): P(chi2(1) > 1.92) =
    16.6% expected (sd 2.6 points), must exceed 11%. Observed 6.0% and 17.0%
    (at R = 300, N = 1000: 5.7% and 19.0%)."""
    n = 500

    def contrib(v):
        return np.column_stack([v, v**2, v**3])

    def sim(theta, rng):
        return contrib(theta[0] + theta[1] * rng.standard_normal(n))

    J = []
    for r in range(200):
        y = np.random.default_rng(10_000 + r).normal(0.5, 1.5, n)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            res = sp.smm(
                sim,
                contrib(y),
                [y.mean(), y.std()],
                n_sim=1,
                seed=20_000 + r,
                bounds=BOUNDS,
            )
        J.append(res.diagnostics["J_stat"])
    J = np.array(J)
    crit = 3.841458820694124  # chi2(1) 95%
    assert 0.02 <= np.mean(J > crit) <= 0.10
    assert np.mean(2.0 * J > crit) > 0.11


def test_seed_none_records_entropy_and_reproduces():
    a = sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], n_sim=3, bounds=BOUNDS)
    ent = a.model_info["seed_entropy"]
    b = sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], n_sim=3, bounds=BOUNDS, seed=ent)
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())


def test_vcov_matches_standard_errors():
    res = sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], n_sim=5, seed=2, bounds=BOUNDS)
    V = res.vcov()
    np.testing.assert_allclose(np.sqrt(np.diag(V)), res.std_errors, rtol=1e-12)
    fit = res.diagnostics["moment_fit"]
    assert list(fit.columns) == ["data", "simulated", "gap", "gap_se"]


def test_parameter_at_bound_warns():
    with pytest.warns(ConvergenceWarning, match="at a bound"):
        res = sp.smm(
            _sim_iid,
            CONTRIB,
            [0.0, 1.0],
            n_sim=5,
            seed=2,
            bounds=[(None, None), (1e-6, 1.5)],  # true sigma = 2
            param_names=["mu", "sigma"],
        )
    assert res.diagnostics["at_bound"] == ["sigma"]
    assert res.params["sigma"] == pytest.approx(1.5, abs=1e-12)


def test_global_rng_simulator_is_rejected():
    def bad(theta, rng):
        ys = theta[0] + theta[1] * np.random.standard_normal(N)
        return np.column_stack([ys, ys**2])

    with pytest.raises(MethodIncompatibility, match="common random numbers"):
        sp.smm(bad, CONTRIB, [0.0, 1.0], n_sim=2, seed=0)


def test_step_function_simulator_is_caught():
    d = (Y > 0).astype(float)

    def freq(theta, rng):
        # a frequency simulator: piecewise constant in theta
        return (theta[0] + rng.standard_normal(N) > 0).astype(float)[:, None]

    with pytest.raises(NumericalInstability, match="rank deficient"):
        sp.smm(freq, d[:, None], [0.4], n_sim=2, seed=0, optimizer="Nelder-Mead")


def test_input_validation():
    with pytest.raises(IdentificationFailure):
        sp.smm(_sim_iid, CONTRIB[:, :1], [0.0, 1.0], seed=0)
    with pytest.raises(DataInsufficient):
        bad = CONTRIB.copy()
        bad[0, 0] = np.nan
        sp.smm(_sim_iid, bad, [0.0, 1.0], seed=0)
    with pytest.raises(MethodIncompatibility, match="positive definite"):
        sp.smm(
            _sim_iid,
            CONTRIB,
            [0.0, 1.0],
            seed=0,
            W=np.array([[1.0, 0], [0, -1.0]]),
        )
    with pytest.raises(MethodIncompatibility, match="cluster"):
        sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], seed=0, vcov="cluster")
    with pytest.raises(MethodIncompatibility, match="data sample size is not"):
        # simulated rows give N_s, but N is unknown
        sp.smm(_sim_iid, CONTRIB.mean(axis=0), [0.0, 1.0], seed=0)
    with pytest.raises(MethodIncompatibility, match="no data covariance"):
        sp.smm(
            lambda t, r: _sim_iid(t, r).mean(axis=0),
            CONTRIB.mean(axis=0),
            [0.0, 1.0],
            seed=0,
        )
    with pytest.raises(MethodIncompatibility, match="returned 3 moments"):
        sp.smm(lambda t, r: np.zeros(3), CONTRIB, [0.0, 1.0], seed=0)
    with pytest.raises(MethodIncompatibility, match="shape"):
        sp.smm(lambda t, r: np.zeros((2, 2, 2)), CONTRIB, [0.0, 1.0], seed=0)
    with pytest.raises(MethodIncompatibility, match="bounds"):
        sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], seed=0, optimizer="BFGS", bounds=BOUNDS)
    with pytest.raises(NumericalInstability, match="singular"):
        sp.smm(
            lambda t, r: np.column_stack([_sim_iid(t, r), _sim_iid(t, r)[:, 0]]),
            np.column_stack([CONTRIB, CONTRIB[:, 0]]),
            [0.0, 1.0],
            seed=0,
        )


def test_cluster_covariance_runs_and_is_recorded():
    cl = np.repeat(np.arange(N // 4), 4)
    res = sp.smm(
        _sim_iid,
        CONTRIB,
        [0.0, 1.0],
        n_sim=5,
        seed=2,
        bounds=BOUNDS,
        vcov="cluster",
        cluster=cl,
    )
    iid = sp.smm(_sim_iid, CONTRIB, [0.0, 1.0], n_sim=5, seed=2, bounds=BOUNDS)
    assert res.model_info["vcov"] == "cluster"
    # the rows are independent, so clustering in groups of 4 changes the SE
    # only by sampling noise of the covariance estimate (~sqrt(4/N) * few)
    np.testing.assert_allclose(res.std_errors, iid.std_errors, rtol=0.15)
