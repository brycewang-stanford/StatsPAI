"""Tests for sp.smm (simulated method of moments): contract and screens.

The known-truth evidence (the standardised-draw identity with sp.gmm, the
iid-draw SE ratio, the AR(1) agreement with the MLE, and the Monte Carlo
coverage of the simulation-corrected intervals) is in
``tests/reference_parity/test_smm_known_truth.py``. Kept here:

* Equivalences. Identity and optimal weights coincide when exactly
  identified; a supplied data-moment covariance equals the row route; the
  simulated-covariance route is two-step and close to the data route.
* Monte Carlo screen (seeded, S-class: reported, not graded). The size of
  the simulation-corrected J test at S = 1, and the over-rejection of the
  uncorrected one.
* Contract: correction factor bookkeeping, seeds, vcov, warnings and input
  validation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

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


def _sim_iid(theta, rng):
    ys = theta[0] + theta[1] * rng.standard_normal(N)
    return np.column_stack([ys, ys**2])


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
