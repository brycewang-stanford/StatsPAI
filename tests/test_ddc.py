"""Tests for sp.ddc / sp.dgp_ddc (dynamic discrete choice)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    IdentificationFailure,
    MethodIncompatibility,
)
from statspai.structural.ddc._core import solve_value

TRUTH = np.array([10.0, 2.5])


@pytest.fixture(scope="module")
def big_panel():
    # 400,000 bus-months; the likelihood only sees the (state, action)
    # counts, so the sample size costs nothing at estimation time.
    return sp.dgp_ddc(n_units=2000, n_periods=200, seed=11)


@pytest.fixture(scope="module")
def small_panel():
    return sp.dgp_ddc(n_units=200, n_periods=200, seed=0)


def _fit(df, **kw):
    kw.setdefault("beta", 0.9999)
    kw.setdefault("n_states", 90)
    return sp.ddc(df, "replace", "mileage", id="bus", time="t", **kw)


# ----------------------------------------------------------------------
# Known-truth recovery (T1)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["nfxp", "ccp", "npl"])
def test_recovers_true_parameters(big_panel, method):
    r = _fit(big_panel, method=method)
    assert r.converged
    z = (r.params.to_numpy() - TRUTH) / r.se.to_numpy()
    # One draw of a correctly specified model: |z| < 4 fails with
    # probability ~6e-5 per parameter.
    assert np.all(np.abs(z) < 4), z
    # ... and the estimates are informative: SE below 3% of the truth
    # (observed: 1.2% for RC, 2.1% for theta_1).
    assert np.all(r.se.to_numpy() < 0.03 * TRUTH)


def test_first_stage_increments_recovered(big_panel):
    r = _fit(big_panel)
    p = np.array(r.transition_params["increment_probs"])
    # 398,000 multinomial draws: sd of each share < 0.001.
    np.testing.assert_allclose(p, [0.35, 0.60, 0.05], atol=0.005)


def test_model_ccp_close_to_true_ccp(big_panel):
    r = _fit(big_panel)
    true_p = np.asarray(big_panel.attrs["true_ccp"])[:, 1]
    visited = big_panel["mileage"].value_counts().index.to_numpy()
    # Replacement probabilities are below 0.06; an absolute error of 0.003
    # is ~ 2 sampling SE of the parameter-implied CCP at the busiest states.
    np.testing.assert_allclose(r.ccp.to_numpy()[visited, 1], true_p[visited], atol=3e-3)


# ----------------------------------------------------------------------
# Identities
# ----------------------------------------------------------------------
def test_npl_fixed_point_is_the_nfxp_mle(small_panel):
    nfxp = _fit(small_panel, method="nfxp")
    npl = _fit(small_panel, method="npl", tol=1e-11)
    # Aguirregabiria-Mira: the NPL fixed point solves the partial-likelihood
    # first-order conditions. Agreement is limited by the Bellman tolerance
    # (1e-11 relative on values of order 1e3).
    np.testing.assert_allclose(npl.params, nfxp.params, rtol=1e-7)
    np.testing.assert_allclose(npl.se, nfxp.se, rtol=1e-5)
    assert npl.loglik == pytest.approx(nfxp.loglik, abs=1e-8)


def test_nfxp_gradient_is_zero_and_likelihood_is_maximal(small_panel):
    r = _fit(small_panel)
    for k in range(2):
        for h in (1e-3, -1e-3):
            theta = r.params.to_numpy().copy()
            theta[k] += h
            u = r._phi @ theta
            _, _, P, _ = solve_value(u, r.transition, r.beta)
            counts = (
                small_panel.assign(j=small_panel["replace"])
                .groupby(["mileage", "j"])
                .size()
                .unstack(fill_value=0)
                .reindex(range(90), fill_value=0)
                .to_numpy()
            )
            ll = float(np.sum(counts[counts > 0] * np.log(P[counts > 0])))
            assert ll < r.loglik


def test_beta_zero_is_a_static_logit():
    df = sp.dgp_ddc(n_units=300, n_periods=100, beta=0.0, RC=3.0, theta_1=40.0, seed=3)
    r = sp.ddc(df, "replace", "mileage", id="bus", time="t", beta=0.0, n_states=90)
    # P(replace | s) = logistic(-RC + theta_1 * 0.001 * s): a logit of the
    # choice on (1, 0.001 s) with coefficients (-RC, theta_1).
    df = df.assign(xs=0.001 * df["mileage"])
    lg = sp.logit("replace ~ xs", data=df)
    b = np.asarray(lg.params)
    se = np.asarray(lg.std_errors)
    # Both are the same MLE found by different Newton iterations.
    np.testing.assert_allclose(r.params.to_numpy(), [-b[0], b[1]], rtol=1e-7)
    np.testing.assert_allclose(r.se.to_numpy(), se, rtol=1e-5)


def test_array_and_callable_utilities_match_builtin(small_panel):
    F = np.asarray(small_panel.attrs["transition"])
    base = _fit(small_panel, transition=F)
    x = 0.001 * np.arange(90)
    phi = np.zeros((90, 2, 2))
    phi[:, 0, 1] = -x
    phi[:, 1, 0] = -1.0
    arr = _fit(small_panel, utility=phi, transition=F, param_names=["RC", "theta_1"])
    fn = _fit(
        small_panel,
        utility=lambda s, j: [-1.0, 0.0] if j == 1 else [0.0, -0.001 * s],
        transition=F,
    )
    np.testing.assert_allclose(
        arr.params.to_numpy(), base.params.to_numpy(), rtol=1e-10
    )
    np.testing.assert_allclose(fn.params.to_numpy(), base.params.to_numpy(), rtol=1e-10)
    assert list(fn.params.index) == ["theta_0", "theta_1"]
    assert base.transition_params["mode"] == "supplied"


def test_frequency_transitions_reproduce_increment_model(big_panel):
    inc = _fit(big_panel, transition="increments")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)
        freq = _fit(big_panel, transition="estimate")
    # Same truth, two first stages: estimates within one SE of each other.
    assert np.all(np.abs(freq.params - inc.params) < inc.se)


def test_three_actions_general_model():
    rng = np.random.default_rng(5)
    S, J = 12, 3
    F = np.zeros((J, S, S))
    for j in range(J):
        F[j] = rng.dirichlet(np.ones(S), size=S)
    phi = np.zeros((S, J, 3))
    s = np.arange(S) / S
    phi[:, 1, 0] = 1.0
    phi[:, 2, 1] = 1.0
    phi[:, 1, 2] = s
    phi[:, 2, 2] = -s
    theta = np.array([-0.5, 0.3, 1.5])
    _, _, P, _ = solve_value(phi @ theta, F, 0.9)
    n, T = 3000, 30
    st = rng.integers(0, S, n)
    rows = []
    for t in range(T):
        a = (rng.random(n)[:, None] > np.cumsum(P[st], axis=1)).sum(axis=1)
        rows.append(pd.DataFrame({"i": np.arange(n), "t": t, "s": st, "a": a}))
        cum = np.cumsum(F[a, st], axis=1)
        st = (rng.random(n)[:, None] > cum).sum(axis=1).clip(max=S - 1)
    df = pd.concat(rows, ignore_index=True)
    for method in ("nfxp", "npl"):
        r = sp.ddc(df, "a", "s", id="i", time="t", utility=phi, beta=0.9, method=method)
        z = (r.params.to_numpy() - theta) / r.se.to_numpy()
        assert np.all(np.abs(z) < 4), (method, z)


def test_counterfactual_identity_and_stationarity(small_panel):
    r = _fit(small_panel)
    same = r.counterfactual()
    np.testing.assert_allclose(same["P_1_cf"], same["P_1_base"], atol=1e-12)
    np.testing.assert_allclose(
        same["stationary_cf"], same["stationary_base"], atol=1e-10
    )
    cf = r.counterfactual(params={"RC": 5.0})
    # The stationary distribution is invariant under the controlled chain.
    P = cf[["P_0_cf", "P_1_cf"]].to_numpy()
    T = np.einsum("sj,jst->st", P, r.transition)
    pi = cf["stationary_cf"].to_numpy()
    np.testing.assert_allclose(pi @ T, pi, atol=1e-10)
    lr = cf.attrs["long_run"]["counterfactual"]
    assert lr[1] == pytest.approx(float(pi @ P[:, 1]), abs=1e-12)


def test_counterfactual_utility_shift_equals_parameter_change(small_panel):
    r = _fit(small_panel)
    shift = np.zeros((90, 2))
    shift[:, 1] = 2.0  # a subsidy of 2 on replacing == RC lower by 2
    a = r.counterfactual(utility_shift=shift)
    b = r.counterfactual(params={"RC": r.params["RC"] - 2.0})
    np.testing.assert_allclose(a["P_1_cf"], b["P_1_cf"], rtol=1e-10)


def test_result_protocol(small_panel):
    r = _fit(small_panel, method="ccp")
    assert "Hotz-Miller" in r.summary()
    td = r.tidy()
    assert list(td["term"]) == ["RC", "theta_1"]
    ci = r.conf_int()
    assert np.all(ci["lower"] < r.params) and np.all(r.params < ci["upper"])
    d = r.to_dict()
    assert d["method"] == "ccp"
    assert r.ccp_first_stage is not None
    np.testing.assert_allclose(r.ccp.sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# Edge cases and failures
# ----------------------------------------------------------------------
def test_unvisited_states_are_reported(small_panel):
    r = _fit(small_panel.query("mileage < 70"), n_states=90)
    assert r.diagnostics["unvisited_states"] == list(range(70, 90))
    assert np.all(np.isfinite(r.params))
    with pytest.warns(AssumptionWarning):
        rf = _fit(small_panel, transition="estimate", n_states=95)
    assert rf.diagnostics["transition_rows_without_data"]


def test_npl_with_unvisited_states_still_matches_nfxp(small_panel):
    sub = small_panel.query("mileage < 70")
    a = _fit(sub, method="npl", tol=1e-11)
    b = _fit(sub, method="nfxp")
    np.testing.assert_allclose(a.params, b.params, rtol=1e-6)


@pytest.mark.parametrize("beta", [1.0, 1.2, -0.1])
def test_beta_outside_unit_interval_rejected(small_panel, beta):
    with pytest.raises(MethodIncompatibility):
        _fit(small_panel, beta=beta)


def test_single_action_rejected(small_panel):
    one = small_panel.query("replace == 0")
    with pytest.raises(MethodIncompatibility):
        _fit(one)


def test_unidentified_features_rejected(small_panel):
    phi = np.zeros((90, 2, 2))
    phi[:, :, 0] = 1.0  # same for both actions
    phi[:, 1, 1] = -1.0
    with pytest.raises(IdentificationFailure):
        _fit(small_panel, utility=phi)


def test_bad_inputs_rejected(small_panel):
    with pytest.raises(MethodIncompatibility):
        _fit(small_panel.assign(mileage=small_panel["mileage"] - 0.5))
    with pytest.raises(MethodIncompatibility):
        _fit(small_panel, n_states=50)  # codes up to 89
    with pytest.raises(MethodIncompatibility):
        _fit(small_panel, transition=np.ones((2, 90, 90)))
    with pytest.raises(MethodIncompatibility):
        sp.ddc(small_panel, "replace", "mileage", beta=0.9)  # no panel ids
    with pytest.raises(MethodIncompatibility):
        sp.dgp_ddc(increment_probs=(0.5, 0.6))


def test_missing_values_rejected(small_panel):
    bad = small_panel.astype({"mileage": float})
    bad.loc[3, "mileage"] = np.nan
    with pytest.raises(Exception, match="missing"):
        _fit(bad)


def test_dgp_is_reproducible():
    a = sp.dgp_ddc(n_units=20, n_periods=10, seed=4)
    b = sp.dgp_ddc(n_units=20, n_periods=10, seed=4)
    pd.testing.assert_frame_equal(a, b)
    P = np.asarray(a.attrs["true_ccp"])
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    # The truth satisfies the Bellman equation it was solved from.
    phi_u = np.zeros((90, 2))
    x = 0.001 * np.arange(90)
    phi_u[:, 0] = -2.5 * x
    phi_u[:, 1] = -10.0
    V, v, P2, _ = solve_value(phi_u, np.asarray(a.attrs["transition"]), 0.9999)
    np.testing.assert_allclose(P2, P, atol=1e-12)
    np.testing.assert_allclose(V, special.logsumexp(v, axis=1), rtol=1e-11)
