"""Sprint-5 tests: OPE + CEVAE."""
import numpy as np
import pytest

import statspai as sp


# ---------- OPE ----------

def _bandit_logging(n=2000, K=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 2))
    # behaviour policy: soft action preferences driven by X[:,0]
    logits_b = np.column_stack([np.zeros(n), X[:, 0], -X[:, 0]])
    exp_b = np.exp(logits_b - logits_b.max(axis=1, keepdims=True))
    pi_b = exp_b / exp_b.sum(axis=1, keepdims=True)
    actions = np.array([rng.choice(K, p=pi_b[i]) for i in range(n)])
    # true reward: depends on X[:,1] and action
    true_means = np.column_stack([
        1.0 + X[:, 1],
        0.5 * X[:, 0],
        -0.2 + 0.5 * X[:, 1],
    ])
    rewards = true_means[np.arange(n), actions] + rng.normal(0, 0.5, n)
    return X, actions, rewards, pi_b, true_means


def test_ips_close_to_true_value():
    X, a, r, pi_b, true_means = _bandit_logging(n=3000)
    # evaluation policy: always action 0
    pi_e = np.zeros_like(pi_b)
    pi_e[:, 0] = 1.0
    res = sp.ope.ips(a, r, pi_b, pi_e)
    true_val = true_means[:, 0].mean()  # ≈ 1.0 since E[X1]=0
    assert abs(res.value - true_val) < 0.3
    assert isinstance(res, sp.OPEResult)


def test_dr_returns_valid_estimate():
    X, a, r, pi_b, true_means = _bandit_logging(n=2000)
    pi_e = np.zeros_like(pi_b)
    pi_e[:, 0] = 1.0
    def rm(X_, k):
        return true_means[:, k][: len(X_)]
    res_dr = sp.ope.doubly_robust(X, a, r, pi_b, pi_e, reward_model=rm)
    # DR should be a finite, unbiased (in expectation) estimate close to 1.0
    assert np.isfinite(res_dr.value)
    assert abs(res_dr.value - true_means[:, 0].mean()) < 0.3
    assert res_dr.se > 0


def test_snips_self_normalized():
    X, a, r, pi_b, true_means = _bandit_logging(n=1500)
    pi_e = pi_b.copy()  # trivially equal policy -> value = mean(r)
    res = sp.ope.snips(a, r, pi_b, pi_e)
    assert abs(res.value - r.mean()) < 0.2


def test_direct_method_runs():
    X, a, r, pi_b, true_means = _bandit_logging(n=500)
    pi_e = np.full_like(pi_b, 1.0 / pi_b.shape[1])
    def rm(X_, k):
        return true_means[:, k][: len(X_)]
    res = sp.ope.direct_method(rm, X, pi_e)
    assert isinstance(res, sp.OPEResult)
    assert res.method == "DM"


def test_switch_dr_produces_valid_result():
    X, a, r, pi_b, true_means = _bandit_logging(n=1200)
    pi_e = np.zeros_like(pi_b)
    pi_e[:, 0] = 1.0
    def rm(X_, k):
        return true_means[:, k][: len(X_)]
    res = sp.ope.switch_dr(X, a, r, pi_b, pi_e, rm, tau=3.0)
    assert res.method == "Switch-DR"
    assert 0.0 <= res.diagnostics["switched_frac"] <= 1.0


def test_evaluate_dispatches_all_methods():
    X, a, r, pi_b, true_means = _bandit_logging(n=500)
    pi_e = pi_b.copy()
    def rm(X_, k): return true_means[:, k][: len(X_)]
    for m in ("DM", "IPS", "SNIPS", "DR", "Switch-DR"):
        kw = dict(X=X, actions=a, rewards=r, pi_b=pi_b, pi_e=pi_e, reward_model=rm)
        res = sp.ope.evaluate(m, **kw)
        assert res.method == m
        assert np.isfinite(res.value)


def test_evaluate_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown"):
        sp.ope.evaluate("XYZ", actions=np.array([0]), rewards=np.array([0.0]),
                        pi_b=np.array([[1.0]]), pi_e=np.array([[1.0]]))


# ---------- CEVAE ----------

def test_cevae_recovers_ate_in_simple_setting():
    rng = np.random.default_rng(0)
    n = 600
    z = rng.normal(0, 1, n)  # latent confounder
    x = np.column_stack([z + rng.normal(0, 0.2, n), z + rng.normal(0, 0.3, n)])
    t = (z + rng.normal(0, 0.5, n) > 0).astype(float)
    y = 2.0 * t + z + rng.normal(0, 0.1, n)

    res = sp.cevae(x, t, y, z_dim=1, n_epochs=80, seed=0)
    assert isinstance(res, sp.CEVAEResult)
    # ATE in truth is 2.0; small-sample numpy/torch VAE should land within 1.0
    assert abs(res.ate - 2.0) < 1.5
    assert res.ite.shape == (n,)
    assert "CEVAE" in res.summary()


def test_cevae_backend_reported():
    rng = np.random.default_rng(1)
    n = 100
    x = rng.normal(0, 1, (n, 2))
    t = rng.binomial(1, 0.5, n)
    y = t + rng.normal(0, 1, n)
    res = sp.cevae(x, t, y, z_dim=2, n_epochs=30, seed=0)
    assert res.backend in ("torch", "numpy")
