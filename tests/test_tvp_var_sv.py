"""TVP-VAR with stochastic volatility: each Gibbs block against an exact
computation, the whole sweep against the joint distribution, and the
public interface.

The sampler is stochastic, so the checks are of three kinds: identities
that hold draw by draw, Monte Carlo moments of one block against the
closed form (tolerances are multiples of the Monte Carlo standard error,
stated at each assert), and the joint-distribution test of Geweke (2004)
on a small model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.mcmc._core import rinvwishart
from statspai.mcmc.diagnostics import _spectrum0_ar
from statspai.timeseries import _tvp_var_sv_core as core
from statspai.timeseries.tvp_var_sv import TVPVARSVResult, tvp_var_sv

KERN = core.kernels()
MIX_Q, MIX_M, MIX_V = core.mixture()


# --------------------------------------------------------------------------
# Block 1 and 2: the Carter-Kohn draw
# --------------------------------------------------------------------------
def test_cholesky_of_singular_matrix() -> None:
    A = np.array([[4.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 9.0]])
    L = KERN["chol"](A)
    # exact arithmetic on small integers: the factor reproduces A
    assert np.allclose(L @ L.T, A, atol=1e-14)
    assert np.allclose(np.triu(L, 1), 0.0)


def test_carter_kohn_moments_match_the_kalman_smoother() -> None:
    rng = np.random.default_rng(3)
    T, n, m, N = 12, 2, 3, 8000
    Z = rng.normal(size=(T, n, m))
    R = np.zeros((T, n, n))
    for t in range(T):
        c = rng.normal(size=(n, n))
        R[t] = c @ c.T + 0.3 * np.eye(n)
    c = rng.normal(size=(m, m))
    Q = 0.2 * (c @ c.T) + 0.05 * np.eye(m)
    m0 = np.array([0.5, -1.0, 0.2])
    P0 = np.diag([1.0, 2.0, 0.5])
    y = rng.normal(size=(T, n)) * 2.0
    # the prior is on the state of the first date: no innovation before it
    Qt = np.repeat(Q[None], T, axis=0)
    Qt[0] = 0.0
    ref = sp.kalman_filter(y, F=np.eye(m), G=Z, Q=Qt, R=R, x0=m0, P0=P0)
    draws = np.empty((N, T, m))
    for i in range(N):
        draws[i] = KERN["ck"](rng, y, Z, R, Q, m0, P0)
    mean = draws.mean(axis=0)
    sd = np.sqrt(np.einsum("tii->ti", ref.smoothed_cov))
    # Monte Carlo error of a mean of N draws is sd / sqrt(N); 5 of them
    assert np.abs(mean - ref.smoothed_state).max() < 5.0 * sd.max() / np.sqrt(N)
    assert (np.abs(mean - ref.smoothed_state) < 5.0 * sd / np.sqrt(N)).all()
    for t in (0, 5, T - 1):
        cov = np.cov(draws[:, t].T)
        scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        # a sample covariance of N normal draws has relative standard
        # error below sqrt(2 / N) = 0.016; 5 of them
        assert np.abs(cov - ref.smoothed_cov[t]).max() / scale.max() < 0.08
    # the joint law, not only the marginals: lag-one cross covariance of
    # the first state, cov(x_t, x_{t+1}) = J_t S_{t+1} for the smoother
    t = 4
    Pf, Pp = ref.filtered_cov[t], ref.predicted_cov[t + 1]
    exact = (Pf @ np.linalg.inv(Pp) @ ref.smoothed_cov[t + 1])[0, 0]
    got = np.cov(draws[:, t, 0], draws[:, t + 1, 0])[0, 1]
    assert abs(got - exact) < 0.08 * sd[t, 0] * sd[t + 1, 0]


# --------------------------------------------------------------------------
# Block 3: mixture indicators
# --------------------------------------------------------------------------
def test_indicator_probabilities_are_bayes_rule() -> None:
    resid = np.array([-9.0, -3.0, -1.27, 0.0, 1.5])
    w = MIX_Q[None, :] * stats.norm.pdf(
        resid[:, None], loc=MIX_M[None, :], scale=np.sqrt(MIX_V)[None, :]
    )
    direct = w / w.sum(axis=1, keepdims=True)
    # same formula evaluated on the log scale: rounding only
    assert np.allclose(core.indicator_probs(resid), direct, rtol=1e-12, atol=1e-300)


def test_indicator_draws_follow_the_probabilities() -> None:
    rng = np.random.default_rng(8)
    N = 200000
    for value in (-6.0, -1.0, 1.0):
        s = np.zeros((N, 1), dtype=np.int64)
        KERN["draw_ind"](rng, np.full((N, 1), value), MIX_Q, MIX_M, MIX_V, s)
        freq = np.bincount(s[:, 0], minlength=7) / N
        p = core.indicator_probs(np.array(value))
        # binomial standard error sqrt(p (1 - p) / N); 5 of them
        assert (np.abs(freq - p) < 5.0 * np.sqrt(p * (1 - p) / N) + 1e-9).all()


# --------------------------------------------------------------------------
# Block 4: inverse-Wishart updates
# --------------------------------------------------------------------------
def test_iw_posterior_parameters() -> None:
    rng = np.random.default_rng(0)
    path = rng.normal(size=(9, 3)).cumsum(axis=0)
    scale, df = core.iw_posterior(2.0 * np.eye(3), 7.0, path)
    d = path[1:] - path[:-1]
    manual = 2.0 * np.eye(3) + sum(np.outer(v, v) for v in d)
    assert np.allclose(scale, manual, rtol=1e-13)  # same sum, other order
    assert df == 7.0 + 8


def test_inverse_wishart_draws_have_the_closed_form_moments() -> None:
    rng = np.random.default_rng(1)
    c = rng.normal(size=(3, 3))
    scale = c @ c.T + np.eye(3)
    df, N = 12.0, 8000
    acc = np.zeros((3, 3))
    acc_inv = np.zeros((3, 3))
    for _ in range(N):
        d = KERN["riw"](rng, df, scale)
        acc += d
        acc_inv += np.linalg.inv(d)
    # E[X] = scale / (df - p - 1) and E[X^{-1}] = df scale^{-1}; the
    # relative Monte Carlo error of a diagonal element at N = 8000 is
    # sqrt(2 / (df - p - 3) / N) = 0.65%, bound 4%
    assert np.allclose(acc / N, scale / (df - 4), rtol=0.04, atol=0.04)
    target = df * np.linalg.inv(scale)
    assert np.allclose(acc_inv / N, target, rtol=0.04, atol=0.04 * target.max())
    # and the sampler agrees with the one of statspai.mcmc
    ref = np.mean([rinvwishart(rng, df, scale) for _ in range(N // 2)], axis=0)
    assert np.allclose(acc / N, ref, rtol=0.06, atol=0.06)


def _tiny_prior(T: int) -> dict:
    m = 6
    return {
        "b0": np.array([0.3, 0.0, 0.0, 0.0, 0.3, 0.0]),
        "PB": 0.02 * np.eye(m),
        "a0": np.array([0.3]),
        "PA": np.array([[0.2]]),
        "h0": np.zeros(2),
        "PH": 0.5 * np.eye(2),
        "Qs": 0.01 * 12 * np.eye(m),
        "Qdf": 12.0,
        "Ss": np.array([[0.05 * 6]]),
        "Sdf": np.array([6.0]),
        "Ws": 0.1 * 8 * np.eye(2),
        "Wdf": 8.0,
    }


def _sweep(rng, Y, Z, st, pr, ordering=0, offset=0.0):  # type: ignore[no-untyped-def]
    return KERN["sweep"](
        rng,
        Y,
        Z,
        st["B"],
        st["a"],
        st["h"],
        st["s"],
        st["Q"],
        st["S"],
        st["W"],
        pr["b0"],
        pr["PB"],
        pr["a0"],
        pr["PA"],
        pr["h0"],
        pr["PH"],
        pr["Qs"],
        pr["Qdf"],
        pr["Ss"],
        pr["Sdf"],
        pr["Ws"],
        pr["Wdf"],
        MIX_Q,
        MIX_M,
        MIX_V,
        offset,
        ordering,
        False,
        1,
        1,
    )


def _prior_draw(rng, pr, T):  # type: ignore[no-untyped-def]
    """States and hyperparameters from the prior (K = 2, p = 1)."""
    Q = rinvwishart(rng, pr["Qdf"], pr["Qs"])
    S = rinvwishart(rng, pr["Sdf"][0], pr["Ss"])
    W = rinvwishart(rng, pr["Wdf"], pr["Ws"])
    B = np.zeros((T, 6))
    a = np.zeros((T, 1))
    h = np.zeros((T, 2))
    B[0] = pr["b0"] + np.linalg.cholesky(pr["PB"]) @ rng.standard_normal(6)
    a[0] = pr["a0"] + np.sqrt(pr["PA"][0, 0]) * rng.standard_normal()
    h[0] = pr["h0"] + np.linalg.cholesky(pr["PH"]) @ rng.standard_normal(2)
    LQ, LW = np.linalg.cholesky(Q), np.linalg.cholesky(W)
    for t in range(1, T):
        B[t] = B[t - 1] + LQ @ rng.standard_normal(6)
        a[t] = a[t - 1] + np.sqrt(S[0, 0]) * rng.standard_normal()
        h[t] = h[t - 1] + LW @ rng.standard_normal(2)
    return {"B": B, "a": a, "h": h, "Q": Q, "S": S, "W": W}


def _simulate(rng, st, s=None):  # type: ignore[no-untyped-def]
    """Data given the states. With indicators ``s`` the shocks come from
    the mixture model conditional on them (what a sampler that carries
    the indicators across sweeps conditions on)."""
    B, a, h = st["B"], st["a"], st["h"]
    T = B.shape[0]
    Y = np.zeros((T, 2))
    Z = np.zeros((T, 2, 6))
    e = rng.standard_normal((T, 2))
    if s is not None:
        sign = np.where(rng.random((T, 2)) < 0.5, -1.0, 1.0)
        e = sign * np.exp(0.5 * (MIX_M[s] + np.sqrt(MIX_V[s]) * e))
    prev = np.zeros(2)
    for t in range(T):
        x = np.array([prev[0], prev[1], 1.0])
        Z[t, 0, :3] = x
        Z[t, 1, 3:] = x
        u1 = np.exp(h[t, 0]) * e[t, 0]
        u2 = -a[t, 0] * u1 + np.exp(h[t, 1]) * e[t, 1]
        Y[t, 0] = B[t, :3] @ x + u1
        Y[t, 1] = B[t, 3:] @ x + u2
        prev = Y[t]
    return Y, Z


def test_sweep_feeds_the_increments_to_the_hyperparameter_draws() -> None:
    # with a million prior degrees of freedom an inverse-Wishart draw is
    # its mean up to a relative error of order sqrt(2 / df) = 0.0014, so
    # the sweep's Q, S and W must equal the closed-form posterior means
    # computed from the paths the same sweep returned
    rng = np.random.default_rng(5)
    T = 30
    pr = _tiny_prior(T)
    st = _prior_draw(rng, pr, T)
    Y, Z = _simulate(rng, st)
    big = 1e6
    pr.update(Qdf=big, Sdf=np.array([big]), Wdf=big)
    pr["Qs"] = pr["Qs"] * big
    pr["Ss"] = pr["Ss"] * big
    pr["Ws"] = pr["Ws"] * big
    st["s"] = np.zeros((T, 2), dtype=np.int64)
    _sweep(rng, Y, Z, st, pr)
    for path, scale, df, got in (
        (st["B"], pr["Qs"], big, st["Q"]),
        (st["a"], pr["Ss"], big, st["S"]),
        (st["h"], pr["Ws"], big, st["W"]),
    ):
        post_scale, post_df = core.iw_posterior(scale, df, path)
        p = post_scale.shape[0]
        mean = post_scale / (post_df - p - 1)
        assert post_df == big + T - 1
        assert np.allclose(got, mean, rtol=0.02, atol=0.02 * np.diag(mean).max())


# --------------------------------------------------------------------------
# The whole sweep: joint-distribution test
# --------------------------------------------------------------------------
def _functionals(st):  # type: ignore[no-untyped-def]
    B, a, h, Q, S, W = (st[k] for k in ("B", "a", "h", "Q", "S", "W"))
    return np.array(
        [
            h[-1, 0],
            h[-1, 1],
            h[-1, 0] ** 2,
            h[-1, 1] ** 2,
            h[0, 0],
            h[-1, 0] * h[-1, 1],
            a[-1, 0],
            a[-1, 0] ** 2,
            B[-1, 0],
            B[-1, 0] ** 2,
            B[-1, 4],
            B[-1, 2],
            np.log(Q[0, 0]),
            np.log(S[0, 0]),
            np.log(W[0, 0]),
            np.log(W[1, 1]),
            W[0, 1] / np.sqrt(W[0, 0] * W[1, 1]),
            (h[-1, 0] - h[0, 0]) ** 2,
        ]
    )


def _successive(ordering, N, seed, T=6):  # type: ignore[no-untyped-def]
    """Alternate data given parameters and one sweep given data.

    If the sweep leaves the posterior invariant, the parameters keep
    their prior distribution along this chain.
    """
    rng = np.random.default_rng(seed)
    pr = _tiny_prior(T)
    st = _prior_draw(rng, pr, T)
    st["s"] = rng.choice(7, size=(T, 2), p=MIX_Q / MIX_Q.sum()).astype(np.int64)
    out = np.zeros((N, 18))
    for it in range(N):
        Y, Z = _simulate(rng, st, st["s"] if ordering == 1 else None)
        if not np.isfinite(Y).all() or np.abs(st["h"]).max() > 15.0:
            return out[:it], True
        _sweep(rng, Y, Z, st, pr, ordering=ordering)
        out[it] = _functionals(st)
    return out, False


def test_joint_distribution_test_passes_with_the_corrected_ordering() -> None:
    N, T = 6000, 6
    rng = np.random.default_rng(99)
    pr = _tiny_prior(T)
    marginal = np.array([_functionals(_prior_draw(rng, pr, T)) for _ in range(N)])
    chain, diverged = _successive(0, N, seed=5)
    assert not diverged
    se = np.sqrt(
        marginal.var(axis=0) / N
        + np.array([_spectrum0_ar(chain[:, j]) for j in range(chain.shape[1])]) / N
    )
    z = (chain.mean(axis=0) - marginal.mean(axis=0)) / se
    # 18 functionals, each z is N(0, 1) if the sweep is right: |z| < 4
    # has probability 0.999 jointly. The seven-normal approximation adds a
    # bias that is far below the Monte Carlo error at this N (runs of
    # 100,000 sweeps, three seeds, gave max |z| 2.9, 2.2 and 3.0).
    assert np.abs(z).max() < 4.0, z


def test_joint_distribution_test_rejects_the_original_ordering() -> None:
    # the 2005 order keeps the indicators across the coefficient and
    # covariance steps. Its chain leaves the prior at once: the log
    # volatilities run off (a prior standard deviation is about 1, the
    # stop is at 15) within a few hundred sweeps on every seed tried.
    for seed in (5, 6, 7):
        chain, diverged = _successive(1, 3000, seed=seed)
        assert diverged, f"seed {seed}: still inside the prior after 3000 sweeps"
        assert len(chain) < 3000


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------
def _data(seed: int = 0, n: int = 130, K: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros((n, K))
    for t in range(1, n):
        sd = 1.0 if t < 85 else 2.0
        y[t] = 0.4 * y[t - 1] + rng.normal(size=K)
        y[t, 0] += (sd - 1.0) * rng.normal()
        y[t, 1] += 0.5 * y[t, 0]
    return pd.DataFrame(y, columns=[f"v{i + 1}" for i in range(K)])


@pytest.fixture(scope="module")
def fit() -> TVPVARSVResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        return tvp_var_sv(_data(), lags=1, training=40, draws=300, burnin=200, seed=3)


def test_shapes_and_labels(fit: TVPVARSVResult) -> None:
    assert fit.coef_draws.shape == (300, 90, 2, 3)
    assert fit.a_draws.shape == (300, 90, 1)
    assert fit.logsig_draws.shape == (300, 90, 2)
    assert fit.terms == ["L1.v1", "L1.v2", "_cons"]
    assert fit.index[0] == 40 and fit.n_obs == 90
    assert list(fit.coefficients().columns) == [
        "date",
        "equation",
        "term",
        "median",
        "mean",
        "lower",
        "upper",
    ]
    vol = fit.volatility()
    assert (vol["lower"] <= vol["median"]).all()
    assert (vol["median"] <= vol["upper"]).all()
    assert (vol["lower"] > 0).all()


def test_same_seed_same_draws() -> None:
    kw = dict(lags=1, training=40, draws=40, burnin=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        a = tvp_var_sv(_data(), seed=11, **kw)
        b = tvp_var_sv(_data(), seed=11, **kw)
        c = tvp_var_sv(_data(), seed=12, **kw)
    assert np.array_equal(a.coef_draws, b.coef_draws)
    assert np.array_equal(a.logsig_draws, b.logsig_draws)
    assert not np.array_equal(a.logsig_draws, c.logsig_draws)


def test_covariance_is_built_from_the_states(fit: TVPVARSVResult) -> None:
    # draw by draw: Omega = A^{-1} diag(sigma^2) A^{-1}'
    d, t = 17, 33
    A = np.array([[1.0, 0.0], [float(fit.a_draws[d, t, 0]), 1.0]])
    sig = np.exp(fit.logsig_draws[d, t].astype(float))
    omega = np.linalg.inv(A) @ np.diag(sig**2) @ np.linalg.inv(A).T
    imp = fit._impact()[d, t]
    assert np.allclose(imp @ imp.T, omega, rtol=1e-12)  # algebra only
    corr = fit.covariance("corr")
    own = corr[corr["row"] == corr["col"]]
    assert np.allclose(own[["median", "lower", "upper"]], 1.0, atol=1e-12)
    sd = fit.covariance("sd")
    assert sd.shape[0] == 90 * 2
    # the first shock is the first reduced-form error
    v = fit.volatility()
    first = sd[sd["variable"] == "v1"]["median"].to_numpy()
    assert np.allclose(first, v[v["variable"] == "v1"]["median"].to_numpy(), rtol=1e-6)


def test_irf_of_one_draw_matches_the_recursion(fit: TVPVARSVResult) -> None:
    draws = fit.irf_draws(at=20, periods=4)
    d = 5
    Bm = fit.coef_draws[d, 20].astype(float)
    A1 = Bm[:, :2]
    imp = fit._impact(20)[d]
    for s in range(5):
        # VAR(1): Phi_s = A1^s; same floating-point products up to order
        assert np.allclose(
            draws[d, s], np.linalg.matrix_power(A1, s) @ imp, rtol=1e-10, atol=1e-14
        )
    unit = fit.irf_draws(at=20, periods=0, shock_size="unit")
    assert np.allclose(unit[:, 0, 0, 0], 1.0) and np.allclose(unit[:, 0, 1, 1], 1.0)
    assert np.allclose(unit[:, 0, 0, 1], 0.0)  # recursive: no impact upwards
    tab = fit.irf(at=[5, -1], periods=3, shock_size="unit", cumulative=True)
    assert len(tab) == 2 * 2 * 2 * 4
    assert set(tab["date"]) == {45, 129}
    cum = fit.irf_draws(at=20, periods=4, cumulative=True)
    assert np.allclose(cum[:, 2], draws[:, :3].sum(axis=1), rtol=1e-12)


def test_stability_diagnostics_summary_plot(fit: TVPVARSVResult) -> None:
    stab = fit.stability()
    assert ((stab["share_explosive"] >= 0) & (stab["share_explosive"] <= 1)).all()
    assert (stab["median_max_root"] > 0).all()
    diag = fit.diagnostics()
    assert {"ess", "inefficiency", "geweke_z", "geweke_p"} <= set(diag.columns)
    assert "tr(Q)" in diag.index and (diag["ess"] > 0).all()
    assert np.allclose(diag["inefficiency"], 300 / diag["ess"])
    text = fit.summary()
    assert "stochastic volatility" in text and "Convergence" in text
    out = fit.to_dict()
    assert out["n_draws"] == 300 and out["lags"] == 1
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for what in ("volatility", "coefficients", "correlation"):
        plt.close(fit.plot(what))
    with pytest.raises(MethodIncompatibility):
        fit.plot("nothing")


def test_three_variables_two_lags_and_user_prior() -> None:
    df = _data(seed=2, n=120, K=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        res = tvp_var_sv(
            df,
            lags=2,
            training=None,
            prior={"b0_var": 0.5, "logsig0": [0.0, 0.0, 0.0], "W_df": 6.0},
            draws=60,
            burnin=40,
            seed=1,
        )
    assert res.coef_draws.shape == (60, 118, 3, 7)
    assert res.a_draws.shape[2] == 3 and res.s_draws.shape[1:] == (3, 3)
    # S is block diagonal: equation 2 has one state, equation 3 has two
    assert np.all(res.s_draws[:, 0, 1:] == 0.0)
    assert np.all(res.s_draws[:, 1, 2] != 0.0)
    assert res.model_info["prior"]["W_df"] == 6.0
    assert np.allclose(res.model_info["prior"]["b0_var"], 0.5 * np.eye(21))


def test_stationary_option_rejects_explosive_paths() -> None:
    rng = np.random.default_rng(4)
    n = 110
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t, 0] = 0.97 * y[t - 1, 0] + rng.normal()
        y[t, 1] = 0.5 * y[t - 1, 1] + rng.normal()
    df = pd.DataFrame(y, columns=["a", "b"])
    kw = dict(lags=1, training=40, draws=150, burnin=50, seed=2, k_Q=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        free = tvp_var_sv(df, **kw)
        kept = tvp_var_sv(df, stationary=True, **kw)
    assert kept.stability(max_draws=150).attrs["share_any"] == 0.0
    assert free.stability(max_draws=150).attrs["share_any"] > 0.0
    assert any("stationary=True" in note for note in kept.model_info["notes"])


def test_errors() -> None:
    df = _data()
    with pytest.raises(MethodIncompatibility):
        tvp_var_sv(df[["v1"]], draws=10, burnin=0)
    with pytest.raises(MethodIncompatibility):
        tvp_var_sv(df.assign(v1=np.nan), draws=10, burnin=0)
    with pytest.raises(DataInsufficient):
        tvp_var_sv(df, lags=4, training=12, draws=10, burnin=0)
    with pytest.raises(DataInsufficient):
        tvp_var_sv(df.iloc[:44], training=40, draws=10, burnin=0)
    with pytest.raises(MethodIncompatibility):
        tvp_var_sv(df, prior={"nonsense": 1.0}, draws=10, burnin=0)
    with pytest.raises(MethodIncompatibility):
        tvp_var_sv(df, prior={"Q_df": 2.0}, draws=10, burnin=0)
    with pytest.raises(MethodIncompatibility):
        tvp_var_sv(df, alpha=1.5, draws=10, burnin=0)
    with pytest.raises(MethodIncompatibility):
        tvp_var_sv(df, k_Q=0.0, draws=10, burnin=0)
    with pytest.raises(MethodIncompatibility):
        tvp_var_sv(df, draws=0)


def test_result_errors(fit: TVPVARSVResult) -> None:
    with pytest.raises(MethodIncompatibility):
        fit.irf(at=500)
    with pytest.raises(MethodIncompatibility):
        fit.irf(at=3, shock_size="big")
    with pytest.raises(MethodIncompatibility):
        fit.covariance("precision")
    with pytest.raises(MethodIncompatibility):
        fit.volatility(alpha=0.0)


def test_slow_chain_warns() -> None:
    with pytest.warns(sp.ConvergenceWarning, match="effective sample size"):
        tvp_var_sv(_data(), lags=1, training=40, draws=30, burnin=0, seed=1)


def test_offset_that_is_not_negligible_warns() -> None:
    # shocks of sd 0.01: variance 1e-4, far below 100 times the offset
    small = _data() * 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        with pytest.warns(sp.exceptions.StatsPAIWarning, match="offset"):
            tvp_var_sv(small, lags=1, training=40, draws=30, burnin=0, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", sp.exceptions.StatsPAIWarning)
            warnings.simplefilter("ignore", sp.ConvergenceWarning)
            res = tvp_var_sv(
                small, lags=1, training=40, draws=30, burnin=0, seed=1, offset=1e-8
            )
    assert res.model_info["offset"] == 1e-8
