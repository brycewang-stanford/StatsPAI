"""Bands around ``sp.irf``: bootstrap behaviour and argument checks."""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

A = np.array([[0.5, 0.2], [0.1, 0.4]])
P = np.array([[1.0, 0.0], [0.5, 0.8]])


def _simulate(seed, T=200):
    rng = np.random.default_rng(seed)
    y = np.zeros((T + 100, 2))
    for t in range(1, T + 100):
        y[t] = A @ y[t - 1] + P @ rng.normal(size=2)
    return pd.DataFrame(y[100:], columns=["a", "b"])


@pytest.fixture(scope="module")
def fit():
    return sp.var(_simulate(0, T=600), lags=1)


def test_no_ci_returns_the_old_dictionary(fit):
    assert sorted(sp.irf(fit, periods=3)) == ["irf", "periods"]


def test_bootstrap_is_reproducible_and_brackets_the_estimate(fit):
    one = sp.irf(fit, periods=6, ci="bootstrap", reps=300, seed=4)
    two = sp.irf(fit, periods=6, ci="bootstrap", reps=300, seed=4)
    for key in one["irf"]:
        np.testing.assert_array_equal(one["lower"][key], two["lower"][key])
        assert np.all(one["lower"][key] <= one["upper"][key])
    assert one["ci"]["reps"] == 300
    # the response the ordering sets to zero has a degenerate band
    assert one["upper"]["b -> a"][0] == 0.0


def test_bootstrap_spread_agrees_with_the_delta_method_in_a_large_sample(fit):
    asy = sp.irf(fit, periods=5, ci="asymptotic")
    boot = sp.irf(fit, periods=5, ci="bootstrap", reps=800, seed=1)
    for key in asy["irf"]:
        a, b = asy["se"][key][1:], boot["se"][key][1:]
        # T = 600 and 800 replications: the two agree within a fifth
        np.testing.assert_allclose(b, a, rtol=0.2)


def test_hall_reflects_efron_about_the_estimate(fit):
    efron = sp.irf(fit, periods=4, ci="bootstrap", reps=200, seed=2, boot="efron")
    hall = sp.irf(fit, periods=4, ci="bootstrap", reps=200, seed=2, boot="hall")
    for key, est in efron["irf"].items():
        np.testing.assert_allclose(
            hall["lower"][key], 2 * est - efron["upper"][key], atol=1e-12
        )
        np.testing.assert_allclose(
            hall["upper"][key], 2 * est - efron["lower"][key], atol=1e-12
        )


@pytest.mark.parametrize("ci", ["asymptotic", "bootstrap"])
def test_bands_cover_the_true_response_at_about_the_nominal_rate(ci):
    # 90% bands for the response of b to a one-s.d. shock in a at horizons
    # 1-3, over 150 samples of T = 200: coverage between 80% and 97%
    # (binomial s.e. of 0.9 over 150 draws is 0.025; the bootstrap is
    # slightly liberal at this sample size).
    truth = np.array([(np.linalg.matrix_power(A, h) @ P)[1, 0] for h in (1, 2, 3)])
    hits = np.zeros(3)
    n_mc = 150
    for s in range(n_mc):
        out = sp.irf(
            sp.var(_simulate(1000 + s), lags=1),
            periods=3,
            ci=ci,
            alpha=0.10,
            reps=99,
            seed=s,
        )
        lo, hi = out["lower"]["a -> b"][1:], out["upper"]["a -> b"][1:]
        hits += (lo <= truth) & (truth <= hi)
    coverage = hits / n_mc
    assert np.all(coverage > 0.80), coverage
    assert np.all(coverage < 0.97), coverage


def test_exogenous_and_trend_terms_survive_the_bootstrap():
    df = _simulate(3)
    df["x"] = np.random.default_rng(9).normal(size=len(df))
    fit = sp.var(df, variables=["a", "b"], lags=2, trend="ct", exog=["x"])
    out = sp.irf(fit, periods=3, ci="bootstrap", reps=50, seed=0)
    assert np.all(np.isfinite(out["se"]["a -> b"]))


def test_argument_errors(fit):
    with pytest.raises(MethodIncompatibility, match="ci="):
        sp.irf(fit, ci="bayes")
    with pytest.raises(MethodIncompatibility, match="reps"):
        sp.irf(fit, ci="bootstrap", reps=5)
    with pytest.raises(MethodIncompatibility, match="boot"):
        sp.irf(fit, ci="bootstrap", boot="bca")
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.irf(fit, ci="asymptotic", alpha=1.5)


# ---------------------------------------------------------------------------
# Structural VAR
# ---------------------------------------------------------------------------


def test_recursive_svar_bootstrap_equals_the_cholesky_bootstrap(fit):
    nan = np.nan
    chol = sp.svar(fit, B=[[nan, 0], [nan, nan]])
    structural = chol.irf(4, ci="bootstrap", reps=60, seed=7)
    reduced = sp.irf(fit, periods=4, ci="bootstrap", reps=60, seed=7)
    rows = structural[(structural.shock == "shock1") & (structural.response == "b")]
    # same bootstrap samples, same identification, two code paths
    np.testing.assert_allclose(rows["lower"], reduced["lower"]["a -> b"], atol=1e-8)
    np.testing.assert_allclose(rows["upper"], reduced["upper"]["a -> b"], atol=1e-8)
    np.testing.assert_allclose(rows["se"], reduced["se"]["a -> b"], atol=1e-8)


def test_long_run_svar_band_respects_the_restriction():
    nan = np.nan
    rng = np.random.default_rng(21)
    T = 400
    u = rng.normal(size=(T, 2))
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = np.array([[0.3, 0.1], [0.0, 0.5]]) @ y[t - 1] + P @ u[t]
    bq = sp.svar(sp.var(pd.DataFrame(y, columns=["dy", "u"]), lags=1),
                 long_run=[[nan, 0], [nan, nan]])  # fmt: skip
    out = bq.irf(60, cumulative=True, ci="bootstrap", reps=80, seed=3)
    last = out[(out.shock == "shock2") & (out.response == "dy") & (out.period == 60)]
    # the second shock has no long-run effect on the first variable in
    # every replicate, so the band collapses there
    assert abs(float(last["lower"].iloc[0])) < 1e-4
    assert abs(float(last["upper"].iloc[0])) < 1e-4
    own = out[(out.shock == "shock1") & (out.response == "dy") & (out.period == 60)]
    assert float(own["lower"].iloc[0]) < float(own["irf"].iloc[0])
    assert float(own["irf"].iloc[0]) < float(own["upper"].iloc[0])


def test_sign_restricted_svar_refuses_a_bootstrap_band(fit):
    res = sp.svar(fit, sign={"s": {"a": "+", "b": "+"}}, n_draws=20, seed=0)
    with pytest.raises(MethodIncompatibility, match="identified set"):
        res.irf(4, ci="bootstrap")
    with pytest.raises(MethodIncompatibility, match="ci="):
        sp.svar(fit, B=[[np.nan, 0], [np.nan, np.nan]]).irf(4, ci="asymptotic")


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------


def test_fevd_without_ci_keeps_its_columns(fit):
    assert list(fit.fevd(3).columns) == ["shock", "response", "period", "fevd"]


def test_fevd_standard_error_matches_a_numerical_delta_method(fit):
    # the analytic gradient against finite differences of the shares in
    # the coefficients and the residual covariance
    from statspai.timeseries import irf_bands
    from statspai.timeseries.svar import fevd_shares

    B, k, p = np.asarray(fit._B, float), 2, 1
    sigma = np.asarray(fit.sigma_u, float)
    xtx = np.asarray(fit._XtX_inv, float)
    se = irf_bands.fevd_se(B, sigma, xtx, fit.n_obs, k, p, 4)

    def shares(a_vec, s_vech):
        Bn = B.copy()
        Bn[: k * p, :] = a_vec.reshape(k, k * p, order="F").T
        S = np.array([[s_vech[0], s_vech[1]], [s_vech[1], s_vech[2]]])
        return fevd_shares(irf_bands.response_array(Bn, S, k, p, 4, True, False))

    a0 = B[: k * p, :].T.reshape(-1, order="F")
    s0 = np.array([sigma[0, 0], sigma[1, 0], sigma[1, 1]])
    x0 = np.concatenate([a0, s0])
    jac = []
    for i in range(x0.size):
        h = 1e-6 * max(abs(x0[i]), 1e-3)
        up, dn = x0.copy(), x0.copy()
        up[i] += h
        dn[i] -= h
        jac.append((shares(up[:4], up[4:]) - shares(dn[:4], dn[4:])) / (2 * h))
    jac = np.stack(jac, axis=-1)  # (h, i, j, parameter)
    D = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]], float)
    Dp = np.linalg.solve(D.T @ D, D.T)
    cov = np.zeros((7, 7))
    cov[:4, :4] = np.kron(xtx[: k * p, : k * p], sigma)
    cov[4:, 4:] = 2 * Dp @ np.kron(sigma, sigma) @ Dp.T / fit.n_obs
    numeric = np.sqrt(np.einsum("hijp,pq,hijq->hij", jac, cov, jac))
    np.testing.assert_allclose(se, numeric, rtol=1e-5, atol=1e-9)


def test_fevd_bands(fit):
    asy = fit.fevd(4, ci="asymptotic")
    boot = fit.fevd(4, ci="bootstrap", reps=200, seed=5)
    # bootstrap bands stay inside the unit interval
    assert boot["lower"].min() >= 0.0 and boot["upper"].max() <= 1.0
    inner = asy[(asy.period >= 2) & (asy.shock == "a") & (asy.response == "b")]
    other = boot[(boot.period >= 2) & (boot.shock == "a") & (boot.response == "b")]
    np.testing.assert_allclose(other["se"], inner["se"], rtol=0.25)
    # the first-ordered variable explains all of its own one-step error
    first = asy[(asy.shock == "a") & (asy.response == "a") & (asy.period == 1)]
    assert float(first["fevd"].iloc[0]) == pytest.approx(1.0)
    assert float(first["se"].iloc[0]) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(MethodIncompatibility, match="ci="):
        fit.fevd(3, ci="bayes")


# ---------------------------------------------------------------------------
# Bias-corrected bootstrap and structural variance decompositions
# ---------------------------------------------------------------------------

PERSISTENT = np.array([[0.92, 0.0], [0.15, 0.5]])


def _persistent(seed, T=60):
    rng = np.random.default_rng(seed)
    y = np.zeros((T + 100, 2))
    for t in range(1, T + 100):
        y[t] = PERSISTENT @ y[t - 1] + P @ rng.normal(size=2)
    return pd.DataFrame(y[100:], columns=["a", "b"])


def test_bias_correction_raises_the_persistence_estimate():
    # OLS underestimates a large autoregressive coefficient in a short
    # sample; the first bootstrap round measures that, and the corrected
    # coefficient moves up without leaving the stationary region.
    from statspai.timeseries import irf_bands

    lifted = 0
    for s in range(20):
        fit = sp.var(_persistent(300 + s), lags=1)
        A_hat = np.asarray(fit._B, float)[:2, :]
        first = [Bb[:2] for Bb, _ in irf_bands._draws(fit, False, 200, s, None)]
        bias = np.mean(first, axis=0) - A_hat
        corrected = irf_bands._bias_corrected(A_hat, bias, 2, 1)
        assert irf_bands._max_root(corrected, 2, 1) < 1.0
        lifted += corrected[0, 0] > A_hat[0, 0]
    assert lifted >= 18


def test_explosive_correction_is_shrunk_and_a_unit_root_left_alone():
    from statspai.timeseries import irf_bands

    A_hat = np.array([[0.97]])
    out = irf_bands._bias_corrected(A_hat, np.array([[-0.08]]), 1, 1)
    assert 0.97 < out[0, 0] < 1.0  # full correction would give 1.05
    unit = np.array([[1.01]])
    assert irf_bands._bias_corrected(unit, np.array([[-0.05]]), 1, 1) is unit


def test_kilian_band_covers_a_persistent_response_better_than_efron():
    # T = 60 and an own root of 0.92: the response of a to its own shock
    # at horizons 4-8. Nominal 90%. Over 100 samples the percentile band
    # undercovers and the bias-corrected one is closer to nominal (the
    # difference is well outside the binomial error of either rate).
    truth = np.array(
        [(np.linalg.matrix_power(PERSISTENT, h) @ P)[0, 0] for h in range(4, 9)]
    )
    hits = {"efron": 0.0, "kilian": 0.0}
    n_mc = 100
    for s in range(n_mc):
        fit = sp.var(_persistent(2000 + s), lags=1)
        for boot in hits:
            with warnings.catch_warnings():
                # the percentile bootstrap warns here, which is the point
                warnings.simplefilter("ignore")
                out = sp.irf(
                    fit,
                    periods=8,
                    ci="bootstrap",
                    alpha=0.10,
                    reps=199,
                    seed=s,
                    boot=boot,
                )
            lo, hi = out["lower"]["a -> a"][4:], out["upper"]["a -> a"][4:]
            hits[boot] += np.mean((lo <= truth) & (truth <= hi))
    efron, kilian = hits["efron"] / n_mc, hits["kilian"] / n_mc
    assert kilian > efron + 0.05, (efron, kilian)
    assert 0.80 < kilian < 0.97, (efron, kilian)


def test_kilian_is_reproducible_and_reaches_fevd_and_svar(fit):
    one = sp.irf(fit, periods=3, ci="bootstrap", reps=60, seed=1, boot="kilian")
    two = sp.irf(fit, periods=3, ci="bootstrap", reps=60, seed=1, boot="kilian")
    np.testing.assert_array_equal(one["lower"]["a -> b"], two["lower"]["a -> b"])
    assert one["ci"]["boot"] == "kilian"
    shares = fit.fevd(3, ci="bootstrap", reps=60, seed=1, boot="kilian")
    assert shares["lower"].min() >= 0 and shares["upper"].max() <= 1
    chol = sp.svar(fit, B=[[np.nan, 0], [np.nan, np.nan]])
    bands = chol.irf(3, ci="bootstrap", reps=60, seed=1, boot="kilian")
    row = bands[(bands.shock == "shock1") & (bands.response == "b")]
    # same generator, same identification
    np.testing.assert_allclose(row["lower"], one["lower"]["a -> b"], atol=1e-8)
    with pytest.raises(MethodIncompatibility, match="boot"):
        fit.fevd(3, ci="bootstrap", boot="hall")


def test_structural_fevd_bands(fit):
    nan = np.nan
    chol = sp.svar(fit, B=[[nan, 0], [nan, nan]])
    structural = chol.fevd(4, ci="bootstrap", reps=80, seed=9)
    assert list(structural.columns) == [
        "shock",
        "response",
        "period",
        "fevd",
        "se",
        "lower",
        "upper",
    ]
    reduced = fit.fevd(4, ci="bootstrap", reps=80, seed=9)
    a = structural[(structural.shock == "shock1") & (structural.response == "b")]
    b = reduced[(reduced.shock == "a") & (reduced.response == "b")]
    # a recursive structural model is the Cholesky decomposition
    np.testing.assert_allclose(a["fevd"], b["fevd"], atol=1e-10)
    np.testing.assert_allclose(a["lower"], b["lower"], atol=1e-8)
    np.testing.assert_allclose(a["upper"], b["upper"], atol=1e-8)
    assert structural["lower"].min() >= 0 and structural["upper"].max() <= 1
    # without ci= the old columns
    assert list(chol.fevd(2).columns) == ["shock", "response", "period", "fevd"]
    sign = sp.svar(fit, sign={"s": {"a": "+", "b": "+"}}, n_draws=20, seed=0)
    with pytest.raises(MethodIncompatibility, match="identified set"):
        sign.fevd(3, ci="bootstrap")


def test_uncorrected_bootstrap_warns_on_a_persistent_var():
    from statspai.exceptions import AssumptionWarning

    fit = sp.var(_persistent(2001, T=200), lags=1)
    with pytest.warns(AssumptionWarning, match="kilian"):
        sp.irf(fit, periods=2, ci="bootstrap", reps=30, seed=0, boot="efron")
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        sp.irf(fit, periods=2, ci="bootstrap", reps=30, seed=0)  # the default


def test_bias_corrected_bootstrap_is_the_default(fit):
    default = sp.irf(fit, periods=3, ci="bootstrap", reps=40, seed=3)
    named = sp.irf(fit, periods=3, ci="bootstrap", reps=40, seed=3, boot="kilian")
    assert default["ci"]["boot"] == "kilian"
    np.testing.assert_array_equal(default["upper"]["a -> b"], named["upper"]["a -> b"])
    plain = sp.irf(fit, periods=3, ci="bootstrap", reps=40, seed=3, boot="efron")
    assert not np.array_equal(plain["upper"]["a -> b"], named["upper"]["a -> b"])
    # the variance decomposition and structural responses follow it
    shares = fit.fevd(3, ci="bootstrap", reps=40, seed=3)
    same = fit.fevd(3, ci="bootstrap", reps=40, seed=3, boot="kilian")
    np.testing.assert_array_equal(shares["upper"], same["upper"])
