"""Tests for sp.recentered_iv (Borusyak and Hull 2023)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.iv.recentered import _ri_confidence_set, _ri_pvalue


def _market_access_draw(rng, n_units=300, n_lines=150, beta=0.5, endogenous=True):
    """Network-exposure DGP with randomly opened lines and central units.

    Central units are linked to more lines (non-random exposure) and also
    have higher unobserved productivity, so the realised exposure is
    correlated with the error although the line openings are random.
    """
    centrality = rng.uniform(0, 1, n_units)
    p_link = 0.05 + 0.3 * centrality
    W = (rng.uniform(size=(n_units, n_lines)) < p_link[:, None]).astype(float)
    g = rng.binomial(1, 0.3, n_lines).astype(float)
    z = W @ g
    cf = np.column_stack([W @ rng.permutation(g) for _ in range(199)])
    eps = 2.0 * centrality + rng.normal(scale=0.5, size=n_units)
    if endogenous:
        x = 0.7 * z + 0.8 * eps + rng.normal(size=n_units)
    else:
        x = z
    y = beta * x + eps
    df = pd.DataFrame({"y": y, "x": x, "z": z, "cen": centrality})
    return df, cf


def _mc(method, endogenous, reps=40, seed=1):
    rng = np.random.default_rng(seed)
    adj, naive = [], []
    for _ in range(reps):
        df, cf = _market_access_draw(rng, endogenous=endogenous)
        treat = "x" if endogenous else "z"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.recentered_iv(df, "y", treat, "z", cf, method=method, ri=False)
        adj.append(res.estimate)
        naive.append(res.diagnostics["unadjusted_estimate"])
    return np.array(adj), np.array(naive)


@pytest.mark.parametrize("method", ["recenter", "control"])
@pytest.mark.parametrize("endogenous", [True, False])
def test_recovers_truth_where_unadjusted_is_biased(method, endogenous):
    adj, naive = _mc(method, endogenous)
    mc_se_adj = adj.std(ddof=1) / np.sqrt(len(adj))
    mc_se_naive = naive.std(ddof=1) / np.sqrt(len(naive))
    # Known truth beta = 0.5: the adjusted estimator is centred on it
    # (4 Monte Carlo SEs) while the unadjusted one is far off.
    assert abs(adj.mean() - 0.5) < 4 * mc_se_adj + 0.02
    assert abs(naive.mean() - 0.5) > 10 * mc_se_naive


def _small_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "z": rng.normal(size=n),
            "c1": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.0, n),
            "g": rng.integers(0, 25, n),
        }
    )
    df["x"] = 0.8 * df.z + 0.3 * df.c1 + rng.normal(size=n)
    df["y"] = 0.4 * df.x + 0.5 * df.c1 + rng.normal(size=n)
    df["mu"] = 0.3 * df.c1 + rng.normal(scale=0.2, size=n)
    return df


@pytest.mark.parametrize(
    "kw,ivkw",
    [
        ({}, {"robust": "hc1"}),
        ({"weights": "w", "cluster": "g"}, {"weights": "w", "cluster": "g"}),
    ],
)
def test_zero_expected_instrument_equals_plain_iv(kw, ivkw):
    df = _small_data()
    cf = np.zeros((len(df), 5))
    with pytest.warns(sp.AssumptionWarning, match="same for every unit"):
        res = sp.recentered_iv(df, "y", "x", "z", cf, covariates=["c1"], ri=False, **kw)
    ref = sp.iv("y ~ (x ~ z) + c1", data=df, **ivkw)
    # Identity: with mu == 0 the recentered instrument is z itself.
    assert res.estimate == pytest.approx(ref.params["x"], abs=1e-10)
    assert res.se == pytest.approx(ref.std_errors["x"], abs=1e-10)


def test_control_method_equals_iv_with_mu_control():
    df = _small_data(seed=3)
    rng = np.random.default_rng(7)
    # Draws whose row mean is exactly df.mu.
    noise = rng.normal(size=(len(df), 6))
    noise -= noise.mean(axis=1, keepdims=True)
    cf = df["mu"].to_numpy()[:, None] + noise
    res = sp.recentered_iv(df, "y", "x", "z", cf, method="control", ri=False)
    ref = sp.iv("y ~ (x ~ z) + mu", data=df, robust="hc1")
    assert res.estimate == pytest.approx(ref.params["x"], abs=1e-10)
    assert res.se == pytest.approx(ref.std_errors["x"], abs=1e-10)
    assert res.diagnostics["mu_coefficient"] == pytest.approx(
        ref.params["mu"], abs=1e-10
    )
    # Recentering: same data, instrument z - mu.
    df["zt"] = df.z - df.mu
    rec = sp.recentered_iv(df, "y", "x", "z", cf, ri=False)
    ref2 = sp.iv("y ~ (x ~ zt)", data=df, robust="hc1")
    assert rec.estimate == pytest.approx(ref2.params["x"], abs=1e-10)


@pytest.mark.parametrize("method", ["recenter", "control"])
def test_hodges_lehmann_equals_estimate(method):
    # Lemma 2 of the paper: the estimator solves T(b) = E*[T*(b)].
    rng = np.random.default_rng(5)
    df, cf = _market_access_draw(rng)
    df["wt"] = rng.uniform(0.5, 1.5, len(df))
    res = sp.recentered_iv(
        df, "y", "x", "z", cf, covariates=["cen"], weights="wt", method=method
    )
    assert res.diagnostics["ri_hodges_lehmann"] == pytest.approx(res.estimate, rel=1e-9)


def test_ri_confidence_set_matches_brute_force_inversion():
    rng = np.random.default_rng(11)
    S = 99
    a, c = 1.3, 2.0
    a_s = rng.normal(scale=1.0, size=S)
    c_s = rng.normal(scale=1.0, size=S)
    sets = _ri_confidence_set(a, c, a_s, c_s, alpha=0.1)
    grid = np.linspace(-20, 20, 4001)
    for b in grid:
        inside = any(lo <= b <= hi for lo, hi in sets)
        near_edge = any(min(abs(b - lo), abs(b - hi)) < 1e-9 for lo, hi in sets)
        if near_edge:
            continue
        assert inside == (_ri_pvalue(a, c, a_s, c_s, b) > 0.1)


def test_ri_set_unbounded_when_shocks_do_not_move_treatment():
    rng = np.random.default_rng(2)
    n = 200
    z = rng.normal(size=n)
    cf = rng.normal(size=(n, 99))
    df = pd.DataFrame({"z": z, "x": rng.normal(size=n)})
    df["y"] = rng.normal(size=n)
    with pytest.warns(sp.AssumptionWarning, match="unbounded"):
        res = sp.recentered_iv(df, "y", "x", "z", cf)
    assert not res.diagnostics["ri_ci_bounded"]


def test_ri_pvalue_valid_under_sharp_null():
    # Exchangeable draws: the realised shocks are one more permutation, so
    # the RI test is exact. Rejection rate at 10% across replications.
    rng = np.random.default_rng(123)
    pvals = []
    for _ in range(120):
        df, cf = _market_access_draw(rng, n_units=120, beta=0.0, endogenous=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.recentered_iv(df, "y", "z", "z", cf[:, :49])
        pvals.append(res.diagnostics["ri_pvalue"])
    pvals = np.array(pvals)
    # Exact test: P(p <= 0.1) <= 0.1. Binomial(120, 0.1) has sd 0.027;
    # allow 3 sd above.
    assert np.mean(pvals <= 0.1) <= 0.1 + 3 * 0.0274
    # And not degenerate: some spread across (0, 1].
    assert 0.3 < np.median(pvals) < 0.7


def test_ri_rejects_with_strong_effect():
    rng = np.random.default_rng(9)
    df, cf = _market_access_draw(rng, beta=3.0, endogenous=False)
    res = sp.recentered_iv(df, "y", "z", "z", cf)
    assert res.diagnostics["ri_pvalue"] <= 0.01
    assert res.diagnostics["ri_ci_bounded"]
    lo, hi = res.diagnostics["ri_ci"][0][0], res.diagnostics["ri_ci"][-1][1]
    assert lo < 3.0 < hi


def test_balance_test_flags_correlation_with_mu_not_with_noise():
    rng = np.random.default_rng(4)
    df, cf = _market_access_draw(rng)
    df["noise"] = rng.normal(size=len(df))
    res = sp.recentered_iv(df, "y", "x", "z", cf, balance=["noise", "cen"])
    tab = pd.DataFrame(res.diagnostics["balance"]).set_index("variable")
    assert set(tab.index) == {"noise", "cen", "joint"}
    assert ((tab["ri_pvalue"] > 0) & (tab["ri_pvalue"] <= 1)).all()


def test_counterfactual_given_as_columns_matches_array():
    rng = np.random.default_rng(6)
    df, cf = _market_access_draw(rng)
    cols = [f"cf{s}" for s in range(20)]
    df = pd.concat([df, pd.DataFrame(cf[:, :20], columns=cols)], axis=1)
    r1 = sp.recentered_iv(df, "y", "x", "z", cols, ri=False)
    r2 = sp.recentered_iv(df, "y", "x", "z", cf[:, :20], ri=False)
    assert r1.estimate == pytest.approx(r2.estimate, abs=1e-12)


def test_missing_rows_are_dropped_with_warning():
    rng = np.random.default_rng(6)
    df, cf = _market_access_draw(rng)
    df.loc[3, "y"] = np.nan
    cf[5, 0] = np.nan
    with pytest.warns(sp.AssumptionWarning, match="dropped 2 rows"):
        res = sp.recentered_iv(df, "y", "x", "z", cf, ri=False)
    assert res.n_obs == len(df) - 2


def test_fewer_than_two_draws_raises():
    df = _small_data()
    with pytest.raises(sp.DataInsufficient):
        sp.recentered_iv(df, "y", "x", "z", df[["mu"]])


def test_mismatched_shape_raises():
    df = _small_data()
    with pytest.raises(sp.MethodIncompatibility):
        sp.recentered_iv(df, "y", "x", "z", np.zeros((len(df) - 1, 10)))


def test_draws_identical_to_realised_raise():
    df = _small_data()
    cf = np.repeat(df[["z"]].to_numpy(), 10, axis=1)
    with pytest.raises(sp.AssumptionViolation):
        sp.recentered_iv(df, "y", "x", "z", cf)


def test_bad_method_and_missing_column():
    df = _small_data()
    cf = np.zeros((len(df), 3))
    with pytest.raises(sp.MethodIncompatibility):
        sp.recentered_iv(df, "y", "x", "z", cf, method="bogus")
    with pytest.raises(sp.StatsPAIError):
        sp.recentered_iv(df, "y", "nope", "z", cf)
