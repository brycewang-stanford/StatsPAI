"""Tests for sp.noisy_covariates (Agarwal-Singh noisy-covariate DR).

Evidence tier T1: known-truth recovery on low-rank designs, a
replication of the paper's own simulation design, and an exact identity
against a hand-written per-arm OLS + balancing-weight computation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.measurement_error.noisy_covariates import gavish_donoho_rank


def _low_rank_data(seed, n=400, p=120, r=3, miss=0.2):
    """Low-rank X, heteroscedastic noise, missing cells, confounded D."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, r))
    X = L @ rng.normal(size=(r, p))
    Z = X + rng.normal(scale=0.4 + 0.2 * np.abs(X), size=X.shape)
    Z[rng.uniform(size=Z.shape) < miss] = np.nan
    cols = [f"z{j}" for j in range(p)]
    df = pd.DataFrame(Z, columns=cols)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.8 * L[:, 0])))
    tau = 2.0 + 0.5 * L[:, 1]
    df["y"] = d * tau + 1.0 + 1.5 * L[:, 0] + rng.normal(size=n)
    df["d"] = d
    return df, cols, L, rng, float(tau.mean())


def test_ate_recovers_truth_with_noise_and_missing():
    # 60 replications; truth is the sample ATE (the estimand conditions
    # on the covariates). MC SE of the mean bias ~ 0.11 / sqrt(60) = 0.014,
    # so |mean bias| < 0.05 is > 3 MC SEs. Coverage SE ~ 0.028 at 60 reps.
    errs, covered, raw_err = [], [], []
    for b in range(60):
        df, cols, _, _, truth = _low_rank_data(b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.noisy_covariates(df, "y", "d", cols, seed=b)
        errs.append(r.estimate - truth)
        covered.append(r.ci[0] <= truth <= r.ci[1])
        raw_err.append(r.model_info["estimate_without_cleaning"] - truth)
    errs = np.asarray(errs)
    assert abs(errs.mean()) < 0.05
    assert np.mean(covered) >= 0.85
    # Without cleaning, p (120) is close to the per-arm training rows and
    # the DR score is unstable: its typical error is several times larger.
    assert np.median(np.abs(raw_err)) > 3 * np.median(np.abs(errs))


def test_replicates_paper_simulation_design():
    # Agarwal-Singh Appendix L design, 20% noise-to-signal, oracle k = r = 5,
    # at n = 400 (coverage at the paper's n = 100 is 0.92 here vs 0.96
    # reported; see the docstring). 80 reps: MC SE of the mean ~ 0.017.
    est, covered = [], []
    n, p, r = 400, 100, 5
    beta = 1.0 / np.arange(1, p + 1) ** 2
    for b in range(80):
        rng = np.random.default_rng(1000 + b)
        X = rng.normal(size=(n, r)) @ rng.normal(size=(r, p))
        prop = 0.9 / (1 + np.exp(-0.25 * X @ beta)) + 0.05
        d = rng.binomial(1, prop)
        yv = 2.2 * d + 1.2 * X @ beta + d * X[:, 0] + rng.normal(size=n)
        Z = X + rng.normal(scale=1.0, size=X.shape)  # variance 0.2 * r
        df = pd.DataFrame(Z, columns=[f"z{j}" for j in range(p)])
        df["y"], df["d"] = yv, d
        res = sp.noisy_covariates(df, "y", "d", list(df.columns[:p]), rank=5, seed=b)
        est.append(res.estimate)
        covered.append(res.ci[0] <= 2.2 <= res.ci[1])
    assert abs(np.mean(est) - 2.2) < 0.06
    assert np.mean(covered) >= 0.87


def test_identity_full_rank_equals_hand_dr():
    # No corruption, k = p: the cleaned estimator is the cross-fitted DR
    # estimator with per-arm OLS on [1, X] and the linear balancing weight
    # solving (sum_arm x x') eta = sum_train x. Exact up to rounding.
    rng = np.random.default_rng(7)
    n, p = 300, 6
    X = rng.normal(size=(n, p))
    d = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    yv = d * (1 + X[:, 1]) + X @ np.arange(1, p + 1) / p + rng.normal(size=n)
    cols = [f"x{j}" for j in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["y"], df["d"] = yv, d
    with pytest.warns(sp.AssumptionWarning, match="low rank"):
        res = sp.noisy_covariates(df, "y", "d", cols, rank=p, seed=3)

    # Fold assignment documented in the module: stratified by arm.
    frng = np.random.default_rng(3)
    folds = np.empty(n, dtype=int)
    for g in (np.flatnonzero(d == 1), np.flatnonzero(d == 0)):
        perm = frng.permutation(g)
        folds[perm] = np.arange(perm.size) % 2
    Xt = np.column_stack([np.ones(n), X])
    psi = np.empty(n)
    for f in (0, 1):
        te, tr = folds == f, folds != f
        tot = Xt[tr].sum(axis=0)
        coef, eta = {}, {}
        for a in (0, 1):
            idx = tr & (d == a)
            coef[a] = np.linalg.lstsq(Xt[idx], yv[idx], rcond=None)[0]
            eta[a] = np.linalg.solve(Xt[idx].T @ Xt[idx], tot)
        g1, g0 = Xt[te] @ coef[1], Xt[te] @ coef[0]
        alpha = np.where(d[te] == 1, Xt[te] @ eta[1], -(Xt[te] @ eta[0]))
        gd = np.where(d[te] == 1, g1, g0)
        psi[te] = g1 - g0 + alpha * (yv[te] - gd)
    np.testing.assert_allclose(res.estimate, psi.mean(), rtol=1e-9)
    se = np.sqrt(np.mean((psi - psi.mean()) ** 2) / n)
    np.testing.assert_allclose(res.se, se, rtol=1e-9)


def test_constant_weights_are_identity_and_weighted_truth():
    df, cols, L, rng, _ = _low_rank_data(11, n=1500, p=200)
    base = sp.noisy_covariates(df, "y", "d", cols, rank=3)
    df["w_const"] = 3.0
    same = sp.noisy_covariates(df, "y", "d", cols, rank=3, weights="w_const")
    np.testing.assert_allclose(same.estimate, base.estimate, rtol=1e-12)
    np.testing.assert_allclose(same.se, base.se, rtol=1e-12)
    # Weights correlated with the effect modifier move the estimand.
    ell = np.exp(L[:, 1])
    df["w"] = ell
    tau = 2.0 + 0.5 * L[:, 1]
    truth = float(np.sum(ell * tau) / np.sum(ell))
    res = sp.noisy_covariates(df, "y", "d", cols, rank=3, weights="w")
    assert abs(res.estimate - truth) < 4 * res.se
    assert truth - float(tau.mean()) > 3 * res.se  # the weighting matters


def test_late_and_partially_linear_known_truth():
    rng = np.random.default_rng(5)
    n, p, r = 1500, 250, 3
    L = rng.normal(size=(n, r))
    X = L @ rng.normal(size=(r, p))
    Z = X + rng.normal(scale=0.4 + 0.2 * np.abs(X), size=X.shape)
    Z[rng.uniform(size=Z.shape) < 0.2] = np.nan
    cols = [f"z{j}" for j in range(p)]
    df = pd.DataFrame(Z, columns=cols)
    # LATE: 60% compliers with constant effect 1.5.
    u = rng.binomial(1, 1 / (1 + np.exp(-0.8 * L[:, 0])))
    typ = rng.uniform(size=n)
    dl = np.where(typ < 0.6, u, np.where(typ < 0.8, 1, 0))
    df["u"], df["dl"] = u, dl
    df["yl"] = 1.5 * dl + L[:, 0] + 0.5 * L[:, 1] + rng.normal(size=n)
    late = sp.noisy_covariates(
        df, "yl", "dl", cols, estimand="LATE", instrument="u", rank=3
    )
    assert abs(late.estimate - 1.5) < 4 * late.se
    assert abs(late.model_info["first_stage"] - 0.6) < 0.08
    # Partially linear: continuous dose confounded by the first factor.
    dd = L[:, 0] + rng.normal(size=n)
    df["dd"] = dd
    df["y2"] = 0.7 * dd + L[:, 0] - L[:, 2] + rng.normal(size=n)
    plr = sp.noisy_covariates(df, "y2", "dd", cols, estimand="partially_linear")
    assert abs(plr.estimate - 0.7) < 4 * plr.se


def test_gavish_donoho_rank_finds_low_rank():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(300, 4)) @ rng.normal(size=(4, 80))
    k, tau, s = gavish_donoho_rank(M + 0.3 * rng.normal(size=M.shape))
    assert k == 4
    assert s[3] > tau > s[4]


def test_result_contract_and_diagnostics():
    df, cols, _, _, _ = _low_rank_data(2)
    df.loc[df.index[:5], "y"] = np.nan
    res = sp.noisy_covariates(df, "y", "d", cols)
    mi = res.model_info
    assert mi["n_dropped_rows"] == 5 and res.n_obs == len(df) - 5
    assert len(mi["rank_by_fold"]) == 2
    assert 0.75 < mi["rho_min"] <= 1.0
    assert mi["p_in_theory_range"] is True
    assert res.ci[0] < res.estimate < res.ci[1]
    assert np.isfinite(res.pvalue)
    assert res._influence_funcs.shape == (res.n_obs,)


def test_edge_rank_too_large():
    df, cols, _, _, _ = _low_rank_data(3, n=100, p=40)
    with pytest.raises(MethodIncompatibility, match="rank"):
        sp.noisy_covariates(df, "y", "d", cols, rank=41)
    with pytest.raises(MethodIncompatibility):
        sp.noisy_covariates(df, "y", "d", cols, rank=0)


def test_edge_all_missing_column():
    df, cols, _, _, _ = _low_rank_data(4, n=200, p=40)
    df["z0"] = np.nan
    with pytest.raises(DataInsufficient, match="z0"):
        sp.noisy_covariates(df, "y", "d", cols)


def test_edge_non_binary_treatment_and_bad_options():
    df, cols, _, _, _ = _low_rank_data(5, n=200, p=40)
    df["dose"] = np.arange(len(df)) % 3
    with pytest.raises(Exception, match="0/1|binary"):
        sp.noisy_covariates(df, "y", "dose", cols)
    with pytest.raises(MethodIncompatibility, match="instrument"):
        sp.noisy_covariates(df, "y", "d", cols, estimand="LATE")
    with pytest.raises(MethodIncompatibility, match="estimand"):
        sp.noisy_covariates(df, "y", "d", cols, estimand="ATT")
    with pytest.raises(MethodIncompatibility, match="n_folds"):
        sp.noisy_covariates(df, "y", "d", cols, n_folds=1)
    df.loc[df.index[0], "z1"] = np.inf
    with pytest.raises(MethodIncompatibility, match="infinite"):
        sp.noisy_covariates(df, "y", "d", cols)


def test_warns_when_covariates_are_not_low_rank():
    rng = np.random.default_rng(9)
    n, p = 300, 12
    X = rng.normal(size=(n, p))  # full rank, no factor structure
    d = rng.binomial(1, 0.5, size=n)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df["y"] = d + X[:, 0] + rng.normal(size=n)
    df["d"] = d
    with pytest.warns(sp.AssumptionWarning, match="low rank"):
        sp.noisy_covariates(df, "y", "d", list(df.columns[:p]), rank=10)
