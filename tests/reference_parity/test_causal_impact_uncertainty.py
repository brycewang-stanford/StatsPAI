"""``sp.causal_impact`` intervals carry parameter uncertainty (2026-10-19).

Before, the standard error of the average effect was the forecast-error
variance *given* the plug-in parameters. In 200 null replications per
design its 95% interval covered 79.5% (n_pre 40, rho 0.6), 70.5%
(n_pre = n_post = 30), 57.5% (rho 0.9), 90.5% (rho 0.2, n_pre 80) and 65%
(four covariates, n_pre 30). It now covers 93.5%, 94%, 89.5%, 92% and 94%
on the same draws: the error of the forecast actually made,
``u_post - L u_pre``, under an AR(1)+noise error covariance fitted by REML,
with a t quantile. Evidence here: two exact identities and a seeded
coverage run.

2026-10-20: the critical value of the average / cumulative effect also
carries the estimation of that error model (bias-corrected parametric
bootstrap-t, never below the t quantile). Coverage on the same draws:
94.5%, 94.5%, 93% (was 89.5% in the hard design n_pre 40, rho 0.9),
92.5% and 95%.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.causal_impact import _uncertainty as U


def _sim(seed, n_pre=40, n_post=20, rho=0.6, k=2, noise=0.5):
    r = np.random.default_rng(seed)
    n = n_pre + n_post
    X = np.cumsum(r.normal(size=(n, k)), axis=0) * 0.3 + r.normal(size=(n, k))
    u = np.zeros(n)
    u[0] = r.normal() / np.sqrt(1 - rho**2)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + r.normal()
    y = 1 + X @ np.arange(1, k + 1) + u + noise * r.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(k)])
    df["y"] = y
    df["t"] = np.arange(n)
    return df


def _fit(df, n_pre):
    cov = [c for c in df if c.startswith("x")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.causal_impact(
            df, y="y", time="t", intervention_time=n_pre, covariates=cov
        )


def test_kalman_reml_equals_dense_reml():
    """The filter-based restricted likelihood is the dense-matrix one."""
    df = _sim(1)
    y = df["y"].to_numpy()[:40]
    X = np.column_stack([np.ones(40), df[["x0", "x1"]].to_numpy()[:40]])
    for rho, psi in [(0.6, 0.7), (-0.3, 0.2), (0.9, 1.0)]:
        nll, s2 = U._reml_objective(y, X, np.array([rho]), np.array([psi]))
        n, k = X.shape
        idx = np.arange(n)
        R = psi / (1 - rho**2) * rho ** np.abs(idx[:, None] - idx[None, :])
        R[idx, idx] += 1 - psi
        Ri = np.linalg.inv(R)
        XtRX = X.T @ Ri @ X
        b = np.linalg.solve(XtRX, X.T @ Ri @ y)
        r = y - X @ b
        s2_d = r @ Ri @ r / (n - k)
        dense = 0.5 * (
            (n - k) * np.log(s2_d)
            + np.linalg.slogdet(R)[1]
            + np.linalg.slogdet(XtRX)[1]
        )
        assert s2[0] == pytest.approx(s2_d, rel=1e-10)
        assert nll[0] == pytest.approx(dense, rel=1e-10)


def test_se_is_the_variance_of_the_forecast_actually_made():
    """y_hat_post = L y_pre with L X_o = X_p (beta cancels), and se_total^2
    is 1' Var(u_post - L u_pre) 1 under the reported error model."""
    df = _sim(2)
    res = _fit(df, 40)
    mi = res.model_info
    pre = df["t"].to_numpy() < 40
    X = np.column_stack([np.ones(len(df)), df[["x0", "x1"]].to_numpy()])
    L = U.predictor_map(X, pre, mi["model_params"])
    np.testing.assert_allclose(L @ X[pre], X[~pre], atol=1e-8)
    np.testing.assert_allclose(
        L @ df["y"].to_numpy()[pre], res.detail["predicted"].to_numpy()[~pre], rtol=1e-9
    )
    err = mi["error_model"]
    T = U.error_covariance(len(df), err)
    e = np.hstack([-L, np.eye((~pre).sum())])
    order = np.r_[np.flatnonzero(pre), np.flatnonzero(~pre)]
    V = e @ T[np.ix_(order, order)] @ e.T
    assert mi["se_total"] ** 2 == pytest.approx(V.sum(), rel=1e-10)
    assert res.se == pytest.approx(np.sqrt(V.sum()) / 20, rel=1e-10)
    assert mi["se_method"] == "reml_error_model"


@pytest.mark.slow
def test_null_coverage_is_close_to_nominal():
    """200 null replications (n_pre 40, n_post 20, rho 0.6, two covariates):
    0.935 observed; the old conditional interval covered 0.795. A binomial
    SE at 0.95 is 0.015, so require at least 0.90."""
    hits = 0
    reps = 200
    for s in range(reps):
        lo, hi = _fit(_sim(s), 40).ci
        hits += lo <= 0 <= hi
    assert hits / reps >= 0.90


def test_without_bootstrap_the_t_quantile_is_used():
    from scipy import stats

    df = _sim(3)
    cov = ["x0", "x1"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.causal_impact(
            df, y="y", time="t", intervention_time=40, covariates=cov, n_boot=0
        )
    mi = r.model_info
    crit = stats.t.ppf(0.975, mi["df"])
    assert mi["crit"] == pytest.approx(crit, rel=1e-12)
    assert r.ci[1] - r.estimate == pytest.approx(crit * r.se, rel=1e-12)
    t = r.estimate / r.se
    assert r.pvalue == pytest.approx(2 * stats.t.sf(abs(t), mi["df"]), rel=1e-12)


def test_bootstrap_is_seeded_and_the_interval_uses_its_quantile():
    df = _sim(4)
    a, b = _fit(df, 40), _fit(df, 40)
    assert a.ci == b.ci and a.pvalue == b.pvalue
    mi = a.model_info
    assert mi["crit_method"] == "bias-corrected parametric bootstrap-t"
    assert a.ci[1] - a.estimate == pytest.approx(mi["crit"] * a.se, rel=1e-12)
    assert 0.0 < a.pvalue <= 1.0


def test_vectorised_bootstrap_reml_is_the_grid_maximum():
    """_grid_reml_many returns, for each replicate, the grid point with the
    smallest REML criterion of _reml_objective (the same likelihood)."""
    r = np.random.default_rng(5)
    X = np.column_stack([np.ones(40), r.normal(size=(40, 2))])
    Z = r.normal(size=(40, 3)).cumsum(axis=0) * 0.3 + r.normal(size=(40, 3))
    got = U._grid_reml_many(Z, X)
    R, P = np.meshgrid(U._BOOT_RHO, U._BOOT_PSI, indexing="ij")
    for j in range(Z.shape[1]):
        nll, s2 = U._reml_objective(Z[:, j], X, R.ravel(), P.ravel())
        g = int(np.argmin(nll))
        assert got[j, 0] == R.ravel()[g] and got[j, 1] == P.ravel()[g]
        assert got[j, 2] == pytest.approx(s2[g], rel=1e-10)


@pytest.mark.slow
def test_null_coverage_in_the_hard_design():
    """n_pre 40, rho 0.9: the t quantile alone covered 0.895; with the
    bootstrap-t critical value 0.93 on these 200 seeded draws."""
    hits = 0
    reps = 200
    for s in range(reps):
        lo, hi = _fit(_sim(s, rho=0.9), 40).ci
        hits += lo <= 0 <= hi
    assert hits / reps >= 0.92
