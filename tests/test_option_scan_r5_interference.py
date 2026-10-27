"""Option scan, round 5: ``interference/`` (outside network_exposure)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import IdentificationFailure


def _rollout(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(30):
        ft = 4 if c < 10 else (6 if c < 20 else 0)
        fe = rng.normal()
        for t in range(9):
            d = 1 if (ft > 0 and t >= ft) else 0
            y = 1.0 + fe + 0.2 * t + 1.5 * d + rng.normal(scale=0.3)
            rows.append({"cluster": c, "time": t, "first_treat": ft, "y": y})
    return pd.DataFrame(rows)


def _fit(df):
    return sp.cluster_staggered_rollout(
        df, y="y", cluster="cluster", time="time", first_treat="first_treat"
    )


def test_staggered_rollout_gappy_time_equals_consecutive():
    """Relabelling periods 0..8 as 2000, 2002, ... changes nothing."""
    df = _rollout()
    base = _fit(df)
    g = df.copy()
    g["time"] = 2000 + 2 * g["time"]
    g["first_treat"] = np.where(g["first_treat"] > 0, 2000 + 2 * g["first_treat"], 0)
    gap = _fit(g)
    assert gap.overall_att == base.overall_att
    assert gap.overall_se == base.overall_se
    np.testing.assert_array_equal(
        gap.event_study.to_numpy(), base.event_study.to_numpy()
    )


@pytest.mark.parametrize("share", ["all", "some"])
def test_staggered_rollout_nan_first_treat_is_never_treated(share):
    df = _rollout()
    base = _fit(df)
    h = df.copy()
    h["first_treat"] = h["first_treat"].astype(float)
    mask = h["first_treat"] == 0
    if share == "some":
        mask &= h["cluster"] >= 25
    h.loc[mask, "first_treat"] = np.nan
    got = _fit(h)
    assert got.n_clusters == base.n_clusters
    assert got.overall_att == base.overall_att


def test_peer_effects_refuses_reflection_design():
    """Complete groups of equal size: I, G and G^2 are linearly dependent and
    the endogenous peer effect is not identified (Bramoulle et al. 2009)."""
    rng = np.random.default_rng(0)
    G, m = 100, 4
    n = G * m
    W = np.zeros((n, n))
    for g in range(G):
        idx = np.arange(g * m, (g + 1) * m)
        W[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(W, 0.0)
    Wn = W / W.sum(1, keepdims=True)
    x = rng.normal(size=n)
    y = np.linalg.solve(
        np.eye(n) - 0.3 * Wn, 1 + 0.8 * x + 0.5 * (Wn @ x) + rng.normal(size=n)
    )
    with pytest.raises(IdentificationFailure):
        sp.peer_effects(pd.DataFrame({"y": y, "x": x}), y="y", covariates=["x"], W=W)


def test_peer_effects_identified_design_still_fits():
    rng = np.random.default_rng(1)
    n = 300
    W = (rng.uniform(size=(n, n)) < 0.02).astype(float)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(1, keepdims=True)
    Wn = W / np.where(rs == 0, 1, rs)
    x = rng.normal(size=n)
    y = np.linalg.solve(np.eye(n) - 0.3 * Wn, 1 + 0.8 * x + 0.5 * (Wn @ x))
    res = sp.peer_effects(pd.DataFrame({"y": y, "x": x}), y="y", covariates=["x"], W=W)
    # noiseless outcome: 2SLS recovers the structural coefficients
    assert res.endogenous_peer == pytest.approx(0.3, abs=1e-8)


def test_network_hte_se_is_the_sandwich():
    """The orthogonal-score variance is the sandwich of the residual
    regression (HC0), not sigma^2 (X'X)^-1."""
    rng = np.random.default_rng(3)
    n = 600
    x1 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.5).astype(float)
    e = rng.uniform(size=n)
    y = 1 + 0.8 * d + 0.5 * e + 0.3 * x1 + (0.2 + 2.0 * d) * rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "e": e, "x1": x1})
    res = sp.network_hte(
        df,
        y="y",
        treatment="d",
        neighbor_exposure="e",
        covariates=["x1"],
        n_folds=3,
        random_state=0,
    )
    # Rebuild the residuals with the same folds and learners.
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import KFold

    X = x1.reshape(-1, 1)
    ry, rd, re = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(3, shuffle=True, random_state=0).split(X):
        gy = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0)
        gd = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
        ge = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0)
        ry[te] = y[te] - gy.fit(X[tr], y[tr]).predict(X[te])
        rd[te] = d[te] - gd.fit(X[tr], d[tr].astype(int)).predict_proba(X[te])[:, 1]
        re[te] = e[te] - ge.fit(X[tr], e[tr]).predict(X[te])
    import statsmodels.api as sm

    fit = sm.OLS(ry, np.column_stack([rd, re])).fit(cov_type="HC0")
    assert res.direct_effect == pytest.approx(fit.params[0], rel=1e-10)
    assert res.direct_se == pytest.approx(fit.bse[0], rel=1e-10)
    assert res.spillover_se == pytest.approx(fit.bse[1], rel=1e-10)
