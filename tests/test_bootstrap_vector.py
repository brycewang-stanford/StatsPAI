"""``sp.bootstrap`` of a coefficient vector (Stata ``bootstrap _b: regress``).

The UCT replication (QJE 2016) needed Stata's clustered bootstrap of a
regression: every coefficient, their covariance, and ``idcluster()`` so a
cluster drawn twice counts as two groups. ``sp.bootstrap`` took one scalar at
a time and returned no covariance. The bootstrap is random, so these tests
check the construction exactly (covariance = covariance of the stored
replicates, normal-based intervals, reproducibility, idcluster semantics)
and screen the scale against the analytic cluster-robust variance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    m, G = 900, 45
    df = pd.DataFrame(dict(x=rng.normal(size=m), g=np.arange(m) % G))
    df["y"] = 1.0 + df.x + rng.normal(size=G)[df.g] + rng.normal(size=m)
    return df


def _fit(d):
    return sp.regress("y ~ x", data=d)


def test_vector_statistic_returns_regression_table(data):
    r = sp.bootstrap(data, _fit, n_boot=300, cluster="g", seed=7)
    assert isinstance(r, sp.EconometricResults)
    draws = r.model_info["boot_distribution"]
    V = np.cov(draws.to_numpy(), rowvar=False, ddof=1)
    np.testing.assert_allclose(r.vcov().to_numpy(), V, rtol=1e-14)
    np.testing.assert_allclose(r.std_errors.to_numpy(), np.sqrt(np.diag(V)))
    # Observed coefficients, normal-based interval (Stata's bootstrap table).
    full = _fit(data)
    pd.testing.assert_series_equal(r.params, full.params, check_names=False)
    ci = r.conf_int()
    half = (ci.iloc[:, 1] - ci.iloc[:, 0]) / 2
    np.testing.assert_allclose(half, 1.959963984540054 * r.std_errors, rtol=1e-12)
    pct = r.model_info["percentile_ci"]
    np.testing.assert_allclose(
        pct["ci_lower"], np.percentile(draws, 2.5, axis=0), rtol=1e-14
    )
    again = sp.bootstrap(data, _fit, n_boot=300, cluster="g", seed=7)
    pd.testing.assert_frame_equal(again.vcov(), r.vcov())


def test_cluster_bootstrap_scale_matches_cluster_robust_variance(data):
    """Screen, not a parity claim: B = 1500 draws, the slope's bootstrap SE
    is within 10% of the CR0 sandwich (the cluster bootstrap has no
    small-sample factor)."""
    r = sp.bootstrap(data, _fit, n_boot=1500, cluster="g", seed=11)
    cr1 = sp.regress("y ~ x", data=data, cluster="g")
    G, n, k = 45, len(data), 2
    cr0 = cr1.std_errors["x"] * np.sqrt((G - 1) / G * (n - k) / (n - 1))
    assert r.std_errors["x"] == pytest.approx(cr0, rel=0.10)


def test_idcluster_gives_each_draw_its_own_group(data):
    seen = []

    def stat(d):
        if "cid" in d:
            per = d.groupby("cid")["g"].nunique()
            seen.append((d["cid"].nunique(), int(per.max())))
        return pd.Series({"m": d.y.mean(), "n": float(d["cid"].nunique())})

    r = sp.bootstrap(data, stat, n_boot=30, cluster="g", idcluster="cid", seed=2)
    # every draw has G distinct ids, each covering exactly one original cluster
    assert all(n_ids == 45 and per == 1 for n_ids, per in seen)
    assert r.model_info["idcluster"] == "cid"
    with pytest.raises(ValueError, match="requires cluster"):
        sp.bootstrap(data, stat, n_boot=10, idcluster="cid")


def test_failed_replications_are_counted_not_hidden(data):
    calls = {"n": 0}

    def flaky(d):
        calls["n"] += 1
        if calls["n"] % 5 == 0:
            raise RuntimeError("boom")
        return _fit(d)

    with pytest.warns(RuntimeWarning, match="failed"):
        r = sp.bootstrap(data, flaky, n_boot=50, cluster="g", seed=1)
    assert r.model_info["n_failed"] == 10
    assert r.model_info["n_boot"] == 40
